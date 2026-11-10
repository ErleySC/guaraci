# -*- coding: utf-8 -*-
"""Cenarios realistas do modo sintetico (`simulador_espectral`)."""
from __future__ import annotations

import os
import re

import numpy as np
import pytest

from guaraci.config import Config
from guaraci.dados_io import generate_synthetic_data
from guaraci.simulador_espectral import (CENARIOS, cenario, decompor_variancia,
                                         gerar_cenario)


def _gerar(nome, **kw):
    args = dict(n_pontos=300, n_por_classe=10, n_replicas=3, seed=5)
    args.update(kw)
    return gerar_cenario(cenario(nome), **args)


@pytest.mark.parametrize("nome", sorted(CENARIOS))
def test_formato_e_reprodutibilidade(nome):
    eixo, X, rot, conc, mae, lote = _gerar(nome)
    assert X.shape == (90, 300) and np.isfinite(X).all()
    assert eixo.shape == (300,)
    assert set(rot) == {"Esp_A", "Esp_B", "Esp_C"}
    assert len(np.unique(mae)) == 30            # 3 classes x 10 amostras
    assert np.all(np.unique(mae, return_counts=True)[1] == 3)
    assert len(np.unique(lote)) == CENARIOS[nome].n_lotes
    np.testing.assert_array_equal(X, _gerar(nome)[1])
    assert not np.array_equal(X, _gerar(nome, seed=6)[1])


@pytest.mark.parametrize("nome", sorted(CENARIOS))
def test_replicas_sao_mais_parecidas_que_amostras_diferentes(nome):
    """Estrutura hierarquica: replicas variam, mas menos que amostras
    fisicas diferentes (senao a validacao agrupada nao faria diferenca;
    se fossem identicas, nao seriam replicas reais)."""
    _, X, _, _, mae, _ = _gerar(nome, n_por_classe=15)
    frac = decompor_variancia(X, mae)["frac_dentro"]
    assert 0.003 < frac < 0.5


def test_adulterantes_mantem_o_formato_de_mae_id():
    _, X, rot, conc, mae, _ = _gerar("nir", adulterantes=("A", "S"))
    puros = [m for m in mae if re.fullmatch(r"ES[ABC]-P\d\d", m)]
    adult = [m for m in mae if re.fullmatch(r"ES[ABC]-[AS]\d+\.\d\d", m)]
    assert len(puros) + len(adult) == len(mae)
    assert np.all(conc[[m in puros for m in mae]] == 0)
    assert conc.max() == pytest.approx(40.0)


def test_teor_e_recuperavel_e_classes_se_sobrepoem():
    """O cenario tem de ser um problema de verdade: o teor e' modelavel
    (R2cv alto) e as classes nao sao separadas trivialmente."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline

    from guaraci.preprocessamento import SNV, SavGol
    _, X, rot, conc, mae, _ = _gerar("nir", n_pontos=400, n_por_classe=20)
    pred = np.zeros_like(conc)
    for tr, va in GroupKFold(5).split(X, conc, mae):
        m = make_pipeline(SNV(), SavGol(15, 2, 1), PLSRegression(5, scale=False))
        pred[va] = m.fit(X[tr], conc[tr]).predict(X[va]).ravel()
    r2 = 1 - np.sum((pred - conc) ** 2) / np.sum((conc - conc.mean()) ** 2)
    assert r2 > 0.6
    # sem pre-processamento, a primeira PC e' espalhamento/linha de base,
    # nao classe: as classes se misturam nela
    from sklearn.decomposition import PCA
    t1 = PCA(1).fit_transform(X).ravel()
    medias = [t1[rot == c].mean() for c in np.unique(rot)]
    assert np.ptp(medias) < 2 * t1.std()


def test_raman_tem_fluorescencia_e_raios_cosmicos():
    _, X, *_ = gerar_cenario(cenario("raman", raios_cosmicos_por_espectro=1.0),
                             n_pontos=400, n_por_classe=10, n_replicas=3, seed=1)
    d = np.abs(np.diff(X, axis=1))
    assert np.mean(d.max(axis=1) > 0.8) > 0.4      # saltos de raio cosmico


def test_modo_sintetico_usa_o_cenario_da_config(monkeypatch):
    cfg = Config()
    cfg.synthetic_scenario = "nir"
    cfg.n_per_class = 6
    cfg.n_synthetic_points = 200
    wn, X, rot, conc, mae = generate_synthetic_data(cfg)
    assert X.shape == (54, 200)
    assert wn.min() >= cfg.wn_min - 1e-9 and wn.max() <= cfg.wn_max + 1e-9
    cfg.synthetic_scenario = "inexistente"
    with pytest.raises(ValueError, match="cenario sintetico desconhecido"):
        generate_synthetic_data(cfg)


def test_padrao_do_usuario_e_nir_e_testes_fixam_classico(monkeypatch):
    assert os.environ.get("GUARACI_SINTETICO_CENARIO") == "classico"
    assert Config().synthetic_scenario == "classico"
    monkeypatch.delenv("GUARACI_SINTETICO_CENARIO")
    assert Config().synthetic_scenario == "nir"


@pytest.mark.slow
def test_pipeline_completo_roda_no_cenario_nir(pq, tmp_path):
    cfg = pq.Config(
        input_folder=str(tmp_path / "dados"), output_root_folder=str(tmp_path / "saida"),
        mode="sintetico", synthetic_scenario="nir", n_per_class=8,
        n_synthetic_points=120, wn_min=400.0, wn_max=4001.0, n_splits_cv=3,
        n_repeats_cv=1, n_permutations=5, n_permutations_wold=5,
        n_bootstrap_vip=3, n_bootstrap_bca=20, n_monte_carlo=3,
        run_benchmark=False, run_monte_carlo=False, run_shap=False,
        run_wold=False, run_cv_anova=False, run_opls=False,
        executar_etapa4=False, comparar_pipelines=False,
        compare_hca_pipelines=False, max_lvs=6)
    os.makedirs(str(tmp_path / "dados"), exist_ok=True)
    pq.executar(cfg)
    assert list((tmp_path / "saida").rglob("resumo_modelo.txt"))
