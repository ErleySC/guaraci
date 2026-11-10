# -*- coding: utf-8 -*-
"""R7: preset `airpls_snv_sg_mc` (AirPLS -> SNV -> SG -> mean-centering), a tela [2.4] do Raman que o grava com
`remover_raios_cosmicos: true`, e a exportacao portatil do modelo com AirPLS e com a etapa de raios cosmicos.

Fase 1 (Raman sintetico de quantificacao, CV agrupada): R2 do teste externo 0,76 -> 0,96 e R2cv 0,879 -> 0,988 com
faixa >= 200 + raios cosmicos + SNV (a normalizacao e' o que faltava: a intensidade Raman absoluta depende de
potencia, foco e densidade da amostra)."""
from __future__ import annotations

import os

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from guaraci import catalogo_tecnicas as ct
from guaraci.config import Config
from guaraci.config_io import _CONFIG_SPEC, _coagir_valor
from guaraci.model_export import (UnsupportedModelError, export_portable_json, load_portable_json,
                                  predict_portable, save_portable_json)
from guaraci.preprocessamento import (AirPLS, SNV, RemocaoRaiosCosmicos, SavGol, build_preprocessor,
                                      descrever_preprocessamento)


def _raman_sintetico(n=40, p=400, seed=0, raios=False):
    """Bandas lorentzianas (meia largura de ~5 pontos, como num Raman real) + fluorescencia de fundo + escala por
    amostra (potencia/foco)."""
    rng = np.random.default_rng(seed)
    eixo = np.linspace(200.0, 3200.0, p)
    y_int = rng.integers(0, 3, size=n)
    X = np.empty((n, p))
    for i in range(n):
        bandas = sum(h / (1 + ((eixo - c) / 40.0) ** 2) for c, h in ((600, 1.0), (1000, 0.6 + 0.3 * y_int[i]), (1600, 0.8)))
        fluo = rng.uniform(0.5, 3.0) * (0.5 + np.exp(-((eixo - 1600) / 1500) ** 2))
        X[i] = rng.lognormal(0, 0.25) * (bandas + fluo) + rng.normal(0, 0.003, p)
    if raios:
        for i in range(0, n, 3):
            X[i, 30 + i] += 5.0
    return eixo, X, y_int


def test_preset_encadeia_airpls_snv_sg_mc_nessa_ordem():
    cfg = Config(default_preprocessing="airpls_snv_sg_mc")
    passos = build_preprocessor(cfg).steps
    assert [n for n, _ in passos] == ["airpls", "snv", "sg", "mc"]
    assert descrever_preprocessamento(cfg)[0] == ["AirPLS", "SNV", "SG1", "MC"]
    cfg.apply_despike = True
    assert [n for n, _ in build_preprocessor(cfg).steps] == ["despike", "airpls", "snv", "sg", "mc"]


def test_preset_equivale_a_composicao_manual():
    _eixo, X, _y = _raman_sintetico()
    cfg = Config(default_preprocessing="airpls_snv_sg_mc")
    obtido = build_preprocessor(cfg).fit_transform(X)
    esperado = AirPLS(lam=cfg.airpls_lam).transform(X)
    esperado = SNV().transform(esperado)
    esperado = SavGol(cfg.sg_window, cfg.sg_polyorder, cfg.sg_deriv).transform(esperado)
    esperado = StandardScaler(with_std=False).fit_transform(esperado)
    np.testing.assert_allclose(obtido, esperado, atol=1e-12)


def test_snv_depois_do_airpls_deixa_o_resultado_invariante_a_escala_da_amostra():
    """O motivo do preset: a escala por amostra (potencia, foco, densidade) sobra no AirPLS+SG+MC e some com o SNV."""
    _eixo, X, _y = _raman_sintetico(seed=3)
    novo = build_preprocessor(Config(default_preprocessing="airpls_snv_sg_mc")).fit(X)
    velho = build_preprocessor(Config(default_preprocessing="airpls_sg_mc")).fit(X)
    amostra = X[:6]
    np.testing.assert_allclose(novo.transform(3.0 * amostra), novo.transform(amostra), atol=1e-6)
    assert np.abs(velho.transform(3.0 * amostra) - velho.transform(amostra)).max() > 0.05


def test_nome_amigavel_do_preset_volta_para_o_preset():
    spec = next(s for s in _CONFIG_SPEC if s["key"] == "pre_processamento")
    assert _coagir_valor(spec, "AirPLS+SNV+SG+MC") == "airpls_snv_sg_mc"
    assert _coagir_valor(spec, "airpls_snv_sg_mc") == "airpls_snv_sg_mc"


def test_catalogo_do_raman_aponta_o_preset_novo_e_liga_raios_cosmicos():
    r = ct.INSTRUMENTOS["raman"]
    assert r["preproc"] == "airpls_snv_sg_mc" and r["raios_cosmicos"] is True
    assert r["faixa_min"] == 200.0
    assert "AirPLS+SNV+SG+MC" in r["preproc_rec"][0] and "AirPLS+SNV+SG+MC" in r["preproc_rec"][1]
    assert not any(v.get("raios_cosmicos") for k, v in ct.INSTRUMENTOS.items() if k != "raman")


def _tela_2_4(monkeypatch, cfg, tk, formato=""):
    from guaraci import guaraci as G
    ordem = G._tecnica_ordem()
    respostas = iter([str(ordem.index(tk) + 1), formato])
    monkeypatch.setattr(G, "_input", lambda *a, **k: next(respostas))
    monkeypatch.setattr(G, "_pause", lambda *a, **k: None)
    monkeypatch.setattr(G, "_cabecalho_tela", lambda *a, **k: None)
    monkeypatch.setattr(G, "_rodape_tela", lambda *a, **k: None)
    G._menu_technique(cfg)
    return cfg


def test_tela_2_4_do_raman_grava_o_preset_e_liga_os_raios_cosmicos(monkeypatch):
    cfg = _tela_2_4(monkeypatch, Config(), "raman", "3")
    assert cfg.default_preprocessing == "airpls_snv_sg_mc"
    assert cfg.apply_despike is True
    assert cfg.wn_min == 200.0 and cfg.mode == "csv"


def test_sair_do_raman_desliga_os_raios_cosmicos_mas_nao_mexe_na_escolha_do_usuario(monkeypatch):
    cfg = _tela_2_4(monkeypatch, Config(), "raman")
    assert cfg.apply_despike is True
    _tela_2_4(monkeypatch, cfg, "mir")
    assert cfg.apply_despike is False             # o despike era da tecnica anterior
    # quem ligou o despike a mao (sem ter passado pelo Raman) e escolhe NIR: a escolha dele fica
    cfg2 = Config(apply_despike=True)
    _tela_2_4(monkeypatch, cfg2, "nir")
    assert cfg2.apply_despike is True


def test_raios_cosmicos_ligados_nao_alteram_espectro_sem_raios():
    """Falso positivo: o despike nao toca espectros limpos de bandas estreitas."""
    _eixo, X, _y = _raman_sintetico(n=60, seed=5)
    np.testing.assert_array_equal(RemocaoRaiosCosmicos().transform(X), X)
    cfg = Config(default_preprocessing="airpls_snv_sg_mc")
    sem = build_preprocessor(cfg).fit_transform(X)
    cfg.apply_despike = True
    com = build_preprocessor(cfg).fit_transform(X)
    np.testing.assert_allclose(com, sem, atol=1e-12)


# ---- exportacao portatil -------------------------------------------------

def _pacote(preset, despike, seed=0):
    eixo, X, y_int = _raman_sintetico(seed=seed, raios=despike)
    cfg = Config(default_preprocessing=preset, apply_despike=despike)
    preproc = build_preprocessor(cfg).fit(X)
    lb = LabelBinarizer().fit(y_int)
    pls = PLSRegression(n_components=3, scale=False).fit(preproc.transform(X), np.asarray(lb.transform(y_int), dtype=float))
    return {"preprocessador": preproc, "pls_final": pls, "label_binarizer": lb, "classes": list(lb.classes_),
            "wavenumbers": eixo, "preset": preset, "wn_min": float(eixo.min()), "wn_max": float(eixo.max())}, eixo


def _oraculo(pkg, X):
    from tests.test_model_export import _prever_via_pacote_real
    return _prever_via_pacote_real(pkg, X, None)


@pytest.mark.parametrize("preset", ["airpls_sg_mc", "airpls_snv_sg_mc"])
@pytest.mark.parametrize("despike", [False, True])
def test_exportacao_portatil_reproduz_a_predicao_com_airpls_e_raios_cosmicos(preset, despike, tmp_path):
    pkg, eixo = _pacote(preset, despike)
    caminho = str(tmp_path / "m.json")
    save_portable_json(pkg, caminho)
    portavel = load_portable_json(caminho)                      # JSON puro, com SHA-256 conferido
    tipos = [p["tipo"] for p in portavel["passos_preprocessamento"]]
    assert tipos[0] == ("RemocaoRaiosCosmicos" if despike else "AirPLS") and "AirPLS" in tipos
    _e, Xn, _y = _raman_sintetico(n=12, seed=99, raios=True)     # espectros novos, alguns COM raio cosmico
    esperado = _oraculo(pkg, Xn)
    obtido = predict_portable(portavel, Xn, eixo)
    assert obtido["classe_pred"] == [str(c) for c in esperado["classe_pred"]]
    assert np.allclose(obtido["confianca_%"], esperado["confianca_%"], atol=1e-6)
    # o passo guarda os parametros usados no treino (lam, limiar...), nao os padroes do codigo
    airpls = next(p for p in portavel["passos_preprocessamento"] if p["tipo"] == "AirPLS")
    assert airpls["lam"] == Config().airpls_lam


def test_exportacao_usa_os_parametros_ajustados_pelo_usuario():
    cfg = Config(default_preprocessing="airpls_snv_sg_mc", apply_despike=True, airpls_lam=1000.0, despike_limiar=9.0, despike_janela=3)
    eixo, X, y_int = _raman_sintetico()
    preproc = build_preprocessor(cfg).fit(X)
    lb = LabelBinarizer().fit(y_int)
    pls = PLSRegression(n_components=2, scale=False).fit(preproc.transform(X), np.asarray(lb.transform(y_int), dtype=float))
    pkg = {"preprocessador": preproc, "pls_final": pls, "label_binarizer": lb, "classes": list(lb.classes_), "wavenumbers": eixo,
           "preset": "airpls_snv_sg_mc", "wn_min": 200.0, "wn_max": 3200.0}
    passos = {p["tipo"]: p for p in export_portable_json(pkg)["passos_preprocessamento"]}
    assert passos["AirPLS"]["lam"] == 1000.0
    assert passos["RemocaoRaiosCosmicos"]["limiar"] == 9.0 and passos["RemocaoRaiosCosmicos"]["janela"] == 3


def test_preset_custom_continua_recusado_na_exportacao():
    pkg, _eixo = _pacote("airpls_sg_mc", False)
    pkg["preset"] = "custom"
    with pytest.raises(UnsupportedModelError, match="custom"):
        export_portable_json(pkg)


@pytest.mark.slow
def test_execucao_real_com_o_preset_novo_exporta_e_o_portatil_bate_com_predict_samples(pq, tmp_path):
    import joblib

    import guaraci.predicao as pr
    from conftest import achar_pastas_run
    cfg = pq.Config(
        input_folder=str(tmp_path / "dados"), output_root_folder=str(tmp_path / "saida"), mode="sintetico",
        n_per_class=8, n_synthetic_points=60, wn_min=400.0, wn_max=4001.0, n_splits_cv=2, n_repeats_cv=1,
        n_permutations=3, n_permutations_wold=3, n_bootstrap_vip=2, n_bootstrap_bca=5, n_monte_carlo=2, max_lvs=4,
        frac_holdout=0.2, run_benchmark=False, run_monte_carlo=False, run_shap=False, run_wold=False,
        run_cv_anova=False, run_opls=False, executar_etapa4=False, show_plots=False,
        default_preprocessing="airpls_snv_sg_mc", apply_despike=True)
    os.makedirs(cfg.input_folder, exist_ok=True)
    pq.executar(cfg)
    run = achar_pastas_run(cfg.output_root_folder)[0]
    cam = os.path.join(run, pq.NOME_MODELOS, "modelo_plsda.portatil.json")
    assert os.path.exists(cam), "o preset novo tem de gerar o JSON portatil"
    pkg = joblib.load(os.path.join(run, pq.NOME_MODELOS, "modelo_plsda.joblib"))
    portavel = load_portable_json(cam)
    wn = np.asarray(pkg["wavenumbers"], dtype=float)
    rng = np.random.default_rng(7)
    X_novos = rng.normal(loc=0.5, scale=0.05, size=(6, len(wn)))
    X_novos[1, 20] += 3.0                                        # um raio cosmico
    real = pr.predict_samples(pkg, X_novos, wn)
    port = predict_portable(portavel, X_novos, wn)
    assert port["classe_pred"] == list(real["classe_pred"])
    assert np.allclose(port["confianca_%"], real["confianca_%"].values, atol=0.06)   # predict_samples arredonda a 0,1
