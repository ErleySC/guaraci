# -*- coding: utf-8 -*-
"""Remocao de raios cosmicos (Whitaker & Hayes 2018, z-score modificado da
primeira diferenca): `preprocessamento.RemocaoRaiosCosmicos` e a opcao
`remover_raios_cosmicos` da Config."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from guaraci.config import Config
from guaraci.preprocessamento import (RemocaoRaiosCosmicos, SavGol, build_preprocessor,
                                      descrever_preprocessamento)
from guaraci.simulador_espectral import cenario, gerar_cenario


def _espectro_limpo(seed: int = 0, n: int = 800) -> np.ndarray:
    """Raman sintetico: fundo largo + bandas lorentzianas de 3 a 8 pontos de
    meia largura + ruido."""
    rng = np.random.default_rng(seed)
    i = np.arange(n, dtype=float)
    x = 0.5 + 0.4 * np.exp(-((i - n / 3) / (n / 2)) ** 2)
    for c, w, h in zip(rng.uniform(40, n - 40, 12), rng.uniform(3, 8, 12), rng.uniform(0.3, 1.0, 12)):
        x += h / (1.0 + ((i - c) / w) ** 2)
    return x + rng.normal(0, 0.005, n)


@pytest.mark.parametrize("largura", [1, 2, 3])
def test_spike_e_removido(largura):
    x = _espectro_limpo()
    sujo = x.copy()
    sujo[300:300 + largura] += 2.0
    sujo[611] += 4.0
    r = RemocaoRaiosCosmicos()
    marcados = r.mascara(sujo)
    assert marcados[300:300 + largura].all() and marcados[611]
    assert marcados.sum() == largura + 1          # nada alem dos spikes
    out = r.fit_transform(sujo[None, :])[0]
    assert np.abs(out - x).max() < 0.05           # spike de 2 a 4 sai inteiro


def test_espectro_limpo_sai_igual_e_sem_nan():
    X = np.vstack([_espectro_limpo(s) for s in range(10)])
    out = RemocaoRaiosCosmicos().fit_transform(X)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, X, atol=1e-12)


def test_bordas_constante_e_curto_nao_quebram():
    r = RemocaoRaiosCosmicos()
    borda = _espectro_limpo()
    borda[0] += 3.0
    borda[-1] += 3.0
    assert np.isfinite(r.transform(borda[None, :])).all()
    np.testing.assert_array_equal(r.transform(np.ones((2, 50))), np.ones((2, 50)))
    np.testing.assert_array_equal(r.transform(np.array([[1.0, 5.0]])), [[1.0, 5.0]])


def test_por_espectro_e_sem_estado():
    """Cada linha e' tratada sozinha: o resultado de uma linha nao depende das
    outras (nem do que foi visto no fit) -- nao ha vazamento entre dobras."""
    X = np.vstack([_espectro_limpo(s) for s in range(4)])
    X[2, 400:402] += 3.0
    r = RemocaoRaiosCosmicos().fit(X[:1])
    np.testing.assert_array_equal(r.transform(X)[2], RemocaoRaiosCosmicos().fit_transform(X[2:3])[0])


def test_config_desligada_por_padrao_e_ligada_vira_primeira_etapa():
    cfg = Config()
    assert cfg.apply_despike is False
    assert "despike" not in [n for n, _ in build_preprocessor(cfg).steps]
    cfg.apply_despike = True
    cfg.despike_limiar = 8.0
    cfg.despike_janela = 3
    passos = build_preprocessor(cfg).steps
    assert passos[0][0] == "despike" and passos[0][1].limiar == 8.0 and passos[0][1].janela == 3
    for preset in ("airpls_sg_mc", "custom", "autoscaling"):
        cfg.default_preprocessing = preset
        assert build_preprocessor(cfg).steps[0][0] == "despike"
    curtos, descr, _ = descrever_preprocessamento(cfg)
    assert curtos[0] == "Despike" and "Whitaker-Hayes" in descr


def test_chaves_do_config_yaml():
    from guaraci.config_io import _CONFIG_SPEC
    chaves = {s["key"]: s["attr"] for s in _CONFIG_SPEC}
    assert chaves["remover_raios_cosmicos"] == "apply_despike"
    assert chaves["raios_cosmicos_limiar"] == "despike_limiar"
    assert chaves["raios_cosmicos_janela"] == "despike_janela"


def _r2cv(X, c, grupos, despike: bool) -> float:
    pred = np.zeros_like(c)
    for tr, va in GroupKFold(5).split(X, c, grupos):
        etapas = ([RemocaoRaiosCosmicos()] if despike else []) + [
            SavGol(25, 2, 1), StandardScaler(with_std=False), PLSRegression(6, scale=False)]
        pred[va] = make_pipeline(*etapas).fit(X[tr], c[tr]).predict(X[va]).ravel()
    return float(1 - np.sum((pred - c) ** 2) / np.sum((c - c.mean()) ** 2))


@pytest.mark.parametrize("seed, off_max, on_min", [(42, 0.82, 0.88), (1, 0.75, 0.90)])
def test_cenario_raman_despike_recupera_o_r2cv(seed, off_max, on_min):
    """Cenario `raman` do modo sintetico (5 % dos espectros com raio cosmico,
    1000 pontos, 10 amostras x 3 replicas por classe). R2cv do teor em CV
    agrupada por amostra fisica, medido em 2026-10-16: semente 42,
    0,798 -> 0,906; semente 1, 0,727 -> 0,939."""
    _, X, _, conc, mae, _ = gerar_cenario(cenario("raman"), n_pontos=1000, n_por_classe=10,
                                          n_replicas=3, seed=seed)
    off = _r2cv(X, conc, mae, despike=False)
    on = _r2cv(X, conc, mae, despike=True)
    assert off < off_max and on > on_min and on - off > 0.08, (off, on)
