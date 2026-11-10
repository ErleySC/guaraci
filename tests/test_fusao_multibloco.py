# -*- coding: utf-8 -*-
"""Testes de guaraci.fusao_multibloco -- fechamento do Grupo 2 (prova de
conceito de fusao multibloco, ver docs/projeto/escopo_fusao_multibloco_e_mspc.md).

Duas camadas: (1) testes SINTETICOS e rapidos da mecanica pura
(`build_multiblock_dataset`/`fit_evaluate_pls_regression`) -- guarda
contra desalinhamento, ausencia de vazamento cal->val, formato do
resultado.

Limite conhecido: fusao de BAIXO NIVEL (concatenacao apos
pre-processamento separado, sem block-scaling) nao garante ganho sobre o
melhor bloco sozinho -- um bloco com muito mais variaveis e sinal mais
fraco pode dominar a decomposicao PLS por simples contagem de variaveis.
Block-scaling NAO foi adotado como feature do modulo (escopo "nivel 1,
nenhum algoritmo novo").
"""
from __future__ import annotations


import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from guaraci.config import Config
from guaraci.fusao_multibloco import (
    BlockMismatchError,
    build_multiblock_dataset,
    fit_evaluate_pls_regression,
)

# ── Mecanica sintetica (rapida, sem dataset externo) ────────────────────


def test_build_multiblock_dataset_recusa_blocos_desalinhados():
    rng = np.random.default_rng(0)
    bloco_a = rng.normal(size=(20, 5))
    bloco_b = rng.normal(size=(19, 4))   # 1 amostra a menos -- deliberado
    with pytest.raises(BlockMismatchError, match="n_amostras diferente"):
        build_multiblock_dataset({"a": bloco_a, "b": bloco_b})


def test_build_multiblock_dataset_concatena_na_ordem_e_reporta_fatias():
    rng = np.random.default_rng(1)
    bloco_a = rng.normal(size=(30, 6))
    bloco_b = rng.normal(size=(30, 4))
    # default_preprocessing="mc" (so' mean-centering) -- o preset default
    # (msc_sg_mc) exige sg_window (25) <= n_variaveis, que estes blocos
    # sinteticos pequenos (6/4 colunas) nao satisfazem; irrelevante para
    # o que este teste verifica (concatenacao/fatias, nao o preset).
    cfg = {"a": Config(default_preprocessing="mc"),
           "b": Config(default_preprocessing="mc")}
    X_fundida, preps, fatias = build_multiblock_dataset(
        {"a": bloco_a, "b": bloco_b}, cfg)
    assert X_fundida.shape == (30, 10)
    assert fatias["a"] == (0, 6)
    assert fatias["b"] == (6, 10)
    assert set(preps.keys()) == {"a", "b"}


def test_build_multiblock_dataset_ajusta_preprocessador_so_no_indices_ajuste():
    """MSC/mean-centering sao stateful (media de referencia vem do fit) --
    ajustar so' com `indices_ajuste` (calibracao) tem que dar um resultado
    DIFERENTE de ajustar com todas as amostras, quando cal e val tem
    medias bem diferentes -- prova de que nao ha' vazamento cal->val."""
    rng = np.random.default_rng(2)
    bloco = np.vstack([
        rng.normal(loc=0.0, scale=0.1, size=(20, 8)),    # "calibracao"
        rng.normal(loc=50.0, scale=0.1, size=(10, 8)),   # "validacao" (media MUITO diferente)
    ])
    idx_cal = np.arange(20)

    cfg = {"x": Config(default_preprocessing="mc")}
    _, preps_cal, _ = build_multiblock_dataset(
        {"x": bloco}, cfg, indices_ajuste=idx_cal)
    _, preps_tudo, _ = build_multiblock_dataset({"x": bloco}, cfg)

    media_cal = preps_cal["x"].named_steps["mc"].mean_
    media_tudo = preps_tudo["x"].named_steps["mc"].mean_
    assert not np.allclose(media_cal, media_tudo), (
        "ajustar com indices_ajuste=calibracao deveria dar uma media "
        "diferente de ajustar com o dataset inteiro -- se bater, o "
        "pre-processador esta' vazando informacao de validacao")


def test_fit_evaluate_pls_regression_recupera_sinal_linear_simples():
    """Contra-prova minima: com y linear numa unica variavel latente
    (sinal facil, sem ruido de dominio real), a regressao tem que
    recuperar R2 alto e RMSEP baixo -- se isto falhar, a mecanica de
    ajuste/avaliacao esta' quebrada, independente do dataset real."""
    rng = np.random.default_rng(3)
    n = 200
    latente = rng.normal(size=n)
    X = latente[:, None] * rng.normal(loc=1.0, scale=0.05, size=(1, 20)) \
        + rng.normal(scale=0.01, size=(n, 20))
    y = 3.0 * latente + rng.normal(scale=0.05, size=n)

    idx_cal, idx_val = train_test_split(
        np.arange(n), test_size=0.25, random_state=0)
    r = fit_evaluate_pls_regression(
        X[idx_cal], y[idx_cal], X[idx_val], y[idx_val], max_lvs=5, seed=0)
    assert r.n_cal == 150 and r.n_val == 50
    assert r.r2val > 0.9, f"R2val={r.r2val:.3f} baixo demais p/ sinal facil"
    assert r.rmsep < 0.5


def _oraculo_pls(X_cal, y_cal, X_val, y_val, max_lvs, n_splits_cv, seed):
    """Reimplementacao independente (KFold manual, sem cross_val_predict)."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    yc = np.asarray(y_cal, float).ravel()
    lv_max = min(max_lvs, max(2, X_cal.shape[0] // 5))
    kf = KFold(n_splits=n_splits_cv, shuffle=True, random_state=seed)
    rm = []
    for n_lv in range(1, lv_max + 1):
        pred = np.zeros(len(yc))
        for tr, va in kf.split(X_cal):
            m = PLSRegression(n_components=n_lv, scale=False).fit(X_cal[tr], yc[tr])
            pred[va] = m.predict(X_cal[va]).ravel()
        rm.append(float(np.sqrt(np.mean((yc - pred) ** 2))))
    n_opt = int(np.argmin(rm)) + 1
    m = PLSRegression(n_components=n_opt, scale=False).fit(X_cal, yc)
    yv = np.asarray(y_val, float).ravel()
    pv = m.predict(X_val).ravel()
    pc = m.predict(X_cal).ravel()

    def r2(a, b):
        return 1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)
    return n_opt, float(np.sqrt(np.mean((yv - pv) ** 2))), r2(yc, pc), r2(yv, pv)


def _dados_pls(n_cal, n_val, p=12, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(4, p))

    def gera(n):
        T = rng.normal(size=(n, 4))
        X = T @ W + rng.normal(scale=0.3, size=(n, p))
        y = T @ np.array([2.0, -1.0, 0.5, 0.2]) + rng.normal(scale=0.3, size=n)
        return X, y
    return (*gera(n_cal), *gera(n_val))


@pytest.mark.parametrize("n_cal,max_lvs,n_splits_cv,seed", [
    (60, 15, 5, 0),     # lv_max limitado por n_cal // 5 = 12
    (60, 3, 5, 1),      # lv_max limitado por max_lvs
    (60, 15, 3, 2),     # n_splits_cv diferente do default
    (8, 15, 2, 0),      # n_cal // 5 = 1 -> piso de 2 LVs
])
def test_fit_evaluate_pls_regression_bate_com_oraculo_independente(
        n_cal, max_lvs, n_splits_cv, seed):
    """Achado da mutacao (Passo 219): 44/110 mutantes sobreviviam porque so'
    um teste com sinal facil cobria a funcao. Aqui TODAS as saidas (n_lv,
    RMSEP, R2cal, R2val, contagens) sao comparadas com uma reimplementacao
    independente, e a formula de `lv_max` e' exercitada nos 3 regimes."""
    Xc, yc, Xv, yv = _dados_pls(n_cal, 20)
    r = fit_evaluate_pls_regression(Xc, yc, Xv, yv, max_lvs=max_lvs,
                                    n_splits_cv=n_splits_cv, seed=seed)
    n_opt, rmsep, r2c, r2v = _oraculo_pls(Xc, yc, Xv, yv, max_lvs, n_splits_cv, seed)
    assert r.n_lv == n_opt
    assert r.rmsep == pytest.approx(rmsep)
    assert r.r2cal == pytest.approx(r2c)
    assert r.r2val == pytest.approx(r2v)
    assert (r.n_cal, r.n_val) == (n_cal, 20)
    assert 1 <= r.n_lv <= min(max_lvs, max(2, n_cal // 5))


def test_fit_evaluate_pls_regression_defaults_documentados():
    import inspect
    p = inspect.signature(fit_evaluate_pls_regression).parameters
    assert (p["max_lvs"].default, p["n_splits_cv"].default, p["seed"].default) == (15, 5, 0)


def test_seed_do_cv_chega_ao_kfold():
    """`random_state=seed` chega ao KFold -- confere pelo oraculo em 2 seeds."""
    Xc, yc, Xv, yv = _dados_pls(40, 15, seed=5)
    for s in (0, 7):
        r = fit_evaluate_pls_regression(Xc, yc, Xv, yv, max_lvs=8, seed=s)
        assert r.n_lv == _oraculo_pls(Xc, yc, Xv, yv, 8, 5, s)[0]


def test_lv_max_e_um_quinto_das_amostras_e_y_1d_de_tamanho_impar():
    """Sinal de posto 20 sem ruido: o RMSECV cai ate' o limite de LVs, entao
    o n_lv escolhido == lv_max == n_cal // 5 (61 // 5 = 12, nao 15 nem 10).
    Tamanhos IMPARES (61 cal / 21 val) com y 1-D: o reshape(-1, 1) nao pode
    virar outra forma."""
    rng = np.random.default_rng(0)
    n, nv, p = 61, 21, 40
    T = rng.normal(size=(n + nv, 20))
    X = T @ rng.normal(size=(20, p))
    y = T @ rng.normal(size=20)
    r = fit_evaluate_pls_regression(X[:n], y[:n], X[n:], y[n:], max_lvs=15)
    assert r.n_lv == 12
    assert (r.n_cal, r.n_val) == (61, 21)


@pytest.mark.parametrize("posto", [2, 4])
def test_piso_de_2_lvs_com_poucas_amostras(posto):
    """n_cal=8 => n_cal // 5 = 1, mas o piso e' 2: com sinal de posto 2 ou 4
    (sem ruido) o modelo escolhido tem SEMPRE 2 LVs (nem 1, nem 3)."""
    rng = np.random.default_rng(0)
    n, nv, p = 8, 6, 10
    T = rng.normal(size=(n + nv, posto))
    X = T @ rng.normal(size=(posto, p))
    y = T @ rng.normal(size=posto)
    r = fit_evaluate_pls_regression(X[:n], y[:n], X[n:], y[n:], max_lvs=15,
                                    n_splits_cv=4)
    assert r.n_lv == 2
