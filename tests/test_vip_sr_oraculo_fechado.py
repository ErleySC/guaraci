# -*- coding: utf-8 -*-
"""VIP e Selectivity Ratio contra formulas fechadas calculadas por outro caminho (auditoria A-10).

VIP: modelo falso com pesos NAO normalizados (expoe a divisao pela norma de W) e soma de VIP^2 = p.
SR: SR_j = r_j^2 / (1 - r_j^2), com r_j a correlacao entre o escore da projecao-alvo e a coluna j.
(O SR e' invariante a escala de b, entao `b / ||b||` nao e' distinguivel por teste: mutante equivalente.)
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from guaraci.chemometric_stats import compute_selectivity_ratio, vip_scores


def test_vip_formula_fechada_com_pesos_nao_normalizados():
    T = np.array([[1.0, 0.5], [-1.0, 0.5], [0.0, -1.0]])
    W = np.array([[3.0, 0.0], [4.0, 2.0], [0.0, 2.0]])          # normas 5 e 2.83: so' a normalizacao os deixa em 1
    Q = np.array([[2.0, 1.0]])
    m = SimpleNamespace(x_scores_=T, x_weights_=W, y_loadings_=Q)
    ss = (T ** 2).sum(axis=0) * (Q ** 2).sum(axis=0)             # [2*4, 1.5*1]
    esperado = np.array([np.sqrt(3 * sum(ss[a] * (W[j, a] / np.linalg.norm(W[:, a])) ** 2 for a in range(2)) / ss.sum())
                         for j in range(3)])
    np.testing.assert_allclose(vip_scores(m), esperado, rtol=1e-10)
    np.testing.assert_allclose((vip_scores(m) ** 2).sum(), 3.0, rtol=1e-9)     # soma de VIP^2 = p


@pytest.mark.parametrize("n_lv", [1, 3])
def test_vip_do_pls_real_soma_p_e_nao_e_negativo(n_lv):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 12)); y = X[:, 2] - 0.5 * X[:, 7] + 0.1 * rng.normal(size=40)
    m = PLSRegression(n_components=n_lv).fit(X, y)
    v = vip_scores(m)
    assert (v >= 0).all() and abs((v ** 2).sum() - 12) < 1e-6
    assert int(np.argmax(v)) == 2


def _sr_por_correlacao(X, b):
    Xc = X - X.mean(axis=0)
    t = Xc @ b
    r = np.array([np.corrcoef(t, Xc[:, j])[0, 1] for j in range(X.shape[1])])
    return r ** 2 / (1 - r ** 2)


@pytest.mark.parametrize("escala", [1.0, 7.5])
def test_sr_igual_a_r2_sobre_1_menos_r2(escala):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 8)) + 10.0                      # media longe de zero
    b = rng.normal(size=8)
    m = SimpleNamespace(coef_=(escala * b).reshape(1, -1))
    np.testing.assert_allclose(compute_selectivity_ratio(m, X), _sr_por_correlacao(X, b), rtol=1e-8)


def test_sr_multiclasse_e_o_maximo_entre_as_saidas_e_ignora_saida_nula():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 6))
    b1, b2 = rng.normal(size=6), rng.normal(size=6)
    m = SimpleNamespace(coef_=np.vstack([b1, b2, np.zeros(6)]))
    esperado = np.maximum(_sr_por_correlacao(X, b1), _sr_por_correlacao(X, b2))
    np.testing.assert_allclose(compute_selectivity_ratio(m, X), esperado, rtol=1e-8)


def test_sr_coef_transposto_do_sklearn_antigo_e_aceito():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(25, 5)); b = rng.normal(size=5)
    a = compute_selectivity_ratio(SimpleNamespace(coef_=b.reshape(1, -1)), X)
    np.testing.assert_allclose(compute_selectivity_ratio(SimpleNamespace(coef_=b.reshape(-1, 1)), X), a, rtol=1e-12)
