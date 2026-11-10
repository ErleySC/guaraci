# -*- coding: utf-8 -*-
"""DS/PDS contra um oraculo independente: `sklearn.linear_model.Ridge` (centra X e y e nao penaliza o intercepto).

Fecha a brecha da mutacao (auditoria A-10): trocas de sinal na centragem e no termo de regularizacao sobreviviam
porque os testes antigos so' checavam propriedades gerais (forma, recuperacao com alpha pequeno).
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from guaraci.transferencia_calibracao import (
    apply_standardization,
    direct_standardization,
    piecewise_direct_standardization,
)


def _dados(n=12, p=9, seed=3):
    rng = np.random.default_rng(seed)
    Xm = rng.normal(size=(n, p)).cumsum(axis=1) * 0.3 + 5.0          # media longe de zero: expoe erro de centragem
    Xs = 1.1 * Xm + 0.4 + 0.05 * rng.normal(size=(n, p)) + 0.2 * np.roll(Xm, 1, axis=1)
    return Xm, Xs


@pytest.mark.parametrize("alpha", [0.0, 0.05, 1.0, 40.0])
def test_ds_igual_ao_ridge_do_sklearn(alpha):
    Xm, Xs = _dados()
    t = direct_standardization(Xm, Xs, alpha=alpha)
    for j in range(Xm.shape[1]):
        r = Ridge(alpha=alpha, solver="cholesky").fit(Xs, Xm[:, j]) if alpha > 0 else None
        if r is None:
            continue
        np.testing.assert_allclose(t.F[:, j], r.coef_, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(t.bias[j], r.intercept_, rtol=1e-8, atol=1e-9)
    novo = Xs + 0.1
    ref = np.column_stack([Ridge(alpha=alpha, solver="cholesky").fit(Xs, Xm[:, j]).predict(novo)
                          for j in range(Xm.shape[1])]) if alpha > 0 else None
    if ref is not None:
        np.testing.assert_allclose(apply_standardization(novo, t), ref, rtol=1e-8, atol=1e-9)


def test_ds_alpha_zero_interpola_o_mestre_quando_ha_mais_canais_que_amostras():
    Xm, Xs = _dados(n=5, p=9)
    t = direct_standardization(Xm, Xs, alpha=1e-9)
    # n < p e alpha ~ 0: o modelo reproduz as amostras de transferencia
    np.testing.assert_allclose(apply_standardization(Xs, t), Xm, atol=1e-4)


@pytest.mark.parametrize("janela", [0, 1, 2, 4, 20])
@pytest.mark.parametrize("alpha", [0.05, 2.0])
def test_pds_igual_ao_ridge_local_do_sklearn(janela, alpha):
    Xm, Xs = _dados()
    p = Xm.shape[1]
    t = piecewise_direct_standardization(Xm, Xs, janela=janela, alpha=alpha)
    F = np.zeros((p, p))
    bias = np.zeros(p)
    for j in range(p):
        lo, hi = max(0, j - janela), min(p, j + janela + 1)
        r = Ridge(alpha=alpha, solver="cholesky").fit(Xs[:, lo:hi], Xm[:, j])
        F[lo:hi, j] = r.coef_
        bias[j] = r.intercept_
    np.testing.assert_allclose(t.F, F, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(t.bias, bias, rtol=1e-8, atol=1e-9)
    assert (t.F[np.triu_indices(p, janela + 1)] == 0).all()           # fora da janela, zero


def test_a_regularizacao_encolhe_os_coeficientes():
    Xm, Xs = _dados()
    normas = [np.linalg.norm(direct_standardization(Xm, Xs, alpha=a).F) for a in (0.01, 1.0, 100.0, 1e4)]
    assert normas == sorted(normas, reverse=True) and normas[-1] < 0.01 * normas[0]
