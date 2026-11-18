# -*- coding: utf-8 -*-
"""Oraculo externo em R (limitacoes.md, secao 8: "oraculos externos em R").

Referencia: pacote `pls` do R (Mevik & Wehrens; NIPALS `oscorespls`), `pls::msc`, `prcomp` e `stats::qf`/
`qchisq`, gravados por `tests/oraculos_r/gen_oraculo_pls_r.R` (versoes no proprio JSON). O teste so' le o
JSON: a CI nao precisa de R. mdatools e ropls (os oraculos da versao antiga) nao puderam ser instalados no
ambiente de geracao (CRAN e Bioconductor inacessiveis); a comparacao com eles segue aberta.

Convencao registrada: o SNV do Guaraci divide pelo desvio-padrao populacional (ddof=0, numpy); o R usa
n-1. A razao entre os dois e' exatamente sqrt((p-1)/p), conferida aqui."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from guaraci.chemometric_stats import (hotelling_t2, hotelling_t2_limit, q_residuals, q_residuals_limit,
                                       vip_scores)
from guaraci.preprocessamento import MSC, SNV

ORACULO = json.loads((Path(__file__).parent / "oraculos_r" / "oraculo_pls_r.json").read_text(encoding="utf-8"))
CASOS = ORACULO["casos"]


def _arr(c, k):
    return np.asarray(c[k], dtype=float)


@pytest.mark.parametrize("c", CASOS, ids=[c["id"] for c in CASOS])
def test_snv_e_msc_iguais_ao_r(c):
    X = _arr(c, "X")
    p = X.shape[1]
    snv = SNV().fit(X).transform(X)
    np.testing.assert_allclose(snv * np.sqrt((p - 1) / p), _arr(c, "snv_sd_n1"), atol=1e-12)
    np.testing.assert_allclose(MSC().fit(X).transform(X), _arr(c, "msc"), atol=1e-12)


@pytest.mark.parametrize("c", CASOS, ids=[c["id"] for c in CASOS])
def test_pls1_coeficientes_predicao_e_vip_iguais_ao_pacote_pls(c):
    X, y, a = _arr(c, "X"), _arr(c, "y"), c["ncomp"]
    m = PLSRegression(n_components=a, scale=False).fit(X, y)
    b = _arr(c, "pls_coef")
    np.testing.assert_allclose(m.coef_.ravel(), b, atol=1e-10 * np.abs(b).max())
    np.testing.assert_allclose(m.predict(X).ravel(), _arr(c, "pls_yhat"), atol=1e-10)
    np.testing.assert_allclose(vip_scores(m), _arr(c, "vip").ravel(), atol=1e-10)


@pytest.mark.parametrize("c", CASOS, ids=[c["id"] for c in CASOS])
def test_t2_q_e_limites_iguais_ao_r(c):
    X, a = _arr(c, "X"), c["ncomp"]
    pca = PCA(n_components=a).fit(X)
    T = pca.transform(X)
    np.testing.assert_allclose(hotelling_t2(T), _arr(c, "pca_T2"), rtol=1e-10)
    Q = q_residuals(X - X.mean(axis=0), T, pca.components_)
    np.testing.assert_allclose(Q, _arr(c, "pca_Q"), rtol=1e-9)
    assert hotelling_t2_limit(X.shape[0], a) == pytest.approx(c["t2_lim_95"], rel=1e-12)
    assert q_residuals_limit(_arr(c, "pca_Q")) == pytest.approx(c["q_lim_95"], rel=1e-12)


def test_versoes_registradas():
    assert set(ORACULO["versoes"]) == {"R", "pls"} and len(CASOS) == 3
