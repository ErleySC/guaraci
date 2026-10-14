# -*- coding: utf-8 -*-
"""Contra-prova de PLS-DA, VIP, SR, DD-SIMCA e OPLS-DA em DADOS REAIS
PUBLICADOS, contra mdatools 0.16.0 e ropls 1.45.0 (R). Complementa
`test_oraculo_referencias.py` (dados sinteticos): la' se prova a conta
num caso controlado; aqui, que ela continua batendo em dados de verdade
(colinearidade real, escalas heterogeneas, classes desbalanceadas).

Datasets (nenhum versionado -- `datasets/README.md`; so' as SAIDAS dos
modelos de referencia ficam em `tests/fixtures/oraculo_dados_reais/`,
junto com o `gen_oracle.R` que as gerou):

  * sacurine -- Thevenot et al. (2015), J. Proteome Res. 14(8):3322-3335:
    urina de 183 adultos, LC-HRMS, 109 metabolitos (log10), classe =
    sexo (F/M). E' o dataset de exemplo do proprio ropls para OPLS-DA.
    Baixar com `scripts/download_datasets/baixar_ropls_sacurine.py`;
    sem `$GUARACI_DATASETS_DIR/ropls_sacurine/`, esses testes PULAM.
  * iris -- Fisher (1936), Ann. Eugenics 7:179-188, embutido no
    scikit-learn (identico ao `datasets::iris` do R, conferido valor a
    valor). Roda sempre.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_iris

from guaraci.avaliacao_modelos import PLSDAClassifier
from guaraci.chemometric_stats import (
    combined_distance,
    compute_selectivity_ratio,
    mean_and_dof_moments,
    vip_scores,
)
from guaraci.classificadores import DDSimca, OPLSDAWrapper

_FIXDIR = Path(__file__).parent / "fixtures" / "oraculo_dados_reais"

_EXATO = dict(rtol=1e-8, atol=1e-10)
_ITERATIVO = dict(rtol=1e-4, atol=1e-6)


def _ler(nome: str) -> np.ndarray:
    return np.loadtxt(_FIXDIR / nome, delimiter=",", skiprows=1, ndmin=2)


def _alinhar_sinal(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(ref.shape)
    sinais = np.sign(np.sum(a * ref, axis=0))
    sinais[sinais == 0] = 1.0
    return a * sinais


# ---------------------------------------------------------------- sacurine

def _pasta_sacurine() -> "Path | None":
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "ropls_sacurine"
    return pasta if (pasta / "sacurine_dataMatrix.tsv").is_file() else None


requer_sacurine = pytest.mark.skipif(
    _pasta_sacurine() is None,
    reason=("sacurine (ropls) ausente. Baixe com 'python scripts/"
            "download_datasets/baixar_ropls_sacurine.py' e aponte "
            "GUARACI_DATASETS_DIR para a pasta que contem ropls_sacurine/."))


def _sacurine():
    """X (183x109, log10) e sexo, na ordem de amostras do sampleMetadata
    (a mesma que o gen_oracle.R confere com stopifnot)."""
    import pandas as pd

    pasta = _pasta_sacurine()
    dm = pd.read_csv(pasta / "sacurine_dataMatrix.tsv", sep="\t", index_col=0)
    sm = pd.read_csv(pasta / "sacurine_sampleMetadata.tsv", sep="\t",
                     index_col=0)
    X = dm.T.loc[sm.index].to_numpy(dtype=float)
    assert X.shape == (183, 109)
    return X, sm["gender"].to_numpy(dtype=str)


def _autoescalar(X: np.ndarray) -> np.ndarray:
    """Mesmo `scale()` do R: centra e divide pelo desvio com ddof=1."""
    return (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)


@requer_sacurine
def test_sacurine_plsda_vip_sr_batem_com_referencias():
    X, sexo = _sacurine()
    Xa = _autoescalar(X)
    clf = PLSDAClassifier(n_components=2).fit(Xa, sexo)
    assert list(clf.classes_) == ["F", "M"]
    pls = clf._pls

    np.testing.assert_allclose(pls.predict(Xa), _ler("sac_mda_ypred.csv"),
                               **_EXATO)
    np.testing.assert_allclose(pls.coef_.T, _ler("sac_mda_coef.csv"),
                               **_EXATO)
    np.testing.assert_allclose(vip_scores(pls),
                               _ler("sac_ropls_vip.csv").ravel(), **_EXATO)
    np.testing.assert_allclose(compute_selectivity_ratio(pls, Xa),
                               _ler("sac_mda_sr.csv").max(axis=1), **_EXATO)


@requer_sacurine
def test_sacurine_oplsda_bate_com_ropls():
    """Mesmo modelo da vinheta do ropls: autoescalado, 1 componente
    preditivo, numero de ortogonais escolhido pelo ropls (2)."""
    X, sexo = _sacurine()
    tp_ref = _ler("sac_opls_tpred.csv")
    to_ref = _ler("sac_opls_torth.csv")
    assert to_ref.shape[1] == 2
    opls = OPLSDAWrapper(n_ortho=2).fit(
        _autoescalar(X), (sexo == "M").astype(float).reshape(-1, 1))
    np.testing.assert_allclose(
        _alinhar_sinal(opls.t_pred_train_, tp_ref), tp_ref, **_ITERATIVO)
    np.testing.assert_allclose(
        _alinhar_sinal(opls.t_orth_train_, to_ref), to_ref, **_ITERATIVO)
    np.testing.assert_allclose(
        np.asarray(opls.predict(_autoescalar(X)), dtype=float).reshape(-1, 1),
        _ler("sac_opls_ypred.csv"), **_ITERATIVO)


# -------------------------------------------------------------------- iris

def _iris():
    d = load_iris()
    return d.data.astype(float), d.target_names[d.target]


def test_iris_plsda_multiclasse_e_vip_batem_com_ropls():
    X, especie = _iris()
    clf = PLSDAClassifier(n_components=3).fit(X, especie)
    y_ref = _ler("iris_ropls_ypred.csv")
    np.testing.assert_allclose(clf._pls.predict(X), y_ref, **_ITERATIVO)
    np.testing.assert_array_equal(clf.predict(X),
                                  clf.classes_[np.argmax(y_ref, axis=1)])
    pls = PLSRegression(n_components=3, scale=False).fit(
        X, np.eye(3)[np.searchsorted(clf.classes_, especie)])
    np.testing.assert_allclose(vip_scores(pls),
                               _ler("iris_ropls_vip.csv").ravel(),
                               **_ITERATIVO)


def test_iris_ddsimca_versicolor_bate_com_mdatools():
    """Treino: versicolor 51-90 (indices R, 1-based); novos: versicolor
    91-100 + virginica 101-150. Distancias, momentos, f e a decisao
    aceita/rejeita batem com o mdatools quando calibrados com o mesmo Q
    in-sample (o GUARACI calibra com Q LOO de proposito -- ver
    `test_oraculo_referencias.py`)."""
    X, _ = _iris()
    tr = np.arange(50, 90)
    nv = np.r_[np.arange(90, 100), np.arange(100, 150)]
    dd = DDSimca(n_components=2, alpha=0.05).fit(
        X[tr], np.array(["versicolor"] * len(tr)))

    T2, Q = dd._t2_q(X[nv], "versicolor")
    hq_nv = _ler("iris_dds_hq_new.csv")
    np.testing.assert_allclose(T2, hq_nv[:, 0], **_EXATO)
    np.testing.assert_allclose(Q, hq_nv[:, 1], **_EXATO)

    T2_tr, Q_tr = dd._t2_q(X[tr], "versicolor")
    hq_tr = _ler("iris_dds_hq_train.csv")
    np.testing.assert_allclose(T2_tr, hq_tr[:, 0], **_EXATO)
    np.testing.assert_allclose(Q_tr, hq_tr[:, 1], **_EXATO)

    h0, Nh = mean_and_dof_moments(T2_tr)
    q0, Nq = mean_and_dof_moments(Q_tr)
    h0_r, Nh_r, q0_r, Nq_r, fcrit_r = _ler("iris_dds_params.csv").ravel()
    np.testing.assert_allclose([h0, Nh, q0, Nq], [h0_r, Nh_r, q0_r, Nq_r],
                               **_EXATO)
    fcrit = chi2.ppf(0.95, Nh + Nq)
    np.testing.assert_allclose(fcrit, fcrit_r, **_EXATO)

    f = combined_distance(T2, Q, h0, q0, Nh, Nq)
    f_ref = _ler("iris_dds_f_new.csv").ravel()
    np.testing.assert_allclose(f, f_ref, **_EXATO)
    aceita = f <= fcrit
    np.testing.assert_array_equal(aceita, f_ref <= fcrit_r)
    assert aceita[:10].any() and not aceita[10:].all()   # caso nao-trivial
