# -*- coding: utf-8 -*-
"""Contra-prova numerica de PLS-DA, VIP, SR, DD-SIMCA e OPLS contra
implementacoes de REFERENCIA EXTERNAS (mesmo padrao do oraculo do
Dual-sPLS em `test_dual_spls.py`).

Oraculos (ver cabecalho de `tests/fixtures/oraculo_referencias/gen_oracle.R`
para versoes, commits e como reproduzir):
  * mdatools 0.16.0 (R) -- PLS/SIMPLS, `plsda`, `vipscores`, `selratio`,
    `ddsimca`, `ddmoments.param`, `dd.crit`;
  * ropls 1.45.0 (R/Bioconductor) -- nucleo NIPALS PLS2 e OPLS (`.coreF`).

Os CSVs foram gerados UMA vez com R 4.3.3 e versionados; estes testes nao
precisam de R. Tolerancias: 1e-8 onde os dois lados fazem a mesma conta
fechada (so' I/O de ~15 digitos do `write.csv` separa os numeros); 1e-4
onde um dos lados e' NIPALS iterativo com criterio de parada proprio
(ropls para em ~1e-6 relativo, sklearn em 1e-6 -- medido: diferencas de
1e-7 a 2e-4 relativas, nunca estruturais).

Diferencas de CONVENCAO (nao bugs) que os testes documentam em vez de
esconder:
  * VIP: o GUARACI usa os pesos W do NIPALS (Wold; Chong & Jun 2005) e bate
    com o ropls a 1e-14. O `vipscores` do mdatools usa os pesos do SIMPLS
    R = W(P'W)^-1 e da' OUTRO numero (34% de diferenca maxima no caso
    binario abaixo) -- `test_vip_mdatools_difere_por_usar_pesos_simpls`
    prova que a diferenca e' exatamente essa troca de W por R.
  * DD-SIMCA: o mdatools arredonda Nh/Nq para inteiro por padrao
    (`do.round=TRUE`, `clamp.dof`); o GUARACI nao arredonda. O oraculo foi
    gerado com `do.round=FALSE`; o valor arredondado fica no CSV para
    referencia. O GUARACI tambem calibra q0/Nq com Q LEAVE-ONE-OUT no
    treino (decisao deliberada, ver docstring de `DDSimca`), entao a
    paridade e' testada nas pecas: T2/Q de amostras novas, formula dos
    momentos, distancia combinada e decisao com os mesmos parametros.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.stats import chi2
from sklearn.cross_decomposition import PLSRegression

from guaraci.avaliacao_modelos import PLSDAClassifier
from guaraci.chemometric_stats import (
    combined_distance,
    compute_selectivity_ratio,
    mean_and_dof_moments,
    vip_scores,
)
from guaraci.classificadores import DDSimca, OPLSDAWrapper

_FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures",
                       "oraculo_referencias")

_EXATO = dict(rtol=1e-8, atol=1e-10)      # mesma conta fechada
_ITERATIVO = dict(rtol=1e-4, atol=1e-6)   # NIPALS x NIPALS (paradas distintas)


def _ler(nome: str) -> np.ndarray:
    """CSV de `write.csv` do R (cabecalho + virgula) como array 2D."""
    return np.loadtxt(os.path.join(_FIXDIR, nome), delimiter=",",
                      skiprows=1, ndmin=2)


def _alinhar_sinal(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Escores/pesos de PLS/PCA sao definidos a menos de sinal por coluna."""
    a = np.asarray(a, dtype=float).reshape(ref.shape)
    sinais = np.sign(np.sum(a * ref, axis=0))
    sinais[sinais == 0] = 1.0
    return a * sinais


def _rotulos(Y: np.ndarray) -> np.ndarray:
    return np.array(["ABC"[k] for k in np.argmax(Y, axis=1)])


# ---------------------------------------------------------------------
# PLS-DA
# ---------------------------------------------------------------------

def test_plsda_binario_bate_com_mdatools():
    """PLSDAClassifier (PLS2 em one-hot 0/1, 3 VL) x mdatools::pls/plsda.
    Para 2 classes, Y one-hot tem posto 1 depois de centrado, entao NIPALS
    (sklearn) e SIMPLS (mdatools) coincidem exatamente."""
    X, Y = _ler("bin_X.csv"), _ler("bin_Y.csv")
    clf = PLSDAClassifier(n_components=3).fit(X, _rotulos(Y))
    y_hat = np.asarray(clf._pls.predict(X), dtype=float)

    np.testing.assert_allclose(y_hat, _ler("bin_mda_ypred.csv"), **_EXATO)
    np.testing.assert_allclose(clf._pls.coef_.T, _ler("bin_mda_coef.csv"),
                               **_EXATO)
    # mdatools::plsda codifica classe como +1/-1: predicao = 2*y_01 - 1
    np.testing.assert_allclose(2 * y_hat - 1,
                               _ler("bin_mda_plsda_ypred.csv"), **_EXATO)


def test_plsda_multiclasse_bate_com_ropls_nipals():
    """3 classes: PLS2 NIPALS do sklearn x NIPALS do ropls (`.coreF`). O
    mdatools nao serve aqui -- SIMPLS e NIPALS divergem para Y com posto>1."""
    X, Y = _ler("multi_X.csv"), _ler("multi_Y.csv")
    clf = PLSDAClassifier(n_components=3).fit(X, _rotulos(Y))
    y_hat = np.asarray(clf._pls.predict(X), dtype=float)
    y_ref = _ler("multi_ropls_ypred.csv")

    np.testing.assert_allclose(y_hat, y_ref, **_ITERATIVO)
    np.testing.assert_allclose(clf._pls.coef_.T, _ler("multi_ropls_coef.csv"),
                               **_ITERATIVO)
    # Escores individuais convergem mais devagar que a predicao no NIPALS
    # (tol=1e-6 do sklearn): medido 6.6e-4 absoluto para |t| ate' ~6.5
    # (cai a 7e-8 com tol=1e-14). Tolerancia absoluta relativa a escala.
    t_ref = _ler("multi_ropls_scores.csv")
    np.testing.assert_allclose(_alinhar_sinal(clf._pls.x_scores_, t_ref),
                               t_ref, rtol=0,
                               atol=1e-3 * np.abs(t_ref).max())
    np.testing.assert_array_equal(clf.predict(X), _rotulos(y_ref))


# ---------------------------------------------------------------------
# VIP
# ---------------------------------------------------------------------

@pytest.mark.parametrize("caso, ncomp", [("reg", 4), ("bin", 3)])
def test_vip_bate_com_ropls(caso, ncomp):
    X = _ler("bin_X.csv")
    Y = _ler("reg_y.csv").ravel() if caso == "reg" else _ler("bin_Y.csv")
    pls = PLSRegression(n_components=ncomp, scale=False).fit(X, Y)
    np.testing.assert_allclose(vip_scores(pls),
                               _ler(f"{caso}_ropls_vip.csv").ravel(), **_EXATO)


def test_vip_multiclasse_bate_com_ropls():
    X, Y = _ler("multi_X.csv"), _ler("multi_Y.csv")
    pls = PLSRegression(n_components=3, scale=False).fit(X, Y)
    np.testing.assert_allclose(vip_scores(pls),
                               _ler("multi_ropls_vip.csv").ravel(),
                               **_ITERATIVO)


@pytest.mark.parametrize("caso, ncomp", [("reg", 4), ("bin", 3)])
def test_vip_mdatools_difere_por_usar_pesos_simpls(caso, ncomp):
    """Documenta a diferenca de convencao: trocar W por R = x_rotations_
    na MESMA formula reproduz o `vipscores(type="combined")` do mdatools.
    Se um dia o GUARACI mudar de convencao, este teste e o de cima mostram
    qual das duas referencias ele passou a seguir."""
    X = _ler("bin_X.csv")
    Y = _ler("reg_y.csv").ravel() if caso == "reg" else _ler("bin_Y.csv")
    pls = PLSRegression(n_components=ncomp, scale=False).fit(X, Y)
    vip_mda = _ler(f"{caso}_mda_vip.csv").ravel()

    R = pls.x_rotations_
    ss = np.sum(pls.x_scores_ ** 2, axis=0) * np.sum(pls.y_loadings_ ** 2,
                                                       axis=0)
    Rn = R / np.linalg.norm(R, axis=0)
    vip_r = np.sqrt(R.shape[0] * (Rn ** 2 @ ss) / ss.sum())
    np.testing.assert_allclose(vip_r, vip_mda, rtol=1e-8, atol=1e-10)
    assert np.max(np.abs(vip_scores(pls) - vip_mda)) > 0.05


# ---------------------------------------------------------------------
# Selectivity Ratio
# ---------------------------------------------------------------------

@pytest.mark.parametrize("caso, ncomp", [("reg", 4), ("bin", 3)])
def test_sr_bate_com_mdatools_com_x_nao_centrado(caso, ncomp):
    """X chega com offset (~2.0), como sai de um pre-processamento sem
    centragem (preset 'custom' com apply_mc=False, ou subconjunto de treino
    em `selecao_variaveis`). O PLS do sklearn centra X por dentro, entao
    b se aplica a X CENTRADO -- a projecao-alvo tambem tem de ser feita em
    X centrado (Rajalahti et al. 2009; `selratio` do mdatools usa o X
    pre-processado do modelo, ja centrado)."""
    X = _ler("bin_X.csv")
    assert abs(X.mean()) > 1.0   # garante que o caso exercita o offset
    Y = _ler("reg_y.csv").ravel() if caso == "reg" else _ler("bin_Y.csv")
    pls = PLSRegression(n_components=ncomp, scale=False).fit(X, Y)
    sr_ref = _ler(f"{caso}_mda_sr.csv")
    # binario: as 2 colunas one-hot dao o mesmo SR (b2 = -b1); o GUARACI
    # agrega por maximo entre colunas
    np.testing.assert_allclose(compute_selectivity_ratio(pls, X),
                               sr_ref.max(axis=1), **_EXATO)


# ---------------------------------------------------------------------
# DD-SIMCA
# ---------------------------------------------------------------------

def _ddsimca_ajustado():
    Xtr = _ler("dds_Xtrain.csv")
    dd = DDSimca(n_components=2, alpha=0.05).fit(
        Xtr, np.array(["alvo"] * len(Xtr)))
    return dd, Xtr


def test_ddsimca_distancias_bate_com_mdatools():
    """h (T2) e q (Q) do mdatools para amostras NOVAS e para o treino
    (in-sample) -- mesma PCA centrada, autovalores com ddof=1."""
    dd, Xtr = _ddsimca_ajustado()
    hq_novo = _ler("dds_mda_hq_new.csv")
    T2, Q = dd._t2_q(_ler("dds_Xnew.csv"), "alvo")
    np.testing.assert_allclose(T2, hq_novo[:, 0], **_EXATO)
    np.testing.assert_allclose(Q, hq_novo[:, 1], **_EXATO)

    hq_tr = _ler("dds_mda_hq_train.csv")
    T2_tr, Q_tr = dd._t2_q(Xtr, "alvo")
    np.testing.assert_allclose(T2_tr, hq_tr[:, 0], **_EXATO)
    np.testing.assert_allclose(Q_tr, hq_tr[:, 1], **_EXATO)


def test_ddsimca_momentos_e_distancia_combinada_batem_com_mdatools():
    """`mean_and_dof_moments` x `ddmoments.param`; `combined_distance` e
    f_crit = chi2(1-alpha, Nh+Nq) x `dd.crit` (sem arredondar Nh/Nq)."""
    hq_tr = _ler("dds_mda_hq_train.csv")
    h0_r, Nh_r, q0_r, Nq_r, fcrit_r, _fcrit_arred = _ler(
        "dds_mda_params.csv").ravel()
    h0, Nh = mean_and_dof_moments(hq_tr[:, 0])
    q0, Nq = mean_and_dof_moments(hq_tr[:, 1])
    np.testing.assert_allclose([h0, Nh, q0, Nq], [h0_r, Nh_r, q0_r, Nq_r],
                               **_EXATO)
    np.testing.assert_allclose(chi2.ppf(0.95, Nh + Nq), fcrit_r, **_EXATO)

    hq_novo = _ler("dds_mda_hq_new.csv")
    f = combined_distance(hq_novo[:, 0], hq_novo[:, 1], h0, q0, Nh, Nq)
    f_ref = _ler("dds_mda_f_new.csv").ravel()
    np.testing.assert_allclose(f, f_ref, **_EXATO)
    aceita_ref = f_ref <= fcrit_r
    assert aceita_ref.any() and not aceita_ref.all()   # caso nao-trivial
    np.testing.assert_array_equal(f <= chi2.ppf(0.95, Nh + Nq), aceita_ref)


def test_ddsimca_calibracao_loo_diverge_do_mdatools_so_em_q():
    """Trava a divergencia DELIBERADA: h0/Nh iguais ao mdatools; q0/Nq
    diferentes porque o GUARACI calibra com Q leave-one-out (o mdatools
    usa Q in-sample). Se este teste quebrar, alguem mudou a calibracao."""
    dd, _ = _ddsimca_ajustado()
    m = dd._modelos["alvo"]
    h0_r, Nh_r, q0_r, Nq_r, _f, _fa = _ler("dds_mda_params.csv").ravel()
    np.testing.assert_allclose([m["h0"], m["Nh"]], [h0_r, Nh_r], **_EXATO)
    assert m["q0"] > q0_r          # Q LOO >= Q in-sample, por construcao
    assert not np.isclose(m["Nq"], Nq_r, rtol=1e-3)


# ---------------------------------------------------------------------
# OPLS-DA (Trygg & Wold 2002)
# ---------------------------------------------------------------------
#
# O oraculo do ropls foi conferido duas vezes antes de virar teste:
# (1) a predicao do OPLS 1+2 do ropls bate com a de um PLS de 3 VL do
#     mdatools a 1e-13 (equivalencia de Trygg & Wold 2002, checada abaixo);
# (2) uma implementacao direta do Algoritmo 1 de Trygg & Wold (X e y
#     centrados, sem Gram-Schmidt) reproduz t_pred e t_orth do ropls a
#     1e-13 -- ou seja, a tolerancia _ITERATIVO abaixo e' alcancavel.
#
# Hoje o `OPLSDAWrapper` NAO bate: erro maximo relativo em t_orth de 2,6x
# a 12x com X nao centrado e de 30% a 72% mesmo com X centrado. A correcao
# esta' no PR #26 (conferido: os 3 testes abaixo passam com o
# classificadores.py daquele PR). Sao `xfail(strict=True)` para que passem a
# FALHAR (XPASS) quando a correcao entrar -- sinal para remover o marcador.

_OPLS_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="OPLSDAWrapper diverge de Trygg & Wold 2002 (Gram-Schmidt extra "
           "em t_orth e deflacao em X nao centrado); correcao em andamento")


def test_oraculo_opls_ropls_equivale_a_pls_k_mais_1():
    """Sanidade do PROPRIO oraculo (nao do GUARACI)."""
    np.testing.assert_allclose(_ler("opls_ropls_ypred.csv"),
                               _ler("opls_mda_pls3_ypred.csv"), **_EXATO)


def _opls_ajustado():
    X, Y = _ler("bin_X.csv"), _ler("bin_Y.csv")
    return OPLSDAWrapper(n_ortho=2).fit(X, Y[:, 1:2]), X


@_OPLS_XFAIL
def test_opls_escores_bate_com_ropls():
    opls, _X = _opls_ajustado()
    tp_ref = _ler("opls_ropls_tpred.csv")
    to_ref = _ler("opls_ropls_torth.csv")
    assert opls.n_ortho_fitted_ == 2
    np.testing.assert_allclose(
        _alinhar_sinal(opls.t_pred_train_, tp_ref), tp_ref, **_ITERATIVO)
    np.testing.assert_allclose(
        _alinhar_sinal(opls.t_orth_train_, to_ref), to_ref, **_ITERATIVO)


@_OPLS_XFAIL
def test_opls_transform_reproduz_escores_de_treino():
    opls, X = _opls_ajustado()
    t_pred, t_orth = opls.transform(X)
    np.testing.assert_allclose(t_pred, opls.t_pred_train_, **_EXATO)
    np.testing.assert_allclose(t_orth, opls.t_orth_train_, **_EXATO)


@_OPLS_XFAIL
def test_opls_predict_bate_com_ropls():
    """Valores ajustados do OPLS 1+2 (ropls) -- iguais aos de PLS(3)."""
    opls, X = _opls_ajustado()
    y_hat = np.asarray(opls.predict(X), dtype=float).reshape(-1, 1)
    np.testing.assert_allclose(y_hat, _ler("opls_ropls_ypred.csv"),
                               **_ITERATIVO)
