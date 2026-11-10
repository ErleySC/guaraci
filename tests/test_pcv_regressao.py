"""PCV para PLS (guaraci.pcv_regressao): comparacao com o pacote R `pcv` 1.2.0 e verdade conhecida simulada.

Os fixtures vem de `tests/oraculos_metodos/gen_oraculo_pcv.R`. A parte ortogonal do pseudo-conjunto e' aleatoria
nas duas implementacoes e por isso so' e' conferida pela definicao (norma e ortogonalidade), nao por valores.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from guaraci import pcv_regressao as P

_ORACULO = json.loads((Path(__file__).parent / "oraculos_metodos" / "oraculo_pcv.json").read_text(encoding="utf-8"))
_CASOS = {c["id"]: c for c in _ORACULO["casos"]}


def _cv(c):
    cv = c["cv"]
    return tuple(cv) if isinstance(cv[0], str) else np.array(cv)


def _rodar(c):
    X, Y = np.array(c["X"]), np.array(c["Y"])
    r = P.pcv_pls(X, Y, c["ncomp_in"], cv=_cv(c), center=c["center"], scale=c["scale"], cv_scope=c["scope"])
    return X, Y, r


def test_fixture_tem_versoes():
    assert _ORACULO["versoes"]["pcv"] == "1.2.0" and len(_CASOS) == 7
    assert all(c["dif_semente"] < 1e-8 for c in _CASOS.values())    # o R afirmou: RMSEP-PV nao depende da parte ortogonal


@pytest.mark.parametrize("cid", sorted(_CASOS))
def test_igual_ao_pacote_pcv(cid):
    c = _CASOS[cid]
    X, Y, r = _rodar(c)
    tol = 1e-5 if cid == "P6_ncomp_truncado" else 1e-8      # P6 usa ncomp = p: mal condicionado
    assert r.ncomp == c["ncomp"]
    np.testing.assert_array_equal(r.segmentos, c["ind"])
    np.testing.assert_allclose(r.Xpv_explicada, c["Xhat"], atol=tol)
    np.testing.assert_allclose(r.D, c["D"], atol=tol)
    np.testing.assert_allclose(r.q, c["q"], atol=1e-9)
    # R e R_k so' a menos do sinal global de cada componente
    s = np.sign(np.sum(r.R * np.array(c["R"]), axis=0))
    np.testing.assert_allclose(r.R * s, c["R"], atol=1e-8)
    np.testing.assert_allclose(r.Rk * s[None, None, :], c["Rk"], atol=1e-7)
    ref = np.array(c["rmsep_pv"])
    ref = ref if ref.shape == (r.ncomp, Y.shape[1]) else ref.T
    np.testing.assert_allclose(P.rmsep_pv(r, Y), ref, atol=tol * 10)


@pytest.mark.parametrize("cid", ["P1_ven5_n30_p12", "P3_ven4_local_scale_pls2"])
def test_parte_ortogonal_pela_definicao(cid):
    c = _CASOS[cid]
    X, Y, r = _rodar(c)
    # sem escala local/padronizacao a norma^2 do residuo ortogonal e' q_k; confere no caso global sem escala
    if cid.startswith("P1"):
        orth = r.Xpv - r.Xpv_explicada
        np.testing.assert_allclose((orth ** 2).sum(axis=1), r.q, rtol=1e-8)
        np.testing.assert_allclose(orth @ r.R, 0.0, atol=1e-8)       # ortogonal ao espaco dos pesos globais
    # o RMSEP-PV nao depende da semente (a parte ortogonal muda, as predicoes nao)
    r2 = P.pcv_pls(X, Y, c["ncomp_in"], cv=_cv(c), center=c["center"], scale=c["scale"], cv_scope=c["scope"], seed=99)
    assert not np.allclose(r.Xpv, r2.Xpv)
    np.testing.assert_allclose(P.rmsep_pv(r, Y), P.rmsep_pv(r2, Y), atol=1e-8)


# ------------------------------------------------------------ verdade conhecida (simulacao)
def _gerar(rng, n, Pm, B, p=30, ruido=0.5):
    T = rng.normal(size=(n, 3)) * np.array([4.0, 3.0, 2.0])
    return T @ Pm.T + rng.normal(size=(n, p)) * 0.3, (T @ B + rng.normal(size=n) * ruido)[:, None]


def test_recupera_numero_de_componentes_em_simulacao():
    """3 fatores verdadeiros, n = 40, teste independente de 5000. Mede UM cenario, nao garante nada em geral:
    nas mesmas 20 sementes a validacao cruzada comum em 5 blocos fica no mesmo nivel (erro log medio 0.17 contra
    0.16), isto e', nao ha' vantagem demonstrada sobre ela aqui."""
    dentro, erros = 0, []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        Pm = np.linalg.qr(rng.normal(size=(30, 3)))[0]
        B = rng.normal(size=3)
        X, Y = _gerar(rng, 40, Pm, B)
        Xt, Yt = _gerar(rng, 5000, Pm, B)
        r = P.pcv_pls(X, Y, 8, cv=("ven", 5), seed=seed)
        pv = P.rmsep_pv(r, Y)[:, 0]
        te = P.rmsep_pv(r, Yt, Xpv=Xt)[:, 0]
        dentro += abs(int(np.argmin(pv)) - int(np.argmin(te))) <= 1
        erros.append(np.abs(np.log(pv / te)).mean())
    assert dentro >= 16                      # medido: 18 de 20 (a semente 19 escolhe 6 contra 2)
    assert float(np.mean(erros)) < 0.25      # medido: 0.159


def test_escolher_componentes():
    rng = np.random.default_rng(0)
    Pm = np.linalg.qr(rng.normal(size=(30, 3)))[0]
    B = rng.normal(size=3)
    X, Y = _gerar(rng, 40, Pm, B)
    r = P.pcv_pls(X, Y, 8, cv=("ven", 5))
    n, curva = P.escolher_componentes(r, Y)
    assert 1 <= n <= r.ncomp and curva.shape == (r.ncomp,) and curva[n - 1] == curva.min()


def test_entradas_invalidas():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(20, 6)), rng.normal(size=20)
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError):
        P.pcv_pls(Xn, Y)
    with pytest.raises(ValueError):
        P.pcv_pls(X, Y[:10])
    with pytest.raises(ValueError):
        P.pcv_pls(X, Y, cv=("ven", 1))
    with pytest.raises(ValueError):
        P.pcv_pls(X, Y, cv_scope="x")
