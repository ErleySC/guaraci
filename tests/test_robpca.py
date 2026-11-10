"""ROBPCA (guaraci.robpca): comparacao com rrcov::PcaHubert e propriedades conhecidas.

Os fixtures vem de `tests/oraculos_metodos/gen_oraculo_robpca.R` (R 4.3.3, rrcov 1.7-5, robustbase 0.99-2).
Isto mostra que a reimplementacao faz o mesmo que o rrcov nesses casos; nao e' prova de que o rrcov esteja
certo nem cobre dados onde o MCD final tem otimos empatados (ver a docstring do modulo).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from guaraci import robpca as R

_ARQ = Path(__file__).parent / "oraculos_metodos" / "oraculo_robpca.json"
_ORACULO = json.loads(_ARQ.read_text(encoding="utf-8"))
_CASOS = {c["id"]: c for c in _ORACULO["casos"]}


def _kwargs(args):
    kw = {}
    if "k" in args:
        kw["k"] = int(args["k"])
    if "alpha" in args:
        kw["alpha"] = float(args["alpha"])
    if "scale" in args:
        kw["scale"] = bool(args["scale"])
    if "mcd" in args:
        kw["mcd_se_p_pequeno"] = bool(args["mcd"])
    return kw


def test_fixture_tem_versoes():
    assert _ORACULO["versoes"]["rrcov"] == "1.7.5" and len(_CASOS) == 9


@pytest.mark.parametrize("cid", sorted(_CASOS))
def test_igual_ao_rrcov(cid):
    c = _CASOS[cid]
    r = R.robpca(np.array(c["X"]), **_kwargs(c["args"]))
    assert r.k == c["k"] and r.h == c["quan"]
    np.testing.assert_allclose(r.eigenvalues, c["eigenvalues"], rtol=1e-8)
    np.testing.assert_allclose(r.center, c["center"], atol=1e-8)
    np.testing.assert_allclose(r.sd, c["sd"], atol=1e-8)
    np.testing.assert_allclose(r.od, c["od"], atol=1e-7)
    assert r.cutoff_sd == pytest.approx(c["cutoff_sd"], abs=1e-10)
    assert r.cutoff_od == pytest.approx(c["cutoff_od"], abs=1e-7)
    assert r.flag.tolist() == c["flag"]
    # cargas e escores so' a menos do sinal de cada coluna
    L, S = np.array(c["loadings"]), np.array(c["scores"])
    sinal = np.sign(np.sum(L * r.loadings, axis=0))
    np.testing.assert_allclose(r.loadings * sinal, L, atol=1e-8)
    np.testing.assert_allclose(r.scores * sinal, S, atol=1e-7)


def test_fixtures_exercitam_todos_os_ramos():
    # caminho MCD direto (p pequeno), ROBPCA com direcoes aleatorias (n > 22), todas as direcoes (n <= 22),
    # n < p (kernel), padronizacao, k automatico, k = 1
    r = {cid: R.robpca(np.array(c["X"]), **_kwargs(c["args"])) for cid, c in _CASOS.items()}
    assert any(not x.flag.all() for x in r.values())          # ha' outliers marcados
    assert np.array(_CASOS["H_n25_p60_kernel"]["X"]).shape[0] < np.array(_CASOS["H_n25_p60_kernel"]["X"]).shape[1]
    assert r["G_n40_p30_k1"].k == 1 and r["B_n20_p8_auto"].k == 2 and r["F_n50_p40_auto"].k == 3
    assert np.all(r["E_n30_p5_scale"].scale > 0) and not np.allclose(r["E_n30_p5_scale"].scale, 1.0)


# ---------------------------------------------------------------- propriedades conhecidas (sem R)
def _contaminado(seed=0, n=80, p=20, nout=12):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(2, p))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    T = rng.normal(size=(n, 2)) * np.array([5.0, 3.0])
    X = T @ base + rng.normal(size=(n, p)) * 0.2
    X[:nout] += rng.normal(8.0, 1.0, size=(nout, p))
    return X, base, nout


def test_recupera_subespaco_com_contaminacao_e_marca_os_outliers():
    X, base, nout = _contaminado()
    r = R.robpca(X, k=2)
    # angulos principais entre o subespaco estimado e o verdadeiro
    q1, _ = np.linalg.qr(r.loadings)
    q2, _ = np.linalg.qr(base.T)
    cos_min = np.linalg.svd(q1.T @ q2, compute_uv=False).min()
    assert cos_min > 0.98
    assert (~r.flag[:nout]).all()                     # todos os contaminados marcados
    assert r.flag[nout:].mean() > 0.9                 # quase nenhum falso alarme
    # a PCA classica, nos mesmos dados, erra o subespaco por bem mais
    U = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:2].T
    cos_cl = np.linalg.svd(np.linalg.qr(U)[0].T @ q2, compute_uv=False).min()
    assert cos_cl < cos_min


def test_equivariancia_a_translacao():
    X, _, _ = _contaminado(1)
    a, b = R.robpca(X, k=2), R.robpca(X + 100.0, k=2)
    np.testing.assert_allclose(a.eigenvalues, b.eigenvalues, rtol=1e-7)
    np.testing.assert_allclose(b.center - 100.0, a.center, atol=1e-6)
    assert a.flag.tolist() == b.flag.tolist()


def test_deterministico_com_a_mesma_semente():
    X, _, _ = _contaminado(2)
    a, b = R.robpca(X, k=2, seed=3), R.robpca(X, k=2, seed=3)
    np.testing.assert_array_equal(a.scores, b.scores)


def test_entradas_invalidas():
    X, _, _ = _contaminado(3)
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError):
        R.robpca(Xn)
    with pytest.raises(ValueError):
        R.robpca(X, alpha=0.3)
    with pytest.raises(ValueError):
        R.robpca(np.ones((10, 4)))                    # todos os pontos colapsam


def test_mcd_resiste_a_contaminacao():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    X[:20] += 10.0
    m = R.mcd(X, alpha=0.75)
    assert np.all(np.abs(m.center) < 0.5) and not m.weights[:20].any()
    assert np.all(np.abs(np.diag(m.cov) - 1.0) < 0.5)


def test_unimcd_univariado():
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(size=60), rng.normal(size=10) + 9.0])
    tm, sm, w = R.unimcd(y, 52)
    assert abs(tm) < 0.5 and 0.7 < sm < 1.4 and not w[60:].any()


def test_h_alpha_n_como_no_robustbase():
    # valores conferidos no R: robustbase::h.alpha.n(0.75, 40, 10) e (0.5, 100, 3)
    assert R.h_alpha_n(0.75, 40, 10) == 32 and R.h_alpha_n(0.5, 100, 3) == 52
