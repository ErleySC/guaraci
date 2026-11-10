"""Selectivity Ratio: invariante a uma constante somada a X (auditoria de 2026-11-10)."""
import numpy as np
from sklearn.cross_decomposition import PLSRegression

from guaraci.chemometric_stats import compute_selectivity_ratio


def _dados(seed=0, n=60, p=80):
    rng = np.random.default_rng(seed)
    wl = np.linspace(0, 1, p)
    c = rng.uniform(0, 1, (n, 3))
    S = np.stack([np.exp(-((wl - m) / 0.05) ** 2) for m in (0.3, 0.5, 0.7)])
    return c @ S + 0.01 * rng.standard_normal((n, p)), c[:, 0]


def _sr_formula(modelo, X):
    """Rajalahti et al. (2009), escrita de forma independente, sobre X centrado."""
    Xc = X - X.mean(0)
    b = modelo.coef_.ravel()
    w = b / np.linalg.norm(b)
    t = Xc @ w
    P = (t @ Xc) / (t @ t)
    Xtp = np.outer(t, P)
    return (Xtp ** 2).sum(0) / ((Xc - Xtp) ** 2).sum(0)


def test_sr_nao_depende_da_media_dos_espectros():
    X, y = _dados()
    pls = PLSRegression(3, scale=False).fit(X, y)
    a = compute_selectivity_ratio(pls, X)
    b = compute_selectivity_ratio(pls, X + 5.0)
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_sr_bate_com_a_formula_publicada():
    X, y = _dados(1)
    Xs = X + 5.0
    pls = PLSRegression(3, scale=False).fit(Xs, y)
    np.testing.assert_allclose(compute_selectivity_ratio(pls, Xs), _sr_formula(pls, Xs), rtol=1e-6)
