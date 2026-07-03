"""
chemometric_stats.py — Diagnósticos quimiométricos puros (VIP, SR, Hotelling T²,
Q-resíduos, variância explicada).

Extraído de pipeline.py como primeiro passo da modularização (Fase H). Funções
PURAS: dependem só de numpy/scipy, sem acoplamento a Config nem ao resto do
pipeline. pipeline.py reexporta estes nomes, então `pipeline.vip_scores(...)`
e chamadas internas continuam funcionando sem alteração.

Coberto por tests/test_pipeline_core.py.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import f as f_dist, chi2
from sklearn.cross_decomposition import PLSRegression


def vip_scores(modelo: PLSRegression) -> np.ndarray:
    """VIP scores per Chong & Jun (2005), Chemom. Intell. Lab. Syst. 78:103-112."""
    T = np.asarray(modelo.x_scores_, dtype=float)
    W = np.asarray(modelo.x_weights_, dtype=float)
    Q = np.asarray(modelo.y_loadings_, dtype=float)
    p, _ = W.shape
    ss = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    normas = np.linalg.norm(W, axis=0); normas[normas == 0] = 1.0
    W_norm = W / normas
    return np.sqrt(p * np.sum(ss * W_norm ** 2, axis=1) / (np.sum(ss) + 1e-12))


def calcular_selectivity_ratio(modelo: PLSRegression,
                                X: np.ndarray) -> np.ndarray:
    """Selectivity Ratio (SR) per Rajalahti et al. (2009),
    Chemom. Intell. Lab. Syst. 95:20-28.

    Para cada variavel j, decompoe X_j em parte explicada pela projecao
    alvo (primeiro peso preditivo PLS) e residuo:
        t_tp  = X @ w1 / ||w1||   (target projection scores)
        p_tp_j = (t_tp^T * X_j) / (t_tp^T * t_tp)
        SR_j  = Var(t_tp * p_tp_j) / Var(X_j - t_tp * p_tp_j)

    Complementa o VIP: SR e mais sensivel a variaveis com correlacao
    direcional com Y no 1o componente; VIP integra todos os LVs.
    Concordancia entre VIP >= 1 e SR alto reforca a relevancia.
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(modelo.x_weights_, dtype=float)   # (p, n_lv)
    w1 = W[:, 0]
    norm_w = float(np.linalg.norm(w1))
    if norm_w < 1e-12:
        return np.zeros(X.shape[1])
    w1_unit = w1 / norm_w

    t_tp = X @ w1_unit                  # (n,)
    tt = float(t_tp @ t_tp)
    if tt < 1e-12:
        return np.zeros(X.shape[1])

    p_tp   = (t_tp @ X) / tt            # (p,) — target projection loadings
    X_tp   = np.outer(t_tp, p_tp)       # (n, p) — target-projected X
    X_res  = X - X_tp                   # (n, p) — residual

    var_tp  = X_tp.var(axis=0, ddof=1)
    var_res = X_res.var(axis=0, ddof=1)
    var_res[var_res < 1e-12] = 1e-12
    return var_tp / var_res


def hotelling_t2(T: np.ndarray) -> np.ndarray:
    var_t = T.var(axis=0, ddof=1)
    var_t[var_t == 0] = 1.0
    return np.sum((T ** 2) / var_t, axis=1)


def hotelling_t2_limite(n: int, k: int, alpha: float = 0.05) -> float:
    """Hotelling T2 upper control limit (Tracy-Young-Mason 1992).

    Correct small-sample formula, valid for both observations
    within the calibration set and new observations:

        T2_UCL = k * (n - 1) * (n + 1) / (n * (n - k)) * F_(alpha, k, n - k)

    Replaces the approximation (k(n-1)/(n-k))*F that underestimated the limit
    by ~5-10% for n<30 (causing false outliers in small datasets).
    """
    if n - k <= 0:
        print(f"[WARNING] Hotelling T2: n={n} too small for k={k} LVs.")
        return float("inf")
    if n < 3 * k:
        print(f"[WARNING] Hotelling T2: n={n} < 3k={3*k}. Limit may be "
              f"imprecise (wide confidence interval).")
    return float(((k * (n - 1) * (n + 1)) / (n * (n - k)))
                  * f_dist.ppf(1 - alpha, k, n - k))


def q_residuos(X: np.ndarray, T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.sum((X - T @ P) ** 2, axis=1)


def q_residuos_limite(q: np.ndarray, alpha: float = 0.05) -> float:
    media = float(q.mean()); var = float(q.var())
    if var <= 0 or media <= 0:
        return float(np.percentile(q, (1 - alpha) * 100)) if q.size else 0.0
    g = var / (2 * media); h = 2 * (media ** 2) / var
    return float(g * chi2.ppf(1 - alpha, h))


def variancia_explicada(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Explained variance (%) of X by each column of T."""
    var_X_total = float(np.var(X, axis=0).sum())
    if var_X_total <= 0:
        return np.zeros(T.shape[1])
    return np.var(T, axis=0) / var_X_total * 100
