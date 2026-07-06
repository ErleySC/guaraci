"""
classificadores.py — Classificadores quimiométricos avançados: DD-SIMCA
(one-class por classe) e OPLS-DA (deflação ortogonal + Gram-Schmidt).

Extraído de pipeline.py como parte da modularização (Fase H). Sem
acoplamento a Config — dependem só de numpy/scipy/sklearn e de
chemometric_stats.py (hotelling_t2_limite, q_residuos_limite). pipeline.py
reexporta estes nomes, então `pipeline.DDSimca(...)`,
`pipeline.OPLSDAWrapper(...)` etc. continuam funcionando sem alteração.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import chi2
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from guaraci.chemometric_stats import hotelling_t2_limite, q_residuos_limite


class DDSimca:
    """Data-Driven SIMCA: per-class one-class classifier via PCA.

    For each class, trains an independent PCA model and defines
    acceptance limits (UCL) for T2 and Q-residuals:
        T2_UCL  — computed by ucl_method:
                    'empirical'  : (1-alpha) percentile of training T2
                    'theoretical': Tracy-Young-Mason (F-distribution)
                    'chi2'       : chi2(1-alpha, n_components)
                  'empirical' is the only one that VARIES PER CLASS
                  (theoretical and chi2 depend only on n,k); recommended.
        Q_UCL   — chi2 approximation (Jackson & Mudholkar) via mean/var of
                  training Q-residuals — naturally data-driven.

    A new sample is 'accepted' by the class if T2 <= UCL **and** Q <= UCL.

    Referencias:
        Rodionova O.Y. & Pomerantsev A.L. (2020). Chemom. Intell. Lab.
        Syst. 200:103958.
    """

    def __init__(self, n_components: int = 3, alpha: float = 0.05,
                 ucl_method: str = "empirical"):
        self.n_components = n_components
        self.alpha = alpha
        self.ucl_method = ucl_method
        self._modelos: Dict[str, Dict[str, Any]] = {}
        self._classes: np.ndarray = np.array([], dtype=str)

    def _compute_t2_ucl(self, T2_train: np.ndarray, n: int, k: int) -> float:
        method = (self.ucl_method or "empirical").lower()
        if method == "empirical":
            if T2_train.size == 0:
                return float("inf")
            return float(np.percentile(T2_train, 100 * (1 - self.alpha)))
        if method == "theoretical":
            return hotelling_t2_limite(n, k, self.alpha)
        if method == "chi2":
            return float(chi2.ppf(1 - self.alpha, k))
        # fallback
        return float(np.percentile(T2_train, 100 * (1 - self.alpha)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DDSimca":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=str)
        self._classes = np.unique(y)
        self._modelos = {}
        for cls in self._classes:
            Xc = X[y == cls]
            nc = len(Xc)
            n_comp = min(self.n_components, nc - 1, Xc.shape[1])
            if n_comp < 1:
                print(f"[DDSimca] Class '{cls}': insufficient samples "
                      f"(n={nc}) — model skipped.")
                continue
            pca = PCA(n_components=n_comp)
            T = pca.fit_transform(Xc)
            var_t = T.var(axis=0, ddof=1)
            var_t[var_t == 0] = 1.0
            X_rec = pca.inverse_transform(T)
            Q_train = np.sum((Xc - X_rec) ** 2, axis=1)
            T2_train = np.sum((T ** 2) / var_t, axis=1)

            t2_ucl = self._compute_t2_ucl(T2_train, nc, n_comp)
            q_ucl  = q_residuos_limite(Q_train, self.alpha)

            # Small-n guard: with nc < 20 two numerical bugs cause training samples
            # to be rejected by their OWN model.
            #
            # BUG A (T2): np.percentile([a,b,c], 95) < max → the max-T2 sample
            #   always gets T2_norm = max/(0.9·max) ≈ 1.11 > 1 → rejected.
            #
            # BUG B (Q): PCA with n_comp = n-1 fits training perfectly → Q≈0
            #   → q_ucl ≈ 0 → Q_norm = Q/1e-12 → ∞ for all samples.
            #
            # Fix: clamp UCLs to at least max(training statistic), then add a
            # tiny RELATIVE tolerance (1 ppm = 1e-6) to absorb floating-point
            # discrepancies between pca.fit_transform() (used during fit) and
            # pca.transform() (used during score_matrix evaluation). Without
            # the tolerance, the exact-max sample gets T2_norm ≈ 1.0000004 > 1
            # and is silently rejected. 1 ppm is imperceptible for real samples
            # (adulterated oils have T2_norm >> 1) but fixes precision artefacts.
            _EPS_UCL = 1e-6   # 1 ppm relative tolerance
            if nc < 20:
                if T2_train.size > 0:
                    t2_ucl = max(t2_ucl,
                                 float(T2_train.max()) * (1.0 + _EPS_UCL) + 1e-12)
                if Q_train.size > 0:
                    q_ucl  = max(q_ucl,
                                 float(Q_train.max())  * (1.0 + _EPS_UCL) + 1e-12)

            self._modelos[cls] = {
                "pca":      pca,
                "var_t":    var_t,
                "T2_ucl":   t2_ucl,
                "Q_ucl":    q_ucl,
                "T_train":  T,
                "T2_train": T2_train,
                "Q_train":  Q_train,
                "n_train":  nc,
                "n_comp":   n_comp,
            }
        return self

    def _t2_q(self, X: np.ndarray, cls: str
              ) -> Tuple[np.ndarray, np.ndarray]:
        m = self._modelos[cls]
        pca = m["pca"]
        T = pca.transform(X)
        X_rec = pca.inverse_transform(T)
        Q  = np.sum((X - X_rec) ** 2, axis=1)
        T2 = np.sum((T ** 2) / m["var_t"], axis=1)
        return T2, Q

    def score_matrix(self, X: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """T2, Q and normalized versions (T2/UCL, Q/UCL) per class."""
        X = np.asarray(X, dtype=float)
        res: Dict[str, Dict[str, Any]] = {}
        for cls in self._classes:
            if cls not in self._modelos:
                continue
            m = self._modelos[cls]
            T2, Q = self._t2_q(X, cls)
            res[cls] = {
                "T2":       T2,
                "Q":        Q,
                "T2_ucl":   m["T2_ucl"],
                "Q_ucl":    m["Q_ucl"],
                "T2_norm":  T2 / max(m["T2_ucl"], 1e-12),
                "Q_norm":   Q  / max(m["Q_ucl"],  1e-12),
                "T_train":  m["T_train"],
                "Q_train":  m["Q_train"],
                "n_train":  m["n_train"],
            }
        return res

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns: class name | 'Ambiguo' | 'Desconhecido'."""
        X = np.asarray(X, dtype=float)
        preds = []
        for i in range(len(X)):
            xi = X[i:i+1]
            aceitas = []
            for cls in self._classes:
                if cls not in self._modelos:
                    continue
                m = self._modelos[cls]
                T2, Q = self._t2_q(xi, cls)
                if T2[0] <= m["T2_ucl"] and Q[0] <= m["Q_ucl"]:
                    aceitas.append(cls)
            if   len(aceitas) == 1: preds.append(aceitas[0])
            elif len(aceitas) >  1: preds.append("Ambiguo")
            else:                   preds.append("Desconhecido")
        return np.array(preds)


class OPLSDAWrapper(BaseEstimator):
    """OPLS-DA: orthogonal deflation + 1 predictive component.

    Implements Algorithm 1 of Trygg & Wold (2002) with explicit Gram-Schmidt
    orthogonalization to guarantee t_orth ⊥ t_pred by construction.

    Steps per orthogonal component:
      1. NIPALS PLS1 on (X_r, y) → w (normalized), t = X_r @ w, p = X_r^T t / (t^T t)
      2. w_orth = p − (p^T w) w   [part of p orthogonal to w, per Trygg 2002]
      3. w_orth /= ||w_orth||
      4. t_orth_raw = X_r @ w_orth
      5. Gram-Schmidt: t_orth = t_orth_raw − (t_orth_raw^T t)/(t^T t) * t
         [explicit enforcement of t_orth ⊥ t_pred; required for valid S-Plots]
      6. p_orth = X_r^T t_orth / (t_orth^T t_orth)
      7. Deflate: X_r = X_r − t_orth p_orth^T

    Outputs:
        t_pred  — predictive score (separates classes)
        t_orth  — orthogonal score(s) (systematic X-variation uncorrelated with Y,
                  e.g., baseline drift, multiplicative scatter in FT-NIR)

    References:
        Trygg J. & Wold S. (2002) J. Chemometrics 16:119-128.
        Bylesjo M. et al. (2006) J. Chemometrics 20:341-351.
        Wiklund S. et al. (2008) Anal. Chem. 80:115-122.
    """

    def __init__(self, n_ortho: int = 1):
        self.n_ortho = n_ortho
        self.W_orth_: List[np.ndarray] = []
        self.P_orth_: List[np.ndarray] = []

    @staticmethod
    def _nipals_pls1(X: np.ndarray, y: np.ndarray,
                     max_iter: int = 500, tol: float = 1e-10
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """NIPALS PLS1 — extracts 1 component from (X, y).

        Returns (w, t, p):
            w  : X-weight (unit norm), shape (n_features,)
            t  : X-score = X @ w,      shape (n_samples,)
            p  : X-loading = X^T t / (t^T t), shape (n_features,)
        """
        u = y.astype(float).copy()
        t_old = np.zeros(X.shape[0])
        w = np.zeros(X.shape[1])
        t = t_old.copy()
        for _ in range(max_iter):
            w = X.T @ u
            nw = float(np.linalg.norm(w))
            if nw < 1e-12:
                break
            w /= nw
            t = X @ w
            nt = float(t @ t)
            if nt < 1e-12:
                break
            c = float(y @ t) / nt
            u_new = y * c
            if float(np.linalg.norm(t - t_old)) / (float(np.linalg.norm(t)) + 1e-12) < tol:
                break
            t_old = t.copy()
            u = u_new
        nt = float(t @ t)
        p = X.T @ t / nt if nt > 1e-12 else np.zeros(X.shape[1])
        return w, t, p

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "OPLSDAWrapper":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        # Build a single continuous y that captures all-class discriminant structure.
        # For binary Y (1 column): use that column directly.
        # For multiclass Y (K columns, one-hot): use the first Linear Discriminant
        # component (LDA), which maximally separates all K classes simultaneously.
        # Using Y[:,0] (first class vs. rest) would silently bias the OPLS toward
        # one class only — a methodological error for 14-class FT-NIR data.
        if Y.ndim == 2 and Y.shape[1] > 1:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _LDA
            y_int_opls = np.argmax(Y, axis=1)
            try:
                _lda = _LDA(n_components=1)
                y = _lda.fit_transform(X, y_int_opls)[:, 0].astype(float)
            except Exception:
                # Fallback: PLS2 first y-score (less optimal but correct for multiclass)
                from sklearn.cross_decomposition import PLSRegression as _PLSr
                _pls2 = _PLSr(n_components=1, scale=False)
                _pls2.fit(X, Y)
                _ys = _pls2.y_scores_
                y = (np.asarray(_ys, dtype=float)[:, 0]
                     if _ys is not None else Y @ np.ones(Y.shape[1]))
        else:
            y = (Y[:, 0] if Y.ndim == 2 else Y.copy()).astype(float)
        y = y - float(y.mean())

        n = X.shape[0]
        Xr = X.copy()
        self.W_orth_ = []
        self.P_orth_ = []
        T_orth_train: List[np.ndarray] = []

        for _ in range(self.n_ortho):
            w, t, p = self._nipals_pls1(Xr, y)

            # Step 2 (Trygg & Wold 2002, Eq. 3):
            # w_orth = part of p orthogonal to w (since ||w||=1, proj = p^T w)
            proj = float(p @ w)
            w_orth = p - proj * w
            no = float(np.linalg.norm(w_orth))
            if no < 1e-10:
                break
            w_orth /= no

            # Step 4: raw orthogonal score
            t_orth_raw = Xr @ w_orth

            # Step 5 — Gram-Schmidt: remove predictive component from t_orth
            # Guarantees t_orth ⊥ t_pred by construction (required for valid S-Plot).
            # Without this, w_orth ⊥ w does NOT imply X@w_orth ⊥ X@w when X^T X ≠ I.
            t_norm_sq = float(t @ t)
            if t_norm_sq > 1e-12:
                t_orth = t_orth_raw - (float(t_orth_raw @ t) / t_norm_sq) * t
            else:
                t_orth = t_orth_raw

            nto = float(t_orth @ t_orth)
            if nto < 1e-12:
                break

            # Step 6: orthogonal loading from Gram-Schmidt-corrected t_orth
            p_orth = Xr.T @ t_orth / nto
            self.W_orth_.append(w_orth.copy())
            self.P_orth_.append(p_orth.copy())
            T_orth_train.append(t_orth.copy())

            # Step 7: deflate X only (y is not deflated — single predictive LV)
            Xr = Xr - np.outer(t_orth, p_orth)

        # 1 predictive component on deflated X
        self._pls_pred = PLSRegression(n_components=1, scale=False)
        self._pls_pred.fit(Xr, Y)

        # Training scores (sklearn >= 1.x returns ndarray directly)
        _t_arr = self._pls_pred.transform(Xr)
        t_pred_tr = _t_arr if isinstance(_t_arr, np.ndarray) else _t_arr[0]
        self.t_pred_train_ = t_pred_tr[:, 0]
        self.t_orth_train_ = (np.column_stack(T_orth_train)
                               if T_orth_train else np.zeros((n, 1)))
        self.n_ortho_fitted_ = len(self.W_orth_)
        return self

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (t_pred, t_orth) with t_orth ⊥ t_pred (Gram-Schmidt applied).

        Deflation uses the raw w_orth/p_orth (consistent with fit), but the
        returned t_orth vectors are Gram-Schmidt orthogonalized against t_pred
        so that S-Plot covariance/correlation axes are rigorously orthogonal.
        """
        X = np.asarray(X, dtype=float)
        Xr = X.copy()
        T_orth_raw: List[np.ndarray] = []
        for w_o, p_o in zip(self.W_orth_, self.P_orth_):
            t_o_raw = Xr @ w_o
            T_orth_raw.append(t_o_raw)
            Xr = Xr - np.outer(t_o_raw, p_o)   # deflation unchanged

        _t_new = self._pls_pred.transform(Xr)
        t_pred_arr = _t_new if isinstance(_t_new, np.ndarray) else _t_new[0]
        t_pred = t_pred_arr[:, 0]

        # Gram-Schmidt: remove predictive component from each raw t_orth
        t_pred_norm_sq = float(t_pred @ t_pred)
        T_orth: List[np.ndarray] = []
        for t_o_raw in T_orth_raw:
            if t_pred_norm_sq > 1e-12:
                t_o = t_o_raw - (float(t_o_raw @ t_pred) / t_pred_norm_sq) * t_pred
            else:
                t_o = t_o_raw
            T_orth.append(t_o)

        t_orth = (np.column_stack(T_orth)
                  if T_orth else np.zeros((len(X), 1)))
        return t_pred, t_orth
