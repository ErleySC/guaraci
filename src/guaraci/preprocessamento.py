"""
preprocessamento.py — Transformers sklearn-compatíveis de pré-processamento
espectral (SNV, Savitzky-Golay, MSC) e o construtor de pipeline de
pré-processamento.

Extraído de pipeline.py como parte da modularização (Fase H). Depende de
Config só para type hint de `construir_preprocessador` (import guardado por
TYPE_CHECKING, para não criar import circular com pipeline.py, que importa
este módulo). pipeline.py reexporta estes nomes, então `pipeline.SNV`,
`pipeline.construir_preprocessador(...)` etc. continuam funcionando sem
alteração.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from guaraci.pipeline import Config


class SNV(BaseEstimator, TransformerMixin):
    """Standard Normal Variate: per-sample z-score (scatter correction)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        return (X - mu) / sd


class SavGol(BaseEstimator, TransformerMixin):
    """Savitzky-Golay filter (smoothing or derivative)."""

    def __init__(self, window_length: int = 25, polyorder: int = 2, deriv: int = 1):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return savgol_filter(np.asarray(X, dtype=float),
                             window_length=self.window_length,
                             polyorder=self.polyorder,
                             deriv=self.deriv, axis=1)


class MSC(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction. Uses mean training spectrum as
    reference; for each sample estimates (a, b) such that X_i ~ a + b * ref and
    returns (X_i - a) / b. Stateful: must remain inside Pipeline+CV."""

    def fit(self, X, y=None):
        self.ref_ = np.asarray(X, dtype=float).mean(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        A = np.column_stack([np.ones_like(self.ref_), self.ref_])
        out = np.zeros_like(X)
        for i in range(X.shape[0]):
            sol, *_ = np.linalg.lstsq(A, X[i], rcond=None)
            a, b = float(sol[0]), float(sol[1])
            out[i] = (X[i] - a) / b if abs(b) > 1e-12 else X[i] - a
        return out


def construir_preprocessador(cfg: "Config") -> Pipeline:
    """Builds preprocessor according to cfg.preprocessamento_padrao.

    Presets:
        'snv_sg_mc'   : SNV -> SG -> mean-centering (Rinnan et al. 2009,
                        recommended for FTIR/NIR with scatter)
        'autoscaling' : StandardScaler (mean + unit variance)
                        — recommended when SG derivative destroys signal
                        or for NIR without pronounced scatter
        'mc'          : mean-centering only
        'custom'      : honors aplicar_snv / aplicar_sg / aplicar_mc

    Mean-centering / autoscaling are kept INSIDE the Pipeline so that
    cross_val_predict does not leak statistics between folds.
    """
    preset = (cfg.preprocessamento_padrao or "custom").lower()

    if preset == "autoscaling":
        return Pipeline([("auto", StandardScaler(with_mean=True, with_std=True))])
    if preset == "mc":
        return Pipeline([("mc", StandardScaler(with_std=False))])
    if preset == "snv_sg_mc":
        return Pipeline([
            ("snv", SNV()),
            ("sg",  SavGol(cfg.sg_window, cfg.sg_polyorder, cfg.sg_deriv)),
            ("mc",  StandardScaler(with_std=False)),
        ])
    if preset == "msc_sg_mc":
        # MSC->SG+MC: best pipeline on the full dataset (0.923 bal.acc).
        # MSC is stateful (reference = training mean) -> kept inside
        # Pipeline to avoid leakage between CV folds.
        return Pipeline([
            ("msc", MSC()),
            ("sg",  SavGol(cfg.sg_window, cfg.sg_polyorder, cfg.sg_deriv)),
            ("mc",  StandardScaler(with_std=False)),
        ])
    # custom — uses individual flags
    etapas: List[Tuple[str, BaseEstimator]] = []
    if cfg.aplicar_snv:
        etapas.append(("snv", SNV()))
    if cfg.aplicar_sg:
        etapas.append(("sg", SavGol(cfg.sg_window, cfg.sg_polyorder, cfg.sg_deriv)))
    if cfg.aplicar_mc:
        etapas.append(("mc", StandardScaler(with_std=False)))
    if not etapas:
        etapas.append(("mc", StandardScaler(with_std=False)))
    return Pipeline(etapas)
