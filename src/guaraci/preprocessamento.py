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
    returns (X_i - a) / b. Stateful: must remain inside Pipeline+CV.

    VETORIZADO em 2026-08-07 (achado da auditoria metodologica -- ver
    docs/auditoria/AUDITORIA_METODOLOGICA_2026-08-07.md, secao "Dívida de
    engenharia observada"): a regressao de 2 parametros (a, b) por amostra
    e' uma regressao linear simples (1 preditor + intercepto), que tem
    forma fechada:
        b = Cov(ref, X_i) / Var(ref)
        a = mean(X_i) - b * mean(ref)
    resolvida para TODAS as amostras de uma vez via operacoes matriciais,
    em vez de um `np.linalg.lstsq` por amostra num loop Python (o mesmo
    resultado, so' mais lento -- desperdicio notavel com 934x8192 pontos
    espectrais reais). Verificado numericamente contra a versao anterior
    em 20 casos aleatorios + casos estruturados (b=0/1/2): diff < 1e-8.

    Unico caso em que o resultado MUDA de proposito: referencia de treino
    com variancia ~0 (espectro medio CONSTANTE em todo o eixo -- nao
    acontece com dado espectral real, exigiria um instrumento sem
    absolutamente nenhum sinal). Nesse caso a regressao e' mal-posta;
    `lstsq` antigo devolvia a solucao de NORMA MINIMA via SVD (um artefato
    numerico, nao uma resposta cientifica definida), a versao atual cai no
    MESMO fallback ja usado por amostra quando b~=0 (so' subtrai a media),
    mais previsivel que o artefato do SVD.
    """

    def fit(self, X, y=None):
        self.ref_ = np.asarray(X, dtype=float).mean(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        ref = self.ref_
        x_mean = float(ref.mean())
        xc = ref - x_mean
        var_x = float(xc @ xc)

        y_mean = X.mean(axis=1)                  # (n,)
        if var_x < 1e-12:
            # Referencia degenerada (variancia ~0) -- ver docstring: sem
            # regressao possivel, so' centra pela media de cada amostra.
            return X - y_mean[:, None]

        Yc = X - y_mean[:, None]                 # (n, p)
        b = (Yc @ xc) / var_x                     # (n,) -- Cov(ref, X_i)/Var(ref)
        a = y_mean - b * x_mean                   # (n,)

        b_seguro = np.where(np.abs(b) > 1e-12, b, 1.0)
        out = (X - a[:, None]) / b_seguro[:, None]
        b_quase_zero = np.abs(b) <= 1e-12
        if b_quase_zero.any():
            out[b_quase_zero] = (X - a[:, None])[b_quase_zero]
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
        # MSC->SG+MC: default preset for diffuse FT-NIR with strong scatter.
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
