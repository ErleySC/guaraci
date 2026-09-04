"""
preprocessamento.py — Transformers sklearn-compatíveis de pré-processamento
espectral (SNV, Savitzky-Golay, MSC) e o construtor de pipeline de
pré-processamento.

Extraído de pipeline.py como parte da modularização (Fase H). Depende de
Config só para type hint de `build_preprocessor` (import guardado por
TYPE_CHECKING, para não criar import circular com pipeline.py, que importa
este módulo). pipeline.py reexporta estes nomes, então `pipeline.SNV`,
`pipeline.build_preprocessor(...)` etc. continuam funcionando sem
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

__all__ = [
    "SNV",
    "SavGol",
    "MSC",
    "EMSC",
    "OSC",
    "build_preprocessor",
]


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
    AUDITORIA_METODOLOGICA_2026-08-07.md, secao "Dívida de
    engenharia observada"): a regressao de 2 parametros (a, b) por amostra
    e' uma regressao linear simples (1 preditor + intercepto), que tem
    forma fechada:
        b = Cov(ref, X_i) / Var(ref)
        a = mean(X_i) - b * mean(ref)
    resolvida para TODAS as amostras de uma vez via operacoes matriciais,
    em vez de um `np.linalg.lstsq` por amostra num loop Python (o mesmo
    resultado, so' mais lento -- desperdicio notavel com matrizes
    espectrais grandes
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


class EMSC(BaseEstimator, TransformerMixin):
    """Extended Multiplicative Signal Correction (Martens & Stark, 1991,
    J. Pharm. Biomed. Anal. 9(8):625-635, DOI 10.1016/0731-7085(91)80188-F
    -- verificado no Crossref em 2026-09-04).

    Generaliza MSC: alem do termo multiplicativo contra o espectro medio
    de referencia (igual ao MSC), ajusta tambem uma linha de base
    POLINOMIAL (ordem `ordem_polinomial`, no EIXO de indice do canal --
    ver nota abaixo) e, opcionalmente, espectros de INTERFERENTES
    conhecidos, tudo numa unica regressao por amostra:

        X_i(canal) ~= a_i + b_i*ref(canal) + sum_k c_ik*canal^k
                      + sum_j d_ij*interferente_j(canal)

    A correcao remove tudo (a_i, os termos polinomiais, os interferentes)
    exceto o termo multiplicativo b_i, igual ao MSC:

        X_corrigido_i = (X_i - a_i - sum_k c_ik*canal^k
                          - sum_j d_ij*interferente_j) / b_i

    NOTA: sem `eixo` explicito, usa o INDICE do canal (0..p-1) normalizado
    -- suficiente para capturar uma tendencia suave de linha de base;
    nao precisa do numero de onda fisico real para isso (a forma
    polinomial de uma linha de base nao muda por reescala/deslocamento
    linear do eixo). `ordem_polinomial=0` sem `interferentes` reduz ao
    MSC (so' com solver por minimos quadrados em vez da forma fechada
    usada em `MSC`, resultado numericamente equivalente).

    VEREDITO DO PORTAO DE ACEITE (Bloco 20, medido em 2026-09-04, ver
    `docs/PROGRESSO.md` Passo 134/`docs/VALIDACAO_PUBLICA.md` secao 9 --
    `portao_correcao_sinal.avaliar_correcao_sinal_pls`, 10 seeds, Wilcoxon
    pareado): APROVADO no acervo privado de oleo (RMSEP 4,70->4,39
    pooled, p=0,002) E no Corn publico (0,164->0,132, p=0,002). Dos dois
    metodos deste modulo, e' o unico com ganho comprovado nos DOIS
    cenarios testados ate' agora -- OSC (abaixo) piora no oleo."""

    def __init__(self, eixo=None, ordem_polinomial: int = 2, interferentes=None):
        self.eixo = eixo
        self.ordem_polinomial = ordem_polinomial
        self.interferentes = interferentes

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        self.ref_ = X.mean(axis=0)

        eixo = np.asarray(self.eixo, dtype=float) if self.eixo is not None \
            else np.arange(p, dtype=float)
        desvio = eixo.std()
        eixo_norm = (eixo - eixo.mean()) / (desvio if desvio > 1e-300 else 1.0)

        colunas = [np.ones(p), self.ref_]
        for k in range(1, int(self.ordem_polinomial) + 1):
            colunas.append(eixo_norm ** k)
        if self.interferentes is not None:
            interf = np.atleast_2d(np.asarray(self.interferentes, dtype=float))
            for linha in interf:
                colunas.append(linha)
        self.base_ = np.column_stack(colunas)   # (p, n_termos)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        base = self.base_
        coefs, *_ = np.linalg.lstsq(base, X.T, rcond=None)   # (n_termos, n)
        b = coefs[1]                                          # termo multiplicativo (ref)
        b_seguro = np.where(np.abs(b) > 1e-12, b, 1.0)
        base_aditiva = base.copy()
        base_aditiva[:, 1] = 0.0                              # zera a contribuicao de b
        reconstrucao_aditiva = base_aditiva @ coefs           # (p, n)
        Xc = (X.T - reconstrucao_aditiva) / b_seguro[None, :]
        return Xc.T


class OSC(BaseEstimator, TransformerMixin):
    """Orthogonal Signal Correction (Wold, Antti, Lindgren & Ohman, 1998,
    Chemom. Intell. Lab. Syst. 44(1-2):175-185, DOI
    10.1016/S0169-7439(98)00109-9 -- verificado no Crossref em 2026-09-04).

    Ao contrario de SNV/MSC/SG (nao-supervisionados, so' olham X), OSC usa
    `y` para remover de X so' a variacao ORTOGONAL ao alvo -- por isso
    `fit` exige `y` (nao e' opcional aqui). Dentro de um `Pipeline`
    (`cross_val_predict`/`cross_val_score`), o sklearn ja' passa `y` de
    treino para o `fit` de cada etapa automaticamente -- sem risco de
    vazamento adicional ao ja' existente no restante do pipeline.

    NIPALS por componente: parte do 1o PC de X (deflacionado pelos
    componentes OSC anteriores), ortogonaliza iterativamente o escore `t`
    em relacao a `y` (projeta fora a parte de `t` correlacionada com y),
    recalcula o loading `w` contra esse `t` ortogonal, ate' convergir;
    deflaciona X por esse componente e repete para `n_componentes`.

    VEREDITO DO PORTAO DE ACEITE (Bloco 20, medido em 2026-09-04, ver
    `docs/PROGRESSO.md` Passo 134/`docs/VALIDACAO_PUBLICA.md` secao 9):
    APROVADO no Corn publico (RMSEP 0,164->0,145, p=0,002), mas
    **REJEITADO** no acervo privado de oleo (quantificacao pooled de
    teor: RMSEP 4,70->4,99, PIOROU, p=0,002). Ganho NAO e' garantido por
    cenario -- confirme com `avaliar_correcao_sinal_pls` no seu proprio
    dataset antes de usar OSC como pre-processamento padrao; nao use so'
    porque ajudou no Corn."""

    def __init__(self, n_componentes: int = 1, max_iter: int = 100, tol: float = 1e-8):
        self.n_componentes = n_componentes
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        self.mean_ = X.mean(axis=0)
        Xr = X - self.mean_
        Yc = Y - Y.mean(axis=0)

        n_comp = int(max(1, min(self.n_componentes, X.shape[1], X.shape[0] - 1)))
        YtY_pinv = np.linalg.pinv(Yc.T @ Yc)

        ws, ps = [], []
        for _ in range(n_comp):
            u, s, _vt = np.linalg.svd(Xr, full_matrices=False)
            t = u[:, 0] * s[0]
            t_ant = None
            for _it in range(self.max_iter):
                t_orth = t - Yc @ (YtY_pinv @ (Yc.T @ t))
                w = Xr.T @ t_orth
                norma_w = np.linalg.norm(w)
                w = w / norma_w if norma_w > 1e-300 else w
                t_novo = Xr @ w
                t_novo_orth = t_novo - Yc @ (YtY_pinv @ (Yc.T @ t_novo))
                if t_ant is not None and np.linalg.norm(t_novo_orth - t_ant) < self.tol:
                    t = t_novo_orth
                    break
                t_ant, t = t_novo_orth, t_novo_orth
            tt = float(t @ t)
            p_load = (Xr.T @ t / tt) if tt > 1e-300 else np.zeros(X.shape[1])
            Xr = Xr - np.outer(t, p_load)
            ws.append(w)
            ps.append(p_load)

        self.w_ = np.column_stack(ws)   # (p, n_comp)
        self.p_ = np.column_stack(ps)   # (p, n_comp)
        return self

    def transform(self, X):
        Xc = np.asarray(X, dtype=float) - self.mean_
        for k in range(self.w_.shape[1]):
            t = Xc @ self.w_[:, k]
            Xc = Xc - np.outer(t, self.p_[:, k])
        return Xc


def build_preprocessor(cfg: "Config") -> Pipeline:
    """Builds preprocessor according to cfg.default_preprocessing.

    Presets:
        'snv_sg_mc'   : SNV -> SG -> mean-centering (Rinnan et al. 2009,
                        recommended for FTIR/NIR with scatter)
        'autoscaling' : StandardScaler (mean + unit variance)
                        — recommended when SG derivative destroys signal
                        or for NIR without pronounced scatter
        'mc'          : mean-centering only
        'custom'      : honors apply_snv / apply_sg / apply_mc

    Mean-centering / autoscaling are kept INSIDE the Pipeline so that
    cross_val_predict does not leak statistics between folds.
    """
    preset = (cfg.default_preprocessing or "custom").lower()

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
    if cfg.apply_snv:
        etapas.append(("snv", SNV()))
    if cfg.apply_emsc:
        etapas.append(("emsc", EMSC(ordem_polinomial=cfg.emsc_ordem_polinomial)))
    if cfg.apply_sg:
        etapas.append(("sg", SavGol(cfg.sg_window, cfg.sg_polyorder, cfg.sg_deriv)))
    if cfg.apply_mc:
        etapas.append(("mc", StandardScaler(with_std=False)))
    if cfg.apply_osc:
        # OSC precisa de y (Y_bin one-hot) -- sklearn.Pipeline.fit(X, y)
        # ja passa y para toda etapa que aceite, entao entra por ultimo
        # sem nenhuma alteracao de assinatura em quem chama build_preprocessor.
        etapas.append(("osc", OSC(n_componentes=cfg.osc_n_componentes)))
    if not etapas:
        etapas.append(("mc", StandardScaler(with_std=False)))
    return Pipeline(etapas)
