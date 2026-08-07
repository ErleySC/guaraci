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

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import chi2
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from guaraci.chemometric_stats import (hotelling_t2_limite, q_residuos_limite,
                                       media_e_dof_momentos, distancia_combinada)

log = logging.getLogger(__name__)

# Graus de liberdade residuais minimos exigidos apos a PCA (nc - n_comp).
# Com nc amostras de treino, PCA com n_comp = nc-1 componentes reconstroi
# o treino EXATAMENTE (Q_train ~= 0 para toda amostra, pois todos os graus
# de liberdade entre as nc amostras centradas foram consumidos). Isso nao
# e' ruido numerico: e' uma propriedade exata de PCA quando n_comp se
# aproxima de nc-1. q_residuos_limite() estima o UCL a partir de
# media/variancia de Q_train (Jackson & Mudholkar) -- com Q_train ~= 0
# para todas as amostras, a variancia colapsa e o UCL colapsa junto,
# rejeitando qualquer amostra nova/retida por um fator de ordens de
# magnitude (achado de auditoria adversarial, 2026-07-19: sensibilidade
# LOGO cai para 0.0 mesmo com grupos estatisticamente identicos). Exigir
# um minimo de graus de liberdade residuais evita o colapso; abaixo do
# minimo, o modelo daquela classe e' pulado (mesmo tratamento que amostras
# insuficientes) em vez de produzir um UCL que nao significa nada.
_MIN_Q_RESIDUAL_DF = 2


class DDSimca:
    """Data-Driven SIMCA: per-class one-class classifier via PCA.

    For each class, trains an independent PCA model. Duas coisas sao
    reportadas por eixo, mas a DECISAO de aceitar/rejeitar usa a
    ESTATISTICA COMBINADA do metodo original (corrigido em 2026-08-08 —
    ver nota abaixo):

        T2_UCL, Q_UCL — limites POR EIXO, so' para diagnostico/plotagem
                  individual (nao usados para aceitar/rejeitar):
                    'empirical'  : (1-alpha) percentile of training T2
                    'theoretical': Tracy-Young-Mason (F-distribution)
                    'chi2'       : chi2(1-alpha, n_components)
                  ('ucl_method' controla so' o T2_UCL exibido; Q_UCL e'
                  sempre chi2 Jackson-Mudholkar via mean/var, como antes.)

        f, f_crit — a DISTANCIA COMBINADA de fato usada em predict():
                    h0, q0 = media de T2_train, Q_train (respectivamente)
                    Nh, Nq = graus de liberdade estimados DOS DADOS via
                             metodo dos momentos (Jackson & Mudholkar
                             1979): N = 2*(media/desvio)^2 -- o "data-
                             driven" que da nome ao metodo, aplicado aos
                             DOIS eixos, nao so' a Q.
                    f       = (T2/h0)*Nh + (Q/q0)*Nq
                    f_crit  = chi2.ppf(1-alpha, Nh+Nq)
                  Um novo objeto e' aceito se f <= f_crit.

    CORRIGIDO em 2026-08-08 (achado por auditoria de figuras + pesquisa de
    literatura atualizada): a versao anterior aceitava um objeto se
    T2<=T2_UCL **e** Q<=Q_UCL independentemente -- uma regiao retangular,
    nao a elipse/reta combinada do metodo publicado. Com alpha independente
    em cada eixo, a taxa de rejeicao conjunta efetiva era ~1-(1-alpha)^2
    (~0.0975 para alpha=0.05), quase o dobro do alpha nominal declarado.
    A formula f/f_crit acima e' a Eq. (3)-(4) de Kucheryavskiy, Rodionova &
    Pomerantsev (2024) -- ver referencia completa abaixo -- reproduzida
    exatamente (Nq/Nh la' chamados N_q/N_h, h0/q0 la' chamados h0/q0).

    Referencias:
        Rodionova O.Y. & Pomerantsev A.L. (2020). Popular decision rules in
        SIMCA: critical review. J. Chemometrics 200:103958.
        Kucheryavskiy S., Rodionova O. & Pomerantsev A. (2024). A
        comprehensive tutorial on Data-Driven SIMCA: theory and
        implementation in web. J. Chemometrics 38(7):e3556.
    """

    def __init__(self, n_components: int = 3, alpha: float = 0.05,
                 ucl_method: str = "empirical"):
        self.n_components = n_components
        self.alpha = alpha
        self.ucl_method = ucl_method
        self._modelos: Dict[str, Dict[str, Any]] = {}
        self._classes: np.ndarray = np.array([], dtype=str)

    @staticmethod
    def _outliers_robustos_mad(valores: np.ndarray,
                               limiar: float = 3.5) -> np.ndarray:
        """Indices sinalizados como possiveis outliers via z-score
        modificado (Iglewicz & Hoaglin 1993): M_i = 0.6745*(x_i-mediana)/MAD.

        Kucheryavskiy, Rodionova & Pomerantsev (2024) recomendam
        explicitamente estimadores ROBUSTOS (mediana/IQR, nao media/desvio)
        para a DETECCAO de outliers no treino, revertendo para os
        estimadores classicos so' DEPOIS de remover o que for encontrado
        ("Once all outliers have been removed, it is recommended to revert
        to the classic estimates for further calculations").

        Aqui SO' sinaliza (nunca remove automaticamente): com nc=3-4
        amostras puras de treino -- o regime real deste projeto -- excluir
        uma amostra pode derrubar o modelo inteiro abaixo do minimo de
        graus de liberdade (`_MIN_Q_RESIDUAL_DF`). Remocao automatica seria
        arriscada demais com um treino ja tao escasso; um AVISO deixa a
        decisao (investigar a replica, ou aceitar o risco) com o usuario,
        em vez de o software decidir sozinho o que descartar.

        `limiar=3.5` e' o valor recomendado pelos autores do metodo.
        MAD=0 (valores identicos -- treino degenerado ou n<2) devolve
        nenhum outlier, nao ZeroDivisionError/NaN.

        LIMITACAO HONESTA (medida, nao suposta): com nc=3, T2/Q_train ja
        sao inerentemente instaveis (so' 2 graus de liberdade residuais,
        `_MIN_Q_RESIDUAL_DF`) mesmo sem outlier real algum -- o "sinal" que
        este detector ve pode ser so' o ruido de amostragem do proprio
        regime de poucas amostras. Aplicado nos dois eixos (T2 e Q, uniao
        dos dois), a taxa de falso positivo medida chega a ~10% mesmo em
        n=20 (3 de 30 seeds testadas). Interpretar o aviso como "vale
        conferir esta replica", nunca como "esta replica esta errada".
        """
        valores = np.asarray(valores, dtype=float)
        if valores.size < 3:
            return np.array([], dtype=int)
        mediana = float(np.median(valores))
        mad = float(np.median(np.abs(valores - mediana)))
        if mad <= 0:
            return np.array([], dtype=int)
        z_mod = 0.6745 * (valores - mediana) / mad
        return np.where(np.abs(z_mod) > limiar)[0]

    @staticmethod
    def _f_distance(T2: np.ndarray, Q: np.ndarray,
                    m: Dict[str, Any]) -> np.ndarray:
        """Distancia combinada f = (T2/h0)*Nh + (Q/q0)*Nq (Eq. 3 de
        Kucheryavskiy/Rodionova/Pomerantsev 2024) -- a estatistica que de
        fato decide aceitar/rejeitar, substituindo o teste retangular
        independente T2<=UCL e Q<=UCL. Delega para
        `chemometric_stats.distancia_combinada` (achado A3 da auditoria de
        2026-08-07: `dominio_aplicabilidade` reimplementava a mesma regra de
        forma independente; unificado numa so' fonte de verdade). Mantida
        como metodo (em vez de chamar `distancia_combinada` direto nos usos
        externos) para preservar a MESMA chamada em predict(), score_matrix()
        e nos usos externos (sensibilidade_ddsimca_logo, resumo do
        pipeline)."""
        return distancia_combinada(T2, Q, m["h0"], m["q0"], m["Nh"], m["Nq"])

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

    @staticmethod
    def _q_residuals_loo(Xc: np.ndarray, n_comp: int) -> np.ndarray:
        """Q-residuo leave-one-out (jackknife) de cada amostra de treino.

        PCA ajustada em TODAS as nc amostras reconstroi cada uma delas de
        forma otimista: a propria amostra ajudou a definir o subespaco que
        depois a reconstroi. Com nc pequeno frente a p (regime deste
        projeto: poucos puros por especie, espectros de milhares de
        variaveis), esse viés faz Q_train colapsar perto de zero -- e o UCL
        derivado dele rejeita qualquer amostra genuinamente nova (achado de
        auditoria adversarial, 2026-07-19).

        Aqui, o Q de cada amostra i e' medido contra um modelo ajustado nas
        OUTRAS nc-1 amostras (i excluida) -- a mesma logica de validacao
        cruzada, aplicada dentro do proprio calculo do limite. Remove o
        viés estruturalmente, sem depender de escolher um limiar de graus
        de liberdade "grande o bastante" (nenhum limiar resolve o viés
        in-sample; so' excluir a amostra do seu proprio ajuste resolve).

        Custo: nc ajustes extras de PCA por classe -- aceitavel porque nc e'
        pequeno justamente no regime em que isso importa.
        """
        nc = Xc.shape[0]
        Q = np.empty(nc)
        idx = np.arange(nc)
        for i in range(nc):
            Xtr = Xc[idx != i]
            n_comp_i = min(n_comp, Xtr.shape[0] - 1, Xtr.shape[1])
            if n_comp_i < 1:
                Q[i] = 0.0
                continue
            pca_i = PCA(n_components=n_comp_i)
            pca_i.fit(Xtr)
            xi = Xc[i:i + 1]
            t_i = pca_i.transform(xi)
            x_rec = pca_i.inverse_transform(t_i)
            Q[i] = float(np.sum((xi - x_rec) ** 2))
        return Q

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DDSimca":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=str)
        self._classes = np.unique(y)
        self._modelos = {}
        for cls in self._classes:
            Xc = X[y == cls]
            nc = len(Xc)
            n_comp = min(self.n_components, nc - 1, Xc.shape[1])
            # Se o n_comp cabivel deixa menos que _MIN_Q_RESIDUAL_DF graus de
            # liberdade residuais, reduz n_comp para preservar o minimo --
            # sem isso, Q_train colapsa para ~0 (PCA reconstroi o treino
            # quase exatamente) e o UCL derivado dele nao significa nada.
            if n_comp >= 1 and (nc - n_comp) < _MIN_Q_RESIDUAL_DF:
                n_comp = nc - _MIN_Q_RESIDUAL_DF
            if n_comp < 1:
                log.warning(
                    "[DDSimca] Class '%s': insufficient samples (n=%d) for a "
                    "non-degenerate Q-residual estimate (needs n >= "
                    "n_components + %d) — model skipped.",
                    cls, nc, _MIN_Q_RESIDUAL_DF)
                continue
            pca = PCA(n_components=n_comp)
            T = pca.fit_transform(Xc)
            var_t = T.var(axis=0, ddof=1)
            var_t[var_t == 0] = 1.0
            T2_train = np.sum((T ** 2) / var_t, axis=1)

            # Q_train usa residuo leave-one-out (ver _q_residuals_loo) em vez
            # do residuo in-sample -- o in-sample colapsa para ~0 quando
            # nc << p, porque a propria amostra ajudou a ajustar a PCA que
            # depois a reconstroi.
            Q_train = self._q_residuals_loo(Xc, n_comp)

            t2_ucl = self._compute_t2_ucl(T2_train, nc, n_comp)
            q_ucl  = q_residuos_limite(Q_train, self.alpha)

            # Estatistica combinada (ver docstring da classe): h0/q0/Nh/Nq
            # data-driven a partir de T2_train/Q_train, f_crit por chi2 com
            # Nf=Nh+Nq graus de liberdade. E' o que predict() usa de fato.
            h0, Nh = media_e_dof_momentos(T2_train)
            q0, Nq = media_e_dof_momentos(Q_train)
            f_crit = float(chi2.ppf(1 - self.alpha, Nh + Nq))

            # Diagnostico robusto (mediana/MAD, Iglewicz & Hoaglin 1993):
            # SO' sinaliza replicas de treino atipicas, NUNCA remove
            # sozinho -- com nc=3-4 (regime real deste projeto), excluir uma
            # amostra pode derrubar o modelo abaixo do minimo de graus de
            # liberdade. Ver docstring de _outliers_robustos_mad.
            idx_out_t2 = self._outliers_robustos_mad(T2_train)
            idx_out_q  = self._outliers_robustos_mad(Q_train)
            idx_out = sorted(set(idx_out_t2) | set(idx_out_q))
            if idx_out:
                log.warning(
                    "[DDSimca] Classe '%s': %d amostra(s) de treino "
                    "atipica(s) (indices %s de %d, deteccao robusta "
                    "mediana/MAD). Nao removidas automaticamente -- "
                    "considere investigar essas replicas.",
                    cls, len(idx_out), idx_out, nc)

            self._modelos[cls] = {
                "pca":      pca,
                "var_t":    var_t,
                "T2_ucl":   t2_ucl,
                "Q_ucl":    q_ucl,
                "h0":       h0,
                "q0":       q0,
                "Nh":       Nh,
                "Nq":       Nq,
                "f_crit":   f_crit,
                "T_train":  T,
                "T2_train": T2_train,
                "Q_train":  Q_train,
                "n_train":  nc,
                "n_comp":   n_comp,
                "outliers_treino": idx_out,
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
        """T2, Q, versoes normalizadas por eixo (T2/UCL, Q/UCL — so'
        diagnostico) e a distancia combinada f/f_crit (o que decide
        aceitar/rejeitar) por classe."""
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
                "f":        self._f_distance(T2, Q, m),
                "f_crit":   m["f_crit"],
                "h0":       m["h0"],
                "q0":       m["q0"],
                "Nh":       m["Nh"],
                "Nq":       m["Nq"],
                "T_train":  m["T_train"],
                "Q_train":  m["Q_train"],
                "n_train":  m["n_train"],
                "n_comp":   m["n_comp"],
                "outliers_treino": m["outliers_treino"],
            }
        return res

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns: class name | 'Ambiguo' | 'Desconhecido'.

        Aceita via distancia combinada f<=f_crit (ver docstring da classe),
        nao mais o teste retangular independente por eixo."""
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
                f = self._f_distance(T2, Q, m)
                if f[0] <= m["f_crit"]:
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
            except (ValueError, np.linalg.LinAlgError) as _e_lda:
                # LDA falha tipicamente com matriz de dispersao intra-classe
                # singular (classe com poucas/colineares amostras) -- cai p/
                # o fallback PLS2 (menos otimo mas correto p/ multiclasse).
                # Registrado pois muda o eixo y do OPLS-DA/S-Plot silenciosamente.
                log.warning("OPLS-DA: LDA falhou (%s); usando fallback PLS2.",
                           _e_lda)
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


def sensibilidade_ddsimca_logo(
    X_puros: np.ndarray,
    grupos_puros: np.ndarray,
    *,
    n_components: int,
    alpha: float = 0.05,
    ucl_method: str = "empirical",
) -> Dict[str, Any]:
    """Sensibilidade DD-SIMCA honesta por leave-one-group-out (LOGO).

    Re-substituicao (treinar o modelo one-class nos puros e medir a
    sensibilidade nos MESMOS puros) infla o valor ate ~100% e nao prova
    nada: mede o modelo reconhecendo dados que ele ja viu. Com poucos grupos
    de replica fisica (``mae_id``), LOGO e a unica estimativa defensavel:
    para cada grupo retira-se um grupo inteiro, treina-se o modelo nos
    demais e verifica-se se as amostras retidas caem na regiao de aceitacao.

    O numero de componentes de cada dobra e limitado internamente pelo
    ``DDSimca`` a ``min(n_components, n_treino - 1, n_variaveis)``, com reducao
    adicional para preservar graus de liberdade residuais minimos (ver
    ``_MIN_Q_RESIDUAL_DF`` em classificadores.py). Dobras cujo treino nao
    comporta um modelo nao-degenerado sao puladas (o mesmo caminho de "classe
    ausente" ja tratado abaixo) em vez de produzir uma sensibilidade
    artificialmente baixa por colapso numerico do limite de Q -- se
    ``n_grupos_validos`` vier menor que ``n_grupos``, e' porque dobras foram
    puladas por dados insuficientes, nao porque o modelo rejeitou tudo.

    Parameters
    ----------
    X_puros : (n, p) espectros das amostras PURAS de UMA classe.
    grupos_puros : (n,) identificador de replica (mae_id) de cada amostra pura.
    n_components, alpha, ucl_method : hiperparametros do ``DDSimca``, iguais
        aos do modelo final para comparabilidade.

    Returns
    -------
    dict com chaves:
        sensibilidade (float|nan), n_grupos (int), n_grupos_validos (int),
        n_amostras (int), aviso (str|None).
    O valor tende a cair abaixo de 100% — esse e o objetivo, nao um defeito.
    """
    X_puros = np.asarray(X_puros, dtype=float)
    grupos = np.asarray(grupos_puros)
    grupos_unicos = np.unique(grupos)
    n_grupos = int(len(grupos_unicos))
    resultado: Dict[str, Any] = {
        "sensibilidade": float("nan"),
        "n_grupos": n_grupos,
        "n_grupos_validos": 0,
        "n_amostras": int(len(X_puros)),
        "aviso": None,
    }
    if n_grupos < 2:
        resultado["aviso"] = (
            f"Sensibilidade nao estimavel por LOGO: apenas {n_grupos} grupo(s) "
            "de replica pura. Sem replicacao independente nao ha validacao "
            "possivel; re-substituicao seria enganosa."
        )
        return resultado

    aceitos: List[bool] = []
    validos = 0
    for g in grupos_unicos:
        treino = grupos != g
        teste = grupos == g
        if int(treino.sum()) < 2 or int(teste.sum()) == 0:
            continue
        modelo = DDSimca(n_components=n_components, alpha=alpha,
                         ucl_method=ucl_method)
        modelo.fit(X_puros[treino], np.array(["_c"] * int(treino.sum())))
        res = modelo.score_matrix(X_puros[teste])
        if "_c" not in res:   # classe pulada (puros de treino insuficientes)
            continue
        m = res["_c"]
        aceito = np.asarray(m["f"]) <= m["f_crit"]
        aceitos.extend(bool(a) for a in aceito)
        validos += 1

    resultado["n_grupos_validos"] = validos
    if validos < 2 or not aceitos:
        resultado["aviso"] = (
            f"Sensibilidade LOGO inconclusiva: {validos} dobra(s) valida(s) de "
            f"{n_grupos} grupos (puros por grupo insuficientes para ajustar o "
            "modelo). Interpretar como nao-validado."
        )
        return resultado

    resultado["sensibilidade"] = float(np.mean(aceitos))
    if n_grupos < 10:
        resultado["aviso"] = (
            f"Sensibilidade estimada por LOGO com apenas {n_grupos} grupos de "
            "replica. Incerteza alta; IC bootstrap nao e confiavel neste "
            "regime. Interpretar como exploratoria."
        )
    return resultado


def sensibilidade_ddsimca_pcv(
    X_puros: np.ndarray,
    grupos_puros: np.ndarray,
    *,
    n_components: int,
    alpha: float = 0.05,
    ucl_method: str = "empirical",
) -> Dict[str, Any]:
    """Sensibilidade DD-SIMCA por Procrustes Cross-Validation (PCV) --
    diagnostico COMPLEMENTAR ao LOGO (`sensibilidade_ddsimca_logo`), NUNCA
    um substituto.

    PCV (Kucheryavskiy, Zhilin, Rodionova & Pomerantsev -- ver referencias)
    gera um "PV-set" por reamostragem que se comporta estatisticamente como
    um conjunto de validacao independente, sem exigir mais amostras reais.
    Isso ajuda quando LOGO fica inconclusivo por FALTA DE DOBRAS validas
    (poucos grupos com >=2 puros cada) -- mas PCV nao fabrica variacao que
    nao existe nos dados: se todas as replicas puras de uma classe vem do
    MESMO grupo `mae_id` (`n_grupos==1`, o caso mais comum neste dataset),
    o PV-set so' pode reproduzir ruido de MEDICAO (variacao entre T1/T2/T3
    da mesma amostra fisica), nunca variacao ENTRE amostras fisicas
    diferentes -- a unica coisa que provaria generalizacao de autenticacao.
    Por isso este diagnostico e' SEMPRE rotulado como exploratorio e NUNCA
    substitui o aviso "nao validado" do LOGO quando `n_grupos<2`.

    O split de CV usado dentro do PCV respeita os grupos `mae_id` quando ha'
    2 ou mais (nao trata cada espectro como independente -- a mesma logica
    group-aware do resto do projeto). Com `n_grupos==1`, cai para
    leave-one-out por AMOSTRA individual (nao ha' estrutura de grupo a
    proteger quando so' existe 1 grupo; testado empiricamente que o split
    por grupo unico faz o PCV falhar -- ValueError de shape).

    Requer o pacote opcional `prcv` (`pip install
    guaraci-chemometrics[robusto]`); ausente, devolve `disponivel=False`
    sem lancar excecao.

    Returns
    -------
    dict com chaves: sensibilidade (float|nan), n_grupos (int),
    n_amostras (int), aviso (str|None), disponivel (bool).

    Referencias:
        Kucheryavskiy S., Zhilin S., Rodionova O. & Pomerantsev A. (2020).
        Procrustes cross-validation -- a bridge between cross-validation
        and independent validation sets. Anal. Chem. 92(17):11842-11850.
        Pomerantsev A.L. & Rodionova O.Y. (2021). Procrustes
        cross-validation of short datasets in PCA context. Talanta
        226:122104.
    """
    X_puros = np.asarray(X_puros, dtype=float)
    grupos = np.asarray(grupos_puros)
    grupos_unicos = np.unique(grupos)
    n_grupos = int(len(grupos_unicos))
    nc = len(X_puros)
    resultado: Dict[str, Any] = {
        "sensibilidade": float("nan"),
        "n_grupos": n_grupos,
        "n_amostras": int(nc),
        "aviso": None,
        "disponivel": True,
    }
    try:
        from prcv.methods import pcvpca
    except ImportError:
        resultado["disponivel"] = False
        resultado["aviso"] = (
            "Pacote opcional 'prcv' nao instalado -- diagnostico PCV "
            "indisponivel (pip install guaraci-chemometrics[robusto])."
        )
        return resultado

    n_comp_pv = min(n_components, nc - 1)
    if n_comp_pv < 1:
        resultado["aviso"] = f"Amostras insuficientes (n={nc}) para gerar PV-set."
        return resultado

    cv_split: Any
    if n_grupos >= 2:
        # Segmentos = grupos mae_id (preserva group-awareness dentro do PCV)
        _, indices = np.unique(grupos, return_inverse=True)
        cv_split = (indices + 1).astype(int)   # prcv espera segmentos >=1
    else:
        # 1 grupo so': nao ha estrutura a proteger, e o split por grupo
        # unico faz pcvpca falhar (ValueError de shape, verificado).
        cv_split = {"type": "loo"}

    try:
        Xpv = pcvpca(X_puros, ncomp=n_comp_pv, cv=cv_split)
    except Exception as e:  # noqa: BLE001 -- diagnostico auxiliar opcional;
        # qualquer falha do PCV (matriz mal condicionada, nc muito pequeno)
        # nao pode derrubar o resto do pipeline, so' reporta.
        resultado["aviso"] = f"PCV falhou: {e}"
        return resultado

    modelo = DDSimca(n_components=n_components, alpha=alpha,
                     ucl_method=ucl_method)
    modelo.fit(X_puros, np.array(["_c"] * nc))
    res = modelo.score_matrix(Xpv)
    if "_c" not in res:
        resultado["aviso"] = "Modelo nao ajustavel com estes puros (ver LOGO)."
        return resultado
    m = res["_c"]
    aceito = np.asarray(m["f"]) <= m["f_crit"]
    resultado["sensibilidade"] = float(np.mean(aceito))

    if n_grupos < 2:
        resultado["aviso"] = (
            "PCV com um unico grupo de replica pura: o PV-set reproduz so' "
            "ruido de MEDICAO (T1/T2/T3 da mesma amostra), nao variacao "
            "entre amostras fisicas diferentes. Nao e' evidencia de "
            "generalizacao -- interpretar como robustez a ruido "
            "instrumental, nunca como autenticacao validada."
        )
    elif n_grupos < 10:
        resultado["aviso"] = (
            f"PCV com {n_grupos} grupos de replica. Diagnostico "
            "exploratorio, complementar ao LOGO -- nao o substitui."
        )
    return resultado
