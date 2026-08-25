"""
classificadores.py — Classificadores quimiométricos avançados: DD-SIMCA
(one-class por classe) e OPLS-DA (deflação ortogonal + Gram-Schmidt).

Extraído de pipeline.py como parte da modularização (Fase H). Sem
acoplamento a Config — dependem só de numpy/scipy/sklearn e de
chemometric_stats.py (hotelling_t2_limit, q_residuals_limit). pipeline.py
reexporta estes nomes, então `pipeline.DDSimca(...)`,
`pipeline.OPLSDAWrapper(...)` etc. continuam funcionando sem alteração.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chi2
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from guaraci.chemometric_stats import (hotelling_t2_limit, q_residuals_limit,
                                       mean_and_dof_moments, combined_distance,
                                       q_residuals_loo)

log = logging.getLogger(__name__)

# Graus de liberdade residuais minimos exigidos apos a PCA (nc - n_comp).
# Com nc amostras de treino, PCA com n_comp = nc-1 componentes reconstroi
# o treino EXATAMENTE (Q_train ~= 0 para toda amostra, pois todos os graus
# de liberdade entre as nc amostras centradas foram consumidos). Isso nao
# e' ruido numerico: e' uma propriedade exata de PCA quando n_comp se
# aproxima de nc-1. q_residuals_limit() estima o UCL a partir de
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

    CORRIGIDO em 2026-08-16 (achado F1/A2-3 da auditoria de gate 0): Nh/Nq
    eram estimados por ESPECTRO (`fit(X, y)` usava `len(Xc)` direto), nao
    por AMOSTRA FISICA -- replicas tecnicas (T1/T2/T3 da mesma amostra)
    inflavam artificialmente os graus de liberdade do proprio limiar que
    decide aceitacao. E' o argumento central do projeto (vazamento de
    replica) nao aplicado ao calculo do limiar. `fit()` aceita `mae_id`
    opcional para calibrar h0/q0/Nh/Nq pela media de T2/Q por amostra
    fisica -- ver docstring de `fit()` para o mecanismo e a consequencia
    medida num dataset de referencia interno (Nh=Nq=1 para a maioria das
    classes, que so'
    tem 1 amostra pura independente).

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
        `chemometric_stats.combined_distance` (achado A3 da auditoria de
        2026-08-07: `applicability_domain` reimplementava a mesma regra de
        forma independente; unificado numa so' fonte de verdade). Mantida
        como metodo (em vez de chamar `combined_distance` direto nos usos
        externos) para preservar a MESMA chamada em predict(), score_matrix()
        e nos usos externos (ddsimca_logo_sensitivity, resumo do
        pipeline)."""
        return combined_distance(T2, Q, m["h0"], m["q0"], m["Nh"], m["Nq"])

    def _compute_t2_ucl(self, T2_train: np.ndarray, n: int, k: int) -> float:
        method = (self.ucl_method or "empirical").lower()
        if method == "empirical":
            if T2_train.size == 0:
                return float("inf")
            return float(np.percentile(T2_train, 100 * (1 - self.alpha)))
        if method == "theoretical":
            return hotelling_t2_limit(n, k, self.alpha)
        if method == "chi2":
            return float(chi2.ppf(1 - self.alpha, k))
        # fallback
        return float(np.percentile(T2_train, 100 * (1 - self.alpha)))

    @staticmethod
    def _q_residuals_loo(Xc: np.ndarray, n_comp: int) -> np.ndarray:
        """Q-residuo leave-one-out (jackknife) de cada amostra de treino.

        Delega para `chemometric_stats.q_residuals_loo`. A implementacao
        nasceu aqui (achado da auditoria adversarial de 2026-07-19) e foi
        promovida a funcao pura em 2026-08-17, quando se descobriu que
        `training_applicability_domain` precisava exatamente da mesma
        correcao -- manter duas copias da mesma regra estatistica foi o que
        permitiu que uma delas ficasse para tras (mesmo padrao do achado A3).
        Mantido como metodo para nao quebrar chamadores/testes existentes.
        """
        return q_residuals_loo(Xc, n_comp)

    @staticmethod
    def _media_por_grupo(valores: np.ndarray,
                         grupos: np.ndarray) -> np.ndarray:
        """Colapsa `valores` (1 por espectro) em 1 valor por `mae_id`
        distinto (media do grupo). Usado para calibrar h0/q0/Nh/Nq por
        AMOSTRA FISICA, nao por espectro -- ver nota em `fit()`."""
        _u, inv = np.unique(grupos, return_inverse=True)
        soma = np.bincount(inv, weights=valores)
        cont = np.bincount(inv)
        return soma / cont

    def fit(self, X: np.ndarray, y: np.ndarray,
            mae_id: Optional[np.ndarray] = None) -> "DDSimca":
        """Ajusta um modelo DD-SIMCA por classe.

        `mae_id` (opcional, mesmo comprimento de X/y): identificador de
        replica fisica (T1/T2/T3 da mesma amostra compartilham o mesmo
        `mae_id`). Quando fornecido, h0/q0/Nh/Nq (a calibracao do LIMIAR
        de aceitacao, ver docstring da classe) sao estimados a partir da
        MEDIA de T2/Q por `mae_id` -- isto e', um valor por AMOSTRA FISICA
        independente, nao um valor por espectro.

        POR QUE ISSO IMPORTA (achado F1/A2-3 da auditoria de 2026-08-16):
        sem `mae_id`, 3 replicas tecnicas da MESMA amostra sao tratadas
        como 3 observacoes independentes ao estimar Nh/Nq pelo metodo dos
        momentos -- exatamente o erro de vazamento de replica que o
        projeto existe para impedir, cometido no proprio calculo do
        limiar que decide aceitacao/rejeicao. Duplicar espectros da MESMA
        amostra fisica (sem adicionar amostra real nenhuma) infla Nh/Nq
        sem fundamento quando `mae_id` esta ausente; com `mae_id`, nao.

        CONSEQUENCIA HONESTA quando ha' so' 1 mae_id de treino por classe
        (regime comum em autenticacao one-class, em que so' se dispoe de
        um ponto de amostragem fisico genuino por classe): Nh=Nq=1.0 (o
        minimo que `mean_and_dof_moments` retorna para entrada degenerada
        de tamanho 1) -- NAO e' um bug desta funcao, e' a calibracao mais
        honesta possivel dado que so' existe 1 amostra fisica independente
        para calibrar contra. E' o mesmo movimento do P1 (sensibilidade
        LOGO honesta substituindo re-substituicao inflada): o numero fica
        mais largo/conservador, que e' o objetivo, nao um problema.

        Sem `mae_id` (None): comportamento anterior preservado (Nh/Nq
        estimados por espectro) -- necessario quando nao ha' identificador
        de replica disponivel (ex.: modo_entrada="imagem", B4-1 da mesma
        auditoria). `calibrado_por_amostra=False` fica marcado no modelo
        para que figuras/relatorios distingam os dois casos.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=str)
        mae_id = np.asarray(mae_id, dtype=str) if mae_id is not None else None
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
            q_ucl  = q_residuals_limit(Q_train, self.alpha)

            # Estatistica combinada (ver docstring da classe): h0/q0/Nh/Nq
            # data-driven a partir de T2_train/Q_train, f_crit por chi2 com
            # Nf=Nh+Nq graus de liberdade. E' o que predict() usa de fato.
            #
            # CALIBRACAO POR AMOSTRA FISICA (ver docstring de fit(), achado
            # F1/A2-3): com mae_id disponivel, h0/q0/Nh/Nq vem da MEDIA de
            # T2/Q por mae_id, nao dos nc espectros individuais -- replicas
            # tecnicas da mesma amostra nao inflam os graus de liberdade
            # estimados.
            mae_id_c = mae_id[y == cls] if mae_id is not None else None
            if mae_id_c is not None and len(np.unique(mae_id_c)) >= 1:
                n_grupos_calib = int(len(np.unique(mae_id_c)))
                T2_calib = self._media_por_grupo(T2_train, mae_id_c)
                Q_calib  = self._media_por_grupo(Q_train, mae_id_c)
                calibrado_por_amostra = True
            else:
                n_grupos_calib = nc
                T2_calib, Q_calib = T2_train, Q_train
                calibrado_por_amostra = False
            h0, Nh = mean_and_dof_moments(T2_calib)
            q0, Nq = mean_and_dof_moments(Q_calib)
            f_crit = float(chi2.ppf(1 - self.alpha, Nh + Nq))
            if calibrado_por_amostra and n_grupos_calib < 3:
                log.warning(
                    "[DDSimca] Classe '%s': limiar calibrado com apenas "
                    "%d amostra(s) fisica(s) independente(s) (mae_id). "
                    "Nh=%.2f, Nq=%.2f -- regiao de aceitacao larga/"
                    "conservadora por construcao, nao um defeito.",
                    cls, n_grupos_calib, Nh, Nq)

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
                "n_grupos_calibracao":    n_grupos_calib,
                "calibrado_por_amostra":  calibrado_por_amostra,
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

    def score_matrix(self, X: np.ndarray,
                     mask_treino: Optional[np.ndarray] = None,
                     y: Optional[np.ndarray] = None
                     ) -> Dict[str, Dict[str, Any]]:
        """T2, Q, versoes normalizadas por eixo (T2/UCL, Q/UCL — so'
        diagnostico) e a distancia combinada f/f_crit (o que decide
        aceitar/rejeitar) por classe.

        `mask_treino` + `y` (opcionais, ambos do tamanho de X): identificam
        quais linhas de X estavam no TREINO. Quando fornecidos, o Q dessas
        linhas vem do residuo LEAVE-ONE-OUT armazenado em `fit()`, nao do
        residuo in-sample recalculado por `_t2_q`.

        POR QUE (achado A1 da auditoria de gate 0, 2026-08-16): `fit()`
        calibra q0/Nq/f_crit a partir de `Q_train` LOO, mas `_t2_q`
        recalcula Q in-sample -- e uma amostra de treino reconstroi a si
        mesma de forma otimista, porque ajudou a definir a PCA que depois
        a reconstroi. Plotar pontos de treino via `_t2_q` contra uma
        fronteira derivada do q0 LOO poe pontos e fronteira em ESCALAS
        DIFERENTES. Medido no regime real do projeto (p=8192, nc=3-4
        puros/classe): o Q in-sample e' **10 a 15x menor** que o LOO --
        em eixo log, mais de uma decada de folga visual inventada.
        Ver scripts/medicoes/medir_ddsimca_loo_vs_insample.py.

        Impacto na DECISAO: nenhum no regime de producao -- a fracao de
        pontos de treino que muda de lado da fronteira foi medida em 0,0%
        com nc=3-4 (sobe a 7-12% com nc>=6). E' defeito de fidelidade da
        FIGURA, nao de numero; a correcao existe para a figura nao mentir
        sobre a folga.

        Sem os dois argumentos: comportamento anterior (tudo in-sample) --
        que continua CORRETO para amostras novas, que nao participaram do
        ajuste. So' o subconjunto de treino precisava do LOO.
        """
        X = np.asarray(X, dtype=float)
        usar_loo = mask_treino is not None and y is not None
        if usar_loo:
            mask_treino = np.asarray(mask_treino, dtype=bool)
            y = np.asarray(y, dtype=str)
        res: Dict[str, Dict[str, Any]] = {}
        for cls in self._classes:
            if cls not in self._modelos:
                continue
            m = self._modelos[cls]
            T2, Q = self._t2_q(X, cls)
            if usar_loo:
                # Linhas de treino DESTA classe, na mesma ordem em que
                # `fit()` as consumiu (indexacao booleana preserva ordem,
                # entao a k-esima linha aqui e' a k-esima de Q_train).
                idx_tr = np.where(mask_treino & (y == cls))[0]
                q_loo = np.asarray(m["Q_train"], dtype=float)
                if idx_tr.size == q_loo.size:
                    Q = Q.copy()
                    Q[idx_tr] = q_loo
                elif idx_tr.size:
                    # Desalinhamento (X diferente do usado em fit): nao
                    # adivinhar a correspondencia -- manter in-sample e
                    # avisar, em vez de trocar Q pelas linhas erradas.
                    log.warning(
                        "[DDSimca] score_matrix: %d linhas de treino da "
                        "classe '%s' contra %d valores de Q_train — X nao "
                        "confere com o usado em fit(); mantendo Q in-sample "
                        "para essa classe.", idx_tr.size, cls, q_loo.size)
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
                "n_grupos_calibracao":   m["n_grupos_calibracao"],
                "calibrado_por_amostra": m["calibrado_por_amostra"],
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

    @staticmethod
    def _alvo_continuo(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Alvo y continuo (1 coluna, centrado) usado para achar a direcao
        preditiva do OPLS via NIPALS PLS1. Captura a direcao de covariancia
        X-Y dominante. Para Y binario (1 coluna): a propria coluna.

        Para Y multiclasse (K colunas, one-hot): CORRIGIDO em 2026-08-07
        (achado A4 da auditoria metodologica -- ver
        AUDITORIA_METODOLOGICA_2026-08-07.md). A versao
        anterior usava o 1o escore de uma LinearDiscriminantAnalysis(X,
        y_int) como alvo -- nao e' o metodo publicado: Trygg & Wold (2002)
        definem OPLS para y binario/continuo; a extensao multiclasse
        publicada e' OPLS/O2PLS com Y multi-coluna via PLS2, nao um alvo
        derivado separadamente de X por um classificador supervisionado (a
        LDA usa so' a estrutura de classes em X, ignorando a covariancia
        X-Y que define o eixo preditivo do (O)PLS). Usa-se agora o escore Y
        da 1a variavel latente de um PLS2 ajustado em (X, Y) -- a direcao
        que capta a covariancia dominante X-Y entre TODAS as K classes
        simultaneamente, o caminho publicado. Using Y[:,0] (first class vs.
        rest) would silently bias the OPLS toward one class only — a
        methodological error for 14-class FT-NIR data; PLS2's y_scores_
        avoids that by construction (all K columns enter the covariance
        direction jointly).
        """
        if Y.ndim == 2 and Y.shape[1] > 1:
            _pls2 = PLSRegression(n_components=1, scale=False)
            _pls2.fit(X, Y)
            _ys = _pls2.y_scores_
            y = (np.asarray(_ys, dtype=float)[:, 0]
                 if _ys is not None else Y @ np.ones(Y.shape[1]))
        else:
            y = (Y[:, 0] if Y.ndim == 2 else Y.copy()).astype(float)
        return y - float(y.mean())

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "OPLSDAWrapper":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        y = self._alvo_continuo(X, Y)

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


def ddsimca_logo_sensitivity(
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
        # mae_id=grupos[treino] (achado F1/A2-3): o limiar interno desta
        # dobra tambem deve ser calibrado por amostra fisica, nao por
        # espectro -- senao o LOGO mede aceitacao contra um limiar com o
        # MESMO vies que o LOGO existe para corrigir.
        modelo.fit(X_puros[treino], np.array(["_c"] * int(treino.sum())),
                   mae_id=grupos[treino])
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


def ddsimca_pcv_sensitivity(
    X_puros: np.ndarray,
    grupos_puros: np.ndarray,
    *,
    n_components: int,
    alpha: float = 0.05,
    ucl_method: str = "empirical",
) -> Dict[str, Any]:
    """Sensibilidade DD-SIMCA por Procrustes Cross-Validation (PCV) --
    diagnostico COMPLEMENTAR ao LOGO (`ddsimca_logo_sensitivity`), NUNCA
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
    # mae_id=grupos (achado F1/A2-3): mesma razao do LOGO acima -- o limiar
    # usado para avaliar o PV-set deve ser calibrado por amostra fisica.
    modelo.fit(X_puros, np.array(["_c"] * nc), mae_id=grupos)
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
