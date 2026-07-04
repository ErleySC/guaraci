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

from typing import Dict, List

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


# =========================================================================
#  Figuras de merito analiticas (calibracao multivariada, UM analito)
# =========================================================================

def figuras_merito_regressao(modelo: PLSRegression, X_cal: np.ndarray,
                              grupos_replicas: List[np.ndarray]
                              ) -> Dict[str, float]:
    """Figuras de merito analiticas para um modelo PLS de calibracao
    multivariada de um analito, seguindo Valderrama, Braga & Poppi (2009),
    Quim. Nova 32(5):1278-1287 ("Estado da arte de figuras de merito em
    calibracao multivariada").

    Usa o vetor de regressao b do modelo (tal que y_hat = b.(x-x_mean)+y_mean,
    no espaco JA PRE-PROCESSADO -- SNV/MSC/SG/MC etc. -- que e o "sinal" que o
    modelo de fato usa):

        Sensibilidade       SEN   = 1 / ||b||
        Sensib. analitica   gamma = SEN / delta_x
        Seletividade        SEL_i = |b.(x_i-xbar)| / (||b|| . ||x_i-xbar||)
                            (reportada como a media sobre as amostras)
        Limite de deteccao  LOD   = 3.3 * delta_x / SEN = 3.3 * delta_x * ||b||
        Limite de quantif.  LOQ   = 10  * delta_x / SEN = 10  * delta_x * ||b||

    delta_x (ruido instrumental/repetibilidade) e estimado empiricamente a
    partir de REPLICAS FISICAS do mesmo ponto amostral (ex.: T1/T2/T3 via
    mae_id) -- a forma mais rigorosa recomendada na literatura, em vez de uma
    especificacao generica do instrumento. Usa a variancia pooled por
    variavel espectral (estilo ANOVA / ISO 5725 de repetibilidade), agregada
    via RMS sobre as variaveis. Requer pelo menos um grupo com >=2 replicas;
    sem isso os campos dependentes de ruido voltam NaN (nao ha como estimar
    ruido instrumental sem medidas repetidas do mesmo ponto).

    `X_cal` e cada array de `grupos_replicas` devem estar no MESMO espaco
    pre-processado usado para ajustar `modelo` (aplicar so `.transform()`,
    nunca reajustar o preprocessador, nas replicas).
    """
    X_cal = np.asarray(X_cal, dtype=float)
    # .coef_ varia de forma (n_features, 1) ou (1, n_features) conforme a
    # versao do sklearn; com y de 1 coluna (um analito) o total de elementos
    # e sempre n_features, entao reshape(-1) e robusto a ambas convencoes.
    b = np.asarray(modelo.coef_, dtype=float).reshape(-1)
    norm_b = float(np.linalg.norm(b))

    resultado: Dict[str, float] = {
        "sensibilidade": float("nan"),
        "sensibilidade_analitica": float("nan"),
        "seletividade_media": float("nan"),
        "delta_x_ruido": float("nan"),
        "lod": float("nan"),
        "loq": float("nan"),
        "n_grupos_replicas": 0.0,
    }
    if norm_b < 1e-12:
        return resultado

    sen = 1.0 / norm_b
    resultado["sensibilidade"] = sen

    # Seletividade: |cos(angulo)| entre b e cada amostra centrada -- fracao
    # do sinal total de cada amostra que e "util" (colinear com a direcao
    # de calibracao), o resto e atribuido a interferentes/matriz.
    x_mean = X_cal.mean(axis=0)
    Xc = X_cal - x_mean
    norm_xi = np.linalg.norm(Xc, axis=1)
    norm_xi_seguro = np.where(norm_xi < 1e-12, 1.0, norm_xi)
    sel_i = np.abs(Xc @ b) / (norm_b * norm_xi_seguro)
    resultado["seletividade_media"] = float(np.mean(sel_i))

    # delta_x: variancia pooled (ANOVA) por variavel, a partir de replicas
    # fisicas -- soma dos quadrados dentro de cada grupo / soma dos graus de
    # liberdade, RMS sobre as variaveis.
    soma_ss = None
    soma_df = 0
    n_grupos_validos = 0
    for grupo in grupos_replicas:
        grupo = np.asarray(grupo, dtype=float)
        if grupo.shape[0] < 2:
            continue
        d = grupo - grupo.mean(axis=0, keepdims=True)
        ss = np.sum(d ** 2, axis=0)
        soma_ss = ss if soma_ss is None else soma_ss + ss
        soma_df += (grupo.shape[0] - 1)
        n_grupos_validos += 1
    resultado["n_grupos_replicas"] = float(n_grupos_validos)
    if soma_ss is not None and soma_df > 0:
        var_pooled = soma_ss / soma_df
        delta_x = float(np.sqrt(np.mean(var_pooled)))
        resultado["delta_x_ruido"] = delta_x
        if delta_x > 1e-12:
            resultado["sensibilidade_analitica"] = sen / delta_x
            resultado["lod"] = 3.3 * delta_x * norm_b
            resultado["loq"] = 10.0 * delta_x * norm_b

    return resultado


# =========================================================================
#  Dominio de Aplicabilidade (Applicability Domain, AD)
# =========================================================================

def dominio_aplicabilidade(pca, X_train: np.ndarray, X_new: np.ndarray,
                           alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Dominio de aplicabilidade via distancia ao modelo PCA/PLS, combinando
    as duas distancias complementares ja usadas para deteccao de outliers:

        Hotelling T2  -> distancia DENTRO do plano do modelo (leverage): quao
                         extrema a amostra e ao longo das direcoes de maior
                         variancia capturadas pela calibracao.
        Q-residuos    -> distancia ORTOGONAL ao plano (residuo espectral): quao
                         mal o modelo reconstroi a amostra (quimica nova, nao
                         vista no treino).

    Uma amostra nova esta DENTRO do dominio se T2 <= T2_limite E Q <= Q_limite,
    ambos os limites derivados EXCLUSIVAMENTE do conjunto de treino (mesma
    formula de Tracy-Young-Mason e chi2-Jackson-Mudholkar ja usadas no
    diagnostico de outliers). Amostras fora do dominio tem predicao pouco
    confiavel — a extrapolacao nao e garantida.

    Referencias: Jaworska, Nikolova-Jeliazkova & Aldenberg (2005), SAR QSAR
    Environ. Res. 16:445-466; Gadaleta et al. (2016), J. Chem. Inf. Model.
    A convencao T2+Q e o "AD baseado em leverage/residuo" padrao em
    espectroscopia (equivalente ao par distance-to-model do SIMCA).

    Parametros
    ----------
    pca      : modelo PCA ja ajustado no treino (precisa de .transform(),
               .components_ (k, p) e .mean_ (p,) — sklearn PCA satisfaz).
    X_train  : matriz de treino no MESMO espaco pre-processado do ajuste.
    X_new    : amostras novas a avaliar (mesmo pre-processamento).
    alpha    : nivel de significancia dos limites (default 0.05 -> 95%).

    Retorna dict com t2/q por amostra nova, os limites, e as mascaras
    booleanas dentro_t2 / dentro_q / dentro_dominio + a fracao dentro.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_new = np.asarray(X_new, dtype=float)
    T_train = np.asarray(pca.transform(X_train), dtype=float)
    T_new = np.asarray(pca.transform(X_new), dtype=float)
    P = np.asarray(pca.components_, dtype=float)          # (k, p)
    mean = np.asarray(pca.mean_, dtype=float)             # (p,)
    n, k = T_train.shape

    # T2 das amostras novas usando a variancia dos scores do TREINO (nunca a
    # das novas — senao o limite deixaria de ser um teste de extrapolacao).
    var_t = T_train.var(axis=0, ddof=1)
    var_t[var_t == 0] = 1.0
    t2_new = np.sum((T_new ** 2) / var_t, axis=1)
    t2_lim = hotelling_t2_limite(n, k, alpha)

    # Q-residuos: reconstrucao no espaco CENTRADO pela media do treino.
    q_train = q_residuos(X_train - mean, T_train, P)
    q_new = q_residuos(X_new - mean, T_new, P)
    q_lim = q_residuos_limite(q_train, alpha)

    dentro_t2 = t2_new <= t2_lim
    dentro_q = q_new <= q_lim
    dentro = dentro_t2 & dentro_q
    return {
        "t2": t2_new,
        "q": q_new,
        "t2_limite": np.asarray(t2_lim, dtype=float),
        "q_limite": np.asarray(q_lim, dtype=float),
        "dentro_t2": dentro_t2,
        "dentro_q": dentro_q,
        "dentro_dominio": dentro,
        "fracao_dentro": np.asarray(
            float(np.mean(dentro)) if dentro.size else float("nan"),
            dtype=float),
    }
