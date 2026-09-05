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

import logging
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from scipy.stats import f as f_dist, chi2, t as t_dist
from sklearn.cross_decomposition import PLSRegression

log = logging.getLogger(__name__)

__all__ = [
    "vip_scores",
    "compute_selectivity_ratio",
    "martens_uncertainty_test",
    "hotelling_t2",
    "hotelling_t2_limit",
    "q_residuals",
    "q_residuals_loo",
    "q_residuals_limit",
    "mean_and_dof_moments",
    "combined_distance",
    "dmodx",
    "dmody",
    "explained_variance",
    "rmse_flat",
    "rpd_rer",
    "interpret_rpd",
    "regression_figures_of_merit",
    "faixa_decisao",
    "FAIXA_NAO_DETECTAVEL",
    "FAIXA_ZONA_CINZENTA",
    "FAIXA_QUANTIFICADO",
    "applicability_domain",
    "training_applicability_domain",
    "applicability_domain_new_samples",
    "diagnose_spectral_range",
    "expandir_binario_um_quente",
]


def expandir_binario_um_quente(Y: np.ndarray) -> np.ndarray:
    """Reconstroi a 2a coluna one-hot que falta quando `Y` vem de
    `LabelBinarizer.fit_transform` para EXATAMENTE 2 classes.

    Para 3+ classes o sklearn devolve uma coluna por classe (ndim=2,
    shape[1]>=2), mas para exatamente 2 classes ele devolve UMA coluna so'
    (shape (n,1), ndim=2 -- nao 1) por convencao binaria. Um check que so'
    testa `Y.ndim == 1` nunca dispara nesse caso, e downstream
    `np.argmax(Y, axis=1)` fica sempre 0: toda predicao colapsa na
    primeira classe, balanced_accuracy trava em exatamente 0.5 (achado do
    Passo 148, RETRATACAO do achado negativo do RMN -- ver
    `docs/VALIDACAO_PUBLICA.md` secao 2e). Este mesmo padrao de correcao
    (`Y.ndim == 1 or Y.shape[1] == 1`) existia duplicado, escrito a mao,
    em `avaliacao_modelos.PLSDAClassifier.fit`,
    `hsi_multiway.NPLSClassifier.fit`, `portao_correcao_sinal.py` e
    `pipeline._construir_ybin_e_yint` -- extraido aqui no Passo 156 para
    um unico ponto (dividia manutencao em 4 lugares que precisavam ficar
    sincronizados manualmente, exatamente o tipo de duplicacao que deixou
    `pipeline.py` desatualizado da correcao por tanto tempo).

    Idempotente: se `Y` ja' tem 2+ colunas (multi-classe), retorna sem
    alteracao."""
    Y = np.asarray(Y)
    if Y.ndim == 1 or Y.shape[1] == 1:
        Y_col = Y.reshape(-1, 1)
        Y = np.hstack([1 - Y_col, Y_col])
    return Y


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


def compute_selectivity_ratio(modelo: PLSRegression,
                                X: np.ndarray) -> np.ndarray:
    """Selectivity Ratio (SR) per Rajalahti et al. (2009),
    Chemom. Intell. Lab. Syst. 95:20-28; Kvalheim (2020),
    J. Chemometrics 34:e3211.

    CORRIGIDO em 2026-08-07 (achado A2 da auditoria metodologica — ver
    AUDITORIA_METODOLOGICA_2026-08-07.md): a projecao-alvo
    (target projection) usa o VETOR DE REGRESSAO NORMALIZADO b/||b||, NAO o
    primeiro peso PLS w1 -- os dois so' coincidem quando o modelo tem 1
    variavel latente. A propriedade que define o metodo (Rajalahti et al.
    2009, Sec. 2.2): o escore projetado t_tp e' PROPORCIONAL ao vetor de
    valores preditos y_hat. Medido com w1 (versao anterior):
    corr(t_tp, y_hat) caia para ~0.92 com >=2 LVs (deveria ser 1.000 exato)
    e o ranking de variaveis selecionadas divergia (Jaccard@20 ~ 0.39 em
    cenario multi-interferente com 3+ LVs) — ver
    scripts/medicoes/medir_sr_ranking.py.

    Para cada variavel j, decompoe X_j em parte explicada pela projecao
    alvo e residuo:
        w_tp   = b / ||b||          (vetor de regressao normalizado)
        t_tp   = X @ w_tp           (target projection scores, prop. a y_hat)
        p_tp_j = (t_tp^T * X_j) / (t_tp^T * t_tp)
        SR_j   = Var(t_tp * p_tp_j) / Var(X_j - t_tp * p_tp_j)

    Y multi-coluna (one-hot, classificacao multiclasse): o metodo publicado
    e' definido para y de 1 coluna. Aqui aplica-se a formula EXATA a cada
    coluna (problema one-vs-rest) independentemente e agrega por MAXIMO
    entre classes -- uma variavel e' reportada como seletiva se discrimina
    PELO MENOS uma classe (mesmo espirito da agregacao multi-saida ja usada
    em `martens_uncertainty_test`, nesta mesma secao do modulo).

    Complementa o VIP: SR e mais sensivel a variaveis com correlacao
    direcional com Y no componente preditivo; VIP integra todos os LVs.
    Concordancia entre VIP >= 1 e SR alto reforca a relevancia.
    """
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    # .coef_ e' (n_targets, n_features) no sklearn atual; versoes antigas
    # usavam a convencao transposta -- normaliza pelo eixo que bate com p.
    coef = np.asarray(modelo.coef_, dtype=float)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    if coef.shape[1] != p and coef.shape[0] == p:
        coef = coef.T
    n_saidas = coef.shape[0]

    sr_por_saida = np.zeros((n_saidas, p))
    for k in range(n_saidas):
        b = coef[k]
        norm_b = float(np.linalg.norm(b))
        if norm_b < 1e-12:
            continue
        w_tp = b / norm_b

        t_tp = X @ w_tp                     # (n,) -- proporcional a y_hat
        tt = float(t_tp @ t_tp)
        if tt < 1e-12:
            continue

        p_tp   = (t_tp @ X) / tt            # (p,) — target projection loadings
        X_tp   = np.outer(t_tp, p_tp)       # (n, p) — target-projected X
        X_res  = X - X_tp                   # (n, p) — residual

        var_tp  = X_tp.var(axis=0, ddof=1)
        var_res = X_res.var(axis=0, ddof=1)
        var_res[var_res < 1e-12] = 1e-12
        sr_por_saida[k] = var_tp / var_res

    return sr_por_saida.max(axis=0) if n_saidas > 1 else sr_por_saida[0]


def martens_uncertainty_test(
        X: np.ndarray, Y: np.ndarray, n_components: int,
        cv_indices: List[Tuple[np.ndarray, np.ndarray]],
        coef_completo: np.ndarray,
        alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Teste de incerteza de Martens (Martens & Martens, 2000) para os
    coeficientes de regressao PLS, via jackknifing group-aware.

    Reajusta um PLSRegression(n_components, scale=False) em cada fold de
    `cv_indices` (a MESMA CV group-aware ja usada p/ selecao de LVs -- sem
    recalcular splits novos), coleta os coeficientes de regressao b_i por
    fold, e estima a variancia jackknife por variavel:

        var_j = (n_folds - 1)/n_folds * sum_i (b_i,j - b_bar_j)^2
        t_j   = b_completo_j / sqrt(var_j)

    t_j e' comparado a distribuicao t de Student com (n_folds - 1) graus de
    liberdade -- um TESTE DE HIPOTESE FORMAL (p-valor) de significancia por
    variavel, mais rigoroso que VIP/Selectivity Ratio (medidas de
    MAGNITUDE, sem p-valor associado).

    Y pode ter multiplas colunas (Y_bin one-hot, classificacao multiclasse)
    -- nesse caso o coeficiente e' uma matriz (k, p) (convencao sklearn
    PLSRegression.coef_) e o resultado agrega por MAXIMO |t| entre as
    classes: uma variavel e' reportada como significativa se discrimina
    PELO MENOS uma classe (mesmo espirito da agregacao multi-saida ja
    embutida no VIP, que soma a contribuicao de todas as colunas de Y).

    Ref: Martens, H. & Martens, M. (2000), "Modified Jack-knife Estimation
    of Parameter Uncertainty in Bilinear Modelling (PLSR)", Food Quality
    and Preference 11:5-16.

    Retorna dict com t_valores/p_valores/significativo (um por variavel,
    ja agregados entre classes se multi-saida), n_folds_validos e
    coef_medio_folds (para diagnostico).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    coef_completo = np.asarray(coef_completo, dtype=float)
    if coef_completo.ndim == 1:
        coef_completo = coef_completo.reshape(1, -1)
    p = coef_completo.shape[1]

    coefs_fold: List[np.ndarray] = []
    for tr, _va in cv_indices:
        try:
            modelo_fold = PLSRegression(n_components=n_components, scale=False)
            modelo_fold.fit(X[tr], Y[tr])
            b_fold = np.asarray(modelo_fold.coef_, dtype=float)
            if b_fold.ndim == 1:
                b_fold = b_fold.reshape(1, -1)
            if b_fold.shape == coef_completo.shape:
                coefs_fold.append(b_fold)
        except (ValueError, np.linalg.LinAlgError):
            # Fold degenerado (ex.: classe ausente no treino, matriz singular)
            # -- descartado; n_folds_validos abaixo reflete a perda. Excecoes
            # fora deste tipo (ex.: bug real) NAO sao engolidas.
            continue

    n_folds = len(coefs_fold)
    if n_folds < 3:
        # Jackknife precisa de >= 3 folds validos p/ estimar variancia com
        # sentido estatistico -- sem isso, retorna NaN em vez de um numero
        # enganoso (ex.: variancia de 1 unica observacao e' sempre 0).
        return {
            "t_valores": np.full(p, np.nan),
            "p_valores": np.full(p, np.nan),
            "significativo": np.zeros(p, dtype=bool),
            "n_folds_validos": np.asarray(n_folds),
            "coef_medio_folds": np.full(p, np.nan),
        }

    B = np.stack(coefs_fold, axis=0)          # (n_folds, k, p)
    b_bar = B.mean(axis=0)                    # (k, p)
    var_jk = ((n_folds - 1) / n_folds) * np.sum((B - b_bar) ** 2, axis=0)  # (k, p)
    se_jk = np.sqrt(var_jk)
    se_jk_seguro = np.where(se_jk < 1e-12, 1.0, se_jk)

    t_por_classe = coef_completo / se_jk_seguro           # (k, p)
    t_por_classe = np.where(se_jk < 1e-12, 0.0, t_por_classe)
    p_por_classe = 2.0 * t_dist.sf(np.abs(t_por_classe), df=n_folds - 1)  # (k, p)

    idx_max = np.argmax(np.abs(t_por_classe), axis=0)     # (p,) -- classe dominante
    t_valores = t_por_classe[idx_max, np.arange(p)]
    p_valores = p_por_classe[idx_max, np.arange(p)]

    return {
        "t_valores": t_valores,
        "p_valores": p_valores,
        "significativo": p_valores < alpha,
        "n_folds_validos": np.asarray(n_folds),
        "coef_medio_folds": b_bar[idx_max, np.arange(p)],
    }


def hotelling_t2(T: np.ndarray) -> np.ndarray:
    var_t = T.var(axis=0, ddof=1)
    var_t[var_t == 0] = 1.0
    return np.sum((T ** 2) / var_t, axis=1)


def hotelling_t2_limit(n: int, k: int, alpha: float = 0.05) -> float:
    """Hotelling T2 upper control limit — Tracy, Young & Mason (1992),
    Technometrics 34(1):46-53, **Phase II** (new/future observations,
    F-distribution).

        T2_UCL = k * (n - 1) * (n + 1) / (n * (n - k)) * F_(alpha, k, n - k)

    Replaces the approximation (k(n-1)/(n-k))*F that underestimated the limit
    by ~5-10% for n<30 (causing false outliers in small datasets).

    CORRIGIDO em 2026-08-07 (achado A5 da auditoria metodologica — ver
    AUDITORIA_METODOLOGICA_2026-08-07.md): a docstring
    anterior afirmava que a formula valia "for both observations within
    the calibration set and new observations". Isso contradiz o proprio
    artigo citado: TYM (1992) e' precisamente o trabalho que estabelece
    que os dois casos usam distribuicoes DIFERENTES -- Fase I (amostras do
    proprio conjunto de treino) usa distribuicao **Beta**
    (T2 ~ ((n-1)^2/n) * Beta(k/2, (n-k-1)/2)); Fase II (amostras novas,
    a formula acima) usa **F**. Esta funcao implementa SO' a de Fase II.

    Onde e' aplicada em contexto de Fase I neste codebase
    (`training_applicability_domain`, `figuras.fig3_outliers`), o erro
    numerico medido (razao limite-F / limite-Beta) e' pequeno para os
    tamanhos de amostra tipicos do projeto (~1.01-1.03x com n~300; sobe a
    ~2-3x so' com n<20) — ver scripts/medicoes/medir_achados.py. Nao
    corrigido nesta rodada (impacto real medido como baixo); se usada com
    n pequeno como limite de FASE I, considerar o limite Beta exato.
    """
    if n - k <= 0:
        log.warning("Hotelling T2: n=%d too small for k=%d LVs.", n, k)
        return float("inf")
    if n < 3 * k:
        log.warning("Hotelling T2: n=%d < 3k=%d. Limit may be imprecise "
                    "(wide confidence interval).", n, 3 * k)
    return float(((k * (n - 1) * (n + 1)) / (n * (n - k)))
                  * f_dist.ppf(1 - alpha, k, n - k))


def q_residuals(X: np.ndarray, T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.sum((X - T @ P) ** 2, axis=1)


def q_residuals_loo(X: np.ndarray, n_comp: int) -> np.ndarray:
    """Q-residuo leave-one-out (jackknife) de cada amostra de treino.

    Uma PCA ajustada em TODAS as n amostras reconstroi cada uma delas de
    forma otimista: a propria amostra ajudou a definir o subespaco que
    depois a reconstroi. Com n pequeno frente a p (regime deste projeto:
    espectros de milhares de variaveis), esse vies faz Q_train colapsar
    perto de zero -- e qualquer limite derivado dele rejeita amostras
    genuinamente novas, mesmo vindas da mesma distribuicao do treino.

    Aqui o Q de cada amostra i e' medido contra um modelo ajustado nas
    OUTRAS n-1 amostras. Remove o vies estruturalmente; nenhum ajuste de
    graus de liberdade resolve, porque o problema esta na estimativa, nao
    no limiar.

    Origem: `DDSimca._q_residuals_loo` (auditoria adversarial 2026-07-19,
    CLAUDE.md P1). Promovida a funcao pura em 2026-08-17 ao se descobrir
    que `training_applicability_domain` -- o caminho que roda em producao
    em predicao.py -- tinha o mesmo vies e nao havia recebido a correcao
    (ver scripts/medicoes/medir_ad_vies_insample.py). DDSimca delega para ca'
    em vez de manter a segunda copia.

    Custo: n ajustes extras de PCA. Aceitavel porque n e' pequeno
    justamente no regime em que este vies importa.
    """
    from sklearn.decomposition import PCA   # local: evita import pesado no topo

    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    Q = np.empty(n, dtype=float)
    idx = np.arange(n)
    for i in range(n):
        X_tr = X[idx != i]
        n_comp_i = min(n_comp, X_tr.shape[0] - 1, X_tr.shape[1])
        if n_comp_i < 1:
            Q[i] = 0.0
            continue
        pca_i = PCA(n_components=n_comp_i).fit(X_tr)
        xi = X[i:i + 1]
        Q[i] = float(np.sum((xi - pca_i.inverse_transform(
            pca_i.transform(xi))) ** 2))
    return Q


def q_residuals_limit(q: np.ndarray, alpha: float = 0.05) -> float:
    # Guarda explicita p/ array vazio ANTES de mean()/var(): numpy retorna
    # NaN com RuntimeWarning nesse caso, e NaN <= 0 e' sempre False em Python
    # (nunca aciona o fallback abaixo) -- sem isso, q vazio silenciosamente
    # devolvia NaN em vez de 0.0 (achado pela rede de seguranca de testes).
    if q.size == 0:
        return 0.0
    media = float(q.mean()); var = float(q.var())
    if var <= 0 or media <= 0:
        return float(np.percentile(q, (1 - alpha) * 100))
    g = var / (2 * media); h = 2 * (media ** 2) / var
    return float(g * chi2.ppf(1 - alpha, h))


def mean_and_dof_moments(valores: np.ndarray) -> Tuple[float, float]:
    """Media e graus de liberdade (N) por metodo dos momentos (Box 1954,
    aproximado por Jackson & Mudholkar 1979 para Q-residuos): N =
    2*(media/desvio)^2. E' o "data-driven" que da nome ao metodo DD-SIMCA
    (Kucheryavskiy, Rodionova & Pomerantsev, 2024), usado para converter uma
    estatistica de distancia (T2 ou Q) numa aproximacao chi-quadrado com
    graus de liberdade estimados dos proprios dados.

    Compartilhada entre `classificadores.DDSimca` e
    `training_applicability_domain` (achado A3 da auditoria de
    2026-08-07: as duas reimplementavam a mesma regra de decisao de forma
    independente e divergente).

    Com desvio<=0 ou media<=0 (dados degenerados: valores identicos ou
    vazio), cai para N=1 -- o minimo que ainda faz sentido como grau de
    liberdade, em vez de propagar NaN/Inf para a estatistica combinada.
    """
    valores = np.asarray(valores, dtype=float)
    if valores.size == 0:
        return 0.0, 1.0
    media = float(valores.mean())
    desvio = float(valores.std(ddof=1)) if valores.size > 1 else 0.0
    if desvio <= 0 or media <= 0:
        return max(media, 1e-12), 1.0
    return media, 2.0 * (media / desvio) ** 2


def combined_distance(T2: np.ndarray, Q: np.ndarray, h0: float, q0: float,
                        Nh: float, Nq: float) -> np.ndarray:
    """Distancia combinada f = (T2/h0)*Nh + (Q/q0)*Nq (Eq. 3 de
    Kucheryavskiy, Rodionova & Pomerantsev 2024, J. Chemometrics 38(7):
    e3556) -- a estatistica de decisao do DD-SIMCA (`f <= chi2.ppf(1-alpha,
    Nh+Nq)` decide aceitacao). Substitui o teste retangular independente
    T2<=UCL e Q<=UCL (alpha por eixo), que infla o alpha CONJUNTO efetivo
    para ~1-(1-alpha)^2 -- achado corrigido no DD-SIMCA em 2026-08-08 e no
    dominio de aplicabilidade PCA/PLS (achado A3 da auditoria de
    2026-08-07: medido 11.6% de rejeicao contra 5% nominal em amostras da
    MESMA distribuicao do treino, ver scripts/medicoes/medir_achados.py).
    """
    return ((np.asarray(T2, dtype=float) / max(h0, 1e-12)) * Nh
            + (np.asarray(Q, dtype=float) / max(q0, 1e-12)) * Nq)


def dmodx(Q: np.ndarray, n_variaveis: int, n_componentes: int,
          n_amostras: int, alpha: float = 0.05) -> Dict[str, object]:
    """DModX (Distance to Model X) -- nomenclatura e normalizacao padrao do
    SIMCA-P/Unscrambler (Eriksson et al. 2006, "Multi- and Megavariate Data
    Analysis", cap. 9) para o MESMO Q-residuo ja calculado por
    `q_residuals()`. NAO e' um diagnostico novo -- e' a mesma distancia
    residual X, normalizada pela variancia residual media do modelo (assim
    DModX ~= 1 significa "residuo tipico"; DModX >> DModX_crit sinaliza
    amostra fora do modelo), na escala que usuarios vindos do SIMCA-P/
    Unscrambler ja reconhecem de outras ferramentas comerciais.

        DModX_i           = sqrt(Q_i / (K - A))
        s0 (residuo medio) = sqrt(sum(Q) / ((N - A - 1)(K - A)))
        DModX_normalizado_i = DModX_i / s0
        DModX_critico       = sqrt(F_crit(alpha; K-A, (N-A-1)(K-A)))

    K = n_variaveis, A = n_componentes, N = n_amostras.
    """
    Q = np.asarray(Q, dtype=float)
    K, A, N = int(n_variaveis), int(n_componentes), int(n_amostras)
    df_num = max(K - A, 1)
    df_den = max((N - A - 1) * df_num, 1)

    dmodx_bruto = np.sqrt(Q / df_num)
    s0 = np.sqrt(float(np.sum(Q)) / df_den) if df_den > 0 else 1.0
    s0 = s0 if s0 > 1e-12 else 1.0
    dmodx_norm = dmodx_bruto / s0

    dmodx_crit = float(np.sqrt(f_dist.ppf(1 - alpha, df_num, df_den)))

    return {
        "dmodx": dmodx_norm,
        "dmodx_crit": dmodx_crit,
        "fora_do_modelo": dmodx_norm > dmodx_crit,
        "n_fora_do_modelo": int(np.sum(dmodx_norm > dmodx_crit)),
    }


def dmody(residuo_y: np.ndarray, n_componentes: int, n_amostras: int,
          alpha: float = 0.05) -> Dict[str, object]:
    """DModY (Distance to Model Y) -- mesma logica/normalizacao do DModX,
    aplicada ao residuo de PREDICAO (y - y_hat) de um modelo de regressao
    PLS (N2/N3). Tambem nomenclatura padrao SIMCA-P/Unscrambler, mesma
    reapresentacao (nao e' um diagnostico novo): o residuo de validacao
    ja calculado (usado no RMSEP), normalizado para a escala DModY.

        DModY_i              = |y_i - yhat_i|
        s0 (residuo medio)    = sqrt(sum(residuo_y^2) / (N - A - 1))
        DModY_normalizado_i   = DModY_i / s0
        DModY_critico         = sqrt(F_crit(alpha; 1, N-A-1))
    """
    residuo_y = np.asarray(residuo_y, dtype=float).flatten()
    A, N = int(n_componentes), int(n_amostras)
    df_den = max(N - A - 1, 1)

    dmody_bruto = np.abs(residuo_y)
    s0 = np.sqrt(float(np.sum(residuo_y ** 2)) / df_den) if df_den > 0 else 1.0
    s0 = s0 if s0 > 1e-12 else 1.0
    dmody_norm = dmody_bruto / s0

    dmody_crit = float(np.sqrt(f_dist.ppf(1 - alpha, 1, df_den)))

    return {
        "dmody": dmody_norm,
        "dmody_crit": dmody_crit,
        "fora_do_modelo": dmody_norm > dmody_crit,
        "n_fora_do_modelo": int(np.sum(dmody_norm > dmody_crit)),
    }


def explained_variance(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Explained variance (%) of X by each column of T."""
    var_X_total = float(np.var(X, axis=0).sum())
    if var_X_total <= 0:
        return np.zeros(T.shape[1])
    return np.var(T, axis=0) / var_X_total * 100


def rmse_flat(a, b) -> float:
    """RMSE entre dois arrays quaisquer (achatados antes de comparar) --
    usada em toda regressao PLS do pipeline (RMSEC/RMSECV/RMSEP) e no
    Auto-Benchmark de regressao (avaliacao_modelos.py), mesma metrica p/
    comparacao apples-to-apples entre PLS-R e os modelos alternativos."""
    return float(np.sqrt(np.mean((np.asarray(a).flatten()
                                  - np.asarray(b).flatten()) ** 2)))


# =========================================================================
#  Figuras de merito analiticas (calibracao multivariada, UM analito)
# =========================================================================

def rpd_rer(y_ref: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """RPD e RER -- as duas razoes que dizem se um RMSEP e' bom ou ruim.

    Um RMSEP sozinho nao e' interpretavel: 0,17 %m/m e' excelente para
    proteina em milho (faixa ~6-10 %) e inutil para um analito que varia
    entre 0 e 0,5 %. As duas razoes normalizam o erro pela variacao do
    proprio conjunto de referencia:

        RPD = SD(y_ref) / SEP     (Residual Prediction Deviation)
        RER = amplitude(y_ref) / SEP   (Range Error Ratio)

    SEP e' o erro-padrao de predicao CORRIGIDO PELO BIAS -- a definicao
    usada nas faixas de interpretacao publicadas. Usar RMSEP no lugar de
    SEP (erro comum) infla RPD quando ha' bias, porque o bias sai da conta.

    Faixas de interpretacao (Williams 2014, em Williams, Dardenne & Flinn,
    *J. Near Infrared Spectrosc.* 22(2):85-93; e AACC 39-00.01):
        RPD < 2,0        nao utilizavel
        2,0 <= RPD < 2,5 triagem grosseira
        2,5 <= RPD < 3,0 triagem
        3,0 <= RPD < 5,0 controle de qualidade
        RPD >= 5,0       controle de processo / quantificacao
    RER < 4 nao utilizavel; RER >= 10 costuma acompanhar RPD >= 3.

    A classificacao textual vem junto de proposito: um numero cru convida a
    comparacoes indevidas entre estudos, enquanto a faixa carrega a
    referencia que a define.

    Devolve NaN nos campos dependentes quando `y_ref` tem desvio zero (nao
    ha' variacao a explicar) ou quando SEP = 0 -- nunca infinito disfarcado
    de desempenho perfeito.
    """
    y_ref = np.asarray(y_ref, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = y_ref.size
    saida: Dict[str, float] = {
        "sep": float("nan"), "bias": float("nan"),
        "rpd": float("nan"), "rer": float("nan"),
    }
    if n < 2 or y_pred.size != n:
        return saida

    residuos = y_pred - y_ref
    bias = float(np.mean(residuos))
    # SEP: desvio-padrao dos residuos apos remover o bias (ddof=1).
    sep = float(np.sqrt(np.sum((residuos - bias) ** 2) / (n - 1)))
    sd_ref = float(np.std(y_ref, ddof=1))
    amplitude = float(np.max(y_ref) - np.min(y_ref))

    saida["bias"] = bias
    saida["sep"] = sep
    if sep > 0 and sd_ref > 0:
        saida["rpd"] = sd_ref / sep
    if sep > 0 and amplitude > 0:
        saida["rer"] = amplitude / sep
    return saida


def interpret_rpd(rpd: float) -> str:
    """Faixa de uso correspondente ao RPD (Williams 2014; AACC 39-00.01).

    Existe para que nenhum relatorio imprima um RPD nu: o numero sem a
    faixa e' o tipo de metrica que vira alegacao exagerada em texto.
    """
    if not np.isfinite(rpd):
        return "nao estimavel"
    if rpd < 2.0:
        return "nao utilizavel"
    if rpd < 2.5:
        return "triagem grosseira"
    if rpd < 3.0:
        return "triagem"
    if rpd < 5.0:
        return "controle de qualidade"
    return "controle de processo / quantificacao"


def regression_figures_of_merit(modelo: PLSRegression, X_cal: np.ndarray,
                              grupos_replicas: List[np.ndarray],
                              alpha_ic: float = 0.05
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

    LOD/LOQ COMO INTERVALO (Bloco 12): `delta_x_ruido` e' uma ESTIMATIVA com
    `soma_df` graus de liberdade (soma de (n_replicas-1) por grupo), nao o
    valor verdadeiro -- reportar so' o LOD/LOQ pontual esconde essa
    incerteza de estimacao. O CONCEITO de nao reduzir o limite de deteccao a
    um numero unico, tratando-o como um intervalo que reflete a incerteza da
    propria calibracao, segue Allegrini & Olivieri (2014), Anal. Chem.
    86(15):7858-7866 (abordagem consistente com IUPAC para LOD em PLS). A
    CONSTRUCAO especifica do intervalo aqui e' nossa, nao necessariamente
    identica ao algoritmo exato do artigo (que trata taxas de erro tipo
    I/II explicitamente) -- [VERIFICAR contra o artigo original antes de
    citar a formula exata deles]: intervalo de (1-`alpha_ic`) para a
    VARIANCIA pooled via qui-quadrado (teoria ANOVA padrao,
    `soma_df * var_pooled / sigma^2 ~ chi2(soma_df)`), propagado linearmente
    para LOD/LOQ (ambos lineares em delta_x). APROXIMACAO: trata a
    variancia agregada (RMS sobre variaveis espectrais correlacionadas)
    como se fosse uma unica variancia pooled com `soma_df` graus de
    liberdade -- ignora a correlacao entre variaveis espectrais vizinhas;
    um intervalo multivariado exato exigiria propagacao mais completa.
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
        # Bloco 12: LOD/LOQ como intervalo -- ver docstring. NaN ate' que
        # `soma_df` seja calculavel (mesma condicao de delta_x_ruido).
        "delta_x_ruido_ic_baixo": float("nan"),
        "delta_x_ruido_ic_alto": float("nan"),
        "lod_ic_baixo": float("nan"),
        "lod_ic_alto": float("nan"),
        "loq_ic_baixo": float("nan"),
        "loq_ic_alto": float("nan"),
        "lod_ic_confianca": float(1.0 - alpha_ic),
        "lod_ic_graus_liberdade": 0.0,
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
        var_media = float(np.mean(var_pooled))
        delta_x = float(np.sqrt(var_media))
        resultado["delta_x_ruido"] = delta_x
        resultado["lod_ic_graus_liberdade"] = float(soma_df)
        if delta_x > 1e-12:
            resultado["sensibilidade_analitica"] = sen / delta_x
            resultado["lod"] = 3.3 * delta_x * norm_b
            resultado["loq"] = 10.0 * delta_x * norm_b

            # IC de (1-alpha_ic) para a variancia pooled via qui-quadrado
            # (df*S^2/sigma^2 ~ chi2(df)) -- ver docstring para o
            # significado e as limitacoes desta aproximacao.
            chi2_baixo = float(chi2.ppf(alpha_ic / 2.0, soma_df))
            chi2_alto = float(chi2.ppf(1.0 - alpha_ic / 2.0, soma_df))
            if chi2_alto > 0:
                sigma_baixo = float(np.sqrt(soma_df * var_media / chi2_alto))
                resultado["delta_x_ruido_ic_baixo"] = sigma_baixo
                resultado["lod_ic_baixo"] = 3.3 * sigma_baixo * norm_b
                resultado["loq_ic_baixo"] = 10.0 * sigma_baixo * norm_b
            if chi2_baixo > 0:
                sigma_alto = float(np.sqrt(soma_df * var_media / chi2_baixo))
                resultado["delta_x_ruido_ic_alto"] = sigma_alto
                resultado["lod_ic_alto"] = 3.3 * sigma_alto * norm_b
                resultado["loq_ic_alto"] = 10.0 * sigma_alto * norm_b

    return resultado


#: Rotulos da faixa de decisao (Bloco 24) -- string estavel, nao texto
#: ja' traduzido (quem exibe decide o idioma).
FAIXA_NAO_DETECTAVEL = "nao_detectavel"
FAIXA_ZONA_CINZENTA = "zona_cinzenta"
FAIXA_QUANTIFICADO = "quantificado_com_confianca"


def faixa_decisao(valor: float, lod: float, loq: float) -> Optional[str]:
    """Categoriza um valor quantificado (teor predito) em 3 estados
    usando os limiares de LOD/LOQ do Bloco 12
    (`regression_figures_of_merit`) -- MESMOS numeros, nunca recalculados
    aqui: abaixo do LOD = "nao detectavel", entre LOD e LOQ = "zona
    cinzenta" (deteccao possivel, quantificacao nao confiavel), acima do
    LOQ = "quantificado com confianca".

    Devolve `None` (nao "nao_detectavel") quando LOD/LOQ nao sao
    computaveis (NaN -- sem replicas fisicas suficientes para estimar
    ruido instrumental, ver `regression_figures_of_merit`) -- categorizar
    contra um limiar que nao existe seria fabricar confianca que os dados
    nao sustentam."""
    if not (np.isfinite(lod) and np.isfinite(loq)):
        return None
    if valor < lod:
        return FAIXA_NAO_DETECTAVEL
    if valor < loq:
        return FAIXA_ZONA_CINZENTA
    return FAIXA_QUANTIFICADO


# =========================================================================
#  Dominio de Aplicabilidade (Applicability Domain, AD)
# =========================================================================

def applicability_domain(pca, X_train: np.ndarray, X_new: np.ndarray,
                           alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """API de CONVENIENCIA (uma chamada so') p/ uso exploratorio/notebook,
    quando X_train e X_new estao ambos disponiveis em memoria ao mesmo
    tempo. Em PRODUCAO (salvar modelo -> prever amostra nova depois, sem
    reexportar X_train inteiro), use `training_applicability_domain` +
    `applicability_domain_new_samples` diretamente -- e' o que
    pipeline.py/predicao.py fazem internamente. Esta funcao so' chama as
    duas em sequencia; nao e' dead code, e' a forma simples da mesma API
    (auditoria de 2026-07-12 apontou como "semi-orfa" por nao ter chamador
    de producao -- e' esperado: os dois usos servem publicos diferentes).

    Dominio de aplicabilidade via distancia ao modelo PCA/PLS, combinando
    as duas distancias complementares ja usadas para deteccao de outliers:

        Hotelling T2  -> distancia DENTRO do plano do modelo (leverage): quao
                         extrema a amostra e ao longo das direcoes de maior
                         variancia capturadas pela calibracao.
        Q-residuos    -> distancia ORTOGONAL ao plano (residuo espectral): quao
                         mal o modelo reconstroi a amostra (quimica nova, nao
                         vista no treino).

    Uma amostra nova esta DENTRO do dominio se a DISTANCIA COMBINADA
    f=(T2/h0)*Nh+(Q/q0)*Nq <= chi2.ppf(1-alpha, Nh+Nq) -- mesma estatistica
    do DD-SIMCA (Kucheryavskiy, Rodionova & Pomerantsev 2024), com h0/q0/Nh/
    Nq estimados EXCLUSIVAMENTE do conjunto de treino. t2/q individuais e
    seus limites por eixo sao mantidos so' para diagnostico/plotagem.

    CORRIGIDO em 2026-08-07 (achado A3 da auditoria metodologica — ver
    AUDITORIA_METODOLOGICA_2026-08-07.md): a versao anterior
    decidia dentro/fora por T2<=T2_limite E Q<=Q_limite independentemente
    (alpha=0.05 em cada eixo) -- a mesma regra retangular corrigida no
    DD-SIMCA em 2026-08-08, com o mesmo efeito: alpha CONJUNTO efetivo
    inflado. Medido (scripts/medicoes/medir_achados.py, 40 simulacoes,
    amostras novas da MESMA distribuicao do treino): rejeicao de 11.6%
    contra 5% nominal.

    Referencias: Jaworska, Nikolova-Jeliazkova & Aldenberg (2005), SAR QSAR
    Environ. Res. 16:445-466; Gadaleta et al. (2016), J. Chem. Inf. Model.
    A convencao T2+Q e o "AD baseado em leverage/residuo" padrao em
    espectroscopia (equivalente ao par distance-to-model do SIMCA);
    Kucheryavskiy, Rodionova & Pomerantsev (2024), J. Chemometrics 38(7):
    e3556, para a distancia combinada.

    Parametros
    ----------
    pca      : modelo PCA ja ajustado no treino (precisa de .transform(),
               .components_ (k, p) e .mean_ (p,) — sklearn PCA satisfaz).
    X_train  : matriz de treino no MESMO espaco pre-processado do ajuste.
    X_new    : amostras novas a avaliar (mesmo pre-processamento).
    alpha    : nivel de significancia (default 0.05 -> 95%).

    Retorna dict com t2/q/f por amostra nova, os limites, e a mascara
    booleana dentro_dominio + a fracao dentro.
    """
    treino = training_applicability_domain(pca, X_train, alpha)
    # cast: o dict é Dict[str, object] (chaves heterogêneas — arrays e
    # floats); os tipos concretos são garantidos na construção do dict.
    return applicability_domain_new_samples(
        pca, X_new,
        cast(np.ndarray, treino["var_t"]),
        cast(float, treino["h0"]), cast(float, treino["q0"]),
        cast(float, treino["Nh"]), cast(float, treino["Nq"]),
        cast(float, treino["f_crit"]))


def training_applicability_domain(pca, X_train: np.ndarray,
                                   alpha: float = 0.05) -> Dict[str, object]:
    """Deriva do TREINO os artefatos leves necessarios para avaliar o
    dominio de aplicabilidade em amostras novas depois, sem precisar
    re-exportar X_train inteiro (que pode ser um artefato pesado -- MB a
    dezenas de MB para datasets espectrais reais): a variancia dos scores
    PCA (var_t) e os parametros da distancia combinada (h0/q0/Nh/Nq/f_crit,
    ver `combined_distance`). Usado ao SALVAR um modelo (ver pipeline.py,
    pacote_modelo); `applicability_domain_new_samples` consome o
    resultado na hora de PREDIZER, sem X_train.

    t2_limite/q_limite (Tracy-Young-Mason / chi2-Jackson-Mudholkar) sao
    mantidos no retorno so' para diagnostico/plotagem por eixo -- a decisao
    dentro/fora usa h0/q0/Nh/Nq/f_crit (ver docstring de
    `applicability_domain`).
    """
    X_train = np.asarray(X_train, dtype=float)
    T_train = np.asarray(pca.transform(X_train), dtype=float)
    n, k = T_train.shape

    var_t = T_train.var(axis=0, ddof=1)
    var_t[var_t == 0] = 1.0
    T2_train = np.sum((T_train ** 2) / var_t, axis=1)

    # Q de treino por leave-one-out, NAO in-sample: com n < p (o regime deste
    # projeto -- espectros de milhares de canais) a PCA reconstroi o proprio
    # treino quase exatamente, q0/Nq saem otimistas e o dominio passa a
    # rejeitar amostras legitimas. Medido antes da correcao: 0,14 a 0,57 de
    # aceitacao contra 0,95 nominal (scripts/medicoes/medir_ad_vies_insample.py).
    # Mesma correcao ja aplicada ao DD-SIMCA em 2026-07-19 (CLAUDE.md P1) --
    # aqui ela faltava, e este e' o caminho que predicao.py usa em producao.
    q_train = q_residuals_loo(X_train, k)

    h0, Nh = mean_and_dof_moments(T2_train)
    q0, Nq = mean_and_dof_moments(q_train)
    f_crit = float(chi2.ppf(1 - alpha, Nh + Nq))

    return {
        "var_t": var_t,
        "h0": h0, "q0": q0, "Nh": Nh, "Nq": Nq, "f_crit": f_crit,
        "t2_limite": float(hotelling_t2_limit(n, k, alpha)),
        "q_limite": float(q_residuals_limit(q_train, alpha)),
    }


def applicability_domain_new_samples(
        pca, X_new: np.ndarray, var_t: np.ndarray,
        h0: float, q0: float, Nh: float, Nq: float, f_crit: float
        ) -> Dict[str, np.ndarray]:
    """Aplica a distancia combinada de dominio de aplicabilidade (ja
    derivada do treino por `training_applicability_domain`) a amostras
    novas -- nao precisa de X_train, so' dos artefatos leves, ideal para
    predicao em producao sem reexportar o dataset de calibracao.
    """
    X_new = np.asarray(X_new, dtype=float)
    T_new = np.asarray(pca.transform(X_new), dtype=float)
    P = np.asarray(pca.components_, dtype=float)          # (k, p)
    mean = np.asarray(pca.mean_, dtype=float)             # (p,)
    var_t = np.asarray(var_t, dtype=float)

    # T2 das amostras novas usando a variancia dos scores do TREINO (nunca a
    # das novas — senao o limite deixaria de ser um teste de extrapolacao).
    t2_new = np.sum((T_new ** 2) / var_t, axis=1)

    # Q-residuos: reconstrucao no espaco CENTRADO pela media do treino.
    q_new = q_residuals(X_new - mean, T_new, P)

    f = combined_distance(t2_new, q_new, h0, q0, Nh, Nq)
    dentro = f <= f_crit
    return {
        "t2": t2_new,
        "q": q_new,
        "f": f,
        "f_crit": np.asarray(f_crit, dtype=float),
        "dentro_dominio": dentro,
        "fracao_dentro": np.asarray(
            float(np.mean(dentro)) if dentro.size else float("nan"),
            dtype=float),
    }


def diagnose_spectral_range(X: np.ndarray, wavenumbers: np.ndarray,
                                  limiar_snr: float = 3.0,
                                  largura_min_cm: float = 150.0,
                                  janela_suave: int = 11
                                  ) -> Dict[str, object]:
    """Detecta regioes espectrais que nao carregam informacao analitica.

    Motivacao (achado 2026-08-07): rodar com uma faixa larga demais inclui
    regioes onde o espectro e' so' linha de base e ruido de detector. Isso
    nao "e' inofensivo": infla o numero de variaveis, dilui metricas por
    variavel (VIP/SR), aumenta o custo de CV e da' ao modelo espaco para
    ajustar ruido. O usuario so' percebe olhando o loading plot e vendo uma
    metade chapada -- este diagnostico automatiza essa leitura.

    Separa DOIS defeitos diferentes, que exigem acoes diferentes:

    - regiao MORTA   : sinal analitico ~ 0 (nada acontece ali).
    - regiao RUIDOSA : ha' variacao, mas dominada por alta frequencia
                       (ruido de detector), nao por banda espectral.

    Metodo: para cada numero de onda, separa o espectro medio-centrado em
    componente suave (sinal) e residuo de alta frequencia (ruido) por media
    movel, e compara a dispersao ENTRE amostras de cada um --
    SNR = sd_entre_amostras(suave) / sd(residuo). Regioes com SNR abaixo de
    `limiar_snr` sao marcadas; blocos contiguos mais estreitos que
    `largura_min_cm` sao descartados (evita marcar ponto isolado).

    Parameters
    ----------
    X : (n_amostras, n_variaveis) — espectros JA na faixa em uso.
    wavenumbers : (n_variaveis,) — em cm-1.

    Returns
    -------
    dict com:
      snr            : (n_variaveis,) SNR por numero de onda
      mascara_util   : (n_variaveis,) bool — True onde ha' sinal
      regioes_ruins  : lista de (wn_ini, wn_fim, tipo) — tipo em
                       {"morta", "ruidosa"}
      frac_util      : fracao de variaveis uteis
      faixa_sugerida : (min, max) contiguo cobrindo a parte util, ou None
    """
    X = np.asarray(X, dtype=float)
    wn = np.asarray(wavenumbers, dtype=float)
    n_var = X.shape[1]
    if n_var < 5 or X.shape[0] < 3:
        return {"snr": np.full(n_var, np.nan),
                "mascara_util": np.ones(n_var, dtype=bool),
                "regioes_ruins": [], "frac_util": 1.0,
                "faixa_sugerida": None,
                "aviso": "espectro curto demais para diagnosticar"}

    # Suavizacao por media movel (kernel impar), sem depender de savgol para
    # manter a funcao pura em numpy/scipy basico.
    jan = int(max(3, min(janela_suave, n_var // 2 * 2 - 1)))
    if jan % 2 == 0:
        jan += 1
    kernel = np.ones(jan) / jan
    suave = np.apply_along_axis(
        lambda linha: np.convolve(linha, kernel, mode="same"), 1, X)
    # Bordas da convolucao 'same' sao atenuadas -> ignora meia janela
    borda = jan // 2
    residuo = X - suave

    sd_sinal = suave.std(axis=0, ddof=1)
    sd_ruido = residuo.std(axis=0, ddof=1)
    # Piso de ruido global evita SNR explodir onde o residuo e' ~0 por acaso
    piso = float(np.median(sd_ruido[sd_ruido > 0])) if np.any(sd_ruido > 0) else 1.0
    snr = sd_sinal / np.maximum(sd_ruido, piso * 1e-3)
    if borda:
        snr[:borda] = snr[borda]
        snr[-borda:] = snr[-borda - 1]

    mascara_util = snr >= limiar_snr

    # Amplitude do sinal (para distinguir "morta" de "ruidosa")
    amp_rel = sd_sinal / max(float(sd_sinal.max()), 1e-12)

    regioes: List[Tuple[float, float, str]] = []
    ruim = ~mascara_util
    i = 0
    while i < n_var:
        if not ruim[i]:
            i += 1
            continue
        j = i
        while j + 1 < n_var and ruim[j + 1]:
            j += 1
        wn_a, wn_b = float(wn[i]), float(wn[j])
        if abs(wn_b - wn_a) >= largura_min_cm:
            # MEDIANA, nao maximo: as bordas de uma regiao morta encostam na
            # cauda da banda vizinha, entao o maximo dentro do bloco fica
            # alto e classificava tudo como "ruidosa" por engano.
            tipo = ("morta" if float(np.median(amp_rel[i:j + 1])) < 0.10
                    else "ruidosa")
            regioes.append((min(wn_a, wn_b), max(wn_a, wn_b), tipo))
        i = j + 1

    frac_util = float(mascara_util.mean())

    faixa_sugerida = None
    if mascara_util.any() and frac_util < 0.95:
        uteis = wn[mascara_util]
        faixa_sugerida = (float(uteis.min()), float(uteis.max()))

    return {"snr": snr, "mascara_util": mascara_util,
            "regioes_ruins": regioes, "frac_util": frac_util,
            "faixa_sugerida": faixa_sugerida, "aviso": None}
