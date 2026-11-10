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
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

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
    "q_residuals_pls_loo",
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
    "ad_rejection_rate_cv",
    "ad_small_n_warning",
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
    Passo 148). Este mesmo padrao de correcao
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
    Chemom. Intell. Lab. Syst. 95:35-48; Kvalheim (2020),
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
    simulacao do mantenedor (script nao versionado).

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
    # Centra em colunas: a projecao-alvo e a variancia por canal sao definidas sobre X
    # centrado na media (Rajalahti et al. 2009). Com X sem centragem (preset `custom` sem
    # mean-centering) o escore t_tp carregava a media dos espectros e o SR saia errado
    # (auditoria de 2026-11-10: correlacao de postos 0,48 contra a formula). Nos presets
    # padrao X ja' vem centrado e nada muda.
    X = X - X.mean(axis=0)
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
    J. Quality Technology 24(2):88-95 (DOI 10.1080/00224065.1992.12015232),
    **Phase II** (new/future observations,
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
    ~2-3x so' com n<20) — ver simulacao do mantenedor (script nao versionado). Nao
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


def q_residuals_loo(X: np.ndarray, n_comp: int,
                    groups: Optional[np.ndarray] = None) -> np.ndarray:
    """Q-residuo de cada amostra de treino contra uma PCA ajustada SEM ela --
    e, com `groups`, sem NENHUMA amostra do mesmo grupo (leave-one-GROUP-out).

    Uma PCA ajustada em TODAS as n amostras reconstroi cada uma delas de
    forma otimista: a propria amostra ajudou a definir o subespaco que
    depois a reconstroi. Com n pequeno frente a p (regime deste projeto:
    espectros de milhares de variaveis), esse vies faz Q_train colapsar
    perto de zero -- e qualquer limite derivado dele rejeita amostras
    genuinamente novas, mesmo vindas da mesma distribuicao do treino.

    LEAVE-ONE-GROUP-OUT: deixar de fora so'
    UM espectro mantem as replicas irmas (mesma amostra fisica) no ajuste, e
    o Q continua otimista para o caso que importa -- uma amostra NOVA, sem
    irmas no treino. Medido: o DD-SIMCA aceitava 0,62 das amostras puras
    novas (3 replicas, 10 grupos de calibracao) contra 0,80 nominal-alcancavel
    com Q-LOO por grupo. Sem `groups` (None) cada amostra e' seu proprio grupo
    (leave-one-out, comportamento anterior).

    Origem: `DDSimca._q_residuals_loo` (o Q de calibracao por exclusao de
    amostra, que fica otimista quando replicas ficam juntas no treino).
    Promovida a funcao pura ao se descobrir
    que `training_applicability_domain` -- o caminho que roda em producao
    em predicao.py -- tinha o mesmo vies e nao havia recebido a correcao
    (ver simulacao do mantenedor (script nao versionado)). DDSimca delega para ca'
    em vez de manter a segunda copia.

    Custo: com n <= p (o regime espectral tipico) usa a matriz de Gram
    X X^T, calculada UMA vez, e resolve cada ajuste sem um grupo por
    autodecomposicao de uma matriz (n-g)x(n-g), sem refazer o produto com
    as p variaveis (`_q_residuals_loo_gram`). E' a mesma PCA, em forma
    fechada: os mesmos componentes principais, so' que obtidos pelo lado
    das amostras. Medido (243x1500): 31x mais rapido que um `PCA.fit` por
    grupo, Q igual a ~1e-12 relativo. Com n > p cai no ajuste de PCA por
    grupo, onde o lado das variaveis ja' e' o menor.
    """
    from sklearn.decomposition import PCA   # local: evita import pesado no topo

    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    Q = np.empty(n, dtype=float)
    if groups is None:
        grupos = np.arange(n)
    else:
        grupos = np.asarray(groups)
        if grupos.shape[0] != n:
            raise ValueError(
                f"groups tem {grupos.shape[0]} elementos, X tem {n} linhas")
        # Com UM so' grupo nao ha' o que deixar de fora: tirar o grupo inteiro
        # deixaria o ajuste vazio (Q=0 para todos -> limite 0 -> o modelo
        # rejeitaria ate' o proprio treino). Cai para leave-one-out por
        # amostra (otimista, mas nao degenerado); o chamador ja' declara que
        # com 1 grupo a validacao nao e' possivel.
        if np.unique(grupos).size < 2:
            grupos = np.arange(n)
    if n <= X.shape[1]:
        return _q_residuals_loo_gram(X, n_comp, grupos)
    for g in np.unique(grupos):
        fora = grupos == g
        X_tr = X[~fora]
        n_comp_i = min(n_comp, X_tr.shape[0] - 1, X_tr.shape[1])
        if n_comp_i < 1:
            Q[fora] = 0.0
            continue
        pca_i = PCA(n_components=n_comp_i).fit(X_tr)
        Xg = X[fora]
        Q[fora] = np.sum((Xg - pca_i.inverse_transform(
            pca_i.transform(Xg))) ** 2, axis=1)
    return Q


def _q_residuals_loo_gram(X: np.ndarray, n_comp: int,
                          grupos: np.ndarray) -> np.ndarray:
    """Nucleo de `q_residuals_loo` pelo lado das amostras.

    Para o treino S (todas as linhas fora do grupo), a PCA centrada de X_S
    tem os mesmos autovalores nao nulos da Gram centrada
    K = (X_S - m)(X_S - m)^T, e o subespaco principal e' gerado por
    (X_S - m)^T u_j / sqrt(lambda_j). O Q de uma amostra z deixada de fora
    e' entao ||z - m||^2 - sum_j (u_j^T (X_S - m)(z - m))^2 / lambda_j, e
    todos esses produtos internos saem de G = X X^T, calculada uma vez.
    """
    n, p = X.shape
    G = X @ X.T
    rs = G.sum(axis=1)
    Q = np.empty(n, dtype=float)
    for g in np.unique(grupos):
        fora = grupos == g
        S = np.flatnonzero(~fora)
        F = np.flatnonzero(fora)
        m1 = S.size
        k = min(n_comp, m1 - 1, p)
        if k < 1:
            Q[F] = 0.0
            continue
        soma_S = rs - G[:, F].sum(axis=1)          # x_j . sum(X_S)
        xm = soma_S / m1                            # x_j . m
        mm = float(soma_S[S].sum()) / m1 ** 2       # m . m
        K = G[np.ix_(S, S)] - xm[S][:, None] - xm[S][None, :] + mm
        lam, U = np.linalg.eigh(K)
        lam = lam[::-1][:k]
        U = U[:, ::-1][:, :k]
        # componentes de variancia numericamente nula nao definem direcao
        # (a PCA do sklearn devolveria uma direcao arbitraria ali)
        ok = lam > max(float(lam[0]), 0.0) * 1e-12
        lam, U = lam[ok], U[:, ok]
        cz = G[np.ix_(S, F)] - xm[S][:, None] - xm[F][None, :] + mm
        zz = G[F, F] - 2.0 * xm[F] + mm
        proj = np.sum((U.T @ cz) ** 2 / lam[:, None], axis=0)
        Q[F] = np.maximum(zz - proj, 0.0)
    return Q


def q_residuals_pls_loo(X: np.ndarray, Y: np.ndarray, n_comp: int,
                        groups: Optional[np.ndarray] = None,
                        max_ajustes: int = 100) -> np.ndarray:
    """Q-residuo de treino NO ESPACO PLS, calculado fora da amostra: cada grupo
    e' reconstruido por um PLS ajustado SEM ele (leave-one-group-out), com X
    centrado pela media do PLS daquele ajuste.

    Por que existe (achado A4-03 da vistoria de 2026-10): o Q de treino do
    espaco PLS vinha da reconstrucao do proprio treino (in-sample) e sem
    centrar X, entao era otimista e o limite de aceitacao derivado dele
    rejeitava amostras novas legitimas (9 % a 46 % com 5 a 10 grupos por
    classe, contra 5 % nominais). Este e' o mesmo remedio de `q_residuals_loo`
    (PCA), mas com o PLS no lugar: o Q de amostra nova, em `predicao.py`, e'
    o residuo contra os loadings do PLS, e treino e predicao precisam medir a
    mesma coisa -- um Q de treino por PCA contra um Q novo por PLS piorou a
    rejeicao com 6 LVs (medido), por isso nao se mistura.

    `groups=None`: cada amostra e' seu proprio grupo. Com mais de
    `max_ajustes` grupos os grupos sao reunidos em `max_ajustes` blocos
    (indice do grupo ordenado modulo `max_ajustes`, determinístico) para
    limitar o custo a `max_ajustes` ajustes de PLS. Com um so' grupo cai para
    leave-one-out por amostra (mesmo criterio de `q_residuals_loo`).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError(f"Y tem {Y.shape[0]} linhas, X tem {n}")
    if groups is None:
        grupos = np.arange(n)
    else:
        grupos = np.asarray(groups)
        if grupos.shape[0] != n:
            raise ValueError(
                f"groups tem {grupos.shape[0]} elementos, X tem {n} linhas")
        if np.unique(grupos).size < 2:
            grupos = np.arange(n)
    unicos, inverso = np.unique(grupos, return_inverse=True)
    if unicos.size > int(max_ajustes):
        inverso = inverso % int(max_ajustes)
    Q = np.zeros(n, dtype=float)
    for b in np.unique(inverso):
        fora = inverso == b
        n_tr = int((~fora).sum())
        n_comp_i = min(int(n_comp), n_tr - 1, X.shape[1])
        if n_comp_i < 1:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pls_i = PLSRegression(n_components=n_comp_i, scale=False).fit(
                X[~fora], Y[~fora])
        Xc = X[fora] - np.asarray(pls_i._x_mean, dtype=float)  # noqa: SLF001
        T = np.asarray(pls_i.transform(X[fora]), dtype=float)
        Q[fora] = np.sum((Xc - T @ np.asarray(pls_i.x_loadings_).T) ** 2, axis=1)
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
    limite = float(g * chi2.ppf(1 - alpha, h))
    if h < 1.0 and limite < float(np.percentile(q, 50)):
        # Cauda pesada (var > 2*media^2: poucos Q enormes, ex.: espectros saturados). O ajuste
        # chi2 por momentos desaba: o limite cai abaixo da MEDIANA e quase toda amostra vira
        # "outlier Q" (medido numa corrida real: limite 4e-8 contra Q tipico 2e-5, 1340 de 1340
        # fora). So' nesse caso cai para o percentil empirico; limite plausivel nao e' tocado.
        return float(np.percentile(q, (1 - alpha) * 100))
    return limite


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
    MESMA distribuicao do treino, ver simulacao do mantenedor (script nao versionado)).
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
    um constituinte majoritario (faixa ~6-10 %) e inutil para um analito que varia
    entre 0 e 0,5 %. As duas razoes normalizam o erro pela variacao do
    proprio conjunto de referencia:

        RPD = SD(y_ref) / SEP     (Residual Prediction Deviation)
        RER = amplitude(y_ref) / SEP   (Range Error Ratio)

    SEP e' o erro-padrao de predicao CORRIGIDO PELO BIAS -- a definicao
    usada nas faixas de interpretacao publicadas. Usar RMSEP no lugar de
    SEP (erro comum) infla RPD quando ha' bias, porque o bias sai da conta.

    Faixas de interpretacao usadas aqui: convencao de NIR agricola,
    SEM fonte unica verificada para os cortes -- NAO atribuir os cortes a
    Williams nem a AACC 39-00.01 (o texto original nao foi conferido e ha'
    escalas publicadas com cortes diferentes). Williams 2014, "The RPD
    statistic: a tutorial note", *NIR news* 25(1):22-26, DOI
    10.1255/nirn.1419, e' citado so' pela DEFINICAO do RPD:
        RPD < 2,0        nao utilizavel
        2,0 <= RPD < 2,5 triagem grosseira
        2,5 <= RPD < 3,0 triagem
        3,0 <= RPD < 5,0 controle de qualidade
        RPD >= 5,0       controle de processo / quantificacao
    RER < 4 nao utilizavel; RER >= 10 costuma acompanhar RPD >= 3.

    Ressalva: o RPD depende da distribuicao e do desvio-padrao do conjunto
    de referencia, e a literatura critica o uso de cortes fixos como se
    valessem para qualquer analito, matriz e aplicacao (Esbensen, Geladi &
    Larsen 2014, "The RPD Myth...", *NIR news* 25(5):24-28, DOI
    10.1255/nirn.1462). Trate a faixa como orientacao, e compare o RMSEP
    com o criterio do seu metodo oficial.

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
    """Faixa de uso correspondente ao RPD (convencao de NIR agricola, nao
    regra geral e sem fonte unica verificada para os cortes; ressalva em `rpd_rer`).

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
    I/II explicitamente). LIMITACAO CONFERIDA:
    o LOD/LOQ daqui usa SO' o ruido instrumental de replica (delta_x) e
    OMITE os termos de leverage da amostra e de erro na referencia (y) que
    a formula de Allegrini & Olivieri incluem -- portanto tende a SUBestimar
    o limite; e' uma aproximacao de primeira ordem, nao a formula do artigo.
    Intervalo de (1-`alpha_ic`) para a
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
        # `keepdims` aqui e' inerte, nao cosmetico: achado 4 da rodada de
        # mutation testing (2026-09-11) apontou a mutacao keepdims=True->
        # False como sobrevivente; investigado e CONFIRMADO mutante
        # equivalente -- para `grupo` 2D (n_replicas, n_variaveis), a
        # regra de broadcasting do numpy alinha (p,) e (1,p) de forma
        # IDENTICA contra (n,p) (verificado por prova e empiricamente,
        # `np.array_equal` em 4 formas distintas). So' importaria se
        # `grupo` deixasse de ser 2D -- nao acontece aqui:
        # `_agrupar_replicas_processadas` (pipeline.py) so' produz
        # arrays 2D. `keepdims` removido por ser sempre um no-op, nao
        # porque mudasse algum resultado.
        d = grupo - grupo.mean(axis=0)
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
                           alpha: float = 0.05,
                           groups: Optional[np.ndarray] = None
                           ) -> Dict[str, np.ndarray]:
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
    inflado. Medido (simulacao do mantenedor, script nao versionado, 40 simulacoes,
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
    treino = training_applicability_domain(pca, X_train, alpha, groups=groups)
    # cast: o dict é Dict[str, object] (chaves heterogêneas — arrays e
    # floats); os tipos concretos são garantidos na construção do dict.
    return applicability_domain_new_samples(
        pca, X_new,
        cast(np.ndarray, treino["var_t"]),
        cast(float, treino["h0"]), cast(float, treino["q0"]),
        cast(float, treino["Nh"]), cast(float, treino["Nq"]),
        cast(float, treino["f_crit"]))


def training_applicability_domain(pca, X_train: np.ndarray,
                                   alpha: float = 0.05,
                                   groups: Optional[np.ndarray] = None
                                   ) -> Dict[str, object]:
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
    # aceitacao contra 0,95 nominal (simulacao do mantenedor, script nao versionado).
    # Mesma correcao ja aplicada ao DD-SIMCA (Q de calibracao por exclusao de
    # grupo, nao in-sample) -- aqui ela faltava, e este e' o caminho que predicao.py usa em producao.
    q_train = q_residuals_loo(X_train, k, groups=groups)

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


#: Grupos (amostras fisicas) por classe abaixo dos quais a taxa de alarmes
#: do dominio de aplicabilidade em amostras legitimas se afasta do nominal.
AD_MIN_GRUPOS_POR_CLASSE = 30


def ad_small_n_warning(n_por_classe: Dict[str, int],
                       minimo: int = AD_MIN_GRUPOS_POR_CLASSE,
                       por_amostra: bool = False) -> Optional[str]:
    """Aviso de n pequeno para o dominio de aplicabilidade, ou `None`.

    `n_por_classe`: n de grupos (amostras fisicas independentes) de cada
    classe da calibracao; com `por_amostra=True` sao n de AMOSTRAS (nao ha'
    identificador de grupo). Achado A4-02 (vistoria de 2026-10-10): com 2 a
    10 grupos por classe o dominio rejeitou de 9% a 98% de amostras
    legitimas (nominal 5%); com 40 grupos, 5% a 6%. Abaixo de `minimo` a
    taxa de alarmes de amostras legitimas e' desconhecida e pode ser alta
    -- ver `ad_rejection_rate_cv` para a estimativa por validacao cruzada.
    """
    if not n_por_classe:
        return None
    pequenas = {c: int(n) for c, n in n_por_classe.items() if int(n) < minimo}
    if not pequenas:
        return None
    menor = min(pequenas.values())
    unidade = "amostras" if por_amostra else "grupos (amostras fisicas)"
    return (
        f"Dominio de aplicabilidade calibrado com poucos {unidade} por "
        f"classe (minimo {menor}; {len(pequenas)} classe(s) abaixo de "
        f"{minimo}). Nesse regime ele costuma rejeitar muitas amostras "
        f"legitimas: a taxa esperada de alarmes e' desconhecida e pode ser "
        f"muito maior que os 5% nominais. Use a taxa de rejeicao em "
        f"validacao cruzada como estimativa e nao trate 'fora do dominio' "
        f"como prova de problema na amostra.")


def ad_rejection_rate_cv(
        X: np.ndarray, n_components: int, alpha: float = 0.05,
        groups: Optional[np.ndarray] = None, max_loo_groups: int = 60,
        n_splits_grupos: int = 10, max_grupos: int = 300,
        progresso: Optional[Callable[[int, int], None]] = None
        ) -> Tuple[int, int]:
    """Taxa de rejeicao do Dominio de Aplicabilidade em VALIDACAO CRUZADA
    da propria calibracao: `(n_rejeitadas, n_avaliadas)`.

    Por que existe (MSPC, `sentinela_deriva`): o dominio calibrado com n
    finito rejeita, em controle, uma taxa diferente do `alpha` nominal e
    diferente a cada calibracao (medido em dado real de validacao: 6,5% em vez de 5%, com
    n=40; falso alarme do sentinela ~21% ao testar contra 5%). E' o efeito
    da estimacao de parametros da Fase I em CEP (Jensen, Jones-Farmer,
    Champ & Woodall 2006, J. Qual. Technol. 38(4):349-364, DOI
    10.1080/00224065.2006.11918623). A referencia correta para "a taxa em
    producao mudou?" e' a taxa que ESTA calibracao rejeita em dados que
    ela nao viu -- estimavel no treino, sem holdout: cada fold reajusta
    PCA + parametros do dominio SO' no treino do fold e julga o fold de
    fora.

    Folds: leave-one-GROUP-out quando ha' ate' `max_loo_groups` grupos
    (replicas do mesmo `groups`, ex. `mae_id`, ficam sempre juntas --
    nunca vazam entre ajuste e julgamento); acima disso, `n_splits_grupos`
    folds de grupos (`StableStratifiedGroupKFold`, deterministico), porque
    o custo por fold inclui o Q leave-one-out do proprio dominio.
    `groups=None` trata cada amostra como seu proprio grupo.

    Custo: cada fold refaz o dominio inteiro (incluindo o Q
    leave-one-out), ~3-4x o custo de `training_applicability_domain` por
    conjunto de 5 folds (medido: n=1000, p=1500 -> 70 s so' o dominio,
    252 s a CV de 5 folds). Por isso, acima de `max_grupos` grupos a
    referencia e' estimada num subconjunto DETERMINISTICO de `max_grupos`
    grupos (ordem por hash blake2b do id do grupo): a rejeicao em CV cai
    com n (o vies de Fase I decai ~1/n), entao a referencia com n menor e'
    levemente CONSERVADORA -- menos sensibilidade, nunca mais falso alarme.

    Folds cujo treino e' pequeno demais para o PCA (< n_components + 2
    amostras) sao ignorados -- `n_avaliadas` diz quantas entraram.

    `progresso(k, total)`, se dado, e' chamado ao fim de cada fold: e' o
    passo mais longo da exportacao do modelo e, sem isso, a barra de
    progresso ficava parada sem nenhuma linha de log.
    """
    from sklearn.decomposition import PCA

    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    g = np.arange(n) if groups is None else np.asarray(groups)
    if g.shape[0] != n:
        raise ValueError(
            f"groups ({g.shape[0]}) e X ({n}) diferem em tamanho")
    unicos = np.unique(g)
    if len(unicos) > max_grupos:
        from hashlib import blake2b
        ordem = sorted(unicos.tolist(), key=lambda u: blake2b(
            str(u).encode("utf-8"), digest_size=8).hexdigest())
        manter = np.isin(g, np.asarray(ordem[:max_grupos], dtype=g.dtype))
        X, g = X[manter], g[manter]
        n = X.shape[0]
        unicos = np.unique(g)
    if len(unicos) <= max_loo_groups:
        folds = [np.flatnonzero(g == u) for u in unicos]
    else:
        from guaraci.validacao_estatistica import StableStratifiedGroupKFold
        k_folds = min(n_splits_grupos, len(unicos))
        folds = [va for _, va in StableStratifiedGroupKFold(
            n_splits=k_folds, seed=0).split(np.zeros(n), np.zeros(n), groups=g)]

    rejeitadas = 0
    avaliadas = 0
    todos = np.arange(n)
    for i_fold, va in enumerate(folds):
        tr = np.setdiff1d(todos, va, assume_unique=True)
        k = int(min(n_components, len(tr) - 1, X.shape[1]))
        if len(tr) < n_components + 2 or k < 1:
            if progresso is not None:
                progresso(i_fold + 1, len(folds))
            continue
        pca = PCA(n_components=k).fit(X[tr])
        art = cast(Dict[str, Any], training_applicability_domain(
            pca, X[tr], alpha=alpha, groups=g[tr]))
        dentro = applicability_domain_new_samples(
            pca, X[va], art["var_t"], art["h0"], art["q0"], art["Nh"],
            art["Nq"], art["f_crit"])["dentro_dominio"]
        rejeitadas += int(np.sum(~dentro))
        avaliadas += int(len(va))
        if progresso is not None:
            progresso(i_fold + 1, len(folds))
    return rejeitadas, avaliadas


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
    snr[:borda] = snr[borda]                  # jan >= 3 => borda >= 1: sempre ha' borda a descartar
    snr[-borda:] = snr[-borda - 1]

    mascara_util = snr >= limiar_snr

    # Amplitude do sinal (para distinguir "morta" de "ruidosa")
    amp_rel = sd_sinal / max(float(np.max(sd_sinal)), 1e-12)

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
