"""
selecao_variaveis.py — Etapa 4: selecao de variaveis espectrais (iPLS por
intervalos, sPLS-DA esparso, SPA/APS, AG) + figuras de intervalos/convergencia/
comparacao de metodos.

Extraido de pipeline.py (Fase H). Usa modulos ja extraidos (chemometric_stats,
preprocessamento, validacao_estatistica, figuras, paleta_cores); Config so em
type hint (TYPE_CHECKING). pipeline.py reexporta (executar() chama
etapa4_selecao_variaveis).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from guaraci.chemometric_stats import vip_scores, compute_selectivity_ratio
from guaraci.validacao_estatistica import (_cv_predict_manual,
                                           StableStratifiedGroupKFold)
from guaraci.figuras import save, color, _ticks_x_inteiros

if TYPE_CHECKING:
    pass

__all__ = [
    "selecao_ipls",
    "sparse_plsda_mask",
    "selecao_spa",
    "ga_selection",
    "cars_selecao",
    "uve_selecao",
    "estabilidade_selecao_entre_repeticoes",
    "fig_etapa4_ag_convergencia",
    "fig_etapa4_ipls",
    "fig_etapa4_comparacao",
    "etapa4_selecao_variaveis",
]


def _avaliar_subset_cv(X_sel: np.ndarray, Y_bin: np.ndarray, y_int: np.ndarray,
                        cv_indices: list, n_lv: int) -> Dict[str, float]:
    """Avalia um subconjunto de variaveis por CV group-aware (mesmos folds
    do pipeline). Mean-centering re-ajustado por fold (sem leakage). PLS-DA.
    Retorna accuracy, balanced_accuracy, Q2 e n_vars."""
    n_lv_eff = int(max(1, min(n_lv, X_sel.shape[1], X_sel.shape[0] - 1)))

    def _fac():
        return Pipeline([
            ("mc",  StandardScaler(with_std=False)),
            ("pls", PLSRegression(n_components=n_lv_eff, scale=False)),
        ])

    y_cv = _cv_predict_manual(_fac, X_sel, Y_bin, cv_indices)
    ss_tot = float(np.sum((Y_bin - Y_bin.mean(axis=0)) ** 2))
    ss_res = float(np.sum((Y_bin - y_cv) ** 2))
    # Guard: non-finite ss_res = numerical blow-up in ill-conditioned PLS fold
    # (common in narrow iPLS intervals with near-collinear variables).
    if ss_tot < 1e-12 or not np.isfinite(ss_res):
        q2 = float("nan")
    else:
        q2 = max(-1.0, 1.0 - ss_res / ss_tot)
    yhat = np.argmax(y_cv, axis=1)
    return {
        "accuracy":          float(accuracy_score(y_int, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y_int, yhat)),
        "q2":                float(q2),
        "n_vars":            int(X_sel.shape[1]),
        "n_lv":              n_lv_eff,
    }


# =========================================================================
#  Selecao ANINHADA (nested-CV) para VIP/SR/sPLS-DA
#
#  Auditoria de 2026-07-12 (CLAUDE.md secao 13, item 03): VIP>=threshold,
#  SR top-fracao e sPLS-DA escolhiam a mascara de variaveis usando um
#  modelo ajustado no dataset INTEIRO (via vip/sr pre-calculados em
#  pipeline.py a partir de pls_final.fit(X_processed, ...)) e so DEPOIS
#  avaliavam o subconjunto ja fixo por CV. Isso e' "double dipping"
#  (Ambroise & McLachlan, 2002, PNAS 99:6562-6566): a selecao ve rotulo
#  de amostras que mais tarde servem de fold de validacao, inflando o
#  balanced_accuracy reportado em relacao a uma selecao re-feita a cada
#  fold. iPLS fica de fora desta correcao: a particao em intervalos NAO
#  usa rotulo (so a escolha do "melhor intervalo" usa CV), o mesmo tipo
#  de vies brando de qualquer selecao de hiperparametro via CV -- nao o
#  double dipping que este bloco corrige.
# =========================================================================

def _mask_vip_threshold(X_train: np.ndarray, Y_train: np.ndarray,
                         n_lv: int, threshold: float) -> np.ndarray:
    """Mascara VIP>=threshold usando SO dados de treino do fold."""
    n_lv_eff = int(max(1, min(n_lv, X_train.shape[1], X_train.shape[0] - 1)))
    pls = PLSRegression(n_components=n_lv_eff, scale=False)
    pls.fit(X_train, Y_train)
    vip = vip_scores(pls)
    return np.asarray(vip) >= threshold


def _mask_melhor_intervalo(X_train: np.ndarray, Y_train: np.ndarray,
                            n_lv: int, n_intervalos: int,
                            seed: int) -> np.ndarray:
    """Mascara do melhor intervalo iPLS escolhido usando SO' dados de treino
    do fold (CV interna propria), nunca a particao que reporta o resultado.

    Achado B1-1 da auditoria de 2026-08-16: `selecao_ipls` escolhia o melhor
    intervalo por balanced_accuracy calculado sobre `cv_indices`, e
    `etapa4_selecao_variaveis` reavaliava o vencedor NA MESMA `cv_indices`
    para a tabela final -- vies de maximo-de-N sobre a particao que reporta.
    O comentario antigo do modulo justificava excluir o iPLS do nested-CV
    porque "a particao em intervalos NAO usa rotulo": verdade, e irrelevante
    -- a ESCOLHA DO MELHOR intervalo usa. Vies medido: **+0,070 pontos de
    balanced accuracy, positivo em 12/12 seeds** (
    scripts/medicoes/medir_selecao_variaveis.py, funcao b1_1), contra um limiar de desempate
    de 1% usado por `etapa4_selecao_variaveis` para eleger o metodo mais
    parcimonioso -- ou seja, o vies era 7x maior que o criterio de decisao,
    e favorecia sistematicamente o iPLS numa tabela onde todos os OUTROS
    metodos ja passavam por nested-CV."""
    p = X_train.shape[1]
    bordas = np.linspace(0, p, n_intervalos + 1).astype(int)
    y_train_int = np.argmax(Y_train, axis=1)
    cv_interna = _cv_local(y_train_int, seed)
    melhor_bal, melhor_ab = -1.0, (0, p)
    for i in range(n_intervalos):
        a, b = bordas[i], bordas[i + 1]
        if b - a < 2:
            continue
        m = _avaliar_subset_cv(X_train[:, a:b], Y_train, y_train_int,
                                cv_interna, n_lv)
        if m["balanced_accuracy"] > melhor_bal:
            melhor_bal, melhor_ab = m["balanced_accuracy"], (a, b)
    mask = np.zeros(p, dtype=bool)
    mask[melhor_ab[0]:melhor_ab[1]] = True
    return mask


def _mask_sr_top_frac(X_train: np.ndarray, Y_train: np.ndarray,
                       n_lv: int, top_frac: float) -> np.ndarray:
    """Mascara top-fracao por Selectivity Ratio usando SO dados de treino."""
    n_lv_eff = int(max(1, min(n_lv, X_train.shape[1], X_train.shape[0] - 1)))
    pls = PLSRegression(n_components=n_lv_eff, scale=False)
    pls.fit(X_train, Y_train)
    sr = compute_selectivity_ratio(pls, X_train)
    p = X_train.shape[1]
    n_top = max(2, int(round(top_frac * p)))
    idx = np.argsort(np.asarray(sr))[::-1][:n_top]
    mask = np.zeros(p, dtype=bool)
    mask[idx] = True
    return mask


def _avaliar_subset_nested_cv(X_proc: np.ndarray, Y_bin: np.ndarray,
                               y_int: np.ndarray, cv_indices: list,
                               n_lv: int, selecionar_fn) -> Dict[str, float]:
    """Como `_avaliar_subset_cv`, mas refaz a SELECAO de variaveis a cada
    fold usando `selecionar_fn(X_treino, Y_treino, n_lv) -> mascara`,
    sem olhar as amostras de validacao daquele fold. Se um fold selecionar
    menos de 2 variaveis (raro, threshold agressivo + fold pequeno), cai
    de volta para todas as variaveis NAQUELE fold, em vez de descartar o
    fold inteiro -- mantem a CV cobrindo 100% das amostras.

    `n_vars` no retorno e' a MEDIA de variaveis selecionadas entre folds
    (pode variar fold a fold, ao contrario da selecao nao-aninhada, que
    tinha um n_vars fixo); `n_vars_min`/`n_vars_max` dao o intervalo.
    """
    p = X_proc.shape[1]
    y_hat = np.zeros_like(Y_bin, dtype=float)
    contador = np.zeros(len(Y_bin), dtype=int)
    n_vars_por_fold: List[int] = []
    for tr, va in cv_indices:
        mask = selecionar_fn(X_proc[tr], Y_bin[tr], n_lv)
        if mask is None or mask.sum() < 2:
            mask = np.ones(p, dtype=bool)
        n_vars_por_fold.append(int(mask.sum()))
        n_lv_eff = int(max(1, min(n_lv, int(mask.sum()), len(tr) - 1)))
        pipe = Pipeline([
            ("mc",  StandardScaler(with_std=False)),
            ("pls", PLSRegression(n_components=n_lv_eff, scale=False)),
        ])
        pipe.fit(X_proc[tr][:, mask], Y_bin[tr])
        y_hat[va] += pipe.predict(X_proc[va][:, mask])
        contador[va] += 1
    contador[contador == 0] = 1
    y_cv = y_hat / contador[:, None]

    ss_tot = float(np.sum((Y_bin - Y_bin.mean(axis=0)) ** 2))
    ss_res = float(np.sum((Y_bin - y_cv) ** 2))
    if ss_tot < 1e-12 or not np.isfinite(ss_res):
        q2 = float("nan")
    else:
        q2 = max(-1.0, 1.0 - ss_res / ss_tot)
    yhat = np.argmax(y_cv, axis=1)
    return {
        "accuracy":          float(accuracy_score(y_int, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y_int, yhat)),
        "q2":                float(q2),
        "n_vars":            float(np.mean(n_vars_por_fold)),
        "n_vars_min":        int(min(n_vars_por_fold)),
        "n_vars_max":        int(max(n_vars_por_fold)),
        "n_lv":              n_lv,
    }


# =========================================================================
#  Selecao ANINHADA (nested-CV) para metodos de BUSCA (AG/SPA)
#
#  Achado colateral da correcao de 2026-07-13 (CLAUDE.md secao 8): a
#  *fitness* do AG (a cada individuo, a cada geracao) e a pontuacao usada
#  p/ escolher a melhor cadeia do SPA sao ambas `_avaliar_subset_cv` na
#  MESMA `cv_indices` cujo resultado e' depois reportado como numero final
#  na tabela da Etapa 4 -- a busca otimiza DIRETAMENTE contra a particao
#  que mede o resultado. E' double dipping mais severo que o do VIP/SR
#  (aqui a busca tem centenas de avaliacoes tentando "acertar" a mesma CV).
#
#  Correcao: nested-CV completo. A CADA fold EXTERNO (cv_indices, group-
#  aware), a busca inteira (GA ou SPA) roda de novo usando SO' os dados de
#  TREINO daquele fold, com uma CV INTERNA propria (StratifiedKFold local,
#  nao group-aware -- mae_id nao chega ate aqui) para guiar a fitness. A
#  mascara resultante e' avaliada no fold de TESTE externo, nunca visto
#  pela busca. Custo: a busca roda ~len(cv_indices) vezes mais (aceitavel
#  -- AG/SPA ja sao opt-in e documentados como mais lentos).
# =========================================================================

def _cv_local(y_local: np.ndarray, seed: int,
              grupos_local: Optional[np.ndarray] = None,
              n_splits: int = 3) -> list:
    """K-fold local (indices 0..len(y_local)-1) p/ guiar a busca DENTRO de
    um fold externo.

    GROUP-AWARE quando `grupos_local` (mae_id do fold externo) e' fornecido
    -- correcao do achado B1-3 da auditoria de 2026-08-16. A versao anterior
    usava sempre `StratifiedKFold`, com a justificativa de que "so' orienta a
    otimizacao; o numero reportado usa o fold externo group-aware". O
    argumento esta' metade certo: o NUMERO reportado e' de fato honesto, mas
    o produto cientifico da Etapa 4 nao e' o bal.acc -- sao as VARIAVEIS
    SELECIONADAS, e uma busca guiada por uma particao onde replicas vazam
    prefere justamente as variaveis que exploram similaridade entre
    replicas. Isto e', o software aplicava a si mesmo, no passo de selecao,
    o erro que existe para combater.

    Sem `grupos_local` (None): cai para `StratifiedKFold` como antes --
    necessario quando nao ha' identificador de replica disponivel."""
    y_local = np.asarray(y_local)
    if grupos_local is not None:
        grupos_local = np.asarray(grupos_local)
        n_grupos = int(len(np.unique(grupos_local)))
        # Cada fold interno precisa de >=1 grupo; com poucos grupos o
        # splitter group-aware nao tem material e cai para o estratificado.
        if n_grupos >= 2:
            n_splits_eff = max(2, min(n_splits, n_grupos))
            gkf = StableStratifiedGroupKFold(n_splits=n_splits_eff, seed=seed)
            return list(gkf.split(np.zeros(len(y_local)), y_local,
                                  groups=grupos_local))
    from sklearn.model_selection import StratifiedKFold
    _classes, contagens = np.unique(y_local, return_counts=True)
    n_splits_eff = max(2, min(n_splits, int(contagens.min())))
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y_local)), y_local))


def _avaliar_busca_nested_cv(X_proc: np.ndarray, Y_bin: np.ndarray,
                              y_int: np.ndarray, cv_indices: list,
                              n_lv: int, buscar_fn, seed: int,
                              mae_id: Optional[np.ndarray] = None
                              ) -> Dict[str, float]:
    """Nested-CV p/ AG/SPA: `buscar_fn(X_treino, Y_treino, y_treino,
    cv_interna) -> mascara` roda a busca completa usando so' o subconjunto
    de TREINO do fold externo (reindexado localmente) + uma CV interna
    propria; a mascara e' avaliada no fold de TESTE externo (nunca visto
    pela busca). Mesma estrutura de retorno de `_avaliar_subset_nested_cv`.

    `mae_id` (opcional): quando fornecido, a CV INTERNA que guia a busca
    tambem e' group-aware (achado B1-3) -- ver docstring de `_cv_local`."""
    p = X_proc.shape[1]
    y_hat = np.zeros_like(Y_bin, dtype=float)
    contador = np.zeros(len(Y_bin), dtype=int)
    n_vars_por_fold: List[int] = []
    for tr, va in cv_indices:
        X_tr, Y_tr, y_tr = X_proc[tr], Y_bin[tr], y_int[tr]
        grupos_tr = mae_id[tr] if mae_id is not None else None
        cv_interna = _cv_local(y_tr, seed, grupos_local=grupos_tr)
        mask = buscar_fn(X_tr, Y_tr, y_tr, cv_interna)
        if mask is None or mask.sum() < 2:
            mask = np.ones(p, dtype=bool)
        n_vars_por_fold.append(int(mask.sum()))
        n_lv_eff = int(max(1, min(n_lv, int(mask.sum()), len(tr) - 1)))
        pipe = Pipeline([
            ("mc",  StandardScaler(with_std=False)),
            ("pls", PLSRegression(n_components=n_lv_eff, scale=False)),
        ])
        pipe.fit(X_proc[tr][:, mask], Y_bin[tr])
        y_hat[va] += pipe.predict(X_proc[va][:, mask])
        contador[va] += 1
    contador[contador == 0] = 1
    y_cv = y_hat / contador[:, None]

    ss_tot = float(np.sum((Y_bin - Y_bin.mean(axis=0)) ** 2))
    ss_res = float(np.sum((Y_bin - y_cv) ** 2))
    if ss_tot < 1e-12 or not np.isfinite(ss_res):
        q2 = float("nan")
    else:
        q2 = max(-1.0, 1.0 - ss_res / ss_tot)
    yhat = np.argmax(y_cv, axis=1)
    return {
        "accuracy":          float(accuracy_score(y_int, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y_int, yhat)),
        "q2":                float(q2),
        "n_vars":            float(np.mean(n_vars_por_fold)),
        "n_vars_min":        int(min(n_vars_por_fold)),
        "n_vars_max":        int(max(n_vars_por_fold)),
        "n_lv":              n_lv,
    }


def selecao_ipls(X_proc, Y_bin, y_int, wavenumbers, cv_indices, n_lv,
                 n_intervalos: int) -> Tuple[list, np.ndarray]:
    """interval-PLS: divide o espectro em n_intervalos contiguos, avalia
    PLS-DA em cada um. Retorna (lista de resultados por intervalo,
    mascara do melhor intervalo)."""
    p = X_proc.shape[1]
    bordas = np.linspace(0, p, n_intervalos + 1).astype(int)
    resultados = []
    for i in range(n_intervalos):
        a, b = bordas[i], bordas[i + 1]
        if b - a < 2:
            continue
        mask = np.zeros(p, dtype=bool); mask[a:b] = True
        m = _avaliar_subset_cv(X_proc[:, mask], Y_bin, y_int, cv_indices, n_lv)
        m["intervalo"] = i + 1
        m["wn_ini"] = float(wavenumbers[a])
        m["wn_fim"] = float(wavenumbers[b - 1])
        m["idx_a"] = int(a)
        m["idx_b"] = int(b)
        resultados.append(m)
    melhor = max(resultados, key=lambda r: r["balanced_accuracy"])
    a, b = int(melhor["idx_a"]), int(melhor["idx_b"])
    mask_melhor = np.zeros(p, dtype=bool); mask_melhor[a:b] = True
    return resultados, mask_melhor


def sparse_plsda_mask(X_proc, Y_bin, n_comp: int,
                      keep_por_comp: int) -> np.ndarray:
    """sPLS-DA (Le Cao et al. 2008): NIPALS com penalizacao por
    SOFT-THRESHOLDING no vetor de loading, mantendo `keep_por_comp`
    variaveis nao-nulas por componente. Retorna mascara da uniao das
    variaveis selecionadas por componente.

    O operador e' o da referencia:

        w_j <- sign(w_j) * max(|w_j| - lambda, 0)

    com `lambda` = o (keep_por_comp+1)-esimo maior |w_j|, o que faz o
    numero de sobreviventes ser exatamente `keep_por_comp` (a mesma
    parametrizacao por contagem que o mixOmics expoe como `keepX`) --
    mas, ao contrario do truncamento duro, as sobreviventes sao
    ENCOLHIDAS por lambda, e e' esse encolhimento que muda a direcao
    normalizada de w, logo o escore t, logo a deflacao, logo o conjunto
    escolhido pelos componentes seguintes.

    CORRIGIDO em 2026-08-16 (achado B1-2): a versao anterior fazia
    truncamento DURO (`argsort(|w|)[:keep]`, zerando o resto sem encolher
    as sobreviventes) enquanto a docstring a descrevia como
    "soft-selection" -- divergencia da referencia citada E contradicao
    interna, a mesma classe do achado A5 da auditoria de 2026-08-07.
    Divergencia medida entre as duas variantes (
    scripts/medicoes/medir_selecao_variaveis.py, funcao b1_2): Jaccard=1,000 com 1
    componente (identicas por construcao) caindo a ~0,87 com 5.

    Referencia:
        Le Cao K.-A., Rossouw D., Robert-Granie C. & Besse P. (2008).
        A Sparse PLS for Variable Selection when Integrating Omics Data.
        Stat. Appl. Genet. Mol. Biol. 7(1):35.
    """
    X = np.asarray(X_proc, dtype=float)
    Y = np.asarray(Y_bin, dtype=float)
    Xr = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    p = X.shape[1]
    selecionadas: set = set()
    n_comp = int(max(1, min(n_comp, p)))
    keep = int(max(1, min(keep_por_comp, p)))
    for _ in range(n_comp):
        M = Xr.T @ Yc                      # (p, m)
        try:
            U, _S, _Vt = np.linalg.svd(M, full_matrices=False)
            w = U[:, 0]
        except np.linalg.LinAlgError:
            break
        # Soft-threshold com lambda = (keep+1)-esimo maior |w|: zera todas
        # menos `keep` e encolhe as que sobram por esse mesmo lambda.
        aw = np.abs(w)
        lam = float(np.sort(aw)[::-1][keep]) if keep < p else 0.0
        w_sp = np.sign(w) * np.maximum(aw - lam, 0.0)
        nw = float(np.linalg.norm(w_sp))
        if nw < 1e-12:
            break
        w_sp /= nw
        t = Xr @ w_sp
        tt = float(t @ t)
        if tt < 1e-12:
            break
        pld = Xr.T @ t / tt
        Xr = Xr - np.outer(t, pld)
        c = Yc.T @ t / tt
        Yc = Yc - np.outer(t, c)
        selecionadas.update(int(i) for i in np.flatnonzero(w_sp))
    mask = np.zeros(p, dtype=bool)
    if selecionadas:
        mask[list(selecionadas)] = True
    return mask


# =========================================================================
#  SPA / APS — Algoritmo das Projecoes Sucessivas (Araujo et al. 2001,
#  Chemom. Intell. Lab. Syst. 57:65-73)
# =========================================================================

def _spa_cadeia(X: np.ndarray, idx_inicial: int, n_vars_max: int) -> np.ndarray:
    """Constroi UMA cadeia SPA a partir de uma variavel inicial.

    A cada passo, projeta TODAS as variaveis candidatas ainda nao
    selecionadas ortogonalmente ao subespaco das ja selecionadas (deflacao
    cumulativa, estilo Gram-Schmidt) e escolhe a de MAIOR norma residual —
    minimiza a colinearidade entre as variaveis escolhidas (Araujo et al.
    2001). Retorna os indices selecionados, na ordem em que entraram.
    """
    n, p = X.shape
    n_vars_max = int(max(1, min(n_vars_max, p, n)))
    Xw = X.astype(float).copy()
    selecionadas = [int(idx_inicial)]
    disponiveis = [j for j in range(p) if j != idx_inicial]
    vetor_atual = Xw[:, idx_inicial]

    for _ in range(n_vars_max - 1):
        if not disponiveis:
            break
        norma_sq = float(vetor_atual @ vetor_atual)
        if norma_sq < 1e-12:
            break
        melhor_j, melhor_norma = None, -1.0
        for j in disponiveis:
            xj = Xw[:, j]
            coef = float(vetor_atual @ xj) / norma_sq
            Xw[:, j] = xj - coef * vetor_atual   # deflacao cumulativa, persiste
            nr = float(np.linalg.norm(Xw[:, j]))
            if nr > melhor_norma:
                melhor_norma, melhor_j = nr, j
        selecionadas.append(int(melhor_j))
        disponiveis.remove(melhor_j)
        vetor_atual = Xw[:, melhor_j]

    return np.array(selecionadas, dtype=int)


def selecao_spa(X_proc: np.ndarray, Y_bin: np.ndarray, y_int: np.ndarray,
                 cv_indices: list, n_lv: int, n_vars_max: int, n_starts: int,
                 seed: int) -> Tuple[List[Dict], np.ndarray]:
    """SPA/APS: constroi cadeias de baixa colinearidade a partir de varios
    pontos de partida (distribuidos uniformemente pelo espectro, para
    limitar o custo sem deixar de cobrir toda a faixa espectral) e escolhe
    a cadeia com maior balanced_accuracy via CV — mesmo esquema de
    avaliacao (`_avaliar_subset_cv`) do resto da Etapa 4.

    Retorna (lista de resultados por ponto de partida, mascara da melhor
    cadeia)."""
    p = X_proc.shape[1]
    n_starts_eff = int(max(1, min(n_starts, p)))
    starts = np.unique(np.linspace(0, p - 1, n_starts_eff).astype(int))

    resultados: List[Dict] = []
    melhor_bal = -1.0
    melhor_mask: np.ndarray = np.zeros(p, dtype=bool)

    for k0 in starts:
        cadeia = _spa_cadeia(X_proc, int(k0), n_vars_max)
        if len(cadeia) < 2:
            continue
        mask = np.zeros(p, dtype=bool)
        mask[cadeia] = True
        m = _avaliar_subset_cv(X_proc[:, mask], Y_bin, y_int, cv_indices, n_lv)
        m["inicio_idx"] = int(k0)
        resultados.append(m)
        if m["balanced_accuracy"] > melhor_bal:
            melhor_bal = m["balanced_accuracy"]
            melhor_mask = mask

    return resultados, melhor_mask


# =========================================================================
#  AG — Algoritmo Genetico para selecao de variaveis (GA-PLS; Leardi 2000
#  e variantes: populacao binaria + fitness via CV + torneio/crossover/
#  mutacao/elitismo)
# =========================================================================

def _torneio_ag(populacao: np.ndarray, fitnesses: np.ndarray,
                 rng: np.random.Generator, k: int = 3) -> np.ndarray:
    """Selecao por torneio: sorteia k cromossomos, devolve o de maior fitness."""
    idxs = rng.choice(len(populacao), size=min(k, len(populacao)), replace=False)
    melhor_idx = idxs[int(np.argmax(fitnesses[idxs]))]
    return populacao[melhor_idx].copy()


def ga_selection(X_proc: np.ndarray, Y_bin: np.ndarray, y_int: np.ndarray,
                cv_indices: list, n_lv: int, tam_populacao: int,
                n_geracoes: int, prob_mutacao: float, frac_inicial: float,
                seed: int) -> Tuple[List[Dict], np.ndarray]:
    """AG (Algoritmo Genetico) para selecao de variaveis: cada cromossomo e
    um vetor binario (1 = variavel selecionada); fitness = balanced_accuracy
    via CV (mesmo `_avaliar_subset_cv` do resto da Etapa 4). Selecao por
    torneio (k=3), crossover de 1 ponto, mutacao bit-flip, elitismo (o
    melhor cromossomo da geracao sempre sobrevive).

    Retorna (historico por geracao [melhor/media fitness, n_vars], mascara
    do melhor cromossomo encontrado em toda a busca)."""
    p = X_proc.shape[1]
    rng = np.random.default_rng(seed)

    def _garantir_min_2(cromo: np.ndarray) -> np.ndarray:
        if cromo.sum() < 2:
            idx = rng.choice(p, size=2, replace=False)
            cromo = cromo.copy()
            cromo[idx] = True
        return cromo

    def _fitness(mask: np.ndarray) -> float:
        if mask.sum() < 2:
            return -1.0
        m = _avaliar_subset_cv(X_proc[:, mask], Y_bin, y_int, cv_indices, n_lv)
        return float(m["balanced_accuracy"])

    populacao = rng.random((tam_populacao, p)) < frac_inicial
    populacao = np.array([_garantir_min_2(populacao[i]) for i in range(tam_populacao)])

    historico: List[Dict] = []
    melhor_mask = np.zeros(p, dtype=bool)
    melhor_fitness = -1.0

    for geracao in range(n_geracoes):
        fitnesses = np.array([_fitness(populacao[i]) for i in range(tam_populacao)])
        idx_melhor_ger = int(np.argmax(fitnesses))
        if fitnesses[idx_melhor_ger] > melhor_fitness:
            melhor_fitness = float(fitnesses[idx_melhor_ger])
            melhor_mask = populacao[idx_melhor_ger].copy()
        historico.append({
            "geracao": geracao + 1,
            "melhor_fitness": float(fitnesses.max()),
            "media_fitness": float(fitnesses.mean()),
            "n_vars_melhor": int(populacao[idx_melhor_ger].sum()),
        })

        nova_populacao = np.empty_like(populacao)
        nova_populacao[0] = melhor_mask   # elitismo
        for i in range(1, tam_populacao):
            pai1 = _torneio_ag(populacao, fitnesses, rng)
            pai2 = _torneio_ag(populacao, fitnesses, rng)
            ponto = int(rng.integers(1, p)) if p > 1 else 1
            filho = np.concatenate([pai1[:ponto], pai2[ponto:]])
            mutar = rng.random(p) < prob_mutacao
            filho[mutar] = ~filho[mutar]
            nova_populacao[i] = _garantir_min_2(filho)
        populacao = nova_populacao

    return historico, melhor_mask


# =========================================================================
#  CARS -- Competitive Adaptive Reweighted Sampling (Li, Liang, Xu & Cao,
#  2009, "Key wavelengths screening using competitive adaptive reweighted
#  sampling method for multivariate calibration", Analytica Chimica Acta
#  648:77-84, DOI 10.1016/j.aca.2009.06.046 -- verificado no Crossref em
#  2026-09-04)
#
#  ADAPTACAO: o artigo original usa RMSECV de PLS de regressao (y
#  continuo, univariado). Este pipeline e' PLS-DA de classificacao (Y_bin
#  one-hot multi-coluna) -- mesma adaptacao ja' usada pelo AG/SPA deste
#  modulo: o criterio de "melhor iteracao" e' balanced_accuracy via CV
#  (`_avaliar_subset_cv`), nao RMSECV bruto. Direcao da otimizacao e'
#  equivalente (ambos escolhem o subconjunto que generaliza melhor).
# =========================================================================

def _edf_contagens(p: int, n_iteracoes: int) -> np.ndarray:
    """Funcao Exponencialmente Decrescente (EDF) do CARS: numero de
    variaveis "candidatas" permitido a cada iteracao, caindo de `p` (i=0)
    a 2 (ultima iteracao) -- agressiva no inicio, fina no fim (formula do
    artigo original: r_i = a*exp(-k*i), a=1, k=ln(p/2)/(n_iteracoes-1))."""
    if n_iteracoes <= 1:
        return np.array([p])
    k = np.log(p / 2.0) / (n_iteracoes - 1)
    i = np.arange(n_iteracoes)
    contagens = np.round(p * np.exp(-k * i)).astype(int)
    return np.clip(contagens, 2, p)


def _coef_por_variavel(pls: PLSRegression, n_vars: int) -> np.ndarray:
    """Norma do coeficiente PLS por variavel, agregando sobre as colunas do
    alvo (Y_bin multi-classe one-hot) -- `pls.coef_` muda de orientacao
    entre versoes do sklearn ((n_features, n_targets) vs
    (n_targets, n_features)), esta funcao normaliza para (n_vars,)."""
    coef = np.asarray(pls.coef_)
    if coef.ndim == 1:
        return np.abs(coef)
    if coef.shape[0] == n_vars:
        return np.linalg.norm(coef, axis=1)
    return np.linalg.norm(coef, axis=0)


def cars_selecao(X_proc: np.ndarray, Y_bin: np.ndarray, y_int: np.ndarray,
                  cv_indices: list, n_lv: int, n_iteracoes: int,
                  frac_amostragem: float, seed: int
                  ) -> Tuple[List[Dict], np.ndarray]:
    """CARS: a cada iteracao, amostra `frac_amostragem` das amostras
    (Monte Carlo sampling SEM reposicao), ajusta PLS no subconjunto,
    calcula |coeficiente| por variavel. A EDF (`_edf_contagens`) diminui o
    numero maximo de variaveis candidatas a cada iteracao; dentre as
    candidatas atuais, ARS (Adaptive Reweighted Sampling) sorteia quais
    sobrevivem via ROLETA ponderada por |coeficiente| (nao corte duro por
    ranking -- e' o que distingue CARS de VIP/SR: uma variavel de
    coeficiente baixo ainda tem chance de sobreviver, uma de coeficiente
    alto tem MAIOR chance mas nao certeza). balanced_accuracy via CV
    decide a melhor iteracao (ver nota de adaptacao no cabecalho da secao).

    Retorna (historico por iteracao, mascara da melhor iteracao)."""
    p = X_proc.shape[1]
    n = X_proc.shape[0]
    rng = np.random.default_rng(seed)
    contagens = _edf_contagens(p, n_iteracoes)
    n_amostras_sub = max(2, int(round(frac_amostragem * n)))

    vars_ativas = np.arange(p)
    historico: List[Dict] = []
    melhor_bal = -1.0
    melhor_mask = np.ones(p, dtype=bool)

    for it, n_manter in enumerate(contagens):
        idx_amostras = rng.choice(n, size=n_amostras_sub, replace=False)
        n_lv_eff = int(max(1, min(n_lv, len(vars_ativas), n_amostras_sub - 1)))
        Xs = X_proc[np.ix_(idx_amostras, vars_ativas)]
        Ys = Y_bin[idx_amostras]
        pls = PLSRegression(n_components=n_lv_eff, scale=False)
        pls.fit(Xs - Xs.mean(axis=0), Ys - Ys.mean(axis=0))
        b = _coef_por_variavel(pls, len(vars_ativas))

        n_manter_local = int(max(2, min(int(n_manter), len(vars_ativas))))
        if n_manter_local < len(vars_ativas):
            pesos = np.abs(b)
            soma = float(pesos.sum())
            probs = (pesos / soma) if soma > 1e-300 else None
            escolhidos_local = rng.choice(len(vars_ativas), size=n_manter_local,
                                          replace=False, p=probs)
            vars_ativas = np.sort(vars_ativas[escolhidos_local])

        mask_atual = np.zeros(p, dtype=bool)
        mask_atual[vars_ativas] = True
        m = _avaliar_subset_cv(X_proc[:, mask_atual], Y_bin, y_int, cv_indices, n_lv)
        historico.append({"iteracao": it + 1, "n_vars": int(mask_atual.sum()),
                          "balanced_accuracy": m["balanced_accuracy"],
                          "q2": m["q2"]})
        if m["balanced_accuracy"] > melhor_bal:
            melhor_bal = m["balanced_accuracy"]
            melhor_mask = mask_atual.copy()

    return historico, melhor_mask


# =========================================================================
#  UVE -- Uninformative Variable Elimination (Centner, Massart, de Noord,
#  de Jong, Vandeginste & Sterna, 1996, "Elimination of Uninformative
#  Variables for Multivariate Calibration", Analytical Chemistry
#  68(21):3851-3858, DOI 10.1021/ac960321m -- verificado no Crossref em
#  2026-09-04)
# =========================================================================

def uve_selecao(X_proc: np.ndarray, Y_bin: np.ndarray, n_lv: int,
                 n_repeticoes: int, frac_amostragem: float, seed: int
                 ) -> Tuple[Dict, np.ndarray]:
    """UVE: concatena `p` variaveis de RUIDO artificial (amplitude ~1e-10x
    menor que o desvio-padrao dos dados reais -- nao deve carregar sinal
    nenhum, so serve de referencia) as `p` variaveis reais. Roda PLS
    repetidas vezes em subamostras Monte Carlo (sem reposicao) das
    amostras recebidas, coleta o coeficiente por variavel a cada
    repeticao, e calcula a estabilidade c_j = media(coef_j) / desvio(coef_j).
    O corte e' o MAIOR |c_j| entre as variaveis de RUIDO: uma variavel
    real com |c_j| <= corte e' estatisticamente indistinguivel do ruido
    artificial e e' eliminada.

    Retorna (dict de diagnostico com c_scores/corte, mascara das
    variaveis reais mantidas)."""
    X = np.asarray(X_proc, dtype=float)
    Y = np.asarray(Y_bin, dtype=float)
    n, p = X.shape
    rng = np.random.default_rng(seed)

    escala_ruido = float(X.std()) * 1e-10
    ruido = rng.normal(0.0, 1.0, size=(n, p)) * escala_ruido
    X_aug = np.hstack([X, ruido])

    n_amostras_sub = max(2, int(round(frac_amostragem * n)))
    coefs = np.empty((n_repeticoes, 2 * p))
    for r in range(n_repeticoes):
        idx = rng.choice(n, size=n_amostras_sub, replace=False)
        n_lv_eff = int(max(1, min(n_lv, 2 * p, n_amostras_sub - 1)))
        Xs, Ys = X_aug[idx], Y[idx]
        pls = PLSRegression(n_components=n_lv_eff, scale=False)
        pls.fit(Xs - Xs.mean(axis=0), Ys - Ys.mean(axis=0))
        coefs[r] = _coef_por_variavel(pls, 2 * p)

    media = coefs.mean(axis=0)
    desvio = coefs.std(axis=0)
    desvio_seguro = np.where(desvio > 1e-300, desvio, 1e-300)
    c = media / desvio_seguro
    c_real, c_ruido = c[:p], c[p:]
    corte = float(np.max(np.abs(c_ruido)))
    mask = np.abs(c_real) > corte

    info = {"c_scores": c_real, "corte": corte}
    return info, mask


def estabilidade_selecao_entre_repeticoes(selecionar_fn, n_repeticoes: int = 5,
                                           seed_base: int = 0) -> Dict:
    """Estabilidade de selecao de variaveis: roda `selecionar_fn(seed) ->
    mascara` `n_repeticoes` vezes com seeds diferentes e mede o indice de
    Jaccard PAREADO entre todas as mascaras resultantes (|intersecao| /
    |uniao|; 1.0 = identico, 0.0 = disjunto). Generico o bastante para
    comparar CARS/UVE/iPLS/VIP -- qualquer metodo cuja selecao dependa de
    aleatoriedade (amostragem Monte Carlo, cadeias) ou de reparticao de
    CV entre execucoes."""
    mascaras = [selecionar_fn(seed_base + k) for k in range(n_repeticoes)]
    jaccards = []
    for i in range(len(mascaras)):
        for j in range(i + 1, len(mascaras)):
            inter = int(np.logical_and(mascaras[i], mascaras[j]).sum())
            uniao = int(np.logical_or(mascaras[i], mascaras[j]).sum())
            jaccards.append(inter / uniao if uniao > 0 else 1.0)
    return {
        "jaccard_medio": float(np.mean(jaccards)) if jaccards else 1.0,
        "jaccard_desvio": float(np.std(jaccards)) if jaccards else 0.0,
        "jaccards_pareados": jaccards,
        "n_vars_por_repeticao": [int(m.sum()) for m in mascaras],
    }


def fig_etapa4_ag_convergencia(historico: List[Dict], cfg, pasta):
    """Convergencia do AG: melhor e media fitness (balanced_accuracy) por
    geracao — diagnostico padrao de algoritmos evolutivos."""
    geracoes = [h["geracao"] for h in historico]
    melhores  = [h["melhor_fitness"] for h in historico]
    medias    = [h["media_fitness"] for h in historico]

    fig, ax = plt.subplots(figsize=(9.0, 4.0), constrained_layout=True)
    ax.plot(geracoes, melhores, color=color(2), lw=1.6, marker="o", ms=3.5,
            label="Melhor da geracao")
    ax.plot(geracoes, medias, color=color(3), lw=1.2, ls="--",
            label="Media da populacao")
    ax.set_xlabel("Geracao")
    ax.set_ylabel("Balanced accuracy (CV)")
    ax.set_title("Etapa 4 — AG: convergencia da busca genetica", loc="left")
    ax.grid(axis="y", color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "fig_etapa4_ag_convergencia", pasta, cfg)


def fig_etapa4_ipls(resultados, wavenumbers, baseline_bal, cfg, pasta):
    """Barras de balanced_acc por intervalo iPLS, com linha do modelo global."""
    intervalos = [r["intervalo"] for r in resultados]
    bals       = [r["balanced_accuracy"] for r in resultados]
    melhor_i   = int(np.argmax(bals))

    fig, ax = plt.subplots(figsize=(11.0, 4.2), constrained_layout=True)
    cores_b = [color(2) if k == melhor_i else "0.6" for k in range(len(bals))]
    ax.bar(intervalos, bals, color=cores_b, edgecolor="white", lw=0.5)
    ax.axhline(baseline_bal, color=color(3), ls="--", lw=1.2,
               label=f"Modelo global ({baseline_bal:.3f})")
    ax.set_xlabel("Intervalo iPLS")
    ax.set_ylabel("Balanced accuracy (CV)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Etapa 4 — iPLS: desempenho por intervalo espectral",
                  loc="left")
    # eixo secundario com faixa de wavenumber do melhor
    r = resultados[melhor_i]
    ax.annotate(f"melhor: {r['wn_ini']:.0f}-{r['wn_fim']:.0f} cm$^{{-1}}$\n"
                f"bal.acc={r['balanced_accuracy']:.3f} ({r['n_vars']} vars)",
                xy=(r["intervalo"], r["balanced_accuracy"]),
                xytext=(0.98, 0.05), textcoords="axes fraction",
                ha="right", va="bottom", fontsize=8.5, color="0.2",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.82"))
    _ticks_x_inteiros(ax, np.array(intervalos))
    ax.grid(axis="y", color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)
    save(fig, "fig_etapa4_ipls_intervalos", pasta, cfg)


def fig_etapa4_comparacao(tabela, cfg, pasta):
    """Compara metodos de selecao: balanced_acc (barras) + n_vars (texto)."""
    nomes = [t["metodo"] for t in tabela]
    bals  = [t["balanced_accuracy"] for t in tabela]
    nvars = [t["n_vars"] for t in tabela]
    pos = np.arange(len(nomes))

    fig, ax = plt.subplots(figsize=(10.0, 0.6 * len(nomes) + 2.0),
                            constrained_layout=True)
    cores_b = [color(0) if n == "Full (todas)" else color(2) for n in nomes]
    ax.barh(pos, bals, color=cores_b, edgecolor="white", lw=0.5, height=0.7)
    for k, (b, nv) in enumerate(zip(bals, nvars)):
        ax.text(min(b + 0.01, 1.0), k, f" {b:.3f} | {nv} vars",
                va="center", ha="left", fontsize=9)
    ax.set_yticks(pos); ax.set_yticklabels(nomes, fontsize=9.5)
    ax.set_xlim(0, 1.12); ax.invert_yaxis()
    ax.set_xlabel("Balanced accuracy (CV group-aware)")
    ax.set_title("Etapa 4 — comparacao de metodos de selecao de variaveis",
                  loc="left")
    ax.grid(axis="x", color="0.93", lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig_etapa4_comparacao_metodos", pasta, cfg)


def etapa4_selecao_variaveis(X_proc, Y_bin, y_int, wavenumbers,
                              cv_indices, n_lv, cfg, pasta, pasta_dados,
                              mae_id=None):
    """Orquestra a Etapa 4: avalia Full vs iPLS vs VIP vs SR vs sPLS-DA (sempre)
    e, se ligados em cfg, SPA/APS e AG (opt-in, mais lentos) — sob o MESMO
    esquema de CV group-aware. Salva tabela (dados/) e figuras (figuras/).
    Retorna dict-resumo para o resumo_modelo.txt.

    iPLS/VIP/SR/sPLS-DA usam selecao ANINHADA (nested-CV,
    `_avaliar_subset_nested_cv`): a mascara de variaveis e' recalculada a cada
    fold usando so as amostras de treino daquele fold, nao um vip/sr
    pre-calculado no dataset inteiro (ver correcao da auditoria de 2026-07-12,
    CLAUDE.md secao 13, item 03). Por isso esta funcao nao recebe mais
    `vip`/`sr` pre-calculados como parametro. O iPLS entrou nesse mesmo
    esquema em 2026-08-16 (achado B1-1: era o unico metodo da tabela cujo
    numero vinha de selecao feita na propria particao que o reportava, vies
    medido de +0,070 bal.acc).

    `mae_id` (opcional): identificador de replica fisica, usado para tornar
    group-aware tambem a CV INTERNA que guia as buscas SPA/AG (achado B1-3).

    SPA/AG (quando ligados) usam nested-CV equivalente (`_avaliar_busca_nested_cv`,
    achado colateral de 2026-07-13): a fitness/pontuacao usada pela BUSCA nunca
    e' a mesma cv_indices do numero reportado. A busca no dataset inteiro
    continua rodando 1x so' para as figuras/CSV de diagnostico (convergencia
    do AG, cadeias do SPA) — nao para o bal.acc reportado na tabela."""
    print("\n[Etapa4] Selecao de variaveis "
          f"(iPLS, VIP, SR, sPLS-DA"
          f"{', SPA' if cfg.run_spa else ''}"
          f"{', AG' if cfg.executar_ag else ''}"
          f"{', CARS' if cfg.run_cars else ''}"
          f"{', UVE' if cfg.run_uve else ''})...")
    p = X_proc.shape[1]
    tabela = []

    # 0) Baseline: all variables (no selection)
    full = _avaliar_subset_cv(X_proc, Y_bin, y_int, cv_indices, n_lv)
    tabela.append({"metodo": "Full (todas)", **full})
    base_bal = full["balanced_accuracy"]
    print(f"  Full: bal.acc={base_bal:.3f} ({p} vars)")

    # 1) iPLS (nested-CV: o melhor intervalo e' reescolhido a cada fold com
    #    dados de treino apenas -- achado B1-1, ver _mask_melhor_intervalo).
    #    A busca no dataset inteiro continua rodando 1x para a figura/CSV de
    #    diagnostico por intervalo, MESMO padrao ja usado por SPA/AG; o
    #    numero REPORTADO na tabela vem do nested-CV.
    ipls_res, _mask_ipls_diagnostico = selecao_ipls(
        X_proc, Y_bin, y_int, wavenumbers, cv_indices, n_lv,
        cfg.ipls_n_intervalos)
    m_ipls = _avaliar_subset_nested_cv(
        X_proc, Y_bin, y_int, cv_indices, n_lv,
        lambda Xtr, Ytr, nlv: _mask_melhor_intervalo(
            Xtr, Ytr, nlv, cfg.ipls_n_intervalos, cfg.seed))
    tabela.append({"metodo": "iPLS (melhor intervalo)", **m_ipls})
    fig_etapa4_ipls(ipls_res, wavenumbers, base_bal, cfg, pasta)
    pd.DataFrame(ipls_res).to_csv(
        os.path.join(pasta_dados, "etapa4_ipls_intervalos.csv"),
        sep=";", decimal=",", index=False)
    print(f"  iPLS: bal.acc={m_ipls['balanced_accuracy']:.3f} "
          f"({m_ipls['n_vars']:.0f} vars, media/fold; "
          f"faixa {m_ipls['n_vars_min']}-{m_ipls['n_vars_max']})")

    # 2) Selection by VIP >= threshold (nested-CV: mascara refeita por fold,
    #    so' com dados de treino -- ver docstring da funcao)
    m_vip = _avaliar_subset_nested_cv(
        X_proc, Y_bin, y_int, cv_indices, n_lv,
        lambda Xtr, Ytr, nlv: _mask_vip_threshold(Xtr, Ytr, nlv, cfg.vip_threshold_sel))
    if m_vip["n_vars_max"] >= 2:
        tabela.append({"metodo": f"VIP>={cfg.vip_threshold_sel:g}", **m_vip})
        print(f"  VIP: bal.acc={m_vip['balanced_accuracy']:.3f} "
              f"({m_vip['n_vars']:.0f} vars, media/fold; "
              f"faixa {m_vip['n_vars_min']}-{m_vip['n_vars_max']})")

    # 3) Selection by SR (top fraction, nested-CV)
    m_sr = _avaliar_subset_nested_cv(
        X_proc, Y_bin, y_int, cv_indices, n_lv,
        lambda Xtr, Ytr, nlv: _mask_sr_top_frac(Xtr, Ytr, nlv, cfg.sr_top_frac))
    tabela.append({"metodo": f"SR top {cfg.sr_top_frac:.0%}", **m_sr})
    print(f"  SR: bal.acc={m_sr['balanced_accuracy']:.3f} "
          f"({m_sr['n_vars']:.0f} vars, media/fold; "
          f"faixa {m_sr['n_vars_min']}-{m_sr['n_vars_max']})")

    # 4) sPLS-DA (nested-CV: sparse_plsda_mask ja' e' fold-agnostica, so'
    #    precisa ser chamada dentro do fold em vez de 1x no dataset inteiro)
    m_sp = _avaliar_subset_nested_cv(
        X_proc, Y_bin, y_int, cv_indices, n_lv,
        lambda Xtr, Ytr, nlv: sparse_plsda_mask(Xtr, Ytr, nlv, cfg.splsda_keep_by_comp))
    if m_sp["n_vars_max"] >= 2:
        tabela.append({"metodo": "sPLS-DA", **m_sp})
        print(f"  sPLS-DA: bal.acc={m_sp['balanced_accuracy']:.3f} "
              f"({m_sp['n_vars']:.0f} vars, media/fold; "
              f"faixa {m_sp['n_vars_min']}-{m_sp['n_vars_max']})")

    # 5) SPA/APS (opt-in — mais lento que os metodos acima: n_starts avaliacoes
    #    de CV, agora vezes len(cv_indices) por causa do nested-CV abaixo).
    #    A chamada no dataset inteiro fica so' p/ diagnostico (CSV de cadeias
    #    avaliadas); o numero REPORTADO na tabela vem do nested-CV, que nunca
    #    deixa a busca ver o fold de teste que mede o resultado final.
    if cfg.run_spa:
        spa_res, _mask_spa_diagnostico = selecao_spa(
            X_proc, Y_bin, y_int, cv_indices, n_lv,
            cfg.spa_n_vars_max, cfg.spa_n_starts, cfg.seed)
        if spa_res:
            pd.DataFrame(spa_res).to_csv(
                os.path.join(pasta_dados, "etapa4_spa_cadeias.csv"),
                sep=";", decimal=",", index=False)
        m_spa = _avaliar_busca_nested_cv(
            X_proc, Y_bin, y_int, cv_indices, n_lv,
            lambda Xtr, Ytr, ytr, cvin: selecao_spa(
                Xtr, Ytr, ytr, cvin, n_lv, cfg.spa_n_vars_max,
                cfg.spa_n_starts, cfg.seed)[1],
            cfg.seed, mae_id=mae_id)
        if m_spa["n_vars_max"] >= 2:
            tabela.append({"metodo": "SPA (APS)", **m_spa})
            print(f"  SPA: bal.acc={m_spa['balanced_accuracy']:.3f} "
                  f"({m_spa['n_vars']:.0f} vars, media/fold; "
                  f"faixa {m_spa['n_vars_min']}-{m_spa['n_vars_max']})")

    # 6) AG (opt-in — o mais lento: tam_populacao x n_geracoes avaliacoes de
    #    CV, agora vezes len(cv_indices) por causa do nested-CV abaixo).
    #    Convergencia (historico/figura) usa a busca no dataset inteiro
    #    (diagnostico de comportamento da busca, nao um numero cientifico);
    #    o bal.acc REPORTADO na tabela vem do nested-CV.
    if cfg.executar_ag:
        historico_ag, _mask_ag_diagnostico = ga_selection(
            X_proc, Y_bin, y_int, cv_indices, n_lv,
            cfg.ag_tam_populacao, cfg.ag_n_geracoes, cfg.ag_prob_mutacao,
            cfg.ag_frac_inicial, cfg.seed)
        if historico_ag:
            pd.DataFrame(historico_ag).to_csv(
                os.path.join(pasta_dados, "etapa4_ag_historico.csv"),
                sep=";", decimal=",", index=False)
            fig_etapa4_ag_convergencia(historico_ag, cfg, pasta)
        m_ag = _avaliar_busca_nested_cv(
            X_proc, Y_bin, y_int, cv_indices, n_lv,
            lambda Xtr, Ytr, ytr, cvin: ga_selection(
                Xtr, Ytr, ytr, cvin, n_lv, cfg.ag_tam_populacao,
                cfg.ag_n_geracoes, cfg.ag_prob_mutacao, cfg.ag_frac_inicial,
                cfg.seed)[1],
            cfg.seed, mae_id=mae_id)
        if m_ag["n_vars_max"] >= 2:
            tabela.append({"metodo": "AG (Genetico)", **m_ag})
            print(f"  AG: bal.acc={m_ag['balanced_accuracy']:.3f} "
                  f"({m_ag['n_vars']:.0f} vars, media/fold; "
                  f"faixa {m_ag['n_vars_min']}-{m_ag['n_vars_max']})")

    # 7) CARS (opt-in — n_iteracoes avaliacoes de CV, vezes len(cv_indices)
    #    por causa do nested-CV abaixo, mesmo esquema do AG/SPA). Historico
    #    por iteracao (dataset inteiro) fica so' pro CSV de diagnostico; o
    #    numero REPORTADO na tabela vem do nested-CV.
    if cfg.run_cars:
        historico_cars, _mask_cars_diagnostico = cars_selecao(
            X_proc, Y_bin, y_int, cv_indices, n_lv,
            cfg.cars_n_iteracoes, cfg.cars_frac_amostragem, cfg.seed)
        if historico_cars:
            pd.DataFrame(historico_cars).to_csv(
                os.path.join(pasta_dados, "etapa4_cars_iteracoes.csv"),
                sep=";", decimal=",", index=False)
        m_cars = _avaliar_busca_nested_cv(
            X_proc, Y_bin, y_int, cv_indices, n_lv,
            lambda Xtr, Ytr, ytr, cvin: cars_selecao(
                Xtr, Ytr, ytr, cvin, n_lv, cfg.cars_n_iteracoes,
                cfg.cars_frac_amostragem, cfg.seed)[1],
            cfg.seed, mae_id=mae_id)
        if m_cars["n_vars_max"] >= 2:
            tabela.append({"metodo": "CARS", **m_cars})
            print(f"  CARS: bal.acc={m_cars['balanced_accuracy']:.3f} "
                  f"({m_cars['n_vars']:.0f} vars, media/fold; "
                  f"faixa {m_cars['n_vars_min']}-{m_cars['n_vars_max']})")

    # 8) UVE (opt-in — n_repeticoes avaliacoes de PLS por fold externo, sem
    #    CV interna guiando busca -- so' compara coeficiente real x ruido,
    #    por isso usa `_avaliar_subset_nested_cv` (mesmo esquema de VIP/SR),
    #    nao `_avaliar_busca_nested_cv`.
    if cfg.run_uve:
        m_uve = _avaliar_subset_nested_cv(
            X_proc, Y_bin, y_int, cv_indices, n_lv,
            lambda Xtr, Ytr, nlv: uve_selecao(
                Xtr, Ytr, nlv, cfg.uve_n_repeticoes,
                cfg.uve_frac_amostragem, cfg.seed)[1])
        if m_uve["n_vars_max"] >= 2:
            tabela.append({"metodo": "UVE", **m_uve})
            print(f"  UVE: bal.acc={m_uve['balanced_accuracy']:.3f} "
                  f"({m_uve['n_vars']:.0f} vars, media/fold; "
                  f"faixa {m_uve['n_vars_min']}-{m_uve['n_vars_max']})")

    # Tabela + figura comparativa
    pd.DataFrame(tabela).to_csv(
        os.path.join(pasta_dados, "etapa4_selecao_variaveis.csv"),
        sep=";", decimal=",", index=False)
    fig_etapa4_comparacao(tabela, cfg, pasta)

    # Resumo: melhor metodo por parcimonia (bal.acc dentro de 1% do max, menos vars)
    bal_max = max(t["balanced_accuracy"] for t in tabela)
    candidatos = [t for t in tabela
                  if t["balanced_accuracy"] >= bal_max - 0.01]
    melhor = min(candidatos, key=lambda t: t["n_vars"])
    print(f"  -> Mais parcimonioso (bal.acc>={bal_max-0.01:.3f}): "
          f"{melhor['metodo']} ({melhor['n_vars']} vars, "
          f"bal.acc={melhor['balanced_accuracy']:.3f})")
    return {"tabela": tabela, "melhor": melhor}
