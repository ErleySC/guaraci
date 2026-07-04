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
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from validacao_estatistica import _cv_predict_manual
from figuras import salvar, cor, _ticks_x_inteiros

if TYPE_CHECKING:
    pass


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
    """sPLS-DA (estilo Le Cao et al. 2008): NIPALS com soft-selection —
    mantem apenas as `keep_por_comp` variaveis de maior |peso| por
    componente. Retorna mascara da uniao das variaveis selecionadas."""
    X = np.asarray(X_proc, dtype=float)
    Y = np.asarray(Y_bin, dtype=float)
    Xr = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    p = X.shape[1]
    selecionadas: set = set()
    n_comp = int(max(1, min(n_comp, p)))
    for _ in range(n_comp):
        M = Xr.T @ Yc                      # (p, m)
        try:
            U, _S, _Vt = np.linalg.svd(M, full_matrices=False)
            w = U[:, 0]
        except np.linalg.LinAlgError:
            break
        idx = np.argsort(np.abs(w))[::-1][:keep_por_comp]
        w_sp = np.zeros_like(w); w_sp[idx] = w[idx]
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
        selecionadas.update(int(i) for i in idx)
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


def selecao_ag(X_proc: np.ndarray, Y_bin: np.ndarray, y_int: np.ndarray,
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


def fig_etapa4_ag_convergencia(historico: List[Dict], cfg, pasta):
    """Convergencia do AG: melhor e media fitness (balanced_accuracy) por
    geracao — diagnostico padrao de algoritmos evolutivos."""
    geracoes = [h["geracao"] for h in historico]
    melhores  = [h["melhor_fitness"] for h in historico]
    medias    = [h["media_fitness"] for h in historico]

    fig, ax = plt.subplots(figsize=(9.0, 4.0), constrained_layout=True)
    ax.plot(geracoes, melhores, color=cor(2), lw=1.6, marker="o", ms=3.5,
            label="Melhor da geracao")
    ax.plot(geracoes, medias, color=cor(3), lw=1.2, ls="--",
            label="Media da populacao")
    ax.set_xlabel("Geracao")
    ax.set_ylabel("Balanced accuracy (CV)")
    ax.set_title("Etapa 4 — AG: convergencia da busca genetica", loc="left")
    ax.grid(axis="y", color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    salvar(fig, "fig_etapa4_ag_convergencia", pasta, cfg)


def fig_etapa4_ipls(resultados, wavenumbers, baseline_bal, cfg, pasta):
    """Barras de balanced_acc por intervalo iPLS, com linha do modelo global."""
    intervalos = [r["intervalo"] for r in resultados]
    bals       = [r["balanced_accuracy"] for r in resultados]
    melhor_i   = int(np.argmax(bals))

    fig, ax = plt.subplots(figsize=(11.0, 4.2), constrained_layout=True)
    cores_b = [cor(2) if k == melhor_i else "0.6" for k in range(len(bals))]
    ax.bar(intervalos, bals, color=cores_b, edgecolor="white", lw=0.5)
    ax.axhline(baseline_bal, color=cor(3), ls="--", lw=1.2,
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
    salvar(fig, "fig_etapa4_ipls_intervalos", pasta, cfg)


def fig_etapa4_comparacao(tabela, cfg, pasta):
    """Compara metodos de selecao: balanced_acc (barras) + n_vars (texto)."""
    nomes = [t["metodo"] for t in tabela]
    bals  = [t["balanced_accuracy"] for t in tabela]
    nvars = [t["n_vars"] for t in tabela]
    pos = np.arange(len(nomes))

    fig, ax = plt.subplots(figsize=(10.0, 0.6 * len(nomes) + 2.0),
                            constrained_layout=True)
    cores_b = [cor(0) if n == "Full (todas)" else cor(2) for n in nomes]
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
    salvar(fig, "fig_etapa4_comparacao_metodos", pasta, cfg)


def etapa4_selecao_variaveis(X_proc, Y_bin, y_int, vip, sr, wavenumbers,
                              cv_indices, n_lv, cfg, pasta, pasta_dados):
    """Orquestra a Etapa 4: avalia Full vs iPLS vs VIP vs SR vs sPLS-DA (sempre)
    e, se ligados em cfg, SPA/APS e AG (opt-in, mais lentos) — sob o MESMO
    esquema de CV group-aware. Salva tabela (dados/) e figuras (figuras/).
    Retorna dict-resumo para o resumo_modelo.txt."""
    print("\n[Etapa4] Selecao de variaveis "
          f"(iPLS, VIP, SR, sPLS-DA"
          f"{', SPA' if cfg.executar_spa else ''}"
          f"{', AG' if cfg.executar_ag else ''})...")
    p = X_proc.shape[1]
    tabela = []

    # 0) Baseline: all variables (no selection)
    full = _avaliar_subset_cv(X_proc, Y_bin, y_int, cv_indices, n_lv)
    tabela.append({"metodo": "Full (todas)", **full})
    base_bal = full["balanced_accuracy"]
    print(f"  Full: bal.acc={base_bal:.3f} ({p} vars)")

    # 1) iPLS
    ipls_res, mask_ipls = selecao_ipls(X_proc, Y_bin, y_int, wavenumbers,
                                        cv_indices, n_lv, cfg.ipls_n_intervalos)
    m_ipls = _avaliar_subset_cv(X_proc[:, mask_ipls], Y_bin, y_int,
                                 cv_indices, n_lv)
    tabela.append({"metodo": "iPLS (melhor intervalo)", **m_ipls})
    fig_etapa4_ipls(ipls_res, wavenumbers, base_bal, cfg, pasta)
    pd.DataFrame(ipls_res).to_csv(
        os.path.join(pasta_dados, "etapa4_ipls_intervalos.csv"),
        sep=";", decimal=",", index=False)
    print(f"  iPLS: bal.acc={m_ipls['balanced_accuracy']:.3f} "
          f"({m_ipls['n_vars']} vars)")

    # 2) Selection by VIP >= threshold
    mask_vip = np.asarray(vip) >= cfg.vip_threshold_sel
    if mask_vip.sum() >= 2:
        m_vip = _avaliar_subset_cv(X_proc[:, mask_vip], Y_bin, y_int,
                                    cv_indices, n_lv)
        tabela.append({"metodo": f"VIP>={cfg.vip_threshold_sel:g}", **m_vip})
        print(f"  VIP: bal.acc={m_vip['balanced_accuracy']:.3f} "
              f"({m_vip['n_vars']} vars)")

    # 3) Selection by SR (top fraction)
    n_top = max(2, int(round(cfg.sr_top_frac * p)))
    idx_sr = np.argsort(np.asarray(sr))[::-1][:n_top]
    mask_sr = np.zeros(p, dtype=bool); mask_sr[idx_sr] = True
    m_sr = _avaliar_subset_cv(X_proc[:, mask_sr], Y_bin, y_int,
                               cv_indices, n_lv)
    tabela.append({"metodo": f"SR top {cfg.sr_top_frac:.0%}", **m_sr})
    print(f"  SR: bal.acc={m_sr['balanced_accuracy']:.3f} "
          f"({m_sr['n_vars']} vars)")

    # 4) sPLS-DA
    mask_sp = sparse_plsda_mask(X_proc, Y_bin, n_lv, cfg.splsda_keep_por_comp)
    if mask_sp.sum() >= 2:
        m_sp = _avaliar_subset_cv(X_proc[:, mask_sp], Y_bin, y_int,
                                   cv_indices, n_lv)
        tabela.append({"metodo": "sPLS-DA", **m_sp})
        print(f"  sPLS-DA: bal.acc={m_sp['balanced_accuracy']:.3f} "
              f"({m_sp['n_vars']} vars)")

    # 5) SPA/APS (opt-in — mais lento que os metodos acima: n_starts avaliacoes de CV)
    if cfg.executar_spa:
        spa_res, mask_spa = selecao_spa(
            X_proc, Y_bin, y_int, cv_indices, n_lv,
            cfg.spa_n_vars_max, cfg.spa_n_starts, cfg.seed)
        if mask_spa.sum() >= 2:
            m_spa = _avaliar_subset_cv(X_proc[:, mask_spa], Y_bin, y_int,
                                        cv_indices, n_lv)
            tabela.append({"metodo": "SPA (APS)", **m_spa})
            if spa_res:
                pd.DataFrame(spa_res).to_csv(
                    os.path.join(pasta_dados, "etapa4_spa_cadeias.csv"),
                    sep=";", decimal=",", index=False)
            print(f"  SPA: bal.acc={m_spa['balanced_accuracy']:.3f} "
                  f"({m_spa['n_vars']} vars)")

    # 6) AG (opt-in — o mais lento: tam_populacao x n_geracoes avaliacoes de CV)
    if cfg.executar_ag:
        historico_ag, mask_ag = selecao_ag(
            X_proc, Y_bin, y_int, cv_indices, n_lv,
            cfg.ag_tam_populacao, cfg.ag_n_geracoes, cfg.ag_prob_mutacao,
            cfg.ag_frac_inicial, cfg.seed)
        if mask_ag.sum() >= 2:
            m_ag = _avaliar_subset_cv(X_proc[:, mask_ag], Y_bin, y_int,
                                       cv_indices, n_lv)
            tabela.append({"metodo": "AG (Genetico)", **m_ag})
            if historico_ag:
                pd.DataFrame(historico_ag).to_csv(
                    os.path.join(pasta_dados, "etapa4_ag_historico.csv"),
                    sep=";", decimal=",", index=False)
                fig_etapa4_ag_convergencia(historico_ag, cfg, pasta)
            print(f"  AG: bal.acc={m_ag['balanced_accuracy']:.3f} "
                  f"({m_ag['n_vars']} vars)")

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
