"""
avaliacao_modelos.py — Comparacao de classificadores e interpretabilidade:
PLS-DA sklearn-compativel, Auto-Benchmark (PLS-DA vs SVM/RF/GBM/XGBoost),
Monte Carlo CV (IC95%), curvas DET e SHAP (TreeExplainer).

Extraido de pipeline.py (Fase H). Usa modulos ja extraidos (preprocessamento,
figuras, paleta_cores, hardware); Config so em type hint (TYPE_CHECKING).
pipeline.py reexporta (executar() chama benchmark_classificadores/
monte_carlo_cv/fig_det_curvas/fig_shap_benchmark).
"""
from __future__ import annotations

import logging
import os
import time
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import balanced_accuracy_score

from guaraci.preprocessamento import construir_preprocessador
from guaraci.figuras import salvar, cor
from guaraci.hardware import _verificar_ram
from guaraci.dados_io import kennard_stone_split_group_aware
from guaraci.config import NOME_TABELAS
from guaraci.chemometric_stats import rmse_flat
from guaraci.model_registry import construir_lista_benchmark

if TYPE_CHECKING:
    from guaraci.pipeline import Config




# =========================================================================
#  Auto-Benchmark — v27: PLS-DA vs SVM vs RF vs XGBoost
# =========================================================================

class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper sklearn para PLS-DA.
    Permite uso transparente em cross_validate e Pipeline.sklearn.
    Normaliza scores por softmax para predict_proba.
    """
    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.preprocessing import LabelBinarizer as _LB
        self._lb = _LB()
        Y_bin: np.ndarray = np.asarray(self._lb.fit_transform(y))  # ensure ndarray (not spmatrix)
        # Binary: LabelBinarizer returns (n,1) — expand to (n,2)
        if Y_bin.ndim == 1 or Y_bin.shape[1] == 1:
            Y_bin = np.hstack([1 - Y_bin.reshape(-1, 1),
                                   Y_bin.reshape(-1, 1)])
        n_comp = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self._pls = PLSRegression(n_components=n_comp, scale=False)
        self._pls.fit(X, Y_bin)
        self.classes_ = self._lb.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Use argmax on raw PLS scores instead of LabelBinarizer.inverse_transform,
        # which expects binary {0,1} input. PLS output is continuous and can be
        # negative or >1, making LB.inverse_transform undefined for 13+ classes.
        return self._lb.classes_[np.argmax(  # type: ignore[return-value]
            np.asarray(self._pls.predict(X), float), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Softmax numericamente estavel — garante distribuicao valida mesmo
        # para valores negativos (frequentes em PLSRegression).
        Y = np.asarray(self._pls.predict(X), float)
        Y -= Y.max(axis=1, keepdims=True)   # shift para estabilidade
        E  = np.exp(Y)
        return E / E.sum(axis=1, keepdims=True)


def fig_benchmark_classificadores(scores_por_clf: Dict[str, np.ndarray],
                                   n_splits: int,
                                   n_classes: int,
                                   cfg: "Config", pasta: str) -> None:
    """Boxplot + swarmplot de balanced accuracy por fold para cada classificador."""
    nomes  = list(scores_por_clf.keys())
    dados  = [scores_por_clf[n] for n in nomes]
    chance = 1.0 / max(n_classes, 1)
    cores  = [cor(i) for i in range(len(nomes))]

    fig, ax = plt.subplots(figsize=(max(7.0, len(nomes) * 1.7), 4.8),
                           constrained_layout=True)

    bp = ax.boxplot(dados, patch_artist=True, notch=False, widths=0.45,
                    medianprops=dict(color="black", lw=2.0),
                    whiskerprops=dict(lw=1.2, color="0.3"),
                    capprops=dict(lw=1.2, color="0.3"),
                    flierprops=dict(marker="x", ms=5, color="0.5", lw=0.8))
    for patch, c in zip(bp["boxes"], cores):
        patch.set_facecolor(c); patch.set_alpha(0.70)

    # Pontos individuais (um por fold) com jitter reprodutivel
    rng = np.random.default_rng(42)
    for i, (nome, dado) in enumerate(zip(nomes, dados), 1):
        jitter = rng.uniform(-0.14, 0.14, len(dado))
        ax.scatter(np.full(len(dado), i) + jitter, dado,
                   color=cores[i - 1], s=38, alpha=0.85, zorder=5,
                   edgecolors="white", linewidths=0.5)

    ax.axhline(chance, color="0.35", ls="--", lw=1.2,
               label=f"Chance = 1/{n_classes} = {chance:.2f}")
    ax.axhline(1.0, color="0.88", ls=":", lw=0.7)

    ax.set_xticks(range(1, len(nomes) + 1))
    ax.set_xticklabels(nomes, fontsize=9)
    ax.set_ylabel("Balanced Accuracy (CV fold)")
    ax.set_title(
        f"Auto-Benchmark — {n_splits}-fold GroupKFold (anti-leakage de replicas)\n"
        f"Preprocessamento: {cfg.preprocessamento_padrao}",
        fontsize=8.5, loc="left")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="0.94", lw=0.5); ax.set_axisbelow(True)

    salvar(fig, "fig_benchmark_classificadores", pasta, cfg)
    plt.close(fig)


def benchmark_classificadores(X_raw: np.ndarray, y_int: np.ndarray,
                               grupos_cv: Optional[np.ndarray],
                               lb: "LabelBinarizer",
                               n_opt: int, cfg: "Config", pasta: str,
                               wavenumbers: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Compara PLS-DA (n_opt LVs) vs SVM RBF vs Random Forest vs XGBoost
    usando a mesma CV group-aware e o mesmo pre-processamento do pipeline.

    Anti-leakage garantido: cada classificador e encapsulado num sklearn
    Pipeline(preproc + clf), de forma que o MSC/SNV/SG e recalibrado
    dentro de cada fold sem vazamento de informacao do conjunto de validacao.

    Ref: Westerhuis et al. (2008) Chemom. Intell. Lab. Syst. 92:58-64.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline as _SKPipeline
    from scipy.stats import wilcoxon

    n_splits = min(cfg.n_splits_cv,
                   int(pd.Series(y_int).value_counts().min()))
    n_splits  = max(n_splits, 2)
    n_classes = len(lb.classes_)
    preproc   = construir_preprocessador(cfg)

    # ── Classifiers (fonte unica: guaraci.model_registry, item 20) ────────
    clfs: List[Tuple[str, Any]] = construir_lista_benchmark(
        n_opt, cfg, incluir_opcionais=True)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                               random_state=cfg.seed)
    # Pre-computar os splits (reutilizado em todos os classificadores)
    splits_list = list(cv.split(X_raw, y_int, grupos_cv))

    resultados: List[Dict] = []
    scores_por_clf: Dict[str, np.ndarray] = {}
    # OOF probabilities coletadas no mesmo loop — evita cross_val_predict extra
    oof_probas: Dict[str, np.ndarray] = {}

    for nome, clf in clfs:
        pipe_base = _SKPipeline([("preproc", clone(preproc)), ("clf", clf)])
        print(f"  {nome:<18s} ... ", end="", flush=True)
        t0 = time.time()
        try:
            ba_folds: List[float] = []
            f1_folds: List[float] = []
            proba_oof = np.zeros((len(y_int), n_classes))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for tr_idx, te_idx in splits_list:
                    pipe_i = clone(pipe_base)
                    pipe_i.fit(X_raw[tr_idx], y_int[tr_idx])
                    y_pred = pipe_i.predict(X_raw[te_idx])
                    ba_folds.append(
                        float(balanced_accuracy_score(y_int[te_idx], y_pred)))
                    f1_folds.append(
                        float(f1_score(y_int[te_idx], y_pred,
                                       average="macro", zero_division=0)))
                    # Coleta OOF proba em um unico passe (sem cross_val_predict extra)
                    if hasattr(pipe_i[-1], "predict_proba"):
                        try:
                            p = pipe_i.predict_proba(X_raw[te_idx])
                            nc = min(p.shape[1], n_classes)
                            proba_oof[te_idx, :nc] = p[:, :nc]
                        except Exception:
                            logging.getLogger(__name__).debug("suppressed non-critical exception", exc_info=True)
            elapsed = time.time() - t0
            ba = np.array(ba_folds)
            f1 = np.array(f1_folds)
            scores_por_clf[nome] = ba
            oof_probas[nome]     = proba_oof
            resultados.append({
                "Classificador":      nome,
                "Bal.Acc media":      round(float(ba.mean()), 4),
                "Bal.Acc std":        round(float(ba.std()),  4),
                "F1 macro media":     round(float(f1.mean()), 4),
                "F1 macro std":       round(float(f1.std()),  4),
                "Tempo total (s)":    round(elapsed, 2),
            })
            print(f"bal.acc={ba.mean():.4f} ± {ba.std():.4f}  [{elapsed:.1f}s]")
        except Exception as _e:
            print(f"FALHA ({_e})")

    # ── Wilcoxon vs PLS-DA ────────────────────────────────────────────────
    ref = scores_por_clf.get("PLS-DA")
    for r in resultados:
        nome_r = r["Classificador"]
        if nome_r == "PLS-DA" or ref is None:
            r["p Wilcoxon (vs PLS-DA)"] = "-"
        else:
            try:
                alt = scores_por_clf.get(nome_r)
                if alt is not None and len(alt) == len(ref):
                    _, pval = wilcoxon(ref, alt, alternative="two-sided",
                                       zero_method="wilcox")
                    r["p Wilcoxon (vs PLS-DA)"] = round(float(pval), 4)  # type: ignore[arg-type]
                else:
                    r["p Wilcoxon (vs PLS-DA)"] = "n/a"
            except Exception:
                r["p Wilcoxon (vs PLS-DA)"] = "n/a"

    df_bench = pd.DataFrame(resultados)

    # ── Salvar CSV ────────────────────────────────────────────────────────
    cam_csv = os.path.join(pasta, NOME_TABELAS, "benchmark_classificadores.csv")
    df_bench.to_csv(cam_csv, index=False, sep=";", decimal=",")
    print(f"  -> {cam_csv}")

    # ── Figura boxplot ────────────────────────────────────────────────────
    if scores_por_clf:
        fig_benchmark_classificadores(scores_por_clf, n_splits, n_classes, cfg, pasta)

    # ── Curvas DET (OOF coletados no loop principal — sem re-execucao) ───
    if len(oof_probas) >= 2:
        try:
            fig_det_curvas(oof_probas, y_int, n_classes, cfg, pasta)
        except Exception as _e_det:
            print(f"\n  [AVISO] DET curves falhou: {_e_det}")

    # ── SHAP values (opcional) ────────────────────────────────────────────
    if cfg.executar_shap:
        # Guard: RF multiclass (14 classes × 500 samples × n_feat) ~600 MB
        if _verificar_ram(3.0, "SHAP TreeExplainer (RF multiclasse 14 classes)"):
            fig_shap_benchmark(X_raw, y_int, n_opt, cfg, pasta, wavenumbers)

    return df_bench


# =========================================================================
#  v28: Monte Carlo CV — IC95% por percentil
# =========================================================================

def fig_monte_carlo_distribuicao(scores_mc: Dict[str, List[float]],
                                  cfg: "Config", pasta: str) -> None:
    """Violin + IC95% percentil das distribuicoes Monte Carlo CV."""
    nomes = list(scores_mc.keys())
    dados = [np.array(scores_mc[n]) for n in nomes]
    cores = [cor(i) for i in range(len(nomes))]

    fig, ax = plt.subplots(figsize=(max(5.5, len(nomes) * 1.9), 5.0),
                           constrained_layout=True)
    parts = ax.violinplot(dados, positions=range(1, len(nomes) + 1),
                          showmedians=True, showextrema=False)
    for pc, c in zip(parts["bodies"], cores):  # type: ignore[arg-type]
        pc.set_facecolor(c); pc.set_alpha(0.50)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2.0)

    for i, d in enumerate(dados, 1):
        ci_lo = float(np.percentile(d, 2.5))
        ci_hi = float(np.percentile(d, 97.5))
        ax.plot([i, i], [ci_lo, ci_hi], color="0.25", lw=2.0, zorder=5)
        ax.plot(i, float(np.mean(d)), "o", ms=6, color="white",
                markeredgecolor="0.2", markeredgewidth=1.2, zorder=6)
        ax.annotate(
            f"IC95%\n[{ci_lo:.3f},{ci_hi:.3f}]",
            xy=(i, ci_hi), xytext=(i + 0.18, ci_hi),
            fontsize=6.5, va="center", color="0.3")

    n_iter = max(len(d) for d in dados)
    ax.set_xticks(range(1, len(nomes) + 1))
    ax.set_xticklabels(nomes, fontsize=9)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(
        f"Monte Carlo CV ({n_iter} iteracoes, test={cfg.monte_carlo_test_size:.0%})\n"
        f"IC95% percentil — pre-processamento: {cfg.preprocessamento_padrao}",
        fontsize=8.5, loc="left")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", color="0.93", lw=0.5)
    ax.set_axisbelow(True)

    salvar(fig, "fig_monte_carlo_cv", pasta, cfg)
    plt.close(fig)


def _stratified_group_shuffle_splits(
        y_int: np.ndarray,
        grupos_cv: np.ndarray,
        n_splits: int,
        test_size: float,
        random_state: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split estratificado no nivel de grupo: cada grupo recebe a classe da maioria
    de suas amostras e o split e feito via StratifiedShuffleSplit sobre grupos.
    Garante (1) anti-leakage de replicas e (2) proporcao de classes no teste.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    grupos_u = np.unique(grupos_cv)
    # Classe dominante por grupo (votacao por maioria)
    cls_por_grupo = np.array([
        int(pd.Series(y_int[grupos_cv == g]).mode().iloc[0])
        for g in grupos_u
    ])

    sss = StratifiedShuffleSplit(n_splits=n_splits,
                                  test_size=test_size,
                                  random_state=random_state)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for grp_tr, grp_te in sss.split(grupos_u, cls_por_grupo):
        grupos_train = set(grupos_u[grp_tr])
        grupos_test  = set(grupos_u[grp_te])
        tr_idx = np.where(np.isin(grupos_cv, list(grupos_train)))[0]
        te_idx = np.where(np.isin(grupos_cv, list(grupos_test)))[0]
        splits.append((tr_idx, te_idx))
    return splits


def monte_carlo_cv(X_raw: np.ndarray, y_int: np.ndarray,
                   grupos_cv: Optional[np.ndarray],
                   lb: "LabelBinarizer",
                   n_opt: int, cfg: "Config", pasta: str) -> pd.DataFrame:
    """
    Monte Carlo CV com split estratificado por grupo (N repeticoes).
    Gera distribuicao empirica de Balanced Accuracy com IC95% por percentil.

    Se cfg.monte_carlo_incluir_todos=True, roda tambem SVM RBF, RF e XGBoost
    (mesmos hiperparametros do benchmark, via guaraci.model_registry — item
    20 da auditoria: fonte unica, antes duplicada e divergente aqui: este
    Grad. Boost. nao tinha subsample=0.8 como o do benchmark); caso
    contrario, apenas PLS-DA.

    Ref: Filzmoser et al. (2009) Anal. Chim. Acta 652:133-142.
    """
    from sklearn.base import clone
    from sklearn.metrics import f1_score
    from sklearn.pipeline import Pipeline as _SKPipeline

    n_iter  = cfg.n_monte_carlo
    test_sz = cfg.monte_carlo_test_size
    preproc = construir_preprocessador(cfg)

    # Montar lista de modelos (fonte unica: guaraci.model_registry, item 20)
    mc_clfs: List[Tuple[str, Any]] = construir_lista_benchmark(
        n_opt, cfg, incluir_opcionais=cfg.monte_carlo_incluir_todos)

    # Gerar splits estratificados por grupo (risco 3 resolvido)
    if grupos_cv is not None:
        splits = _stratified_group_shuffle_splits(
            y_int, grupos_cv, n_iter, test_sz, cfg.seed)
    else:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=n_iter,
                                     test_size=test_sz,
                                     random_state=cfg.seed)
        splits = list(sss.split(X_raw, y_int))

    # Correr modelos
    resultados_mc: List[Dict] = []
    scores_mc: Dict[str, List[float]] = {}

    t0_total = time.time()
    for nome, clf in mc_clfs:
        pipe = _SKPipeline([("preproc", clone(preproc)), ("clf", clf)])
        ba_list: List[float] = []
        f1_list: List[float] = []
        print(f"  MC CV [{nome}]: {n_iter} iteracoes ... ", end="", flush=True)
        t0 = time.time()
        for tr_idx, te_idx in splits:
            X_tr, X_te = X_raw[tr_idx], X_raw[te_idx]
            y_tr, y_te = y_int[tr_idx], y_int[te_idx]
            # Ensure all classes are in training and at least 1 sample in test
            if len(np.unique(y_tr)) < len(lb.classes_):
                continue
            if len(np.unique(y_te)) < 2:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe_i = clone(pipe)
                    pipe_i.fit(X_tr, y_tr)
                    y_pred = pipe_i.predict(X_te)
                    ba_list.append(float(balanced_accuracy_score(y_te, y_pred)))
                    f1_list.append(
                        float(f1_score(y_te, y_pred, average="macro", zero_division=0)))
            except Exception:
                logging.getLogger(__name__).debug("suppressed non-critical exception", exc_info=True)

        elapsed = time.time() - t0
        arr_ba = np.array(ba_list) if ba_list else np.array([np.nan])
        arr_f1 = np.array(f1_list) if f1_list else np.array([np.nan])
        ci_lo  = float(np.nanpercentile(arr_ba, 2.5))
        ci_hi  = float(np.nanpercentile(arr_ba, 97.5))
        scores_mc[nome] = ba_list

        print(f"media={np.nanmean(arr_ba):.4f}  IC95%=[{ci_lo:.4f},{ci_hi:.4f}]"
              f"  [{elapsed:.1f}s]")
        resultados_mc.append({
            "Classificador":    nome,
            "Iteracoes validas": len(ba_list),
            "Media BA":         round(float(np.nanmean(arr_ba)), 4),
            "Mediana BA":       round(float(np.nanmedian(arr_ba)), 4),
            "Std BA":           round(float(np.nanstd(arr_ba)), 4),
            "IC95% inf":        round(ci_lo, 4),
            "IC95% sup":        round(ci_hi, 4),
            "Media F1 macro":   round(float(np.nanmean(arr_f1)), 4),
            "IC95% F1 inf":     round(float(np.nanpercentile(arr_f1, 2.5)), 4),
            "IC95% F1 sup":     round(float(np.nanpercentile(arr_f1, 97.5)), 4),
            "Fracao teste":     test_sz,
        })

    print(f"  [MC CV total: {time.time()-t0_total:.1f}s]")
    df_mc = pd.DataFrame(resultados_mc)

    cam_csv = os.path.join(pasta, NOME_TABELAS, "monte_carlo_cv.csv")
    df_mc.to_csv(cam_csv, index=False, sep=";", decimal=",")
    print(f"  -> {cam_csv}")

    # Figura violin — apenas modelos com >= 5 iteracoes validas
    scores_plot = {n: v for n, v in scores_mc.items() if len(v) >= 5}
    if scores_plot:
        fig_monte_carlo_distribuicao(scores_plot, cfg, pasta)

    return df_mc


# =========================================================================
#  v28: Curvas DET — Detection Error Tradeoff
# =========================================================================

def fig_det_curvas(oof_probas: Dict[str, np.ndarray],
                   y_int: np.ndarray,
                   n_classes: int,
                   cfg: "Config", pasta: str) -> None:
    """
    Curvas DET (Detection Error Tradeoff) macro-OvR por classificador.
    Cada curva e a media das curvas binarias One-vs-Rest por classe.

    Ref: Martin et al. (1997) NIST/SEMATECH Engineering Statistics Handbook.
    """
    try:
        from sklearn.metrics import det_curve
        from sklearn.preprocessing import label_binarize
    except ImportError:
        print("  [AVISO] det_curve indisponivel (sklearn < 0.24) — DET pulada.")
        return

    nomes = list(oof_probas.keys())
    cores = [cor(i) for i in range(len(nomes))]

    y_bin: np.ndarray = np.asarray(label_binarize(y_int, classes=range(n_classes)))
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    # Gerar duas versoes: escala linear (0-100%) + escala log (padrao NIST)
    for log_scale in (False, True):
        eps  = 0.1   # evitar log(0)
        lo   = eps if log_scale else 0.0
        hi   = 100.0
        grid = np.logspace(np.log10(eps), np.log10(hi), 500) if log_scale \
               else np.linspace(0.0, hi, 500)
        fmr_grid_pct = grid
        fmr_grid_frac = fmr_grid_pct / 100.0

        fig, ax = plt.subplots(figsize=(6.0, 5.5), constrained_layout=True)

        for nome, c in zip(nomes, cores):
            proba = oof_probas[nome]
            fnmr_acum = np.zeros(len(fmr_grid_frac))
            n_valid = 0
            for k in range(n_classes):
                y_k = y_bin[:, k]
                if y_k.sum() < 2 or k >= proba.shape[1]:
                    continue
                try:
                    fmr, fnmr, _ = det_curve(y_k, proba[:, k])
                    fnmr_acum += np.interp(fmr_grid_frac, fmr, fnmr,
                                           left=fnmr[0], right=fnmr[-1])
                    n_valid += 1
                except Exception:
                    logging.getLogger(__name__).debug("suppressed non-critical exception", exc_info=True)
            if n_valid == 0:
                continue
            fnmr_media = fnmr_acum / n_valid
            ax.plot(fmr_grid_pct, fnmr_media * 100,
                    lw=1.8, color=c, label=nome, alpha=0.85)

        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.35,
                label="Ref. diagonal")
        ax.set_xlabel("False Match Rate — FMR (%)")
        ax.set_ylabel("False Non-Match Rate — FNMR (%)")
        escala_str = "log" if log_scale else "linear"
        ax.set_title(
            f"Curvas DET (Detection Error Tradeoff) — macro OvR [{escala_str}]\n"
            f"OOF predictions, {cfg.n_splits_cv}-fold GroupKFold",
            fontsize=8.5, loc="left")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        if log_scale:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(color="0.93", lw=0.5, which="both"); ax.set_axisbelow(True)

        sufixo = "_log" if log_scale else ""
        salvar(fig, f"fig_det_curvas{sufixo}", pasta, cfg)
        plt.close(fig)


# =========================================================================
#  v28: SHAP values — interpretabilidade espectral dos ensembles
# =========================================================================

def fig_shap_benchmark(X_raw: np.ndarray, y_int: np.ndarray,
                        n_opt: int, cfg: "Config", pasta: str,
                        wavenumbers: Optional[np.ndarray] = None) -> None:
    """
    SHAP TreeExplainer para RF, GBM e XGBoost (se disponivel).
    Plota barplot horizontal dos top-20 wavenumbers por mean |SHAP|.

    Ref: Lundberg & Lee (2017) NeurIPS — SHAP (SHapley Additive exPlanations).
    """
    try:
        import shap  # type: ignore
    except ImportError:
        print("  [AVISO] shap nao instalado — pip install shap. SHAP pulado.")
        return

    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    preproc   = construir_preprocessador(cfg)
    X_proc    = clone(preproc).fit(X_raw).transform(X_raw)
    feat_names = ([f"{w:.0f}" for w in wavenumbers]
                  if wavenumbers is not None
                  else [f"X{i}" for i in range(X_proc.shape[1])])

    # Memory cap: random subsample of shap_max_amostras samples for TreeExplainer
    n_max = cfg.shap_max_amostras
    if X_proc.shape[0] > n_max:
        rng_shap = np.random.default_rng(cfg.seed)
        idx_shap = rng_shap.choice(X_proc.shape[0], n_max, replace=False)
        X_shap = X_proc[idx_shap]
        y_shap = y_int[idx_shap]
    else:
        X_shap = X_proc
        y_shap = y_int

    n_classes_shap = len(np.unique(y_int))
    tree_clfs: List[Tuple[str, Any]] = [
        ("Random Forest",
         RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                class_weight="balanced_subsample",
                                n_jobs=1, random_state=cfg.seed)),
    ]
    # GradientBoostingClassifier: SHAP TreeExplainer does NOT support multiclass
    # (raises InvalidModelError when n_classes > 2). Skip for multiclass datasets.
    if n_classes_shap == 2:
        tree_clfs.append(
            ("Grad. Boost.",
             GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                        max_depth=3, random_state=cfg.seed))
        )
    try:
        from xgboost import XGBClassifier  # type: ignore
        tree_clfs.append(("XGBoost",
                           XGBClassifier(n_estimators=300, learning_rate=0.05,
                                         max_depth=4, subsample=0.8,
                                         colsample_bytree=0.8,
                                         eval_metric="mlogloss", verbosity=0,
                                         n_jobs=1, random_state=cfg.seed)))
    except ImportError:
        pass

    for idx, (nome, clf) in enumerate(tree_clfs):
        print(f"  SHAP {nome} ({len(X_shap)} amostras) ... ", end="", flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Treinar no conjunto completo; calcular SHAP no subconjunto
                clf.fit(X_proc, y_int)
                explainer = shap.TreeExplainer(clf)
                sv = explainer.shap_values(X_shap)

                # sv: lista (RF antigo), ndarray 3D (XGBoost>=2 multiclass),
                # ndarray 2D (GBM binario / single output).
                # XGBoost>=2.0 e shap>=0.41 retornam ndarray diretamente.
                if isinstance(sv, list):
                    # RF classico (shap<0.41) — lista de arrays por classe
                    importance = np.mean([np.abs(s) for s in sv], axis=0).mean(axis=0)
                else:
                    sv_arr = np.asarray(sv)
                    if sv_arr.ndim == 3:
                        # SHAP 0.4x+: shape (n_samples, n_features, n_classes)
                        # mean over samples (axis=0) and classes (axis=2) → (n_features,)
                        importance = np.abs(sv_arr).mean(axis=(0, 2))
                    elif sv_arr.ndim == 2:
                        # (n_samples, n_features) — GBM / XGBoost binario
                        importance = np.abs(sv_arr).mean(axis=0)
                    else:
                        # Fallback: aplanar dimensoes extras
                        importance = np.abs(sv_arr).reshape(sv_arr.shape[-2], sv_arr.shape[-1]).mean(axis=0)

                top_n   = min(20, len(importance))
                top_idx = np.argsort(importance)[-top_n:][::-1]
                top_imp = importance[top_idx]
                top_lbl = [feat_names[i] for i in top_idx]

                # ── Barplot de importancia SHAP ───────────────────────────
                fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
                ax.barh(range(top_n), top_imp[::-1],
                        color=cor(idx), alpha=0.76, edgecolor="white", lw=0.5)
                ax.set_yticks(range(top_n))
                ax.set_yticklabels(top_lbl[::-1], fontsize=8)
                ax.set_xlabel("Mean |SHAP value|")
                ax.set_title(
                    f"SHAP — {nome} (top-{top_n} bandas espectrais)\n"
                    f"Unidade: cm⁻¹  |  pre-proc: {cfg.preprocessamento_padrao}"
                    f"  |  n={len(X_shap)}",
                    fontsize=8.5, loc="left")
                ax.grid(axis="x", color="0.93", lw=0.5); ax.set_axisbelow(True)
                tag = nome.lower().replace(" ", "_").replace(".", "").replace("/", "_")
                salvar(fig, f"fig_shap_{tag}", pasta, cfg)
                plt.close(fig)

                # ── Dependence plots: top-3 features × classe ─────────────
                n_dep = min(3, top_n)
                unique_cls = np.unique(y_shap)
                cores_dep  = [cor(c) for c in unique_cls]
                for rank, feat_i in enumerate(top_idx[:n_dep]):
                    feat_vals = X_shap[:, feat_i]
                    # Para multiclass, usar shap medio sobre classes
                    if isinstance(sv, list):
                        shap_feat = np.mean([sv_k[:, feat_i] for sv_k in sv], axis=0)
                    else:
                        sv_arr2 = np.array(sv)
                        if sv_arr2.ndim == 3:
                            # (n_samples, n_features, n_classes): media sobre classes
                            # (axis=2) -> (n_samples, n_features); depois seleciona a feature
                            shap_feat = sv_arr2.mean(axis=2)[:, feat_i]
                        else:
                            shap_feat = sv_arr2[:, feat_i]

                    fig2, ax2 = plt.subplots(figsize=(5.5, 4.0),
                                             constrained_layout=True)
                    for ci, (cls_v, cc) in enumerate(zip(unique_cls, cores_dep)):
                        mask_c = y_shap == cls_v
                        ax2.scatter(feat_vals[mask_c], shap_feat[mask_c],
                                    color=cc, alpha=0.60, s=18, lw=0,
                                    label=f"Cls {cls_v}")
                    ax2.axhline(0, color="0.5", lw=0.8, ls="--")
                    ax2.set_xlabel(f"{top_lbl[rank]} cm⁻¹")
                    ax2.set_ylabel("SHAP value (medio)")
                    ax2.set_title(
                        f"SHAP Dependence — {nome}\n"
                        f"Feature #{rank+1}: {top_lbl[rank]}",
                        fontsize=8.5, loc="left")
                    if len(unique_cls) <= 10:
                        ax2.legend(fontsize=6.5, ncol=2, markerscale=1.2)
                    ax2.grid(color="0.93", lw=0.5); ax2.set_axisbelow(True)
                    salvar(fig2, f"fig_shap_dep_{tag}_feat{rank+1}", pasta, cfg)
                    plt.close(fig2)

                print("salvo.")
        except Exception as _e:
            print(f"FALHA ({_e})")


# =========================================================================
#  Auto-Benchmark de REGRESSAO (N2/N3) -- PLS-R vs Ridge/Lasso/EN/SVR/RF
# =========================================================================

def fig_benchmark_regressores(rmsep_por_modelo: Dict[str, np.ndarray],
                               n_especies: int,
                               cfg: "Config", pasta: str) -> None:
    """Boxplot de RMSEP por especie para cada modelo de regressao -- ao
    contrario do benchmark de classificacao (bal.acc, maior=melhor), aqui
    MENOR e melhor."""
    nomes = list(rmsep_por_modelo.keys())
    dados = [rmsep_por_modelo[n] for n in nomes]
    cores = [cor(i) for i in range(len(nomes))]

    fig, ax = plt.subplots(figsize=(max(7.0, len(nomes) * 1.7), 4.8),
                           constrained_layout=True)

    bp = ax.boxplot(dados, patch_artist=True, notch=False, widths=0.45,
                    medianprops=dict(color="black", lw=2.0),
                    whiskerprops=dict(lw=1.2, color="0.3"),
                    capprops=dict(lw=1.2, color="0.3"),
                    flierprops=dict(marker="x", ms=5, color="0.5", lw=0.8))
    for patch, c in zip(bp["boxes"], cores):
        patch.set_facecolor(c); patch.set_alpha(0.70)

    rng = np.random.default_rng(42)
    for i, (nome, dado) in enumerate(zip(nomes, dados), 1):
        if len(dado) == 0:
            continue
        jitter = rng.uniform(-0.14, 0.14, len(dado))
        ax.scatter(np.full(len(dado), i) + jitter, dado,
                   color=cores[i - 1], s=38, alpha=0.85, zorder=5,
                   edgecolors="white", linewidths=0.5)

    ax.set_xticks(range(1, len(nomes) + 1))
    ax.set_xticklabels(nomes, fontsize=9)
    ax.set_ylabel("RMSEP por especie (menor = melhor)")
    ax.set_title(
        f"Auto-Benchmark de regressao -- {n_especies} especies\n"
        f"Pre-processamento: {cfg.preprocessamento_padrao}",
        fontsize=8.5, loc="left")
    ax.grid(axis="y", color="0.94", lw=0.5); ax.set_axisbelow(True)

    salvar(fig, "fig_benchmark_regressores", pasta, cfg)
    plt.close(fig)


def benchmark_regressao_por_especie(
        X_raw: np.ndarray, conc: np.ndarray, rotulos: np.ndarray,
        mae_id: Optional[np.ndarray], classes_unicas: np.ndarray,
        cfg: "Config", pasta: str,
        reg_esp_pls: Dict[str, Any],
        min_amostras_adult: int = 6) -> Optional[pd.DataFrame]:
    """
    Compara PLS-R (baseline, ja calibrado por `pls_regressao_por_especie` --
    reaproveitado SEM refit) vs Ridge / Lasso / Elastic Net / SVR (RBF) /
    Random Forest Regressor, calibrando UM MODELO POR ESPECIE (mesma
    arquitetura da quantificacao do pipeline: calibracao separada evita que
    a variacao inter-especies confunda o sinal de adulteracao).

    Cada modelo usa O MESMO split cal/val por especie (reproduzido
    deterministicamente com o mesmo cfg.seed/cfg.divisao_cal_val do PLS-R
    ja calculado) e o mesmo pre-processamento dentro de um sklearn Pipeline
    (sem vazamento entre cal/val) -- comparacao honesta apples-to-apples.

    Hiperparametros por heuristica de literatura (sem tuning por CV interna,
    mesmo padrao de benchmark_classificadores/PLS-DA -- ver nota
    metodologica em pipeline.salvar_resumo_modelo). Ref: Hastie, Tibshirani
    & Friedman (2009), The Elements of Statistical Learning, 2nd ed.

    Retorna None se nenhuma especie tiver dados suficientes (mesmo criterio
    de `pls_regressao_por_especie`: min_amostras_adult adulteradas e
    variancia de teor > 0).
    """
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline as _SKPipeline
    from sklearn.metrics import r2_score

    modelos: List[Tuple[str, Any]] = [
        ("Ridge",
         Ridge(alpha=1.0, random_state=cfg.seed)),
        ("Lasso",
         Lasso(alpha=0.1, random_state=cfg.seed, max_iter=5000)),
        ("Elastic Net",
         ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=cfg.seed,
                    max_iter=5000)),
        ("SVR (RBF)",
         SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")),
        ("Random Forest",
         RandomForestRegressor(n_estimators=300, max_features="sqrt",
                               n_jobs=1, random_state=cfg.seed)),
    ]

    conc = np.asarray(conc, dtype=float)
    yv_pool: Dict[str, List[np.ndarray]] = {nome: [] for nome, _ in modelos}
    yvh_pool: Dict[str, List[np.ndarray]] = {nome: [] for nome, _ in modelos}
    rmsep_por_especie: Dict[str, List[float]] = {nome: [] for nome, _ in modelos}
    n_especies_ok = 0

    for cls in classes_unicas:
        idx = np.where(rotulos == cls)[0]
        if idx.size == 0:
            continue
        conc_c = conc[idx].copy()
        conc_c = np.where(np.isnan(conc_c), 0.0, conc_c)
        n_adult_c = int(np.sum(conc_c > 0))
        if n_adult_c < min_amostras_adult or float(conc_c.std()) < 1e-8:
            continue

        X_c = X_raw[idx]
        mae_c = mae_id[idx] if mae_id is not None else None
        Y_c = conc_c.reshape(-1, 1)

        # MESMO split (deterministico) usado em pls_regressao_por_especie --
        # mesma logica de decisao, reproduzida aqui p/ evitar acoplamento
        # circular com pipeline.py (que importaria de volta este modulo).
        try:
            if cfg.divisao_cal_val == "kennard_stone":
                ic, iv = kennard_stone_split_group_aware(
                    X_c, mae_c, cfg.frac_cal)
            elif mae_c is not None and len(np.unique(mae_c)) >= 4:
                gss = GroupShuffleSplit(n_splits=1, train_size=cfg.frac_cal,
                                        random_state=cfg.seed)
                ic, iv = next(gss.split(X_c, Y_c, groups=mae_c))
            else:
                rng = np.random.default_rng(cfg.seed)
                perm = rng.permutation(len(conc_c))
                ncal = max(2, int(cfg.frac_cal * len(conc_c)))
                ic, iv = perm[:ncal], perm[ncal:]
        except Exception:
            continue
        if len(ic) < 4 or len(iv) < 2:
            continue

        Xc, Yc = X_c[ic], Y_c[ic].ravel()
        Xv, Yv = X_c[iv], Y_c[iv].ravel()
        n_especies_ok += 1

        for nome, modelo in modelos:
            try:
                pipe = _SKPipeline([
                    ("preproc", clone(construir_preprocessador(cfg))),
                    ("reg", clone(modelo)),
                ])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(Xc, Yc)
                    Yv_hat = np.asarray(pipe.predict(Xv),
                                        dtype=float).flatten()
                rmsep_por_especie[nome].append(rmse_flat(Yv, Yv_hat))
                yv_pool[nome].append(Yv)
                yvh_pool[nome].append(Yv_hat)
            except Exception as _e:
                print(f"  [AVISO] {nome} falhou em especie '{cls}': {_e}")

    if n_especies_ok == 0:
        return None

    linhas: List[Dict[str, Any]] = []

    # PLS-R baseline: reaproveita reg_esp_pls (ja calculado, SEM refit)
    rmsep_pls_por_especie = [
        t["rmsep"] for t in reg_esp_pls.get("tabela_especie", [])
        if np.isfinite(t.get("rmsep", np.nan))
    ]
    linhas.append({
        "Modelo":         "PLS-R",
        "RMSEP (pooled)": round(float(reg_esp_pls["rmsep"]), 3),
        "R2val (pooled)": round(float(reg_esp_pls["r2v"]), 4),
        "N especies":     int(reg_esp_pls["n_especies"]),
        "RMSEP std (entre especies)": (
            round(float(np.std(rmsep_pls_por_especie)), 3)
            if rmsep_pls_por_especie else float("nan")),
    })
    rmsep_boxplot: Dict[str, np.ndarray] = {}
    if rmsep_pls_por_especie:
        rmsep_boxplot["PLS-R"] = np.array(rmsep_pls_por_especie)

    for nome, _ in modelos:
        if not yv_pool[nome]:
            continue
        Yv_p  = np.concatenate(yv_pool[nome])
        Yvh_p = np.concatenate(yvh_pool[nome])
        rmsep_pooled = rmse_flat(Yv_p, Yvh_p)
        r2_pooled = (float(r2_score(Yv_p, Yvh_p))
                    if len(np.unique(Yv_p)) > 1 else float("nan"))
        linhas.append({
            "Modelo":         nome,
            "RMSEP (pooled)": round(rmsep_pooled, 3),
            "R2val (pooled)": round(r2_pooled, 4),
            "N especies":     len(rmsep_por_especie[nome]),
            "RMSEP std (entre especies)": round(
                float(np.std(rmsep_por_especie[nome])), 3),
        })
        rmsep_boxplot[nome] = np.array(rmsep_por_especie[nome])

    df_bench = pd.DataFrame(linhas)

    cam_csv = os.path.join(pasta, NOME_TABELAS, "benchmark_regressao.csv")
    df_bench.to_csv(cam_csv, index=False, sep=";", decimal=",")
    print(f"  -> {cam_csv}")

    if rmsep_boxplot:
        fig_benchmark_regressores(rmsep_boxplot, n_especies_ok, cfg, pasta)

    return df_bench
