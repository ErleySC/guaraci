"""
figuras.py — Toda a camada de plotagem do pipeline: setup do matplotlib,
salvar(), helpers de plot e as ~30 funcoes fig_* (scores, confusao, VIP/SR,
HCA, ROC, S-Plot, DD-SIMCA, etc.).

Extraido de pipeline.py como parte da modularizacao (Fase H). Depende so de
matplotlib/scipy/sklearn + paleta_cores + chemometric_stats; nao importa
pipeline (Config so em type hint, via TYPE_CHECKING). pipeline.py reexporta os
nomes publicos, entao `pipeline.fig1_pca_scores(...)`, `pipeline.salvar(...)`
etc. continuam funcionando sem alteracao. Coberto por
tests/test_figuras_regressao.py.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import numpy as np
from scipy.signal import savgol_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from scipy.stats import chi2
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

from guaraci.paleta_cores import (
    cor, edge_para_cor,
)
from guaraci.chemometric_stats import (
    hotelling_t2, hotelling_t2_limite, q_residuos, q_residuos_limite,
)
from guaraci.config import NOME_GRAFICOS

if TYPE_CHECKING:
    from guaraci.pipeline import Config


def setup_matplotlib(cfg: Config) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "axes.labelsize": 9,
        "axes.labelpad": 4,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "figure.dpi": 110,
        "savefig.dpi": cfg.dpi_salvar,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        "mathtext.default": "regular",
    })


_AVISO_MOSTRAR_GRAFICOS_EMITIDO = False


def salvar(fig, nome: str, pasta: str, cfg: Config,
           subpasta: str = "") -> None:
    """Always saves figure under pasta/Graficos/[subpasta]/ (auditoria jul/2026
    item 4). subpasta groups detailed figures (e.g. 'ddsimca')."""
    base = os.path.join(pasta, NOME_GRAFICOS)
    destino = os.path.join(base, subpasta) if subpasta else base
    os.makedirs(destino, exist_ok=True)
    caminho = os.path.join(destino, f"{nome}.{cfg.formato_saida}")
    try:
        fig.savefig(caminho)
        print(f"  -> {caminho}")
    except OSError as e:   # disco cheio, permissao negada, path invalido
        print(f"  [ERROR] {caminho}: {e}")
    if cfg.mostrar_graficos:
        global _AVISO_MOSTRAR_GRAFICOS_EMITIDO
        if not _AVISO_MOSTRAR_GRAFICOS_EMITIDO:
            print("  [AVISO] 'abrir_figuras_na_tela' esta ligado, mas o backend "
                  "grafico e sempre headless (Agg) — as figuras nao abrem em "
                  "janela, apenas sao salvas em disco normalmente.")
            _AVISO_MOSTRAR_GRAFICOS_EMITIDO = True
    # Always release the figure: in a headless/Agg backend show() is a no-op,
    # and skipping close() leaks the figure in a long-lived server process
    # (e.g. Streamlit Cloud), growing RAM across runs.
    plt.close(fig)


def especificidade_por_classe(cm: np.ndarray) -> np.ndarray:
    """Specificity = TN / (TN + FP), por classe (one-vs-rest)."""
    n = cm.shape[0]
    total = cm.sum()
    spec = np.zeros(n)
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        spec[i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return spec


# =========================================================================
#  Plot helpers
# =========================================================================

def elipse_t2(ax, x, y, color, lw=1.4, alpha=0.85,
              max_excentricidade: float = 50.0,
              limite_dispersao: Optional[Tuple[float, float]] = None):
    """Hotelling T2 95% ellipse (chi2_{2,0.95}). No fill. Returns False
    in degenerate cases: <4 points, zero minimum eigenvalue, excessive
    eccentricity, or ellipse larger than the data range itself."""
    if len(x) < 4:
        return False
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if vals[1] <= 1e-12 or vals[0] / max(vals[1], 1e-12) > max_excentricidade:
        return False
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    escala = np.sqrt(chi2.ppf(0.95, df=2))
    w = 2 * escala * np.sqrt(vals[0])
    h = 2 * escala * np.sqrt(vals[1])
    if limite_dispersao is not None:
        max_w, max_h = limite_dispersao
        if w > max_w * 2.5 or h > max_h * 2.5:
            return False
    ell = Ellipse(xy=(x.mean(), y.mean()), width=w, height=h, angle=angle,
                  edgecolor=color, facecolor="none", lw=lw,
                  alpha=alpha, zorder=2)
    ax.add_patch(ell)
    return True


def convex_hull_contorno(ax, x, y, color, lw=1.4, alpha=0.85):
    """Convex hull (fallback when n_per_class < 4 for ellipse)."""
    pts = np.column_stack([x, y])
    if len(pts) < 3:
        return False
    try:
        from scipy.spatial import ConvexHull, QhullError
        hull = ConvexHull(pts)
        verts = pts[hull.vertices]
        verts = np.vstack([verts, verts[0]])
        ax.plot(verts[:, 0], verts[:, 1], color=color, lw=lw, alpha=alpha,
                zorder=2)
        return True
    except QhullError:
        return False   # pontos colineares/degenerados -- sem contorno a tracar


def parametros_scatter_adaptativos(n_total: int, n_classes: int
                                    ) -> Tuple[float, float, float]:
    """Marker size, alpha and edge width as a function of point density."""
    n_pc = n_total / max(n_classes, 1)
    if n_pc < 12:
        return 60.0, 0.92, 0.6
    if n_pc < 40:
        return 44.0, 0.82, 0.5
    if n_pc < 100:
        return 30.0, 0.65, 0.4
    if n_pc < 250:
        return 20.0, 0.50, 0.35
    return 14.0, 0.40, 0.30


def _ticks_x_inteiros(ax, valores, limiar: int = 15, nbins: int = 10):
    """Avoids overlapping ticks on the X axis. If there are more than
    `limiar` values, uses MaxNLocator (integer steps, ~nbins divisions).
    Otherwise shows all values. Used in LV selection plots."""
    valores = np.asarray(valores)
    if len(valores) > limiar:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=nbins))
    else:
        ax.set_xticks(valores)


def plot_scores_panel(ax, scores, rotulos, mapa_cores, var_exp,
                      titulo, xlabel, ylabel, puros_mask=None,
                      mapa_marcadores=None, desenhar_elipses=True):
    """Score plot WITHOUT internal legend. The legend must be drawn
    externally via _legenda_lateral using the scatter handles.

    Distinction channels: COLOR (class) + SHAPE (mapa_marcadores, optional)
    + smart edge (light/dark by luminance). If puros_mask is given, pure
    samples get a black edge (highlight) keeping the class shape."""
    rotulos = np.asarray(rotulos, dtype=str)
    scores  = np.asarray(scores,  dtype=float)

    classes_unicas = np.unique(rotulos)
    s, alpha, lw_e = parametros_scatter_adaptativos(len(rotulos),
                                                      len(classes_unicas))

    xrange = float(scores[:, 0].max() - scores[:, 0].min())
    yrange = float(scores[:, 1].max() - scores[:, 1].min())
    limite = (xrange, yrange)

    pmask = (np.asarray(puros_mask, dtype=bool)
             if puros_mask is not None else None)

    for cls in classes_unicas:
        idx = rotulos == cls
        c = mapa_cores[cls]
        mk = mapa_marcadores.get(cls, "o") if mapa_marcadores else "o"
        edge = edge_para_cor(c)
        n_cls = int(idx.sum())
        if pmask is not None:
            idx_puro  = idx & pmask
            idx_adult = idx & (~pmask)
            if idx_adult.any():
                ax.scatter(scores[idx_adult, 0], scores[idx_adult, 1],
                           color=c, s=s, marker=mk, edgecolors=edge,
                           linewidths=lw_e, zorder=3, alpha=alpha,
                           label=f"{cls} (n={n_cls})")
            # Pure samples: same shape/color, thick black edge = highlight
            if idx_puro.any():
                ax.scatter(scores[idx_puro, 0], scores[idx_puro, 1],
                           color=c, s=s * 1.15, marker=mk,
                           edgecolors="black", linewidths=1.2, zorder=5,
                           label=(f"{cls} (n={n_cls})"
                                  if not idx_adult.any() else None))
        else:
            ax.scatter(scores[idx, 0], scores[idx, 1],
                       color=c, s=s, marker=mk, edgecolors=edge,
                       linewidths=lw_e, zorder=3, alpha=alpha,
                       label=f"{cls} (n={n_cls})")
        if desenhar_elipses:
            ok = elipse_t2(ax, scores[idx, 0], scores[idx, 1], c,
                            lw=1.4, limite_dispersao=limite)
            if not ok:
                convex_hull_contorno(ax, scores[idx, 0], scores[idx, 1],
                                       c, lw=1.3)

    ax.axhline(0, color="0.82", lw=0.5, ls="-", zorder=1)
    ax.axvline(0, color="0.82", lw=0.5, ls="-", zorder=1)
    ax.set_xlabel(f"{xlabel} ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"{ylabel} ({var_exp[1]:.1f}%)")
    ax.set_title(titulo, loc="left")

    xmin, xmax = float(scores[:, 0].min()), float(scores[:, 0].max())
    ymin, ymax = float(scores[:, 1].min()), float(scores[:, 1].max())
    px = max((xmax - xmin) * 0.08, 1e-3)
    py = max((ymax - ymin) * 0.08, 1e-3)
    ax.set_xlim(xmin - px, xmax + px)
    ax.set_ylim(ymin - py, ymax + py)
    ax.grid(color="0.94", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def _legenda_lateral(ax_leg, ax_dados, titulo: str = "Class",
                      max_col_alta: int = 18):
    """Draws external legend in a dedicated axes, without occupying the data area."""
    ax_leg.axis("off")
    handles, labels = ax_dados.get_legend_handles_labels()
    ncol = 1 if len(labels) <= max_col_alta else 2
    ax_leg.legend(handles, labels, loc="center left", frameon=False,
                   title=titulo, title_fontsize=9.5, fontsize=8.5,
                   ncol=ncol, borderaxespad=0, handletextpad=0.6,
                   labelspacing=0.6)


# =========================================================================
#  FIGURAS
# =========================================================================

def fig1_selecao_lvs(erros_rmsecv, metricas_por_lv, n_opt, cfg, pasta):
    """LV selection + CV metrics per LV."""
    n_max = len(erros_rmsecv)
    lvs = np.arange(1, n_max + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                              constrained_layout=True)

    ax = axes[0]
    ax.plot(lvs, erros_rmsecv, "o-", color=cor(0), ms=5.5, lw=1.6)
    ax.axvline(n_opt, color=cor(3), ls="--", lw=1.3,
               label=f"Optimal: {n_opt} LVs")
    ax.set_xlabel("Number of latent variables")
    ax.set_ylabel("RMSECV")
    ax.set_title("(a) LV selection", loc="left")
    _ticks_x_inteiros(ax, lvs)
    ax.grid(axis="y", color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(lvs, [m["accuracy"]          for m in metricas_por_lv],
            "o-", color=cor(0), ms=4.5, lw=1.3, label="Accuracy")
    ax.plot(lvs, [m["balanced_accuracy"] for m in metricas_por_lv],
            "s-", color=cor(2), ms=4.5, lw=1.3, label="Balanced acc.")
    ax.plot(lvs, [m["f1_macro"]          for m in metricas_por_lv],
            "^-", color=cor(1), ms=4.5, lw=1.3, label="F1 (macro)")
    ax.plot(lvs, [m["cohen_kappa"]       for m in metricas_por_lv],
            "d-", color=cor(3), ms=4.5, lw=1.3, label="Cohen's $\\kappa$")
    ax.axvline(n_opt, color="0.55", ls=":", lw=1)
    ax.set_xlabel("Number of latent variables")
    ax.set_ylabel("Metric (CV)")
    ax.set_ylim(-0.05, 1.05)
    _ticks_x_inteiros(ax, lvs)
    ax.set_title("(b) Cross-validation metrics", loc="left")
    ax.grid(axis="y", color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower right", ncol=2, frameon=False)

    salvar(fig, "figS1_selecao_lvs", pasta, cfg)


def _centroides_pca(X, rotulos, n_pcs):
    """Per-class centroid in the PCA(n_pcs) space. Reduces spectral noise
    before clustering. Returns (M_centroides, classes)."""
    rotulos = np.asarray(rotulos, dtype=str)
    classes = np.unique(rotulos)
    n_comp = int(min(n_pcs, X.shape[1], X.shape[0]))
    scores = PCA(n_components=n_comp).fit_transform(X)
    M = np.vstack([scores[rotulos == c].mean(axis=0) for c in classes])
    return M, classes


def fig_hca_dendrograma(X_processed, rotulos, mapa_cores, cfg, pasta,
                         metodo="ward"):
    """HCA dendrogram (Ward, Euclidean) on CENTROIDS per species in
    the PCA(hca_n_pcs components) space — N1 required.

    Axes (V3): TOP orientation — species on the X axis (bottom, rotated
    and colored labels per species); distance on the Y axis (left).

    HCA complements PCA: reveals the hierarchy of spectral similarity
    among species (potential confusions in PLS-DA).
    """
    rotulos = np.asarray(rotulos, dtype=str)
    M, classes = _centroides_pca(X_processed, rotulos, cfg.hca_n_pcs)
    if len(M) < 2:
        print("[WARNING] HCA: <2 groups, dendrogram skipped.")
        return

    Z = linkage(pdist(M, metric="euclidean"), method=metodo)

    largura = max(8.0, 0.55 * len(classes) + 2.0)
    fig, ax = plt.subplots(figsize=(largura, 5.2), constrained_layout=True)
    dendrogram(
        Z, labels=list(classes), orientation="top", ax=ax,
        color_threshold=0, above_threshold_color="0.55",
        leaf_font_size=10,
    )
    # Species labels on the X axis (bottom), colored and rotated
    mapa_lbl_cor = {str(c): mapa_cores.get(c, "0.4") for c in classes}
    for lbl in ax.get_xmajorticklabels():
        lbl.set_color(mapa_lbl_cor.get(lbl.get_text(), "0.2"))
        lbl.set_fontweight("bold")
        lbl.set_rotation(40)
        lbl.set_horizontalalignment("right")
    ax.set_ylabel(f"Distance ({metodo}, Euclidean)")
    ax.set_xlabel("Species", labelpad=6)
    ax.set_title(f"HCA — dendrogram (centroids, PCA {M.shape[1]} comp.)",
                  loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    salvar(fig, "fig_hca_dendrograma", pasta, cfg)

    # Automatic interpretation of main clusters (k=2)
    try:
        from scipy.cluster.hierarchy import fcluster
        grupos = fcluster(Z, t=2, criterion="maxclust")
        comp = {}
        for g in np.unique(grupos):
            comp[int(g)] = [str(classes[i]) for i in range(len(classes))
                            if grupos[i] == g]
        print("  [HCA] Main clusters (k=2):")
        for g, membros in comp.items():
            print(f"    Cluster {g}: {', '.join(membros)}")
    except ValueError as _e_fc:
        # Interpretacao de cluster e' so' um printout auxiliar no console;
        # a figura do dendrograma ja foi salva acima, intacta.
        logging.getLogger(__name__).debug(
            "HCA: interpretacao de clusters (k=2) falhou: %s", _e_fc)


def fig_hca_comparacao_pipelines(X_raw, rotulos, mapa_cores, cfg, pasta,
                                  metodo="ward"):
    """Compares HCA (centroids in PCA hca_n_pcs) under several spectral
    preprocessing methods in a dendrogram panel. Evaluates cluster
    stability and the impact of preprocessing."""
    rotulos = np.asarray(rotulos, dtype=str)

    def _snv(X):
        mu = X.mean(axis=1, keepdims=True); sd = X.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0; return (X - mu) / sd

    def _msc(X):
        ref = X.mean(axis=0); A = np.column_stack([np.ones_like(ref), ref])
        out = np.zeros_like(X)
        for i in range(X.shape[0]):
            sol, *_ = np.linalg.lstsq(A, X[i], rcond=None)
            a, b = float(sol[0]), float(sol[1])
            out[i] = (X[i] - a) / b if abs(b) > 1e-12 else X[i] - a
        return out

    def _sg(X, d):
        return savgol_filter(X, cfg.sg_window, cfg.sg_polyorder, deriv=d, axis=1)

    presets = {
        "Raw":           lambda X: X,
        "SNV":           lambda X: _snv(X),
        "MSC":           lambda X: _msc(X),
        "SG 1st deriv":  lambda X: _sg(X, 1),
        "SG 2nd deriv":  lambda X: _sg(X, 2),
        "SNV+SG1":       lambda X: _sg(_snv(X), 1),
        "MSC+SG1":       lambda X: _sg(_msc(X), 1),
        "Normalization": lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12),
    }

    n = len(presets); ncols = 4; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows),
                              constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    mapa_lbl_cor = {str(c): mapa_cores.get(c, "0.4")
                    for c in np.unique(rotulos)}
    for k, (nome, fn) in enumerate(presets.items()):
        ax = axes[k]
        try:
            Xp = np.asarray(fn(X_raw), dtype=float)
            M, classes = _centroides_pca(Xp, rotulos, cfg.hca_n_pcs)
            Z = linkage(pdist(M, metric="euclidean"), method=metodo)
            dendrogram(Z, labels=list(classes), orientation="top", ax=ax,
                       color_threshold=0, above_threshold_color="0.6",
                       leaf_font_size=6.5)
            for lbl in ax.get_xmajorticklabels():
                lbl.set_color(mapa_lbl_cor.get(lbl.get_text(), "0.2"))
                lbl.set_rotation(80); lbl.set_horizontalalignment("right")
                lbl.set_fontsize(6.5)
            ax.set_title(nome, loc="left", fontsize=9, fontweight="bold")
            ax.set_yticks([])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        except Exception as e:  # noqa: BLE001 -- 1 painel de N (comparacao de
            # pre-processamentos); erro exibido DENTRO do proprio painel
            # (mais visivel que print), os demais paineis continuam.
            ax.text(0.5, 0.5, f"failed\n{e}", ha="center", va="center",
                    fontsize=7, transform=ax.transAxes); ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("HCA by preprocessing — cluster stability",
                  fontsize=11, fontweight="bold", x=0.01, ha="left")
    salvar(fig, "fig_hca_comparacao_pipelines", pasta, cfg)


def fig1_pca_scores(scores_pca, var_pca, rotulos, mapa_cores, cfg, pasta,
                     puros_mask=None, mapa_marcadores=None):
    """Figure 1: PCA scores. External legend (outside the data area).
    Color + shape per class; pure samples with black edge (if puros_mask)."""
    fig = plt.figure(figsize=(8.5, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[5.0, 1.1])
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    plot_scores_panel(ax, scores_pca[:, :2], rotulos, mapa_cores,
                       var_pca[:2],
                       titulo="PCA — exploratory (unsupervised)",
                       xlabel="PC1", ylabel="PC2", puros_mask=puros_mask,
                       mapa_marcadores=mapa_marcadores,
                       desenhar_elipses=cfg.mostrar_elipses_grupo)
    _legenda_lateral(ax_leg, ax)
    salvar(fig, "fig1_pca_scores", pasta, cfg)


def fig2_plsda_scores(T_pls, var_lv_pls, rotulos, mapa_cores, cfg, pasta,
                       puros_mask=None, mapa_marcadores=None):
    """Figure 2: PLS-DA scores. Shows LV1xLV2 and LV2xLV3 (if >=3 LVs).
    Color + shape per class; pure samples with black edge (if puros_mask)."""
    n_lv_avail = T_pls.shape[1]
    if n_lv_avail < 2:
        print("[WARNING] PLS-DA with <2 LVs: scores plot skipped.")
        return

    if n_lv_avail >= 3:
        fig = plt.figure(figsize=(13.5, 5.6), constrained_layout=True)
        gs = fig.add_gridspec(1, 3, width_ratios=[5.0, 5.0, 1.1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax_leg = fig.add_subplot(gs[2])
        plot_scores_panel(ax1, T_pls[:, [0, 1]], rotulos, mapa_cores,
                           [var_lv_pls[0], var_lv_pls[1]],
                           titulo="(a) PLS-DA — LV1 × LV2",
                           xlabel="LV1", ylabel="LV2", puros_mask=puros_mask,
                           mapa_marcadores=mapa_marcadores,
                           desenhar_elipses=cfg.mostrar_elipses_grupo)
        plot_scores_panel(ax2, T_pls[:, [1, 2]], rotulos, mapa_cores,
                           [var_lv_pls[1], var_lv_pls[2]],
                           titulo="(b) PLS-DA — LV2 × LV3",
                           xlabel="LV2", ylabel="LV3", puros_mask=puros_mask,
                           mapa_marcadores=mapa_marcadores,
                           desenhar_elipses=cfg.mostrar_elipses_grupo)
        _legenda_lateral(ax_leg, ax1)
    else:
        fig = plt.figure(figsize=(8.5, 5.8), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[5.0, 1.1])
        ax = fig.add_subplot(gs[0])
        ax_leg = fig.add_subplot(gs[1])
        plot_scores_panel(ax, T_pls[:, :2], rotulos, mapa_cores,
                           var_lv_pls[:2],
                           titulo="PLS-DA — LV1 × LV2",
                           xlabel="LV1", ylabel="LV2", puros_mask=puros_mask,
                           mapa_marcadores=mapa_marcadores,
                           desenhar_elipses=cfg.mostrar_elipses_grupo)
        _legenda_lateral(ax_leg, ax)
    salvar(fig, "fig2_plsda_scores", pasta, cfg)


def fig3_outliers(T_scores, P_loadings, X_processed, rotulos, mapa_cores,
                   n_lv, cfg, pasta):
    n = X_processed.shape[0]
    T2 = hotelling_t2(T_scores[:, :n_lv])
    Q  = q_residuos(X_processed, T_scores[:, :n_lv], P_loadings[:n_lv])
    t2_lim = hotelling_t2_limite(n, n_lv)
    q_lim  = q_residuos_limite(Q)

    rotulos = np.asarray(rotulos, dtype=str)
    classes_unicas = np.unique(rotulos)
    s_pt, alpha_pt, lw_pt = parametros_scatter_adaptativos(
        len(rotulos), len(classes_unicas))

    fig = plt.figure(figsize=(13.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[5.0, 5.0, 1.1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax_leg = fig.add_subplot(gs[2])

    # Log scale on Y: a few samples with T2 ~1000 compress the rest to
    # the bottom. Log distributes values and keeps the 95% limit visible.
    for cls in classes_unicas:
        idx = rotulos == cls
        ax1.scatter(np.where(idx)[0], np.clip(T2[idx], 1e-2, None),
                    color=mapa_cores[cls],
                    s=s_pt, edgecolors="white", linewidths=lw_pt,
                    label=str(cls), zorder=3, alpha=alpha_pt)
    ax1.axhline(t2_lim, color="0.30", ls="--", lw=1.0,
                label=f"95% limit ({t2_lim:.1f})")
    ax1.set_yscale("log")
    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("Hotelling T$^2$ (log)")
    ax1.set_title(f"(a) Hotelling T$^2$ ({n_lv} LVs) — 95% limit",
                   loc="left")
    ax1.grid(axis="y", color="0.94", lw=0.5); ax1.set_axisbelow(True)

    # Panel (b): log-log centers the cloud (it was in the lower-left corner).
    for cls in classes_unicas:
        idx = rotulos == cls
        ax2.scatter(np.clip(T2[idx], 1e-2, None), np.clip(Q[idx], 1e-12, None),
                    color=mapa_cores[cls],
                    s=s_pt, edgecolors="white", linewidths=lw_pt,
                    zorder=3, alpha=alpha_pt)
    ax2.axvline(t2_lim, color="0.30", ls="--", lw=1.0)
    ax2.axhline(q_lim,  color="0.30", ls="--", lw=1.0)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Hotelling T$^2$ (log)")
    ax2.set_ylabel("Q-residuals / SPE (log)")
    ax2.set_title("(b) T$^2$ vs Q — outlier detection", loc="left")
    ax2.grid(color="0.94", lw=0.5); ax2.set_axisbelow(True)

    _legenda_lateral(ax_leg, ax1)

    salvar(fig, "fig3_outliers_T2_Q", pasta, cfg)

    outliers_t2 = np.where(T2 > t2_lim)[0]
    outliers_q  = np.where(Q  > q_lim)[0]
    return T2, Q, t2_lim, q_lim, outliers_t2, outliers_q


def fig4_confusao(cm_mat, classes, y_true, y_pred, cfg, pasta):
    """Confusion matrix + per-class metrics (precision, sensitivity,
    specificity, F1). Size scales with number of classes."""
    n_cls = len(classes)
    largura = max(10.5, 5.5 + 0.55 * n_cls)
    altura  = max(4.8, 4.2 + 0.10 * n_cls)
    fig = plt.figure(figsize=(largura, altura), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])

    # ---- (a) Confusion matrix -------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    cm_norm = cm_mat.astype(float) / np.maximum(
        cm_mat.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    thresh = 0.55

    fs_anot = 10 if n_cls <= 6 else (8.5 if n_cls <= 10 else 7)
    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j, i, f"{cm_mat[i, j]}\n{cm_norm[i, j]*100:.1f}%",
                    ha="center", va="center",
                    fontsize=fs_anot, fontweight="bold",
                    color="white" if cm_norm[i, j] > thresh else "0.15")

    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    fs_tick = 9 if n_cls <= 8 else 8
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=fs_tick)
    ax.set_yticklabels(classes, fontsize=fs_tick)
    ax.set_xlabel("Predicted", labelpad=8)
    ax.set_ylabel("True",      labelpad=8)
    ax.set_title("(a) Confusion matrix (CV)", loc="left")

    ax.set_xticks(np.arange(-0.5, n_cls, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_cls, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion per true class", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    # ---- (b) Per-class metrics -----------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    prec = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    rec  = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    spec = especificidade_por_classe(cm_mat)
    f1   = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

    nomes_m   = ["Precision", "Sensitivity", "Specificity", "F1-score"]
    valores_m = [prec, rec, spec, f1]
    cores_m   = [cor(0), cor(2), cor(1), cor(3)]

    x = np.arange(n_cls)
    width = 0.20
    for k, (nome, val, cm_) in enumerate(zip(nomes_m, valores_m, cores_m)):
        ax.bar(x + (k - 1.5) * width, val, width, label=nome,
                color=cm_, edgecolor="white", lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=fs_tick)
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1.10)
    ax.set_title("(b) Per-class metrics (one-vs-rest)", loc="left", pad=22)
    ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    # Legend ABOVE the plot (horizontal, 4 cols) so it never overlaps the bars
    # (which reach ~1.0 for well-classified species like Pracaxi/Andiroba).
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
              fontsize=7.5, frameon=False, columnspacing=1.0,
              handletextpad=0.4)

    salvar(fig, "fig4_confusao_e_metricas_por_classe", pasta, cfg)


# fig4b_metricas_globais: REMOVIDA — as metricas globais (accuracy, balanced,
# F1, kappa) e o resultado do teste de permutacao ja constam em
# logs/resumo_modelo.txt; a figura era redundante e nao era mais chamada.


# Chemical band assignments for FT-NIR of vegetable oils (M3).
# References: Workman & Weyer (2012) Practical Guide to Interpretive
# Near-IR Spectroscopy; Cen & He (2007) Trends Food Sci Technol 18:72.
BANDAS_NIR: List[Tuple[float, str]] = [
    (4255, "C-H comb.\n(ac. graxos)"),
    (4325, "O-H comb."),
    (4665, "C-H/C=C"),
    (5180, "O-H (agua)"),
    (5800, "C=O 1o overt."),
    (5900, "C-H 1o overt."),
    (7050, "O-H 1o overt."),
    (8500, "C-H 2o overt."),
]


def _anotar_sem_sobreposicao(ax, xs, ys, labels, ref_x=None, ref_y=None,
                              fontsize=6.5, color="0.2", min_dist_frac=0.05,
                              max_labels=None, offset=(3, 3)):
    """Annotate points but SKIP labels that would overlap already-placed ones.

    Greedy spatial thinning: processes points in the given order (assumed
    sorted by importance) and places a label only if the point is at least
    `min_dist_frac` of the data range away — in BOTH x and y — from every
    label already placed. Prevents the cluttered stacks of wavenumber labels
    that appear when many top features cluster in the same plot region.

    ref_x/ref_y: full data arrays used to compute the spatial scale (so the
    threshold reflects the whole plot, not just the labeled subset).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    rx = np.asarray(ref_x if ref_x is not None else xs, dtype=float)
    ry = np.asarray(ref_y if ref_y is not None else ys, dtype=float)
    dx = float(np.ptp(rx)) or 1.0
    dy = float(np.ptp(ry)) or 1.0
    placed: List[Tuple[float, float]] = []
    for k in range(len(xs)):
        xk, yk = float(xs[k]), float(ys[k])
        if any(abs(xk - px) / dx < min_dist_frac and
               abs(yk - py) / dy < min_dist_frac for px, py in placed):
            continue
        ax.annotate(str(labels[k]), xy=(xk, yk), xytext=offset,
                    textcoords="offset points", fontsize=fontsize, color=color)
        placed.append((xk, yk))
        if max_labels is not None and len(placed) >= max_labels:
            break
    return len(placed)


def _anotar_bandas_vip(ax, wavenumbers, vip, limiar=2.0, janela=120.0,
                        sep_frac=0.052):
    """M3: annotates known chemical bands at VIP peaks, with GUARANTEED
    non-overlap and a safe margin between labels.

    All labels are vertical text placed in a single row above the peaks. Their
    x-positions are spread by iterative relaxation so consecutive labels keep a
    minimum horizontal gap (sep_frac of the spectral span) — no overlap even
    for bands only 100 cm-1 apart (C=O 5800 & C-H 5900). A short, thin arrow
    links each spread label to its true spectral peak (no long crossing lines).
    """
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    vip = np.asarray(vip, dtype=float)
    if vip.size == 0:
        return
    span = float(np.ptp(wavenumbers)) or 1.0
    vmax = float(np.nanmax(vip))

    # 1. Collect annotatable bands (local VIP peak above threshold).
    cand = []
    for centro, rotulo in BANDAS_NIR:
        viz = np.abs(wavenumbers - centro) <= janela
        if not viz.any():
            continue
        idx_viz = np.where(viz)[0]
        i_pico = idx_viz[int(np.argmax(vip[idx_viz]))]
        if vip[i_pico] < limiar:
            continue
        # Single-line label: a newline in 90°-rotated text would create two
        # side-by-side columns. Join into one compact vertical string.
        rot = rotulo.replace("\n", " ")
        cand.append([float(wavenumbers[i_pico]), float(vip[i_pico]), rot])
    if not cand:
        return
    cand.sort(key=lambda c: c[0])

    # 2. Spread x-positions by relaxation until every gap >= min_sep.
    min_sep = sep_frac * span
    xs = [c[0] for c in cand]
    for _ in range(300):
        moved = False
        for i in range(len(xs) - 1):
            gap = xs[i + 1] - xs[i]
            if gap < min_sep:
                d = (min_sep - gap) / 2.0
                xs[i] -= d
                xs[i + 1] += d
                moved = True
        if not moved:
            break
    xlo, xhi = float(wavenumbers.min()), float(wavenumbers.max())
    xs = [min(max(x, xlo), xhi) for x in xs]

    # 3. Uniform label base ABOVE all peaks; vertical text grows upward.
    y_base = vmax * 1.04
    for x_lbl, (x_pk, y_pk, rot) in zip(xs, cand):
        ax.annotate(
            rot,
            xy=(x_pk, y_pk),               # arrow tip at the real peak
            xytext=(x_lbl, y_base),        # spread, non-overlapping label base
            ha="center", va="bottom", rotation=90, rotation_mode="anchor",
            fontsize=6.2, color="0.20", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="0.55", lw=0.5,
                            shrinkA=1.5, shrinkB=1.5),
            zorder=6,
        )


def fig_espectros_medios_classe(wavenumbers, X_raw, rotulos, mapa_cores,
                                 cfg, pasta):
    """Espectros medios por classe (banda = +-1 desvio-padrao) do dado
    BRUTO -- contexto quimico antes de qualquer modelagem (item 3 da
    lista de figuras que faltavam, CLAUDE.md secao 5). Ao contrario de
    fig6_preprocessamento (comparacao antes/depois do pre-processamento,
    restrita ao objetivo Exploratorio), esta e' CONTEXTO valido em
    qualquer objetivo -- mesma logica de fig1_pca_scores/fig3_outliers,
    por isso e' chamada incondicionalmente pelo executar()."""
    rotulos = np.asarray(rotulos, dtype=str)
    fig = plt.figure(figsize=(10.5, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[6.0, 1.3])
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])

    for cls in np.unique(rotulos):
        idx = rotulos == cls
        m = X_raw[idx].mean(axis=0)
        s = X_raw[idx].std(axis=0)
        c = mapa_cores.get(cls, "0.4")
        ax.plot(wavenumbers, m, color=c, lw=1.2, label=str(cls))
        ax.fill_between(wavenumbers, m - s, m + s, color=c, alpha=0.15, lw=0)

    ax.set_xlabel("Número de onda (cm$^{-1}$)")
    ax.set_ylabel("Absorbância")
    ax.set_title("Espectros médios por classe (banda = ±1 DP) — "
                 "dado bruto, antes da modelagem", loc="left")
    if len(wavenumbers) > 1 and wavenumbers[0] < wavenumbers[-1]:
        ax.invert_xaxis()
    ax.grid(axis="y", color="0.94", lw=0.5); ax.set_axisbelow(True)
    _legenda_lateral(ax_leg, ax)

    salvar(fig, "fig0_espectros_medios_classe", pasta, cfg)


def fig6_preprocessamento(wavenumbers, X_raw, X_processed, rotulos,
                           mapa_cores, cfg, pasta):
    rotulos = np.asarray(rotulos, dtype=str)
    fig = plt.figure(figsize=(11.5, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[6.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_leg = fig.add_subplot(gs[:, 1])

    for cls in np.unique(rotulos):
        idx = rotulos == cls
        m_raw = X_raw[idx].mean(axis=0)
        s_raw = X_raw[idx].std(axis=0)
        m_pp  = X_processed[idx].mean(axis=0)
        s_pp  = X_processed[idx].std(axis=0)
        c = mapa_cores[cls]
        ax_a.plot(wavenumbers, m_raw, color=c, lw=1.2, label=str(cls))
        ax_a.fill_between(wavenumbers, m_raw - s_raw, m_raw + s_raw,
                           color=c, alpha=0.15, lw=0)
        ax_b.plot(wavenumbers, m_pp, color=c, lw=1.2)
        ax_b.fill_between(wavenumbers, m_pp - s_pp, m_pp + s_pp,
                           color=c, alpha=0.15, lw=0)

    ax_a.set_ylabel("Absorbance")
    ax_a.set_title("(a) Raw spectra (mean $\\pm$ SD per class)",
                    loc="left")
    ax_a.grid(axis="y", color="0.94", lw=0.5); ax_a.set_axisbelow(True)

    descricao = []
    if cfg.aplicar_snv: descricao.append("SNV")
    if cfg.aplicar_sg:
        descricao.append(f"SG(w={cfg.sg_window},p={cfg.sg_polyorder},d={cfg.sg_deriv})")
    if cfg.aplicar_mc:  descricao.append("mean-centering")
    ax_b.set_ylabel("Preprocessed signal")
    ax_b.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax_b.set_title(f"(b) After {' → '.join(descricao)}", loc="left")
    ax_b.invert_xaxis()
    ax_b.axhline(0, color="0.75", lw=0.5, ls=":")
    ax_b.grid(axis="y", color="0.94", lw=0.5); ax_b.set_axisbelow(True)

    _legenda_lateral(ax_leg, ax_a)

    salvar(fig, "fig6_preprocessamento", pasta, cfg)


def fig_extra_wold(wold: Dict[str, object], cfg, pasta):
    """Wold-style permutation plot: R2Y and Q2Y vs similarity of permuted Y."""
    sims = np.asarray(cast(Any, wold["sims"]))
    r2s  = np.asarray(cast(Any, wold["r2s"]))
    q2s  = np.asarray(cast(Any, wold["q2s"]))
    r2_obs   = cast(float, wold["r2_obs"])
    q2_obs   = cast(float, wold["q2_obs"])
    int_r2   = cast(float, wold["intercept_r2"])
    int_q2   = cast(float, wold["intercept_q2"])
    slope_r2 = cast(float, wold["slope_r2"])
    slope_q2 = cast(float, wold["slope_q2"])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                              constrained_layout=True)

    x_line = np.linspace(0, 1, 50)

    ax = axes[0]
    ax.scatter(sims, r2s, color=cor(0), s=28, alpha=0.55,
                edgecolors="white", linewidths=0.4, label="Permutations")
    ax.scatter([1.0], [r2_obs], color=cor(3), s=90, marker="D",
                edgecolors="black", linewidths=0.8, zorder=5,
                label="Observed")
    if np.isfinite(slope_r2):
        ax.plot(x_line, slope_r2 * x_line + int_r2, color="0.35", lw=1.2,
                 ls="--", label=f"Line (intercept = {int_r2:.3f})")
    ax.axhline(0.40, color=cor(1), lw=0.9, ls=":",
                label="Threshold R2Y = 0.40")
    cor_status = cor(2) if (np.isfinite(int_r2) and int_r2 < 0.40) else cor(3)
    status = "VALID" if (np.isfinite(int_r2) and int_r2 < 0.40) else ("N/A" if not np.isfinite(int_r2) else "FAILED")
    ax.text(0.02, 0.97, f"R2Y intercept: {status}",
             transform=ax.transAxes, ha="left", va="top",
             fontsize=9, fontweight="bold", color=cor_status,
             bbox=dict(boxstyle="round,pad=0.35", fc="white",
                        ec=cor_status, lw=0.8))
    ax.set_xlabel("Similarity (permuted Y, original Y)")
    ax.set_ylabel("R$^2$Y (training fit)")
    ax.set_xlim(-0.05, 1.08); ax.set_ylim(-0.5, 1.08)
    ax.set_title("(a) Wold — R$^2$Y vs permutation", loc="left")
    ax.grid(color="0.94", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    ax = axes[1]
    ax.scatter(sims, q2s, color=cor(2), s=28, alpha=0.55,
                edgecolors="white", linewidths=0.4, label="Permutations")
    ax.scatter([1.0], [q2_obs], color=cor(3), s=90, marker="D",
                edgecolors="black", linewidths=0.8, zorder=5,
                label="Observed")
    if np.isfinite(slope_q2):
        ax.plot(x_line, slope_q2 * x_line + int_q2, color="0.35", lw=1.2,
                 ls="--", label=f"Line (intercept = {int_q2:.3f})")
    ax.axhline(0.05, color=cor(1), lw=0.9, ls=":",
                label="Threshold Q2Y = 0.05")
    cor_status = cor(2) if (np.isfinite(int_q2) and int_q2 < 0.05) else cor(3)
    status = "VALID" if (np.isfinite(int_q2) and int_q2 < 0.05) else ("N/A" if not np.isfinite(int_q2) else "FAILED")
    ax.text(0.02, 0.97, f"Q2Y intercept: {status}",
             transform=ax.transAxes, ha="left", va="top",
             fontsize=9, fontweight="bold", color=cor_status,
             bbox=dict(boxstyle="round,pad=0.35", fc="white",
                        ec=cor_status, lw=0.8))
    ax.set_xlabel("Similarity (permuted Y, original Y)")
    ax.set_ylabel("Q$^2$Y (CV)")
    ax.set_xlim(-0.05, 1.08); ax.set_ylim(-0.6, 1.08)
    ax.set_title("(b) Wold — Q$^2$Y vs permutation", loc="left")
    ax.grid(color="0.94", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    salvar(fig, "fig_extra_wold_permutacao", pasta, cfg)


def fig_extra_holdout(metricas_cv: Dict[str, float],
                       metricas_holdout: Dict[str, float],
                       cm_holdout: np.ndarray, classes: np.ndarray,
                       n_holdout: int, cfg, pasta):
    """Holdout confusion matrix + CV vs holdout comparison (overfitting check)."""
    n_cls = len(classes)
    fig = plt.figure(figsize=(13.0, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1])

    # (a) Holdout confusion matrix
    ax = fig.add_subplot(gs[0])
    cm_norm = cm_holdout.astype(float) / np.maximum(
        cm_holdout.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    fs_an = 10 if n_cls <= 6 else (8.5 if n_cls <= 10 else 7)
    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j, i, f"{cm_holdout[i, j]}\n{cm_norm[i, j]*100:.1f}%",
                    ha="center", va="center", fontsize=fs_an,
                    fontweight="bold",
                    color="white" if cm_norm[i, j] > 0.55 else "0.15")
    ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted", labelpad=8); ax.set_ylabel("True", labelpad=8)
    ax.set_title(f"(a) Confusion matrix — holdout ({n_holdout} samples)",
                  loc="left")
    ax.set_xticks(np.arange(-0.5, n_cls, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_cls, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion per true class", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    # (b) CV vs Holdout comparison
    ax = fig.add_subplot(gs[1])
    nomes  = ["Accuracy", "Balanced acc.", "F1 (macro)", "Cohen's $\\kappa$"]
    chaves = ["accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa"]
    cv_vals = [metricas_cv[k]      for k in chaves]
    ho_vals = [metricas_holdout[k] for k in chaves]
    x = np.arange(len(nomes)); w = 0.36
    ax.bar(x - w/2, cv_vals, w, color=cor(0), label="CV (training)",
            edgecolor="white", lw=0.5)
    ax.bar(x + w/2, ho_vals, w, color=cor(1), label="Holdout (test)",
            edgecolor="white", lw=0.5)
    for k, (cv_v, ho_v) in enumerate(zip(cv_vals, ho_vals)):
        ax.text(k - w/2, cv_v + 0.015, f"{cv_v:.3f}", ha="center",
                va="bottom", fontsize=8)
        ax.text(k + w/2, ho_v + 0.015, f"{ho_v:.3f}", ha="center",
                va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(nomes, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Value")
    ax.set_title("(b) Cross-validation vs holdout (overfitting check)",
                  loc="left")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(axis="y", color="0.93", lw=0.5); ax.set_axisbelow(True)

    salvar(fig, "fig_extra_holdout", pasta, cfg)


def fig_extra_comparacao_pipelines(resultados, cfg, pasta):
    """Comparison of preprocessing pipelines (Q2, Acc, BalAcc)."""
    nomes = list(resultados.keys())
    accs = [resultados[n]["accuracy"]     for n in nomes]
    bals = [resultados[n]["balanced_acc"] for n in nomes]
    q2s  = [max(resultados[n]["q2"], -0.2) for n in nomes]
    n_lvs = [resultados[n]["n_lv"]        for n in nomes]

    fig, ax = plt.subplots(figsize=(11.0, 0.55 * len(nomes) + 2.2),
                            constrained_layout=True)
    pos = np.arange(len(nomes))
    h = 0.26
    ax.barh(pos - h, accs, h, color=cor(0), label="Accuracy",
             edgecolor="white", lw=0.5)
    ax.barh(pos,     bals, h, color=cor(2), label="Balanced acc.",
             edgecolor="white", lw=0.5)
    ax.barh(pos + h, q2s,  h, color=cor(1), label="Q$^2$",
             edgecolor="white", lw=0.5)

    for k, n_lv in enumerate(n_lvs):
        ax.text(1.005, k, f" {n_lv} LVs", va="center", ha="left",
                 fontsize=8.5, color="0.30",
                 transform=ax.get_yaxis_transform())

    ax.set_yticks(pos)
    ax.set_yticklabels(nomes, fontsize=9.5)
    ax.set_xlim(-0.05, 1.12)
    ax.set_xlabel("Value (cross-validation)")
    ax.set_title("Comparison of preprocessing pipelines",
                  loc="left")
    ax.axvline(0, color="0.7", lw=0.5)
    ax.invert_yaxis()
    ax.grid(axis="x", color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=3)

    salvar(fig, "fig_extra_comparacao_pipelines", pasta, cfg)


def fig5b_vip_estabilidade(boot: Dict[str, object], wavenumbers,
                            top_n, cfg, pasta):
    """Stratified VIP bootstrap: mean, CI95 and selection frequency."""
    vip_mean = np.asarray(cast(Any, boot["mean"]))
    ci_lo    = np.asarray(cast(Any, boot["ci95_low"]))
    ci_hi    = np.asarray(cast(Any, boot["ci95_high"]))
    sel_freq = np.asarray(cast(Any, boot["selection_frequency"]))

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3),
                              constrained_layout=True,
                              gridspec_kw={"width_ratios": [1.7, 1.0, 1.0]})

    ax = axes[0]
    ax.fill_between(wavenumbers, ci_lo, ci_hi,
                     color=cor(0), alpha=0.22, lw=0, zorder=2,
                     label="95% CI (bootstrap)")
    ax.plot(wavenumbers, vip_mean, color="0.25", lw=1.0, alpha=0.95, zorder=3)
    ax.axhline(1.0, color=cor(3), ls="--", lw=1.0, label="VIP = 1")
    _anotar_bandas_vip(ax, wavenumbers, vip_mean, limiar=2.0)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("VIP (stratified bootstrap mean)")
    ax.set_title("(a) Mean VIP with 95% CI", loc="left")
    ax.invert_xaxis()
    # Generous headroom so vertical band labels do not clip at the top of
    # the subplot.
    _vm = float(np.nanmax(vip_mean)) if vip_mean.size else 1.0
    ax.set_ylim(bottom=0.0, top=_vm * 1.45)
    ax.grid(axis="y", color="0.94", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    ax = axes[1]
    ord_vip = np.argsort(vip_mean)[::-1]
    top_n = min(top_n, len(wavenumbers))
    idx_top = ord_vip[:top_n][::-1]
    valores = vip_mean[idx_top]
    erro_inf = valores - ci_lo[idx_top]
    erro_sup = ci_hi[idx_top] - valores
    pos = np.arange(top_n)
    cores_b = [cor(1) if v >= 1.0 else "0.7" for v in valores]
    ax.barh(pos, valores, color=cores_b, edgecolor="white", lw=0.5,
             height=0.78,
             xerr=np.vstack([erro_inf, erro_sup]),
             error_kw=dict(ecolor="0.35", lw=0.8, capsize=2))
    ax.axvline(1.0, color=cor(3), ls="--", lw=1.0)
    ax.set_yticks(pos)
    ax.set_yticklabels([f"{wavenumbers[i]:.0f}" for i in idx_top], fontsize=8)
    ax.set_xlabel("VIP score (with 95% CI)")
    ax.set_ylabel("Wavenumber (cm$^{-1}$)")
    ax.set_title(f"(b) Top {top_n} variables", loc="left")
    ax.grid(axis="x", color="0.94", lw=0.5); ax.set_axisbelow(True)

    ax = axes[2]
    freq_top = sel_freq[idx_top]
    ax.barh(pos, freq_top, color=cor(2), edgecolor="white", lw=0.5,
             height=0.78)
    ax.axvline(0.5, color="0.55", ls=":", lw=0.9)
    ax.set_yticks(pos)
    ax.set_yticklabels([f"{wavenumbers[i]:.0f}" for i in idx_top], fontsize=8)
    ax.set_xlabel("Selection frequency (VIP $\\geq$ 1)")
    ax.set_xlim(0, 1.05)
    ax.set_title("(c) Selection stability", loc="left")
    ax.grid(axis="x", color="0.94", lw=0.5); ax.set_axisbelow(True)

    salvar(fig, "fig5b_vip_bootstrap", pasta, cfg)


def fig7_pls_regressao(Yc, Yc_hat, Yv, Yv_hat, erros_reg, n_opt_reg,
                        r2c, r2v, rmsec, rmsecv, rmsep, bias_v, cfg, pasta):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2),
                              constrained_layout=True)

    n_max = len(erros_reg)
    lvs = np.arange(1, n_max + 1)
    ax = axes[0]
    ax.plot(lvs, erros_reg, "o-", color=cor(0), ms=5, lw=1.4)
    ax.axvline(n_opt_reg, color=cor(1), ls="--", lw=1.2,
               label=f"Optimal: {n_opt_reg} LVs")
    ax.set_xlabel("Number of latent variables")
    ax.set_ylabel("RMSECV")
    _ticks_x_inteiros(ax, lvs)
    ax.set_title("(a) LV selection", loc="left")
    ax.legend()

    Yc_f = np.asarray(Yc).flatten(); Yc_h = np.asarray(Yc_hat).flatten()
    Yv_f = np.asarray(Yv).flatten(); Yv_h = np.asarray(Yv_hat).flatten()
    todos = np.concatenate([Yc_f, Yc_h, Yv_f, Yv_h])
    lim = [todos.min() - 1, todos.max() + 1]

    ax = axes[1]
    ax.scatter(Yc_f, Yc_h, color=cor(0), s=36, edgecolors="white",
               linewidths=0.5, label="Calibration", zorder=3, alpha=0.9)
    ax.scatter(Yv_f, Yv_h, color=cor(1), s=44, marker="^",
               edgecolors="white", linewidths=0.5,
               label="Validation", zorder=3, alpha=0.9)
    ax.plot(lim, lim, "k--", lw=0.8, label="y = x")
    ax.set_xlabel("Reference value")
    ax.set_ylabel("Predicted value")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(
        f"(b) Predicted vs Reference\n"
        f"$R^2_{{cal}}$={r2c:.3f}  $R^2_{{val}}$={r2v:.3f}",
        loc="left")
    ax.legend(loc="best")

    res = Yv_f - Yv_h
    ax = axes[2]
    ax.scatter(Yv_h, res, color=cor(2), s=44, edgecolors="white",
               linewidths=0.5, zorder=3, alpha=0.9)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axhline( rmsep, color=cor(3), lw=0.8, ls=":",
                label=f"$\\pm$RMSEP ({rmsep:.2f})")
    ax.axhline(-rmsep, color=cor(3), lw=0.8, ls=":")
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual")
    ax.set_title(f"(c) Residuals — Validation\nBias = {bias_v:.3f}", loc="left")
    ax.legend(loc="best")

    salvar(fig, "figS2_pls_regressao", pasta, cfg)


def fig_merito_regressao(tabela_especie: List[Dict[str, Any]], cfg, pasta) -> None:
    """Figura de merito analitica dedicada (auditoria jul/2026, item 5):
    LOD/LOQ e Seletividade media por especie, lado a lado — Valderrama,
    Braga & Poppi (2009), Quim. Nova 32(5):1278-1287. Ate aqui, LOD/LOQ/SEN/
    SEL so apareciam como tabela de TEXTO em resumo_modelo.txt/model_card.md;
    esta e' a primeira representacao visual.

    `tabela_especie` e' a mesma lista de dicts usada no resumo (chaves
    'especie'/'lod'/'loq'/'seletividade_media'; ver
    chemometric_stats.figuras_merito_regressao). Especies sem replicas
    fisicas suficientes tem lod/loq/seletividade = NaN — aparecem no eixo
    com rotulo 'n/a' em vez de quebrar o layout ou sumir silenciosamente.
    """
    if not tabela_especie:
        return
    especies = [str(t.get("especie", "?")) for t in tabela_especie]
    lod = np.array([t.get("lod", float("nan")) for t in tabela_especie], dtype=float)
    loq = np.array([t.get("loq", float("nan")) for t in tabela_especie], dtype=float)
    sel = np.array([t.get("seletividade_media", float("nan"))
                    for t in tabela_especie], dtype=float)

    n = len(especies)
    x = np.arange(n)
    largura = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(max(8.5, 1.05 * n), 4.6),
                              constrained_layout=True)

    ax = axes[0]
    tem_lod = np.isfinite(lod)
    if tem_lod.any():
        ax.bar(x[tem_lod] - largura / 2, lod[tem_lod], largura,
               label="LOD", color=cor(0))
        ax.bar(x[tem_lod] + largura / 2, loq[tem_lod], largura,
               label="LOQ", color=cor(1))
        ax.legend()
    for xi, ok in zip(x, tem_lod):
        if not ok:
            ax.text(xi, 0, "n/a", ha="center", va="bottom", fontsize=8,
                    color="gray", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(especies, rotation=40, ha="right")
    ax.set_ylabel("Concentration (%)")
    ax.set_title("(a) LOD / LOQ per species", loc="left")

    ax = axes[1]
    tem_sel = np.isfinite(sel)
    if tem_sel.any():
        ax.bar(x[tem_sel], sel[tem_sel], color=cor(2))
    for xi, ok in zip(x, tem_sel):
        if not ok:
            ax.text(xi, 0, "n/a", ha="center", va="bottom", fontsize=8,
                    color="gray", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(especies, rotation=40, ha="right")
    ax.set_ylabel("Selectivity ratio (mean)")
    ax.set_title("(b) Analytical selectivity", loc="left")

    salvar(fig, "figS3_merito_regressao", pasta, cfg)


# =========================================================================
#  Sprint 3 Figures
# =========================================================================

def fig_sprint3_sr_vip(vip: np.ndarray, sr: np.ndarray,
                        wavenumbers: np.ndarray, top_n: int,
                        cfg: Config, pasta: str) -> None:
    """VIP (all LVs) vs SR (1st component) side by side.

    Agreement between VIP >= 1 and high SR reinforces that the variable is
    relevant both globally (VIP) and in the main discriminant component (SR).
    Disagreements reveal variables important only in secondary LVs.

    Reference: Rajalahti et al. (2009), Chemom. Intell. Lab. Syst.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4),
                              constrained_layout=True)

    # (a) Spectrum: VIP + SR on dual axes
    ax = axes[0]
    ax2 = ax.twinx()
    c_vip, c_sr = cor(0), cor(1)

    l1, = ax.plot(wavenumbers, vip, color=c_vip, lw=1.1,
                   alpha=0.9, label="VIP")
    ax.axhline(1.0, color=c_vip, ls="--", lw=0.8, alpha=0.5)
    mask_vip = vip >= 1.0
    ax.scatter(wavenumbers[mask_vip], vip[mask_vip],
               color=c_vip, s=9, zorder=3, alpha=0.8)

    l2, = ax2.plot(wavenumbers, sr, color=c_sr, lw=1.1,
                    alpha=0.8, label="SR")
    sr_thr = float(np.percentile(sr, 75))   # 75th percentile as reference
    ax2.axhline(sr_thr, color=c_sr, ls="--", lw=0.8, alpha=0.5,
                label=f"SR Q75 = {sr_thr:.2f}")

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("VIP score", color=c_vip)
    ax2.set_ylabel("Selectivity Ratio", color=c_sr)
    ax.tick_params(axis="y", labelcolor=c_vip)
    ax2.tick_params(axis="y", labelcolor=c_sr)
    ax.invert_xaxis()
    ax.set_title("(a) VIP and SR across the spectrum", loc="left")
    ax.grid(axis="y", color="0.93", lw=0.5); ax.set_axisbelow(True)
    ax.legend(handles=[l1, l2], loc="upper left", frameon=False,
               fontsize=8.5)

    # (b) Top N by SR (dual bars: normalized VIP + SR)
    ax = axes[1]
    top_n = min(top_n, len(wavenumbers))
    idx_sr = np.argsort(sr)[::-1][:top_n][::-1]
    pos     = np.arange(top_n)
    sr_max  = max(float(sr.max()),  1e-12)
    vip_max = max(float(vip.max()), 1e-12)
    sr_top  = sr[idx_sr]  / sr_max
    vip_top = vip[idx_sr] / vip_max
    w_b = 0.38
    ax.barh(pos - w_b / 2, sr_top,  w_b, color=c_sr,
             edgecolor="white", lw=0.5, label="SR (norm.)")
    ax.barh(pos + w_b / 2, vip_top, w_b, color=c_vip,
             edgecolor="white", lw=0.5, label="VIP (norm.)")
    ax.axvline(1.0 / vip_max, color=c_vip, ls=":", lw=0.8, alpha=0.6)
    ax.set_yticks(pos)
    ax.set_yticklabels([f"{wavenumbers[i]:.0f}" for i in idx_sr],
                        fontsize=8)
    ax.set_xlabel("Normalized value (global max = 1)")
    ax.set_ylabel("Wavenumber (cm$^{-1}$)")
    ax.set_title(f"(b) Top {top_n} by SR vs VIP", loc="left")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.grid(axis="x", color="0.93", lw=0.5); ax.set_axisbelow(True)

    salvar(fig, "fig_sprint3_sr_vip", pasta, cfg)


def fig_sprint3_score_contribution(pls_model: PLSRegression,
                                    X_processed: np.ndarray,
                                    rotulos: np.ndarray,
                                    wavenumbers: np.ndarray,
                                    mapa_cores: Dict[str, str],
                                    top_n: int,
                                    cfg: Config, pasta: str) -> None:
    """Per-variable contribution to LV1 score per class (B8).

    Contribution of variable j to the score of sample i on LV1:
        c_ij = x_ij * w*_j   (normalized PLS weight)

    Class mean reveals which spectral regions 'push' each class toward
    its side in score space — complementary information to VIP (global)
    and SR (scalar).

    Panels:
        (a) Mean contribution spectrum ± SE per class
        (b) Top N variables with highest discriminant power between classes
    """
    rotulos = np.asarray(rotulos, dtype=str)
    classes = np.unique(rotulos)
    W = np.asarray(pls_model.x_weights_, dtype=float)   # (p, n_lv)
    if W.shape[1] < 1:
        return

    w1 = W[:, 0] / (float(np.linalg.norm(W[:, 0])) + 1e-12)
    contrib = X_processed * w1[np.newaxis, :]   # (n, p)

    means: Dict[str, np.ndarray] = {}
    sems:  Dict[str, np.ndarray] = {}
    for cls in classes:
        idx = rotulos == cls
        c = contrib[idx]
        n_cls_amostras = int(idx.sum())
        means[cls] = c.mean(axis=0)
        # ddof=1 divide por (n-1); com 1 amostra isso vira 0/0 = NaN e a banda
        # de erro some silenciosamente. Classe singleton -> SEM = 0 (sem
        # dispersao estimavel), evitando NaN no grafico.
        if n_cls_amostras > 1:
            sems[cls] = c.std(axis=0, ddof=1) / float(np.sqrt(n_cls_amostras))
        else:
            sems[cls] = np.zeros(c.shape[1])

    n_cls = len(classes)

    # ===== FIGURE 1: mean contribution spectrum (lateral legend) ==========
    fig = plt.figure(figsize=(10.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[6.0, 1.1])
    ax = fig.add_subplot(gs[0]); ax_leg = fig.add_subplot(gs[1])
    for cls in classes:
        c = mapa_cores.get(cls, "0.5")
        m = means[cls]; s = sems[cls]
        ax.plot(wavenumbers, m, color=c, lw=1.2, label=str(cls))
        ax.fill_between(wavenumbers, m - s, m + s, color=c, alpha=0.18, lw=0)
    ax.axhline(0, color="0.70", lw=0.7, ls=":")
    ax.invert_xaxis()
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("$c_j = x_j \\cdot w^*_j$  (LV1)")
    ax.set_title("Mean spectral contribution per class — LV1", loc="left")
    ax.grid(axis="y", color="0.94", lw=0.5); ax.set_axisbelow(True)
    _legenda_lateral(ax_leg, ax)
    salvar(fig, "fig_score_contribution_espectro", pasta, cfg)

    # ===== FIGURE 2: Top N discriminant power (separate, tall, readable) ===
    mean_stack = np.vstack([means[c] for c in classes])   # (n_cls, p)
    disc_power = mean_stack.var(axis=0)                    # (p,)
    top_idx = np.argsort(disc_power)[::-1][:top_n][::-1]
    pos = np.arange(top_n)
    w_bar = 0.84 / max(n_cls, 1)

    altura = max(6.0, 0.42 * top_n + 1.6)   # scales with top_n -> readable
    fig2 = plt.figure(figsize=(11.0, altura), constrained_layout=True)
    gs2 = fig2.add_gridspec(1, 2, width_ratios=[6.0, 1.2])
    ax = fig2.add_subplot(gs2[0]); ax_leg2 = fig2.add_subplot(gs2[1])
    for k, cls in enumerate(classes):
        offset = (k - (n_cls - 1) / 2.0) * w_bar
        ax.barh(pos + offset, means[cls][top_idx], w_bar,
                color=mapa_cores.get(cls, "0.5"),
                edgecolor="white", lw=0.4, label=str(cls),
                xerr=sems[cls][top_idx],
                error_kw=dict(ecolor="0.45", lw=0.6, capsize=1.2))
    ax.axvline(0, color="0.40", lw=0.8, ls="--")
    ax.set_yticks(pos)
    ax.set_yticklabels([f"{wavenumbers[i]:.0f}" for i in top_idx], fontsize=8.5)
    ax.set_ylim(-0.5, top_n - 0.5)
    ax.set_xlabel("Mean contribution $\\pm$ SE per class")
    ax.set_ylabel("Wavenumber (cm$^{-1}$)")
    ax.set_title(f"Top {top_n} variables — discriminant power between classes",
                  loc="left")
    ax.grid(axis="x", color="0.93", lw=0.5); ax.set_axisbelow(True)
    _legenda_lateral(ax_leg2, ax)
    salvar(fig2, "fig_score_contribution_top_discriminante", pasta, cfg)


def fig_sprint3_ddsimca_acceptance(scores: Dict[str, Dict[str, Any]],
                                    rotulos: np.ndarray,
                                    mapa_cores: Dict[str, str],
                                    cfg: Config, pasta: str,
                                    sens_esp: Optional[Dict[str, Any]] = None
                                    ) -> None:
    """DD-SIMCA acceptance plot: T2/UCL vs Q/UCL per class model.

    Each panel = 1 one-class model. Unit square = acceptance region.
    Points colored by true class.

    If sens_esp is provided {class: (sens, spec, n_pure, n_adult, n_grupos,
    aviso)}, the title shows 'sens.=XX% (LOGO n=G) | spec.=YY%' (M2): sens =
    honest leave-one-group-out sensitivity (accepted held-out pure replicate
    groups), spec = rejected adulterated samples, n_grupos = number of
    independent pure replicate groups the LOGO estimate is based on.
    """
    classes = [c for c in scores.keys()]
    n_cls = len(classes)
    if n_cls == 0:
        return

    ncols = min(3, n_cls)
    nrows = int(np.ceil(n_cls / ncols))
    has_leg = n_cls > 1
    fig_w = 4.4 * ncols + (1.2 if has_leg else 0)
    fig_h = max(4.2 * nrows, 4.2)

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    total_cols = ncols + (1 if has_leg else 0)
    gs = fig.add_gridspec(nrows, total_cols,
                           width_ratios=[4.2] * ncols + ([1.1] if has_leg else []))

    rotulos = np.asarray(rotulos, dtype=str)
    all_classes = np.unique(rotulos)
    s_pt, alpha_pt, lw_pt = parametros_scatter_adaptativos(
        len(rotulos), len(all_classes))

    ax_leg_ref = None
    for k, cls in enumerate(classes):
        row, col = divmod(k, ncols)
        ax = fig.add_subplot(gs[row, col])
        m = scores[cls]
        t2n = np.asarray(m["T2_norm"])
        qn  = np.asarray(m["Q_norm"])

        # Log-log scale (Pomerantsev): clamp at small floor to avoid
        # log(0) and make acceptance region visible (lower-left corner).
        piso = 1e-2
        t2p = np.clip(t2n, piso, None)
        qp  = np.clip(qn,  piso, None)
        for true_cls in all_classes:
            idx = rotulos == true_cls
            ax.scatter(t2p[idx], qp[idx],
                       color=mapa_cores.get(true_cls, "0.5"),
                       s=s_pt, alpha=alpha_pt,
                       edgecolors="white", linewidths=lw_pt,
                       label=str(true_cls), zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")

        # Acceptance boundary at (1,1): lower-left quadrant accepted
        ax.axvline(1.0, color="0.20", ls="--", lw=1.2, zorder=4)
        ax.axhline(1.0, color="0.20", ls="--", lw=1.2, zorder=4)
        ax.axvspan(piso, 1.0, ymin=0, ymax=1, color=mapa_cores.get(cls, cor(0)),
                   alpha=0.0)  # placeholder to keep color in title

        # Title: uses sens/spec from one-class model if available (M2);
        # otherwise falls back to fraction of own class accepted.
        if sens_esp is not None and cls in sens_esp:
            _info = sens_esp[cls]
            sens_c, esp_c = _info[0], _info[1]
            n_grp = _info[4] if len(_info) > 4 else None
            s_txt = f"{sens_c*100:.0f}%" if sens_c == sens_c else "n/a"
            e_txt = f"{esp_c*100:.0f}%"  if esp_c == esp_c else "n/a"
            g_txt = f" (LOGO n={n_grp})" if n_grp is not None else ""
            titulo_painel = (f"Model: {cls}  sens.={s_txt}{g_txt} "
                             f"| spec.={e_txt}")
        else:
            idx_cls   = rotulos == cls
            n_cls_tot = int(idx_cls.sum())
            n_aceitos = int(np.sum((t2n[idx_cls] <= 1.0) & (qn[idx_cls] <= 1.0)))
            titulo_painel = f"Model: {cls}  sens.={n_aceitos/max(n_cls_tot,1):.0%}"

        lim_hi = max(float(np.percentile(np.concatenate([t2p, qp]), 99)) * 1.5,
                     3.0)
        ax.set_xlim(piso * 0.8, lim_hi)
        ax.set_ylim(piso * 0.8, lim_hi)
        ax.set_xlabel(r"$T^2$ / UCL($T^2$)  (log)", fontsize=8.5)
        ax.set_ylabel("$Q$ / UCL($Q$)  (log)", fontsize=8.5)
        ax.set_title(titulo_painel, loc="left",
                      fontsize=8.5, fontweight="bold")
        ax.text(0.98, 0.98,
                f"UCL($T^2$)={float(m['T2_ucl']):.1f}\n"
                f"UCL($Q$)={float(m['Q_ucl']):.2g}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color="0.35",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                           ec="0.82", lw=0.5))
        ax.grid(color="0.94", lw=0.5); ax.set_axisbelow(True)
        if ax_leg_ref is None:
            ax_leg_ref = ax

    if has_leg and ax_leg_ref is not None:
        ax_leg = fig.add_subplot(gs[:, ncols])
        _legenda_lateral(ax_leg, ax_leg_ref)

    salvar(fig, "fig_sprint3_ddsimca_acceptance", pasta, cfg)


def fig_ddsimca_individuais(scores: Dict[str, Dict[str, Any]],
                             rotulos: np.ndarray,
                             mapa_cores: Dict[str, str],
                             cfg: Config, pasta: str,
                             sens_esp: Optional[Dict[str, Any]] = None
                             ) -> None:
    """One DD-SIMCA acceptance plot per class, saved in pasta/ddsimca/.
    Log-log scale, lateral legend, UCL annotation — individual readable
    version (the 5x3 grid is too small for detailed inspection)."""
    rotulos = np.asarray(rotulos, dtype=str)
    all_classes = np.unique(rotulos)
    s_pt, alpha_pt, lw_pt = parametros_scatter_adaptativos(
        len(rotulos), len(all_classes))
    piso = 1e-2
    for cls in scores.keys():
        m = scores[cls]
        t2p = np.clip(np.asarray(m["T2_norm"]), piso, None)
        qp  = np.clip(np.asarray(m["Q_norm"]),  piso, None)
        fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[5.0, 1.2])
        ax = fig.add_subplot(gs[0]); ax_leg = fig.add_subplot(gs[1])
        for true_cls in all_classes:
            idx = rotulos == true_cls
            ax.scatter(t2p[idx], qp[idx], color=mapa_cores.get(true_cls, "0.5"),
                       s=s_pt, alpha=alpha_pt, edgecolors="white",
                       linewidths=lw_pt, label=str(true_cls), zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axvline(1.0, color="0.20", ls="--", lw=1.2, zorder=4)
        ax.axhline(1.0, color="0.20", ls="--", lw=1.2, zorder=4)
        lim_hi = max(float(np.percentile(np.concatenate([t2p, qp]), 99)) * 1.5,
                     3.0)
        ax.set_xlim(piso * 0.8, lim_hi); ax.set_ylim(piso * 0.8, lim_hi)
        if sens_esp is not None and cls in sens_esp:
            _info = sens_esp[cls]
            sc, ec = _info[0], _info[1]
            n_grp = _info[4] if len(_info) > 4 else None
            st = f"{sc*100:.0f}%" if sc == sc else "n/a"
            et = f"{ec*100:.0f}%" if ec == ec else "n/a"
            gt = f" (LOGO n={n_grp})" if n_grp is not None else ""
            tt = f"DD-SIMCA: {cls}  sens.={st}{gt} | spec.={et}"
        else:
            tt = f"DD-SIMCA: {cls}"
        ax.set_xlabel(r"$T^2$ / UCL($T^2$)  (log)")
        ax.set_ylabel("$Q$ / UCL($Q$)  (log)")
        ax.set_title(tt, loc="left", fontsize=9.5, fontweight="bold")
        ax.text(0.98, 0.98, f"UCL($T^2$)={float(m['T2_ucl']):.1f}\n"
                f"UCL($Q$)={float(m['Q_ucl']):.2g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color="0.35", bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                         ec="0.82", lw=0.5))
        ax.grid(color="0.94", lw=0.5); ax.set_axisbelow(True)
        _legenda_lateral(ax_leg, ax)
        nome_seguro = str(cls).replace(" ", "_").replace("/", "-")
        salvar(fig, f"ddsimca_{nome_seguro}", pasta, cfg, subpasta="ddsimca")


def fig_sprint3_opls_scores(t_pred: np.ndarray, t_orth: np.ndarray,
                             rotulos: np.ndarray,
                             mapa_cores: Dict[str, str],
                             n_ortho: int,
                             cfg: Config, pasta: str) -> None:
    """OPLS-DA scores: tp (predictive) vs to1 (1st orthogonal).

    tp axis: true separation between classes.
    to axis: systematic variation in X uncorrelated with Y
    (baseline, multiplicative scatter, optical path length).

    The plot decomposes total spectral variation into 'useful for
    discrimination' (tp) vs 'structured interference' (to), helping to
    interpret how pure the separation is and how much variation can be
    attributed to instrumental artefacts.
    """
    rotulos = np.asarray(rotulos, dtype=str)
    t_orth1 = t_orth[:, 0] if t_orth.ndim == 2 and t_orth.shape[1] > 0 \
              else np.asarray(t_orth).flatten()

    var_p = float(np.var(t_pred,  ddof=1)) if len(t_pred)  > 1 else 1.0
    var_o = float(np.var(t_orth1, ddof=1)) if len(t_orth1) > 1 else 1.0
    total = var_p + var_o + 1e-12
    pct_p = var_p / total * 100
    pct_o = var_o / total * 100

    fig = plt.figure(figsize=(8.5, 5.8), constrained_layout=True)
    gs  = fig.add_gridspec(1, 2, width_ratios=[5.0, 1.1])
    ax     = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])

    scores2d = np.column_stack([t_pred, t_orth1])
    plot_scores_panel(
        ax, scores2d, rotulos, mapa_cores,
        var_exp=[pct_p, pct_o],
        titulo=f"OPLS-DA — predictive × orthogonal ({n_ortho} orth. comp.)",
        xlabel="$t_p$ (predictive)",
        ylabel="$t_o$ (orthogonal 1)",
        desenhar_elipses=cfg.mostrar_elipses_grupo,
    )
    _legenda_lateral(ax_leg, ax)
    salvar(fig, "fig_sprint3_opls_scores", pasta, cfg)


# =========================================================================
#  Sprint v24 — Publication Figures
#  (PCA Loading Plot, ROC/AUC, OPLS-DA S-Plot, DD-SIMCA Cooman's Plot)
# =========================================================================

def fig_loadings_pca(pca, wavenumbers: np.ndarray, cfg: "Config",
                      pasta: str, n_pcs: int = 2) -> None:
    """PCA Loading Plot: spectral contribution of each variable per component.

    Bars in red (positive) and blue (negative). X axis inverted per
    NIR/FTIR convention (wavenumber decreasing from left to right).

    Ref: Bro & Smilde (2014) Anal. Methods 6:2812-2831.
    """
    n_comp = min(n_pcs, int(pca.n_components_))
    fig, axes = plt.subplots(n_comp, 1,
                              figsize=(12.0, 3.4 * n_comp),
                              constrained_layout=True)
    if n_comp == 1:
        axes = [axes]

    dx = float(abs(wavenumbers[1] - wavenumbers[0])) if len(wavenumbers) > 1 else 1.0

    for i, ax in enumerate(axes):
        loadings = pca.components_[i]
        var_exp  = float(pca.explained_variance_ratio_[i]) * 100
        cores_b  = [cor(0) if v >= 0 else cor(1) for v in loadings]
        ax.bar(wavenumbers, loadings, width=dx * 0.9,
               color=cores_b, alpha=0.80, edgecolor="none")
        ax.axhline(0, color="0.45", lw=0.7, ls="--")
        ax.set_xlabel("Número de onda (cm$^{-1}$)")
        ax.set_ylabel(f"Loading PC{i + 1}")
        ax.set_title(f"PC{i + 1} — {var_exp:.1f}% da variância explicada  "
                     "| vermelho = positivo · azul = negativo",
                     loc="left", fontsize=9)
        # Convencao NIR: wavenumbers crescentes no array -> inverte eixo X
        if len(wavenumbers) > 1 and wavenumbers[0] < wavenumbers[-1]:
            ax.invert_xaxis()

    fig.suptitle("Loading Plot PCA — contribuição espectral por componente",
                  fontsize=10, fontweight="bold")
    salvar(fig, "fig_loadings_pca", pasta, cfg)


def _escala_vetores_biplot(scores2: np.ndarray, loadings: np.ndarray,
                            frac: float = 0.8) -> float:
    """Fator de escala UNICO p/ desenhar vetores de loading sobre scores,
    tal que NENHUM vetor (em nenhum dos 2 eixos) ultrapasse `frac` do maior
    score visivel NAQUELE eixo especifico.

    Extraida como funcao pura (testavel sem renderizar figura) apos um bug
    real: usar um unico fator calibrado pelo maior score CONJUNTO (max sobre
    as 2 colunas juntas) deixava vetores com componente forte no eixo de
    menor alcance (ex.: PC2, quando PC1 domina a variancia) desenhados MUITO
    alem da area visivel daquele eixo -- rotulos apareciam flutuando fora do
    grafico. Calibrando por eixo e usando o mais restritivo, todo vetor cabe
    dentro do alcance visivel dos DOIS eixos, preservando um fator uniforme
    (o angulo entre vetores continua interpretavel).
    """
    max_score_x = float(np.abs(scores2[:, 0]).max()) if scores2.size else 1.0
    max_score_y = float(np.abs(scores2[:, 1]).max()) if scores2.size else 1.0
    max_load_x  = float(np.abs(loadings[:, 0]).max()) if loadings.size else 1e-12
    max_load_y  = float(np.abs(loadings[:, 1]).max()) if loadings.size else 1e-12
    escala_x = frac * max_score_x / max(max_load_x, 1e-12)
    escala_y = frac * max_score_y / max(max_load_y, 1e-12)
    return min(escala_x, escala_y)


def fig_biplot_pca(pca, scores_pca: np.ndarray, wavenumbers: np.ndarray,
                    rotulos, mapa_cores, cfg: "Config", pasta: str,
                    n_vars_destacadas: int = 12) -> None:
    """Biplot PCA classico: scores (PC1 x PC2, por amostra) + vetores de
    loading das variaveis mais influentes SOBREPOSTOS no mesmo painel --
    interpretacao quimica direta de quais regioes espectrais empurram cada
    grupo de amostras na direcao observada (item 4 da lista de figuras que
    faltavam, CLAUDE.md secao 5). Como um espectro tem centenas/milhares
    de variaveis, mostra so' as `n_vars_destacadas` de maior magnitude
    conjunta em PC1/PC2 -- senao o painel vira um emaranhado ilegivel
    (pratica padrao em biplots de dados de alta dimensao).

    Ref: Bro & Smilde (2014) Anal. Methods 6:2812-2831 (mesma referencia
    de fig_loadings_pca).
    """
    rotulos = np.asarray(rotulos, dtype=str)
    loadings = np.asarray(pca.components_[:2]).T   # (p, 2)
    scores2  = np.asarray(scores_pca)[:, :2]

    fig = plt.figure(figsize=(9.5, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[6.0, 1.3])
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])

    for cls in np.unique(rotulos):
        idx = rotulos == cls
        ax.scatter(scores2[idx, 0], scores2[idx, 1], s=32, alpha=0.65,
                   color=mapa_cores.get(cls, "0.4"), edgecolors="white",
                   linewidths=0.4, label=str(cls), zorder=2)

    escala = _escala_vetores_biplot(scores2, loadings)
    mag = np.sqrt((loadings ** 2).sum(axis=1))
    idx_top = np.argsort(mag)[::-1][:min(n_vars_destacadas, len(mag))]

    for i in idx_top:
        vx, vy = loadings[i, 0] * escala, loadings[i, 1] * escala
        ax.annotate("", xy=(vx, vy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.1,
                                    shrinkA=0, shrinkB=0), zorder=4)
        ax.text(vx * 1.08, vy * 1.08, f"{wavenumbers[i]:.0f}",
               fontsize=7, color="0.15", ha="center", va="center",
               fontweight="bold", zorder=5)

    ax.axhline(0, color="0.75", lw=0.5, ls=":")
    ax.axvline(0, color="0.75", lw=0.5, ls=":")
    var1 = float(pca.explained_variance_ratio_[0]) * 100
    var2 = float(pca.explained_variance_ratio_[1]) * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var2:.1f}%)")
    ax.set_title(
        f"Biplot PCA — scores + top-{len(idx_top)} loadings "
        f"(número de onda, cm$^{{-1}}$)", loc="left")
    ax.grid(color="0.94", lw=0.5); ax.set_axisbelow(True)
    _legenda_lateral(ax_leg, ax)

    salvar(fig, "fig_biplot_pca", pasta, cfg)


def fig_roc_auc(Y_bin: np.ndarray, Y_cv: np.ndarray,
                classes: np.ndarray, cfg: "Config",
                pasta: str) -> Dict[str, float]:
    """Curvas ROC multiclasse One-vs-Rest usando predicoes CV group-aware.

    Os scores continuos do PLS-DA (Y_cv) sao usados como funcao discriminante
    — sem re-calibracao, sem leakage. AUC macro calculado por roc_auc_score.

    Retorna dict {classe: AUC, 'macro': AUC_macro}.

    Ref: Fawcett (2006) Pattern Recognit. Lett. 27:861-874.
         Hand & Till (2001) Machine Learning 45:171-186.
    """
    from sklearn.metrics import auc as sk_auc

    n_cls = Y_bin.shape[1]
    if n_cls < 2:
        return {}

    aucs: Dict[str, float] = {}
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)

    for i, cls in enumerate(classes):
        y_true_i  = Y_bin[:, i]
        y_score_i = Y_cv[:, i]
        if y_true_i.sum() == 0 or y_true_i.sum() == len(y_true_i):
            continue
        fpr, tpr, _ = roc_curve(y_true_i, y_score_i)
        auc_i        = float(sk_auc(fpr, tpr))
        aucs[str(cls)] = auc_i
        ax.plot(fpr, tpr, color=cor(i), lw=1.4,
                label=f"{cls}  (AUC = {auc_i:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Aleatório (AUC = 0.500)")
    ax.set_xlabel("Taxa de Falso Positivo  (1 – Especificidade)")
    ax.set_ylabel("Taxa de Verdadeiro Positivo  (Sensibilidade)")
    ax.set_xlim((-0.01, 1.01)); ax.set_ylim((-0.01, 1.02))

    try:
        macro_auc = float(roc_auc_score(
            Y_bin, Y_cv, average="macro", multi_class="ovr"))
        aucs["macro"] = macro_auc
        ax.set_title(
            f"Curvas ROC — PLS-DA (CV group-aware) | AUC macro = {macro_auc:.3f}",
            loc="left")
    except ValueError:
        # AUC macro degenerado (ex.: classe ausente em Y_bin/Y_cv) -- titulo
        # so' omite o numero, nao mostra um AUC inventado; aucs["macro"]
        # fica ausente do dict (o chamador ja trata a chave como opcional).
        ax.set_title("Curvas ROC — PLS-DA (CV group-aware)", loc="left")

    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax.grid(color="0.94", lw=0.5); ax.set_axisbelow(True)
    salvar(fig, "fig_roc_auc_multiclasse", pasta, cfg)
    return aucs


def fig_splot_opls(X_proc: np.ndarray, t_pred: np.ndarray,
                   wavenumbers: np.ndarray, cfg: "Config",
                   pasta: str, top_n: int = 15) -> None:
    """S-Plot OPLS-DA: covariancia x correlacao de cada variavel com t_pred.

    Superior-direito: variaveis discriminantes positivas (classe 1 > classe 0).
    Inferior-esquerdo: discriminantes negativas.
    Centro (~0, ~0): ruido espectral sem poder discriminante.

    Anotacao automatica dos top_n wavenumbers por |correlacao|.

    Ref: Bylesjo M. et al. (2006) J. Chemometrics 20:341-351.
         Wiklund S. et al. (2008) Anal. Chem. 80:115-122.
    """
    n = X_proc.shape[0]
    X_c    = X_proc - X_proc.mean(axis=0)
    t_c    = t_pred - float(t_pred.mean())
    cov_xj = (X_c * t_c[:, None]).sum(axis=0) / max(n - 1, 1)

    std_x = X_proc.std(axis=0, ddof=1)
    std_t = float(t_pred.std(ddof=1))
    std_x[std_x < 1e-12] = 1.0
    if std_t < 1e-12:
        std_t = 1.0
    corr_xj = np.clip(cov_xj / (std_x * std_t), -1.0, 1.0)

    fig, ax = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)
    sc = ax.scatter(cov_xj, corr_xj,
                    c=corr_xj, cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                    s=10, alpha=0.75, edgecolors="none", zorder=3)
    plt.colorbar(sc, ax=ax, label="Correlação($X_j$, $t_p$)",
                 fraction=0.035, pad=0.02)

    ax.axhline(0, color="0.55", lw=0.8, ls="--")
    ax.axvline(0, color="0.55", lw=0.8, ls="--")

    # Anota top wavenumbers por |correlacao|, com anti-sobreposicao
    # (evita o empilhamento de rotulos quando varias variaveis-top se
    # agrupam na mesma regiao do grafico).
    idx_top = np.argsort(np.abs(corr_xj))[::-1][:max(top_n * 3, 30)]
    _anotar_sem_sobreposicao(
        ax, cov_xj[idx_top], corr_xj[idx_top],
        [f"{wavenumbers[j]:.0f}" for j in idx_top],
        ref_x=cov_xj, ref_y=corr_xj,
        fontsize=6.5, color="0.2", min_dist_frac=0.055, max_labels=top_n)

    ax.set_xlabel("Covariância($X_j$, $t_p$)")
    ax.set_ylabel("Correlação($X_j$, $t_p$)")
    ax.set_title(
        "S-Plot OPLS-DA — quadrante sup-dir: discriminantes positivos; "
        "inf-esq: negativos; centro: ruído",
        loc="left", fontsize=9)
    ax.grid(color="0.95", lw=0.5); ax.set_axisbelow(True)
    salvar(fig, "fig_splot_opls", pasta, cfg)


def fig_heatmap_especie_adulterante(resultado: Dict[str, Any], cfg,
                                     pasta: str) -> None:
    """Heatmap R2cv por especie (linhas) x adulterante (colunas).

    Cor = R2cv numa escala divergente ancorada no limiar de aceite. Celula
    ABAIXO do limiar fica HACHURADA e com o valor em negrito (nunca some);
    celula sem dados/replicas suficientes vira cinza com 'n/a'. O titulo tras
    o contador de falhas (ex.: '16/37 combinacoes abaixo de R²cv = 0.70'),
    o mesmo numero registrado no relatorio -- para que uma quantificacao que
    so funciona em parte das combinacoes nao seja lida como sucesso geral.
    """
    especies     = resultado["especies"]
    adulterantes = resultado["adulterantes"]
    matriz       = resultado["matriz"]
    limiar       = float(resultado["limiar_r2"])
    n_falhas     = int(resultado["n_falhas"])
    n_total      = int(resultado["n_total"])

    n_lin, n_col = len(especies), len(adulterantes)
    M = np.full((n_lin, n_col), np.nan)
    for i, esp in enumerate(especies):
        for j, ad in enumerate(adulterantes):
            M[i, j] = matriz.get((esp, ad), np.nan)

    fig, ax = plt.subplots(figsize=(max(4.5, 1.6 * n_col + 2.5),
                                    max(3.0, 0.55 * n_lin + 1.6)))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="0.85")                 # n/a -> cinza explicito
    vmin = min(0.0, 2.0 * limiar - 1.0)        # escala divergente centra no limiar
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=vmin, vmax=1.0,
                   aspect="auto")

    for i in range(n_lin):
        for j in range(n_col):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=8, color="0.35")
                continue
            falha = v < limiar
            if falha:                          # hachura marca a falha
                ax.add_patch(mpl.patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                    edgecolor="0.12", lw=0.0, zorder=3))
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if falha else "normal", color="0.10")

    ax.set_xticks(range(n_col)); ax.set_xticklabels(adulterantes, fontsize=9)
    ax.set_yticks(range(n_lin)); ax.set_yticklabels(especies, fontsize=9)
    ax.set_xlabel("Adulterante"); ax.set_ylabel("Especie")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"R²cv  (limiar de aceite = {limiar:.2f})")
    ax.set_title(
        f"Quantificacao especie x adulterante — {n_falhas}/{n_total} "
        f"combinacoes abaixo de R²cv = {limiar:.2f}",
        fontsize=10, fontweight="bold", loc="left")
    fig.tight_layout()
    salvar(fig, "figN3_heatmap_especie_adulterante", pasta, cfg)


def fig_cooman_ddsimca(ddsimca_res: Dict[str, Dict[str, Any]],
                        rotulos: np.ndarray,
                        mapa_cores: Dict[str, str],
                        cfg: "Config", pasta: str,
                        max_pares: int = 6) -> None:
    """Cooman's Plot: distancia normalizada ao modelo de classe A vs classe B.

    Regioes do plano (por par A x B):
      sqrt(dQ_A)<=1 e sqrt(dQ_B)<=1  -> Ambiguo (aceito por ambos)
      sqrt(dQ_A)<=1 e sqrt(dQ_B)>1   -> pertence a A
      sqrt(dQ_A)>1  e sqrt(dQ_B)<=1  -> pertence a B
      sqrt(dQ_A)>1  e sqrt(dQ_B)>1   -> Desconhecido

    Escala raiz-quadrada: preserva estrutura proxima ao UCL sem comprimir
    o eixo para amostras muito distantes (adulteracoes altas).

    Ref: Rodionova & Pomerantsev (2020) Chemom. Intell. Lab. Syst. 200:103958.
    """
    classes_dd = sorted(ddsimca_res.keys())
    pares = [(classes_dd[i], classes_dd[j])
             for i in range(len(classes_dd))
             for j in range(i + 1, len(classes_dd))]
    if not pares:
        return

    pares = pares[:max_pares]
    n_p   = len(pares)
    ncols = min(3, n_p)
    nrows = (n_p + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.2 * ncols, 4.5 * nrows),
                              constrained_layout=True)
    if n_p == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (clsA, clsB) in enumerate(pares):
        ax  = axes[idx // ncols, idx % ncols]
        qA  = np.sqrt(np.clip(np.asarray(ddsimca_res[clsA]["Q_norm"]), 0, None))
        qB  = np.sqrt(np.clip(np.asarray(ddsimca_res[clsB]["Q_norm"]), 0, None))

        for cls in sorted(set(rotulos)):
            mask = rotulos == cls
            ax.scatter(qA[mask], qB[mask],
                       color=mapa_cores.get(cls, "#999999"),
                       s=20, alpha=0.80, label=cls,
                       edgecolors="none", zorder=3)

        ax.axhline(1.0, color="black", lw=0.9, ls="--")
        ax.axvline(1.0, color="black", lw=0.9, ls="--")
        ax.text(0.03, 0.97, clsA, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color="0.35")
        ax.text(0.97, 0.03, clsB, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="0.35")
        ax.set_xlabel(f"$\\sqrt{{d_Q}}$ — modelo {clsA}")
        ax.set_ylabel(f"$\\sqrt{{d_Q}}$ — modelo {clsB}")
        ax.set_title(f"{clsA} × {clsB}", fontsize=9)
        ax.grid(color="0.94", lw=0.5); ax.set_axisbelow(True)

    for idx_extra in range(n_p, nrows * ncols):
        axes[idx_extra // ncols, idx_extra % ncols].set_visible(False)

    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_leg, loc="lower center",
                   ncol=min(len(handles), 5), frameon=False,
                   fontsize=8, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle("Cooman's Plot — DD-SIMCA (escala $\\sqrt{d_Q}$)",
                  fontsize=11, fontweight="bold")
    salvar(fig, "fig_cooman_ddsimca", pasta, cfg, subpasta="ddsimca")
