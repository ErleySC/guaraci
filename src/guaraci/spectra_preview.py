"""spectra_preview.py — Carregamento/plotagem de amostra de espectros para
prévia na UI web (abas Data e Preprocessing).

Extraído de app_quimiometria.py (item 18 da auditoria): usado por duas abas,
então vira módulo de serviço próprio em vez de duplicado ou passado por
parâmetro entre elas. `preview_espectros_dx`/`preview_espectros_csv` mantêm
o cache do Streamlit (`st.cache_data`) — não são puras no sentido de
app_logic.py, mas não fazem I/O de widget algum, só leitura/parsing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import guaraci.pipeline as pq


@st.cache_data(show_spinner=False, ttl=120)
def preview_espectros_dx(pasta: str, wn_min: float, wn_max: float,
                          max_por_classe: int = 5):
    """Loads up to max_por_classe samples per subfolder for visualization."""
    try:
        subpastas = sorted(
            p for p in Path(pasta).iterdir()
            if p.is_dir() and list(p.glob("*.dx"))
        )
        if not subpastas:
            # flat folder with .dx files
            subpastas = [Path(pasta)]
        wn_ref, specs, labs = None, [], []
        for sp in subpastas:
            arqs = sorted(sp.glob("*.dx"))[:max_por_classe]
            for arq in arqs:
                try:
                    wn_a, sp_a = pq.parse_dx(str(arq))
                    mask = (wn_a >= wn_min) & (wn_a <= wn_max)
                    wn_a, sp_a = wn_a[mask], sp_a[mask]
                    if wn_ref is None:
                        wn_ref = wn_a
                    else:
                        # np.interp replaces deprecated scipy.interpolate.interp1d
                        sp_a = np.interp(wn_ref, wn_a, sp_a)
                    specs.append(sp_a)
                    labs.append(sp.name)
                except Exception:
                    continue
        if not specs or wn_ref is None:
            return None, None, None
        return wn_ref, np.array(specs), np.array(labs)
    except Exception:
        return None, None, None


@st.cache_data(show_spinner=False, ttl=120)
def preview_espectros_csv(caminho: str, col_cls: str,
                          wn_min: float, wn_max: float,
                          max_n: int = 50):
    """Loads up to max_n CSV rows for visualization."""
    try:
        df = pd.read_csv(caminho, sep=None, engine="python", nrows=max_n)
        meta = {col_cls}
        num_cols = [c for c in df.columns if c not in meta]
        try:
            wn = np.array([float(c) for c in num_cols])
        except ValueError:
            return None, None, None
        mask = (wn >= wn_min) & (wn <= wn_max)
        X = df[num_cols].values[:, mask].astype(float)
        labs = df[col_cls].astype(str).values if col_cls in df.columns else \
               np.array(["?"] * len(df))
        return wn[mask], X, labs
    except Exception:
        return None, None, None


def plot_espectros_media(wn: np.ndarray, X: np.ndarray,
                          rotulos: np.ndarray, titulo: str = ""):
    """Plots mean ± std per class."""
    classes = np.unique(np.asarray(rotulos))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    for i, cls in enumerate(classes[:10]):
        cor = cmap(i / 10)
        mask = rotulos == cls
        med = X[mask].mean(axis=0)
        std = X[mask].std(axis=0)
        ax.plot(wn, med, color=cor, lw=1.3, label=f"{cls} (n={mask.sum()})")
        ax.fill_between(wn, med - std, med + std, color=cor, alpha=0.15)
    if len(wn) > 1 and wn[0] > wn[-1]:
        ax.invert_xaxis()
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorbance")
    if titulo:
        ax.set_title(titulo, fontsize=9)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(color="0.93", lw=0.5)
    return fig


__all__ = ["preview_espectros_dx", "preview_espectros_csv", "plot_espectros_media"]
