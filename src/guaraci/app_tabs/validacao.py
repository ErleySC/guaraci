"""app_tabs/validacao.py — Aba 5 (Validation): figuras e métricas da última
execução. Extraído de app_quimiometria.py (item 18).
"""
from __future__ import annotations

import os
import re as _re_acc
from typing import Callable, Dict

import pandas as pd
import streamlit as st

_CATS = {
    "All":              "",
    "PCA":              "pca",
    "PLS-DA scores":    "plsda",
    "Outliers (T²/Q)":  "outlier",
    "Confusion":        "confus",
    "ROC / AUC":        "roc",
    "VIP / SR":         "vip",
    "Loading":          "loading",
    "HCA":              "hca",
    "OPLS-DA":          "opls",
    "DD-SIMCA":         "ddsimca",
    "Cooman's Plot":    "cooman",
    "S-Plot":           "splot",
    "Permutation":      "permut",
    "Wold":             "wold",
    "Benchmark":        "benchmark",
    "Monte Carlo CV":   "monte_carlo",
    "DET curves":       "fig_det",
    "SHAP":             "fig_shap",
}


def render(T: Callable[[str], str], tok: Callable[[], Dict[str, str]],
           ler_resumo: Callable[[str], str], listar_figuras: Callable[[str], list]) -> None:
    """Renderiza a aba Validation. `ler_resumo`/`listar_figuras` são as
    versões cacheadas (@st.cache_data) definidas em app_quimiometria.py."""
    st.subheader(T("Validation Results"))
    pasta_v = st.session_state.get("ultima_pasta")

    if not pasta_v or not os.path.isdir(pasta_v):
        st.info(T("No results yet. Run the pipeline in the Model tab."))
        return

    st.caption(f"Folder: `{os.path.abspath(pasta_v)}`")

    # Numeric summary
    resumo_txt = ler_resumo(pasta_v)
    if resumo_txt:
        with st.expander("📋 Model summary (resumo_modelo.txt)", expanded=True):
            # Split into main section and notes section for cleaner display
            if "Methodological Notes" in resumo_txt:
                parts = resumo_txt.split("---")
                main_part = parts[0].strip() if parts else resumo_txt
                notes_part = resumo_txt[resumo_txt.find("Methodological Notes"):].strip() if "Methodological Notes" in resumo_txt else ""
                st.code(main_part, language="text")
                if notes_part:
                    with st.expander("📋 Methodological Notes (for peer review)", expanded=False):
                        st.code(notes_part, language="text")
            else:
                st.code(resumo_txt, language="text")

    # ── Per-Class Accuracy table (extracted from summary) ───────────────
    if resumo_txt:
        acc_map: Dict[str, float] = {}
        for _linha in resumo_txt.splitlines():
            _m = _re_acc.match(r"\s*Acc\s+(.+?)\s*[:=]\s*([\d.]+)", _linha)
            if _m:
                acc_map[_m.group(1).strip()] = float(_m.group(2))
        if acc_map:
            with st.expander("📊 Accuracy by Class", expanded=True):
                _df_acc = pd.DataFrame(
                    list(acc_map.items()), columns=["Class", "Accuracy (recall)"]
                ).sort_values("Accuracy (recall)")
                # Color: red < 0.7, yellow 0.7-0.9, green >= 0.9
                # Cores vindas dos design tokens do tema ativo → contraste
                # correto em claro E escuro (não mais hex claros fixos).
                _tkc = tok()
                def _cor_acc(v: object) -> str | None:  # Scalar compat
                    try:
                        fv = float(v)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        return None
                    if fv >= 0.90:
                        return (f"background-color:{_tkc['success_bg']};"
                                f"color:{_tkc['success']}")
                    if fv >= 0.70:
                        return (f"background-color:{_tkc['warn_bg']};"
                                f"color:{_tkc['warn']}")
                    return (f"background-color:{_tkc['error_bg']};"
                            f"color:{_tkc['error']}")
                st.dataframe(
                    _df_acc.style.map(
                        _cor_acc, subset=["Accuracy (recall)"]),
                    use_container_width=True, height=min(400, 35 * len(_df_acc) + 38))

    # ── Benchmark and MC CV tables (if they exist) ───────────────────
    _bench_csv_v = os.path.join(pasta_v, "dados",
                                "benchmark_classificadores.csv")
    _mc_csv_v    = os.path.join(pasta_v, "dados", "monte_carlo_cv.csv")
    if os.path.exists(_bench_csv_v) or os.path.exists(_mc_csv_v):
        with st.expander("🏅 Classifier Benchmark", expanded=False):
            if os.path.exists(_bench_csv_v):
                try:
                    _df_b = pd.read_csv(_bench_csv_v, sep=";", decimal=",")
                    st.markdown("**Auto-Benchmark — Balanced Accuracy by model (GroupKFold)**")
                    st.dataframe(_df_b, use_container_width=True)
                except Exception as _e_b:
                    st.warning(f"Error reading benchmark CSV: {_e_b}")
            if os.path.exists(_mc_csv_v):
                try:
                    _df_mc = pd.read_csv(_mc_csv_v, sep=";", decimal=",")
                    st.markdown("**Monte Carlo CV — 95% CI by percentile**")
                    st.dataframe(_df_mc, use_container_width=True)
                except Exception as _e_mc:
                    st.warning(f"Error reading MC CV CSV: {_e_mc}")

    # Gallery filtered by figure category
    imgs_v = listar_figuras(pasta_v)
    if imgs_v:
        st.markdown(f"**{len(imgs_v)} figures generated**")
        filtro_v = st.selectbox("Filter by category",
                                list(_CATS.keys()), key="filtro_valid")
        token_v  = _CATS[filtro_v].lower()
        imgs_filt = [im for im in imgs_v
                     if token_v in os.path.basename(im).lower()] \
                    if token_v else imgs_v

        if imgs_filt:
            cols_v = st.columns(2)
            for j, img in enumerate(imgs_filt):
                with cols_v[j % 2]:
                    st.image(img, caption=os.path.basename(img),
                             use_container_width=True)
        else:
            st.info(f"No figures found for '{filtro_v}'.")
    else:
        st.info("Figures saved in non-displayable format (PDF/SVG). "
                "Use the download button in the Reports tab.")
