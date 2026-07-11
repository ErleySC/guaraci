"""app_tabs/preprocessamento.py — Aba 3 (Preprocessing): preset espectral +
prévia antes/depois. Extraído de app_quimiometria.py (item 18).
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from guaraci.spectra_preview import preview_espectros_dx, preview_espectros_csv, plot_espectros_media
from guaraci.app_logic import coletar_config

_PRESET_INFO = {
    "MSC+SG+MC":      "MSC (scatter correction) → 1st derivative SG (Savitzky-Golay) → Mean-Centering. **Best for FT-NIR with pronounced scatter.** Acc=0.923 on 1807 Amazonian oil samples.",
    "SNV+SG+MC":      "SNV (variance normalization) → SG → Mean-Centering. Robust alternative to MSC when global reference is not stable.",
    "Autoscaling":    "Mean-Centering + division by standard deviation. **Caution**: collapses spectral noise when SG is not applied first.",
    "Mean-centering": "Mean centering only. Recommended as a comparative baseline.",
}


def render(pq, cfg_base, specs: Dict, valores: Dict,
           widget_para_campo: Callable) -> None:
    """Renderiza a aba Preprocessing. `valores` é o dict compartilhado com
    Data/Model (mesmo objeto, mutado em sequência)."""
    st.subheader("Spectral Preprocessing")

    _PREPROC_KEYS = ["pre_processamento", "faixa_min_cm", "faixa_max_cm"]

    col_p1, col_p2 = st.columns(2)
    for i, k in enumerate(_PREPROC_KEYS):
        s = specs.get(k)
        if s is None:
            continue
        with (col_p1 if i % 2 == 0 else col_p2):
            valores[k] = widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    # Information about each preset
    preset_selecionado = valores.get("pre_processamento", "")
    if preset_selecionado in _PRESET_INFO:
        st.info(_PRESET_INFO[preset_selecionado])

    # ---- Before/after preview ---------------------------------------------
    st.divider()
    st.markdown("**Before / after preprocessing visualization**")
    cfg_pp, _ = coletar_config(cfg_base, valores)
    ok_pp, _ = pq._validar_pasta_dados(cfg_pp)

    if not ok_pp:
        st.info("Configure and validate data input (Data tab) to enable the preview.")
    elif st.button("⚗️ Generate before/after preview", key="btn_prev_preproc"):
        with st.spinner("Loading and processing spectra..."):
            modo_pp = cfg_pp.modo
            wn_mn_pp = float(cfg_pp.wn_min)
            wn_mx_pp = float(cfg_pp.wn_max)
            if modo_pp == "dx":
                wn_raw, X_raw, labs_raw = preview_espectros_dx(
                    cfg_pp.pasta_entrada, wn_mn_pp, wn_mx_pp)
            elif modo_pp == "csv":
                csv_cam_pp = st.session_state.get("_csv_upload_path",
                                                   cfg_pp.arquivo_csv)
                wn_raw, X_raw, labs_raw = preview_espectros_csv(
                    csv_cam_pp, cfg_pp.coluna_classe, wn_mn_pp, wn_mx_pp)
            else:
                wn_raw, X_raw, labs_raw = None, None, None

        if wn_raw is not None and X_raw is not None:
            try:
                preproc_pp = pq.construir_preprocessador(cfg_pp)
                preproc_pp.fit(X_raw)
                X_proc_pp = preproc_pp.transform(X_raw)
                labs_raw_arr = np.asarray(labs_raw)
                fig_antes = plot_espectros_media(
                    wn_raw, X_raw, labs_raw_arr, "Before preprocessing")
                fig_depois = plot_espectros_media(
                    wn_raw, X_proc_pp, labs_raw_arr,
                    f"After: {preset_selecionado}")
                col_ant, col_dep = st.columns(2)
                with col_ant:
                    st.pyplot(fig_antes, use_container_width=True)
                    plt.close(fig_antes)
                with col_dep:
                    st.pyplot(fig_depois, use_container_width=True)
                    plt.close(fig_depois)
            except Exception as e_pp:
                st.error(f"Error applying preprocessing: {e_pp}")
        else:
            st.warning("Could not load spectra. Check the Data tab.")
