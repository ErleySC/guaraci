"""app_tabs/preprocessamento.py — Tela (Preprocessing): preset espectral +
prévia antes/depois. Extraído de app_quimiometria.py (item 18).
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from guaraci import web_seguranca as _ws

from guaraci.spectra_preview import preview_spectra_dx, preview_spectra_csv, plot_mean_spectra
from guaraci.app_logic import collect_config

_PRESET_INFO = {
    "MSC+SG+MC":      "MSC (scatter correction) → 1st derivative SG (Savitzky-Golay) → Mean-Centering. **Best for FT-NIR with pronounced scatter.** Compare presets on your own data by turning on 'Compare preprocessing' in the Model tab.",
    "SNV+SG+MC":      "SNV (variance normalization) → SG → Mean-Centering. Robust alternative to MSC when global reference is not stable.",
    "Autoscaling":    "Mean-Centering + division by standard deviation. **Caution**: collapses spectral noise when SG is not applied first.",
    "Mean-centering": "Mean centering only. Recommended as a comparative baseline.",
}


#: Campos desta aba (nivel de modulo: a aba Modelo usa para achar os que sobram).
_PREPROC_KEYS = ["pre_processamento", "faixa_min_cm", "faixa_max_cm", "normalizacao", "ims_rip_janela"]


def render(pq, cfg_base, specs: Dict, valores: Dict,
           widget_para_campo: Callable,
           T: Callable[[str], str] = lambda s: s) -> None:
    """Renderiza a aba Preprocessing. `valores` é o dict compartilhado com
    Data/Model (mesmo objeto, mutado em sequência).

    `T`: traducao PT/EN (mesma funcao `_T` de app_quimiometria.py). Default
    no-op so' para chamada direta em teste isolado.
    """
    st.subheader(T("Spectral Preprocessing"))
    st.caption(T("⚗️ Choose the spectral preprocessing preset and preview "
               "before/after → then go to **Model** tab."))

    col_p1, col_p2 = st.columns(2)
    for i, k in enumerate(_PREPROC_KEYS):
        s = specs[k]
        with (col_p1 if i % 2 == 0 else col_p2):
            valores[k] = widget_para_campo(s, pq.attr_para_yaml(s, cfg_base))

    # Information about each preset
    preset_selecionado = valores.get("pre_processamento", "")
    if preset_selecionado in _PRESET_INFO:
        st.info(T(_PRESET_INFO[preset_selecionado]))

    # ---- Before/after preview ---------------------------------------------
    st.divider()
    st.markdown(T("**Before / after preprocessing visualization**"))
    cfg_pp, _ = collect_config(cfg_base, valores)
    ok_pp, _ = pq.validar_fonte_de_dados(cfg_pp)

    if not ok_pp:
        st.info(T("Configure and validate data input (Data tab) to enable the preview."))
    elif st.button(T("⚗️ Generate before/after preview"), key="btn_prev_preproc"):
        with st.spinner(T("Loading and processing spectra...")):
            modo_pp = cfg_pp.mode
            wn_mn_pp = float(cfg_pp.wn_min)
            wn_mx_pp = float(cfg_pp.wn_max)
            if modo_pp == "dx":
                wn_raw, X_raw, labs_raw = preview_spectra_dx(
                    cfg_pp.input_folder, wn_mn_pp, wn_mx_pp)
            elif modo_pp == "csv":
                csv_cam_pp = st.session_state.get("_csv_upload_path",
                                                   cfg_pp.csv_file)
                wn_raw, X_raw, labs_raw = preview_spectra_csv(
                    csv_cam_pp, cfg_pp.class_column, wn_mn_pp, wn_mx_pp)
            elif modo_pp == "sintetico":
                # R5-01: no modo padrao (sintetico) a previa dava so' aviso de falha.
                try:
                    from guaraci.io_registry import get_reader
                    _wn_s, _X_s, _rot_s, *_ = get_reader("sintetico")(cfg_pp)
                    _m = (_wn_s >= wn_mn_pp) & (_wn_s <= wn_mx_pp)
                    wn_raw, X_raw, labs_raw = _wn_s[_m], _X_s[:50][:, _m], _rot_s[:50]
                except Exception:  # noqa: BLE001 -- previa best-effort
                    wn_raw, X_raw, labs_raw = None, None, None
            else:
                wn_raw, X_raw, labs_raw = None, None, None

        if wn_raw is not None and X_raw is not None:
            try:
                preproc_pp = pq.build_preprocessor(cfg_pp)
                preproc_pp.fit(X_raw)
                X_proc_pp = preproc_pp.transform(X_raw)
                labs_raw_arr = np.asarray(labs_raw)
                fig_antes = plot_mean_spectra(
                    wn_raw, X_raw, labs_raw_arr, T("Before preprocessing"))
                fig_depois = plot_mean_spectra(
                    wn_raw, X_proc_pp, labs_raw_arr,
                    T("After: {preset}").format(preset=preset_selecionado))
                col_ant, col_dep = st.columns(2)
                with col_ant:
                    st.pyplot(fig_antes, use_container_width=True)
                    plt.close(fig_antes)
                with col_dep:
                    st.pyplot(fig_depois, use_container_width=True)
                    plt.close(fig_depois)
            except Exception as e_pp:  # noqa: BLE001 -- previa ao vivo
                # (multi-etapa: fit+transform+2 figuras); erro exibido ao
                # usuario, nao afeta o pipeline real (so' esta previa).
                st.error(T("Error applying preprocessing: {e}").format(e=_ws.msg_segura(e_pp)))
        else:
            st.warning(T("Could not load spectra. Check the Data tab."))
