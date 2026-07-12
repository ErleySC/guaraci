"""app_tabs/dados.py — Aba 2 (Data): entrada de dados (.dx, CSV local/upload)
+ prévia dos espectros. Extraído de app_quimiometria.py (item 18).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from guaraci.spectra_preview import preview_espectros_dx, preview_espectros_csv, plot_espectros_media
from guaraci.app_logic import coletar_config


def render(pq, cfg_base, specs: Dict, valores: Dict,
           widget_para_campo: Callable, cfg_path: str) -> None:
    """Renderiza a aba Data.

    `valores` é o dict COMPARTILHADO entre Data/Preprocessing/Model (o mesmo
    objeto é mutado por cada aba — mesma semântica de antes da extração,
    já que o Config final só é montado depois que as 3 abas rodaram).
    """
    st.subheader("Data Input")
    st.caption("📂 Step 2: Upload or select spectra folder → then go to **Model** tab.")

    # ---- CSV Upload (at top for easy access) -----------------------------
    st.markdown("**Upload CSV** *(alternative to the local path below)*")
    upld = st.file_uploader(
        "Drag or select a CSV file",
        type=["csv", "txt"],
        key="csv_upload_widget",
        help="The file will be saved to a temporary folder and the path adjusted automatically.",
    )
    if upld is not None:
        tmp_dir = Path(tempfile.gettempdir()) / "pq_uploads"
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = str(tmp_dir / Path(upld.name).name)  # basename only — blocks path traversal
        if (st.session_state.get("_csv_upload_name") != upld.name
                or not os.path.exists(tmp_path)):
            with open(tmp_path, "wb") as f:
                f.write(upld.getvalue())
            st.session_state["_csv_upload_name"] = upld.name
            st.session_state["_csv_upload_path"] = tmp_path
        st.success(f"File saved: `{tmp_path}`")
        st.info("Mode automatically set to 'csv'. "
                "The path above will be overridden when running.")

    # ---- Path / config fields ---------------------------------------------
    st.divider()
    _DADOS_KEYS = ["modo_entrada", "pasta_dados", "arquivo_csv",
                   "coluna_classe", "coluna_concentracao",
                   "pasta_saida", "excluir_classes", "imagem_incluir_textura"]

    col_d1, col_d2 = st.columns(2)
    for i, k in enumerate(_DADOS_KEYS):
        s = specs.get(k)
        if s is None:
            continue
        with (col_d1 if i % 2 == 0 else col_d2):
            valores[k] = widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    # ---- Data statistics preview -------------------------------------------
    st.divider()
    st.markdown("**Data preview**")
    cfg_prev, _ = coletar_config(cfg_base, valores)
    ok_dados, msg_dados = pq._validar_pasta_dados(cfg_prev)
    (st.success if ok_dados else st.warning)(f"Status: {msg_dados}")

    if ok_dados and st.button("🔍 Load spectra preview", key="btn_prev_dados"):
        modo = cfg_prev.modo
        wn_mn = float(cfg_prev.wn_min)
        wn_mx = float(cfg_prev.wn_max)
        with st.spinner("Loading spectra sample..."):
            if modo == "dx":
                wn_p, X_p, labs_p = preview_espectros_dx(
                    cfg_prev.pasta_entrada, wn_mn, wn_mx)
            elif modo == "csv":
                csv_cam = st.session_state.get("_csv_upload_path",
                                               cfg_prev.arquivo_csv)
                wn_p, X_p, labs_p = preview_espectros_csv(
                    csv_cam, cfg_prev.coluna_classe, wn_mn, wn_mx)
            else:
                wn_p, X_p, labs_p = None, None, None

        if wn_p is not None and X_p is not None:
            cls_u = np.unique(np.asarray(labs_p))
            st.markdown(f"**{len(X_p)} spectra** · {len(cls_u)} classes: "
                        f"`{'`, `'.join(cls_u[:8])}`"
                        + (" ..." if len(cls_u) > 8 else ""))
            fig_p = plot_espectros_media(wn_p, X_p, np.asarray(labs_p), titulo="Raw spectra (sample)")
            st.pyplot(fig_p, use_container_width=True)
            plt.close(fig_p)
        else:
            st.warning("Could not load spectra for preview. "
                       "Check the path/mode.")

    # ---- Save / Reload config.yaml ---------------------------------
    st.divider()
    cfg_dados, erros_dados = coletar_config(cfg_base, valores)
    if erros_dados:
        st.warning("Fields with errors:\n- " + "\n- ".join(erros_dados))
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        if st.button("💾 Save config.yaml", key="btn_salvar_cfg_dados",
                     use_container_width=True):
            if erros_dados:
                st.error("Fix the fields before saving.")
            else:
                pq.salvar_config(cfg_dados, cfg_path)
                st.session_state.cfg_base = cfg_dados
                st.success(f"Saved to {cfg_path}")
    with c_s2:
        if st.button("↺ Reload config.yaml", key="btn_reload_cfg_dados",
                     use_container_width=True):
            try:
                st.session_state.cfg_base = pq.carregar_config(cfg_path)
                st.success("Config reloaded.")
                st.rerun()
            except (RuntimeError, FileNotFoundError, ValueError) as e:
                st.error(f"Error: {e}")
