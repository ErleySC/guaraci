"""app_tabs/relatorios.py — Aba 7 (Reports): downloads (ZIP/PDF/Word/Excel/
LaTeX/PPTX), limpeza de execuções antigas, Model Card, resumo e galeria de
figuras. Extraído de app_quimiometria.py (item 18).

Geração de bytes dos relatórios delegada às funções cacheadas passadas por
parâmetro (definidas em app_quimiometria.py, que por sua vez chamam
guaraci.reports — ver item 18, primeira fatia).
"""
from __future__ import annotations

import os
from typing import Callable, Dict

import streamlit as st

_CATS_R = {
    "All":              "",
    "PCA":              "pca",
    "PLS-DA":           "plsda",
    "Outliers":         "outlier",
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
    "Wold/ANOVA":       "wold",
    "Regression":       "regressao",
    "Benchmark":        "benchmark",
    "Monte Carlo CV":   "monte_carlo",
    "DET curves":       "fig_det",
    "SHAP":             "fig_shap",
}


def _tamanho_pasta_mb(pasta_p: str) -> float:
    tot = 0
    for raiz, _, arqs in os.walk(pasta_p):
        for a in arqs:
            try: tot += os.path.getsize(os.path.join(raiz, a))
            except OSError:
                pass   # arquivo removido/inacessivel entre o walk e o stat
    return round(tot / (1024 * 1024), 1)


def render(pq, modo_analise_rotulo: Dict[str, str],
           zip_da_pasta: Callable, pdf_bytes: Callable, word_bytes: Callable,
           excel_bytes: Callable, latex_bytes: Callable, pptx_bytes: Callable,
           ler_resumo: Callable, ler_model_card: Callable,
           listar_figuras: Callable) -> None:
    """Renderiza a aba Reports. Os `*_bytes`/`ler_*`/`listar_figuras` são as
    versões cacheadas (@st.cache_data) definidas em app_quimiometria.py."""
    st.subheader("Reports and Downloads")
    pasta_r = st.session_state.get("ultima_pasta")

    if not pasta_r or not os.path.isdir(pasta_r):
        st.info("Run the pipeline (Model tab) to generate reports.")
        return

    st.caption(f"Results folder: `{os.path.abspath(pasta_r)}`")

    # ── Downloads ─────────────────────────────────────────────────────
    st.markdown("### ⬇️ Downloads")

    _nome_base = os.path.basename(pasta_r)
    # "Study type" na capa do relatório é DERIVADO do Modo de análise
    # escolhido (N1/N2/N3) — sem campo duplicado na aba Project.
    _tipo_estudo = modo_analise_rotulo.get(
        st.session_state.get("w_nivel", ""), "")
    _projeto_info = {
        "nome":     st.session_state.get("proj_nome", ""),
        "autor":    st.session_state.get("proj_autor", ""),
        "inst":     st.session_state.get("proj_inst", "GEAAp / UFPA"),
        "tipo":     _tipo_estudo,
        "objetivo": st.session_state.get("proj_objetivo", ""),
    }
    # Sorted tuple used as cache key (Dict is not hashable)
    _proj_items = tuple(sorted(_projeto_info.items()))

    # Row 1: ZIP + PDF
    col_a, col_b = st.columns(2)
    with col_a:
        try:
            st.download_button(
                "📦 Full results (.zip)",
                data=zip_da_pasta(pasta_r),
                file_name=_nome_base + ".zip",
                mime="application/zip",
                use_container_width=True,
            )
        except Exception as e_zip:  # noqa: BLE001 -- 1 botao de download de 6
            # (ZIP/PDF/Word/Excel/LaTeX/PPTX, cada um c/ gerador proprio);
            # erro exibido ao usuario via st.warning, os demais continuam.
            st.warning(f"ZIP: {e_zip}")

    with col_b:
        try:
            st.download_button(
                "📄 PDF Report",
                data=pdf_bytes(pasta_r, _proj_items),
                file_name=_nome_base + "_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        except Exception as e_pdf:  # noqa: BLE001 -- mesma logica: 1 de 6
            # botoes de download, erro exibido via st.error.
            st.error(f"PDF: {e_pdf}")

    # Row 2: Word + Excel
    col_c, col_d = st.columns(2)
    with col_c:
        try:
            st.download_button(
                "📝 Word Report (.docx)",
                data=word_bytes(pasta_r, _proj_items),
                file_name=_nome_base + "_report.docx",
                mime="application/vnd.openxmlformats-officedocument"
                     ".wordprocessingml.document",
                use_container_width=True,
            )
        except Exception as e_word:  # noqa: BLE001 -- 1 de 6 botoes de download.
            st.error(f"Word: {e_word}")

    with col_d:
        try:
            st.download_button(
                "📊 Data in Excel (.xlsx)",
                data=excel_bytes(pasta_r),
                file_name=_nome_base + "_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument"
                     ".spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e_xlsx:  # noqa: BLE001 -- 1 de 6 botoes de download.
            st.error(f"Excel: {e_xlsx}")

    # Row 3: LaTeX + PowerPoint
    col_e, col_f = st.columns(2)
    with col_e:
        try:
            st.download_button(
                "🔬 LaTeX Template (Talanta / Food Chemistry / J. Chemom.)",
                data=latex_bytes(pasta_r, _proj_items),
                file_name=_nome_base + "_template.tex",
                mime="text/plain",
                use_container_width=True,
            )
        except Exception as e_tex:  # noqa: BLE001 -- 1 de 6 botoes de download.
            st.error(f"LaTeX: {e_tex}")

    with col_f:
        try:
            from pptx import Presentation as _PPTXCheck  # noqa: F401
            st.download_button(
                "🎯 PowerPoint Presentation (.pptx)",
                data=pptx_bytes(pasta_r, _proj_items),
                file_name=_nome_base + "_presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument"
                     ".presentationml.presentation",
                use_container_width=True,
            )
        except ImportError:
            st.warning(
                "python-pptx not installed. "
                "Run: `pip install python-pptx>=1.1`",
                icon="⚠️",
            )
        except Exception as e_pptx:  # noqa: BLE001 -- 1 de 6 botoes de download
            # (ImportError ja tratado acima separadamente).
            st.error(f"PowerPoint: {e_pptx}")

    st.divider()

    # ── Clean up old results ─────────────────────────────────────────
    with st.expander("🗑️ Free space — Clean up old results",
                     expanded=False):
        _pasta_base_lim = os.path.dirname(pasta_r)
        _pastas_exist = sorted(
            [p for p in os.scandir(_pasta_base_lim) if p.is_dir()],
            key=lambda p: p.stat().st_mtime, reverse=True)
        n_pastas = len(_pastas_exist)

        st.caption(f"Results folder: `{_pasta_base_lim}`  "
                   f"({n_pastas} runs stored)")
        if n_pastas > 1:
            _manter = st.slider(
                "Keep N most recent runs",
                min_value=1, max_value=max(1, n_pastas - 1),
                value=min(3, n_pastas - 1), key="lim_manter")
            _n_remover = n_pastas - _manter
            _tam_est = sum(
                _tamanho_pasta_mb(p.path)
                for p in _pastas_exist[_manter:])
            st.info(
                f"**{_n_remover}** old run(s) will be removed "
                f"(~{_tam_est:.0f} MB freed). "
                f"The current run **will not be affected**.")
            if st.button("🗑️ Confirm cleanup",
                         key="btn_limpar_resultados",
                         type="secondary"):
                _res = pq.limpar_resultados_antigos(
                    _pasta_base_lim, _manter)
                if _res["removidas"]:
                    st.success(
                        f"Removed {len(_res['removidas'])} folder(s), "
                        f"freed {_res['liberado_mb']:.0f} MB.")
                else:
                    st.info("No folders removed.")
                if _res["erro"]:
                    st.warning(f"Errors: {_res['erro']}")
        else:
            st.info("Only one run stored. Nothing to clean.")

    st.divider()

    # Model Card (Mitchell et al. 2019) — intended use, data, metrics,
    # caveats in one shareable document. Rendered as Markdown (tables
    # display nicely), with its own download button.
    st.markdown("### 🪪 Model Card")
    model_card_r = ler_model_card(pasta_r)
    if model_card_r:
        with st.expander("View Model Card", expanded=False):
            st.markdown(model_card_r)
        st.download_button(
            "⬇️ Model Card (.md)",
            data=model_card_r.encode("utf-8"),
            file_name=_nome_base + "_model_card.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.info("File model_card.md not found.")

    st.divider()

    # Model summary
    st.markdown("### 📋 Model summary")
    resumo_r = ler_resumo(pasta_r)
    if resumo_r:
        st.text_area("resumo_modelo.txt", resumo_r, height=400)
    else:
        st.info("File resumo_modelo.txt not found.")

    st.divider()

    # Full gallery with filter
    st.markdown("### 🖼️ Figure gallery")
    imgs_r = listar_figuras(pasta_r)
    if imgs_r:
        filtro_r = st.selectbox("Filter figures",
                                list(_CATS_R.keys()), key="filtro_rel")
        token_r  = _CATS_R[filtro_r].lower()
        imgs_filt_r = [im for im in imgs_r
                       if token_r in os.path.basename(im).lower()] \
                      if token_r else imgs_r
        st.caption(f"{len(imgs_filt_r)} figure(s) displayed.")
        n_cols_r = st.slider("Columns", 1, 3, 2, key="slider_cols_rel")
        cols_r   = st.columns(n_cols_r)
        for j, img in enumerate(imgs_filt_r):
            with cols_r[j % n_cols_r]:
                st.image(img, caption=os.path.basename(img),
                         use_container_width=True)
    else:
        st.info("No PNG/JPG images found in the results folder.")

    st.divider()

    # Execution log
    if st.session_state.get("ultimo_log"):
        with st.expander("📜 Execution log (terminal output)"):
            st.code(st.session_state.ultimo_log, language="text")
