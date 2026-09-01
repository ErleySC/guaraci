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
           load_summary: Callable, load_model_card: Callable,
           list_figures: Callable,
           T: Callable[[str], str] = lambda s: s) -> None:
    """Renderiza a aba Reports. Os `*_bytes`/`ler_*`/`list_figures` são as
    versões cacheadas (@st.cache_data) definidas em app_quimiometria.py.

    `T`: traducao PT/EN (mesma funcao `_T` de app_quimiometria.py). Default
    no-op so' para chamada direta em teste isolado.
    """
    st.subheader(T("Reports and Downloads"))
    st.caption(T("📄 Download reports (ZIP/PDF/Word/Excel/LaTeX/PowerPoint), "
               "browse the figure gallery, and clean up old result folders."))
    pasta_r = st.session_state.get("ultima_pasta")

    if not pasta_r or not os.path.isdir(pasta_r):
        st.info(T("Run the pipeline (Model tab) to generate reports."))
        return

    st.caption(T("Results folder: `{pasta}`").format(pasta=os.path.abspath(pasta_r)))

    # ── Downloads ─────────────────────────────────────────────────────
    st.markdown(T("### ⬇️ Downloads"))

    _nome_base = os.path.basename(pasta_r)
    # "Study type" na capa do relatório é DERIVADO do Modo de análise
    # escolhido (N1/N2/N3) — sem campo duplicado na aba Project.
    _tipo_estudo = modo_analise_rotulo.get(
        st.session_state.get("w_nivel", ""), "")
    _projeto_info = {
        "nome":     st.session_state.get("proj_nome", ""),
        "autor":    st.session_state.get("proj_autor", ""),
        "inst":     st.session_state.get("proj_inst", ""),
        "tipo":     _tipo_estudo,
        "objetivo": st.session_state.get("proj_objetivo", ""),
    }
    # Sorted tuple used as cache key (Dict is not hashable)
    _proj_items = tuple(sorted(_projeto_info.items()))

    # Generation is cached (@st.cache_data) after the first call, but the
    # first time a report type is requested for this run (PDF/Word/PPTX
    # embed up to ~14 figures) it can take a few seconds — a spinner avoids
    # the tab looking frozen while nothing is on screen yet.
    with st.spinner(T("Preparing report files (cached after the first time)...")):
        # Row 1: ZIP + PDF
        col_a, col_b = st.columns(2)
        with col_a:
            try:
                st.download_button(
                    T("📦 Full results (.zip)"),
                    data=zip_da_pasta(pasta_r),
                    file_name=_nome_base + ".zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            except Exception as e_zip:  # noqa: BLE001 -- 1 botao de download de 6
                # (ZIP/PDF/Word/Excel/LaTeX/PPTX, cada um c/ gerador proprio);
                # erro exibido ao usuario via st.warning, os demais continuam.
                st.warning(T("ZIP: {e}").format(e=e_zip))

        with col_b:
            try:
                st.download_button(
                    T("📄 PDF Report"),
                    data=pdf_bytes(pasta_r, _proj_items),
                    file_name=_nome_base + "_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                )
            except Exception as e_pdf:  # noqa: BLE001 -- mesma logica: 1 de 6
                # botoes de download, erro exibido via st.error.
                st.error(T("PDF: {e}").format(e=e_pdf))

        # Row 2: Word + Excel
        col_c, col_d = st.columns(2)
        with col_c:
            try:
                st.download_button(
                    T("📝 Word Report (.docx)"),
                    data=word_bytes(pasta_r, _proj_items),
                    file_name=_nome_base + "_report.docx",
                    mime="application/vnd.openxmlformats-officedocument"
                         ".wordprocessingml.document",
                    use_container_width=True,
                )
            except Exception as e_word:  # noqa: BLE001 -- 1 de 6 botoes de download.
                st.error(T("Word: {e}").format(e=e_word))

        with col_d:
            try:
                st.download_button(
                    T("📊 Data in Excel (.xlsx)"),
                    data=excel_bytes(pasta_r),
                    file_name=_nome_base + "_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument"
                         ".spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e_xlsx:  # noqa: BLE001 -- 1 de 6 botoes de download.
                st.error(T("Excel: {e}").format(e=e_xlsx))

        # Row 3: LaTeX + PowerPoint
        col_e, col_f = st.columns(2)
        with col_e:
            try:
                st.download_button(
                    T("🔬 LaTeX Template (Talanta / Food Chemistry / J. Chemom.)"),
                    data=latex_bytes(pasta_r, _proj_items),
                    file_name=_nome_base + "_template.tex",
                    mime="text/plain",
                    use_container_width=True,
                )
            except Exception as e_tex:  # noqa: BLE001 -- 1 de 6 botoes de download.
                st.error(T("LaTeX: {e}").format(e=e_tex))

        with col_f:
            try:
                from pptx import Presentation as _PPTXCheck  # noqa: F401
                st.download_button(
                    T("🎯 PowerPoint Presentation (.pptx)"),
                    data=pptx_bytes(pasta_r, _proj_items),
                    file_name=_nome_base + "_presentation.pptx",
                    mime="application/vnd.openxmlformats-officedocument"
                         ".presentationml.presentation",
                    use_container_width=True,
                )
            except ImportError:
                st.warning(
                    T("python-pptx not installed. "
                    "Run: `pip install python-pptx>=1.1`"),
                    icon="⚠️",
                )
            except Exception as e_pptx:  # noqa: BLE001 -- 1 de 6 botoes de download
                # (ImportError ja tratado acima separadamente).
                st.error(T("PowerPoint: {e}").format(e=e_pptx))

    st.divider()

    # ── Clean up old results ─────────────────────────────────────────
    with st.expander(T("🗑️ Free space — Clean up old results"),
                     expanded=False):
        _pasta_base_lim = os.path.dirname(pasta_r)
        _pastas_exist = sorted(
            [p for p in os.scandir(_pasta_base_lim) if p.is_dir()],
            key=lambda p: p.stat().st_mtime, reverse=True)
        n_pastas = len(_pastas_exist)

        st.caption(T("Results folder: `{pasta}`  ({n} runs stored)").format(
            pasta=_pasta_base_lim, n=n_pastas))
        if n_pastas > 1:
            _manter = st.slider(
                T("Keep N most recent runs"),
                min_value=1, max_value=max(1, n_pastas - 1),
                value=min(3, n_pastas - 1), key="lim_manter")
            _n_remover = n_pastas - _manter
            _tam_est = sum(
                _tamanho_pasta_mb(p.path)
                for p in _pastas_exist[_manter:])
            st.info(T(
                "**{n_remover}** old run(s) will be removed "
                "(~{tam_est:.0f} MB freed). "
                "The current run **will not be affected**.").format(
                    n_remover=_n_remover, tam_est=_tam_est))
            if st.button(T("🗑️ Confirm cleanup"),
                         key="btn_limpar_resultados",
                         type="secondary"):
                _res = pq.clear_old_results(
                    _pasta_base_lim, _manter)
                if _res["removidas"]:
                    st.success(T(
                        "Removed {n} folder(s), freed {mb:.0f} MB.").format(
                            n=len(_res['removidas']), mb=_res['liberado_mb']))
                else:
                    st.info(T("No folders removed."))
                if _res["erro"]:
                    st.warning(T("Errors: {erro}").format(erro=_res["erro"]))
        else:
            st.info(T("Only one run stored. Nothing to clean."))

    st.divider()

    # Model Card (Mitchell et al. 2019) — intended use, data, metrics,
    # caveats in one shareable document. Rendered as Markdown (tables
    # display nicely), with its own download button.
    st.markdown(T("### 🪪 Model Card"))
    model_card_r = load_model_card(pasta_r)
    if model_card_r:
        with st.expander(T("View Model Card"), expanded=False):
            st.markdown(model_card_r)
        st.download_button(
            T("⬇️ Model Card (.md)"),
            data=model_card_r.encode("utf-8"),
            file_name=_nome_base + "_model_card.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.info(T("File model_card.md not found."))

    st.divider()

    # Model summary
    st.markdown(T("### 📋 Model summary"))
    resumo_r = load_summary(pasta_r)
    if resumo_r:
        st.text_area("resumo_modelo.txt", resumo_r, height=400)
    else:
        st.info(T("File resumo_modelo.txt not found."))

    st.divider()

    # Full gallery with filter
    st.markdown(T("### 🖼️ Figure gallery"))
    imgs_r = list_figures(pasta_r)
    if imgs_r:
        filtro_r = st.selectbox(
            T("Filter figures"), list(_CATS_R.keys()), key="filtro_rel",
            help=T("Show only figures of one analysis type (e.g. PCA scores, "
                 "confusion matrix, DD-SIMCA acceptance). 'All' shows every "
                 "figure generated by the run."))
        token_r  = _CATS_R[filtro_r].lower()
        imgs_filt_r = [im for im in imgs_r
                       if token_r in os.path.basename(im).lower()] \
                      if token_r else imgs_r
        st.caption(T("{n} figure(s) displayed.").format(n=len(imgs_filt_r)))
        n_cols_r = st.slider(T("Columns"), 1, 3, 2, key="slider_cols_rel")
        cols_r   = st.columns(n_cols_r)
        for j, img in enumerate(imgs_filt_r):
            with cols_r[j % n_cols_r]:
                st.image(img, caption=os.path.basename(img),
                         use_container_width=True)
    else:
        st.info(T("No PNG/JPG images found in the results folder."))

    st.divider()

    # Execution log
    if st.session_state.get("ultimo_log"):
        with st.expander(T("📜 Execution log (terminal output)")):
            st.code(st.session_state.ultimo_log, language="text")
