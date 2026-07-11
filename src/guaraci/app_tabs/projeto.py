"""app_tabs/projeto.py — Aba 1 (Project): identificação do projeto + status
de hardware. Extraído de app_quimiometria.py (item 18 da auditoria).
"""
from __future__ import annotations

from typing import Callable

import streamlit as st


def _hardware_status_widget(pq, is_public_demo: bool = False) -> None:
    """Exibe o painel de hardware com alertas de compatibilidade.

    No deploy publico (Streamlit Community Cloud), psutil le a RAM da
    maquina HOSPEDEIRA fisica compartilhada, nao a fatia real alocada ao
    container — o numero absoluto pode enganar (ex.: "125.8 GB" quando o
    container so tem ~1-2GB). hardware_probe() ja tenta corrigir isso lendo
    o limite via cgroup (Linux); quando o cgroup NAO esta exposto (limitacao
    da propria sandbox do Streamlit Cloud, fora do nosso controle), o numero
    continua sem confirmacao — nesse caso, no deploy publico, escondemos o
    valor absoluto (que pode ser falso) e mostramos so o essencial e
    verdadeiro: que os limites sao ajustados automaticamente.
    """
    try:
        hw = pq.hardware_probe()
        ram_t = hw["ram_total_gb"]
        ram_l = hw["ram_livre_gb"]
        cpu_f = hw["cpu_fisicos"]
        cpu_l = hw["cpu_logicos"]
        disco = hw["disco_livre_gb"]
        psutil_ok = hw["psutil_ok"]
        limitada_por_container = hw.get("ram_limitada_por_container", False)

        if ram_l < 2.0:
            cor_ram = "🔴"
            dica = "Critical RAM. Disable Benchmark, SHAP and MC CV."
        elif ram_l < 4.0:
            cor_ram = "🟠"
            dica = "Low RAM. Benchmark and SHAP will be disabled automatically."
        elif ram_l < 8.0:
            cor_ram = "🟡"
            dica = "Moderate RAM. Limits will be adjusted automatically."
        else:
            cor_ram = "🟢"
            dica = "Sufficient RAM for all operations."

        # Numero absoluto de RAM so eh confiavel se: (a) nao estamos no
        # deploy publico, ou (b) o cgroup confirmou o limite real do
        # container. Fora isso (deploy publico + cgroup nao exposto), o
        # valor pode ser o da maquina host inteira — escondido.
        ram_confiavel = (not is_public_demo) or limitada_por_container

        c_hw1, c_hw2, c_hw3 = st.columns(3)
        with c_hw1:
            if ram_confiavel:
                _ram_note = " (container limit)" if limitada_por_container else ""
                st.metric("Total RAM", f"{ram_t:.1f} GB{_ram_note}",
                          delta=f"{cor_ram} {ram_l:.1f} GB free",
                          delta_color="off")
            else:
                st.metric("Total RAM", "Managed by host",
                          delta="Cloud demo — limits applied automatically",
                          delta_color="off")
        with c_hw2:
            st.metric("CPU", f"{cpu_f} cores",
                      delta=f"{cpu_l} logical threads",
                      delta_color="off")
        with c_hw3:
            st.metric("Free disk", f"{disco:.0f} GB",
                      delta="working folder",
                      delta_color="off")

        if ram_confiavel and ram_l < 8.0:
            st.warning(f"**Limited hardware detected.** {dica}")
        if not psutil_ok:
            st.caption("⚠️ psutil not available — approximate readings. "
                       "Install with `pip install psutil`.")
    except Exception:
        st.caption("Hardware: could not detect hardware specifications.")


def render(pq, T: Callable[[str], str], is_public_demo: bool = False) -> None:
    """Renderiza a aba Project. `T` é a função de tradução `_T` do app."""
    st.subheader(T("Project Identification"))
    st.caption(
        "Descriptive only — used in the report cover and saved automatically in "
        "this session. What the pipeline actually runs is set by the "
        "**Analysis mode** (Model tab)."
        if st.session_state.get("lang") != "PT" else
        "Apenas descritivo — vai na capa dos relatórios e é salvo automaticamente "
        "nesta sessão. O que o pipeline executa é definido pelo **Modo de "
        "análise** (aba Modelo).")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Project name", key="proj_nome",
                      placeholder="e.g.: Authentication of Amazonian Oils FT-NIR")
        st.text_input("Author(s)", key="proj_autor",
                      placeholder="e.g.: Silva, J.A.; Costa, M.B.")
        st.text_input("Institution / Laboratory", key="proj_inst",
                      placeholder="e.g.: GEAAp / UFPA")
    with c2:
        st.text_area("Objective", key="proj_objetivo", height=182,
                     placeholder="Describe the objective of the chemometric analysis...")

    with st.expander("💻 Hardware Status", expanded=False):
        _hardware_status_widget(pq, is_public_demo=is_public_demo)

    run_proj = st.session_state.get("proj_nome", "")
    if run_proj:
        st.caption(f"✅ Active project: **{run_proj}**")
