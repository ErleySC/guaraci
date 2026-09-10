"""app_tabs/projeto.py — Tela (Project): identificação do projeto e status
de hardware. Extraído de app_quimiometria.py (item 18 da auditoria).

Histórico: o painel de status ficou AQUI, no topo desta tela, entre
2026-09-08 (Passo 190) e a reestruturação do mesmo dia (Passo 195), que o
promoveu a tela própria -- `app_tabs/inicio.py`. Com a navegação lateral
uma tela a mais não empurra as outras, que era o motivo de ele estar
embutido aqui.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import streamlit as st

from guaraci.app_logic import load_summary
from guaraci.config import NOME_RELATORIOS
from guaraci.resultados_io import load_design_audit
from guaraci.resumo_parse import parse_dataset_counts
from guaraci.validacao_publica import find_rows_for_matrix, load_consolidated_table

_RAIZ_REPO = Path(__file__).resolve().parents[3]

# Severidade da auditoria de delineamento -> como mostrar. "silenciado" NAO
# vira "ok": a checagem continua no painel, com a justificativa anexada
# (mesma regra de auditoria_delineamento.run_audit).
_ICONE_SEVERIDADE = {"critico": "🔴", "aviso": "🟠",
                     "ok": "🟢", "silenciado": "⚪"}


def _pasta_relatorios(pasta_run: str) -> str:
    """Subpasta de relatórios da execução (layout atual), com fallback para
    o layout `logs/` das execuções anteriores a jul/2026."""
    for candidata in (os.path.join(pasta_run, NOME_RELATORIOS),
                      os.path.join(pasta_run, "logs"),
                      pasta_run):
        if os.path.isdir(candidata):
            return candidata
    return pasta_run


def _linha_validacao_publica(matriz: Optional[str]) -> Optional[Dict[str, str]]:
    """Primeira linha da tabela de validações públicas cuja matriz casa com a
    do perfil ativo, ou None. Usa a MESMA leitura do gerador do vault
    (`guaraci.validacao_publica`), nunca uma cópia."""
    linhas = find_rows_for_matrix(load_consolidated_table(_RAIZ_REPO), matriz)
    return linhas[0] if linhas else None


def _painel_status(pq, T: Callable[[str], str]) -> None:
    """Resumo do estado REAL da sessão. Todo número vem de execução ou de
    `st.session_state`; nada é fixo no código."""
    pt = st.session_state.get("lang") == "PT"
    pasta_run = st.session_state.get("ultima_pasta")
    tem_run = bool(pasta_run) and os.path.isdir(pasta_run or "")
    previa = st.session_state.get("previa_dados")

    if not tem_run and not previa:
        st.info(T("No data loaded yet — start on the **Data** tab: point to a "
                  "spectra folder or upload a CSV. This panel fills in with "
                  "the real numbers as soon as there is data."))
        return

    st.markdown(T("### 📌 Project status"))

    if tem_run:
        contagens = parse_dataset_counts(load_summary(str(pasta_run)) or "")
        origem = T("last completed run")
    else:
        contagens = {"amostras": previa.get("n_espectros"),
                     "variaveis": None,
                     "classes": previa.get("n_classes")}
        origem = T("data preview (no run yet)")

    c1, c2, c3 = st.columns(3)
    for col, chave, rotulo in (
        (c1, "amostras", T("spectra")),
        (c2, "classes", T("classes")),
        (c3, "variaveis", T("spectral variables")),
    ):
        with col:
            valor = contagens.get(chave)
            # "—" em vez de 0: um zero se leria como "nenhuma amostra", que e'
            # diferente de "esse numero nao esta' disponivel nesta origem".
            st.metric(rotulo, str(valor) if valor is not None else "—")
    st.caption(T("Source: {origem}.").format(origem=origem))

    _painel_tecnica(pq, pt, T)

    if tem_run:
        _painel_auditoria(_pasta_relatorios(str(pasta_run)), T)
        st.caption(T("Results folder: `{pasta}`").format(
            pasta=os.path.abspath(str(pasta_run))))
    st.divider()


def _painel_tecnica(pq, pt: bool, T: Callable[[str], str]) -> None:
    """Técnica/matriz do perfil ativo + estado de validação pública, quando
    houver linha correspondente em docs/VALIDACAO_PUBLICA.md."""
    try:
        cfg = st.session_state.get("cfg_base")
        from guaraci.perfil_matriz import cfg_profile
        perfil = cfg_profile(cfg)
        matriz = perfil.vocabulario.matriz
        st.markdown(T("**Matrix / technique:** {matriz}").format(matriz=matriz))
    except Exception:  # noqa: BLE001 -- linha informativa do painel; sem
        # perfil resolvido nao ha' o que cruzar, e o resto do painel segue.
        return

    linha = _linha_validacao_publica(matriz)
    if linha:
        st.markdown(T(
            "**Public validation for this matrix:** {estado} — {dataset} "
            "({metrica})").format(estado=linha.get("estado", "—"),
                                  dataset=linha.get("dataset", "—"),
                                  metrica=linha.get("metrica", "—")))
        st.caption(T("From the consolidated table in "
                     "`docs/VALIDACAO_PUBLICA.md`."))
    else:
        st.caption(T("No public validation registered for this matrix in "
                     "`docs/VALIDACAO_PUBLICA.md` — the pipeline still runs, "
                     "but there is no external benchmark to compare against."))


def _painel_auditoria(pasta_rel: str, T: Callable[[str], str]) -> None:
    """Achados REAIS da auditoria de delineamento da execução (Bloco 11),
    lidos do JSON gravado pelo pipeline — nunca um texto fixo."""
    achados: List[Dict[str, str]] = load_design_audit(pasta_rel)
    if not achados:
        st.caption(T("Design audit not available for this run (run produced "
                     "before this record existed)."))
        return

    n_crit = sum(1 for a in achados if a.get("severidade") == "critico")
    n_aviso = sum(1 for a in achados if a.get("severidade") == "aviso")
    resumo = T("{crit} critical · {aviso} warning(s) · {n} checks").format(
        crit=n_crit, aviso=n_aviso, n=len(achados))
    if n_crit:
        st.error(T("Design audit: {resumo}").format(resumo=resumo))
    elif n_aviso:
        st.warning(T("Design audit: {resumo}").format(resumo=resumo))
    else:
        st.success(T("Design audit: {resumo}").format(resumo=resumo))

    with st.expander(T("Design audit findings"), expanded=bool(n_crit)):
        for a in achados:
            icone = _ICONE_SEVERIDADE.get(a.get("severidade", ""), "•")
            st.markdown(f"{icone} **{a.get('nome', '?')}** — "
                        f"{a.get('mensagem', '')}")


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
    except Exception:  # noqa: BLE001 -- painel cosmetico best-effort (mesmo
        # padrao de hardware_probe/psutil em outros pontos da UI); fallback
        # textual documentado, nunca impede a aba de renderizar.
        st.caption("Hardware: could not detect hardware specifications.")


def render(pq, T: Callable[[str], str], is_public_demo: bool = False) -> None:
    """Renderiza a aba Project. `T` é a função de tradução `_T` do app."""
    _painel_status(pq, T)

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
                      placeholder="e.g.: Analytical Chemistry Laboratory")
    with c2:
        st.text_area("Objective", key="proj_objetivo", height=182,
                     placeholder="Describe the objective of the chemometric analysis...")

    with st.expander("💻 Hardware Status", expanded=False):
        _hardware_status_widget(pq, is_public_demo=is_public_demo)

    run_proj = st.session_state.get("proj_nome", "")
    if run_proj:
        st.caption(f"✅ Active project: **{run_proj}**")
