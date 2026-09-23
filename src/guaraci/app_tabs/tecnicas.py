"""app_tabs/tecnicas.py — Tela (Advanced Techniques): ASCA, EPO/GLSW,
MCR-ALS (com/sem restrição de correlação) e fusão multibloco.

Fecha a lacuna encontrada na auditoria de acessibilidade CLI/web: essas
técnicas existiam implementadas e testadas em isolamento, mas sem NENHUM
caminho de execução real (nem CLI, nem web). Lógica real em
`tecnicas_avancadas.py` (testável sem Streamlit) — este módulo só coleta
entrada e formata saída, mesmo padrão de `guaraci.py::_menu_tecnicas_
avancadas` (fonte única entre as duas interfaces).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import streamlit as st


def _carregar_dataset_atual(pq, cfg_base, T):
    """Carrega+valida o dataset configurado na aba Data (mesmo caminho de
    `pq.load_data`/`pq.validate_input` usado pela aba Validation)."""
    try:
        wn, X, rotulos, conc, mae_id, metadados = pq.load_data(cfg_base)
        X, wn, rotulos, conc, mae_id, _rel = pq.validate_input(
            X, wn, rotulos, conc, mae_id)
    except Exception as e:  # noqa: BLE001 -- dado externo, mesma disciplina do CLI
        st.error(T("Error loading data") + f": {e}")
        return None
    return wn, X, rotulos, conc, mae_id, metadados


def _secao_asca(pq, cfg_base, T) -> None:
    from guaraci.tecnicas_avancadas import (
        fatores_categoricos_disponiveis, rodar_asca)

    st.caption(T(
        "Decomposes the loaded spectra by design factors (species, and any "
        "low-cardinality metadata column) and tests each factor's "
        "significance by permutation — group-aware when mae_id is "
        "available."))
    # Persistido em session_state: `if st.button(...): ...` so' e' True no
    # MESMO rerun do clique -- sem guardar o dataset carregado, o resto da
    # UI (multiselect de fatores, botao Run) sumiria no rerun seguinte
    # (acionado pelo proprio multiselect/Run), achado real via AppTest
    # (nunca exercitado manualmente antes desta rodada).
    if st.button(T("Load dataset"), key="btn_tec_asca_load"):
        st.session_state["tec_asca_dataset"] = _carregar_dataset_atual(pq, cfg_base, T)
    carregado = st.session_state.get("tec_asca_dataset")
    if carregado is None:
        return
    wn, X, rotulos, conc, mae_id, metadados = carregado
    fatores_disp = fatores_categoricos_disponiveis(rotulos, metadados)
    escolhidos = st.multiselect(
        T("Factors"), list(fatores_disp), default=list(fatores_disp)[:1],
        key="tec_asca_fatores")
    if not escolhidos:
        st.info(T("Choose at least 1 factor."))
        return
    if st.button(T("Run ASCA"), key="btn_tec_asca_run"):
        fatores = {k: fatores_disp[k] for k in escolhidos}
        try:
            rel = rodar_asca(X, fatores, mae_id=mae_id)
        except ValueError as e:
            st.error(str(e)); return
        st.success(T("Group-aware (mae_id)") + f": {'yes' if rel.group_aware else 'NO'}")
        for nome, efeito in rel.decomposicao["efeitos"].items():
            p_txt = ""
            if rel.permutacao is not None:
                p = rel.permutacao[nome]["p_value"]
                p_txt = f" — p={p:.4f}" + ("  **SIGNIFICANT**" if p < 0.05 else "")
            st.write(f"**{nome}**: {efeito['fracao_ss_total']*100:.1f}% "
                     f"{T('of total variance')}{p_txt}")
        st.caption(T("ss_desbalanco") + f": {rel.decomposicao['ss_desbalanco']:.3g} — "
                   + T("large = correlated factors, see module scope limit"))


def _secao_epo_glsw(pq, cfg_base, T) -> None:
    from guaraci.tecnicas_avancadas import (
        fatores_categoricos_disponiveis, rodar_epo_glsw)

    st.caption(T(
        "Removes/attenuates the variation of a KNOWN nuisance factor "
        "(session, host species, instrument), estimated from difference "
        "spectra between samples that share the condition to preserve but "
        "differ in the nuisance factor. Exploratory diagnostic, not wired "
        "into the automatic training pipeline."))
    if st.button(T("Load dataset"), key="btn_tec_epo_load"):
        st.session_state["tec_epo_dataset"] = _carregar_dataset_atual(pq, cfg_base, T)
    carregado = st.session_state.get("tec_epo_dataset")
    if carregado is None:
        return
    wn, X, rotulos, conc, mae_id, metadados = carregado
    fatores_disp = fatores_categoricos_disponiveis(rotulos, metadados)
    nomes = list(fatores_disp)
    col1, col2 = st.columns(2)
    with col1:
        f1 = st.selectbox(T("grupo_interesse (preserve)"), nomes, key="tec_epo_f1")
    with col2:
        f2 = st.selectbox(T("fator_incomodo (remove)"), nomes,
                          index=min(1, len(nomes) - 1), key="tec_epo_f2")
    metodo = st.radio(T("Method"), ["EPO", "GLSW"], horizontal=True, key="tec_epo_metodo")
    if f1 == f2:
        st.warning(T("The two factors must be different."))
        return
    if st.button(T("Run"), key="btn_tec_epo_run"):
        try:
            rel = rodar_epo_glsw(X, fatores_disp[f1], fatores_disp[f2], f1, f2,
                                 metodo=metodo)
        except ValueError as e:
            st.error(str(e)); return
        st.success(f"{rel.metodo}: {rel.n_pares_diferenca} "
                  f"{T('difference pairs')}")
        st.write(f"{T('Total variance fraction removed')}: "
                f"{rel.variancia_removida_fracao*100:.1f}%")


def _secao_mcr_als(pq, cfg_base, T) -> None:
    from guaraci.mcr_als import MCRALSResultadoSupervisionado
    from guaraci.tecnicas_avancadas import rodar_mcr_als

    st.caption(T(
        "Resolves the spectral matrix into pure concentration/spectrum "
        "profiles (unsupervised), or anchors one component to known "
        "reference values when content/concentration is available "
        "(correlation-constrained variant). Interpretive tool, does NOT "
        "replace PLS-R for quantification."))
    if st.button(T("Load dataset"), key="btn_tec_mcr_load"):
        st.session_state["tec_mcr_dataset"] = _carregar_dataset_atual(pq, cfg_base, T)
    carregado = st.session_state.get("tec_mcr_dataset")
    if carregado is None:
        return
    wn, X, rotulos, conc, mae_id, metadados = carregado
    n_comp = st.number_input(T("Number of components"), min_value=1,
                              max_value=10, value=2, key="tec_mcr_ncomp")
    tem_conc = conc is not None and bool(np.any(~np.isnan(conc)))
    usar_restricao = False
    indice_alvo = 0
    if tem_conc:
        usar_restricao = st.checkbox(
            T("Use correlation restriction with known content"),
            key="tec_mcr_restricao")
        if usar_restricao:
            indice_alvo = st.number_input(
                T("Target component index (0-indexed)"), min_value=0,
                max_value=int(n_comp) - 1, value=0, key="tec_mcr_alvo")
    if st.button(T("Run MCR-ALS"), key="btn_tec_mcr_run"):
        X_pos = np.clip(X, 0, None) if np.any(X < 0) else X
        try:
            resultado = rodar_mcr_als(
                X_pos, int(n_comp),
                conc=conc if usar_restricao else None,
                indice_componente_alvo=int(indice_alvo) if usar_restricao else None)
        except ValueError as e:
            st.error(str(e)); return
        st.success(f"{resultado.n_iter} {T('iterations')}, "
                  f"{'converged' if resultado.convergiu else 'did NOT converge'}")
        st.write(f"lack-of-fit: {resultado.lof_percent:.2f}%")
        if isinstance(resultado, MCRALSResultadoSupervisionado):
            st.write(f"{T('Target component x reference correlation')}: "
                    f"{resultado.correlacao_calibracao:.3f}")


def _secao_fusao_multibloco(pq, T) -> None:
    from guaraci.tecnicas_avancadas import rodar_fusao_multibloco

    st.caption(T(
        "Fuses >=2 blocks measured on the SAME physical sample (e.g. NIR + "
        "MIR) into a single PLS-R regression. Needs 2 folders/files with "
        "the same samples in the same row order. Proof-of-concept: on the "
        "public Mendeley NIR+MIR pair, fusion did NOT beat the best single "
        "block."))
    blocos_cfg = []
    for i in (1, 2):
        with st.expander(f"{T('Block')} {i}", expanded=True):
            nome = st.text_input(T("Block name"), value=f"bloco{i}",
                                 key=f"tec_fusao_nome_{i}")
            modo = st.selectbox(
                T("Input mode"),
                ["dx", "csv", "spc", "sp", "opus", "rmn", "hplc", "gcms"],
                key=f"tec_fusao_modo_{i}")
            caminho = st.text_input(T("Folder (or CSV file)"),
                                    key=f"tec_fusao_caminho_{i}")
            blocos_cfg.append((nome, modo, caminho))

    if st.button(T("Run multiblock fusion"), key="btn_tec_fusao_run"):
        Config = pq.Config
        blocos = {}
        y_bloco = None
        for nome, modo, caminho in blocos_cfg:
            if not caminho:
                st.error(T("Fill in both folders/files."))
                return
            cfg_bloco = Config()
            cfg_bloco.mode = modo
            cfg_bloco.input_folder = caminho
            cfg_bloco.csv_file = caminho
            try:
                wn_b, X_b, rot_b, conc_b, mae_b, meta_b = pq.load_data(cfg_bloco)
            except Exception as e:  # noqa: BLE001 -- mesma disciplina acima
                st.error(f"{T('Error loading')} {nome}: {e}")
                return
            blocos[nome] = X_b
            st.write(f"✔ {nome}: {X_b.shape[0]} {T('samples')}, "
                    f"{X_b.shape[1]} {T('variables')}")
            if y_bloco is None and conc_b is not None:
                y_bloco = conc_b
        if y_bloco is None:
            st.error(T("Neither block has content/concentration (y) — "
                       "multiblock fusion here only supports regression."))
            return
        try:
            resultado, fatias = rodar_fusao_multibloco(blocos, y_bloco)
        except ValueError as e:
            st.error(str(e)); return
        st.success(f"RMSEP={resultado.rmsep:.4g}  R²cal={resultado.r2cal:.3f}  "
                  f"R²val={resultado.r2val:.3f}  n_lv={resultado.n_lv}  "
                  f"({resultado.n_cal} cal / {resultado.n_val} val)")


def render(pq, cfg_base, T: Callable[[str], str]) -> None:
    """Renderiza a aba Advanced Techniques."""
    st.subheader(T("Advanced Techniques"))
    st.caption(T(
        "🧪 ASCA, EPO/GLSW, MCR-ALS and multiblock fusion — exploratory/"
        "diagnostic techniques, not wired into the automatic training "
        "pipeline. ASCA/EPO-GLSW/MCR-ALS use the dataset configured in the "
        "Data tab; multiblock fusion asks for 2 folders on the spot."))

    aba_asca, aba_epo, aba_mcr, aba_fusao = st.tabs(
        ["ASCA", "EPO / GLSW", "MCR-ALS", T("Multiblock fusion")])
    with aba_asca:
        _secao_asca(pq, cfg_base, T)
    with aba_epo:
        _secao_epo_glsw(pq, cfg_base, T)
    with aba_mcr:
        _secao_mcr_als(pq, cfg_base, T)
    with aba_fusao:
        _secao_fusao_multibloco(pq, T)
