"""app_tabs/sobre.py — Aba 8 (About): identidade do projeto, diferenciais,
licença e como citar. Fonte da versão/licença: `pq.__version__` (config.py,
fonte única) — nunca hardcoded, para não repetir o drift de versão já
corrigido no resto do projeto (ver roadmap_mercado).
"""
from __future__ import annotations

from typing import Callable

import streamlit as st

_REPO         = "https://github.com/ErleySC/guaraci"
_LICENCA      = "GPL-3.0-or-later"
_INST         = "GEAAp/UFPA"
_AUTOR_NOME   = "Erley S. da Costa"
_AUTOR_LATTES = "http://lattes.cnpq.br/5755582193284309"
_AUTOR_GITHUB = "https://github.com/ErleySC"
_AUTOR_EMAIL  = "erleysdacosta@gmail.com"
_ANO          = "2026"


def _bibtex(versao: str) -> str:
    return (
        f"@software{{guaraci_{_ANO},\n"
        f"  author      = {{Costa, Erley S. da}},\n"
        f"  title       = {{{{GUARACI: Inteligencia Quimiometrica"
        f" para Matrizes Amazonicas}}}},\n"
        f"  version     = {{{versao}}},\n"
        f"  year        = {{{_ANO}}},\n"
        f"  institution = {{{_INST}}},\n"
        f"  url         = {{{_REPO}}},\n"
        f"  license     = {{{_LICENCA}}}\n"
        f"}}"
    )


def render(pq, T: Callable[[str], str]) -> None:
    """Renderiza a aba About. `T` é a função de tradução `_T` do app."""
    pt = st.session_state.get("lang") == "PT"
    versao = getattr(pq, "__version__", "?")

    st.markdown("### GUARACI")
    st.caption(
        "Inteligência Quimiométrica para Matrizes Amazônicas"
        if pt else
        "Chemometric Intelligence for Amazonian Matrices")
    c_b1, c_b2, c_b3 = st.columns(3)
    c_b1.metric("Versão" if pt else "Version", versao)
    c_b2.metric("Licença" if pt else "License", _LICENCA)
    c_b3.metric("Instituição" if pt else "Institution", _INST)

    st.divider()

    st.markdown("#### " + ("Propósito" if pt else "Purpose"))
    if pt:
        st.markdown(
            "Democratizar o acesso a análises quimiométricas de alta "
            "qualidade para pesquisadores que não dominam programação. "
            "Oferece um ambiente confiável, reprodutível e bilíngue "
            "(PT/EN) para classificação, autenticação e exploração de "
            "matrizes complexas — do FT-NIR ao GC-MS, sem escrever uma "
            "linha de código.\n\n"
            "Desenvolvido no âmbito de uma pesquisa PIBIC/UFPA sobre "
            "óleos vegetais amazônicos, com metodologia generalizável "
            "para qualquer técnica analítica com dados multivariados."
        )
    else:
        st.markdown(
            "Democratize access to high-quality chemometric analyses "
            "for researchers without a programming background. Provides "
            "a reliable, reproducible and bilingual (PT/EN) environment "
            "for classification, authentication and exploration of "
            "complex matrices — from FT-NIR to GC-MS, without writing "
            "a single line of code.\n\n"
            "Developed within a PIBIC/UFPA research project on Amazonian "
            "vegetable oils, with a methodology generalized to any "
            "analytical technique with multivariate data."
        )
    st.caption(
        "Técnicas: FT-NIR · NIR · MIR/FTIR · Raman · UV-Vis · "
        "Fluorescência · HPLC · GC-MS · NMR · IMS · Genérica"
        if pt else
        "Techniques: FT-NIR · NIR · MIR/FTIR · Raman · UV-Vis · "
        "Fluorescence · HPLC · GC-MS · NMR · IMS · Generic")

    st.divider()

    st.markdown("#### " + ("Por que o GUARACI?" if pt else "Why GUARACI?"))
    st.caption(
        "Comparativo informativo com softwares comerciais como "
        "MATLAB/PLS_Toolbox, The Unscrambler, SIMCA e similares."
        if pt else
        "Informative comparison with commercial software such as "
        "MATLAB/PLS_Toolbox, The Unscrambler, SIMCA and similar.")
    if pt:
        linhas = [
            ("Custo de licença", "Gratuito", "Alto"),
            ("Código aberto / auditável", "Sim", "Não"),
            ("Validação anti-vazamento (group-aware)", "Padrão", "Manual"),
            ("Reprodutibilidade (seeds, versionado)", "Sim", "Parcial"),
            ("Uso sem programar", "Sim", "Sim (GUI)"),
            ("Bilíngue PT / EN", "Sim", "Raro"),
            ("Relatórios prontos (PDF/Word/PPTX)", "Sim", "Parcial"),
            ("Roda offline, sem nuvem obrigatória", "Sim", "Varia"),
        ]
        cols = ["Critério", "GUARACI", "Pagos*"]
    else:
        linhas = [
            ("License cost", "Free", "High"),
            ("Open source / auditable", "Yes", "No"),
            ("Leakage-safe validation (group-aware)", "Default", "Manual"),
            ("Reproducibility (seeds, versioned)", "Yes", "Partial"),
            ("Usable without coding", "Yes", "Yes (GUI)"),
            ("Bilingual PT / EN", "Yes", "Rare"),
            ("Ready-made reports (PDF/Word/PPTX)", "Yes", "Partial"),
            ("Runs offline, no mandatory cloud", "Yes", "Varies"),
        ]
        cols = ["Criterion", "GUARACI", "Paid*"]
    st.table({cols[0]: [r[0] for r in linhas],
              cols[1]: [r[1] for r in linhas],
              cols[2]: [r[2] for r in linhas]})

    st.divider()

    c_lic, c_cit = st.columns(2)
    with c_lic:
        st.markdown("#### " + ("Licença" if pt else "License"))
        st.markdown(
            f"**{_LICENCA}** — código aberto, uso e modificação livres "
            f"sob os termos da GPLv3. Uso comercial dual-license: ver "
            f"[COMMERCIAL.md]({_REPO}/blob/master/docs/COMMERCIAL.md)."
            if pt else
            f"**{_LICENCA}** — free and open source under the GPLv3 "
            f"terms. Dual-license for commercial use: see "
            f"[COMMERCIAL.md]({_REPO}/blob/master/docs/COMMERCIAL.md).")
        st.markdown(f"[LICENSE]({_REPO}/blob/master/LICENSE) · "
                     f"[Repositório]({_REPO})" if pt else
                     f"[LICENSE]({_REPO}/blob/master/LICENSE) · "
                     f"[Repository]({_REPO})")
    with c_cit:
        st.markdown("#### " + ("Autor / Contato" if pt else "Author / Contact"))
        st.markdown(f"**{_AUTOR_NOME}**  \n"
                     f"{'Pesquisador / Desenvolvedor' if pt else 'Researcher / Developer'} "
                     f"— {_INST}")
        st.markdown(f"[Lattes]({_AUTOR_LATTES}) · [GitHub]({_AUTOR_GITHUB}) · "
                     f"[{_AUTOR_EMAIL}](mailto:{_AUTOR_EMAIL})")

    st.divider()

    st.markdown("#### " + ("Como Citar" if pt else "How to Cite"))
    tit_full = ("GUARACI: Inteligência Quimiométrica para Matrizes Amazônicas"
                if pt else
                "GUARACI: Chemometric Intelligence for Amazonian Matrices")
    apa = (f"Costa, E. S. da. ({_ANO}). {tit_full} (v{versao})"
           f" [Software]. {_INST}. {_REPO}")
    abnt = (f"COSTA, E. S. da. GUARACI: Inteligência Quimiométrica para "
            f"Matrizes Amazônicas. Versão {versao}. {_INST}, {_ANO}. "
            f"Disponível em: <{_REPO}>.")
    with st.expander("APA", expanded=True):
        st.code(apa, language=None)
    with st.expander("ABNT (NBR 6023:2018)"):
        st.code(abnt, language=None)
    with st.expander("BibTeX"):
        st.code(_bibtex(versao), language="bibtex")
    st.caption(
        "Detalhes completos em CITATION.cff (raiz do repositório)."
        if pt else
        "Full details in CITATION.cff (repository root).")
