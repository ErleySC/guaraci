"""app_tabs/sobre.py — Tela (About): identidade do projeto, diferenciais,
licença e como citar. Fonte da versão/licença: `pq.__version__` (config.py,
fonte única) — nunca hardcoded, para não repetir o drift de versão já
corrigido no resto do projeto (ver roadmap_mercado).
"""
from __future__ import annotations

from typing import Callable

import streamlit as st

from guaraci.config import (AUTOR_GITHUB_URL, AUTOR_NOME, AUTOR_ORCID, COPYRIGHT_LINHA, DISCUSSIONS_URL, LICENCA_SPDX,
                            REPO_URL, SEGURANCA_URL)

_REPO         = REPO_URL
_LICENCA      = LICENCA_SPDX
_AUTOR_NOME   = AUTOR_NOME
_AUTOR_ORCID  = f"https://orcid.org/{AUTOR_ORCID}"
_AUTOR_GITHUB = AUTOR_GITHUB_URL
_ANO          = "2026"


def _bibtex(versao: str) -> str:
    return (
        f"@software{{guaraci_{_ANO},\n"
        f"  author      = {{Costa, Erley Silva da}},\n"
        f"  title       = {{{{GUARACI: Plataforma quimiométrica com"
        f" validação anti-vazamento por padrão}}}},\n"
        f"  version     = {{{versao}}},\n"
        f"  year        = {{{_ANO}}},\n"
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
        "Plataforma quimiométrica com validação anti-vazamento por padrão"
        if pt else
        "Chemometrics platform with leakage-safe validation by default")
    c_b1, c_b2 = st.columns(2)
    c_b1.metric("Versão" if pt else "Version", versao)
    c_b2.metric("Licença" if pt else "License", _LICENCA)

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
            "Metodologia geral, aplicável a qualquer "
            "técnica analítica com dados multivariados."
        )
    else:
        st.markdown(
            "Democratize access to high-quality chemometric analyses "
            "for researchers without a programming background. Provides "
            "a reliable, reproducible and bilingual (PT/EN) environment "
            "for classification, authentication and exploration of "
            "complex matrices — from FT-NIR to GC-MS, without writing "
            "a single line of code.\n\n"
            "General methodology, applicable to any analytical "
            "technique with multivariate data."
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
            ("Validação anti-vazamento (group-aware)", "Padrão", "Disponível, configurada pelo usuário"),
            ("Reprodutibilidade (seeds, versionado)", "Sim", "Possível via script, não no fluxo padrão da GUI"),
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
            ("Leakage-safe validation (group-aware)", "Default", "Available, user-configured"),
            ("Reproducibility (seeds, versioned)", "Yes", "Possible via scripting, not the default GUI workflow"),
            ("Usable without coding", "Yes", "Yes (GUI)"),
            ("Bilingual PT / EN", "Yes", "Rare"),
            ("Ready-made reports (PDF/Word/PPTX)", "Yes", "Partial"),
            ("Runs offline, no mandatory cloud", "Yes", "Varies"),
        ]
        cols = ["Criterion", "GUARACI", "Paid*"]
    st.table({cols[0]: [r[0] for r in linhas],
              cols[1]: [r[1] for r in linhas],
              cols[2]: [r[2] for r in linhas]})
    st.caption(
        "Fontes: as mesmas da tabela \"Why it exists\" do README (documentação "
        "pública dos fabricantes, consultada em 2026-08; itens que exigem "
        "login não foram verificados). Esta tabela deve concordar com o README."
        if pt else
        "Sources: the same as the README's \"Why it exists\" table (vendors' "
        "public documentation, consulted 2026-08; items behind a login were "
        "not verified). This table must agree with the README.")
    st.caption(
        "* Faixa típica declarada pelos próprios fabricantes/usuários, "
        "não medida por este projeto. \"Validação anti-vazamento: "
        "Padrão\" é também por que alguns números deste projeto parecem "
        "mais modestos do que uma validação ingênua sugeriria — ver "
        "\"Por que os números aqui às vezes parecem modestos\" no "
        "README."
        if pt else
        "* Typical range as stated by the vendors/users themselves, not "
        "measured by this project. \"Leakage-safe validation: Default\" "
        "is also why some numbers in this project look more modest than "
        "a naive validation would suggest — see \"Why the numbers here "
        "sometimes look modest\" in the README.")

    st.divider()

    c_lic, c_cit = st.columns(2)
    with c_lic:
        st.markdown("#### " + ("Licença" if pt else "License"))
        st.markdown(
            f"**{_LICENCA}** — código aberto, uso e modificação livres "
            f"sob os termos da GPLv3, inclusive para uso comercial."
            if pt else
            f"**{_LICENCA}** — free and open source under the GPLv3 "
            f"terms, commercial use included.")
        st.caption(COPYRIGHT_LINHA)
        st.markdown(f"[LICENSE]({_REPO}/blob/master/LICENSE) · "
                     f"[Repositório]({_REPO})" if pt else
                     f"[LICENSE]({_REPO}/blob/master/LICENSE) · "
                     f"[Repository]({_REPO})")
    with c_cit:
        st.markdown("#### " + ("Autor / Contato" if pt else "Author / Contact"))
        st.markdown(f"**{_AUTOR_NOME}**  \n"
                     f"{'Pesquisador / Desenvolvedor' if pt else 'Researcher / Developer'}")
        st.markdown(f"[ORCID]({_AUTOR_ORCID}) · [GitHub]({_AUTOR_GITHUB}) · "
                     f"[{'Dúvidas e suporte (Discussions)' if pt else 'Questions and support (Discussions)'}]({DISCUSSIONS_URL}) · "
                     f"[{'Relatar vulnerabilidade (privado)' if pt else 'Report a vulnerability (private)'}]({SEGURANCA_URL})")

    st.divider()

    st.markdown("#### " + ("Como Citar" if pt else "How to Cite"))
    tit_full = ("GUARACI: Plataforma quimiométrica com validação"
                " anti-vazamento por padrão"
                if pt else
                "GUARACI: Chemometrics platform with leakage-safe"
                " validation by default")
    apa = (f"Costa, Erley Silva da. ({_ANO}). {tit_full} (v{versao})"
           f" [Software]. {_REPO}")
    abnt = (f"COSTA, Erley Silva da. GUARACI: Plataforma quimiométrica com "
            f"validação anti-vazamento por padrão. Versão {versao}. "
            f"{_ANO}. Disponível em: <{_REPO}>.")
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
