"""app_nav.py — Definição PURA da navegação do app web (sem Streamlit).

A interface passou de 8 abas horizontais para navegação lateral agrupada
(mockup de 2026-09-08, instrução "siga à risca"): dois itens fixos no topo
(Início, Visualização) e quatro grupos numerados — Preparar, Executar,
Analisar, Referência — exatamente os mesmos agrupamentos que antes eram só
uma legenda acima das abas.

Aqui ficam só os DADOS da navegação (ids, rótulos, ordem, grupos), sem
`import streamlit`, para que a estrutura seja testável em isolamento e para
que a UI não tenha a lista duplicada em dois lugares.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional


class Pagina(NamedTuple):
    """Uma tela do app. `chave` é o id interno usado em `session_state`."""
    chave: str
    rotulo_pt: str
    rotulo_en: str
    icone: str

    def rotulo(self, pt: bool) -> str:
        return self.rotulo_pt if pt else self.rotulo_en


class Grupo(NamedTuple):
    numero: int
    rotulo_pt: str
    rotulo_en: str
    paginas: List[Pagina]

    def rotulo(self, pt: bool) -> str:
        return self.rotulo_pt if pt else self.rotulo_en


# Itens fixos, fora dos grupos (topo da barra lateral).
PAGINAS_FIXAS: List[Pagina] = [
    Pagina("inicio", "Início", "Home", "◆"),
    Pagina("visualizacao", "Visualização", "Visualisation", "🎨"),
]

GRUPOS: List[Grupo] = [
    Grupo(1, "Preparar", "Prepare", [
        Pagina("projeto", "Projeto", "Project", "📋"),
        Pagina("dados", "Dados", "Data", "📂"),
        Pagina("preprocessamento", "Pré-processamento", "Preprocessing", "⚗️"),
    ]),
    Grupo(2, "Executar", "Run", [
        Pagina("modelo", "Modelo", "Model", "🧮"),
    ]),
    Grupo(3, "Analisar", "Analyse", [
        Pagina("validacao", "Validação", "Validation", "📊"),
        Pagina("predicao", "Predição", "Prediction", "🔮"),
        Pagina("relatorios", "Relatórios", "Reports", "📄"),
    ]),
    Grupo(4, "Referência", "Reference", [
        Pagina("sobre", "Sobre", "About", "ℹ️"),
    ]),
]

# Subtítulo da barra superior, por tela.
SUBTITULOS: Dict[str, Dict[str, str]] = {
    "inicio": {
        "PT": "Painel de status — resume o estado atual do projeto",
        "EN": "Status panel — summarises the current state of the project",
    },
    "visualizacao": {
        "PT": "Personalize as cores dos gráficos — aplica-se a todo o relatório",
        "EN": "Customise figure colours — applies to the whole report",
    },
}

PAGINA_INICIAL = "inicio"


def todas_paginas() -> List[Pagina]:
    """Todas as telas, na ordem em que aparecem na barra lateral."""
    paginas = list(PAGINAS_FIXAS)
    for grupo in GRUPOS:
        paginas.extend(grupo.paginas)
    return paginas


def pagina_por_chave(chave: str) -> Optional[Pagina]:
    for pagina in todas_paginas():
        if pagina.chave == chave:
            return pagina
    return None


def grupo_da_pagina(chave: str) -> Optional[Grupo]:
    """Grupo que contém a tela (para deixar o grupo aberto ao entrar nela)."""
    for grupo in GRUPOS:
        if any(p.chave == chave for p in grupo.paginas):
            return grupo
    return None


def subtitulo(chave: str, pt: bool) -> str:
    """Subtítulo da barra superior; vazio quando a tela não define um."""
    return SUBTITULOS.get(chave, {}).get("PT" if pt else "EN", "")
