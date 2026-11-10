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

#: Paginas do web e a tela do catalogo que cada uma espelha. Os GRUPOS (rotulo e ordem)
#: vem do catalogo -- a mesma taxonomia por objetivo do menu do CLI (paridade testada).
PAGINAS_CATALOGO: List[tuple] = [
    (Pagina("projeto", "Projeto", "Project", "📋"), "projeto"),
    (Pagina("dados", "Dados", "Data", "📂"), "dados"),
    (Pagina("preprocessamento", "Pré-processamento", "Preprocessing", "⚗️"), "preproc"),
    (Pagina("tecnicas", "Técnicas especiais", "Special techniques", "🧪"), "tecnicas_avancadas"),
    (Pagina("modelo", "Modelo", "Model", "🧮"), "modelo"),
    (Pagina("validacao", "Validação", "Validation", "📊"), "validacao"),
    (Pagina("predicao", "Predição", "Prediction", "🔮"), "predicao"),
    (Pagina("relatorios", "Relatórios", "Reports", "📄"), "relatorios"),
    (Pagina("sobre", "Sobre", "About", "ℹ️"), "sobre"),
]

#: Telas do CLI sem pagina propria no web, com o motivo (tests/test_app_nav.py exige a lista completa).
TELAS_SO_NO_CLI: Dict[str, str] = {
    "rodar": "no web, rodar fica dentro da pagina Modelo",
    "exemplo": "exemplos geram arquivos e configuracao na maquina; uso por linha de comando",
    "perfis": "no web, os perfis ficam na pagina Dados",
    "salvar_config": "acao do terminal: grava a configuracao da sessao do CLI num arquivo YAML",
    "carregar_config": "acao do terminal: le um arquivo YAML para a sessao do CLI",
    "assistente": "assistente de texto do terminal; o web tem a ajuda ao lado de cada campo",
    "fluxo_planejamento": "fluxo guiado de terminal (planejar, selecionar, auditar)",
    "planejamento": "planejamento de coleta: so' no CLI nesta versao",
    "tecnica": "no web, a tecnica analitica entra pela pagina Dados (modo de entrada e perfil)",
    "codificacao": "preferencia de nomes de arquivo .dx: so' no CLI",
    "selecao_amostras": "selecao de amostras: so' no CLI nesta versao",
    "auditoria": "no web, a auditoria aparece no resultado da execucao",
    "checar": "pre-voo e ficha anti-vazamento do assistente: so' no CLI (tela [6.5] e `guaraci checar`) nesta versao",
    "avancado": "no web, benchmark e Monte Carlo ficam na pagina Modelo",
    "comparar": "ANOVA por linha de comando ou pelo CLI nesta versao",
    "monitorar": "monitoramento de lotes por linha de comando ou pelo CLI nesta versao",
    "hsi": "imageamento hiperespectral: pagina web ainda nao existe (decisao do autor)",
    "visualizacao": "no web e' uma pagina fixa do topo (Visualizacao)",
    "confianca": "selo e evidencia aparecem ao lado das opcoes no web",
    "ajuda": "no web, a ajuda fica ao lado de cada campo",
    "hardware": "diagnostico do computador local: so' no CLI",
}


def _grupos_do_catalogo() -> List[Grupo]:
    from guaraci import catalogo as _cat
    out: List[Grupo] = []
    for g in _cat.grupos():
        paginas = [p for p, tid in PAGINAS_CATALOGO if _cat.TELAS[tid]["grupo"] == g["id"]]
        if paginas:
            out.append(Grupo(len(out) + 1, g["nome"][0], g["nome"][1], paginas))
    return out


GRUPOS: List[Grupo] = _grupos_do_catalogo()

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
