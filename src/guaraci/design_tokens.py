"""
design_tokens.py — Fonte UNICA de cor da plataforma GUARACI (web + CLI + figuras).

Modulo proposital-mente SEM dependencias (nem rich, nem streamlit): pode ser
importado pelo app Streamlit (deploy na nuvem via requirements.txt, que nao tem
rich) e pelos CLIs (guaraci.py / cli_assistente.py via guaraci_theme.py) sem
arrastar nenhuma biblioteca extra.

Tokens SEMANTICOS (bg, surface, text, primary...), nao widget-a-widget. Um tema
novo = adicionar uma chave em TOKENS; nada mais precisa mudar. Derivados da
mesma paleta de identidade (laranja/dourado, extraida por amostragem real de
pixel da mascote em assets/guaraci_icon.png -- ver docs/DESIGN.md) usada pelo
terminal, para que CLI, web e graficos pareçam o mesmo produto. `success`/
`warn`/`error` continuam verde/ambar/vermelho -- migrar a cor de marca nao
muda o significado das cores de status.
"""
from __future__ import annotations

from typing import Dict

__all__ = [
    "TOKENS",
    "tokens",
]

TOKENS: Dict[str, Dict[str, str]] = {
    "light": {
        "bg":         "#F7F9FB",   # fundo da pagina
        "surface":    "#FFFFFF",   # cartoes / paineis
        "text":       "#1A2B22",   # texto principal (verde-navy)
        "text_muted": "#5A6B62",   # texto secundario
        "primary":    "#D95700",   # laranja (mascote) — cor de marca/acao
        "accent":     "#B8850A",   # dourado (mascote) — destaque/borda
        "success":    "#2F7A48",   "success_bg": "#E6F2EA",
        "warn":       "#8A6D1E",   "warn_bg":    "#FBF3E0",
        "error":      "#9E3F22",   "error_bg":   "#F7E6E0",
        "border":     "#E2E8E4",
    },
    "dark": {
        "bg":         "#0F1613",
        "surface":    "#18211C",
        "text":       "#E6EDE8",
        "text_muted": "#9DB0A5",
        "primary":    "#FF8A3D",   # laranja (mascote), mais claro p/ fundo escuro
        "accent":     "#FFC100",   # dourado (mascote), tom vivido
        "success":    "#8FD69F",   "success_bg": "#16301F",
        "warn":       "#E0C978",   "warn_bg":    "#332B14",
        "error":      "#E39070",   "error_bg":   "#331C14",
        "border":     "#2A362F",
    },
}


def tokens(theme: str = "light") -> Dict[str, str]:
    """Retorna o conjunto de tokens do tema ('light' ou 'dark')."""
    return TOKENS.get(theme, TOKENS["light"])
