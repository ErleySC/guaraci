"""
design_tokens.py — Fonte UNICA de cor da plataforma GUARACI (web + CLI + figuras).

Modulo proposital-mente SEM dependencias (nem rich, nem streamlit): pode ser
importado pelo app Streamlit (deploy na nuvem via requirements.txt, que nao tem
rich) e pelos CLIs (guaraci.py / cli_assistente.py via guaraci_theme.py) sem
arrastar nenhuma biblioteca extra.

Tokens SEMANTICOS (bg, surface, text, primary...), nao widget-a-widget. Um tema
novo = adicionar uma chave em TOKENS; nada mais precisa mudar. Derivados da
mesma paleta profissional (forest/amber/rust) usada pelo terminal, para que
CLI, web e graficos pareçam o mesmo produto.
"""
from __future__ import annotations

from typing import Dict

TOKENS: Dict[str, Dict[str, str]] = {
    "light": {
        "bg":         "#F7F9FB",   # fundo da pagina
        "surface":    "#FFFFFF",   # cartoes / paineis
        "text":       "#1A2B22",   # texto principal (verde-navy)
        "text_muted": "#5A6B62",   # texto secundario
        "primary":    "#3D8B57",   # forest — cor de marca
        "accent":     "#B8963E",   # amber — destaque
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
        "primary":    "#55B06A",
        "accent":     "#C9A94E",
        "success":    "#8FD69F",   "success_bg": "#16301F",
        "warn":       "#E0C978",   "warn_bg":    "#332B14",
        "error":      "#E39070",   "error_bg":   "#331C14",
        "border":     "#2A362F",
    },
}


def tokens(theme: str = "light") -> Dict[str, str]:
    """Retorna o conjunto de tokens do tema ('light' ou 'dark')."""
    return TOKENS.get(theme, TOKENS["light"])
