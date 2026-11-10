"""
design_tokens.py — Fonte UNICA de cor da plataforma GUARACI (web + CLI + figuras).

Modulo proposital-mente SEM dependencias (nem rich, nem streamlit): pode ser
importado pelo app Streamlit (deploy na nuvem via requirements.txt, que nao tem
rich) e pelos CLIs (guaraci.py / cli_assistente.py via guaraci_theme.py) sem
arrastar nenhuma biblioteca extra.

Tokens SEMANTICOS (bg, surface, text, primary...), nao widget-a-widget. Um tema
novo = adicionar uma chave em TOKENS; nada mais precisa mudar. Derivados da
mesma paleta de identidade (laranja/dourado, extraida por amostragem real de
pixel da mascote em assets/guaraci_icon.png -- ver docs/projeto/identidade_visual.md) usada pelo
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
        "primary":    "#C24A00",   # laranja (mascote) — cor de marca/acao; texto branco 4,9:1 (WCAG AA)
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
        "primary":    "#C24A00",   # laranja (mascote); o #FF8A3D antigo dava 2,35:1 com o texto branco do botao
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


def _lum(hexcor: str) -> float:
    h = hexcor.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contraste(a: str, b: str) -> float:
    """Razao de contraste WCAG entre duas cores '#RRGGBB' (1 a 21)."""
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def escurecer_para_contraste(hexcor: str, sobre: str = "#FFFFFF",
                             minimo: float = 4.5) -> str:
    """Escurece `hexcor` (mantendo o matiz) ate' o texto `sobre` ter contraste
    >= `minimo` (WCAG AA = 4,5). Devolve a cor original se ja' passa."""
    h = hexcor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    f = 1.0
    while contraste(f"#{int(r*f):02X}{int(g*f):02X}{int(b*f):02X}", sobre) < minimo and f > 0.05:
        f -= 0.02
    return f"#{int(r*f):02X}{int(g*f):02X}{int(b*f):02X}"
