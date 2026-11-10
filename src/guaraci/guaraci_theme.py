"""
guaraci_theme.py — Tema visual compartilhado da plataforma GUARACI.

Fonte unica da paleta de cores, do Console Rich e de helpers de UI, para que
guaraci.py e cli_assistente.py tenham exatamente o mesmo visual profissional
(sem duplicar constantes nem divergir de aparencia).

Importado por guaraci.py e cli_assistente.py. Nao importa nenhum dos dois
(evita import circular).
"""
from __future__ import annotations

import shutil
from typing import Dict

from rich.console import Console
from rich.theme import Theme

__all__ = [
    "PA", "PF", "PS", "PR", "PW", "PM", "PD", "PG",
    "TOKENS", "tokens",
    "ansi",
    "THEME",
    "console",
    "ok", "warn", "err", "info",
]

# ---------------------------------------------------------------------------
# PALETA DE IDENTIDADE — extraida por amostragem real de pixel da mascote
# (assets/guaraci_icon.png, ver docs/projeto/identidade_visual.md secao 1). PF (marca/bordas) e
# PA (destaque/aviso) migraram de forest/amber para laranja/dourado da
# mascote em 2026-09-01; PG (sucesso) e PR (alerta/erro) NAO mudaram de
# significado -- so' a cor de marca migrou, os status continuam verde/rust.
# ---------------------------------------------------------------------------
PA = "#FFC100"   # Dourado (mascote — destaque, aviso, bordas de identidade)
PF = "#FF7F2B"   # Laranja (mascote — marca/acao primaria, secoes)
PS = "#4A9E5C"   # Sage  (verde medio — substitui ciano)
PR = "#B85030"   # Rust  (laranja-vermelho — alertas/erro, distinto do laranja de marca)
PW = "#C8C8C8"   # Light (texto principal)
PM = "#686868"   # Muted (texto secundario)
PD = "#3A3A3A"   # Dim   (linhas de separacao)
PG = "#55B06A"   # Green (ok / sucesso)



def terminal_claro() -> bool:
    """Terminal de FUNDO CLARO? (M-18c: a paleta acima supoe fundo escuro e dava
    1,67:1 em fundo branco.) `GUARACI_TERMINAL=claro|escuro` decide; sem ela,
    usa a convencao `COLORFGBG="fg;bg"` (bg 7 ou 15 = claro)."""
    import os as _os
    v = _os.environ.get("GUARACI_TERMINAL", "").strip().lower()
    if v in ("claro", "light"):
        return True
    if v in ("escuro", "dark"):
        return False
    bg = _os.environ.get("COLORFGBG", "").split(";")[-1].strip()
    return bg in ("7", "15")


if terminal_claro():
    # Mesmos papeis, tons escuros: todos >= 5:1 sobre branco (medido com
    # design_tokens.contraste); PD (linhas) fica claro de proposito.
    PA, PF, PS, PR = "#8A6400", "#B34700", "#2E7040", "#9E3F22"
    PW, PM, PD, PG = "#1A1A1A", "#595959", "#BBBBBB", "#2F7A48"


def _hex_rgb(h: str) -> tuple:
    return tuple(int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))


# RGB de cada tom (para gerar sequencias de escape ANSI truecolor onde Rich nao e usado).
_RGB: Dict[str, tuple] = {k: _hex_rgb(v) for k, v in (("PA", PA), ("PF", PF), ("PS", PS), ("PR", PR), ("PW", PW), ("PM", PM), ("PD", PD), ("PG", PG))}

# ---------------------------------------------------------------------------
# DESIGN TOKENS — reexportados do modulo puro `design_tokens` (sem dependencias).
# Fonte unica de cor para web (Streamlit), figuras e CLI. Mantidos em modulo
# separado para que o app Streamlit possa importar tokens sem arrastar `rich`.
# ---------------------------------------------------------------------------
from guaraci.design_tokens import TOKENS, tokens  # noqa: E402,F401  (re-export)


def ansi(tom: str) -> str:
    """Escape ANSI truecolor (\\033[38;2;R;G;Bm) para um tom da paleta.

    Permite recolorir codigo legado baseado em print()/ANSI com exatamente as
    mesmas cores do tema Rich, sem reescrever as telas.
    """
    r, g, b = _RGB.get(tom, (200, 200, 200))
    return f"\033[38;2;{r};{g};{b}m"


_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_DIM = "\033[2m"

THEME = Theme({
    "a":     PA,           # amber / titulo
    "f":     PF,           # forest / ok
    "s":     PS,           # spectral / info
    "r":     PR,           # risk / alert
    "w":     PW,           # white text
    "m":     PM,           # muted
    "d":     PD,           # dim
    "g":     PG,           # green / success
    "ok":    f"bold {PG}",
    "warn":  f"bold {PA}",
    "err":   f"bold {PR}",
    "info":  PS,
    "hdr":   f"bold {PA}",
    "dim":   PM,
})

import os as _os_env
import sys as _sys

console = Console(
    theme=THEME,
    highlight=False,
    legacy_windows=False,
    # R5-16: forcar cor so' quando ha terminal de verdade; em pipe/arquivo (CI,
    # `> saida.txt`) saem sequencias ANSI cruas se for sempre True.
    force_terminal=(True if _sys.stdout.isatty() else None),
    # R5-16: NO_COLOR (https://no-color.org) tira TODO estilo, nao so' a cor
    # (o Rich mantinha negrito/italico).
    **({"color_system": None} if _os_env.environ.get("NO_COLOR") else {}),
)


# ---------------------------------------------------------------------------
# Largura responsiva
# ---------------------------------------------------------------------------
def _W() -> int:
    """Largura util do console (minimo 60, maximo 100)."""
    try:
        w = shutil.get_terminal_size((80, 24)).columns
    except (OSError, ValueError):   # COLUMNS/LINES malformado no ambiente
        w = 80
    return max(60, min(w, 100))


# ---------------------------------------------------------------------------
# Helpers de status — convencao unica de sucesso/aviso/erro/info
# ---------------------------------------------------------------------------
def ok(msg: str) -> None:
    """Linha de sucesso, prefixo verde."""
    console.print(f"[ok]✓[/ok] {msg}")


def warn(msg: str) -> None:
    """Linha de aviso, prefixo ambar."""
    console.print(f"[warn]⚠[/warn] {msg}")


def err(msg: str) -> None:
    """Linha de erro, prefixo rust."""
    console.print(f"[err]✗[/err] {msg}")


def info(msg: str) -> None:
    """Linha informativa, tom sage."""
    console.print(f"[info]{msg}[/info]")
