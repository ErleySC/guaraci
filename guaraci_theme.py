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

# ---------------------------------------------------------------------------
# PALETA PROFISSIONAL — tons discretos, aspecto cientifico
# ---------------------------------------------------------------------------
PA = "#B8963E"   # Amber (ouro discreto — destaques)
PF = "#3D8B57"   # Forest (verde escuro — bordas, secoes)
PS = "#4A9E5C"   # Sage  (verde medio — substitui ciano)
PR = "#B85030"   # Rust  (laranja-vermelho — alertas)
PW = "#C8C8C8"   # Light (texto principal)
PM = "#686868"   # Muted (texto secundario)
PD = "#3A3A3A"   # Dim   (linhas de separacao)
PG = "#55B06A"   # Green (ok / sucesso)

# RGB de cada tom (para gerar escapes ANSI truecolor onde Rich nao e usado).
_RGB: Dict[str, tuple] = {
    "PA": (184, 150, 62), "PF": (61, 139, 87), "PS": (74, 158, 92),
    "PR": (184, 80, 48),  "PW": (200, 200, 200), "PM": (104, 104, 104),
    "PD": (58, 58, 58),   "PG": (85, 176, 106),
}

# ---------------------------------------------------------------------------
# DESIGN TOKENS — reexportados do modulo puro `design_tokens` (sem dependencias).
# Fonte unica de cor para web (Streamlit), figuras e CLI. Mantidos em modulo
# separado para que o app Streamlit possa importar tokens sem arrastar `rich`.
# ---------------------------------------------------------------------------
from design_tokens import TOKENS, tokens  # noqa: E402,F401  (re-export)


def ansi(tom: str) -> str:
    """Escape ANSI truecolor (\\033[38;2;R;G;Bm) para um tom da paleta.

    Permite recolorir codigo legado baseado em print()/ANSI com exatamente as
    mesmas cores do tema Rich, sem reescrever as telas.
    """
    r, g, b = _RGB.get(tom, (200, 200, 200))
    return f"\033[38;2;{r};{g};{b}m"


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"

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

console = Console(
    theme=THEME,
    highlight=False,
    legacy_windows=False,
    force_terminal=True,
)


# ---------------------------------------------------------------------------
# Largura responsiva
# ---------------------------------------------------------------------------
def _W() -> int:
    """Largura util do console (minimo 60, maximo 100)."""
    try:
        w = shutil.get_terminal_size((80, 24)).columns
    except Exception:
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
