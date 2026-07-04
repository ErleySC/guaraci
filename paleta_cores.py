"""
paleta_cores.py — Paleta de cores e marcadores de máxima distintividade
perceptual para gráficos quimiométricos (classes/espécies).

Extraído de pipeline.py como parte da modularização (Fase H). Funções PURAS:
dependem só de matplotlib.colors/pyplot, sem acoplamento a Config. pipeline.py
reexporta estes nomes, então `pipeline.cor(...)`, `pipeline.PALETA` etc.
continuam funcionando sem alteração.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# MAXIMUM PERCEPTUAL DISTINCTIVENESS palette (base: Trubetskoy "20 distinct
# colors" + Glasbey). Ordered by contrast on white background: strong/
# saturated colors first, light colors last. NO near-identical blues/greens.
# For printing/colorblindness the SHAPE channel (MARCADORES) complements color.
PALETA = [
    "#E6194B",  # red
    "#4363D8",  # blue
    "#3CB44B",  # green
    "#F58231",  # orange
    "#911EB4",  # purple
    "#42D4F4",  # cyan
    "#F032E6",  # magenta
    "#9A6324",  # brown
    "#469990",  # teal
    "#800000",  # maroon
    "#808000",  # olive
    "#000075",  # navy
    "#E6A000",  # amber/gold
    "#BFEF45",  # lime
    "#FABED4",  # light pink
    "#DCBEFF",  # lavender
    "#AAFFC3",  # mint
    "#FFD8B1",  # peach
    "#A9A9A9",  # gray
    "#FFE119",  # yellow
]

# Secondary channel: marker shapes (identity by SHAPE, robust to
# colorblindness/B&W printing). 14 distinct shapes before repeating.
MARCADORES = ["o", "s", "^", "D", "v", "P", "X", "*",
              "<", ">", "h", "p", "8", "d"]


def _paleta_externa(n: int) -> Optional[List[str]]:
    """Tries to generate a max-distinctiveness palette via optional libs (glasbey,
    colorcet). Returns a list of hex colors or None if none available."""
    try:
        import glasbey as _gb  # type: ignore
        return list(_gb.create_palette(palette_size=n))
    except Exception:
        pass
    try:
        import colorcet as _cc  # type: ignore
        base = _cc.glasbey_category10
        return [base[i % len(base)] for i in range(n)]
    except Exception:
        pass
    return None


def _luminancia(hex_cor: str) -> float:
    """Relative luminance (0=dark, 1=light) to decide edge color."""
    r, g, b = mcolors.to_rgb(hex_cor)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def edge_para_cor(hex_cor: str) -> str:
    """Smart edge color: dark gray for light fills (visible on white
    background), white for dark fills."""
    return "0.25" if _luminancia(hex_cor) > 0.65 else "white"


def cor(i: int) -> str:
    """Color i from the maximum-distinctiveness palette. Beyond palette size,
    uses external lib (if available) or cycles with slight luminance variation via HSV."""
    if i < len(PALETA):
        return PALETA[i]
    ext = _paleta_externa(i + 1)
    if ext is not None and i < len(ext):
        return mcolors.to_hex(ext[i])
    # fallback: shifted tab20
    cmap = plt.get_cmap("tab20")
    return mcolors.to_hex(cmap(((i - len(PALETA)) % 20) / 20))


def mapear_cores_classes(classes) -> Dict[str, str]:
    """Assigns color in alphabetical order from the maximum-distinctiveness
    palette. SEQUENTIAL (non-hash) assignment ensures adjacent classes
    receive well-separated colors — the palette is already ordered to
    maximize contrast between neighboring indices. Deterministic."""
    classes_sorted = sorted({str(c) for c in classes})
    n = len(classes_sorted)
    externa = _paleta_externa(n) if n > len(PALETA) else None
    mapa: Dict[str, str] = {}
    for idx, cls in enumerate(classes_sorted):
        if externa is not None:
            mapa[cls] = mcolors.to_hex(externa[idx])
        else:
            mapa[cls] = cor(idx)
    return mapa


def mapear_marcadores_classes(classes) -> Dict[str, str]:
    """Assigns marker shape per class (secondary channel). Combined
    with color, ensures distinctiveness even in B&W/colorblindness and high density."""
    classes_sorted = sorted({str(c) for c in classes})
    return {cls: MARCADORES[i % len(MARCADORES)]
            for i, cls in enumerate(classes_sorted)}
