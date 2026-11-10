# -*- coding: utf-8 -*-
"""O eixo Y das figuras de espectro nao presume absorbancia (sobra do V75).

O motor serve NIR, MIR, Raman, fluorescencia...: reflectancia, intensidade ou
absorbancia dependem do dado, que o perfil ainda nao declara (v1.1).
"""
from __future__ import annotations

from pathlib import Path


def test_previa_e_figuras_sem_absorbance_fixo():
    raiz = Path(__file__).resolve().parents[1] / "src" / "guaraci"
    for nome in ("spectra_preview.py", "figuras.py"):
        assert "Absorbance" not in (raiz / nome).read_text(encoding="utf-8"), nome
