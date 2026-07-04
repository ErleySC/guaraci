"""GUARACI — plataforma quimiométrica multitécnica (pacote).

A versão canônica vive em `guaraci.pipeline.__version__`; reexportada aqui
para permitir `from guaraci import __version__`.
"""
from __future__ import annotations

try:
    from guaraci.pipeline import __version__  # noqa: F401
except Exception:  # pragma: no cover - durante bootstrap parcial de import
    __version__ = "unknown"
