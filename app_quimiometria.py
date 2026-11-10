# -*- coding: utf-8 -*-
"""Ponto de entrada fino do app web (Streamlit Cloud, Docker, CI, testes).

O corpo do app mora no pacote, em `src/guaraci/app_web.py`, para que
`guaraci web` funcione tambem a partir do pacote instalado por pip. Este
arquivo so' localiza esse corpo e o executa de novo a cada rerun do Streamlit
(um `import` simples rodaria o app uma unica vez). Uso a partir de uma copia
do repositorio, sem instalar: `streamlit run app_quimiometria.py`.
"""
import importlib.util
import os
import runpy
import sys

# Copia do repositorio sem `pip install`: o pacote `guaraci` esta em ./src.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_spec = importlib.util.find_spec("guaraci.app_web")
if _spec is None or not _spec.origin:
    raise RuntimeError("pacote guaraci nao encontrado: instale com "
                       "`pip install '.[web]'` ou rode a "
                       "partir de uma copia completa do repositorio.")
runpy.run_path(_spec.origin, run_name="__main__")
