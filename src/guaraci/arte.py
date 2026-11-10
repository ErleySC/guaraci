"""arte.py -- arte em braille do GUARACI (sol), mostrada na tela Sobre (6.3).

Fica fora da tela inicial de proposito: tem 20 linhas e empurraria o menu para
baixo da janela. Caracteres do bloco Unicode "Braille Patterns" (U+2800-U+28FF):
o Terminal do Windows (fonte Cascadia) mostra; o console antigo com Consolas
mostra quadrados. Por isso ha' como desligar (GUARACI_ARTE=0) e o modo linear
(leitor de tela) nunca a mostra.
"""
from __future__ import annotations

import os
import sys

SOL_BRAILLE = """\
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣿⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣾⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡄⠀⠀⠀⢻⣷⣦⠀⢸⠀⠀⠀⠀⢾⡿⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢦⣄⡀⠈⢻⣿⣀⠘⠃⠀⠀⣀⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠐⠢⢤⣄⡀⠀⠀⠀⠀⠀⠉⠋⢀⣼⣿⠿⠿⠬⠙⠻⣿⣿⠀⢀⡶⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠙⢿⣶⣶⣶⣦⣴⣶⠿⠿⠟⣦⡾⠾⣶⣤⣀⠈⠿⣾⡌⠀⠀⠀⢀⣴⣶⣤⣤⡶⠊⠀
⠀⠀⠀⠀⠀⠀⠀⠐⠚⠃⠀⡿⢃⣥⣴⣾⡿⠁⢀⣀⡙⢿⣦⣀⠈⠻⣶⣶⡾⠟⠃⠀⠉⠁⠀⠀⠀
⠀⠀⠀⠀⠐⠒⠲⠶⣦⣤⣤⡇⢸⣿⡅⣰⣷⣾⠿⠻⣿⣆⠉⣿⡇⠀⣡⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣠⣄⢾⣿⠁⠻⣿⣧⣤⠆⢸⣿⠀⣿⡇⢸⡿⠛⣷⡶⢲⡆⠀⠀⠀⠀⠀
⠀⠀⠀⢀⣀⣀⣀⣼⡿⠛⠛⣿⠀⢻⣷⣷⣜⣉⣁⣤⣾⢏⣠⡿⠋⢸⣧⣀⠀⠉⣑⡀⠀⠀⠀⠀⠀
⠠⠴⠚⠛⠛⠛⠛⠋⠀⢀⡠⠀⢦⣀⣀⡈⠙⠛⠻⠿⢁⣼⠟⠀⠀⠾⠛⠻⠿⠿⠿⢿⣆⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢀⡤⠛⠀⠀⣼⣿⡿⠛⠻⠿⢶⣶⡿⢋⣠⡴⠏⢀⠀⠀⠀⠀⠀⠈⠛⠛⠒⠦⠄
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⡏⠀⠀⠐⠚⠲⠶⣾⣿⡟⠀⠘⢺⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⡷⠀⠀⢠⠆⠀⠀⣿⣿⣷⡀⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⡿⠏⠀⠀⠀⡸⠀⠀⠀⠀⠉⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣗⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀"""

LARGURA = max(len(linha) for linha in SOL_BRAILLE.splitlines())


def pode_mostrar(largura_console: int, linear: bool) -> bool:
    """A arte so' aparece se couber, se o terminal codifica Unicode, se o modo
    linear (leitor de tela) estiver desligado e se GUARACI_ARTE nao for 0."""
    if linear or os.environ.get("GUARACI_ARTE", "1").strip() == "0":
        return False
    enc = (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "")
    if enc and not enc.startswith("utf"):
        return False
    return largura_console >= LARGURA + 6
