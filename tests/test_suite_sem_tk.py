"""A3-04/Tk (vistoria 2026-10): a suite nao depende de Tk. O conftest fixa MPLBACKEND=Agg (salvo se quem roda ja' exportou outro);
sem isso, num Python sem Tk utilizavel o matplotlib escolhe TkAgg sozinho e a primeira figura de um teste levanta TclError."""
from __future__ import annotations

import os
import subprocess
import sys


def test_figura_sai_mesmo_com_tk_quebrado():
    env = dict(os.environ, TCL_LIBRARY=os.path.join(os.sep, "nao", "existe", "tcl"), TK_LIBRARY=os.path.join(os.sep, "nao", "existe", "tk"))
    codigo = ("import matplotlib, matplotlib.pyplot as plt; f = plt.figure(); f.gca().plot([0, 1]); "
              "print(matplotlib.get_backend())")
    r = subprocess.run([sys.executable, "-c", codigo], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-600:]
