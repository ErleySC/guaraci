"""O menu, `--help`, `--version`, ajuda e assistentes nao importam o motor (sklearn/scipy/pandas).

Medido em 2026-10-03 (mediana de 4 a 5 execucoes, `python -m guaraci` ate aparecer o
menu): 5,44 s antes, 0,55 s depois. O tempo em si depende da maquina e nao e' testado;
o que e' testado e' a causa: nenhuma biblioteca pesada no caminho do menu.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
PESADOS = ("guaraci.pipeline", "sklearn", "scipy", "pandas", "matplotlib")


def _roda(codigo: str, entrada: str = "") -> str:
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-c", codigo], input=entrada, capture_output=True, text=True,
                       cwd=str(RAIZ), env=env, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout.strip().splitlines()[-1]


def _pesados_carregados(corpo: str, entrada: str = "") -> list:
    cod = ("import sys\n" + corpo + "\nprint(sorted(m for m in %r if m in sys.modules))" % (PESADOS,))
    return eval(_roda(cod, entrada))  # noqa: S307 -- saida controlada do proprio teste


def test_import_do_cli_nao_carrega_o_motor():
    assert _pesados_carregados("import guaraci.guaraci") == []


@pytest.mark.parametrize("args", [["--help"], ["--version"], ["ajuda", "dpi"], ["buscar", "censura"],
                                  ["confianca", "pls_da"], ["assistente", "pseudo-replicacao"]])
def test_comandos_leves_nao_carregam_o_motor(args):
    corpo = ("import io, contextlib, guaraci.guaraci as g\n"
             "with contextlib.redirect_stdout(io.StringIO()):\n"
             f"    g.main({args!r})")
    assert _pesados_carregados(corpo) == []


def test_menu_principal_e_telas_de_configuracao_nao_carregam_o_motor():
    corpo = ("import io, contextlib, builtins, guaraci.guaraci as g\n"
             "g._cls = lambda: None\n"
             "g._pause = lambda *a, **k: None\n"
             "it = iter(['2', 'v', '4', 'v', '5', 'v', 'c', '', 'q'])\n"
             "builtins.input = lambda *a, **k: next(it, 'q')\n"
             "with contextlib.redirect_stdout(io.StringIO()):\n"
             "    g.main([])")
    assert _pesados_carregados(corpo) == []


def test_a_execucao_continua_importando_o_motor_so_quando_precisa():
    corpo = "import guaraci.guaraci as g\ng.pq.executar"
    assert "guaraci.pipeline" in _pesados_carregados(corpo)
