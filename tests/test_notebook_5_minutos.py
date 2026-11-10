"""B5 do parecer de 2026-09-25: o notebook anunciado ("zero local install")
quebrava com TypeError porque usava nomes de campo anteriores a renomeacao PT->EN.

Executa as celulas de codigo do notebook pelo pacote do repositorio (sem
`!pip`/`!guaraci`, que dependem do Colab; e sem `display` de imagem).
"""
import pytest
import json
import os
import subprocess
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
_NB = _RAIZ / "notebooks" / "guaraci_5_minutos.ipynb"


def _celulas_de_codigo():
    nb = json.loads(_NB.read_text(encoding="utf-8"))
    out = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        linhas = [l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith("!")]
        codigo = "\n".join(linhas).strip()
        if codigo and "IPython.display" not in codigo:
            out.append(codigo)
    return out


@pytest.mark.slow
def test_notebook_executa_de_ponta_a_ponta(tmp_path):
    programa = "\n\n".join(_celulas_de_codigo())
    env = dict(os.environ, PYTHONPATH=str(_RAIZ / "src"), MPLBACKEND="Agg",
               HOME=str(tmp_path), USERPROFILE=str(tmp_path), PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-c", programa], cwd=str(tmp_path), env=env,
                       capture_output=True, text=True, encoding="utf-8", timeout=900)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "Resultados em:" in r.stdout
    assert any((tmp_path / "GUARACI_Demo").rglob("resumo_modelo.txt"))


def test_notebook_nao_usa_nomes_de_campo_antigos():
    texto = "\n".join(_celulas_de_codigo())
    for velho in ("pasta_entrada", "pasta_saida_raiz", "modo=", "executar_benchmark",
                  "n_permutacoes", "sint_adulterantes"):
        assert velho not in texto, velho
