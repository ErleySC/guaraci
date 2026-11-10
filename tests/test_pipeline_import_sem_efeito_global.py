"""M-15(a) do parecer de 2026-09-25: importar guaraci.pipeline nao troca o
backend do matplotlib nem silencia warnings do sklearn no processo inteiro."""
import pytest
import os
import subprocess
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]


def _py(codigo, **env_extra):
    env = dict(os.environ, **env_extra)
    env["PYTHONPATH"] = str(_RAIZ / "src")
    r = subprocess.run([sys.executable, "-c", codigo], env=env, capture_output=True,
                       text=True, cwd=str(_RAIZ), timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout.strip().splitlines()[-1]


def test_importar_nao_muda_o_backend():
    saida = _py("import matplotlib as m; import guaraci.pipeline; print(m.get_backend())",
                MPLBACKEND="svg")
    assert saida.lower() == "svg"


def test_importar_nao_silencia_warnings_do_sklearn():
    codigo = (
        "import warnings\n"
        "from sklearn.exceptions import ConvergenceWarning\n"
        "import guaraci.pipeline\n"
        "ig=[f for f in warnings.filters if f[0]=='ignore' and f[2] is ConvergenceWarning]\n"
        "print(len(ig))")
    assert _py(codigo) == "0"


@pytest.mark.slow
def test_executar_usa_agg_so_durante_a_execucao_e_restaura_o_backend():
    codigo = (
        "import matplotlib as m, warnings\n"
        "import guaraci.pipeline as pq, tempfile\n"
        "with tempfile.TemporaryDirectory() as d:\n"
        "    cfg=pq.Config(mode='sintetico', output_root_folder=d, show_plots=False,\n"
        "        n_permutations=3, run_benchmark=False, run_monte_carlo=False, run_shap=False,\n"
        "        run_wold=False, run_cv_anova=False, run_opls=False, run_ddsimca=False,\n"
        "        executar_etapa4=False, max_lvs=3, selecao_lv_cv_aninhada=False, frac_holdout=0.0)\n"
        "    pq.executar(cfg)\n"
        "print(m.get_backend())")
    assert _py(codigo, MPLBACKEND="svg").lower() == "svg"
