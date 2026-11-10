"""PARAFAC (EEM) e COW na tela de tecnicas especiais, e `guaraci relatorio`."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from guaraci import relatorio_cli
from guaraci.tecnicas_avancadas import rodar_cow, rodar_parafac_eem

RAIZ = Path(__file__).resolve().parents[1]


def _eem_sintetico(n=8, seed=0):
    rng = np.random.default_rng(seed)
    ex, em = np.linspace(250, 400, 12), np.linspace(300, 500, 20)
    A = rng.uniform(0.2, 1.0, (n, 2))
    Ex = np.stack([np.exp(-((ex - 300) / 30) ** 2), np.exp(-((ex - 360) / 25) ** 2)], 1)
    Em = np.stack([np.exp(-((em - 350) / 30) ** 2), np.exp(-((em - 450) / 35) ** 2)], 1)
    T = np.einsum("ir,jr,kr->ijk", A, Ex, Em)                 # amostra, excitacao, emissao
    X = np.array([T[i].T.ravel() for i in range(n)])          # como o leitor eem: (emissao, excitacao) achatada
    return X, list(ex), list(em), A


def test_parafac_recupera_os_escores_de_uma_mistura_sintetica():
    pytest.importorskip("tensorly")
    X, ex, em, A = _eem_sintetico()
    r = rodar_parafac_eem(X, ex, em, 2)
    assert r.erro_reconstrucao_relativo < 1e-3
    corr = np.abs(np.corrcoef(np.hstack([A, r.fator_amostra]).T)[:2, 2:])
    assert corr.max(axis=1).min() > 0.99                       # cada escore real tem um fator correspondente


def test_parafac_recusa_grade_errada_e_componentes_fora_da_faixa():
    X, ex, em, _ = _eem_sintetico()
    with pytest.raises(ValueError, match="mesma grade"):
        rodar_parafac_eem(X, ex[:-1], em, 2)
    with pytest.raises(ValueError, match="fora de"):
        rodar_parafac_eem(X, ex, em, 0)


def test_cow_aumenta_a_correlacao_de_picos_deslocados_e_escolhe_a_referencia_central():
    t = np.linspace(0, 10, 400)
    X = np.array([np.exp(-((t - 4 - 0.15 * i) / 0.3) ** 2) + np.exp(-((t - 7 + 0.1 * i) / 0.4) ** 2) for i in range(6)])
    r = rodar_cow(X, n_segmentos=20, slack=8)
    assert r.correlacao_depois > r.correlacao_antes + 0.2
    assert r.X_alinhado.shape == X.shape and r.indice_referencia in (2, 3)
    with pytest.raises(ValueError):
        rodar_cow(X[:1])


def test_relatorio_formatos_e_erros_de_uso(tmp_path, capsys):
    assert relatorio_cli.formatos_pedidos("todos") == list(relatorio_cli.FORMATOS)
    assert relatorio_cli.formatos_pedidos("pdf, xlsx") == ["pdf", "xlsx"]
    with pytest.raises(relatorio_cli._Uso):
        relatorio_cli.formatos_pedidos("odt")
    assert relatorio_cli.executar([str(tmp_path)]) == 2
    assert "nao parece a pasta de uma execucao" in capsys.readouterr().err
    assert relatorio_cli.executar(["--ajuda"]) == 0


@pytest.mark.slow
def test_relatorio_pela_cli_depois_de_uma_execucao_de_exemplo(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    def cli(*a):
        return subprocess.run([sys.executable, "-m", "guaraci", *a], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=env, cwd=str(tmp_path), timeout=600)
    assert cli("exemplo", "classificar", "ex", "--rodar").returncode == 0
    run = next((tmp_path / "ex").rglob("resumo_modelo.txt")).parents[1]
    r = cli("relatorio", str(run), "--formato", "todos")
    assert r.returncode == 0, r.stdout + r.stderr
    for ext in ("pdf", "docx", "xlsx", "tex", "pptx"):
        assert (run / "Relatorios" / f"relatorio.{ext}").stat().st_size > 1000
