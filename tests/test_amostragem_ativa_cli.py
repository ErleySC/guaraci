"""Amostragem ativa: selo `limitado` no catalogo e prova de ponta a ponta pelo CLI (tela J).

Sem dado real: o mae_id sintetico reproduz o formato do modo dx (codigo da especie, data da
coleta com o ano sentinela 2099 e token de teor do adulterante), o unico formato em que a
identificacao sabe contar sessoes e adulterantes.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from guaraci import catalogo as cat

RAIZ = Path(__file__).resolve().parents[1]


def test_selo_limitado_nao_use_e_so_evidencia_que_vai_junto():
    t = cat.TECNICAS["amostragem_ativa"]
    assert t["selo"] == "amostragem_ativa"
    assert cat.SELOS["amostragem_ativa"]["estado"] == "limitado"
    assert "modo dx" in t["nao_use"][0] and "dx-mode" in t["nao_use"][1]
    c = cat.confianca("tecnica", "amostragem_ativa")
    assert c["evidencias"] and all(e["tipo"] in ("simulacao", "testes") for e in c["evidencias"])
    assert "ponto de partida" in c["calibracao"]


def _mae_id_formato_dx(especie_idx: int, sessao: int, teor: float) -> str:
    return "S" + str(especie_idx).zfill(2) + "-0" + str(sessao) + "-03-" + "2099" + "-A" + f"{teor:.2f}"


@pytest.mark.slow
def test_tarefa_amostragem_ativa_pela_tela_j_com_mae_id_no_formato_dx(tmp_path):
    import pandas as pd
    from tests.test_tarefas_cli import _roda
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "USERPROFILE": str(tmp_path / "home"), "HOME": str(tmp_path / "home")}
    (tmp_path / "home").mkdir()
    subprocess.run([sys.executable, "-m", "guaraci", "exemplo", "quantificar", "q"], capture_output=True, env=env,
                   cwd=str(tmp_path), timeout=300, check=True)
    csv = tmp_path / "q" / "dados.csv"
    d = pd.read_csv(csv, sep=None, engine="python")
    especies = sorted(d["classe"].unique())
    lotes = sorted(d["lote"].unique())
    d["lote"] = [_mae_id_formato_dx(especies.index(c), 1 + lotes.index(lt) % 2, t)
                 for c, lt, t in zip(d["classe"], d["lote"], d["conc"])]
    d.to_csv(csv, index=False)
    r = subprocess.run([sys.executable, "-m", "guaraci", "run", "q/config.yaml"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, cwd=str(tmp_path), timeout=600)
    assert r.returncode == 0, r.stdout[-800:]
    modelo = next((tmp_path / "q").rglob("*.joblib"))
    roteiro = ["2.7", "A,B", "3", "C", "0.10", str(tmp_path / "plano"), "n", "s", str(modelo), "s", "", "q", "S"]
    assert len(roteiro) <= 13
    saida, _, rc = _roda(roteiro, tmp_path, timeout=300)
    assert rc == 0, saida[-800:]
    assert "Faltam" in saida and "nada a priorizar" not in saida
    for esp in especies:
        assert esp in saida
