"""Cada perfil (objetivo x rigor) roda de ponta a ponta pelo CLI no exemplo sintetico do objetivo."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from guaraci import catalogo as cat

RAIZ = Path(__file__).resolve().parents[1]
COMBINACOES = [(o, r) for o, po in cat.PERFIS_OBJETIVO.items() if po["cfg"] for r in cat.PERFIS_RIGOR]


def test_toda_combinacao_tem_tempo_medido():
    assert set(COMBINACOES) == set(cat.TEMPO_PERFIL) and cat.TEMPO_PERFIL_MEDIDO_EM


def _yaml(v):
    return ("true" if v else "false") if isinstance(v, bool) else str(v)


@pytest.mark.slow
@pytest.mark.parametrize("objetivo,rigor", COMBINACOES)
def test_perfil_roda_de_ponta_a_ponta(tmp_path, objetivo, rigor):
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    po, pr = cat.PERFIS_OBJETIVO[objetivo], cat.PERFIS_RIGOR[rigor]
    pasta = tmp_path / "ex"
    subprocess.run([sys.executable, "-m", "guaraci", "exemplo", po["exemplo"], str(pasta)], env=env,
                   capture_output=True, timeout=600, check=True)
    chaves = dict(po["cfg"], **pr["cfg"])
    cfg = pasta / "config.yaml"
    linhas = [ln for ln in cfg.read_text(encoding="utf-8").splitlines() if ln.split(":")[0] not in chaves]
    cfg.write_text("\n".join(linhas + [f"{k}: {_yaml(v)}" for k, v in chaves.items()]) + "\n", encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "guaraci", "run", str(cfg)], env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-800:]
    assert list(pasta.rglob("resumo_modelo.txt")) and list(pasta.rglob("config_efetiva.yaml"))
