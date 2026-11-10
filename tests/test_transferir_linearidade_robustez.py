"""`guaraci transferir` (DS/PDS), linearidade formal no N3 e protocolo de robustez -- tres tecnicas
do catalogo que eram importadas pelo motor mas nunca chamadas."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from guaraci import exemplos, transferir_cli
from guaraci.config import Config
from guaraci.robustness import protocolo_padrao

RAIZ = Path(__file__).resolve().parents[1]


def test_transferir_reduz_a_diferenca_e_grava_o_convertido(tmp_path):
    exemplos.gerar("transferir", str(tmp_path))
    for metodo in ("pds", "ds"):
        saida = tmp_path / f"r_{metodo}"
        assert transferir_cli.executar(["--mestre", str(tmp_path / "mestre.csv"), "--escravo", str(tmp_path / "escravo.csv"),
                                        "--novos", str(tmp_path / "novos_escravo.csv"), "--metodo", metodo,
                                        "--saida", str(saida)]) == 0
        assert (saida / "novos_no_espaco_do_mestre.csv").is_file() and (saida / "config_efetiva.yaml").is_file()
    import pandas as pd
    Xm = pd.read_csv(tmp_path / "mestre.csv").to_numpy()
    Xs = pd.read_csv(tmp_path / "escravo.csv").to_numpy()
    antes, depois = transferir_cli.erro_loo("pds", Xm, Xs, 5, 1.0)
    assert depois < antes / 3


def test_transferir_erros_de_uso(tmp_path, capsys):
    exemplos.gerar("transferir", str(tmp_path))
    assert transferir_cli.executar(["--mestre", str(tmp_path / "mestre.csv")]) == 2
    assert transferir_cli.executar(["--mestre", str(tmp_path / "mestre.csv"), "--escravo", str(tmp_path / "novos_escravo.csv"),
                                    "--novos", str(tmp_path / "novos_escravo.csv")]) == 2
    assert "mesmas amostras" in capsys.readouterr().err
    assert transferir_cli.executar(["--metodo", "xx"]) == 2
    assert transferir_cli.executar(["--ajuda"]) == 0


def test_protocolo_de_robustez_devolve_intervalo_por_perturbacao():
    rng = np.random.default_rng(0)
    eixo = np.linspace(0, 1, 60)
    conc = np.repeat(np.linspace(0, 10, 10), 3)
    X = np.array([np.exp(-((eixo - 0.5) / 0.1) ** 2) * (1 + 0.05 * c) + rng.normal(0, 0.005, 60) for c in conc])
    mae = np.repeat(np.arange(10), 3).astype(str)
    cfg = Config(default_preprocessing="mc")
    r = protocolo_padrao(X, conc, mae, cfg, "quantificacao", 3, n_replicas=2)
    assert any(k.startswith("ruido_gaussiano") for k in r) and any(k.startswith("deriva_linha_base") for k in r)
    assert all(np.isfinite(v.baseline) and v.minimo <= v.maximo for v in r.values())


@pytest.mark.slow
def test_n3_anexa_linearidade_e_robustez_ao_model_card(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    exemplos.gerar("quantificar", str(tmp_path / "ex"))
    cfg = tmp_path / "ex" / "config.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8") + "protocolo_robustez: true\n", encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "guaraci", "run", str(cfg)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, timeout=900)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    card = next((tmp_path / "ex").rglob("model_card.md")).read_text(encoding="utf-8")
    assert "Teste de falta de ajuste" in card and "Protocolo de robustez" in card and "ruido_gaussiano" in card


