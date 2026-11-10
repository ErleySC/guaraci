"""R3-08 (revisao cega 2026-09-26): quantificacao de UM material (uma classe)
nao imprime acuracia 1,0 / kappa nan como se fossem desempenho."""
import numpy as np
import pandas as pd

import guaraci.pipeline as pq
from tests._resumo import canonico


def test_resumo_de_uma_classe_marca_metricas_de_classificacao_como_na(tmp_path):
    rng = np.random.default_rng(0)
    wn = np.linspace(1100, 2498, 120)
    y = rng.uniform(5, 15, 60)
    base = np.exp(-((wn - 1700) / 80) ** 2)
    X = 1 + np.outer(y, base) * 0.05 + 0.01 * rng.standard_normal((60, 120))
    df = pd.DataFrame(X, columns=[f"{w:.1f}" for w in wn])
    df.insert(0, "classe", "adulterante M")
    df.insert(1, "conc", y)
    csv = tmp_path / "q.csv"
    df.to_csv(csv, index=False)
    cfg = pq.Config(mode="csv", csv_file=str(csv), class_column="classe", conc_column="conc",
                    objective="quantificacao", level="N3", group_by_mae_id=False,
                    output_root_folder=str(tmp_path / "o"), show_plots=False,
                    n_permutations=3, wn_min=1100, wn_max=2498, frac_holdout=0.25)
    pq.executar(cfg)
    txt = canonico(next((tmp_path / "o").rglob("resumo_modelo.txt")))
    linha = next(l for l in txt.splitlines() if l.strip().startswith("Accuracy (CV)"))
    assert "n/a (uma unica classe)" in linha
    assert "RMSEP" in txt
