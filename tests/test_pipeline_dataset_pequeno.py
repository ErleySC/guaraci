# -*- coding: utf-8 -*-
"""Regressao (achado do roteiro de teste externo, 2026-09-23): com o
default `max_lvs=40` e um dataset pequeno (60 amostras -> folds de treino
de 38), a etapa [2/7] tentava PLSRegression(n_components=39) e a execucao
inteira falhava com "n_components upper bound is 38". Nenhum teste
anterior usava max_lvs default com dado pequeno (todos passavam
max_lvs=4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from guaraci.config import Config
from guaraci.pipeline import executar


def _csv_pequeno(caminho, n_por_classe=20, p=120, seed=7):
    rng = np.random.default_rng(seed)
    wn = np.linspace(4000, 9000, p)
    picos = {"A": (5200, 7000), "B": (5600, 7400), "C": (4800, 8200)}
    linhas, rot = [], []
    for c, (p1, p2) in picos.items():
        base = np.exp(-((wn - p1) / 250) ** 2) + 0.7 * np.exp(-((wn - p2) / 300) ** 2)
        for _ in range(n_por_classe):
            linhas.append(rng.normal(1.0, 0.05) * base + rng.normal(0, 0.01, p))
            rot.append(c)
    df = pd.DataFrame(linhas, columns=[f"{w:.2f}" for w in wn])
    df["classe"] = rot
    df.to_csv(caminho, index=False)


def test_max_lvs_default_nao_estoura_em_dataset_pequeno(tmp_path):
    csv = tmp_path / "pequeno.csv"
    _csv_pequeno(csv)
    cfg = Config(
        mode="csv", csv_file=str(csv), class_column="classe",
        wn_min=4000.0, wn_max=9000.0,
        output_root_folder=str(tmp_path / "saida"),
        n_splits_cv=5, n_repeats_cv=1, n_permutations=3,
        n_permutations_wold=3, n_bootstrap_vip=2, n_bootstrap_bca=5,
        n_monte_carlo=2, run_benchmark=False, run_monte_carlo=False,
        run_shap=False, run_wold=False, run_cv_anova=False, run_opls=False,
        executar_etapa4=False, show_plots=False,
        selecao_lv_cv_aninhada=False)
    assert cfg.max_lvs == 40          # o default e' o que quebrava
    executar(cfg)                      # nao deve levantar
