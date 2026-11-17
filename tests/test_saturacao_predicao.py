# -*- coding: utf-8 -*-
"""Saturacao na predicao e nos artefatos do modelo (pacote, model_card.md, JSON portatil)."""
from __future__ import annotations

import copy
import glob
import os

import numpy as np
import pandas as pd
import pytest

import guaraci.pipeline as pl
import guaraci.predicao as pr
from guaraci.config import Config
from guaraci.saturacao import fracao_no_teto, sinalizar_saturacao_nova


def test_fracao_no_teto_ignora_regioes_ja_descartadas():
    wn = np.linspace(4000, 7000, 100)
    X = np.zeros((2, 100))
    X[0, 40:60] = 3.0                  # saturado so' na regiao que o modelo ja' descartou
    X[1, 10:60] = 3.0                  # saturado tambem fora dela
    reg = [[float(wn[40]), float(wn[59])]]
    f = fracao_no_teto(X, wn, 3.0, 0.01, reg)
    assert f[0] == 0.0 and f[1] > 0.3


def test_pacote_antigo_ou_sem_saturacao_devolve_none():
    assert sinalizar_saturacao_nova({}, np.zeros((1, 5)), np.arange(5.0)) is None
    assert sinalizar_saturacao_nova({"saturacao": None}, np.zeros((1, 5)), np.arange(5.0)) is None


def test_predizer_sinaliza_e_nao_aceita_espectro_no_teto(modelo_e_dados):
    pkg, X_novos, wn = modelo_e_dados
    pkg = copy.deepcopy(pkg)
    teto = float(np.max(X_novos)) + 1.0
    pkg["saturacao"] = {"teto": teto, "tolerancia": 0.01, "regioes": [], "aplicado": True}
    X = np.array(X_novos, dtype=float)
    X[0, :] = teto                     # espectro morto
    df = pr.predict_samples(pkg, X, wn)
    assert bool(df.loc[0, "saturado"]) and not bool(df.loc[0, "aceito"])
    assert "saturado" in df.loc[0, "criterio"]
    assert not df.loc[1:, "saturado"].any()


def test_pacote_sem_chave_nao_ganha_colunas(modelo_e_dados):
    pkg, X_novos, wn = modelo_e_dados
    assert "saturado" not in pr.predict_samples({k: v for k, v in pkg.items() if k != "saturacao"}, X_novos, wn).columns


def test_fluxo_cego_nao_quantifica_espectro_saturado(modelo_e_dados):
    pkg, X_novos, wn = modelo_e_dados
    pkg = copy.deepcopy(pkg)
    teto = float(np.max(X_novos)) + 1.0
    pkg["saturacao"] = {"teto": teto, "tolerancia": 0.01, "regioes": [], "aplicado": True}
    X = np.array(X_novos, dtype=float)
    X[0, :] = teto
    _, res = pr.predict_blind(pkg, X, wn)
    assert res[0].quantificacao.motivo_bloqueio == "espectro_saturado"
    assert res[0].quantificacao.teor_estimado is None


def test_treino_saturado_grava_teto_no_pacote_model_card_e_json(tmp_path):
    rng = np.random.default_rng(0)
    n, p = 40, 300
    X = 0.5 + 0.05 * rng.normal(size=(n, p)) + np.linspace(0, 1, p)
    X[:, 100:130] = 3.0
    cols = [f"{w:.1f}" for w in np.linspace(4000, 7000, p)]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "classe", ["A", "B"] * (n // 2))
    df.insert(1, "grupo", [f"g{i // 2}" for i in range(n)])
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    cfg = Config(mode="csv", csv_file=str(csv), class_column="classe", group_column="grupo", level="N1",
                 objective="classificacao", wn_min=4000.0, wn_max=7000.0,
                 output_root_folder=str(tmp_path / "saida"), n_splits_cv=2, n_repeats_cv=1,
                 n_permutations=5, n_permutations_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20,
                 n_monte_carlo=3, max_lvs=4)
    pl.executar(cfg)
    import joblib
    cards = glob.glob(str(tmp_path / "saida" / "**" / "model_card.md"), recursive=True)
    mods = glob.glob(str(tmp_path / "saida" / "**" / "modelo_plsda.joblib"), recursive=True)
    assert cards and mods
    assert "Saturacao do detector" in open(cards[0], encoding="utf-8").read()
    pkg = joblib.load(mods[0])
    assert pkg["saturacao"]["teto"] == pytest.approx(3.0, abs=0.01) and pkg["saturacao"]["regioes"]
    from guaraci.model_export import load_portable_json, predict_portable
    port = load_portable_json(os.path.join(os.path.dirname(mods[0]), "modelo_plsda.portatil.json"))
    assert port["saturacao"]["teto"] == pkg["saturacao"]["teto"]
    wn = np.asarray(pkg["wavenumbers"], dtype=float)
    Xn = np.full((2, len(wn)), 1.0)
    Xn[1, :] = 3.0
    saida = predict_portable(port, Xn, wn)
    assert saida["saturado"] == [False, True]
