"""Ramos de borda de guaraci.pipeline, parte B (regressao por especie e regressao pooled)."""
from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")

from guaraci import pipeline as pl  # noqa: E402
from guaraci.config import Config  # noqa: E402


def _dados(n_por_especie=24, p=12, seed=0, especies=("A", "B")):
    rng = np.random.default_rng(seed)
    rotulos = np.repeat(np.array(especies), n_por_especie)
    conc = np.tile(np.concatenate([np.zeros(6), np.linspace(1, 10, n_por_especie - 6)]), len(especies))
    X = rng.normal(size=(len(rotulos), p)) + conc[:, None] * 0.3
    mae = np.array([f"{r}-{i // 3:03d}-S{c:.2f}" for r, i, c in zip(rotulos, np.tile(np.arange(n_por_especie), len(especies)), conc)])
    return X, conc, rotulos, mae


def _cfg(**kw):
    base = dict(seed=0, max_lvs=3, n_splits_cv=2, frac_cal=0.7, default_preprocessing="mc")
    base.update(kw)
    return Config(**base)


def test_regressao_por_especie_pula_ausente_pouca_adulterada_e_sem_split_e_funciona_sem_grupo(tmp_path):
    X, conc, rot, mae = _dados()
    extra_X = np.random.default_rng(1).normal(size=(5, X.shape[1]))
    X = np.vstack([X, extra_X])
    conc = np.concatenate([conc, [0.0, 0.0, 2.0, 3.0, 4.0]])                    # espécie C: 5 amostras (3 adulteradas): sem split viavel
    rot = np.concatenate([rot, ["C"] * 5])
    classes = np.array(["A", "B", "C", "ausente"])
    r = pl.pls_regression_by_species(X, conc, rot, None, classes, _cfg(), str(tmp_path), n_splits=2, min_amostras_adult=3)
    assert r is not None and set(r["pipelines_especie"]) == {"A", "B"} and r["n_especies"] == 2


def test_regressao_por_especie_com_cv_que_falha_em_todas_devolve_none(monkeypatch, tmp_path):
    X, conc, rot, mae = _dados()

    def falha(*a, **k):
        raise ValueError("grupos insuficientes")
    monkeypatch.setattr(pl, "cross_val_predict", falha)
    assert pl.pls_regression_by_species(X, conc, rot, mae, np.array(["A", "B"]), _cfg(), str(tmp_path), n_splits=2) is None


def test_regressao_pooled_com_split_kennard_stone_e_protocolo_de_robustez(tmp_path):
    X, conc, rot, mae = _dados()
    pasta_logs = tmp_path / "Logs"
    os.makedirs(pasta_logs, exist_ok=True)
    cfg = _cfg(cal_val_split="kennard_stone", robustness_protocol=True)
    r = pl.pls_regressao_pooled(X, conc, rot, mae, cfg, str(tmp_path), str(pasta_logs), n_splits=2)
    assert r["robustez"] is not None and r["linearidade"] is not None and np.isfinite(r["rmsep"])


def test_limite_de_duplicatas_aproximadas_igual_as_exatas_nao_avisa_de_novo(caplog):
    rng = np.random.default_rng(3)
    base = rng.normal(size=(6, 40))
    X = np.vstack([base, base[:3]])                                          # 3 copias exatas
    rot = np.array(list("aabbccaab"))
    with caplog.at_level("INFO", logger=pl.log.name):
        *_, rel = pl.validate_input(X, np.arange(40.0), rot, None, None)
    assert rel["n_duplicatas_exatas"] == 3 and "samples with correlation" not in caplog.text


def test_duplicatas_exatas_de_linhas_constantes_nao_fazem_o_aviso_de_quase_duplicata_valer_de_novo(caplog):
    rng = np.random.default_rng(4)
    p = 40
    par = rng.normal(size=p)
    X = np.vstack([np.ones((5, p)), par, par + rng.normal(scale=1e-6, size=p), rng.normal(size=(3, p))])
    rot = np.array(list("aaaaabbbbbb")[:11])
    with caplog.at_level("INFO", logger=pl.log.name):
        *_, rel = pl.validate_input(X, np.arange(float(p)), rot, None, None)
    assert rel["n_duplicatas_exatas"] == 4 and rel["n_duplicatas_aproximadas"] == 2        # 2 <= 4: o aviso de replicas nao se repete
    assert "samples with corr" not in caplog.text


def test_regressao_por_especie_pula_a_especie_com_poucas_amostras_adulteradas(tmp_path):
    X, conc, rot, mae = _dados()
    X = np.vstack([X, np.random.default_rng(2).normal(size=(4, X.shape[1]))])
    conc = np.concatenate([conc, [0.0, 0.0, 1.0, 2.0]])                         # espécie D: so' 2 adulteradas (< 3)
    rot = np.concatenate([rot, ["D"] * 4])
    r = pl.pls_regression_by_species(X, conc, rot, None, np.array(["A", "D"]), _cfg(), str(tmp_path), n_splits=2, min_amostras_adult=3)
    assert set(r["pipelines_especie"]) == {"A"}
