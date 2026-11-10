# -*- coding: utf-8 -*-
"""`dados_io._split_cal_val_por_especie` -- fonte unica do split cal/val por
especie (extraida em 2026-09-19 de duas copias inline de ~36 linhas em
`pipeline.pls_regression_by_species` e `avaliacao_modelos.benchmark_
regression_by_species`)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from guaraci.dados_io import _split_cal_val_por_especie


def _cfg(split="aleatorio", frac=0.7, seed=0):
    return SimpleNamespace(cal_val_split=split, frac_cal=frac, seed=seed)


def _dados(n=24, p=6, grupos=True, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    Y = rng.normal(size=(n, 1))
    mae = np.repeat(np.arange(n // 3), 3).astype(str) if grupos else None
    return X, Y, mae


def test_replicas_do_mesmo_grupo_nunca_sao_separadas():
    X, Y, mae = _dados()
    for split in ("aleatorio", "kennard_stone"):
        ic, iv = _split_cal_val_por_especie(X, Y, mae, _cfg(split))
        assert not set(mae[ic]) & set(mae[iv]), split
        assert sorted(np.concatenate([ic, iv])) == list(range(len(X)))


def test_split_e_deterministico_para_a_mesma_semente():
    X, Y, mae = _dados()
    a = _split_cal_val_por_especie(X, Y, mae, _cfg(seed=3))
    b = _split_cal_val_por_especie(X, Y, mae, _cfg(seed=3))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_sem_mae_id_usa_permutacao_simples_com_fracao_pedida():
    X, Y, _ = _dados(n=20, grupos=False)
    ic, iv = _split_cal_val_por_especie(X, Y, None, _cfg(frac=0.7))
    assert len(ic) == 14 and len(iv) == 6


def test_poucos_grupos_cai_na_permutacao_simples():
    X, Y, _ = _dados(n=20)
    mae = np.repeat(["a", "b", "c"], [7, 7, 6])       # 3 grupos (< 4)
    r = _split_cal_val_por_especie(X, Y, mae, _cfg())
    assert r is not None and len(r[0]) + len(r[1]) == 20


def test_especie_pequena_demais_devolve_none():
    X, Y, _ = _dados(n=5, grupos=False)               # 5 amostras: val < 2
    assert _split_cal_val_por_especie(X, Y, None, _cfg(frac=0.9)) is None
    X2, Y2, _ = _dados(n=5, grupos=False)             # cal < 4
    assert _split_cal_val_por_especie(X2, Y2, None, _cfg(frac=0.4)) is None
