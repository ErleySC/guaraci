# -*- coding: utf-8 -*-
"""Textos da quantificacao sem unidade nem material presumidos (V78).

O motor quantifica qualquer resposta numerica; nao ha como declarar a unidade
dela ainda (v1.1), entao nenhum texto/rotulo pode afirmar "%" nem "teor" nem
"adulterados". A convencao de pasta `PLSDA_OE_` fica: e' o nome que a busca da
ultima execucao procura (ver tests/test_pipeline_core.py) e existe em pastas
de usuarios.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np

import guaraci.pipeline as pq

RAIZ = Path(__file__).resolve().parents[1]


def _dados(seed=0, n_cls=2, n_grupos=6, n_rep=3, p=30):
    rng = np.random.default_rng(seed)
    X, conc, rot, mae = [], [], [], []
    for c in range(n_cls):
        base = rng.normal(size=p) * 3
        for g in range(n_grupos):
            teor = 0.0 if g < 2 else float(g)
            for r in range(n_rep):
                X.append(base + 0.1 * teor * np.arange(p) / p + rng.normal(scale=0.05, size=p))
                conc.append(teor); rot.append(f"C{c}"); mae.append(f"C{c}g{g}")
    return np.array(X), np.array(conc), np.array(rot), np.array(mae)


def test_regressao_pooled_nao_afirma_percentual_nem_teor(tmp_path, caplog):
    X, conc, rot, mae = _dados()
    cfg = pq.Config(seed=0, max_lvs=4, frac_cal=0.7)
    pasta = str(tmp_path / "r")
    os.makedirs(os.path.join(pasta, pq.NOME_TABELAS), exist_ok=True)
    os.makedirs(os.path.join(pasta, pq.NOME_GRAFICOS), exist_ok=True)
    with caplog.at_level(logging.INFO):
        pq.pls_regressao_pooled(X, conc, rot, mae, cfg, pasta, pasta, n_splits=3)
    texto = caplog.text
    assert "[7/7] PLS regressao" in texto
    assert "teor%" not in texto and "adulterados" not in texto and "puros)" not in texto
    assert "resposta" in texto


def test_rotulo_de_lod_loq_sem_percentual_presumido():
    fonte = (RAIZ / "src" / "guaraci" / "figuras.py").read_text(encoding="utf-8")
    assert "Concentration (%)" not in fonte


def test_faixa_de_validacao_do_log_sem_percentual():
    fonte = (RAIZ / "src" / "guaraci" / "pipeline.py").read_text(encoding="utf-8")
    assert not re.search(r"validacao_teor_(max|dp)'\]:\.2f\}\]?%", fonte)
