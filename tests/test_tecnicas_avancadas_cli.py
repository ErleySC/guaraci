# -*- coding: utf-8 -*-
"""Testes de ponta a ponta (via CLI, `builtins.input` mockado) do menu [T]
Tecnicas Avancadas em guaraci.py -- confirma que ASCA/EPO-GLSW/MCR-ALS/
fusao multibloco sao alcancaveis e executaveis de verdade pelo terminal,
nao so' pela camada de orquestracao testada em test_tecnicas_avancadas.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def guaraci_mod():
    import guaraci.guaraci as mod
    return mod


def _dataset_fake(n_por_classe=8, p=30, seed=0, com_metadados=True):
    rng = np.random.default_rng(seed)
    classes = np.repeat(["Copaiba", "Andiroba", "Buriti"], n_por_classe)
    n = len(classes)
    base = {"Copaiba": 1.0, "Andiroba": 2.0, "Buriti": 3.0}
    X = np.array([base[c] + rng.normal(0, 0.1, p) for c in classes])
    wn = np.linspace(4000, 10000, p)
    mae_id = np.array([f"G{i // 2}" for i in range(n)])
    conc = np.where(classes == "Andiroba", rng.uniform(1, 10, n), np.nan)
    metadados = None
    if com_metadados:
        sessao = np.tile(["S1", "S2"], n // 2 + 1)[:n]
        metadados = pd.DataFrame({"especie": classes, "sessao": sessao})
    return wn, X, classes, conc, mae_id, metadados


def _mock_input(monkeypatch, respostas):
    it = iter(respostas)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(it))


def _mock_load_data(monkeypatch, guaraci_mod, retorno):
    monkeypatch.setattr(guaraci_mod.pq, "load_data", lambda cfg: retorno)


# ---------------------------------------------------------------------------
# ASCA
# ---------------------------------------------------------------------------

def test_menu_tec_asca_end_to_end(guaraci_mod, monkeypatch, capsys):
    wn, X, classes, conc, mae_id, metadados = _dataset_fake()
    _mock_load_data(monkeypatch, guaraci_mod, (wn, X, classes, conc, mae_id, metadados))
    # menu principal "1"=ASCA; escolhe fator "1" (especie); Enter final (_pause)
    _mock_input(monkeypatch, ["1", "1", ""])
    cfg = guaraci_mod.Config(mode="sintetico")
    guaraci_mod._menu_tecnicas_avancadas(cfg)
    saida = capsys.readouterr().out
    assert "ASCA" in saida or "especie" in saida


def test_menu_tec_asca_sem_fator_valido_nao_quebra(guaraci_mod, monkeypatch, capsys):
    wn, X, classes, conc, mae_id, metadados = _dataset_fake()
    _mock_load_data(monkeypatch, guaraci_mod, (wn, X, classes, conc, mae_id, metadados))
    _mock_input(monkeypatch, ["1", "", ""])   # escolhe nenhum fator
    cfg = guaraci_mod.Config(mode="sintetico")
    guaraci_mod._menu_tecnicas_avancadas(cfg)   # nao deve levantar excecao


# ---------------------------------------------------------------------------
# EPO/GLSW
# ---------------------------------------------------------------------------

def test_menu_tec_epo_glsw_end_to_end(guaraci_mod, monkeypatch, capsys):
    wn, X, classes, conc, mae_id, metadados = _dataset_fake()
    _mock_load_data(monkeypatch, guaraci_mod, (wn, X, classes, conc, mae_id, metadados))
    # "2"=EPO/GLSW; fator1="1"(especie); fator2="2"(sessao); metodo Enter=EPO; Enter final
    _mock_input(monkeypatch, ["2", "1", "2", "", ""])
    cfg = guaraci_mod.Config(mode="sintetico")
    guaraci_mod._menu_tecnicas_avancadas(cfg)
    saida = capsys.readouterr().out
    assert "EPO" in saida


def test_menu_tec_glsw_end_to_end(guaraci_mod, monkeypatch, capsys):
    wn, X, classes, conc, mae_id, metadados = _dataset_fake()
    _mock_load_data(monkeypatch, guaraci_mod, (wn, X, classes, conc, mae_id, metadados))
    _mock_input(monkeypatch, ["2", "1", "2", "2", ""])   # metodo="2"=GLSW
    cfg = guaraci_mod.Config(mode="sintetico")
    guaraci_mod._menu_tecnicas_avancadas(cfg)
    saida = capsys.readouterr().out
    assert "GLSW" in saida


def test_menu_tec_epo_glsw_mesmo_fator_duas_vezes_da_erro(guaraci_mod, monkeypatch, capsys):
    wn, X, classes, conc, mae_id, metadados = _dataset_fake()
    _mock_load_data(monkeypatch, guaraci_mod, (wn, X, classes, conc, mae_id, metadados))
    _mock_input(monkeypatch, ["2", "1", "1", ""])   # fator1=fator2=especie
    cfg = guaraci_mod.Config(mode="sintetico")
    guaraci_mod._menu_tecnicas_avancadas(cfg)
    saida = capsys.readouterr().out
    assert "diferentes" in saida.lower()


# ---------------------------------------------------------------------------
# MCR-ALS
# ---------------------------------------------------------------------------

def test_menu_tec_mcr_als_puro_end_to_end(guaraci_mod, monkeypatch, capsys):
    wn, X, classes, conc, mae_id, metadados = _dataset_fake()
    X_pos = np.abs(X)
    _mock_load_data(monkeypatch, guaraci_mod, (wn, X_pos, classes, None, mae_id, metadados))
    # "3"=MCR-ALS; n_componentes Enter=2; Enter final
    _mock_input(monkeypatch, ["3", "", ""])
    cfg = guaraci_mod.Config(mode="sintetico")
    guaraci_mod._menu_tecnicas_avancadas(cfg)
    saida = capsys.readouterr().out
    assert "MCR-ALS" in saida
    assert "lack-of-fit" in saida.lower()


def test_menu_tec_mcr_als_com_restricao_correlacao_end_to_end(guaraci_mod, monkeypatch, capsys):
    wn, X, classes, conc, mae_id, metadados = _dataset_fake()
    X_pos = np.abs(X)
    _mock_load_data(monkeypatch, guaraci_mod, (wn, X_pos, classes, conc, mae_id, metadados))
    # "3"=MCR-ALS; n_comp Enter=2; usar restricao="s"; indice Enter=0; Enter final
    _mock_input(monkeypatch, ["3", "", "s", "", ""])
    cfg = guaraci_mod.Config(mode="sintetico")
    guaraci_mod._menu_tecnicas_avancadas(cfg)
    saida = capsys.readouterr().out
    assert "correlacao" in saida.lower() or "referencia" in saida.lower()


# ---------------------------------------------------------------------------
# Fusao multibloco
# ---------------------------------------------------------------------------

def test_menu_tec_fusao_multibloco_end_to_end(guaraci_mod, monkeypatch, capsys):
    rng = np.random.default_rng(7)
    # p1/p2 realistas (>= janela padrao do Savitzky-Golay do pre-processador
    # default msc_sg_mc -- poucas variaveis, como um bloco sintetico
    # minusculo, faz o SG estourar ValueError "window_length must be <=
    # size of x", mesmo comportamento ja documentado em config.py para o
    # modo imagem).
    n, p1, p2 = 30, 250, 200
    y = rng.uniform(0, 10, n)
    bloco_a = y[:, None] * 0.1 + rng.normal(0, 0.05, (n, p1))
    bloco_b = y[:, None] * 0.2 + rng.normal(0, 0.05, (n, p2))
    wn_a = np.linspace(4000, 10000, p1)
    wn_b = np.linspace(700, 4000, p2)
    rot = np.array(["oleo"] * n)
    mae = np.array([f"G{i}" for i in range(n)])

    chamadas = {"n": 0}

    def _fake_load_data(cfg):
        chamadas["n"] += 1
        if chamadas["n"] == 1:
            return wn_a, bloco_a, rot, y, mae, None
        return wn_b, bloco_b, rot, None, mae, None

    monkeypatch.setattr(guaraci_mod.pq, "load_data", _fake_load_data)
    # "4"=fusao; nome1 Enter; modo1 Enter=dx; pasta1="x"; nome2 Enter; modo2 Enter; pasta2="y"; Enter final
    _mock_input(monkeypatch, ["4", "", "", "x", "", "", "y", ""])
    cfg = guaraci_mod.Config(mode="sintetico")
    guaraci_mod._menu_tecnicas_avancadas(cfg)
    saida = capsys.readouterr().out
    assert "RMSEP" in saida


def test_menu_tec_fusao_multibloco_sem_y_da_erro_claro(guaraci_mod, monkeypatch, capsys):
    n, p = 20, 15
    X = np.zeros((n, p))
    wn = np.linspace(4000, 10000, p)
    rot = np.array(["oleo"] * n)
    mae = np.array([f"G{i}" for i in range(n)])

    monkeypatch.setattr(guaraci_mod.pq, "load_data",
                         lambda cfg: (wn, X, rot, None, mae, None))
    _mock_input(monkeypatch, ["4", "", "", "x", "", "", "y", ""])
    cfg = guaraci_mod.Config(mode="sintetico")
    guaraci_mod._menu_tecnicas_avancadas(cfg)
    saida = capsys.readouterr().out
    assert "teor" in saida.lower() or "concentracao" in saida.lower()
