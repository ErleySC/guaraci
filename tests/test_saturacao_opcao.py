# -*- coding: utf-8 -*-
"""Opcao `saturacao: auto | avisar | nenhuma` no pipeline (substitui o gatilho por ajuste_automatico)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import guaraci.pipeline as pl
from guaraci.config import Config
from guaraci.config_io import _CONFIG_SPEC

N, P = 40, 300


def _csv(tmp_path):
    rng = np.random.default_rng(0)
    X = 0.5 + 0.05 * rng.normal(size=(N, P)) + np.linspace(0, 1, P)
    X[:, 100:140] = 3.0            # banda no teto em todas as amostras (40 variaveis)
    X[7, :] = 3.0                  # espectro inteiro saturado
    cols = [f"{w:.1f}" for w in np.linspace(4000, 7000, P)]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "classe", ["A", "B"] * (N // 2))
    caminho = tmp_path / "d.csv"
    df.to_csv(caminho, index=False)
    return str(caminho)


def _rodar(tmp_path, saturacao, **extra):
    cfg = Config(mode="csv", csv_file=_csv(tmp_path), class_column="classe", conc_column="", level="N1",
                 objective="classificacao", wn_min=4000.0, wn_max=7000.0, sg_window=301,
                 default_preprocessing="msc_sg_mc", output_root_folder=str(tmp_path / "saida"),
                 saturacao=saturacao, **extra)
    with pytest.raises(ValueError, match="sg_window"):
        pl.executar(cfg)           # a janela grande interrompe logo depois da etapa de saturacao
    return cfg


def test_opcao_existe_no_spec_com_os_tres_valores():
    spec = {d["key"]: d for d in _CONFIG_SPEC}["saturacao"]
    assert spec["opcoes"] == ["auto", "avisar", "nenhuma"] and Config().saturacao == "auto"


def test_auto_remove_variaveis_e_espectro(tmp_path):
    info = _rodar(tmp_path, "auto").saturacao_info
    assert info["aplicado"] is True
    assert info["variaveis_removidas"] >= 40 and info["espectros_removidos"] == 1


def test_avisar_detecta_mas_nao_remove(tmp_path, capsys):
    info = _rodar(tmp_path, "avisar").saturacao_info
    assert info["aplicado"] is False and info["variaveis_removidas"] >= 40
    cap = capsys.readouterr()
    assert "saturacao=avisar" in cap.out + cap.err


def test_nenhuma_nao_procura(tmp_path):
    assert _rodar(tmp_path, "nenhuma").saturacao_info == {}


def test_independe_de_ajuste_automatico(tmp_path):
    assert _rodar(tmp_path, "auto", auto_adjust=False).saturacao_info["aplicado"] is True
