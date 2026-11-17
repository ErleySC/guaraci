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


def _csv_mortos(tmp_path, n_mortos):
    rng = np.random.default_rng(1)
    X = 0.5 + 0.05 * rng.normal(size=(N, P)) + np.linspace(0, 1, P)
    X[:, 100:110] = 3.0
    X[N - n_mortos:, :] = 3.0       # espectros inteiros no teto, todos da classe B
    cols = [f"{w:.1f}" for w in np.linspace(4000, 7000, P)]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "classe", ["A"] * (N - n_mortos) + ["B"] * n_mortos)
    caminho = tmp_path / "m.csv"
    df.to_csv(caminho, index=False)
    return str(caminho)


def _cfg(tmp_path, csv, saturacao):
    return Config(mode="csv", csv_file=csv, class_column="classe", conc_column="", level="N1",
                  objective="classificacao", wn_min=4000.0, wn_max=7000.0, sg_window=301,
                  default_preprocessing="msc_sg_mc", output_root_folder=str(tmp_path / "saida"),
                  saturacao=saturacao)


def test_metade_morta_sai_e_classe_vazia_e_avisada(tmp_path, capsys):
    cfg = _cfg(tmp_path, _csv_mortos(tmp_path, 20), "auto")
    with pytest.raises(ValueError, match="sg_window"):
        pl.executar(cfg)
    info = cfg.saturacao_info
    assert info["espectros_removidos"] == 20 and info["espectros_mortos"] == 20
    assert info["removidos_por_classe"] == {"A": [0, 20], "B": [20, 20]}
    assert info["espectros_total"] == N and not info["inutilizavel"]
    cap = capsys.readouterr()
    assert "TODOS" in cap.out + cap.err and "sem nenhum espectro" in cap.out + cap.err


def test_inutilizavel_recusa_em_auto_e_so_avisa_em_avisar(tmp_path):
    csv = _csv_mortos(tmp_path, N - 2)
    with pytest.raises(ValueError, match="nao ha' o que modelar"):
        pl.executar(_cfg(tmp_path, csv, "auto"))
    cfg = _cfg(tmp_path, csv, "avisar")
    with pytest.raises(ValueError, match="sg_window"):
        pl.executar(cfg)
    assert cfg.saturacao_info["inutilizavel"] is True and cfg.saturacao_info["aplicado"] is False


def _csv_quant(tmp_path, teto_a_partir_de):
    """Teor alto -> espectro inteiro no teto (detector cego nas amostras mais concentradas)."""
    rng = np.random.default_rng(2)
    teores = np.repeat(np.arange(0, 20, 1.0), 2)          # 0..19 %, 2 replicas
    X = 0.5 + 0.05 * rng.normal(size=(len(teores), P)) + np.linspace(0, 1, P) + 0.01 * teores[:, None]
    X[:, 100:110] = 3.0
    X[teores >= teto_a_partir_de, :] = 3.0
    cols = [f"{w:.1f}" for w in np.linspace(4000, 7000, P)]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "teor", teores)
    df.insert(0, "classe", ["A"] * len(teores))
    caminho = tmp_path / "q.csv"
    df.to_csv(caminho, index=False)
    return str(caminho)


def _cfg_quant(tmp_path, csv, saturacao="auto"):
    return Config(mode="csv", csv_file=csv, class_column="classe", conc_column="teor", level="N1",
                  objective="quantificacao", wn_min=4000.0, wn_max=7000.0, sg_window=301,
                  default_preprocessing="msc_sg_mc", output_root_folder=str(tmp_path / "saida"),
                  saturacao=saturacao)


def test_quantificacao_registra_faixa_que_encolheu(tmp_path, capsys):
    cfg = _cfg_quant(tmp_path, _csv_quant(tmp_path, 10.0))
    with pytest.raises(ValueError, match="sg_window"):
        pl.executar(cfg)
    info = cfg.saturacao_info
    assert info["faixa_teor_antes"] == [0.0, 19.0] and info["faixa_teor_depois"] == [0.0, 9.0]
    assert info["concentracoes_afetadas"][0] == 10.0 and not info["inutilizavel"]
    cap = capsys.readouterr()
    assert "encolheu a faixa de teor" in cap.out + cap.err


def test_quantificacao_sem_teores_suficientes_e_recusada(tmp_path):
    with pytest.raises(ValueError, match="nao ha' o que modelar"):
        pl.executar(_cfg_quant(tmp_path, _csv_quant(tmp_path, 2.0)))   # sobram so' 0 e 1 %
