# -*- coding: utf-8 -*-
"""Aviso quando a faixa espectral remove mais de 20% das variaveis (V81).

A faixa padrao 4000-10000 (herdada do FT-NIR em cm-1) vale para qualquer csv/dx
e cortava parte do espectro de outra tecnica sem aviso. O padrao neutro e' v1.1;
aqui so' o aviso. O corte em si nao muda.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import guaraci.pipeline as pl
from guaraci.config import Config
from guaraci.eixo_espectral import LIMITE_CORTE_FAIXA, aviso_de_corte_de_faixa


def test_limite_e_de_20_por_cento():
    assert LIMITE_CORTE_FAIXA == 0.20


def test_aviso_dispara_acima_de_20_por_cento_e_nao_abaixo():
    assert aviso_de_corte_de_faixa(100, 21, (4000, 10000), (400, 4000))
    assert not aviso_de_corte_de_faixa(100, 20, (4000, 10000), (400, 4000))   # exatamente 20%: sem aviso
    assert not aviso_de_corte_de_faixa(100, 5, (4000, 10000), (400, 4000))
    assert not aviso_de_corte_de_faixa(0, 0, (4000, 10000), (0, 1))


def test_aviso_diz_quanto_foi_cortado_e_so_usa_a_unidade_declarada():
    txt = aviso_de_corte_de_faixa(200, 100, (4000, 10000), (400.0, 4000.0), "nm")
    assert "100 de 200" in txt and "50%" in txt and "nm" in txt
    assert "cm-1" not in aviso_de_corte_de_faixa(200, 100, (4000, 10000), (400.0, 4000.0), "")


def _csv(tmp_path):
    rng = np.random.default_rng(0)
    classes = ["A"] * 6 + ["B"] * 6
    df = pd.DataFrame(rng.normal(size=(12, 40)) + 1, columns=[f"{w:.1f}" for w in np.linspace(4000, 7000, 40)])
    df.insert(0, "classe", classes)
    caminho = tmp_path / "d.csv"
    df.to_csv(caminho, index=False)
    return str(caminho)


def _rodar_ate_o_corte(tmp_path, wn_min):
    """Roda ate' a etapa de corte; a janela SG grande interrompe logo depois (barato)."""
    cfg = Config(mode="csv", csv_file=_csv(tmp_path), class_column="classe", conc_column="", level="N1",
                 auto_adjust=False, objective="classificacao", wn_min=wn_min, wn_max=7000.0, sg_window=101,
                 default_preprocessing="msc_sg_mc", output_root_folder=str(tmp_path / "saida"))
    with pytest.raises(ValueError, match="sg_window"):
        pl.executar(cfg)


def _saida(capsys) -> str:
    cap = capsys.readouterr()
    return cap.out + cap.err


def test_execucao_avisa_quando_a_faixa_corta_um_terco_dos_pontos(tmp_path, capsys):
    _rodar_ate_o_corte(tmp_path, wn_min=5000.0)              # remove ~1/3 dos 40 pontos
    saida = _saida(capsys)
    assert "[AVISO] A faixa espectral [5000, 7000] removeu" in saida


def test_execucao_nao_avisa_com_corte_pequeno(tmp_path, capsys):
    _rodar_ate_o_corte(tmp_path, wn_min=4100.0)              # remove ~3 de 40
    saida = _saida(capsys)
    assert "Truncamento espectral" in saida                  # a etapa rodou
    assert "removeu" not in saida
