# -*- coding: utf-8 -*-
"""Leitor .dx escolhe os arquivos que cobrem a janela pedida; divisao SPXY na regressao."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from guaraci import dados_io
from guaraci.config import Config
from guaraci.config_io import _CONFIG_SPEC


def _dx(path, titulo, x0, x1, n):
    wn = np.linspace(x0, x1, n)
    y = np.sin(wn / 900.0) + 2
    corpo = "\n".join(f"{wn[k]:.4f} " + " ".join(f"{v:.6f}" for v in y[k:k + 5]) for k in range(0, n, 5))
    Path(path).write_text(
        f"##TITLE={titulo}\n##JCAMP-DX=4.24\n##DATA TYPE=NEAR INFRARED SPECTRUM\n##XUNITS=1/CM\n##YUNITS=ABSORBANCE\n"
        f"##XFACTOR=1\n##YFACTOR=1\n##FIRSTX={wn[0]}\n##LASTX={wn[-1]}\n##NPOINTS={n}\n##XYDATA=(X++(Y..Y))\n{corpo}\n##END=\n",
        encoding="latin-1")


@pytest.fixture()
def pasta_mista(tmp_path):
    """Puros em faixa larga (0-15800) e misturas exportadas cortadas em 4000 (299-4000)."""
    d = tmp_path / "oleo"
    d.mkdir()
    for i in range(3):
        _dx(d / f"X-01-01-2020_T{i + 1}.dx", f"X-01-01-2020_T{i + 1}", 0, 15800, 400)
    for i in range(6):
        _dx(d / f"X-01-01-2020-AD-M-{i + 1},0%_T1.dx", f"X-01-01-2020-AD-M-{i + 1},0%_T1", 299, 4000, 100)
    return str(d)


def test_janela_nir_fica_com_os_arquivos_que_cobrem_e_nao_com_a_moda(pasta_mista, capsys):
    wn, X, *_ = dados_io.load_dx(pasta_mista, extrair_conc=True, janela=(4540.0, 9500.0))
    assert X.shape[0] == 3 and wn.max() > 9500          # os 3 puros de faixa larga; os 6 cortados saem
    assert "DISCARDING 6 files" in capsys.readouterr().out


def test_janela_mir_mantem_todos(pasta_mista):
    wn, X, *_ = dados_io.load_dx(pasta_mista, extrair_conc=True, janela=(300.0, 4000.0))
    assert X.shape[0] == 9 and wn.max() <= 4001


def test_sem_janela_ou_sem_cobertura_vale_a_moda_como_antes(pasta_mista):
    assert dados_io.load_dx(pasta_mista, extrair_conc=True)[1].shape[0] == 6            # moda: os 6 cortados
    assert dados_io.load_dx(pasta_mista, extrair_conc=True, janela=(20000.0, 30000.0))[1].shape[0] == 6


def test_divisao_spxy_existe_no_spec_e_cobre_os_extremos_do_teor():
    spec = {d["key"]: d for d in _CONFIG_SPEC}["divisao_calibracao"]
    assert "spxy" in spec["opcoes"]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 30))
    y = np.linspace(0, 15, 40)
    mae = np.repeat(np.arange(20), 2).astype(str)
    cfg = Config(cal_val_split="spxy", frac_cal=0.7)
    ic, iv = dados_io._split_cal_val_por_especie(X, y.reshape(-1, 1), mae, cfg)
    assert len(set(ic) & set(iv)) == 0 and len(ic) + len(iv) == 40
    assert y[ic].max() == y.max() and y[ic].min() == y.min()     # SPXY: o treino cobre a faixa de teor
    assert len(set(mae[ic]) & set(mae[iv])) == 0                  # replicas juntas


def test_titulo_com_codigo_errado_e_arquivo_sem_titulo_voltam_para_a_classe_da_subpasta(tmp_path, capsys):
    d = tmp_path / "Maracuja"
    d.mkdir()
    outra = tmp_path / "Coco"
    outra.mkdir()
    for i in range(6):
        _dx(outra / f"COC-01-01-2020-AD-A-{i + 1},0%_T1.dx", f"COC-01-01-2020-AD-A-{i + 1},0%_T1", 4000, 10000, 100)
    for i in range(20):
        _dx(d / f"MAR-01-01-2020-AD-A-{i + 1},0%_T1.dx", f"MAR-01-01-2020-AD-A-{i + 1},0%_T1", 4000, 10000, 100)
    for i in range(3):    # titulo com o codigo errado: viraria a classe-fantasma "AR"
        _dx(d / f"MAR-01-01-2020-AD-A-{i + 30},0%_T2.dx", f"AR-01-01-2020-AD-A-{i + 30},0%_T2", 4000, 10000, 100)
    _dx(d / "medida solta.dx", "medida solta", 4000, 10000, 100)    # sem padrao: caia no nome da subpasta
    _, _, rot, _, _, meta = dados_io.load_dx(str(tmp_path), extrair_conc=True)
    assert set(rot) == {"MAR", "COC"}                # sem classe-fantasma "AR" nem "medida solta"
    assert "reatribuidos a 'MAR'" in capsys.readouterr().out
