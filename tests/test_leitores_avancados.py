# -*- coding: utf-8 -*-
"""Testes de leitores_avancados.py -- registro no io_registry dos formatos
de instrumento (opus/spc/sp/rmn/hplc/gcms/eem), fechando a lacuna
encontrada na auditoria de acessibilidade CLI/web (os parsers de baixo
nivel existiam testados em test_importadores_proprietarios.py, mas nenhum
`cfg.mode` os alcancava).

Reusa os MESMOS arquivos/pastas reais ja' usados por
test_importadores_proprietarios.py (nao sinteticos) para spc/sp/rmn/hplc;
OPUS usa a mesma limitacao honesta e o mesmo double daquele arquivo (sem
binario OPUS real disponivel neste ambiente)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import guaraci.leitores_avancados  # noqa: F401 -- registra os leitores
from guaraci.config import Config
from guaraci.io_registry import get_reader, registered_modes

_FIXTURES = Path(__file__).parent / "fixtures"


def _cfg(mode: str, pasta: str) -> Config:
    cfg = Config()
    cfg.mode = mode
    cfg.input_folder = pasta
    return cfg


def test_todos_os_novos_modos_registrados():
    modos = registered_modes()
    for m in ("opus", "spc", "sp", "rmn", "hplc", "gcms", "eem"):
        assert m in modos


def test_spc_carrega_fixture_real():
    cfg = _cfg("spc", str(_FIXTURES / "spc"))
    wn, X, rot, conc, mae, meta = get_reader("spc")(cfg)
    assert X.shape[0] == 2          # spectra.spc + nir.spc
    assert wn.shape[0] == X.shape[1]
    assert mae is None
    assert set(rot.tolist()) == {"spc"}   # sem subpasta -> 1 classe (nome da pasta)


def test_sp_carrega_fixture_real():
    cfg = _cfg("sp", str(_FIXTURES / "sp"))
    wn, X, rot, conc, mae, meta = get_reader("sp")(cfg)
    assert X.shape[0] == 1
    assert wn.shape[0] == X.shape[1]


def test_rmn_carrega_fixture_real():
    cfg = _cfg("rmn", str(_FIXTURES / "rmn_bruker"))
    wn, X, rot, conc, mae, meta = get_reader("rmn")(cfg)
    assert X.shape[0] == 1
    assert wn.shape[0] == X.shape[1]


def test_hplc_carrega_fixture_real():
    cfg = _cfg("hplc", str(_FIXTURES / "hplc_agilent"))
    wn, X, rot, conc, mae, meta = get_reader("hplc")(cfg)
    assert X.shape[0] == 1
    assert wn.shape[0] == X.shape[1]


def test_hplc_detector_fid_ausente_da_erro_claro(capsys):
    cfg = _cfg("hplc", str(_FIXTURES / "hplc_agilent"))
    cfg.hplc_detector = "FID"
    with pytest.raises(ValueError, match="Nenhuma amostra"):
        get_reader("hplc")(cfg)
    assert "detector 'FID' nao encontrado" in capsys.readouterr().out


def test_eem_carrega_fixture_real_horiba():
    cfg = _cfg("eem", str(_FIXTURES / "eem_horiba_aqualog"))
    wn, X, rot, conc, mae, meta = get_reader("eem")(cfg)
    assert X.shape[0] == 1
    assert wn.shape[0] == X.shape[1]
    assert meta.attrs["grade_excitacao"] is not None


def test_opus_sem_brukeropus_da_erro_claro_no_dataset_inteiro(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "brukeropus", None)
    (tmp_path / "amostra.0").write_bytes(b"nao importa, brukeropus ausente")
    cfg = _cfg("opus", str(tmp_path))
    with pytest.raises(ValueError, match="Nenhuma amostra"):
        get_reader("opus")(cfg)


def test_opus_com_double_carrega_por_arquivo(monkeypatch, tmp_path):
    eixo = np.linspace(4000, 10000, 50)

    def _bloco(y):
        return SimpleNamespace(x=eixo, y=np.asarray(y, dtype=float))

    def _fake_read_opus(fp, **kw):
        return SimpleNamespace(is_opus=True, data_keys=["a"],
                                a=_bloco(np.sin(eixo / 500.0)))

    monkeypatch.setattr("brukeropus.read_opus", _fake_read_opus, raising=False)
    monkeypatch.setitem(sys.modules, "brukeropus",
                         SimpleNamespace(read_opus=_fake_read_opus))

    (tmp_path / "amostra1.0").write_bytes(b"fake opus bytes")
    (tmp_path / "amostra2.0").write_bytes(b"fake opus bytes")
    cfg = _cfg("opus", str(tmp_path))
    wn, X, rot, conc, mae, meta = get_reader("opus")(cfg)
    assert X.shape == (2, 50)
    assert len(set(rot.tolist())) == 1   # sem subpasta -> 1 classe so'


def test_subpastas_viram_classes_leitor_generico_por_arquivo(tmp_path):
    """Duas subpastas com arquivo .sp real (copiado da fixture) -> 2 classes."""
    import shutil
    origem = _FIXTURES / "sp" / "spectra.sp"
    for classe in ("Copaiba", "Andiroba"):
        pasta = tmp_path / classe
        pasta.mkdir()
        shutil.copy(origem, pasta / "amostra1.sp")
    cfg = _cfg("sp", str(tmp_path))
    wn, X, rot, conc, mae, meta = get_reader("sp")(cfg)
    assert X.shape[0] == 2
    assert set(rot.tolist()) == {"Copaiba", "Andiroba"}


def test_pasta_vazia_da_erro_claro(tmp_path):
    cfg = _cfg("spc", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        get_reader("spc")(cfg)


def test_pasta_inexistente_da_erro_claro():
    cfg = _cfg("spc", str(_FIXTURES / "nao_existe_de_verdade"))
    with pytest.raises(FileNotFoundError):
        get_reader("spc")(cfg)


def test_gcms_pasta_vazia_da_erro_claro(tmp_path):
    cfg = _cfg("gcms", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        get_reader("gcms")(cfg)


def test_rmn_pasta_sem_pdata_da_erro_claro(tmp_path):
    (tmp_path / "nao_e_amostra_rmn").mkdir()
    cfg = _cfg("rmn", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        get_reader("rmn")(cfg)
