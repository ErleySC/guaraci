# -*- coding: utf-8 -*-
"""Testes de leitores_avancados.py -- registro no io_registry dos formatos
de instrumento (opus/spc/sp/rmn/hplc/gcms/eem), fechando a lacuna
encontrada na auditoria de acessibilidade CLI/web (os parsers de baixo
nivel existiam testados em test_importadores_proprietarios.py, mas nenhum
`cfg.mode` os alcancava).

Usa arquivos SINTETICOS (tests/_formatos_binarios.py, verdade conhecida) para spc/sp/rmn/eem e um `rainbow` falso para o HPLC/GC;
OPUS usa a mesma limitacao honesta e o mesmo double daquele arquivo (sem binario OPUS disponivel). A leitura de arquivos REAIS
de fabricante e' validacao do mantenedor (`validacao/tests/test_formatos_reais_locais.py`)."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import guaraci.leitores_avancados  # noqa: F401 -- registra os leitores
from guaraci.config import Config
from guaraci.io_registry import get_reader, registered_modes
from tests._formatos_binarios import escrever_rmn_bruker, escrever_sp, escrever_spc, instalar_rainbow_falso



def _cfg(mode: str, pasta: str) -> Config:
    cfg = Config()
    cfg.mode = mode
    cfg.input_folder = pasta
    return cfg


def test_todos_os_novos_modos_registrados():
    modos = registered_modes()
    for m in ("opus", "spc", "sp", "rmn", "hplc", "gcms", "eem"):
        assert m in modos


def test_spc_carrega_pasta_sintetica(tmp_path):
    pytest.importorskip("spcfile")
    escrever_spc(tmp_path / "a.spc", 400.0, 1800.0, [np.linspace(0, 1, 40)])
    escrever_spc(tmp_path / "b.spc", 1100.0, 2500.0, [np.linspace(0, 2, 40), np.linspace(0, 3, 40)])
    wn, X, rot, conc, mae, meta = get_reader("spc")(_cfg("spc", str(tmp_path)))
    assert X.shape[0] == 2 and wn.shape[0] == X.shape[1] == 40
    assert mae is None
    assert len(set(rot.tolist())) == 1                 # sem subpasta -> 1 classe (nome da pasta)


def test_sp_carrega_pasta_sintetica(tmp_path):
    escrever_sp(tmp_path / "a.sp", 4000.0, 700.0, np.linspace(0.1, 0.9, 33))
    wn, X, rot, conc, mae, meta = get_reader("sp")(_cfg("sp", str(tmp_path)))
    assert X.shape == (1, 33) and wn.shape[0] == 33


def test_rmn_carrega_pasta_sintetica(tmp_path):
    pytest.importorskip("nmrglue")
    escrever_rmn_bruker(tmp_path / "exp1", 9.5, -0.5, np.round(np.abs(np.sin(np.linspace(0, 9, 128))) * 1e5))
    escrever_rmn_bruker(tmp_path / "exp2", 9.5, -0.5, np.round(np.abs(np.cos(np.linspace(0, 9, 128))) * 1e5))
    wn, X, rot, conc, mae, meta = get_reader("rmn")(_cfg("rmn", str(tmp_path)))
    assert X.shape == (2, 128) and wn.shape[0] == 128


def test_hplc_carrega_pasta_sintetica_com_rainbow_falso(tmp_path, monkeypatch):
    t = np.linspace(0.0, 30.0, 300)
    for nome in ("amostra1.D", "amostra2.D"):
        (tmp_path / nome).mkdir()
    instalar_rainbow_falso(monkeypatch, {"UV": [(t, np.exp(-((t - 12.0) / 0.4) ** 2))]})
    wn, X, rot, conc, mae, meta = get_reader("hplc")(_cfg("hplc", str(tmp_path)))
    assert X.shape == (2, 300) and wn.shape[0] == 300


def test_hplc_detector_fid_ausente_da_erro_claro_sintetico(tmp_path, monkeypatch, capsys):
    (tmp_path / "amostra1.D").mkdir()
    instalar_rainbow_falso(monkeypatch, {"UV": [(np.linspace(0, 10, 20), np.ones(20))]})
    cfg = _cfg("hplc", str(tmp_path))
    cfg.hplc_detector = "FID"
    with pytest.raises(ValueError, match="Nenhuma amostra"):
        get_reader("hplc")(cfg)
    assert "detector 'FID' nao encontrado" in capsys.readouterr().out


def test_eem_carrega_pasta_sintetica_de_fluorimetro(tmp_path):
    from guaraci.formatos_sinteticos import gerar_eem
    gerar_eem(str(tmp_path), n_classes=2, n_amostras=3)
    wn, X, rot, conc, mae, meta = get_reader("eem")(_cfg("eem", str(tmp_path)))
    assert X.shape[0] == 6 and wn.shape[0] == X.shape[1]
    assert meta.attrs["grade_excitacao"] is not None
    assert set(rot.tolist()) == {"classe_1", "classe_2"}


def test_subpastas_viram_classes_leitor_generico_por_arquivo_sintetico(tmp_path):
    """Duas subpastas com um `.sp` cada -> 2 classes (o nome da subpasta)."""
    for classe in ("classe_a", "classe_b"):
        (tmp_path / classe).mkdir()
        escrever_sp(tmp_path / classe / "amostra1.sp", 4000.0, 700.0, np.linspace(0.1, 0.9, 33))
    wn, X, rot, conc, mae, meta = get_reader("sp")(_cfg("sp", str(tmp_path)))
    assert X.shape[0] == 2
    assert set(rot.tolist()) == {"classe_a", "classe_b"}


def test_opus_sem_brukeropus_da_erro_claro_no_dataset_inteiro(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "brukeropus", None)
    (tmp_path / "amostra.0").write_bytes(b"nao importa, brukeropus ausente")
    cfg = _cfg("opus", str(tmp_path))
    # pacote opcional ausente falha no 1o arquivo, com a instrucao de instalacao
    # (antes: N erros iguais e "Nenhuma amostra valida", que a CLI mapeava a GUA-E008 "pasta nao encontrada")
    with pytest.raises(ImportError, match="brukeropus"):
        get_reader("opus")(cfg)


def test_opus_com_double_carrega_por_arquivo(monkeypatch, tmp_path):
    pytest.importorskip("brukeropus")
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


def test_pasta_vazia_da_erro_claro(tmp_path):
    cfg = _cfg("spc", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        get_reader("spc")(cfg)


def test_pasta_inexistente_da_erro_claro():
    cfg = _cfg("spc", "pasta_que_nao_existe_de_verdade")
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


# --- checkup 2026-11 (RMN): bucketing integrado e layout TopSpin <nome>/<expno>/pdata ----------

def _rmn_estreito(n: int, centro: float, fase: float) -> np.ndarray:
    """Linha de RMN (FWHM ~0,003 ppm) num eixo 10..0 ppm de `n` pontos, deslocada por `fase`."""
    x = np.linspace(10.0, 0.0, n)
    g = 0.0015
    return np.round(1e6 * g * g / ((x - centro - fase) ** 2 + g * g) + 10)


def test_rmn_reamostra_integrando_nao_por_ponto(tmp_path):
    """Com mais de 2000 pontos o leitor reduz a grade (~0,005 ppm aqui): uma linha
    estreita deslocada de meio passo nao pode mudar de altura ao acaso -- a area
    no bucket tem de se conservar (antes: np.interp pontual = aliasing)."""
    pytest.importorskip("nmrglue")
    n = 16384
    for i, fase in enumerate((0.0, 0.0023)):
        escrever_rmn_bruker(tmp_path / "cls" / f"exp{i}", 10.0, 0.0, _rmn_estreito(n, 1.33, fase))
    wn, X, rot, conc, mae, meta = get_reader("rmn")(_cfg("rmn", str(tmp_path)))
    assert X.shape == (2, 2000)
    area = (X - 10).sum(axis=1)
    assert area[1] == pytest.approx(area[0], rel=0.02)


def test_rmn_aceita_layout_topspin_com_expno(tmp_path):
    """Layout real do TopSpin: <classe>/<amostra>/<expno>/pdata/1."""
    pytest.importorskip("nmrglue")
    for cls in ("A", "B"):
        for k in range(2):
            escrever_rmn_bruker(tmp_path / cls / f"am{k}" / "10", 9.5, -0.5,
                                np.round(np.abs(np.sin(np.linspace(0, 9 + k, 128))) * 1e5))
    wn, X, rot, conc, mae, meta = get_reader("rmn")(_cfg("rmn", str(tmp_path)))
    assert X.shape == (4, 128) and sorted(set(rot.tolist())) == ["A", "B"]
