# -*- coding: utf-8 -*-
"""Testes de importadores_proprietarios.py (Bloco 18).

LIMITACAO HONESTA (ver docstring do modulo): nao ha' arquivo OPUS binario
real disponivel neste ambiente para um teste fim-a-fim genuino. Os testes
de OPUS abaixo cobrem duas coisas SEPARADAS:

  1. O contrato de import opcional (brukeropus ausente -> ImportError com
     mensagem clara), que nao depende de nenhum arquivo.
  2. A logica de extracao/preferencia de bloco de `parse_opus`, testada
     contra um DOUBLE que reproduz EXATAMENTE a forma documentada e
     verificada no codigo-fonte da biblioteca instalada nesta sessao
     (`brukeropus.file.data.Data`: atributos `x`/`y` como ndarray 1D;
     `OPUSFile`: atributos `is_opus`/`data_keys` + um atributo por chave
     de `data_keys`) -- NAO um binario OPUS de verdade.

Ja os testes de `parse_spc` e `parse_sp` (ver mais abaixo) rodam contra
ARQUIVOS REAIS (nao sinteticos, nao doubles) em `tests/fixtures/spc/` e
`tests/fixtures/sp/` -- ver `PROVENANCIA.md` em cada pasta.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from guaraci.importadores_proprietarios import parse_opus, parse_sp, parse_spc

_FIXTURES = Path(__file__).parent / "fixtures"


def _bloco_fake(x, y):
    return SimpleNamespace(x=np.asarray(x, dtype=float), y=np.asarray(y, dtype=float))


def _opus_file_fake(data_keys, is_opus=True, **blocos):
    ns = SimpleNamespace(is_opus=is_opus, data_keys=list(data_keys))
    for chave, bloco in blocos.items():
        setattr(ns, chave, bloco)
    return ns


def test_parse_opus_sem_brukeropus_instalado_da_importerror_claro(monkeypatch):
    monkeypatch.setitem(sys.modules, "brukeropus", None)
    with pytest.raises(ImportError, match="brukeropus"):
        parse_opus("qualquer.0")


def test_parse_opus_prefere_absorbancia_quando_disponivel(monkeypatch):
    eixo = np.linspace(4000, 10000, 50)
    y_abs = np.sin(eixo / 500.0)
    y_trans = np.cos(eixo / 500.0)
    fake = _opus_file_fake(["a", "t"],
                            a=_bloco_fake(eixo, y_abs),
                            t=_bloco_fake(eixo, y_trans))
    monkeypatch.setattr("brukeropus.read_opus", lambda fp, **kw: fake)

    X, Y = parse_opus("amostra.0")
    np.testing.assert_allclose(X, eixo)
    np.testing.assert_allclose(Y, y_abs)


def test_parse_opus_cai_para_transmitancia_sem_absorbancia(monkeypatch):
    eixo = np.linspace(4000, 10000, 30)
    y_trans = np.cos(eixo / 300.0)
    fake = _opus_file_fake(["t", "sm"],
                            t=_bloco_fake(eixo, y_trans),
                            sm=_bloco_fake(eixo, eixo * 0.0))
    monkeypatch.setattr("brukeropus.read_opus", lambda fp, **kw: fake)

    X, Y = parse_opus("amostra.0")
    np.testing.assert_allclose(Y, y_trans)


def test_parse_opus_usa_primeira_chave_disponivel_fora_da_preferencia(monkeypatch):
    eixo = np.linspace(4000, 10000, 20)
    y_igsm = np.arange(20, dtype=float)
    fake = _opus_file_fake(["igsm"], igsm=_bloco_fake(eixo, y_igsm))
    monkeypatch.setattr("brukeropus.read_opus", lambda fp, **kw: fake)

    X, Y = parse_opus("amostra.0")
    np.testing.assert_allclose(Y, y_igsm)


def test_parse_opus_arquivo_nao_opus_levanta_valueerror(monkeypatch):
    fake = _opus_file_fake([], is_opus=False)
    monkeypatch.setattr("brukeropus.read_opus", lambda fp, **kw: fake)
    with pytest.raises(ValueError, match="nao reconhecido"):
        parse_opus("nao_e_opus.txt")


def test_parse_opus_sem_blocos_de_dados_levanta_valueerror(monkeypatch):
    fake = _opus_file_fake([])
    monkeypatch.setattr("brukeropus.read_opus", lambda fp, **kw: fake)
    with pytest.raises(ValueError, match="nenhum bloco"):
        parse_opus("vazio.0")


def test_parse_opus_x_y_com_formas_incompativeis_levanta_valueerror(monkeypatch):
    fake = _opus_file_fake(["a"], a=SimpleNamespace(
        x=np.linspace(0, 1, 10), y=np.linspace(0, 1, 5)))
    monkeypatch.setattr("brukeropus.read_opus", lambda fp, **kw: fake)
    with pytest.raises(ValueError, match="inconsistentes"):
        parse_opus("corrompido.0")


def test_parse_opus_retorna_arrays_1d_numpy(monkeypatch):
    eixo = np.linspace(4000, 10000, 15)
    y = np.random.default_rng(0).normal(size=15)
    fake = _opus_file_fake(["a"], a=_bloco_fake(eixo, y))
    monkeypatch.setattr("brukeropus.read_opus", lambda fp, **kw: fake)

    X, Y = parse_opus("amostra.0")
    assert isinstance(X, np.ndarray) and X.ndim == 1
    assert isinstance(Y, np.ndarray) and Y.ndim == 1


# ---------------------------------------------------------------------------
# SPC (Galactic/Thermo) -- testado com arquivos REAIS, ver
# tests/fixtures/spc/PROVENANCIA.md.
# ---------------------------------------------------------------------------


def test_parse_spc_sem_spcfile_instalado_da_importerror_claro(monkeypatch):
    monkeypatch.setitem(sys.modules, "spcfile", None)
    with pytest.raises(ImportError, match="spcfile"):
        parse_spc(str(_FIXTURES / "spc" / "spectra.spc"))


def test_parse_spc_arquivo_real_subarquivo_unico():
    X, Y = parse_spc(str(_FIXTURES / "spc" / "spectra.spc"))
    assert isinstance(X, np.ndarray) and X.ndim == 1
    assert isinstance(Y, np.ndarray) and Y.ndim == 1
    assert X.shape == Y.shape
    assert X.size == 1911
    # eixo Raman crescente, mesmo intervalo do arquivo original
    assert X[0] == pytest.approx(400.62109375)
    np.testing.assert_allclose(Y[:3], [1487.0, 1385.0, 1441.0])


def test_parse_spc_arquivo_real_multi_subarquivo_retorna_so_o_primeiro():
    X, Y = parse_spc(str(_FIXTURES / "spc" / "nir.spc"))
    assert X.shape == Y.shape
    assert X.size == 700
    assert X[0] == pytest.approx(1100.0)


def test_parse_spc_arquivo_nao_spc_levanta_valueerror(tmp_path):
    arquivo_ruim = tmp_path / "nao_e_spc.txt"
    arquivo_ruim.write_bytes(b"isto nao e um SPC binario" * 20)
    with pytest.raises(ValueError, match="SPC valido"):
        parse_spc(str(arquivo_ruim))


# ---------------------------------------------------------------------------
# PerkinElmer .sp -- testado com arquivo REAL, ver
# tests/fixtures/sp/PROVENANCIA.md. Sem dependencia opcional (struct puro).
# ---------------------------------------------------------------------------


def test_parse_sp_arquivo_real():
    X, Y = parse_sp(str(_FIXTURES / "sp" / "spectra.sp"))
    assert isinstance(X, np.ndarray) and X.ndim == 1
    assert isinstance(Y, np.ndarray) and Y.ndim == 1
    assert X.shape == Y.shape
    assert X.size == 3301
    # conferido contra o docstring publicado de specio.plugins.sp:
    # spectra.wavelength -> [4000, 3999, 3998, ..., 702, 701, 700]
    # spectra.amplitudes -> [0.03723936, 0.03718614, 0.03713289, ...]
    np.testing.assert_allclose(X[:3], [4000.0, 3999.0, 3998.0])
    np.testing.assert_allclose(X[-3:], [702.0, 701.0, 700.0])
    np.testing.assert_allclose(Y[:3], [0.03723936, 0.03718614, 0.03713289], atol=1e-8)


def test_parse_sp_sem_assinatura_pepe_levanta_valueerror(tmp_path):
    arquivo_ruim = tmp_path / "nao_e_sp.txt"
    arquivo_ruim.write_bytes(b"XXXX" + b"\x00" * 100)
    with pytest.raises(ValueError, match="PEPE"):
        parse_sp(str(arquivo_ruim))


def test_parse_sp_truncado_apos_assinatura_levanta_valueerror(tmp_path):
    arquivo_truncado = tmp_path / "truncado.sp"
    conteudo_real = (_FIXTURES / "sp" / "spectra.sp").read_bytes()
    arquivo_truncado.write_bytes(conteudo_real[:60])
    with pytest.raises(ValueError, match="truncado ou corrompido"):
        parse_sp(str(arquivo_truncado))
