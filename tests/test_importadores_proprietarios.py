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

Os testes de `parse_spc`/`parse_sp`/`parse_rmn_bruker`/`parse_cromatograma_hplc` (mais abaixo) rodam contra arquivos SINTETICOS gerados
por `tests/_formatos_binarios.py` (verdade conhecida, nenhum arquivo de terceiros versionado). A conformidade com arquivos REAIS de
fabricante e' validacao do mantenedor: `validacao/tests/test_formatos_reais_locais.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from guaraci.importadores_proprietarios import (
    parse_cromatograma_hplc,
    parse_opus,
    parse_rmn_bruker,
    parse_sp,
    parse_spc,
)



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
    pytest.importorskip("brukeropus")
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
    pytest.importorskip("brukeropus")
    eixo = np.linspace(4000, 10000, 30)
    y_trans = np.cos(eixo / 300.0)
    fake = _opus_file_fake(["t", "sm"],
                            t=_bloco_fake(eixo, y_trans),
                            sm=_bloco_fake(eixo, eixo * 0.0))
    monkeypatch.setattr("brukeropus.read_opus", lambda fp, **kw: fake)

    X, Y = parse_opus("amostra.0")
    np.testing.assert_allclose(Y, y_trans)


def test_parse_opus_usa_primeira_chave_de_amostra_fora_da_preferencia(monkeypatch):
    pytest.importorskip("brukeropus")
    eixo = np.linspace(4000, 10000, 20)
    y_x = np.arange(20, dtype=float)
    fake = _opus_file_fake(["rf", "igrf", "xyz"], rf=_bloco_fake(eixo, eixo * 0.0), igrf=_bloco_fake(eixo, eixo * 0.0),
                           xyz=_bloco_fake(eixo, y_x))
    monkeypatch.setattr("brukeropus.read_opus", lambda fp, **kw: fake)

    X, Y = parse_opus("amostra.0")
    np.testing.assert_allclose(Y, y_x)             # o desconhecido, nunca a referencia que vem antes


def test_parse_opus_prefere_espectros_derivados_a_referencia_e_ao_single_channel(monkeypatch):
    pytest.importorskip("brukeropus")
    eixo = np.linspace(500, 4000, 25)
    for chave in ("km", "ra", "atr", "logr"):
        fake = _opus_file_fake(["rf", "igrf", chave, "sm"], rf=_bloco_fake(eixo, eixo * 0.0), igrf=_bloco_fake(eixo, eixo * 0.0),
                               sm=_bloco_fake(eixo, eixo * 0.0), **{chave: _bloco_fake(eixo, eixo + 1.0)})
        monkeypatch.setattr("brukeropus.read_opus", lambda fp, _f=fake, **kw: _f)
        _x, y = parse_opus("amostra.0")
        np.testing.assert_allclose(y, eixo + 1.0)


@pytest.mark.parametrize("chaves", [["rf", "igrf"], ["igsm"], ["igrf", "phrf"], ["rf", "igsm", "phsm"]])
def test_parse_opus_recusa_arquivo_so_de_referencia_interferograma_ou_fase(monkeypatch, chaves):
    """Um `background.0` (so' rf/igrf) ou um interferograma lido como espectro entrava no conjunto como uma amostra."""
    pytest.importorskip("brukeropus")
    eixo = np.linspace(500, 4000, 10)
    fake = _opus_file_fake(chaves, **{k: _bloco_fake(eixo, eixo) for k in chaves})
    monkeypatch.setattr("brukeropus.read_opus", lambda fp, **kw: fake)
    with pytest.raises(ValueError, match="sem espectro de amostra"):
        parse_opus("background.0")


def test_parse_opus_serie_de_espectros_que_o_brukeropus_nao_abre_da_erro_explicito(monkeypatch):
    """Arquivo OPUS "assembled" (serie): `all_data_keys` lista sm/a, mas o leitor so' abre rf/igrf. Antes devolvia o
    background como se fosse a amostra."""
    pytest.importorskip("brukeropus")
    eixo = np.linspace(900, 4000, 10)
    fake = _opus_file_fake(["rf", "igrf"], rf=_bloco_fake(eixo, eixo), igrf=_bloco_fake(eixo, eixo))
    fake.all_data_keys = ["igsm", "sm", "a", "tr", "phsm", "rf", "igrf"]
    monkeypatch.setattr("brukeropus.read_opus", lambda fp, **kw: fake)
    with pytest.raises(ValueError, match="serie de espectros"):
        parse_opus("serie.0")


def test_parse_opus_arquivo_nao_opus_levanta_valueerror(monkeypatch):
    pytest.importorskip("brukeropus")
    fake = _opus_file_fake([], is_opus=False)
    monkeypatch.setattr("brukeropus.read_opus", lambda fp, **kw: fake)
    with pytest.raises(ValueError, match="nao reconhecido"):
        parse_opus("nao_e_opus.txt")


def test_parse_opus_sem_blocos_de_dados_levanta_valueerror(monkeypatch):
    pytest.importorskip("brukeropus")
    fake = _opus_file_fake([])
    monkeypatch.setattr("brukeropus.read_opus", lambda fp, **kw: fake)
    with pytest.raises(ValueError, match="nenhum bloco"):
        parse_opus("vazio.0")


def test_parse_opus_x_y_com_formas_incompativeis_levanta_valueerror(monkeypatch):
    pytest.importorskip("brukeropus")
    fake = _opus_file_fake(["a"], a=SimpleNamespace(
        x=np.linspace(0, 1, 10), y=np.linspace(0, 1, 5)))
    monkeypatch.setattr("brukeropus.read_opus", lambda fp, **kw: fake)
    with pytest.raises(ValueError, match="inconsistentes"):
        parse_opus("corrompido.0")


def test_parse_opus_retorna_arrays_1d_numpy(monkeypatch):
    pytest.importorskip("brukeropus")
    eixo = np.linspace(4000, 10000, 15)
    y = np.random.default_rng(0).normal(size=15)
    fake = _opus_file_fake(["a"], a=_bloco_fake(eixo, y))
    monkeypatch.setattr("brukeropus.read_opus", lambda fp, **kw: fake)

    X, Y = parse_opus("amostra.0")
    assert isinstance(X, np.ndarray) and X.ndim == 1
    assert isinstance(Y, np.ndarray) and Y.ndim == 1


# ---------------------------------------------------------------------------
# SPC (Galactic/Thermo) -- arquivos sinteticos (tests/_formatos_binarios.py).
# ---------------------------------------------------------------------------


def test_parse_spc_sem_spcfile_instalado_da_importerror_claro(monkeypatch):
    monkeypatch.setitem(sys.modules, "spcfile", None)
    with pytest.raises(ImportError, match="spcfile"):
        parse_spc("qualquer.spc")


def test_parse_spc_arquivo_nao_spc_levanta_valueerror(tmp_path):
    pytest.importorskip("spcfile")
    arquivo_ruim = tmp_path / "nao_e_spc.txt"
    arquivo_ruim.write_bytes(b"isto nao e um SPC binario" * 20)
    with pytest.raises(ValueError, match="SPC valido"):
        parse_spc(str(arquivo_ruim))


# ---------------------------------------------------------------------------
# PerkinElmer .sp -- arquivo sintetico. Sem dependencia opcional (struct puro).
# ---------------------------------------------------------------------------


def test_parse_sp_sem_assinatura_pepe_levanta_valueerror(tmp_path):
    arquivo_ruim = tmp_path / "nao_e_sp.txt"
    arquivo_ruim.write_bytes(b"XXXX" + b"\x00" * 100)
    with pytest.raises(ValueError, match="PEPE"):
        parse_sp(str(arquivo_ruim))


# ---------------------------------------------------------------------------
# RMN bruto Bruker (espectro ja processado, pdata/<N>/1r) -- diretorio sintetico.
# ---------------------------------------------------------------------------


def test_parse_rmn_bruker_sem_nmrglue_instalado_da_importerror_claro(monkeypatch):
    monkeypatch.setitem(sys.modules, "nmrglue", None)
    with pytest.raises(ImportError, match="nmrglue"):
        parse_rmn_bruker("qualquer/pdata/1")


def test_parse_rmn_bruker_diretorio_inexistente_levanta_valueerror(tmp_path):
    pytest.importorskip("nmrglue")
    with pytest.raises(ValueError, match="pdata Bruker valido"):
        parse_rmn_bruker(str(tmp_path / "nao_existe"))


def test_parse_rmn_bruker_diretorio_sem_pdata_levanta_valueerror(tmp_path):
    pytest.importorskip("nmrglue")
    (tmp_path / "vazio").mkdir()
    with pytest.raises(ValueError, match="pdata Bruker valido"):
        parse_rmn_bruker(str(tmp_path / "vazio"))


# ---------------------------------------------------------------------------
# HPLC/GC cromatograma bruto de fabricante (Agilent/Waters, via rainbow-api) -- com um `rainbow` falso
# (tests/_formatos_binarios.py): testa o NOSSO codigo; a leitura do formato real e' validacao local.
# ---------------------------------------------------------------------------


def test_parse_cromatograma_hplc_sem_rainbow_instalado_da_importerror_claro(monkeypatch):
    monkeypatch.setitem(sys.modules, "rainbow", None)
    with pytest.raises(ImportError, match="rainbow-api"):
        parse_cromatograma_hplc("qualquer.D")


def test_parse_cromatograma_hplc_diretorio_invalido_levanta_valueerror(tmp_path):
    pytest.importorskip("rainbow")
    with pytest.raises(ValueError, match="Agilent/Waters valido"):
        parse_cromatograma_hplc(str(tmp_path / "nao_existe.D"))


# --- Endurecimento de parse_sp (auditoria de seguranca 2026-09-19) ---------


# ---------------------------------------------------------------------------
# Arquivos sinteticos de verdade conhecida (nenhum arquivo de terceiros)
# ---------------------------------------------------------------------------
from tests._formatos_binarios import (  # noqa: E402
    escrever_rmn_bruker, escrever_sp, escrever_spc, instalar_rainbow_falso)


def test_parse_spc_sintetico_subarquivo_unico(tmp_path):
    pytest.importorskip("spcfile")
    y = np.array([1487.0, 1385.0, 1441.0, 1500.0, 1390.0, 1422.0, 1388.0, 1401.0])
    X, Y = parse_spc(escrever_spc(tmp_path / "a.spc", 400.0, 1800.0, [y]))
    assert X.shape == Y.shape == (8,)
    assert X[0] == pytest.approx(400.0) and X[-1] == pytest.approx(1800.0)       # eixo crescente
    np.testing.assert_allclose(Y, y)


def test_parse_spc_sintetico_multi_subarquivo_retorna_so_o_primeiro(tmp_path):
    pytest.importorskip("spcfile")
    y = np.linspace(0.0, 2.0, 20)
    X, Y = parse_spc(escrever_spc(tmp_path / "m.spc", 1100.0, 2500.0, [y, 2 * y, 3 * y]))
    assert X.size == Y.size == 20 and X[0] == pytest.approx(1100.0)
    np.testing.assert_allclose(Y, y, atol=1e-6)                                  # so' o subarquivo 0


def test_parse_sp_sintetico_ida_e_volta(tmp_path):
    y = np.linspace(0.03723936, 0.9, 33)
    X, Y = parse_sp(escrever_sp(tmp_path / "a.sp", 4000.0, 700.0, y))
    assert X.shape == Y.shape == (33,)
    np.testing.assert_allclose([X[0], X[1], X[-1]], [4000.0, 4000.0 - 3300.0 / 32, 700.0])   # eixo decrescente, como o fabricante grava
    np.testing.assert_allclose(Y, y, atol=1e-12)


def test_parse_sp_sintetico_truncado_apos_assinatura_levanta_valueerror(tmp_path):
    bruto = (tmp_path / "a.sp")
    escrever_sp(bruto, 4000.0, 700.0, np.linspace(0, 1, 33))
    cortado = tmp_path / "truncado.sp"
    cortado.write_bytes(bruto.read_bytes()[:60])
    with pytest.raises(ValueError, match="truncado ou corrompido"):
        parse_sp(str(cortado))


def _sp_com_n_pontos(caminho, n_novo: int) -> str:
    import struct
    escrever_sp(caminho, 4000.0, 700.0, np.linspace(0, 1, 33))
    bruto = caminho.read_bytes()
    cab = struct.pack("<H", 35701)
    achados = [i for i in range(len(bruto) - 12) if bruto[i:i + 2] == cab and struct.unpack("<H", bruto[i + 6:i + 8])[0] == 29995]
    assert achados, "bloco n_points nao encontrado no .sp sintetico"
    i = achados[0] + 6 + 2
    caminho.write_bytes(bruto[:i] + struct.pack("<I", n_novo) + bruto[i + 4:])
    return str(caminho)


def test_parse_sp_sintetico_recusa_n_points_gigante_sem_alocar_o_eixo(tmp_path):
    """`n_points` vem do arquivo (u32): um `.sp` de poucas centenas de bytes declarando 2**32-1 pontos nao pode alocar o eixo."""
    with pytest.raises(ValueError, match="n_points"):
        parse_sp(_sp_com_n_pontos(tmp_path / "malicioso.sp", 2**32 - 1))


def test_parse_sp_sintetico_malformado_sempre_levanta_valueerror_nunca_typeerror(tmp_path):
    """Corrompe cada byte do miolo, um por vez: o contrato e' ValueError (ou sucesso), NUNCA TypeError/IndexError vazando."""
    bruto = escrever_sp(tmp_path / "a.sp", 4000.0, 700.0, np.linspace(0, 1, 33))
    dados = Path(bruto).read_bytes()
    alvo = tmp_path / "fuzz.sp"
    for pos in range(4, min(len(dados), 400)):
        b = bytearray(dados)
        b[pos] = (b[pos] + 117) % 256                  # inclui o marcador 117 do ramo de recuo
        alvo.write_bytes(bytes(b))
        try:
            parse_sp(str(alvo))
        except ValueError:
            pass


def test_parse_rmn_bruker_sintetico_eixo_ppm_decrescente(tmp_path):
    pytest.importorskip("nmrglue")
    y = np.round(np.abs(np.sin(np.linspace(0, 20, 256))) * 1e6)
    pdata = escrever_rmn_bruker(tmp_path / "exp", 9.5, -0.5, y)
    X, Y = parse_rmn_bruker(pdata)
    assert X.shape == Y.shape == (256,)
    assert X[0] > X[-1] and X[0] == pytest.approx(9.5, abs=0.05) and X[-1] == pytest.approx(-0.5, abs=0.05)   # convencao de RMN
    np.testing.assert_allclose(Y, y)
    assert np.max(np.abs(Y)) > 1e5                       # sinal real, nao degenerado


def test_parse_cromatograma_hplc_com_rainbow_falso_primeiro_canal_uv(tmp_path, monkeypatch):
    pasta = tmp_path / "amostra.D"
    pasta.mkdir()
    t = np.linspace(0.0, 60.0, 9000)
    instalar_rainbow_falso(monkeypatch, {"UV": [(t, np.exp(-((t - 20.0) / 0.5) ** 2)), (t, 2 * t)]})
    X, Y = parse_cromatograma_hplc(str(pasta))
    assert X.shape == Y.shape == (9000,)
    assert X[0] == pytest.approx(0.0, abs=0.01) and X[-1] == pytest.approx(60.0, abs=0.01)       # minutos, crescente
    assert Y.max() == pytest.approx(1.0, abs=1e-3)       # o primeiro canal, nao o segundo


def test_parse_cromatograma_hplc_detector_inexistente_levanta_valueerror_sintetico(tmp_path, monkeypatch):
    pasta = tmp_path / "amostra.D"
    pasta.mkdir()
    t = np.linspace(0.0, 10.0, 50)
    instalar_rainbow_falso(monkeypatch, {"UV": [(t, t)]})
    with pytest.raises(ValueError, match="detector 'FID' nao encontrado"):
        parse_cromatograma_hplc(str(pasta), detector="FID")



def test_parse_cromatograma_hplc_pasta_sem_nenhum_detector_explica_masshunter_icpms(tmp_path, monkeypatch):
    """Pastas MassHunter Q-TOF / ICP-MS da Agilent nao tem detector cromatografico no `rainbow`: a mensagem deve dizer
    isso, nao so' "disponiveis: []"."""
    pasta = tmp_path / "amostra.D"
    pasta.mkdir()
    instalar_rainbow_falso(monkeypatch, {})
    with pytest.raises(ValueError, match="MassHunter"):
        parse_cromatograma_hplc(str(pasta), detector="UV")
    from guaraci.importadores_proprietarios import parse_cromatograma_hplc_canais
    with pytest.raises(ValueError, match="ICP-MS"):
        parse_cromatograma_hplc_canais(str(pasta), detector="UV")
