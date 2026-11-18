# -*- coding: utf-8 -*-
"""Leitor de .mat contra arquivos GRAVADOS PELO MATLAB de verdade, e robustez a arquivo estranho.

Arquivos reais: a instalacao do scipy traz ~110 .mat gravados pelo MATLAB 4.2c a 8 (Solaris, Linux,
Windows), inclusive um v7.3 (HDF5), objetos, celulas, esparsas, complexos, string do MATLAB e
arquivos corrompidos de proposito. Nao sao copiados para o repositorio (sao do scipy, BSD): o teste
os procura na instalacao e pula com motivo quando o pacote vem sem a pasta de testes.

O que se exige de CADA arquivo: ou lista as variaveis, ou falha com ValueError que cita o arquivo;
cada variavel listada carrega com a forma anunciada e tipo numerico real; nada de excecao crua.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from guaraci import arquivo_matrizes as am


def _pasta_scipy() -> Path:
    import scipy.io.matlab as m
    return Path(os.path.dirname(m.__file__)) / "tests" / "data"


PASTA = _pasta_scipy()
REAIS = sorted(PASTA.glob("*.mat")) if PASTA.is_dir() else []
precisa_reais = pytest.mark.skipif(not REAIS, reason="esta instalacao do scipy nao traz os .mat de teste")


@precisa_reais
@pytest.mark.parametrize("caminho", REAIS, ids=lambda p: p.name)
def test_arquivo_gravado_pelo_matlab_le_ou_falha_com_mensagem(caminho):
    try:
        _fmt, variaveis = am.listar_variaveis(str(caminho))
    except ValueError as e:
        assert caminho.name in str(e)
        return
    for v in variaveis:
        try:
            a = np.asarray(v.ler())
        except ValueError as e:
            assert caminho.name in str(e)
            continue
        assert tuple(a.shape) == v.forma, v.nome
        assert a.dtype.kind in "iufb", v.nome


@precisa_reais
def test_versoes_do_matlab_de_4_a_7_4_dao_o_mesmo_cubo_3d():
    cubos = [am.listar_variaveis(str(PASTA / f"test3dmatrix_{ver}.mat"))[1]
             for ver in ("6.1_SOL2", "6.5.1_GLNX86", "7.1_GLNX86", "7.4_GLNX86")]
    ref = cubos[0][0].ler()
    assert ref.shape == (2, 3, 4)
    for vs in cubos[1:]:
        np.testing.assert_array_equal(vs[0].ler(), ref)
    # o v4 e o v7.3 do MATLAB (sem 3D) tambem batem com o v7
    v4 = am.listar_variaveis(str(PASTA / "testdouble_4.2c_SOL2.mat"))[1][0].ler()
    fmt, v73 = am.listar_variaveis(str(PASTA / "testhdf5_7.4_GLNX86.mat"))
    assert fmt == "mat73"
    np.testing.assert_allclose(v73[0].ler(), v4)


@precisa_reais
def test_esparsa_vira_densa_e_complexo_e_recusado_com_motivo():
    _f, vs = am.listar_variaveis(str(PASTA / "testsparse_7.4_GLNX86.mat"))
    a = vs[0].ler()
    assert isinstance(a, np.ndarray) and a.shape == (3, 5)
    _f, vs = am.listar_variaveis(str(PASTA / "logical_sparse.mat"))
    assert vs[0].ler().shape == (5, 4)
    _f, vs = am.listar_variaveis(str(PASTA / "testcomplex_7.4_GLNX86.mat"))
    with pytest.raises(ValueError, match="complexos"):
        vs[0].ler()


@precisa_reais
def test_string_do_matlab_nao_derruba_o_arquivo_e_aparece_como_ignorada():
    # whosmat do scipy quebra com o objeto string (MCOS); antes o arquivo inteiro ficava ilegivel
    _f, vs = am.listar_variaveis(str(PASTA / "testmatlabstring_7_WIN64.mat"))
    assert len(vs) == 0
    texto = am.descrever_variaveis(vs)
    assert "matstring1" in texto and "objeto MATLAB" in texto and "table2array" in texto


@precisa_reais
def test_celula_e_struct_viram_variaveis_com_indice_do_matlab():
    _f, vs = am.listar_variaveis(str(PASTA / "testcell_7.4_GLNX86.mat"))
    nomes = {v.nome for v in vs}
    assert {"testcell{3}", "testcell{4}"} <= nomes
    assert any(n == "testcell{1}" for n, _c in vs.ignoradas)


@precisa_reais
@pytest.mark.parametrize("nome", ["corrupted_zlib_checksum.mat", "corrupted_zlib_data.mat",
                                  "debigged_m4.mat", "malformed1.mat"])
def test_arquivo_corrompido_falha_com_o_nome_do_arquivo(nome):
    with pytest.raises(ValueError, match=nome.replace(".", r"\.")):
        _f, vs = am.listar_variaveis(str(PASTA / nome))
        for v in vs:
            v.ler()


# ---------------------------------------------------------------- casos que o MATLAB real produz

def test_v4_le(tmp_path):
    p = tmp_path / "v4.mat"
    sio.savemat(p, {"X": np.arange(12.0).reshape(3, 4)}, format="4")
    _f, vs = am.listar_variaveis(str(p))
    np.testing.assert_array_equal(vs[0].ler(), np.arange(12.0).reshape(3, 4))


def test_arquivo_truncado_falha_com_nome(tmp_path):
    p = tmp_path / "cortado.mat"
    sio.savemat(p, {"cubo": np.random.default_rng(1).random((20, 20, 30))}, do_compression=True)
    dados = p.read_bytes()
    p.write_bytes(dados[: len(dados) // 2])
    with pytest.raises(ValueError, match="cortado"):
        _f, vs = am.listar_variaveis(str(p))
        for v in vs:
            v.ler()


def test_hdf5_puro_com_extensao_mat_do_octave(tmp_path):
    h5py = pytest.importorskip("h5py", reason="h5py (extra [hdf5]) nao instalado")
    p = tmp_path / "octave.mat"
    with h5py.File(p, "w") as f:           # layout do `save -hdf5` do Octave: grupo com type/value
        g = f.create_group("cubo")
        g.create_dataset("type", data=np.bytes_("matrix"))
        g.create_dataset("value", data=np.ones((4, 5, 6)))
    fmt, vs = am.listar_variaveis(str(p))
    assert fmt == "h5"
    assert [v.nome for v in vs] == ["cubo/value"]


def test_v73_esparsa_e_string_nao_viram_lixo_numerico(tmp_path):
    h5py = pytest.importorskip("h5py", reason="h5py (extra [hdf5]) nao instalado")
    p = tmp_path / "v73.mat"
    with h5py.File(p, "w", userblock_size=512) as f:
        sp = f.create_group("A")                         # esparsa: grupo com data/ir/jc
        sp.attrs["MATLAB_class"] = np.bytes_("double")
        sp.attrs["MATLAB_sparse"] = np.uint64(3)
        sp.create_dataset("data", data=np.array([1.0, 2.0]))
        sp.create_dataset("ir", data=np.array([0, 2], dtype=np.uint64))
        sp.create_dataset("jc", data=np.array([0, 1, 2], dtype=np.uint64))
        s = f.create_dataset("rotulos", data=np.arange(6, dtype=np.uint32)[:, None])  # string do MATLAB
        s.attrs["MATLAB_class"] = np.bytes_("string")
        s.attrs["MATLAB_object_decode"] = np.int32(3)
        x = f.create_dataset("X", data=np.ones((7, 3)))
        x.attrs["MATLAB_class"] = np.bytes_("double")
    with open(p, "r+b") as fh:
        fh.write(b"MATLAB 7.3 MAT-file, Platform: PCWIN64".ljust(116, b" "))
    _f, vs = am.listar_variaveis(str(p))
    assert [v.nome for v in vs] == ["X"] and vs[0].forma == (3, 7)
    classes = dict(vs.ignoradas)
    assert classes["A"].startswith("esparsa") and classes["rotulos"] == "string"


def test_npy_de_objeto_recusa_sem_pickle(tmp_path):
    p = tmp_path / "obj.npy"
    np.save(p, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="pickle"):
        am.listar_variaveis(str(p))
