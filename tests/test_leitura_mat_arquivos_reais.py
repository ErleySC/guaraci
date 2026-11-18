# -*- coding: utf-8 -*-
"""Leitor de .mat contra arquivos GRAVADOS PELO MATLAB de verdade, e robustez a arquivo estranho.

Arquivos reais: algumas instalacoes do scipy trazem ~110 .mat gravados pelo MATLAB 4.2c a 8 (Solaris, Linux,
Windows). Nao sao copiados para o repositorio (sao do scipy, BSD) e a CI pode nao te-los: so' a varredura
os usa, sobre o que existir (sem pular nada). Os casos que a varredura achou estao reproduzidos aqui com
arquivos gerados na hora; a conferencia completa (126 arquivos) e' `scripts/conferir_leitores_publicos.py`.

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


def test_todo_arquivo_gravado_pelo_matlab_le_ou_falha_com_mensagem():
    """Varre os .mat reais que a instalacao do scipy traz (varia com a versao e com o pacote: a roda da CI pode
    vir sem eles, e entao nao ha o que varrer). Nenhum teste depende de um arquivo especifico."""
    for caminho in REAIS:
        try:
            _fmt, variaveis = am.listar_variaveis(str(caminho))
        except ValueError as e:
            assert caminho.name in str(e), caminho.name
            continue
        for v in variaveis:
            try:
                a = np.asarray(v.ler())
            except ValueError as e:
                assert caminho.name in str(e), caminho.name
                continue
            assert tuple(a.shape) == v.forma, (caminho.name, v.nome)
            assert a.dtype.kind in "iufb", (caminho.name, v.nome)


def test_versoes_do_formato_dao_o_mesmo_cubo_3d(tmp_path):
    cubo = np.random.default_rng(4).random((2, 3, 4))
    for nome, fmt, comp in (("v5.mat", "5", False), ("v5z.mat", "5", True)):
        sio.savemat(tmp_path / nome, {"cubo": cubo}, format=fmt, do_compression=comp)
        _f, vs = am.listar_variaveis(str(tmp_path / nome))
        assert vs[0].forma == (2, 3, 4)
        np.testing.assert_array_equal(vs[0].ler(), cubo)
    sio.savemat(tmp_path / "v4.mat", {"x": cubo[0]}, format="4")
    np.testing.assert_array_equal(am.listar_variaveis(str(tmp_path / "v4.mat"))[1][0].ler(), cubo[0])


def test_esparsa_vira_densa_e_complexo_e_recusado_com_motivo(tmp_path):
    import scipy.sparse as sp
    esp = sp.random(3, 5, density=0.5, random_state=1, format="csc")
    sio.savemat(tmp_path / "e.mat", {"S": esp, "L": sp.csc_matrix(np.eye(4, dtype=bool)),
                                     "Z": np.array([[1 + 2j, 3 - 1j]])})
    _f, vs = am.listar_variaveis(str(tmp_path / "e.mat"))
    por_nome = {v.nome: v for v in vs}
    np.testing.assert_allclose(por_nome["S"].ler(), esp.toarray())
    assert por_nome["L"].ler().shape == (4, 4)
    with pytest.raises(ValueError, match="complexos"):
        por_nome["Z"].ler()


def test_string_do_matlab_nao_derruba_o_arquivo_e_aparece_como_ignorada(tmp_path, monkeypatch):
    """whosmat do scipy quebra com o objeto string (MCOS) e o loadmat devolve um `MatlabOpaque`. Simulado aqui
    (o scipy nao sabe gravar MCOS): foi o que o arquivo real do scipy fez na varredura."""
    class MatlabOpaque(np.ndarray):
        pass
    opaco = np.zeros(1, dtype=[("_TypeSystem", "O"), ("_Class", "O")]).view(MatlabOpaque)
    p = tmp_path / "string.mat"
    sio.savemat(p, {"X": np.ones((2, 2))})

    def whosmat_falha(*a, **k):
        raise TypeError("'NoneType' object is not iterable")
    monkeypatch.setattr(sio, "whosmat", whosmat_falha)
    monkeypatch.setattr(am, "_loadmat", lambda *a, **k: {"matstring1": opaco, "__header__": b""})
    _f, vs = am.listar_variaveis(str(p))
    assert len(vs) == 0
    texto = am.descrever_variaveis(vs)
    assert "matstring1" in texto and "objeto MATLAB" in texto and "table2array" in texto


def test_celula_e_struct_viram_variaveis_com_indice_do_matlab(tmp_path):
    celula = np.empty((1, 3), dtype=object)
    celula[0, 0], celula[0, 1], celula[0, 2] = "texto", np.arange(4.0), np.ones((2, 2))
    sio.savemat(tmp_path / "c.mat", {"c": celula, "s": {"a": np.arange(3.0), "nome": "x"}})
    _f, vs = am.listar_variaveis(str(tmp_path / "c.mat"))
    nomes = {v.nome for v in vs}
    assert {"c{2}", "c{3}", "s.a"} <= nomes
    assert ("c{1}", "texto") in vs.ignoradas and ("s.nome", "texto") in vs.ignoradas


def test_arquivo_corrompido_falha_com_o_nome_do_arquivo(tmp_path):
    p = tmp_path / "quebrado.mat"
    sio.savemat(p, {"x": np.random.default_rng(2).random((50, 50))}, do_compression=True)
    dados = bytearray(p.read_bytes())
    dados[200:260] = b"\xff" * 60                 # estraga o fluxo comprimido
    p.write_bytes(bytes(dados))
    with pytest.raises(ValueError, match="quebrado"):
        _f, vs = am.listar_variaveis(str(p))
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
