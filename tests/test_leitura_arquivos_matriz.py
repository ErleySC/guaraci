# -*- coding: utf-8 -*-
"""Leitura de cubo hiperespectral (modo hsi) e de tabela (modo csv) a partir de MATLAB .mat (v5 e v7.3),
NumPy (.npy/.npz), HDF5 (.h5) e Excel (.xlsx): deteccao do dado e do eixo espectral, ordem dos eixos,
mensagens de ambiguidade e o extra opcional h5py (pula com motivo quando falta).

Tudo sintetico. O .mat v7.3 e' gravado como o MATLAB grava: arquivo HDF5 com userblock de 512 bytes e
cabecalho "MATLAB 7.3 MAT-file", eixos em ordem de coluna (transpostos para o h5py) e atributo
MATLAB_class em cada variavel -- o mesmo que `save(..., '-v7.3')` produz.
"""
from __future__ import annotations

import builtins
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import scipy.io as sio

from guaraci import arquivo_matrizes as am
from guaraci.hsi_arquivo import ler_cubo_arquivo
from guaraci.hsi_io import load_hsi_folder_dataset
from guaraci.tabela_arquivo import ler_tabela_arquivo

RNG = np.random.default_rng(2099)
WL = np.linspace(400.0, 1000.0, 24)


def _cubo(l: int = 6, c: int = 9, b: int = 24) -> np.ndarray:
    return RNG.random((l, c, b)).astype(np.float32)


_CLASSE_MATLAB = {"float64": "double", "float32": "single", "uint16": "uint16", "uint8": "uint8",
                  "int32": "int32", "bool": "logical"}


def _salvar_mat73(caminho: Path, variaveis: Dict[str, object]) -> None:
    """.mat v7.3 como o MATLAB grava (ver docstring do modulo). Textos: char (uint16) e cellstr (#refs#)."""
    h5py = pytest.importorskip("h5py", reason="h5py (extra [hdf5]) nao instalado")
    with h5py.File(caminho, "w", userblock_size=512) as f:
        refs = f.create_group("#refs#")
        for nome, valor in variaveis.items():
            if isinstance(valor, list):               # cellstr n x 1
                rs = []
                for i, s in enumerate(valor):
                    d = refs.create_dataset(f"{nome}_{i}", data=np.array([[ord(ch)] for ch in s], dtype=np.uint16))
                    d.attrs["MATLAB_class"] = np.bytes_("char")
                    rs.append(d.ref)
                ds = f.create_dataset(nome, data=np.array([rs], dtype=h5py.ref_dtype))
                ds.attrs["MATLAB_class"] = np.bytes_("cell")
                continue
            a = np.asarray(valor)
            if a.ndim == 1:
                a = a[:, None]                         # vetor coluna N x 1 do MATLAB
            ds = f.create_dataset(nome, data=(a.astype(np.uint8) if a.dtype == bool else a).T)
            ds.attrs["MATLAB_class"] = np.bytes_(_CLASSE_MATLAB[str(a.dtype)])
    with open(caminho, "r+b") as fh:
        fh.write(b"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: Thu Jan  1 00:00:00 2099 HDF5 schema 1.00 ."
                 .ljust(116, b" "))


# ---------------------------------------------------------------- deteccao do formato
def test_formato_distingue_mat_v5_de_v73(tmp_path):
    sio.savemat(tmp_path / "v5.mat", {"cubo": _cubo()})
    _salvar_mat73(tmp_path / "v73.mat", {"cubo": _cubo()})
    assert am.formato_do_arquivo(str(tmp_path / "v5.mat")) == "mat5"
    assert am.formato_do_arquivo(str(tmp_path / "v73.mat")) == "mat73"


# ---------------------------------------------------------------- cubo HSI
@pytest.mark.parametrize("versao", ["v5", "v73"])
def test_cubo_mat_linhas_colunas_bandas_com_eixo(tmp_path, versao):
    cubo = _cubo()
    arq = tmp_path / "amostra.mat"
    vars_ = {"cubo": cubo, "wavelength": WL, "mascara": cubo[..., 0] > 0.5}
    if versao == "v5":
        sio.savemat(arq, vars_)
    else:
        _salvar_mat73(arq, vars_)
    r = ler_cubo_arquivo(str(arq))
    assert r.cubo.shape == (6, 9, 24) and r.cubo.dtype == np.float32
    np.testing.assert_allclose(r.cubo, cubo)
    np.testing.assert_allclose(r.wavelengths, WL)
    assert r.info["variavel_dados"] == "cubo" and r.info["eixo_espectral"] == "ultimo"
    assert r.info["formato"] == ("mat5" if versao == "v5" else "mat73")


@pytest.mark.parametrize("versao", ["v5", "v73"])
def test_bandas_no_primeiro_eixo_achadas_pelo_vetor(tmp_path, versao):
    cubo = _cubo()
    arq = tmp_path / "x.mat"
    vars_ = {"hsi": np.transpose(cubo, (2, 0, 1)).copy(), "wl": WL}
    (sio.savemat if versao == "v5" else _salvar_mat73)(arq, vars_)
    r = ler_cubo_arquivo(str(arq))
    assert r.info["eixo_espectral"] == "primeiro"
    np.testing.assert_allclose(r.cubo, cubo)


def test_eixo_decrescente_vira_crescente_com_as_bandas(tmp_path):
    cubo = _cubo()
    sio.savemat(tmp_path / "a.mat", {"cubo": cubo[..., ::-1].copy(), "numero_onda": WL[::-1].copy()})
    r = ler_cubo_arquivo(str(tmp_path / "a.mat"))
    np.testing.assert_allclose(r.wavelengths, WL)
    np.testing.assert_allclose(r.cubo, cubo)


def test_struct_do_matlab(tmp_path):
    cubo = _cubo()
    sio.savemat(tmp_path / "s.mat", {"dados": {"cubo": cubo, "wl": WL}})
    r = ler_cubo_arquivo(str(tmp_path / "s.mat"))
    assert r.info["variavel_dados"] == "dados.cubo"
    np.testing.assert_allclose(r.cubo, cubo)


def test_npy_e_npz_e_h5(tmp_path):
    h5py = pytest.importorskip("h5py", reason="h5py (extra [hdf5]) nao instalado")
    cubo = _cubo()
    np.save(tmp_path / "a.npy", cubo)
    np.savez(tmp_path / "b.npz", cube=cubo, wavelengths=WL)
    with h5py.File(tmp_path / "c.h5", "w") as f:
        ds = f.create_dataset("grupo/cubo", data=cubo)
        ds.attrs["wavelength"] = WL
    for nome in ("a.npy", "b.npz", "c.h5"):
        r = ler_cubo_arquivo(str(tmp_path / nome))
        np.testing.assert_allclose(r.cubo, cubo)
    assert ler_cubo_arquivo(str(tmp_path / "a.npy")).wavelengths is None
    np.testing.assert_allclose(ler_cubo_arquivo(str(tmp_path / "c.h5")).wavelengths, WL)


def test_sem_eixo_assume_ultimo_e_avisa(tmp_path):
    np.save(tmp_path / "a.npy", _cubo(6, 9, 24))
    r = ler_cubo_arquivo(str(tmp_path / "a.npy"))
    assert r.info["eixo_espectral"] == "ultimo"
    assert any("assumido que as bandas sao o ULTIMO eixo" in a for a in r.info["avisos"])


def test_forma_bandas_linhas_colunas_sem_eixo_para_e_pede_escolha(tmp_path):
    cubo = _cubo(8, 8, 24)
    np.save(tmp_path / "a.npy", np.transpose(cubo, (2, 0, 1)).copy())       # 24 x 8 x 8
    with pytest.raises(ValueError, match="hsi_eixo_espectral"):
        ler_cubo_arquivo(str(tmp_path / "a.npy"))
    r = ler_cubo_arquivo(str(tmp_path / "a.npy"), eixo_espectral="primeiro")
    np.testing.assert_allclose(r.cubo, cubo)


def test_varios_cubos_lista_e_usuario_escolhe(tmp_path):
    a, b = _cubo(), _cubo()
    sio.savemat(tmp_path / "a.mat", {"cubo_a": a, "cubo_b": b, "wl": WL})
    with pytest.raises(ValueError, match="arquivo_variavel_dados") as e:
        ler_cubo_arquivo(str(tmp_path / "a.mat"))
    assert "cubo_a" in str(e.value) and "cubo_b" in str(e.value)
    np.testing.assert_allclose(ler_cubo_arquivo(str(tmp_path / "a.mat"), variavel_dados="cubo_b").cubo, b)


def test_referencias_no_mesmo_arquivo_nao_confundem(tmp_path):
    cubo = _cubo()
    sio.savemat(tmp_path / "a.mat", {"amostra": cubo, "white_ref": _cubo(), "dark_ref": _cubo(), "wl": WL})
    assert ler_cubo_arquivo(str(tmp_path / "a.mat")).info["variavel_dados"] == "amostra"


def test_vetor_com_tamanho_de_dois_eixos_e_ambiguo(tmp_path):
    sio.savemat(tmp_path / "a.mat", {"cubo": _cubo(24, 9, 24), "wl": WL})
    with pytest.raises(ValueError, match="mais de um eixo"):
        ler_cubo_arquivo(str(tmp_path / "a.mat"))
    r = ler_cubo_arquivo(str(tmp_path / "a.mat"), eixo_espectral="ultimo")
    assert r.cubo.shape == (24, 9, 24)


def test_nome_inexistente_lista_as_variaveis(tmp_path):
    sio.savemat(tmp_path / "a.mat", {"cubo": _cubo(), "wl": WL})
    with pytest.raises(ValueError, match="Variaveis encontradas: 'cubo'"):
        ler_cubo_arquivo(str(tmp_path / "a.mat"), variavel_dados="nao_existe")


def test_matriz_desdobrada_explica_o_reshape(tmp_path):
    sio.savemat(tmp_path / "a.mat", {"X": RNG.random((54, 24)), "wl": WL})
    with pytest.raises(ValueError, match="reshape"):
        ler_cubo_arquivo(str(tmp_path / "a.mat"))


def test_cubo_inteiro_avisa_contagem_bruta(tmp_path):
    sio.savemat(tmp_path / "a.mat", {"cubo": (RNG.random((4, 5, 24)) * 4000).astype(np.uint16), "wl": WL})
    r = ler_cubo_arquivo(str(tmp_path / "a.mat"))
    assert any("contagem bruta" in a for a in r.info["avisos"])


def test_sem_h5py_mensagem_com_comando_de_instalacao(tmp_path, monkeypatch):
    _salvar_mat73(tmp_path / "a.mat", {"cubo": _cubo(), "wl": WL})
    real = builtins.__import__

    def _sem_h5py(nome, *a, **k):
        if nome == "h5py":
            raise ImportError("sem h5py")
        return real(nome, *a, **k)
    monkeypatch.setattr(builtins, "__import__", _sem_h5py)
    with pytest.raises(ImportError, match=r'pip install "\.\[hdf5\]"') as e:
        ler_cubo_arquivo(str(tmp_path / "a.mat"))
    assert "-v7" in str(e.value)


# ---------------------------------------------------------------- pasta HSI (modo hsi)
@pytest.mark.parametrize("versao", ["v5", "v73"])
def test_pasta_hsi_com_mat_por_classe(tmp_path, versao, capsys):
    salvar = sio.savemat if versao == "v5" else _salvar_mat73
    for classe in ("classe_a", "classe_b"):
        (tmp_path / classe).mkdir()
        for i in range(2):
            salvar(tmp_path / classe / f"objeto_{i}.mat", {"cubo": _cubo(), "wl": WL})
    cubos, rotulos, mae_id, wl, meta = load_hsi_folder_dataset(str(tmp_path))
    assert len(cubos) == 4 and sorted(set(rotulos)) == ["classe_a", "classe_b"]
    assert all(c.shape == (6, 9, 24) for c in cubos)
    np.testing.assert_allclose(wl, WL)
    assert set(meta["formato"]) == {"mat5" if versao == "v5" else "mat73"}


def test_pasta_hsi_calibracao_branco_escuro_recusa_mat(tmp_path):
    sio.savemat(tmp_path / "a.mat", {"cubo": _cubo(), "wl": WL})
    with pytest.raises(ValueError, match="so' vale para cubos ENVI"):
        load_hsi_folder_dataset(str(tmp_path), calibracao="branco_escuro")


def test_pasta_hsi_eixos_diferentes_recusa(tmp_path):
    sio.savemat(tmp_path / "a.mat", {"cubo": _cubo(), "wl": WL})
    sio.savemat(tmp_path / "b.mat", {"cubo": _cubo(), "wl": WL + 50})
    with pytest.raises(ValueError, match="difere"):
        load_hsi_folder_dataset(str(tmp_path))


def test_pasta_hsi_opcoes_chegam_ao_leitor(tmp_path):
    cubo = _cubo(8, 8, 24)
    np.save(tmp_path / "a.npy", np.transpose(cubo, (2, 0, 1)).copy())
    cubos, *_ = load_hsi_folder_dataset(str(tmp_path), eixo_espectral="primeiro")
    np.testing.assert_allclose(cubos[0], cubo)


# ---------------------------------------------------------------- tabela (modo csv)
@pytest.mark.parametrize("versao", ["v5", "v73"])
def test_tabela_mat_com_eixo_classe_e_y(tmp_path, versao):
    X = RNG.random((10, 24))
    classes = ["A"] * 5 + ["B"] * 5
    arq = tmp_path / "t.mat"
    if versao == "v5":
        sio.savemat(arq, {"X": X, "wl": WL, "classe": np.array(classes, dtype=object), "y": np.arange(10.0)})
    else:
        _salvar_mat73(arq, {"X": X, "wl": WL, "classe": classes, "y": np.arange(10.0)})
    df = ler_tabela_arquivo(str(arq))
    assert list(df["classe"]) == classes and list(df["y"]) == list(np.arange(10.0))
    num = [c for c in df.columns if c not in ("classe", "y")]
    np.testing.assert_allclose([float(c) for c in num], WL)
    np.testing.assert_allclose(df[num].to_numpy(), X)


def test_tabela_transposta_pelo_eixo(tmp_path):
    X = RNG.random((10, 24))
    sio.savemat(tmp_path / "t.mat", {"espectros": X.T, "wavelength": WL})
    df = ler_tabela_arquivo(str(tmp_path / "t.mat"))
    assert df.shape == (10, 24)


def test_tabela_xcal_xval_pede_escolha(tmp_path):
    sio.savemat(tmp_path / "t.mat", {"Xcal": RNG.random((10, 24)), "Xval": RNG.random((4, 24)), "wl": WL})
    with pytest.raises(ValueError, match="Xcal"):
        ler_tabela_arquivo(str(tmp_path / "t.mat"))
    assert ler_tabela_arquivo(str(tmp_path / "t.mat"), variavel_dados="Xval").shape[0] == 4


def test_modo_csv_le_mat_e_xlsx_pelo_mesmo_caminho(tmp_path):
    from guaraci.dados_io import load_csv_com_grupos
    X = RNG.random((10, 24))
    classes = np.array(["A"] * 5 + ["B"] * 5, dtype=object)
    sio.savemat(tmp_path / "t.mat", {"X": X, "wl": WL, "classe": classes})
    wn, Xl, rot, _conc, _g = load_csv_com_grupos(str(tmp_path / "t.mat"), "classe", None)
    np.testing.assert_allclose(wn, WL)
    np.testing.assert_allclose(Xl, X)
    assert list(rot) == list(classes)
    pytest.importorskip("openpyxl", reason="openpyxl (extra [reports]) nao instalado")
    import pandas as pd
    df = pd.DataFrame(X, columns=[float(w) for w in WL])
    df.insert(0, "classe", classes)
    df.to_excel(tmp_path / "t.xlsx", index=False)
    wn2, X2, rot2, _c, _g2 = load_csv_com_grupos(str(tmp_path / "t.xlsx"), "classe", None)
    np.testing.assert_allclose(wn2, WL)
    np.testing.assert_allclose(X2, X)
    assert list(rot2) == list(classes)


def test_validacao_da_fonte_aceita_mat_e_pasta_hsi(tmp_path):
    from guaraci.config import Config
    from guaraci.config_io import validar_fonte_de_dados
    sio.savemat(tmp_path / "t.mat", {"X": RNG.random((10, 24)), "wl": WL,
                                     "classe": np.array(["A"] * 10, dtype=object)})
    cfg = Config()
    cfg.mode, cfg.csv_file, cfg.class_column = "csv", str(tmp_path / "t.mat"), "classe"
    ok, msg = validar_fonte_de_dados(cfg)
    assert ok, msg
    (tmp_path / "hsi").mkdir()
    sio.savemat(tmp_path / "hsi" / "a.mat", {"cubo": _cubo(), "wl": WL})
    cfg.mode, cfg.hsi_dataset_folder = "hsi", str(tmp_path / "hsi")
    ok, msg = validar_fonte_de_dados(cfg)
    assert ok and "1 cubos" in msg


# ---------------------------------------------------------------- ponta a ponta (modo hsi com .mat)
def test_pipeline_hsi_roda_com_cubos_mat_v5_e_v73(tmp_path):
    """O mesmo cenario de `test_hsi_pipeline` (2 classes x 3 objetos, 64 x 64 pixels), com os cubos em
    .mat -- metade v5, metade v7.3 -- gravados BANDAS x LINHAS x COLUNAS com vetor `lambda`: a leitura acha
    o eixo pelo vetor e o pipeline inteiro roda."""
    pytest.importorskip("h5py", reason="h5py (extra [hdf5]) nao instalado")
    from guaraci.config import Config
    from guaraci.hsi_pipeline import run_hsi_pipeline
    rng = np.random.default_rng(0)
    n_lin, n_col, n_b = 64, 64, 6
    wl = np.array([450.0, 550.0, 650.0, 750.0, 850.0, 950.0])
    yy, xx = np.ogrid[:n_lin, :n_col]
    dist = np.sqrt((yy - 31.5) ** 2 + (xx - 31.5) ** 2)
    alpha = np.clip((0.35 * 64 - dist) / 4.0 + 0.5, 0.0, 1.0)[..., None]
    raiz = tmp_path / "hsi"
    for classe, nivel in (("classe_a", 0.85), ("classe_b", 0.45)):
        for a in range(3):
            pasta = raiz / classe / f"objeto{a}"
            pasta.mkdir(parents=True)
            fundo = rng.normal(0.05, 0.01, (n_lin, n_col, n_b))
            obj = rng.normal(nivel, 0.01, (n_lin, n_col, n_b))
            cubo = np.transpose(alpha * obj + (1 - alpha) * fundo, (2, 0, 1)).astype(np.float32)
            vars_ = {"hsi": cubo, "lambda": wl}
            (sio.savemat if a % 2 == 0 else _salvar_mat73)(pasta / "vista0.mat", vars_)
    cfg = Config(mode="hsi", hsi_dataset_folder=str(raiz), output_root_folder=str(tmp_path / "saida"),
                 output_format="png")
    resumo = run_hsi_pipeline(cfg)
    assert resumo["n_gravacoes_total"] == 6 and resumo["n_gravacoes_aceitas"] == 6
    assert resumo["grouping_guarantee"] == "high"
