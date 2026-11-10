"""Testes de hsi_io.py (Passo 94) -- leitor ENVI (.hdr + .bin) e leitor
generico de pasta de cubos do proprio usuario.

Os testes de round-trip ENVI sao 100% sinteticos (nao dependem de rede) --
gravam um .hdr+.bin de verdade em disco com a MESMA estrutura confirmada
por leitura direta de cubos reais (ver docstring de hsi_io.py) e conferem
que `load_envi_cube` devolve o cubo original.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from guaraci.hsi_io import (HSICubeMetadata,
                            load_envi_cube,
                            load_hsi_folder_dataset, parse_envi_header)


def _gravar_envi(tmp_path: Path, nome: str, cubo_bip: np.ndarray,
                  interleave: str = "bip", data_type: int = 4,
                  incluir_wavelength: bool = False) -> Path:
    """Grava um par .hdr+.bin real em disco a partir de um cubo
    (linhas, colunas, bandas) — reordena para o interleave pedido antes de
    gravar, exatamente como um instrumento real gravaria."""
    n_lines, n_samples, n_bands = cubo_bip.shape
    if interleave == "bip":
        bruto = cubo_bip
    elif interleave == "bil":
        bruto = cubo_bip.transpose(0, 2, 1)  # (linha, banda, coluna)
    else:  # bsq
        bruto = cubo_bip.transpose(2, 0, 1)  # (banda, linha, coluna)

    caminho_bin = tmp_path / f"{nome}.bin"
    caminho_bin.write_bytes(bruto.astype("<f4").tobytes())

    linhas_hdr = [
        "ENVI", f"samples = {n_samples}", f"lines = {n_lines}",
        f"bands = {n_bands}", "header offset = 0", "file type = ENVI Standard",
        f"data type = {data_type}", f"interleave = {interleave}",
        "byte order = 0",
    ]
    if incluir_wavelength:
        wl = ", ".join(f"{400.0 + 2.5 * i:.2f}" for i in range(n_bands))
        linhas_hdr.append(f"wavelength = {{{wl}}}")
    caminho_hdr = tmp_path / f"{nome}.hdr"
    caminho_hdr.write_text("\n".join(linhas_hdr), encoding="utf-8")
    return caminho_bin


# ── parse_envi_header ──────────────────────────────────────────────────────

def test_parse_envi_header_campos_simples():
    texto = "ENVI\nsamples = 64\nlines = 64\nbands = 224\ninterleave = bip\n"
    campos = parse_envi_header(texto)
    assert campos["samples"] == "64"
    assert campos["bands"] == "224"
    assert campos["interleave"] == "bip"


def test_parse_envi_header_lista_multilinha():
    texto = ("ENVI\nsamples = 2\nwavelength = {\n400.0,\n410.0,\n420.0\n}\n"
              "bands = 3\n")
    campos = parse_envi_header(texto)
    assert campos["wavelength"].replace(" ", "") == "400.0,410.0,420.0"


# ── load_envi_cube: round-trip por interleave ──────────────────────────────

@pytest.mark.parametrize("interleave", ["bip", "bil", "bsq"])
def test_load_envi_cube_roundtrip_por_interleave(tmp_path, interleave):
    rng = np.random.default_rng(0)
    cubo_original = rng.normal(size=(5, 4, 10)).astype("<f4")
    caminho_bin = _gravar_envi(tmp_path, "amostra", cubo_original,
                               interleave=interleave)
    cubo_lido, meta = load_envi_cube(str(caminho_bin))
    assert cubo_lido.shape == (5, 4, 10)
    np.testing.assert_allclose(cubo_lido, cubo_original, atol=1e-5)
    assert isinstance(meta, HSICubeMetadata)
    assert meta.interleave == interleave
    assert meta.n_lines == 5 and meta.n_samples == 4 and meta.n_bands == 10


def test_load_envi_cube_le_wavelength_do_proprio_header(tmp_path):
    rng = np.random.default_rng(1)
    cubo = rng.normal(size=(3, 3, 4)).astype("<f4")
    caminho_bin = _gravar_envi(tmp_path, "com_wl", cubo,
                               incluir_wavelength=True)
    _, meta = load_envi_cube(str(caminho_bin))
    assert meta.wavelengths is not None
    assert meta.wavelengths.shape == (4,)
    np.testing.assert_allclose(meta.wavelengths, [400.0, 402.5, 405.0, 407.5])


def test_load_envi_cube_usa_wavelength_externo_quando_header_nao_tem(tmp_path):
    """Caso real de camera cujo .hdr nao traz `wavelength` -- quem
    chama fornece."""
    rng = np.random.default_rng(2)
    cubo = rng.normal(size=(2, 2, 3)).astype("<f4")
    caminho_bin = _gravar_envi(tmp_path, "sem_wl", cubo,
                               incluir_wavelength=False)
    wl_externo = np.array([500.0, 510.0, 520.0])
    _, meta = load_envi_cube(str(caminho_bin), wavelengths=wl_externo)
    np.testing.assert_allclose(meta.wavelengths, wl_externo)


def test_load_envi_cube_wavelength_com_contagem_errada_levanta_erro(tmp_path):
    rng = np.random.default_rng(3)
    cubo = rng.normal(size=(2, 2, 3)).astype("<f4")
    caminho_bin = _gravar_envi(tmp_path, "wl_errado", cubo)
    with pytest.raises(ValueError, match="nao correspondem"):
        load_envi_cube(str(caminho_bin), wavelengths=np.array([1.0, 2.0]))


def test_load_envi_cube_header_incompleto_levanta_erro_claro(tmp_path):
    caminho_hdr = tmp_path / "quebrado.hdr"
    caminho_hdr.write_text("ENVI\nsamples = 4\n", encoding="utf-8")
    (tmp_path / "quebrado.bin").write_bytes(b"\x00" * 64)
    with pytest.raises(ValueError, match="obrigatorio"):
        load_envi_cube(str(tmp_path / "quebrado.bin"))


def test_load_envi_cube_bin_truncado_levanta_erro_claro(tmp_path):
    caminho_bin = _gravar_envi(tmp_path, "trunc",
                               np.zeros((4, 4, 4), dtype="<f4"))
    # Trunca o .bin depois de gravado -- simula transferencia incompleta.
    dados = caminho_bin.read_bytes()
    caminho_bin.write_bytes(dados[:-100])
    with pytest.raises(ValueError, match="truncado"):
        load_envi_cube(str(caminho_bin))


def test_load_envi_cube_data_type_nao_suportado_levanta_erro(tmp_path):
    caminho_bin = _gravar_envi(tmp_path, "tipo_ruim",
                               np.zeros((2, 2, 2), dtype="<f4"),
                               data_type=99)
    with pytest.raises(ValueError, match="data type.*99"):
        load_envi_cube(str(caminho_bin))


# ── load_hsi_folder_dataset (Passo 111): cubo do PROPRIO usuario, sem
# manifest.json nem layout de terceiros -- Bloco 8 reaproveitado ──────────

def test_pacote_so_tem_leitores_genericos_de_hsi():
    """Leitores dedicados a conjuntos de calibracao sairam do pacote: so' ficam o ENVI e a pasta do usuario."""
    import guaraci.hsi_io as m
    leitores = sorted(n for n in dir(m) if n.startswith("load_"))
    assert leitores == ["load_envi_cube", "load_hsi_folder_dataset"]
    assert sorted(m.__all__) == ["HSICubeMetadata", "load_envi_cube", "load_hsi_folder_dataset", "parse_envi_header"]


def _cubo_sintetico(seed: int, n_lin=3, n_col=3, n_bandas=5) -> np.ndarray:
    return np.random.default_rng(seed).normal(
        size=(n_lin, n_col, n_bandas)).astype("<f4")


def test_load_hsi_folder_dataset_pasta_inexistente_levanta_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pasta nao existe"):
        load_hsi_folder_dataset(str(tmp_path / "nao_existe"))


def test_load_hsi_folder_dataset_pasta_vazia_levanta_filenotfound(tmp_path):
    (tmp_path / "vazia").mkdir()
    with pytest.raises(FileNotFoundError, match="nao contem cubos"):
        load_hsi_folder_dataset(str(tmp_path / "vazia"))


def test_load_hsi_folder_dataset_nivel_none_flat_sem_estrutura(tmp_path):
    """Subpasta por classe, sem subpasta-de-amostra nem CSV: nivel 'none',
    mae_id=None, mas processa mesmo assim (uso pratico "so jogar e rodar")."""
    raiz = tmp_path / "sem_estrutura"
    pasta_classe = raiz / "ClasseA"
    pasta_classe.mkdir(parents=True)
    for i in range(3):
        _gravar_envi(pasta_classe, f"cubo{i}", _cubo_sintetico(i))

    cubos, rotulos, mae_id, wavelengths, meta_df = load_hsi_folder_dataset(str(raiz))
    assert len(cubos) == 3
    assert set(rotulos) == {"ClasseA"}
    assert mae_id is None
    assert meta_df.attrs["grouping_guarantee"] == "none"
    assert wavelengths.shape == (5,)  # eixo simbolico (nenhum .hdr trouxe wavelength)


def test_load_hsi_folder_dataset_nivel_high_subpasta_por_amostra(tmp_path):
    """Classe/Amostra/*.hdr (2 niveis): nivel 'high', mae_id real -- cada
    amostra fisica (subpasta) vira 1 grupo, replicas dentro dela
    compartilham o grupo -- mesma hierarquia ja validada em dados_imagem.py."""
    raiz = tmp_path / "por_amostra"
    for classe in ("Verde", "Madura"):
        for a in range(2):
            pasta_amostra = raiz / classe / f"amostra{a}"
            pasta_amostra.mkdir(parents=True)
            for r in range(2):  # 2 replicas (vistas) da mesma amostra
                _gravar_envi(pasta_amostra, f"vista{r}",
                            _cubo_sintetico(hash((classe, a, r)) % 1000))

    cubos, rotulos, mae_id, wavelengths, meta_df = load_hsi_folder_dataset(str(raiz))
    assert len(cubos) == 8  # 2 classes x 2 amostras x 2 vistas
    assert mae_id is not None
    assert meta_df.attrs["grouping_guarantee"] == "high"
    grupos, contagens = np.unique(mae_id, return_counts=True)
    assert len(grupos) == 4
    assert set(contagens) == {2}
    assert len({g for g in grupos if g.startswith("Verde/")}) == 2


def test_load_hsi_folder_dataset_nivel_medium_csv_associacao(tmp_path):
    raiz = tmp_path / "com_csv"
    pasta_classe = raiz / "Classe"
    pasta_classe.mkdir(parents=True)
    for i in range(4):
        _gravar_envi(pasta_classe, f"c{i}", _cubo_sintetico(i))
    (raiz / "amostras.csv").write_text(
        "arquivo,id_amostra\n"
        "Classe/c0.hdr,S1\nClasse/c1.hdr,S1\n"
        "Classe/c2.hdr,S2\nClasse/c3.hdr,S2\n",
        encoding="utf-8",
    )
    cubos, rotulos, mae_id, wavelengths, meta_df = load_hsi_folder_dataset(str(raiz))
    assert len(cubos) == 4
    assert meta_df.attrs["grouping_guarantee"] == "medium"
    grupos, contagens = np.unique(mae_id, return_counts=True)
    assert sorted(grupos) == ["S1", "S2"]
    assert set(contagens) == {2}


def test_load_hsi_folder_dataset_bandas_incompativeis_levanta_erro(tmp_path):
    """2 cubos com numero de bandas DIFERENTE na mesma pasta -- nao da pra'
    formar uma matriz X coerente (sensores/cameras incompativeis)."""
    raiz = tmp_path / "bandas_mistas"
    pasta_classe = raiz / "Classe"
    pasta_classe.mkdir(parents=True)
    _gravar_envi(pasta_classe, "a", _cubo_sintetico(0, n_bandas=5))
    _gravar_envi(pasta_classe, "b", _cubo_sintetico(1, n_bandas=7))
    with pytest.raises(ValueError, match="nao da pra"):
        load_hsi_folder_dataset(str(raiz))


def test_load_hsi_folder_dataset_usa_wavelength_do_header_quando_presente(tmp_path):
    raiz = tmp_path / "com_wl"
    pasta_classe = raiz / "Classe"
    pasta_classe.mkdir(parents=True)
    _gravar_envi(pasta_classe, "a", _cubo_sintetico(0), incluir_wavelength=True)
    _, _, _, wavelengths, _ = load_hsi_folder_dataset(str(raiz))
    np.testing.assert_allclose(
        wavelengths, [400.0, 402.5, 405.0, 407.5, 410.0])


def test_load_hsi_folder_dataset_arquivo_corrompido_e_pulado_com_aviso(
        tmp_path, capsys):
    raiz = tmp_path / "com_corrompida"
    pasta_classe = raiz / "Classe"
    pasta_classe.mkdir(parents=True)
    _gravar_envi(pasta_classe, "boa", _cubo_sintetico(0))
    (pasta_classe / "quebrada.hdr").write_text(
        "ENVI\nsamples = 4\n", encoding="utf-8")
    (pasta_classe / "quebrada.bin").write_bytes(b"\x00" * 4)

    cubos, *_ = load_hsi_folder_dataset(str(raiz))
    assert len(cubos) == 1
    assert "ERROR" in capsys.readouterr().out
