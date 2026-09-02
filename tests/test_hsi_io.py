"""Testes de hsi_io.py (Passo 94) -- leitor ENVI (.hdr + .bin) e o leitor
especifico do subconjunto DeepHS Fruit/Kaki/VIS (Passo 93).

Os testes de round-trip ENVI sao 100% sinteticos (nao dependem de rede) --
gravam um .hdr+.bin de verdade em disco com a MESMA estrutura confirmada
por leitura direta do dataset real (ver docstring de hsi_io.py) e conferem
que `load_envi_cube` devolve o cubo original. O teste contra o dataset
publico de verdade (`test_load_deephs_kaki_dataset_real`) segue o MESMO
padrao de `test_validacao_publica_mendeley.py`: PULA se
`GUARACI_DATASETS_DIR` nao apontar para o subconjunto ja baixado, nunca
falha por ausencia nem baixa nada por conta propria dentro da suite.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from guaraci.hsi_io import (HSICubeMetadata, load_deephs_kaki_dataset,
                            load_envi_cube, parse_envi_header)


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
    """Caso REAL do dataset DeepHS: o .hdr nao traz `wavelength` -- quem
    chama fornece (mesma ponte que load_deephs_kaki_dataset faz)."""
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


# ── load_deephs_kaki_dataset: contra o dataset publico de verdade ─────────

def _pasta_deephs_kaki():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "deephs_kaki_vis"
    return pasta if (pasta / "manifest.json").is_file() else None


requer_deephs_kaki = pytest.mark.skipif(
    _pasta_deephs_kaki() is None,
    reason=("dataset publico DeepHS Fruit/Kaki/VIS ausente. Baixe com "
            "'python scripts/download_datasets/baixar_deephs_kaki.py' e "
            "aponte GUARACI_DATASETS_DIR para a pasta que contem "
            "deephs_kaki_vis/."))


@requer_deephs_kaki
def test_load_deephs_kaki_dataset_real():
    pasta = str(_pasta_deephs_kaki())
    cubos, rotulos, grupos, wavelengths, meta_df = load_deephs_kaki_dataset(pasta)

    assert len(cubos) == len(rotulos) == len(grupos) == len(meta_df)
    assert len(cubos) > 0
    for cubo in cubos:
        assert cubo.shape == (64, 64, 224)
    assert wavelengths.shape == (224,)
    assert wavelengths.min() > 390 and wavelengths.max() < 1010  # Specim FX10

    assert set(rotulos) <= {"unripe", "perfect", "overripe"}

    # Objeto fisico ("frente"/"costas" da MESMA fruta) compartilha group_id.
    from collections import Counter
    contagem = Counter(grupos)
    assert any(n >= 2 for n in contagem.values()), (
        "esperado ao menos 1 grupo com >=2 gravacoes (frente+costas da "
        "mesma fruta) -- se isso falhar, a premissa de agrupamento por "
        "objeto fisico do Passo 97 precisa ser revista.")
