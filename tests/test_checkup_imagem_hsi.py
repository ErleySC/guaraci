"""Checkup pre-lancamento (familia imagem): leitores ENVI/fotos e fluxo HSI."""
import os

import numpy as np
import pytest

from guaraci.dados_imagem import extract_color_features, load_image_file
from guaraci.hsi_io import load_envi_cube, load_hsi_folder_dataset


def _hdr(path, nl, ns, nb, dt, inter, extra=()):
    with open(path, "w") as f:
        f.write("\n".join(["ENVI", f"samples = {ns}", f"lines = {nl}", f"bands = {nb}",
                           "header offset = 0", f"data type = {dt}", f"interleave = {inter}",
                           "byte order = 0", *extra]) + "\n")


def test_envi_dados_em_raw_e_escala_e_micrometros(tmp_path):
    cubo = np.random.default_rng(0).uniform(0.1, 0.9, (5, 6, 4))
    d = tmp_path / "classeA" / "obj0"
    d.mkdir(parents=True)
    (d / "v0.raw").write_bytes(np.round(cubo * 10000).astype("<i2").tobytes())   # BIP
    _hdr(d / "v0.hdr", 5, 6, 4, 2, "bip",
         ("reflectance scale factor = 10000", "wavelength units = Micrometers",
          "wavelength = {0.4, 0.6, 0.8, 1.0}"))
    cubos, rot, _g, wl, _m = load_hsi_folder_dataset(str(tmp_path))
    assert len(cubos) == 1 and rot[0] == "classeA"
    np.testing.assert_allclose(cubos[0], cubo, atol=1e-4)
    np.testing.assert_allclose(wl, [400, 600, 800, 1000])
    assert cubos[0].dtype == np.float32


def test_envi_bsq_big_endian(tmp_path):
    cubo = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    (tmp_path / "c.bin").write_bytes(cubo.transpose(2, 0, 1).astype(">f4").tobytes())
    with open(tmp_path / "c.hdr", "w") as f:
        f.write("ENVI\nsamples = 3\nlines = 2\nbands = 4\ndata type = 4\ninterleave = bsq\nbyte order = 1\n")
    lido, _ = load_envi_cube(str(tmp_path / "c.bin"))
    np.testing.assert_array_equal(lido, cubo)


def test_foto_cinza_16bit_nao_satura(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    g = np.full((8, 8), 32768, dtype=np.uint16)
    tifffile.imwrite(tmp_path / "f.tif", g)
    img = load_image_file(str(tmp_path / "f.tif"))
    assert img.shape == (8, 8, 3) and 120 <= int(img[0, 0, 0]) <= 135


def test_matiz_vermelho_usa_media_circular():
    import colorsys
    rgb = [colorsys.hsv_to_rgb(h % 1.0, 0.9, 0.7) for h in (-0.02, 0.02) * 50]
    img = (np.array(rgb).reshape(10, 10, 3) * 255).round().astype(np.uint8)
    f = extract_color_features(img)
    assert min(f["H_media"], 1 - f["H_media"]) < 0.01      # vermelho, nao ~0.5 (ciano)
    assert f["H_dp"] < 0.05


def test_pipeline_hsi_aplica_snv_e_faixa(tmp_path):
    from guaraci.config import Config
    from guaraci.exemplos import _cubos_hsi
    from guaraci.hsi_pipeline import gravar_relatorio_hsi, run_hsi_pipeline
    _cubos_hsi(str(tmp_path / "cubos"))
    cfg = Config()
    cfg.hsi_dataset_folder = str(tmp_path / "cubos")
    cfg.output_folder = str(tmp_path / "saida")
    cfg.default_preprocessing = "snv_mc"
    cfg.wn_min, cfg.wn_max = 500.0, 900.0
    resumo = run_hsi_pipeline(cfg)
    wl = np.asarray(resumo["wavelengths"])
    assert wl.min() >= 500 and wl.max() <= 900
    assert any("SNV" in n for n in resumo["pre_processamento_aplicado"])
    rel = open(gravar_relatorio_hsi(cfg, resumo), encoding="utf-8").read()
    assert "SNV por pixel" in rel
    assert os.path.isfile(os.path.join(cfg.output_folder, "Graficos", "hsi",
                                       "hsi_mapa_classificacao_amostra.png"))


def test_metrica_de_classe_ausente_no_teste_e_indefinida():
    from guaraci.hsi_validation import _metricas_por_classe
    m = _metricas_por_classe({"o1": "a"}, {"o1": "a"}, ["a", "b"])
    assert m["sensibilidade"]["a"] == 1.0
    assert np.isnan(m["sensibilidade"]["b"])
