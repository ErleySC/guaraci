"""Ramos de borda dos leitores de dados (CSV, ANDI-MS, EEM, ENVI, imagem, formatos de fabricante): um teste por ramo."""
from __future__ import annotations

import io
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from guaraci import csv_robusto as cr
from guaraci import dados_imagem as di
from guaraci import eem_io
from guaraci import gcms_io
from guaraci import hsi_io
from guaraci import importadores_proprietarios as ip

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _formatos_binarios import escrever_rmn_bruker, escrever_sp, instalar_rainbow_falso  # noqa: E402


# ── csv_robusto ─────────────────────────────────────────────────────────────
def test_csv_le_bytes_buffer_binario_e_buffer_de_texto():
    assert cr._texto("﻿a;b\n1;2\n".encode("utf-8")) == "a;b\n1;2\n"                    # bytes: BOM sai
    buf = io.BytesIO("a;b\n1;2\n".encode("utf-8"))
    buf.read(3)                                                                             # posicao no meio: _texto volta ao inicio
    assert cr._texto(buf) == "a;b\n1;2\n" and buf.tell() == 0
    assert cr._texto(io.StringIO("﻿x,y\n1,2\n")) == "x,y\n1,2\n"                      # str: so' tira o BOM


def test_csv_aceita_objeto_que_so_sabe_ler_sem_seek():
    class _SoLeitura:
        def read(self):
            return b"a;b\n1;2\n"
    assert cr._texto(_SoLeitura()) == "a;b\n1;2\n"


def test_csv_em_latin1_cai_para_latin1_e_sem_separador_consistente_nao_chuta():
    assert cr._texto("amostra;aço\n".encode("latin-1")) == "amostra;aço\n"
    assert cr.detectar_separador("a b\nc d e\nf\n") is None
    assert cr.ler_csv(io.BytesIO(b"a;b\n1;2\n3;4\n")).shape == (2, 2)


# ── gcms_io (ANDI-MS/netCDF clássico gerado pelo próprio teste) ─────────────
def _escrever_andi(caminho, tempo, intensidade, com_intensidade=True):
    from scipy.io import netcdf_file
    with netcdf_file(str(caminho), "w") as f:
        f.createDimension("scan_number", len(tempo))
        v = f.createVariable("scan_acquisition_time", "d", ("scan_number",))
        v[:] = tempo
        if com_intensidade:
            f.createDimension("outra", len(intensidade))
            w = f.createVariable("total_intensity", "d", ("outra",))
            w[:] = intensidade


def test_andi_ms_le_o_tic_e_converte_segundos_em_minutos(tmp_path):
    _escrever_andi(tmp_path / "a.CDF", [0.0, 60.0, 120.0], [1.0, 5.0, 2.0])
    c = gcms_io.ler_tic_andi_ms(tmp_path / "a.CDF")
    np.testing.assert_allclose(c.tempo_min, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(c.intensidade, [1.0, 5.0, 2.0])


def test_andi_ms_recusa_variavel_ausente_e_formas_diferentes(tmp_path):
    _escrever_andi(tmp_path / "sem.CDF", [0.0, 60.0], [], com_intensidade=False)
    with pytest.raises(ValueError, match="ausente"):
        gcms_io.ler_tic_andi_ms(tmp_path / "sem.CDF")
    _escrever_andi(tmp_path / "forma.CDF", [0.0, 60.0, 120.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="formas diferentes"):
        gcms_io.ler_tic_andi_ms(tmp_path / "forma.CDF")


def test_andi_ms_dataset_acha_maiusculo_e_minusculo_e_recusa_pasta_vazia(tmp_path):
    _escrever_andi(tmp_path / "um.CDF", [0.0, 60.0], [1.0, 2.0])
    _escrever_andi(tmp_path / "dois.cdf", [0.0, 60.0], [3.0, 4.0])
    assert sorted(gcms_io.carregar_dataset_gcms(tmp_path)) == ["dois", "um"]
    with pytest.raises(ValueError, match="nenhum arquivo"):
        gcms_io.carregar_dataset_gcms(tmp_path / "vazia")


# ── eem_io ──────────────────────────────────────────────────────────────────
def _eem(tmp_path, corpo, nome="a.dat"):
    p = tmp_path / nome
    p.write_text(corpo, encoding="utf-8")
    return p


def test_eem_descarta_linha_de_tamanho_errado_ou_nao_numerica_e_ignora_linha_em_branco(tmp_path):
    corpo = "nome\t250\t260\nx\nx\n300\t1\t2\n310\t3\n\n320\t5\tabc\n330\t7\t8\n"
    exc, em, m, rel = eem_io.parse_eem_dat(_eem(tmp_path, corpo))
    assert list(em) == [300.0, 330.0] and m.shape == (2, 2)
    assert rel.n_validas == 2 and rel.n_descartadas == 2
    assert [motivo.split()[0] for _, motivo in rel.motivos_descarte] == ["esperava", "campo"]
    assert rel.taxa_descarte == pytest.approx(0.5)


def test_eem_recusa_arquivo_curto_cabecalho_sem_excitacao_e_sem_linha_valida(tmp_path):
    with pytest.raises(ValueError, match="so' 2 linhas"):
        eem_io.parse_eem_dat(_eem(tmp_path, "a\nb\n"))
    with pytest.raises(ValueError, match="sem colunas de excitacao"):
        eem_io.parse_eem_dat(_eem(tmp_path, "nome\nx\nx\n300\t1\n", "b.dat"))
    with pytest.raises(ValueError, match="nenhuma linha de dado valida"):
        eem_io.parse_eem_dat(_eem(tmp_path, "nome\t250\nx\nx\n300\tabc\n", "c.dat"))


def test_taxa_de_descarte_de_relatorio_vazio_e_zero():
    assert eem_io.ParseEEMRelatorio(n_esperado=0, n_validas=0, n_descartadas=0).taxa_descarte == 0.0


# ── hsi_io ──────────────────────────────────────────────────────────────────
def _hdr(extra="", interleave="bip", byte_order=0, dtype=4, n=(2, 2, 3)):
    return "\n".join(["ENVI", f"samples = {n[1]}", f"lines = {n[0]}", f"bands = {n[2]}", "header offset = 0", f"data type = {dtype}",
                      f"interleave = {interleave}", f"byte order = {byte_order}", extra])


def _cubo(pasta, nome, hdr, big_endian=False, n=(2, 2, 3)):
    dados = np.arange(np.prod(n), dtype=">f4" if big_endian else "<f4").reshape(n)
    (pasta / f"{nome}.bin").write_bytes(dados.tobytes())
    (pasta / f"{nome}.hdr").write_text(hdr, encoding="utf-8")
    return dados


def test_header_envi_com_wavelength_ilegivel_e_lista_nao_fechada():
    assert hsi_io._wavelengths_do_header({"wavelength": "400, abc, 410"}) is None
    campos = hsi_io.parse_envi_header("ENVI\nwavelength = {400, 410\n420")             # chave `}` nunca fecha: vai ate' o fim
    assert "420" in campos["wavelength"]


def test_cubo_envi_big_endian_e_interleave_desconhecido(tmp_path):
    esperado = _cubo(tmp_path, "be", _hdr(byte_order=1), big_endian=True)
    cubo, _ = hsi_io.load_envi_cube(str(tmp_path / "be.bin"))
    np.testing.assert_allclose(cubo, esperado)
    _cubo(tmp_path, "ruim", _hdr(interleave="xyz"))
    with pytest.raises(ValueError, match="interleave 'xyz'"):
        hsi_io.load_envi_cube(str(tmp_path / "ruim.bin"))


def test_pasta_de_cubos_soltos_na_raiz_e_subpasta_de_classe_com_arquivo_perdido(tmp_path):
    _cubo(tmp_path, "solto1", _hdr())
    _cubo(tmp_path, "solto2", _hdr())
    cubos, rotulos, *_ = hsi_io.load_hsi_folder_dataset(str(tmp_path))
    assert len(cubos) == 2 and sorted(rotulos) == ["solto1", "solto2"]
    classe = tmp_path / "pasta2" / "classe"
    (classe / "amostra1").mkdir(parents=True)
    (classe / "leiame.txt").write_text("nao e' cubo nem pasta", encoding="utf-8")       # arquivo solto dentro da classe: ignorado
    _cubo(classe / "amostra1", "c", _hdr())
    cubos, rotulos, *_ = hsi_io.load_hsi_folder_dataset(str(tmp_path / "pasta2"))
    assert len(cubos) == 1 and rotulos[0] == "classe"


def test_pasta_em_que_todo_cubo_falha_levanta_erro_claro(tmp_path, capsys):
    (tmp_path / "quebrado.bin").write_bytes(b"\x00" * 4)                                 # truncado
    (tmp_path / "quebrado.hdr").write_text(_hdr(), encoding="utf-8")
    with pytest.raises(ValueError, match="Nenhum cubo hiperespectral valido"):
        hsi_io.load_hsi_folder_dataset(str(tmp_path))
    assert "quebrado.hdr" in capsys.readouterr().out


# ── dados_imagem ────────────────────────────────────────────────────────────
def _png(caminho, valor):
    arr = np.full((24, 24, 3), valor, dtype=np.uint8)
    arr[::2] //= 2
    Image.fromarray(arr, "RGB").save(caminho)


def test_textura_glcm_devolve_as_quatro_medidas_ou_vazio_sem_scikit_image():
    img = np.random.default_rng(0).integers(0, 255, size=(20, 20, 3), dtype=np.uint8)
    try:
        import skimage  # noqa: F401
    except ImportError:
        assert di.extract_texture_features(img) == {}                       # ambiente sem o opcional: dict vazio + aviso
    else:
        f = di.extract_texture_features(img)
        assert set(f) == {"GLCM_contraste", "GLCM_homogeneidade", "GLCM_energia", "GLCM_correlacao"}
        assert di.extract_texture_features(img[..., 0])["GLCM_energia"] > 0     # imagem 2-D (ja' em cinza) tambem


def test_load_images_com_textura_e_subpasta_de_classe_com_arquivo_perdido(tmp_path):
    for i in range(2):
        (tmp_path / "A").mkdir(exist_ok=True)
        _png(tmp_path / "A" / f"a{i}.png", 100 + 20 * i)
    (tmp_path / "B" / "amostra1").mkdir(parents=True)
    (tmp_path / "B" / "leiame.txt").write_text("x", encoding="utf-8")                    # sem imagem direta e nao e' pasta
    _png(tmp_path / "B" / "amostra1" / "b0.png", 200)
    _w, X, rotulos, *_ = di.load_images(str(tmp_path), incluir_textura=True)
    assert X.shape[0] == 3 and set(rotulos) == {"A", "B"}
    try:
        import skimage  # noqa: F401
        assert X.shape[1] == len(di._NOMES_FEATURES_COR) + len(di._NOMES_FEATURES_TEXTURA)
    except ImportError:
        assert X.shape[1] >= len(di._NOMES_FEATURES_COR)


def test_load_images_sem_scikit_image_segue_so_com_as_features_de_cor_e_avisa(tmp_path, monkeypatch, capsys):
    for i in range(2):
        (tmp_path / "A").mkdir(exist_ok=True)
        _png(tmp_path / "A" / f"a{i}.png", 100 + 20 * i)
    monkeypatch.setitem(sys.modules, "skimage.feature", None)             # `from skimage.feature import ...` levanta ImportError
    _w, X, rotulos, *_ = di.load_images(str(tmp_path), incluir_textura=True)
    assert X.shape == (2, len(di._NOMES_FEATURES_COR)) and set(rotulos) == {"A"}
    assert "scikit-image nao instalado" in capsys.readouterr().out


# ── importadores de fabricante ──────────────────────────────────────────────
def _modulo_falso(monkeypatch, nome, **attrs):
    m = ModuleType(nome)
    for k, v in attrs.items():
        setattr(m, k, v)
    monkeypatch.setitem(sys.modules, nome, m)
    return m


def test_spc_sem_subarquivo_e_com_eixo_inconsistente(monkeypatch, tmp_path):
    class _Vazio:
        def __init__(self, caminho):
            pass

        def __getitem__(self, i):
            raise IndexError(i)
    _modulo_falso(monkeypatch, "spcfile", SPCFile=_Vazio)
    with pytest.raises(ValueError, match="sem nenhum subarquivo"):
        ip.parse_spc(str(tmp_path / "x.spc"))

    class _Torto(_Vazio):
        def __getitem__(self, i):
            return SimpleNamespace(x=np.arange(3.0), y=np.arange(5.0))
    _modulo_falso(monkeypatch, "spcfile", SPCFile=_Torto)
    with pytest.raises(ValueError, match="inconsistentes"):
        ip.parse_spc(str(tmp_path / "x.spc"))


def test_sp_sem_nenhum_ponto_e_recusado(tmp_path):
    caminho = escrever_sp(tmp_path / "vazio.sp", 4000.0, 400.0, [])
    with pytest.raises(ValueError, match="inconsistentes"):
        ip.parse_sp(caminho)


def test_rmn_bruker_eixo_ppm_ausente_e_eixo_de_outro_tamanho(monkeypatch, tmp_path):
    pytest.importorskip("nmrglue")
    import nmrglue as ng
    pdata = escrever_rmn_bruker(tmp_path / "exp", 9.5, -0.5, np.arange(64.0))

    def sem_parametro(dic, dados):
        raise KeyError("SW_p")
    monkeypatch.setattr(ng.bruker, "guess_udic", sem_parametro)
    with pytest.raises(ValueError, match="ausentes/invalidos"):
        ip.parse_rmn_bruker(pdata)
    monkeypatch.undo()
    monkeypatch.setattr(ng.fileiobase, "uc_from_udic", lambda udic: SimpleNamespace(ppm_scale=lambda: np.arange(10.0)))
    with pytest.raises(ValueError, match="inconsistentes"):
        ip.parse_rmn_bruker(pdata)


def test_cromatograma_hplc_com_eixo_e_intensidade_de_tamanhos_diferentes(monkeypatch, tmp_path):
    pasta = tmp_path / "amostra.D"
    pasta.mkdir()
    instalar_rainbow_falso(monkeypatch, {"UV": [(np.linspace(0, 10, 50), np.zeros(60))]})
    with pytest.raises(ValueError, match="inconsistentes"):
        ip.parse_cromatograma_hplc(str(pasta))
