"""Rede de segurança do parsing de espectros (guaraci.dados_io) — cobre o que
tests/test_dados_io_jcamp.py (arquivos .dx completos) e test_pipeline_core.py
(parse_title, Kennard-Stone) não exercitam: o codec ASDF (DIF/DUP), o parser
ASCII genérico (parse_spectrum), extração de concentração do nome do arquivo,
e a varredura de pastas por técnica.

Uma corrupção silenciosa aqui = espectro errado entrando no pipeline sem
nenhum erro visível — o pior tipo de bug num projeto sobre medição.
"""
import numpy as np
import pytest

from guaraci.dados_io import (
    _decodificar_linha_asdf, _flush_asdf, parse_spectrum,
    _extrair_conc_filename, _listar_arquivos_espectro, _detectar_subpastas_classe,
    carregar_csv,
)


# ── ASDF codec: SQZ (dígito único) ───────────────────────────────────────────
def _tabelas():
    SQZ, DIF, DUP = {"@": 0}, {"%": 0}, {}
    for i, c in enumerate("ABCDEFGHI", 1): SQZ[c] = i
    for i, c in enumerate("abcdefghi", 1): SQZ[c] = -i
    for i, c in enumerate("JKLMNOPQR", 1): DIF[c] = i
    for i, c in enumerate("jklmnopqr", 1): DIF[c] = -i
    for i, c in enumerate("STUVWXYZs", 1): DUP[c] = i
    return SQZ, DIF, DUP


def test_asdf_sqz_simples():
    SQZ, DIF, DUP = _tabelas()
    # "100" e' o x_check; A=1, B=2, C=3 (SQZ direto, sem DIF)
    x, y = _decodificar_linha_asdf("100ABC", SQZ, DIF, DUP)
    assert x == 100.0
    assert y == [1.0, 2.0, 3.0]


def test_asdf_sqz_negativo_letras_minusculas():
    SQZ, DIF, DUP = _tabelas()
    x, y = _decodificar_linha_asdf("0abc", SQZ, DIF, DUP)
    assert x == 0.0
    assert y == [-1.0, -2.0, -3.0]


def test_asdf_zero_via_arroba():
    SQZ, DIF, DUP = _tabelas()
    x, y = _decodificar_linha_asdf("0@A@", SQZ, DIF, DUP)
    assert y == [0.0, 1.0, 0.0]


def test_asdf_multi_digito_apos_sqz():
    SQZ, DIF, DUP = _tabelas()
    # A + "23" (digitos extras apos o codigo SQZ) = valor 123
    x, y = _decodificar_linha_asdf("0A23", SQZ, DIF, DUP)
    assert y == [123.0]


def test_asdf_dif_soma_ao_anterior():
    SQZ, DIF, DUP = _tabelas()
    # A=1 (base), depois J=+1 em DIF: proximo valor = anterior(1) + 1 = 2
    x, y = _decodificar_linha_asdf("0AJ", SQZ, DIF, DUP)
    assert y == [1.0, 2.0]


def test_asdf_dif_negativo():
    SQZ, DIF, DUP = _tabelas()
    # A=5 base (usando E=5), depois j=-1 (DIF negativo): 5 + (-1) = 4
    x, y = _decodificar_linha_asdf("0Ej", SQZ, DIF, DUP)
    assert y == [5.0, 4.0]


def test_asdf_dup_repete_ultimo_valor():
    SQZ, DIF, DUP = _tabelas()
    # C=3 (SQZ), depois T=DUP[2] (segunda letra de STUVWXYZs): repete o
    # ultimo valor (3) mais 2 vezes -> total 3 ocorrencias de 3.0
    x, y = _decodificar_linha_asdf("0CT", SQZ, DIF, DUP)
    assert y == [3.0, 3.0, 3.0]


def test_asdf_linha_sem_x_check_retorna_vazio():
    SQZ, DIF, DUP = _tabelas()
    x, y = _decodificar_linha_asdf("ABC", SQZ, DIF, DUP)  # sem digito inicial
    assert x is None and y == []


def test_asdf_x_check_negativo():
    SQZ, DIF, DUP = _tabelas()
    x, y = _decodificar_linha_asdf("-50A", SQZ, DIF, DUP)
    assert x == -50.0


def test_flush_asdf_sign_zero_nao_adiciona():
    y_raw = []
    _flush_asdf(y_raw, sign=0, digits=[5], is_dif=False)
    assert y_raw == []


def test_flush_asdf_dif_sem_historico_usa_zero():
    y_raw = []
    _flush_asdf(y_raw, sign=1, digits=[7], is_dif=True)
    assert y_raw == [7]  # 0 (base vazia) + 7


# ── parse_spectrum: ASCII genérico ───────────────────────────────────────────
def test_parse_spectrum_espaco_como_separador(tmp_path):
    p = tmp_path / "esp.txt"
    p.write_text("4000.0 0.123\n4001.0 0.130\n4002.0 0.140\n")
    x, y = parse_spectrum(str(p))
    np.testing.assert_allclose(x, [4000.0, 4001.0, 4002.0])
    np.testing.assert_allclose(y, [0.123, 0.130, 0.140])


def test_parse_spectrum_virgula_decimal_arquivo(tmp_path):
    p = tmp_path / "esp.csv"
    p.write_text("4000,0;0,123\n4001,0;0,130\n")
    x, y = parse_spectrum(str(p))
    np.testing.assert_allclose(x, [4000.0, 4001.0])
    np.testing.assert_allclose(y, [0.123, 0.130])


def test_parse_spectrum_ignora_linhas_de_cabecalho(tmp_path):
    p = tmp_path / "esp.txt"
    p.write_text("Wavenumber Absorbance\n---\n4000.0 0.1\n4001.0 0.2\n")
    x, y = parse_spectrum(str(p))
    assert len(x) == 2
    np.testing.assert_allclose(x, [4000.0, 4001.0])


def test_parse_spectrum_ignora_linha_com_um_token_so(tmp_path):
    p = tmp_path / "esp.txt"
    p.write_text("4000.0\n4001.0 0.2\n")
    x, y = parse_spectrum(str(p))
    assert len(x) == 1
    assert x[0] == 4001.0


def test_parse_spectrum_detecta_bomem_binario(tmp_path):
    p = tmp_path / "esp.spectrum"
    conteudo = "Bomem File Header".encode("utf-16-le")
    p.write_bytes(conteudo + b"\x00" * 240)
    with pytest.raises(ValueError, match="Bomem"):
        parse_spectrum(str(p))


def test_parse_spectrum_detecta_perkinelmer(tmp_path):
    p = tmp_path / "esp.sp"
    p.write_bytes(b"PEPE" + b"\x00" * 60)
    with pytest.raises(ValueError, match="binary format detected"):
        parse_spectrum(str(p))


def test_parse_spectrum_detecta_binario_generico(tmp_path):
    p = tmp_path / "esp.txt"
    # bytes de alta entropia / nao-ASCII imprimivel -> detectado como binario
    p.write_bytes(bytes(range(200, 256)) * 3)
    with pytest.raises(ValueError, match="Unrecognized binary"):
        parse_spectrum(str(p))


def test_parse_spectrum_arquivo_vazio_retorna_arrays_vazios(tmp_path):
    p = tmp_path / "vazio.txt"
    p.write_text("")
    x, y = parse_spectrum(str(p))
    assert len(x) == 0 and len(y) == 0


# ── _extrair_conc_filename ───────────────────────────────────────────────────
@pytest.mark.parametrize("nome,esperado", [
    ("CAP-04-11-2020-AD-S-4.13%-T2.dx", 4.13),
    ("amostra_10%_teste.dx", 10.0),
    ("valor_com_virgula_3,5%.dx", 3.5),
    ("sem_percentual.dx", None),
    ("multiplos_5%_e_10%.dx", 5.0),  # pega a PRIMEIRA ocorrência
])
def test_extrair_conc_filename(nome, esperado):
    resultado = _extrair_conc_filename(nome)
    if esperado is None:
        assert resultado is None
    else:
        assert resultado == pytest.approx(esperado)


# ── _listar_arquivos_espectro: prioridade de extensão ────────────────────────
def test_listar_arquivos_prioriza_dx_sobre_outros(tmp_path):
    (tmp_path / "a.dx").write_text("x")
    (tmp_path / "b.txt").write_text("x")
    arqs, ext = _listar_arquivos_espectro(str(tmp_path))
    assert ext == ".dx"
    assert len(arqs) == 1


def test_listar_arquivos_cai_para_txt_sem_dx(tmp_path):
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / "c.txt").write_text("x")
    arqs, ext = _listar_arquivos_espectro(str(tmp_path))
    assert ext == ".txt"
    assert len(arqs) == 2


def test_listar_arquivos_pasta_vazia(tmp_path):
    arqs, ext = _listar_arquivos_espectro(str(tmp_path))
    assert arqs == [] and ext is None


# ── _detectar_subpastas_classe ───────────────────────────────────────────────
def test_detectar_subpastas_com_espectros(tmp_path):
    (tmp_path / "Andiroba").mkdir()
    (tmp_path / "Andiroba" / "a.dx").write_text("x")
    (tmp_path / "PastaVazia").mkdir()  # sem espectros -> nao conta
    (tmp_path / "nota.txt").write_text("x")  # arquivo solto na raiz, nao e subpasta

    subpastas = _detectar_subpastas_classe(str(tmp_path))
    nomes = [__import__("os").path.basename(p) for p in subpastas]
    assert nomes == ["Andiroba"]


def test_detectar_subpastas_raiz_inexistente():
    assert _detectar_subpastas_classe("/caminho/que/nao/existe") == []


# ── carregar_csv ─────────────────────────────────────────────────────────────
def test_carregar_csv_sem_coluna_concentracao(tmp_path):
    p = tmp_path / "dados.csv"
    p.write_text("classe,4000.0,4001.0,4002.0\n"
                 "Andiroba,0.1,0.2,0.3\n"
                 "Copaiba,0.4,0.5,0.6\n")
    wn, X, rot, conc = carregar_csv(str(p), "classe", None)
    np.testing.assert_allclose(wn, [4000.0, 4001.0, 4002.0])
    assert X.shape == (2, 3)
    assert list(rot) == ["Andiroba", "Copaiba"]
    assert conc is None


def test_carregar_csv_com_coluna_concentracao(tmp_path):
    p = tmp_path / "dados.csv"
    p.write_text("classe,teor,4000.0,4001.0\n"
                 "Andiroba,0.0,0.1,0.2\n"
                 "Andiroba,5.5,0.15,0.25\n")
    wn, X, rot, conc = carregar_csv(str(p), "classe", "teor")
    assert X.shape == (2, 2)  # colunas classe+teor excluidas do espectro
    np.testing.assert_allclose(conc, [0.0, 5.5])
    np.testing.assert_allclose(wn, [4000.0, 4001.0])
