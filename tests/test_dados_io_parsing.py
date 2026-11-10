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
    load_csv,
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
    # C=3 (SQZ), depois T=DUP 2: JCAMP-DX 4.24 -- a contagem DUP e' o TOTAL de
    # ocorrencias, contando a primeira -> 2 ocorrencias de 3.0 (antes: 3)
    x, y = _decodificar_linha_asdf("0CT", SQZ, DIF, DUP)
    assert y == [3.0, 3.0]


def test_asdf_dup_apos_dif_repete_a_diferenca():
    """DIFDUP (export do OPUS e da maioria dos softwares FTIR): depois de um DIF o
    DUP repete a DIFERENCA. 10, +2 (x3) -> 10, 12, 14, 16 (antes: 10, 12, 12, 12, 12)."""
    SQZ, DIF, DUP = _tabelas()
    x, y = _decodificar_linha_asdf("0A0KU", SQZ, DIF, DUP)
    assert y == [10.0, 12.0, 14.0, 16.0]


def test_asdf_dup_com_contagem_de_dois_digitos():
    SQZ, DIF, DUP = _tabelas()
    x, y = _decodificar_linha_asdf("0A%S1", SQZ, DIF, DUP)     # 1, depois diferenca 0 onze vezes
    assert y == [1.0] * 12


def test_parse_dx_difdup_decrescente_reconstroi_o_espectro(tmp_path):
    """FTIR decrescente (4000 -> 3990) em DIFDUP com Y-check (cada linha comeca repetindo o ultimo Y da anterior)."""
    from guaraci.dados_io import parse_dx
    linhas = ["##TITLE=t", "##XFACTOR=1", "##YFACTOR=0.001", "##FIRSTX=4000", "##LASTX=3990", "##NPOINTS=11",
              "##XYDATA=(X++(Y..Y))",
              "4000A00KU",        # 100, +2 (3 ocorrencias) -> 100 102 104 106
              "3997A06%Ull",      # Y-check 106, 0 (3 ocorrencias) -> 106 106 106, -3, -3 -> 103 100
              "3992A00lJ",        # Y-check 100, -3 -> 97, +1 -> 98
              "##END="]
    arq = tmp_path / "difdup.dx"
    arq.write_text("\n".join(linhas) + "\n", encoding="ascii")
    x, y = parse_dx(str(arq))
    np.testing.assert_allclose(x, np.linspace(4000, 3990, 11))
    np.testing.assert_allclose(y, np.array([100, 102, 104, 106, 106, 106, 106, 103, 100, 97, 98]) * 0.001, atol=1e-12)


def test_parse_dx_affn_texto_puro(tmp_path):
    """AFFN (numeros em texto separados por espaco: OMNIC, Spectrum, WiRE, LabSpec) era lido como todo NaN."""
    from guaraci.dados_io import parse_dx
    arq = tmp_path / "affn.dx"
    arq.write_text("\n".join(["##TITLE=t", "##XFACTOR=1", "##YFACTOR=1", "##FIRSTX=4000.0", "##LASTX=3990.0",
                               "##NPOINTS=6", "##XYDATA=(X++(Y..Y))", "4000.0 0.1 0.2 1.5E-01",
                               "3994.0 -0.4 .5 6e-1", "##END="]) + "\n", encoding="ascii")
    x, y = parse_dx(str(arq))
    np.testing.assert_allclose(y, [0.1, 0.2, 0.15, -0.4, 0.5, 0.6])
    arq.write_text(arq.read_text().replace("4000.0 0.1 0.2 1.5E-01", "4000+100+200+150").replace("3994.0 -0.4 .5 6e-1", "3994-400+500+600")
                   .replace("##YFACTOR=1", "##YFACTOR=0.001"), encoding="ascii")          # PAC
    x, y = parse_dx(str(arq))
    np.testing.assert_allclose(y, [0.1, 0.2, 0.15, -0.4, 0.5, 0.6])


def test_parse_dx_xypoints_pares(tmp_path):
    from guaraci.dados_io import parse_dx
    arq = tmp_path / "xy.dx"
    arq.write_text("##TITLE=t\n##XFACTOR=1\n##YFACTOR=1\n##NPOINTS=3\n##XYPOINTS=(XY..XY)\n"
                   "200.0, 0.5; 201.0, 0.6\n202.0, 0.7\n##END=\n", encoding="ascii")
    x, y = parse_dx(str(arq))
    np.testing.assert_allclose(x, [200, 201, 202]); np.testing.assert_allclose(y, [0.5, 0.6, 0.7])


def test_parse_dx_sem_nenhum_ponto_decodificado_levanta(tmp_path):
    """Espectro todo NaN nao pode sair do parser em silencio (load_dx so' conta erro se o parser levantar)."""
    from guaraci.dados_io import parse_dx
    arq = tmp_path / "ruim.dx"
    arq.write_text("##TITLE=t\n##FIRSTX=1\n##LASTX=3\n##NPOINTS=3\n##XYDATA=(X++(Y..Y))\n?? ?? ??\n##END=\n", encoding="ascii")
    with pytest.raises(ValueError, match="nenhum ponto"):
        parse_dx(str(arq))


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


@pytest.mark.parametrize("marca", ["Bomem File Header", "Horizon MB Spectrum"])
def test_parse_spectrum_recusa_binario_bomem_por_qualquer_das_duas_marcas_com_a_instrucao_de_exportar(tmp_path, marca):
    """Cada marca ('Bomem' OU 'Horizon') basta; a recusa traz o formato e a instrucao de reexportar em JCAMP-DX (nao cai no parser ASCII)."""
    p = tmp_path / "esp.spectrum"
    p.write_bytes(marca.encode("utf-16-le") + b"\x00" * 240)
    with pytest.raises(ValueError, match="ABB Bomem Horizon MB") as e:
        parse_spectrum(str(p))
    assert "JCAMP-DX" in str(e.value) and "esp.spectrum" in str(e.value)


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
    ("ESB-03-01-2099-AD-S-4.13%-T2.dx", 4.13),
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
    (tmp_path / "Classe A").mkdir()
    (tmp_path / "Classe A" / "a.dx").write_text("x")
    (tmp_path / "PastaVazia").mkdir()  # sem espectros -> nao conta
    (tmp_path / "nota.txt").write_text("x")  # arquivo solto na raiz, nao e subpasta

    subpastas = _detectar_subpastas_classe(str(tmp_path))
    nomes = [__import__("os").path.basename(p) for p in subpastas]
    assert nomes == ["Classe A"]


def test_detectar_subpastas_raiz_inexistente():
    assert _detectar_subpastas_classe("/caminho/que/nao/existe") == []


# ── load_csv ─────────────────────────────────────────────────────────────
def test_carregar_csv_sem_coluna_concentracao(tmp_path):
    p = tmp_path / "dados.csv"
    p.write_text("classe,4000.0,4001.0,4002.0\n"
                 "Classe A,0.1,0.2,0.3\n"
                 "Classe B,0.4,0.5,0.6\n")
    wn, X, rot, conc = load_csv(str(p), "classe", None)
    np.testing.assert_allclose(wn, [4000.0, 4001.0, 4002.0])
    assert X.shape == (2, 3)
    assert list(rot) == ["Classe A", "Classe B"]
    assert conc is None


def test_carregar_csv_com_coluna_concentracao(tmp_path):
    p = tmp_path / "dados.csv"
    p.write_text("classe,teor,4000.0,4001.0\n"
                 "Classe A,0.0,0.1,0.2\n"
                 "Classe A,5.5,0.15,0.25\n")
    wn, X, rot, conc = load_csv(str(p), "classe", "teor")
    assert X.shape == (2, 2)  # colunas classe+teor excluidas do espectro
    np.testing.assert_allclose(conc, [0.0, 5.5])
    np.testing.assert_allclose(wn, [4000.0, 4001.0])
