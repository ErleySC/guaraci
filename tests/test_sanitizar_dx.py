# -*- coding: utf-8 -*-
"""A ferramenta de sanitizacao de `.dx` nao pode: deixar passar um campo
sensivel, quebrar o agrupamento de replicas, ou tocar nos originais.

O ultimo item e' o que mais importa: um script de sanitizacao que apaga o
original em caso de bug destroi a unica copia da evidencia.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sanitizar_dx  # noqa: E402

OPERADOR = "FULANO_DE_TAL"
LOCAL = "INSTITUICAO_QUALQUER"
DETECTOR = "DETECTOR-123456"


def _dx(titulo: str) -> str:
    return "\n".join([
        f"##TITLE={titulo}",
        "##JCAMP-DX=4.24",
        "##$Spectrometer model=MB3600",
        f"##$Detector model={DETECTOR}",
        "##AUDIT TRAIL= $$ (NUMBER, WHEN, WHO, WHERE, WHAT)",
        f"( 1, <2021/01/06 14:15:37 -03>,  <{OPERADOR}>,  <{LOCAL}>,  <x>)",
        f"( 2, <2021/01/06 14:15:37 -03>,  <{OPERADOR}>,  <{LOCAL}>,  <y>)",
        "##XUNITS=1/CM",
        "##FIRSTX=4000",
        "##LASTX=10000",
        "##NPOINTS=4",
        "##XYDATA=(X++(Y..Y))",
        "4000A1B2C3D4",
        "##END=",
    ])


@pytest.fixture
def acervo(tmp_path):
    entrada = tmp_path / "originais"
    (entrada / "ClasseA").mkdir(parents=True)
    for ponto in (1, 2):
        for trip in (1, 2, 3):
            titulo = f"ACA-0{ponto}-01-2099-T{trip}"
            (entrada / "ClasseA" / f"{titulo}.dx").write_text(
                _dx(titulo), encoding="latin-1")
    return entrada


def test_remove_audit_trail_inteiro_inclusive_as_continuacoes(acervo, tmp_path):
    """`AUDIT TRAIL` e' um bloco MULTILINHA: remover so' a linha do cabecalho
    deixaria as entradas `( 1, <...>, <operador>, <local>, ...)` no arquivo --
    que e' exatamente onde o nome da pessoa esta'."""
    saida = tmp_path / "limpos"
    sanitizar_dx.sanitizar_pasta(acervo, saida)
    for arq in saida.rglob("*.dx"):
        texto = arq.read_text(encoding="latin-1")
        assert OPERADOR not in texto
        assert LOCAL not in texto
        assert DETECTOR not in texto
        assert "AUDIT" not in texto.upper()
        # o espectro e o TITLE sobrevivem -- sem eles o arquivo e' inutil
        assert "##XYDATA" in texto and "4000A1B2C3D4" in texto
        assert "##TITLE=" in texto


def test_conferir_nao_encontra_nada_apos_sanitizar(acervo, tmp_path):
    saida = tmp_path / "limpos"
    sanitizar_dx.sanitizar_pasta(acervo, saida)
    assert sanitizar_dx.conferir(saida) == 0


def test_conferir_detecta_arquivo_nao_sanitizado(acervo, tmp_path):
    """Contra-prova: se `conferir` devolve 0 para qualquer coisa, ele nao
    esta' conferindo nada."""
    assert sanitizar_dx.conferir(acervo) > 0


def test_nunca_sobrescreve_os_originais(acervo, tmp_path):
    antes = {p: p.read_bytes() for p in acervo.rglob("*.dx")}
    sanitizar_dx.sanitizar_pasta(acervo, tmp_path / "limpos")
    for p, conteudo in antes.items():
        assert p.read_bytes() == conteudo, f"{p.name} foi modificado"


def test_recusa_escrever_dentro_da_entrada(acervo):
    """Saida dentro da entrada acabaria misturando limpos e sujos na mesma
    arvore -- e a proxima varredura nao saberia dizer quais sao quais."""
    with pytest.raises(SystemExit, match="saida"):
        sanitizar_dx.sanitizar_pasta(acervo, acervo / "sub")
    with pytest.raises(SystemExit, match="saida"):
        sanitizar_dx.sanitizar_pasta(acervo, acervo)


def test_anonimizar_titulo_preserva_o_agrupamento_de_replicas(acervo, tmp_path):
    """Se a anonimizacao quebrar o vinculo entre T1/T2/T3 do mesmo ponto, a
    validacao group-aware vai embora junto -- e e' o diferencial do projeto."""
    saida = tmp_path / "anon"
    res = sanitizar_dx.sanitizar_pasta(acervo, saida, anonimizar_titulo=True)
    assert res["grupos"] == 2, "2 pontos fisicos deveriam virar 2 grupos"

    titulos = []
    for arq in sorted(saida.rglob("*.dx")):
        texto = arq.read_text(encoding="latin-1")
        titulos.append(sanitizar_dx._ler_titulo(texto))
    assert all(t.startswith("AMOSTRA_") for t in titulos)
    # 3 replicas por grupo, e nenhum identificador original sobrevive
    grupos = {t.rsplit("_T", 1)[0] for t in titulos}
    assert len(grupos) == 2
    assert not any("ACA" in t for t in titulos)
    for g in grupos:
        assert sum(1 for t in titulos if t.startswith(g + "_T")) == 3


def test_mapa_de_titulos_e_escrito_e_avisado(acervo, tmp_path, capsys):
    saida = tmp_path / "anon"
    sanitizar_dx.sanitizar_pasta(acervo, saida, anonimizar_titulo=True)
    mapa = saida / "mapa_titulos.csv"
    assert mapa.is_file()
    conteudo = mapa.read_text(encoding="utf-8")
    assert "titulo_original;titulo_anonimo" in conteudo
    assert "ACA-01-01-2099-T1" in conteudo      # permite reverter localmente
    # e o script tem que AVISAR que esse arquivo nao pode ser publicado
    assert "NUNCA publique" in capsys.readouterr().out


def test_conferir_nao_alarma_em_parametro_de_aquisicao(tmp_path):
    """`##$Detector Gain` e' ganho do detector -- parametro da MEDIDA, nao
    identificacao. Tem que sobreviver a sanitizacao E nao disparar o alarme.

    A primeira versao do `--conferir` casava qualquer `##$Detector*` e
    disparava em 100% dos arquivos reais (achado rodando contra `.dx` de
    verdade). Um alarme que dispara sempre ensina quem o le' a ignora-lo.
    """
    entrada = tmp_path / "orig"; entrada.mkdir()
    (entrada / "a.dx").write_text("\n".join([
        "##TITLE=ACA-01-01-2099-T1",
        "##$Spectrometer model=MB3600",
        "##$Detector model=114690-655370",     # numero de serie -> sai
        "##$Detector Gain=1",                  # parametro da medida -> fica
        "##$Resolution=4",
        "##$Scans=16",
        "##AUDIT TRAIL= $$ (NUMBER, WHEN, WHO, WHERE, WHAT)",
        "( 1, <2021/01/06 14:15:37 -03>,  <FULANO>,  <LUGAR>,  "
        "<Measured with Instrument ID: '1389360-1'>)",
        "##XYDATA=(X++(Y..Y))",
        "4000A1B2",
        "##END=",
    ]), encoding="latin-1")

    saida = tmp_path / "limpo"
    sanitizar_dx.sanitizar_pasta(entrada, saida)
    texto = (saida / "a.dx").read_text(encoding="latin-1")

    # o que descreve a MEDIDA fica
    assert "##$Detector Gain=1" in texto
    assert "##$Resolution=4" in texto and "##$Scans=16" in texto
    # o que IDENTIFICA sai
    assert "114690-655370" not in texto
    assert "MB3600" not in texto
    assert "FULANO" not in texto and "LUGAR" not in texto
    assert "Instrument ID" not in texto

    assert sanitizar_dx.conferir(saida) == 0, (
        "alarme falso: parametro de aquisicao nao e' proveniencia")
