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
    assert sanitizar_dx.conferir(saida)[0] == 0


def test_conferir_detecta_arquivo_nao_sanitizado(acervo, tmp_path):
    """Contra-prova: se `conferir` devolve 0 para qualquer coisa, ele nao
    esta' conferindo nada."""
    assert sanitizar_dx.conferir(acervo)[0] > 0


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

    assert sanitizar_dx.conferir(saida)[0] == 0, (
        "alarme falso: parametro de aquisicao nao e' proveniencia")


# ─────────────────────────────────────────────────────────────────────────
#  Os quatro furos achados na verificacao independente de 2026-08-20.
#
#  Tres eram SILENCIOSOS: o campo sobrevivia e o `--conferir` reportava
#  limpo. Falso negativo aqui vaza nome de pessoa, porque este e' o script
#  que o ACKNOWLEDGMENTS manda rodar antes de publicar espectros brutos.
# ─────────────────────────────────────────────────────────────────────────

SENT_OP = "OPERADOR_XYZZY"
SENT_LOCAL = "LOCAL_XYZZY"


def _dx_furo(corpo: str) -> str:
    return ("##TITLE=AMOSTRA_0001_T1\n"
            "##JCAMP-DX=4.24\n"
            f"{corpo}"
            "##XUNITS=1/CM\n##YUNITS=ABSORBANCE\n"
            "##FIRSTX=100\n##LASTX=102\n##NPOINTS=3\n"
            "##XFACTOR=1\n##YFACTOR=1\n"
            "##XYDATA=(X++(Y..Y))\n100 A1 A2\n##END=\n")


def test_furo_rotulo_sem_espaco_nao_vaza():
    """JCAMP normaliza espaco/hifen no nome do rotulo: `##AUDITTRAIL` e
    `##AUDIT-TRAIL` sao o MESMO campo que `##AUDIT TRAIL`. A versao anterior
    comparava a string crua e deixava as duas variantes passarem inteiras."""
    for rotulo in ("##AUDITTRAIL", "##AUDIT-TRAIL", "##audit trail"):
        texto = _dx_furo(f"{rotulo}=$$ inicio\n( 1, <2099/01/01>, <{SENT_OP}>, <{SENT_LOCAL}>)\n")
        limpo = sanitizar_dx.sanitizar_texto(texto)
        assert SENT_OP not in limpo, rotulo
        assert SENT_LOCAL not in limpo, rotulo


def test_furo_continuacao_sem_parentese_nao_vaza():
    """O pior dos quatro: `_e_continuacao_de_audit` assumia que toda linha de
    continuacao comeca com '('. Sem o parentese, o CABECALHO saia e a linha
    com operador e local FICAVA -- e o --conferir dizia limpo, porque o
    padrao que ele procurava (`##AUDIT`) tinha saido junto."""
    texto = _dx_furo(f"##AUDIT TRAIL=$$ inicio\n 1, 2099/01/01, {SENT_OP}, {SENT_LOCAL}\n")
    limpo = sanitizar_dx.sanitizar_texto(texto)
    assert SENT_OP not in limpo
    assert SENT_LOCAL not in limpo


def test_furo_comments_nao_vaza():
    """Rotulo de prosa livre com valor MULTILINHA.

    A variante de uma linha era a facil, e era a unica que este teste
    exercitava. A multilinha seguia vazando: o rastreio de continuacao so'
    ligava para `##AUDIT TRAIL`, entao o cabecalho saia e a linha seguinte
    -- com operador e local -- ficava. Escrevi o teste que passa, nao o que
    prova. Achado na verificacao independente de 2026-08-20.
    """
    for corpo in (
        f"##COMMENTS=amostra lida por {SENT_OP} no {SENT_LOCAL}\n",
        f"##COMMENTS=descricao\nLida por {SENT_OP} no {SENT_LOCAL}\n",
        f"##SAMPLE DESCRIPTION=lote 3\nColetado por {SENT_OP} em {SENT_LOCAL}\n",
        f"##SOURCE REFERENCE=ref\n{SENT_OP} / {SENT_LOCAL}\n",
    ):
        limpo = sanitizar_dx.sanitizar_texto(_dx_furo(corpo))
        assert SENT_OP not in limpo, corpo
        assert SENT_LOCAL not in limpo, corpo
        assert "##XYDATA" in limpo, f"dados perdidos: {corpo}"


def test_furo_prefixo_cifrao_nao_vaza():
    """`##$OWNER` escapava porque `"##$OWNER".startswith("##OWNER")` e'
    False. A lista ja' trazia `##$OPERATOR` e `##$PATH` escritos a mao --
    prova de que a variante existe e de que enumerar por rotulo nao escala.
    """
    for rotulo in ("##$OWNER", "##$ORIGIN", "##$SOURCE REFERENCE"):
        limpo = sanitizar_dx.sanitizar_texto(
            _dx_furo(f"{rotulo}={SENT_LOCAL}\n"))
        assert SENT_LOCAL not in limpo, rotulo


def test_furo_detector_model_sem_espaco_nao_vaza():
    texto = _dx_furo(f"##$DetectorModel={SENT_OP}\n")
    limpo = sanitizar_dx.sanitizar_texto(texto)
    assert SENT_OP not in limpo


def test_conferir_lista_rotulo_desconhecido_para_inspecao(tmp_path):
    """O conferidor deixou de ser veredito binario. Um campo que nenhuma
    lista de proibicao conhece tem de aparecer para um humano olhar."""
    saida = tmp_path / "saida"
    saida.mkdir()
    (saida / "a.dx").write_text(_dx_furo("##$CampoNovoDoGravador=qualquer coisa\n"),
                                encoding="latin-1")
    problemas, inspecionar = sanitizar_dx.conferir(saida)
    assert problemas == 0, "nao e' um padrao conhecido de proveniencia"
    assert any("CampoNovoDoGravador" in linha for linha in inspecionar), (
        "campo fora da lista de rotulos esperados tem de ser listado; "
        f"listadas: {inspecionar}"
    )


def test_conferir_nao_lista_parametro_de_aquisicao(tmp_path):
    """Contrapeso: alarme que dispara sempre ensina quem le' a ignora-lo.
    Ganho, resolucao e scans descrevem a MEDIDA e nao podem aparecer."""
    saida = tmp_path / "saida"
    saida.mkdir()
    (saida / "a.dx").write_text(
        _dx_furo("##$Detector Gain=1\n##$Resolution=4\n##$Scans=32\n"),
        encoding="latin-1")
    problemas, inspecionar = sanitizar_dx.conferir(saida)
    assert problemas == 0
    assert inspecionar == [], f"falso positivo: {inspecionar}"

def test_anonimizar_limpa_caminho_e_nome_de_arquivo(tmp_path):
    """O CONTEUDO era anonimizado e o CAMINHO nao: a saida reproduzia
    `<identificador_real>/<arquivo>.dx`. E um `.dx` sem `##TITLE` parseavel
    mantinha o nome ORIGINAL, que e' identificador tanto quanto o TITLE.
    `conferir()` nao via nenhum dos dois, porque so' lia conteudo.
    """
    corpo = ("##XUNITS=1/CM\n##FIRSTX=100\n##LASTX=102\n##NPOINTS=3\n"
             "##XFACTOR=1\n##YFACTOR=1\n##XYDATA=(X++(Y..Y))\n100 A1 A2\n##END=\n")
    ident_pasta = _id_sintetico("CAP", "04", "11")
    ent = tmp_path / "in" / ident_pasta
    ent.mkdir(parents=True)
    (ent / "a.dx").write_text(
        f"##TITLE={ident_pasta}-T1\n##JCAMP-DX=4.24\n" + corpo,
        encoding="latin-1")
    sem_titulo = _id_sintetico("AND", "10", "06") + "-T2.dx"
    (ent / sem_titulo).write_text("##JCAMP-DX=4.24\n" + corpo,
                                  encoding="latin-1")

    saida = tmp_path / "out"
    sanitizar_dx.sanitizar_pasta(tmp_path / "in", saida, anonimizar_titulo=True)

    caminhos = [p.relative_to(saida).as_posix()
                for p in saida.rglob("*.dx")]
    assert caminhos, "nada foi escrito"
    for rel in caminhos:
        assert not sanitizar_dx._RE_ID_AMOSTRA.search(rel), rel
    assert not any(p.name == sem_titulo for p in saida.rglob("*.dx")), (
        "arquivo sem TITLE parseavel manteve o nome original")

    problemas, inspecionar = sanitizar_dx.conferir(saida)
    assert problemas == 0
    assert inspecionar == [], inspecionar


def test_conferir_lista_identificador_no_caminho(tmp_path):
    """Contra-prova do teste acima: se um identificador ESTIVER no caminho,
    `conferir()` tem de acusar -- senao o teste anterior passa por vacuidade.
    """
    saida = tmp_path / "out" / _id_sintetico("CAP", "04", "11")
    saida.mkdir(parents=True)
    (saida / "x.dx").write_text(_dx_furo(""), encoding="latin-1")
    _problemas, inspecionar = sanitizar_dx.conferir(tmp_path / "out")
    assert any("identificador de amostra" in linha for linha in inspecionar), (
        f"caminho com identificador nao foi listado: {inspecionar}")


def _id_sintetico(cod: str, dia: str, mes: str) -> str:
    """Monta o identificador em runtime: escreve-lo literal aqui violaria
    `tests/test_sem_identificador_real.py`. Ano sentinela: `_RE_ID_AMOSTRA`
    nao isenta 2099, entao a contra-prova continua valendo."""
    return f"{cod}-{dia}-{mes}-2099"
