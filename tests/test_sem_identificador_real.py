# -*- coding: utf-8 -*-
"""Nenhum arquivo versionado pode conter identificador de amostra real.

POR QUE ESTE TESTE EXISTE. Julgar codigo a codigo se um identificador de
amostra e' real so' alcanca os codigos que quem varre ja' conhece.

A regra que fica nao julga codigo nenhum: **o padrao inteiro e' proibido,
com uma unica excecao declarada.** Identificador de exemplo usa o ano
sentinela 2099, que nao e' data de aquisicao possivel. Qualquer outro ano
reprova, independente de quem o escreveu saber se e' real.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parents[1]

#: `COD-DD-MM-AAAA` -- especie + data de coleta, o formato de `mae_id`
#: reconhecido pelo parser. `(19|20)` cobre qualquer ano plausivel de aquisicao.
#: O separador entre codigo e data e' `-` OU `_`: e' o que `_RE_TITLE`
#: (`src/guaraci/dados_io.py`) aceita, porque as duas formas existem em
#: arquivos reais. A primeira versao deste guarda exigia `-`, entao um titulo
#: inteiro como `ESD_03-01-2099_T1` -- que o parser aceita e converte em
#: `mae_id` -- passava sem alarme. Achado na verificacao independente de
#: 2026-08-19.
#:
#: `\b` no fim tambem NAO serve: o identificador aparece colado ao sufixo
#: de replicata (`ESA-04-01-2099_T1.dx`), e `_` e' caractere de palavra --
#: entre `0` e `_` nao ha fronteira, entao `\b` perdia justamente o caso
#: real que motivou este teste. Guardas explicitas no lugar.
#:
#: IGNORECASE: sem ele o padrao cobria so' a metade MAIUSCULA do que a
#: docstring diz proibir. Um titulo em caixa baixa passava livre -- e o
#: proprio repositorio usa fixtures assim (`and_T1.dx`), enquanto
#: `dados_io.py` chama `.upper()` justamente porque a caixa varia no
#: dado real. Em NTFS, que ignora caixa, renomear para minuscula nem
#: seria visivel. Achado na verificacao independente de 2026-08-20.
#:
#: Extraido para `scripts/privacidade_amostras.py` (Passo 169) para que o
#: gerador do vault Obsidian reaproveite a MESMA regra em vez de duplica-la.
_SCRIPTS = str(_RAIZ / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from privacidade_amostras import (  # noqa: E402
    ANO_SENTINELA as _ANO_SENTINELA,
    PADRAO_IDENTIFICADOR as _PADRAO,
    CATEGORIAS_REAIS as _CATEGORIAS_REAIS,
    USUARIO_SENTINELA as _USUARIO_SENTINELA,
    caminhos_absolutos_em_texto,
    classificar_ocorrencia as _classificar,
    usuario_do_caminho as _usuario_do_caminho,
)

#: Extensoes que nao sao texto -- lidas em mode binario dariam ruido.
#: `.sp`/`.spc` adicionados (Grupo 1, 2026-09-12): fixtures binarias reais
#: de instrumento em validacao/fixtures_locais/sp/ e validacao/fixtures_locais/spc/ (ver
#: PROVENANCIA.md em cada pasta) -- nao sao UTF-8, mesma categoria de
#: `.joblib`/`.pdf` acima.
#: `.ch` (Agilent ChemStation, canal de detector UV/FID binario --
#: validacao/fixtures_locais/hplc_agilent/, Grupo 1, 2026-09-12) adicionado junto.
_BINARIO = {".png", ".ico", ".jpg", ".jpeg", ".gif", ".pdf", ".joblib",
            ".xlsx", ".docx", ".pptx", ".woff", ".woff2", ".zip", ".gz",
            ".sp", ".spc", ".ch"}

#: Convencao de nomes do formato Bruker NAO usa extensao -- `fid`/`1r`/`1i`
#: sao dados binarios reais (int32/float64), diferente de `acqus`/`procs`/
#: `pulseprogram`/`method` (que sao texto JCAMP-DX-like e DEVEM continuar
#: sendo varridos normalmente). So' os basenames exatos abaixo, nao a pasta
#: inteira -- adicionar so' `validacao/fixtures_locais/rmn_bruker/` a `_BINARIO`
#: esconderia um identificador real que por acaso aparecesse num arquivo de
#: texto novo dentro dela no futuro. Ver validacao/fixtures_locais/rmn_bruker/
#: PROVENANCIA.md (Grupo 1, 2026-09-12).
_BINARIO_NOMES_SEM_EXTENSAO = {"fid", "ser", "1r", "1i", "2rr", "2ii"}


def _arquivos_versionados() -> list[Path]:
    """Lista o que o Git rastreia. Arquivo ignorado (dado local do usuario)
    esta' fora de escopo: o teste protege o que e' PUBLICADO."""
    saida = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=_RAIZ, capture_output=True, check=True,
    ).stdout.decode("utf-8", "replace")
    return [_RAIZ / n for n in saida.split("\0") if n]


def _ocorrencias() -> list[tuple[str, int, str]]:
    achados: list[tuple[str, int, str]] = []
    for caminho in _arquivos_versionados():
        rel = str(caminho.relative_to(_RAIZ)).replace("\\", "/")
        # O NOME do arquivo tambem e' identificador -- varrer so' o conteudo
        # deixaria passar um arquivo chamado `ESC-03-01-2099_T1.md`.
        for m in _PADRAO.finditer(rel):
            if not m.group(0).endswith(_ANO_SENTINELA):
                achados.append((rel, 0, m.group(0)))
        if (caminho.suffix.lower() in _BINARIO
                or caminho.name in _BINARIO_NOMES_SEM_EXTENSAO
                or not caminho.is_file()):
            continue
        try:
            texto = caminho.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Antes isto era `continue` silencioso -- um arquivo nao-UTF-8
            # saia da varredura sem deixar rastro. Agora vira achado: o
            # teste nao pode declarar limpo o que nao conseguiu ler.
            achados.append((rel, 0, "<ILEGIVEL: nao e' UTF-8 nem binario declarado>"))
            continue
        except OSError:
            achados.append((rel, 0, "<ILEGIVEL: erro de leitura>"))
            continue
        for n, linha in enumerate(texto.splitlines(), start=1):
            for m in _PADRAO.finditer(linha):
                if m.group(0).endswith(_ANO_SENTINELA):
                    continue
                achados.append((
                    str(caminho.relative_to(_RAIZ)).replace("\\", "/"),
                    n, m.group(0),
                ))
    return achados


def test_nenhum_identificador_real_em_arquivo_versionado() -> None:
    achados = _ocorrencias()
    if achados:
        linhas = "\n".join(f"  {a}:{n}  ->  {ident}" for a, n, ident in achados)
        pytest.fail(
            f"{len(achados)} identificador(es) de amostra em arquivo "
            f"versionado:\n{linhas}\n\n"
            f"Identificador de exemplo deve usar o ano {_ANO_SENTINELA}. "
            "Nao julgue caso a caso se o codigo e' real -- o padrao inteiro "
            "e' proibido justamente porque esse julgamento ja' falhou duas "
            "vezes (ver docstring)."
        )


def _id(especie: str, dia: str, mes: str, ano: str, sep: str = "-") -> str:
    """Monta um identificador em tempo de execucao, em vez de escrever um
    literal aqui.

    Um literal de ano plausivel neste arquivo seria barrado pela propria
    varredura, que tambem vale aqui (sem auto-isencao).
    """
    return f"{especie}{sep}{dia}-{mes}-{ano}"


def test_padrao_reconhece_o_formato_que_pretende_barrar() -> None:
    """Contra-prova: sem isto, um regex quebrado passaria como 'limpo'."""
    assert _PADRAO.search(f"dados/{_id('ESC', '03', '01', '2020')}_T1.dx")
    assert _PADRAO.search(f"{_id('ESB', '03', '01', '2020')}-A1.03")
    assert _PADRAO.search(f"titulo {_id('ESF', '06', '01', '2020')}_AD-S-20_T1")
    assert _PADRAO.search(f"{_id('ESA', '01', '01', '2022')}-S5.00")


def test_padrao_cobre_qualquer_caixa() -> None:
    """O guarda era case-sensitive e cobria so' a metade maiuscula do padrao
    que a docstring declara proibir. Estas quatro variacoes passavam livres,
    e o repositorio usa fixtures em caixa baixa."""
    for variacao in (
        f"dados/{_id('and', '03', '01', '2020', '_')}_T1.dx",
        f"{_id('cap', '03', '01', '2020')}-A1.03",
        _id("Cap", "03", "01", "2020"),
        f"{_id('go', '03', '01', '2020', '_')}_T1",
    ):
        assert _PADRAO.search(variacao), variacao


def test_padrao_cobre_o_separador_que_o_parser_aceita() -> None:
    """`_RE_TITLE` (dados_io.py) aceita `-` OU `_` depois do codigo. Um
    guarda que so' conhecesse `-` deixaria passar metade dos titulos reais
    -- foi assim que a primeira versao deste teste furou."""
    for sep in ("-", "_"):
        titulo = f"{_id('ESD', '03', '01', '2020', sep)}_T1"
        assert _PADRAO.search(titulo), titulo


def test_sentinela_2099_e_a_unica_excecao() -> None:
    achado = _PADRAO.search(f"{_id('ESB', '03', '01', '2099')}-A1.03")
    assert achado.group(0).endswith(_ANO_SENTINELA)
    # Ano vizinho nao e' excecao -- a folga tem que ser exatamente uma.
    vizinho = _PADRAO.search(_id("ESB", "03", "01", "2098"))
    assert not vizinho.group(0).endswith(_ANO_SENTINELA)


# ─────────────────────────────────────────────────────────────────────────
#  Caminho absoluto de maquina em arquivo versionado
#
#  Um caminho absoluto de maquina (nome de usuario + pasta de dados) num
#  arquivo versionado expoe quem gerou o arquivo; a varredura abaixo barra.
# ─────────────────────────────────────────────────────────────────────────


def _caminhos_absolutos_versionados() -> list[tuple[str, int, str]]:
    achados: list[tuple[str, int, str]] = []
    for caminho in _arquivos_versionados():
        rel = str(caminho.relative_to(_RAIZ)).replace("\\", "/")
        for cam in caminhos_absolutos_em_texto(rel):
            if _usuario_do_caminho(cam) != _USUARIO_SENTINELA:
                achados.append((rel, 0, cam))
        if (caminho.suffix.lower() in _BINARIO
                or caminho.name in _BINARIO_NOMES_SEM_EXTENSAO
                or not caminho.is_file()):
            continue
        try:
            texto = caminho.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # ja' reportado como <ILEGIVEL> pela varredura acima
        for n, linha in enumerate(texto.splitlines(), start=1):
            for cam in caminhos_absolutos_em_texto(linha):
                if _usuario_do_caminho(cam) != _USUARIO_SENTINELA:
                    achados.append((rel, n, cam))
    return achados


def test_nenhum_caminho_absoluto_de_maquina_em_arquivo_versionado() -> None:
    achados = _caminhos_absolutos_versionados()
    if achados:
        linhas = "\n".join(f"  {a}:{n}  ->  {cam}" for a, n, cam in achados)
        pytest.fail(
            f"{len(achados)} caminho(s) absoluto(s) de maquina em arquivo "
            f"versionado:\n{linhas}\n\n"
            f"Caminho de exemplo deve usar o usuario '{_USUARIO_SENTINELA}'; "
            "caminho real vira descricao relativa (ex.: 'a pasta_dados do "
            "config.yaml local')."
        )


def _caminho(sistema: str, usuario: str) -> str:
    """Monta o caminho em tempo de execucao -- um literal aqui seria pego
    pela propria varredura acima (mesma armadilha documentada em `_id`)."""
    if sistema == "win":
        return "C:" + "\\" + "Users" + "\\" + usuario + "\\projeto\\x.md"
    return "/" + "home" + "/" + usuario + "/projeto/x.md"


def test_usuario_sentinela_e_a_unica_excecao_de_caminho() -> None:
    """Contra-prova: o regex acusa caminho real, e a folga e' exatamente o
    usuario sentinela -- nem prefixo, nem vizinho."""
    for sistema in ("win", "unix"):
        (sentinela,) = caminhos_absolutos_em_texto(_caminho(sistema, _USUARIO_SENTINELA))
        assert _usuario_do_caminho(sentinela) == _USUARIO_SENTINELA
        for outro in ("fulano", _USUARIO_SENTINELA + "2", "x" + _USUARIO_SENTINELA):
            (cam,) = caminhos_absolutos_em_texto(_caminho(sistema, outro))
            assert _usuario_do_caminho(cam) != _USUARIO_SENTINELA, cam


def test_caminho_com_separador_repetido_ou_escapado_e_detectado() -> None:
    """Achado de 2026-10-04: um caminho de usuario com barras dobradas (string escapada num
    printf de documento versionado) passava pela guarda, que so' aceitava uma barra entre os
    segmentos. Montado em tempo de execucao (mesma armadilha de `_caminho`)."""
    b = "\\"
    for sep in (b + b, "//", b + "/"):
        for usuario in ("fulano", _USUARIO_SENTINELA):
            (cam,) = caminhos_absolutos_em_texto("C:" + sep + "Users" + sep + usuario + sep + "x.md")
            assert _usuario_do_caminho(cam) == usuario, cam
    (cam,) = caminhos_absolutos_em_texto("//" + "home" + "//" + "fulano//x")
    assert _usuario_do_caminho(cam) == "fulano"


def test_classificacao_por_categoria_reprova_so_o_real() -> None:
    """Uma entrada sintetica por categoria; so' as reais reprovam (sentinelas passam)."""
    b = "\\"
    casos = {
        "C:" + b + "Users" + b + _USUARIO_SENTINELA: "placeholder",
        "C:" + b + "Users" + b + "fulano": "caminho real",
        "/" + "home" + "/" + "fulano": "caminho real",
        "C" + "--" + "Users-" + _USUARIO_SENTINELA + "-proj": "caminho real (forma achatada)",
        "ABC-01-02-" + "2019": "codigo de amostra real",
        "ABC-01-02-" + _ANO_SENTINELA: "codigo de exemplo (ano sentinela)",
    }
    for achado, esperado in casos.items():
        assert _classificar(achado) == esperado, achado
    assert {c for c in casos.values() if c in _CATEGORIAS_REAIS} == {
        "caminho real", "caminho real (forma achatada)", "codigo de amostra real"}


def test_forma_achatada_com_hifens_e_detectada_sem_excecao() -> None:
    """Achado de 2026-10-02: saidas versionadas traziam o caminho do
    projeto na forma que ferramentas usam para nomear pasta temporaria
    (letra, '--Users-', usuario, '-One' 'Drive-...'), invisivel para o padrao
    com barras. Montado em tempo de execucao (mesma armadilha de `_caminho`)."""
    for usuario in ("fulano", _USUARIO_SENTINELA):
        texto = "tmp/claude/C" + "--" + "Users" + "-" + usuario + "-One" + "Drive-Doc/x.txt"
        assert caminhos_absolutos_em_texto(texto), texto
    assert not caminhos_absolutos_em_texto("opcao --Users-foo sem letra de unidade")


def test_padrao_nao_barra_texto_legitimo() -> None:
    """Falso positivo tambem e' defeito: alarme que dispara sempre ensina
    quem le a ignora-lo (licao registrada no CLAUDE.md, 2026-08-18)."""
    for legitimo in (
        "AUDITORIA_MESTRE_2026-08-17.md",
        "versao 31.9.0 de 2026-08-19",
        "ISO-8601: 2026-08-19",
        "RMSEP 0,144 %m/m",
    ):
        assert not _PADRAO.search(legitimo), legitimo


# ─────────────────────────────────────────────────────────────────────────
#  Varredura de VALOR GERADO
#
#  A varredura textual acima pega o identificador ESCRITO. Nao pega o
#  identificador PRODUZIDO -- e o guarda ja' foi furado tres vezes por essa
#  fronteira: auto-isencao, separador `_`, e (verificacao independente de
#  2026-08-20) identificadores montados por f-string
#  (`f"{cod}-0{i//2+1}-11-2020_T{i%2+1}"`) ou partidos entre campos
#  (`"cod": "ESB", "data": "04-11-2020"`). Casamento de string sobre
#  codigo-fonte nao ve o que o codigo produz.
#
#  Este bloco fecha a fronteira pelo outro lado: executa a suite de testes
#  que cria arquivos com titulo, e varre os NOMES e TITULOS efetivamente
#  gerados.
# ─────────────────────────────────────────────────────────────────────────


def _titulos_gerados_pelos_testes() -> list[str]:
    """Roda os modulos de teste que escrevem `.dx` e devolve todo titulo e
    nome de arquivo que eles produziram de fato."""
    import tempfile

    gerados: list[str] = []
    alvo = _RAIZ / "tests"
    # A pasta ja' nasce curta: um `--basetemp` longo seria trocado pelo conftest (A3-05, tests/_basetemp_curto.py) e os
    # artefatos iriam para outra pasta, deixando esta varredura vazia.
    from tests._basetemp_curto import raiz_curta
    base = None
    if os.name == "nt":
        base = raiz_curta()
        base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as tmp:
        r = subprocess.run(
            [os.sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             str(alvo / "test_dados_io_jcamp.py"),
             str(alvo / "test_sanitizacao_metadados.py"),
             str(alvo / "test_sanitizar_dx.py"),
             "--basetemp", tmp],
            cwd=_RAIZ, capture_output=True,
        )
        assert r.returncode == 0, (
            "os testes que geram `.dx` falharam; sem eles esta varredura nao "
            f"prova nada:\n{r.stdout.decode('utf-8', 'replace')[-2000:]}"
        )
        for raiz, _dirs, arquivos in os.walk(tmp):
            for nome in arquivos:
                gerados.append(nome)
                if nome.lower().endswith((".dx", ".jdx")):
                    try:
                        texto = Path(raiz, nome).read_text(
                            encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                    for linha in texto.splitlines():
                        if linha.upper().lstrip("#").strip().startswith("TITLE"):
                            gerados.append(linha)
    return gerados


@pytest.mark.slow
def test_nenhum_identificador_real_nos_valores_gerados() -> None:
    gerados = _titulos_gerados_pelos_testes()
    assert gerados, "nenhum artefato gerado -- a varredura seria vacua"
    achados = [
        m.group(0)
        for valor in gerados
        for m in _PADRAO.finditer(valor)
        if not m.group(0).endswith(_ANO_SENTINELA)
    ]
    if achados:
        pytest.fail(
            f"{len(achados)} identificador(es) de ano real PRODUZIDO(S) em "
            f"tempo de execucao: {sorted(set(achados))}\n"
            "Escrever o identificador por partes ou por f-string nao o torna "
            "sintetico -- use o ano " + _ANO_SENTINELA + "."
        )
