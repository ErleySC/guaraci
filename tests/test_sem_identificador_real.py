# -*- coding: utf-8 -*-
"""Nenhum arquivo versionado pode conter identificador de amostra real.

POR QUE ESTE TESTE EXISTE. Duas varreduras manuais anteriores tentaram
limpar identificadores de amostra do repositorio publico julgando **codigo
a codigo**: para cada string encontrada, decidiam se aquela amostra existia
ou nao no acervo. As duas falharam pelo mesmo motivo -- o julgamento so'
alcanca os codigos que quem varre ja' conhece, e ninguem conhece o
acervo inteiro.

A varredura de 2026-08-17 converteu 2 codigos e deixou 15 linhas passarem,
incluindo um identificador **real** no texto de ajuda do CLI, entregue ao
usuario final. A varredura tambem publicava, ao justificar-se, a janela de
aquisicao do acervo -- o proprio dado que tentava proteger.

A regra que fica nao julga codigo nenhum: **o padrao inteiro e' proibido,
com uma unica excecao declarada.** Identificador de exemplo usa o ano
sentinela 2099, que nao existe em acervo nenhum. Qualquer outro ano
reprova, independente de quem o escreveu saber se e' real.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parents[1]

#: `COD-DD-MM-AAAA` -- especie + data de coleta, o formato de `mae_id` do
#: acervo de origem. `(19|20)` cobre qualquer ano plausivel de aquisicao.
#: O separador entre codigo e data e' `-` OU `_`: e' o que `_RE_TITLE`
#: (`src/guaraci/dados_io.py`) aceita, porque as duas formas existem no
#: dado real. A primeira versao deste guarda exigia `-`, entao um titulo
#: inteiro como `GOI_04-11-2099_T1` -- que o parser aceita e converte em
#: `mae_id` -- passava sem alarme. Achado na verificacao independente de
#: 2026-08-19.
#:
#: `\b` no fim tambem NAO serve: o identificador aparece colado ao sufixo
#: de replicata (`AND-10-06-2099_T1.dx`), e `_` e' caractere de palavra --
#: entre `0` e `_` nao ha fronteira, entao `\b` perdia justamente o caso
#: real que motivou este teste. Guardas explicitas no lugar.
#:
#: IGNORECASE: sem ele o padrao cobria so' a metade MAIUSCULA do que a
#: docstring diz proibir. Um titulo em caixa baixa passava livre -- e o
#: proprio repositorio usa fixtures assim (`and_T1.dx`), enquanto
#: `dados_io.py` chama `.upper()` justamente porque a caixa varia no
#: dado real. Em NTFS, que ignora caixa, renomear para minuscula nem
#: seria visivel. Achado na verificacao independente de 2026-08-20.
_PADRAO = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]{2,5}[-_]\d{2}-\d{2}-(?:19|20)\d{2}(?!\d)",
    re.IGNORECASE,
)

#: Unica excecao. Ano que nao pode corresponder a leitura nenhuma, entao um
#: identificador com ele e' inequivocamente inventado para exemplo/fixture.
_ANO_SENTINELA = "2099"

#: Extensoes que nao sao texto -- lidas em modo binario dariam ruido.
_BINARIO = {".png", ".ico", ".jpg", ".jpeg", ".gif", ".pdf", ".joblib",
            ".xlsx", ".docx", ".pptx", ".woff", ".woff2", ".zip", ".gz"}


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
        # deixaria passar um arquivo chamado `ACA-04-11-2099_T1.md`.
        for m in _PADRAO.finditer(rel):
            if not m.group(0).endswith(_ANO_SENTINELA):
                achados.append((rel, 0, m.group(0)))
        if caminho.suffix.lower() in _BINARIO or not caminho.is_file():
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

    A primeira versao deste arquivo continha 5 identificadores literais de
    ano real -- dois deles confirmados como amostra do acervo -- e se
    auto-isentava da varredura (`if caminho.name == __file__: continue`)
    para poder conte-los. Isso e' a mesma falha que o teste existe para
    barrar, cometida pelo proprio teste. Achado na verificacao independente
    de 2026-08-19; a auto-isencao foi removida e as contra-provas passaram
    a montar as strings, que e' o que faz a varredura valer tambem aqui.
    """
    return f"{especie}{sep}{dia}-{mes}-{ano}"


def test_padrao_reconhece_o_formato_que_pretende_barrar() -> None:
    """Contra-prova: sem isto, um regex quebrado passaria como 'limpo'."""
    assert _PADRAO.search(f"dados/{_id('ACA', '04', '11', '2020')}_T1.dx")
    assert _PADRAO.search(f"{_id('CAP', '04', '11', '2020')}-A1.03")
    assert _PADRAO.search(f"titulo {_id('BCB', '03', '03', '2020')}_AD-S-20_T1")
    assert _PADRAO.search(f"{_id('AND', '01', '01', '2022')}-S5.00")


def test_padrao_cobre_qualquer_caixa() -> None:
    """O guarda era case-sensitive e cobria so' a metade maiuscula do padrao
    que a docstring declara proibir. Estas quatro variacoes passavam livres,
    e o repositorio usa fixtures em caixa baixa."""
    for variacao in (
        f"dados/{_id('and', '04', '11', '2020', '_')}_T1.dx",
        f"{_id('cap', '04', '11', '2020')}-A1.03",
        _id("Cap", "04", "11", "2020"),
        f"{_id('go', '04', '11', '2020', '_')}_T1",
    ):
        assert _PADRAO.search(variacao), variacao


def test_padrao_cobre_o_separador_que_o_parser_aceita() -> None:
    """`_RE_TITLE` (dados_io.py) aceita `-` OU `_` depois do codigo. Um
    guarda que so' conhecesse `-` deixaria passar metade dos titulos reais
    -- foi assim que a primeira versao deste teste furou."""
    for sep in ("-", "_"):
        titulo = f"{_id('GOI', '04', '11', '2020', sep)}_T1"
        assert _PADRAO.search(titulo), titulo


def test_sentinela_2099_e_a_unica_excecao() -> None:
    achado = _PADRAO.search(f"{_id('CAP', '04', '11', '2099')}-A1.03")
    assert achado.group(0).endswith(_ANO_SENTINELA)
    # Ano vizinho nao e' excecao -- a folga tem que ser exatamente uma.
    vizinho = _PADRAO.search(_id("CAP", "04", "11", "2098"))
    assert not vizinho.group(0).endswith(_ANO_SENTINELA)


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
#  (`"cod": "CAP", "data": "04-11-2020"`). Casamento de string sobre
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
    with tempfile.TemporaryDirectory() as tmp:
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
