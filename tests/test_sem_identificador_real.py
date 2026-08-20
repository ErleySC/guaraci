# -*- coding: utf-8 -*-
"""Nenhum arquivo versionado pode conter identificador de amostra real.

POR QUE ESTE TESTE EXISTE. Duas varreduras manuais anteriores tentaram
limpar identificadores de amostra do repositorio publico julgando **codigo
a codigo**: para cada string encontrada, decidiam se aquela amostra existia
ou nao no acervo. As duas falharam pelo mesmo motivo -- o julgamento so'
alcanca os codigos que quem varre ja' conhece, e ninguem conhece os 652.

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

import re
import subprocess
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parents[1]

#: `COD-DD-MM-AAAA` -- especie + data de coleta, o formato de `mae_id` do
#: acervo de origem. `(19|20)` cobre qualquer ano plausivel de aquisicao.
#: `\b` no fim NAO serve: o identificador aparece colado ao sufixo de
#: replicata (`AND-10-06-2099_T1.dx`), e `_` e' caractere de palavra --
#: entre `0` e `_` nao ha fronteira, entao `\b` perdia justamente o caso
#: real que motivou este teste. Guardas explicitas no lugar.
_PADRAO = re.compile(
    r"(?<![A-Za-z0-9])[A-Z]{2,5}-\d{2}-\d{2}-(?:19|20)\d{2}(?!\d)"
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
        if caminho.suffix.lower() in _BINARIO or not caminho.is_file():
            continue
        try:
            texto = caminho.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binario nao declarado ou ilegivel: fora de escopo
        # O proprio teste cita o padrao na docstring e no regex.
        if caminho.name == Path(__file__).name:
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


def test_padrao_reconhece_o_formato_que_pretende_barrar() -> None:
    """Contra-prova: sem isto, um regex quebrado passaria como 'limpo'."""
    assert _PADRAO.search("dados/ACA-04-11-2020_T1.dx")
    assert _PADRAO.search("CAP-04-11-2020-A1.03")
    assert _PADRAO.search("titulo BCB-03-03-2020_AD-S-20_T1")
    assert _PADRAO.search("AND-01-01-2022-S5.00")


def test_sentinela_2099_e_a_unica_excecao() -> None:
    assert _PADRAO.search("CAP-04-11-2099-A1.03").group(0).endswith("2099")
    # Ano vizinho nao e' excecao -- a folga tem que ser exatamente uma.
    assert not _PADRAO.search("CAP-04-11-2098").group(0).endswith("2099")


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
