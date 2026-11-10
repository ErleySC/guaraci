"""Leitura de CSV tolerante ao que o Excel e a locale pt-BR realmente geram.

Achados R2-03/R2-04 da revisao cega de 2026-09-26: (i) o "CSV UTF-8" do Excel
traz BOM, que virava um caractere invisivel no nome da primeira coluna; (ii) com
separador `;` ou TAB e virgula decimal, o detector do pandas (`sep=None`) via
virgulas em toda parte e escolhia `,`, partindo os numeros ao meio.

Aqui o separador e' decidido pela CONSISTENCIA de contagem por linha (TAB, depois
`;`, depois `,`): um separador de verdade aparece o mesmo numero de vezes em toda
linha; a virgula decimal, nao. Sem candidato consistente, cai no sniffer do pandas.
"""
from __future__ import annotations

import io
import os
from typing import Any, Optional

import pandas as pd

_CANDIDATOS = ("\t", ";", ",")


class CsvDecimalAmbiguoError(ValueError):
    """Linhas com mais campos que o cabecalho: virgula decimal igual ao separador."""


def _texto(fonte: Any) -> str:
    """Conteudo da fonte (caminho, bytes ou buffer/UploadedFile) como texto;
    remove o BOM; UTF-8 e, se falhar, latin-1."""
    if isinstance(fonte, bytes):
        dados: Any = fonte
    elif isinstance(fonte, (str, os.PathLike)):
        with open(fonte, "rb") as f:
            dados = f.read()
    else:
        if hasattr(fonte, "seek"):
            fonte.seek(0)
        dados = fonte.read()
        if hasattr(fonte, "seek"):
            fonte.seek(0)
    if isinstance(dados, str):
        return dados.lstrip("﻿")
    try:
        return dados.decode("utf-8-sig")
    except UnicodeDecodeError:
        return dados.decode("latin-1")


def detectar_separador(texto: str) -> Optional[str]:
    linhas = [ln for ln in texto.splitlines() if ln.strip()][:25]
    if len(linhas) < 2:
        return None
    for sep in _CANDIDATOS:
        cont = [ln.count(sep) for ln in linhas]
        if cont[0] >= 1 and len(set(cont)) == 1:
            return sep
    return None


def ler_csv(fonte: Any, **kw: Any) -> pd.DataFrame:
    """`pd.read_csv` com BOM removido e separador decidido por consistencia.
    Levanta os mesmos erros do pandas (`EmptyDataError`, `ParserError`,
    `csv.Error` do sniffer) para os chamadores tratarem como antes."""
    texto = _texto(fonte)
    sep = detectar_separador(texto)
    df = pd.read_csv(io.StringIO(texto), sep=sep if sep else None,
                     engine="python", **kw)
    _recusar_indice_inesperado(df, kw)
    return df


def _recusar_indice_inesperado(df: pd.DataFrame, kw: dict) -> None:
    """Linhas com MAIS campos que o cabecalho: o pandas joga o excedente para o
    indice, em silencio. O caso real e' o "CSV separado por virgulas" do Excel
    em pt-BR, em que a virgula e' ao mesmo tempo separador de colunas e decimal
    ("0,00123" vira dois campos). Sem esta checagem o arquivo era lido como
    lixo e a execucao morria depois, com uma explicacao errada sobre classes."""
    ic = kw.get("index_col")
    if ic is not None and ic is not False:      # `0 in (None, False)` seria True
        return                      # o chamador pediu um indice: nada a recusar
    if isinstance(df.index, pd.RangeIndex):
        return
    n_cab = len(df.columns)
    n_linha = n_cab + df.index.nlevels
    raise CsvDecimalAmbiguoError(
        f"As linhas de dados tem {n_linha} campos e o cabecalho tem {n_cab}. "
        f"Isso acontece quando a virgula e' ao mesmo tempo o separador de "
        f"colunas e o separador decimal (\"0,00123\" vira dois campos). "
        f"Salve o CSV com ponto e virgula (;) como separador de colunas "
        f"(no Excel: \"CSV (separado por ponto e virgula)\"), ou com ponto "
        f"como separador decimal.")
