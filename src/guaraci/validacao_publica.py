"""validacao_publica.py — Leitura da tabela consolidada de validacoes
publicas (`docs/VALIDACAO_PUBLICA.md` §1).

Extraido de `scripts/gerar_vault_obsidian.py` (`parse_tabela_consolidada`)
para virar fonte UNICA: o gerador do vault e o painel de status da aba
Projeto do app web precisam da MESMA leitura -- duas copias divergiriam na
primeira mudanca de formato da tabela.

Parsing PURO (texto -> dados), no espirito de `resumo_parse.py`: `_ler`/I-O
fica no chamador, e a versao com caminho (`load_consolidated_table`) e' um
utilitario fino em cima da funcao pura.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

__all__ = [
    "CABECALHO_TABELA",
    "parse_consolidated_table",
    "load_consolidated_table",
    "find_rows_for_matrix",
]

# Ordem das colunas da tabela em docs/VALIDACAO_PUBLICA.md §1.
CABECALHO_TABELA = ["dataset", "matriz", "n", "canais", "alvo", "metrica",
                    "referencia", "estado"]

_RE_SECAO = re.compile(r"^## 1\. Tabela consolidada\n\n(.*?)\n\n", re.S | re.M)


def parse_consolidated_table(texto: str) -> List[Dict[str, str]]:
    """Linhas da tabela consolidada como dicts (uma por dataset validado).

    Lista vazia quando a secao nao existe no texto -- quem chama deve tratar
    isso como "nao ha' informacao", nunca como "nao ha' validacao".
    """
    m = _RE_SECAO.search(texto or "")
    if not m:
        return []
    linhas = [ln for ln in m.group(1).splitlines() if ln.startswith("|")]
    linhas_dados = linhas[2:]   # pula cabecalho + separador
    resultado: List[Dict[str, str]] = []
    for ln in linhas_dados:
        campos = [c.strip() for c in ln.strip("|").split("|")]
        if len(campos) < len(CABECALHO_TABELA):
            continue
        resultado.append(dict(zip(CABECALHO_TABELA, campos)))
    return resultado


def load_consolidated_table(raiz: Path) -> List[Dict[str, str]]:
    """Le `docs/VALIDACAO_PUBLICA.md` sob `raiz` e devolve a tabela. Arquivo
    ausente/ilegivel -> lista vazia (mesmo contrato acima)."""
    try:
        texto = (Path(raiz) / "docs" / "VALIDACAO_PUBLICA.md").read_text(
            encoding="utf-8")
    except OSError:
        return []
    return parse_consolidated_table(texto)


def find_rows_for_matrix(tabela: List[Dict[str, str]],
                         matriz: Optional[str]) -> List[Dict[str, str]]:
    """Linhas cuja coluna `matriz` casa com `matriz` (comparacao literal,
    minusculas, subcadeia nos dois sentidos).

    De proposito SEM heuristica esperta: um casamento aproximado que erra
    exibiria a validacao de OUTRA matriz como se fosse a desta execucao. Sem
    casamento, o chamador diz que nao ha' validacao publica registrada para
    o perfil -- que e' a resposta honesta.
    """
    alvo = _normalizar(matriz)
    if not alvo:
        return []
    achadas = []
    for linha in tabela:
        campo = _normalizar(linha.get("matriz"))
        if campo and (campo in alvo or alvo in campo):
            achadas.append(linha)
    return achadas


def _normalizar(texto: Optional[str]) -> str:
    """Minusculas sem acento — "milho em grão" e "milho em grao" sao o mesmo
    texto para efeito de comparacao."""
    nfkd = unicodedata.normalize("NFKD", (texto or "").strip().lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))
