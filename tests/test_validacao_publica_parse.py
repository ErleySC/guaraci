"""Leitura da tabela consolidada de `docs/VALIDACAO_PUBLICA.md`.

O parsing saiu de `scripts/gerar_vault_obsidian.py` para
`guaraci.validacao_publica` (2026-09-08) porque o painel de status da aba
Projeto precisa da MESMA leitura. O teste de paridade abaixo e' o que impede
as duas de divergirem de novo.
"""
from __future__ import annotations

import sys
from pathlib import Path

from guaraci.validacao_publica import (
    find_rows_for_matrix,
    load_consolidated_table,
    parse_consolidated_table,
)

_RAIZ = Path(__file__).resolve().parents[1]

_TABELA = """\
# Validacoes

## 1. Tabela consolidada

| Dataset | Matriz | n | Canais | Alvo | Metrica | Referencia | Estado |
|---|---|---|---|---|---|---|---|
| **Corn** (m5) | milho em grão | 80 | 700 | umidade | RMSEP 0,14 | Eigenvector | ✅ dentro da faixa |
| **Mel** | mel | 100 | 1200 | origem | acc 0,91 | Mendeley | ⚠ parcial |

## 2. Outra secao
"""


def test_parse_le_linhas_e_colunas_da_tabela():
    linhas = parse_consolidated_table(_TABELA)
    assert len(linhas) == 2
    assert linhas[0]["matriz"] == "milho em grão"
    assert linhas[0]["estado"] == "✅ dentro da faixa"
    assert linhas[1]["dataset"] == "**Mel**"


def test_texto_sem_a_secao_devolve_lista_vazia():
    assert parse_consolidated_table("# Doc sem tabela\n\ntexto\n") == []
    assert parse_consolidated_table("") == []


def test_busca_por_matriz_ignora_acento_e_caixa():
    linhas = parse_consolidated_table(_TABELA)
    assert find_rows_for_matrix(linhas, "MILHO EM GRAO")[0]["dataset"] == "**Corn** (m5)"
    assert find_rows_for_matrix(linhas, "mel") != []


def test_busca_sem_casamento_nao_devolve_validacao_de_outra_matriz():
    """Casar por aproximacao exibiria a validacao de OUTRA matriz como se
    fosse a desta execucao -- prefere-se nao achar nada."""
    linhas = parse_consolidated_table(_TABELA)
    assert find_rows_for_matrix(linhas, "óleo vegetal amazônico") == []
    assert find_rows_for_matrix(linhas, None) == []
    assert find_rows_for_matrix(linhas, "  ") == []


def test_le_o_documento_real_do_repositorio():
    linhas = load_consolidated_table(_RAIZ)
    assert linhas, "docs/VALIDACAO_PUBLICA.md deveria ter a tabela consolidada"
    assert all(l["dataset"] and l["matriz"] for l in linhas)


def test_gerador_do_vault_usa_a_mesma_leitura():
    """Anti-duplicacao: o script do vault tem que devolver exatamente o que a
    funcao do pacote devolve, nao uma copia do parser."""
    sys.path.insert(0, str(_RAIZ / "scripts"))
    try:
        import gerar_vault_obsidian as vault
    finally:
        sys.path.pop(0)
    assert vault.parse_tabela_consolidada() == load_consolidated_table(_RAIZ)
