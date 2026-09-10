"""Paridade CLI x web na tabela do fluxo cego (Detectar -> Identificar ->
Quantificar).

REGRESSAO (2026-09-08): a faixa de decisao do Bloco 24 (`faixa_decisao`,
`lod`, `loq`) ja' era calculada em `QuantificationResult` e ja' aparecia na
saida da CLI (`guaraci.py:_menu_prediction`), mas a aba Predicao do app web
montava a tabela sem essas 3 colunas -- o usuario da web via o teor cru sem
saber se estava abaixo do LOD, na zona cinzenta ou quantificavel. Duas
interfaces, dois resultados diferentes para a MESMA predicao.

O contrato aqui e' de paridade de COLUNAS, nao de layout: a web pode
apresentar como quiser (metricas, barras, cores), mas nao pode omitir um
campo que a CLI entrega.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Set

_RAIZ = Path(__file__).resolve().parents[1]
_CLI = _RAIZ / "src" / "guaraci" / "guaraci.py"
_WEB = _RAIZ / "src" / "guaraci" / "app_tabs" / "predicao.py"


def _colunas_df_res(arquivo: Path) -> Set[str]:
    """Nomes usados em `df_res["<nome>"] = ...` no arquivo."""
    tree = ast.parse(arquivo.read_text(encoding="utf-8"))
    nomes: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for alvo in node.targets:
            if (isinstance(alvo, ast.Subscript)
                    and isinstance(alvo.value, ast.Name)
                    and alvo.value.id == "df_res"
                    and isinstance(alvo.slice, ast.Constant)
                    and isinstance(alvo.slice.value, str)):
                nomes.add(alvo.slice.value)
    return nomes


def test_web_expoe_todas_as_colunas_do_fluxo_cego_que_a_cli_expoe():
    faltando = _colunas_df_res(_CLI) - _colunas_df_res(_WEB)
    assert not faltando, (
        "a aba Predicao do app web nao expoe coluna(s) que a CLI entrega "
        f"na mesma predicao: {sorted(faltando)}")


def test_faixa_de_decisao_esta_nas_duas_interfaces():
    """Ancora explicita do Bloco 24 -- o teor nunca deve aparecer sozinho,
    sem dizer em que faixa de decisao ele cai."""
    for arquivo in (_CLI, _WEB):
        colunas = _colunas_df_res(arquivo)
        assert {"teor_estimado", "faixa_decisao", "lod", "loq"} <= colunas, (
            f"{arquivo.name} nao expoe a faixa de decisao junto do teor")
