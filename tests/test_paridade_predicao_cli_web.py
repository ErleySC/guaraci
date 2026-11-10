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

ATUALIZADO (2026-09-19, varredura de duplicacao): as ~12 colunas eram
montadas por DUAS copias identicas (CLI e web), e este teste comparava as
duas por AST. Agora ha' UMA fonte (`predicao.anexar_colunas_fluxo_cego`) e
a paridade vale POR CONSTRUCAO -- o teste passa a garantir que as duas
interfaces de fato a CHAMAM (nenhuma volta a montar colunas por conta
propria) e que a fonte unica contem a faixa de decisao.
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


def _chama(arquivo: Path, nome: str) -> bool:
    """O arquivo referencia `nome` (chamada direta, atributo ou alias de
    import) -- `anexar_colunas_fluxo_cego` ou o alias da web."""
    texto = arquivo.read_text(encoding="utf-8")
    return nome in texto


_PRED = _RAIZ / "src" / "guaraci" / "predicao.py"


def test_as_duas_interfaces_usam_a_fonte_unica_das_colunas_do_fluxo_cego():
    assert _chama(_CLI, "anexar_colunas_fluxo_cego")
    assert _chama(_WEB, "anexar_colunas_fluxo_cego")


def test_nenhuma_interface_monta_de_novo_as_colunas_do_fluxo_cego():
    """Regressao da duplicacao: `classe_identificada`/`faixa_decisao` etc.
    so' podem ser ATRIBUIDAS em `predicao.py` -- reaparecer em `guaraci.py`
    ou na aba web significa uma segunda copia nascendo."""
    cobertas = {"classe_identificada", "faixa_decisao", "teor_estimado",
                "identificacao_cobertura", "alpha_total"}
    for arquivo in (_CLI, _WEB):
        duplicadas = _colunas_df_res(arquivo) & cobertas
        assert not duplicadas, (
            f"{arquivo.name} voltou a montar coluna(s) do fluxo cego por "
            f"conta propria: {sorted(duplicadas)} -- use "
            "predicao.anexar_colunas_fluxo_cego")


def test_faixa_de_decisao_esta_na_fonte_unica():
    """Ancora explicita do Bloco 24 -- o teor nunca deve aparecer sozinho,
    sem dizer em que faixa de decisao ele cai."""
    assert {"teor_estimado", "faixa_decisao", "lod", "loq"} <= _colunas_df_res(_PRED)
