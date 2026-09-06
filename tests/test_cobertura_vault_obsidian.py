# -*- coding: utf-8 -*-
"""Auditoria de cobertura do vault Obsidian (Passo 172).

Compara a lista REAL de módulos/técnicas/passos/datasets contra o que o
gerador efetivamente colocou no plano de escrita -- independente da
lógica interna do gerador (cada teste recalcula sua própria "verdade",
não reaproveita o resultado que está checando). Se algo faltar aqui, a
correção é no GERADOR, não neste teste nem no motor de consulta.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parents[1]
_SCRIPTS = str(_RAIZ / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import gerar_vault_obsidian as gvo  # noqa: E402


@pytest.fixture(scope="module")
def plano():
    p, _contagens, _avisos = gvo.montar_plano()
    return p


def _fontes_progresso(plano: dict[str, str], pasta: str) -> list[int]:
    """Linhas de `docs/PROGRESSO.md` citadas em `fonte:` das notas de `pasta`."""
    linhas = []
    for rel, conteudo in plano.items():
        if not rel.startswith(pasta + "/"):
            continue
        for m in re.finditer(r"docs/PROGRESSO\.md:(\d+)", conteudo):
            linhas.append(int(m.group(1)))
    return linhas


# ─────────────────────────────────────────────────────────────────────────
#  Módulos
# ─────────────────────────────────────────────────────────────────────────

def test_todo_modulo_de_src_guaraci_tem_nota(plano):
    modulos_reais = {p.stem for p in (_RAIZ / "src" / "guaraci").glob("*.py")}
    modulos_no_vault = {
        rel[len("20-Modulos/"):-len(".py.md")]
        for rel in plano if rel.startswith("20-Modulos/") and rel.endswith(".py.md")
    }
    faltando = modulos_reais - modulos_no_vault
    sobrando = modulos_no_vault - modulos_reais
    assert not faltando, f"módulo(s) sem nota em 20-Modulos/: {sorted(faltando)}"
    assert not sobrando, f"nota de módulo sem módulo real correspondente: {sorted(sobrando)}"


# ─────────────────────────────────────────────────────────────────────────
#  Técnicas
# ─────────────────────────────────────────────────────────────────────────

def test_todas_as_tecnicas_do_catalogo_tem_nota(plano):
    tecnicas_reais = gvo.parse_tecnicas_catalog()
    notas_tecnicas = [rel for rel in plano if rel.startswith("10-Tecnicas/")]
    assert len(notas_tecnicas) == len(tecnicas_reais), (
        f"{len(tecnicas_reais)} técnicas em TECNICAS, mas {len(notas_tecnicas)} "
        "notas em 10-Tecnicas/"
    )
    slugs_esperados = {
        gvo._slug(dados.get("PT", {}).get("nome", chave))
        for chave, dados in tecnicas_reais.items()
    }
    slugs_no_vault = {rel[len("10-Tecnicas/"):-len(".md")] for rel in notas_tecnicas}
    assert slugs_esperados == slugs_no_vault


# ─────────────────────────────────────────────────────────────────────────
#  Passos numerados de docs/PROGRESSO.md
# ─────────────────────────────────────────────────────────────────────────

def test_todo_passo_do_progresso_e_rastreavel_em_achados_ou_decisoes(plano):
    blocos = gvo.parse_blocos_passo()
    assert blocos, "nenhum '## Passo' encontrado em docs/PROGRESSO.md -- parser quebrou"

    linhas_citadas = (_fontes_progresso(plano, "60-Achados")
                       + _fontes_progresso(plano, "50-Decisoes"))
    assert linhas_citadas, "nenhuma nota em 60-Achados/50-Decisoes cita docs/PROGRESSO.md"

    sem_rastro = []
    for bloco in blocos:
        if not any(bloco.linha_inicio <= ln < bloco.linha_fim for ln in linhas_citadas):
            sem_rastro.append(bloco.titulo)
    assert not sem_rastro, (
        f"{len(sem_rastro)} passo(s) sem nenhuma nota em 60-Achados/50-Decisoes: "
        f"{sem_rastro[:10]}{'...' if len(sem_rastro) > 10 else ''}"
    )


def test_contagem_de_passos_bate_com_grep_direto():
    """Confere o parser de blocos contra uma contagem independente (grep
    puro em texto), para não confiar só na própria lógica do parser."""
    texto = (_RAIZ / "docs" / "PROGRESSO.md").read_text(encoding="utf-8")
    n_grep = len(re.findall(r"^## Passo", texto, re.M))
    n_parser = len(gvo.parse_blocos_passo())
    assert n_grep == n_parser == 84 or n_grep == n_parser, (
        f"grep direto achou {n_grep} '## Passo', parser achou {n_parser}"
    )


# ─────────────────────────────────────────────────────────────────────────
#  Datasets públicos (VALIDACAO_PUBLICA.md)
# ─────────────────────────────────────────────────────────────────────────

def test_toda_linha_da_tabela_consolidada_tem_nota_de_validacao(plano):
    tabela = gvo.parse_tabela_consolidada()
    notas_validacao = [rel for rel in plano if rel.startswith("40-Validacoes/")]
    assert tabela, "tabela consolidada de VALIDACAO_PUBLICA.md não encontrada/vazia"
    assert len(notas_validacao) == len(tabela), (
        f"{len(tabela)} linhas na tabela consolidada, {len(notas_validacao)} "
        "notas em 40-Validacoes/"
    )


def test_contagem_de_datasets_bate_com_grep_direto():
    """A tabela consolidada é uma seção markdown única (`## 1. Tabela
    consolidada`) -- confere a contagem de linhas de dado por grep direto,
    independente do parser do gerador."""
    texto = (_RAIZ / "docs" / "VALIDACAO_PUBLICA.md").read_text(encoding="utf-8")
    m = re.search(r"^## 1\. Tabela consolidada\n\n(.*?)\n\n", texto, re.S | re.M)
    assert m, "seção '## 1. Tabela consolidada' não encontrada"
    linhas_tabela = [ln for ln in m.group(1).splitlines() if ln.startswith("|")]
    n_dados_grep = len(linhas_tabela) - 2  # cabeçalho + separador
    n_parser = len(gvo.parse_tabela_consolidada())
    assert n_dados_grep == n_parser


# ─────────────────────────────────────────────────────────────────────────
#  Conceitos -- todo módulo referenciado por CONCEITOS precisa existir
# ─────────────────────────────────────────────────────────────────────────

def test_todos_os_modulos_referenciados_por_conceitos_existem():
    modulos_reais = {p.stem for p in (_RAIZ / "src" / "guaraci").glob("*.py")}
    faltando = []
    for c in gvo.CONCEITOS:
        for m in c["modulos"]:
            if m not in modulos_reais:
                faltando.append((c["titulo"], m))
    assert not faltando, (
        f"conceito(s) apontando para módulo que não existe mais: {faltando} "
        "-- corrigir a tabela CONCEITOS em gerar_vault_obsidian.py"
    )
