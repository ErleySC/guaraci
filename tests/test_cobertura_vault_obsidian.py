# -*- coding: utf-8 -*-
"""Auditoria de cobertura do vault Obsidian (Passo 172).

Compara a lista REAL de módulos/técnicas/passos/datasets contra o que o
gerador efetivamente colocou no plano de escrita -- independente da
lógica interna do gerador (cada teste recalcula sua própria "verdade",
não reaproveita o resultado que está checando). Se algo faltar aqui, a
correção é no GERADOR, não neste teste nem no motor de consulta.
"""
from __future__ import annotations

import ast
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
    """`10-Tecnicas/` cobre as chaves de `TECNICAS` MAIS as modalidades
    reais fora do catálogo (`Config.mode`: `imagem`/`hsi` -- ver
    `MODOS_FORA_DO_CATALOGO` em gerar_vault_obsidian.py). `sintetico` fica
    de fora de propósito (dado de teste/demo, não técnica analítica)."""
    tecnicas_reais = gvo.parse_tecnicas_catalog()
    notas_tecnicas = [rel for rel in plano if rel.startswith("10-Tecnicas/")]
    esperado = len(tecnicas_reais) + len(gvo.MODOS_FORA_DO_CATALOGO)
    assert len(notas_tecnicas) == esperado, (
        f"{len(tecnicas_reais)} técnicas em TECNICAS + "
        f"{len(gvo.MODOS_FORA_DO_CATALOGO)} fora do catálogo = {esperado} esperadas, "
        f"mas {len(notas_tecnicas)} notas em 10-Tecnicas/"
    )
    slugs_esperados = {
        gvo._slug(dados.get("PT", {}).get("nome", chave))
        for chave, dados in tecnicas_reais.items()
    } | {gvo._slug(item["nome"]) for item in gvo.MODOS_FORA_DO_CATALOGO}
    slugs_no_vault = {rel[len("10-Tecnicas/"):-len(".md")] for rel in notas_tecnicas}
    assert slugs_esperados == slugs_no_vault


def test_modalidades_fora_do_catalogo_tem_modulo_principal_real():
    """Cada item de `MODOS_FORA_DO_CATALOGO` precisa apontar para um
    módulo que realmente existe -- se o módulo for renomeado/removido, a
    nota correspondente é omitida silenciosamente (ver
    `gerar_tecnicas_fora_do_catalogo`); este teste garante que isso não
    acontece sem ninguém perceber."""
    modulos_reais = {p.stem for p in (_RAIZ / "src" / "guaraci").glob("*.py")}
    faltando = [item["modulo_principal"] for item in gvo.MODOS_FORA_DO_CATALOGO
                if item["modulo_principal"] not in modulos_reais]
    assert not faltando, f"módulo principal inexistente: {faltando}"


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


# ─────────────────────────────────────────────────────────────────────────
#  25-Funcoes -- 1 nota por módulo que tem função pública em __all__
#  (Passo 178, escopo reduzido de "toda função de __all__" para "1 por
#  módulo" -- ver docstring da seção no gerador)
# ─────────────────────────────────────────────────────────────────────────

def _modulos_com_funcao_publica_em_all() -> list[str]:
    """Recalcula, direto do AST (independente de `gvo.parse_funcoes_principais`),
    quais módulos têm ao menos 1 função top-level cujo nome está em
    `__all__` -- verdade própria deste teste, não reaproveitada."""
    modulos = []
    for caminho in sorted((_RAIZ / "src" / "guaraci").glob("*.py")):
        texto = caminho.read_text(encoding="utf-8")
        arvore = ast.parse(texto)
        all_publico: list[str] = []
        for node in ast.walk(arvore):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if getattr(t, "id", None) == "__all__":
                        try:
                            all_publico = list(ast.literal_eval(node.value))
                        except (ValueError, SyntaxError):
                            pass
        funcoes_top = {n.name for n in arvore.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if any(n in funcoes_top for n in all_publico):
            modulos.append(caminho.stem)
    return modulos


def test_todo_modulo_com_funcao_publica_em_all_tem_nota_em_funcoes(plano):
    modulos_com_funcao = _modulos_com_funcao_publica_em_all()
    notas_funcoes = [rel for rel in plano if rel.startswith("25-Funcoes/")]
    assert len(notas_funcoes) == len(modulos_com_funcao), (
        f"{len(modulos_com_funcao)} módulo(s) com função pública em __all__, "
        f"{len(notas_funcoes)} nota(s) em 25-Funcoes/"
    )


def test_toda_nota_de_funcao_referencia_seu_modulo_de_verdade(plano):
    modulos_reais = {p.stem for p in (_RAIZ / "src" / "guaraci").glob("*.py")}
    sem_modulo_real = []
    for rel, conteudo in plano.items():
        if not rel.startswith("25-Funcoes/"):
            continue
        m = re.search(r"^modulo: '\[\[(.+?)\.py\]\]'$", conteudo, re.M)
        if not m or m.group(1) not in modulos_reais:
            sem_modulo_real.append(rel)
    assert not sem_modulo_real, f"nota(s) de função sem módulo real: {sem_modulo_real}"


# ─────────────────────────────────────────────────────────────────────────
#  05-Identidade e 06-Autoria-e-Seguranca -- conjunto fixo de notas
# ─────────────────────────────────────────────────────────────────────────

def test_05_identidade_tem_as_notas_esperadas(plano):
    esperado = {
        "05-Identidade/Missao.md", "05-Identidade/Identidade-Visual.md",
        "05-Identidade/Mascote.md", "05-Identidade/Tom-de-Voz.md",
        "05-Identidade/MOC-Identidade.md",
    }
    real = {rel for rel in plano if rel.startswith("05-Identidade/")}
    assert real == esperado


def test_06_autoria_seguranca_tem_as_notas_esperadas(plano):
    esperado = {
        "06-Autoria-e-Seguranca/Autoria.md", "06-Autoria-e-Seguranca/Proveniencia.md",
        "06-Autoria-e-Seguranca/Seguranca-de-Dados.md",
        "06-Autoria-e-Seguranca/MOC-Autoria-Seguranca.md",
    }
    real = {rel for rel in plano if rel.startswith("06-Autoria-e-Seguranca/")}
    assert real == esperado


def test_autoria_le_nome_e_email_reais_do_citation_cff(plano):
    """Contra-prova de que Autoria.md vem de CITATION.cff de verdade, não
    de um literal escrito no gerador -- recalcula direto do arquivo."""
    citation = gvo.parse_citation_cff()
    autor = citation["authors"][0]
    nome_esperado = f"{autor['given-names']} {autor['family-names']}"
    conteudo = plano["06-Autoria-e-Seguranca/Autoria.md"]
    assert nome_esperado in conteudo
    assert autor["email"] in conteudo
    assert autor["orcid"] in conteudo


def test_identidade_visual_le_paleta_real_de_design_md(plano):
    paleta = gvo.parse_paleta_mascote()
    assert paleta, "docs/DESIGN.md §1 não produziu nenhuma linha de paleta"
    conteudo = plano["05-Identidade/Identidade-Visual.md"]
    for cor in paleta:
        assert cor["hex_moda"] in conteudo, f"hex {cor['hex_moda']} não apareceu na nota"
