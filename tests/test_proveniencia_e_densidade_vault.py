# -*- coding: utf-8 -*-
"""Testes do vault Obsidian para os Passos 179/180 e 183/184 da auditoria
de rede de conhecimento (2026-09-06):

- Passo 179/180: todo link do plano carrega uma relação explícita
  (`[[alvo]] — relação`), e um teste de densidade de grafo sinaliza
  notas órfãs (sem link de saída ou de entrada) por categoria --
  "sinalizar, não necessariamente bloquear" (a instrução original).
- Passo 183/184: toda nota gerada (exceto `Autoria.md` nela mesma e
  `README-VAULT.md`, que é documentação SOBRE o vault) carrega
  `autor: "[[Autoria]]"` no frontmatter -- proveniência universal.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
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


# ─────────────────────────────────────────────────────────────────────────
#  Passo 183/184 -- campo `autor:` universal
# ─────────────────────────────────────────────────────────────────────────

def test_toda_nota_md_exceto_excecoes_tem_autor_universal(plano):
    excecoes = {"README-VAULT.md", "06-Autoria-e-Seguranca/Autoria.md"}
    sem_autor = []
    for rel, conteudo in plano.items():
        if not rel.endswith(".md") or rel in excecoes:
            continue
        if not re.search(r'^autor: "\[\[Autoria\]\]"$', conteudo, re.M):
            sem_autor.append(rel)
    assert not sem_autor, (
        f"{len(sem_autor)} nota(s) sem `autor: \"[[Autoria]]\"` no frontmatter: "
        f"{sem_autor[:10]}{'...' if len(sem_autor) > 10 else ''}"
    )


def test_autoria_md_nao_se_autorreferencia_no_campo_autor(plano):
    conteudo = plano["06-Autoria-e-Seguranca/Autoria.md"]
    assert not re.search(r"^autor:", conteudo, re.M), (
        "Autoria.md não deve ter campo `autor:` apontando pra ela mesma"
    )


def test_readme_vault_nao_tem_campo_autor():
    """README-VAULT.md é documentação SOBRE o vault, não uma nota de
    conteúdo -- decisão explícita (ver docstring do módulo), não descuido."""
    plano, _contagens, _avisos = gvo.montar_plano()
    conteudo = plano["README-VAULT.md"]
    assert not re.search(r"^autor:", conteudo, re.M)


# ─────────────────────────────────────────────────────────────────────────
#  Passo 179/180 -- todo link carrega relação; densidade de grafo
# ─────────────────────────────────────────────────────────────────────────

def _links_de(conteudo: str) -> list[tuple[str, bool]]:
    """`[(alvo, tem_relacao)]` para cada wikilink de `conteudo`."""
    resultado = []
    for m in re.finditer(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]([^\n]*)", conteudo):
        alvo = m.group(1).strip()
        resto = m.group(2)
        tem_relacao = resto.strip().startswith("—")
        resultado.append((alvo, tem_relacao))
    return resultado


def test_a_grande_maioria_dos_links_de_lista_carrega_relacao_explicita(plano):
    """Não é 100% (autolink em prosa dentro de docstring/achado citando um
    módulo entre crases fica sem relação por design -- a frase ao redor já
    dá o contexto); mas os links estruturados (Ver também, Depende de,
    Usado por, MOCs etc.) devem carregar relação na grande maioria dos
    casos. Sinaliza regressão sem exigir 100% cego."""
    total = 0
    com_relacao = 0
    for conteudo in plano.values():
        for _alvo, tem_relacao in _links_de(conteudo):
            total += 1
            com_relacao += tem_relacao
    assert total > 0
    proporcao = com_relacao / total
    assert proporcao >= 0.6, (
        f"só {proporcao:.0%} dos {total} links do plano carregam relação "
        "explícita (\"— relação\") -- abaixo do esperado após o Passo 180"
    )


def test_densidade_de_grafo_por_categoria(plano, capsys):
    """Calcula grau de saída médio por categoria e lista notas órfãs (sem
    link de saída OU sem link de entrada) -- SINALIZA (imprime um
    relatório), não bloqueia por nota individual (a própria instrução
    original diz "não necessariamente bloquear"). Só falha se uma
    categoria inteira de conteúdo (excluindo MOC/Canvas/README/Estado, que
    são meta-navegação) ficar com grau de saída médio zero -- isso sim é
    sinal de bug no gerador, não de nota legitimamente terminal."""
    saida: dict[str, set[str]] = defaultdict(set)
    entrada: dict[str, set[str]] = defaultdict(set)
    stems = {Path(rel).stem: rel for rel in plano if rel.endswith(".md")}

    for rel, conteudo in plano.items():
        if not rel.endswith(".md"):
            continue
        origem = Path(rel).stem
        for alvo, _rel_txt in _links_de(conteudo):
            if alvo in stems and alvo != origem:
                saida[rel].add(alvo)
                entrada[alvo].add(rel)

    categorias: dict[str, list[str]] = defaultdict(list)
    for rel in plano:
        if rel.endswith(".md"):
            categorias[rel.split("/")[0]].append(rel)

    linhas_relatorio = []
    orfas: list[str] = []
    medias_por_categoria: dict[str, float] = {}
    for categoria, notas in sorted(categorias.items()):
        graus_saida = [len(saida.get(n, set())) for n in notas]
        media = sum(graus_saida) / len(graus_saida) if graus_saida else 0.0
        medias_por_categoria[categoria] = media
        linhas_relatorio.append(f"{categoria}: {len(notas)} nota(s), "
                                 f"grau de saída médio {media:.2f}")
        for n in notas:
            stem = Path(n).stem
            if not saida.get(n) or not entrada.get(stem):
                orfas.append(n)

    print("\n".join(linhas_relatorio))
    print(f"{len(orfas)} nota(s) órfã(s) (sem link de saída OU sem link de "
          f"entrada) de {sum(len(v) for v in categorias.values())} nota(s) totais")
    if orfas:
        print("Órfãs:", orfas[:30], "..." if len(orfas) > 30 else "")

    categorias_conteudo = {"05-Identidade", "10-Tecnicas", "20-Modulos", "25-Funcoes",
                            "30-Conceitos", "40-Validacoes", "50-Decisoes", "60-Achados",
                            "06-Autoria-e-Seguranca"}
    zeradas = [c for c in categorias_conteudo
               if c in medias_por_categoria and medias_por_categoria[c] == 0.0]
    assert not zeradas, (
        f"categoria(s) de conteúdo com grau de saída médio ZERO (toda nota "
        f"sem nenhum link de saída): {zeradas} -- provável bug no gerador, "
        "não notas legitimamente terminais"
    )
