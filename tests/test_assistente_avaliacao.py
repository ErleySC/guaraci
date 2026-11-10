"""Avaliacao do assistente com perguntas que NAO foram escritas junto com as regras.

Quatro conjuntos em tests/perguntas_assistente/ (fora de tests/dados, que o .gitignore exclui):
  semente.tsv    41 perguntas do relatorio de 2026-10-15, escritas ANTES das regras; as regras foram
                 ajustadas olhando para elas (nao e' cego).
  reservado.tsv  48 perguntas escritas depois e medidas uma vez sem ajuste (35 certas, 5 erradas com
                 confianca); depois disso entraram correcoes de mecanismo. Nao e' mais cego.
  cego2.tsv      34 perguntas escritas antes da busca BM25; baseline 20 certas, 2 erradas. Serviu para
                 escolher os limiares da busca, entao tambem nao e' mais cego.
  cego3.tsv      27 perguntas escritas antes da busca BM25 e medidas uma vez com ela: 16 certas, 1 errada
                 (as 2 erradas do baseline ja eram conhecidas). Depois entraram so' um limiar minimo de
                 relevancia e a correcao de um gatilho; nao e' cego para esses dois ajustes.
O conjunto que vale de verdade e' o de perguntas de usuarios reais: acrescente linhas a um novo arquivo
(pergunta, lang, esperado) e nao mexa nas regras ao mesmo tempo.

`esperado` e' `tipo:id` (ou varios separados por `|`), ou `nao_sei`. Resultado de uma pergunta:
  certa      resposta direta com o id esperado
  busca      lista de resultados (nao e' resposta direta)
  nao_sei    o assistente disse que nao sabe
  ERRADA     resposta direta com outro id -- o pior caso: o usuario confia e esta errado
"""
from __future__ import annotations

import csv
import os
from collections import Counter
from typing import Dict, List, Tuple

import pytest

from guaraci import assistente as ass

_PASTA = os.path.join(os.path.dirname(__file__), "perguntas_assistente")


def _ler(nome: str) -> List[Dict[str, str]]:
    with open(os.path.join(_PASTA, nome), encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _classificar(linha: Dict[str, str]) -> Tuple[str, str]:
    r = ass.responder_estruturado(linha["pergunta"], linha["lang"])
    esperado = linha["esperado"].split("|")
    obtido = "nao_sei" if r["tipo"] == "nao_sei" else f"{r['tipo']}:{r['id']}"
    if r["tipo"] == "busca":
        return "busca", obtido
    if obtido in esperado:
        return "certa", obtido
    if r["tipo"] == "nao_sei":
        return "nao_sei", obtido
    return "ERRADA", obtido


def _medir(nome: str) -> Tuple[Counter, List[str]]:
    c: Counter = Counter()
    erradas = []
    for linha in _ler(nome):
        estado, obtido = _classificar(linha)
        c[estado] += 1
        if estado == "ERRADA":
            erradas.append(f"{linha['pergunta']!r} -> {obtido} (esperado {linha['esperado']})")
    return c, erradas


@pytest.mark.parametrize("nome,minimo", [("semente.tsv", 0.85),
                                         ("reservado.tsv", 0.75),
                                         ("cego2.tsv", 0.65),
                                         ("cego3.tsv", 0.60)])
def test_nenhuma_resposta_errada_com_confianca_e_acerto_minimo(nome, minimo):
    c, erradas = _medir(nome)
    total = sum(c.values())
    assert not erradas, "respostas erradas com confianca:\n" + "\n".join(erradas)
    assert c["certa"] / total >= minimo, f"{nome}: {dict(c)} de {total}"


def test_conjuntos_bem_formados():
    for nome in ("semente.tsv", "reservado.tsv", "cego2.tsv", "cego3.tsv"):
        linhas = _ler(nome)
        assert len(linhas) >= 27
        for ln in linhas:
            assert ln["lang"] in ("PT", "EN"), ln
            for e in ln["esperado"].split("|"):
                assert e == "nao_sei" or ":" in e, ln


def test_fora_do_assunto_nao_vira_resposta():
    for p in ("qual a capital da Franca?", "como troco o pneu do carro?"):
        assert ass.responder_estruturado(p)["tipo"] == "nao_sei"
