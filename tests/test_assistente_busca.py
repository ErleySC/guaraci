"""Busca BM25 do assistente: ranqueia por idf, expande sinonimos e recusa o que nao tem relacao com o catalogo."""
from __future__ import annotations

import pytest

from guaraci import assistente as ass
from guaraci import assistente_busca as B

DOCS = [
    B.Documento("duvida", "outliers", "como achar outlier amostra anomala espectro atipico"),
    B.Documento("duvida", "lote", "efeito de lote ou sessao: amostras de dias diferentes se separam"),
    B.Documento("duvida", "amostras", "amostras amostras amostras de treino e teste"),
    B.Documento("termo", "vip", "importancia da variavel no modelo PLS"),
]


def test_radical_e_sinonimos():
    assert B.radical("variaveis") == B.radical("variavel") == "varia"
    assert B.radical("pls") == "pls"
    c = B.conceitos("espectros anomalos")
    assert len(c) == 2
    assert any(r == B.radical("outlier") for r, _ in c[1])           # anomalo -> outlier, com peso menor
    assert all(w < 1.0 for r, w in c[1] if r == B.radical("outlier"))


def test_sinonimo_acha_o_documento():
    rk = B.ranquear("amostra anomala", DOCS)
    assert rk[0].id == "outliers"
    rk = B.ranquear("os dias de medicao mudam o resultado", DOCS)
    assert rk[0].id == "lote"


def test_idf_palavra_comum_pesa_menos():
    rk = B.ranquear("amostras outlier", DOCS)
    assert rk[0].id == "outliers"


def test_consulta_sem_relacao_nao_tem_cobertura():
    assert B.ranquear("receita de bolo de cenoura", DOCS) == []
    assert B.decidir([]) == "nada"


def test_decidir_direta_lista_nada():
    R = B.Resultado
    assert B.decidir([R("duvida", "a", 10.0, 0.8, 1.0), R("duvida", "b", 5.0, 0.4, 0.8)]) == "direta"
    assert B.decidir([R("duvida", "a", 10.0, 0.8, 1.0), R("duvida", "b", 9.5, 0.4, 0.8)]) == "lista"   # sem folga
    assert B.decidir([R("duvida", "a", 10.0, 0.3, 1.0)]) == "lista"                                     # pouco relevante
    assert B.decidir([R("duvida", "a", 10.0, 0.8, 0.2)]) == "nada"                                      # nao cobre
    assert B.decidir([R("duvida", "a", 1.0, 0.1, 1.0)]) == "nada"                                       # relevancia minima


def test_deterministico():
    a = B.ranquear("como achar anomalias", DOCS)
    b = B.ranquear("como achar anomalias", DOCS)
    assert a == b


# ------------------------------------------------------------ ligacao com o assistente
def test_fora_do_assunto_vira_nao_sei():
    for p in ("como faço um bolo?", "escreva um poema sobre espectroscopia", "who won the football world cup in 2014?"):
        assert ass.responder_estruturado(p)["tipo"] == "nao_sei", p


def test_negacao_nunca_vira_resposta_direta_do_bm25():
    d, direta = ass._duvida_por_bm25("qual tamanho de janela do Savitzky-Golay devo escolher?", "PT")
    assert d["id"] == "janela_sg" and direta
    d, direta = ass._duvida_por_bm25("a janela do Savitzky-Golay não é a que devo escolher", "PT")
    assert not direta


def test_palavra_solta_so_em_pergunta_curta_ou_de_definicao():
    # antes: 'pre-processamento' sozinho respondia a opcao com confianca para uma pergunta sobre outra coisa
    r = ass.responder_estruturado("tem como saber se o pré-processamento ajudou?")
    assert not (r["tipo"] == "opcao")
    assert ass.responder_estruturado("o que é VIP")["tipo"] == "termo"
    assert ass.responder_estruturado("quando usar OPLS-DA")["tipo"] in ("opcao", "tecnica")


def test_gatilho_erra_nao_casa_errada():
    r = ass.responder_estruturado("amostras novas de outra espécie aparecem como pertencentes à classe errada")
    assert r["id"] != "overfitting"
    assert ass.responder_estruturado("o modelo erra amostras novas")["id"] == "overfitting"


def test_arbitragem_regra_x_bm25():
    r = ass.responder_estruturado("my model says 100% accuracy with very few samples", "EN")
    assert (r["tipo"], r["id"]) == ("duvida", "acuracia_perfeita")


@pytest.mark.parametrize("pergunta,esperado", [
    ("qual tamanho de janela do Savitzky-Golay devo escolher?", "janela_sg"),
    ("how do I cite this software in a paper?", "citar_guaraci"),
])
def test_resposta_direta_por_bm25(pergunta, esperado):
    r = ass.responder_estruturado(pergunta, "PT" if "qual" in pergunta else "EN")
    assert (r["tipo"], r["id"]) == ("duvida", esperado)
