# -*- coding: utf-8 -*-
"""Contrato de texto do catalogo de tecnicas (`cli_assistente.TECNICAS`,
tela `[1]` do menu principal) e do aviso de maturidade do HSI -- Passo
164/165 da auditoria de documentacao (2026-09-05).

Achado que motivou este teste: o `preproc_rec` do Raman e o aviso de
maturidade do HSI embutiam detalhe de auditoria interna na tela de uso
corrente (data especifica de aprovacao em portao de aceite; contagem
`n=12`/`n=2` por classe) -- informacao que pertence a
`docs/VALIDACAO_PUBLICA.md`, nao a uma tela que o usuario le em segundos
para decidir como configurar uma execucao. Este teste fixa o criterio:
texto de tela pode dizer QUE foi validado e apontar pro documento
completo, nunca reproduzir o numero/data/metodologia em si."""
from __future__ import annotations

import re

from guaraci.cli_assistente import TECNICAS
from guaraci.guaraci import _AVISO_MATURIDADE_HSI_EN, _AVISO_MATURIDADE_HSI_PT

# Padroes que NAO devem aparecer em texto de tela de uso corrente --
# pertencem a docs/VALIDACAO_PUBLICA.md, nao ao catalogo/aviso.
_PADRAO_DATA = re.compile(r"202\d-\d\d-\d\d")
_PADRAO_PASSO = re.compile(r"\bpasso\s*\d+\b", re.IGNORECASE)
_PADRAO_N_IGUAL = re.compile(r"\bn\s*=\s*\d+\b", re.IGNORECASE)
_PADRAO_P_VALOR = re.compile(r"\bp\s*[<=]\s*0[.,]\d+\b", re.IGNORECASE)
_PADRAO_ANTES_DE_USAR = re.compile(
    r"antes de usar para resultado publicavel|before using.*publishable",
    re.IGNORECASE)


def _sem_detalhe_de_auditoria(texto: str) -> list:
    """Retorna a lista de padroes proibidos encontrados em `texto`."""
    achados = []
    for nome, padrao in (
        ("data (AAAA-MM-DD)", _PADRAO_DATA),
        ("referencia a 'Passo N'", _PADRAO_PASSO),
        ("contagem 'n=N'", _PADRAO_N_IGUAL),
        ("p-valor", _PADRAO_P_VALOR),
        ("instrucao 'antes de usar para publicar'", _PADRAO_ANTES_DE_USAR),
    ):
        if padrao.search(texto):
            achados.append(nome)
    return achados


def test_catalogo_de_tecnicas_nao_embute_detalhe_de_auditoria():
    for chave, entrada in TECNICAS.items():
        for lang in ("PT", "EN"):
            for campo in ("desc", "preproc_rec", "faixa", "nome"):
                texto = str(entrada.get(lang, {}).get(campo, ""))
                achados = _sem_detalhe_de_auditoria(texto)
                assert not achados, (
                    f"TECNICAS['{chave}']['{lang}']['{campo}'] tem detalhe de "
                    f"auditoria que deveria estar so' em "
                    f"docs/VALIDACAO_PUBLICA.md: {achados} -- texto: {texto!r}")


def test_catalogo_de_tecnicas_valida_ou_declara_fallback():
    """As 10 tecnicas com validacao real (todas exceto 'generico') devem
    mencionar isso de forma curta, com link -- 'generico' e' o fallback
    e NAO deve alegar ser uma tecnica validada por si so'."""
    for chave, entrada in TECNICAS.items():
        desc_pt = entrada["PT"]["desc"]
        if chave == "generico":
            assert "validado" not in desc_pt.lower() and \
                   "validated" not in entrada["EN"]["desc"].lower(), (
                "'generico' e' o fallback, nao deve alegar validacao propria")
            continue
        assert "docs/VALIDACAO_PUBLICA.md" in desc_pt, (
            f"TECNICAS['{chave}']['PT']['desc'] nao referencia "
            f"docs/VALIDACAO_PUBLICA.md -- ver Passo 165.")


def test_aviso_maturidade_hsi_nao_embute_detalhe_de_auditoria():
    for texto in (_AVISO_MATURIDADE_HSI_PT, _AVISO_MATURIDADE_HSI_EN):
        achados = _sem_detalhe_de_auditoria(texto)
        assert not achados, (
            f"Aviso de maturidade do HSI tem detalhe de auditoria que "
            f"deveria estar so' em docs/VALIDACAO_PUBLICA.md: {achados} -- "
            f"texto: {texto!r}")
    # Ainda precisa citar a limitacao especifica (nao virar frase generica
    # de "prototipo") e apontar pro documento completo -- so' NAO o numero.
    assert "Kaki" in _AVISO_MATURIDADE_HSI_PT and "VIS" in _AVISO_MATURIDADE_HSI_PT
    assert "docs/VALIDACAO_PUBLICA.md" in _AVISO_MATURIDADE_HSI_PT
    assert "docs/VALIDACAO_PUBLICA.md" in _AVISO_MATURIDADE_HSI_EN
