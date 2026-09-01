"""Rede de seguranca do catalogo de tecnicas do assistente `G` (Agente 6).

Dois contratos, cada um pegando um jeito diferente do catalogo mentir:

1. Toda `referencia` de TechniqueEntry precisa apontar pra um simbolo que
   EXISTE de verdade (pega entrada que sobrevive a uma renomeacao/remocao
   do simbolo real -- o catalogo afirmaria um metodo que nao existe mais).
2. Para os modulos de proposito UNICO (todo simbolo publico e' de fato uma
   tecnica, listados em MODULOS_COBERTURA_TOTAL), todo `__all__` precisa
   estar OU no registry OU na lista de excecoes documentada -- pega metodo
   novo adicionado sem atualizar o catalogo (o motivo desta rodada existir:
   `_guaraci_navegar_secoes` tinha virado uma segunda fonte, desatualizada,
   do que ja existia em `_CONFIG_SPEC`/`_HELP_DB` -- ver docs/DESIGN.md).
"""
from __future__ import annotations

import importlib

from guaraci.technique_registry import MODULOS_COBERTURA_TOTAL, REGISTRY


def test_toda_referencia_do_catalogo_resolve_para_simbolo_real():
    faltando = []
    for entry in REGISTRY:
        modulo_nome, _, simbolo = entry.referencia.rpartition(".")
        modulo = importlib.import_module(modulo_nome)
        if not hasattr(modulo, simbolo):
            faltando.append(f"{entry.id}: {entry.referencia}")
    assert not faltando, (
        f"Entrada(s) do catalogo apontando pra simbolo inexistente: "
        f"{faltando}. O metodo foi renomeado/removido -- atualize a "
        f"`referencia` em technique_registry.py.")


def test_ids_do_catalogo_sao_unicos():
    ids = [e.id for e in REGISTRY]
    assert len(ids) == len(set(ids)), f"id duplicado em REGISTRY: {ids}"


def test_modulos_de_proposito_unico_tem_cobertura_completa():
    referenciados = {e.referencia.rsplit(".", 1)[-1] for e in REGISTRY}
    faltando = {}
    for modulo_nome, regra in MODULOS_COBERTURA_TOTAL.items():
        modulo = importlib.import_module(modulo_nome)
        todos = set(getattr(modulo, "__all__", ()))
        cobertos = regra["incluir"] | regra["excecoes"]
        sobra = todos - cobertos
        if sobra:
            faltando[modulo_nome] = sorted(sobra)
        # Contra-prova reversa: tudo que a regra diz "incluir" precisa
        # de fato estar referenciado por alguma entrada do catalogo --
        # senao a regra promete cobertura que o REGISTRY nao cumpre.
        nao_registrado = regra["incluir"] - referenciados
        assert not nao_registrado, (
            f"{modulo_nome}: simbolo(s) em 'incluir' sem entrada "
            f"correspondente no REGISTRY: {sorted(nao_registrado)}")
    assert not faltando, (
        f"Modulo(s) de proposito unico com simbolo novo em __all__ sem "
        f"entrada no REGISTRY nem na lista de excecoes de "
        f"MODULOS_COBERTURA_TOTAL (technique_registry.py): {faltando}. "
        f"Adicione uma TechniqueEntry para o metodo novo, ou -- se ele "
        f"genuinamente nao for uma 'tecnica' (dataclass de resultado, "
        f"excecao, helper) -- inclua o nome em 'excecoes'.")


def test_categorias_do_catalogo_sao_um_conjunto_pequeno_e_fechado():
    """Categoria e' o agrupamento que o assistente usa pra apresentar o
    catalogo -- uma categoria nova precisa ser deliberada, nao surgir de
    um typo numa entrada nova."""
    categorias_conhecidas = {
        "classificacao_deteccao", "quantificacao",
        "identificacao_conjunto_aberto", "selecao_amostras",
        "transferencia_calibracao", "figuras_de_merito",
        "robustez_linearidade", "perfis",
    }
    usadas = {e.categoria for e in REGISTRY}
    assert usadas <= categorias_conhecidas, (
        f"Categoria(s) nao reconhecida(s): {usadas - categorias_conhecidas}. "
        f"Se for deliberada, adicione a categoria_conhecidas neste teste.")
