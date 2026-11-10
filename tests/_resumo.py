"""Leitura do resumo da execucao como os leitores do programa fazem.

Desde que o resumo_modelo.txt sai no idioma da sessao (pedido de 2026-11-06), as chaves fixas do
pipeline ("Grouping guarantee", "DD-SIMCA X sens(LOGO)/esp"...) moram em resumo_modelo.json, ao lado.
Testes que conferem essas chaves leem o texto canonico por aqui."""
from __future__ import annotations

from guaraci.resumo_parse import ler_resumo_canonico


def canonico(caminho) -> str:
    texto = ler_resumo_canonico(str(caminho))
    assert texto is not None, caminho
    return texto
