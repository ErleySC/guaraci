# -*- coding: utf-8 -*-
"""Nenhum modulo do pacote pode depender da ordem de import.

CONTEXTO. Uma varredura de AST sobre `src/guaraci` acusa 14 ciclos de
import. Ela conta TODO `from guaraci.x import y`, inclusive os que estao
dentro de funcoes ou sob `TYPE_CHECKING` -- que sao justamente a tecnica
usada para QUEBRAR ciclo, nao para cria-lo. Considerando so' os imports de
nivel de modulo, os ciclos sao **zero** (medido em 2026-08-18; retifica a
contagem de 14 que a auditoria da vespera reportou como divida).

Este teste trava a propriedade que de fato importa: importar qualquer
modulo PRIMEIRO, num interpretador limpo, funciona. E' a unica forma de
detectar ciclo real -- num processo unico, o primeiro import povoa
`sys.modules` e mascara o problema para todos os seguintes.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

_RAIZ = pathlib.Path(__file__).resolve().parents[1] / "src" / "guaraci"
_MODULOS = sorted(p.stem for p in _RAIZ.glob("*.py") if p.stem != "__init__")


@pytest.mark.parametrize("modulo", _MODULOS)
def test_modulo_importa_primeiro_em_processo_limpo(modulo):
    """Cada modulo, sozinho, num interpretador que ainda nao viu o pacote."""
    src = str(_RAIZ.parent)
    r = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {src!r}); import guaraci.{modulo}"],
        capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, (
        f"guaraci.{modulo} nao importa sozinho:\n{r.stderr[-1500:]}")


def test_nenhum_ciclo_entre_imports_de_NIVEL_DE_MODULO():
    """Ciclo entre imports de topo trava o interpretador; entre imports
    locais, nao. Este teste conta apenas os primeiros.

    Se ele falhar, um `from guaraci.x import y` novo subiu para o topo de um
    modulo que `x` tambem importa -- mova-o para dentro da funcao que o usa,
    ou para um bloco `if TYPE_CHECKING:` se for so' anotacao.
    """
    grafo: dict[str, set[str]] = {}
    for arquivo in _RAIZ.rglob("*.py"):
        alvos = set()
        arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
        for no in arvore.body:                       # SO' o nivel de modulo
            if isinstance(no, ast.ImportFrom) and no.module \
                    and "guaraci" in no.module:
                destino = no.module.split(".")[-1]
                if destino != arquivo.stem:
                    alvos.add(destino)
        grafo[arquivo.stem] = alvos

    encontrados: set[tuple] = set()

    def _dfs(no: str, caminho: list[str], visitados: set[str]) -> None:
        for vizinho in grafo.get(no, ()):
            if vizinho in caminho:
                ciclo = caminho[caminho.index(vizinho):] + [vizinho]
                encontrados.add(tuple(sorted(set(ciclo))))
            elif vizinho not in visitados:
                visitados.add(vizinho)
                _dfs(vizinho, caminho + [vizinho], visitados)

    for no in list(grafo):
        _dfs(no, [no], set())

    assert not encontrados, (
        f"ciclo(s) entre imports de nivel de modulo: "
        f"{[' <-> '.join(c) for c in sorted(encontrados)]}")
