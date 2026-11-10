#!/usr/bin/env python3
"""cobertura_excecoes.py -- toda exclusao de cobertura precisa de uma entrada no registro, com motivo e categoria.

    python scripts/cobertura_excecoes.py             # lista o que o codigo exclui e o que o registro declara
    python scripts/cobertura_excecoes.py --verificar # sai 1 se houver exclusao sem registro ou registro sem exclusao

Exclusao = linha de `src/guaraci/` que o coverage exclui por padrao (`# pragma: no cover`, `if TYPE_CHECKING:`, corpo de stub `...`)
ou `omit`/`exclude_lines` na configuracao de cobertura do `pyproject.toml`. O registro
(`tests/cobertura_excecoes.json`) declara cada uma como `arquivo | funcao | quantidade | motivo | categoria`. O numero de exclusoes
nao pode crescer sem entrada no registro (e o registro nao pode manter entrada de uma exclusao que ja saiu).

Categorias: A = coberto por teste (nao deveria estar aqui), B = codigo inalcancavel (deveria ser removido), C = depende de plataforma ou
de dependencia opcional (injecao de ambiente nao alcanca), D = interface web sem teste possivel.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

RAIZ = Path(__file__).resolve().parents[1]
REGISTRO = RAIZ / "tests" / "cobertura_excecoes.json"
CATEGORIAS = ("A", "B", "C", "D")
# os padroes que o coverage (7.10+) exclui sem configuracao nenhuma; o teste confere que nenhum padrao do coverage instalado fica de fora
PADROES_DO_COVERAGE = (r"#\s*(pragma|PRAGMA)[:\s]?\s*(no|NO)\s*(cover|COVER)",
                       r"^\s*(((async )?def .*?)?[\])]+(\s*->.*?)?:\s*)?\.\.\.\s*(#|$)",
                       r"if (typing\.)?TYPE_CHECKING:",
                       r"^\s*if (typing\.)?TYPE_CHECKING:")   # grafia do coverage 7.17+ (mesmo efeito, outra regex)
_PRAGMA = re.compile("|".join(f"(?:{p})" for p in PADROES_DO_COVERAGE + (r"#\s*pragma:\s*no\s*cover",)), re.I)

Chave = Tuple[str, str]


def _funcao_que_contem(arvore: ast.AST, linha: int) -> str:
    melhor = None
    for no in ast.walk(arvore):
        if isinstance(no, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and no.lineno <= linha <= (no.end_lineno or no.lineno):
            if melhor is None or no.lineno >= melhor.lineno:
                melhor = no
    return melhor.name if melhor else "<modulo>"


def pragmas_no_codigo(raiz: Path = RAIZ) -> Dict[Chave, int]:
    """{(arquivo, funcao): quantidade} de linhas de src/guaraci que o coverage exclui (uma por linha, mesmo com dois padroes)."""
    achados: Counter = Counter()
    for caminho in sorted((raiz / "src" / "guaraci").rglob("*.py")):
        texto = caminho.read_text(encoding="utf-8")
        if not _PRAGMA.search(texto):
            continue
        arvore = ast.parse(texto)
        rel = caminho.relative_to(raiz).as_posix()
        for n, linha in enumerate(texto.splitlines(), 1):
            if _PRAGMA.search(linha):
                achados[(rel, _funcao_que_contem(arvore, n))] += 1
    return dict(achados)


def linhas_excluidas(raiz: Path = RAIZ) -> int:
    """Linhas de src/guaraci que o coverage tira da conta (a linha-gatilho e o corpo do bloco), pelo parser do proprio coverage."""
    from coverage.parser import PythonParser
    padrao = "|".join(f"(?:{p})" for p in PADROES_DO_COVERAGE)
    total = 0
    for caminho in sorted((raiz / "src" / "guaraci").rglob("*.py")):
        parser = PythonParser(filename=str(caminho), exclude=padrao)
        parser.parse_source()
        total += len(parser.excluded)
    return total


def frase_de_escopo(n: int) -> str:
    """Como os documentos versionados afirmam a cobertura: escopo, exclusoes e o que fica fora."""
    return (f"src/guaraci, com {n} linhas excluídas e registradas (≈99,90 % sem exclusões); "
            "app_quimiometria.py e scripts fora do escopo")


def exclusoes_no_pyproject(raiz: Path = RAIZ) -> List[str]:
    """`omit`, `exclude_lines` e `exclude_also` da configuracao de cobertura (vazio quando nao ha)."""
    try:
        import tomllib
    except ModuleNotFoundError:     # Python 3.10
        import tomli as tomllib     # type: ignore[no-redef]
    dados = tomllib.loads((raiz / "pyproject.toml").read_text(encoding="utf-8"))
    cobertura = dados.get("tool", {}).get("coverage", {})
    out: List[str] = []
    for secao in ("run", "report"):
        for campo in ("omit", "exclude_lines", "exclude_also"):
            for item in cobertura.get(secao, {}).get(campo, []) or []:
                out.append(f"{secao}.{campo}: {item}")
    return out


def carregar_registro(caminho: Path = REGISTRO) -> Dict:
    return json.loads(caminho.read_text(encoding="utf-8"))


def comparar(encontrados: Dict[Chave, int], pyproject: List[str], registro: Dict) -> Dict[str, List[str]]:
    """Diferencas entre o que o codigo exclui e o que o registro declara."""
    declarados: Dict[Chave, int] = {}
    for e in registro.get("pragmas", []):
        declarados[(e["arquivo"], e["funcao"])] = int(e.get("quantidade", 1))
    sem_registro = [f"{a}:{f} ({n} pragma(s), registrado {declarados.get((a, f), 0)})"
                    for (a, f), n in sorted(encontrados.items()) if n > declarados.get((a, f), 0)]
    obsoletas = [f"{a}:{f} (registrado {n}, no codigo {encontrados.get((a, f), 0)})"
                 for (a, f), n in sorted(declarados.items()) if n > encontrados.get((a, f), 0)]
    declarado_py = list(registro.get("exclusoes_pyproject", []))
    py_novas = [x for x in pyproject if x not in declarado_py]
    py_obsoletas = [x for x in declarado_py if x not in pyproject]
    incompletas = []
    for e in registro.get("pragmas", []):
        if not str(e.get("motivo", "")).strip() or e.get("categoria") not in CATEGORIAS:
            incompletas.append(f"{e.get('arquivo')}:{e.get('funcao')} (motivo vazio ou categoria fora de {CATEGORIAS})")
    return {"sem_registro": sem_registro + [f"pyproject: {x}" for x in py_novas],
            "obsoletas": obsoletas + [f"pyproject: {x}" for x in py_obsoletas],
            "incompletas": incompletas}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verificar", action="store_true")
    ap.parse_args(argv)
    achados, py = pragmas_no_codigo(), exclusoes_no_pyproject()
    d = comparar(achados, py, carregar_registro())
    print(f"{sum(achados.values())} linha(s) excluida(s) no codigo; {len(py)} exclusao(oes) na configuracao de cobertura.")
    for rotulo, itens in d.items():
        for item in itens:
            print(f"{rotulo}: {item}")
    return 1 if any(d.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
