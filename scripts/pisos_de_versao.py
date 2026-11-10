"""Imprime, no formato `pacote==versao`, o PISO de versao de cada dependencia
declarada no pyproject.toml (nucleo + extras). Usado pela CI para rodar
`pip-audit` sobre o MENOR conjunto que o projeto admite instalar -- e' o que
protege quem instala com pisos, nao so' quem usa o lock.

    python scripts/pisos_de_versao.py > requirements-pisos.txt
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*>=\s*([0-9][^,;\s]*)")


def pisos(pyproject: Path) -> list[str]:
    dados = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    projeto = dados["project"]
    reqs = list(projeto.get("dependencies", []))
    for extra, lista in projeto.get("optional-dependencies", {}).items():
        reqs += [r for r in lista if not r.startswith("guaraci-chemometrics")]
    saida = {}
    for r in reqs:
        if "@" in r:          # dependencia por URL (ex.: spcfile @ git+...)
            continue
        m = _RE.match(r)
        if m:
            saida[m.group(1).lower()] = m.group(2)
    return [f"{k}=={v}" for k, v in sorted(saida.items())]


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parents[1]
    print("\n".join(pisos(raiz / "pyproject.toml")))
    sys.exit(0)
