#!/usr/bin/env python3
"""verificar_links_docs.py -- todo caminho e todo link citado num documento versionado aponta para um arquivo versionado.

    python scripts/verificar_links_docs.py [--json]

Varre os documentos de entrada (README, MANUAL, CHANGELOG...) e todo `.md` versionado em `docs/` atras de (a) caminhos escritos em
texto (`src/guaraci/...`, `tests/...`, `scripts/...`, `docs/...`, `paper/...`) e (b) links `[texto](alvo)`. Imprime documento e caminho
quebrado; codigo de saida 1 se houver algum e 2 se nao achar documento nenhum (nada verificado nao e' "0 quebrados"). Fonte unica: `tests/test_links_docs.py` e `scripts/estado_do_projeto.py` importam daqui.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

RAIZ = Path(__file__).resolve().parents[1]

RAIZ_MD = ["README.md", "README.pt-BR.md", "CONTRIBUTING.md", "MANUAL.md", "CHANGELOG.md", "SECURITY.md", "CODE_OF_CONDUCT.md", "SUPPORT.md",
           "GOVERNANCE.md"]
#: O mapa de migracao existe justamente para citar caminhos que nao existem mais.
DOCS_QUE_CITAM_CAMINHO_ANTIGO = {"docs/projeto/MAPA_DE_MIGRACAO.md"}
EXTENSOES = ("py", "md", "yaml", "yml", "json", "sh", "csv", "toml", "cff", "txt")
#: Copiados da raiz pelo job de docs do CI (nao sao versionados em docs/).
ISENTOS = {"docs/SECURITY.md", "docs/MANUAL.md", "docs/CHANGELOG.md"}
_RX = re.compile(r"(?<![\w/.-])((?:src/guaraci|tests|scripts|docs|paper)/[\w./-]*[\w])")
_RX_LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def versionados(raiz: Path = RAIZ) -> Set[str]:
    r = subprocess.run(["git", "ls-files"], cwd=raiz, capture_output=True, text=True, encoding="utf-8")
    return set(r.stdout.split()) if r.returncode == 0 else set()


def documentos_vivos(v: Set[str]) -> List[str]:
    return [f for f in RAIZ_MD if f in v] + sorted(f for f in v if f.startswith("docs/") and f.endswith(".md")
                                                     and f not in DOCS_QUE_CITAM_CAMINHO_ANTIGO)


def caminhos_quebrados(texto: str, doc: str, v: Set[str]) -> List[str]:
    ruins: Set[str] = set()
    for m in _RX.finditer(texto):
        p = m.group(1).rstrip(".")
        if p.rsplit(".", 1)[-1] not in EXTENSOES or p in ISENTOS or "*" in p:
            continue
        if p not in v:
            ruins.add(p)
    base = Path(doc).parent
    pastas = {str(Path(f).parent.as_posix()) for f in v}
    for m in _RX_LINK.finditer(texto):
        alvo = m.group(1)
        if re.match(r"^[a-z]+:", alvo) or alvo.startswith("#"):
            continue
        partes: List[str] = []
        for seg in (base / alvo).as_posix().split("/"):
            if seg == "..":
                if partes:
                    partes.pop()
            elif seg not in ("", "."):
                partes.append(seg)
        resolvido = "/".join(partes)
        if resolvido and resolvido not in v and resolvido.rstrip("/") not in pastas and resolvido not in ISENTOS and not alvo.startswith("/"):
            ruins.add(resolvido)
    return sorted(ruins)


class SemDocumentos(RuntimeError):
    """Nenhum documento encontrado: sem `git` ou fora de um clone, a varredura nao viu nada e "0 links quebrados" seria falso."""


def quebrados_na_arvore(raiz: Path = RAIZ, v: Optional[Set[str]] = None) -> Dict[str, List[str]]:
    v = versionados(raiz) if v is None else v
    docs = documentos_vivos(v)
    if not docs:
        raise SemDocumentos("nenhum documento versionado encontrado (git ausente ou pasta que nao e' um clone): nada foi verificado")
    out: Dict[str, List[str]] = {}
    for doc in docs:
        ruins = caminhos_quebrados((raiz / doc).read_text(encoding="utf-8", errors="replace"), doc, v)
        if ruins:
            out[doc] = ruins
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        verificados = len(documentos_vivos(versionados()))
        q = quebrados_na_arvore()
    except SemDocumentos as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 2
    total = sum(len(x) for x in q.values())
    if a.json:
        print(json.dumps({"verificados": verificados, "quebrados": q, "total": total}, ensure_ascii=False, indent=2))
    else:
        for doc, ruins in sorted(q.items()):
            print(f"{doc}: {', '.join(ruins)}")
        print(f"{verificados} documento(s) verificado(s); links quebrados: {total} em {len(q)} documento(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
