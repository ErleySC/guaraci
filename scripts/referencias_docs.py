#!/usr/bin/env python3
"""referencias_docs.py -- toda referencia a documento em TEXTO DE EXECUCAO do pacote, com o veredito `no repo | no pacote`.

    python scripts/referencias_docs.py            # tabela `origem | alvo | no repo | no pacote`
    python scripts/referencias_docs.py --verificar # sai 1 se alguma referencia nao resolve dentro do pacote instalado

Varre (via `ast`) as cadeias de caracteres de `src/guaraci/` que NAO sao docstring: o que o programa mostra ao usuario.
Comentarios e docstrings sao texto de desenvolvedor, lido no repositorio, e ficam de fora. As abas web (`app_tabs/`) so'
rodam a partir do repositorio (o app web nao faz parte do wheel) e entram na tabela como `so repo (web)`.

Uma referencia resolve no pacote quando e' `guaraci manual TOPICO|SECAO` com topico/secao existente, ou um arquivo gerado pela
execucao (`model_card.md`). Caminho de `docs/`, `MANUAL.md`, `SECURITY.md` etc. em texto de execucao NAO resolve.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

#: arquivos que a EXECUCAO gera (nao sao documentos de projeto)
ARTEFATOS_DE_SAIDA = {"model_card.md", "_model_card.md", "ficha_vazamento.md"}
_DOC = re.compile(r"(docs/[\w./-]+\.(?:md|yaml)|\b[A-Z][A-Z_]{3,}\.md\b|\bMANUAL\b|[\w/-]+\.md\b)")
_MANUAL = re.compile(r"guaraci manual ([\w.]+)")

Linha = Tuple[str, str, bool, bool]       # origem, alvo, no_repo, no_pacote


def _strings_de_execucao(arquivo: Path):
    arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
    docstrings = set()
    for n in ast.walk(arvore):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body \
                and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant):
            docstrings.add(id(n.body[0].value))
    for n in ast.walk(arvore):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings:
            yield n.lineno, n.value


def _manual_resolve(alvo: str) -> bool:
    from guaraci import docs_pacote as dp
    try:
        return alvo in dp.TOPICOS or dp.secao_do_manual(alvo) is not None
    except FileNotFoundError:
        return False


def referencias() -> List[Linha]:
    out: List[Linha] = []
    for f in sorted((RAIZ / "src" / "guaraci").rglob("*.py")):
        rel = f.relative_to(RAIZ).as_posix()
        for ln, txt in _strings_de_execucao(f):
            for alvo in _MANUAL.findall(txt):
                out.append((f"{rel}:{ln}", f"guaraci manual {alvo}", True, _manual_resolve(alvo)))
            sem_manual = _MANUAL.sub("", txt)
            for alvo in _DOC.findall(sem_manual):
                if alvo in ARTEFATOS_DE_SAIDA:
                    continue
                if (RAIZ / "src" / "guaraci" / "dados_docs" / alvo).is_file():     # nome de arquivo do proprio pacote
                    out.append((f"{rel}:{ln}", alvo, True, True))
                    continue
                existe = (RAIZ / alvo).exists() if ("/" in alvo or alvo.endswith(".md")) else (RAIZ / "MANUAL.md").exists()
                out.append((f"{rel}:{ln}", alvo, existe, False))
    return out


def quebradas() -> List[Linha]:
    """Referencias de execucao que nao resolvem no pacote instalado (as abas web ficam de fora: declaradas so' do repo)."""
    return [r for r in referencias() if not r[3] and "/app_tabs/" not in r[0]]


def main(argv: List[str] | None = None) -> int:
    ruins = quebradas()
    if "--verificar" in (argv if argv is not None else sys.argv[1:]):
        for o, a, _, _ in ruins:
            print(f"nao resolve no pacote: {o} -> {a}")
        return 1 if ruins else 0
    print("origem | alvo | no repo | no pacote")
    for o, a, r, p in referencias():
        web = "/app_tabs/" in o
        print(f"{o} | {a} | {'sim' if r else 'nao'} | {'sim' if p else ('so repo (web)' if web else 'nao')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
