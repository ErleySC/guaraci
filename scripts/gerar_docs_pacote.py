#!/usr/bin/env python3
"""gerar_docs_pacote.py -- copia para `src/guaraci/dados_docs/` os documentos que o programa cita ao usuario.

    python scripts/gerar_docs_pacote.py             # grava as copias
    python scripts/gerar_docs_pacote.py --verificar # sai 1 se alguma copia estiver defasada ou faltar

O wheel nao leva `docs/` nem `MANUAL.md`; a CLI (`guaraci manual`, assistente, ajuda) le estas copias. A fonte de verdade
continua sendo o arquivo do repositorio: a copia e' sempre regerada, nunca editada.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "src" / "guaraci" / "dados_docs"

#: arquivo no pacote -> fonte no repositorio. Deve casar com `guaraci.docs_pacote.TOPICOS`.
FONTES: Dict[str, str] = {
    "manual.md": "MANUAL.md",
    "seguranca.md": "SECURITY.md",
    "limitacoes.md": "docs/cientifico/limitacoes.md",
    "suporte.md": "SUPPORT.md",
    "privacidade.md": "docs/usuario/privacidade.md",
    "termos_de_uso.md": "docs/usuario/termos_de_uso.md",
}


def conteudo_esperado(nome: str) -> str:
    return (RAIZ / FONTES[nome]).read_text(encoding="utf-8")


def defasados() -> List[str]:
    """Nomes cuja copia falta ou difere da fonte."""
    ruins = []
    for nome in FONTES:
        alvo = DESTINO / nome
        if not alvo.is_file() or alvo.read_text(encoding="utf-8") != conteudo_esperado(nome):
            ruins.append(nome)
    return ruins


def gravar() -> List[str]:
    DESTINO.mkdir(parents=True, exist_ok=True)
    mudou = []
    for nome in defasados():
        (DESTINO / nome).write_text(conteudo_esperado(nome), encoding="utf-8", newline="\n")
        mudou.append(nome)
    return mudou


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verificar", action="store_true")
    a = ap.parse_args(argv)
    if a.verificar:
        ruins = defasados()
        for n in ruins:
            print(f"defasado ou ausente: dados_docs/{n} (fonte {FONTES[n]})")
        return 1 if ruins else 0
    for n in gravar():
        print(f"gravado: dados_docs/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
