#!/usr/bin/env python3
"""padronizar_cabecalhos.py -- todo documento versionado abre com: titulo, publico, estado (gerado/manual) e "atualizado em".

    python scripts/padronizar_cabecalhos.py             # insere o cabecalho nos documentos que nao o tem
    python scripts/padronizar_cabecalhos.py --verificar # sai 1 e lista os documentos sem cabecalho

O cabecalho e' UMA linha logo depois do titulo (H1):
    **Público:** <quem le> · **Estado:** manual|gerado ... · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`
O hash verificado vive num so' lugar (o estado do projeto, gerado), nao em cada documento: assim o carimbo nunca fica velho nem exige um
commit extra por documento. `docs/api/` (stubs do mkdocstrings) e arquivos de licenca ficam de fora.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

RAIZ = Path(__file__).resolve().parents[1]
ATUALIZADO = "**Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`"
RX_CABECALHO = re.compile(r"\*\*P[úu]blico:?\*\*|\*\*P[úu]blico\.\*\*")

ENTRADA = ["README.md", "README.pt-BR.md", "MANUAL.md", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md", "CODE_OF_CONDUCT.md", "ACKNOWLEDGMENTS.md", "SUPPORT.md",
           "GOVERNANCE.md"]
#: Documentos que abrem a vitrine do repositorio e por isso NAO levam a linha interna (publico/estado): o README em ingles (V38).
SEM_CABECALHO = {"README.md"}
PUBLICO = {"usuario": "quem usa ou testa o programa", "cientifico": "quem avalia os métodos e os números", "contribuir": "quem mantém ou contribui com o código",
           "projeto": "quem decide o rumo do projeto", "qualidade": "quem confere o que foi verificado"}


def alvos(arquivos: List[str]) -> List[str]:
    docs = [f for f in arquivos if f not in SEM_CABECALHO and (f in ENTRADA or (f.startswith("docs/") and f.endswith(".md") and not f.startswith("docs/api/")))]
    return sorted(docs)


def cabecalho_padrao(doc: str) -> str:
    if doc in ENTRADA or doc == "docs/index.md":
        publico = "todos"
    elif doc == "docs/POLITICA_DE_DOCUMENTACAO.md":
        publico = "quem escreve ou revisa documentação"
    else:
        publico = PUBLICO.get(doc.split("/")[1], "todos")
    gerado = doc.startswith("docs/qualidade/")
    estado = "gerado por script (não editar à mão)" if gerado else "manual"
    return f"**Público:** {publico} · **Estado:** {estado} · {ATUALIZADO}"


def tem_cabecalho(texto: str) -> bool:
    """O cabecalho esta ate 5 linhas depois do primeiro titulo (H1)."""
    linhas = texto.splitlines()
    h1 = next((i for i, ln in enumerate(linhas) if ln.startswith("# ")), 0)
    return bool(RX_CABECALHO.search(chr(10).join(linhas[h1:h1 + 6])))


def inserir(texto: str, doc: str) -> Optional[str]:
    """Texto com o cabecalho logo depois do H1, ou None se ja tem (ou nao ha H1)."""
    if tem_cabecalho(texto):
        return None
    linhas = texto.split("\n")
    for i, ln in enumerate(linhas):
        if ln.startswith("# "):
            return "\n".join(linhas[:i + 1] + ["", cabecalho_padrao(doc)] + ([""] if i + 1 < len(linhas) and linhas[i + 1].strip() else []) + linhas[i + 1:])
    return None


def sem_cabecalho(raiz: Path = RAIZ) -> List[Tuple[str, bool]]:
    r = subprocess.run(["git", "ls-files"], cwd=raiz, capture_output=True, text=True, encoding="utf-8")
    out = []
    for doc in alvos(r.stdout.split()):
        t = (raiz / doc).read_bytes().decode("utf-8", errors="replace").replace("\r\n", "\n")
        if not tem_cabecalho(t):
            out.append((doc, inserir(t, doc) is not None))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verificar", action="store_true")
    a = ap.parse_args(argv)
    faltam = sem_cabecalho()
    if a.verificar:
        for d, _ok in faltam:
            print(f"sem cabecalho: {d}")
        return 1 if faltam else 0
    for d, _ok in faltam:
        p = RAIZ / d
        raw = p.read_bytes().decode("utf-8")
        crlf = "\r\n" in raw
        novo = inserir(raw.replace("\r\n", "\n"), d)
        if novo is not None:
            p.write_bytes((novo.replace("\n", "\r\n") if crlf else novo).encode("utf-8"))
            print(f"cabecalho inserido: {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
