# -*- coding: utf-8 -*-
"""Nenhum texto do produto manda instalar por nome pelo PyPI (o pacote nao esta no PyPI e o nome `guaraci` e' de outro
projeto: quem registrasse `guaraci-chemometrics` depois faria o usuario rodar codigo de terceiro). A instalacao e'
sempre a partir do clone (`pip install ".[extra]"`)."""
from __future__ import annotations

import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PADRAO = re.compile(r"pip install\s+[\"'`]?guaraci(-chemometrics)?(\[|\b)", re.I)
PERMITIDO = ("Do not run `pip install guaraci`", "Não rode `pip install guaraci`", "pip install guaraci")


def _arquivos():
    for sub in ("src/guaraci",):
        for p in (RAIZ / sub).rglob("*"):
            if p.suffix in {".py", ".md"} and "__pycache__" not in p.parts:
                yield p
    for nome in ("README.md", "README.pt-BR.md", "MANUAL.md", "app_quimiometria.py", "CHANGELOG.md"):
        yield RAIZ / nome
    yield from (RAIZ / "docs" / "usuario").glob("*.md")


def test_nenhum_texto_manda_instalar_pelo_nome_no_pypi():
    achados = []
    for p in _arquivos():
        for i, linha in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if PADRAO.search(linha) and not any(a in linha and "unrelated" in linha or a in linha and "outro projeto" in linha
                                                for a in PERMITIDO) and p.name != "CHANGELOG.md":
                achados.append(f"{p.relative_to(RAIZ)}:{i}")
    assert not achados, achados


def test_comandos_de_instalar_extras_usam_o_clone():
    from guaraci import extras_opcionais as ext
    assert ext.COMANDO_TUDO == 'pip install ".[all]"'
