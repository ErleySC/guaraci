# -*- coding: utf-8 -*-
"""Nenhum texto do produto (codigo, scripts, documentos) cita um `scripts/<nome>.py` que nao existe.

Achado da rodada de fechamento: o docstring de `scripts/privacidade_amostras.py` citava um gerador que ja nao existia; o mesmo
acontece quando uma ferramenta sai do repositorio (por exemplo, para o ferramental privado) e fica uma remissao para tras.
"""
from __future__ import annotations

import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RX = re.compile(r"(?<![\w/.-])scripts/([A-Za-z0-9_]+\.py)\b")
#: o historico de versoes cita scripts de epocas passadas; os resultados gerados citam o que rodou naquele dia.
ISENTOS = {"CHANGELOG.md"}


def _textos():
    for pasta in ("scripts", "src/guaraci", "docs", "paper", ".github"):
        for p in (RAIZ / pasta).rglob("*"):
            if p.is_file() and p.suffix in {".py", ".md", ".yml", ".yaml", ".toml", ".sh"} and "__pycache__" not in p.parts:
                if p.name.startswith("resultado_"):
                    continue
                yield p
    for nome in ("README.md", "README.pt-BR.md", "CONTRIBUTING.md", "MANUAL.md", "SECURITY.md", "SUPPORT.md", "GOVERNANCE.md", "pyproject.toml"):
        if (RAIZ / nome).exists():
            yield RAIZ / nome


def test_nenhuma_remissao_a_script_inexistente():
    ruins = {}
    for p in _textos():
        if p.name in ISENTOS:
            continue
        try:
            texto = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for m in RX.finditer(texto):
            if not (RAIZ / "scripts" / m.group(1)).exists():
                ruins.setdefault(str(p.relative_to(RAIZ)).replace("\\", "/"), set()).add(m.group(1))
    assert ruins == {}, {k: sorted(v) for k, v in ruins.items()}
