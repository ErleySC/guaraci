"""'State of the field' do paper (lote 8b: V122), conforme a matriz comparativa com fonte e data.
"""
from __future__ import annotations

import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def _le(rel: str) -> str:
    return (RAIZ / rel).read_text(encoding="utf-8")


def _paper_corpo() -> str:
    return re.sub(r"<!--.*?-->", "", _le("paper/paper.md"), flags=re.S)


def _secao(nome: str) -> str:
    txt = _paper_corpo()
    m = re.search(rf"^# {nome}\s*$(.*?)(?=^# |\Z)", txt, re.M | re.S)
    assert m, nome
    return " ".join(m.group(1).split())


# ── V122: State of the field ─────────────────────────────────────────────────

def test_state_of_the_field_descreve_os_pares_com_os_recursos_documentados():
    campo = _secao("State of the field")
    mdatools = campo.split("mdatools", 1)[1].split("chemotools", 1)[0]
    for termo in ("DD-SIMCA", "MCR", "iPLS", "Procrustes"):
        assert termo in mdatools, f"mdatools sem {termo}"
    assert "web" in mdatools
    chemotools = campo.split("chemotools", 1)[1][:600]
    assert "DS" in chemotools and "PDS" in chemotools


def test_state_of_the_field_sem_superlativos_nem_afirmacao_de_superioridade():
    campo = _secao("State of the field").lower()
    for proibido in ("best", "superior", "outperform", "better than", "unique", "only tool"):
        assert proibido not in campo, proibido


def test_state_of_the_field_posiciona_o_guaraci_pelo_que_oferece():
    campo = _secao("State of the field")
    for termo in ("report", "conformal", "grouped"):
        assert termo in campo.lower(), termo


