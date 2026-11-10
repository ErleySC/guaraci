"""Dados bibliograficos do paper (lote 8b: V100).
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


# ── V100: bibliografia ───────────────────────────────────────────────────────

def _entradas_bib() -> dict[str, tuple[str, str]]:
    bib = _le("paper/paper.bib")
    saida: dict[str, tuple[str, str]] = {}
    for m in re.finditer(r"^@(\w+)\{([^,]+),(.*?)^\}", bib, re.M | re.S):
        saida[m.group(2)] = (m.group(1).lower(), m.group(3))
    return saida


def test_bib_tipo_de_entrada_confere_com_os_campos():
    for chave, (tipo, corpo) in _entradas_bib().items():
        if "booktitle" in corpo:
            assert tipo == "inproceedings", f"{chave}: booktitle exige @inproceedings"
        if re.search(r"^\s*journal\s*=", corpo, re.M):
            assert tipo == "article", chave


def test_bib_rajalahti_com_as_paginas_do_registro_oficial():
    """Crossref 10.1016/j.chemolab.2008.08.004: vol 95, no 1, pp 35-48."""
    corpo = _entradas_bib()["Rajalahti2009"][1]
    assert re.search(r"pages\s*=\s*\{35--48\}", corpo)


def test_dd_simca_citado_pela_fonte_do_metodo():
    """O DD-SIMCA vem de Pomerantsev 2008 (cem.1147) e do tutorial de Kucheryavskiy
    et al. 2024 (cem.3556), nao do artigo sobre objetos extremos."""
    bib = _le("paper/paper.bib")
    assert "10.1002/cem.1147" in bib and "10.1002/cem.3556" in bib
    resumo = _secao("Summary")
    assert "DD-SIMCA" in resumo
    trecho = resumo.split("DD-SIMCA", 1)[1][:160]
    assert "Pomerantsev2008" in trecho and "Kucheryavskiy2024" in trecho
