"""Coerencia do paper com a CI (lote 8b: V99; V116 so' o registro).
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


# ── V99: macOS ───────────────────────────────────────────────────────────────

def test_paper_nao_promete_macos_que_o_readme_nega():
    """O README diz que macOS nao e' suportado/verificado; o paper nao pode listar
    macOS entre as plataformas em que a CI/os testes dao garantia."""
    readme = " ".join(_le("README.md").split())
    assert "macOS is not a supported or verified platform" in readme
    projeto = _secao("Software design")
    assert not re.search(r"Linux,\s*Windows\s+and\s+macOS", projeto)
    for frase in re.split(r"(?<=[.])\s+", projeto):
        if "macOS" in frase:
            assert "not a supported or verified platform" in frase, frase


def test_paper_so_cita_na_ci_o_que_o_workflow_roda():
    wf = _le(".github/workflows/test.yml")
    projeto = _secao("Software design")
    for so, rotulo in (("ubuntu", "Linux"), ("windows", "Windows"), ("macos", "macOS")):
        no_workflow = f"{so}-latest" in wf
        if rotulo in projeto and rotulo != "macOS":
            assert no_workflow, f"paper cita {rotulo} mas o workflow nao roda {so}"




# ── V116 (do autor): nenhum numero de commits escrito a mao fora da declaracao ──

def test_paper_sem_contagem_de_commits_fora_da_declaracao_de_ia():
    txt = _paper_corpo()
    fora = txt.split("# AI usage disclosure", 1)[0] + txt.split("# Acknowledgements", 1)[1]
    assert not re.search(r"\b\d+\s+(of\s+\d+\s+)?commits?\b", fora, re.I)
