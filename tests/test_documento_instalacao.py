"""O guia de instalação cobre os ambientes reais e não promete o que não foi testado (V37)."""
from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
GUIA = RAIZ / "docs" / "usuario" / "instalacao.md"
TEXTO = GUIA.read_text(encoding="utf-8")


def test_guia_existe_e_esta_no_indice_dos_documentos():
    assert GUIA.is_file()
    assert "docs/usuario/instalacao.md" in (RAIZ / "docs/projeto/INDICE.md").read_text(encoding="utf-8")


def test_cobre_windows_macos_linux_wsl_e_diz_que_macos_nao_foi_usado():
    for secao in ("## Windows", "## macOS", "## Linux", "## WSL", "## Requisitos mínimos"):
        assert secao in TEXTO, secao
    # O macOS so' tem testes automaticos; o guia diz que nao houve uso real la'.
    assert "macOS: verificado só pelos testes automáticos" in TEXTO and "ninguém usou o programa de verdade lá" in TEXTO
    assert "brew install libomp" in TEXTO                          # xgboost do .[all] no Mac


def test_trata_os_tropecos_comuns():
    assert "Set-ExecutionPolicy -Scope Process" in TEXTO           # política de execução do PowerShell
    assert "python3.12-venv" in TEXTO and "python -m guaraci" in TEXTO                              # venv separado no Debian/Ubuntu
    assert "python3-tk" in TEXTO and "Tk não é necessário" in TEXTO
    assert '".[web]"' in TEXTO                                     # aspas, por causa do zsh
    assert "4 GB" in TEXTO and "2 núcleos" in TEXTO                # requisitos do cenário típico
    assert "GUARACI_TERMINAL=claro" in TEXTO                       # fundo claro do terminal


def test_nao_promete_pacote_publicado_nem_numeros_que_envelhecem():
    assert "Ainda não há pacote publicado no PyPI" in TEXTO
    assert "pip install guaraci" not in TEXTO
    assert "vistoria" not in TEXTO.lower()

