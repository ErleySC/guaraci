"""Lote 6 da vistoria: autoria uniforme, licenca SPDX, avisos de terceiros e `--version` completo (V29, V54, V104, V105, V111, V119)."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import auditar_pacote_construido as apc  # noqa: E402

from guaraci.config import AUTOR_NOME, AUTOR_ORCID, LICENCA_SPDX  # noqa: E402


def _ler(rel: str) -> str:
    return (RAIZ / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# V29: --version
# ---------------------------------------------------------------------------
def test_version_mostra_versao_autor_licenca_garantia_e_pasta():
    from guaraci import __version__
    from guaraci.guaraci import _texto_versao
    texto = _texto_versao()
    assert f"GUARACI v{__version__}" in texto
    assert AUTOR_NOME in texto and AUTOR_ORCID in texto
    assert LICENCA_SPDX in texto
    assert "SEM GARANTIA" in texto
    assert str(Path(sys.modules["guaraci.guaraci"].__file__).resolve().parent) in texto


def test_version_pela_linha_de_comando_tem_a_licenca():
    r = subprocess.run([sys.executable, "-m", "guaraci", "--version"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    assert r.returncode == 0
    assert "GPL-3.0-or-later" in r.stdout and AUTOR_NOME in r.stdout and "SEM GARANTIA" in r.stdout


# ---------------------------------------------------------------------------
# V119: forma unica do nome + ORCID
# ---------------------------------------------------------------------------
_ARQUIVOS_DE_AUTORIA = ["pyproject.toml", "CITATION.cff", "README.md", "README.pt-BR.md", "paper/paper.md",
                        "src/guaraci/app_tabs/sobre.py", "src/guaraci/guaraci.py", "src/guaraci/config.py"]


@pytest.mark.parametrize("rel", _ARQUIVOS_DE_AUTORIA)
def test_nenhuma_forma_abreviada_do_nome(rel):
    assert not re.search(r"Erley S\.|Costa, Erley S\b|E\. S\. da Costa", _ler(rel))


def test_forma_unica_do_nome_em_pyproject_readme_e_citation():
    assert AUTOR_NOME == "Erley Silva da Costa"
    assert f'name = "{AUTOR_NOME}"' in _ler("pyproject.toml")
    assert f"**{AUTOR_NOME}**" in _ler("README.md") and f"**{AUTOR_NOME}**" in _ler("README.pt-BR.md")
    cff = _ler("CITATION.cff")
    assert 'given-names: "Erley Silva"' in cff and 'name-particle: "da"' in cff and 'family-names: "Costa"' in cff
    assert "Costa, Erley Silva da" in _ler("README.md")


# ---------------------------------------------------------------------------
# Copyright e licenca unica
# ---------------------------------------------------------------------------
def test_linha_de_copyright_tem_a_forma_combinada():
    from guaraci.config import COPYRIGHT_ANO_ATUAL, COPYRIGHT_ANO_INICIAL, COPYRIGHT_LINHA
    anos = str(COPYRIGHT_ANO_INICIAL) if COPYRIGHT_ANO_INICIAL == COPYRIGHT_ANO_ATUAL else f"{COPYRIGHT_ANO_INICIAL}–{COPYRIGHT_ANO_ATUAL}"
    assert COPYRIGHT_LINHA == f"Copyright (C) {anos} {AUTOR_NOME}"
    assert COPYRIGHT_ANO_INICIAL <= COPYRIGHT_ANO_ATUAL


def test_ano_inicial_do_copyright_nao_e_posterior_ao_primeiro_commit():
    """O ano inicial e' o do primeiro commit do historico (num clone raso, o commit mais antigo visivel nunca e' anterior a ele)."""
    from guaraci.config import COPYRIGHT_ANO_INICIAL
    if not (RAIZ / ".git").exists():
        return
    r = subprocess.run(["git", "log", "--reverse", "--format=%ad", "--date=format:%Y"], cwd=RAIZ, capture_output=True, text=True)
    anos = [int(x) for x in r.stdout.split() if x.isdigit()]
    assert not anos or COPYRIGHT_ANO_INICIAL <= min(anos)


def test_copyright_aparece_no_version_nos_readmes_e_nos_avisos_de_terceiros():
    from guaraci.config import COPYRIGHT_LINHA
    from guaraci.guaraci import _texto_versao
    assert COPYRIGHT_LINHA in _texto_versao()
    for rel in ("README.md", "README.pt-BR.md", "THIRD_PARTY_NOTICES.md"):
        assert COPYRIGHT_LINHA in _ler(rel), rel


@pytest.mark.parametrize("rel", ["README.md", "README.pt-BR.md", "src/guaraci/app_tabs/sobre.py", "src/guaraci/guaraci.py", "MANUAL.md"])
def test_nenhuma_oferta_de_outros_termos_de_licenca(rel):
    texto = _ler(rel).lower()
    for frase in ("other than the gplv3", "termos diferentes da gplv3", "commercial license", "licenca comercial", "licença comercial",
                  "dual licens", "dual-licens", "licenca dupla", "licença dupla"):
        assert frase not in texto, (rel, frase)


@pytest.mark.parametrize("rel", ["CITATION.cff", "README.md", "README.pt-BR.md", "paper/paper.md"])
def test_orcid_presente(rel):
    assert AUTOR_ORCID in _ler(rel)


def test_orcid_na_tela_sobre_da_web_e_da_cli():
    assert "AUTOR_ORCID" in _ler("src/guaraci/app_tabs/sobre.py") and "[ORCID]" in _ler("src/guaraci/app_tabs/sobre.py")
    assert "ORCID:" in _ler("src/guaraci/guaraci.py")


def test_citation_nao_descreve_uma_matriz_especifica():
    cff = _ler("CITATION.cff").lower()
    # literais partidos: o auditor de termos de matriz le' o proprio teste e nao deve tomar a proibicao por uso
    assert "vegetable" " oil" not in cff and "amaz" "on" not in cff


# ---------------------------------------------------------------------------
# V54 / V111: metadados do pyproject
# ---------------------------------------------------------------------------
def test_pyproject_licenca_spdx_e_classificadores():
    t = _ler("pyproject.toml")
    assert re.search(r'^license = "GPL-3.0-or-later"$', t, re.M)
    assert re.search(r'^license-files = \[.*"LICENSES/\*".*"THIRD_PARTY_NOTICES.md".*\]$', t, re.M)
    assert re.search(r"setuptools>=(7[7-9]|[89]\d)", t)          # PEP 639 exige setuptools >= 77
    assert "License ::" not in t                                   # classificador de licenca + SPDX = erro do setuptools
    assert "Development Status :: 4 - Beta" in t and "Production/Stable" not in t
    assert "Operating System :: Microsoft :: Windows" in t and "Operating System :: POSIX :: Linux" in t
    assert "OS Independent" not in t


# ---------------------------------------------------------------------------
# V104 / V105: avisos de licenca de terceiros
# ---------------------------------------------------------------------------
def test_texto_bsd3_do_specio_esta_integral():
    t = _ler("LICENSES/BSD-3-Clause-specio.txt")
    assert "Guillaume Lemaitre" in t
    for trecho in ("Redistributions of source code must retain", "Redistributions in binary form must reproduce",
                   "Neither the name of the copyright holder", 'PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"'):
        assert trecho in t


def test_cabecalho_do_leitor_sp_aponta_para_os_avisos():
    t = _ler("src/guaraci/importadores_proprietarios.py")
    assert "LICENSES/BSD-3-Clause-specio.txt" in t and "THIRD_PARTY_NOTICES.md" in t
    assert "BSD-3-Clause-specio" in _ler("THIRD_PARTY_NOTICES.md") and "dual.spls" in _ler("THIRD_PARTY_NOTICES.md")


def test_licenca_mit_do_oraculo_dual_spls_esta_completa():
    t = _ler("tests/fixtures/dual_spls_oracle/LICENSE.dual.spls")
    assert t.startswith("MIT License")
    assert "Permission is hereby granted, free of charge" in t and "WITHOUT WARRANTY OF ANY KIND" in t
    assert "2021" in t and "Louna Alsouki" in t


# ---------------------------------------------------------------------------
# V104 / V111 no pacote de verdade (constroi wheel + sdist)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_wheel_construido_tem_licenca_spdx_e_avisos_de_terceiros():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d, "dist")
        assert apc.construir(out) == 0
        whl = next(out.glob("*.whl"))
        with zipfile.ZipFile(whl) as z:
            nomes = z.namelist()
            meta = next(z.read(n).decode("utf-8") for n in nomes if n.endswith(".dist-info/METADATA"))
        assert "License-Expression: GPL-3.0-or-later" in meta.splitlines()
        assert "Classifier: License ::" not in meta and "Classifier: Development Status :: 4 - Beta" in meta.splitlines()
        for exigido in apc.LICENCAS_EXIGIDAS_NO_WHEEL:
            assert any(n.endswith("/licenses/" + exigido) for n in nomes), exigido
