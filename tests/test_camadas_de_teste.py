"""Um comando por camada de teste: a linha de comando de cada camada e o numero de processos."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))

import camadas_de_teste as ct  # noqa: E402


@pytest.mark.parametrize("nucleos, esperado", [(1, 1), (2, 1), (4, 2), (8, 4), (32, 4), (None, ct.workers(None))])
def test_processos_sao_metade_dos_nucleos_no_maximo_quatro(nucleos, esperado):
    assert ct.workers(nucleos) == esperado >= 1


def test_camada_rapida_exclui_slow_e_nao_mede_cobertura():
    cmd = ct.comando("rapido", 4, "base")
    assert "not slow" in cmd and not any(a.startswith("--cov") for a in cmd) and cmd[cmd.index("-n") + 1] == "4"


def test_camada_de_cobertura_inclui_slow_e_mede_ramos_com_relatorio_xml():
    cmd = ct.comando("cobertura", 2, "base", "saida.xml")
    assert "not slow" not in cmd and "--cov-branch" in cmd and "--cov-report=xml:saida.xml" in cmd


def test_camada_completa_inclui_slow_e_nao_mede_cobertura():
    cmd = ct.comando("completo", 4, "base")
    assert "not slow" not in cmd and not any(a.startswith("--cov") for a in cmd)


def test_um_processo_so_nao_usa_xdist_e_extras_vao_ao_pytest():
    cmd = ct.comando("rapido", 1, "base", extra=["tests/test_x.py", "-k", "nome"])
    assert "-n" not in cmd and cmd[-3:] == ["tests/test_x.py", "-k", "nome"] and "--basetemp=base" in cmd


def test_camada_desconhecida_e_recusada():
    with pytest.raises(ValueError, match="camada desconhecida"):
        ct.comando("lenta", 2, "base")


def test_resumo_da_cobertura_le_o_xml_no_formato_do_verificador(tmp_path):
    xml = tmp_path / "c.xml"
    xml.write_text('<coverage><packages><package><classes><class filename="src/guaraci/pipeline.py"><lines>'
                   '<line number="1" hits="1"/><line number="2" hits="0" branch="true" condition-coverage="50% (1/2)"/></lines></class></classes></package></packages></coverage>',
                   encoding="utf-8")
    texto = ct.resumo_da_cobertura(xml)
    assert texto.startswith("global:") and "nucleo:" in texto
