"""Gerador de `docs/qualidade/cobertura_lacunas.md`: leitura do coverage.xml, camadas, faixas de linha e classificacao das lacunas."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))

import gerar_cobertura_lacunas as g  # noqa: E402

_XML = """<coverage><packages><package><classes>
<class filename="src/guaraci/pipeline.py"><lines><line number="1" hits="1"/><line number="2" hits="0"/><line number="3" hits="1" branch="true" condition-coverage="50% (1/2)"/></lines></class>
<class filename="src/guaraci/app_tabs/inicio.py"><lines><line number="1" hits="1"/></lines></class>
<class filename="src/guaraci/guaraci.py"><lines><line number="1" hits="0"/></lines></class>
<class filename="src/guaraci/figuras.py"><lines><line number="1" hits="1"/></lines></class>
</classes></package></packages></coverage>"""


@pytest.fixture
def xml(tmp_path):
    p = tmp_path / "c.xml"
    p.write_text(_XML, encoding="utf-8")
    return p


def test_leitura_do_xml_conta_linhas_ramos_e_o_que_falta(xml):
    d = g.ler_xml(xml)
    assert d["pipeline.py"]["linhas"] == 3 and d["pipeline.py"]["cobertas"] == 2
    assert d["pipeline.py"]["ramos"] == 2 and d["pipeline.py"]["ramos_cobertos"] == 1
    assert d["pipeline.py"]["linhas_perdidas"] == [2] and d["pipeline.py"]["linhas_de_ramo_perdidas"] == [3]


def test_camadas_separam_nucleo_web_cli_e_resto(xml):
    nucleo = {"pipeline.py"}
    assert [g.camada(a, nucleo) for a in ("pipeline.py", "app_tabs/inicio.py", "guaraci.py", "figuras.py")] == ["nucleo", "web", "cli", "resto"]
    agg = g.agregar(g.ler_xml(xml), nucleo)
    assert agg["nucleo"]["linhas"] == 66.67 and agg["nucleo"]["ramos"] == 50.0 and agg["web"]["combinada"] == 100.0
    assert agg["global"]["linhas"] == pytest.approx(100 * 4 / 6, abs=0.01)


@pytest.mark.parametrize("numeros, esperado", [([], "—"), ([5], "5"), ([1, 2, 3, 7], "1–3, 7"), ([9, 3, 4, 4], "3–4, 9")])
def test_faixas_de_linha_agrupam_sequencias(numeros, esperado):
    assert g._faixas(numeros) == esperado


def test_o_documento_marca_lacuna_sem_registro_e_usa_o_motivo_registrado(xml, tmp_path):
    reg = {"medicao_inicial": {"global": {"linhas": 50, "ramos": 40, "combinada": 45}}, "pragmas": [], "lacunas_aceitas": []}
    texto = g.gerar(xml, reg, RAIZ)
    assert "**SEM CLASSIFICAÇÃO**" in texto and "50% → 66.67%" in texto and "Nenhuma exclusão." in texto
    reg["lacunas_aceitas"] = [{"arquivo": "pipeline.py", "funcao": "x", "categoria": "C", "motivo": "depende do Windows"}]
    reg["pragmas"] = [{"arquivo": "src/guaraci/x.py", "funcao": "f", "quantidade": 2, "categoria": "C", "motivo": "so tipagem"}]
    texto = g.gerar(xml, reg, RAIZ)
    assert "C · depende do Windows" in texto and "`src/guaraci/x.py:f` | 2 | C | so tipagem" in texto


def test_sem_nenhuma_lacuna_o_documento_diz_que_nada_falta(tmp_path):
    p = tmp_path / "ok.xml"
    p.write_text('<coverage><packages><package><classes><class filename="src/guaraci/figuras.py"><lines><line number="1" hits="1"/></lines></class>'
                 "</classes></package></packages></coverage>", encoding="utf-8")
    assert "Nada: todas as linhas e todos os ramos medidos estão cobertos." in g.gerar(p, {"pragmas": [], "lacunas_aceitas": []}, RAIZ)


def test_a_verificacao_aponta_documento_desatualizado(xml, tmp_path, monkeypatch, capsys):
    saida = tmp_path / "cobertura_lacunas.md"
    monkeypatch.setattr(g, "SAIDA", saida)
    monkeypatch.setattr(g, "REGISTRO", RAIZ / "tests" / "cobertura_excecoes.json")
    assert g.main(["--cobertura", str(xml)]) == 0 and saida.exists()
    assert g.main(["--cobertura", str(xml), "--verificar"]) == 0
    saida.write_text("velho\n", encoding="utf-8")
    assert g.main(["--cobertura", str(xml), "--verificar"]) == 1 and "desatualizado" in capsys.readouterr().out
