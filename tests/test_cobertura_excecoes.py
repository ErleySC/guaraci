"""Guarda das exclusoes de cobertura: o numero de `# pragma: no cover` e de omit/exclude nao cresce sem entrada no registro.

Alem do repositorio real, cada defeito da guarda (exclusao nova sem registro, quantidade maior que a registrada, registro de exclusao que
saiu, entrada sem motivo, exclusao na configuracao) e' reproduzido em copias sinteticas, para que a guarda nao possa deixar de reclamar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))

import cobertura_excecoes as ce  # noqa: E402


def _registro(*pragmas, py=()):
    return {"pragmas": list(pragmas), "exclusoes_pyproject": list(py)}


def _entrada(arquivo="src/guaraci/x.py", funcao="f", quantidade=1, motivo="porque sim", categoria="C"):
    return {"arquivo": arquivo, "funcao": funcao, "quantidade": quantidade, "motivo": motivo, "categoria": categoria}


def test_o_repositorio_nao_tem_exclusao_sem_registro_nem_registro_sem_exclusao():
    d = ce.comparar(ce.pragmas_no_codigo(), ce.exclusoes_no_pyproject(), ce.carregar_registro())
    assert d == {"sem_registro": [], "obsoletas": [], "incompletas": []}


def test_todo_campo_do_spec_tem_entrada_no_catalogo():
    from guaraci import catalogo
    from guaraci.config_io import _CONFIG_SPEC
    assert [s["key"] for s in _CONFIG_SPEC if s["key"] not in catalogo.OPCOES] == []


def test_pragma_novo_sem_registro_e_apontado():
    d = ce.comparar({("src/guaraci/x.py", "f"): 1}, [], _registro())
    assert d["sem_registro"] and "x.py:f" in d["sem_registro"][0]


def test_mais_pragmas_do_que_o_registrado_e_apontado():
    d = ce.comparar({("src/guaraci/x.py", "f"): 3}, [], _registro(_entrada(quantidade=2)))
    assert d["sem_registro"] and "3 pragma(s), registrado 2" in d["sem_registro"][0]


def test_registro_de_exclusao_que_ja_saiu_e_apontado_como_obsoleto():
    d = ce.comparar({}, [], _registro(_entrada()))
    assert d["obsoletas"] and "registrado 1, no codigo 0" in d["obsoletas"][0]
    d = ce.comparar({("src/guaraci/x.py", "f"): 1}, [], _registro(_entrada(quantidade=2)))
    assert d["obsoletas"]


def test_igualdade_entre_codigo_e_registro_nao_reclama():
    assert ce.comparar({("src/guaraci/x.py", "f"): 2}, [], _registro(_entrada(quantidade=2))) == {"sem_registro": [], "obsoletas": [], "incompletas": []}


@pytest.mark.parametrize("campo, valor", [("motivo", "   "), ("motivo", ""), ("categoria", "Z"), ("categoria", None)])
def test_entrada_sem_motivo_ou_com_categoria_invalida_e_apontada(campo, valor):
    e = _entrada()
    e[campo] = valor
    assert ce.comparar({("src/guaraci/x.py", "f"): 1}, [], _registro(e))["incompletas"]


def test_exclusao_na_configuracao_de_cobertura_exige_registro_e_some_quando_sai():
    d = ce.comparar({}, ["report.exclude_lines: if x:"], _registro())
    assert d["sem_registro"] == ["pyproject: report.exclude_lines: if x:"]
    d = ce.comparar({}, [], _registro(py=["report.omit: foo.py"]))
    assert d["obsoletas"] == ["pyproject: report.omit: foo.py"]


def test_a_varredura_conta_pragmas_por_funcao_e_le_a_configuracao(tmp_path):
    (tmp_path / "src" / "guaraci").mkdir(parents=True)
    (tmp_path / "src" / "guaraci" / "m.py").write_text(
        "def f():\n    pass  # pragma: no cover\n\n\ndef g():\n    pass  # pragma:no cover\n    pass  # PRAGMA: NO COVER\n\nx = 1  # pragma: no cover\n",
        encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[tool.coverage.report]\nexclude_lines = ["if TYPE_CHECKING:"]\n[tool.coverage.run]\nomit = ["a.py"]\n', encoding="utf-8")
    assert ce.pragmas_no_codigo(tmp_path) == {("src/guaraci/m.py", "f"): 1, ("src/guaraci/m.py", "g"): 2, ("src/guaraci/m.py", "<modulo>"): 1}
    assert ce.exclusoes_no_pyproject(tmp_path) == ["run.omit: a.py", "report.exclude_lines: if TYPE_CHECKING:"]


def test_type_checking_e_stub_excluidos_pelo_padrao_do_coverage_contam_como_exclusao(tmp_path):
    (tmp_path / "src" / "guaraci").mkdir(parents=True)
    (tmp_path / "src" / "guaraci" / "m.py").write_text(
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:  # pragma: no cover\n    import os\n\n\nclass P:\n    def f(self) -> int: ...\n",
        encoding="utf-8")
    assert ce.pragmas_no_codigo(tmp_path) == {("src/guaraci/m.py", "<modulo>"): 1, ("src/guaraci/m.py", "f"): 1}


def test_todo_padrao_que_o_coverage_instalado_exclui_sozinho_e_vigiado_pela_guarda():
    # um coverage novo que passe a excluir outro padrao por conta propria criaria exclusao invisivel ao registro
    config = pytest.importorskip("coverage.config")
    assert [p for p in config.DEFAULT_EXCLUDE if p not in ce.PADROES_DO_COVERAGE] == []


def test_o_registro_do_repositorio_e_json_valido_com_as_categorias_declaradas():
    registro = json.loads((RAIZ / "tests" / "cobertura_excecoes.json").read_text(encoding="utf-8"))
    assert set(registro) >= {"pragmas", "exclusoes_pyproject"}
    assert all(e["categoria"] in ce.CATEGORIAS for e in registro["pragmas"])
    assert ce.main(["--verificar"]) == 0


def test_pragma_na_linha_do_proprio_def_e_em_funcao_aninhada_vai_para_a_funcao_certa(tmp_path):
    (tmp_path / "src" / "guaraci").mkdir(parents=True)
    (tmp_path / "src" / "guaraci" / "m.py").write_text(
        "def h(): pass  # pragma: no cover\n\n\ndef externa():\n    def interna():\n        pass  # pragma: no cover\n    return interna\n", encoding="utf-8")
    assert ce.pragmas_no_codigo(tmp_path) == {("src/guaraci/m.py", "h"): 1, ("src/guaraci/m.py", "interna"): 1}


def test_entrada_sem_quantidade_vale_um_pragma():
    e = _entrada()
    del e["quantidade"]
    assert ce.comparar({("src/guaraci/x.py", "f"): 1}, [], _registro(e)) == {"sem_registro": [], "obsoletas": [], "incompletas": []}
    assert ce.comparar({("src/guaraci/x.py", "f"): 2}, [], _registro(e))["sem_registro"]


def test_a_linha_de_comando_sai_com_codigo_1_quando_ha_diferenca(monkeypatch):
    monkeypatch.setattr(ce, "carregar_registro", lambda: {"pragmas": [], "exclusoes_pyproject": []})
    assert ce.main(["--verificar"]) == 1


def test_documento_que_afirma_o_escopo_da_cobertura_traz_o_numero_medido_de_linhas_excluidas():
    # o numero escrito a mao num documento nao pode ficar para tras quando uma exclusao entra ou sai
    pytest.importorskip("coverage.parser")
    frase = ce.frase_de_escopo(ce.linhas_excluidas())
    for doc in ("docs/cientifico/limitacoes.md", "src/guaraci/dados_docs/limitacoes.md"):
        assert frase in (RAIZ / doc).read_text(encoding="utf-8"), doc


def test_linhas_excluidas_conta_a_linha_gatilho_e_o_corpo_do_bloco(tmp_path):
    pytest.importorskip("coverage.parser")
    (tmp_path / "src" / "guaraci").mkdir(parents=True)
    (tmp_path / "src" / "guaraci" / "m.py").write_text(
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import os\n    import re\nx = 1\n", encoding="utf-8")
    assert ce.linhas_excluidas(tmp_path) == 3
