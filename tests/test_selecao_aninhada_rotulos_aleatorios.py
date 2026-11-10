"""Com rotulos sem sinal, a selecao aninhada nao passa do acaso; a ingenua passa (controle de poder)."""
import importlib.util
import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prova():
    spec = importlib.util.spec_from_file_location("prova_selecao_aninhada", RAIZ / "scripts" / "prova_selecao_aninhada.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.medir(6)


def test_aninhado_fica_no_acaso_em_todos_os_metodos(prova):
    acaso = 1 / 3
    for nome, v in prova["metodos"].items():
        assert abs(v["aninhado"]["media"] - acaso) < 0.08, (nome, v["aninhado"]["media"])


@pytest.mark.parametrize("nome", ["VIP>=1", "SR top 10%", "sPLS-DA", "CARS"])
def test_controle_de_poder_a_selecao_ingenua_vaza_e_o_experimento_enxerga(prova, nome):
    v = prova["metodos"][nome]
    assert v["ingenuo"]["media"] > 0.45 and v["ingenuo"]["media"] - v["aninhado"]["media"] > 0.15


def test_evidencia_versionada_das_30_sementes_confirma_o_mesmo():
    r = json.loads((RAIZ / "docs" / "qualidade" / "selecao_aninhada_rotulos_aleatorios.json").read_text(encoding="utf-8"))
    assert r["n_sementes"] == 30 and r["acaso"] == pytest.approx(1 / 3)
    assert all(abs(v["aninhado"]["media"] - r["acaso"]) < 0.05 for v in r["metodos"].values())
    assert "Users" not in json.dumps(r)


def test_tabela_de_limitacoes_bate_com_a_evidencia():
    r = json.loads((RAIZ / "docs" / "qualidade" / "selecao_aninhada_rotulos_aleatorios.json").read_text(encoding="utf-8"))
    texto = (RAIZ / "docs" / "cientifico" / "limitacoes.md").read_text(encoding="utf-8")
    rotulos = {"VIP>=1": "VIP ≥ 1", "SR top 10%": "SR (10% melhores)", "sPLS-DA": "sPLS-DA", "CARS": "CARS",
               "iPLS": "iPLS (melhor intervalo)", "SPA": "SPA/APS"}
    for chave, nome in rotulos.items():
        v = r["metodos"][chave]
        linha = f"| {nome} | {v['aninhado']['media']:.3f} | {v['ingenuo']['media']:.3f} |".replace(".", ",")
        assert linha in texto, linha
