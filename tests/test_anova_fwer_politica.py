"""Politica padrao do `guaraci anova`: a escolha, o erro por familia (FWER) medido e o limite declarado.

A escolha vem de `scripts/simular_politicas_anova.py` (calibracao em uma grade, confirmacao em outra, independente). Aqui:
  1. a evidencia versionada sustenta o padrao do codigo e a frase do manual/limitacoes;
  2. uma simulacao CURTA e deterministica refaz dois cenarios (um que o padrao cumpre e um que o padrao NAO cumpre) -- se alguem
     "consertar" o Games-Howell ou piorar o controle, a documentacao do limite precisa mudar junto;
  3. as bordas da regra do padrao (com e sem bloco).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
JSON = RAIZ / "docs" / "qualidade" / "politicas_anova_simulacao.json"


@pytest.fixture(scope="module")
def evidencia():
    return json.loads(JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sim():
    sys.path.insert(0, str(RAIZ / "scripts"))
    spec = importlib.util.spec_from_file_location("simular_politicas_anova", RAIZ / "scripts" / "simular_politicas_anova.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_o_padrao_do_codigo_e_a_politica_escolhida_pela_simulacao(evidencia):
    from guaraci.anova_cli import METODO_PADRAO
    assert evidencia["escolha"]["politica"] == METODO_PADRAO == "games-howell"


def test_a_evidencia_declara_que_nenhuma_politica_cumpre_o_criterio_e_o_manual_tambem(evidencia):
    assert evidencia["escolha"]["cumpre_o_criterio"] is False
    texto = (RAIZ / "docs" / "cientifico" / "limitacoes.md").read_text(encoding="utf-8")
    assert "nenhuma política cumpre" in texto.lower()


@pytest.mark.parametrize("grade", ["calibracao", "confirmacao"])
def test_o_padrao_nao_tem_mais_violacoes_que_tukey_auto_e_recomendar(evidencia, grade):
    pol = evidencia[grade]["resumo"]["politicas"]
    gh = pol["games-howell"]["n_acima_limite_com_margem"]
    for outra in ("tukey", "recomendar"):
        assert gh <= pol[outra]["n_acima_limite_com_margem"], (grade, outra)
    assert gh <= pol["auto"]["n_acima_limite_com_margem"] + 2, grade        # na confirmacao empata com o auto


def test_a_recomendacao_perdeu_para_o_padrao_nas_duas_grades(evidencia):
    for grade in ("calibracao", "confirmacao"):
        pol = evidencia[grade]["resumo"]["politicas"]
        assert pol["recomendar"]["n_acima_limite_com_margem"] > pol["games-howell"]["n_acima_limite_com_margem"], grade
        assert pol["recomendar"]["fwer_max"] > pol["games-howell"]["fwer_max"], grade


def test_simulacao_minima_e_deterministica_e_tem_a_forma_esperada(sim, monkeypatch):
    """Gemeo RAPIDO das duas simulacoes `slow` abaixo: mesmo caminho de codigo (cenario -> amostra -> politicas -> FWER/poder), com
    poucas repeticoes, so' confere a forma do resultado e que a mesma semente da o mesmo numero. A precisao estatistica (FWER medido
    contra o limite) fica nos testes `slow`."""
    import guaraci.anova_grupos as ag
    monkeypatch.setattr(ag, "_q_critico", ag._q_critico)
    sc = sim._cenario("normal", 16, 4, 4, "var_alta_n_pequeno")
    r1 = sim.simular_cenario((sc, 4, 20261007))
    r2 = sim.simular_cenario((sc, 4, 20261007))
    assert r1 == r2
    assert set(r1["fwer"]) == set(sim.POLITICAS) == set(r1["poder"]) == set(r1["discordancia"])
    assert all(0.0 <= v <= 1.0 and (v * 4) == int(v * 4) for v in r1["fwer"].values())   # multiplos de 1/4
    assert r1["discordancia"]["games-howell"] == 0.0         # o padrao nao discorda de si mesmo


@pytest.mark.slow          # simulacao Monte Carlo (100 repeticoes); gemeo rapido: test_simulacao_minima_*
def test_simulacao_curta_o_padrao_cumpre_onde_as_premissas_valem(sim, monkeypatch):
    import guaraci.anova_grupos as ag
    monkeypatch.setattr(ag, "_q_critico", ag._q_critico)                    # restaurado no fim (a simulacao troca por um stub)
    reps = 100
    sc = sim._cenario("normal", 1.0, 10, 3, "igual")
    r = sim.simular_cenario((sc, reps, 20261007))
    ep = (0.05 * 0.95 / reps) ** 0.5
    assert r["fwer"]["games-howell"] <= sim.LIMITE_FWER + 3 * ep, r["fwer"]


@pytest.mark.slow          # simulacao Monte Carlo (100 repeticoes, ~45 s); gemeo rapido: test_simulacao_minima_*
def test_simulacao_curta_o_limite_declarado_e_real(sim, monkeypatch):
    """Variancia alta no grupo de poucas unidades (n pequeno): o padrao passa muito do limite. Se este teste falhar porque o erro
    caiu, atualize docs/cientifico/limitacoes.md e a simulacao junto -- nao relaxe o teste."""
    import guaraci.anova_grupos as ag
    monkeypatch.setattr(ag, "_q_critico", ag._q_critico)
    sc = sim._cenario("normal", 16, 4, 4, "var_alta_n_pequeno")
    r = sim.simular_cenario((sc, 100, 20261007))
    assert r["fwer"]["games-howell"] > sim.LIMITE_FWER + 0.03, r["fwer"]


def test_bordas_do_padrao_com_e_sem_bloco():
    from guaraci.anova_cli import METODO_PADRAO_COM_BLOCO, metodo_padrao
    assert metodo_padrao(False) == "games-howell"
    assert metodo_padrao(True) == "auto" == METODO_PADRAO_COM_BLOCO
