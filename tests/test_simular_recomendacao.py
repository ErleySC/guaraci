"""`scripts/simular_recomendacao.py`: o atalho de velocidade nao muda o p, e a evidencia versionada diz o que o texto afirma."""
import importlib.util
import json
from pathlib import Path

import pytest

import guaraci.anova_grupos as ag

RAIZ = Path(__file__).resolve().parents[1]
CASO_LIBERAL = "normal, variancias 1:10, n desigual (grupo de maior variancia tem n=5)"


@pytest.fixture(scope="module")
def sim():
    spec = importlib.util.spec_from_file_location("simular_recomendacao", RAIZ / "scripts" / "simular_recomendacao.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_atalho_do_valor_critico_nao_muda_o_p_ajustado(sim, monkeypatch):
    gerador = sim.CENARIOS["normal, variancias 1:10, n=10"][0]
    y, g, u = sim._amostra(gerador, 7, 0.0)
    antes = {m: [c["p_ajustado"] for c in sim._rodar(m, y, g, u)] for m in ("tukey", "games-howell")}
    monkeypatch.setattr(ag, "_q_critico", ag._q_critico)        # restaurado ao fim do teste
    sim._sem_ic_caro()
    depois = {m: [c["p_ajustado"] for c in sim._rodar(m, y, g, u)] for m in ("tukey", "games-howell")}
    assert antes == depois


def test_simulacao_pequena_devolve_taxas_validas_para_todos_os_cenarios(sim, monkeypatch):
    monkeypatch.setattr(ag, "_q_critico", ag._q_critico)
    r = sim.simular(2)
    assert set(r["cenarios"]) == set(sim.CENARIOS) and r["repeticoes"] == 2
    for c in r["cenarios"].values():
        assert sum(c["recomendacao"].values()) == pytest.approx(1.0)
        assert all(0.0 <= v <= 1.0 for d in (c["fwer"], c["poder"]) for v in d.values())
    md = sim._md(r)
    assert "## 1. A recomendacao acerta?" in md and "## 2. Erro tipo I" in md


def test_evidencia_versionada_de_2000_repeticoes_sustenta_o_que_o_texto_afirma():
    r = json.loads((RAIZ / "docs" / "qualidade" / "anova" / "recomendacao_simulacao.json").read_text(encoding="utf-8"))
    assert r["repeticoes"] == 2000 and len(r["cenarios"]) == 8 and "Users" not in json.dumps(r)
    ep = (0.05 * 0.95 / r["repeticoes"]) ** 0.5
    bom = r["cenarios"]["normal, variancias iguais, n=10"]
    assert abs(bom["fwer"]["tukey"] - 0.05) < 3 * ep                  # o controle: Tukey honra o nivel quando as premissas valem
    liberal = r["cenarios"][CASO_LIBERAL]
    assert liberal["fwer"]["tukey"] > 0.20                            # a premissa que justifica recomendar: Tukey infla muito aqui
    assert liberal["fwer"]["recomendado"] < liberal["fwer"]["tukey"] / 2
    assert liberal["fwer"]["games-howell"] < 0.07
    # limites declarados no manual: o acerto e' baixo em n=4 e em caudas pesadas
    assert r["cenarios"]["normal, variancias iguais, n=4"]["acerto"] < 0.6
    assert r["cenarios"]["t com 3 gl (caudas pesadas), n=15"]["acerto"] < 0.2
    assert r["cenarios"]["lognormal (assimetrica), n=15"]["acerto"] > 0.9


def test_numeros_do_manual_batem_com_a_evidencia():
    r = json.loads((RAIZ / "docs" / "qualidade" / "anova" / "recomendacao_simulacao.json").read_text(encoding="utf-8"))
    manual = (RAIZ / "MANUAL.md").read_text(encoding="utf-8")
    for nome in ("normal, variancias iguais, n=10", "normal, variancias iguais, n=4", "normal, variancias 1:10, n=10",
                 CASO_LIBERAL, "lognormal (assimetrica), n=15", "t com 3 gl (caudas pesadas), n=15"):
        c = r["cenarios"][nome]
        linha = f"{c['acerto']:.0%} | {c['fwer']['recomendado']:.3f} ({c['fwer']['tukey']:.3f})".replace(".", ",")
        assert linha in manual, (nome, linha)
