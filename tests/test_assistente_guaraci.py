"""Testes do Assistente Guaraci (Agente 6, Fase 1) -- diagnostico de dados
carregados, catalogo de tecnicas, e o unico caso de "sugerir+executar"
implementado nesta rodada (alpha minimo alcancavel para a classe mais
fraca). Antes desta rodada `_abrir_assistente` nao tinha nenhuma cobertura
de teste (confirmado por busca no repo antes de escrever este arquivo).
"""
from __future__ import annotations

import numpy as np
import pytest

import guaraci.guaraci as guaraci_mod
from guaraci.config import Config


# ── _sugestao_alpha_classe_fraca (funcao pura, sem console) ──────────────

def test_sugestao_alpha_identifica_a_classe_mais_fraca():
    rotulos = np.array(["A"] * 3 + ["B"] * 5)
    mae_id = np.array([f"A-0{i}-01-2099-S1.00" for i in range(3)] +
                      [f"B-0{i}-01-2099-S1.00" for i in range(5)])
    info = guaraci_mod._sugestao_alpha_classe_fraca(rotulos, mae_id)

    from guaraci.conformal import achievable_alpha, n_minimum_for_alpha
    assert str(info["classe"]) == "A"
    assert info["n"] == 3
    assert info["alpha_alcancavel"] == pytest.approx(achievable_alpha(3))
    assert info["n_para_ref"] == n_minimum_for_alpha(0.05)


def test_sugestao_alpha_sem_amostras_devolve_none():
    assert guaraci_mod._sugestao_alpha_classe_fraca(
        np.array([], dtype=str), np.array([], dtype=str)) is None


# ── _guaraci_diagnosticar ───────────────────────────────────────────────

def test_diagnosticar_roda_sem_excecao_com_dados_sinteticos(monkeypatch):
    """Mesmo padrao de test_menu_audit_cli_end_to_end_roda_sem_pipeline_
    completo: Config sintetica real, so' monkeypatcha input() para o
    _pause() final."""
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    cfg = Config(mode="sintetico", seed=0)
    guaraci_mod._guaraci_diagnosticar(cfg)   # nao deve lancar excecao


def test_diagnosticar_sem_fonte_de_dados_nao_quebra(monkeypatch):
    """Contra-prova: sem pasta de dados configurada, reporta erro amigavel
    e retorna -- nunca stack trace cru."""
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    cfg = Config()  # mode="dx" default, pasta_dados inexistente
    guaraci_mod._guaraci_diagnosticar(cfg)   # nao deve lancar excecao


# ── _guaraci_tecnicas ────────────────────────────────────────────────────

def test_tecnicas_roda_sem_excecao(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    guaraci_mod._guaraci_tecnicas()   # nao deve lancar excecao


# ── dispatch do assistente ([3]/[4] novos, ver Agente 6) ─────────────────

def test_assistente_opcao_3_chama_diagnosticar(monkeypatch):
    chamado = {}
    monkeypatch.setattr(guaraci_mod, "_guaraci_diagnosticar",
                        lambda cfg: chamado.setdefault("ok", True))
    monkeypatch.setattr("builtins.input", lambda *a, **k: "3")
    guaraci_mod._abrir_assistente("teste", Config())
    assert chamado.get("ok") is True


def test_assistente_opcao_4_chama_tecnicas(monkeypatch):
    chamado = {}
    monkeypatch.setattr(guaraci_mod, "_guaraci_tecnicas",
                        lambda: chamado.setdefault("ok", True))
    monkeypatch.setattr("builtins.input", lambda *a, **k: "4")
    guaraci_mod._abrir_assistente("teste", Config())
    assert chamado.get("ok") is True
