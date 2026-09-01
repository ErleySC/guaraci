# -*- coding: utf-8 -*-
"""Testes de guaraci.auditoria_delineamento (Bloco 11).

O invariante central: SILENCIAR uma checagem nunca a faz desaparecer do
relatorio -- so' muda a severidade para "silenciado" e anexa a
justificativa. Sem justificativa, silenciar falha alto (nunca some sem
motivo registrado).
"""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.auditoria_delineamento import (
    AuditFinding,
    check_grouping,
    check_class_session_confounding,
    check_duplicates,
    check_validation_use_range,
    check_insufficient_n,
    check_external_validation,
    run_audit,
)
from guaraci.config import Config


class _CfgFake:
    """Substituto minimo de Config -- so' os atributos que as checagens
    realmente leem, para nao acoplar estes testes ao Config completo."""

    def __init__(self, grouping_guarantee="high", matrix_profile="generico"):
        self.grouping_guarantee = grouping_guarantee
        self.matrix_profile = matrix_profile


# ── AuditFinding: contrato de severidade ──────────────────────────────

def test_severidade_invalida_falha_alto():
    with pytest.raises(ValueError, match="severidade"):
        AuditFinding("x", "nao_existe", "msg")


def test_severidades_validas_aceitas():
    for sev in ("ok", "aviso", "critico", "silenciado"):
        AuditFinding("x", sev, "msg")   # nao deve lancar


# ── check_grouping ────────────────────────────────────────────────

def test_checar_agrupamento_high_e_ok():
    a = check_grouping(_CfgFake(grouping_guarantee="high"))
    assert a.severidade == "ok"


def test_checar_agrupamento_none_e_critico():
    a = check_grouping(_CfgFake(grouping_guarantee="none"))
    assert a.severidade == "critico"


def test_checar_agrupamento_medium_e_aviso():
    a = check_grouping(_CfgFake(grouping_guarantee="medium"))
    assert a.severidade == "aviso"


# ── check_class_session_confounding ───────────────────────────────

def test_classe_confinada_a_1_sessao_e_critico():
    rotulos = np.array(["A"] * 6 + ["B"] * 6)
    # todas as amostras de A vem da MESMA sessao (mesma data); B tem 2.
    mae_id = np.array(
        ["A-01-01-2099-S1.00"] * 6 +
        ["B-01-01-2099-S1.00"] * 3 + ["B-02-01-2099-S1.00"] * 3)
    a = check_class_session_confounding(rotulos, mae_id)
    assert a.severidade == "critico"
    assert "A" in a.mensagem


def test_sem_classe_confinada_e_ok():
    rotulos = np.array(["A"] * 6 + ["B"] * 6)
    mae_id = np.array(
        ["A-01-01-2099-S1.00"] * 3 + ["A-02-01-2099-S1.00"] * 3 +
        ["B-01-01-2099-S1.00"] * 3 + ["B-02-01-2099-S1.00"] * 3)
    a = check_class_session_confounding(rotulos, mae_id)
    assert a.severidade == "ok"


def test_dataset_inteiro_de_1_sessao_e_critico_estrutural():
    rotulos = np.array(["A", "A", "B", "B"])
    mae_id = np.array(["X-01-01-2099-S1.00"] * 4)
    a = check_class_session_confounding(rotulos, mae_id)
    assert a.severidade == "critico"
    assert "1 UNICA sessao" in a.mensagem


def test_sem_mae_id_vira_aviso_nao_crash():
    a = check_class_session_confounding(np.array(["A", "B"]), None)
    assert a.severidade == "aviso"


# ── check_duplicates ─────────────────────────────────────────────────

def test_checar_duplicatas_detecta_exata():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 20))
    X[5] = X[0]   # duplicata exata
    wn = np.linspace(4000, 10000, 20)
    rotulos = np.array(["A"] * 10)
    a = check_duplicates(X, wn, rotulos)
    assert a.severidade == "critico"


def test_checar_duplicatas_sem_duplicata_e_ok():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 20))
    wn = np.linspace(4000, 10000, 20)
    rotulos = np.array(["A"] * 10)
    a = check_duplicates(X, wn, rotulos)
    assert a.severidade == "ok"


# ── check_insufficient_n ────────────────────────────────────────────

def test_checar_n_insuficiente_classe_fraca():
    rotulos = np.array(["A"] * 3 + ["B"] * 3)
    mae_id = np.array([f"A-0{i}-01-2099-S1.00" for i in range(3)] +
                      [f"B-0{i}-01-2099-S1.00" for i in range(3)])
    a = check_insufficient_n(rotulos, mae_id, alpha_conformal_referencia=0.10)
    assert a.severidade == "aviso"
    assert "A" in a.mensagem and "B" in a.mensagem


def test_checar_n_insuficiente_ok_quando_ha_sessoes_suficientes():
    from guaraci.conformal import n_minimum_for_alpha
    n_min = n_minimum_for_alpha(0.25)   # pequeno, facil de satisfazer
    rotulos = np.array(["A"] * n_min)
    mae_id = np.array([f"A-{i:02d}-01-2099-S1.00" for i in range(n_min)])
    a = check_insufficient_n(rotulos, mae_id, alpha_conformal_referencia=0.25)
    assert a.severidade == "ok"


# ── check_validation_use_range ────────────────────────────────────────

def test_checar_faixa_validacao_uso_sem_conc_e_aviso():
    a = check_validation_use_range(None, _CfgFake())
    assert a.severidade == "aviso"


# ── check_external_validation ──────────────────────────────────────────

def test_checar_validacao_externa_e_sempre_aviso_informativo():
    a = check_external_validation()
    assert a.severidade == "aviso"


# ── run_audit: agregacao + silenciamento ───────────────────────

def _dados_minimos():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 15))
    wn = np.linspace(4000, 10000, 15)
    rotulos = np.array(["A"] * 6 + ["B"] * 6)
    mae_id = np.array(
        ["A-01-01-2099-S1.00"] * 3 + ["A-02-01-2099-S1.00"] * 3 +
        ["B-01-01-2099-S1.00"] * 3 + ["B-02-01-2099-S1.00"] * 3)
    conc = np.array([0.0] * 12)
    return X, wn, rotulos, conc, mae_id


def test_rodar_auditoria_retorna_todas_as_checagens():
    X, wn, rotulos, conc, mae_id = _dados_minimos()
    achados = run_audit(X, wn, rotulos, _CfgFake(), conc, mae_id)
    nomes = {a.nome for a in achados}
    assert nomes == {"agrupamento", "confundimento_classe_sessao",
                      "duplicatas", "n_insuficiente",
                      "faixa_validacao_uso", "validacao_externa"}


def test_silenciar_sem_justificativa_falha_alto():
    X, wn, rotulos, conc, mae_id = _dados_minimos()
    with pytest.raises(ValueError, match="justificativa"):
        run_audit(X, wn, rotulos, _CfgFake(), conc, mae_id,
                  silenciar={"validacao_externa": ""})
    with pytest.raises(ValueError, match="justificativa"):
        run_audit(X, wn, rotulos, _CfgFake(), conc, mae_id,
                  silenciar={"validacao_externa": None})


def test_silenciar_com_justificativa_nao_remove_do_relatorio():
    """Contra-prova central: a checagem SILENCIADA continua no relatorio,
    so' muda de severidade -- nunca desaparece sem rastro."""
    X, wn, rotulos, conc, mae_id = _dados_minimos()
    justificativa = "Benchmark externo fora do escopo deste projeto -- decisao da orientadora."
    achados = run_audit(
        X, wn, rotulos, _CfgFake(), conc, mae_id,
        silenciar={"validacao_externa": justificativa})

    nomes = {a.nome for a in achados}
    assert "validacao_externa" in nomes   # continua no relatorio


# ── Comando CLI dedicado (Passo 76, Bloco 11) ─────────────────────────────
# `_menu_audit` roda so' a auditoria sobre o dataset configurado, sem
# exigir rodar classificacao/quantificacao inteira -- mode="sintetico" nao
# precisa de arquivo em disco, entao o teste end-to-end fica rapido.

def test_menu_audit_cli_end_to_end_roda_sem_pipeline_completo(monkeypatch):
    import guaraci.guaraci as guaraci_mod

    # 1a resposta "" = Enter no gate [G]/[0]/[Enter]=Rodar (auditoria nesta
    # varredura); 2a "" = Enter no _pause() final.
    respostas = iter(["", ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))

    cfg = Config(mode="sintetico", seed=0)
    guaraci_mod._menu_audit(cfg)   # nao deve lancar excecao


def test_menu_audit_cli_sem_fonte_de_dados_nao_quebra(monkeypatch):
    """Contra-prova: pasta_dados vazia/inexistente (mode dx padrao, sem
    configurar nada) tem que reportar erro amigavel e retornar -- nunca
    stack trace cru numa ferramenta interativa."""
    import guaraci.guaraci as guaraci_mod

    respostas = iter([""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))

    cfg = Config(mode="dx", input_folder="/caminho/que/nao/existe/nunca")
    guaraci_mod._menu_audit(cfg)   # nao deve lancar excecao
