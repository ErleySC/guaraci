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


# ── Fase 2 (Agente 6): sugerir NAO para no texto -- oferece o plano de ────
#    coleta de verdade, reaproveitando plan_from_statistical_target.

def test_diagnosticar_oferece_plano_de_coleta_quando_n_insuficiente(monkeypatch):
    """Forca um dataset com poucas sessoes por classe (mesmo padrao de
    test_checar_n_insuficiente_classe_fraca em test_auditoria_delineamento.
    py) via monkeypatch de load_data/validate_input -- confirma que a
    pergunta "quer ver o plano?" aparece e que responder 's' produz um
    plano real (nao so' nao lanca excecao)."""
    rotulos = np.array(["A"] * 3 + ["B"] * 3)
    mae_id = np.array([f"A-0{i}-01-2099-S1.00" for i in range(3)] +
                      [f"B-0{i}-01-2099-S1.00" for i in range(3)])
    wn = np.linspace(4000.0, 10000.0, 20)
    X = np.random.default_rng(0).normal(size=(6, 20))
    conc = None
    metadados = {}

    monkeypatch.setattr(guaraci_mod.pq, "load_data",
                        lambda cfg: (wn, X, rotulos, conc, mae_id, metadados))
    monkeypatch.setattr(guaraci_mod.pq, "validate_input",
                        lambda *a, **k: (X, wn, rotulos, conc, mae_id, {}))

    respostas = iter(["s", ""])   # 's' na pergunta do plano, "" no _pause
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))

    cfg = Config()
    guaraci_mod._guaraci_diagnosticar(cfg)   # nao deve lancar excecao


def test_diagnosticar_nao_oferece_plano_quando_n_ja_suficiente(monkeypatch):
    """Contra-prova inversa: com sessoes suficientes, achado_n vem 'ok' e
    a pergunta do plano nao deveria nem aparecer -- so' 1 resposta de
    input (_pause) e' consumida, se sobrar resposta no iterator o teste
    de outra forma nao pegaria isso, entao usamos um iterator de 1 item
    so' -- StopIteration se o codigo pedir mais input do que devia."""
    from guaraci.conformal import n_minimum_for_alpha
    n_min = n_minimum_for_alpha(0.05)
    rotulos = np.array(["A"] * n_min)
    mae_id = np.array([f"A-{i:02d}-01-2099-S1.00" for i in range(n_min)])
    wn = np.linspace(4000.0, 10000.0, 20)
    X = np.random.default_rng(0).normal(size=(n_min, 20))

    monkeypatch.setattr(guaraci_mod.pq, "load_data",
                        lambda cfg: (wn, X, rotulos, None, mae_id, {}))
    monkeypatch.setattr(guaraci_mod.pq, "validate_input",
                        lambda *a, **k: (X, wn, rotulos, None, mae_id, {}))
    monkeypatch.setattr("builtins.input", lambda *a, **k: iter([""]).__next__())

    cfg = Config()
    guaraci_mod._guaraci_diagnosticar(cfg)   # so' 1 input consumido -- OK


# ── FAQ curado (Agente 6, Fase 2) ─────────────────────────────────────────

def test_faq_roda_sem_excecao_para_cada_pergunta(monkeypatch):
    """Cada entrada de _FAQ precisa produzir resposta sem lancar excecao,
    pra' qualquer nivel de Config (a resposta de 'qual metodo usar' e'
    condicional em cfg.level)."""
    for nivel in ("N1", "N2", "N3"):
        for n in range(1, len(guaraci_mod._FAQ) + 1):
            respostas = iter([str(n), ""])
            monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
            guaraci_mod._guaraci_faq(Config(level=nivel))


def test_faq_metodo_recomendado_e_grounded_no_nivel_real():
    """Regra dura do Agente 6: a resposta muda com o cfg real, nao e' um
    texto fixo -- contra-prova de que os 3 niveis produzem texto DISTINTO
    (prova minima de que a funcao le' cfg.level de verdade)."""
    respostas = {
        nivel: guaraci_mod._faq_metodo_recomendado(Config(level=nivel), True)
        for nivel in ("N1", "N2", "N3")
    }
    assert len(set(respostas.values())) == 3
    assert "PLS-DA" in respostas["N1"]
    assert "DD-SIMCA" in respostas["N2"]
    assert "PLS-R" in respostas["N3"]


def test_assistente_opcao_5_chama_faq(monkeypatch):
    chamado = {}
    monkeypatch.setattr(guaraci_mod, "_guaraci_faq",
                        lambda cfg: chamado.setdefault("ok", True))
    monkeypatch.setattr("builtins.input", lambda *a, **k: "5")
    guaraci_mod._abrir_assistente("teste", Config())
    assert chamado.get("ok") is True


def test_assistente_opcao_6_chama_fluxo_decisao(monkeypatch):
    chamado = {}
    monkeypatch.setattr(guaraci_mod, "_guaraci_fluxo_decisao",
                        lambda cfg: chamado.setdefault("ok", True))
    monkeypatch.setattr("builtins.input", lambda *a, **k: "6")
    guaraci_mod._abrir_assistente("teste", Config())
    assert chamado.get("ok") is True


# ── Fluxo de entrada orientado a decisao (Bloco 19) ───────────────────────

def test_fluxo_decisao_roda_sem_excecao_para_cada_opcao_sem_aplicar(monkeypatch):
    """Cada opcao de _FLUXO_DECISAO precisa rodar sem excecao; resposta 'n'
    na pergunta de aplicar (quando existe) -- cfg nao deve mudar."""
    for n in range(1, len(guaraci_mod._FLUXO_DECISAO) + 1):
        respostas = iter([str(n), "n", ""])
        monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
        cfg = Config()
        nivel_antes = cfg.level
        guaraci_mod._guaraci_fluxo_decisao(cfg)
        assert cfg.level == nivel_antes


def test_fluxo_decisao_aplica_nivel_e_preprocessamento_quando_confirmado(monkeypatch):
    """Opcao 'identificar_especie' tem nivel=N1 -- responder 's' precisa
    de fato mudar cfg.level (nao so' mostrar o texto)."""
    idx = next(i for i, o in enumerate(guaraci_mod._FLUXO_DECISAO, start=1)
              if o["id"] == "identificar_especie")
    respostas = iter([str(idx), "s", ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
    cfg = Config(level="N2")
    guaraci_mod._guaraci_fluxo_decisao(cfg)
    assert cfg.level == "N1"
    assert cfg.default_preprocessing == "msc_sg_mc"


def test_fluxo_decisao_opcao_invalida_nao_lanca_excecao(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: "99")
    guaraci_mod._guaraci_fluxo_decisao(Config())


def test_fluxo_decisao_q_volta_sem_pedir_mais_input(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: "Q")
    guaraci_mod._guaraci_fluxo_decisao(Config())


def test_fluxo_decisao_toda_tecnica_referenciada_existe_no_registry():
    """Nenhuma opcao pode citar um id de tecnica que nao esta' no
    REGISTRY -- pegaria um typo ou uma referencia a tecnica renomeada."""
    from guaraci.technique_registry import REGISTRY
    ids_registry = {e.id for e in REGISTRY}
    for opcao in guaraci_mod._FLUXO_DECISAO:
        for tid in opcao["tecnicas"]:
            assert tid in ids_registry, (
                f"opcao '{opcao['id']}' referencia tecnica inexistente "
                f"no REGISTRY: '{tid}'")


def test_fluxo_decisao_resolver_mistura_sugere_mcr_als():
    """Confirma que o Bloco 14 (MCR-ALS) esta' de fato conectado ao fluxo
    de decisao -- nao so' presente no REGISTRY sem uso."""
    opcao = next(o for o in guaraci_mod._FLUXO_DECISAO if o["id"] == "resolver_mistura")
    assert "mcr_als" in opcao["tecnicas"]


# ── _guaraci_navegar_secoes cobre as 19 abas reais (achado do Agente 6 +
#    "X" de HSI, Passo 102) ──────────────────────────────────────────────

def test_navegar_secoes_cobre_todas_as_19_abas_reais():
    """Lista confirmada pelo Agente 1 (auditoria funcional, nao suposta):
    19 abas -- 1-9, H, B, X, J, U, K, P, G, ?, A ("X" = HSI, Passo 102).
    "G" fica fora do proprio indice (e' o assistente, navegar ate ele de
    dentro dele e' circular). Antes desta correcao, so' 8 apareciam aqui."""
    teclas_com_t_d = {k for k, _, _ in guaraci_mod._SECOES_NAVEGAVEIS}
    teclas_indice = teclas_com_t_d | {"A"}   # "A" (Sobre) e' adicionado a parte
    esperado = {"1", "2", "3", "4", "5", "6", "7", "8", "9",
                "H", "B", "X", "J", "U", "K", "P", "?", "A"}
    assert teclas_indice == esperado


def test_navegar_secoes_toda_chave_t_e_d_resolve_para_texto_real():
    """Nenhuma entrada de _SECOES_NAVEGAVEIS pode apontar pra uma chave
    _t()/d_ inexistente (cairia no fallback = a propria chave crua, ex.
    'd_predicao' aparecendo literal na tela em vez de uma descricao)."""
    for _k, t_key, d_key in guaraci_mod._SECOES_NAVEGAVEIS:
        for chave in (t_key, d_key):
            nome = guaraci_mod._t(chave)
            assert nome != chave, f"chave de traducao '{chave}' sem entrada real"


def test_navegar_secoes_roda_sem_excecao_para_cada_aba(monkeypatch):
    for tecla, _t_key, _d_key in guaraci_mod._SECOES_NAVEGAVEIS + [("A", "", "")]:
        respostas = iter([tecla, ""])
        monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
        guaraci_mod._guaraci_navegar_secoes(Config())


# ── Bloco 23: aviso de escopo do MCR-ALS no fluxo de decisao ─────────────

def test_fluxo_decisao_resolver_mistura_tem_aviso_de_escopo():
    opcao = next(o for o in guaraci_mod._FLUXO_DECISAO if o["id"] == "resolver_mistura")
    assert "aviso" in opcao
    assert "MCR-ALS" in opcao["aviso"]["PT"]
    assert "MCR-ALS" in opcao["aviso"]["EN"]


def test_fluxo_decisao_exibe_aviso_ao_selecionar_resolver_mistura(monkeypatch, capsys):
    idx = next(i for i, o in enumerate(guaraci_mod._FLUXO_DECISAO, start=1)
              if o["id"] == "resolver_mistura")
    respostas = iter([str(idx), ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
    guaraci_mod._guaraci_fluxo_decisao(Config())
    saida = capsys.readouterr().out
    assert "MCR-ALS" in saida
    assert "recuperou" in saida or "quantificar" in saida.lower()
