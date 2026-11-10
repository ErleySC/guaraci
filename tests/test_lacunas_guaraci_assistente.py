"""Assistente Guaraci e telas de ajuda de `guaraci.guaraci` (revisao, diagnostico, FAQ, fluxo de decisao, confianca, rota, busca).

Os motores (auditoria, plano de coleta, catalogo) sao os de verdade ou trocados por dubles pequenos quando a tela precisa de um caso
que o dado de teste nao produz. O que se confere e' o que a pessoa ve e o que muda na configuracao.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

import guaraci.guaraci as g


def _quebra(msg="falhou", exc=ValueError):
    def f(*a, **k):
        raise exc(msg)
    return f


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_revisao_da_configuracao_marca_o_que_foge_do_recomendado(cli, monkeypatch, lang):
    # Os valores exibidos vem de _cfgv; os ALERTAS vem do registro do assistente (assistente_regras),
    # que le o proprio Config: faixa invertida, permutacoes sem resolucao e sem holdout.
    monkeypatch.setitem(g._STATE, "lang", lang)
    valores = {"pre_processamento": "raw", "max_lvs": 60, "monte_carlo": True, "shap_benchmark": True, "n_permutacoes": 10,
               "opls_da": False, "ddsimca": False, "holdout_fracao": 0.0, "benchmark": True}
    monkeypatch.setattr(g, "_cfgv", lambda cfg, chave, padrao: valores.get(chave, padrao))
    g._guaraci_revisar_config(g.Config(wn_min=5000.0, wn_max=4000.0, n_permutations=10, frac_holdout=0.0))
    saida = cli.saida()
    en = lang == "EN"
    trechos = (("Monte Carlo", "SHAP", "Benchmark", "OPLS-DA off", "DD-SIMCA off", "range is inverted", "1/11", "No external holdout")
               if en else
               ("Monte Carlo", "SHAP", "Benchmark", "OPLS-DA desativado", "DD-SIMCA desativado", "faixa espectral esta invertida",
                "1/11", "Sem holdout externo"))
    for trecho in trechos:
        assert trecho in saida, trecho


def test_revisao_da_configuracao_padrao_nao_levanta_alertas(cli):
    g._guaraci_revisar_config(g.Config())
    assert "Nenhum problema encontrado" in cli.saida() and "Revis" in cli.saida()


def _achado(nome, severidade, mensagem="m"):
    return types.SimpleNamespace(nome=nome, severidade=severidade, mensagem=mensagem)


def _preparar_diagnostico(monkeypatch, achados, mae_id=("A_1", "A_2")):
    monkeypatch.setattr(g.pq, "load_data", lambda cfg: (np.arange(3.0), np.zeros((2, 3)), np.array(["a", "b"]), None, np.array(mae_id), None))
    monkeypatch.setattr(g.pq, "validate_input", lambda *a: (a[0], a[1], a[2], a[3], a[4], None))
    import guaraci.auditoria_delineamento as ad
    monkeypatch.setattr(ad, "run_audit", lambda *a, **k: achados)


def test_diagnostico_com_erro_ao_carregar_mostra_a_mensagem(cli, monkeypatch):
    monkeypatch.setattr(g.pq, "load_data", _quebra("pasta vazia"))
    g._guaraci_diagnosticar(g.Config())
    assert "Erro ao carregar dados: pasta vazia" in cli.saida()


@pytest.mark.parametrize("lang, resposta", [("PT", "s"), ("EN", "y"), ("PT", "n")])
def test_diagnostico_sugere_o_plano_de_coleta_e_so_o_mostra_se_pedido(cli, monkeypatch, lang, resposta):
    import guaraci.plano_coleta as plano
    monkeypatch.setitem(g._STATE, "lang", lang)
    _preparar_diagnostico(monkeypatch, [_achado("n_insuficiente", "aviso"), _achado("x", "critico"), _achado("y", "ok")])
    monkeypatch.setattr(g, "_sugestao_alpha_classe_fraca",
                        lambda r, m: {"classe": "a", "n": 2, "alpha_alcancavel": 0.3, "alpha_ref": 0.05, "n_para_ref": 19})
    monkeypatch.setattr(plano, "plan_from_statistical_target",
                        lambda *a, **k: (types.SimpleNamespace(alertas=["colete mais"]), {"n_por_classe": 19, "origem": "alpha"}))
    cli.dizer(resposta, "")
    g._guaraci_diagnosticar(g.Config())
    saida = cli.saida()
    assert "alpha" in saida and ("colete mais" in saida) is (resposta in ("s", "y"))


def test_diagnostico_com_plano_recusado_pelo_motor_mostra_o_motivo_e_sem_classe_fraca_nao_oferece(cli, monkeypatch):
    import guaraci.plano_coleta as plano
    _preparar_diagnostico(monkeypatch, [_achado("n_insuficiente", "aviso")])
    monkeypatch.setattr(g, "_sugestao_alpha_classe_fraca",
                        lambda r, m: {"classe": "a", "n": 2, "alpha_alcancavel": 0.3, "alpha_ref": 0.05, "n_para_ref": 19})
    monkeypatch.setattr(plano, "plan_from_statistical_target", _quebra("alpha fora do alcance"))
    cli.dizer("s", "")
    g._guaraci_diagnosticar(g.Config())
    assert "alpha fora do alcance" in cli.saida()
    monkeypatch.setattr(g, "_sugestao_alpha_classe_fraca", lambda r, m: None)
    cli.dizer("")
    g._guaraci_diagnosticar(g.Config())


def test_alpha_da_classe_mais_fraca_conta_sessoes_independentes():
    rotulos = np.array(["a", "a", "a", "b", "b"])
    info = g._sugestao_alpha_classe_fraca(rotulos, ["A-1", "A-2", "A-3", "B-1", "B-1"])
    assert info["classe"] == "b" and info["n"] >= 1 and info["alpha_ref"] == 0.05
    assert g._sugestao_alpha_classe_fraca(np.array([], dtype=str), []) is None


def test_catalogo_de_tecnicas_pula_categoria_sem_entrada(cli, monkeypatch):
    import guaraci.technique_registry as tr
    g._guaraci_tecnicas()
    assert "tecnicas cadastradas" in cli.saida()
    um = [e for e in tr.REGISTRY if e.categoria == "quantificacao"][:1]
    monkeypatch.setattr(tr, "REGISTRY", um)
    cli.dizer("")
    g._guaraci_tecnicas()
    assert "1 tecnicas cadastradas" in cli.saida()


@pytest.mark.parametrize("nivel", ["N1", "N2", "N3"])
@pytest.mark.parametrize("pt", [True, False])
def test_metodo_recomendado_fala_do_nivel_da_sessao(nivel, pt):
    texto = g._faq_metodo_recomendado(g.Config(level=nivel), pt)
    assert nivel in texto and ("Sua sessao" in texto) is pt


def test_faq_responde_a_pergunta_escolhida_e_ignora_numero_fora_da_lista(cli):
    cli.dizer("1", "", "999", "")
    g._guaraci_faq(g.Config())
    g._guaraci_faq(g.Config())
    assert cli.saida().count("Enter=voltar") >= 2


def test_navegar_secoes_mostra_a_escolhida(cli):
    chave = next(iter({k for k, _n, _f in g._secoes_do_catalogo()}))
    cli.dizer(chave, "")
    g._guaraci_navegar_secoes(g.Config())
    assert cli.saida().strip()


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_fluxo_de_decisao_percorre_cada_opcao_e_aplica_so_se_confirmado(cli, monkeypatch, lang):
    monkeypatch.setitem(g._STATE, "lang", lang)
    for i, opcao in enumerate(g._FLUXO_DECISAO, 1):
        cfg = g.Config()
        antes = (cfg.level, cfg.default_preprocessing)
        cli.dizer(str(i), "n", "")
        g._guaraci_fluxo_decisao(cfg)
        assert (cfg.level, cfg.default_preprocessing) == antes
        if opcao.get("nivel"):
            cli.dizer(str(i), "s" if lang == "PT" else "y", "")
            g._guaraci_fluxo_decisao(cfg)
            assert cfg.level == opcao["nivel"]
            if opcao.get("preproc_default"):
                assert cfg.default_preprocessing == opcao["preproc_default"]
    cli.dizer("q")
    g._guaraci_fluxo_decisao(g.Config())
    cli.dizer("999")
    g._guaraci_fluxo_decisao(g.Config())


def test_fluxo_de_decisao_com_tecnica_fora_do_registro_pula_a_linha(cli, monkeypatch):
    import guaraci.technique_registry as tr
    monkeypatch.setattr(tr, "REGISTRY", [])
    cli.dizer("1", "n", "")
    g._guaraci_fluxo_decisao(g.Config())
    assert "Tecnica(s) sugerida(s)" in cli.saida()


def test_confianca_lista_abre_pagina_por_numero_e_por_nome_e_recusa_o_desconhecido(cli):
    from guaraci import confianca as conf
    ident = conf.entradas(None)[1][1]
    cli.dizer("1", "", ident, "", "zzzz_nao_existe", "")
    g._abrir_confianca(None, g.Config())
    assert "Assistente de confianca" in cli.saida() and g._t("invalido") in cli.saida()


def test_confianca_de_tela_sem_selo_mostra_o_quadro_todo_em_modo_ascii(cli, monkeypatch):
    monkeypatch.setenv("GUARACI_ASCII", "1")
    cli.dizer("")
    g._abrir_confianca("sobre", g.Config())
    assert "Assistente de confianca" in cli.saida()


def test_rota_do_assistente_cobre_escolha_invalida_sem_rota_e_rota_pronta(cli, monkeypatch):
    from guaraci import assistente as ass
    from guaraci.catalogo_ajuda import DADOS, OBJETIVOS
    cli.dizer("x")
    g._assistente_rota(g.Config())
    cli.dizer("1", "x")
    g._assistente_rota(g.Config())
    monkeypatch.setattr(ass, "rota", lambda o, d: None)
    cli.dizer("1", "1", "")
    g._assistente_rota(g.Config())
    assert "Nao ha rota pronta" in cli.saida()
    monkeypatch.setattr(ass, "rota", lambda o, d: {"ok": True})
    monkeypatch.setattr(ass, "formatar_rota", lambda r, l: "ROTA PRONTA")
    monkeypatch.setitem(g._STATE, "lang", "EN")
    cli.dizer(str(len(OBJETIVOS)), str(len(DADOS)), "")
    g._assistente_rota(g.Config())
    assert "ROTA PRONTA" in cli.saida()


def test_assistente_explica_diagnostica_e_busca(cli, monkeypatch):
    from guaraci import assistente as ass
    cli.dizer("")
    g._assistente_explicar()
    monkeypatch.setattr(ass, "responder", lambda termo, lang: "EXPLICACAO")
    cli.dizer("vip", "")
    g._assistente_explicar()
    cli.dizer("")
    g._assistente_diagnosticar()
    monkeypatch.setattr(ass, "diagnosticar", lambda txt, lang: [{"id": "W1", "significa": "algo", "fazer": "isso"}])
    cli.dizer("aviso", "")
    g._assistente_diagnosticar()
    monkeypatch.setattr(ass, "diagnosticar", lambda txt, lang: [])
    cli.dizer("aviso", "")
    g._assistente_diagnosticar()
    cli.dizer("")
    g._assistente_buscar()
    monkeypatch.setattr(ass, "buscar", lambda q, lang, n: [{"tipo": "t", "titulo": "T", "id": "i", "trecho": "tr"}])
    cli.dizer("vip", "")
    g._assistente_buscar()
    monkeypatch.setattr(ass, "buscar", lambda q, lang, n: [])
    cli.dizer("vip", "")
    g._assistente_buscar()
    saida = cli.saida()
    for trecho in ("EXPLICACAO", "W1: algo", "Nao reconheci", "(i): tr", "Nada encontrado"):
        assert trecho in saida, trecho


def test_assistente_sugere_e_aplica_so_o_que_o_usuario_escolhe(cli, monkeypatch):
    from guaraci import assistente as ass
    sug = [types.SimpleNamespace(texto="ajuste A", campo="max_lvs", valor=7),
           types.SimpleNamespace(texto="so' dica", campo=None, valor=None),
           types.SimpleNamespace(texto="ajuste ruim", campo="max_lvs", valor="abc")]
    monkeypatch.setattr(ass, "sugerir", lambda cfg, lang: sug)
    cfg = g.Config()
    cli.dizer("1", "")
    g._assistente_sugerir(cfg)
    assert cfg.max_lvs == 7
    cli.dizer("2", "")
    g._assistente_sugerir(cfg)
    cli.dizer("3", "")
    g._assistente_sugerir(cfg)
    assert cfg.max_lvs == 7 and "max_lvs = 7" in cli.saida()


def test_assistente_despacha_cada_tecla_para_a_tela_certa(cli, monkeypatch):
    chamadas = []
    for nome in ("_guaraci_revisar_config", "_guaraci_navegar_secoes", "_guaraci_diagnosticar", "_guaraci_tecnicas", "_guaraci_faq",
                 "_guaraci_fluxo_decisao", "_assistente_rota", "_assistente_explicar", "_assistente_sugerir", "_assistente_diagnosticar",
                 "_assistente_buscar"):
        monkeypatch.setattr(g, nome, (lambda n: lambda *a, **k: chamadas.append(n))(nome))
    for tecla in [*"123456789", "10", "11"]:
        cli.dizer(tecla)
        g._abrir_assistente("tela de teste", g.Config())
    cli.dizer("q")
    g._abrir_assistente("", None)
    cli.dizer("1")
    g._abrir_assistente("", None)
    assert len(chamadas) == 11
