"""As duvidas de quimiometria respondidas por regra (catalogo_duvidas) e o 'nao sei' honesto."""
import socket

import pytest

from guaraci import assistente as ass
from guaraci import catalogo as cat
from guaraci import cli_comandos as cmd
from guaraci.catalogo_duvidas import DUVIDAS

_CASOS = [(k, q, "PT") for k, d in DUVIDAS.items() for q in d["exemplos"][0]] + \
         [(k, q, "EN") for k, d in DUVIDAS.items() for q in d["exemplos"][1]]


def test_ha_pelo_menos_dez_duvidas_com_exemplos_em_pt_e_en():
    assert len(DUVIDAS) >= 10
    for k, d in DUVIDAS.items():
        assert len(d["exemplos"][0]) >= 2 and len(d["exemplos"][1]) >= 1, k


@pytest.mark.parametrize("ident,pergunta,lang", _CASOS)
def test_cada_exemplo_cai_na_propria_regra(ident, pergunta, lang):
    assert ass.duvida(pergunta, lang)["id"] == ident


@pytest.mark.parametrize("lang,rotulos", [("PT", ("Isto significa:", "Por que acontece:", "O que fazer agora:", "Onde ver na CLI:")),
                                          ("EN", ("This means:", "Why it happens:", "What to do now:", "Where to look in the CLI:"))])
def test_resposta_vem_no_gabarito_dos_quatro_campos(lang, rotulos):
    for k, d in DUVIDAS.items():
        r = ass.responder(d["exemplos"][0 if lang == "PT" else 1][0], lang)
        assert all(rot in r for rot in rotulos), (k, r)
        assert d["nome"][0 if lang == "PT" else 1] in r


def test_por_que_meu_q2_e_baixo_nao_devolve_so_a_tela_modelagem():
    r = ass.responder("por que meu modelo tem Q2 baixo")
    assert "Isto significa" in r and "O que fazer agora" in r
    assert not r.startswith("Modelagem")


def test_r2_alto_com_q2_baixo_vence_a_regra_de_q2_baixo_sozinho():
    assert ass.duvida("meu R2 de calibracao e alto mas o Q2 e baixo")["id"] == "r2_alto_q2_baixo"
    assert ass.duvida("meu Q2 e baixo")["id"] == "q2_baixo"


def test_derivada_do_preprocessamento_nao_vira_alarme_de_deriva():
    assert ass.duvida("qual pre-processamento devo escolher, derivada ou SNV")["id"] == "escolha_preproc"


def test_pergunta_de_definicao_continua_indo_para_o_termo():
    assert ass.duvida("o que e Q2") is None
    assert "VIP" in ass.responder("o que e VIP")


@pytest.mark.parametrize("pergunta", ["qual a capital da franca", "receita de bolo de cenoura", "meu carro nao liga"])
def test_fora_de_escopo_diz_que_nao_sabe_e_aponta_o_manual(pergunta):
    r = ass.responder(pergunta)
    assert r.startswith("Nao sei responder isso com seguranca")
    assert "guaraci manual" in r and "?" in r
    assert all(nome in r for nome in ass.duvidas())


def test_fora_de_escopo_em_ingles():
    r = ass.responder("what is the capital of France", "EN")
    assert r.startswith("I cannot answer that reliably")


def test_funciona_sem_rede(monkeypatch):
    def sem_rede(*a, **k):
        raise AssertionError("o assistente tentou usar a rede")
    monkeypatch.setattr(socket, "socket", sem_rede)
    monkeypatch.setattr(socket, "create_connection", sem_rede)
    assert "Isto significa" in ass.responder("por que meu Q2 e baixo")


def test_nenhuma_resposta_relata_validacao_e_o_catalogo_continua_valido():
    assert cat.validar_catalogo() == []
    for k, d in DUVIDAS.items():
        for campo in ("significa", "por_que", "fazer"):
            for txt in d[campo]:
                assert cat.texto_proibido(txt) is None, (k, campo)


def test_onde_ver_na_cli_aponta_para_tela_comando_grafico_ou_manual_que_existem():
    for k, d in DUVIDAS.items():
        for tipo, ident in d["onde"]:
            assert tipo in ("tela", "comando", "grafico", "manual"), k
        assert ass.duvida(d["exemplos"][0][0])["onde"], k


def test_comando_assistente_responde_a_duvida(capsys):
    assert cmd.subcomando("assistente", ["por", "que", "meu", "Q2", "e", "baixo"]) == 0
    assert "Isto significa" in capsys.readouterr().out


def test_buscar_acha_a_duvida_pelo_nome():
    achados = ass.buscar("sobreajuste overfitting", "PT", 5)
    assert any(a["tipo"] == "duvida" and a["id"] == "overfitting" for a in achados)


def test_alternativa_curta_so_casa_a_palavra_inteira_e_a_longa_casa_por_prefixo():
    assert ass.duvida("meu q2x e baixo") is None            # 'q2' tem 2 letras: exato
    assert ass.duvida("meu q2 esta baixa")["id"] == "q2_baixo"  # 'baix' casa 'baixa'
