"""Assistente: tarefas basicas no catalogo, `--help` que cumpre o que diz e `ajuda vip` que cita o comando (V24, lote 4)."""
import pytest

from guaraci import assistente as ass
from guaraci import catalogo as cat
from guaraci import cli_comandos as cmd


def test_help_do_assistente_nao_promete_abrir_nas_telas_sem_argumento(capsys):
    assert "abre o assistente nas telas" not in cmd.texto_ajuda("PT")
    assert cmd.subcomando("assistente", []) == 0
    msg = capsys.readouterr().out
    assert "tecla G" in msg and "tecla a)" not in msg


def test_a_tecla_do_assistente_nas_telas_e_g():
    assert cat.TELAS["assistente"]["tecla"] == "1.3" and cat.ACOES_MENU["G"] == "assistente"


@pytest.mark.parametrize("pergunta,trecho", [
    ("como eu quantifico um adulterante?", "guaraci exemplo quantificar"),
    ("como eu classifico minhas amostras?", "guaraci exemplo classificar"),
    ("como prevejo amostras novas com o modelo salvo?", "predicao"),
])
def test_assistente_responde_as_tres_tarefas_basicas(pergunta, trecho):
    r = ass.responder(pergunta)
    assert trecho in r, r
    assert "sintetico_adulterantes" not in r


def test_assistente_responde_as_tarefas_basicas_em_ingles():
    assert "guaraci exemplo quantificar" in ass.responder("how do I quantify an adulterant?", "EN")
    assert "guaraci exemplo classificar" in ass.responder("how do I classify my samples?", "EN")


def test_ajuda_vip_cita_o_comando_vip():
    e = ass.explicar("vip")
    assert e is not None and "guaraci vip TABELA.csv --classe COL --grupo COL" in e["texto"]
