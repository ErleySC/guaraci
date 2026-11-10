"""`guaraci confianca`: cada id uma vez, legenda dos selos e "metodo" em vez de "tecnica" (V32, lote 4)."""
import re

from guaraci import cli_comandos as cmd


def test_confianca_lista_cada_id_uma_vez_e_tem_legenda_dos_selos(capsys):
    assert cmd.subcomando("confianca", []) == 0
    saida = capsys.readouterr().out
    ids = re.findall(r"\(([a-z0-9_]+)\)$", saida, re.M)
    assert ids and len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
    assert "Selos:" in saida and "verificacao parcial" in saida and "informativo" in saida
    assert "METODO" in saida and "TECNICA" not in saida


def test_confianca_explica_que_ft_nir_nao_e_um_metodo(capsys):
    assert cmd.subcomando("confianca", ["ft-nir"]) == 2
    err = capsys.readouterr().err
    assert "tecnica de medicao" in err and "metodo ou opcao" in err


def test_help_chama_de_metodo_e_nao_de_tecnica_o_argumento_de_confianca():
    h = cmd.texto_ajuda("PT")
    assert "confianca [METODO]" in h and "confianca [TECNICA]" not in h
