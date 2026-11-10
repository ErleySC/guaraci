"""M-07 do parecer de 2026-09-25: opcao desconhecida e' erro (rc 2); --modo = --mode."""
import pytest

import guaraci.guaraci as g


def test_opcao_desconhecida_sai_com_codigo_2(capsys):
    with pytest.raises(SystemExit) as e:
        g.main(["--perfl=nir_dispersivo_1100_2498nm", "version"])
    assert e.value.code == 2
    assert "opcao desconhecida" in capsys.readouterr().err


def test_modo_e_sinonimo_de_mode():
    ops = {"modo": "controle"}
    g._validar_opcoes(ops)
    assert ops == {"mode": "controle"}


def test_modo_invalido_continua_sendo_recusado(capsys):
    with pytest.raises(SystemExit) as e:
        g._validar_opcoes({"modo": "qualquer"})
    assert e.value.code == 2


def test_opcoes_validas_passam():
    g._validar_opcoes({"mode": "cego"})
