"""Erros de arquivo de `vip`, `monitorar` e `transferir` sugerem o exemplo do PROPRIO comando, nao o da ANOVA (V30, lote 4)."""
from pathlib import Path

import pytest

from guaraci import cli_comandos as cmd
from guaraci import erros_entrada as ee


@pytest.mark.parametrize("cenario", ["vip", "monitorar", "transferir", "comparar"])
def test_erro_de_arquivo_sugere_o_exemplo_do_cenario_pedido(cenario):
    a = ee.arquivo_nao_encontrado("x.csv", "pt", cenario)
    b = ee.leitura_csv(str(Path(__file__)), UnicodeError("x"), "en", cenario)
    assert f"guaraci exemplo {cenario}`" in a and f"guaraci exemplo {cenario}`" in b


def test_erro_padrao_continua_sugerindo_o_exemplo_de_anova():
    assert "guaraci exemplo comparar" in ee.arquivo_nao_encontrado("x.csv", "pt")


@pytest.mark.parametrize("args,cenario", [
    (["vip", "nao_existe.csv", "--classe", "a", "--grupo", "b"], "vip"),
    (["monitorar", "--referencia", "nao_existe.csv", "--novos", "n.csv", "--lote", "l"], "monitorar"),
    (["transferir", "--mestre", "nao_existe.csv", "--escravo", "e.csv", "--novos", "n.csv"], "transferir"),
])
def test_comandos_nao_sugerem_o_exemplo_da_anova(args, cenario, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    codigo = cmd.subcomando(args[0], args[1:])
    err = capsys.readouterr().err
    assert codigo == 2
    assert f"guaraci exemplo {cenario}`" in err and "exemplo comparar" not in err
