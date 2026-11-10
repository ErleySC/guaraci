# -*- coding: utf-8 -*-
"""`guaraci COMANDO --help` mostra a ajuda e sai com 0; nunca executa o comando.

Achado da rodada de fechamento (wheel instalado em venv limpo): `demo --help` rodava a demonstracao inteira (68 s), `web --help`
subia o servidor e ficava preso, e `run`, `confianca` e `manual` tratavam `--help` como argumento e saiam com erro.
"""
from __future__ import annotations

import pytest

import guaraci.guaraci as g

COMANDOS = ("run", "demo", "web", "doctor", "confianca", "manual")


@pytest.mark.parametrize("comando", COMANDOS)
@pytest.mark.parametrize("pedido", ("--help", "-h"))
def test_pedir_ajuda_mostra_a_ajuda_geral_e_nao_executa(comando, pedido, monkeypatch, capsys):
    def _nao_deve_rodar(*a, **k):
        raise AssertionError(f"{comando} --help executou o comando")

    for nome in ("_comando_demo", "_comando_run", "_comando_web", "_comando_doctor"):
        monkeypatch.setattr(g, nome, _nao_deve_rodar)
    monkeypatch.setattr(g._cmd, "subcomando", _nao_deve_rodar)
    assert g.main([comando, pedido]) is None            # nenhum SystemExit: codigo de saida 0
    saida = capsys.readouterr().out
    assert saida.startswith("Uso: guaraci") and comando in saida


def test_comando_com_argumento_de_verdade_continua_executando(monkeypatch):
    chamadas = []
    monkeypatch.setattr(g, "_comando_run", lambda cfg: chamadas.append(cfg))
    g.main(["run", "meu_config.yaml"])
    assert chamadas == ["meu_config.yaml"]
