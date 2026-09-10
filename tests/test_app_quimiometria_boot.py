"""Smoke test de boot do app Streamlit (app_quimiometria.py).

Contra-prova de 3 bugs reais achados por execucao (nao suposicao) na
auditoria de 2026-09-01 (Agente 1): `pq.Config(modo=...)` (deveria ser
`mode=`), `guaraci.app_logic.listar_figuras/ler_resumo/ler_model_card`
(nao existem -- os nomes reais sao `list_figures/load_summary/
load_model_card`) e `pq.carregar_config` (nao existe -- o nome real e
`pq.load_config`). Nenhum teste anterior importava/executava
app_quimiometria.py de fato (so' um scan estatico de texto em
test_interfaces_configuraveis.py), entao a suite inteira passava com o app
completamente quebrado no primeiro render. Este teste fecha essa lacuna.
"""
from __future__ import annotations

import os

from streamlit.testing.v1 import AppTest

_APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "app_quimiometria.py",
)


def test_app_quimiometria_carrega_sem_excecao():
    at = AppTest.from_file(_APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
