# -*- coding: utf-8 -*-
"""Testes de guaraci.sentinela_deriva (Bloco 13b).

O invariante central: o alerta de deriva e' um TESTE ESTATISTICO, nao um
limiar cru -- prova-se isso em dois sentidos (1) uma sequencia GERADA
exatamente na taxa nominal nao deve disparar alerta na maioria das vezes
(taxa de falso alarme proxima do `significancia` declarado); (2) uma
sequencia com taxa de rejeicao CLARAMENTE mais alta que o nominal, com n
suficiente, dispara o alerta.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from guaraci.conformal import n_minimum_for_alpha
from guaraci.sentinela_deriva import (
    EstadoSentinela,
    atualizar_com_predicoes,
    carregar_estado,
    checar_deriva,
    salvar_estado,
)


# ── EstadoSentinela: registro e janela ────────────────────────────────

def test_registrar_acumula_historico():
    estado = EstadoSentinela(alpha_nominal=0.05)
    for d in [True, True, False, True]:
        estado.registrar(d)
    assert estado.n == 4
    assert estado.n_fora_do_dominio == 1
    assert estado.taxa_rejeicao_observada == pytest.approx(0.25)


def test_janela_deslizante_descarta_o_mais_antigo():
    estado = EstadoSentinela(alpha_nominal=0.05, janela=3)
    for d in [True, True, True, False]:
        estado.registrar(d)
    assert estado.n == 3
    # o primeiro True foi descartado; sobra [True, True, False]
    assert estado.n_fora_do_dominio == 1


def test_sem_janela_e_cumulativo_sem_limite():
    estado = EstadoSentinela(alpha_nominal=0.05, janela=None)
    for _ in range(500):
        estado.registrar(True)
    assert estado.n == 500


def test_taxa_rejeicao_vazia_e_nan():
    estado = EstadoSentinela()
    assert np.isnan(estado.taxa_rejeicao_observada)


# ── atualizar_com_predicoes ────────────────────────────────────────────

def test_atualizar_com_predicoes_le_coluna_ad_dentro_dominio():
    estado = EstadoSentinela(alpha_nominal=0.05)
    df = pd.DataFrame({"AD_dentro_dominio": [True, True, False, True, True]})
    n_registrado = atualizar_com_predicoes(estado, df)
    assert n_registrado == 5
    assert estado.n == 5
    assert estado.n_fora_do_dominio == 1


def test_atualizar_com_predicoes_sem_coluna_nao_lanca_excecao():
    estado = EstadoSentinela(alpha_nominal=0.05)
    df = pd.DataFrame({"classe_pred": ["A", "B"]})
    n_registrado = atualizar_com_predicoes(estado, df)
    assert n_registrado == 0
    assert estado.n == 0


# ── checar_deriva: poder estatistico e falso alarme ───────────────────

def test_n_abaixo_do_minimo_nao_testa_e_nao_alerta():
    estado = EstadoSentinela(alpha_nominal=0.05)
    for _ in range(5):
        estado.registrar(False)   # 100% rejeicao, mas n so' 5
    alerta = checar_deriva(estado)
    assert alerta.alerta is False
    assert np.isnan(alerta.p_valor)
    assert "minimo" in alerta.mensagem


def test_n_minimo_default_reaproveita_n_minimum_for_alpha():
    """O n_minimo usado por padrao TEM que ser exatamente
    conformal.n_minimum_for_alpha(alpha_nominal) -- reaproveitado, nao
    escolhido a dedo para este modulo."""
    estado = EstadoSentinela(alpha_nominal=0.05)
    n_esperado = n_minimum_for_alpha(0.05)
    for _ in range(n_esperado - 1):
        estado.registrar(True)
    assert str(n_esperado) in checar_deriva(estado).mensagem


def test_taxa_nominal_verdadeira_raramente_dispara_falso_alarme():
    """Contra-prova de calibracao do teste: gerando exatamente na taxa
    NOMINAL (H0 verdadeiro), a fracao de alertas disparados ao longo de
    muitas repeticoes tem que ficar PROXIMA do `significancia` declarado
    -- nao muito acima (o que indicaria um teste maldisenhado, alarmando
    demais)."""
    rng = np.random.default_rng(0)
    alpha_nominal = 0.05
    n_janela = 200
    n_repeticoes = 300
    alertas = 0
    for _ in range(n_repeticoes):
        estado = EstadoSentinela(alpha_nominal=alpha_nominal)
        amostras = rng.random(n_janela) >= alpha_nominal   # True = dentro
        for d in amostras:
            estado.registrar(bool(d))
        if checar_deriva(estado, significancia=0.05).alerta:
            alertas += 1
    taxa_falso_alarme = alertas / n_repeticoes
    # unilateral, alpha=0.05 -> esperado ~5%; tolerancia generosa para nao
    # tornar o teste flaky (Monte Carlo com n_repeticoes finito).
    assert taxa_falso_alarme < 0.12, (
        f"taxa de falso alarme {taxa_falso_alarme:.3f} muito acima do "
        "nominal 0.05 -- teste de deriva mal calibrado")


def test_deriva_real_com_n_suficiente_dispara_alerta():
    """Populacao com taxa de rejeicao MUITO acima do nominal (deriva real
    e' forte, nao sutil) -- com n suficiente, o alerta TEM que disparar."""
    rng = np.random.default_rng(1)
    estado = EstadoSentinela(alpha_nominal=0.05)
    taxa_real_com_deriva = 0.30   # 6x o nominal
    amostras = rng.random(200) >= taxa_real_com_deriva
    for d in amostras:
        estado.registrar(bool(d))
    alerta = checar_deriva(estado)
    assert alerta.alerta is True
    assert alerta.p_valor < 0.05
    assert "DERIVA PROVAVEL" in alerta.mensagem


def test_alpha_nominal_customizado_e_respeitado():
    estado = EstadoSentinela(alpha_nominal=0.20)
    rng = np.random.default_rng(2)
    amostras = rng.random(100) >= 0.20   # taxa == nominal, H0 verdadeiro
    for d in amostras:
        estado.registrar(bool(d))
    alerta = checar_deriva(estado)
    assert alerta.alpha_nominal == pytest.approx(0.20)


# ── Persistencia ────────────────────────────────────────────────────────

def test_salvar_e_carregar_estado_preserva_historico(tmp_path):
    estado = EstadoSentinela(alpha_nominal=0.07, janela=50)
    for d in [True, False, True, True, False]:
        estado.registrar(d)
    caminho = str(tmp_path / "sentinela.json")
    salvar_estado(estado, caminho)

    recarregado = carregar_estado(caminho)
    assert recarregado.alpha_nominal == pytest.approx(0.07)
    assert recarregado.janela == 50
    assert recarregado.historico == estado.historico
    assert recarregado.n == estado.n
    assert recarregado.taxa_rejeicao_observada == pytest.approx(
        estado.taxa_rejeicao_observada)


def test_salvar_estado_grava_json_legivel(tmp_path):
    estado = EstadoSentinela(alpha_nominal=0.05)
    estado.registrar(True)
    caminho = str(tmp_path / "sentinela.json")
    salvar_estado(estado, caminho)
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)
    assert dados["alpha_nominal"] == pytest.approx(0.05)
    assert dados["historico"] == [True]


def test_continuar_registrando_apos_carregar(tmp_path):
    """Simula o uso real: sentinela persistida entre execucoes do
    pipeline, continua acumulando -- nao reseta ao recarregar."""
    estado = EstadoSentinela(alpha_nominal=0.05)
    for _ in range(10):
        estado.registrar(True)
    caminho = str(tmp_path / "sentinela.json")
    salvar_estado(estado, caminho)

    estado2 = carregar_estado(caminho)
    estado2.registrar(False)
    assert estado2.n == 11
    assert estado2.n_fora_do_dominio == 1
