# -*- coding: utf-8 -*-
"""Testes de mcr_als.py (Bloco 14) -- mistura SINTETICA de espectros puros
conhecidos, combinados em proporcoes conhecidas por construcao. Dataset real
de oleo com adulterante em teor declarado NAO esta disponivel neste
checkout (pasta dados/ vazia -- dado de terceiro, nunca versionado, ver
.gitignore) -- essa validacao fica pendente ate' o dado estar acessivel.
"""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.mcr_als import (
    mcr_als,
    avaliar_incerteza_rotacional,
    mcr_als_com_restricao_correlacao,
)


def _mistura_sintetica(seed=0, n=30, p=60, ruido=0.002):
    """3 espectros puros (picos gaussianos em posicoes bem separadas),
    misturados em proporcoes aleatorias que somam 1 por amostra (fecho de
    composicao, como um teor declarado real). Retorna D, C_verdadeiro,
    S_verdadeiro."""
    rng = np.random.default_rng(seed)
    eixo = np.arange(p, dtype=float)
    centros = [10.0, 30.0, 50.0]
    larguras = [3.0, 3.5, 3.0]
    S_true = np.array([np.exp(-0.5 * ((eixo - c) / w) ** 2)
                        for c, w in zip(centros, larguras)])  # (3, p)
    S_true = S_true / S_true.sum(axis=1, keepdims=True)

    proporcoes = rng.dirichlet(alpha=[1.5, 1.5, 1.5], size=n)  # (n, 3), soma 1
    D = proporcoes @ S_true
    D += rng.normal(0, ruido, size=D.shape)
    D = np.clip(D, 0.0, None)
    return D, proporcoes, S_true


def test_mcr_als_recupera_perfis_espectrais_dentro_de_tolerancia():
    D, C_true, S_true = _mistura_sintetica()
    res = mcr_als(D, n_componentes=3)

    # Piso de LOF dado o ruido injetado (sinal e' pequeno na maior parte dos
    # 60 canais -- so' perto dos picos -- entao mesmo o ruido "verdadeiro"
    # (contra C_true@S_true) produz um LOF% que nao e' perto de zero; o
    # criterio honesto e' o ajuste nao ficar muito PIOR que esse piso, nao
    # um numero fixo arbitrario.
    piso_ruido = 100.0 * float(np.linalg.norm(D - C_true @ S_true)) / float(np.linalg.norm(D))
    assert res.lof_percent < piso_ruido * 1.3, (
        f"lack-of-fit ({res.lof_percent:.2f}%) muito acima do piso de ruido "
        f"({piso_ruido:.2f}%)")

    # Casa cada componente recuperado com o verdadeiro por correlacao maxima
    # (MCR-ALS nao preserva rotulo/ordem dos componentes).
    corr = np.array([[abs(np.corrcoef(res.S[i], S_true[j])[0, 1])
                       for j in range(3)] for i in range(3)])
    from scipy.optimize import linear_sum_assignment
    linhas, colunas = linear_sum_assignment(-corr)
    correlacoes_casadas = corr[linhas, colunas]

    assert (correlacoes_casadas > 0.9).all(), (
        f"correlacao espectro recuperado x verdadeiro abaixo do esperado: "
        f"{correlacoes_casadas}")


def test_mcr_als_recupera_proporcoes_dentro_de_tolerancia():
    D, C_true, S_true = _mistura_sintetica(seed=1)
    res = mcr_als(D, n_componentes=3)

    prop_rec = res.C / res.C.sum(axis=1, keepdims=True)

    corr = np.array([[abs(np.corrcoef(prop_rec[:, i], C_true[:, j])[0, 1])
                       for j in range(3)] for i in range(3)])
    from scipy.optimize import linear_sum_assignment
    linhas, colunas = linear_sum_assignment(-corr)
    prop_rec_alinhada = prop_rec[:, colunas]

    erro_abs = np.abs(prop_rec_alinhada - C_true)
    assert erro_abs.mean() < 0.10, (
        f"erro medio de proporcao acima da tolerancia: {erro_abs.mean():.4f}")


def test_mcr_als_respeita_nao_negatividade():
    D, _, _ = _mistura_sintetica(seed=2)
    res = mcr_als(D, n_componentes=3, nao_negativo_c=True, nao_negativo_s=True)
    assert (res.C >= -1e-9).all()
    assert (res.S >= -1e-9).all()


def test_mcr_als_normalizacao_soma_unitaria_aplicada():
    D, _, _ = _mistura_sintetica(seed=3)
    res = mcr_als(D, n_componentes=3, normalizacao="soma_unitaria")
    somas = res.S.sum(axis=1)
    np.testing.assert_allclose(somas, np.ones(3), atol=1e-6)


def test_mcr_als_reporta_aviso_de_ambiguidade_rotacional():
    D, _, _ = _mistura_sintetica(seed=4)
    res = mcr_als(D, n_componentes=3)
    assert "nao tem solucao unica" in res.aviso_ambiguidade_rotacional


def test_mcr_als_lof_diminui_ou_estabiliza_ao_longo_das_iteracoes():
    D, _, _ = _mistura_sintetica(seed=5)
    res = mcr_als(D, n_componentes=3, max_iter=50)
    hist = np.array(res.historico_lof)
    # Tolerancia pequena: ALS pode oscilar residualmente entre passos C/S,
    # mas a tendencia geral tem que ser de queda (compara inicio com fim).
    assert hist[-1] <= hist[0] + 1e-6


def test_mcr_als_rejeita_matriz_toda_zero():
    D = np.zeros((10, 20))
    with pytest.raises(ValueError):
        mcr_als(D, n_componentes=2)


def test_mcr_als_rejeita_n_componentes_invalido():
    D, _, _ = _mistura_sintetica(seed=6)
    with pytest.raises(ValueError):
        mcr_als(D, n_componentes=0)


def test_avaliar_incerteza_rotacional_estrutura_do_retorno():
    D, _, _ = _mistura_sintetica(seed=7, n=20, p=40)
    diag = avaliar_incerteza_rotacional(D, n_componentes=3, n_inicializacoes=3, seed=0)

    assert len(diag["resultados"]) == 3
    assert diag["desvio_padrao_proporcao"].shape == (20, 3)
    assert diag["desvio_padrao_medio"] >= 0.0
    assert "MCR-BANDS" in diag["aviso"]


def test_avaliar_incerteza_rotacional_baixa_para_mistura_bem_separada():
    # Componentes bem separados espectralmente (picos distantes, pouco
    # overlap) -- espera-se baixa sensibilidade a' inicializacao.
    D, _, _ = _mistura_sintetica(seed=8, n=25, p=60, ruido=0.001)
    diag = avaliar_incerteza_rotacional(D, n_componentes=3, n_inicializacoes=5, seed=1)
    assert diag["desvio_padrao_medio"] < 0.15


# ── mcr_als_com_restricao_correlacao (T9, rodada multiagente 2026-09-10) ──

def test_restricao_correlacao_ancora_componente_na_calibracao():
    D, C_true, _S_true = _mistura_sintetica(seed=10, n=30, p=60)
    y_ref = C_true[:, 0]   # teor "de referencia" do componente 0
    idx_calib = np.arange(len(y_ref))

    res = mcr_als_com_restricao_correlacao(
        D, n_componentes=3, indice_componente_alvo=0,
        y_referencia=y_ref, indices_calibracao=idx_calib)

    assert res.correlacao_calibracao == pytest.approx(1.0, abs=0.05)
    assert np.isfinite(res.coef_regressao[0])
    assert np.isfinite(res.coef_regressao[1])


def test_restricao_correlacao_predicao_a_partir_da_reta():
    """A reta ancorada (a, b) tem que servir para PREVER y de uma amostra
    fora da calibracao a partir do C que o MCR-ALS lhe atribuir --
    y_pred = (c - a) / b deve ficar perto do y verdadeiro."""
    D, C_true, _S_true = _mistura_sintetica(seed=11, n=40, p=60)
    idx_calib = np.arange(30)          # 30 primeiras p/ calibracao
    idx_teste = np.arange(30, 40)      # 10 restantes, fora da restricao
    y_ref_calib = C_true[idx_calib, 0]

    res = mcr_als_com_restricao_correlacao(
        D, n_componentes=3, indice_componente_alvo=0,
        y_referencia=y_ref_calib, indices_calibracao=idx_calib)

    a, b = res.coef_regressao
    c_teste = res.C[idx_teste, 0]
    y_pred = (c_teste - a) / b
    y_true = C_true[idx_teste, 0]
    erro = np.abs(y_pred - y_true)
    assert erro.mean() < 0.15   # tolerancia generosa (mistura sintetica com ruido)


def test_restricao_correlacao_nao_forca_os_demais_componentes():
    """So o componente ALVO e' ancorado -- os outros dois continuam livres
    (nao ficam artificialmente correlacionados com y_ref por construcao)."""
    D, C_true, _S_true = _mistura_sintetica(seed=12, n=30, p=60)
    y_ref = C_true[:, 0]
    idx_calib = np.arange(len(y_ref))

    res = mcr_als_com_restricao_correlacao(
        D, n_componentes=3, indice_componente_alvo=0,
        y_referencia=y_ref, indices_calibracao=idx_calib)

    # componentes 1 e 2 nao precisam estar perfeitamente correlacionados
    # com y_ref (que e' o teor do componente 0) -- diferente do alvo.
    corr_outro = abs(float(np.corrcoef(res.C[:, 1], y_ref)[0, 1]))
    assert corr_outro < 0.999   # nao e' 1.0 "de graca" como o alvo e'


def test_restricao_correlacao_respeita_nao_negatividade():
    D, C_true, _S_true = _mistura_sintetica(seed=13, n=30, p=60)
    y_ref = C_true[:, 0]
    idx_calib = np.arange(len(y_ref))
    res = mcr_als_com_restricao_correlacao(
        D, n_componentes=3, indice_componente_alvo=0,
        y_referencia=y_ref, indices_calibracao=idx_calib,
        nao_negativo_c=True)
    assert (res.C >= -1e-9).all()


def test_restricao_correlacao_indice_alvo_fora_do_intervalo():
    D, _, _ = _mistura_sintetica(seed=14, n=10, p=20)
    with pytest.raises(ValueError, match="indice_componente_alvo"):
        mcr_als_com_restricao_correlacao(
            D, n_componentes=3, indice_componente_alvo=5,
            y_referencia=np.zeros(10), indices_calibracao=np.arange(10))


def test_restricao_correlacao_exige_mesmo_comprimento():
    D, _, _ = _mistura_sintetica(seed=15, n=10, p=20)
    with pytest.raises(ValueError, match="comprimento"):
        mcr_als_com_restricao_correlacao(
            D, n_componentes=3, indice_componente_alvo=0,
            y_referencia=np.zeros(5), indices_calibracao=np.arange(10))


def test_restricao_correlacao_resultado_carrega_aviso_de_ambiguidade():
    """A variante supervisionada NAO fica isenta do aviso de ambiguidade
    rotacional -- ancorar 1 componente nao resolve a ambiguidade dos
    outros n_componentes-1."""
    D, C_true, _S_true = _mistura_sintetica(seed=16, n=20, p=40)
    y_ref = C_true[:, 0]
    res = mcr_als_com_restricao_correlacao(
        D, n_componentes=3, indice_componente_alvo=0,
        y_referencia=y_ref, indices_calibracao=np.arange(20))
    assert "nao tem solucao unica" in res.aviso_ambiguidade_rotacional.lower()
