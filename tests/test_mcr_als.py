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

from guaraci.mcr_als import mcr_als, avaliar_incerteza_rotacional


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
