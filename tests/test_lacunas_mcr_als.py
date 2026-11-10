"""Ramos de borda de guaraci.mcr_als (inicializacao, restricoes, normalizacao, entrada invalida): um teste por ramo."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from guaraci import mcr_als as m


def _mistura(n=18, p=30, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, p)
    S = np.vstack([np.exp(-((x - 0.3) / 0.08) ** 2), np.exp(-((x - 0.7) / 0.08) ** 2)])
    C = np.column_stack([np.linspace(0.1, 1.0, n), np.linspace(1.0, 0.1, n)])
    return C @ S + rng.normal(scale=1e-4, size=(n, p)), C


def test_inicializacao_de_dados_todo_zero_nao_normaliza():
    assert not m._init_variaveis_puras(np.zeros((4, 5)), 2).any()


def test_minimos_quadrados_sem_restricao_resolve_por_lstsq():
    base = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    X = m._passo_minimos_quadrados(base @ np.array([[2.0, -1.0], [0.5, 3.0]]), base, nao_negativo=False)
    np.testing.assert_allclose(X, [[2.0, -1.0], [0.5, 3.0]], atol=1e-9)           # coeficientes negativos so' existem sem a restricao


def test_normalizacao_por_norma_unitaria_e_modo_desconhecido():
    S = np.array([[3.0, 4.0], [0.0, 2.0]])
    C = np.ones((3, 2))
    C2, S2 = m._normalizar_S(S, C, "norma_unitaria")
    np.testing.assert_allclose(np.linalg.norm(S2, axis=1), 1.0)
    np.testing.assert_allclose(C2 @ S2, C @ S)                                    # a escala passa para C: o produto nao muda
    with pytest.raises(ValueError, match="desconhecido"):
        m._normalizar_S(S, C, "outra")


def test_unimodalidade_torna_o_perfil_nao_crescente_dos_dois_lados_do_pico():
    C = np.array([[1.0], [3.0], [2.0], [5.0], [4.0], [6.0], [2.0]])
    out = m._aplicar_unimodalidade(C)[:, 0]
    pico = int(np.argmax(out))
    assert np.all(np.diff(out[:pico + 1]) >= 0) and np.all(np.diff(out[pico:]) <= 0)


def test_mcr_als_com_unimodalidade_com_ponto_de_partida_em_s_e_sem_restricao_de_sinal():
    D, C = _mistura()
    S0 = np.vstack([np.exp(-((np.linspace(0, 1, 30) - c) / 0.1) ** 2) for c in (0.3, 0.7)])
    r = m.mcr_als(D, 2, unimodal_c=True, s_inicial=S0, nao_negativo_c=False, nao_negativo_s=False, max_iter=20)
    assert r.C.shape == (18, 2) and r.S.shape == (2, 30)


def test_mcr_als_recusa_entrada_invalida():
    with pytest.raises(ValueError, match="2D"):
        m.mcr_als(np.ones(5), 1)
    with pytest.raises(ValueError, match="todo zero"):
        m.mcr_als(np.zeros((4, 4)), 1)


def _kw(D, C, **extra):
    return dict(indice_componente_alvo=0, y_referencia=C[:6, 0], indices_calibracao=np.arange(6), **extra)


def test_restricao_de_correlacao_recusa_entrada_invalida():
    D, C = _mistura()
    with pytest.raises(ValueError, match="2D"):
        m.mcr_als_com_restricao_correlacao(np.ones(5), 2, **_kw(D, C))
    with pytest.raises(ValueError, match=">= 1"):
        m.mcr_als_com_restricao_correlacao(D, 0, **_kw(D, C))
    with pytest.raises(ValueError, match=">=2 amostras"):
        m.mcr_als_com_restricao_correlacao(D, 2, indice_componente_alvo=0, y_referencia=C[:1, 0], indices_calibracao=np.array([0]))
    with pytest.raises(ValueError, match="todo zero"):
        m.mcr_als_com_restricao_correlacao(np.zeros((18, 30)), 2, **_kw(D, C))


def test_restricao_de_correlacao_com_s_inicial_unimodal_e_sem_nao_negatividade():
    D, C = _mistura()
    S0 = np.vstack([np.exp(-((np.linspace(0, 1, 30) - c) / 0.1) ** 2) for c in (0.3, 0.7)])
    r = m.mcr_als_com_restricao_correlacao(D, 2, **_kw(D, C, s_inicial=S0, unimodal_c=True, nao_negativo_c=False), max_iter=15)
    assert np.isfinite(r.C).all()


def test_restricao_de_correlacao_com_referencia_constante_nao_tem_correlacao():
    D, C = _mistura()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                                          # polyfit avisa mal condicionamento: e' o caso testado
        r = m.mcr_als_com_restricao_correlacao(D, 2, indice_componente_alvo=0, y_referencia=np.ones(6), indices_calibracao=np.arange(6), max_iter=10)
    assert np.isnan(r.correlacao_calibracao)
