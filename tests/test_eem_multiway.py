# -*- coding: utf-8 -*-
"""Testes de eem_multiway.py (PARAFAC generalizado p/ EEM de
fluorescencia) -- Passo 144/145 da auditoria das 11 tecnicas
(2026-09-04). Contra-prova sintetica exigida pela instrucao: EEM
simulada como mistura de 2 componentes puros conhecidos, PARAFAC
precisa recuperar as proporcoes verdadeiras."""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.eem_multiway import ParafacEEMResultado, construir_tensor_eem, parafac_eem


def test_construir_tensor_eem_empilha_na_ordem_e_devolve_ids():
    m1 = np.arange(6).reshape(2, 3).astype(float)
    m2 = (np.arange(6).reshape(2, 3) * 2).astype(float)
    tensor, ids = construir_tensor_eem({"a1": m1, "a2": m2})
    assert tensor.shape == (2, 2, 3)
    assert ids == ["a1", "a2"]
    np.testing.assert_array_equal(tensor[0], m1)
    np.testing.assert_array_equal(tensor[1], m2)


def test_construir_tensor_eem_rejeita_dict_vazio():
    with pytest.raises(ValueError, match="vazio"):
        construir_tensor_eem({})


def test_construir_tensor_eem_rejeita_forma_inconsistente():
    m1 = np.zeros((2, 3))
    m2 = np.zeros((2, 4))   # grade de emissao diferente
    with pytest.raises(ValueError, match="forma"):
        construir_tensor_eem({"a1": m1, "a2": m2})


def _melhor_correlacao_absoluta(fator_amostra: np.ndarray, verdadeiro: np.ndarray) -> float:
    """PARAFAC tem ambiguidade de permutacao/sinal entre componentes --
    compara `verdadeiro` (1 coluna) contra CADA coluna recuperada e
    devolve a maior correlacao absoluta (o componente que -- em
    QUALQUER ordem/sinal -- melhor corresponde)."""
    melhores = []
    for k in range(fator_amostra.shape[1]):
        r = np.corrcoef(fator_amostra[:, k], verdadeiro)[0, 1]
        melhores.append(abs(r))
    return float(max(melhores))


def test_parafac_eem_recupera_2_componentes_puros_de_mistura_sintetica():
    """Contra-prova central: constroi uma EEM sintetica como mistura
    LINEAR de 2 pares (excitacao, emissao) puros CONHECIDOS, com
    proporcoes por amostra tambem conhecidas -- PARAFAC (n_componentes=2)
    tem que recuperar essas proporcoes por amostra com alta correlacao,
    mesmo sem nunca ver as proporcoes verdadeiras (nao-supervisionado)."""
    rng = np.random.default_rng(0)
    n_amostras, n_ex, n_em = 30, 12, 20

    eixo_ex = np.linspace(0, 1, n_ex)
    eixo_em = np.linspace(0, 1, n_em)

    # 2 perfis de excitacao/emissao puros, bem separados espectralmente
    # (picos em posicoes diferentes) -- premissa minima de identificabilidade
    # do PARAFAC (componentes genuinamente distintos, nao colineares).
    ex1 = np.exp(-0.5 * ((eixo_ex - 0.25) / 0.10) ** 2)
    ex2 = np.exp(-0.5 * ((eixo_ex - 0.75) / 0.10) ** 2)
    em1 = np.exp(-0.5 * ((eixo_em - 0.30) / 0.08) ** 2)
    em2 = np.exp(-0.5 * ((eixo_em - 0.70) / 0.08) ** 2)

    conc1 = rng.uniform(0.2, 1.0, n_amostras)
    conc2 = rng.uniform(0.2, 1.0, n_amostras)

    matrizes = {}
    for n in range(n_amostras):
        eem = (conc1[n] * np.outer(ex1, em1) + conc2[n] * np.outer(ex2, em2))
        eem += rng.normal(0, 0.01, eem.shape)   # ruido de medicao pequeno
        matrizes[f"amostra_{n:02d}"] = eem

    tensor, ids = construir_tensor_eem(matrizes)
    assert tensor.shape == (n_amostras, n_ex, n_em)

    resultado = parafac_eem(tensor, n_componentes=2, seed=0)
    assert isinstance(resultado, ParafacEEMResultado)
    assert resultado.fator_amostra.shape == (n_amostras, 2)
    assert resultado.fator_excitacao.shape == (n_ex, 2)
    assert resultado.fator_emissao.shape == (n_em, 2)

    assert resultado.erro_reconstrucao_relativo < 0.1, (
        f"erro de reconstrucao alto demais: {resultado.erro_reconstrucao_relativo:.4f} "
        f"-- PARAFAC nao ajustou a mistura sintetica simples")

    r1 = _melhor_correlacao_absoluta(resultado.fator_amostra, conc1)
    r2 = _melhor_correlacao_absoluta(resultado.fator_amostra, conc2)
    assert r1 > 0.9, f"componente 1 nao recuperado: melhor |r|={r1:.3f}"
    assert r2 > 0.9, f"componente 2 nao recuperado: melhor |r|={r2:.3f}"
