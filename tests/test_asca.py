# -*- coding: utf-8 -*-
"""Testes de asca.py -- proposta T3 da rodada multiagente de 2026-09-10.

Referências: Smilde et al. (2005), Bioinformatics 21:3043-3048,
DOI 10.1093/bioinformatics/bti476; Thiel, Féraud & Govaerts (2017),
J. Chemometrics 31:e2895, DOI 10.1002/cem.2895 (ASCA+, não implementada
aqui -- ver docstring do módulo).
"""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.asca import asca_decompose, asca_permutation_test


def _dataset_2_fatores_balanceado(seed=0, n_por_celula=10, p=20):
    """Delineamento 2x2 BALANCEADO e ORTOGONAL: fator A domina o sinal,
    fator B nao tem efeito nenhum -- exatamente o cenario "especie explica
    muito mais que adulterante" do achado real do projeto."""
    rng = np.random.default_rng(seed)
    niveis_a = np.repeat(["a1", "a2"], n_por_celula * 2)
    niveis_b = np.tile(np.repeat(["b1", "b2"], n_por_celula), 2)
    n = len(niveis_a)
    efeito_a = np.where(niveis_a == "a1", 5.0, -5.0)
    X = rng.normal(scale=0.3, size=(n, p))
    X += efeito_a[:, None]   # so' fator A tem efeito real
    return X, {"A": niveis_a, "B": niveis_b}


def test_decompose_recupera_fator_dominante_e_fator_nulo():
    X, fatores = _dataset_2_fatores_balanceado()
    dec = asca_decompose(X, fatores)
    ss_a = dec["efeitos"]["A"]["soma_quadrados"]
    ss_b = dec["efeitos"]["B"]["soma_quadrados"]
    assert ss_a > 50 * ss_b   # A domina, B e' essencialmente ruido


def test_decompose_aditividade_exata_com_fatores_ortogonais():
    """Delineamento balanceado/ortogonal: ss_total == soma dos efeitos +
    residuo, sem sobra -- o proprio ss_desbalanco prova isso."""
    X, fatores = _dataset_2_fatores_balanceado()
    dec = asca_decompose(X, fatores)
    assert dec["ss_desbalanco"] == pytest.approx(0.0, abs=1e-6)
    soma = sum(e["soma_quadrados"] for e in dec["efeitos"].values())
    assert dec["ss_total"] == pytest.approx(soma + dec["ss_residuo"], rel=1e-6)


def test_decompose_fracao_ss_total_entre_0_e_1():
    X, fatores = _dataset_2_fatores_balanceado()
    dec = asca_decompose(X, fatores)
    for efeito in dec["efeitos"].values():
        assert 0.0 <= efeito["fracao_ss_total"] <= 1.0


def test_decompose_exige_mesmo_n_de_amostras():
    X = np.zeros((10, 5))
    with pytest.raises(ValueError, match="rótulos"):
        asca_decompose(X, {"A": np.array(["a"] * 5)})


def test_decompose_recusa_fatores_vazio():
    X = np.zeros((10, 5))
    with pytest.raises(ValueError, match="vazio"):
        asca_decompose(X, {})


def test_desbalanco_cresce_com_fatores_correlacionados():
    """Contra-prova do limite documentado: quando os dois fatores sao
    QUASE o mesmo rotulo (fortemente correlacionados/confundidos), o
    ss_desbalanco deixa de ser ~0 -- o sinal de que ASCA+ seria
    necessario."""
    rng = np.random.default_rng(1)
    n, p = 60, 15
    niveis_a = np.repeat(["a1", "a2", "a3"], 20)
    # B quase identico a A (confundido), so' 3 trocas
    niveis_b = niveis_a.copy()
    niveis_b[[0, 20, 40]] = np.roll(np.unique(niveis_a), 1)[[0, 1, 2]]
    efeito_a = {"a1": 5.0, "a2": -3.0, "a3": 1.0}
    X = rng.normal(scale=0.2, size=(n, p))
    X += np.array([efeito_a[a] for a in niveis_a])[:, None]

    dec = asca_decompose(X, {"A": niveis_a, "B": niveis_b})
    assert abs(dec["ss_desbalanco"]) > 1e-6


# ── asca_permutation_test ────────────────────────────────────────────────

def test_permutation_fator_dominante_da_p_valor_baixo():
    X, fatores = _dataset_2_fatores_balanceado(n_por_celula=15)
    r = asca_permutation_test(X, fatores, n_perm=200, seed=0)
    assert r["A"]["p_value"] < 0.01
    assert r["B"]["p_value"] > 0.05   # fator nulo nao deveria "significar"


def test_permutation_por_unidade_evita_pseudo_replicacao():
    """Sem unidade de permutacao, replicas TECNICAS contam como
    independentes e inflam a significancia de um fator sem efeito real
    (mesmo viés documentado em validacao_estatistica.permutation_test).
    Com a unidade certa, o p-valor do fator nulo fica bem menos
    otimista."""
    rng = np.random.default_rng(7)
    n_grupos, n_replicas, p = 10, 4, 15
    grupos = np.repeat(np.arange(n_grupos), n_replicas)
    # "fator" sorteado por GRUPO (nao por replica) -- nao tem efeito real
    nivel_por_grupo = rng.choice(["x", "y"], size=n_grupos)
    niveis = nivel_por_grupo[grupos]
    X = rng.normal(scale=1.0, size=(n_grupos, p))[grupos]   # so' ruido por grupo
    X += rng.normal(scale=0.05, size=(n_grupos * n_replicas, p))  # + ruido fino por replica

    r_sem_unidade = asca_permutation_test(X, {"F": niveis}, n_perm=300, seed=1)
    r_com_unidade = asca_permutation_test(
        X, {"F": niveis}, unidade_permutacao=grupos, n_perm=300, seed=1)

    # com unidade correta, a permutacao usa so' 10 "moedas" (grupos) --
    # a distribuicao nula fica mais dispersa/realista; sem unidade, 40
    # "moedas" (replicas) dao uma nula artificialmente estreita.
    assert r_com_unidade["F"]["ss_permutado"].std() >= r_sem_unidade["F"]["ss_permutado"].std()


def test_permutation_p_value_nunca_e_zero():
    """Correcao +1/+1: p-valor nunca sai exatamente 0, mesmo com efeito
    muito forte -- e' a convencao padrao pra' teste de permutacao com
    numero finito de reamostragens (Davison & Hinkley 1997)."""
    X, fatores = _dataset_2_fatores_balanceado(n_por_celula=15)
    r = asca_permutation_test(X, fatores, n_perm=50, seed=0)
    assert r["A"]["p_value"] > 0.0
