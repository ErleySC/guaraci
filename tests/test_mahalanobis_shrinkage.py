# -*- coding: utf-8 -*-
"""Testes de `chemometric_stats.mahalanobis_distance_shrinkage` -- proposta
T7 da rodada multiagente de 2026-09-10.

Referência: Ledoit & Wolf (2004), J. Multivariate Anal. 88:365-411,
DOI 10.1016/S0047-259X(03)00096-4.
"""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.chemometric_stats import mahalanobis_distance_shrinkage


def test_distancia_zero_para_classes_com_mesma_media():
    rng = np.random.default_rng(0)
    X_a = rng.normal(size=(30, 5))
    X_b = rng.normal(size=(30, 5))
    d = mahalanobis_distance_shrinkage(X_a, X_b)
    assert d >= 0
    assert d < 3.0   # mesma media populacional -- distancia pequena, nao zero exato


def test_distancia_grande_para_classes_bem_separadas():
    rng = np.random.default_rng(1)
    X_a = rng.normal(loc=0.0, size=(30, 5))
    X_b = rng.normal(loc=10.0, size=(30, 5))
    d = mahalanobis_distance_shrinkage(X_a, X_b)
    assert d > 3.0


def test_shrinkage_fica_mais_estavel_que_raw_com_dimensao_alta_e_n_pequeno():
    """Contra-prova central (achado do Passo 112): a distancia RAW infla
    so' por mal-condicionamento ao acrescentar dimensoes de RUIDO PURO
    (sem separacao real); a distancia SHRINKAGE deve subir bem menos."""
    rng = np.random.default_rng(2)
    n_por_classe = 15   # poucas amostras, mesmo espirito do Passo 112 (n~30-40)

    def _distancias(n_dim_ruido):
        base_a = rng.normal(loc=0.0, size=(n_por_classe, 2))
        base_b = rng.normal(loc=1.5, size=(n_por_classe, 2))
        ruido_a = rng.normal(size=(n_por_classe, n_dim_ruido))
        ruido_b = rng.normal(size=(n_por_classe, n_dim_ruido))
        X_a = np.hstack([base_a, ruido_a])
        X_b = np.hstack([base_b, ruido_b])
        d_raw = mahalanobis_distance_shrinkage(X_a, X_b, estimador="raw")
        d_shr = mahalanobis_distance_shrinkage(X_a, X_b, estimador="shrinkage")
        return d_raw, d_shr

    d_raw_baixa, d_shr_baixa = _distancias(n_dim_ruido=0)
    d_raw_alta, d_shr_alta = _distancias(n_dim_ruido=12)   # dimensao total 14, n=15

    aumento_raw = d_raw_alta - d_raw_baixa
    aumento_shr = d_shr_alta - d_shr_baixa
    assert aumento_raw > 0   # a raw realmente infla
    assert aumento_shr < aumento_raw   # a shrinkage infla bem menos


def test_estimador_invalido_levanta_erro_explicito():
    X_a = np.zeros((10, 3))
    X_b = np.ones((10, 3))
    with pytest.raises(ValueError, match="estimador"):
        mahalanobis_distance_shrinkage(X_a, X_b, estimador="inexistente")


def test_funciona_com_1_variavel():
    X_a = np.array([[1.0], [1.1], [0.9], [1.05]])
    X_b = np.array([[5.0], [5.2], [4.8], [5.1]])
    d = mahalanobis_distance_shrinkage(X_a, X_b)
    assert d > 0 and np.isfinite(d)
