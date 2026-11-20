# -*- coding: utf-8 -*-
"""Kennard-Stone e DUPLEX com semente EXATA (par mais distante), contra a forma de livro com a matriz de
distancias completa (scipy pdist). Ate a 1.0.0 a semente era aproximada pelo centroide e o conjunto mudava
em 45 de 200 sorteios (limitacoes.md, secao 8)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from guaraci.dados_io import _par_mais_distante, duplex_split, kennard_stone, kennard_stone_split


def _ks_livro(X: np.ndarray, k: int) -> np.ndarray:
    D = squareform(pdist(X))
    n = len(X)
    iu = np.triu_indices(n, 1)
    d = D[iu]
    i, j = min((a, b) for a, b, v in zip(*iu, d) if v == d.max())
    sel = [int(i), int(j)]
    md = np.minimum(D[i], D[j])
    while len(sel) < k:
        c = md.copy()
        c[sel] = -np.inf
        p = int(np.argmax(c))
        sel.append(p)
        md = np.minimum(md, D[p])
    return np.array(sel[:k])


@pytest.mark.parametrize("semente", range(40))
def test_ks_igual_ao_algoritmo_de_livro(semente):
    rng = np.random.default_rng(semente)
    n, p = int(rng.integers(5, 90)), int(rng.integers(2, 40))
    X = rng.normal(size=(n, p)) * rng.uniform(0.01, 1000)
    if semente % 4 == 0:
        X = np.round(X / X.std())          # muitos empates
    np.testing.assert_array_equal(kennard_stone(X, n), _ks_livro(X, n))


def test_par_mais_distante_exato_com_deslocamento_grande_e_blocos_pequenos():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(700, 8)) + 1e6      # cancelamento na identidade |a|^2+|b|^2-2ab
    D = squareform(pdist(X))
    i, j = np.unravel_index(np.argmax(np.triu(D)), D.shape)
    assert _par_mais_distante(X, bloco=64) == (int(min(i, j)), int(max(i, j)))


def test_dados_identicos_nao_explodem_memoria():
    assert _par_mais_distante(np.ones((4000, 20))) == (0, 1)
    np.testing.assert_array_equal(kennard_stone(np.ones((5, 2)), 5), np.arange(5))


def test_conjunto_nao_depende_da_ordem_das_linhas_sem_empate():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(60, 12))
    perm = rng.permutation(60)
    tr1, _ = kennard_stone_split(X, 0.7)
    tr2, _ = kennard_stone_split(X[perm], 0.7)
    assert set(tr1.tolist()) == set(perm[tr2].tolist())


def test_duplex_semente_exata_nos_dois_conjuntos():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 6))
    tr, va = duplex_split(X, 0.5)
    i, j = _par_mais_distante(X)
    assert {i, j} <= set(tr.tolist())
    resto = np.array(sorted(set(range(40)) - {i, j}))
    a, b = _par_mais_distante(X[resto])
    assert {int(resto[a]), int(resto[b])} <= set(va.tolist())
