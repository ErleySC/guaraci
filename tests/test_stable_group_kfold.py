# -*- coding: utf-8 -*-
"""Testes de `StableStratifiedGroupKFold` (partição congelada).

Achado da rodada de mutação (Passo 219): 23 de 72 mutantes do splitter
sobreviveram -- nenhum teste fixava a PARTIÇÃO em si (só propriedades
gerais), embora a promessa da classe seja justamente "mesmo
`(y, groups, n_splits, seed)` => mesma partição, em qualquer versão".
Estes testes fixam a partição por valores literais (golden) e as
fronteiras de validação.
"""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.validacao_estatistica import StableStratifiedGroupKFold

_TAM = [3, 3, 2, 2, 2, 1, 3, 2, 1, 2, 3, 1]
_CLS = ["a", "a", "a", "b", "b", "b", "c", "c", "c", "a", "b", "c"]


def _dados():
    g, y = [], []
    for i, (t, c) in enumerate(zip(_TAM, _CLS)):
        g += [f"g{i}"] * t
        y += [c] * t
    return np.array(y), np.array(g)


def _fold_por_amostra(n_splits, seed):
    y, g = _dados()
    f = np.full(len(y), -1)
    for k, (_, va) in enumerate(
            StableStratifiedGroupKFold(n_splits, seed).split(np.zeros(len(y)), y, groups=g)):
        f[va] = k
    return f.tolist()


# Valores gerados com a implementacao atual (2026-09-19) e CONGELADOS: se
# mudarem, a reprodutibilidade entre versoes -- razao de existir da classe --
# foi quebrada.
_GOLDEN_SEED42 = [0, 0, 0, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 0, 0, 0, 1, 1, 2, 2, 2, 0, 0, 0, 2]
_GOLDEN_SEED0 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2, 0, 0, 0, 2]


def test_particao_congelada_seed_42():
    assert _fold_por_amostra(3, 42) == _GOLDEN_SEED42


def test_particao_congelada_seed_0_difere_da_42():
    assert _fold_por_amostra(3, 0) == _GOLDEN_SEED0
    assert _GOLDEN_SEED0 != _GOLDEN_SEED42       # o seed realmente importa


def test_seed_default_e_42_e_n_splits_default_e_5():
    s = StableStratifiedGroupKFold()
    assert (s.n_splits, s.seed) == (5, 42)
    assert s.get_n_splits() == 5


def test_hash_de_desempate_congelado():
    """blake2b(seed:gid, 8 bytes) -- digest_size ou formato diferentes
    mudariam todas as particoes com empate."""
    ch = StableStratifiedGroupKFold._chave_ordem
    assert ch("g1", 42) == "fc1c70aa1c1340be"
    assert ch("g1", 7) == "d6059b68c8ddd6ff"
    assert ch(3, 42) == "eccac922b758abd1"
    assert len(ch("x", 1)) == 16


def test_replicas_do_mesmo_grupo_nunca_se_separam_e_cada_fold_tem_validacao():
    y, g = _dados()
    for tr, va in StableStratifiedGroupKFold(3, 42).split(np.zeros(len(y)), y, groups=g):
        assert not (set(g[tr]) & set(g[va]))
        assert len(va) > 0 and len(tr) + len(va) == len(y)


def test_n_splits_2_e_valido_e_menor_que_2_falha():
    assert StableStratifiedGroupKFold(n_splits=2).n_splits == 2
    for ruim in (1, 0, -3):
        with pytest.raises(ValueError, match="n_splits deve ser >= 2"):
            StableStratifiedGroupKFold(n_splits=ruim)


def test_groups_obrigatorio():
    with pytest.raises(ValueError, match="exige `groups`"):
        list(StableStratifiedGroupKFold(2).split(np.zeros(4), np.array([0, 0, 1, 1])))


def test_tamanhos_diferentes_de_y_e_groups_falham_nos_dois_sentidos():
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="diferem em tamanho"):
        list(StableStratifiedGroupKFold(2).split(np.zeros(4), y, groups=np.array([1, 2, 3])))
    with pytest.raises(ValueError, match="diferem em tamanho"):
        list(StableStratifiedGroupKFold(2).split(np.zeros(4), y, groups=np.array([1, 2, 3, 4, 5])))


def test_n_splits_igual_ao_numero_de_grupos_e_aceito_e_maior_falha():
    y = np.array([0, 0, 1, 1])
    g = np.array([1, 1, 2, 2])
    folds = list(StableStratifiedGroupKFold(2).split(np.zeros(4), y, groups=g))
    assert len(folds) == 2
    with pytest.raises(ValueError, match="numero de grupos"):
        list(StableStratifiedGroupKFold(3).split(np.zeros(4), y, groups=g))


def test_empate_de_custo_fica_no_fold_de_menor_indice():
    """Grupos identicos, folds vazios: o 1o grupo (menor hash) tem que ir
    ao fold 0 (empate resolvido por `<` estrito)."""
    y = np.array([0, 0, 0])
    g = np.array(["a", "b", "c"])
    fold = {}
    for k, (_, va) in enumerate(StableStratifiedGroupKFold(3, 1).split(np.zeros(3), y, groups=g)):
        for i in va:
            fold[g[i]] = k
    primeiro = min("abc", key=lambda gid: StableStratifiedGroupKFold._chave_ordem(gid, 1))
    assert fold[primeiro] == 0
    assert sorted(fold.values()) == [0, 1, 2]


def test_particao_congelada_em_caso_onde_o_melhor_fold_nao_e_o_ultimo():
    """Pequeno caso em que o custo NAO e' monotono nos folds: o fold
    escolhido tem que ser o de MENOR custo (`<`), nao 'o ultimo diferente'
    (mutante `!=`)."""
    tam = [2, 3, 1, 2, 1, 1, 3, 3]
    cls = [2, 0, 2, 2, 0, 0, 2, 2]
    g = np.repeat(np.arange(len(tam)), tam)
    y = np.repeat(cls, tam)
    f = np.full(len(y), -1)
    for k, (_, va) in enumerate(
            StableStratifiedGroupKFold(3, 42).split(np.zeros(len(y)), y, groups=g)):
        f[va] = k
    assert f.tolist() == [2, 2, 0, 0, 0, 0, 2, 2, 2, 1, 0, 0, 0, 1, 1, 1]
