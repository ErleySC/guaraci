# -*- coding: utf-8 -*-
"""Testes de CARS/UVE em selecao_variaveis.py (Bloco 17) -- dado SINTETICO
group-aware (replicas do mesmo grupo/objeto fisico nunca separadas entre
treino/validacao), com um subconjunto conhecido de variaveis informativas
em meio a ruido puro.
"""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.selecao_variaveis import (
    cars_selecao,
    uve_selecao,
    estabilidade_selecao_entre_repeticoes,
    _avaliar_busca_nested_cv,
    _avaliar_subset_nested_cv,
    _mask_vip_threshold,
    _mask_melhor_intervalo,
    _edf_contagens,
)
from guaraci.validacao_estatistica import StableStratifiedGroupKFold


def _dataset_com_variaveis_informativas(seed=0, n_grupos=24, replicas=3, p=40,
                                        n_informativas=3, deslocamento=2.5):
    """n_grupos objetos fisicos (mae_id), cada um com `replicas` medidas
    (mesmo objeto -- nunca podem ser separadas entre treino/validacao).
    `n_informativas` variaveis carregam sinal de classe; o resto e' ruido
    gaussiano puro."""
    rng = np.random.default_rng(seed)
    n = n_grupos * replicas
    y_grupo = rng.integers(0, 2, size=n_grupos)
    y = np.repeat(y_grupo, replicas)
    grupos = np.repeat(np.arange(n_grupos), replicas)

    X = rng.normal(0, 1, size=(n, p))
    idx_informativas = np.sort(rng.choice(p, size=n_informativas, replace=False))
    X[:, idx_informativas] += (y[:, None] * deslocamento)

    Y_bin = np.eye(2)[y]
    return X, Y_bin, y, grupos, idx_informativas


def _cv_group_aware(y, grupos, n_splits=4, seed=0):
    return list(StableStratifiedGroupKFold(n_splits=n_splits, seed=seed)
                .split(np.zeros(len(y)), y, groups=grupos))


# ---------------------------------------------------------------------------
#  EDF (funcao exponencialmente decrescente do CARS)
# ---------------------------------------------------------------------------

def test_edf_contagens_decresce_de_p_a_2():
    contagens = _edf_contagens(40, 15)
    assert contagens[0] == 40
    assert contagens[-1] == 2
    assert (np.diff(contagens) <= 0).all(), "EDF precisa ser monotona nao-crescente"


def test_edf_contagens_uma_iteracao():
    contagens = _edf_contagens(40, 1)
    assert list(contagens) == [40]


# ---------------------------------------------------------------------------
#  CARS
# ---------------------------------------------------------------------------

def test_cars_recupera_maioria_das_variaveis_informativas():
    X, Y_bin, y, grupos, idx_info = _dataset_com_variaveis_informativas(seed=1)
    cv_indices = _cv_group_aware(y, grupos, seed=1)

    historico, mask = cars_selecao(X, Y_bin, y, cv_indices, n_lv=2,
                                    n_iteracoes=20, frac_amostragem=0.8, seed=1)

    assert mask.sum() < X.shape[1], "CARS precisa reduzir o numero de variaveis"
    recall = mask[idx_info].sum() / len(idx_info)
    assert recall >= 2 / 3, (
        f"CARS recuperou so' {mask[idx_info].sum()}/{len(idx_info)} "
        f"variaveis informativas")
    assert len(historico) == 20
    assert all(h["balanced_accuracy"] >= 0.0 for h in historico)


def test_cars_reduz_numero_de_variaveis_ao_longo_das_iteracoes():
    X, Y_bin, y, grupos, _idx_info = _dataset_com_variaveis_informativas(seed=2)
    cv_indices = _cv_group_aware(y, grupos, seed=2)
    historico, _mask = cars_selecao(X, Y_bin, y, cv_indices, n_lv=2,
                                     n_iteracoes=15, frac_amostragem=0.8, seed=2)
    n_vars = [h["n_vars"] for h in historico]
    assert n_vars[0] > n_vars[-1]
    assert n_vars[-1] >= 2


# ---------------------------------------------------------------------------
#  UVE
# ---------------------------------------------------------------------------

def test_uve_recupera_maioria_das_variaveis_informativas():
    X, Y_bin, y, grupos, idx_info = _dataset_com_variaveis_informativas(seed=3)

    info, mask = uve_selecao(X, Y_bin, n_lv=2, n_repeticoes=40,
                              frac_amostragem=0.8, seed=3)

    assert mask.sum() < X.shape[1], "UVE precisa eliminar alguma variavel"
    recall = mask[idx_info].sum() / len(idx_info)
    assert recall >= 2 / 3, (
        f"UVE recuperou so' {mask[idx_info].sum()}/{len(idx_info)} "
        f"variaveis informativas")
    assert info["corte"] >= 0.0
    assert info["c_scores"].shape == (X.shape[1],)


def test_uve_corte_vem_das_variaveis_de_ruido_nao_das_reais():
    # Dataset SO' ruido (sem nenhuma variavel informativa de verdade): UVE
    # nao deve reter quase nada -- o corte (max |c| do ruido artificial) e'
    # da MESMA distribuicao das variaveis reais nesse caso degenerado.
    rng = np.random.default_rng(4)
    n, p = 60, 30
    X = rng.normal(0, 1, size=(n, p))
    y = rng.integers(0, 2, size=n)
    Y_bin = np.eye(2)[y]
    _info, mask = uve_selecao(X, Y_bin, n_lv=2, n_repeticoes=40,
                               frac_amostragem=0.8, seed=4)
    # Sem sinal real, a fracao retida deve ser proxima da taxa de falso-
    # positivo esperada de um corte por maximo de p variaveis de ruido
    # (folga generosa: nao e' um teste de taxa exata, so' de ordem de
    # grandeza -- nao pode reter a maioria das variaveis).
    assert mask.mean() < 0.5


# ---------------------------------------------------------------------------
#  Regra 5 (group-aware): nested-CV nunca deixa a SELECAO ver o fold de
#  validacao -- propriedade verificada diretamente instrumentando as
#  funcoes de selecao com um espiao que registra o tamanho de X recebido.
# ---------------------------------------------------------------------------

def test_cars_nested_cv_nunca_ve_o_fold_de_validacao():
    X, Y_bin, y, grupos, _idx = _dataset_com_variaveis_informativas(seed=5)
    cv_indices = _cv_group_aware(y, grupos, seed=5)
    tamanhos_esperados = [len(tr) for tr, _va in cv_indices]

    vistos = []

    def buscar_fn(Xtr, Ytr, ytr, cv_interna):
        vistos.append(Xtr.shape[0])
        return cars_selecao(Xtr, Ytr, ytr, cv_interna, n_lv=2,
                            n_iteracoes=5, frac_amostragem=0.8, seed=5)[1]

    _avaliar_busca_nested_cv(X, Y_bin, y, cv_indices, n_lv=2,
                              buscar_fn=buscar_fn, seed=5, mae_id=grupos)

    assert vistos == tamanhos_esperados
    assert all(v < X.shape[0] for v in vistos), (
        "selecao viu o dataset inteiro em algum fold -- vazamento")


def test_uve_nested_cv_nunca_ve_o_fold_de_validacao():
    X, Y_bin, y, grupos, _idx = _dataset_com_variaveis_informativas(seed=6)
    cv_indices = _cv_group_aware(y, grupos, seed=6)
    tamanhos_esperados = [len(tr) for tr, _va in cv_indices]

    vistos = []

    def selecionar_fn(Xtr, Ytr, nlv):
        vistos.append(Xtr.shape[0])
        return uve_selecao(Xtr, Ytr, nlv, n_repeticoes=10,
                           frac_amostragem=0.8, seed=6)[1]

    _avaliar_subset_nested_cv(X, Y_bin, y, cv_indices, n_lv=2,
                              selecionar_fn=selecionar_fn)

    assert vistos == tamanhos_esperados
    assert all(v < X.shape[0] for v in vistos), (
        "selecao viu o dataset inteiro em algum fold -- vazamento")


# ---------------------------------------------------------------------------
#  Estabilidade entre repeticoes: CARS/UVE (estocasticos, amostragem Monte
#  Carlo) vs VIP/iPLS (o numero reportado no fold-nested nao muda por seed
#  externo -- VIP e' puramente determinístico, iPLS so' varia pela CV
#  interna que guia a escolha do melhor intervalo).
# ---------------------------------------------------------------------------

def test_estabilidade_cars_uve_menor_que_vip_deterministico():
    X, Y_bin, y, grupos, _idx = _dataset_com_variaveis_informativas(seed=7)
    cv_indices = _cv_group_aware(y, grupos, seed=7)

    est_cars = estabilidade_selecao_entre_repeticoes(
        lambda seed: cars_selecao(X, Y_bin, y, cv_indices, 2, 15, 0.8, seed)[1],
        n_repeticoes=5, seed_base=100)
    est_uve = estabilidade_selecao_entre_repeticoes(
        lambda seed: uve_selecao(X, Y_bin, 2, 30, 0.8, seed)[1],
        n_repeticoes=5, seed_base=200)
    est_vip = estabilidade_selecao_entre_repeticoes(
        lambda seed: _mask_vip_threshold(X, Y_bin, 2, 1.0),
        n_repeticoes=5, seed_base=300)

    assert est_vip["jaccard_medio"] == pytest.approx(1.0), (
        "VIP e' deterministico (mesmos dados, mesmo threshold) -- "
        "estabilidade tem que ser perfeita entre 'repeticoes'")
    assert est_cars["jaccard_medio"] < 1.0, (
        "CARS usa amostragem Monte Carlo -- alguma variabilidade e' esperada")
    assert est_uve["jaccard_medio"] < 1.0, (
        "UVE usa amostragem Monte Carlo -- alguma variabilidade e' esperada")


def test_estabilidade_ipls_varia_pela_cv_interna():
    X, Y_bin, y, _grupos, _idx = _dataset_com_variaveis_informativas(seed=8)
    est_ipls = estabilidade_selecao_entre_repeticoes(
        lambda seed: _mask_melhor_intervalo(X, Y_bin, 2, 5, seed),
        n_repeticoes=6, seed_base=400)
    # Nao afirma que sempre varia (poderia convergir pro mesmo intervalo em
    # todo seed) -- so' que a funcao produz um numero valido e as mascaras
    # tem o formato certo.
    assert 0.0 <= est_ipls["jaccard_medio"] <= 1.0
    assert len(est_ipls["n_vars_por_repeticao"]) == 6
