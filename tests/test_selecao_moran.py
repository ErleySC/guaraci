# -*- coding: utf-8 -*-
"""Testes do I de Moran (selecao geoestatistica de variavel) em
`selecao_variaveis.py` -- Passo 148 (Fase B, auditoria das 11 tecnicas,
2026-09-04). Referencia: Lamanna et al. (2017), Magnetic Resonance in
Chemistry 55(7):639-647, DOI 10.1002/mrc.4566 (confirmado no Crossref).

Contra-prova obrigatoria (regra 9 da instrucao): dado sintetico com
variaveis GEOGRAFICAS conhecidas (valor correlacionado com a coordenada
da amostra) misturadas com ruido puro (sem nenhuma relacao com
coordenada) -- o metodo tem que separar as duas categorias. Sem essa
contra-prova, "o I de Moran funciona" seria alegacao sem lastro (mesma
disciplina que pegou o bug do AirPLS)."""
from __future__ import annotations

import numpy as np

from guaraci.selecao_variaveis import (
    _pesos_knn,
    moran_i_mask,
    _avaliar_subset_nested_cv_moran,
)


def _dataset_geografico(seed=0, n_por_cluster=30, p=40, n_geograficas=4,
                         deslocamento=3.0):
    """2 clusters geograficos bem separados (coordenadas fake, unidades
    arbitrarias). `n_geograficas` variaveis tem media deslocada por
    cluster (autocorrelacao espacial real); o resto e' ruido gaussiano
    puro (sem relacao com posicao). Retorna (X, coords, idx_geograficas,
    y_cluster) -- y_cluster so' para diagnostico, o metodo NUNCA ve rotulo."""
    rng = np.random.default_rng(seed)
    n = 2 * n_por_cluster
    cluster = np.repeat([0, 1], n_por_cluster)
    coords = np.empty((n, 2))
    coords[cluster == 0] = rng.normal([0.0, 0.0], 0.3, size=(n_por_cluster, 2))
    coords[cluster == 1] = rng.normal([10.0, 10.0], 0.3, size=(n_por_cluster, 2))

    X = rng.normal(0, 1, size=(n, p))
    idx_geo = np.sort(rng.choice(p, size=n_geograficas, replace=False))
    X[:, idx_geo] += (cluster[:, None] * deslocamento)
    return X, coords, idx_geo, cluster


def test_pesos_knn_e_row_standardized_e_ignora_diagonal():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 10.0]])
    W = _pesos_knn(coords, k_vizinhos=2)
    assert W.shape == (4, 4)
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
    assert np.all(np.diag(W) == 0.0), "uma amostra nunca pode ser vizinha de si mesma"
    # amostra 0 (x=0) tem os 2 vizinhos mais proximos em x=1 e x=2, nunca x=10.
    assert W[0, 3] == 0.0
    assert W[0, 1] > 0.0 and W[0, 2] > 0.0


def test_moran_i_seleciona_variaveis_geograficas_e_rejeita_ruido():
    X, coords, idx_geo, _ = _dataset_geografico(seed=1)
    info, mask = moran_i_mask(X, coords, k_vizinhos=6, alpha=0.05,
                               n_permutacoes=299, seed=1)

    # Toda variavel geografica de verdade tem I de Moran claramente positivo.
    assert (info["moran_i"][idx_geo] > 0.3).all(), (
        f"I de Moran das variaveis geograficas deveria ser alto: "
        f"{info['moran_i'][idx_geo]}")

    # A mascara tem que PEGAR a maioria das variaveis geograficas reais...
    recuperadas = mask[idx_geo].sum()
    assert recuperadas >= len(idx_geo) - 1, (
        f"so' {recuperadas}/{len(idx_geo)} variaveis geograficas recuperadas")

    # ...e nao pode virar "seleciona tudo" (senao nao filtrou nada de verdade).
    idx_ruido = np.setdiff1d(np.arange(X.shape[1]), idx_geo)
    taxa_falso_positivo = mask[idx_ruido].mean()
    assert taxa_falso_positivo < 0.15, (
        f"taxa de falso-positivo em ruido puro alta demais: "
        f"{taxa_falso_positivo:.3f} (FDR deveria manter isso baixo)")


def test_moran_i_contraprova_embaralhar_coordenadas_destroi_o_sinal():
    """Contra-prova (regra 9): embaralhando as COORDENADAS (mantendo os
    valores espectrais intactos) destroi a correspondencia geografica --
    se o metodo continuasse selecionando as mesmas variaveis com a MESMA
    forca depois disso, ele estaria pegando alguma outra coisa (ex.: a
    ordem dos dados, um artefato de implementacao), nao autocorrelacao
    espacial de verdade."""
    X, coords, idx_geo, _ = _dataset_geografico(seed=2)
    info_real, mask_real = moran_i_mask(X, coords, k_vizinhos=6, alpha=0.05,
                                         n_permutacoes=299, seed=2)

    rng = np.random.default_rng(99)
    coords_embaralhadas = coords[rng.permutation(len(coords))]
    info_embaralhado, mask_embaralhado = moran_i_mask(
        X, coords_embaralhadas, k_vizinhos=6, alpha=0.05,
        n_permutacoes=299, seed=2)

    assert info_real["moran_i"][idx_geo].mean() > info_embaralhado["moran_i"][idx_geo].mean() + 0.2, (
        "I de Moran das variaveis geograficas deveria cair bastante ao "
        "embaralhar as coordenadas -- se nao caiu, o metodo nao esta "
        "medindo autocorrelacao espacial de verdade.")
    assert mask_embaralhado[idx_geo].sum() <= mask_real[idx_geo].sum(), (
        "embaralhar coordenadas nao deveria SELECIONAR MAIS variaveis "
        "geograficas do que com as coordenadas corretas.")


def test_moran_i_alpha_mais_apertado_seleciona_menos_ou_igual():
    X, coords, _idx_geo, _ = _dataset_geografico(seed=3)
    _info_a, mask_frouxo = moran_i_mask(X, coords, alpha=0.20, n_permutacoes=299, seed=3)
    _info_b, mask_apertado = moran_i_mask(X, coords, alpha=0.01, n_permutacoes=299, seed=3)
    assert mask_apertado.sum() <= mask_frouxo.sum()


def test_avaliar_subset_nested_cv_moran_roda_e_devolve_formato_esperado():
    from sklearn.model_selection import KFold

    X, coords, _idx_geo, cluster = _dataset_geografico(seed=4, n_por_cluster=15)
    y_int = cluster
    Y_bin = np.eye(2)[y_int]
    cv = list(KFold(n_splits=4, shuffle=True, random_state=4).split(X))

    resultado = _avaliar_subset_nested_cv_moran(
        X, Y_bin, y_int, coords, cv, n_lv=3,
        k_vizinhos=6, alpha=0.05, n_permutacoes=99, seed=4)

    assert set(resultado) >= {"accuracy", "balanced_accuracy", "n_vars",
                               "n_vars_min", "n_vars_max", "n_lv"}
    assert 0.0 <= resultado["balanced_accuracy"] <= 1.0
    assert resultado["n_vars_min"] >= 2
    # sinal geografico forte + clusters bem separados -> deveria aprender
    # a diferenciar os 2 clusters bem acima do acaso (~0.5).
    assert resultado["balanced_accuracy"] > 0.8
