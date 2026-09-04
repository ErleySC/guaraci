# -*- coding: utf-8 -*-
"""Testes de hsi_multiway.py (Bloco 15) -- cubos hiperespectrais SINTETICOS
com resolucao espacial VARIAVEL entre objetos (mesma irregularidade de
dado real, ver docstring do modulo) e estrutura de sinal conhecida.

Dataset publico real (DeepHS Fruit, Kiwi/VIS + outra combinacao) NAO foi
baixado neste ambiente -- Kiwi sozinho tem ~44GB na fonte (ver
`scripts/download_datasets/baixar_deephs_fruit_todas.py`), inviavel nesta
sessao. A comparacao N-PLS vs. PLS-DA por pixel e' validada aqui em dado
sintetico com a MESMA estrutura de agrupamento (group_id por objeto,
gravacoes front/back); validacao contra o dataset publico real fica
pendente, mesma classe de limitacao ja' documentada no Bloco 14
(MCR-ALS) para o dataset de oleo.
"""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.hsi_multiway import (
    construir_tensor_amostras,
    parafac_hsi,
    NPLS,
    NPLSClassifier,
    comparar_npls_vs_pixelwise,
)


def _cubo_uniforme(altura, largura, espectro, seed, ruido=0.02):
    rng = np.random.default_rng(seed)
    n_bandas = len(espectro)
    cubo = np.tile(espectro, (altura, largura, 1))
    cubo = cubo + rng.normal(0, ruido, size=(altura, largura, n_bandas))
    return np.clip(cubo, 0.0, None)


def _dataset_hsi_sintetico(seed=0, n_objetos_por_classe=8, n_bandas=15):
    """Simula um dataset HSI: 2 classes, espectros puros bem separados,
    cada objeto fisico com 2 gravacoes (front/back, MESMO group_id,
    resolucao espacial levemente diferente entre gravacoes -- irregular
    de proposito, como dado real)."""
    rng = np.random.default_rng(seed)
    eixo = np.arange(n_bandas, dtype=float)
    espectro_a = np.exp(-0.5 * ((eixo - 4) / 2) ** 2)
    espectro_b = np.exp(-0.5 * ((eixo - 11) / 2) ** 2)

    cubos, mascaras, rotulos, grupos = [], [], [], []
    for classe, espectro in (("A", espectro_a), ("B", espectro_b)):
        for k in range(n_objetos_por_classe):
            gid = f"{classe}_{k}"
            for face in ("front", "back"):
                h = int(rng.integers(8, 14))
                w = int(rng.integers(8, 14))
                cubo = _cubo_uniforme(h, w, espectro, seed=hash((gid, face, seed)) % (2**31))
                mascara = np.ones((h, w), dtype=bool)
                # recorta um canto pra' simular fundo/ROI irregular
                mascara[0, 0] = False
                cubos.append(cubo)
                mascaras.append(mascara)
                rotulos.append(classe)
                grupos.append(gid)
    return cubos, mascaras, rotulos, grupos


# ---------------------------------------------------------------------------
#  construir_tensor_amostras
# ---------------------------------------------------------------------------

def test_construir_tensor_amostras_forma_regular_com_cubos_de_tamanhos_diferentes():
    cubos, mascaras, _rotulos, _grupos = _dataset_hsi_sintetico(seed=1, n_objetos_por_classe=2)
    tensor = construir_tensor_amostras(cubos, mascaras, n_linhas=4, n_colunas=4)
    n_bandas = cubos[0].shape[2]
    assert tensor.shape == (len(cubos), 16, n_bandas)
    assert np.all(np.isfinite(tensor))


def test_construir_tensor_amostras_recupera_espectro_uniforme():
    eixo = np.arange(10, dtype=float)
    espectro = np.exp(-0.5 * ((eixo - 5) / 1.5) ** 2)
    cubo = _cubo_uniforme(12, 12, espectro, seed=0, ruido=0.0)
    mascara = np.ones((12, 12), dtype=bool)
    tensor = construir_tensor_amostras([cubo], [mascara], n_linhas=3, n_colunas=3)
    for pos in range(tensor.shape[1]):
        np.testing.assert_allclose(tensor[0, pos, :], espectro, atol=1e-8)


def test_construir_tensor_amostras_rejeita_comprimentos_diferentes():
    cubo = _cubo_uniforme(5, 5, np.ones(4), seed=0)
    with pytest.raises(ValueError):
        construir_tensor_amostras([cubo], [], n_linhas=2, n_colunas=2)


# ---------------------------------------------------------------------------
#  PARAFAC
# ---------------------------------------------------------------------------

def test_parafac_hsi_reconstroi_tensor_de_baixo_posto_com_erro_pequeno():
    rng = np.random.default_rng(2)
    I, J, K, R = 10, 6, 8, 2
    fator_a = rng.uniform(0.5, 1.5, size=(I, R))
    fator_j = rng.uniform(-1, 1, size=(J, R))
    fator_k = rng.uniform(-1, 1, size=(K, R))
    tensor = np.einsum("ir,jr,kr->ijk", fator_a, fator_j, fator_k)
    tensor += rng.normal(0, 0.001, size=tensor.shape)

    res = parafac_hsi(tensor, n_componentes=R, seed=0)
    assert res.fator_amostra.shape == (I, R)
    assert res.fator_espacial.shape == (J, R)
    assert res.fator_espectral.shape == (K, R)
    assert res.erro_reconstrucao_relativo < 0.05


def test_parafac_hsi_rejeita_tensor_nao_3d():
    with pytest.raises(ValueError):
        parafac_hsi(np.zeros((5, 5)), n_componentes=2)


def test_parafac_hsi_rejeita_tensor_todo_zero():
    with pytest.raises(ValueError):
        parafac_hsi(np.zeros((4, 4, 4)), n_componentes=2)


def test_parafac_hsi_em_tensor_construido_de_dado_hsi_sintetico():
    cubos, mascaras, _rotulos, _grupos = _dataset_hsi_sintetico(seed=3, n_objetos_por_classe=4)
    tensor = construir_tensor_amostras(cubos, mascaras, n_linhas=3, n_colunas=3)
    res = parafac_hsi(tensor, n_componentes=2, seed=0)
    assert np.isfinite(res.erro_reconstrucao_relativo)
    assert res.erro_reconstrucao_relativo < 0.5


# ---------------------------------------------------------------------------
#  N-PLS
# ---------------------------------------------------------------------------

def _tensor_supervisionado(seed=0, I=40, J=5, K=10):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=I)
    w_j_true = rng.normal(size=J); w_j_true /= np.linalg.norm(w_j_true)
    w_k_true = rng.normal(size=K); w_k_true /= np.linalg.norm(w_k_true)
    t_true = np.where(y == 1, 2.0, -2.0) + rng.normal(0, 0.3, size=I)
    X = np.einsum("i,j,k->ijk", t_true, w_j_true, w_k_true)
    X += rng.normal(0, 0.05, size=X.shape)
    Y_bin = np.eye(2)[y]
    return X, Y_bin, y


def test_npls_recupera_estrutura_e_separa_classes():
    X, Y_bin, y = _tensor_supervisionado(seed=4)
    modelo = NPLS(n_componentes=1).fit(X, Y_bin)
    assert modelo.T_.shape == (X.shape[0], 1)
    assert modelo.pesos_espaciais_.shape == (X.shape[1], 1)
    assert modelo.pesos_espectrais_.shape == (X.shape[2], 1)

    y_hat = np.argmax(modelo.predict(X), axis=1)
    acc = float(np.mean(y_hat == y))
    assert acc > 0.85, f"N-PLS nao separou as classes (acc={acc:.3f})"


def test_npls_classifier_predict_em_dado_novo():
    X, Y_bin, y = _tensor_supervisionado(seed=5, I=60)
    y_labels = np.array(["classe_a", "classe_b"])[y]
    X_treino, X_teste = X[:40], X[40:]
    y_treino, y_teste = y_labels[:40], y_labels[40:]

    clf = NPLSClassifier(n_componentes=1).fit(X_treino, y_treino)
    pred = clf.predict(X_teste)
    acc = float(np.mean(pred == y_teste))
    assert acc > 0.75, f"NPLSClassifier nao generalizou (acc={acc:.3f})"


def test_npls_rejeita_x_nao_3d():
    with pytest.raises(ValueError):
        NPLS().fit(np.zeros((10, 5)), np.zeros(10))


# ---------------------------------------------------------------------------
#  Comparacao N-PLS vs. PLS-DA por pixel + propriedade group-aware
# ---------------------------------------------------------------------------

def test_comparar_npls_vs_pixelwise_nunca_vaza_grupo_entre_treino_teste():
    cubos, mascaras, rotulos, grupos = _dataset_hsi_sintetico(seed=6, n_objetos_por_classe=6)
    # nao deve levantar RuntimeError (a funcao propria verifica e levanta
    # se algum fold vazar group_id -- ver comparar_npls_vs_pixelwise)
    resultado = comparar_npls_vs_pixelwise(
        cubos, mascaras, rotulos, grupos,
        n_linhas_grade=3, n_colunas_grade=3, n_componentes=2,
        n_splits=3, seed=0, max_pixels_por_gravacao=200)

    assert 0.0 <= resultado["balanced_accuracy_npls"] <= 1.0
    assert 0.0 <= resultado["balanced_accuracy_pixelwise"] <= 1.0
    assert len(resultado["por_fold"]) == 3
    for fold in resultado["por_fold"]:
        assert fold["n_objetos_teste"] > 0


def test_comparar_npls_vs_pixelwise_relata_ambos_os_numeros_honestamente():
    """Nao afirma qual metodo e' melhor -- so' confirma que ambos os
    numeros sao relatados e sao finitos, dado sintetico bem separavel
    (sinal de classe forte, ambos os metodos deveriam performar bem)."""
    cubos, mascaras, rotulos, grupos = _dataset_hsi_sintetico(seed=7, n_objetos_por_classe=6)
    resultado = comparar_npls_vs_pixelwise(
        cubos, mascaras, rotulos, grupos,
        n_linhas_grade=3, n_colunas_grade=3, n_componentes=2,
        n_splits=3, seed=1, max_pixels_por_gravacao=200)
    assert np.isfinite(resultado["balanced_accuracy_npls"])
    assert np.isfinite(resultado["balanced_accuracy_pixelwise"])
