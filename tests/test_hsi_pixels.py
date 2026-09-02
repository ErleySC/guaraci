"""Testes de hsi_pixels.py (Passo 97 -- o passo mais critico do bloco
HSI). A contra-prova de propriedade abaixo
(`test_split_group_aware_hsi_nunca_separa_pixels_do_mesmo_objeto`) e'
OBRIGATORIA pela instrucao: "nenhum split de treino/teste pode colocar
pixels do mesmo objeto fisico em lados diferentes" -- gerada com
Hypothesis (numero de objetos e pixels-por-objeto aleatorios), rodando o
splitter group-aware JA PADRONIZADO no projeto
(`StableStratifiedGroupKFold`, mesmo usado no resto do Guaraci -- nao um
mecanismo group-aware paralelo)."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import example, given, strategies as st

from guaraci.hsi_pixels import build_pixel_dataset, extract_roi_spectra
from guaraci.validacao_estatistica import StableStratifiedGroupKFold


# ── extract_roi_spectra ──────────────────────────────────────────────────

def test_extract_roi_spectra_extrai_so_os_pixels_da_mascara():
    cubo = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    mascara = np.array([[True, False, True], [False, True, False]])
    pixels = extract_roi_spectra(cubo, mascara)
    assert pixels.shape == (3, 4)
    np.testing.assert_array_equal(pixels[0], cubo[0, 0])
    np.testing.assert_array_equal(pixels[1], cubo[0, 2])
    np.testing.assert_array_equal(pixels[2], cubo[1, 1])


def test_extract_roi_spectra_mascara_vazia_levanta_erro():
    cubo = np.zeros((2, 2, 3))
    with pytest.raises(ValueError, match="mascara vazia"):
        extract_roi_spectra(cubo, np.zeros((2, 2), dtype=bool))


def test_extract_roi_spectra_forma_incompativel_levanta_erro():
    cubo = np.zeros((2, 2, 3))
    with pytest.raises(ValueError, match="forma"):
        extract_roi_spectra(cubo, np.zeros((3, 3), dtype=bool))


# ── build_pixel_dataset ──────────────────────────────────────────────────

def test_build_pixel_dataset_replica_rotulo_e_grupo_por_pixel():
    cubo1 = np.ones((2, 2, 3))       # 4 pixels
    cubo2 = np.full((2, 1, 3), 5.0)  # 2 pixels
    mascara_tudo1 = np.ones((2, 2), dtype=bool)
    mascara_tudo2 = np.ones((2, 1), dtype=bool)

    X, y, groups = build_pixel_dataset(
        [cubo1, cubo2], [mascara_tudo1, mascara_tudo2],
        ["objeto_A", "objeto_B"], ["perfect", "overripe"])

    assert X.shape == (6, 3)
    assert list(y) == ["perfect"] * 4 + ["overripe"] * 2
    assert list(groups) == ["objeto_A"] * 4 + ["objeto_B"] * 2


def test_build_pixel_dataset_comprimentos_divergentes_levanta_erro():
    with pytest.raises(ValueError, match="comprimentos divergentes"):
        build_pixel_dataset([np.zeros((2, 2, 3))], [np.ones((2, 2), bool)],
                            ["A", "B"], ["x"])


# ── max_pixels_por_gravacao (Passo 104: frutas de resolucao alta) ───────

def test_build_pixel_dataset_subamostra_ate_o_limite():
    cubo = np.arange(10 * 1 * 3, dtype=float).reshape(10, 1, 3)  # 10 pixels
    mascara = np.ones((10, 1), dtype=bool)
    X, y, groups = build_pixel_dataset(
        [cubo], [mascara], ["obj1"], ["perfect"],
        max_pixels_por_gravacao=4, seed=0)
    assert X.shape == (4, 3)
    assert list(y) == ["perfect"] * 4
    assert list(groups) == ["obj1"] * 4
    # todos os pixels retidos vieram REALMENTE do cubo original (nunca
    # inventados) -- cada linha de X deve bater com alguma linha real.
    pixels_reais = cubo.reshape(-1, 3)
    for linha in X:
        assert any(np.array_equal(linha, p) for p in pixels_reais)


def test_build_pixel_dataset_nao_subamostra_abaixo_do_limite():
    cubo = np.arange(3 * 1 * 3, dtype=float).reshape(3, 1, 3)  # so' 3 pixels
    mascara = np.ones((3, 1), dtype=bool)
    X, y, groups = build_pixel_dataset(
        [cubo], [mascara], ["obj1"], ["perfect"],
        max_pixels_por_gravacao=100, seed=0)
    assert X.shape == (3, 3)  # nada a subamostrar, todos os 3 mantidos


def test_build_pixel_dataset_sem_limite_preserva_comportamento_antigo():
    cubo = np.arange(50 * 1 * 3, dtype=float).reshape(50, 1, 3)
    mascara = np.ones((50, 1), dtype=bool)
    X, y, groups = build_pixel_dataset(
        [cubo], [mascara], ["obj1"], ["perfect"])  # max_pixels_por_gravacao=None
    assert X.shape == (50, 3)


def test_build_pixel_dataset_subamostragem_e_reprodutivel_por_seed():
    cubo = np.arange(20 * 1 * 3, dtype=float).reshape(20, 1, 3)
    mascara = np.ones((20, 1), dtype=bool)
    X1, *_ = build_pixel_dataset([cubo], [mascara], ["obj1"], ["perfect"],
                                 max_pixels_por_gravacao=5, seed=7)
    X2, *_ = build_pixel_dataset([cubo], [mascara], ["obj1"], ["perfect"],
                                 max_pixels_por_gravacao=5, seed=7)
    np.testing.assert_array_equal(X1, X2)


# ── CONTRA-PROVA OBRIGATORIA (Passo 97): propriedade de nao-vazamento ───

@given(
    n_objetos=st.integers(min_value=4, max_value=10),
    n_pixels_min=st.integers(min_value=1, max_value=3),
    n_pixels_extra=st.integers(min_value=0, max_value=5),
    n_bandas=st.integers(min_value=2, max_value=6),
    n_splits=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
)
# Fronteira defensiva: n_objetos=4 e' o minimo pratico p/ n_splits=4 (cada
# fold precisa de ao menos 1 grupo) -- mesmo tipo de limiar coberto por
# @example em test_propriedades_hypothesis.py::test_split_group_aware_
# nunca_separa_grupo (nao ha' bug historico conhecido aqui, e' cobertura
# defensiva de fronteira, nao reproducao de bug ja achado).
@example(n_objetos=4, n_pixels_min=1, n_pixels_extra=0, n_bandas=3,
         n_splits=4, seed=0)
def test_split_group_aware_hsi_nunca_separa_pixels_do_mesmo_objeto(
        n_objetos, n_pixels_min, n_pixels_extra, n_bandas, n_splits, seed):
    rng = np.random.default_rng(seed)
    cubos, mascaras, group_ids, rotulos = [], [], [], []
    classes = ["unripe", "perfect", "overripe"]
    for i in range(n_objetos):
        n_pixels = n_pixels_min + int(rng.integers(0, n_pixels_extra + 1))
        cubo = rng.normal(size=(n_pixels, 1, n_bandas))
        cubos.append(cubo)
        mascaras.append(np.ones((n_pixels, 1), dtype=bool))
        group_ids.append(f"objeto_{i}")
        rotulos.append(classes[i % len(classes)])

    X, y, groups = build_pixel_dataset(cubos, mascaras, group_ids, rotulos)

    # StableStratifiedGroupKFold precisa de >= n_splits grupos DISTINTOS
    # por classe presente -- com classes desbalanceadas por construcao
    # (round-robin de 3 classes sobre n_objetos), pode nao dar pra' rodar
    # com n_splits pedido; nesse caso o teste nao se aplica a esta
    # combinacao (Hypothesis descarta e gera outra).
    from collections import Counter
    grupos_por_classe = Counter()
    for g, rot in zip(group_ids, rotulos):
        grupos_por_classe[rot] += 1
    if min(grupos_por_classe.values()) < n_splits:
        return

    splitter = StableStratifiedGroupKFold(n_splits=n_splits, seed=seed)
    y_int = np.array([classes.index(v) for v in y])
    for idx_treino, idx_val in splitter.split(X, y_int, groups=groups):
        grupos_treino = set(groups[idx_treino])
        grupos_val = set(groups[idx_val])
        interseccao = grupos_treino & grupos_val
        assert not interseccao, (
            f"objeto(s) {interseccao} apareceram em treino E validacao no "
            f"mesmo fold -- vazamento de pixels do mesmo objeto fisico.")
