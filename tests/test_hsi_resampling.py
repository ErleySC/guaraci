"""Testes de hsi_resampling.py (Passo 105) -- reamostragem group-aware
+ relatorio de avaliabilidade por classe. A contra-prova de propriedade
(`test_split_group_aware_apos_oversample_nunca_separa_objeto`) e'
OBRIGATORIA pela instrucao: confirma que a reamostragem muda a
distribuicao de treino SEM quebrar o agrupamento por objeto fisico --
mesmo splitter (`StableStratifiedGroupKFold`) e mesmo espirito de
propriedade do Passo 97, generalizado para o caso pos-reamostragem."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from hypothesis import given, strategies as st

from guaraci.conformal import n_minimum_for_alpha
from guaraci.hsi_pixels import build_pixel_dataset
from guaraci.hsi_resampling import (class_evaluability_report,
                                    oversample_minority_groups)
from guaraci.validacao_estatistica import StableStratifiedGroupKFold


def _dataset_desbalanceado(n_por_classe=(10, 3, 1), n_pixels_por_objeto=5,
                            n_bandas=6, seed=0):
    rng = np.random.default_rng(seed)
    classes = ["perfect", "overripe", "unripe"]
    cubos, mascaras, group_ids, rotulos = [], [], [], []
    obj_idx = 0
    for classe, n_objetos in zip(classes, n_por_classe):
        for _ in range(n_objetos):
            gid = f"objeto_{obj_idx}"
            obj_idx += 1
            cubo = rng.normal(size=(n_pixels_por_objeto, 1, n_bandas))
            cubos.append(cubo)
            mascaras.append(np.ones((n_pixels_por_objeto, 1), dtype=bool))
            group_ids.append(gid)
            rotulos.append(classe)
    return build_pixel_dataset(cubos, mascaras, group_ids, rotulos)


# ── oversample_minority_groups ────────────────────────────────────────────

def test_oversample_iguala_peso_em_pixels_sem_criar_objeto_novo():
    """O que `oversample_minority_groups` de fato equaliza e' o PESO em
    PIXELS de cada classe na matriz de treino (o que influencia o que o
    PLS-DA aprende a priorizar) -- NUNCA o numero de OBJETOS FISICOS
    distintos por classe, que e' estrutural (nao da pra' inventar um
    2o abacate onde so' existe 1 sem fabricar dado, o que a instrucao
    proibe). Por isso a contagem de GRUPOS por classe fica INALTERADA
    (contra-prova em test_oversample_duplicatas_mantem_mesmo_group_id),
    so' a contagem de PIXELS muda."""
    X, y, groups = _dataset_desbalanceado(n_por_classe=(10, 3, 1))
    X_res, y_res, groups_res = oversample_minority_groups(X, y, groups)

    n_pixels_por_classe = {c: int(np.sum(y_res == c)) for c in set(y_res.tolist())}
    assert len(set(n_pixels_por_classe.values())) == 1, (
        f"contagem de PIXELS por classe deveria ficar igual apos "
        f"oversample: {n_pixels_por_classe}")

    n_objetos_por_classe = {}
    for classe in set(y.tolist()):
        n_objetos_por_classe[classe] = len(set(groups[y == classe].tolist()))
    assert n_objetos_por_classe == {"perfect": 10, "overripe": 3, "unripe": 1}, (
        "contagem de OBJETOS FISICOS (nao pixels) e' estrutural -- "
        "reamostragem nunca fabrica um objeto novo, so' pesa mais os que "
        "existem de verdade")


def test_oversample_duplicatas_mantem_mesmo_group_id():
    """Achado central do design (Passo 105): duplicatas NAO recebem um
    id sintetico novo -- continuam com o group_id ORIGINAL. Confirma
    contando pixels por grupo: um grupo duplicado deve ter MAIS pixels
    que o original tinha (multiplo exato do tamanho original), nunca um
    group_id novo aparecendo na lista de grupos unicos."""
    X, y, groups = _dataset_desbalanceado(n_por_classe=(10, 1),
                                          n_pixels_por_objeto=5)
    grupos_originais = set(groups.tolist())
    X_res, y_res, groups_res = oversample_minority_groups(X, y, groups)
    grupos_apos = set(groups_res.tolist())
    assert grupos_apos == grupos_originais, (
        "oversample nao deveria criar NENHUM group_id novo -- so' "
        "replicar pixels dos grupos existentes")

    contagem_apos = Counter(groups_res.tolist())
    # grupo da classe minoritaria (1 objeto so') deve ter MAIS pixels agora
    gid_minoria = next(g for g in grupos_originais
                       if y[groups == g][0] == "overripe")
    assert contagem_apos[gid_minoria] > 5  # > tamanho original (5 pixels)
    assert contagem_apos[gid_minoria] % 5 == 0  # multiplo exato (copias inteiras)


def test_oversample_sem_desbalanceamento_nao_altera_dados():
    X, y, groups = _dataset_desbalanceado(n_por_classe=(5, 5, 5))
    X_res, y_res, groups_res = oversample_minority_groups(X, y, groups)
    assert len(X_res) == len(X)


def test_oversample_grupo_com_2_classes_levanta_erro():
    X = np.zeros((4, 3))
    y = np.array(["a", "a", "b", "a"])   # grupo "g1" com 2 classes -- invalido
    groups = np.array(["g1", "g1", "g1", "g2"])
    with pytest.raises(ValueError, match="mais de 1 classe"):
        oversample_minority_groups(X, y, groups)


# ── class_evaluability_report ────────────────────────────────────────────

def test_class_evaluability_report_usa_limiar_do_conformal():
    X, y, groups = _dataset_desbalanceado(n_por_classe=(30, 3, 1))
    relatorio = class_evaluability_report(y, groups, alpha=0.05)
    n_min = n_minimum_for_alpha(0.05)

    assert relatorio["unripe"].n_grupos == 1
    assert relatorio["unripe"].avaliavel is False
    assert "nao avaliavel" in relatorio["unripe"].nota.lower()
    assert relatorio["unripe"].n_minimo == n_min

    if 30 >= n_min:
        assert relatorio["perfect"].avaliavel is True
        assert relatorio["perfect"].nota == ""


# ── CONTRA-PROVA OBRIGATORIA (Passo 105): oversample nao quebra agrupamento ──

@given(
    n_maioria=st.integers(min_value=6, max_value=12),
    n_minoria=st.integers(min_value=1, max_value=4),
    n_pixels=st.integers(min_value=1, max_value=4),
    n_splits=st.integers(min_value=2, max_value=3),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_split_group_aware_apos_oversample_nunca_separa_objeto(
        n_maioria, n_minoria, n_pixels, n_splits, seed):
    X, y, groups = _dataset_desbalanceado(
        n_por_classe=(n_maioria, n_minoria), n_pixels_por_objeto=n_pixels,
        seed=seed)
    X_res, y_res, groups_res = oversample_minority_groups(X, y, groups, seed=seed)

    n_grupos_por_classe = Counter()
    for g in set(groups_res.tolist()):
        n_grupos_por_classe[y_res[groups_res == g][0]] += 1
    if min(n_grupos_por_classe.values()) < n_splits:
        return  # Hypothesis descarta combinacoes onde o split nem se aplica

    classes_ordenadas = sorted(set(y_res.tolist()))
    y_int = np.array([classes_ordenadas.index(v) for v in y_res])
    splitter = StableStratifiedGroupKFold(n_splits=n_splits, seed=seed)
    for idx_treino, idx_val in splitter.split(X_res, y_int, groups=groups_res):
        grupos_treino = set(groups_res[idx_treino])
        grupos_val = set(groups_res[idx_val])
        interseccao = grupos_treino & grupos_val
        assert not interseccao, (
            f"objeto(s) {interseccao} apareceram em treino E validacao "
            f"apos oversample -- reamostragem quebrou o agrupamento.")
