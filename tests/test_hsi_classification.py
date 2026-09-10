"""Testes de hsi_classification.py (Passo 98) -- PLS-DA por pixel,
selecao de LVs por parsimonia de Wold, agregacao por voto majoritario
com heterogeneidade. Inclui um teste fim-a-fim contra o dataset publico
real (segmentacao -> extracao de pixel -> classificacao), gated por
GUARACI_DATASETS_DIR como os demais modulos HSI."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from guaraci.hsi_classification import (aggregate_predictions_by_object,
                                        fit_predict_pixel_plsda,
                                        select_n_components_wold)


def _dataset_sintetico_facil(seed: int = 0, n_objetos_por_classe: int = 6,
                              n_pixels_por_objeto: int = 15, n_bandas: int = 12):
    """3 classes com assinaturas espectrais bem separadas -- caso facil,
    so' confirma que o pipeline PLS-DA por pixel funciona mecanicamente
    e converge para uma boa separacao quando o sinal e' forte."""
    rng = np.random.default_rng(seed)
    classes = ["unripe", "perfect", "overripe"]
    centros = {c: rng.normal(scale=1.0, size=n_bandas) * 3.0 for c in classes}

    X_list, y_list, g_list = [], [], []
    obj_idx = 0
    for classe in classes:
        for _ in range(n_objetos_por_classe):
            gid = f"objeto_{obj_idx}"
            obj_idx += 1
            for _ in range(n_pixels_por_objeto):
                X_list.append(centros[classe] + rng.normal(scale=0.3, size=n_bandas))
                y_list.append(classe)
                g_list.append(gid)
    X = np.array(X_list)
    y = np.array(y_list, dtype=str)
    groups = np.array(g_list, dtype=str)
    return X, y, groups


# ── select_n_components_wold ─────────────────────────────────────────────

def test_select_n_components_wold_devolve_valor_no_intervalo_valido():
    X, y, groups = _dataset_sintetico_facil()
    from sklearn.preprocessing import LabelEncoder
    y_int = LabelEncoder().fit_transform(y)
    n_comp = select_n_components_wold(X, y_int, groups, max_lvs=8, n_splits=3)
    assert 1 <= n_comp <= 8


def test_select_n_components_wold_dados_insuficientes_levanta_erro():
    X = np.zeros((0, 5))
    with pytest.raises(ValueError, match="insuficientes"):
        select_n_components_wold(X, np.array([], dtype=int),
                                 np.array([], dtype=str))


# ── aggregate_predictions_by_object ──────────────────────────────────────

def test_aggregate_predictions_by_object_maioria_e_heterogeneidade():
    preds = np.array(["A", "A", "A", "B", "A", "B", "B", "B"])
    groups = np.array(["obj1", "obj1", "obj1", "obj1",
                       "obj2", "obj2", "obj2", "obj2"])
    resultado = aggregate_predictions_by_object(preds, groups)

    assert resultado["obj1"].classe_predita == "A"
    assert resultado["obj1"].heterogeneidade == pytest.approx(0.25)  # 1/4 em desacordo
    assert resultado["obj1"].n_pixels == 4

    assert resultado["obj2"].classe_predita == "B"
    assert resultado["obj2"].heterogeneidade == pytest.approx(0.25)


def test_aggregate_predictions_by_object_unanimidade_heterogeneidade_zero():
    preds = np.array(["A", "A", "A"])
    groups = np.array(["obj1", "obj1", "obj1"])
    resultado = aggregate_predictions_by_object(preds, groups)
    assert resultado["obj1"].heterogeneidade == 0.0


# ── fit_predict_pixel_plsda: fim-a-fim sintetico ─────────────────────────

def test_fit_predict_pixel_plsda_caso_facil_classifica_bem():
    X, y, groups = _dataset_sintetico_facil(seed=1)
    objetos_unicos = np.unique(groups)
    rng = np.random.default_rng(2)
    objetos_teste = set(rng.choice(objetos_unicos, size=6, replace=False))
    mascara_teste = np.array([g in objetos_teste for g in groups])

    resultado = fit_predict_pixel_plsda(
        X[~mascara_teste], y[~mascara_teste], groups[~mascara_teste],
        X[mascara_teste], groups[mascara_teste],
        max_lvs=6, n_splits_wold=3)

    assert resultado["n_components"] >= 1
    assert len(resultado["predicoes_objeto"]) == len(objetos_teste)

    # Caso sintetico facil (classes bem separadas) -- exige acerto na
    # MAIORIA dos objetos de teste, nao 100% (PLS-DA por pixel com ruido
    # nunca e' perfeito, e o teste nao deve exigir mais do que o metodo
    # realmente entrega).
    y_por_objeto = {g: y[groups == g][0] for g in objetos_teste}
    acertos = sum(1 for gid, r in resultado["predicoes_objeto"].items()
                  if r.classe_predita == y_por_objeto[gid])
    assert acertos >= len(objetos_teste) * 0.7


# ── fim-a-fim contra o dataset publico real ──────────────────────────────

def _pasta_deephs_kaki():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "deephs_kaki_vis"
    return pasta if (pasta / "manifest.json").is_file() else None


requer_deephs_kaki = pytest.mark.skipif(
    _pasta_deephs_kaki() is None,
    reason=("dataset publico DeepHS Fruit/Kaki/VIS ausente. Baixe com "
            "'python scripts/download_datasets/baixar_deephs_kaki.py' e "
            "aponte GUARACI_DATASETS_DIR."))


@requer_deephs_kaki
def test_pipeline_hsi_fim_a_fim_contra_kaki_real():
    """Segmentacao (Passo 96) -> extracao de ROI + agrupamento (Passo 97)
    -> PLS-DA por pixel + agregacao (Passo 98), tudo encadeado contra o
    dataset publico real. NAO afirma uma acuracia especifica -- so'
    confirma que o pipeline roda de ponta a ponta sobre dado real e
    reporta (nao esconde) o desempenho obtido, mesmo que modesto (classes
    desbalanceadas: 42 perfect / 12 overripe / 2 unripe nas 56 gravacoes,
    ver docs/PROGRESSO.md)."""
    from guaraci.hsi_io import load_deephs_kaki_dataset
    from guaraci.hsi_segmentation import segment_object_pca_otsu
    from guaraci.hsi_pixels import build_pixel_dataset

    pasta = str(_pasta_deephs_kaki())
    cubos, rotulos, grupos, wavelengths, meta_df = load_deephs_kaki_dataset(pasta)

    mascaras = [segment_object_pca_otsu(c).mascara for c in cubos]
    X, y, pixel_groups = build_pixel_dataset(cubos, mascaras, grupos, rotulos)

    objetos_unicos = np.unique(pixel_groups)
    rng = np.random.default_rng(0)
    n_teste = max(4, len(objetos_unicos) // 4)
    objetos_teste = set(rng.choice(objetos_unicos, size=n_teste, replace=False))
    mascara_teste = np.array([g in objetos_teste for g in pixel_groups])

    resultado = fit_predict_pixel_plsda(
        X[~mascara_teste], y[~mascara_teste], pixel_groups[~mascara_teste],
        X[mascara_teste], pixel_groups[mascara_teste],
        max_lvs=8, n_splits_wold=2)

    assert len(resultado["predicoes_objeto"]) == len(objetos_teste)
    for r in resultado["predicoes_objeto"].values():
        assert r.classe_predita in {"unripe", "perfect", "overripe"}
        assert 0.0 <= r.heterogeneidade <= 1.0
        assert r.n_pixels > 0
    print(f"\n[HSI real] n_components={resultado['n_components']}, "
          f"objetos de teste={len(objetos_teste)}")
