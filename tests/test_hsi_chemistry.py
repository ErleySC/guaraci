"""Testes de hsi_chemistry.py (Passo 100) -- cruzamento VIP x tabela de
atribuicao quimica. Inclui um teste fim-a-fim contra o dataset publico
real (gated por GUARACI_DATASETS_DIR)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from guaraci.hsi_chemistry import (ATRIBUICAO_QUIMICA_VIS_FRUTA,
                                    BandAttribution,
                                    cross_reference_vip_with_chemistry)


def test_cross_reference_encontra_banda_dentro_da_faixa():
    wavelengths = np.array([400.0, 670.0, 900.0])
    vip = np.array([0.5, 2.0, 0.3])
    tabela = [BandAttribution(660.0, 680.0, "clorofila-a", "ref-teste")]
    achados = cross_reference_vip_with_chemistry(wavelengths, vip, tabela, top_n=1)
    assert len(achados) == 1
    assert achados[0].wavelength_nm == 670.0
    assert "clorofila-a" in achados[0].atribuicao
    assert achados[0].referencia == "ref-teste"


def test_cross_reference_sem_atribuicao_quando_fora_de_qualquer_faixa():
    wavelengths = np.array([300.0, 800.0])
    vip = np.array([2.0, 0.1])
    tabela = [BandAttribution(660.0, 680.0, "clorofila-a", "ref-teste")]
    achados = cross_reference_vip_with_chemistry(wavelengths, vip, tabela, top_n=1)
    assert "sem atribuicao" in achados[0].atribuicao
    assert achados[0].referencia == ""


def test_cross_reference_ordena_por_vip_decrescente():
    wavelengths = np.array([400.0, 500.0, 600.0, 700.0])
    vip = np.array([0.1, 0.9, 0.5, 0.3])
    achados = cross_reference_vip_with_chemistry(
        wavelengths, vip, [], top_n=4)
    vips_ordenados = [a.vip for a in achados]
    assert vips_ordenados == sorted(vips_ordenados, reverse=True)


def test_cross_reference_forma_incompativel_levanta_erro():
    with pytest.raises(ValueError, match="mesma forma"):
        cross_reference_vip_with_chemistry(
            np.array([1.0, 2.0]), np.array([1.0]), [])


def test_cross_reference_top_n_maior_que_disponivel_nao_quebra():
    wavelengths = np.array([400.0, 500.0])
    vip = np.array([0.1, 0.9])
    achados = cross_reference_vip_with_chemistry(
        wavelengths, vip, [], top_n=100)
    assert len(achados) == 2


def test_tabela_default_referencias_nao_vazias():
    """Cada entrada da tabela default precisa citar uma referencia real
    -- nenhuma atribuicao "solta" sem fonte."""
    for entrada in ATRIBUICAO_QUIMICA_VIS_FRUTA:
        assert entrada.referencia.strip() != ""
        assert entrada.banda_min_nm < entrada.banda_max_nm


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
def test_explicabilidade_cruzada_contra_kaki_real():
    from guaraci.avaliacao_modelos import PLSDAClassifier
    from guaraci.chemometric_stats import vip_scores
    from guaraci.hsi_io import load_deephs_kaki_dataset
    from guaraci.hsi_pixels import build_pixel_dataset
    from guaraci.hsi_segmentation import segment_object_pca_otsu

    pasta = str(_pasta_deephs_kaki())
    cubos, rotulos, grupos, wavelengths, meta_df = load_deephs_kaki_dataset(pasta)
    mascaras = [segment_object_pca_otsu(c).mascara for c in cubos]
    X, y, _groups = build_pixel_dataset(cubos, mascaras, grupos, rotulos)

    clf = PLSDAClassifier(n_components=5)
    clf.fit(X, y)
    vip = vip_scores(clf._pls)
    assert vip.shape == wavelengths.shape

    achados = cross_reference_vip_with_chemistry(
        wavelengths, vip, ATRIBUICAO_QUIMICA_VIS_FRUTA, top_n=5)
    assert len(achados) == 5
    for a in achados:
        assert wavelengths.min() <= a.wavelength_nm <= wavelengths.max()
    print("\n[HSI VIP x quimica]")
    for a in achados:
        print(f"  {a.wavelength_nm:.1f}nm (VIP={a.vip:.2f}): {a.atribuicao}"
              f"{' -- ' + a.referencia if a.referencia else ''}")
