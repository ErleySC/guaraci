"""Testes de hsi_validation.py (Passo 101) -- validacao externa por
particao nativa de dia/lote. Inclui teste fim-a-fim contra o dataset
publico real (gated por GUARACI_DATASETS_DIR)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from guaraci.hsi_validation import run_external_validation_by_day


def _dataset_sintetico_multi_dia(seed: int = 0):
    """3 dias, 3 classes bem separadas, objetos distintos por dia --
    caso facil o suficiente para o pipeline rodar e produzir metricas
    validas, sem exigir desempenho perfeito."""
    rng = np.random.default_rng(seed)
    classes = ["unripe", "perfect", "overripe"]
    centros = {c: rng.normal(scale=1.0, size=8) * 3.0 for c in classes}

    cubos, mascaras, group_ids, rotulos, dias = [], [], [], [], []
    obj_idx = 0
    for dia in ("dia1", "dia2", "dia3"):
        for classe in classes:
            for _ in range(4):  # 4 objetos por classe por dia
                gid = f"objeto_{obj_idx}"
                obj_idx += 1
                n_pixels = 10
                cubo = (centros[classe] +
                        rng.normal(scale=0.3, size=(n_pixels, 1, 8)))
                cubos.append(cubo)
                mascaras.append(np.ones((n_pixels, 1), dtype=bool))
                group_ids.append(gid)
                rotulos.append(classe)
                dias.append(dia)
    return cubos, mascaras, group_ids, rotulos, dias


def test_run_external_validation_by_day_devolve_metricas_validas():
    cubos, mascaras, group_ids, rotulos, dias = _dataset_sintetico_multi_dia()
    relatorio = run_external_validation_by_day(
        cubos, mascaras, group_ids, rotulos, dias, dias_externos=["dia3"],
        max_lvs=5, n_splits_wold=2)

    assert relatorio.n_objetos_teste_externo == 12  # 3 classes x 4 objetos
    assert relatorio.n_objetos_teste_interno >= 1
    for classe in relatorio.classes:
        for metrica in (relatorio.sensibilidade_interna,
                        relatorio.especificidade_interna,
                        relatorio.precisao_interna,
                        relatorio.sensibilidade_externa,
                        relatorio.especificidade_externa,
                        relatorio.precisao_externa):
            assert 0.0 <= metrica[classe] <= 1.0


def test_run_external_validation_by_day_nenhuma_gravacao_no_dia_externo_levanta_erro():
    cubos, mascaras, group_ids, rotulos, dias = _dataset_sintetico_multi_dia()
    with pytest.raises(ValueError, match="Nenhuma gravacao"):
        run_external_validation_by_day(
            cubos, mascaras, group_ids, rotulos, dias,
            dias_externos=["dia_inexistente"])


def test_run_external_validation_by_day_comprimentos_divergentes_levanta_erro():
    cubos, mascaras, group_ids, rotulos, dias = _dataset_sintetico_multi_dia()
    with pytest.raises(ValueError, match="mesmo comprimento"):
        run_external_validation_by_day(
            cubos, mascaras, group_ids, rotulos, dias[:-1],
            dias_externos=["dia3"])


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
def test_validacao_externa_contra_kaki_real_dias_8_e_9():
    """Teste externo = day_8_m3 + day_9_m3 (nunca vistos no treino);
    teste interno = objetos held-out dos demais dias. NAO afirma uma
    metrica-alvo -- so' confirma que o relatorio roda e reporta
    interno/externo SEPARADOS (nunca uma media unica que esconderia
    queda de desempenho)."""
    from guaraci.hsi_io import load_deephs_kaki_dataset
    from guaraci.hsi_segmentation import segment_object_pca_otsu

    pasta = str(_pasta_deephs_kaki())
    cubos, rotulos, grupos, wavelengths, meta_df = load_deephs_kaki_dataset(pasta)
    mascaras = [segment_object_pca_otsu(c).mascara for c in cubos]
    dias = meta_df["day"].values

    relatorio = run_external_validation_by_day(
        cubos, mascaras, grupos, rotulos, dias,
        dias_externos=["day_8_m3", "day_9_m3"],
        max_lvs=8, n_splits_wold=2)

    assert relatorio.n_objetos_teste_externo > 0
    assert relatorio.n_objetos_teste_interno > 0
    print(f"\n[HSI validacao externa] interno n={relatorio.n_objetos_teste_interno}, "
          f"externo n={relatorio.n_objetos_teste_externo}")
    for classe in relatorio.classes:
        print(f"  {classe}: sens(int)={relatorio.sensibilidade_interna[classe]:.2f} "
              f"sens(ext)={relatorio.sensibilidade_externa[classe]:.2f} "
              f"espec(int)={relatorio.especificidade_interna[classe]:.2f} "
              f"espec(ext)={relatorio.especificidade_externa[classe]:.2f} "
              f"prec(int)={relatorio.precisao_interna[classe]:.2f} "
              f"prec(ext)={relatorio.precisao_externa[classe]:.2f}")
