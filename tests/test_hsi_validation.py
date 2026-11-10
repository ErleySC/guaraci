"""Testes de hsi_validation.py (Passo 101) -- validacao externa por
particao nativa de dia/lote, em dados sinteticos."""
from __future__ import annotations


import numpy as np
import pytest

from guaraci.hsi_validation import (run_external_validation_by_day,
                                    run_internal_validation_group_aware)


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


# ── run_internal_validation_group_aware (Passo 111): sem particao por dia ──

def test_run_internal_validation_group_aware_devolve_so_metricas_internas():
    cubos, mascaras, group_ids, rotulos, _dias = _dataset_sintetico_multi_dia()
    relatorio = run_internal_validation_group_aware(
        cubos, mascaras, group_ids, rotulos, max_lvs=5, n_splits_wold=2)

    assert relatorio.n_objetos_teste_externo == 0
    assert relatorio.n_objetos_teste_interno >= 1
    assert relatorio.sensibilidade_externa == {}
    assert relatorio.especificidade_externa == {}
    assert relatorio.precisao_externa == {}
    for classe in relatorio.classes:
        for metrica in (relatorio.sensibilidade_interna,
                        relatorio.especificidade_interna,
                        relatorio.precisao_interna):
            assert 0.0 <= metrica[classe] <= 1.0


def test_run_internal_validation_group_aware_fracao_extrema_leva_a_erro_claro():
    """fracao_teste=0.99 deixa so' 1 objeto de treino (o holdout e' sempre
    no maximo n_objetos-1, nunca esvazia o treino) -- mas 1 grupo so' e'
    insuficiente para a selecao de LVs por Wold (precisa de >=2 grupos
    pra' cross-validar): erro explicito da camada de baixo, nao um crash
    silencioso nem um resultado forjado com dado de treino insuficiente."""
    cubos, mascaras, group_ids, rotulos, _dias = _dataset_sintetico_multi_dia()
    with pytest.raises(ValueError, match="grupos"):
        run_internal_validation_group_aware(
            cubos, mascaras, group_ids, rotulos, fracao_teste=0.99,
            max_lvs=5, n_splits_wold=2)


def test_run_internal_validation_group_aware_comprimentos_divergentes_levanta_erro():
    cubos, mascaras, group_ids, rotulos, _dias = _dataset_sintetico_multi_dia()
    with pytest.raises(ValueError, match="mesmo comprimento"):
        run_internal_validation_group_aware(
            cubos, mascaras, group_ids, rotulos[:-1])


def test_run_internal_validation_group_aware_menos_de_2_objetos_levanta_erro():
    cubos, mascaras, group_ids, rotulos, _dias = _dataset_sintetico_multi_dia()
    with pytest.raises(ValueError, match="pelo menos 2"):
        run_internal_validation_group_aware(
            cubos[:1], mascaras[:1], group_ids[:1], rotulos[:1])




