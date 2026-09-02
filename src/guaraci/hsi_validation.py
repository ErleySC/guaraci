"""hsi_validation.py — Validacao externa por particao nativa de origem
(Passo 101 da `INSTRUCAO_HSI_MINIMO_VIAVEL.md`).

O dataset publico usado nesta integracao (DeepHS Fruit/Kaki/VIS, ver
`hsi_io.py`) tem particao nativa por DIA de medicao (`day_1_m3` ..
`day_9_m3`, 8 dias distintos, `storage_days` crescente por dia --
achado por leitura direta do JSON de anotacoes: cada dia e' uma sessao
de medicao/lote separada, nao um agrupamento arbitrario nosso). Usamos
essa particao como teste de GENERALIZACAO REAL (Passo 101): treinar
so' com dias mais cedo, testar em dias nunca vistos -- diferente do
Passo 98 (que treina/testa objetos aleatorios misturados entre TODOS
os dias).

Reaproveita `figuras.specificity_by_class` (ja' existente) + sklearn
p/ sensibilidade (recall)/precisao por classe -- reportadas SEMPRE
separadas (teste interno x externo), nunca uma media unica que
esconderia queda de desempenho no externo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, cast

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from guaraci.figuras import specificity_by_class
from guaraci.hsi_classification import (ObjectAggregationResult,
                                        fit_predict_pixel_plsda)
from guaraci.hsi_pixels import build_pixel_dataset

__all__ = ["ExternalValidationReport", "run_external_validation_by_day"]


@dataclass
class ExternalValidationReport:
    classes: List[str]
    n_objetos_teste_interno: int
    n_objetos_teste_externo: int
    sensibilidade_interna: Dict[str, float]
    especificidade_interna: Dict[str, float]
    precisao_interna: Dict[str, float]
    sensibilidade_externa: Dict[str, float]
    especificidade_externa: Dict[str, float]
    precisao_externa: Dict[str, float]


def _metricas_por_classe(y_real_por_objeto: Dict[str, str],
                          y_pred_por_objeto: Dict[str, str],
                          classes: Sequence[str]) -> Dict[str, Dict[str, float]]:
    gids = sorted(y_real_por_objeto)
    y_real = [y_real_por_objeto[g] for g in gids]
    y_pred = [y_pred_por_objeto[g] for g in gids]

    cm = confusion_matrix(y_real, y_pred, labels=list(classes))
    sens = recall_score(y_real, y_pred, labels=list(classes),
                        average=None, zero_division=0)
    prec = precision_score(y_real, y_pred, labels=list(classes),
                           average=None, zero_division=0)
    espec = specificity_by_class(cm)

    return {
        "sensibilidade": dict(zip(classes, sens.tolist())),
        "precisao": dict(zip(classes, prec.tolist())),
        "especificidade": dict(zip(classes, espec.tolist())),
    }


def run_external_validation_by_day(
        cubos: Sequence[np.ndarray], mascaras: Sequence[np.ndarray],
        group_ids: Sequence[str], rotulos: Sequence[str],
        dias: Sequence[str], dias_externos: Sequence[str], *,
        fracao_teste_interno: float = 0.25, seed: int = 42,
        max_lvs: int = 8, n_splits_wold: int = 2,
        max_pixels_por_gravacao: Optional[int] = None,
        ) -> ExternalValidationReport:
    """Treina com objetos de dias FORA de `dias_externos`, valida:
      - internamente: numa fracao dos MESMOS dias de treino (objetos
        nunca vistos, mas do mesmo lote/sessao) -- split group-aware por
        objeto.
      - externamente: em TODOS os objetos de `dias_externos` (lote/sessao
        nunca vista em nenhuma etapa do treino).

    `dias` deve ter o MESMO comprimento de `cubos`/`mascaras`/
    `group_ids`/`rotulos` (1 entrada por gravacao).

    `max_pixels_por_gravacao`: repassado a `build_pixel_dataset` (Passo
    104) -- necessario para frutas de resolucao alta (ex. Avocado/VIS,
    ~97000 pixels/imagem vs. ~4096 do Kaki) nao estourarem memoria.
    `None` (default) preserva o comportamento antigo."""
    dias_arr = np.asarray(dias)
    n = len(cubos)
    if not (len(mascaras) == len(group_ids) == len(rotulos) == len(dias_arr) == n):
        raise ValueError(
            "run_external_validation_by_day: todas as sequencias de "
            "entrada devem ter o mesmo comprimento (1 por gravacao).")

    e_externo = np.isin(dias_arr, list(dias_externos))
    if not e_externo.any():
        raise ValueError(
            f"Nenhuma gravacao pertence a dias_externos={list(dias_externos)} "
            f"-- confira os valores de `dias` (dias disponiveis: "
            f"{sorted(set(dias_arr.tolist()))}).")
    if e_externo.all():
        raise ValueError(
            "Todas as gravacoes pertencem a dias_externos -- nao sobra "
            "nada para treinar.")

    def _sel(seq, mascara_bool):
        return [v for v, m in zip(seq, mascara_bool) if m]

    cubos_dev = _sel(cubos, ~e_externo)
    mascaras_dev = _sel(mascaras, ~e_externo)
    grupos_dev = _sel(group_ids, ~e_externo)
    rotulos_dev = _sel(rotulos, ~e_externo)

    cubos_ext = _sel(cubos, e_externo)
    mascaras_ext = _sel(mascaras, e_externo)
    grupos_ext = _sel(group_ids, e_externo)
    rotulos_ext = _sel(rotulos, e_externo)

    X_dev, y_dev, pg_dev = build_pixel_dataset(
        cubos_dev, mascaras_dev, grupos_dev, rotulos_dev,
        max_pixels_por_gravacao=max_pixels_por_gravacao, seed=seed)
    X_ext, y_ext, pg_ext = build_pixel_dataset(
        cubos_ext, mascaras_ext, grupos_ext, rotulos_ext,
        max_pixels_por_gravacao=max_pixels_por_gravacao, seed=seed)

    objetos_dev = np.unique(pg_dev)
    rng = np.random.default_rng(seed)
    n_teste_interno = max(1, int(round(len(objetos_dev) * fracao_teste_interno)))
    objetos_teste_interno = set(
        rng.choice(objetos_dev, size=n_teste_interno, replace=False))
    mascara_teste_interno = np.array([g in objetos_teste_interno for g in pg_dev])

    resultado = fit_predict_pixel_plsda(
        X_dev[~mascara_teste_interno], y_dev[~mascara_teste_interno],
        pg_dev[~mascara_teste_interno],
        X_ext, pg_ext,   # predicoes no conjunto EXTERNO
        max_lvs=max_lvs, n_splits_wold=n_splits_wold, seed=seed)
    predicoes_externo = cast(Dict[str, ObjectAggregationResult],
                             resultado["predicoes_objeto"])
    n_components = cast(int, resultado["n_components"])

    resultado_interno = fit_predict_pixel_plsda(
        X_dev[~mascara_teste_interno], y_dev[~mascara_teste_interno],
        pg_dev[~mascara_teste_interno],
        X_dev[mascara_teste_interno], pg_dev[mascara_teste_interno],
        n_components=n_components)   # MESMO n_components do modelo externo
    predicoes_interno = cast(Dict[str, ObjectAggregationResult],
                             resultado_interno["predicoes_objeto"])

    classes = sorted(set(rotulos))
    y_real_interno = {g: y_dev[pg_dev == g][0] for g in objetos_teste_interno}
    y_pred_interno = {g: r.classe_predita for g, r in predicoes_interno.items()}
    y_real_externo = {g: y_ext[pg_ext == g][0] for g in np.unique(pg_ext)}
    y_pred_externo = {g: r.classe_predita for g, r in predicoes_externo.items()}

    m_interno = _metricas_por_classe(y_real_interno, y_pred_interno, classes)
    m_externo = _metricas_por_classe(y_real_externo, y_pred_externo, classes)

    return ExternalValidationReport(
        classes=classes,
        n_objetos_teste_interno=len(y_real_interno),
        n_objetos_teste_externo=len(y_real_externo),
        sensibilidade_interna=m_interno["sensibilidade"],
        especificidade_interna=m_interno["especificidade"],
        precisao_interna=m_interno["precisao"],
        sensibilidade_externa=m_externo["sensibilidade"],
        especificidade_externa=m_externo["especificidade"],
        precisao_externa=m_externo["precisao"])
