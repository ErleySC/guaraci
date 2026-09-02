"""hsi_classification.py — PLS-DA por pixel + agregacao por voto
majoritario (Passo 98 da `INSTRUCAO_HSI_MINIMO_VIAVEL.md`).

Reaproveita a implementacao de PLS-DA JA existente (`avaliacao_modelos.
PLSDAClassifier`) -- nao reimplementa. O split e' group-aware por objeto
fisico (`StableStratifiedGroupKFold`, Passo 97). O numero de variaveis
latentes usa o MESMO criterio ja' padronizado no projeto (parsimonia de
Wold, 1978: menor n_componentes cuja taxa de erro de CV fica dentro de
2% da minima -- ver `pipeline.py`, comentario "Wold parsimony
criterion") em vez de introduzir "Venetian Blinds" como metodo novo
(a instrucao permite so' se uma comparacao mostrar vantagem real
documentada -- nao ha' essa comparacao nesta rodada).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from guaraci.avaliacao_modelos import PLSDAClassifier
from guaraci.validacao_estatistica import StableStratifiedGroupKFold

__all__ = [
    "ObjectAggregationResult",
    "select_n_components_wold",
    "aggregate_predictions_by_object",
    "fit_predict_pixel_plsda",
]


def select_n_components_wold(
        X: np.ndarray, y_int: np.ndarray, groups: np.ndarray, *,
        max_lvs: int = 10, n_splits: int = 3, seed: int = 42,
        ) -> int:
    """Escolhe o numero de variaveis latentes do PLS-DA por CV group-aware
    + criterio de parsimonia de Wold (1978): a MENOR quantidade de LVs
    cuja taxa de erro de CV fica dentro de 2% da taxa minima observada
    (mesma regra ja' usada para regressao em `pipeline.py`, generalizada
    aqui para classificacao via taxa de erro = 1 - balanced accuracy)."""
    max_lvs_viavel = min(max_lvs, X.shape[1], X.shape[0] - 1)
    if max_lvs_viavel < 1:
        raise ValueError(
            f"select_n_components_wold: dados insuficientes para nenhum "
            f"componente (n_amostras={X.shape[0]}, n_bandas={X.shape[1]}).")

    splitter = StableStratifiedGroupKFold(n_splits=n_splits, seed=seed)
    folds = list(splitter.split(X, y_int, groups=groups))

    erros_por_lv = np.full(max_lvs_viavel, np.nan)
    for n_comp in range(1, max_lvs_viavel + 1):
        erros_fold = []
        for idx_treino, idx_val in folds:
            clf = PLSDAClassifier(n_components=n_comp)
            clf.fit(X[idx_treino], y_int[idx_treino])
            pred = clf.predict(X[idx_val])
            erros_fold.append(1.0 - balanced_accuracy_score(y_int[idx_val], pred))
        erros_por_lv[n_comp - 1] = float(np.mean(erros_fold))

    erro_min = float(np.nanmin(erros_por_lv))
    tolerancia = erro_min * 1.02
    candidatos = np.where(erros_por_lv <= tolerancia)[0]
    return int(candidatos[0]) + 1  # +1: indice 0 -> 1 componente


@dataclass
class ObjectAggregationResult:
    classe_predita: str
    heterogeneidade: float   # fracao de pixels EM DESACORDO com a classe majoritaria
    n_pixels: int


def aggregate_predictions_by_object(
        predicoes_pixel: np.ndarray, group_ids: np.ndarray,
        ) -> Dict[str, ObjectAggregationResult]:
    """Agrega predicoes por-pixel em UMA predicao por objeto fisico
    (classe majoritaria) + grau de heterogeneidade (fracao de pixels em
    desacordo -- sinal de qualidade, nao descartado)."""
    resultado: Dict[str, ObjectAggregationResult] = {}
    for gid in np.unique(group_ids):
        preds_obj = predicoes_pixel[group_ids == gid]
        contagem = Counter(preds_obj.tolist())
        classe_majoritaria, n_maioria = contagem.most_common(1)[0]
        n_total = len(preds_obj)
        resultado[str(gid)] = ObjectAggregationResult(
            classe_predita=str(classe_majoritaria),
            heterogeneidade=1.0 - (n_maioria / n_total),
            n_pixels=n_total)
    return resultado


def fit_predict_pixel_plsda(
        X_treino: np.ndarray, y_treino: np.ndarray, groups_treino: np.ndarray,
        X_teste: np.ndarray, groups_teste: np.ndarray, *,
        n_components: Optional[int] = None, max_lvs: int = 10,
        n_splits_wold: int = 3, seed: int = 42,
        ) -> Dict[str, object]:
    """Treina PLS-DA por-pixel (com selecao de LVs por Wold se
    `n_components` nao for fornecido) e devolve predicoes agregadas por
    objeto para `X_teste`/`groups_teste`.

    Devolve dict com `n_components` (usado), `predicoes_pixel` (rotulo
    predito de CADA pixel de teste) e `predicoes_objeto` (dict
    group_id -> ObjectAggregationResult)."""
    encoder = LabelEncoder()
    y_int = encoder.fit_transform(y_treino)

    if n_components is None:
        n_components = select_n_components_wold(
            X_treino, y_int, groups_treino,
            max_lvs=max_lvs, n_splits=n_splits_wold, seed=seed)

    clf = PLSDAClassifier(n_components=n_components)
    clf.fit(X_treino, y_treino)
    predicoes_pixel = clf.predict(X_teste)

    predicoes_objeto = aggregate_predictions_by_object(
        predicoes_pixel, groups_teste)

    return {
        "n_components": n_components,
        "predicoes_pixel": predicoes_pixel,
        "predicoes_objeto": predicoes_objeto,
    }
