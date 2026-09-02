"""hsi_resampling.py — Reamostragem estratificada group-aware para
mitigar o colapso na classe majoritaria (Passo 105 da
`INSTRUCAO_HSI_ROBUSTEZ_E_VALIDACAO.md`) + relatorio de avaliabilidade
estatistica por classe.

`oversample_minority_groups` duplica GRUPOS INTEIROS (objetos fisicos,
nunca pixels soltos de fora do grupo) das classes minoritarias ate'
igualar a contagem de OBJETOS da classe majoritaria -- contagem por
OBJETO, nao por pixel, porque um objeto com segmentacao maior (mais
pixels de ROI) nao deveria dominar so' por isso. As copias mantem o
MESMO `group_id` do original (nao um id sintetico novo) -- e' o que
garante que qualquer split group-aware (`StableStratifiedGroupKFold`,
usado tanto no split externo quanto na selecao interna de LVs por
Wold) NUNCA separa uma copia do original em lados diferentes: o
splitter so' enxerga "mais pixels do mesmo grupo", nunca "um grupo
novo e independente". Contra-prova de propriedade em
`tests/test_hsi_resampling.py` (reaproveita o mesmo splitter do Passo
97, generalizado para o caso pos-reamostragem).

`class_evaluability_report` reusa o MESMO limiar quantitativo ja'
padronizado no projeto para "n insuficiente para uma garantia formal"
(`conformal.n_minimum_for_alpha`, o mesmo usado no assistente Guaraci
e no gate de cobertura DD-SIMCA/conjunto aberto) -- nao inventa um
limiar novo so' para HSI.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from guaraci.conformal import n_minimum_for_alpha

__all__ = [
    "oversample_minority_groups",
    "ClassEvaluability",
    "class_evaluability_report",
]


def oversample_minority_groups(
        X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, seed: int = 42,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Duplica objetos fisicos das classes minoritarias -- a contagem
    igualada e' a de OBJETOS (grupos) representados no treino, nunca o
    numero de objetos fisicos DISTINTOS (isso e' estrutural: nao da pra'
    inventar um 2o abacate onde so' existe 1 sem fabricar dado). Como
    objetos reais tem tamanhos de ROI diferentes apos segmentacao, isso
    NAO garante contagem de PIXELS exatamente igual entre classes em
    dado real (so' em cenarios sinteticos com pixels/objeto uniforme) --
    ainda assim rebalanceia o peso efetivo de cada classe no ajuste do
    PLS-DA, que e' o objetivo. Devolve `(X, y, groups)` expandidos --
    duplicatas tem o MESMO `group_id`, nunca um id sintetico (ver
    docstring do modulo)."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    if len(X) == 0:
        return X, y, groups

    rng = np.random.default_rng(seed)
    grupos_unicos = np.unique(groups)
    classe_por_grupo: Dict[str, str] = {}
    for g in grupos_unicos:
        rotulos_do_grupo = set(y[groups == g].tolist())
        if len(rotulos_do_grupo) != 1:
            raise ValueError(
                f"oversample_minority_groups: grupo {g!r} tem mais de 1 "
                f"classe ({rotulos_do_grupo}) -- cada objeto fisico deve "
                f"ter UM rotulo so'.")
        classe_por_grupo[g] = rotulos_do_grupo.pop()

    grupos_por_classe: Dict[str, list] = defaultdict(list)
    for g, c in classe_por_grupo.items():
        grupos_por_classe[c].append(g)

    n_alvo = max(len(v) for v in grupos_por_classe.values())

    indices_por_grupo = {g: np.where(groups == g)[0] for g in grupos_unicos}
    indices_extra = []
    for _classe, grupos_classe in grupos_por_classe.items():
        deficit = n_alvo - len(grupos_classe)
        if deficit <= 0:
            continue
        escolhidos = rng.choice(grupos_classe, size=deficit, replace=True)
        for g in escolhidos:
            indices_extra.append(indices_por_grupo[g])

    if not indices_extra:
        return X.copy(), y.copy(), groups.copy()

    idx_extra = np.concatenate(indices_extra)
    X_res = np.concatenate([X, X[idx_extra]], axis=0)
    y_res = np.concatenate([y, y[idx_extra]], axis=0)
    groups_res = np.concatenate([groups, groups[idx_extra]], axis=0)
    return X_res, y_res, groups_res


@dataclass
class ClassEvaluability:
    n_grupos: int
    n_minimo: int
    avaliavel: bool
    nota: str


def class_evaluability_report(
        y: np.ndarray, groups: np.ndarray, *, alpha: float = 0.05,
        ) -> Dict[str, ClassEvaluability]:
    """Para cada classe presente, conta OBJETOS FISICOS distintos
    (grupos) e compara com `conformal.n_minimum_for_alpha(alpha)` -- o
    MESMO limiar quantitativo ja' usado no resto do projeto para "n
    insuficiente para garantia formal" (nunca um limiar novo/arbitrario
    so' para HSI). Classes abaixo do minimo vem com `avaliavel=False` e
    uma nota explicita ("nao avaliavel estatisticamente"), MESMA
    linguagem ja' usada em DD-SIMCA/conjunto aberto -- nunca reportada
    como se fosse uma metrica confiavel."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_min = n_minimum_for_alpha(alpha)
    resultado: Dict[str, ClassEvaluability] = {}
    for classe in sorted(set(y.tolist())):
        n_grupos = len(set(groups[y == classe].tolist()))
        avaliavel = n_grupos >= n_min
        nota = ("" if avaliavel else
                f"nao avaliavel estatisticamente (n={n_grupos} objetos "
                f"< minimo {n_min} para alpha={alpha:g})")
        resultado[classe] = ClassEvaluability(
            n_grupos=n_grupos, n_minimo=n_min, avaliavel=avaliavel, nota=nota)
    return resultado
