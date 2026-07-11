"""model_registry.py — Registry de modelos do Auto-Benchmark / Monte Carlo CV
(item 20 da auditoria).

Antes desta extração, a MESMA lista de classificadores de comparação
(PLS-DA, SVM RBF, Random Forest, Gradient Boosting, XGBoost opcional) estava
hardcoded DUAS VEZES — em `benchmark_classificadores()` e `monte_carlo_cv()`
(avaliacao_modelos.py) — e havia divergido silenciosamente: o Gradient
Boosting do Monte Carlo CV não tinha `subsample=0.8` (o do benchmark tinha),
apesar da docstring de `monte_carlo_cv` afirmar "mesmos hiperparâmetros do
benchmark". Este módulo é a fonte ÚNICA de verdade; as duas funções agora
chamam `construir_lista_benchmark()` — a divergência foi corrigida
alinhando ao benchmark (fonte mais completa/documentada).

Adicionar, remover ou re-parametrizar um modelo do Auto-Benchmark e do
Monte Carlo CV (que sempre usam o MESMO conjunto de modelos "extras") passa
a ser: editar `_REGISTRO` aqui, em um único lugar.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Tuple

if TYPE_CHECKING:
    from guaraci.pipeline import Config

Construtor = Callable[[int, "Config"], Any]


def _construir_pls_da(n_opt: int, cfg: "Config"):
    from guaraci.avaliacao_modelos import PLSDAClassifier
    return PLSDAClassifier(n_components=n_opt)


def _construir_svm_rbf(n_opt: int, cfg: "Config"):
    from sklearn.svm import SVC
    return SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
               random_state=cfg.seed, class_weight="balanced")


def _construir_random_forest(n_opt: int, cfg: "Config"):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                   class_weight="balanced_subsample",
                                   n_jobs=1, random_state=cfg.seed)


def _construir_grad_boost(n_opt: int, cfg: "Config"):
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                       max_depth=3, subsample=0.8,
                                       random_state=cfg.seed)


def _construir_xgboost(n_opt: int, cfg: "Config"):
    from xgboost import XGBClassifier  # type: ignore  -- ImportError se nao instalado
    return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="mlogloss", verbosity=0,
                          n_jobs=1, random_state=cfg.seed)


# Fonte UNICA de verdade: (nome, construtor, obrigatorio).
# obrigatorio=True (so' PLS-DA) -> sempre incluido, mesmo com
# incluir_opcionais=False (caso de monte_carlo_cv quando
# cfg.monte_carlo_incluir_todos=False: so' PLS-DA participa).
_REGISTRO: List[Tuple[str, Construtor, bool]] = [
    ("PLS-DA",        _construir_pls_da,        True),
    ("SVM RBF",       _construir_svm_rbf,       False),
    ("Random Forest", _construir_random_forest, False),
    ("Grad. Boost.",  _construir_grad_boost,    False),
    ("XGBoost",       _construir_xgboost,       False),
]


def construir_lista_benchmark(n_opt: int, cfg: "Config",
                               incluir_opcionais: bool = True
                               ) -> List[Tuple[str, Any]]:
    """Monta a lista (nome, instância) de classificadores para comparação.

    incluir_opcionais=False: só o modelo obrigatório (PLS-DA) — usado por
    monte_carlo_cv quando cfg.monte_carlo_incluir_todos=False. Modelos cujo
    pacote opcional não está instalado (ex.: xgboost) são pulados
    silenciosamente (ImportError), preservando o comportamento histórico.
    """
    lista: List[Tuple[str, Any]] = []
    for nome, construtor, obrigatorio in _REGISTRO:
        if not obrigatorio and not incluir_opcionais:
            continue
        try:
            lista.append((nome, construtor(n_opt, cfg)))
        except ImportError:
            continue
    return lista


def nomes_modelos_benchmark(incluir_opcionais: bool = True) -> Tuple[str, ...]:
    """Nomes registrados, sem instanciar (útil para UI/documentação)."""
    return tuple(nome for nome, _, obrig in _REGISTRO
                 if obrig or incluir_opcionais)


__all__ = ["construir_lista_benchmark", "nomes_modelos_benchmark", "Construtor"]
