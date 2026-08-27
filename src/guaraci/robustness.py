# -*- coding: utf-8 -*-
"""robustness.py -- Protocolo de robustez (Bloco 13d, Frente 1, decisao
pre-aprovada R1-R3).

QUANTO o resultado final (RMSEP na quantificacao, bal.acc na
classificacao) varia sob perturbacao CONTROLADA de parametros que o
usuario pode variar sem intencao: (a) parametros de pre-processamento
(janela/ordem do Savitzky-Golay, escolha de preset SNV/MSC/MC), (b)
pequena perturbacao sintetica simulando variacao de instrumento (ruido
gaussiano, deriva de linha de base), (c) reamostragem group-aware
(bootstrap ja implementado no projeto) como terceiro eixo.

R2 -- FILOSOFIA: reporta a variacao como INTERVALO (minimo/mediana/
maximo), NUNCA como aprovado/reprovado binario -- mesmo espirito de
`chemometric_stats.rpd_rer`/`regression_figures_of_merit` (LOD/LOQ como
intervalo, Bloco 12): declarar o que os dados sustentam, sem esconder a
incerteza atras de um numero unico ou de um veredito.

R3 -- ESCOPO: cobre o caminho de quantificacao por especie (PLS-R) e o
caminho de classificacao (PLS-DA) -- os dois casos representativos
combinados nesta rodada. DD-SIMCA e conformal ja tem sua propria forma de
declarar incerteza/cobertura (nao duplicado aqui).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from guaraci.config import Config

__all__ = [
    "RobustnessResult",
    "gaussian_noise_variants",
    "baseline_drift_variants",
    "preprocessing_config_variants",
    "run_robustness_protocol",
    "avaliar_rmsep_plsr",
    "avaliar_bal_acc_plsda",
]


@dataclass
class RobustnessResult:
    """Resultado de UMA familia de perturbacao. `valores` e' a lista bruta
    (uma entrada por replica da perturbacao) -- min/mediana/max sao
    derivados dela, nunca um veredito aprovado/reprovado (R2)."""
    perturbacao: str
    baseline: float
    valores: List[float]
    minimo: float
    maximo: float
    mediana: float
    variacao_absoluta: float
    n_replicas: int


def gaussian_noise_variants(X: np.ndarray, niveis: Sequence[float] = (0.001, 0.005, 0.01),
                             n_replicas: int = 5, seed: int = 0
                             ) -> Dict[str, List[np.ndarray]]:
    """R1(b): ruido gaussiano de baixa magnitude somado ao espectro --
    simula degradacao de SNR do instrumento. `niveis` sao fracoes do
    desvio-padrao GLOBAL do espectro (0.001-0.01 tipico de detector NIR
    entre bem calibrado e levemente degradado)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    escala_base = float(X.std())
    variantes: Dict[str, List[np.ndarray]] = {}
    for nivel in niveis:
        variantes[f"ruido_gaussiano_{nivel:g}"] = [
            X + rng.normal(scale=nivel * escala_base, size=X.shape)
            for _ in range(n_replicas)
        ]
    return variantes


def baseline_drift_variants(X: np.ndarray, niveis: Sequence[float] = (0.001, 0.005, 0.01),
                             n_replicas: int = 5, seed: int = 1
                             ) -> Dict[str, List[np.ndarray]]:
    """R1(b): deriva de linha de base sintetica -- rampa linear (por
    amostra, inclinacao aleatoria) somada ao espectro. Magnitude relativa
    ao desvio-padrao global do espectro, mesma escala de
    `gaussian_noise_variants` para comparabilidade entre as duas
    perturbacoes."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    escala_base = float(X.std())
    eixo_norm = np.linspace(-1.0, 1.0, p)
    variantes: Dict[str, List[np.ndarray]] = {}
    for nivel in niveis:
        lista = []
        for _ in range(n_replicas):
            inclinacoes = rng.normal(scale=nivel * escala_base, size=n)
            lista.append(X + np.outer(inclinacoes, eixo_norm))
        variantes[f"deriva_linha_base_{nivel:g}"] = lista
    return variantes


def preprocessing_config_variants(cfg_base: "Config") -> Dict[str, "Config"]:
    """R1(a): variantes de PRE-PROCESSAMENTO que o usuario pode escolher
    sem intencao de testar robustez -- janela/ordem do Savitzky-Golay (se
    o preset atual usa SG) e presets alternativos (troca de SNV/MSC/MC).
    Devolve `Config`s (nao arrays) -- quem avalia decide como reconstruir
    o pre-processador (`preprocessamento.build_preprocessor`) a partir de
    cada variante."""
    variantes: Dict[str, "Config"] = {}
    janela_atual = int(cfg_base.sg_window)
    for delta in (-4, 4):
        nova_janela = max(5, janela_atual + delta)
        if nova_janela % 2 == 0:
            nova_janela += 1   # savgol exige janela impar
        if nova_janela != janela_atual:
            variantes[f"sg_window_{nova_janela}"] = dataclasses.replace(
                cfg_base, sg_window=nova_janela)

    ordem_atual = int(cfg_base.sg_polyorder)
    for delta in (-1, 1):
        nova_ordem = max(1, ordem_atual + delta)
        if nova_ordem != ordem_atual and nova_ordem < janela_atual:
            variantes[f"sg_polyorder_{nova_ordem}"] = dataclasses.replace(
                cfg_base, sg_polyorder=nova_ordem)

    presets = ("msc_sg_mc", "snv_sg_mc", "mc", "autoscaling")
    for preset in presets:
        if preset != cfg_base.default_preprocessing:
            variantes[f"preprocessamento_{preset}"] = dataclasses.replace(
                cfg_base, default_preprocessing=preset)
    return variantes


def run_robustness_protocol(
        avaliar_baseline: Callable[[], float],
        variantes: Dict[str, List[Callable[[], float]]]
        ) -> Dict[str, RobustnessResult]:
    """Roda o protocolo: `avaliar_baseline()` UMA vez (referencia sem
    perturbacao) + `avaliar_baseline`-like para cada callable em cada
    lista de `variantes` -- um `RobustnessResult` por chave.

    Cada valor de `variantes` e' uma lista de callables SEM ARGUMENTO
    (zero-arg) que devolvem a metrica (RMSEP ou bal.acc) para aquela
    replica daquela perturbacao -- desacopla o gerador de perturbacao
    (dados ou config) de COMO avaliar (RMSEP de PLS-R, bal.acc de PLS-DA,
    ou qualquer outra metrica futura).
    """
    baseline = float(avaliar_baseline())
    resultados: Dict[str, RobustnessResult] = {}
    for nome, chamadas in variantes.items():
        valores = [float(c()) for c in chamadas]
        resultados[nome] = RobustnessResult(
            perturbacao=nome, baseline=baseline, valores=valores,
            minimo=min(valores), maximo=max(valores),
            mediana=float(np.median(valores)),
            variacao_absoluta=max(valores) - min(valores),
            n_replicas=len(valores))
    return resultados


# =========================================================================
#  R3 -- caminhos representativos: PLS-R (quantificacao) e PLS-DA
#  (classificacao). Split holdout group-aware simples (nao CV completa) --
#  mantem o protocolo rapido o bastante para repetir por perturbacao;
#  cada perturbacao ja tem `n_replicas` proprias para capturar variacao.
# =========================================================================

def avaliar_rmsep_plsr(X: np.ndarray, y: np.ndarray, mae_id: Optional[np.ndarray],
                        n_components: int = 5, frac_cal: float = 0.7) -> float:
    """RMSEP de um PLS-R com split cal/val group-aware (Kennard-Stone) --
    metrica usada no caminho de QUANTIFICACAO do protocolo de robustez."""
    from sklearn.cross_decomposition import PLSRegression

    from guaraci.dados_io import kennard_stone_split_group_aware

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    idx_cal, idx_val = kennard_stone_split_group_aware(X, mae_id, frac_cal)
    if len(idx_cal) < 2 or len(idx_val) < 1:
        return float("nan")
    nc = max(1, min(n_components, X.shape[1], len(idx_cal) - 1))
    pls = PLSRegression(n_components=nc, scale=False)
    pls.fit(X[idx_cal], y[idx_cal])
    pred = np.asarray(pls.predict(X[idx_val])).ravel()
    return float(np.sqrt(np.mean((pred - y[idx_val]) ** 2)))


def avaliar_bal_acc_plsda(X: np.ndarray, y: np.ndarray, mae_id: Optional[np.ndarray],
                           n_components: int = 5, frac_cal: float = 0.7) -> float:
    """Balanced accuracy de um PLS-DA com split cal/val group-aware
    (Kennard-Stone) -- metrica usada no caminho de CLASSIFICACAO do
    protocolo de robustez."""
    from sklearn.metrics import balanced_accuracy_score

    from guaraci.avaliacao_modelos import PLSDAClassifier
    from guaraci.dados_io import kennard_stone_split_group_aware

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    idx_cal, idx_val = kennard_stone_split_group_aware(X, mae_id, frac_cal)
    if len(idx_cal) < 2 or len(idx_val) < 1:
        return float("nan")
    clf = PLSDAClassifier(n_components=n_components)
    clf.fit(X[idx_cal], y[idx_cal])
    pred = clf.predict(X[idx_val])
    return float(balanced_accuracy_score(y[idx_val], pred))
