# -*- coding: utf-8 -*-
"""linearity.py -- Teste formal de linearidade da curva de calibracao
(Bloco 13d, Frente 1, decisao pre-aprovada L1-L3).

Teste de falta de ajuste (lack-of-fit) classico: ajusta uma reta
y_predito ~ a + b*y_referencia e particiona a soma de quadrados residual
em FALTA DE AJUSTE (desvio sistematico da media de cada nivel em relacao
a reta -- indica curvatura) + ERRO PURO (variacao entre replicas do MESMO
nivel -- ruido que nenhum modelo, por melhor que seja, elimina). F = razao
das duas variancias; H0 = "a reta e' adequada" (sem falta de ajuste
significativa).

Referencia: Draper, N.R. & Smith, H. (1998), Applied Regression Analysis,
3rd ed., Wiley, cap. 2.6 ("Lack of Fit and Estimation of Pure Error").
Mesmo padrao de citacao/reporte (F, p-valor, graus de liberdade) de
`validacao_estatistica.cv_anova_eriksson`, usado aqui como precedente de
estilo do projeto (decisao pre-aprovada L1).

NIVEL da curva = grupo de replica fisica (`mae_id`), nao o valor bruto de
y_referencia -- e' a mesma nocao de "replica verdadeira" ja usada por
`chemometric_stats.regression_figures_of_merit` (LOD/LOQ, Bloco 12): duas
amostras diferentes que por acaso tem a mesma referencia NAO contam como
replica uma da outra (L2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.stats import f as f_dist

__all__ = [
    "LackOfFitResult",
    "lack_of_fit_test",
]


@dataclass
class LackOfFitResult:
    """Resultado do teste de falta de ajuste. `computavel=False` (L2) --
    dataset sem replica verdadeira suficiente -- todos os campos numericos
    ficam `None`; use `motivo` para reportar por que, na MESMA linguagem
    de "nao computavel" ja usada para DD-SIMCA/conformal em regime de n
    insuficiente (nunca forcar o teste, nunca inventar replica)."""
    computavel: bool
    motivo: Optional[str]
    F: Optional[float]
    p_value: Optional[float]
    df_lof: Optional[int]
    df_pe: Optional[int]
    ss_lof: Optional[float]
    ss_pe: Optional[float]
    n_niveis: int
    n_niveis_com_replica: int
    alpha: float
    linear: Optional[bool]   # None se nao computavel; True = H0 nao rejeitada


def _filtrar_faixa_trabalho(y_ref: np.ndarray, faixa_trabalho: Optional[Tuple[float, float]]
                             ) -> np.ndarray:
    """L3: mascara das amostras dentro da faixa de trabalho declarada pelo
    perfil de matriz -- `None`/faixa invalida mantem TODAS as amostras
    (mesmo fallback conservador de `PerfilMatriz.dentro_da_faixa`)."""
    n = len(y_ref)
    if not faixa_trabalho or len(faixa_trabalho) != 2:
        return np.ones(n, dtype=bool)
    lo, hi = faixa_trabalho
    if lo is None or hi is None:
        return np.ones(n, dtype=bool)
    return (y_ref >= lo) & (y_ref <= hi)


def lack_of_fit_test(y_ref: np.ndarray, y_pred: np.ndarray, grupos: np.ndarray,
                      faixa_trabalho: Optional[Tuple[float, float]] = None,
                      alpha: float = 0.05) -> LackOfFitResult:
    """Teste F de falta de ajuste da curva de calibracao (L1).

    y_ref: valor de referencia (verdadeiro) por amostra.
    y_pred: valor predito pelo modelo (PLS) por amostra, MESMA unidade de
        y_ref (nao residuo, nao z-score).
    grupos: identificador de replica fisica por amostra (`mae_id`) -- cada
        grupo e' um NIVEL da curva; L2 exige pelo menos 2 niveis com >=2
        replicas para separar falta-de-ajuste de erro puro.
    faixa_trabalho: (min, max) opcional, filtra para a faixa de trabalho
        do perfil de matriz ANTES de testar (L3) -- nao a faixa total dos
        dados brutos.
    alpha: nivel de significancia para a decisao `linear` (H0 nao
        rejeitada = sem evidencia de falta de ajuste ao nivel `alpha`).
    """
    y_ref = np.asarray(y_ref, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    grupos = np.asarray(grupos)
    if not (len(y_ref) == len(y_pred) == len(grupos)):
        raise ValueError(
            f"y_ref ({len(y_ref)}), y_pred ({len(y_pred)}) e grupos "
            f"({len(grupos)}) precisam ter o MESMO comprimento -- uma "
            f"linha por amostra.")

    mascara = _filtrar_faixa_trabalho(y_ref, faixa_trabalho)
    y_ref, y_pred, grupos = y_ref[mascara], y_pred[mascara], grupos[mascara]

    def _inapto(motivo: str, n_niveis: int = 0, n_com_replica: int = 0) -> LackOfFitResult:
        return LackOfFitResult(
            computavel=False, motivo=motivo, F=None, p_value=None,
            df_lof=None, df_pe=None, ss_lof=None, ss_pe=None,
            n_niveis=n_niveis, n_niveis_com_replica=n_com_replica,
            alpha=alpha, linear=None)

    if len(y_ref) < 4:
        return _inapto(
            f"amostras insuficientes dentro da faixa de trabalho (n={len(y_ref)}, "
            f"minimo pratico 4) -- nao computavel")

    grupos_unicos = np.unique(grupos)
    n_niveis = len(grupos_unicos)
    if n_niveis < 3:
        return _inapto(
            f"niveis insuficientes na curva (n_niveis={n_niveis}, minimo 3 -- "
            f"precisa de pelo menos 1 grau de liberdade de falta-de-ajuste "
            f"alem dos 2 parametros da reta) -- nao computavel", n_niveis)

    niveis_x, niveis_y = [], []
    n_niveis_com_replica = 0
    for g in grupos_unicos:
        m = grupos == g
        niveis_x.append(float(y_ref[m].mean()))
        niveis_y.append(y_pred[m])
        if m.sum() >= 2:
            n_niveis_com_replica += 1

    if n_niveis_com_replica == 0:
        return _inapto(
            "nenhum nivel da curva tem replica fisica verdadeira (todo "
            "grupo de mae_id tem so' 1 amostra dentro da faixa de "
            "trabalho) -- sem replica nao ha' como separar erro puro de "
            "falta de ajuste (L2)", n_niveis, 0)

    x_flat = np.concatenate([np.full(len(arr), x) for x, arr in zip(niveis_x, niveis_y)])
    y_flat = np.concatenate(niveis_y)

    b, a = np.polyfit(x_flat, y_flat, 1)
    y_hat_flat = a + b * x_flat
    ss_resid_total = float(np.sum((y_flat - y_hat_flat) ** 2))

    ss_pe = 0.0
    df_pe = 0
    for arr in niveis_y:
        if len(arr) >= 2:
            ss_pe += float(np.sum((arr - arr.mean()) ** 2))
            df_pe += len(arr) - 1

    df_lof = n_niveis - 2
    if df_lof <= 0 or df_pe <= 0:
        return _inapto(
            f"graus de liberdade insuficientes (df_lof={df_lof}, df_pe={df_pe}) "
            f"-- nao computavel", n_niveis, n_niveis_com_replica)

    ss_lof = max(0.0, ss_resid_total - ss_pe)
    if ss_pe <= 0:
        # Replicas identicas (variancia de erro puro zero) -- F->inf, p->0
        # se houver QUALQUER falta de ajuste; F=0/p=1 se ss_lof tambem for 0.
        F = float("inf") if ss_lof > 0 else 0.0
        p = 0.0 if ss_lof > 0 else 1.0
    else:
        F = (ss_lof / df_lof) / (ss_pe / df_pe)
        p = float(1.0 - f_dist.cdf(F, df_lof, df_pe))

    return LackOfFitResult(
        computavel=True, motivo=None, F=float(F), p_value=p,
        df_lof=int(df_lof), df_pe=int(df_pe), ss_lof=float(ss_lof),
        ss_pe=float(ss_pe), n_niveis=n_niveis,
        n_niveis_com_replica=n_niveis_com_replica, alpha=alpha,
        linear=bool(p >= alpha))
