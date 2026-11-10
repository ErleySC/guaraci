# -*- coding: utf-8 -*-
"""politica_pooled_local.py -- Politica automatica pooled vs. local
(Bloco 26): formaliza a decisao entre um UNICO modelo de quantificacao
pooled (todas as especies juntas, `pipeline.pls_regressao_pooled`) e um
modelo LOCAL por especie (`pipeline.pls_regression_by_species`) --
reaproveitando o portao de aceite do Bloco 20
(`portao_correcao_sinal.avaliar_correcao_sinal`), nao um mecanismo
paralelo.

Ate' esta rodada, o pipeline ja calibra os DOIS (pooled sempre; local
quando a especie tem `min_amostras_adult` amostras adulteradas, ver
`pipeline.pls_regression_by_species`) mas nunca decide formalmente qual
USAR -- a escolha ficava implicita (o pacote persiste os dois, quem
consome escolhe). Este modulo fecha essa lacuna: local so' e'
recomendado quando (a) a especie tem amostras suficientes (MESMO limiar
ja' usado em `pls_regression_by_species`, `min_amostras_adult=6` --
nunca reinventado aqui) E (b) o portao aprova o ganho do modelo local
sobre o pooled, no MESMO split group-aware bloqueado, com poder
estatistico suficiente."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from guaraci.portao_correcao_sinal import VeredictoCorrecaoSinal, avaliar_correcao_sinal
from guaraci.validacao_estatistica import StableStratifiedGroupKFold

__all__ = [
    "DecisaoPooledLocal",
    "MIN_AMOSTRAS_LOCAL_PADRAO",
    "decidir_pooled_vs_local",
]

#: MESMO limiar ja' usado em `pipeline.pls_regression_by_species`
#: (`min_amostras_adult`) -- nao reinventado aqui, so' referenciado.
MIN_AMOSTRAS_LOCAL_PADRAO = 6


@dataclass
class DecisaoPooledLocal:
    """Decisao para UMA especie. `recomendacao` e' sempre `"pooled"`
    (conservador, o comportamento ja' existente) a menos que o portao
    aprove `"local"` com poder estatistico suficiente -- nunca troca pra
    local so' porque o RMSEP local saiu numericamente menor num unico
    split (isso e' exatamente o vies que o portao existe pra' evitar)."""
    especie: str
    n_amostras: int
    n_minimo: int
    dados_suficientes: bool
    veredito: Optional[VeredictoCorrecaoSinal]
    recomendacao: str


def decidir_pooled_vs_local(
        X: np.ndarray, y: np.ndarray, rotulos: np.ndarray,
        grupos: np.ndarray, especie: str, *,
        n_componentes: int = 5, n_splits: int = 3, n_seeds: int = 10,
        min_amostras: int = MIN_AMOSTRAS_LOCAL_PADRAO,
        alpha: float = 0.05,
        ) -> DecisaoPooledLocal:
    """`X`/`y`/`rotulos`/`grupos`: dataset POOLED completo (todas as
    especies com amostra adulterada, mesmo formato de
    `pipeline.pls_regressao_pooled`). Compara, sob o MESMO split
    group-aware bloqueado (repetido em `n_seeds` particoes), o RMSEP de
    `especie` quando avaliada por um modelo treinado em TODO o pooled
    ("sem" o refinamento local) vs. um modelo treinado SO' nos dados de
    `especie` ("com" o refinamento local) -- os dois avaliados nas MESMAS
    amostras de teste de `especie` em cada seed, comparacao pareada
    honesta."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rotulos = np.asarray(rotulos, dtype=str)
    grupos = np.asarray(grupos, dtype=str)

    mask_especie = rotulos == especie
    n_amostras = int(mask_especie.sum())
    dados_suficientes = n_amostras >= min_amostras

    if not dados_suficientes:
        return DecisaoPooledLocal(
            especie=especie, n_amostras=n_amostras, n_minimo=min_amostras,
            dados_suficientes=False, veredito=None, recomendacao="pooled")

    def _fit_predict(X_tr, y_tr, X_va, n_lv_max):
        mc = StandardScaler(with_std=False)
        X_tr_c = mc.fit_transform(X_tr)
        X_va_c = mc.transform(X_va)
        n_lv = int(max(1, min(n_lv_max, X_tr_c.shape[1], len(X_tr_c) - 1)))
        pls = PLSRegression(n_components=n_lv, scale=False)
        pls.fit(X_tr_c, y_tr)
        return np.asarray(pls.predict(X_va_c)).ravel()

    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def _rodar(seed: int, usar_local: bool) -> float:
        splitter = StableStratifiedGroupKFold(n_splits=n_splits, seed=seed)
        folds = list(splitter.split(np.zeros(len(X)), np.zeros(len(X)), groups=grupos))

        y_hat_especie = np.full(n_amostras, np.nan)
        idx_especie_global = np.where(mask_especie)[0]
        posicao = {g: k for k, g in enumerate(idx_especie_global)}

        for idx_tr, idx_va in folds:
            idx_va_especie = np.array([i for i in idx_va if mask_especie[i]], dtype=int)
            if idx_va_especie.size == 0:
                continue
            if usar_local:
                idx_tr_uso = np.array([i for i in idx_tr if mask_especie[i]], dtype=int)
            else:
                idx_tr_uso = idx_tr
            if len(idx_tr_uso) < 2:
                continue
            pred = _fit_predict(X[idx_tr_uso], y[idx_tr_uso],
                                X[idx_va_especie], n_componentes)
            for gi, p in zip(idx_va_especie, pred):
                y_hat_especie[posicao[gi]] = p

        validos = ~np.isnan(y_hat_especie)
        if validos.sum() == 0:
            return float("nan")
        return _rmse(y[idx_especie_global][validos], y_hat_especie[validos])

    veredito = avaliar_correcao_sinal(
        f"local_{especie}",
        avaliar_sem_fn=lambda seed: _rodar(seed, usar_local=False),
        avaliar_com_fn=lambda seed: _rodar(seed, usar_local=True),
        metrica="RMSEP", n_seeds=n_seeds, alpha=alpha)

    recomendacao = ("local" if (veredito.veredito == "aprovado"
                                and veredito.poder_suficiente)
                    else "pooled")

    return DecisaoPooledLocal(
        especie=especie, n_amostras=n_amostras, n_minimo=min_amostras,
        dados_suficientes=True, veredito=veredito, recomendacao=recomendacao)
