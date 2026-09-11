# -*- coding: utf-8 -*-
"""asca.py — ASCA (ANOVA-Simultaneous Component Analysis), Smilde, Jansen,
Hoefsloot, Lamers, van der Greef & Timmerman (2005), *Bioinformatics*
21:3043-3048, DOI 10.1093/bioinformatics/bti476.

Proposta T3 da rodada multiagente de 2026-09-10: formaliza como FUNÇÃO DE
PRODUTO o que até então só existia como script de medição pontual
(`scripts/medicoes/medir_especie_vs_adulterante_permanova.py`, R² one-way,
UM fator por vez). ASCA decompõe a matriz espectral SIMULTANEAMENTE pelos
fatores do delineamento (ex.: espécie, adulterante, sessão) e testa a
significância de cada um por permutação — por UNIDADE EXPERIMENTAL (grupo
de réplica física), nunca por espectro individual (pseudo-replicação).

ESCOPO E LIMITE HONESTO (documentado, não escondido): esta implementação
decompõe por EFEITO MARGINAL — a média do próprio nível do fator, menos a
média geral —, a forma clássica de Smilde et al. 2005. É exata para
fatores ORTOGONAIS (delineamento balanceado). Com fatores fortemente
CORRELACIONADOS ou delineamento muito desbalanceado, variância de um fator
pode "vazar" para outro (mesma ressalva já registrada na literatura
original). A extensão **ASCA+** de Thiel, Féraud & Govaerts (2017,
*J. Chemometrics* 31:e2895, DOI 10.1002/cem.2895), que corrige isso via
modelo linear geral, **NÃO está implementada aqui** — `asca_decompose`
expõe `ss_desbalanco` justamente para que o chamador veja quando essa
correção passaria a ser necessária (valor grande = fatores correlacionados
demais para a decomposição marginal ser confiável).

Referências adicionais:
    Zwanenburg G. et al. (2011). *Anal. Chim. Acta* 719-720:100-108,
    DOI 10.1016/j.aca.2011.11.026 — guia prático de ASCA/permutação.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

__all__ = [
    "asca_decompose",
    "asca_permutation_test",
]


def asca_decompose(X: np.ndarray, fatores: Dict[str, np.ndarray],
                    n_componentes: int = 2) -> Dict[str, Any]:
    """Decompõe `X` (n amostras × p variáveis) em efeitos marginais por
    fator do delineamento (`fatores`: `{nome: array de rótulos (n,)}`).

    Para cada fator, o efeito de cada amostra é a média do seu PRÓPRIO
    nível naquele fator, menos a média geral de `X` — a decomposição
    clássica de Smilde et al. 2005 (ver limite no docstring do módulo).

    Returns
    -------
    dict com:
        media_geral       : (p,) média de X
        efeitos[nome]     : dict por fator, com
            matriz            : (n, p) efeito marginal
            soma_quadrados    : ||matriz||² (estatística de teste)
            fracao_ss_total   : soma_quadrados / ss_total
            scores, var_explicada : PCA (SVD) do efeito, `n_componentes`
        residuo, ss_residuo, ss_total, ss_desbalanco
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X precisa ser 2D (n, p), recebido shape {X.shape}")
    if not fatores:
        raise ValueError("`fatores` nao pode ser vazio -- nada a decompor.")

    n = X.shape[0]
    media_geral = X.mean(axis=0)
    X_centrado = X - media_geral
    ss_total = float(np.sum(X_centrado ** 2))

    soma_efeitos = np.zeros_like(X_centrado)
    efeitos: Dict[str, Any] = {}
    for nome, niveis in fatores.items():
        niveis = np.asarray(niveis)
        if len(niveis) != n:
            raise ValueError(
                f"fator {nome!r}: {len(niveis)} rótulos, esperado {n} "
                f"(mesmo n de amostras de X).")
        X_f = np.zeros_like(X_centrado)
        for nivel in np.unique(niveis):
            mask = niveis == nivel
            X_f[mask] = X[mask].mean(axis=0) - media_geral
        ss_f = float(np.sum(X_f ** 2))

        k = max(1, min(n_componentes, min(X_f.shape) - 1)) if min(X_f.shape) > 1 else 1
        try:
            U, S, _Vt = np.linalg.svd(X_f, full_matrices=False)
            k = min(k, S.size)
            scores = U[:, :k] * S[:k]
            ss_svd = float(np.sum(S ** 2))
            var_explicada = (S[:k] ** 2) / ss_svd if ss_svd > 1e-300 else np.zeros(k)
        except np.linalg.LinAlgError:
            scores = np.zeros((n, k))
            var_explicada = np.zeros(k)

        efeitos[nome] = {
            "matriz": X_f,
            "soma_quadrados": ss_f,
            "fracao_ss_total": ss_f / ss_total if ss_total > 1e-300 else float("nan"),
            "scores": scores,
            "var_explicada": var_explicada,
        }
        soma_efeitos += X_f

    residuo = X_centrado - soma_efeitos
    ss_residuo = float(np.sum(residuo ** 2))
    # Sob fatores ORTOGONAIS, ss_total == sum(ss_efeitos) + ss_residuo
    # exatamente. O que sobra e' a covariancia cruzada entre efeitos (e
    # entre efeito e residuo) -- um numero GRANDE aqui e' o sinal de que
    # os fatores estao correlacionados demais para a decomposicao
    # marginal (ver docstring do modulo: e' o caso que pediria ASCA+).
    ss_desbalanco = ss_total - sum(e["soma_quadrados"] for e in efeitos.values()) - ss_residuo

    return {
        "media_geral": media_geral,
        "efeitos": efeitos,
        "residuo": residuo,
        "ss_residuo": ss_residuo,
        "ss_total": ss_total,
        "ss_desbalanco": ss_desbalanco,
    }


def asca_permutation_test(X: np.ndarray, fatores: Dict[str, np.ndarray],
                           unidade_permutacao: Optional[np.ndarray] = None,
                           n_perm: int = 200, seed: int = 42
                           ) -> Dict[str, Any]:
    """Testa a significância de CADA fator de `asca_decompose` por
    permutação, um fator por vez (os demais ficam fixos nos rótulos
    observados -- isola a contribuição marginal daquele fator).

    GROUP-AWARE por construção (achado #1/#16 da rodada multiagente de
    2026-09-10 -- pseudo-replicação infla significância): com
    `unidade_permutacao` (ex.: `mae_id` ou `session_from_mae_id`), o nível
    do fator é permutado por UNIDADE EXPERIMENTAL inteira -- todas as
    réplicas físicas da mesma unidade trocam de nível JUNTAS, nunca
    separadas. Sem `unidade_permutacao`, permuta por AMOSTRA (réplicas
    técnicas contam como observações independentes -- infla a
    significância se houver réplicas, mesmo viés já medido em
    `validacao_estatistica._gerar_permutacoes_rotulo`).

    Returns: `{nome_do_fator: {"ss_observado", "ss_permutado" (array de
    tamanho n_perm), "p_value"}}`. `p_value` é a fração de permutações com
    soma de quadrados >= a observada (teste de cauda superior -- mais
    estrutura é mais "significativo").
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    obs = asca_decompose(X, fatores)

    if unidade_permutacao is not None:
        unidade_permutacao = np.asarray(unidade_permutacao)
        if len(unidade_permutacao) != n:
            raise ValueError(
                f"unidade_permutacao: {len(unidade_permutacao)} rótulos, "
                f"esperado {n}.")
        unidades_unicas = np.unique(unidade_permutacao)

    resultado: Dict[str, Any] = {}
    for nome, niveis in fatores.items():
        niveis = np.asarray(niveis)
        ss_perm = np.empty(n_perm)
        for p in range(n_perm):
            if unidade_permutacao is not None:
                nivel_por_unidade = np.array([
                    niveis[unidade_permutacao == u][0] for u in unidades_unicas
                ])
                nivel_por_unidade_perm = rng.permutation(nivel_por_unidade)
                mapa = dict(zip(unidades_unicas, nivel_por_unidade_perm))
                niveis_perm = np.array([mapa[u] for u in unidade_permutacao])
            else:
                niveis_perm = rng.permutation(niveis)
            fatores_perm = {**fatores, nome: niveis_perm}
            dec = asca_decompose(X, fatores_perm)
            ss_perm[p] = dec["efeitos"][nome]["soma_quadrados"]

        ss_obs = obs["efeitos"][nome]["soma_quadrados"]
        p_value = float((np.sum(ss_perm >= ss_obs) + 1) / (n_perm + 1))
        resultado[nome] = {
            "ss_observado": ss_obs,
            "ss_permutado": ss_perm,
            "p_value": p_value,
        }
    return resultado
