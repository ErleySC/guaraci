# -*- coding: utf-8 -*-
"""epo_glsw.py — EPO e GLSW: removem/atenuam a variação de um fator de
perturbação CONHECIDO (ex.: espécie-hospedeira, rodada de sessão,
instrumento), estimada de espectros DIFERENÇA entre pares que
compartilham a mesma condição de interesse mas diferem no fator.

Proposta T2 da rodada multiagente de 2026-09-10.

Referências:
    EPO  — Roger J.-M., Chauchard F. & Bellon-Maurel V. (2003).
           *Chemom. Intell. Lab. Syst.* 66:191-204,
           DOI 10.1016/S0169-7439(03)00051-0.
    GLSW — Martens H., Høy M., Wise B. M., Bro R. & Brockhoff P. B. (2003).
           *J. Chemometrics* 17:153-165, DOI 10.1002/cem.780.

DIFERENÇA para o que já existe no GUARACI: o OSC (`preprocessamento.OSC`)
remove variação ortogonal ao **y** (rótulo/teor) — foi REJEITADO no óleo
(Passo 134, RMSEP 4,70→4,99). O EMSC remove linha de base/interferentes
CONHECIDOS a priori. EPO/GLSW usam o RÓTULO DO FATOR DE PERTURBAÇÃO (não
o espectro do interferente) e não precisam de `y`.

POR QUE NÃO É UM TRANSFORMER DE `sklearn.Pipeline` PADRÃO: `Pipeline.fit(X,
y)` só encaminha UM alvo (`y`) a cada etapa — aqui o que se precisa não é
o alvo de classificação/regressão, é o RÓTULO DO FATOR DE PERTURBAÇÃO, e a
matriz de ajuste não é `X` diretamente, é uma matriz de DIFERENÇAS entre
pares. `EPO`/`GLSW` recebem essa matriz de diferenças já pronta em `fit` —
monte-a com `build_difference_matrix`, chamada só com o TREINO de cada
fold de CV (nunca com o fold de validação, para não vazar a estrutura do
fator de perturbação).

LIMITE DE ESCOPO, testado no dataset EEM Zenodo (marca de azeite como
hospedeira, rodada como sessão — ver `docs/VALIDACAO_PUBLICA.md`), NÃO no
dataset próprio de óleo: no dataset próprio, a ordem de leitura é
COLINEAR com o teor (achado já registrado, `~/.guaraci_local/CLAUDE.md`
P12) — remover as direções desse fator removeria o próprio sinal que se
quer quantificar. EPO/GLSW só fazem sentido quando o fator de perturbação
é separável do alvo, não quando são confundidos.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["EPO", "GLSW", "build_difference_matrix"]


def build_difference_matrix(X: np.ndarray, grupo_interesse: np.ndarray,
                             fator_incomodo: np.ndarray) -> np.ndarray:
    """Monta a matriz de espectros DIFERENÇA para `EPO`/`GLSW`.

    Para cada nível de `grupo_interesse` (a condição que se quer
    PRESERVAR — ex.: mesma espécie/mesmo teor) com pelo menos 2 níveis de
    `fator_incomodo` (o que se quer remover — ex.: hospedeira/sessão),
    calcula a MÉDIA por nível de `fator_incomodo` dentro daquele grupo e
    entra com a diferença de cada nível contra o primeiro como uma linha
    de D. Usar a média (não pares individuais) evita que um grupo com
    muitas réplicas domine a estimativa.

    Levanta `ValueError` se nenhum grupo tiver >=2 níveis do fator de
    perturbação — nesse caso não há como estimar a direção a remover.
    """
    X = np.asarray(X, dtype=float)
    grupo_interesse = np.asarray(grupo_interesse)
    fator_incomodo = np.asarray(fator_incomodo)
    if not (len(X) == len(grupo_interesse) == len(fator_incomodo)):
        raise ValueError(
            f"X ({len(X)}), grupo_interesse ({len(grupo_interesse)}) e "
            f"fator_incomodo ({len(fator_incomodo)}) precisam do mesmo "
            f"comprimento.")

    diffs = []
    for g in np.unique(grupo_interesse):
        idx = np.where(grupo_interesse == g)[0]
        niveis = fator_incomodo[idx]
        niveis_unicos = np.unique(niveis)
        if len(niveis_unicos) < 2:
            continue
        medias = {niv: X[idx][niveis == niv].mean(axis=0) for niv in niveis_unicos}
        ref = medias[niveis_unicos[0]]
        for niv in niveis_unicos[1:]:
            diffs.append(medias[niv] - ref)

    if not diffs:
        raise ValueError(
            "Nenhum par formável: todo grupo_interesse tem só 1 nível de "
            "fator_incomodo -- EPO/GLSW precisam de pelo menos um grupo "
            "com >=2 níveis do fator de perturbação para estimar a "
            "direção a remover/atenuar.")
    return np.asarray(diffs)


class EPO(BaseEstimator, TransformerMixin):
    """External Parameter Orthogonalisation. Projeta os espectros no
    complemento ortogonal das `n_componentes` direções principais da
    matriz de diferenças `D` (ver `build_difference_matrix`) -- remove
    essas direções por completo.

    `fit(D)`: `D` é a matriz de DIFERENÇAS, não os espectros originais.
    """

    def __init__(self, n_componentes: int = 1):
        self.n_componentes = n_componentes

    def fit(self, D, y=None):
        D = np.asarray(D, dtype=float)
        if D.ndim != 2 or D.shape[0] == 0:
            raise ValueError(f"D precisa ser 2D não-vazia, recebido shape {D.shape}")
        _, _, Vt = np.linalg.svd(D, full_matrices=False)
        k = max(1, min(self.n_componentes, Vt.shape[0]))
        P = Vt[:k].T   # (p, k) -- direções do fator de perturbação
        p = D.shape[1]
        self.projetor_ = np.eye(p) - P @ P.T
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.projetor_


class GLSW(BaseEstimator, TransformerMixin):
    """Generalized Least Squares Weighting. Versão "suave" de `EPO`: em
    vez de remover as direções do fator de perturbação por completo,
    ATENUA cada uma proporcionalmente ao seu autovalor na covariância das
    diferenças -- `alpha` controla a intensidade (menor `alpha` =
    atenuação mais forte; `alpha` grande tende à identidade, sem efeito).

    `fit(D)`: `D` é a matriz de DIFERENÇAS, não os espectros originais.
    """

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha

    def fit(self, D, y=None):
        D = np.asarray(D, dtype=float)
        if D.ndim != 2 or D.shape[0] == 0:
            raise ValueError(f"D precisa ser 2D não-vazia, recebido shape {D.shape}")
        n = D.shape[0]
        C = (D.T @ D) / max(n, 1)
        autovalores, autovetores = np.linalg.eigh(C)
        autovalores = np.clip(autovalores, 0.0, None)
        pesos = 1.0 / (autovalores / self.alpha + 1.0)
        self.matriz_peso_ = autovetores @ np.diag(pesos) @ autovetores.T
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.matriz_peso_
