# -*- coding: utf-8 -*-
"""dual_spls.py -- Dual-sPLS (PLS esparso via norma dual), variante lasso.

O QUE E'. Dual-sPLS generaliza o PLS1 classico substituindo a etapa de
extracao de componente (que normalmente maximiza a covariancia sob a
restricao ||w||_2=1) por um problema de otimizacao com uma NORMA DUAL
que combina um termo L1 (que zera coeficientes -- selecao de variaveis)
e um termo L2, controlados por um unico parametro intuitivo: `ppnu`, a
PROPORCAO de variaveis que se quer zerar em cada componente (em vez de um
lambda de penalizacao sem escala clara, como em Lasso/Elastic Net
classicos). Referencia primaria (DOI confirmado, sem retratacao):

    Alsouki, L., Duval, L., Marteau, C., El Haddad, R. & Wahl, F. (2023).
    "Dual-sPLS: a family of Dual Sparse Partial Least Squares regressions
    for feature selection and prediction with tunable sparsity;
    evaluation on simulated and near-infrared (NIR) data."
    Chemometrics and Intelligent Laboratory Systems, 237, 104813.
    DOI: 10.1016/j.chemolab.2023.104813  (mirror arXiv:2301.07206)

ESCOPO DESTA IMPLEMENTACAO. O paper descreve uma FAMILIA de normas duais
(least-squares, ridge, lasso, group-lasso A/B/C). Esta classe implementa
so' a variante **lasso** (a mais usada e' a que da' nome ao metodo,
Omega(w) = lambda*||w||_1 + ||w||_2), reimplementada em numpy a partir da
descricao matematica do metodo e VALIDADA numericamente contra a
implementacao de referencia dos proprios autores -- ver
`tests/test_dual_spls.py::test_dual_spls_matches_r_oracle`. As demais
normas (LS, ridge, group-lasso) ficam fora do escopo desta rodada (nao
foram portadas nem validadas) -- nao inferir que elas existem aqui.

CODIGO E DADO DE REFERENCIA. O pacote R `dual.spls` (dos proprios
autores, Alsouki & Wahl, MIT + file LICENSE) foi removido do CRAN em
2024-04-20, mas seu codigo-fonte esta' preservado no mirror somente-leitura
do CRAN no GitHub: https://github.com/cran/dual.spls (tambem
https://github.com/AlsoukiL/dual.spls, repositorio original dos autores).
O algoritmo abaixo (metodo `fit`) e' uma reescrita independente em numpy
da funcao `d.spls.lasso` desse pacote (arquivo `R/d.spls.lasso.R`), NAO
uma traducao linha-a-linha -- mas segue a MESMA sequencia matematica de
operacoes, verificada termo a termo contra a saida do R real (R 4.3.3,
instalado neste ambiente especificamente para gerar o oraculo numerico;
CRAN, o site do paper e arXiv estao bloqueados nesta sessao, entao nao
foi possivel ler o texto completo do paper -- a validacao usa o codigo-
fonte do pacote R como referencia, nao o paper). Uma copia do codigo-
fonte R usado (`d.spls.lasso.R`, `d.spls.norm.R`, MIT + file LICENSE,
copyright Alsouki & Wahl 2021) e os scripts que geraram os dois
oraculos numericos ficam em `tests/fixtures/dual_spls_oracle/` para
reproducao independente.

ALGORITMO (por componente, apos centralizar X e y):
    1. z = X_deflacionado^T @ y_centralizado
    2. nu = o `ppnu`-quantil (por posicao, igual ao R: ordena |z| e pega
       o elemento cujo indice/p mais se aproxima de ppnu) dos valores
       absolutos de z -- o LIMIAR de shrinkage.
    3. z_nu = soft-threshold(z, nu) = sign(z)*max(|z|-nu, 0)
    4. w = (mu / (nu*||z_nu||_1 + mu^2)) * z_nu,  onde mu = ||z_nu||_2
       (solucao fechada do problema dual para a norma lasso)
    5. t = X_deflacionado @ w, normalizado (||t||_2=1) -- o novo escore
    6. Deflacao: X_deflacionado -= t @ t^T @ X_deflacionado
    7. Coeficientes: mesma reconstrucao por minimos quadrados triangulares
       do PLS1 classico (R = T^T @ X_centralizado @ W e' triangular
       SUPERIOR por construcao; resolvida por substituicao regressiva) --
       entrega Bhat cumulativo por numero de componentes.

INTEGRACAO. Ver `avaliacao_modelos.benchmark_regression_by_species`
(entrada "Dual-sPLS (lasso)" na lista `modelos`) e o portao de aceite em
`scripts/benchmark_dual_spls_tecator.py` (Tecator real, via `sktime`
--extraido so' o arquivo de dado, sktime NAO e' dependencia do projeto).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

__all__ = ["DualSPLS"]


def _norma1(x: np.ndarray) -> float:
    return float(np.sum(np.abs(x)))


def _norma2(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(x ** 2)))


class DualSPLS(BaseEstimator, RegressorMixin):
    """Dual-sPLS (norma lasso), interface sklearn-compativel (`fit`/`predict`).

    Parametros
    ----------
    n_components : int
        Numero de componentes Dual-sPLS a extrair (equivalente a `ncp` no
        pacote R e a `n_components` de `PLSRegression`). O modelo ajustado
        guarda os coeficientes de TODOS os componentes de 1 a
        `n_components` (`self.bhat_path_`); `predict`/`coef_` usam por
        padrao o ultimo (numero maximo de componentes), mas
        `predict(X, n_components=k)` permite avaliar um `k` menor sem
        reajustar -- util para escolher `n_components` por CV.
    sparsity : float, em (0, 1)
        Proporcao desejada de variaveis zeradas por componente (`ppnu` no
        paper/pacote R). 0 = sem selecao (degenera para PLS1 padrao);
        proximo de 1 = quase todas as variaveis zeradas (agressivo).

    Atributos apos `fit`
    ---------------------
    x_mean_ : media de cada coluna de X (usada para centralizar em predict)
    y_mean_ : media de y
    coef_ : coeficientes do ultimo componente, shape (p,)
    intercept_ : intercepto do ultimo componente
    bhat_path_ : coeficientes de CADA componente, shape (n_components, p)
    intercept_path_ : intercepto de cada componente, shape (n_components,)
    scores_ : escores (T), shape (n, n_components)
    loadings_ : cargas duais (W), shape (p, n_components)
    n_zeros_path_ : numero de coeficientes zerados por componente
    lambda_path_ : razao nu/mu (equivalente a lambda no paper) por componente

    Notas de fidelidade numerica
    -----------------------------
    Segue o pacote R passo a passo, incluindo o detalhe (facilmente
    perdido numa reimplementacao "pela formula geral do paper") de que o
    limiar `nu` e' escolhido por POSICAO no vetor ordenado (indice mais
    proximo de `ppnu*p`), nao por interpolacao de quantil -- e' assim que
    `d.spls.lasso.R` funciona e e' o que a validacao contra o oraculo
    real (R 4.3.3, pacote `dual.spls` 0.1.4) confirma bit a bit (ver
    `tests/test_dual_spls.py`).
    """

    def __init__(self, n_components: int = 2, sparsity: float = 0.9):
        self.n_components = n_components
        self.sparsity = sparsity

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DualSPLS":
        X, y = check_X_y(X, y, dtype=float, y_numeric=True)
        n, p = X.shape
        ncp = int(self.n_components)
        ppnu = float(self.sparsity)
        if ncp < 1:
            raise ValueError(f"n_components deve ser >= 1, recebido {ncp}")
        if ncp > min(n, p):
            raise ValueError(
                f"n_components={ncp} excede min(n_amostras={n}, "
                f"n_variaveis={p})")
        if not (0.0 <= ppnu < 1.0):
            raise ValueError(f"sparsity deve estar em [0, 1), recebido {ppnu}")

        x_mean = X.mean(axis=0)
        Xc = X - x_mean
        y_mean = float(np.mean(y))
        yc = y - y_mean

        WW = np.zeros((p, ncp))
        TT = np.zeros((n, ncp))
        Bhat = np.zeros((p, ncp))
        intercepts = np.zeros(ncp)
        n_zeros = np.zeros(ncp, dtype=int)
        lambdas = np.zeros(ncp)
        indices_nao_zero: List[np.ndarray] = []

        Xdef = Xc.copy()
        posicoes = (np.arange(1, p + 1) / p)
        iz = int(np.argmin(np.abs(posicoes - ppnu)))

        for ic in range(ncp):
            z = Xdef.T @ yc

            z_ordenado = np.sort(np.abs(z))
            nu = float(z_ordenado[iz])

            z_nu = np.sign(z) * np.maximum(np.abs(z) - nu, 0.0)
            mu = _norma2(z_nu)
            znu1 = _norma1(z_nu)
            if mu <= 0.0:
                raise FloatingPointError(
                    f"Dual-sPLS: componente {ic + 1} degenerou (||z_nu||_2=0"
                    f" -- sparsity={ppnu} zera todas as variaveis; reduza "
                    "sparsity ou n_components).")
            denom = nu * znu1 + mu ** 2
            w = (mu / denom) * z_nu
            WW[:, ic] = w

            t = Xdef @ w
            t = t / _norma2(t)
            TT[:, ic] = t

            Xdef = Xdef - np.outer(t, t) @ Xdef

            # Reconstrucao dos coeficientes cumulativos (mesma triangularizacao
            # do PLS1 classico -- ver nota abaixo sobre R ser triangular
            # SUPERIOR por construcao).
            Tc = TT[:, :ic + 1]
            Wc = WW[:, :ic + 1]
            R = Tc.T @ Xc @ Wc
            # O R original zera `R[row(R)>col(R)]` (a parte ABAIXO da
            # diagonal, por convencao de indexacao de matriz em R: linha >
            # coluna e' o triangulo inferior) -- mantendo o SUPERIOR, so'
            # por estabilidade numerica (R e' triangular superior por
            # construcao; a parte inferior deveria ser ~0 em aritmetica
            # exata). `backsolve()` do R resolve sistemas triangulares
            # SUPERIORES, coerente com isso.
            R = np.triu(R)
            L = np.linalg.solve(R, np.eye(ic + 1))
            Bhat[:, ic] = Wc @ (L @ (Tc.T @ yc))

            lambdas[ic] = nu / mu
            intercepts[ic] = y_mean - x_mean @ Bhat[:, ic]
            n_zeros[ic] = int(np.sum(Bhat[:, ic] == 0.0))
            indices_nao_zero.append(np.flatnonzero(Bhat[:, ic] != 0.0))

        self.x_mean_ = x_mean
        self.y_mean_ = y_mean
        self.bhat_path_ = Bhat.T  # (ncp, p) -- 1 linha por numero de componentes
        self.intercept_path_ = intercepts
        self.scores_ = TT
        self.loadings_ = WW
        self.n_zeros_path_ = n_zeros
        self.lambda_path_ = lambdas
        self.indices_nao_zero_ = indices_nao_zero
        self.n_features_in_ = p

        # Interface sklearn "flat" -- ultimo componente (n_components pedido).
        self.coef_ = Bhat[:, -1]
        self.intercept_ = intercepts[-1]
        return self

    def predict(self, X: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        check_is_fitted(self, "coef_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X tem {X.shape[1]} variaveis, modelo foi ajustado com "
                f"{self.n_features_in_}")
        if n_components is None:
            coef, intercept = self.coef_, self.intercept_
        else:
            k = int(n_components)
            if not (1 <= k <= self.bhat_path_.shape[0]):
                raise ValueError(
                    f"n_components={k} fora do intervalo ajustado "
                    f"[1, {self.bhat_path_.shape[0]}]")
            coef = self.bhat_path_[k - 1]
            intercept = self.intercept_path_[k - 1]
        return X @ coef + intercept
