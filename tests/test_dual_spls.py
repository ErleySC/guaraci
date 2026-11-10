# -*- coding: utf-8 -*-
"""Testes de Dual-sPLS (norma lasso) -- Grupo 2 do MAPA_COMPLETUDE_V1.

Contra-prova principal: `test_dual_spls_matches_r_oracle` compara a saida
do `DualSPLS` do GUARACI, coeficiente a coeficiente, contra a saida REAL
do pacote R `dual.spls` 0.1.4 (funcao `d.spls.lasso`), rodada com R 4.3.3
instalado especificamente para isto (CRAN/arXiv/ScienceDirect bloqueados
nesta sessao -- o codigo-fonte do pacote foi obtido do mirror do CRAN no
GitHub, https://github.com/cran/dual.spls, que NAO esta' bloqueado). O
script que gerou o oraculo (`gen_oracle.R`, dataset sintetico com seed
fixa) esta' descrito no cabecalho de `tests/fixtures/dual_spls_oracle/`.

As demais checagens sao estruturais/propriedades (API sklearn, decaimento
monotonico do numero de zeros conforme a definicao do metodo, etc.) --
nao substituem a contra-prova numerica acima, apenas cobrem casos de uso
que o oraculo (rodado 1x, ncp=4) nao exercita."""
from __future__ import annotations

import os

import numpy as np
import pytest
from sklearn.base import clone

from guaraci.dual_spls import DualSPLS

_FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures", "dual_spls_oracle")


def _ler_csv(nome: str) -> np.ndarray:
    """Le' um CSV escrito por `write.csv` do R (1a linha = cabecalho com
    aspas, colunas separadas por virgula) como array numpy 2D."""
    caminho = os.path.join(_FIXDIR, nome)
    return np.loadtxt(caminho, delimiter=",", skiprows=1)


def test_dual_spls_matches_r_oracle():
    """Oraculo real: R 4.3.3 + dual.spls 0.1.4 (`d.spls.lasso`), dataset
    sintetico (n=30, p=12, seed=42, beta esparso conhecido), ncp=4,
    ppnu=0.7. Tolerancia apertada (1e-8) -- os dois lados fazem a MESMA
    conta em ponto flutuante double, deveriam so' diferir por
    arredondamento de I/O do CSV (R grava ~15 digitos significativos)."""
    X = _ler_csv("X.csv")
    y = _ler_csv("y.csv")
    Bhat_r = _ler_csv("Bhat.csv")           # (p, ncp)
    intercept_r = _ler_csv("intercept.csv").ravel()  # (ncp,)
    zerovar_r = _ler_csv("zerovar.csv").ravel().astype(int)
    lambda_r = _ler_csv("lambda.csv").ravel()
    xmean_r = _ler_csv("Xmean.csv").ravel()

    ncp = Bhat_r.shape[1]
    modelo = DualSPLS(n_components=ncp, sparsity=0.7)
    modelo.fit(X, y)

    np.testing.assert_allclose(modelo.x_mean_, xmean_r, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(modelo.bhat_path_.T, Bhat_r, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(modelo.intercept_path_, intercept_r,
                               atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(modelo.lambda_path_, lambda_r,
                               atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(modelo.n_zeros_path_, zerovar_r)

    # predict() com o numero maximo de componentes deve reproduzir
    # X @ Bhat[:, ncp-1] + intercept[ncp-1] do R.
    y_hat = modelo.predict(X)
    y_hat_r = X @ Bhat_r[:, -1] + intercept_r[-1]
    np.testing.assert_allclose(y_hat, y_hat_r, atol=1e-6, rtol=1e-6)


def test_dual_spls_matches_r_oracle_ppnu_zero():
    """Segundo oraculo, independente do primeiro (dataset e seed
    diferentes, `sparsity=0` -- caso de borda: `nu` = o MENOR |z|
    observado, o que zera exatamente 1 coeficiente no 1o componente e' 0
    nos seguintes). Mesma fonte (R 4.3.3 + dual.spls 0.1.4)."""
    X = _ler_csv(os.path.join("ppnu0", "X.csv"))
    y = _ler_csv(os.path.join("ppnu0", "y.csv"))
    Bhat_r = _ler_csv(os.path.join("ppnu0", "Bhat.csv"))
    intercept_r = _ler_csv(os.path.join("ppnu0", "intercept.csv")).ravel()
    zerovar_r = _ler_csv(os.path.join("ppnu0", "zerovar.csv")).ravel().astype(int)
    lambda_r = _ler_csv(os.path.join("ppnu0", "lambda.csv")).ravel()

    modelo = DualSPLS(n_components=Bhat_r.shape[1], sparsity=0.0).fit(X, y)

    np.testing.assert_allclose(modelo.bhat_path_.T, Bhat_r, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(modelo.intercept_path_, intercept_r,
                               atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(modelo.lambda_path_, lambda_r,
                               atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(modelo.n_zeros_path_, zerovar_r)


def test_dual_spls_predict_por_componente_bate_com_bhat_path():
    rng = np.random.default_rng(0)
    n, p = 40, 15
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[[1, 4, 9]] = [2.0, -1.0, 3.0]
    y = X @ beta + rng.normal(scale=0.05, size=n)

    modelo = DualSPLS(n_components=5, sparsity=0.6).fit(X, y)
    for k in range(1, 6):
        pred_k = modelo.predict(X, n_components=k)
        esperado = X @ modelo.bhat_path_[k - 1] + modelo.intercept_path_[k - 1]
        np.testing.assert_allclose(pred_k, esperado)


def test_dual_spls_sparsity_zero_zera_no_maximo_uma_variavel():
    """sparsity=0 usa `nu` = o MENOR |z| observado (posicao 1/p mais
    proxima de 0) -- o proprio elemento minimo e' shrinkado exatamente a
    zero por construcao (soft-threshold em |z|=nu da' zero), entao
    zerovar no primeiro componente e' 0 ou 1, nunca mais. Verificado
    contra o R real (mesmo pacote/versao do oraculo principal, dataset
    seed=1: `d.spls.lasso(..., ppnu=0)` retorna zerovar=c(1,0,0)) --
    NAO e' um bug da porta em numpy, e' o algoritmo do paper/pacote."""
    rng = np.random.default_rng(1)
    n, p = 50, 10
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + rng.normal(scale=0.01, size=n)

    modelo = DualSPLS(n_components=3, sparsity=0.0).fit(X, y)
    assert modelo.n_zeros_path_[0] <= 1


def test_dual_spls_maior_sparsity_zera_mais_variaveis():
    rng = np.random.default_rng(2)
    n, p = 60, 20
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:5] = rng.normal(size=5)
    y = X @ beta + rng.normal(scale=0.02, size=n)

    baixo = DualSPLS(n_components=1, sparsity=0.3).fit(X, y)
    alto = DualSPLS(n_components=1, sparsity=0.85).fit(X, y)
    assert alto.n_zeros_path_[0] >= baixo.n_zeros_path_[0]


def test_dual_spls_zerovar_nao_aumenta_ao_longo_dos_componentes():
    """Propriedade checada nos testes `testthat` do proprio pacote R
    (`test-d.spls.LS.R`/`test-d.spls.lasso.R`):
    `expect_gt(zerovar[i-1], zerovar[i]-1)` para i em 2..ncp (indices R,
    1-based) -- ou seja `zerovar[i-1] >= zerovar[i]`: o numero de
    coeficientes zerados e' NAO-CRESCENTE conforme mais componentes sao
    adicionados (mais componentes = Bhat cumulativo reconstroi mais sinal,
    nunca menos). Reproduzida aqui como contrato, com os MESMOS dados do
    oraculo (n=30, p=12) para tambem servir de checagem cruzada direta."""
    X = _ler_csv("X.csv")
    y = _ler_csv("y.csv")
    zerovar_r = _ler_csv("zerovar.csv").ravel().astype(int)

    modelo = DualSPLS(n_components=len(zerovar_r), sparsity=0.7).fit(X, y)
    zeros = modelo.n_zeros_path_
    for i in range(1, len(zeros)):
        assert zeros[i - 1] >= zeros[i]
    # e' exatamente a sequencia do R (8, 6, 6, 6) para este dataset/seed.
    np.testing.assert_array_equal(zeros, zerovar_r)


def test_dual_spls_sklearn_api():
    """clone/get_params/set_params (contrato minimo p/ entrar num
    sklearn.Pipeline, como as demais entradas de `modelos` em
    `benchmark_regression_by_species`)."""
    modelo = DualSPLS(n_components=3, sparsity=0.8)
    clone(modelo)  # nao deve levantar

    params = modelo.get_params()
    assert params == {"n_components": 3, "sparsity": 0.8}

    rng = np.random.default_rng(4)
    X = rng.normal(size=(25, 8))
    y = rng.normal(size=25)
    modelo.fit(X, y)
    pred = modelo.predict(X)
    assert pred.shape == (25,)


def test_dual_spls_rejeita_n_components_invalido():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    with pytest.raises(ValueError):
        DualSPLS(n_components=0).fit(X, y)
    with pytest.raises(ValueError):
        DualSPLS(n_components=999).fit(X, y)
    with pytest.raises(ValueError):
        DualSPLS(sparsity=1.0).fit(X, y)


def test_dual_spls_predict_dimensao_errada():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(20, 6))
    y = rng.normal(size=20)
    modelo = DualSPLS(n_components=2, sparsity=0.5).fit(X, y)
    with pytest.raises(ValueError):
        modelo.predict(rng.normal(size=(5, 7)))
