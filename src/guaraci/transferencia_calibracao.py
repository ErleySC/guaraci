# -*- coding: utf-8 -*-
"""transferencia_calibracao.py -- Transferencia de calibracao entre
instrumentos (Passo 86).

PROBLEMA QUE RESOLVE. Um modelo PLS calibrado com espectros do
instrumento A tipicamente degrada quando aplicado a espectros do
instrumento B (mesma amostra, espectrometro diferente): deriva de
comprimento de onda, resposta de detector, otica -- tudo produz um
deslocamento sistematico que o modelo nunca viu no treino. Transferencia
de calibracao aprende esse deslocamento a partir de um PEQUENO conjunto de
amostras medidas nos DOIS instrumentos (amostras de transferencia) e
corrige espectros novos do instrumento B antes de passa-los pelo modelo
calibrado em A -- sem recalibrar do zero.

METODOS (Wang, Veltkamp & Kowalski 1991, "Multivariate instrument
standardization", Analytical Chemistry 63(23):2750-2756,
DOI: 10.1021/ac00023a016 -- verificado no Crossref em 2026-08-27):

- Direct Standardization (DS): UMA regressao ridge global, todos os canais
  do instrumento mestre contra TODOS os canais do escravo de uma vez
  (`X_mestre ~= X_escravo @ F + bias`). Simples, mas F denso (p x p) tende
  a superajustar deslocamentos que nao existem -- precisa de bastante
  regularizacao ou muitas amostras de transferencia.
- Piecewise Direct Standardization (PDS): UMA regressao ridge POR CANAL do
  mestre, contra so' uma JANELA local de canais vizinhos do escravo (o
  deslocamento entre instrumentos e' predominantemente LOCAL -- um pico
  desloca poucos canais, nao o espectro inteiro). F sai em BANDA (a
  maioria das entradas de cada coluna e' zero). E' a variante que o artigo
  original recomenda para espectros contiguos (NIR/FT-NIR); usada aqui como
  o metodo primario.

REGULARIZACAO: o artigo original usa PCR/PLS por janela (reducao de
dimensionalidade) porque a janela pode ter mais canais que amostras de
transferencia disponiveis, o que deixaria OLS indeterminado. Esta
implementacao usa ridge (regressao regularizada em vez de reduzida) para o
MESMO proposito -- estabilidade numerica com poucas amostras de
transferencia relativas a largura da janela -- com a vantagem de nao
precisar escolher um numero de componentes por janela.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "StandardizationTransform",
    "direct_standardization",
    "piecewise_direct_standardization",
    "apply_standardization",
]


@dataclass
class StandardizationTransform:
    """Transformacao aprendida: `X_padronizado = X_escravo_novo @ F + bias`.

    `F` e' (p, p): denso p/ DS, em banda (maioria de entradas zero) p/ PDS.
    `janela` e' None p/ DS (nao ha' janela -- e' global).
    """
    F: np.ndarray
    bias: np.ndarray
    metodo: str
    n_amostras_transferencia: int
    janela: Optional[int] = None
    alpha: float = 0.0


def _regressao_local_ridge(X_local: np.ndarray, y: np.ndarray, alpha: float
                            ) -> "tuple[np.ndarray, float]":
    """Ridge de UM alvo `y` contra as colunas de `X_local`, centrado (nao
    penaliza o intercepto). `alpha=0` cai em OLS (minimos quadrados puro)."""
    X_local = np.asarray(X_local, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = X_local.mean(axis=0)
    ym = float(y.mean())
    Xc = X_local - xm
    yc = y - ym
    p = Xc.shape[1]
    coefs = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(p), Xc.T @ yc)
    bias = ym - float(xm @ coefs)
    return coefs, bias


def direct_standardization(X_master: np.ndarray, X_slave: np.ndarray,
                            alpha: float = 1.0) -> StandardizationTransform:
    """DS: UMA ridge global (todos os canais de uma vez -- `np.linalg.solve`
    resolve para todas as colunas-alvo simultaneamente, MESMA fatoracao).

    `X_master`/`X_slave`: (n_transferencia, p) -- as MESMAS `n` amostras,
    medidas nos dois instrumentos, na MESMA ordem de linha (pareadas).
    `alpha`: forca da regularizacao ridge (>=0; 0 = OLS puro, arriscado
    p/ p grande com poucas amostras de transferencia -- ver docstring do
    modulo).
    """
    X_master = np.asarray(X_master, dtype=float)
    X_slave = np.asarray(X_slave, dtype=float)
    if X_master.shape != X_slave.shape:
        raise ValueError(
            f"X_master {X_master.shape} e X_slave {X_slave.shape} precisam "
            f"ter a MESMA forma -- sao as mesmas amostras nos dois "
            f"instrumentos, pareadas por linha.")
    n, p = X_slave.shape
    slave_mean = X_slave.mean(axis=0)
    master_mean = X_master.mean(axis=0)
    Xs_c = X_slave - slave_mean
    Xm_c = X_master - master_mean
    F = np.linalg.solve(Xs_c.T @ Xs_c + alpha * np.eye(p), Xs_c.T @ Xm_c)
    bias = master_mean - slave_mean @ F
    return StandardizationTransform(F=F, bias=bias, metodo="DS",
                                     n_amostras_transferencia=n, alpha=alpha)


def piecewise_direct_standardization(X_master: np.ndarray, X_slave: np.ndarray,
                                      janela: int = 5, alpha: float = 1.0
                                      ) -> StandardizationTransform:
    """PDS: uma ridge POR CANAL do mestre, contra a janela local
    `[j-janela, j+janela]` (recortada nas bordas) dos canais do escravo.

    `janela`: meia-largura em NUMERO DE CANAIS (nao em unidade espectral --
    depende da resolucao do eixo). `janela=0` degenera em ridge canal-a-
    canal sem vizinhanca (equivalente a uma correcao multiplicativa/aditiva
    por canal, sem informacao dos vizinhos).
    """
    X_master = np.asarray(X_master, dtype=float)
    X_slave = np.asarray(X_slave, dtype=float)
    if X_master.shape != X_slave.shape:
        raise ValueError(
            f"X_master {X_master.shape} e X_slave {X_slave.shape} precisam "
            f"ter a MESMA forma -- sao as mesmas amostras nos dois "
            f"instrumentos, pareadas por linha.")
    if janela < 0:
        raise ValueError(f"janela={janela} invalida -- use >= 0")
    n, p = X_slave.shape
    F = np.zeros((p, p))
    bias = np.zeros(p)
    for j in range(p):
        lo = max(0, j - janela)
        hi = min(p, j + janela + 1)
        coefs, b = _regressao_local_ridge(X_slave[:, lo:hi], X_master[:, j], alpha)
        F[lo:hi, j] = coefs
        bias[j] = b
    return StandardizationTransform(F=F, bias=bias, metodo="PDS",
                                     n_amostras_transferencia=n,
                                     janela=janela, alpha=alpha)


def apply_standardization(X_novo: np.ndarray, transform: StandardizationTransform
                           ) -> np.ndarray:
    """Aplica a transformacao aprendida a espectros NOVOS do instrumento
    escravo -- devolve a versao padronizada (equivalente-ao-mestre), pronta
    para entrar num modelo calibrado no mestre."""
    X_novo = np.asarray(X_novo, dtype=float)
    if X_novo.shape[-1] != transform.F.shape[0]:
        raise ValueError(
            f"X_novo tem {X_novo.shape[-1]} canais, mas a transformacao foi "
            f"aprendida com {transform.F.shape[0]} -- instrumento/faixa "
            f"espectral incompativel.")
    return X_novo @ transform.F + transform.bias
