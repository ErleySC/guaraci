# -*- coding: utf-8 -*-
"""alinhamento_retencao.py -- COW (Correlation Optimized Warping) para
cromatogramas (GC-MS/GC-FID/HPLC), Passo 150 da auditoria das 11
tecnicas analiticas (Fase D, 2026-09-04).

Referencia (confirmada no Crossref em 2026-09-04): Nielsen, N.-P. V.,
Carstensen, J. M. & Smedsgaard, J. (1998). "Aligning of single and
multiple wavelength chromatographic profiles for chemometric data
analysis using correlation optimised warping." Journal of
Chromatography A, 805(1-2), 17-35. DOI: 10.1016/S0021-9673(98)00021-1.

O QUE O ALGORITMO FAZ: alinha um cromatograma AMOSTRA a um cromatograma
REFERENCIA por deformacao linear POR PARTES (piecewise-linear warping).
Os dois sao divididos em `n_segmentos` (a referencia em segmentos de
tamanho FIXO; a amostra em segmentos cujo limite pode se deslocar
dentro de uma folga `slack`). Para cada escolha de limites, cada
segmento da amostra e' esticado/comprimido (interpolacao linear) para o
mesmo tamanho do segmento correspondente da referencia, e a correlacao
de Pearson entre os dois e' calculada. Programacao dinamica escolhe os
limites que MAXIMIZAM a soma das correlacoes segmento-a-segmento,
respeitando continuidade (o fim de um segmento e' o inicio do proximo).

LIMITACAO HONESTA: implementacao da formulacao classica (limites
inteiros na grade de amostragem, slack simetrico, correlacao de
Pearson como criterio) -- nao implementa as extensões multi-lambda
(multiplos comprimentos de onda simultaneos) nem pesos por segmento do
artigo original, que este projeto nao precisa (alinhamento de TIC de
GC-MS, um unico canal por cromatograma)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

__all__ = ["ResultadoCOW", "cow"]


@dataclass
class ResultadoCOW:
    """`amostra_alinhada` tem o MESMO comprimento da referencia (cada
    segmento da amostra foi reamostrado para o tamanho do segmento
    correspondente da referencia -- e' assim que COW produz saida numa
    grade comum, pre-requisito para empilhar amostras numa tabela).
    `nos_amostra`/`nos_referencia` sao os indices de fronteira de
    segmento escolhidos (mesmo comprimento, `n_segmentos + 1`);
    `correlacao_media` e' a media das correlacoes por segmento no
    alinhamento otimo -- diagnostico de qualidade (1.0 = perfeito)."""
    amostra_alinhada: np.ndarray
    nos_amostra: np.ndarray
    nos_referencia: np.ndarray
    correlacao_media: float


def _correlacao_segura(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson, mas 0.0 (nem NaN) se um dos dois for constante --
    segmento sem variacao nao pode contribuir sinal de alinhamento, mas
    tambem nao pode quebrar o max() da DP com NaN."""
    if a.std() < 1e-300 or b.std() < 1e-300:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def _interpolar_para(segmento: np.ndarray, novo_tamanho: int) -> np.ndarray:
    if len(segmento) == novo_tamanho:
        return segmento
    x_orig = np.linspace(0.0, 1.0, len(segmento))
    x_novo = np.linspace(0.0, 1.0, novo_tamanho)
    return np.interp(x_novo, x_orig, segmento)


def cow(referencia: np.ndarray, amostra: np.ndarray, n_segmentos: int,
        slack: int) -> ResultadoCOW:
    """Alinha `amostra` a `referencia` por COW. `n_segmentos` >= 2;
    `slack` >= 0 (folga em NUMERO DE PONTOS da grade da amostra ao
    redor da posicao nominal de cada limite intermediario).

    Levanta `ValueError` se `referencia`/`amostra` tiverem menos de
    `n_segmentos + 1` pontos (nao ha' como formar segmentos de
    comprimento >= 1)."""
    referencia = np.asarray(referencia, dtype=float)
    amostra = np.asarray(amostra, dtype=float)
    n_ref, n_am = len(referencia), len(amostra)
    if n_segmentos < 2:
        raise ValueError(f"n_segmentos={n_segmentos} -- precisa ser >= 2")
    if n_ref < n_segmentos + 1 or n_am < n_segmentos + 1:
        raise ValueError(
            f"referencia ({n_ref} pontos) ou amostra ({n_am} pontos) "
            f"curta demais para {n_segmentos} segmentos")

    nos_ref = np.round(np.linspace(0, n_ref - 1, n_segmentos + 1)).astype(int)
    nos_nominais_am = np.round(np.linspace(0, n_am - 1, n_segmentos + 1)).astype(int)

    # Janela de posicoes candidatas por no intermediario (0 e N sao fixos).
    janelas: List[np.ndarray] = [np.array([0])]
    for i in range(1, n_segmentos):
        lo = max(1, nos_nominais_am[i] - slack)
        hi = min(n_am - 2, nos_nominais_am[i] + slack)
        janelas.append(np.arange(lo, hi + 1))
    janelas.append(np.array([n_am - 1]))

    # dp[i] indexado pela POSICAO na janela do no i (nao pelo valor bruto).
    neg_inf = float("-inf")
    dp: List[np.ndarray] = [np.zeros(1)]
    voltar: List[np.ndarray] = [np.zeros(1, dtype=int)]

    for i in range(1, n_segmentos + 1):
        tam_ref_seg = nos_ref[i] - nos_ref[i - 1] + 1
        seg_ref = referencia[nos_ref[i - 1]:nos_ref[i] + 1]
        atual = np.full(len(janelas[i]), neg_inf)
        origem = np.zeros(len(janelas[i]), dtype=int)
        for k_atual, p_atual in enumerate(janelas[i]):
            melhor_val, melhor_k_prev = neg_inf, 0
            for k_prev, p_prev in enumerate(janelas[i - 1]):
                if p_atual - p_prev < 1:
                    continue   # segmento precisa de >=2 pontos (comprimento >=1)
                seg_am = amostra[p_prev:p_atual + 1]
                seg_am_interp = _interpolar_para(seg_am, tam_ref_seg)
                corr = _correlacao_segura(seg_am_interp, seg_ref)
                val = dp[i - 1][k_prev] + corr
                if val > melhor_val:
                    melhor_val, melhor_k_prev = val, k_prev
            atual[k_atual] = melhor_val
            origem[k_atual] = melhor_k_prev
        dp.append(atual)
        voltar.append(origem)

    # Backtrack a partir do (unico) no final.
    nos_escolhidos = [0] * (n_segmentos + 1)
    k = 0   # janelas[n_segmentos] tem 1 elemento so'
    nos_escolhidos[n_segmentos] = janelas[n_segmentos][0]
    for i in range(n_segmentos, 0, -1):
        k = voltar[i][k]
        nos_escolhidos[i - 1] = janelas[i - 1][k]

    nos_amostra = np.asarray(nos_escolhidos, dtype=int)
    correlacoes: List[float] = []
    pedacos: List[np.ndarray] = []
    for i in range(1, n_segmentos + 1):
        tam_ref_seg = nos_ref[i] - nos_ref[i - 1] + 1
        seg_am = amostra[nos_amostra[i - 1]:nos_amostra[i] + 1]
        seg_am_interp = _interpolar_para(seg_am, tam_ref_seg)
        seg_ref = referencia[nos_ref[i - 1]:nos_ref[i] + 1]
        correlacoes.append(_correlacao_segura(seg_am_interp, seg_ref))
        # Evita duplicar o ponto de fronteira compartilhado entre segmentos.
        pedacos.append(seg_am_interp[:-1] if i < n_segmentos else seg_am_interp)

    amostra_alinhada = np.concatenate(pedacos)
    return ResultadoCOW(
        amostra_alinhada=amostra_alinhada,
        nos_amostra=nos_amostra, nos_referencia=nos_ref,
        correlacao_media=float(np.mean(correlacoes)))
