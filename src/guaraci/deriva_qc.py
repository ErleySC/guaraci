# -*- coding: utf-8 -*-
"""deriva_qc.py — Correção de deriva de sinal ao longo da ordem de
aquisição usando amostras de controle de qualidade (QC/branco)
intercaladas, o método conhecido como QC-RLSC (*QC-based Robust LOESS
Signal Correction*), Dunn, Broadhurst, Begley, Zelena, Francis-McIntyre,
Anderson, Brown, Knowles, Halsall, Haselden, Nicholls, Wilson, Kell &
Goodacre (2011), *Nature Protocols* 6:1060-1083, DOI 10.1038/nprot.2011.335.

Proposta T4 da rodada multiagente de 2026-09-10.

PRÉ-REQUISITO DE DADO -- NÃO DISPONÍVEL HOJE NO GUARACI (documentado, não
contornado): esta correção exige leituras de QC/branco com ORDEM (ou
timestamp) de aquisição conhecida. O dataset PRIVADO de óleo não tem
isso -- o parser DX (`dados_io.parse_dx`) não expõe o timestamp do
`##AUDIT TRAIL` (achado #4 da rodada multiagente: a ordem de leitura só
foi medida por um script privado, nunca portada para o produto). O
candidato público mais próximo, GC-IMS Zenodo `19209004`, tem 14 QCs
(`docs/VALIDACAO_PUBLICA.md` §2j), mas a ordem de injeção real desses QCs
**não foi confirmada disponível** nos metadados públicos. Esta função é
utilizável hoje só com dado que o CHAMADOR forneça (ordem/timestamp real),
não é ligada a nenhum fluxo automático do pacote enquanto esse
pré-requisito não for satisfeito.

É a única via de correção compatível com o achado de que a ordem de
leitura confunde o teor no dataset próprio (`~/.guaraci_local/CLAUDE.md`,
P12): a deriva é estimada SÓ pelos QCs, que não dependem do teor da
amostra -- diferente de qualquer correção que olhasse para o teor
declarado, o que reintroduziria a colinearidade ordem×teor no próprio
mecanismo de correção.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import UnivariateSpline

__all__ = ["corrigir_deriva_por_qc"]


def corrigir_deriva_por_qc(X_amostras: np.ndarray, ordem_amostras: np.ndarray,
                            X_qc: np.ndarray, ordem_qc: np.ndarray, *,
                            suavizacao: Optional[float] = None) -> np.ndarray:
    """Corrige a deriva de `X_amostras` ao longo de `ordem_amostras`,
    canal a canal, usando uma curva SUAVE ajustada ao sinal de QC
    (`X_qc`, medido nas ordens `ordem_qc`) contra a ordem de aquisição.

    Para cada canal: ajusta uma spline suave ao sinal do QC em função da
    ordem, avalia essa curva na ordem de cada amostra, e divide a
    amostra pelo valor da curva naquele ponto (renormalizado pela média
    geral do QC no canal, para manter a escala original). O resultado é
    que o nível médio de QC fica CONSTANTE ao longo da ordem -- a mesma
    lógica de correção de lote/sessão do QC-RLSC (Dunn et al. 2011).

    `suavizacao` é o parâmetro `s` de `scipy.interpolate.UnivariateSpline`
    (maior = mais suave); por padrão é estimado da variância do próprio
    QC (heurística simples, não uma otimização por validação cruzada).

    Amostra com ordem FORA do intervalo coberto pelos QCs usa o valor da
    curva no extremo mais próximo (constante além da borda) -- splines
    extrapolam mal, e um valor absurdo na correção seria pior que assumir
    a última deriva conhecida.

    Levanta `ValueError` se houver menos de 4 QCs em ordens distintas --
    não dá para ajustar uma curva suave confiável com menos que isso.
    """
    X_amostras = np.asarray(X_amostras, dtype=float)
    ordem_amostras = np.asarray(ordem_amostras, dtype=float)
    X_qc = np.asarray(X_qc, dtype=float)
    ordem_qc = np.asarray(ordem_qc, dtype=float)

    if X_amostras.ndim != 2 or X_qc.ndim != 2:
        raise ValueError("X_amostras e X_qc precisam ser 2D (n, p)")
    if X_amostras.shape[1] != X_qc.shape[1]:
        raise ValueError(
            f"X_amostras tem {X_amostras.shape[1]} canais, X_qc tem "
            f"{X_qc.shape[1]} -- precisam ser o mesmo eixo espectral.")
    if len(ordem_amostras) != len(X_amostras):
        raise ValueError("ordem_amostras precisa ter o mesmo n de X_amostras")
    if len(ordem_qc) != len(X_qc):
        raise ValueError("ordem_qc precisa ter o mesmo n de X_qc")
    if len(np.unique(ordem_qc)) < 4:
        raise ValueError(
            f"{len(np.unique(ordem_qc))} ordem(ns) distinta(s) de QC -- "
            f"precisa de pelo menos 4 para ajustar uma curva suave de "
            f"deriva com confiança mínima.")

    ordem_sorted_idx = np.argsort(ordem_qc)
    ordem_qc_ordenada = ordem_qc[ordem_sorted_idx]
    n_canais = X_amostras.shape[1]
    referencia = X_qc.mean(axis=0)

    # Splines EXTRAPOLAM mal (podem disparar para valores absurdos fora do
    # intervalo observado de QC). Uma amostra com ordem fora do intervalo
    # [min(ordem_qc), max(ordem_qc)] usa o valor da curva no EXTREMO mais
    # proximo (constante alem da borda) -- estavel, e' a suposicao honesta
    # de "sem QC mais perto, assume a ultima deriva conhecida", nao uma
    # extrapolacao livre.
    ordem_amostras_dentro = np.clip(
        ordem_amostras, ordem_qc_ordenada[0], ordem_qc_ordenada[-1])

    X_corrigido = np.empty_like(X_amostras)
    for canal in range(n_canais):
        y = X_qc[ordem_sorted_idx, canal]
        grau = min(3, len(y) - 1)
        s = suavizacao if suavizacao is not None else len(y) * float(np.var(y)) * 0.5
        try:
            curva = UnivariateSpline(ordem_qc_ordenada, y, k=grau, s=s)
            curva_em_amostras = curva(ordem_amostras_dentro)
        except Exception:  # noqa: BLE001 -- spline degenerada (ex.: QC ~constante,
            # poucos pontos distintos apos remover duplicatas de ordem);
            # cai no fallback de nao corrigir esse canal (fator neutro),
            # nunca propaga NaN/excecao para o chamador.
            curva_em_amostras = np.full(len(ordem_amostras), y.mean())

        fator = np.where(np.abs(curva_em_amostras) > 1e-300, curva_em_amostras, 1.0)
        X_corrigido[:, canal] = X_amostras[:, canal] * referencia[canal] / fator

    return X_corrigido
