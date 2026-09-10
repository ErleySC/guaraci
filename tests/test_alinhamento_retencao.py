# -*- coding: utf-8 -*-
"""Testes de COW (Correlation Optimized Warping) em
`alinhamento_retencao.py` -- Passo 150 (Fase D, auditoria das 11
tecnicas, 2026-09-04). Referencia: Nielsen, Carstensen & Smedsgaard
(1998), J. Chromatogr. A 805:17-35, DOI 10.1016/S0021-9673(98)00021-1
(confirmada no Crossref).

Contra-prova obrigatoria (regra 9): cromatograma sintetico com um WARP
NAO-LINEAR CONHECIDO aplicado -- o COW tem que recuperar o alinhamento
(correlacao apos > correlacao antes), sem nunca ver o warp verdadeiro."""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.alinhamento_retencao import ResultadoCOW, cow


def _picos_gaussianos(centros, larguras, x):
    y = np.zeros_like(x)
    for c, w in zip(centros, larguras):
        y = y + np.exp(-((x - c) ** 2) / (2 * w ** 2))
    return y


def _cromatograma_com_warp(seed: int, amplitude_warp: float, n: int = 400):
    """Referencia = 3 picos gaussianos numa grade regular. Amostra = OS
    MESMOS 3 picos, mas medidos numa grade de tempo distorcida
    (warp senoidal suave, `amplitude_warp` controla a intensidade) --
    simula deriva nao-linear de tempo de retencao entre corridas de
    GC."""
    x = np.linspace(0.0, 1.0, n)
    centros, larguras = [0.2, 0.5, 0.8], [0.02, 0.03, 0.02]
    referencia = _picos_gaussianos(centros, larguras, x)

    warp = x + amplitude_warp * np.sin(2 * np.pi * x)
    warp = np.clip(warp, 0.0, 1.0)
    tempos_amostra = np.sort(warp)
    amostra_no_tempo_verdadeiro = _picos_gaussianos(centros, larguras, tempos_amostra)
    # Reamostra na MESMA grade de indices que a referencia (sem corrigir
    # o warp ainda) -- e' isso que chega de um instrumento real.
    amostra_bruta = np.interp(x, np.linspace(0.0, 1.0, n), amostra_no_tempo_verdadeiro)
    return referencia, amostra_bruta


def test_cow_recupera_alinhamento_de_warp_nao_linear_conhecido():
    referencia, amostra_bruta = _cromatograma_com_warp(seed=0, amplitude_warp=0.05)

    corr_antes = float(np.corrcoef(referencia, amostra_bruta)[0, 1])
    resultado = cow(referencia, amostra_bruta, n_segmentos=20, slack=8)
    corr_depois = float(np.corrcoef(referencia, resultado.amostra_alinhada)[0, 1])

    assert corr_depois > corr_antes + 0.15, (
        f"COW deveria melhorar substancialmente a correlacao com um warp "
        f"conhecido: antes={corr_antes:.3f}, depois={corr_depois:.3f}")
    assert resultado.correlacao_media > 0.7
    assert resultado.amostra_alinhada.shape == referencia.shape


def test_cow_sem_warp_nao_piora_a_correlacao():
    """Contra-prova complementar: se a amostra JA' estiver alinhada
    (warp=0), o COW nao pode DESTRUIR o alinhamento -- a correlacao
    depois tem que ficar tao alta quanto antes (dentro de folga
    numerica da interpolacao)."""
    referencia, amostra_sem_warp = _cromatograma_com_warp(seed=1, amplitude_warp=0.0)
    corr_antes = float(np.corrcoef(referencia, amostra_sem_warp)[0, 1])
    resultado = cow(referencia, amostra_sem_warp, n_segmentos=20, slack=8)
    corr_depois = float(np.corrcoef(referencia, resultado.amostra_alinhada)[0, 1])
    assert corr_depois >= corr_antes - 0.02


def test_cow_rejeita_poucos_segmentos_ou_serie_curta_demais():
    x = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="n_segmentos"):
        cow(x, x, n_segmentos=1, slack=1)
    with pytest.raises(ValueError, match="curta demais"):
        cow(x, x, n_segmentos=20, slack=1)


def test_resultado_cow_e_dataclass_com_campos_esperados():
    referencia, amostra = _cromatograma_com_warp(seed=2, amplitude_warp=0.03)
    resultado = cow(referencia, amostra, n_segmentos=10, slack=5)
    assert isinstance(resultado, ResultadoCOW)
    assert len(resultado.nos_referencia) == 11
    assert len(resultado.nos_amostra) == 11
    assert resultado.nos_amostra[0] == 0
    assert resultado.nos_amostra[-1] == len(amostra) - 1
