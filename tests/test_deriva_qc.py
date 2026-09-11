# -*- coding: utf-8 -*-
"""Testes de deriva_qc.py -- proposta T4 da rodada multiagente de
2026-09-10 (QC-RLSC, Dunn et al. 2011, Nature Protocols 6:1060-1083,
DOI 10.1038/nprot.2011.335).
"""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.deriva_qc import corrigir_deriva_por_qc


def _dataset_com_deriva(seed=0, n_qc=12, n_amostras=30, p=15):
    """Deriva SUAVE e MONOTÔNICA ao longo da ordem de aquisição, igual
    para QCs e amostras (mesmo instrumento/sessão) -- exatamente o
    cenário que QC-RLSC corrige: o QC não carrega informação de teor, só
    de deriva instrumental."""
    rng = np.random.default_rng(seed)
    eixo = np.arange(p, dtype=float)
    espectro_base = np.exp(-0.5 * ((eixo - 7) / 2) ** 2) + 1.0

    ordem_total = n_qc + n_amostras
    deriva = 1.0 + 0.5 * (np.arange(ordem_total) / ordem_total)  # +50% ao longo da corrida

    ordem_qc = np.sort(rng.choice(ordem_total, size=n_qc, replace=False)).astype(float)
    ordem_amostras = np.setdiff1d(np.arange(ordem_total), ordem_qc).astype(float)[:n_amostras]

    ruido_qc = rng.normal(scale=0.01, size=(n_qc, p))
    X_qc = deriva[ordem_qc.astype(int)][:, None] * espectro_base[None, :] + ruido_qc

    amplitude_amostra = rng.uniform(0.5, 2.0, size=n_amostras)   # o "sinal" de interesse
    ruido_amostras = rng.normal(scale=0.01, size=(n_amostras, p))
    X_amostras = (deriva[ordem_amostras.astype(int)][:, None]
                  * amplitude_amostra[:, None] * espectro_base[None, :] + ruido_amostras)

    return X_amostras, ordem_amostras, X_qc, ordem_qc, amplitude_amostra


def test_corrige_reduz_a_tendencia_com_a_ordem():
    """Contra-prova central: apos a correcao, a norma das amostras nao
    deve mais crescer sistematicamente com a ordem de aquisicao."""
    X_amostras, ordem_amostras, X_qc, ordem_qc, _amp = _dataset_com_deriva()
    X_corrigido = corrigir_deriva_por_qc(X_amostras, ordem_amostras, X_qc, ordem_qc)

    normas_antes = np.linalg.norm(X_amostras, axis=1)
    normas_depois = np.linalg.norm(X_corrigido, axis=1)
    corr_antes = abs(float(np.corrcoef(ordem_amostras, normas_antes)[0, 1]))
    corr_depois = abs(float(np.corrcoef(ordem_amostras, normas_depois)[0, 1]))
    assert corr_depois < corr_antes


def test_corrige_preserva_a_amplitude_relativa_entre_amostras():
    """A correcao nao pode apagar o SINAL de interesse (amplitude
    diferente por amostra) -- so' a tendencia com a ordem."""
    X_amostras, ordem_amostras, X_qc, ordem_qc, amplitude = _dataset_com_deriva(seed=2)
    X_corrigido = corrigir_deriva_por_qc(X_amostras, ordem_amostras, X_qc, ordem_qc)
    normas_depois = np.linalg.norm(X_corrigido, axis=1)
    corr_amplitude = float(np.corrcoef(amplitude, normas_depois)[0, 1])
    assert corr_amplitude > 0.9   # ainda reflete a amplitude real de cada amostra


def test_shape_preservado():
    X_amostras, ordem_amostras, X_qc, ordem_qc, _ = _dataset_com_deriva(seed=3)
    Xc = corrigir_deriva_por_qc(X_amostras, ordem_amostras, X_qc, ordem_qc)
    assert Xc.shape == X_amostras.shape


def test_recusa_com_poucos_qcs_em_ordens_distintas():
    X_amostras = np.zeros((5, 10))
    ordem_amostras = np.arange(5, dtype=float)
    X_qc = np.zeros((3, 10))
    ordem_qc = np.array([0.0, 1.0, 2.0])   # so' 3 -- precisa de >=4
    with pytest.raises(ValueError, match="4"):
        corrigir_deriva_por_qc(X_amostras, ordem_amostras, X_qc, ordem_qc)


def test_recusa_canais_incompativeis():
    X_amostras = np.zeros((5, 10))
    ordem_amostras = np.arange(5, dtype=float)
    X_qc = np.zeros((6, 8))   # 8 canais, nao 10
    ordem_qc = np.arange(6, dtype=float)
    with pytest.raises(ValueError, match="canais"):
        corrigir_deriva_por_qc(X_amostras, ordem_amostras, X_qc, ordem_qc)


def test_amostra_fora_do_intervalo_de_qc_nao_explode():
    """Achado real desta rodada: sem o clamp de extrapolacao, uma amostra
    com ordem FORA do intervalo coberto pelos QCs fazia a spline
    extrapolar para um valor absurdo (dezenas de vezes o esperado). O
    clamp usa o extremo mais proximo em vez de extrapolar livremente."""
    rng = np.random.default_rng(9)
    p = 8
    ordem_qc = np.arange(8, dtype=float)          # QCs so' em 0..7
    X_qc = rng.normal(loc=5.0, scale=0.05, size=(8, p))
    ordem_amostras = np.arange(8, 18, dtype=float)  # amostras TODAS fora, em 8..17
    X_amostras = rng.normal(loc=5.0, scale=0.05, size=(10, p))
    Xc = corrigir_deriva_por_qc(X_amostras, ordem_amostras, X_qc, ordem_qc)
    # nada deveria fugir de uma faixa razoavel perto do nivel real (~5)
    assert np.all(np.abs(Xc) < 50.0)


def test_qc_constante_nao_altera_as_amostras():
    """Sem deriva real (QC constante), a correcao deve ser proxima da
    identidade -- nao inventa tendencia onde nao ha'."""
    rng = np.random.default_rng(4)
    p = 10
    ordem_qc = np.arange(8, dtype=float)
    X_qc = np.tile(rng.normal(size=p), (8, 1)) + rng.normal(scale=1e-4, size=(8, p))
    ordem_amostras = np.arange(8, 18, dtype=float)
    X_amostras = rng.normal(size=(10, p)) + 5.0
    Xc = corrigir_deriva_por_qc(X_amostras, ordem_amostras, X_qc, ordem_qc)
    assert np.allclose(Xc, X_amostras, rtol=0.05)
