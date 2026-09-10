# -*- coding: utf-8 -*-
"""Testes de transferencia_calibracao.py (Passo 86) -- dados SINTETICOS,
com deslocamento entre instrumentos conhecido por construcao (o teste
contra o dataset publico Corn, multiinstrumento de verdade, esta em
test_validacao_publica.py, guardado por @requer_corn).
"""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.transferencia_calibracao import (
    apply_standardization,
    direct_standardization,
    piecewise_direct_standardization,
)


def _espectros_com_deslocamento(seed=0, n=60, p=50, deslocamento=2,
                                escala_extra=1.08, ruido=0.01):
    """Espectros MESTRE (picos gaussianos, loadings aleatorios por amostra)
    e uma versao ESCRAVA deslocada em `deslocamento` CANAIS + ganho extra --
    o tipo de distorcao que PDS foi desenhado para corrigir (deslocamento
    predominantemente LOCAL, nao um embaralhamento global)."""
    rng = np.random.default_rng(seed)
    eixo = np.arange(p, dtype=float)
    centros_pico = rng.uniform(5, p - 5, size=4)
    larguras = rng.uniform(2.0, 4.0, size=4)
    bases = np.array([np.exp(-0.5 * ((eixo - c) / w) ** 2)
                       for c, w in zip(centros_pico, larguras)])   # (4, p)

    loadings = rng.uniform(0.5, 2.0, size=(n, 4))
    X_master = loadings @ bases                                    # (n, p)

    # Escravo: MESMO espectro, canal j do escravo == canal (j-deslocamento)
    # do mestre (pad com borda) * ganho extra + ruido -- desloca o PICO
    # inteiro por `deslocamento` canais, efeito predominantemente local.
    X_slave = np.zeros_like(X_master)
    for j in range(p):
        j_fonte = min(max(j - deslocamento, 0), p - 1)
        X_slave[:, j] = X_master[:, j_fonte]
    X_slave = X_slave * escala_extra
    X_master = X_master + rng.normal(scale=ruido, size=X_master.shape)
    X_slave = X_slave + rng.normal(scale=ruido, size=X_slave.shape)
    return X_master, X_slave


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# =========================================================================
#  Contrato de forma / erro
# =========================================================================

def test_direct_standardization_exige_mesma_forma():
    X_master, X_slave = _espectros_com_deslocamento(n=10, p=20)
    with pytest.raises(ValueError, match="MESMA forma"):
        direct_standardization(X_master, X_slave[:, :-1])


def test_piecewise_exige_mesma_forma():
    X_master, X_slave = _espectros_com_deslocamento(n=10, p=20)
    with pytest.raises(ValueError, match="MESMA forma"):
        piecewise_direct_standardization(X_master[:-1], X_slave)


def test_apply_standardization_exige_mesmo_numero_de_canais():
    X_master, X_slave = _espectros_com_deslocamento(n=10, p=20)
    transform = direct_standardization(X_master, X_slave)
    with pytest.raises(ValueError, match="canais"):
        apply_standardization(X_slave[:, :-1], transform)


# =========================================================================
#  A transferencia de fato reduz o erro entre instrumentos
# =========================================================================

def test_pds_reduz_erro_entre_instrumentos():
    X_master, X_slave = _espectros_com_deslocamento(seed=1, n=80, p=60,
                                                     deslocamento=2)
    idx = np.arange(80)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    idx_transf, idx_teste = idx[:20], idx[20:]

    erro_sem_transferencia = _rmse(X_master[idx_teste], X_slave[idx_teste])
    assert erro_sem_transferencia > 0.05, (
        "o deslocamento sintetico precisa ser grande o bastante p/ o teste "
        "nao ser vacuo -- se o erro sem transferencia ja' e' minusculo, "
        "qualquer metodo pareceria 'funcionar'")

    transform = piecewise_direct_standardization(
        X_master[idx_transf], X_slave[idx_transf], janela=5, alpha=1.0)
    X_slave_padronizado = apply_standardization(X_slave[idx_teste], transform)
    erro_com_pds = _rmse(X_master[idx_teste], X_slave_padronizado)

    assert erro_com_pds < erro_sem_transferencia * 0.5, (
        f"PDS deveria reduzir o erro entre instrumentos por pelo menos "
        f"metade (sem transferencia={erro_sem_transferencia:.4f}, "
        f"com PDS={erro_com_pds:.4f})")


def test_ds_reduz_erro_entre_instrumentos():
    X_master, X_slave = _espectros_com_deslocamento(seed=2, n=80, p=60,
                                                     deslocamento=2)
    idx = np.arange(80)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    idx_transf, idx_teste = idx[:30], idx[30:]

    erro_sem_transferencia = _rmse(X_master[idx_teste], X_slave[idx_teste])

    transform = direct_standardization(X_master[idx_transf], X_slave[idx_transf],
                                       alpha=5.0)
    X_slave_padronizado = apply_standardization(X_slave[idx_teste], transform)
    erro_com_ds = _rmse(X_master[idx_teste], X_slave_padronizado)

    assert erro_com_ds < erro_sem_transferencia, (
        f"DS deveria reduzir o erro entre instrumentos (sem="
        f"{erro_sem_transferencia:.4f}, com DS={erro_com_ds:.4f})")


# =========================================================================
#  Contra-prova: SEM amostras de transferencia suficientes/uteis (mestre e
#  escravo TROCADOS por ruido puro, sem relacao nenhuma), a transferencia
#  nao pode reduzir erro -- prova que os testes acima nao passariam so'
#  porque "qualquer F reduz erro por acaso".
# =========================================================================

def test_contraprova_sem_relacao_real_pds_nao_melhora_erro():
    rng = np.random.default_rng(7)
    n, p = 60, 40
    X_master = rng.normal(size=(n, p))
    X_slave = rng.normal(size=(n, p))   # SEM NENHUMA relacao com X_master

    idx_transf, idx_teste = np.arange(0, 15), np.arange(15, 60)
    erro_sem_transferencia = _rmse(X_master[idx_teste], X_slave[idx_teste])

    transform = piecewise_direct_standardization(
        X_master[idx_transf], X_slave[idx_transf], janela=5, alpha=1.0)
    X_slave_padronizado = apply_standardization(X_slave[idx_teste], transform)
    erro_com_pds = _rmse(X_master[idx_teste], X_slave_padronizado)

    assert erro_com_pds >= erro_sem_transferencia * 0.9, (
        f"com mestre/escravo SEM relacao real, PDS nao deveria conseguir "
        f"reduzir o erro de forma relevante (sem={erro_sem_transferencia:.4f}, "
        f"com PDS={erro_com_pds:.4f}) -- se reduziu, o teste positivo acima "
        f"poderia estar passando por coincidencia numerica, nao por "
        f"recuperar uma relacao real")
