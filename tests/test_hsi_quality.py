"""Testes de hsi_quality.py (Passo 95) -- contra-prova obrigatoria da
instrucao: injetar cubo sintetico saturado/com SNR baixo e confirmar
rejeicao (nunca processamento silencioso)."""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.hsi_quality import estimate_noise_sigma, evaluate_cube_quality


def _cubo_limpo(seed: int = 0, altura: int = 32, largura: int = 32,
                 bandas: int = 16) -> np.ndarray:
    """Cubo sintetico "bem comportado": reflectancia em [0,1], variacao
    espacial suave (cena real nunca e' ruido puro) + ruido pequeno."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:altura, 0:largura]
    base = 0.3 + 0.2 * np.sin(x / 6.0) * np.cos(y / 6.0)
    cubo = np.stack([base + 0.01 * rng.normal(size=(altura, largura))
                     for _ in range(bandas)], axis=-1)
    return cubo


# ── estimate_noise_sigma ────────────────────────────────────────────────

def test_estimate_noise_sigma_cresce_com_ruido_injetado():
    rng = np.random.default_rng(0)
    limpa = np.full((40, 40), 0.5)
    sigma_limpa = estimate_noise_sigma(limpa)
    ruidosa = limpa + rng.normal(scale=0.05, size=limpa.shape)
    sigma_ruidosa = estimate_noise_sigma(ruidosa)
    assert sigma_limpa == pytest.approx(0.0, abs=1e-9)
    assert sigma_ruidosa > sigma_limpa
    # A estimativa deve ficar na mesma ordem de grandeza do sigma injetado
    # (Immerkaer 1996 e' uma estimativa, nao um valor exato).
    assert 0.02 < sigma_ruidosa < 0.10


def test_estimate_noise_sigma_imagem_pequena_demais_levanta_erro():
    with pytest.raises(ValueError, match="3D|3x3"):
        estimate_noise_sigma(np.zeros((2, 2)))


# ── evaluate_cube_quality: aceitacao no caso normal ─────────────────────

def test_evaluate_cube_quality_aceita_cubo_limpo():
    resultado = evaluate_cube_quality(_cubo_limpo())
    assert resultado.aceito is True
    assert resultado.motivo is None
    assert resultado.fracao_pixels_validos == 1.0
    assert resultado.fracao_pixels_saturados == 0.0


# ── contra-prova OBRIGATORIA (Passo 95): saturacao ──────────────────────

def test_evaluate_cube_quality_rejeita_cubo_saturado():
    cubo = _cubo_limpo()
    cubo[:20, :20, :] = 5.0   # bloco bem fora de [0,1] -- clipping simulado
    resultado = evaluate_cube_quality(cubo)
    assert resultado.aceito is False
    assert "satura" in resultado.motivo.lower()


# ── contra-prova OBRIGATORIA (Passo 95): SNR baixo ──────────────────────

def test_evaluate_cube_quality_rejeita_snr_baixo():
    rng = np.random.default_rng(1)
    altura, largura, bandas = 32, 32, 16
    # Sinal fraco + ruido forte -- SNR ruim de proposito.
    cubo = 0.05 + 0.5 * rng.normal(size=(altura, largura, bandas))
    resultado = evaluate_cube_quality(cubo, fracao_saturacao_max=0.5)
    assert resultado.aceito is False
    assert "snr" in resultado.motivo.lower()


def test_evaluate_cube_quality_rejeita_pixels_invalidos_em_excesso():
    cubo = _cubo_limpo()
    cubo[:, :, 0] = np.nan  # 1 banda inteira invalida
    resultado = evaluate_cube_quality(cubo, fracao_validos_minima=0.99)
    assert resultado.aceito is False
    assert "validos" in resultado.motivo.lower()


def test_evaluate_cube_quality_rejeicao_e_fail_fast_motivo_unico():
    """Cubo com VARIOS problemas ao mesmo tempo -- motivo deve ser um so'
    (o primeiro criterio verificado), nunca uma lista vaga."""
    cubo = _cubo_limpo()
    cubo[:, :, 0] = np.nan
    cubo[:20, :20, :] = 5.0
    resultado = evaluate_cube_quality(cubo, fracao_validos_minima=0.99)
    assert resultado.aceito is False
    assert resultado.motivo is not None and resultado.motivo.count(".") <= 3


def test_evaluate_cube_quality_forma_errada_levanta_erro_claro():
    with pytest.raises(ValueError, match="3D"):
        evaluate_cube_quality(np.zeros((10, 10)))
