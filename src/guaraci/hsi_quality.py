"""hsi_quality.py — Quality gate de cubo hiperespectral (Passo 95 da
`INSTRUCAO_HSI_MINIMO_VIAVEL.md`): saturacao, SNR minimo, proporcao de
pixels validos. Resultado e' sempre "aceitar" ou "rejeitar, motivo X" --
NUNCA processa uma cena que falha no gate em silencio (mesmo espirito do
`log.warning` explicito em `figuras.py`/`pipeline.py` para falha de
salvamento, ver Passo residual anterior desta auditoria).

Calibracao radiometrica (converter para reflectancia relativa com
referencias de branco/preto): o dataset publico usado nesta integracao
(DeepHS Fruit, ver `hsi_io.py`) ja' vem calibrado -- os valores de cubo
inspecionados diretamente (Kaki/VIS) ficam no intervalo aproximado
[-0.12, 0.60], compativel com reflectancia relativa ja' corrigida (a
propria camera "VIS_COR" do dataset e' descrita pelos autores como
"corrected"). Por isso este modulo NAO implementa a etapa de calibracao
por referencia de branco/preto (nao ha' cubo de referencia bruto
disponivel nesta integracao para testa-la de verdade) -- documentado
aqui, nao escondido; ver `docs/PROGRESSO.md`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import convolve2d

__all__ = [
    "QualityGateResult",
    "estimate_noise_sigma",
    "evaluate_cube_quality",
]

# Mascara Laplaciana de Immerkaer (1996), "Fast Noise Variance Estimation",
# CVGIP: Graphical Models and Image Processing 64(2):300-302 -- estimativa
# de sigma de ruido de UMA UNICA imagem (sem referencia/replica), robusta a
# textura de baixa frequencia da cena.
_LAPLACIANO_IMMERKAER = np.array([[1.0, -2.0, 1.0],
                                   [-2.0, 4.0, -2.0],
                                   [1.0, -2.0, 1.0]])


def estimate_noise_sigma(imagem_2d: np.ndarray) -> float:
    """Estimativa rapida de sigma de ruido de UMA imagem 2D (Immerkaer
    1996) -- nao precisa de referencia/replica/cena plana. Convolucao com
    o Laplaciano 3x3 acima anula sinal de baixa frequencia (a cena em si)
    e deixa passar predominantemente ruido de alta frequencia."""
    img = np.asarray(imagem_2d, dtype=float)
    if img.ndim != 2 or img.shape[0] < 3 or img.shape[1] < 3:
        raise ValueError(
            f"estimate_noise_sigma espera uma imagem 2D com >=3x3 pixels, "
            f"recebeu shape {img.shape}.")
    conv = convolve2d(img, _LAPLACIANO_IMMERKAER, mode="valid")
    n = (img.shape[0] - 2) * (img.shape[1] - 2)
    return float(np.sqrt(np.pi / 2.0) * np.sum(np.abs(conv)) / (6.0 * n))


@dataclass
class QualityGateResult:
    aceito: bool
    motivo: Optional[str]
    fracao_pixels_validos: float
    fracao_pixels_saturados: float
    snr_estimado: float


def evaluate_cube_quality(
        cubo: np.ndarray, *,
        fracao_saturacao_max: float = 0.01,
        snr_minimo: float = 3.0,
        fracao_validos_minima: float = 0.95,
        limite_saturacao_inferior: float = -0.5,
        limite_saturacao_superior: float = 1.5,
        ) -> QualityGateResult:
    """Avalia um cubo `(altura, largura, bandas)` contra 3 criterios --
    devolve SEMPRE um veredito (`aceito=True/False` + `motivo`), nunca
    processa silenciosamente uma cena degradada.

    - Pixels validos: fracao de valores finitos (nao-NaN/Inf) em todas as
      bandas. Cena com muitos pixels invalidos indica falha de aquisicao
      (linha morta do sensor, corrupcao de transferencia).
    - Saturacao: fracao de valores fora de
      [limite_saturacao_inferior, limite_saturacao_superior] -- em
      reflectancia relativa ja' calibrada, valores bem fora de [0,1]
      (com folga p/ ruido/correcao) indicam clipping do sensor ou erro de
      calibracao. Os limites default sao FOLGADOS (nao [0,1] estrito)
      porque reflectancia calibrada pode ter leve overshoot fisico
      legitimo (ver intervalo real observado no Kaki/VIS, docstring do
      modulo).
    - SNR: estimado (Immerkaer 1996, `estimate_noise_sigma`) sobre a
      imagem media ao longo das bandas -- `signal/noise` em razao linear
      (nao dB), usando a media dos pixels validos como "sinal".

    Rejeita no PRIMEIRO criterio que falhar (fail-fast, motivo unico e
    especifico -- nunca uma lista vaga de "varios problemas").
    """
    cubo = np.asarray(cubo, dtype=float)
    if cubo.ndim != 3:
        raise ValueError(f"evaluate_cube_quality espera um cubo 3D "
                          f"(altura, largura, bandas), recebeu shape "
                          f"{cubo.shape}.")

    validos = np.isfinite(cubo)
    fracao_validos = float(np.mean(validos))
    if fracao_validos < fracao_validos_minima:
        return QualityGateResult(
            aceito=False,
            motivo=(f"fracao de pixels validos ({fracao_validos:.3f}) "
                    f"abaixo do minimo ({fracao_validos_minima:.3f}) -- "
                    f"cena com NaN/Inf em excesso."),
            fracao_pixels_validos=fracao_validos,
            fracao_pixels_saturados=float("nan"),
            snr_estimado=float("nan"))

    cubo_valido = cubo[validos]
    saturados = ((cubo_valido < limite_saturacao_inferior) |
                (cubo_valido > limite_saturacao_superior))
    fracao_saturados = float(np.mean(saturados))
    if fracao_saturados > fracao_saturacao_max:
        return QualityGateResult(
            aceito=False,
            motivo=(f"fracao de pixels saturados/fora de faixa "
                    f"({fracao_saturados:.4f}) acima do maximo "
                    f"({fracao_saturacao_max:.4f}) -- limites "
                    f"[{limite_saturacao_inferior}, "
                    f"{limite_saturacao_superior}]."),
            fracao_pixels_validos=fracao_validos,
            fracao_pixels_saturados=fracao_saturados,
            snr_estimado=float("nan"))

    imagem_media = np.nanmean(cubo, axis=2)
    sinal = float(np.nanmean(np.abs(imagem_media)))
    ruido = estimate_noise_sigma(imagem_media)
    snr = sinal / ruido if ruido > 1e-12 else float("inf")
    if snr < snr_minimo:
        return QualityGateResult(
            aceito=False,
            motivo=(f"SNR estimado ({snr:.2f}) abaixo do minimo "
                    f"({snr_minimo:.2f})."),
            fracao_pixels_validos=fracao_validos,
            fracao_pixels_saturados=fracao_saturados,
            snr_estimado=snr)

    return QualityGateResult(
        aceito=True, motivo=None,
        fracao_pixels_validos=fracao_validos,
        fracao_pixels_saturados=fracao_saturados,
        snr_estimado=snr)
