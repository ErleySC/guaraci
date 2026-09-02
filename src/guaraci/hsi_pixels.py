"""hsi_pixels.py — Extracao de espectros de pixel da ROI (mascara do Passo
96) e montagem do dataset por-pixel com agrupamento por OBJETO FISICO
(Passo 97 da `INSTRUCAO_HSI_MINIMO_VIAVEL.md`) -- o passo mais critico do
bloco HSI: nenhum split de treino/teste pode colocar pixels do MESMO
objeto fisico em lados diferentes (vazamento silencioso).

Reaproveita o MESMO conceito de `group_id`/`session_from_mae_id` ja'
validado no resto do projeto (`dados_io.py`) -- aqui um "grupo" e' um
objeto fisico (uma fruta), e TODOS os pixels de TODAS as gravacoes
daquele objeto (ex.: frente + costas da mesma fruta, ver
`hsi_io.load_deephs_kaki_dataset`) compartilham o mesmo `group_id`. O
splitter usado para provar a ausencia de vazamento e'
`validacao_estatistica.StableStratifiedGroupKFold` -- o MESMO ja'
padronizado no resto do projeto (partição estável entre versões de
scikit-learn), nunca um mecanismo group-aware paralelo.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

__all__ = [
    "extract_roi_spectra",
    "build_pixel_dataset",
]


def extract_roi_spectra(cubo: np.ndarray, mascara: np.ndarray,
                         ) -> np.ndarray:
    """Extrai os espectros dos pixels onde `mascara` e' True -- devolve
    `(n_pixels_roi, n_bandas)`. `mascara` deve ter a MESMA forma espacial
    de `cubo` (altura, largura)."""
    cubo = np.asarray(cubo, dtype=float)
    mascara = np.asarray(mascara, dtype=bool)
    if cubo.ndim != 3:
        raise ValueError(f"extract_roi_spectra espera um cubo 3D, "
                          f"recebeu shape {cubo.shape}.")
    if mascara.shape != cubo.shape[:2]:
        raise ValueError(
            f"mascara tem forma {mascara.shape}, esperado "
            f"{cubo.shape[:2]} (mesma forma espacial do cubo).")
    if not mascara.any():
        raise ValueError("extract_roi_spectra: mascara vazia (nenhum "
                          "pixel de ROI) -- nada a extrair.")
    return cubo[mascara]


def build_pixel_dataset(
        cubos: Sequence[np.ndarray], mascaras: Sequence[np.ndarray],
        group_ids: Sequence[str], rotulos: Sequence[str],
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monta o dataset POR-PIXEL a partir de N gravacoes (cada uma com seu
    cubo, mascara de ROI, group_id de objeto fisico e rotulo de
    classe) -- devolve `(X, y, groups)`:

      - X:      `(n_pixels_total, n_bandas)`, um pixel de ROI por linha.
      - y:      rotulo da gravacao de origem, REPLICADO para cada pixel
                daquele cubo (todos os pixels de UM cubo compartilham o
                mesmo rotulo -- e' a mesma fruta, mesmo estado de
                maturacao).
      - groups: `group_id` da gravacao de origem, REPLICADO por pixel --
                usado pelo Passo 98 (split group-aware) e verificado pela
                contra-prova de propriedade abaixo (nunca 2 lados de um
                split com o MESMO group_id).

    As 4 sequencias de entrada devem ter o MESMO comprimento (1 entrada
    por gravacao) -- comprimentos divergentes levantam erro explicito em
    vez de zip() truncar em silencio."""
    n = len(cubos)
    if not (len(mascaras) == len(group_ids) == len(rotulos) == n):
        raise ValueError(
            f"build_pixel_dataset: comprimentos divergentes -- "
            f"cubos={n}, mascaras={len(mascaras)}, "
            f"group_ids={len(group_ids)}, rotulos={len(rotulos)}. "
            f"Uma entrada por gravacao, os 4 devem bater.")

    partes_X: List[np.ndarray] = []
    partes_y: List[np.ndarray] = []
    partes_g: List[np.ndarray] = []
    for cubo, mascara, gid, rotulo in zip(cubos, mascaras, group_ids, rotulos):
        pixels = extract_roi_spectra(cubo, mascara)
        partes_X.append(pixels)
        partes_y.append(np.full(len(pixels), rotulo, dtype=object))
        partes_g.append(np.full(len(pixels), gid, dtype=object))

    X = np.concatenate(partes_X, axis=0)
    y = np.concatenate(partes_y, axis=0).astype(str)
    groups = np.concatenate(partes_g, axis=0).astype(str)
    return X, y, groups
