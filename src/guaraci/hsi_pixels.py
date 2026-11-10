"""hsi_pixels.py — Extracao de espectros de pixel da ROI (mascara do Passo
96) e montagem do dataset por-pixel com agrupamento por OBJETO FISICO
(Passo 97 da `INSTRUCAO_HSI_MINIMO_VIAVEL.md`) -- o passo mais critico do
bloco HSI: nenhum split de treino/teste pode colocar pixels do MESMO
objeto fisico em lados diferentes (vazamento silencioso).

Reaproveita o MESMO conceito de `group_id`/`session_from_mae_id` ja'
validado no resto do projeto (`dados_io.py`) -- aqui um "grupo" e' um
objeto fisico, e TODOS os pixels de TODAS as gravacoes
daquele objeto (ex.: frente + costas do mesmo objeto, ver
o carregador do dataset) compartilham o mesmo `group_id`. O
splitter usado para provar a ausencia de vazamento e'
`validacao_estatistica.StableStratifiedGroupKFold` -- o MESMO ja'
padronizado no resto do projeto (partição estável entre versões de
scikit-learn), nunca um mecanismo group-aware paralelo.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from guaraci.hsi_blocos import CuboHSI, extrair_pixels

__all__ = [
    "extract_roi_spectra",
    "build_pixel_dataset",
]


def _coordenadas_roi(cubo, mascara) -> "tuple[np.ndarray, np.ndarray]":
    """(linhas, colunas) dos pixels da ROI, na ordem C (a mesma de `cubo[mascara]`)."""
    if not isinstance(cubo, CuboHSI):
        cubo = np.asarray(cubo)
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
    linhas, colunas = np.nonzero(mascara)
    return linhas, colunas


def extract_roi_spectra(cubo: np.ndarray, mascara: np.ndarray, dtype=float,
                         ) -> np.ndarray:
    """Extrai os espectros dos pixels onde `mascara` e' True -- devolve
    `(n_pixels_roi, n_bandas)` (float64 por padrao; `dtype=np.float32` para ROIs enormes). `mascara` deve ter a MESMA forma
    espacial de `cubo` (altura, largura); `cubo` pode ser um ndarray ou um
    `hsi_blocos.CuboHSI` (leitura por memmap: so' os pixels da ROI sao lidos)."""
    linhas, colunas = _coordenadas_roi(cubo, mascara)
    return np.asarray(extrair_pixels(cubo, linhas, colunas), dtype=dtype)


def build_pixel_dataset(
        cubos: Sequence[np.ndarray], mascaras: Sequence[np.ndarray],
        group_ids: Sequence[str], rotulos: Sequence[str], *,
        max_pixels_por_gravacao: Optional[int] = None, seed: int = 42,
        transformar: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monta o dataset POR-PIXEL a partir de N gravacoes (cada uma com seu
    cubo, mascara de ROI, group_id de objeto fisico e rotulo de
    classe) -- devolve `(X, y, groups)`:

      - X:      `(n_pixels_total, n_bandas)`, um pixel de ROI por linha.
      - y:      rotulo da gravacao de origem, REPLICADO para cada pixel
                daquele cubo (todos os pixels de UM cubo compartilham o
                mesmo rotulo -- e' o mesmo objeto, mesmo estado de
                maturacao).
      - groups: `group_id` da gravacao de origem, REPLICADO por pixel --
                usado pelo Passo 98 (split group-aware) e verificado pela
                contra-prova de propriedade abaixo (nunca 2 lados de um
                split com o MESMO group_id).

    As 4 sequencias de entrada devem ter o MESMO comprimento (1 entrada
    por gravacao) -- comprimentos divergentes levantam erro explicito em
    vez de zip() truncar em silencio.

    `max_pixels_por_gravacao` (Passo 104, achado real): resolucao de
    imagem varia MUITO entre cameras (uma: 64x64=4096
    pixels/imagem; outra camera: ~286x294=~97000 pixels/imagem,
    ~24x mais) -- sem limite, o dataset por-pixel de um objeto de alta
    resolucao estoura memoria (medido: ~2,8GB so' para 1 fit de PLS-DA
    dentro do loop de selecao de LVs, com Avocado/VIS). Quando setado,
    cada gravacao e' subamostrada (sem reposicao, RNG semeado por
    `seed` -- reprodutivel) para no maximo esse numero de pixels ANTES
    de concatenar -- os pixels retidos sao reais (nunca inventados), so'
    uma fracao aleatoria da ROI real daquela gravacao. `None` (default)
    preserva o comportamento antigo -- nenhuma gravacao e' truncada.

    A subamostra e' sorteada sobre as COORDENADAS da ROI antes de ler os
    espectros (mesmos indices de antes, porque o RNG recebe o mesmo `n`); com
    cubos `CuboHSI` so' esses pixels saem do disco.

    `transformar(pixels) -> pixels`: etapa por pixel aplicada a cada cubo ja'
    subamostrado (faixa de bandas, SNV por pixel): comuta com a escolha dos
    pixels, entao nao precisa processar o cubo inteiro."""
    n = len(cubos)
    if not (len(mascaras) == len(group_ids) == len(rotulos) == n):
        raise ValueError(
            f"build_pixel_dataset: comprimentos divergentes -- "
            f"cubos={n}, mascaras={len(mascaras)}, "
            f"group_ids={len(group_ids)}, rotulos={len(rotulos)}. "
            f"Uma entrada por gravacao, os 4 devem bater.")

    rng = np.random.default_rng(seed) if max_pixels_por_gravacao else None

    partes_X: List[np.ndarray] = []
    partes_y: List[np.ndarray] = []
    partes_g: List[np.ndarray] = []
    for cubo, mascara, gid, rotulo in zip(cubos, mascaras, group_ids, rotulos):
        linhas, colunas = _coordenadas_roi(cubo, mascara)
        if (rng is not None and max_pixels_por_gravacao is not None
                and len(linhas) > max_pixels_por_gravacao):
            idx = rng.choice(len(linhas), size=max_pixels_por_gravacao,
                             replace=False)
            linhas, colunas = linhas[idx], colunas[idx]
        pixels = np.asarray(extrair_pixels(cubo, linhas, colunas), dtype=float)
        if transformar is not None:
            pixels = np.asarray(transformar(pixels), dtype=float)
        partes_X.append(pixels)
        partes_y.append(np.full(len(pixels), rotulo, dtype=object))
        partes_g.append(np.full(len(pixels), gid, dtype=object))

    X = np.concatenate(partes_X, axis=0)
    y = np.concatenate(partes_y, axis=0).astype(str)
    groups = np.concatenate(partes_g, axis=0).astype(str)
    return X, y, groups
