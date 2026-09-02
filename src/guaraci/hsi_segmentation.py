"""hsi_segmentation.py — Segmentacao espacial de cubo hiperespectral por
PCA (PC1) + limiar de Otsu (Passo 96 da `INSTRUCAO_HSI_MINIMO_VIAVEL.md`):
separa o objeto (fruta) do fundo da cena, produzindo a mascara usada pelo
Passo 97 para extrair so' os pixels do objeto fisico.

IMPORTANTE — distincao deliberada de `chemometric_stats.applicability_
domain`: aquele PCA e' de DOMINIO DE APLICABILIDADE (mede se uma AMOSTRA
inteira, ja' reduzida a um vetor de variaveis, esta' dentro da regiao
coberta pelo treino de um modelo PLS/PCA calibrado). O PCA deste modulo
e' de USO ESPACIAL: cada PIXEL de UMA UNICA imagem vira uma "amostra" (o
espectro daquele pixel), o PCA e' ajustado e aplicado NA MESMA cena (sem
nocao de treino/teste), e o resultado e' uma imagem 2D (PC1 por pixel),
nao uma metrica de distancia a um modelo pre-existente. Sao usos
matematicamente relacionados (ambos PCA) mas SEMANTICAMENTE diferentes
-- nunca reutilizar `applicability_domain` aqui sem adaptar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

__all__ = [
    "SegmentationResult",
    "otsu_threshold",
    "segment_object_pca_otsu",
]


def otsu_threshold(valores: np.ndarray, n_bins: int = 256) -> float:
    """Limiar de Otsu (Otsu, 1979, IEEE Trans. Systems, Man, and
    Cybernetics 9(1):62-66) -- maximiza a variancia ENTRE as duas classes
    (acima/abaixo do limiar) sobre um histograma de `valores`. Implementado
    aqui em vez de depender de scikit-image (que so' e' dependencia
    OPCIONAL do projeto -- extra `imagem`, ver `pyproject.toml`; a
    segmentacao HSI e' parte do "minimo viavel", nao deveria exigir uma
    dependencia extra so' para isso)."""
    valores = np.asarray(valores, dtype=float)
    valores = valores[np.isfinite(valores)]
    if valores.size == 0:
        raise ValueError("otsu_threshold: nenhum valor finito para limiarizar.")
    if np.ptp(valores) == 0:
        return float(valores[0])  # cena constante -- qualquer limiar serve

    hist, bordas = np.histogram(valores, bins=n_bins)
    centros = (bordas[:-1] + bordas[1:]) / 2.0
    hist = hist.astype(float)
    peso_total = hist.sum()

    peso_fundo = np.cumsum(hist)
    peso_objeto = peso_total - peso_fundo
    # Evita divisao por zero nos extremos (todo o histograma de um lado so').
    with np.errstate(invalid="ignore", divide="ignore"):
        media_fundo = np.cumsum(hist * centros) / peso_fundo
        soma_total = np.sum(hist * centros)
        media_objeto = (soma_total - np.cumsum(hist * centros)) / peso_objeto

    with np.errstate(invalid="ignore"):
        variancia_entre_classes = (peso_fundo * peso_objeto *
                                   (media_fundo - media_objeto) ** 2)
    variancia_entre_classes = np.nan_to_num(variancia_entre_classes, nan=-1.0)
    idx_melhor = int(np.argmax(variancia_entre_classes))
    return float(centros[idx_melhor])


@dataclass
class SegmentationResult:
    mascara: np.ndarray          # bool, (altura, largura) -- True = objeto
    imagem_pc1: np.ndarray       # float, (altura, largura), score de PC1
    limiar: float
    variancia_explicada_pc1: float
    fracao_objeto: float


def segment_object_pca_otsu(cubo: np.ndarray, *,
                             objeto_e_pico_maior: Optional[bool] = None,
                             ) -> SegmentationResult:
    """Segmenta `cubo` (altura, largura, bandas) em objeto/fundo via PC1 +
    Otsu. `objeto_e_pico_maior`: quando `None` (default), assume que o
    objeto (fruta) ocupa uma MINORIA da cena -- convencao comum em bancada
    de laboratorio (fundo/mesa dominam a area) -- e escolhe entre `PC1 >
    limiar` e `PC1 <= limiar` o lado com MENOS pixels como "objeto". Passe
    `True`/`False` explicitamente se a convencao da sua cena for diferente
    (objeto preenchendo a maior parte do quadro)."""
    cubo = np.asarray(cubo, dtype=float)
    if cubo.ndim != 3:
        raise ValueError(f"segment_object_pca_otsu espera um cubo 3D "
                          f"(altura, largura, bandas), recebeu shape "
                          f"{cubo.shape}.")
    altura, largura, n_bandas = cubo.shape
    pixels = cubo.reshape(-1, n_bandas)

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(pixels).ravel()
    imagem_pc1 = pc1.reshape(altura, largura)

    limiar = otsu_threshold(pc1)
    acima = imagem_pc1 > limiar

    if objeto_e_pico_maior is None:
        mascara = acima if acima.sum() <= (~acima).sum() else ~acima
    elif objeto_e_pico_maior:
        mascara = acima if acima.sum() >= (~acima).sum() else ~acima
    else:
        mascara = acima if acima.sum() <= (~acima).sum() else ~acima

    return SegmentationResult(
        mascara=mascara, imagem_pc1=imagem_pc1, limiar=limiar,
        variancia_explicada_pc1=float(pca.explained_variance_ratio_[0]),
        fracao_objeto=float(np.mean(mascara)))
