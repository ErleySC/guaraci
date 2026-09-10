"""hsi_figures.py — Mapa de classificacao espacial de cubo HSI (Passo 99
da `INSTRUCAO_HSI_MINIMO_VIAVEL.md`): imagem do objeto com cada pixel
colorido pela classe prevista. Reaproveita a infraestrutura de figuras
JA existente (`figuras.save`, mesma pasta/formato/carimbo de
prototipo) e a paleta extraida da mascote (`paleta_cores.color`), por
consistencia visual com o resto do projeto -- nao introduz uma paleta
nova so' para HSI."""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from guaraci.figuras import save
from guaraci.paleta_cores import color

if TYPE_CHECKING:
    from guaraci.config import Config

__all__ = ["fig_hsi_classification_map"]

_COR_FORA_DA_ROI = "0.85"   # cinza claro -- mesmo tom usado em figuras.py p/ n/a


def fig_hsi_classification_map(
        mascara: np.ndarray, predicoes_pixel: np.ndarray,
        classes_ordenadas: Sequence[str], cfg: "Config", pasta: str,
        nome: str = "hsi_mapa_classificacao") -> None:
    """Gera e salva (via `figuras.save`, mesma pasta Graficos/ + carimbo
    de prototipo quando aplicavel) um mapa espacial: cada pixel da ROI
    (`mascara`) colorido pela sua classe predita
    (`predicoes_pixel[i]` corresponde ao i-esimo pixel True de
    `mascara`, na mesma ordem que `cubo[mascara]` devolve -- ver
    `hsi_pixels.extract_roi_spectra`); pixels fora da ROI ficam em cinza
    claro (fundo, nao classificado)."""
    mascara = np.asarray(mascara, dtype=bool)
    predicoes_pixel = np.asarray(predicoes_pixel)
    n_roi = int(mascara.sum())
    if len(predicoes_pixel) != n_roi:
        raise ValueError(
            f"fig_hsi_classification_map: {len(predicoes_pixel)} predicoes "
            f"para {n_roi} pixels de ROI na mascara -- devem bater.")

    indice_por_classe = {c: i for i, c in enumerate(classes_ordenadas)}
    desconhecidas = set(predicoes_pixel.tolist()) - set(classes_ordenadas)
    if desconhecidas:
        raise ValueError(
            f"fig_hsi_classification_map: classe(s) preditas {desconhecidas} "
            f"nao estao em classes_ordenadas {list(classes_ordenadas)}.")

    mapa_indices = np.full(mascara.shape, -1, dtype=int)
    mapa_indices[mascara] = [indice_por_classe[r] for r in predicoes_pixel]

    cores = [_COR_FORA_DA_ROI] + [color(i) for i in range(len(classes_ordenadas))]
    cmap = ListedColormap(cores)

    fig, ax = plt.subplots(figsize=(5.0, 5.0), constrained_layout=True)
    ax.imshow(mapa_indices + 1, cmap=cmap, vmin=0, vmax=len(cores) - 1,
              interpolation="nearest")
    ax.axis("off")
    ax.set_title("Mapa de classificacao HSI por pixel")
    legenda = [Patch(facecolor=color(i), label=c)
              for i, c in enumerate(classes_ordenadas)]
    ax.legend(handles=legenda, loc="upper center",
             bbox_to_anchor=(0.5, -0.02), ncol=len(classes_ordenadas),
             frameon=False, fontsize=8)

    save(fig, nome, pasta, cfg, subpasta="hsi")
