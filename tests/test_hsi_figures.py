"""Testes de hsi_figures.py (Passo 99) -- mapa de classificacao espacial
por pixel, reaproveitando figuras.save() e a paleta da mascote."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from guaraci.config import Config, NOME_GRAFICOS
from guaraci.hsi_figures import fig_hsi_classification_map


def test_fig_hsi_classification_map_salva_arquivo(tmp_path):
    mascara = np.zeros((6, 6), dtype=bool)
    mascara[2:4, 2:4] = True   # 4 pixels de ROI
    predicoes = np.array(["perfect", "perfect", "overripe", "unripe"])
    cfg = Config(output_format="png")

    fig_hsi_classification_map(
        mascara, predicoes, ["unripe", "perfect", "overripe"], cfg,
        str(tmp_path), nome="mapa_teste")

    caminho = tmp_path / NOME_GRAFICOS / "hsi" / "mapa_teste.png"
    assert caminho.is_file()


def test_fig_hsi_classification_map_contagem_divergente_levanta_erro(tmp_path):
    mascara = np.zeros((4, 4), dtype=bool)
    mascara[0, 0] = True
    mascara[1, 1] = True
    cfg = Config(output_format="png")
    with pytest.raises(ValueError, match="devem bater"):
        fig_hsi_classification_map(
            mascara, np.array(["perfect"]),  # so' 1 predicao p/ 2 pixels
            ["perfect"], cfg, str(tmp_path))


def test_fig_hsi_classification_map_classe_desconhecida_levanta_erro(tmp_path):
    mascara = np.zeros((3, 3), dtype=bool)
    mascara[0, 0] = True
    cfg = Config(output_format="png")
    with pytest.raises(ValueError, match="nao estao em classes_ordenadas"):
        fig_hsi_classification_map(
            mascara, np.array(["classe_fantasma"]), ["perfect", "overripe"],
            cfg, str(tmp_path))
