"""Testes de hsi_pipeline.py (Passo 102) -- orquestracao ponta-a-ponta
do modo `hsi`, distinto do modo `imagem` (ver docstring do modulo)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from guaraci.config import Config
from guaraci.hsi_pipeline import run_hsi_pipeline


def test_run_hsi_pipeline_sem_pasta_configurada_levanta_erro_claro():
    cfg = Config(mode="hsi", hsi_dataset_folder="")
    with pytest.raises(ValueError, match="hsi_dataset_folder"):
        run_hsi_pipeline(cfg)


def _pasta_deephs_kaki():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "deephs_kaki_vis"
    return pasta if (pasta / "manifest.json").is_file() else None


requer_deephs_kaki = pytest.mark.skipif(
    _pasta_deephs_kaki() is None,
    reason=("dataset publico DeepHS Fruit/Kaki/VIS ausente. Baixe com "
            "'python scripts/download_datasets/baixar_deephs_kaki.py' e "
            "aponte GUARACI_DATASETS_DIR."))


@requer_deephs_kaki
def test_run_hsi_pipeline_fim_a_fim_contra_kaki_real(tmp_path):
    """A prova final do Passo 102: TODO o pipeline HSI (leitura -> quality
    gate -> segmentacao -> pixel -> classificacao -> figura -> quimica ->
    validacao externa) acessivel por UMA chamada, contra dado real."""
    cfg = Config(mode="hsi", hsi_dataset_folder=str(_pasta_deephs_kaki()),
                output_root_folder=str(tmp_path), output_format="png")

    resumo = run_hsi_pipeline(cfg)

    assert resumo["n_gravacoes_total"] == 56
    assert resumo["n_gravacoes_aceitas"] > 0
    assert resumo["n_gravacoes_aceitas"] + resumo["n_gravacoes_rejeitadas"] == 56
    assert resumo["n_components"] >= 1
    assert resumo["validacao_externa"].n_objetos_teste_externo > 0
    assert len(resumo["achados_quimica"]) == 5

    caminho_figura = (Path(cfg.output_folder) / "Graficos" / "hsi" /
                      "hsi_mapa_classificacao_amostra.png")
    assert caminho_figura.is_file()
    print(f"\n[HSI pipeline] {resumo['n_gravacoes_aceitas']}/"
          f"{resumo['n_gravacoes_total']} aceitas no quality gate, "
          f"n_components={resumo['n_components']}")
