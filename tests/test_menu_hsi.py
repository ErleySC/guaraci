"""Testes de _menu_hsi (Passo 102): reachability real do modo `hsi` via
CLI -- "nenhuma funcao implementada sem que o usuario consiga usar",
mesmo requisito ja aplicado ao resto desta auditoria."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import guaraci.guaraci as guaraci_mod


def test_menu_hsi_cancelar_com_0_nao_lanca_excecao(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: "0")
    guaraci_mod._menu_hsi(guaraci_mod.Config())


def test_menu_hsi_pasta_invalida_reporta_erro_e_nao_lanca(monkeypatch, tmp_path):
    pasta_sem_manifest = str(tmp_path)
    respostas = iter([pasta_sem_manifest, ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
    guaraci_mod._menu_hsi(guaraci_mod.Config())


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
def test_menu_hsi_roda_pipeline_completo_via_cli(monkeypatch, tmp_path):
    """A prova final de reachability: o usuario aciona o pipeline HSI
    inteiro digitando so' o caminho da pasta na tela do menu -- sem
    chamar hsi_pipeline diretamente."""
    pasta_dataset = str(_pasta_deephs_kaki())
    respostas = iter([pasta_dataset, ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))

    cfg = guaraci_mod.Config(output_root_folder=str(tmp_path))
    guaraci_mod._menu_hsi(cfg)

    assert cfg.mode == "hsi"
    assert cfg.hsi_dataset_folder == pasta_dataset
    assert cfg.output_folder != ""
    caminho_figura = (Path(cfg.output_folder) / "Graficos" / "hsi" /
                      "hsi_mapa_classificacao_amostra.png")
    assert caminho_figura.is_file()
