"""Ramos de borda de guaraci.hsi_pipeline: gravacao rejeitada no quality gate, todas rejeitadas e relatorio em texto."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from guaraci import hsi_pipeline as hp  # noqa: E402
from guaraci.config import Config  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_hsi_pipeline import _gravar_cubo_sintetico, _mascara_objeto_suave  # noqa: E402


def _cubo_limpo(nivel=0.8, seed=0):
    rng = np.random.default_rng(seed)
    alpha = _mascara_objeto_suave(64, 64)[..., None]
    return alpha * rng.normal(nivel, 0.01, (64, 64, 6)) + (1 - alpha) * rng.normal(0.05, 0.01, (64, 64, 6))


def test_gate_rejeita_a_gravacao_saturada_e_deixa_passar_a_limpa():
    limpo, saturado = _cubo_limpo(), np.full((64, 64, 6), 5.0)                     # 5.0 esta' fora de [-0.5, 1.5]
    r = hp.apply_quality_gate_and_segment([limpo, saturado], ["g1", "g2"], ["madura", "madura"], ["d1", "d1"])
    assert r["n_rejeitados"] == 1 and len(r["cubos"]) == 1 and r["motivos_rejeicao"][0].startswith("g2:")


def test_pipeline_generico_com_todas_as_gravacoes_rejeitadas_levanta_erro_com_os_motivos(tmp_path):
    raiz = tmp_path / "ruins"
    for classe in ("a", "b"):
        for i in range(2):
            _gravar_cubo_sintetico(raiz / classe / f"am{i}" / "v0", np.full((64, 64, 6), 5.0), 6)
    cfg = Config(mode="hsi", hsi_dataset_folder=str(raiz), output_root_folder=str(tmp_path / "saida"), output_format="png")
    with pytest.raises(ValueError, match="TODAS as 4 gravacoes falharam"):
        hp.run_hsi_pipeline(cfg)


def _resumo(grouping):
    val = SimpleNamespace(n_objetos_teste_interno=3, n_objetos_teste_externo=0, classes=["madura", "verde"],
                          sensibilidade_interna={"madura": 1.0, "verde": 0.5}, especificidade_interna={"madura": 0.5, "verde": 1.0})
    return {"n_gravacoes_aceitas": 5, "n_gravacoes_total": 6, "n_gravacoes_rejeitadas": 1, "grouping_guarantee": grouping,
            "validacao_externa": val, "n_components": 2, "motivos_rejeicao": ["g2: saturada"]}


@pytest.mark.parametrize("grouping, avisa", [("none", True), ("high", False)])
def test_relatorio_em_texto_lista_as_classes_e_avisa_quando_nao_ha_garantia_de_agrupamento(tmp_path, grouping, avisa):
    cfg = Config(mode="hsi", output_folder=str(tmp_path))
    caminho = Path(hp.gravar_relatorio_hsi(cfg, _resumo(grouping)))
    texto = caminho.read_text(encoding="utf-8")
    assert "madura: sensibilidade interna 1.00" in texto and "verde: sensibilidade interna 0.50" in texto
    assert ("Sem garantia de agrupamento" in texto) is avisa
    assert (tmp_path / "config_efetiva.yaml").is_file()


def test_pasta_de_saida_ja_definida_nao_e_trocada():
    from guaraci.config import Config
    from guaraci.hsi_pipeline import _preparar_pasta_saida
    cfg = Config(output_folder="minha_pasta")
    _preparar_pasta_saida(cfg)
    assert cfg.output_folder == "minha_pasta"
