"""Testes de hsi_pipeline.py (Passo 102) -- orquestracao ponta-a-ponta
do modo `hsi`, distinto do modo `imagem` (ver docstring do modulo)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from guaraci.config import Config
from guaraci.hsi_pipeline import run_hsi_pipeline


def test_run_hsi_pipeline_sem_pasta_configurada_levanta_erro_claro():
    cfg = Config(mode="hsi", hsi_dataset_folder="")
    with pytest.raises(ValueError, match="hsi_dataset_folder"):
        run_hsi_pipeline(cfg)


# ── caminho generico (Passo 111): cubo do proprio usuario, sem manifest.json ──

def _gravar_cubo_sintetico(caminho_hdr_sem_ext, cubo, n_bandas=6):
    """Grava um .hdr+.bin ENVI sintetico com um sinal de classe embutido
    (cada classe tem uma media espectral bem diferente -- resolucao
    espacial pequena o suficiente pra' rodar rapido, grande o suficiente
    pra' passar no quality gate/segmentacao real)."""
    n_lin, n_col, _ = cubo.shape
    caminho_hdr_sem_ext.parent.mkdir(parents=True, exist_ok=True)
    (caminho_hdr_sem_ext.with_suffix(".bin")).write_bytes(
        cubo.astype("<f4").tobytes())
    (caminho_hdr_sem_ext.with_suffix(".hdr")).write_text(
        f"ENVI\nsamples = {n_col}\nlines = {n_lin}\nbands = {n_bandas}\n"
        f"header offset = 0\nfile type = ENVI Standard\ndata type = 4\n"
        f"interleave = bip\nbyte order = 0\n", encoding="utf-8")


def _mascara_objeto_suave(n_lin, n_col, raio_frac=0.35, largura_borda=4.0):
    """Mascara circular de objeto com transicao SUAVE (nao um degrau
    binario) -- um degrau abrupto seria lido pelo estimador de ruido de
    Immerkaer (Laplaciano, ver hsi_quality.py) como ruido de alta
    frequencia, reprovando cenas sinteticas perfeitamente limpas no
    quality gate por um artefato do gerador, nao um problema real de
    qualidade. `largura_borda` (pixels) controla a suavidade."""
    yy, xx = np.ogrid[:n_lin, :n_col]
    cy, cx = (n_lin - 1) / 2.0, (n_col - 1) / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    raio = raio_frac * min(n_lin, n_col)
    return np.clip((raio - dist) / largura_borda + 0.5, 0.0, 1.0)


def _montar_pasta_hsi_generica(tmp_path, n_amostras_por_classe=3, seed=0):
    """2 classes, cada uma com N amostras fisicas (nivel 'high' -- 1
    subpasta por amostra), cada amostra com 1 gravacao -- sinal espectral
    bem separado entre classes (media alta vs. media baixa) pra' o
    pipeline ter algo real pra' classificar. Resolucao 64x64, igual ao
    dataset publico real (Kaki/VIS) -- conhecida por passar no quality
    gate/segmentacao real, ao contrario de resolucoes minusculas
    inventadas que sofrem de artefatos de borda (ver
    `_mascara_objeto_suave`)."""
    rng = np.random.default_rng(seed)
    raiz = tmp_path / "meu_dataset_hsi"
    n_lin, n_col, n_bandas = 64, 64, 6
    alpha = _mascara_objeto_suave(n_lin, n_col)[..., None]
    # Valores dentro da faixa de reflectancia relativa calibrada aceita
    # pelo quality gate ([-0.5, 1.5], ver hsi_quality.py) -- fundo escuro
    # uniforme vs. objeto claro, com o NIVEL do objeto (nao a separacao
    # objeto/fundo) sendo o que distingue as 2 classes.
    for classe, nivel_objeto in (("madura", 0.85), ("verde", 0.45)):
        for a in range(n_amostras_por_classe):
            pasta_amostra = raiz / classe / f"amostra{a}"
            fundo = rng.normal(loc=0.05, scale=0.01,
                               size=(n_lin, n_col, n_bandas))
            objeto = rng.normal(loc=nivel_objeto, scale=0.01,
                                size=(n_lin, n_col, n_bandas))
            cubo = alpha * objeto + (1.0 - alpha) * fundo
            _gravar_cubo_sintetico(pasta_amostra / "vista0", cubo, n_bandas)
    return raiz


def test_run_hsi_pipeline_generico_sem_manifest_roda_ponta_a_ponta(tmp_path):
    """Passo 111: pasta SEM manifest.json (dado proprio do usuario) roda
    o pipeline inteiro -- leitura generica -> quality gate -> segmentacao
    -> classificacao -> mapa -> validacao INTERNA (sem particao por dia,
    que so' o dataset publico tem)."""
    raiz = _montar_pasta_hsi_generica(tmp_path)
    cfg = Config(mode="hsi", hsi_dataset_folder=str(raiz),
                output_root_folder=str(tmp_path / "saida"),
                output_format="png")

    resumo = run_hsi_pipeline(cfg)

    assert resumo["n_gravacoes_total"] == 6  # 2 classes x 3 amostras
    assert resumo["n_gravacoes_aceitas"] == 6  # cenas limpas, todas passam
    assert resumo["grouping_guarantee"] == "high"
    assert resumo["achados_quimica"] is None  # nao aplicavel sem manifest
    val = resumo["validacao_externa"]
    assert val.n_objetos_teste_externo == 0  # sem particao por dia
    assert val.sensibilidade_externa == {}

    caminho_figura = (Path(cfg.output_folder) / "Graficos" / "hsi" /
                      "hsi_mapa_classificacao_amostra.png")
    assert caminho_figura.is_file()


def test_run_hsi_pipeline_generico_nivel_none_avisa_e_processa_mesmo_assim(
        tmp_path, capsys):
    """Pasta flat (1 subpasta por classe, sem subpasta-de-amostra nem
    CSV): nivel 'none' -- processa mesmo assim, mas avisa explicitamente
    (nunca silenciosamente) e cada gravacao vira seu proprio grupo
    (nunca um placeholder compartilhado que colapsaria objetos
    diferentes)."""
    rng = np.random.default_rng(1)
    raiz = tmp_path / "flat"
    n_lin, n_col, n_bandas = 64, 64, 6
    alpha = _mascara_objeto_suave(n_lin, n_col)[..., None]
    for classe, nivel_objeto in (("A", 0.85), ("B", 0.45)):
        pasta_classe = raiz / classe
        for i in range(3):
            fundo = rng.normal(loc=0.05, scale=0.01,
                               size=(n_lin, n_col, n_bandas))
            objeto = rng.normal(loc=nivel_objeto, scale=0.01,
                                size=(n_lin, n_col, n_bandas))
            cubo = alpha * objeto + (1.0 - alpha) * fundo
            _gravar_cubo_sintetico(pasta_classe / f"g{i}", cubo, n_bandas)

    cfg = Config(mode="hsi", hsi_dataset_folder=str(raiz),
                output_root_folder=str(tmp_path / "saida"),
                output_format="png")
    resumo = run_hsi_pipeline(cfg)

    assert resumo["grouping_guarantee"] == "none"
    saida = capsys.readouterr().out
    assert "SEM garantia de agrupamento" in saida


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
