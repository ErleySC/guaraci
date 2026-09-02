"""hsi_pipeline.py — Orquestracao ponta-a-ponta do modo de entrada `hsi`
(Passo 102 da `INSTRUCAO_HSI_MINIMO_VIAVEL.md`): leitura -> quality gate
-> segmentacao -> extracao de ROI/agrupamento -> classificacao por
pixel -> mapa espacial -> explicabilidade cruzada -> validacao externa.

Modo `hsi` e' DISTINTO do modo `imagem` (colorimetria digital de foto
comum, `dados_imagem.py`) -- nunca confundir os dois: `imagem` extrai
UMA linha de features por FOTO inteira e reusa `pipeline.executar()`
sem alteracao (mesma matriz amostras x variaveis de sempre); `hsi`
opera POR PIXEL de um cubo hiperespectral, com agregacao por objeto
fisico -- forma de dado fundamentalmente diferente, por isso tem sua
propria orquestracao aqui em vez de entrar em `pipeline.executar()`.

Esta e' a fatia "minimo viavel" (Passos 94-101): 1 dataset publico
(DeepHS Fruit/Kaki/VIS), 1 formato (ENVI), 1 leitor especifico
(`hsi_io.load_deephs_kaki_dataset`). Um dataset/formato diferente
precisaria de seu proprio leitor em `hsi_io.py` -- o resto do pipeline
(quality gate, segmentacao, classificacao, figuras, quimica, validacao)
ja' e' generico o suficiente para reaproveitar sem alteracao.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List

import numpy as np

from guaraci.hsi_chemistry import (ATRIBUICAO_QUIMICA_VIS_FRUTA,
                                    cross_reference_vip_with_chemistry)
from guaraci.hsi_classification import fit_predict_pixel_plsda
from guaraci.hsi_figures import fig_hsi_classification_map
from guaraci.hsi_io import load_deephs_kaki_dataset
from guaraci.hsi_pixels import build_pixel_dataset
from guaraci.hsi_quality import evaluate_cube_quality
from guaraci.hsi_segmentation import segment_object_pca_otsu
from guaraci.hsi_validation import run_external_validation_by_day

if TYPE_CHECKING:
    from guaraci.config import Config

__all__ = ["run_hsi_pipeline"]


def run_hsi_pipeline(cfg: "Config", *, fracao_teste_interno: float = 0.2,
                     n_dias_externos: int = 2, seed: int = 42,
                     ) -> Dict[str, object]:
    """Roda o pipeline HSI completo sobre `cfg.hsi_dataset_folder`
    (pasta com `manifest.json` + arquivos ENVI, ver
    `scripts/download_datasets/baixar_deephs_kaki.py`). Devolve um dict
    de resumo (nunca lanca excecao por dado individual ruim -- cenas que
    falham no quality gate sao PULADAS e contadas, nunca processadas em
    silencio) e salva o mapa de classificacao espacial em
    `cfg.output_folder`/Graficos/hsi/.

    `n_dias_externos`: quantos dos dias mais RECENTES (ordenados por
    nome, que no dataset usado correspondem a ordem cronologica --
    ver `hsi_io.load_deephs_kaki_dataset`) viram o teste de validacao
    externa (Passo 101)."""
    pasta = getattr(cfg, "hsi_dataset_folder", "") or ""
    if not pasta:
        raise ValueError(
            "run_hsi_pipeline: cfg.hsi_dataset_folder nao configurado -- "
            "aponte para a pasta com manifest.json (ver "
            "scripts/download_datasets/baixar_deephs_kaki.py).")

    if not getattr(cfg, "output_folder", ""):
        # Nome dedicado -- a convencao PLSDA_OE_.../pipeline.generate_
        # output_name() e' especifica do fluxo tabular (preset de
        # pre-processamento, objetivo N1/N2/N3), nao se aplica ao HSI.
        raiz = getattr(cfg, "output_root_folder", "") or "resultados_tcc"
        cfg.output_folder = os.path.join(
            raiz, "hsi", f"HSI_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    cubos, rotulos, grupos, wavelengths, meta_df = load_deephs_kaki_dataset(pasta)

    cubos_ok: List[np.ndarray] = []
    mascaras_ok: List[np.ndarray] = []
    grupos_ok: List[str] = []
    rotulos_ok: List[str] = []
    dias_ok: List[str] = []
    n_rejeitados = 0
    motivos_rejeicao: List[str] = []

    for cubo, gid, rotulo, dia in zip(cubos, grupos, rotulos, meta_df["day"]):
        qualidade = evaluate_cube_quality(cubo)
        if not qualidade.aceito:
            n_rejeitados += 1
            motivos_rejeicao.append(f"{gid}: {qualidade.motivo}")
            continue
        mascara = segment_object_pca_otsu(cubo).mascara
        cubos_ok.append(cubo)
        mascaras_ok.append(mascara)
        grupos_ok.append(gid)
        rotulos_ok.append(rotulo)
        dias_ok.append(dia)

    if not cubos_ok:
        raise ValueError(
            f"run_hsi_pipeline: TODAS as {len(cubos)} gravacoes falharam "
            f"no quality gate -- nada a processar. Motivos: "
            f"{motivos_rejeicao[:5]}")

    dias_unicos = sorted(set(dias_ok))
    dias_externos = dias_unicos[-min(n_dias_externos, len(dias_unicos) - 1):]

    relatorio_validacao = run_external_validation_by_day(
        cubos_ok, mascaras_ok, grupos_ok, rotulos_ok, dias_ok,
        dias_externos=dias_externos,
        fracao_teste_interno=fracao_teste_interno, seed=seed)

    X, y, pixel_groups = build_pixel_dataset(
        cubos_ok, mascaras_ok, grupos_ok, rotulos_ok)
    resultado_modelo = fit_predict_pixel_plsda(
        X, y, pixel_groups, X, pixel_groups, seed=seed)

    classes_ordenadas = sorted(set(y.tolist()))
    idx_amostra = 0
    n_roi_amostra = int(mascaras_ok[idx_amostra].sum())
    preds_amostra = resultado_modelo["predicoes_pixel"][:n_roi_amostra]
    fig_hsi_classification_map(
        mascaras_ok[idx_amostra], preds_amostra, classes_ordenadas, cfg,
        cfg.output_folder, nome="hsi_mapa_classificacao_amostra")

    from guaraci.avaliacao_modelos import PLSDAClassifier
    from guaraci.chemometric_stats import vip_scores
    clf_explicabilidade = PLSDAClassifier(
        n_components=resultado_modelo["n_components"])
    clf_explicabilidade.fit(X, y)
    vip = vip_scores(clf_explicabilidade._pls)
    achados_quimica = cross_reference_vip_with_chemistry(
        wavelengths, vip, ATRIBUICAO_QUIMICA_VIS_FRUTA, top_n=5)

    return {
        "n_gravacoes_total": len(cubos),
        "n_gravacoes_aceitas": len(cubos_ok),
        "n_gravacoes_rejeitadas": n_rejeitados,
        "motivos_rejeicao": motivos_rejeicao,
        "n_components": resultado_modelo["n_components"],
        "validacao_externa": relatorio_validacao,
        "achados_quimica": achados_quimica,
    }
