"""hsi_pipeline.py — Orquestracao ponta-a-ponta do modo de entrada `hsi`
(Passo 102 da `INSTRUCAO_HSI_MINIMO_VIAVEL.md`): leitura -> quality gate
-> segmentacao -> extracao de ROI/agrupamento -> classificacao por
pixel -> mapa espacial -> explicabilidade cruzada -> validacao.

Modo `hsi` e' DISTINTO do modo `imagem` (colorimetria digital de foto
comum, `dados_imagem.py`) -- nunca confundir os dois: `imagem` extrai
UMA linha de features por FOTO inteira e reusa `pipeline.executar()`
sem alteracao (mesma matriz amostras x variaveis de sempre); `hsi`
opera POR PIXEL de um cubo hiperespectral, com agregacao por objeto
fisico -- forma de dado fundamentalmente diferente, por isso tem sua
propria orquestracao aqui em vez de entrar em `pipeline.executar()`.

Passo 111 (INSTRUCAO_HSI_DADO_PROPRIO.md) generalizou a entrada: ANTES,
`run_hsi_pipeline` exigia `manifest.json` de um dataset publico
especifico (DeepHS Fruit) -- agora aceita QUALQUER pasta com cubos ENVI
do proprio usuario (`hsi_io.load_hsi_folder_dataset`), e o dataset
publico vira so' um FIXTURE opcional de validacao. `run_hsi_pipeline`
detecta qual caso e' (presenca de `manifest.json` na raiz da pasta) e
despacha pra' uma das 2 orquestracoes internas:

  - `_run_hsi_pipeline_deephs_fixture`: caminho ORIGINAL (Passos 94-104),
    inalterado -- usa `load_deephs_kaki_dataset`/`load_deephs_fruit_
    dataset`, particao de validacao externa por DIA de medicao (metadado
    que so' o dataset publico tem), e explicabilidade cruzada com a
    tabela de atribuicao quimica especifica de VIS-fruta.
  - `_run_hsi_pipeline_generico`: caminho NOVO -- usa `load_hsi_folder_
    dataset` (Bloco 8: hierarquia de agrupamento por subpasta/CSV/nenhuma
    fonte), so' validacao INTERNA group-aware (nao ha' particao por dia
    num dataset generico -- `hsi_validation.run_internal_validation_
    group_aware` declara isso explicitamente, nao finge ter validacao
    externa), e SEM explicabilidade quimica cruzada (a tabela `ATRIBUICAO_
    QUIMICA_VIS_FRUTA` e' conhecimento de dominio especifico do dataset
    publico -- aplica-la a comprimentos de onda arbitrarios do usuario,
    ou a um eixo simbolico quando o `.hdr` nao traz `wavelength`, seria
    uma alegacao cientifica falsa, nao uma limitacao honesta).
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
from guaraci.hsi_io import load_deephs_kaki_dataset, load_hsi_folder_dataset
from guaraci.hsi_pixels import build_pixel_dataset
from guaraci.hsi_quality import evaluate_cube_quality
from guaraci.hsi_segmentation import segment_object_pca_otsu
from guaraci.hsi_uncertainty import enrich_object_results
from guaraci.hsi_validation import (run_external_validation_by_day,
                                    run_internal_validation_group_aware)

if TYPE_CHECKING:
    from guaraci.config import Config

__all__ = ["apply_quality_gate_and_segment", "run_hsi_pipeline"]


def apply_quality_gate_and_segment(
        cubos: List[np.ndarray], group_ids: List[str], rotulos: List[str],
        dias: List[str],
        ) -> Dict[str, object]:
    """Aplica o quality gate (Passo 95) a cada gravacao e segmenta
    (Passo 96) as que passam -- extraido de `run_hsi_pipeline` (Passo
    104) para ser reaproveitado tambem pela validacao contra outras
    frutas/cameras do DeepHS Fruit (`tests/
    test_validacao_publica_deephs_fruit.py`), sem duplicar a logica.

    Devolve um dict com `cubos`/`mascaras`/`group_ids`/`rotulos`/`dias`
    (so' as gravacoes ACEITAS, alinhados por indice) + `n_rejeitados`/
    `motivos_rejeicao` (nunca descarta silenciosamente -- toda rejeicao
    fica rastreavel)."""
    cubos_ok: List[np.ndarray] = []
    mascaras_ok: List[np.ndarray] = []
    grupos_ok: List[str] = []
    rotulos_ok: List[str] = []
    dias_ok: List[str] = []
    n_rejeitados = 0
    motivos_rejeicao: List[str] = []

    for cubo, gid, rotulo, dia in zip(cubos, group_ids, rotulos, dias):
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

    return {
        "cubos": cubos_ok, "mascaras": mascaras_ok, "group_ids": grupos_ok,
        "rotulos": rotulos_ok, "dias": dias_ok,
        "n_rejeitados": n_rejeitados, "motivos_rejeicao": motivos_rejeicao,
    }


def run_hsi_pipeline(cfg: "Config", *, fracao_teste_interno: float = 0.2,
                     n_dias_externos: int = 2, seed: int = 42,
                     ) -> Dict[str, object]:
    """Roda o pipeline HSI completo sobre `cfg.hsi_dataset_folder`.

    Aceita 2 casos (Passo 111, detectado automaticamente pela presenca
    de `manifest.json` na raiz da pasta -- nunca uma flag manual que o
    usuario precisaria lembrar de setar certo):

      - Pasta SEM `manifest.json` (caso normal -- dado proprio do
        usuario): qualquer pasta com cubos ENVI (.hdr+.bin), convencao
        de subpasta-por-classe (ver `hsi_io.load_hsi_folder_dataset`).
        So' validacao interna (nao ha' particao por dia num dataset
        generico).
      - Pasta COM `manifest.json` (dataset publico DeepHS Fruit, usado
        so' como FIXTURE de validacao do projeto -- ver
        `scripts/download_datasets/baixar_deephs_kaki.py`/
        `baixar_deephs_fruit_todas.py`): caminho original com validacao
        externa por dia de medicao + explicabilidade quimica cruzada.

    Devolve um dict de resumo (nunca lanca excecao por dado individual
    ruim -- cenas que falham no quality gate sao PULADAS e contadas,
    nunca processadas em silencio) e salva o mapa de classificacao
    espacial em `cfg.output_folder`/Graficos/hsi/.

    `n_dias_externos`: so' usado no caminho do dataset publico -- quantos
    dos dias mais RECENTES (ordenados por nome, que no dataset usado
    correspondem a ordem cronologica) viram o teste de validacao externa
    (Passo 101)."""
    pasta = getattr(cfg, "hsi_dataset_folder", "") or ""
    if not pasta:
        raise ValueError(
            "run_hsi_pipeline: cfg.hsi_dataset_folder nao configurado -- "
            "aponte para uma pasta com seus cubos hiperespectrais ENVI "
            "(.hdr/.bin).")

    if os.path.isfile(os.path.join(pasta, "manifest.json")):
        return _run_hsi_pipeline_deephs_fixture(
            cfg, pasta, fracao_teste_interno=fracao_teste_interno,
            n_dias_externos=n_dias_externos, seed=seed)
    return _run_hsi_pipeline_generico(
        cfg, pasta, fracao_teste_interno=fracao_teste_interno, seed=seed)


def _preparar_pasta_saida(cfg: "Config") -> None:
    if not getattr(cfg, "output_folder", ""):
        # Nome dedicado -- a convencao PLSDA_OE_.../pipeline.generate_
        # output_name() e' especifica do fluxo tabular (preset de
        # pre-processamento, objetivo N1/N2/N3), nao se aplica ao HSI.
        raiz = getattr(cfg, "output_root_folder", "") or "resultados_tcc"
        cfg.output_folder = os.path.join(
            raiz, "hsi", f"HSI_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


def _run_hsi_pipeline_deephs_fixture(
        cfg: "Config", pasta: str, *, fracao_teste_interno: float,
        n_dias_externos: int, seed: int) -> Dict[str, object]:
    """Caminho ORIGINAL (Passos 94-104), inalterado -- dataset publico
    DeepHS Fruit/Kaki/VIS (`manifest.json` no formato de `load_deephs_
    kaki_dataset`), usado como fixture de validacao do projeto."""
    _preparar_pasta_saida(cfg)

    cubos, rotulos, grupos, wavelengths, meta_df = load_deephs_kaki_dataset(pasta)

    filtrado = apply_quality_gate_and_segment(
        cubos, list(grupos), list(rotulos), list(meta_df["day"]))
    cubos_ok = filtrado["cubos"]
    mascaras_ok = filtrado["mascaras"]
    grupos_ok = filtrado["group_ids"]
    rotulos_ok = filtrado["rotulos"]
    dias_ok = filtrado["dias"]
    n_rejeitados = filtrado["n_rejeitados"]
    motivos_rejeicao = filtrado["motivos_rejeicao"]

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

    relatorio_confianca = enrich_object_results(resultado_modelo["predicoes_objeto"])

    return {
        "n_gravacoes_total": len(cubos),
        "n_gravacoes_aceitas": len(cubos_ok),
        "n_gravacoes_rejeitadas": n_rejeitados,
        "motivos_rejeicao": motivos_rejeicao,
        "n_components": resultado_modelo["n_components"],
        "validacao_externa": relatorio_validacao,
        "achados_quimica": achados_quimica,
        "confianca_por_objeto": relatorio_confianca,
        "grouping_guarantee": "high",  # objeto fisico via nome de arquivo (Passo 97)
    }


def _run_hsi_pipeline_generico(
        cfg: "Config", pasta: str, *, fracao_teste_interno: float,
        seed: int) -> Dict[str, object]:
    """Caminho NOVO (Passo 111) -- pasta com cubos ENVI do proprio
    usuario, sem `manifest.json`. Ver docstring do modulo para o porque
    de nao ter validacao externa nem explicabilidade quimica cruzada
    aqui (nenhuma das duas faz sentido sem metadado especifico do
    dataset publico)."""
    _preparar_pasta_saida(cfg)

    cubos, rotulos, mae_id, wavelengths, meta_df = load_hsi_folder_dataset(pasta)
    nivel_agrupamento = meta_df.attrs.get("grouping_guarantee", "none")

    # Nivel "none" (Bloco 8): mae_id vem None do loader -- cada gravacao
    # precisa de um id UNICO aqui (nunca um placeholder compartilhado,
    # que colapsaria objetos fisicos DIFERENTES no mesmo grupo e
    # destruiria a garantia que build_pixel_dataset/hsi_validation
    # existem pra' dar).
    if mae_id is None:
        grupos: List[str] = [f"__sem_grupo_{i}__" for i in range(len(cubos))]
    else:
        grupos = [str(g) for g in mae_id]

    filtrado = apply_quality_gate_and_segment(
        cubos, grupos, list(rotulos), [""] * len(cubos))
    cubos_ok = filtrado["cubos"]
    mascaras_ok = filtrado["mascaras"]
    grupos_ok = filtrado["group_ids"]
    rotulos_ok = filtrado["rotulos"]
    n_rejeitados = filtrado["n_rejeitados"]
    motivos_rejeicao = filtrado["motivos_rejeicao"]

    if not cubos_ok:
        raise ValueError(
            f"run_hsi_pipeline: TODAS as {len(cubos)} gravacoes falharam "
            f"no quality gate -- nada a processar. Motivos: "
            f"{motivos_rejeicao[:5]}")

    relatorio_validacao = run_internal_validation_group_aware(
        cubos_ok, mascaras_ok, grupos_ok, rotulos_ok,
        fracao_teste=fracao_teste_interno, seed=seed)

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

    relatorio_confianca = enrich_object_results(resultado_modelo["predicoes_objeto"])

    if nivel_agrupamento == "none":
        print("[WARNING] Validacao roda SEM garantia de agrupamento por "
              "amostra fisica (nivel 'none' -- ver hsi_io.py). Resultados "
              "devem ser tratados como exploratorios.")

    return {
        "n_gravacoes_total": len(cubos),
        "n_gravacoes_aceitas": len(cubos_ok),
        "n_gravacoes_rejeitadas": n_rejeitados,
        "motivos_rejeicao": motivos_rejeicao,
        "n_components": resultado_modelo["n_components"],
        "validacao_externa": relatorio_validacao,
        "achados_quimica": None,  # nao aplicavel: sem tabela de atribuicao
                                  # quimica generica p/ dado do proprio usuario
        "confianca_por_objeto": relatorio_confianca,
        "grouping_guarantee": nivel_agrupamento,
        "wavelengths": wavelengths,
    }
