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
`run_hsi_pipeline` exigia `manifest.json` de um layout de terceiros --
agora aceita QUALQUER pasta com cubos ENVI do proprio usuario
(`hsi_io.load_hsi_folder_dataset`). Um `manifest.json` na pasta e' so'
mais um arquivo. Orquestracao:

  - `_run_hsi_pipeline_generico`: usa `load_hsi_folder_
    dataset` (Bloco 8: hierarquia de agrupamento por subpasta/CSV/nenhuma
    fonte), so' validacao INTERNA group-aware (nao ha' particao por dia
    num dataset generico -- `hsi_validation.run_internal_validation_
    group_aware` declara isso explicitamente, nao finge ter validacao
    externa), e SEM explicabilidade quimica cruzada (uma tabela de
    atribuicao quimica e' conhecimento de dominio de uma matriz
    especifica -- aplica-la a comprimentos de onda arbitrarios do usuario,
    ou a um eixo simbolico quando o `.hdr` nao traz `wavelength`, seria
    uma alegacao cientifica falsa, nao uma limitacao honesta).
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, TypedDict

import numpy as np

from guaraci.hsi_classification import fit_predict_pixel_plsda
from guaraci.hsi_figures import _fig_hsi_classification_map
from guaraci.hsi_io import load_hsi_folder_dataset
from guaraci.hsi_pixels import build_pixel_dataset
from guaraci.hsi_quality import evaluate_cube_quality
from guaraci.hsi_uncertainty import enrich_object_results
from guaraci.hsi_segmentation import segment_object_pca_otsu
from guaraci.hsi_validation import (run_cv_by_object,
                                    run_internal_validation_group_aware)

if TYPE_CHECKING:
    from guaraci.config import Config

__all__ = ["QualityGateResult", "apply_quality_gate_and_segment", "run_hsi_pipeline", "gravar_relatorio_hsi",
           "dados_validacao_hsi"]


class QualityGateResult(TypedDict):
    """Formato devolvido por `apply_quality_gate_and_segment` -- substitui
    o `Dict[str, object]` genérico que antes se propagava, sem tipo real,
    por ~10 call sites deste módulo (achado R2 da rodada multiagente
    2026-09-10, Passo 202: a raiz do tipo `object` solto estava aqui, não
    nos pontos de uso)."""
    cubos: List[np.ndarray]
    mascaras: List[np.ndarray]
    group_ids: List[str]
    rotulos: List[str]
    dias: List[str]
    n_rejeitados: int
    motivos_rejeicao: List[str]


def apply_quality_gate_and_segment(
        cubos: List[np.ndarray], group_ids: List[str], rotulos: List[str],
        dias: List[str],
        ) -> QualityGateResult:
    """Aplica o quality gate (Passo 95) a cada gravacao e segmenta
    (Passo 96) as que passam -- extraido de `run_hsi_pipeline` (Passo
    104) para ser reaproveitado por outros fluxos de HSI, sem duplicar a
    logica.

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


#: Teto de pixels de ROI por gravacao usados no AJUSTE/validacao (amostra
#: aleatoria semeada; ver `hsi_pixels.build_pixel_dataset`). Sem ele, cubos de
#: camera real (10^5-10^6 pixels de ROI x 200+ bandas) estouram memoria/tempo:
#: 6 cubos 300x400x224 levavam 9 min e 3,5 GB. Pixels vizinhos do mesmo objeto
#: sao fortemente correlacionados; 5000 por gravacao ja' cobrem a variabilidade.
MAX_PIXELS_POR_GRAVACAO = 5000


def _snv_por_pixel(espectros: np.ndarray) -> np.ndarray:
    """SNV (Barnes, Dhanoa & Lister 1989) aplicado a CADA pixel (espectro ao
    longo das bandas, ultimo eixo): remove o fator multiplicativo de
    iluminacao/curvatura do objeto, o efeito dominante em HSI de bancada."""
    media = espectros.mean(axis=-1, keepdims=True)
    dp = espectros.std(axis=-1, keepdims=True)
    return ((espectros - media) / np.where(dp > 1e-12, dp, 1.0)).astype(espectros.dtype, copy=False)


def _preparar_espectros(cfg: "Config", wavelengths: np.ndarray
                        ) -> "tuple[np.ndarray, Any, List[str]]":
    """Faixa espectral (faixa_min_cm/faixa_max_cm, lida em nm para HSI) e SNV por
    pixel quando o preset de pre-processamento o pede (o preset da tela [2.4]
    para HSI e' 'snv_mc'; a centragem fica no PLS). Devolve (eixo, transformar,
    notas): `transformar(pixels (n, bandas)) -> (n, bandas_da_faixa)` e' aplicado
    SO' aos pixels ja' escolhidos (a faixa e o SNV sao por pixel e comutam com a
    escolha dos pixels), de modo que nenhum cubo e' copiado inteiro -- a
    segmentacao continua no cubo bruto, onde o brilho separa objeto e fundo."""
    notas: List[str] = []
    wl = np.asarray(wavelengths, dtype=float)
    fmin, fmax = float(getattr(cfg, "wn_min", 0.0)), float(getattr(cfg, "wn_max", 1e9))
    manter = (wl >= fmin) & (wl <= fmax)
    if manter.all():
        notas.append(f"Faixa: todas as {wl.size} bandas ({wl.min():g}-{wl.max():g})")
        manter = None
    elif int(manter.sum()) < 2:
        notas.append(f"Faixa [{fmin:g}, {fmax:g}] IGNORADA: nao cobre o eixo do sensor "
                     f"({wl.min():g}-{wl.max():g} nm); usadas todas as bandas")
        print(f"[AVISO] HSI: {notas[-1]}.")
        manter = None
    else:
        wl = wl[manter]
        notas.append(f"Faixa: {int(manter.sum())} bandas ({wl.min():g}-{wl.max():g})")
    preset = str(getattr(cfg, "default_preprocessing", "") or "").lower()
    usar_snv = "snv" in preset
    if usar_snv:
        notas.append("Pre-processamento: SNV por pixel + centragem na media (PLS)")
    elif preset == "mc":
        notas.append("Pre-processamento: so' centragem na media (PLS)")
    else:
        notas.append(f"Pre-processamento: so' centragem na media (PLS); preset "
                     f"'{preset}' nao se aplica ao modo hsi (use snv_mc)")
        print(f"[AVISO] HSI: {notas[-1]}.")

    def transformar(px: np.ndarray) -> np.ndarray:
        if manter is not None:
            px = px[:, manter]
        return _snv_por_pixel(px) if usar_snv else px

    return wl, transformar, notas


def run_hsi_pipeline(cfg: "Config", *, fracao_teste_interno: float = 0.2, seed: int = 42) -> Dict[str, object]:
    """Roda o pipeline HSI completo sobre `cfg.hsi_dataset_folder`: qualquer pasta com cubos ENVI (.hdr+.bin), convencao de
    subpasta-por-classe (ver `hsi_io.load_hsi_folder_dataset`). So' validacao interna group-aware: num dataset generico nao ha'
    particao por dia, e `hsi_validation.run_internal_validation_group_aware` declara isso, nao finge ter validacao externa.
    Sem explicabilidade quimica cruzada: uma tabela de atribuicao e' conhecimento de dominio da matriz; aplica-la a comprimentos de
    onda arbitrarios do usuario seria uma alegacao cientifica falsa.

    Devolve um dict de resumo (nunca lanca excecao por dado individual ruim -- cenas que falham no quality gate sao PULADAS e
    contadas, nunca processadas em silencio) e salva o mapa de classificacao espacial em `cfg.output_folder`/Graficos/hsi/."""
    pasta = getattr(cfg, "hsi_dataset_folder", "") or ""
    if not pasta:
        raise ValueError(
            "run_hsi_pipeline: cfg.hsi_dataset_folder nao configurado -- "
            "aponte para uma pasta com seus cubos hiperespectrais ENVI "
            "(.hdr/.bin).")
    return _run_hsi_pipeline_generico(cfg, pasta, fracao_teste_interno=fracao_teste_interno, seed=seed)


def _preparar_pasta_saida(cfg: "Config") -> None:
    if not getattr(cfg, "output_folder", ""):
        # Nome dedicado -- a convencao PLSDA_OE_.../pipeline.generate_
        # output_name() e' especifica do fluxo tabular (preset de
        # pre-processamento, objetivo N1/N2/N3), nao se aplica ao HSI.
        raiz = getattr(cfg, "output_root_folder", "") or "resultados_guaraci"
        cfg.output_folder = os.path.join(
            raiz, "hsi", f"HSI_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


def _run_hsi_pipeline_generico(
        cfg: "Config", pasta: str, *, fracao_teste_interno: float,
        seed: int) -> Dict[str, object]:
    """Caminho NOVO (Passo 111) -- pasta com cubos ENVI do proprio
    usuario, sem `manifest.json`. Ver docstring do modulo para o porque
    de nao ter validacao externa nem explicabilidade quimica cruzada
    aqui (nenhuma das duas faz sentido sem metadado especifico de
    particao por dia)."""
    _preparar_pasta_saida(cfg)

    # lazy=True: os cubos ficam no disco (memmap) e cada etapa le blocos de linhas.
    cubos, rotulos, mae_id, wavelengths, meta_df = load_hsi_folder_dataset(
        pasta, lazy=True,
        calibracao=str(getattr(cfg, "hsi_calibration", "nenhuma") or "nenhuma"),
        referencia_branco=str(getattr(cfg, "hsi_white_ref", "") or ""),
        referencia_escuro=str(getattr(cfg, "hsi_dark_ref", "") or ""),
        eixo_bandas=str(getattr(cfg, "hsi_band_axis", "recusar") or "recusar"))
    nivel_agrupamento = meta_df.attrs.get("grouping_guarantee", "none")
    info_calibracao = dict(meta_df.attrs.get("calibracao") or {"modo": "nenhuma"})
    info_eixo = dict(meta_df.attrs.get("eixo_bandas") or {})

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
    # Segmentacao no cubo ORIGINAL (o brilho separa objeto/fundo); faixa e
    # SNV so' nos pixels escolhidos, para a modelagem.
    cubos_ok = filtrado["cubos"]
    wavelengths, transformar, notas_preproc = _preparar_espectros(cfg, wavelengths)
    for nota in notas_preproc:
        print(f"[INFO] HSI: {nota}.")
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

    X, y, pixel_groups = build_pixel_dataset(
        cubos_ok, mascaras_ok, grupos_ok, rotulos_ok,
        max_pixels_por_gravacao=MAX_PIXELS_POR_GRAVACAO, seed=seed,
        transformar=transformar)

    tipo_pedido = str(getattr(cfg, "hsi_validation", "cv_objetos") or "cv_objetos")
    relatorio_validacao = None
    avisos_validacao: List[str] = []
    if tipo_pedido == "cv_objetos":
        try:
            relatorio_validacao = run_cv_by_object(
                X, y, pixel_groups, n_dobras=int(getattr(cfg, "hsi_cv_folds", 5)), seed=seed)
        except ValueError as e:     # poucos objetos/classes para a CV: cai no holdout, dizendo
            avisos_validacao.append(f"CV por objeto indisponivel ({e}); usado o holdout de objetos.")
            print(f"[AVISO] HSI: {avisos_validacao[-1]}")
    if relatorio_validacao is None:
        relatorio_validacao = run_internal_validation_group_aware(
            cubos_ok, mascaras_ok, grupos_ok, rotulos_ok,
            fracao_teste=fracao_teste_interno, seed=seed,
            max_pixels_por_gravacao=MAX_PIXELS_POR_GRAVACAO, pixels=(X, y, pixel_groups))
        relatorio_validacao.avisos = avisos_validacao + list(relatorio_validacao.avisos)

    # Modelo FINAL (todos os objetos): so' fornece as VL e o mapa. A confianca por objeto
    # vem da CV (fora da amostra) quando ela existe; no holdout continua sendo a do
    # modelo final aplicado aos proprios objetos de treino (resubstituicao), declarado no relatorio.
    resultado_modelo = fit_predict_pixel_plsda(
        X, y, pixel_groups, X, pixel_groups, seed=seed)

    classes_ordenadas = sorted(set(y.tolist()))
    _mapa_fora_da_amostra(X, y, pixel_groups, mascaras_ok, grupos_ok,
                          rotulos_ok, classes_ordenadas,
                          int(resultado_modelo["n_components"]), cfg, seed,
                          cubos_ok=cubos_ok, transformar=transformar)

    if relatorio_validacao.confianca_fora_da_amostra:
        relatorio_confianca = enrich_object_results(relatorio_validacao.predicoes_objeto)
    else:
        relatorio_confianca = enrich_object_results(resultado_modelo["predicoes_objeto"])

    if nivel_agrupamento == "none":
        print("[WARNING] Validacao roda SEM garantia de agrupamento por "
              "amostra fisica (nivel 'none'). Resultados "
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
        "confianca_fora_da_amostra": bool(relatorio_validacao.confianca_fora_da_amostra),
        "grouping_guarantee": nivel_agrupamento,
        "wavelengths": wavelengths,
        "pre_processamento_aplicado": notas_preproc,
        "calibracao": info_calibracao,
        "eixo_bandas": info_eixo,
        "max_pixels_por_gravacao": MAX_PIXELS_POR_GRAVACAO,
        "n_pixels_ajuste": int(len(y)),
    }


def _amostra_do_mapa(y, pixel_groups, grupos_ok) -> int:
    """Indice da 1a gravacao cujo objeto fisico, se retirado do treino,
    ainda deixa >= 2 classes para ajustar o modelo do mapa (-1: nenhuma)."""
    for i, g in enumerate(grupos_ok):
        if len(set(np.asarray(y)[np.asarray(pixel_groups) != g].tolist())) >= 2:
            return i
    return -1


def _mapa_fora_da_amostra(X, y, pixel_groups, mascaras_ok, grupos_ok,
                          rotulos_ok, classes_ordenadas, n_comp_global: int,
                          cfg, seed: int, cubos_ok=None, transformar=None) -> None:
    """Mapa espacial de UMA gravacao com predicao FORA DA AMOSTRA: o PLS-DA
    do mapa e' ajustado em todos os OUTROS objetos fisicos (todo pixel com o
    group_id da gravacao mostrada sai do treino) e prediz so' essa gravacao.
    Antes o mapa vinha do modelo ajustado em X e aplicado ao mesmo X
    (in-sample, otimista). So' a FIGURA usa este ajuste; os numeros do
    resumo continuam vindo de `resultado_modelo`/validacao interna."""
    idx = _amostra_do_mapa(y, pixel_groups, grupos_ok)
    if idx < 0:
        print("[AVISO] Mapa HSI nao gerado: retirar qualquer objeto do treino "
              "deixa menos de 2 classes (predicao fora da amostra impossivel).")
        return
    g = grupos_ok[idx]
    pixel_groups = np.asarray(pixel_groups)
    treino = pixel_groups != g
    if cubos_ok is not None:   # ROI inteira da gravacao (X pode estar subamostrado)
        from guaraci.hsi_pixels import extract_roi_spectra
        X_mapa = extract_roi_spectra(cubos_ok[idx], mascaras_ok[idx], dtype=np.float32)
        if transformar is not None:
            X_mapa = transformar(X_mapa)
    else:
        inicio = int(sum(int(np.asarray(m).sum()) for m in mascaras_ok[:idx]))
        X_mapa = X[inicio:inicio + int(np.asarray(mascaras_ok[idx]).sum())]
    g_mapa = np.full(len(X_mapa), g, dtype=object).astype(str)
    try:                       # LVs re-escolhidas SO' com os objetos de treino
        res = fit_predict_pixel_plsda(
            X[treino], y[treino], pixel_groups[treino],
            X_mapa, g_mapa,
            seed=seed)
    except ValueError:         # poucos grupos p/ o Wold: usa o n de LVs global
        res = fit_predict_pixel_plsda(
            X[treino], y[treino], pixel_groups[treino],
            X_mapa, g_mapa,
            n_components=n_comp_global, seed=seed)
    nome_amostra = (f"gravação {idx + 1}" if str(g).startswith("__sem_grupo_")
                    else f"objeto '{g}' (gravação {idx + 1})")
    titulo = (f"Mapa de classificação HSI por pixel — {nome_amostra}\n"
              f"classe verdadeira: {rotulos_ok[idx]}  |  predição FORA DA "
              f"AMOSTRA\n(modelo ajustado sem este objeto, "
              f"{int(res['n_components'])} VL)")
    _fig_hsi_classification_map(
        mascaras_ok[idx], res["predicoes_pixel"], classes_ordenadas, cfg,
        cfg.output_folder, nome="hsi_mapa_classificacao_amostra",
        titulo=titulo)


def _fmt_metrica(v: float) -> str:
    return "n/d (sem objeto desta classe no teste)" if not np.isfinite(v) else f"{v:.2f}"


def _ic_txt(ic) -> str:
    lo, hi = ic
    return "IC95% n/d" if not (np.isfinite(lo) and np.isfinite(hi)) else f"IC95% {lo:.2f}-{hi:.2f}"


def _num(v: float):
    return None if v is None or not np.isfinite(v) else float(v)


def dados_validacao_hsi(resumo: Dict[str, Any]) -> Dict[str, Any]:
    """Campos que descrevem a validacao (tipo, n de objetos, metricas por classe com IC),
    na forma gravada em `hsi_relatorio.json` e lida pela ficha de vazamento/relatorio formal."""
    val: Any = resumo["validacao_externa"]
    tipo = getattr(val, "tipo_validacao", "holdout")
    por_classe = {}
    for c in getattr(val, "classes", []):
        ic_s = getattr(val, "ic_sensibilidade_interna", {}).get(c, (float("nan"), float("nan")))
        ic_e = getattr(val, "ic_especificidade_interna", {}).get(c, (float("nan"), float("nan")))
        por_classe[c] = {
            "n_objetos": int(getattr(val, "n_objetos_por_classe", {}).get(c, 0)),
            "sensibilidade": _num(val.sensibilidade_interna[c]),
            "sensibilidade_ic95": [_num(ic_s[0]), _num(ic_s[1])],
            "especificidade": _num(val.especificidade_interna[c]),
            "especificidade_ic95": [_num(ic_e[0]), _num(ic_e[1])],
            "precisao": _num(getattr(val, "precisao_interna", {}).get(c, float("nan")))}
    return {
        "tipo": tipo,                                       # "cv_objetos" | "holdout" | "externa_por_dia"
        "descricao": ("validacao cruzada por objeto fisico: cada objeto previsto por um modelo ajustado sem ele"
                      if tipo == "cv_objetos" else
                      "holdout unico de objetos fisicos sorteados (estimativa indicativa)"),
        "n_dobras": int(getattr(val, "n_dobras", 0)),       # 0 = holdout; na CV = n de dobras (n objetos se leave-one-out)
        "n_objetos_total": int(getattr(val, "n_objetos_total", 0)),
        "n_objetos_avaliados": int(getattr(val, "n_objetos_avaliados", 0)),
        "n_objetos_por_classe": {k: int(v) for k, v in getattr(val, "n_objetos_por_classe", {}).items()},
        "confianca_por_objeto_fora_da_amostra": bool(resumo.get("confianca_fora_da_amostra", False)),
        "acuracia_objetos": _num(getattr(val, "acuracia_objetos", float("nan"))),
        "acuracia_objetos_ic95": [_num(v) for v in getattr(val, "ic_acuracia_objetos", (float("nan"),) * 2)],
        "acuracia_balanceada_objetos": _num(getattr(val, "acuracia_balanceada_objetos", float("nan"))),
        "nivel_ic": float(getattr(val, "nivel_ic", 0.95)),
        "metodo_ic": "Clopper-Pearson (cada objeto e' uma tentativa; aproximado na CV, pois as dobras compartilham treino)",
        "por_classe": por_classe,
        "avisos": list(getattr(val, "avisos", [])),
    }


def gravar_relatorio_hsi(cfg: "Config", resumo: Dict[str, Any]) -> str:
    """Grava Relatorios/hsi_relatorio.txt, hsi_relatorio.json e config_efetiva.yaml na pasta da
    execucao HSI (o que `guaraci run` faz para os outros modos). Devolve o caminho do relatorio .txt.
    O .json traz os campos da validacao/calibracao/eixo para a ficha de vazamento e o relatorio formal."""
    import json
    from guaraci.config import NOME_RELATORIOS, pasta_execucao
    from guaraci.config_io import save_config
    pasta_rel = pasta_execucao(cfg.output_folder, NOME_RELATORIOS)
    os.makedirs(pasta_rel, exist_ok=True)
    val: Any = resumo["validacao_externa"]
    pre_aplicado: Any = resumo.get("pre_processamento_aplicado") or []
    dv = dados_validacao_hsi(resumo)
    calib: Any = resumo.get("calibracao") or {"modo": "nenhuma"}
    eixo: Any = resumo.get("eixo_bandas") or {}
    linhas = [
        f"Gravacoes aceitas: {resumo['n_gravacoes_aceitas']}/{resumo['n_gravacoes_total']} "
        f"(rejeitadas pelo controle de qualidade: {resumo['n_gravacoes_rejeitadas']})",
        f"Variaveis latentes (Wold): {resumo['n_components']}",
        f"Garantia de agrupamento por objeto: {resumo['grouping_guarantee']}",
        *[str(n) for n in pre_aplicado],
    ]
    if calib.get("modo") == "branco_escuro":
        linhas.append(f"Calibracao radiometrica: branco/escuro, R = (I - D)/(W - D), "
                      f"{calib.get('n_cubos_calibrados', 0)} cubo(s) calibrado(s)")
        linhas.extend(f"  aviso de calibracao: {a}" for a in calib.get("avisos", []))
    else:
        linhas.append("Calibracao radiometrica: nenhuma (cubos usados como estao; presume-se reflectancia)")
    if eixo:
        linhas.append(f"Eixo de bandas: {eixo.get('n_bandas')} bandas"
                      + (f", {eixo['faixa_nm'][0]:g}-{eixo['faixa_nm'][1]:g} nm" if eixo.get("faixa_nm") else "")
                      + f"; cubos reamostrados: {eixo.get('n_reamostrados', 0)} (modo {eixo.get('modo')})")
    if dv["tipo"] == "cv_objetos":
        nd = dv["n_dobras"]
        linhas.append(
            f"Validacao: CV por objeto, {'leave-one-object-out' if nd >= dv['n_objetos_total'] else str(nd) + ' dobras'}; "
            f"{dv['n_objetos_avaliados']} de {dv['n_objetos_total']} objetos previstos fora da amostra "
            f"(LV reescolhidas dentro de cada treino)")
    else:
        linhas.append(
            f"Validacao: holdout unico de {dv['n_objetos_avaliados']} objeto(s) de {dv['n_objetos_total']} "
            f"(estimativa indicativa; a CV por objeto e' hsi_validacao: cv_objetos)")
    linhas.append(f"Objetos de teste: interno {getattr(val, 'n_objetos_teste_interno', 0)}, "
                  f"externo {getattr(val, 'n_objetos_teste_externo', 0)}")
    for classe in getattr(val, "classes", []):
        linhas.append(
            f"  {classe}: sensibilidade interna {_fmt_metrica(val.sensibilidade_interna[classe])} "
            f"({_ic_txt(getattr(val, 'ic_sensibilidade_interna', {}).get(classe, (float('nan'),) * 2))}), "
            f"especificidade interna {_fmt_metrica(val.especificidade_interna[classe])} "
            f"({_ic_txt(getattr(val, 'ic_especificidade_interna', {}).get(classe, (float('nan'),) * 2))}); "
            f"n objetos = {getattr(val, 'n_objetos_por_classe', {}).get(classe, 0)}")
    if np.isfinite(dv["acuracia_objetos"] if dv["acuracia_objetos"] is not None else float("nan")):
        lo, hi = dv["acuracia_objetos_ic95"]
        linhas.append(f"Acuracia por objeto: {dv['acuracia_objetos']:.2f} (IC95% {lo:.2f}-{hi:.2f}); "
                      f"acuracia balanceada: {dv['acuracia_balanceada_objetos']:.2f}")
    linhas.append("Confianca por objeto: " + ("fora da amostra (previsao por modelo que nao viu o objeto)"
                                              if dv["confianca_por_objeto_fora_da_amostra"] else
                                              "DENTRO da amostra (modelo final aplicado aos proprios objetos): otimista"))
    linhas.append(f"Pixels no ajuste: {resumo.get('n_pixels_ajuste', 'n/d')} "
                  f"(no maximo {resumo.get('max_pixels_por_gravacao', 'n/d')} por gravacao, amostra aleatoria semeada)")
    linhas.append("IC: Clopper-Pearson a 95%, cada objeto e' uma tentativa; na CV e' aproximado (dobras compartilham treino).")
    linhas.extend(f"AVISO: {a}" for a in dv["avisos"])
    n_porclasse = getattr(val, "n_objetos_por_classe", None) or {}
    n_teste = int(sum(n_porclasse.values())) if n_porclasse else int(getattr(val, "n_objetos_teste_interno", 0) or 0)
    from guaraci.resumo_parse import N_TESTE_HSI_INDICATIVO
    if n_teste < N_TESTE_HSI_INDICATIVO:
        linhas.append(f"Aviso: so' {n_teste} objeto(s) avaliado(s) fora da amostra (menos de {N_TESTE_HSI_INDICATIVO}): "
                      "sensibilidade e especificidade sao metrica indicativa, nao estimativa de desempenho.")
    if resumo["grouping_guarantee"] == "none":
        linhas.append("Sem garantia de agrupamento por objeto: resultado exploratorio.")
    caminho = os.path.join(pasta_rel, "hsi_relatorio.txt")
    with open(caminho, "w", encoding="utf-8") as fh:
        fh.write("\n".join(linhas) + "\n")
    dados_json = {
        "versao": 1,
        "n_gravacoes_total": int(resumo["n_gravacoes_total"]),
        "n_gravacoes_aceitas": int(resumo["n_gravacoes_aceitas"]),
        "n_gravacoes_rejeitadas": int(resumo["n_gravacoes_rejeitadas"]),
        "n_components": int(resumo["n_components"]),
        "grouping_guarantee": str(resumo["grouping_guarantee"]),
        "pre_processamento": [str(n) for n in pre_aplicado],
        "calibracao": calib,
        "eixo_bandas": eixo,
        "validacao": dv,
        "n_pixels_ajuste": resumo.get("n_pixels_ajuste"),
        "max_pixels_por_gravacao": resumo.get("max_pixels_por_gravacao"),
    }
    with open(os.path.join(pasta_rel, "hsi_relatorio.json"), "w", encoding="utf-8") as fh:
        json.dump(dados_json, fh, ensure_ascii=False, indent=2)
    save_config(cfg, os.path.join(cfg.output_folder, "config_efetiva.yaml"))
    return caminho
