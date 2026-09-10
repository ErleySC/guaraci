"""app_tabs/inicio.py — Tela (Home): painel de status do projeto.

Layout do mockup de 2026-09-08 seguido à risca: faixa "próxima ação
sugerida" no topo e quatro cartões (dados carregados, espectros médios por
classe, resultado da predição, faixa de decisão).

O que NÃO foi seguido à risca, de propósito: os números. O mockup traz
"934 espectros / 311 amostras / 14 classes / 7,4%" como valores de exemplo
— aqui cada um vem de execução real ou de `st.session_state`, e o cartão
diz que está vazio quando não há dado, em vez de mostrar um número que não
existe.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from guaraci.app_logic import load_summary, next_action
from guaraci.chemometric_stats import (
    FAIXA_NAO_DETECTAVEL,
    FAIXA_QUANTIFICADO,
    FAIXA_ZONA_CINZENTA,
)
from guaraci.config import NOME_RELATORIOS
from guaraci.resultados_io import load_design_audit
from guaraci.resumo_parse import extract_metric, parse_dataset_counts


def _pasta_relatorios(pasta_run: str) -> str:
    for candidata in (os.path.join(pasta_run, NOME_RELATORIOS),
                      os.path.join(pasta_run, "logs"), pasta_run):
        if os.path.isdir(candidata):
            return candidata
    return pasta_run


def _cartao(titulo: str):
    """Abre um cartão (container com borda) e escreve o título em caixa alta,
    como no mockup."""
    caixa = st.container(border=True)
    caixa.markdown(
        f"<div style='font-size:.72rem;font-weight:800;letter-spacing:.07em;"
        f"text-transform:uppercase;color:rgba(128,128,128,1);"
        f"margin-bottom:.4rem'>{titulo}</div>",
        unsafe_allow_html=True)
    return caixa


def _numero(caixa, valor: Optional[object], rotulo: str) -> None:
    """Número grande + rótulo pequeno. `None` vira '—', nunca zero: zero se
    leria como 'nenhum', que é uma afirmação diferente de 'não informado'."""
    texto = "—" if valor is None else str(valor)
    caixa.markdown(
        f"<div style='font-size:1.7rem;font-weight:800;line-height:1.1'>"
        f"{texto}</div><div style='font-size:.68rem;"
        f"color:rgba(128,128,128,1)'>{rotulo}</div>",
        unsafe_allow_html=True)


def _linha(caixa, esquerda: str, direita: str) -> None:
    caixa.markdown(
        f"<div style='display:flex;justify-content:space-between;gap:10px;"
        f"padding:.35rem 0;border-bottom:1px solid rgba(128,128,128,.22);"
        f"font-size:.82rem'><span>{esquerda}</span>"
        f"<b style='text-align:right'>{direita}</b></div>",
        unsafe_allow_html=True)


def _selo(texto: str, cor_fundo: str, cor_texto: str) -> str:
    return (f"<span style='font-size:.66rem;font-weight:700;padding:.1rem .45rem;"
            f"border-radius:999px;background:{cor_fundo};color:{cor_texto}'>"
            f"{texto}</span>")


def _faixa_do_resultado(resultados_cego) -> Optional[Dict[str, object]]:
    """Primeiro resultado com teor estimado, para o cartão de faixa de
    decisão. `None` quando nenhuma amostra foi quantificada."""
    for r in resultados_cego or []:
        if r.quantificacao.teor_estimado is not None:
            return {
                "teor": r.quantificacao.teor_estimado,
                "lod": r.quantificacao.lod,
                "loq": r.quantificacao.loq,
                "faixa": r.quantificacao.faixa_decisao,
                "especie": r.quantificacao.especie_usada,
                "cobertura": (r.identificacao.cobertura_status.value
                              if r.identificacao.cobertura_status else None),
            }
    return None


def render(pq, T: Callable[[str], str], ir_para: Callable[[str], None],
           tok: Callable[[], Dict[str, str]]) -> None:
    """Desenha a tela Início. `ir_para` navega para outra tela (a barra
    lateral e os botões desta tela compartilham a mesma função)."""
    pt = st.session_state.get("lang") != "EN"
    tk = tok()

    pasta_run = st.session_state.get("ultima_pasta")
    tem_execucao = bool(pasta_run) and os.path.isdir(pasta_run or "")
    previa = st.session_state.get("previa_dados")
    df_pred = st.session_state.get("pred_resultados")
    resultados_cego = st.session_state.get("pred_resultados_cego")
    achados: List[Dict[str, str]] = (
        load_design_audit(_pasta_relatorios(str(pasta_run)))
        if tem_execucao else [])

    # ── Faixa "próxima ação sugerida" ────────────────────────────────────
    acao = next_action(
        tem_dados=bool(previa) or tem_execucao,
        tem_execucao=tem_execucao,
        achados_auditoria=achados,
        tem_predicao=df_pred is not None,
        pt=pt)
    cor = {"critico": tk["error"], "aviso": tk["warn"],
           "ok": tk["success"], "info": tk["accent"]}[acao.severidade]
    faixa = st.container(border=True)
    col_txt, col_btn = faixa.columns([5, 1])
    col_txt.markdown(
        f"<div style='border-left:4px solid {cor};padding-left:.7rem'>"
        f"<b>{acao.titulo}</b><br>"
        f"<span style='font-size:.82rem;color:rgba(128,128,128,1)'>"
        f"{acao.detalhe}</span></div>", unsafe_allow_html=True)
    if acao.destino:
        col_btn.write("")
        if col_btn.button(T("Go there"), key="btn_acao_sugerida",
                          use_container_width=True):
            ir_para(acao.destino)

    esq, dir_ = st.columns(2)

    # ── Cartão 1: dados carregados ───────────────────────────────────────
    with esq:
        card = _cartao(T("DATA LOADED"))
        if tem_execucao:
            resumo = load_summary(str(pasta_run)) or ""
            contagens = parse_dataset_counts(resumo)
            origem = T("last completed run")
        elif previa:
            resumo = ""
            contagens = {"amostras": previa.get("n_espectros"),
                         "classes": previa.get("n_classes"),
                         "amostras_fisicas": None, "variaveis": None}
            origem = T("data preview (no run yet)")
        else:
            resumo, contagens, origem = "", {}, ""

        if not contagens:
            card.info(T("No data loaded yet — start on the **Data** screen: "
                        "point to a spectra folder or upload a CSV."))
        else:
            c1, c2, c3 = card.columns(3)
            _numero(c1, contagens.get("amostras"), T("spectra"))
            _numero(c2, contagens.get("amostras_fisicas"),
                    T("physical samples"))
            _numero(c3, contagens.get("classes"), T("classes"))
            card.write("")
            _linha(card, T("Technique"), _tecnica(pq, tk))
            _linha(card, T("Preprocessing"),
                   extract_metric(resumo, r"Pre.?processamento\s*[:=]\s*(.+)",
                                  "—") if resumo else "—")
            _linha(card, T("Grouping"), _agrupamento(resumo, tk, T))
            card.caption(T("Source: {origem}.").format(origem=origem))

    # ── Cartão 2: espectros médios por classe ────────────────────────────
    with dir_:
        card = _cartao(T("MEAN SPECTRA BY CLASS"))
        espectros = st.session_state.get("previa_espectros")
        if espectros is not None:
            from guaraci.spectra_preview import plot_mean_spectra
            fig = plot_mean_spectra(espectros["wn"], espectros["X"],
                                    espectros["labels"])
            card.pyplot(fig, use_container_width=True)
            plt.close(fig)
            card.caption(T("Wavenumber (cm⁻¹) — decreasing"))
        else:
            card.info(T("Load the spectra preview on the **Data** screen to "
                        "see the mean spectrum of each class here."))

    esq2, dir2 = st.columns(2)

    # ── Cartão 3: resultado da predição ──────────────────────────────────
    with esq2:
        card = _cartao(T("PREDICTION RESULT"))
        if df_pred is None:
            card.info(T("No prediction yet — the **Prediction** screen "
                        "applies a saved model to new samples."))
        else:
            n_tot = len(df_pred)
            if resultados_cego:
                n_adult = sum(1 for r in resultados_cego
                              if r.pureza.aceito is False)
                n_ident = sum(1 for r in resultados_cego
                              if r.identificacao.classe_identificada is not None)
                n_quant = sum(1 for r in resultados_cego
                              if r.quantificacao.teor_estimado is not None)
                _linha(card, T("Detection"),
                       T("{n} of {t} adulterated").format(n=n_adult, t=n_tot))
                _linha(card, T("Identification"),
                       (T("{n} of {t} identified").format(n=n_ident, t=n_tot)
                        if n_ident else
                        f"<span style='color:{tk['warn']}'>"
                        f"{T('Unknown adulterant')}</span>"))
                _linha(card, T("Quantification"),
                       (T("{n} of {t} quantified").format(n=n_quant, t=n_tot)
                        if n_quant else
                        f"<span style='color:{tk['text_muted']}'>"
                        f"{T('Not performed')}</span>"))
                if not n_quant:
                    card.warning(T(
                        "Quantification was blocked because identification "
                        "found no match with a known adulterant. A number "
                        "here would have no statistical backing."))
            else:
                dentro = (int(df_pred["aceito"].sum())
                          if "aceito" in df_pred.columns else None)
                _linha(card, T("Samples"), str(n_tot))
                _linha(card, T("Within PLS-DA model fit"),
                       "—" if dentro is None else f"{dentro} / {n_tot}")
                card.caption(T("This model predates the blind flow, so there "
                               "is no identification/quantification to show."))

    # ── Cartão 4: faixa de decisão · teor estimado ───────────────────────
    with dir2:
        card = _cartao(T("DECISION RANGE · ESTIMATED CONTENT"))
        dados_faixa = _faixa_do_resultado(resultados_cego)
        if dados_faixa is None:
            card.info(T("No quantified sample yet — the decision range "
                        "appears once a content is estimated."))
        else:
            _desenhar_faixa(card, dados_faixa, tk, T)


def _tecnica(pq, tk: Dict[str, str]) -> str:
    """Matriz/técnica do perfil ativo + selo de validação pública."""
    try:
        from pathlib import Path

        from guaraci.perfil_matriz import cfg_profile
        from guaraci.validacao_publica import (find_rows_for_matrix,
                                               load_consolidated_table)
        perfil = cfg_profile(st.session_state.get("cfg_base"))
        matriz = perfil.vocabulario.matriz
        raiz = Path(__file__).resolve().parents[3]
        linhas = find_rows_for_matrix(load_consolidated_table(raiz), matriz)
        if linhas:
            return matriz + " " + _selo("validada", tk["success_bg"],
                                        tk["success"])
        return matriz + " " + _selo("sem validação pública", tk["warn_bg"],
                                    tk["warn"])
    except Exception:  # noqa: BLE001 -- linha informativa do painel; sem
        # perfil resolvido nao ha' o que exibir, e o cartao segue.
        return "—"


def _agrupamento(resumo: str, tk: Dict[str, str], T: Callable[[str], str]) -> str:
    """Garantia de agrupamento da execução (é o que diz se a validação está
    protegida contra vazamento entre réplicas)."""
    if not resumo:
        return "—"
    grupo = extract_metric(resumo, r"Group-aware \(mae_id\)\s*[:=]\s*(\w+)", "")
    garantia = extract_metric(resumo, r"Grouping guarantee\s*[:=]\s*(\w+)", "")
    if grupo.lower() in ("sim", "yes", "true"):
        return (T("By physical sample") + " " +
                _selo(T("protected"), tk["success_bg"], tk["success"]))
    if garantia:
        return (T("Guarantee: {g}").format(g=garantia) + " " +
                _selo(T("unprotected"), tk["warn_bg"], tk["warn"]))
    return "—"


def _desenhar_faixa(card, dados: Dict[str, object], tk: Dict[str, str],
                    T: Callable[[str], str]) -> None:
    """Barra de 3 zonas (< LOD | zona cinzenta | quantificável) com o marcador
    na posição real do teor estimado."""
    teor = float(dados["teor"])
    lod, loq = dados["lod"], dados["loq"]
    card.markdown(
        f"<div style='display:flex;align-items:baseline;gap:8px'>"
        f"<span style='font-size:1.7rem;font-weight:800'>{teor:.1f}%</span>"
        f"</div>", unsafe_allow_html=True)
    especie = dados.get("especie") or "—"
    card.markdown(
        f"<div style='font-size:.68rem;color:rgba(128,128,128,1);"
        f"margin-bottom:.35rem'>{T('species used')}: {especie}</div>",
        unsafe_allow_html=True)

    if lod is None or loq is None or not (np.isfinite(lod) and np.isfinite(loq)):
        card.info(T("No LOD/LOQ persisted for this species model, so no "
                    "decision range is drawn — a bar without limits would "
                    "suggest a confidence the data does not support."))
        return

    fim = max(float(loq) * 1.6, teor * 1.25, float(loq) + float(lod))
    p_lod, p_loq = float(lod) / fim * 100, float(loq) / fim * 100
    pos = min(max(teor / fim * 100, 0), 100)
    rotulos = {FAIXA_NAO_DETECTAVEL: T("< LOD"),
               FAIXA_ZONA_CINZENTA: T("grey zone"),
               FAIXA_QUANTIFICADO: T("quantifiable")}
    card.markdown(
        f"<div style='position:relative;display:flex;height:26px;"
        f"border-radius:6px;overflow:hidden;font-size:.62rem;font-weight:700;"
        f"color:#fff;text-align:center;line-height:26px'>"
        f"<div style='background:{tk['text_muted']};width:{p_lod}%'>"
        f"{rotulos[FAIXA_NAO_DETECTAVEL]}</div>"
        f"<div style='background:{tk['warn']};width:{p_loq - p_lod}%'>"
        f"{rotulos[FAIXA_ZONA_CINZENTA]}</div>"
        f"<div style='background:{tk['success']};width:{100 - p_loq}%'>"
        f"{rotulos[FAIXA_QUANTIFICADO]}</div>"
        f"<div style='position:absolute;left:{pos}%;top:-4px;width:3px;"
        f"height:34px;background:{tk['text']};border-radius:2px'></div>"
        f"</div>"
        f"<div style='display:flex;justify-content:space-between;"
        f"font-size:.66rem;color:rgba(128,128,128,1);margin-top:.2rem'>"
        f"<span>0</span><span>LOD {float(lod):.1f}</span>"
        f"<span>LOQ {float(loq):.1f}</span><span>{fim:.1f}</span></div>",
        unsafe_allow_html=True)
    cobertura = dados.get("cobertura")
    card.caption(T("Coverage of this combination: {c}.").format(
        c=cobertura or T("not declared")))
