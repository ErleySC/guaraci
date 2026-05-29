# -*- coding: utf-8 -*-
"""
============================================================================
 Plataforma Quimiometrica — Interface Streamlit v25 · 7 abas
============================================================================
Organizacao:
   1. Projeto    — identificacao e objetivo do estudo
   2. Dados      — entrada (FT-NIR .dx, CSV local, CSV upload)
   3. Pre-proc   — preset espectral + visualizacao antes/depois
   4. Modelo     — parametros avancados + execucao com progresso ao vivo
   5. Validacao  — figuras e metricas do ultimo processamento
   6. Predicao   — aplicar modelo salvo em amostras desconhecidas
   7. Relatorios — download ZIP, resumo, galeria de figuras, log

Motor: pineline_quimiometria_14.py (importado dinamicamente).
Nao e preciso editar codigo para usar: configure, rode, baixe.
============================================================================
"""
from __future__ import annotations

import io
import os
import re
import sys
import time
import tempfile
import zipfile
import threading
import contextlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────
# Pagina (deve ser o primeiro comando Streamlit)
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plataforma Quimiometrica",
    page_icon="🧪",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────
# Motor do pipeline
# ──────────────────────────────────────────────────────────────────────────
_AQUI = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_PATH = os.path.join(_AQUI, "pineline_quimiometria_14.py")
_CFG_PATH = os.path.join(_AQUI, "config.yaml")


@st.cache_resource(show_spinner="Carregando motor do pipeline...")
def _carregar_motor():
    if _AQUI not in sys.path:
        sys.path.insert(0, _AQUI)
    spec = importlib.util.spec_from_file_location("pq_engine", _PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Motor nao encontrado em {_PIPELINE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pq = _carregar_motor()

# ──────────────────────────────────────────────────────────────────────────
# Helpers de config (_CONFIG_SPEC como fonte unica de verdade)
# ──────────────────────────────────────────────────────────────────────────

def _spec_por_key() -> Dict:
    return {s["key"]: s for s in pq._CONFIG_SPEC}


def _widget_para_campo(s: Dict, valor_atual, prefixo: str = "w_"):
    """Renderiza UM widget conforme tipo do campo e retorna valor atual."""
    chave = prefixo + s["key"]
    rotulo = s["key"].replace("_", " ").capitalize()
    ajuda = s.get("desc", "")
    t = s["tipo"]
    if t == "bool":
        return st.checkbox(rotulo, value=bool(valor_atual), help=ajuda, key=chave)
    if t in ("choice", "preproc"):
        ops = list(s.get("opcoes") or [])
        idx = ops.index(valor_atual) if valor_atual in ops else 0
        return st.selectbox(rotulo, ops, index=idx, help=ajuda, key=chave)
    if t == "int":
        return st.number_input(rotulo, value=int(valor_atual), step=1,
                               help=ajuda, key=chave)
    if t == "float":
        return st.number_input(rotulo, value=float(valor_atual),
                               help=ajuda, key=chave, format="%.4f")
    if t == "list":
        txt = ", ".join(str(x) for x in (valor_atual or ()))
        return st.text_input(rotulo + " (separe por virgula)", value=txt,
                             help=ajuda, key=chave)
    return st.text_input(rotulo, value=str(valor_atual), help=ajuda, key=chave)


def _coletar_config(cfg_base, valores: Dict):
    """Aplica valores dos widgets numa copia da Config."""
    import copy
    cfg = copy.deepcopy(cfg_base)
    erros: List[str] = []
    for s in pq._CONFIG_SPEC:
        if s["key"] not in valores:
            continue
        try:
            setattr(cfg, s["attr"], pq._coagir_valor(s, valores[s["key"]]))
        except Exception as e:
            erros.append(f"{s['key']}: {e}")
    return cfg, erros


# ──────────────────────────────────────────────────────────────────────────
# Helpers de arquivo
# ──────────────────────────────────────────────────────────────────────────

def _zip_da_pasta(pasta: str) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for raiz, _dirs, arqs in os.walk(pasta):
            for a in arqs:
                cam = os.path.join(raiz, a)
                z.write(cam, os.path.relpath(cam, os.path.dirname(pasta)))
    buf.seek(0)
    return buf


def _listar_figuras(pasta: str) -> List[str]:
    imgs: List[str] = []
    for raiz, _dirs, arqs in os.walk(pasta):
        for a in sorted(arqs):
            if a.lower().endswith((".png", ".jpg", ".jpeg")):
                imgs.append(os.path.join(raiz, a))
    return sorted(imgs)


def _ler_resumo(pasta: str) -> Optional[str]:
    for candidato in [
        os.path.join(pasta, "logs", "resumo_modelo.txt"),
        os.path.join(pasta, "resumo_modelo.txt"),
    ]:
        if os.path.exists(candidato):
            with open(candidato, encoding="utf-8", errors="replace") as f:
                return f.read()
    return None


def _gerar_pdf_relatorio(pasta: str, projeto: Dict,
                          max_figuras: int = 14) -> io.BytesIO:
    """
    Gera relatorio PDF completo com fpdf2.
    Estrutura: Capa | Metricas | Figuras (2/pag) | Referencias.
    Retorna BytesIO pronto para st.download_button.
    """
    import re as _re
    import unicodedata
    from fpdf import FPDF

    def _a(txt: str) -> str:
        """Remove acentos para fontes Latin-1 do fpdf2."""
        return unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode("ascii")

    # ── Parse do resumo_modelo.txt ───────────────────────────────────────
    resumo_raw = _ler_resumo(pasta) or ""

    def _ex(padrao: str, default: str = "-") -> str:
        m = _re.search(padrao, resumo_raw, _re.IGNORECASE | _re.MULTILINE)
        return m.group(1).strip() if m else default

    metricas = {
        "Balanced Accuracy (CV)":   _ex(r"[Bb]alanced[_ ]?[Aa]ccuracy.*?[:=]\s*([\d.]+)"),
        "AUC macro OvR":            _ex(r"ROC AUC macro.*?[:=]\s*([\d.]+)"),
        "R2Y":                      _ex(r"\bR2Y\b.*?[:=]\s*([\d.]+)"),
        "Q2Y":                      _ex(r"\bQ2\b.*?[:=]\s*([\d.E+-]+)"),
        "R2X":                      _ex(r"\bR2X\b.*?[:=]\s*([\d.]+)"),
        "LVs otimos":               _ex(r"LVs?\s+otim[ao].*?[:=]\s*(\d+)"),
        "p-valor permutacao":       _ex(r"p.?value.*?[:=]\s*([\d.E+-]+)"),
        "Pre-processamento":        _ex(r"[Pp]re.?[Pp]rocessamento.*?[:=]\s*([A-Za-z0-9_+]+)"),
        "Hotelling T2 UCL (95%)":   _ex(r"[Hh]otelling.*?[:=]\s*([\d.]+)"),
        "Q-residual UCL (95%)":     _ex(r"Q.residual.*?[:=]\s*([\d.E+-]+)"),
        "n amostras (treino)":      _ex(r"[Nn]\s+treino.*?[:=]\s*(\d+)"),
        "n classes":                _ex(r"[Nn]\.?\s*[Cc]lasses.*?[:=]\s*(\d+)"),
    }

    imgs = _listar_figuras(pasta)[:max_figuras]

    # ── Classe PDF ──────────────────────────────────────────────────────
    class RelatorioPDF(FPDF):
        def header(self):
            if self.page_no() == 1:
                return
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(100, 100, 100)
            proj_nome = _a(projeto.get("nome", "Plataforma Quimiometrica"))
            self.cell(130, 6, proj_nome[:60], border=0, align="L")
            self.cell(0, 6, f"Gerado: {time.strftime('%d/%m/%Y')}",
                      border=0, align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(200, 200, 200)
            self.line(15, self.get_y(), 195, self.get_y())
            self.ln(2)
            self.set_text_color(0, 0, 0)

        def footer(self):
            if self.page_no() == 1:
                return
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(140, 140, 140)
            self.cell(0, 6, f"Pagina {self.page_no()} / {{nb}}", align="C")

    pdf = RelatorioPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(15, 20, 15)
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── CAPA ────────────────────────────────────────────────────────────
    pdf.add_page()
    # Faixa azul superior
    pdf.set_fill_color(30, 80, 140)
    pdf.rect(0, 0, 210, 55, style="F")
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(15, 12)
    pdf.cell(0, 12, "Plataforma Quimiometrica", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_xy(15, 30)
    pdf.cell(0, 8, "PLS-DA  |  PCA  |  OPLS-DA  |  DD-SIMCA", align="C")

    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(15, 65)
    pdf.set_font("Helvetica", "B", 17)
    pdf.multi_cell(180, 9,
                   _a(projeto.get("nome", "Relatorio de Analise Quimiometrica")),
                   align="C")

    pdf.ln(8)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_fill_color(240, 244, 250)
    campos_capa = [
        ("Autor(es)",      projeto.get("autor", "-")),
        ("Instituicao",    projeto.get("inst", "-")),
        ("Tipo de estudo", projeto.get("tipo", "-")),
        ("Data",           time.strftime("%d/%m/%Y")),
        ("Pasta",          os.path.basename(pasta)),
    ]
    for label, valor in campos_capa:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(55, 7, f"{label}:", border="B", fill=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(125, 7, _a(str(valor))[:80], border="B", fill=True,
                 new_x="LMARGIN", new_y="NEXT")

    obj = projeto.get("objetivo", "")
    if obj:
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Objetivo:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(180, 5, _a(obj))

    # Rodape da capa
    pdf.set_y(-30)
    pdf.set_fill_color(30, 80, 140)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "I", 8)
    pdf.rect(0, pdf.get_y(), 210, 30, style="F")
    pdf.set_xy(15, pdf.get_y() + 5)
    pdf.cell(0, 5,
             "Gerado por: Plataforma Quimiometrica v26  |  GEAAp / UFPA  |  Projeto PIBIC",
             align="C")

    # ── SECAO 1: METRICAS ───────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 80, 140)
    pdf.cell(0, 9, "1. Sumario Executivo", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(30, 80, 140)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(4)

    # Tabela de metricas
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(30, 80, 140)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(100, 7, "Metrica", border=1, fill=True)
    pdf.cell(75, 7, "Valor (validacao CV interna)", border=1, fill=True,
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    for i, (k, v) in enumerate(metricas.items()):
        if i % 2 == 0:
            pdf.set_fill_color(240, 245, 255)
        else:
            pdf.set_fill_color(252, 252, 252)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(100, 6, _a(k), border=1, fill=True)
        pdf.cell(75, 6, _a(str(v)), border=1, fill=True,
                 new_x="LMARGIN", new_y="NEXT")

    # Criterios de qualidade
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 80, 140)
    pdf.cell(0, 8, "Criterios de qualidade (PLS-DA — literatura):",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(0, 0, 0)
    criterios = [
        "R2Y > 0.60  e  Q2Y > 0.50 → modelo preditivo robusto [Eriksson 2008]",
        "Q2Y / R2Y > 0.70 → baixo risco de overfitting",
        "p-valor permutacao < 0.05 → resultado nao-aleatorio (Y-randomization)",
        "Intercept R2Y < 0.40 e intercept Q2Y < 0.05 → validacao geometrica [Wold 1984]",
        "BCa IC 95% de Bal.Acc nao inclui 1/n_classes → discriminacao significativa [Efron 1993]",
    ]
    for c in criterios:
        pdf.set_x(18)
        pdf.cell(4, 5, "-")
        pdf.multi_cell(173, 5, _a(c))

    # ── SECAO 2: FIGURAS ────────────────────────────────────────────────
    if imgs:
        # 2 figuras por pagina, empilhadas verticalmente
        fig_w = 175
        fig_h = 100   # altura maxima de cada figura
        cap_h = 6     # altura da legenda

        for par in range(0, len(imgs), 2):
            pdf.add_page()
            if par == 0:
                pdf.set_font("Helvetica", "B", 14)
                pdf.set_text_color(30, 80, 140)
                pdf.cell(0, 9, "2. Figuras", new_x="LMARGIN", new_y="NEXT")
                pdf.set_draw_color(30, 80, 140)
                pdf.set_line_width(0.5)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                pdf.set_line_width(0.2)
                pdf.ln(3)

            # Figura superior
            y_top = pdf.get_y()
            nome1 = os.path.splitext(os.path.basename(imgs[par]))[0]
            try:
                pdf.image(imgs[par], x=15, y=y_top, w=fig_w, h=fig_h)
            except Exception:
                pass
            pdf.set_xy(15, y_top + fig_h + 1)
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(fig_w, cap_h,
                     f"Fig. {par + 1}: {_a(nome1[:70])}",
                     align="C", new_x="LMARGIN", new_y="NEXT")

            # Figura inferior (se existir)
            if par + 1 < len(imgs):
                pdf.ln(3)
                y_bot = pdf.get_y()
                nome2 = os.path.splitext(os.path.basename(imgs[par + 1]))[0]
                try:
                    pdf.image(imgs[par + 1], x=15, y=y_bot, w=fig_w, h=fig_h)
                except Exception:
                    pass
                pdf.set_xy(15, y_bot + fig_h + 1)
                pdf.set_font("Helvetica", "I", 7)
                pdf.cell(fig_w, cap_h,
                         f"Fig. {par + 2}: {_a(nome2[:70])}",
                         align="C")

            pdf.set_text_color(0, 0, 0)

    # ── SECAO 3: REFERENCIAS ────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 80, 140)
    pdf.cell(0, 9, "3. Referencias", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(30, 80, 140)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(0, 0, 0)

    refs = [
        "Bylesjo, M. et al. (2006) OPLS discriminant analysis: combining the strengths of PLS-DA and SIMCA. J. Chemom. 20:341-351.",
        "Chong, I.G. & Jun, C.H. (2005) Performance of some variable selection methods when multicollinearity is present. Chemom. Intell. Lab. Syst. 78:103-112.",
        "Efron, B. & Tibshirani, R.J. (1993) An Introduction to the Bootstrap. CRC Press, Boca Raton.",
        "Eriksson, L. et al. (2008) CV-ANOVA for significance testing of PLS and OPLS models. J. Chemom. 22:594-600.",
        "Pomerantsev, A.L. & Rodionova, O.Y. (2020) Concept and role of extreme objects in PCA/SIMCA. J. Chemom. 34:e3227.",
        "Rajalahti, T. et al. (2009) Discriminating variable test and selectivity ratio plot: quantitative tools for interpretation and variable (biomarker) selection in complex spectral or chromatographic profiles. Anal. Chem. 81:2581-2590.",
        "Rinnan, A. et al. (2009) Review of the most common pre-processing techniques for near-infrared spectra. TrAC Trends Anal. Chem. 28:1201-1222.",
        "Tracy, N.D. et al. (1992) Multivariate control charts for individual observations. J. Qual. Tech. 24:88-95.",
        "Wold, S. (1978) Cross-validatory estimation of the number of components in factor and principal components models. Technometrics 20:397-405.",
    ]
    for i, ref in enumerate(refs, 1):
        pdf.multi_cell(0, 5, f"[{i}] {_a(ref)}")
        pdf.ln(1)

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5,
                   "Relatorio gerado automaticamente pela Plataforma Quimiometrica v26. "
                   "Motor: pineline_quimiometria_14.py | Interface: app_quimiometria.py. "
                   "GEAAp/UFPA — Projeto PIBIC.")

    buf = io.BytesIO()
    buf.write(pdf.output())
    buf.seek(0)
    return buf


def _gerar_word_relatorio(pasta: str, projeto: Dict,
                           max_figuras: int = 14) -> io.BytesIO:
    """
    Gera relatorio Word editavel (.docx) com python-docx.
    Mesma estrutura do PDF: capa, metricas, figuras, referencias.
    """
    import re as _re
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    resumo_raw = _ler_resumo(pasta) or ""

    def _ex(padrao: str, default: str = "-") -> str:
        m = _re.search(padrao, resumo_raw, _re.IGNORECASE | _re.MULTILINE)
        return m.group(1).strip() if m else default

    metricas = {
        "Balanced Accuracy (CV)":  _ex(r"[Bb]alanced[_ ]?[Aa]ccuracy.*?[:=]\s*([\d.]+)"),
        "AUC macro OvR":           _ex(r"ROC AUC macro.*?[:=]\s*([\d.]+)"),
        "R2Y":                     _ex(r"\bR2Y\b.*?[:=]\s*([\d.]+)"),
        "Q2Y":                     _ex(r"\bQ2\b.*?[:=]\s*([\d.E+-]+)"),
        "R2X":                     _ex(r"\bR2X\b.*?[:=]\s*([\d.]+)"),
        "LVs otimos":              _ex(r"LVs?\s+otim[ao].*?[:=]\s*(\d+)"),
        "p-valor permutacao":      _ex(r"p.?value.*?[:=]\s*([\d.E+-]+)"),
        "Pre-processamento":       _ex(r"[Pp]re.?[Pp]rocessamento.*?[:=]\s*([A-Za-z0-9_+]+)"),
        "Hotelling T2 UCL (95%)":  _ex(r"[Hh]otelling.*?[:=]\s*([\d.]+)"),
        "Q-residual UCL (95%)":    _ex(r"Q.residual.*?[:=]\s*([\d.E+-]+)"),
        "n amostras (treino)":     _ex(r"[Nn]\s+treino.*?[:=]\s*(\d+)"),
        "n classes":               _ex(r"[Nn]\.?\s*[Cc]lasses.*?[:=]\s*(\d+)"),
    }
    imgs = _listar_figuras(pasta)[:max_figuras]

    doc = Document()

    # ── Estilos de cabecalho ──
    for i, (level, size) in enumerate([(0, 20), (1, 16), (2, 13)]):
        style = doc.styles[f"Heading {i + 1}"]
        style.font.size = Pt(size)           # type: ignore[union-attr]
        style.font.color.rgb = RGBColor(30, 80, 140)  # type: ignore[union-attr]
        style.font.bold = True               # type: ignore[union-attr]

    # ── CAPA ──────────────────────────────────────────────────────────────
    doc.add_heading("Plataforma Quimiometrica", 0)
    doc.add_heading(projeto.get("nome", "Relatorio de Analise Quimiometrica"), 1)

    t_capa = doc.add_table(rows=5, cols=2)
    t_capa.style = "Table Grid"
    campos_capa = [
        ("Autor(es)",      projeto.get("autor", "-")),
        ("Instituicao",    projeto.get("inst", "GEAAp / UFPA")),
        ("Tipo de estudo", projeto.get("tipo", "-")),
        ("Data",           time.strftime("%d/%m/%Y %H:%M")),
        ("Pasta",          os.path.basename(pasta)),
    ]
    for i, (label, valor) in enumerate(campos_capa):
        c0 = t_capa.cell(i, 0)
        c1 = t_capa.cell(i, 1)
        c0.text = label
        c1.text = str(valor)
        c0.paragraphs[0].runs[0].bold = True

    obj = projeto.get("objetivo", "")
    if obj:
        doc.add_heading("Objetivo", 2)
        doc.add_paragraph(obj)

    doc.add_page_break()

    # ── SECAO 1: METRICAS ─────────────────────────────────────────────────
    doc.add_heading("1. Sumario Executivo — Metricas do Modelo", 1)

    t_met = doc.add_table(rows=1 + len(metricas), cols=2)
    t_met.style = "Table Grid"
    hdr = t_met.rows[0].cells
    hdr[0].text = "Metrica"
    hdr[1].text = "Valor (validacao CV interna)"
    for cell in hdr:
        run = cell.paragraphs[0].runs
        if run:
            run[0].bold = True
        cell.paragraphs[0].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for i, (k, v) in enumerate(metricas.items(), 1):
        t_met.cell(i, 0).text = k
        t_met.cell(i, 1).text = str(v)

    doc.add_heading("Criterios de qualidade", 2)
    criterios = [
        "R2Y > 0.60 e Q2Y > 0.50: modelo preditivo robusto [Eriksson 2008]",
        "Q2Y / R2Y > 0.70: baixo risco de overfitting",
        "p-valor permutacao < 0.05: resultado nao-aleatorio (Y-randomization)",
        "Intercept R2Y < 0.40 e intercept Q2Y < 0.05 [Wold 1984]",
        "BCa IC 95% de Bal.Acc nao inclui 1/n_classes [Efron 1993]",
    ]
    for c in criterios:
        doc.add_paragraph(c, style="List Bullet")

    # ── SECAO 2: FIGURAS ──────────────────────────────────────────────────
    if imgs:
        doc.add_page_break()
        doc.add_heading("2. Figuras", 1)
        for i, img in enumerate(imgs, 1):
            nome_fig = os.path.splitext(os.path.basename(img))[0]
            try:
                doc.add_picture(img, width=Inches(5.5))
                p = doc.paragraphs[-1]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                leg = doc.add_paragraph(f"Fig. {i}: {nome_fig}")
                leg.alignment = WD_ALIGN_PARAGRAPH.CENTER
                if leg.runs:
                    leg.runs[0].italic = True
                    leg.runs[0].font.size = Pt(9)
            except Exception:
                doc.add_paragraph(f"[Fig. {i}: {nome_fig} — imagem nao disponivel]")
            if i % 2 == 0 and i < len(imgs):
                doc.add_page_break()

    # ── SECAO 3: REFERENCIAS ─────────────────────────────────────────────
    doc.add_page_break()
    doc.add_heading("3. Referencias", 1)
    refs = [
        "Bylesjo, M. et al. (2006) OPLS discriminant analysis. J. Chemom. 20:341-351.",
        "Chong, I.G. & Jun, C.H. (2005) VIP scores. Chemom. Intell. Lab. Syst. 78:103-112.",
        "Efron, B. & Tibshirani, R.J. (1993) An Introduction to the Bootstrap. CRC Press.",
        "Eriksson, L. et al. (2008) CV-ANOVA for PLS/OPLS significance. J. Chemom. 22:594-600.",
        "Pomerantsev, A.L. & Rodionova, O.Y. (2020) Cooman's Plot. J. Chemom. 34:e3227.",
        "Rajalahti, T. et al. (2009) Selectivity Ratio. Anal. Chem. 81:2581-2590.",
        "Rinnan, A. et al. (2009) NIR pre-processing review. TrAC 28:1201-1222.",
        "Tracy, N.D. et al. (1992) Multivariate control charts. J. Qual. Tech. 24:88-95.",
        "Wold, S. (1978) Cross-validatory estimation. Technometrics 20:397-405.",
    ]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph(f"[{i}] {ref}")
        p.paragraph_format.space_after = Pt(4)

    doc.add_paragraph()
    nota = doc.add_paragraph(
        "Relatorio gerado pela Plataforma Quimiometrica v26. "
        "Motor: pineline_quimiometria_14.py | GEAAp/UFPA — Projeto PIBIC."
    )
    if nota.runs:
        nota.runs[0].italic = True
        nota.runs[0].font.size = Pt(8)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


def _gerar_excel_relatorio(pasta: str) -> io.BytesIO:
    """
    Gera relatorio Excel com 4 abas via openpyxl:
      - Metricas: metricas extraidas do resumo
      - Identificadores: amostras com T2, Q, classe (CSV do pipeline)
      - VIP_Selecao: escores VIP/SR (CSV do pipeline, se existir)
      - Resumo_Bruto: texto completo do resumo_modelo.txt
    """
    import re as _re
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    resumo_raw = _ler_resumo(pasta) or ""

    def _ex(padrao: str, default: str = "-") -> str:
        m = _re.search(padrao, resumo_raw, _re.IGNORECASE | _re.MULTILINE)
        return m.group(1).strip() if m else default

    metricas_dict = {
        "Balanced Accuracy (CV)":  _ex(r"[Bb]alanced[_ ]?[Aa]ccuracy.*?[:=]\s*([\d.]+)"),
        "AUC macro OvR":           _ex(r"ROC AUC macro.*?[:=]\s*([\d.]+)"),
        "R2Y":                     _ex(r"\bR2Y\b.*?[:=]\s*([\d.]+)"),
        "Q2Y":                     _ex(r"\bQ2\b.*?[:=]\s*([\d.E+-]+)"),
        "R2X":                     _ex(r"\bR2X\b.*?[:=]\s*([\d.]+)"),
        "LVs otimos":              _ex(r"LVs?\s+otim[ao].*?[:=]\s*(\d+)"),
        "p-valor permutacao":      _ex(r"p.?value.*?[:=]\s*([\d.E+-]+)"),
        "Pre-processamento":       _ex(r"[Pp]re.?[Pp]rocessamento.*?[:=]\s*([A-Za-z0-9_+]+)"),
        "Hotelling T2 UCL (95%)":  _ex(r"[Hh]otelling.*?[:=]\s*([\d.]+)"),
        "Q-residual UCL (95%)":    _ex(r"Q.residual.*?[:=]\s*([\d.E+-]+)"),
        "n amostras (treino)":     _ex(r"[Nn]\s+treino.*?[:=]\s*(\d+)"),
        "n classes":               _ex(r"[Nn]\.?\s*[Cc]lasses.*?[:=]\s*(\d+)"),
    }

    _AZUL      = PatternFill("solid", fgColor="1E508C")
    _AZUL_CLAR = PatternFill("solid", fgColor="DCE6F5")
    _BRANCO    = PatternFill("solid", fgColor="FFFFFF")
    _HDR_FONT  = Font(bold=True, color="FFFFFF", size=10)
    _BODY_FONT = Font(size=9)
    _THIN      = Side(style="thin", color="AAAAAA")
    _BORDER    = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)

    def _cabecalho(ws, colunas: List[str]):
        for j, col in enumerate(colunas, 1):
            cell = ws.cell(1, j, col)
            cell.font = _HDR_FONT
            cell.fill = _AZUL
            cell.alignment = Alignment(horizontal="center")
            cell.border = _BORDER

    def _preencher_df(ws, df: pd.DataFrame):
        _cabecalho(ws, list(df.columns))
        for i, row in enumerate(df.itertuples(index=False), 2):
            fill = _AZUL_CLAR if i % 2 == 0 else _BRANCO
            for j, val in enumerate(row, 1):
                c = ws.cell(i, j, val)
                c.font = _BODY_FONT
                c.fill = fill
                c.border = _BORDER

    def _preencher_df_ws(ws, df: pd.DataFrame, row_start: int = 1):
        """Variante de _preencher_df com row_start configuravel."""
        cols = list(df.columns)
        for j, col in enumerate(cols, 1):
            cell = ws.cell(row_start, j, col)
            cell.font = _HDR_FONT; cell.fill = _AZUL
            cell.alignment = Alignment(horizontal="center"); cell.border = _BORDER
        for i, row in enumerate(df.itertuples(index=False), row_start + 1):
            fill = _AZUL_CLAR if i % 2 == 0 else _BRANCO
            for j, val in enumerate(row, 1):
                c = ws.cell(i, j, val)
                c.font = _BODY_FONT; c.fill = fill; c.border = _BORDER

    def _auto_width(ws):
        for col in ws.columns:
            max_len = max((len(str(c.value)) for c in col if c.value), default=8)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 3, 40)

    wb = openpyxl.Workbook()

    # ── ABA 1: Metricas ───────────────────────────────────────────────────
    ws1 = wb.active
    assert ws1 is not None
    ws1.title = "Metricas"
    _cabecalho(ws1, ["Metrica", "Valor (CV interno)"])
    for i, (k, v) in enumerate(metricas_dict.items(), 2):
        fill = _AZUL_CLAR if i % 2 == 0 else _BRANCO
        for j, val in enumerate([k, v], 1):
            c = ws1.cell(i, j, val)
            c.font = _BODY_FONT
            c.fill = fill
            c.border = _BORDER
    _auto_width(ws1)

    # ── ABA 2: Identificadores ────────────────────────────────────────────
    ws2 = wb.create_sheet("Identificadores")
    id_csv = os.path.join(pasta, "dados", "amostras_identificadores.csv")
    if os.path.exists(id_csv):
        try:
            df_id = pd.read_csv(id_csv, sep=";", decimal=",")
            _preencher_df(ws2, df_id)
            _auto_width(ws2)
        except Exception:
            ws2.cell(1, 1, "Erro ao ler amostras_identificadores.csv")
    else:
        ws2.cell(1, 1, "Arquivo nao encontrado (execute a Etapa 5 do pipeline).")

    # ── ABA 3: VIP_Selecao ────────────────────────────────────────────────
    ws3 = wb.create_sheet("VIP_Selecao")
    vip_csv = os.path.join(pasta, "dados", "etapa4_selecao_variaveis.csv")
    if os.path.exists(vip_csv):
        try:
            df_vip = pd.read_csv(vip_csv, sep=";", decimal=",")
            _preencher_df(ws3, df_vip)
            _auto_width(ws3)
        except Exception:
            ws3.cell(1, 1, "Erro ao ler etapa4_selecao_variaveis.csv")
    else:
        ws3.cell(1, 1, "Etapa 4 (selecao de variaveis) nao foi executada.")

    # ── ABA 4: Resumo_Bruto ───────────────────────────────────────────────
    ws4 = wb.create_sheet("Resumo_Bruto")
    ws4.cell(1, 1, "resumo_modelo.txt — conteudo completo").font = Font(bold=True)
    for i, linha in enumerate(resumo_raw.splitlines(), 2):
        ws4.cell(i, 1, linha).font = Font(name="Courier New", size=9)
    ws4.column_dimensions["A"].width = 80

    # ── ABA 5: Benchmark (se existir) ─────────────────────────────────────
    bench_csv = os.path.join(pasta, "dados", "benchmark_classificadores.csv")
    mc_csv    = os.path.join(pasta, "dados", "monte_carlo_cv.csv")
    if os.path.exists(bench_csv) or os.path.exists(mc_csv):
        ws5 = wb.create_sheet("Benchmark")
        row_cursor = 1
        if os.path.exists(bench_csv):
            try:
                df_bench = pd.read_csv(bench_csv, sep=";", decimal=",")
                ws5.cell(row_cursor, 1,
                         "Auto-Benchmark — Bal.Acc por classificador (GroupKFold)").font = Font(bold=True)
                row_cursor += 1
                _preencher_df_ws(ws5, df_bench, row_start=row_cursor)
                row_cursor += len(df_bench) + 3
                _auto_width(ws5)
            except Exception:
                ws5.cell(row_cursor, 1, "Erro ao ler benchmark_classificadores.csv")
                row_cursor += 2
        if os.path.exists(mc_csv):
            try:
                df_mc = pd.read_csv(mc_csv, sep=";", decimal=",")
                ws5.cell(row_cursor, 1,
                         "Monte Carlo CV — IC95% por percentil").font = Font(bold=True)
                row_cursor += 1
                _preencher_df_ws(ws5, df_mc, row_start=row_cursor)
                _auto_width(ws5)
            except Exception:
                ws5.cell(row_cursor, 1, "Erro ao ler monte_carlo_cv.csv")

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


def _gerar_latex_template(pasta: str, projeto: Dict) -> bytes:
    """
    Gera template LaTeX pronto para periódicos (Talanta, Food Chemistry,
    Journal of Chemometrics). Inclui metricas auto-preenchidas, blocos
    \\includegraphics das figuras e bibliografia completa.
    Retorna bytes UTF-8 (arquivo .tex).
    """
    import re as _re

    resumo_raw = _ler_resumo(pasta) or ""

    def _ex(padrao: str, default: str = "-") -> str:
        m = _re.search(padrao, resumo_raw, _re.IGNORECASE | _re.MULTILINE)
        return m.group(1).strip() if m else default

    def _esc(txt: str) -> str:
        """Escapa caracteres especiais do LaTeX."""
        for old, new in [
            ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
            ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
            ("{", r"\{"), ("}", r"\}"),
            ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
        ]:
            txt = txt.replace(old, new)
        return txt

    met = {
        "bal_acc":   _ex(r"[Bb]alanced.*?[:=]\s*([\d.]+)"),
        "auc":       _ex(r"ROC AUC macro.*?[:=]\s*([\d.]+)"),
        "r2y":       _ex(r"\bR2Y\b.*?[:=]\s*([\d.]+)"),
        "q2y":       _ex(r"\bQ2\b.*?[:=]\s*([\d.E+-]+)"),
        "r2x":       _ex(r"\bR2X\b.*?[:=]\s*([\d.]+)"),
        "lvs":       _ex(r"LVs?\s+otim[ao].*?[:=]\s*(\d+)"),
        "perm_p":    _ex(r"p.?value.*?[:=]\s*([\d.E+-]+)"),
        "preproc":   _ex(r"[Pp]re.?[Pp]rocessamento.*?[:=]\s*([A-Za-z0-9_+]+)"),
        "n_train":   _ex(r"[Nn]\s+treino.*?[:=]\s*(\d+)"),
        "n_classes": _ex(r"[Nn]\.?\s*[Cc]lasses.*?[:=]\s*(\d+)"),
    }

    imgs = _listar_figuras(pasta)[:8]
    nome_proj = _esc(projeto.get("nome", "Analise Quimiometrica por FT-NIR"))
    autor     = _esc(projeto.get("autor", "Sobrenome, N."))
    inst      = _esc(projeto.get("inst", "GEAAp, Universidade Federal do Para"))

    # Tabela de metricas
    linhas_met = [
        r"        \textbf{Metrica} & \textbf{Valor (CV interno)} \\",
        r"        \midrule",
        f"        Balanced Accuracy & {met['bal_acc']} \\\\",
        f"        AUC macro (OvR)   & {met['auc']} \\\\",
        f"        $R^2Y$            & {met['r2y']} \\\\",
        f"        $Q^2Y$            & {met['q2y']} \\\\",
        f"        $R^2X$            & {met['r2x']} \\\\",
        f"        LVs otimos        & {met['lvs']} \\\\",
        f"        $p$ (permutacao)  & {met['perm_p']} \\\\",
        f"        Pre-processamento & {_esc(met['preproc'])} \\\\",
        f"        $n$ treino        & {met['n_train']} \\\\",
        f"        $n$ classes       & {met['n_classes']} \\\\",
    ]
    tabela = "\n".join(linhas_met)

    # Blocos de figuras
    blocos_fig = []
    for i, img in enumerate(imgs, 1):
        nome_f = os.path.splitext(os.path.basename(img))[0]
        label  = nome_f.replace(" ", "_").replace("-", "_")
        # LaTeX prefere barras normais
        img_tex = img.replace("\\", "/")
        blocos_fig.append(f"""
\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=0.85\\linewidth]{{{img_tex}}}
    \\caption{{{_esc(nome_f)}. % TODO: adicione interpretacao quimica.}}
    \\label{{fig:{label}}}
\\end{{figure}}""")
    figs_block = "\n".join(blocos_fig)

    tex = f"""% ================================================================
% Template LaTeX — Plataforma Quimiometrica v26
% Compativel com: Talanta, Food Chemistry, J. Chemometrics (Elsevier)
% Gerado em: {time.strftime('%d/%m/%Y %H:%M')}
% ================================================================

\\documentclass[12pt,a4paper]{{article}}

%% Pacotes
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage[english,brazil]{{babel}}
\\usepackage{{amsmath, amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{hyperref}}[colorlinks=true,citecolor=blue,linkcolor=blue]
\\usepackage{{geometry}}
\\usepackage{{caption}}
\\usepackage{{float}}
\\usepackage{{siunitx}}
\\usepackage{{natbib}}

\\geometry{{a4paper, left=2.5cm, right=2.5cm, top=3cm, bottom=3cm}}
\\captionsetup{{font=small, labelfont=bf}}
\\setlength{{\\parindent}}{{0.5cm}}

%% Metadados
\\title{{{nome_proj}}}
\\author{{{autor} \\\\
         \\small {inst}}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

%% ── Resumo ─────────────────────────────────────────────────────────────
\\begin{{abstract}}
A autenticidade de oleos vegetais amazonicos foi investigada por espectroscopia
FT-NIR combinada a metodos quimiometricos (PLS-DA, PCA, OPLS-DA e DD-SIMCA).
O modelo PLS-DA com {_esc(met['lvs'])} variaveis latentes e pre-processamento
{_esc(met['preproc'])} apresentou acuracia balanceada de {met['bal_acc']}
($R^2Y = {met['r2y']}$; $Q^2Y = {met['q2y']}$; $p = {met['perm_p']}$,
Y-randomization), evidenciando capacidade discriminatoria significativa.
% TODO: limite a 250 palavras para Talanta / 200 para Food Chemistry.
\\end{{abstract}}

\\textbf{{Keywords:}} FT-NIR; PLS-DA; quimiometria; autenticacao; oleos vegetais.

%% ── Introducao ──────────────────────────────────────────────────────────
\\section{{Introducao}}
% TODO: contextualize, justifique o metodo e enuncie o objetivo em ~3 paragrafos.
A adulteracao de oleos vegetais e um problema de seguranca alimentar de escala
global \\citep{{rinnan2009}}.
A espectroscopia FT-NIR aliada a quimiometria tem demonstrado potencial como
tecnica rapida, nao-destrutiva e de baixo custo para autenticacao de oleos
\\citep{{bylesjo2006, eriksson2008}}.

%% ── Material e Metodos ──────────────────────────────────────────────────
\\section{{Material e Metodos}}

\\subsection{{Amostras e aquisicao espectral}}
% TODO: descreva numero de amostras, equipamento (ABB MB3600 ou similar),
% faixa espectral, resolucao, numero de varreduras, temperatura.
Os espectros FT-NIR foram adquiridos na faixa de \\SIrange{{4000}}{{10000}}{{\\per\\centi\\meter}}.

\\subsection{{Pre-processamento espectral}}
O pre-processamento {_esc(met['preproc'])} foi selecionado com base na comparacao
de pipelines (balanced accuracy na validacao CV) conforme \\citet{{rinnan2009}}.

\\subsection{{Modelagem PLS-DA}}
O modelo PLS-DA foi calibrado com {met['lvs']} variaveis latentes, selecionados
por validacao cruzada grupal (GroupKFold, agrupando triplicatas tecnicas para
prevenir vazamento de dados) \\citep{{chong2005}}.

\\subsection{{Validacao estatistica}}
A robustez foi avaliada por:
\\begin{{itemize}}
    \\item \\textbf{{Y-randomization}} (200 permutacoes) \\citep{{eriksson2008}};
    \\item \\textbf{{Teste de Wold}} (interceptos $R^2Y < 0.40$ e $Q^2Y < 0.05$);
    \\item \\textbf{{CV-ANOVA}} de Eriksson \\citep{{eriksson2008}};
    \\item \\textbf{{Bootstrap BCa}} 95\\% para a acuracia balanceada \\citep{{efron1993}}.
\\end{{itemize}}

%% ── Resultados e Discussao ──────────────────────────────────────────────
\\section{{Resultados e Discussao}}

\\subsection{{Desempenho do modelo}}
As metricas de desempenho sao apresentadas na Tabela~\\ref{{tab:metricas}}.

\\begin{{table}}[htbp]
    \\centering
    \\caption{{Metricas de desempenho do modelo PLS-DA.}}
    \\label{{tab:metricas}}
    \\begin{{tabular}}{{ll}}
        \\toprule
{tabela}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}

\\subsection{{Figuras}}
% Figuras geradas pelo pipeline. Insira interpretacao quimica em cada legenda.

{figs_block}

%% ── Conclusao ───────────────────────────────────────────────────────────
\\section{{Conclusao}}
% TODO: sintetize resultados e implicacoes em 1-2 paragrafos.
O modelo PLS-DA com pre-processamento {_esc(met['preproc'])} apresentou
desempenho satisfatorio ($Q^2Y = {met['q2y']}$; $p = {met['perm_p']}$),
demonstrando viabilidade da espectroscopia FT-NIR para autenticacao rapida
de oleos vegetais amazonicos.

%% ── Agradecimentos ──────────────────────────────────────────────────────
\\section*{{Agradecimentos}}
% TODO: CNPq, CAPES, PIBIC/UFPA, laboratorio.
Ao GEAAp/UFPA e ao CNPq pelo suporte financeiro (Projeto PIBIC).

%% ── Referencias ─────────────────────────────────────────────────────────
\\bibliographystyle{{elsarticle-num}}  %% Elsevier (Talanta, Food Chemistry)
%% \\bibliographystyle{{apalike}}       %% Alternativa APA

\\begin{{thebibliography}}{{99}}

\\bibitem{{bylesjo2006}}
Bylesjo,~M. et~al. (2006).
OPLS discriminant analysis: combining the strengths of PLS-DA and SIMCA.
\\textit{{Journal of Chemometrics}}, 20(8--10), 341--351.

\\bibitem{{chong2005}}
Chong,~I.G. \\& Jun,~C.H. (2005).
Performance of some variable selection methods when multicollinearity is present.
\\textit{{Chemometrics and Intelligent Laboratory Systems}}, 78(1--2), 103--112.

\\bibitem{{efron1993}}
Efron,~B. \\& Tibshirani,~R.J. (1993).
\\textit{{An Introduction to the Bootstrap}}. CRC Press.

\\bibitem{{eriksson2008}}
Eriksson,~L., Trygg,~J. \\& Wold,~S. (2008).
CV-ANOVA for significance testing of PLS and OPLS models.
\\textit{{Journal of Chemometrics}}, 22(11--12), 594--600.

\\bibitem{{pomerantsev2020}}
Pomerantsev,~A.L. \\& Rodionova,~O.Y. (2020).
Concept and role of extreme objects in PCA/SIMCA.
\\textit{{Journal of Chemometrics}}, 34(3), e3227.

\\bibitem{{rajalahti2009}}
Rajalahti,~T. et~al. (2009).
Discriminating variable test and selectivity ratio plot.
\\textit{{Analytical Chemistry}}, 81(7), 2581--2590.

\\bibitem{{rinnan2009}}
Rinnan,~\\AA., van~den~Berg,~F. \\& Engelsen,~S.B. (2009).
Review of the most common pre-processing techniques for near-infrared spectra.
\\textit{{TrAC Trends in Analytical Chemistry}}, 28(10), 1201--1222.

\\bibitem{{tracy1992}}
Tracy,~N.D., Young,~J.C. \\& Mason,~R.L. (1992).
Multivariate control charts for individual observations.
\\textit{{Journal of Quality Technology}}, 24(2), 88--95.

\\bibitem{{wold1978}}
Wold,~S. (1978).
Cross-validatory estimation of the number of components.
\\textit{{Technometrics}}, 20(4), 397--405.

\\end{{thebibliography}}

\\end{{document}}
"""
    return tex.encode("utf-8")


# ──────────────────────────────────────────────────────────────────────────
# Execucao com feedback ao vivo (thread de fundo + barra de progresso)
# ──────────────────────────────────────────────────────────────────────────

class _LogThreadSafe:
    """Captura stdout/stderr do pipeline numa lista protegida por lock."""
    def __init__(self, tee=None):
        self._buf: List[str] = []
        self._lock = threading.Lock()
        self._tee = tee

    def write(self, s: str):
        with self._lock:
            self._buf.append(s)
        if self._tee is not None:
            try: self._tee.write(s)
            except Exception: pass
        return len(s)

    def flush(self):
        if self._tee is not None:
            try: self._tee.flush()
            except Exception: pass

    def text(self) -> str:
        with self._lock:
            return "".join(self._buf)


_RE_ETAPA = re.compile(r"\[(\d+)[a-z]?/7\]")
_ETAPA_NOMES = {
    0: "Validando a entrada",
    1: "Pre-processamento espectral",
    2: "Selecao de variaveis latentes (LVs)",
    3: "PCA exploratoria",
    4: "Testes de validacao (permutacao / Wold / CV-ANOVA)",
    5: "Metricas finais + IC bootstrap",
    6: "Figuras, DD-SIMCA, OPLS-DA, holdout",
    7: "Regressao / finalizacao e modelo salvo",
}
# Sub-etapas apos o passo 7 (benchmark / MC CV)
_ETAPA_SUBSTEP = {
    "[7b/7]": "Auto-Benchmark (SVM / RF / XGBoost vs PLS-DA)...",
    "[7c/7]": "Monte Carlo CV (IC95% por percentil)...",
}


def _progresso_do_log(txt: str):
    achados = _RE_ETAPA.findall(txt)
    if not achados:
        return 0.0, "Iniciando..."
    n = max(int(a) for a in achados)
    nome = _ETAPA_NOMES.get(n, f"Etapa {n}/7")
    # Sub-etapas pesadas: exibe nome especifico para benchmark e MC CV
    if n >= 7:
        for tag, descricao in _ETAPA_SUBSTEP.items():
            if tag in txt:
                nome = descricao
                break
    return min(0.99, n / 7.0), nome


def _fmt_tempo(seg) -> str:
    if seg is None:
        return "—"
    try:
        seg = float(seg)
    except (TypeError, ValueError):
        return "—"
    if seg != seg or seg < 0:
        return "0s"
    seg = int(round(seg))
    d, r = divmod(seg, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    if d: return f"{d}d {h}h"
    if h: return f"{h}h {m:02d}min"
    if m: return f"{m}min {s:02d}s"
    return f"{s}s"


def _ram_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _rodar_worker(cfg, logger: _LogThreadSafe, estado: Dict):
    try:
        with contextlib.redirect_stdout(logger), \
             contextlib.redirect_stderr(logger):
            pq.executar(cfg)
        estado["pasta"] = getattr(cfg, "pasta_saida", None)
        estado["erro"] = None
    except Exception:
        import traceback
        estado["erro"] = traceback.format_exc()
    finally:
        estado["fim"] = True


# ──────────────────────────────────────────────────────────────────────────
# Preview de espectros (cache por caminho)
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=120)
def _preview_espectros_dx(pasta: str, wn_min: float, wn_max: float,
                           max_por_classe: int = 5):
    """Carrega ate max_por_classe amostras por subpasta para visualizacao."""
    try:
        subpastas = sorted(
            p for p in Path(pasta).iterdir()
            if p.is_dir() and list(p.glob("*.dx"))
        )
        if not subpastas:
            # pasta plana com .dx
            subpastas = [Path(pasta)]
        wn_ref, specs, labs = None, [], []
        from scipy.interpolate import interp1d
        for sp in subpastas:
            arqs = sorted(sp.glob("*.dx"))[:max_por_classe]
            for arq in arqs:
                try:
                    wn_a, sp_a = pq.parse_dx(str(arq))
                    mask = (wn_a >= wn_min) & (wn_a <= wn_max)
                    wn_a, sp_a = wn_a[mask], sp_a[mask]
                    if wn_ref is None:
                        wn_ref = wn_a
                    else:
                        f = interp1d(wn_a, sp_a, kind="linear",
                                     bounds_error=False, fill_value="extrapolate")  # type: ignore
                        sp_a = f(wn_ref)
                    specs.append(sp_a)
                    labs.append(sp.name)
                except Exception:
                    continue
        if not specs or wn_ref is None:
            return None, None, None
        return wn_ref, np.array(specs), np.array(labs)
    except Exception:
        return None, None, None


@st.cache_data(show_spinner=False, ttl=120)
def _preview_espectros_csv(caminho: str, col_cls: str,
                            wn_min: float, wn_max: float,
                            max_n: int = 50):
    """Carrega ate max_n linhas de CSV para visualizacao."""
    try:
        df = pd.read_csv(caminho, sep=None, engine="python", nrows=max_n)
        meta = {col_cls}
        num_cols = [c for c in df.columns if c not in meta]
        try:
            wn = np.array([float(c) for c in num_cols])
        except ValueError:
            return None, None, None
        mask = (wn >= wn_min) & (wn <= wn_max)
        X = df[num_cols].values[:, mask].astype(float)
        labs = df[col_cls].astype(str).values if col_cls in df.columns else \
               np.array(["?" ] * len(df))
        return wn[mask], X, labs
    except Exception:
        return None, None, None


def _plot_espectros_media(wn: np.ndarray, X: np.ndarray,
                           rotulos: np.ndarray, titulo: str = ""):  # -> matplotlib.figure.Figure
    """Plota media ± std por classe."""
    classes = np.unique(np.asarray(rotulos))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    for i, cls in enumerate(classes[:10]):
        cor = cmap(i / 10)
        mask = rotulos == cls
        med = X[mask].mean(axis=0)
        std = X[mask].std(axis=0)
        ax.plot(wn, med, color=cor, lw=1.3, label=f"{cls} (n={mask.sum()})")
        ax.fill_between(wn, med - std, med + std, color=cor, alpha=0.15)
    if len(wn) > 1 and wn[0] > wn[-1]:
        ax.invert_xaxis()
    ax.set_xlabel("Numero de onda (cm$^{-1}$)")
    ax.set_ylabel("Absorbancia")
    if titulo:
        ax.set_title(titulo, fontsize=9)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(color="0.93", lw=0.5)
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Predicao de amostras desconhecidas
# ──────────────────────────────────────────────────────────────────────────

def _predizer(pkg: Dict, X_new_raw: np.ndarray,
              wn_new: Optional[np.ndarray]) -> pd.DataFrame:
    """
    Aplica o pacote de modelo salvo em novos espectros.
    Interpola para o eixo de referencia do treino, aplica pre-processador,
    calcula classe predita (softmax-normalizado), T2 e Q residuais.
    Retorna DataFrame com diagnosticos por amostra.
    """
    from scipy.interpolate import interp1d

    preproc = pkg["preprocessador"]
    pls     = pkg["pls_final"]
    lb      = pkg["label_binarizer"]
    wn_train = np.asarray(pkg["wavenumbers"], dtype=float)
    if wn_new is None:
        raise ValueError("wn_new nao pode ser None")
    wn_min   = float(pkg.get("wn_min", wn_train.min()))
    wn_max   = float(pkg.get("wn_max", wn_train.max()))

    # Eixo de referencia do treino (faixa usada no treino)
    mask_ref = (wn_train >= wn_min) & (wn_train <= wn_max)
    wn_ref   = wn_train[mask_ref]

    # Interpola novos espectros para eixo do treino
    X_interp = np.zeros((X_new_raw.shape[0], len(wn_ref)))
    for i in range(X_new_raw.shape[0]):
        f = interp1d(wn_new.astype(float), X_new_raw[i].astype(float),
                     kind="linear", bounds_error=False, fill_value="extrapolate")  # type: ignore
        X_interp[i] = f(wn_ref)

    # Pre-processamento do treino
    X_proc = preproc.transform(X_interp)

    # Scores PLS (aplica centering interno do modelo)
    T_new  = np.asarray(pls.transform(X_proc), dtype=float)
    P      = np.asarray(pls.x_loadings_, dtype=float)   # (p, k)
    P_T    = P.T                                          # (k, p)

    # T2 de Hotelling — mesma formula do pipeline (escala por variancia treino)
    T_train = np.asarray(pls.x_scores_, dtype=float)
    var_t   = T_train.var(axis=0, ddof=1)
    var_t[var_t == 0] = 1.0
    T2_new  = np.sum((T_new ** 2) / var_t, axis=1)

    # Q residuais — mesma convencao do pipeline: X_proc nao subtraido
    X_rec  = T_new @ P_T                                  # (n_new, p)
    Q_new  = np.sum((X_proc - X_rec) ** 2, axis=1)

    # UCL do pacote (gerados pelo pipeline v25+) ou fallback conservador
    t2_ucl = float(pkg.get("t2_ucl", np.percentile(
        np.sum((T_train ** 2) / var_t, axis=1), 95)))
    q_ucl  = float(pkg.get("q_ucl", np.percentile(Q_new, 99) * 1.5
                            if len(Q_new) > 0 else 1e6))

    # Predicao de classe via softmax-normalizado dos scores PLS
    Y_soft  = np.asarray(pls.predict(X_proc), dtype=float)
    Y_clip  = np.clip(Y_soft, 0.0, 1.0)
    totais  = Y_clip.sum(axis=1, keepdims=True)
    totais[totais < 1e-12] = 1.0
    Y_norm  = Y_clip / totais

    classes    = list(lb.classes_)
    idx_pred   = Y_norm.argmax(axis=1)
    classe_pred = [classes[i] if i < len(classes) else "?" for i in idx_pred]
    confianca  = Y_norm.max(axis=1)

    n = X_new_raw.shape[0]
    return pd.DataFrame({
        "amostra":    [f"S{i+1:03d}" for i in range(n)],
        "classe_pred": classe_pred,
        "confianca_%": np.round(confianca * 100, 1),
        "T2":          np.round(T2_new, 3),
        "T2_ucl":      round(t2_ucl, 3),
        "Q":           np.round(Q_new, 6),
        "Q_ucl":       round(q_ucl, 6),
        "T2_ok":       T2_new <= t2_ucl,
        "Q_ok":        Q_new  <= q_ucl,
        "aceito":      (T2_new <= t2_ucl) & (Q_new <= q_ucl),
    })


# ──────────────────────────────────────────────────────────────────────────
# Estado inicial
# ──────────────────────────────────────────────────────────────────────────

if "cfg_base" not in st.session_state:
    try:
        st.session_state.cfg_base = (
            pq.carregar_config(_CFG_PATH) if os.path.exists(_CFG_PATH)
            else pq.Config())
    except Exception:
        st.session_state.cfg_base = pq.Config()

cfg_base = st.session_state.cfg_base
specs    = _spec_por_key()

# ──────────────────────────────────────────────────────────────────────────
# Cabecalho
# ──────────────────────────────────────────────────────────────────────────

st.title("🧪 Plataforma Quimiometrica")
st.caption(
    "PLS-DA · PCA · OPLS-DA · DD-SIMCA · selecao de variaveis · "
    "validacao group-aware (anti-vazamento de replicas). "
    "FT-NIR (.dx) ou tabela CSV (Raman, UV-Vis, FTIR, cromatografia…)."
)

# ──────────────────────────────────────────────────────────────────────────
# 7 Abas
# ──────────────────────────────────────────────────────────────────────────

(tab_proj, tab_dados, tab_preproc, tab_modelo,
 tab_valid, tab_pred, tab_rel) = st.tabs([
    "📋 Projeto",
    "📂 Dados",
    "⚗️ Pré-proc",
    "🧮 Modelo",
    "📊 Validação",
    "🔮 Predição",
    "📄 Relatórios",
])

valores: Dict = {}  # acumulado pelos widgets de cada aba

# ==========================================================================
#  ABA 1 — PROJETO
# ==========================================================================
def _hardware_status_widget():
    """Exibe painel de hardware com alertas de compatibilidade."""
    try:
        hw = pq.hardware_probe()
        ram_t = hw["ram_total_gb"]
        ram_l = hw["ram_livre_gb"]
        cpu_f = hw["cpu_fisicos"]
        cpu_l = hw["cpu_logicos"]
        disco = hw["disco_livre_gb"]
        psutil_ok = hw["psutil_ok"]

        # Cor do semaforo de RAM livre
        if ram_l < 2.0:
            cor_ram = "🔴"
            dica = "RAM critica. Desabilite Benchmark, SHAP e MC CV."
        elif ram_l < 4.0:
            cor_ram = "🟠"
            dica = "RAM baixa. Benchmark e SHAP serao desabilitados automaticamente."
        elif ram_l < 8.0:
            cor_ram = "🟡"
            dica = "RAM moderada. Limites serao ajustados automaticamente."
        else:
            cor_ram = "🟢"
            dica = "RAM suficiente para todas as operacoes."

        c_hw1, c_hw2, c_hw3 = st.columns(3)
        with c_hw1:
            st.metric("RAM total", f"{ram_t:.1f} GB",
                      delta=f"{cor_ram} {ram_l:.1f} GB livre",
                      delta_color="off")
        with c_hw2:
            st.metric("CPU", f"{cpu_f} nucleos",
                      delta=f"{cpu_l} threads logicos",
                      delta_color="off")
        with c_hw3:
            st.metric("Disco livre", f"{disco:.0f} GB",
                      delta="pasta de trabalho",
                      delta_color="off")

        if ram_l < 8.0:
            st.warning(f"**Hardware limitado detectado.** {dica}")
        if not psutil_ok:
            st.caption("⚠️ psutil nao disponivel — leituras aproximadas. "
                       "Instale com `pip install psutil`.")
    except Exception:
        st.caption("Hardware: nao foi possivel detectar especificacoes.")


with tab_proj:
    st.subheader("Identificacao do Projeto")
    with st.expander("💻 Status do Hardware", expanded=False):
        _hardware_status_widget()
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Nome do projeto", key="proj_nome",
                      placeholder="ex: Autenticacao de Oleos Amazonicos FT-NIR")
        st.text_input("Autor(es)", key="proj_autor",
                      placeholder="ex: Silva, J.A.; Costa, M.B.")
        st.text_input("Instituicao / Laboratorio", key="proj_inst",
                      placeholder="ex: GEAAp / UFPA")
    with c2:
        st.selectbox("Tipo de estudo",
                     ["Classificacao de especies",
                      "Autenticacao (puro vs adulterado)",
                      "Quantificacao (regressao)",
                      "Outro"],
                     key="proj_tipo")
        st.text_area("Objetivo", key="proj_objetivo", height=120,
                     placeholder="Descreva o objetivo da analise quimiometrica...")

    st.divider()
    st.markdown("**Salvar / Exportar identificacao**")
    if st.button("💾 Salvar identificacao na sessao"):
        st.success("Identificacao registrada. Sera incluida nos relatorios desta sessao.")

    run_proj = st.session_state.get("proj_nome", "")
    if run_proj:
        st.info(f"Projeto ativo: **{run_proj}** — {st.session_state.get('proj_tipo', '')}")

# ==========================================================================
#  ABA 2 — DADOS
# ==========================================================================
with tab_dados:
    st.subheader("Entrada de Dados")

    _DADOS_KEYS = ["modo_entrada", "pasta_dados", "arquivo_csv",
                   "coluna_classe", "coluna_concentracao",
                   "pasta_saida", "excluir_classes"]

    col_d1, col_d2 = st.columns(2)
    for i, k in enumerate(_DADOS_KEYS):
        s = specs.get(k)
        if s is None:
            continue
        with (col_d1 if i % 2 == 0 else col_d2):
            valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    # ---- Upload de CSV (opcao adicional ao caminho local) -----------------
    st.divider()
    st.markdown("**Upload de CSV** *(alternativa ao caminho local acima)*")
    upld = st.file_uploader(
        "Arraste ou selecione um arquivo CSV",
        type=["csv", "txt"],
        key="csv_upload_widget",
        help="O arquivo sera salvo em pasta temporaria e o caminho ajustado automaticamente.",
    )
    if upld is not None:
        tmp_dir = Path(tempfile.gettempdir()) / "pq_uploads"
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = str(tmp_dir / upld.name)
        if (st.session_state.get("_csv_upload_name") != upld.name
                or not os.path.exists(tmp_path)):
            with open(tmp_path, "wb") as f:
                f.write(upld.getvalue())
            st.session_state["_csv_upload_name"] = upld.name
            st.session_state["_csv_upload_path"] = tmp_path
        st.success(f"Arquivo salvo: `{tmp_path}`")
        st.info("Modo automaticamente ajustado para 'csv'. "
                "Caminho acima sera sobreposto ao rodar.")

    # ---- Preview de estatisticas -----------------------------------------
    st.divider()
    st.markdown("**Preview dos dados**")
    cfg_prev, _ = _coletar_config(cfg_base, valores)
    ok_dados, msg_dados = pq._validar_pasta_dados(cfg_prev)
    (st.success if ok_dados else st.warning)(f"Status: {msg_dados}")

    if ok_dados and st.button("🔍 Carregar preview de espectros", key="btn_prev_dados"):
        modo = cfg_prev.modo
        wn_mn = float(cfg_prev.wn_min)
        wn_mx = float(cfg_prev.wn_max)
        with st.spinner("Carregando amostra de espectros..."):
            if modo == "dx":
                wn_p, X_p, labs_p = _preview_espectros_dx(
                    cfg_prev.pasta_entrada, wn_mn, wn_mx)
            elif modo == "csv":
                csv_cam = st.session_state.get("_csv_upload_path",
                                               cfg_prev.arquivo_csv)
                wn_p, X_p, labs_p = _preview_espectros_csv(
                    csv_cam, cfg_prev.coluna_classe, wn_mn, wn_mx)
            else:
                wn_p, X_p, labs_p = None, None, None

        if wn_p is not None and X_p is not None:
            cls_u = np.unique(np.asarray(labs_p))
            st.markdown(f"**{len(X_p)} espectros** · {len(cls_u)} classes: "
                        f"`{'`, `'.join(cls_u[:8])}`"
                        + (" ..." if len(cls_u) > 8 else ""))
            fig_p = _plot_espectros_media(wn_p, X_p, np.asarray(labs_p), titulo="Espectros brutos (amostra)")
            st.pyplot(fig_p, use_container_width=True)
            plt.close(fig_p)
        else:
            st.warning("Nao foi possivel carregar espectros para preview. "
                       "Verifique o caminho/modo.")

    # ---- Salvar / Recarregar config.yaml ---------------------------------
    st.divider()
    cfg_dados, erros_dados = _coletar_config(cfg_base, valores)
    if erros_dados:
        st.warning("Campos com erro:\n- " + "\n- ".join(erros_dados))
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        if st.button("💾 Salvar config.yaml", key="btn_salvar_cfg_dados",
                     use_container_width=True):
            if erros_dados:
                st.error("Corrija os campos antes de salvar.")
            else:
                pq.salvar_config(cfg_dados, _CFG_PATH)
                st.session_state.cfg_base = cfg_dados
                st.success(f"Salvo em {_CFG_PATH}")
    with c_s2:
        if st.button("↺ Recarregar config.yaml", key="btn_reload_cfg_dados",
                     use_container_width=True):
            try:
                st.session_state.cfg_base = pq.carregar_config(_CFG_PATH)
                cfg_base = st.session_state.cfg_base
                st.success("Config recarregada.")
                st.rerun()
            except Exception as e:
                st.error(f"Erro: {e}")

# ==========================================================================
#  ABA 3 — PRE-PROCESSAMENTO
# ==========================================================================
with tab_preproc:
    st.subheader("Pre-processamento Espectral")

    _PREPROC_KEYS = ["pre_processamento", "faixa_min_cm", "faixa_max_cm"]

    col_p1, col_p2 = st.columns(2)
    for i, k in enumerate(_PREPROC_KEYS):
        s = specs.get(k)
        if s is None:
            continue
        with (col_p1 if i % 2 == 0 else col_p2):
            valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    # Informativo sobre cada preset
    preset_selecionado = valores.get("pre_processamento", "")
    _PRESET_INFO = {
        "MSC+SG+MC":      "MSC (correcao de scatter) → 1a derivada SG (Savitzky-Golay) → Mean-Centering. **Melhor para FT-NIR com scatter pronunciado.** Acc=0.923 na base de 1807 amostras de oleos amazonicos.",
        "SNV+SG+MC":      "SNV (normalizacao por variancia) → SG → Mean-Centering. Alternativa robusta ao MSC quando a referencia global nao e estavel.",
        "Autoscaling":    "Mean-Centering + divisao pelo desvio padrao. **Cuidado**: colapsa ruido espectral quando SG nao e aplicado antes.",
        "Mean-centering": "Apenas centragem pela media. Recomendado como baseline comparativo.",
    }
    if preset_selecionado in _PRESET_INFO:
        st.info(_PRESET_INFO[preset_selecionado])

    # ---- Preview antes/depois -------------------------------------------
    st.divider()
    st.markdown("**Visualizacao antes / depois do pre-processamento**")
    cfg_pp, _ = _coletar_config(cfg_base, valores)
    ok_pp, _ = pq._validar_pasta_dados(cfg_pp)

    if not ok_pp:
        st.info("Configure e valide a entrada de dados (Aba 'Dados') para habilitar o preview.")
    elif st.button("⚗️ Gerar preview antes/depois", key="btn_prev_preproc"):
        with st.spinner("Carregando e processando espectros..."):
            modo_pp = cfg_pp.modo
            wn_mn_pp = float(cfg_pp.wn_min)
            wn_mx_pp = float(cfg_pp.wn_max)
            if modo_pp == "dx":
                wn_raw, X_raw, labs_raw = _preview_espectros_dx(
                    cfg_pp.pasta_entrada, wn_mn_pp, wn_mx_pp)
            elif modo_pp == "csv":
                csv_cam_pp = st.session_state.get("_csv_upload_path",
                                                   cfg_pp.arquivo_csv)
                wn_raw, X_raw, labs_raw = _preview_espectros_csv(
                    csv_cam_pp, cfg_pp.coluna_classe, wn_mn_pp, wn_mx_pp)
            else:
                wn_raw, X_raw, labs_raw = None, None, None

        if wn_raw is not None and X_raw is not None:
            try:
                preproc_pp = pq.construir_preprocessador(cfg_pp)
                preproc_pp.fit(X_raw)
                X_proc_pp = preproc_pp.transform(X_raw)
                labs_raw_arr = np.asarray(labs_raw)
                fig_antes = _plot_espectros_media(
                    wn_raw, X_raw, labs_raw_arr, "Antes do pre-processamento")
                fig_depois = _plot_espectros_media(
                    wn_raw, X_proc_pp, labs_raw_arr,
                    f"Apos: {preset_selecionado}")
                col_ant, col_dep = st.columns(2)
                with col_ant:
                    st.pyplot(fig_antes, use_container_width=True)
                    plt.close(fig_antes)
                with col_dep:
                    st.pyplot(fig_depois, use_container_width=True)
                    plt.close(fig_depois)
            except Exception as e_pp:
                st.error(f"Erro ao aplicar pre-processamento: {e_pp}")
        else:
            st.warning("Nao foi possivel carregar espectros. Verifique a Aba Dados.")

# ==========================================================================
#  ABA 4 — MODELO (parametros avancados + execucao)
# ==========================================================================
with tab_modelo:
    st.subheader("Parametros do Modelo e Execucao")

    _MODELO_KEYS_ANALISE  = ["nivel", "max_lvs", "holdout_fracao",
                              "validacao_group_aware"]
    _MODELO_KEYS_VALID    = ["n_permutacoes", "teste_wold", "teste_cv_anova"]
    _MODELO_KEYS_EXTRAS   = ["selecao_variaveis_etapa4", "ddsimca", "opls_da",
                              "comparar_pre_processamentos", "benchmark",
                              "monte_carlo", "n_monte_carlo",
                              "monte_carlo_incluir_todos",
                              "shap_benchmark", "shap_max_amostras"]
    _MODELO_KEYS_FIGURAS  = ["figuras_mostrar_marcadores",
                              "figuras_mostrar_elipses",
                              "formato_figura", "dpi",
                              "abrir_figuras_na_tela"]

    with st.expander("Analise e particionamento", expanded=True):
        cols_a = st.columns(2)
        for i, k in enumerate(_MODELO_KEYS_ANALISE):
            s = specs.get(k)
            if s is None: continue
            with cols_a[i % 2]:
                valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    with st.expander("Validacao estatistica", expanded=False):
        cols_v = st.columns(2)
        for i, k in enumerate(_MODELO_KEYS_VALID):
            s = specs.get(k)
            if s is None: continue
            with cols_v[i % 2]:
                valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    with st.expander("Modulos extras", expanded=False):
        # Aviso de hardware para operacoes pesadas
        try:
            _hw_mod = pq.hardware_probe()
            _ram_l  = _hw_mod.get("ram_livre_gb", 16.0)
            if _ram_l < 4.0:
                st.error(
                    f"⚠️ RAM livre: **{_ram_l:.1f} GB** — "
                    "Benchmark e SHAP serao **desabilitados automaticamente** "
                    "pelo pipeline para evitar travamento.")
            elif _ram_l < 8.0:
                st.warning(
                    f"⚠️ RAM livre: **{_ram_l:.1f} GB** — "
                    "Limites de SHAP e Monte Carlo CV serao ajustados "
                    "automaticamente. Recomenda-se fechar outros programas.")
        except Exception:
            pass

        cols_e = st.columns(2)
        for i, k in enumerate(_MODELO_KEYS_EXTRAS):
            s = specs.get(k)
            if s is None: continue
            with cols_e[i % 2]:
                valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    with st.expander("Figuras", expanded=False):
        cols_f = st.columns(2)
        for i, k in enumerate(_MODELO_KEYS_FIGURAS):
            s = specs.get(k)
            if s is None: continue
            with cols_f[i % 2]:
                valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    st.divider()

    # ---- Montagem final da Config e execucao ----------------------------
    cfg_run, erros_run = _coletar_config(cfg_base, valores)

    # Se usuario fez upload de CSV, sobrepoe o caminho
    if st.session_state.get("_csv_upload_path"):
        csv_upld_path = st.session_state["_csv_upload_path"]
        if os.path.exists(csv_upld_path):
            cfg_run.modo = "csv"
            cfg_run.arquivo_csv = csv_upld_path

    ok_run, msg_run = pq._validar_pasta_dados(cfg_run)
    st.write("**Entrada:**", msg_run)
    if erros_run:
        st.error("Campos invalidos na configuracao:\n- " + "\n- ".join(erros_run))

    pode_rodar = ok_run and not erros_run
    rodar = st.button("▶️ Rodar pipeline", type="primary",
                      disabled=not pode_rodar, use_container_width=True,
                      key="btn_rodar")
    if not pode_rodar:
        st.info("Corrija a entrada de dados (Aba 'Dados') para habilitar.")

    if rodar:
        try:
            pq.salvar_config(cfg_run, _CFG_PATH)
        except Exception:
            pass

        logger = _LogThreadSafe(tee=sys.__stdout__)
        estado: Dict = {"fim": False, "erro": None, "pasta": None}
        worker = threading.Thread(
            target=_rodar_worker, args=(cfg_run, logger, estado), daemon=True)

        barra    = st.progress(0.0, text="Iniciando...")
        ph_info  = st.empty()
        ph_log   = st.empty()

        t0 = time.monotonic()
        eta_best: Optional[float] = None
        worker.start()

        while not estado["fim"]:
            txt = logger.text()
            frac, nome = _progresso_do_log(txt)
            elapsed = time.monotonic() - t0
            if frac >= 0.10:
                eta = elapsed / frac - elapsed
                eta_best = eta if eta_best is None else min(eta_best, eta)
            ram = _ram_mb()
            barra.progress(frac, text=f"[{int(frac * 100)}%] {nome}")
            ph_info.markdown(
                f"⏱️ **Decorrido:** {_fmt_tempo(elapsed)}  |  "
                f"⏳ **Falta:** "
                f"{_fmt_tempo(eta_best) if eta_best is not None else 'calculando…'}  |  "
                f"💾 **RAM:** {f'{ram:.0f} MB' if ram else 'n/d'}")
            linhas = txt.strip().splitlines()
            ph_log.code("\n".join(linhas[-12:]) if linhas else "...",
                        language="text")
            time.sleep(0.5)

        txt     = logger.text()
        elapsed = time.monotonic() - t0
        ram     = _ram_mb()
        barra.progress(1.0, text="Concluido")
        ph_info.markdown(
            f"⏱️ **Tempo total:** {_fmt_tempo(elapsed)}  |  "
            f"⏳ **Falta:** 0s  |  "
            f"💾 **RAM:** {f'{ram:.0f} MB' if ram else 'n/d'}")
        linhas = txt.strip().splitlines()
        ph_log.code("\n".join(linhas[-12:]) if linhas else "...",
                    language="text")

        st.session_state.ultimo_log  = txt
        if estado["erro"]:
            st.session_state.erro_run  = estado["erro"]
            st.session_state.ultima_pasta = None
            st.error(f"Pipeline falhou apos {_fmt_tempo(elapsed)}.")
        else:
            st.session_state.erro_run  = None
            st.session_state.ultima_pasta = estado["pasta"]
            st.success(f"Concluido em {_fmt_tempo(elapsed)}! "
                       "Veja os resultados nas abas Validacao e Relatorios.")

    if st.session_state.get("erro_run"):
        st.subheader("Traceback do erro")
        st.code(st.session_state.erro_run, language="text")

# ==========================================================================
#  ABA 5 — VALIDACAO
# ==========================================================================
with tab_valid:
    st.subheader("Resultados de Validacao")
    pasta_v = st.session_state.get("ultima_pasta")

    if not pasta_v or not os.path.isdir(pasta_v):
        st.info("Execute o pipeline (Aba 'Modelo') para visualizar os resultados aqui.")
    else:
        st.caption(f"Pasta: `{os.path.abspath(pasta_v)}`")

        # Resumo numerico
        resumo_txt = _ler_resumo(pasta_v)
        if resumo_txt:
            with st.expander("📋 Resumo do modelo (resumo_modelo.txt)", expanded=True):
                st.code(resumo_txt, language="text")

        # ── Tabela Accuracy por Classe (extraida do resumo) ───────────────
        if resumo_txt:
            import re as _re_acc
            acc_map: Dict[str, float] = {}
            for _linha in resumo_txt.splitlines():
                _m = _re_acc.match(r"\s*Acc\s+(.+?)\s*[:=]\s*([\d.]+)", _linha)
                if _m:
                    acc_map[_m.group(1).strip()] = float(_m.group(2))
            if acc_map:
                with st.expander("📊 Accuracy por Classe", expanded=True):
                    _df_acc = pd.DataFrame(
                        list(acc_map.items()), columns=["Classe", "Accuracy (recall)"]
                    ).sort_values("Accuracy (recall)")
                    # Colorir: vermelho < 0.7, amarelo 0.7-0.9, verde >= 0.9
                    def _cor_acc(v: float) -> str:
                        if v >= 0.90: return "background-color:#d4edda"
                        if v >= 0.70: return "background-color:#fff3cd"
                        return "background-color:#f8d7da"
                    st.dataframe(
                        _df_acc.style.map(
                            _cor_acc, subset=["Accuracy (recall)"]),
                        use_container_width=True, height=min(400, 35 * len(_df_acc) + 38))

        # ── Tabelas de Benchmark e MC CV (se existirem) ───────────────────
        _bench_csv_v = os.path.join(pasta_v, "dados",
                                    "benchmark_classificadores.csv")
        _mc_csv_v    = os.path.join(pasta_v, "dados", "monte_carlo_cv.csv")
        if os.path.exists(_bench_csv_v) or os.path.exists(_mc_csv_v):
            with st.expander("🏅 Benchmark de Classificadores", expanded=False):
                if os.path.exists(_bench_csv_v):
                    try:
                        _df_b = pd.read_csv(_bench_csv_v, sep=";", decimal=",")
                        st.markdown("**Auto-Benchmark — Balanced Accuracy por modelo (GroupKFold)**")
                        st.dataframe(_df_b, use_container_width=True)
                    except Exception as _e_b:
                        st.warning(f"Erro ao ler benchmark CSV: {_e_b}")
                if os.path.exists(_mc_csv_v):
                    try:
                        _df_mc = pd.read_csv(_mc_csv_v, sep=";", decimal=",")
                        st.markdown("**Monte Carlo CV — IC95% por percentil**")
                        st.dataframe(_df_mc, use_container_width=True)
                    except Exception as _e_mc:
                        st.warning(f"Erro ao ler MC CV CSV: {_e_mc}")

        # Galeria filtrada por categoria de figura
        imgs_v = _listar_figuras(pasta_v)
        if imgs_v:
            st.markdown(f"**{len(imgs_v)} figuras geradas**")
            _CATS = {
                "Todas":            "",
                "PCA":              "pca",
                "PLS-DA scores":    "plsda",
                "Outliers (T²/Q)":  "outlier",
                "Confusao":         "confus",
                "ROC / AUC":        "roc",
                "VIP / SR":         "vip",
                "Loading":          "loading",
                "HCA":              "hca",
                "OPLS-DA":          "opls",
                "DD-SIMCA":         "ddsimca",
                "Cooman's Plot":    "cooman",
                "S-Plot":           "splot",
                "Permutacao":       "permut",
                "Wold":             "wold",
                "Benchmark":        "benchmark",
                "Monte Carlo CV":   "monte_carlo",
                "DET curves":       "fig_det",
                "SHAP":             "fig_shap",
            }
            filtro_v = st.selectbox("Filtrar por categoria",
                                    list(_CATS.keys()), key="filtro_valid")
            token_v  = _CATS[filtro_v].lower()
            imgs_filt = [im for im in imgs_v
                         if token_v in os.path.basename(im).lower()] \
                        if token_v else imgs_v

            if imgs_filt:
                cols_v = st.columns(2)
                for j, img in enumerate(imgs_filt):
                    with cols_v[j % 2]:
                        st.image(img, caption=os.path.basename(img),
                                 use_container_width=True)
            else:
                st.info(f"Nenhuma figura encontrada para '{filtro_v}'.")
        else:
            st.info("Figuras salvas em formato nao visualizavel (PDF/SVG). "
                    "Use o download na aba Relatorios.")

# ==========================================================================
#  ABA 6 — PREDICAO
# ==========================================================================
with tab_pred:
    st.subheader("Predicao de Amostras Desconhecidas")
    st.markdown(
        "Carregue um modelo `.joblib` gerado pelo pipeline e um CSV "
        "com novos espectros (colunas = numeros de onda, sem coluna de classe)."
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**1. Modelo treinado (.joblib)**")
        upld_jbl = st.file_uploader("Upload do modelo .joblib",
                                    type=["joblib", "pkl"],
                                    key="pred_model_upload")
        cam_jbl  = st.text_input("Ou caminho local do modelo",
                                  key="pred_model_path",
                                  placeholder="C:/resultados/modelo_pls.joblib")

    with col_m2:
        st.markdown("**2. Espectros novos (CSV)**")
        upld_csv_pred = st.file_uploader("Upload do CSV com novos espectros",
                                          type=["csv", "txt"],
                                          key="pred_csv_upload")
        cam_csv_pred  = st.text_input("Ou caminho local do CSV",
                                       key="pred_csv_path",
                                       placeholder="C:/dados/novos_espectros.csv")
        col_wn_pred   = st.text_input(
            "Primeira coluna a usar como numero de onda (deixe vazio = auto)",
            key="pred_col_wn",
            placeholder="ex: 4000.0  (nome ou indice da primeira coluna espectral)")

    # ---- Botao Predizer -------------------------------------------------
    st.divider()
    if st.button("🔮 Predizer", type="primary", key="btn_predizer",
                 use_container_width=True):
        erros_pred: List[str] = []
        pkg_pred = None

        # Carrega modelo
        try:
            import joblib
            if upld_jbl is not None:
                tmp_jbl = Path(tempfile.gettempdir()) / "pq_pred_model.joblib"
                with open(tmp_jbl, "wb") as f:
                    f.write(upld_jbl.getvalue())
                pkg_pred = joblib.load(str(tmp_jbl))
            elif cam_jbl and os.path.exists(cam_jbl):
                pkg_pred = joblib.load(cam_jbl)
            else:
                erros_pred.append("Nenhum modelo valido fornecido (upload ou caminho).")
        except Exception as e_jbl:
            erros_pred.append(f"Erro ao carregar modelo: {e_jbl}")

        # Carrega CSV de predicao
        X_pred_raw = None
        wn_pred    = None
        try:
            if upld_csv_pred is not None:
                df_pred = pd.read_csv(upld_csv_pred, sep=None, engine="python")
            elif cam_csv_pred and os.path.exists(cam_csv_pred):
                df_pred = pd.read_csv(cam_csv_pred, sep=None, engine="python")
            else:
                erros_pred.append("Nenhum CSV de espectros fornecido.")
                df_pred = None

            if df_pred is not None:
                # Detecta colunas numericas (wavenumbers)
                num_cols_pred = []
                for c in df_pred.columns:
                    try:
                        float(c); num_cols_pred.append(c)
                    except ValueError:
                        pass
                if num_cols_pred:
                    wn_pred    = np.array([float(c) for c in num_cols_pred])
                    X_pred_raw = df_pred[num_cols_pred].values.astype(float)
                    st.session_state["pred_amostras"] = df_pred.drop(
                        columns=num_cols_pred, errors="ignore")
                else:
                    erros_pred.append(
                        "Nao foram encontradas colunas com nomes numericos (wavenumbers). "
                        "Certifique-se de que os cabecalhos das colunas espectrais "
                        "sao os numeros de onda (ex: 4000.5, 4001.0...).")
        except Exception as e_csv:
            erros_pred.append(f"Erro ao ler CSV: {e_csv}")

        if erros_pred:
            for e in erros_pred:
                st.error(e)
        elif pkg_pred is not None and X_pred_raw is not None:
            try:
                with st.spinner("Aplicando modelo..."):
                    df_res = _predizer(pkg_pred, X_pred_raw, wn_pred)
                    # Recoloca metadados (nomes de amostras) se disponiveis
                    meta_df = st.session_state.get("pred_amostras")
                    if meta_df is not None and len(meta_df) == len(df_res):
                        df_res = pd.concat(
                            [meta_df.reset_index(drop=True), df_res], axis=1)
                st.session_state["pred_resultados"] = df_res
                st.success(f"Predicao concluida: {len(df_res)} amostras.")
            except Exception as e_pred:
                st.error(f"Erro na predicao: {e_pred}")

    # ---- Exibicao dos resultados ----------------------------------------
    df_show = st.session_state.get("pred_resultados")
    if df_show is not None:
        st.divider()
        st.markdown("**Resultados da predicao**")

        # Colorir aceito/rejeitado
        def _colorir_aceito(val):
            if val is True:
                return "background-color:#d4edda; color:#155724"
            if val is False:
                return "background-color:#f8d7da; color:#721c24"
            return ""

        aceito_col = "aceito" if "aceito" in df_show.columns else None
        if aceito_col:
            st.dataframe(
                df_show.style.map(_colorir_aceito, subset=[aceito_col]),
                use_container_width=True,
            )
        else:
            st.dataframe(df_show, use_container_width=True)

        # Download CSV
        csv_bytes = df_show.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
        st.download_button(
            "⬇️ Baixar resultados (.csv)",
            data=csv_bytes,
            file_name="predicao_resultados.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Resumo rapido
        if "aceito" in df_show.columns:
            n_ac = int(df_show["aceito"].sum())
            n_tot = len(df_show)
            st.metric("Amostras aceitas (T² ≤ UCL e Q ≤ UCL)",
                      f"{n_ac} / {n_tot}",
                      delta=f"{n_ac/n_tot*100:.1f}%")

def _gerar_pptx_relatorio(pasta: str, projeto: Dict,
                           max_figuras: int = 12) -> io.BytesIO:
    """
    Gera apresentacao PowerPoint com design científico profissional.
    Paleta: Navy #0F172A + Accent #0369A1 (UI Pro Max — B2B Professional).
    Estrutura: Capa | Metodologia | Metricas | Figuras | Benchmark | Conclusoes
    """
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    import re as _re

    # ── Paleta UI Pro Max — B2B Professional ──────────────────────────────
    _NAVY   = RGBColor(0x0F, 0x17, 0x2A)
    _SLATE  = RGBColor(0x33, 0x41, 0x55)
    _ACCENT = RGBColor(0x03, 0x69, 0xA1)
    _LIGHT  = RGBColor(0xF8, 0xFA, 0xFC)
    _WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
    _MUTED  = RGBColor(0x64, 0x74, 0x8B)

    resumo_raw = _ler_resumo(pasta) or ""

    def _ex(padrao: str, default: str = "—") -> str:
        m = _re.search(padrao, resumo_raw, _re.IGNORECASE | _re.MULTILINE)
        return m.group(1).strip() if m else default

    metricas = {
        "Balanced Accuracy (CV)": _ex(r"[Bb]alanced[_ ]?[Aa]cc.*?[:=]\s*([\d.]+)"),
        "AUC macro OvR":          _ex(r"ROC AUC macro.*?[:=]\s*([\d.]+)"),
        "R2Y":                    _ex(r"\bR2Y\b.*?[:=]\s*([\d.]+)"),
        "Q2Y":                    _ex(r"\bQ2\b.*?[:=]\s*([\d.E+-]+)"),
        "LVs otimos":             _ex(r"LVs?\s+otim.*?[:=]\s*(\d+)"),
        "Pre-processamento":      _ex(r"[Pp]re.?[Pp]rocess.*?[:=]\s*([A-Za-z0-9_+]+)"),
        "N amostras":             _ex(r"[Nn]\s+treino.*?[:=]\s*(\d+)"),
        "N classes":              _ex(r"[Nn]\.?\s*[Cc]lasses.*?[:=]\s*(\d+)"),
        "p permutacao":           _ex(r"p.?value.*?[:=]\s*([\d.E+-]+)"),
    }

    # ── Helpers de layout ─────────────────────────────────────────────────
    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)
    W = prs.slide_width
    H = prs.slide_height

    def _rect(slide, l, t, w, h, fill_rgb: RGBColor):
        from pptx.util import Emu
        shape = slide.shapes.add_shape(1, Emu(l), Emu(t), Emu(w), Emu(h))
        shape.fill.solid(); shape.fill.fore_color.rgb = fill_rgb
        shape.line.fill.background()
        return shape

    def _txt(slide, texto: str, l, t, w, h,
             bold=False, size=18, color=_WHITE,
             align=PP_ALIGN.LEFT, wrap=True):
        tb = slide.shapes.add_textbox(Emu(l), Emu(t), Emu(w), Emu(h))
        tf = tb.text_frame; tf.word_wrap = wrap
        p = tf.paragraphs[0]
        p.alignment = align
        run = p.add_run(); run.text = texto
        run.font.bold = bold; run.font.size = Pt(size)
        run.font.color.rgb = color
        return tb

    def _slide_novo(layout_idx=6):
        layout = prs.slide_layouts[layout_idx]
        return prs.slides.add_slide(layout)

    def _fundo(slide, cor=_LIGHT):
        bg = slide.background; fill = bg.fill
        fill.solid(); fill.fore_color.rgb = cor

    def _barra_topo(slide, titulo: str):
        _rect(slide, 0, 0, int(W), int(Inches(1.0)), _NAVY)
        _rect(slide, 0, int(Inches(1.0)), int(W), int(Inches(0.06)), _ACCENT)
        _txt(slide, titulo, int(Inches(0.4)), int(Inches(0.15)),
             int(W - Inches(0.8)), int(Inches(0.75)),
             bold=True, size=22, color=_WHITE)

    def _rodape(slide):
        _rect(slide, 0, int(H - Inches(0.4)), int(W), int(Inches(0.4)), _SLATE)
        data_str = time.strftime("%d/%m/%Y")
        inst = projeto.get("inst", "GEAAp / UFPA")
        _txt(slide,
             f"{inst}  •  Plataforma Quimiometrica  •  {data_str}",
             int(Inches(0.3)), int(H - Inches(0.35)),
             int(W - Inches(0.6)), int(Inches(0.35)),
             size=9, color=_LIGHT, align=PP_ALIGN.CENTER)

    # ── SLIDE 1: Capa ─────────────────────────────────────────────────────
    slide1 = _slide_novo()
    _fundo(slide1, _NAVY)
    # Faixa accent horizontal
    _rect(slide1, 0, int(Inches(3.5)), int(W), int(Inches(0.12)), _ACCENT)
    # Titulo principal
    _txt(slide1, projeto.get("nome", "Plataforma Quimiometrica"),
         int(Inches(1.0)), int(Inches(1.2)),
         int(W - Inches(2.0)), int(Inches(1.5)),
         bold=True, size=36, color=_WHITE)
    # Subtitulo
    _txt(slide1, projeto.get("tipo", "Analise Quimiometrica FT-NIR"),
         int(Inches(1.0)), int(Inches(2.8)),
         int(W - Inches(2.0)), int(Inches(0.8)),
         size=20, color=RGBColor(0xCB, 0xD5, 0xE1))
    # Metadados
    autor  = projeto.get("autor", "")
    inst   = projeto.get("inst", "GEAAp / UFPA")
    data_s = time.strftime("%d/%m/%Y")
    _txt(slide1, f"{autor}\n{inst}\n{data_s}",
         int(Inches(1.0)), int(Inches(4.0)),
         int(Inches(6.0)), int(Inches(2.0)),
         size=14, color=RGBColor(0x94, 0xA3, 0xB8))

    # ── SLIDE 2: Metodologia ──────────────────────────────────────────────
    slide2 = _slide_novo()
    _fundo(slide2)
    _barra_topo(slide2, "Metodologia")
    _rodape(slide2)

    metod_items = [
        f"Pre-processamento: {metricas['Pre-processamento']}",
        f"Classificador principal: PLS-DA ({metricas['LVs otimos']} LVs otimos)",
        f"Validacao cruzada: GroupKFold anti-leakage de replicas (mae_id)",
        f"Testes estatisticos: Y-permutacao, Wold (R2Y/Q2Y), CV-ANOVA",
        f"Selecao de variaveis: iPLS, VIP >= 1, SR top-20%, sPLS-DA",
        f"Validacao externa: holdout estratificado (puros sempre no treino)",
        f"Benchmark: PLS-DA vs SVM RBF vs RF vs GBM vs XGBoost",
        f"Dados: {metricas['N amostras']} amostras  |  {metricas['N classes']} classes",
    ]
    for i, item in enumerate(metod_items):
        _txt(slide2, f"• {item}",
             int(Inches(0.7)), int(Inches(1.3) + i * Inches(0.62)),
             int(W - Inches(1.4)), int(Inches(0.6)),
             size=13, color=_SLATE)

    # ── SLIDE 3: Metricas de Desempenho ───────────────────────────────────
    slide3 = _slide_novo()
    _fundo(slide3)
    _barra_topo(slide3, "Resultados — Metricas de Desempenho")
    _rodape(slide3)

    met_exibir = [
        ("Balanced Accuracy (CV)", metricas["Balanced Accuracy (CV)"], _ACCENT),
        ("AUC macro OvR",          metricas["AUC macro OvR"],          _NAVY),
        ("R2Y",                    metricas["R2Y"],                    _SLATE),
        ("Q2Y",                    metricas["Q2Y"],                    _SLATE),
        ("LVs otimos",             metricas["LVs otimos"],             _MUTED),
        ("p permutacao",           metricas["p permutacao"],           _MUTED),
    ]
    cols = 3; w_box = int((W - Inches(1.0)) / cols)
    h_box = int(Inches(1.6))
    for idx, (lbl, val, cor) in enumerate(met_exibir):
        col = idx % cols; row = idx // cols
        x = int(Inches(0.5) + col * w_box)
        y = int(Inches(1.3)  + row * Inches(2.0))
        _rect(slide3, x, y, w_box - int(Inches(0.15)), h_box, cor)
        _txt(slide3, val, x + int(Inches(0.2)), y + int(Inches(0.2)),
             w_box - int(Inches(0.4)), int(Inches(0.8)),
             bold=True, size=28, color=_WHITE, align=PP_ALIGN.CENTER)
        _txt(slide3, lbl, x + int(Inches(0.1)), y + int(Inches(0.95)),
             w_box - int(Inches(0.2)), int(Inches(0.55)),
             size=11, color=_LIGHT, align=PP_ALIGN.CENTER)

    # ── SLIDES 4+: Figuras ────────────────────────────────────────────────
    imgs = _listar_figuras(pasta)
    # Priorizar figuras relevantes
    prioridade = ["scores","confus","vip","pca","outlier","splot","cooman",
                  "roc","hca","opls","ddsimca","benchmark","shap","monte_carlo"]
    def _prio(p: str) -> int:
        nome_l = os.path.basename(p).lower()
        for i, t in enumerate(prioridade):
            if t in nome_l: return i
        return 99
    imgs_sorted = sorted(imgs, key=_prio)[:max_figuras]

    for i in range(0, len(imgs_sorted), 2):
        slide_f = _slide_novo()
        _fundo(slide_f)
        _barra_topo(slide_f, "Resultados — Figuras")
        _rodape(slide_f)
        for j, img_path in enumerate(imgs_sorted[i:i+2]):
            x = int(Inches(0.3) + j * Inches(6.4))
            try:
                slide_f.shapes.add_picture(img_path,
                    Emu(x), Emu(int(Inches(1.15))),
                    width=Emu(int(Inches(6.1))),
                    height=Emu(int(Inches(5.6))))
            except Exception:
                _txt(slide_f, f"[Figura: {os.path.basename(img_path)}]",
                     x, int(Inches(2.0)), int(Inches(6.0)), int(Inches(1.0)),
                     size=11, color=_MUTED)
            cap = os.path.splitext(os.path.basename(img_path))[0].replace("_", " ")
            _txt(slide_f, cap, x, int(H - Inches(0.85)),
                 int(Inches(6.0)), int(Inches(0.45)),
                 size=9, color=_MUTED, align=PP_ALIGN.CENTER)

    # ── SLIDE Benchmark (se CSV existir) ──────────────────────────────────
    bench_csv = os.path.join(pasta, "dados", "benchmark_classificadores.csv")
    if os.path.exists(bench_csv):
        try:
            df_b = pd.read_csv(bench_csv, sep=";", decimal=",")
            slide_b = _slide_novo()
            _fundo(slide_b)
            _barra_topo(slide_b, "Benchmark de Classificadores")
            _rodape(slide_b)
            # Tabela
            cols_b = list(df_b.columns)
            n_rows = min(len(df_b) + 1, 8)
            col_w  = int((W - Inches(1.0)) / len(cols_b))
            row_h  = int(Inches(0.52))
            # Cabecalho
            for ci, col_n in enumerate(cols_b):
                _rect(slide_b, int(Inches(0.5) + ci * col_w),
                      int(Inches(1.3)), col_w - 2, row_h, _NAVY)
                _txt(slide_b, str(col_n),
                     int(Inches(0.5) + ci * col_w + Inches(0.05)),
                     int(Inches(1.35)), col_w - int(Inches(0.1)),
                     int(Inches(0.45)), bold=True, size=10, color=_WHITE)
            # Linhas de dados
            fill_alts = [_LIGHT, RGBColor(0xE8, 0xEC, 0xF1)]
            for ri, row_data in enumerate(df_b.itertuples(index=False)):
                if ri >= 7: break
                fy = int(Inches(1.3) + (ri + 1) * row_h)
                for ci, val in enumerate(row_data):
                    _rect(slide_b, int(Inches(0.5) + ci * col_w),
                          fy, col_w - 2, row_h, fill_alts[ri % 2])
                    _txt(slide_b, str(val),
                         int(Inches(0.5) + ci * col_w + Inches(0.05)),
                         fy + int(Inches(0.05)),
                         col_w - int(Inches(0.1)), int(Inches(0.45)),
                         size=10, color=_SLATE)
        except Exception:
            pass

    # ── SLIDE Final: Conclusoes ───────────────────────────────────────────
    slide_c = _slide_novo()
    _fundo(slide_c, _NAVY)
    _rect(slide_c, 0, int(Inches(3.2)), int(W), int(Inches(0.1)), _ACCENT)
    _txt(slide_c, "Conclusoes",
         int(Inches(1.0)), int(Inches(1.0)),
         int(W - Inches(2.0)), int(Inches(0.9)),
         bold=True, size=30, color=_WHITE)
    ba   = metricas.get("Balanced Accuracy (CV)", "—")
    auc  = metricas.get("AUC macro OvR", "—")
    r2y  = metricas.get("R2Y", "—"); q2y = metricas.get("Q2Y", "—")
    conc = [
        f"• Pipeline MSC-SG-MC atingiu Balanced Accuracy = {ba} (GroupKFold anti-leakage)",
        f"• AUC macro OvR = {auc}  |  R2Y = {r2y}  |  Q2Y = {q2y}",
        f"• {metricas['N amostras']} amostras, {metricas['N classes']} classes "
          f"de oleos vegetais amazonicos — espectroscopia FT-NIR",
        f"• Validacao: Y-permutacao (p = {metricas['p permutacao']}), "
          f"Wold, CV-ANOVA e holdout externo",
        "• Plataforma exporta modelos .joblib para predicao de novas amostras",
    ]
    for i, c in enumerate(conc):
        _txt(slide_c, c,
             int(Inches(1.0)), int(Inches(2.0) + i * Inches(0.78)),
             int(W - Inches(2.0)), int(Inches(0.75)),
             size=14, color=RGBColor(0xCB, 0xD5, 0xE1))

    _rodape(slide_c)

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf


# ==========================================================================
#  ABA 7 — RELATORIOS
# ==========================================================================
with tab_rel:
    st.subheader("Relatorios e Downloads")
    pasta_r = st.session_state.get("ultima_pasta")

    if not pasta_r or not os.path.isdir(pasta_r):
        st.info("Execute o pipeline (Aba 'Modelo') para gerar relatorios.")
    else:
        st.caption(f"Pasta de resultados: `{os.path.abspath(pasta_r)}`")

        # ── Downloads ─────────────────────────────────────────────────────
        st.markdown("### ⬇️ Downloads")

        _nome_base = os.path.basename(pasta_r)
        _projeto_info = {
            "nome":     st.session_state.get("proj_nome", ""),
            "autor":    st.session_state.get("proj_autor", ""),
            "inst":     st.session_state.get("proj_inst", "GEAAp / UFPA"),
            "tipo":     st.session_state.get("proj_tipo", ""),
            "objetivo": st.session_state.get("proj_objetivo", ""),
        }

        # Linha 1: ZIP + PDF
        col_a, col_b = st.columns(2)
        with col_a:
            try:
                st.download_button(
                    "📦 Resultados completos (.zip)",
                    data=_zip_da_pasta(pasta_r),
                    file_name=_nome_base + ".zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            except Exception as e_zip:
                st.warning(f"ZIP: {e_zip}")

        with col_b:
            try:
                pdf_buf = _gerar_pdf_relatorio(pasta_r, _projeto_info)
                st.download_button(
                    "📄 Relatorio PDF",
                    data=pdf_buf,
                    file_name=_nome_base + "_relatorio.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                )
            except Exception as e_pdf:
                st.error(f"PDF: {e_pdf}")

        # Linha 2: Word + Excel
        col_c, col_d = st.columns(2)
        with col_c:
            try:
                word_buf = _gerar_word_relatorio(pasta_r, _projeto_info)
                st.download_button(
                    "📝 Relatorio Word (.docx)",
                    data=word_buf,
                    file_name=_nome_base + "_relatorio.docx",
                    mime="application/vnd.openxmlformats-officedocument"
                         ".wordprocessingml.document",
                    use_container_width=True,
                )
            except Exception as e_word:
                st.error(f"Word: {e_word}")

        with col_d:
            try:
                xlsx_buf = _gerar_excel_relatorio(pasta_r)
                st.download_button(
                    "📊 Dados em Excel (.xlsx)",
                    data=xlsx_buf,
                    file_name=_nome_base + "_dados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument"
                         ".spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e_xlsx:
                st.error(f"Excel: {e_xlsx}")

        # Linha 3: LaTeX + PowerPoint
        col_e, col_f = st.columns(2)
        with col_e:
            try:
                latex_bytes = _gerar_latex_template(pasta_r, _projeto_info)
                st.download_button(
                    "🔬 Template LaTeX (Talanta / Food Chemistry / J. Chemom.)",
                    data=latex_bytes,
                    file_name=_nome_base + "_template.tex",
                    mime="text/plain",
                    use_container_width=True,
                )
            except Exception as e_tex:
                st.error(f"LaTeX: {e_tex}")

        with col_f:
            try:
                from pptx import Presentation as _PPTXCheck  # noqa: F401
                pptx_buf = _gerar_pptx_relatorio(pasta_r, _projeto_info)
                st.download_button(
                    "🎯 Apresentacao PowerPoint (.pptx)",
                    data=pptx_buf,
                    file_name=_nome_base + "_apresentacao.pptx",
                    mime="application/vnd.openxmlformats-officedocument"
                         ".presentationml.presentation",
                    use_container_width=True,
                )
            except ImportError:
                st.warning(
                    "python-pptx nao instalado. "
                    "Execute: `pip install python-pptx>=1.1`",
                    icon="⚠️",
                )
            except Exception as e_pptx:
                st.error(f"PowerPoint: {e_pptx}")

        st.divider()

        # ── Limpeza de resultados antigos ─────────────────────────────────
        with st.expander("🗑️ Liberar espaco — Limpar resultados antigos",
                         expanded=False):
            _pasta_base_lim = os.path.dirname(pasta_r)
            _pastas_exist = sorted(
                [p for p in os.scandir(_pasta_base_lim) if p.is_dir()],
                key=lambda p: p.stat().st_mtime, reverse=True)
            n_pastas = len(_pastas_exist)
            # Calcular tamanho total
            def _tamanho_pasta_mb(pasta_p: str) -> float:
                tot = 0
                for raiz, _, arqs in os.walk(pasta_p):
                    for a in arqs:
                        try: tot += os.path.getsize(os.path.join(raiz, a))
                        except Exception: pass
                return round(tot / (1024 * 1024), 1)

            st.caption(f"Pasta de resultados: `{_pasta_base_lim}`  "
                       f"({n_pastas} execucoes armazenadas)")
            if n_pastas > 1:
                _manter = st.slider(
                    "Manter N execucoes mais recentes",
                    min_value=1, max_value=max(1, n_pastas - 1),
                    value=min(3, n_pastas - 1), key="lim_manter")
                _n_remover = n_pastas - _manter
                _tam_est = sum(
                    _tamanho_pasta_mb(p.path)
                    for p in _pastas_exist[_manter:])
                st.info(
                    f"Serao removidas **{_n_remover}** execucoes antigas "
                    f"(~{_tam_est:.0f} MB liberados). "
                    f"A execucao atual **nao sera afetada**.")
                if st.button("🗑️ Confirmar limpeza",
                             key="btn_limpar_resultados",
                             type="secondary"):
                    _res = pq.limpar_resultados_antigos(
                        _pasta_base_lim, _manter)
                    if _res["removidas"]:
                        st.success(
                            f"Removidas {len(_res['removidas'])} pastas, "
                            f"liberado {_res['liberado_mb']:.0f} MB.")
                    else:
                        st.info("Nenhuma pasta removida.")
                    if _res["erro"]:
                        st.warning(f"Erros: {_res['erro']}")
            else:
                st.info("Apenas uma execucao armazenada. Nada a limpar.")

        st.divider()

        # Resumo do modelo
        st.markdown("### 📋 Resumo do modelo")
        resumo_r = _ler_resumo(pasta_r)
        if resumo_r:
            st.text_area("resumo_modelo.txt", resumo_r, height=400)
        else:
            st.info("Arquivo resumo_modelo.txt nao encontrado.")

        st.divider()

        # Galeria completa com filtro
        st.markdown("### 🖼️ Galeria de figuras")
        imgs_r = _listar_figuras(pasta_r)
        if imgs_r:
            _CATS_R = {
                "Todas":            "",
                "PCA":              "pca",
                "PLS-DA":           "plsda",
                "Outliers":         "outlier",
                "Confusao":         "confus",
                "ROC / AUC":        "roc",
                "VIP / SR":         "vip",
                "Loading":          "loading",
                "HCA":              "hca",
                "OPLS-DA":          "opls",
                "DD-SIMCA":         "ddsimca",
                "Cooman's Plot":    "cooman",
                "S-Plot":           "splot",
                "Permutacao":       "permut",
                "Wold/ANOVA":       "wold",
                "Regressao":        "regressao",
                "Benchmark":        "benchmark",
                "Monte Carlo CV":   "monte_carlo",
                "DET curves":       "fig_det",
                "SHAP":             "fig_shap",
            }
            filtro_r = st.selectbox("Filtrar figuras",
                                    list(_CATS_R.keys()), key="filtro_rel")
            token_r  = _CATS_R[filtro_r].lower()
            imgs_filt_r = [im for im in imgs_r
                           if token_r in os.path.basename(im).lower()] \
                          if token_r else imgs_r
            st.caption(f"{len(imgs_filt_r)} figura(s) exibidas.")
            n_cols_r = st.slider("Colunas", 1, 3, 2, key="slider_cols_rel")
            cols_r   = st.columns(n_cols_r)
            for j, img in enumerate(imgs_filt_r):
                with cols_r[j % n_cols_r]:
                    st.image(img, caption=os.path.basename(img),
                             use_container_width=True)
        else:
            st.info("Nenhuma imagem PNG/JPG encontrada na pasta de resultados.")

        st.divider()

        # Log de execucao
        if st.session_state.get("ultimo_log"):
            with st.expander("📜 Log de execucao (saida do terminal)"):
                st.code(st.session_state.ultimo_log, language="text")
