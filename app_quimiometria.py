# -*- coding: utf-8 -*-
"""
============================================================================
 Chemometrics Platform — Streamlit Interface v25 · 7 tabs
============================================================================
Organization:
   1. Project      — study identification and objective
   2. Data         — input (FT-NIR .dx, local CSV, CSV upload)
   3. Preprocessing — spectral preset + before/after visualization
   4. Model        — advanced parameters + execution with live progress
   5. Validation   — figures and metrics from the last run
   6. Prediction   — apply saved model to unknown samples
   7. Reports      — download ZIP, summary, figure gallery, log

Engine: pineline_quimiometria_14.py (dynamically imported).
No code editing required: configure, run, download.
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
# Page config (must be the first Streamlit command)
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chemometrics Platform",
    page_icon="🧪",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────
# Pipeline engine
# ──────────────────────────────────────────────────────────────────────────
_AQUI = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_PATH = os.path.join(_AQUI, "pineline_quimiometria_14.py")
_CFG_PATH = os.path.join(_AQUI, "config.yaml")


@st.cache_resource(show_spinner="Loading pipeline engine...")
def _carregar_motor():
    if _AQUI not in sys.path:
        sys.path.insert(0, _AQUI)
    spec = importlib.util.spec_from_file_location("pq_engine", _PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Engine not found at {_PIPELINE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pq = _carregar_motor()

# ──────────────────────────────────────────────────────────────────────────
# Config helpers (_CONFIG_SPEC as single source of truth)
# ──────────────────────────────────────────────────────────────────────────

def _spec_por_key() -> Dict:
    return {s["key"]: s for s in pq._CONFIG_SPEC}


def _widget_para_campo(s: Dict, valor_atual, prefixo: str = "w_"):
    """Renders ONE widget according to field type and returns current value."""
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
        return st.text_input(rotulo + " (comma-separated)", value=txt,
                             help=ajuda, key=chave)
    return st.text_input(rotulo, value=str(valor_atual), help=ajuda, key=chave)


def _coletar_config(cfg_base, valores: Dict):
    """Applies widget values to a deep copy of Config."""
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
# File helpers
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
    Generates a complete PDF report with fpdf2.
    Structure: Cover | Metrics | Figures (2/page) | References.
    Returns BytesIO ready for st.download_button.
    """
    import re as _re
    import unicodedata
    from fpdf import FPDF

    def _a(txt: str) -> str:
        """Removes accents for fpdf2 Latin-1 fonts."""
        return unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode("ascii")

    # ── Parse resumo_modelo.txt ───────────────────────────────────────
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
        "Optimal LVs":              _ex(r"LVs?\s+otim[ao].*?[:=]\s*(\d+)"),
        "p-value (permutation)":    _ex(r"p.?value.*?[:=]\s*([\d.E+-]+)"),
        "Preprocessing":            _ex(r"[Pp]re.?[Pp]rocessamento.*?[:=]\s*([A-Za-z0-9_+]+)"),
        "Hotelling T2 UCL (95%)":   _ex(r"[Hh]otelling.*?[:=]\s*([\d.]+)"),
        "Q-residual UCL (95%)":     _ex(r"Q.residual.*?[:=]\s*([\d.E+-]+)"),
        "n samples (training)":     _ex(r"[Nn]\s+treino.*?[:=]\s*(\d+)"),
        "n classes":                _ex(r"[Nn]\.?\s*[Cc]lasses.*?[:=]\s*(\d+)"),
    }

    imgs = _listar_figuras(pasta)[:max_figuras]

    # ── PDF class ──────────────────────────────────────────────────────
    class RelatorioPDF(FPDF):
        def header(self):
            if self.page_no() == 1:
                return
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(100, 100, 100)
            proj_nome = _a(projeto.get("nome", "Plataforma Quimiometrica"))
            self.cell(130, 6, proj_nome[:60], border=0, align="L")
            self.cell(0, 6, f"Generated: {time.strftime('%Y-%m-%d')}",
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
            self.cell(0, 6, f"Page {self.page_no()} / {{nb}}", align="C")

    pdf = RelatorioPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(15, 20, 15)
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── COVER ────────────────────────────────────────────────────────────
    pdf.add_page()
    # Blue top band
    pdf.set_fill_color(30, 80, 140)
    pdf.rect(0, 0, 210, 55, style="F")
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(15, 12)
    pdf.cell(0, 12, "Chemometrics Platform", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_xy(15, 30)
    pdf.cell(0, 8, "PLS-DA  |  PCA  |  OPLS-DA  |  DD-SIMCA", align="C")

    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(15, 65)
    pdf.set_font("Helvetica", "B", 17)
    pdf.multi_cell(180, 9,
                   _a(projeto.get("nome", "Chemometric Analysis Report")),
                   align="C")

    pdf.ln(8)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_fill_color(240, 244, 250)
    campos_capa = [
        ("Author(s)",      projeto.get("autor", "-")),
        ("Institution",    projeto.get("inst", "-")),
        ("Study type",     projeto.get("tipo", "-")),
        ("Date",           time.strftime("%Y-%m-%d")),
        ("Folder",         os.path.basename(pasta)),
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
        pdf.cell(0, 7, "Objective:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(180, 5, _a(obj))

    # Cover footer
    pdf.set_y(-30)
    pdf.set_fill_color(30, 80, 140)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "I", 8)
    pdf.rect(0, pdf.get_y(), 210, 30, style="F")
    pdf.set_xy(15, pdf.get_y() + 5)
    pdf.cell(0, 5,
             "Generated by: Chemometrics Platform v26  |  GEAAp / UFPA  |  PIBIC Project",
             align="C")

    # ── SECTION 1: METRICS ───────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 80, 140)
    pdf.cell(0, 9, "1. Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(30, 80, 140)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(4)

    # Metrics table
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(30, 80, 140)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(100, 7, "Metric", border=1, fill=True)
    pdf.cell(75, 7, "Value (internal CV validation)", border=1, fill=True,
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

    # Quality criteria
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 80, 140)
    pdf.cell(0, 8, "Quality criteria (PLS-DA — literature):",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(0, 0, 0)
    criterios = [
        "R2Y > 0.60 and Q2Y > 0.50 → robust predictive model [Eriksson 2008]",
        "Q2Y / R2Y > 0.70 → low overfitting risk",
        "permutation p-value < 0.05 → non-random result (Y-randomization)",
        "Intercept R2Y < 0.40 and intercept Q2Y < 0.05 → geometric validation [Wold 1984]",
        "BCa 95% CI of Bal.Acc does not include 1/n_classes → significant discrimination [Efron 1993]",
    ]
    for c in criterios:
        pdf.set_x(18)
        pdf.cell(4, 5, "-")
        pdf.multi_cell(173, 5, _a(c))

    # ── SECTION 2: FIGURES ────────────────────────────────────────────────
    if imgs:
        # 2 figures per page, stacked vertically
        fig_w = 175
        fig_h = 100   # max height per figure
        cap_h = 6     # caption height

        for par in range(0, len(imgs), 2):
            pdf.add_page()
            if par == 0:
                pdf.set_font("Helvetica", "B", 14)
                pdf.set_text_color(30, 80, 140)
                pdf.cell(0, 9, "2. Figures", new_x="LMARGIN", new_y="NEXT")
                pdf.set_draw_color(30, 80, 140)
                pdf.set_line_width(0.5)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                pdf.set_line_width(0.2)
                pdf.ln(3)

            # Top figure
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

            # Bottom figure (if it exists)
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

    # ── SECTION 3: REFERENCES ────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 80, 140)
    pdf.cell(0, 9, "3. References", new_x="LMARGIN", new_y="NEXT")
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
                   "Report automatically generated by the Chemometrics Platform v26. "
                   "Engine: pineline_quimiometria_14.py | Interface: app_quimiometria.py. "
                   "GEAAp/UFPA — PIBIC Project.")

    buf = io.BytesIO()
    buf.write(pdf.output())
    buf.seek(0)
    return buf


def _gerar_word_relatorio(pasta: str, projeto: Dict,
                           max_figuras: int = 14) -> io.BytesIO:
    """
    Generates an editable Word report (.docx) with python-docx.
    Same structure as the PDF: cover, metrics, figures, references.
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
        "Optimal LVs":             _ex(r"LVs?\s+otim[ao].*?[:=]\s*(\d+)"),
        "p-value (permutation)":   _ex(r"p.?value.*?[:=]\s*([\d.E+-]+)"),
        "Preprocessing":           _ex(r"[Pp]re.?[Pp]rocessamento.*?[:=]\s*([A-Za-z0-9_+]+)"),
        "Hotelling T2 UCL (95%)":  _ex(r"[Hh]otelling.*?[:=]\s*([\d.]+)"),
        "Q-residual UCL (95%)":    _ex(r"Q.residual.*?[:=]\s*([\d.E+-]+)"),
        "n samples (training)":    _ex(r"[Nn]\s+treino.*?[:=]\s*(\d+)"),
        "n classes":               _ex(r"[Nn]\.?\s*[Cc]lasses.*?[:=]\s*(\d+)"),
    }
    imgs = _listar_figuras(pasta)[:max_figuras]

    doc = Document()

    # ── Heading styles ──
    for i, (level, size) in enumerate([(0, 20), (1, 16), (2, 13)]):
        style = doc.styles[f"Heading {i + 1}"]
        style.font.size = Pt(size)           # type: ignore[union-attr]
        style.font.color.rgb = RGBColor(30, 80, 140)  # type: ignore[union-attr]
        style.font.bold = True               # type: ignore[union-attr]

    # ── COVER ──────────────────────────────────────────────────────────────
    doc.add_heading("Chemometrics Platform", 0)
    doc.add_heading(projeto.get("nome", "Chemometric Analysis Report"), 1)

    t_capa = doc.add_table(rows=5, cols=2)
    t_capa.style = "Table Grid"
    campos_capa = [
        ("Author(s)",      projeto.get("autor", "-")),
        ("Institution",    projeto.get("inst", "GEAAp / UFPA")),
        ("Study type",     projeto.get("tipo", "-")),
        ("Date",           time.strftime("%Y-%m-%d %H:%M")),
        ("Folder",         os.path.basename(pasta)),
    ]
    for i, (label, valor) in enumerate(campos_capa):
        c0 = t_capa.cell(i, 0)
        c1 = t_capa.cell(i, 1)
        c0.text = label
        c1.text = str(valor)
        c0.paragraphs[0].runs[0].bold = True

    obj = projeto.get("objetivo", "")
    if obj:
        doc.add_heading("Objective", 2)
        doc.add_paragraph(obj)

    doc.add_page_break()

    # ── SECTION 1: METRICS ─────────────────────────────────────────────────
    doc.add_heading("1. Executive Summary — Model Metrics", 1)

    t_met = doc.add_table(rows=1 + len(metricas), cols=2)
    t_met.style = "Table Grid"
    hdr = t_met.rows[0].cells
    hdr[0].text = "Metric"
    hdr[1].text = "Value (internal CV validation)"
    for cell in hdr:
        run = cell.paragraphs[0].runs
        if run:
            run[0].bold = True
        cell.paragraphs[0].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for i, (k, v) in enumerate(metricas.items(), 1):
        t_met.cell(i, 0).text = k
        t_met.cell(i, 1).text = str(v)

    doc.add_heading("Quality criteria", 2)
    criterios = [
        "R2Y > 0.60 and Q2Y > 0.50: robust predictive model [Eriksson 2008]",
        "Q2Y / R2Y > 0.70: low overfitting risk",
        "permutation p-value < 0.05: non-random result (Y-randomization)",
        "Intercept R2Y < 0.40 and intercept Q2Y < 0.05 [Wold 1984]",
        "BCa 95% CI of Bal.Acc does not include 1/n_classes [Efron 1993]",
    ]
    for c in criterios:
        doc.add_paragraph(c, style="List Bullet")

    # ── SECTION 2: FIGURES ──────────────────────────────────────────────────
    if imgs:
        doc.add_page_break()
        doc.add_heading("2. Figures", 1)
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
                doc.add_paragraph(f"[Fig. {i}: {nome_fig} — image not available]")
            if i % 2 == 0 and i < len(imgs):
                doc.add_page_break()

    # ── SECTION 3: REFERENCES ─────────────────────────────────────────────
    doc.add_page_break()
    doc.add_heading("3. References", 1)
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
        "Report generated by the Chemometrics Platform v26. "
        "Engine: pineline_quimiometria_14.py | GEAAp/UFPA — PIBIC Project."
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
    Generates an Excel report with 4 sheets via openpyxl:
      - Metrics: metrics extracted from the summary
      - Identifiers: samples with T2, Q, class (pipeline CSV)
      - VIP_Selection: VIP/SR scores (pipeline CSV, if present)
      - Raw_Summary: full text of resumo_modelo.txt
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
        "Optimal LVs":             _ex(r"LVs?\s+otim[ao].*?[:=]\s*(\d+)"),
        "p-value (permutation)":   _ex(r"p.?value.*?[:=]\s*([\d.E+-]+)"),
        "Preprocessing":           _ex(r"[Pp]re.?[Pp]rocessamento.*?[:=]\s*([A-Za-z0-9_+]+)"),
        "Hotelling T2 UCL (95%)":  _ex(r"[Hh]otelling.*?[:=]\s*([\d.]+)"),
        "Q-residual UCL (95%)":    _ex(r"Q.residual.*?[:=]\s*([\d.E+-]+)"),
        "n samples (training)":    _ex(r"[Nn]\s+treino.*?[:=]\s*(\d+)"),
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
        """Variant of _preencher_df with configurable row_start."""
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

    # ── SHEET 1: Metrics ───────────────────────────────────────────────────
    ws1 = wb.active
    assert ws1 is not None
    ws1.title = "Metrics"
    _cabecalho(ws1, ["Metric", "Value (internal CV)"])
    for i, (k, v) in enumerate(metricas_dict.items(), 2):
        fill = _AZUL_CLAR if i % 2 == 0 else _BRANCO
        for j, val in enumerate([k, v], 1):
            c = ws1.cell(i, j, val)
            c.font = _BODY_FONT
            c.fill = fill
            c.border = _BORDER
    _auto_width(ws1)

    # ── SHEET 2: Identifiers ────────────────────────────────────────────────
    ws2 = wb.create_sheet("Identifiers")
    id_csv = os.path.join(pasta, "dados", "amostras_identificadores.csv")
    if os.path.exists(id_csv):
        try:
            df_id = pd.read_csv(id_csv, sep=";", decimal=",")
            _preencher_df(ws2, df_id)
            _auto_width(ws2)
        except Exception:
            ws2.cell(1, 1, "Error reading amostras_identificadores.csv")
    else:
        ws2.cell(1, 1, "File not found (run pipeline Step 5 first).")

    # ── SHEET 3: VIP_Selection ────────────────────────────────────────────────
    ws3 = wb.create_sheet("VIP_Selection")
    vip_csv = os.path.join(pasta, "dados", "etapa4_selecao_variaveis.csv")
    if os.path.exists(vip_csv):
        try:
            df_vip = pd.read_csv(vip_csv, sep=";", decimal=",")
            _preencher_df(ws3, df_vip)
            _auto_width(ws3)
        except Exception:
            ws3.cell(1, 1, "Error reading etapa4_selecao_variaveis.csv")
    else:
        ws3.cell(1, 1, "Step 4 (variable selection) was not executed.")

    # ── SHEET 4: Raw_Summary ───────────────────────────────────────────────
    ws4 = wb.create_sheet("Raw_Summary")
    ws4.cell(1, 1, "resumo_modelo.txt — full content").font = Font(bold=True)
    for i, linha in enumerate(resumo_raw.splitlines(), 2):
        ws4.cell(i, 1, linha).font = Font(name="Courier New", size=9)
    ws4.column_dimensions["A"].width = 80

    # ── SHEET 5: Benchmark (if it exists) ─────────────────────────────────
    bench_csv = os.path.join(pasta, "dados", "benchmark_classificadores.csv")
    mc_csv    = os.path.join(pasta, "dados", "monte_carlo_cv.csv")
    if os.path.exists(bench_csv) or os.path.exists(mc_csv):
        ws5 = wb.create_sheet("Benchmark")
        row_cursor = 1
        if os.path.exists(bench_csv):
            try:
                df_bench = pd.read_csv(bench_csv, sep=";", decimal=",")
                ws5.cell(row_cursor, 1,
                         "Auto-Benchmark — Bal.Acc by classifier (GroupKFold)").font = Font(bold=True)
                row_cursor += 1
                _preencher_df_ws(ws5, df_bench, row_start=row_cursor)
                row_cursor += len(df_bench) + 3
                _auto_width(ws5)
            except Exception:
                ws5.cell(row_cursor, 1, "Error reading benchmark_classificadores.csv")
                row_cursor += 2
        if os.path.exists(mc_csv):
            try:
                df_mc = pd.read_csv(mc_csv, sep=";", decimal=",")
                ws5.cell(row_cursor, 1,
                         "Monte Carlo CV — 95% CI by percentile").font = Font(bold=True)
                row_cursor += 1
                _preencher_df_ws(ws5, df_mc, row_start=row_cursor)
                _auto_width(ws5)
            except Exception:
                ws5.cell(row_cursor, 1, "Error reading monte_carlo_cv.csv")

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


def _gerar_latex_template(pasta: str, projeto: Dict) -> bytes:
    """
    Generates a LaTeX template ready for journals (Talanta, Food Chemistry,
    Journal of Chemometrics). Includes auto-filled metrics, \\includegraphics
    blocks for figures, and complete bibliography.
    Returns UTF-8 bytes (.tex file).
    """
    import re as _re

    resumo_raw = _ler_resumo(pasta) or ""

    def _ex(padrao: str, default: str = "-") -> str:
        m = _re.search(padrao, resumo_raw, _re.IGNORECASE | _re.MULTILINE)
        return m.group(1).strip() if m else default

    def _esc(txt: str) -> str:
        """Escapes LaTeX special characters."""
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
    nome_proj = _esc(projeto.get("nome", "Chemometric Analysis by FT-NIR"))
    autor     = _esc(projeto.get("autor", "Surname, N."))
    inst      = _esc(projeto.get("inst", "GEAAp, Federal University of Para"))

    # Metrics table
    linhas_met = [
        r"        \textbf{Metric} & \textbf{Value (internal CV)} \\",
        r"        \midrule",
        f"        Balanced Accuracy & {met['bal_acc']} \\\\",
        f"        AUC macro (OvR)   & {met['auc']} \\\\",
        f"        $R^2Y$            & {met['r2y']} \\\\",
        f"        $Q^2Y$            & {met['q2y']} \\\\",
        f"        $R^2X$            & {met['r2x']} \\\\",
        f"        Optimal LVs       & {met['lvs']} \\\\",
        f"        $p$ (permutation) & {met['perm_p']} \\\\",
        f"        Preprocessing     & {_esc(met['preproc'])} \\\\",
        f"        $n$ training      & {met['n_train']} \\\\",
        f"        $n$ classes       & {met['n_classes']} \\\\",
    ]
    tabela = "\n".join(linhas_met)

    # Figure blocks
    blocos_fig = []
    for i, img in enumerate(imgs, 1):
        nome_f = os.path.splitext(os.path.basename(img))[0]
        label  = nome_f.replace(" ", "_").replace("-", "_")
        # LaTeX prefers forward slashes
        img_tex = img.replace("\\", "/")
        blocos_fig.append(f"""
\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=0.85\\linewidth]{{{img_tex}}}
    \\caption{{{_esc(nome_f)}. % TODO: add chemical interpretation.}}
    \\label{{fig:{label}}}
\\end{{figure}}""")
    figs_block = "\n".join(blocos_fig)

    tex = f"""% ================================================================
% LaTeX Template — Chemometrics Platform v26
% Compatible with: Talanta, Food Chemistry, J. Chemometrics (Elsevier)
% Generated: {time.strftime('%Y-%m-%d %H:%M')}
% ================================================================

\\documentclass[12pt,a4paper]{{article}}

%% Packages
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage[english]{{babel}}
\\usepackage{{amsmath, amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage[colorlinks=true,citecolor=blue,linkcolor=blue]{{hyperref}}
\\usepackage{{geometry}}
\\usepackage{{caption}}
\\usepackage{{float}}
\\usepackage{{siunitx}}
\\usepackage{{natbib}}

\\geometry{{a4paper, left=2.5cm, right=2.5cm, top=3cm, bottom=3cm}}
\\captionsetup{{font=small, labelfont=bf}}
\\setlength{{\\parindent}}{{0.5cm}}

%% Metadata
\\title{{{nome_proj}}}
\\author{{{autor} \\\\
         \\small {inst}}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

%% ── Abstract ─────────────────────────────────────────────────────────────
\\begin{{abstract}}
The authenticity of Amazonian vegetable oils was investigated by FT-NIR
spectroscopy combined with chemometric methods (PLS-DA, PCA, OPLS-DA and DD-SIMCA).
The PLS-DA model with {_esc(met['lvs'])} latent variables and {_esc(met['preproc'])}
preprocessing achieved a balanced accuracy of {met['bal_acc']}
($R^2Y = {met['r2y']}$; $Q^2Y = {met['q2y']}$; $p = {met['perm_p']}$,
Y-randomization), evidencing significant discriminatory power.
% TODO: limit to 250 words for Talanta / 200 for Food Chemistry.
\\end{{abstract}}

\\textbf{{Keywords:}} FT-NIR; PLS-DA; chemometrics; authentication; vegetable oils.

%% ── Introduction ──────────────────────────────────────────────────────────
\\section{{Introduction}}
% TODO: contextualize, justify the method and state the objective in ~3 paragraphs.
Adulteration of vegetable oils is a global food safety problem \\citep{{rinnan2009}}.
FT-NIR spectroscopy combined with chemometrics has shown potential as a rapid,
non-destructive, low-cost technique for oil authentication
\\citep{{bylesjo2006, eriksson2008}}.

%% ── Material and Methods ──────────────────────────────────────────────────
\\section{{Material and Methods}}

\\subsection{{Samples and spectral acquisition}}
% TODO: describe number of samples, instrument (ABB MB3600 or similar),
% spectral range, resolution, number of scans, temperature.
FT-NIR spectra were acquired in the range \\SIrange{{4000}}{{10000}}{{\\per\\centi\\meter}}.

\\subsection{{Spectral preprocessing}}
The {_esc(met['preproc'])} preprocessing was selected based on pipeline comparison
(balanced accuracy in CV validation) according to \\citet{{rinnan2009}}.

\\subsection{{PLS-DA modelling}}
The PLS-DA model was calibrated with {met['lvs']} latent variables, selected
by group-aware cross-validation (GroupKFold, grouping technical replicates to
prevent data leakage) \\citep{{chong2005}}.

\\subsection{{Statistical validation}}
Robustness was assessed by:
\\begin{{itemize}}
    \\item \\textbf{{Y-randomization}} (200 permutations) \\citep{{eriksson2008}};
    \\item \\textbf{{Wold's test}} (intercepts $R^2Y < 0.40$ and $Q^2Y < 0.05$);
    \\item \\textbf{{CV-ANOVA}} by Eriksson \\citep{{eriksson2008}};
    \\item \\textbf{{BCa Bootstrap}} 95\\% for balanced accuracy \\citep{{efron1993}}.
\\end{{itemize}}

%% ── Results and Discussion ──────────────────────────────────────────────
\\section{{Results and Discussion}}

\\subsection{{Model performance}}
Performance metrics are presented in Table~\\ref{{tab:metricas}}.

\\begin{{table}}[htbp]
    \\centering
    \\caption{{PLS-DA model performance metrics.}}
    \\label{{tab:metricas}}
    \\begin{{tabular}}{{ll}}
        \\toprule
{tabela}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}

\\subsection{{Figures}}
% Figures generated by the pipeline. Insert chemical interpretation in each caption.

{figs_block}

%% ── Conclusion ───────────────────────────────────────────────────────────
\\section{{Conclusion}}
% TODO: summarise results and implications in 1-2 paragraphs.
The PLS-DA model with {_esc(met['preproc'])} preprocessing achieved satisfactory
performance ($Q^2Y = {met['q2y']}$; $p = {met['perm_p']}$), demonstrating the
viability of FT-NIR spectroscopy for rapid authentication of Amazonian vegetable oils.

%% ── Acknowledgements ──────────────────────────────────────────────────────
\\section*{{Acknowledgements}}
% TODO: CNPq, CAPES, PIBIC/UFPA, laboratory.
To GEAAp/UFPA and CNPq for financial support (PIBIC Project).

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
# Execution with live feedback (background thread + progress bar)
# ──────────────────────────────────────────────────────────────────────────

class _LogThreadSafe:
    """Captures pipeline stdout/stderr into a lock-protected list."""
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
    0: "Validating input",
    1: "Spectral preprocessing",
    2: "Latent variable (LV) selection",
    3: "Exploratory PCA",
    4: "Validation tests (permutation / Wold / CV-ANOVA)",
    5: "Final metrics + bootstrap CI",
    6: "Figures, DD-SIMCA, OPLS-DA, holdout",
    7: "Regression / finalization and model saved",
}
# Sub-steps after step 7 (benchmark / MC CV)
_ETAPA_SUBSTEP = {
    "[7b/7]": "Auto-Benchmark (SVM / RF / XGBoost vs PLS-DA)...",
    "[7c/7]": "Monte Carlo CV (95% CI by percentile)...",
}


def _progresso_do_log(txt: str):
    achados = _RE_ETAPA.findall(txt)
    if not achados:
        return 0.0, "Starting..."
    n = max(int(a) for a in achados)
    nome = _ETAPA_NOMES.get(n, f"Step {n}/7")
    # Heavy sub-steps: show specific name for benchmark and MC CV
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
# Spectra preview (cached by path)
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=120)
def _preview_espectros_dx(pasta: str, wn_min: float, wn_max: float,
                           max_por_classe: int = 5):
    """Loads up to max_por_classe samples per subfolder for visualization."""
    try:
        subpastas = sorted(
            p for p in Path(pasta).iterdir()
            if p.is_dir() and list(p.glob("*.dx"))
        )
        if not subpastas:
            # flat folder with .dx files
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
    """Loads up to max_n CSV rows for visualization."""
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
    """Plots mean ± std per class."""
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
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorbance")
    if titulo:
        ax.set_title(titulo, fontsize=9)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(color="0.93", lw=0.5)
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Prediction on unknown samples
# ──────────────────────────────────────────────────────────────────────────

def _predizer(pkg: Dict, X_new_raw: np.ndarray,
              wn_new: Optional[np.ndarray]) -> pd.DataFrame:
    """
    Applies the saved model package to new spectra.
    Interpolates to the training reference axis, applies preprocessor,
    computes predicted class (softmax-normalized), T2 and Q residuals.
    Returns a DataFrame with per-sample diagnostics.
    """
    from scipy.interpolate import interp1d

    preproc = pkg["preprocessador"]
    pls     = pkg["pls_final"]
    lb      = pkg["label_binarizer"]
    wn_train = np.asarray(pkg["wavenumbers"], dtype=float)
    if wn_new is None:
        raise ValueError("wn_new cannot be None")
    wn_min   = float(pkg.get("wn_min", wn_train.min()))
    wn_max   = float(pkg.get("wn_max", wn_train.max()))

    # Training reference axis (range used during training)
    mask_ref = (wn_train >= wn_min) & (wn_train <= wn_max)
    wn_ref   = wn_train[mask_ref]

    # Interpolate new spectra onto training axis
    X_interp = np.zeros((X_new_raw.shape[0], len(wn_ref)))
    for i in range(X_new_raw.shape[0]):
        f = interp1d(wn_new.astype(float), X_new_raw[i].astype(float),
                     kind="linear", bounds_error=False, fill_value="extrapolate")  # type: ignore
        X_interp[i] = f(wn_ref)

    # Apply training preprocessing
    X_proc = preproc.transform(X_interp)

    # PLS scores (applies internal centering of the model)
    T_new  = np.asarray(pls.transform(X_proc), dtype=float)
    P      = np.asarray(pls.x_loadings_, dtype=float)   # (p, k)
    P_T    = P.T                                          # (k, p)

    # Hotelling T2 — same formula as pipeline (scaled by training variance)
    T_train = np.asarray(pls.x_scores_, dtype=float)
    var_t   = T_train.var(axis=0, ddof=1)
    var_t[var_t == 0] = 1.0
    T2_new  = np.sum((T_new ** 2) / var_t, axis=1)

    # Q residuals — same convention as pipeline: X_proc not subtracted
    X_rec  = T_new @ P_T                                  # (n_new, p)
    Q_new  = np.sum((X_proc - X_rec) ** 2, axis=1)

    # UCL from package (generated by pipeline v25+) or conservative fallback
    t2_ucl = float(pkg.get("t2_ucl", np.percentile(
        np.sum((T_train ** 2) / var_t, axis=1), 95)))
    q_ucl  = float(pkg.get("q_ucl", np.percentile(Q_new, 99) * 1.5
                            if len(Q_new) > 0 else 1e6))

    # Class prediction via softmax-normalized PLS scores
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
# Initial state
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
# Header
# ──────────────────────────────────────────────────────────────────────────

st.title("🧪 Chemometrics Platform")
st.caption(
    "PLS-DA · PCA · OPLS-DA · DD-SIMCA · variable selection · "
    "group-aware validation (anti-leakage of replicates). "
    "FT-NIR (.dx) or CSV table (Raman, UV-Vis, FTIR, chromatography…)."
)

# ──────────────────────────────────────────────────────────────────────────
# 7 Tabs
# ──────────────────────────────────────────────────────────────────────────

(tab_proj, tab_dados, tab_preproc, tab_modelo,
 tab_valid, tab_pred, tab_rel) = st.tabs([
    "📋 Project",
    "📂 Data",
    "⚗️ Preprocessing",
    "🧮 Model",
    "📊 Validation",
    "🔮 Prediction",
    "📄 Reports",
])

valores: Dict = {}  # accumulated by widgets from each tab

# ==========================================================================
#  TAB 1 — PROJECT
# ==========================================================================
def _hardware_status_widget():
    """Displays hardware panel with compatibility alerts."""
    try:
        hw = pq.hardware_probe()
        ram_t = hw["ram_total_gb"]
        ram_l = hw["ram_livre_gb"]
        cpu_f = hw["cpu_fisicos"]
        cpu_l = hw["cpu_logicos"]
        disco = hw["disco_livre_gb"]
        psutil_ok = hw["psutil_ok"]

        # RAM traffic-light color
        if ram_l < 2.0:
            cor_ram = "🔴"
            dica = "Critical RAM. Disable Benchmark, SHAP and MC CV."
        elif ram_l < 4.0:
            cor_ram = "🟠"
            dica = "Low RAM. Benchmark and SHAP will be disabled automatically."
        elif ram_l < 8.0:
            cor_ram = "🟡"
            dica = "Moderate RAM. Limits will be adjusted automatically."
        else:
            cor_ram = "🟢"
            dica = "Sufficient RAM for all operations."

        c_hw1, c_hw2, c_hw3 = st.columns(3)
        with c_hw1:
            st.metric("Total RAM", f"{ram_t:.1f} GB",
                      delta=f"{cor_ram} {ram_l:.1f} GB free",
                      delta_color="off")
        with c_hw2:
            st.metric("CPU", f"{cpu_f} cores",
                      delta=f"{cpu_l} logical threads",
                      delta_color="off")
        with c_hw3:
            st.metric("Free disk", f"{disco:.0f} GB",
                      delta="working folder",
                      delta_color="off")

        if ram_l < 8.0:
            st.warning(f"**Limited hardware detected.** {dica}")
        if not psutil_ok:
            st.caption("⚠️ psutil not available — approximate readings. "
                       "Install with `pip install psutil`.")
    except Exception:
        st.caption("Hardware: could not detect hardware specifications.")


with tab_proj:
    st.subheader("Project Identification")
    with st.expander("💻 Hardware Status", expanded=False):
        _hardware_status_widget()
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Project name", key="proj_nome",
                      placeholder="e.g.: Authentication of Amazonian Oils FT-NIR")
        st.text_input("Author(s)", key="proj_autor",
                      placeholder="e.g.: Silva, J.A.; Costa, M.B.")
        st.text_input("Institution / Laboratory", key="proj_inst",
                      placeholder="e.g.: GEAAp / UFPA")
    with c2:
        st.selectbox("Study type",
                     ["Species classification",
                      "Authentication (pure vs. adulterated)",
                      "Quantification (regression)",
                      "Other"],
                     key="proj_tipo")
        st.text_area("Objective", key="proj_objetivo", height=120,
                     placeholder="Describe the objective of the chemometric analysis...")

    st.divider()
    st.markdown("**Save / Export identification**")
    if st.button("💾 Save identification to session"):
        st.success("Identification saved. It will be included in the reports for this session.")

    run_proj = st.session_state.get("proj_nome", "")
    if run_proj:
        st.info(f"Active project: **{run_proj}** — {st.session_state.get('proj_tipo', '')}")

# ==========================================================================
#  TAB 2 — DATA
# ==========================================================================
with tab_dados:
    st.subheader("Data Input")

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

    # ---- CSV Upload (additional option to local path) -----------------
    st.divider()
    st.markdown("**Upload CSV** *(alternative to the local path above)*")
    upld = st.file_uploader(
        "Drag or select a CSV file",
        type=["csv", "txt"],
        key="csv_upload_widget",
        help="The file will be saved to a temporary folder and the path adjusted automatically.",
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
        st.success(f"File saved: `{tmp_path}`")
        st.info("Mode automatically set to 'csv'. "
                "The path above will be overridden when running.")

    # ---- Data statistics preview -----------------------------------------
    st.divider()
    st.markdown("**Data preview**")
    cfg_prev, _ = _coletar_config(cfg_base, valores)
    ok_dados, msg_dados = pq._validar_pasta_dados(cfg_prev)
    (st.success if ok_dados else st.warning)(f"Status: {msg_dados}")

    if ok_dados and st.button("🔍 Load spectra preview", key="btn_prev_dados"):
        modo = cfg_prev.modo
        wn_mn = float(cfg_prev.wn_min)
        wn_mx = float(cfg_prev.wn_max)
        with st.spinner("Loading spectra sample..."):
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
            st.markdown(f"**{len(X_p)} spectra** · {len(cls_u)} classes: "
                        f"`{'`, `'.join(cls_u[:8])}`"
                        + (" ..." if len(cls_u) > 8 else ""))
            fig_p = _plot_espectros_media(wn_p, X_p, np.asarray(labs_p), titulo="Raw spectra (sample)")
            st.pyplot(fig_p, use_container_width=True)
            plt.close(fig_p)
        else:
            st.warning("Could not load spectra for preview. "
                       "Check the path/mode.")

    # ---- Save / Reload config.yaml ---------------------------------
    st.divider()
    cfg_dados, erros_dados = _coletar_config(cfg_base, valores)
    if erros_dados:
        st.warning("Fields with errors:\n- " + "\n- ".join(erros_dados))
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        if st.button("💾 Save config.yaml", key="btn_salvar_cfg_dados",
                     use_container_width=True):
            if erros_dados:
                st.error("Fix the fields before saving.")
            else:
                pq.salvar_config(cfg_dados, _CFG_PATH)
                st.session_state.cfg_base = cfg_dados
                st.success(f"Saved to {_CFG_PATH}")
    with c_s2:
        if st.button("↺ Reload config.yaml", key="btn_reload_cfg_dados",
                     use_container_width=True):
            try:
                st.session_state.cfg_base = pq.carregar_config(_CFG_PATH)
                cfg_base = st.session_state.cfg_base
                st.success("Config reloaded.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================================================
#  TAB 3 — PREPROCESSING
# ==========================================================================
with tab_preproc:
    st.subheader("Spectral Preprocessing")

    _PREPROC_KEYS = ["pre_processamento", "faixa_min_cm", "faixa_max_cm"]

    col_p1, col_p2 = st.columns(2)
    for i, k in enumerate(_PREPROC_KEYS):
        s = specs.get(k)
        if s is None:
            continue
        with (col_p1 if i % 2 == 0 else col_p2):
            valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    # Information about each preset
    preset_selecionado = valores.get("pre_processamento", "")
    _PRESET_INFO = {
        "MSC+SG+MC":      "MSC (scatter correction) → 1st derivative SG (Savitzky-Golay) → Mean-Centering. **Best for FT-NIR with pronounced scatter.** Acc=0.923 on 1807 Amazonian oil samples.",
        "SNV+SG+MC":      "SNV (variance normalization) → SG → Mean-Centering. Robust alternative to MSC when global reference is not stable.",
        "Autoscaling":    "Mean-Centering + division by standard deviation. **Caution**: collapses spectral noise when SG is not applied first.",
        "Mean-centering": "Mean centering only. Recommended as a comparative baseline.",
    }
    if preset_selecionado in _PRESET_INFO:
        st.info(_PRESET_INFO[preset_selecionado])

    # ---- Before/after preview -------------------------------------------
    st.divider()
    st.markdown("**Before / after preprocessing visualization**")
    cfg_pp, _ = _coletar_config(cfg_base, valores)
    ok_pp, _ = pq._validar_pasta_dados(cfg_pp)

    if not ok_pp:
        st.info("Configure and validate data input (Data tab) to enable the preview.")
    elif st.button("⚗️ Generate before/after preview", key="btn_prev_preproc"):
        with st.spinner("Loading and processing spectra..."):
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
                    wn_raw, X_raw, labs_raw_arr, "Before preprocessing")
                fig_depois = _plot_espectros_media(
                    wn_raw, X_proc_pp, labs_raw_arr,
                    f"After: {preset_selecionado}")
                col_ant, col_dep = st.columns(2)
                with col_ant:
                    st.pyplot(fig_antes, use_container_width=True)
                    plt.close(fig_antes)
                with col_dep:
                    st.pyplot(fig_depois, use_container_width=True)
                    plt.close(fig_depois)
            except Exception as e_pp:
                st.error(f"Error applying preprocessing: {e_pp}")
        else:
            st.warning("Could not load spectra. Check the Data tab.")

# ==========================================================================
#  TAB 4 — MODEL (advanced parameters + execution)
# ==========================================================================
with tab_modelo:
    st.subheader("Model Parameters and Execution")

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

    with st.expander("Analysis and partitioning", expanded=True):
        cols_a = st.columns(2)
        for i, k in enumerate(_MODELO_KEYS_ANALISE):
            s = specs.get(k)
            if s is None: continue
            with cols_a[i % 2]:
                valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    with st.expander("Statistical validation", expanded=False):
        cols_v = st.columns(2)
        for i, k in enumerate(_MODELO_KEYS_VALID):
            s = specs.get(k)
            if s is None: continue
            with cols_v[i % 2]:
                valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    with st.expander("Extra modules", expanded=False):
        # Hardware warning for heavy operations
        try:
            _hw_mod = pq.hardware_probe()
            _ram_l  = _hw_mod.get("ram_livre_gb", 16.0)
            if _ram_l < 4.0:
                st.error(
                    f"⚠️ Free RAM: **{_ram_l:.1f} GB** — "
                    "Benchmark and SHAP will be **automatically disabled** "
                    "by the pipeline to prevent freezing.")
            elif _ram_l < 8.0:
                st.warning(
                    f"⚠️ Free RAM: **{_ram_l:.1f} GB** — "
                    "SHAP and Monte Carlo CV limits will be adjusted "
                    "automatically. Closing other programs is recommended.")
        except Exception:
            pass

        cols_e = st.columns(2)
        for i, k in enumerate(_MODELO_KEYS_EXTRAS):
            s = specs.get(k)
            if s is None: continue
            with cols_e[i % 2]:
                valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    with st.expander("Figures", expanded=False):
        cols_f = st.columns(2)
        for i, k in enumerate(_MODELO_KEYS_FIGURAS):
            s = specs.get(k)
            if s is None: continue
            with cols_f[i % 2]:
                valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    st.divider()

    # ---- Final Config assembly and execution ----------------------------
    cfg_run, erros_run = _coletar_config(cfg_base, valores)

    # If user uploaded a CSV, override the path
    if st.session_state.get("_csv_upload_path"):
        csv_upld_path = st.session_state["_csv_upload_path"]
        if os.path.exists(csv_upld_path):
            cfg_run.modo = "csv"
            cfg_run.arquivo_csv = csv_upld_path

    ok_run, msg_run = pq._validar_pasta_dados(cfg_run)
    st.write("**Input:**", msg_run)
    if erros_run:
        st.error("Invalid fields in configuration:\n- " + "\n- ".join(erros_run))

    pode_rodar = ok_run and not erros_run
    rodar = st.button("▶️ Run pipeline", type="primary",
                      disabled=not pode_rodar, use_container_width=True,
                      key="btn_rodar")
    if not pode_rodar:
        st.info("Fix the data input (Data tab) to enable.")

    if rodar:
        try:
            pq.salvar_config(cfg_run, _CFG_PATH)
        except Exception:
            pass

        logger = _LogThreadSafe(tee=sys.__stdout__)
        estado: Dict = {"fim": False, "erro": None, "pasta": None}
        worker = threading.Thread(
            target=_rodar_worker, args=(cfg_run, logger, estado), daemon=True)

        t0 = time.monotonic()
        eta_best: Optional[float] = None
        # Max runtime guard: 2 hours (prevents infinite blocking on Cloud)
        _MAX_RUNTIME = 7200

        worker.start()

        # Use st.status() + single placeholder to avoid simultaneous DOM
        # mutations that cause React's removeChild error on Streamlit Cloud.
        with st.status("⚙️ Running pipeline...", expanded=True) as _run_status:
            ph = st.empty()
            while not estado["fim"]:
                # Safety timeout
                if time.monotonic() - t0 > _MAX_RUNTIME:
                    estado["fim"] = True
                    estado["erro"] = "Pipeline exceeded maximum runtime (2 h)."
                    break
                txt = logger.text()
                frac, nome = _progresso_do_log(txt)
                elapsed = time.monotonic() - t0
                if frac >= 0.10:
                    eta = elapsed / frac - elapsed
                    eta_best = eta if eta_best is None else min(eta_best, eta)
                ram = _ram_mb()
                linhas = txt.strip().splitlines()
                log_tail = "\n".join(linhas[-10:]) if linhas else "..."
                # Single atomic DOM update per iteration
                ph.markdown(
                    f"**[{int(frac * 100)}%] {nome}**\n\n"
                    f"⏱️ `{_fmt_tempo(elapsed)}` elapsed  |  "
                    f"⏳ `{_fmt_tempo(eta_best) if eta_best else 'calculating…'}` remaining  |  "
                    f"💾 `{f'{ram:.0f} MB' if ram else 'n/a'}`\n\n"
                    f"```text\n{log_tail}\n```"
                )
                time.sleep(0.5)

            # Final state update
            txt     = logger.text()
            elapsed = time.monotonic() - t0
            ph.empty()
            if estado["erro"]:
                _run_status.update(
                    label=f"❌ Pipeline failed after {_fmt_tempo(elapsed)}.",
                    state="error", expanded=True)
            else:
                _run_status.update(
                    label=f"✅ Completed in {_fmt_tempo(elapsed)}!",
                    state="complete", expanded=False)

        st.session_state.ultimo_log  = txt
        if estado["erro"]:
            st.session_state.erro_run  = estado["erro"]
            st.session_state.ultima_pasta = None
            st.error(f"Pipeline failed after {_fmt_tempo(elapsed)}.")
        else:
            st.session_state.erro_run  = None
            st.session_state.ultima_pasta = estado["pasta"]
            st.success("✅ Completed! View results in the Validation and Reports tabs.")

    if st.session_state.get("erro_run"):
        st.subheader("Error traceback")
        st.code(st.session_state.erro_run, language="text")

# ==========================================================================
#  TAB 5 — VALIDATION
# ==========================================================================
with tab_valid:
    st.subheader("Validation Results")
    pasta_v = st.session_state.get("ultima_pasta")

    if not pasta_v or not os.path.isdir(pasta_v):
        st.info("Run the pipeline (Model tab) to view results here.")
    else:
        st.caption(f"Folder: `{os.path.abspath(pasta_v)}`")

        # Numeric summary
        resumo_txt = _ler_resumo(pasta_v)
        if resumo_txt:
            with st.expander("📋 Model summary (resumo_modelo.txt)", expanded=True):
                st.code(resumo_txt, language="text")

        # ── Per-Class Accuracy table (extracted from summary) ───────────────
        if resumo_txt:
            import re as _re_acc
            acc_map: Dict[str, float] = {}
            for _linha in resumo_txt.splitlines():
                _m = _re_acc.match(r"\s*Acc\s+(.+?)\s*[:=]\s*([\d.]+)", _linha)
                if _m:
                    acc_map[_m.group(1).strip()] = float(_m.group(2))
            if acc_map:
                with st.expander("📊 Accuracy by Class", expanded=True):
                    _df_acc = pd.DataFrame(
                        list(acc_map.items()), columns=["Class", "Accuracy (recall)"]
                    ).sort_values("Accuracy (recall)")
                    # Color: red < 0.7, yellow 0.7-0.9, green >= 0.9
                    def _cor_acc(v: object) -> str | None:  # Scalar compat
                        try:
                            fv = float(v)  # type: ignore[arg-type]
                        except (TypeError, ValueError):
                            return None
                        if fv >= 0.90: return "background-color:#d4edda"
                        if fv >= 0.70: return "background-color:#fff3cd"
                        return "background-color:#f8d7da"
                    st.dataframe(
                        _df_acc.style.map(
                            _cor_acc, subset=["Accuracy (recall)"]),
                        use_container_width=True, height=min(400, 35 * len(_df_acc) + 38))

        # ── Benchmark and MC CV tables (if they exist) ───────────────────
        _bench_csv_v = os.path.join(pasta_v, "dados",
                                    "benchmark_classificadores.csv")
        _mc_csv_v    = os.path.join(pasta_v, "dados", "monte_carlo_cv.csv")
        if os.path.exists(_bench_csv_v) or os.path.exists(_mc_csv_v):
            with st.expander("🏅 Classifier Benchmark", expanded=False):
                if os.path.exists(_bench_csv_v):
                    try:
                        _df_b = pd.read_csv(_bench_csv_v, sep=";", decimal=",")
                        st.markdown("**Auto-Benchmark — Balanced Accuracy by model (GroupKFold)**")
                        st.dataframe(_df_b, use_container_width=True)
                    except Exception as _e_b:
                        st.warning(f"Error reading benchmark CSV: {_e_b}")
                if os.path.exists(_mc_csv_v):
                    try:
                        _df_mc = pd.read_csv(_mc_csv_v, sep=";", decimal=",")
                        st.markdown("**Monte Carlo CV — 95% CI by percentile**")
                        st.dataframe(_df_mc, use_container_width=True)
                    except Exception as _e_mc:
                        st.warning(f"Error reading MC CV CSV: {_e_mc}")

        # Gallery filtered by figure category
        imgs_v = _listar_figuras(pasta_v)
        if imgs_v:
            st.markdown(f"**{len(imgs_v)} figures generated**")
            _CATS = {
                "All":              "",
                "PCA":              "pca",
                "PLS-DA scores":    "plsda",
                "Outliers (T²/Q)":  "outlier",
                "Confusion":        "confus",
                "ROC / AUC":        "roc",
                "VIP / SR":         "vip",
                "Loading":          "loading",
                "HCA":              "hca",
                "OPLS-DA":          "opls",
                "DD-SIMCA":         "ddsimca",
                "Cooman's Plot":    "cooman",
                "S-Plot":           "splot",
                "Permutation":      "permut",
                "Wold":             "wold",
                "Benchmark":        "benchmark",
                "Monte Carlo CV":   "monte_carlo",
                "DET curves":       "fig_det",
                "SHAP":             "fig_shap",
            }
            filtro_v = st.selectbox("Filter by category",
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
                st.info(f"No figures found for '{filtro_v}'.")
        else:
            st.info("Figures saved in non-displayable format (PDF/SVG). "
                    "Use the download button in the Reports tab.")

# ==========================================================================
#  TAB 6 — PREDICTION
# ==========================================================================
with tab_pred:
    st.subheader("Prediction on Unknown Samples")
    st.markdown(
        "Upload a `.joblib` model generated by the pipeline and a CSV "
        "with new spectra (columns = wavenumbers, no class column)."
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**1. Trained model (.joblib)**")
        upld_jbl = st.file_uploader("Upload the .joblib model",
                                    type=["joblib", "pkl"],
                                    key="pred_model_upload")
        cam_jbl  = st.text_input("Or local path to model",
                                  key="pred_model_path",
                                  placeholder="C:/results/model_pls.joblib")

    with col_m2:
        st.markdown("**2. New spectra (CSV)**")
        upld_csv_pred = st.file_uploader("Upload CSV with new spectra",
                                          type=["csv", "txt"],
                                          key="pred_csv_upload")
        cam_csv_pred  = st.text_input("Or local path to CSV",
                                       key="pred_csv_path",
                                       placeholder="C:/data/new_spectra.csv")
        col_wn_pred   = st.text_input(
            "First column to use as wavenumber (leave empty = auto)",
            key="pred_col_wn",
            placeholder="e.g.: 4000.0  (name or index of first spectral column)")

    # ---- Predict button -------------------------------------------------
    st.divider()
    if st.button("🔮 Predict", type="primary", key="btn_predizer",
                 use_container_width=True):
        erros_pred: List[str] = []
        pkg_pred = None

        # Load model
        try:
            import joblib
            if upld_jbl is not None:
                tmp_jbl = Path(tempfile.gettempdir()) / "pq_pred_model.joblib"
                with open(tmp_jbl, "wb") as f:
                    f.write(upld_jbl.getvalue())
                pkg_pred = joblib.load(str(tmp_jbl))
            elif cam_jbl and os.path.exists(cam_jbl):
                pkg_pred = joblib.load(cam_jbl)
            # Minimal structure validation to avoid RCE via malicious pickle
            if pkg_pred is not None:
                _chaves_req = {"preprocessador", "pls_final", "label_binarizer", "wavenumbers"}
                if not _chaves_req.issubset(pkg_pred.keys()):
                    raise ValueError(
                        f"Invalid model: expected keys {_chaves_req}, "
                        f"found {set(pkg_pred.keys())}"
                    )
            else:
                erros_pred.append("No valid model provided (upload or path).")
        except Exception as e_jbl:
            erros_pred.append(f"Error loading model: {e_jbl}")

        # Load prediction CSV
        X_pred_raw = None
        wn_pred    = None
        try:
            if upld_csv_pred is not None:
                df_pred = pd.read_csv(upld_csv_pred, sep=None, engine="python")
            elif cam_csv_pred and os.path.exists(cam_csv_pred):
                df_pred = pd.read_csv(cam_csv_pred, sep=None, engine="python")
            else:
                erros_pred.append("No spectra CSV provided.")
                df_pred = None

            if df_pred is not None:
                # Detect numeric columns (wavenumbers)
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
                        "No columns with numeric names (wavenumbers) were found. "
                        "Ensure that the spectral column headers are wavenumbers "
                        "(e.g.: 4000.5, 4001.0...).")
        except Exception as e_csv:
            erros_pred.append(f"Error reading CSV: {e_csv}")

        if erros_pred:
            for e in erros_pred:
                st.error(e)
        elif pkg_pred is not None and X_pred_raw is not None:
            try:
                with st.spinner("Applying model..."):
                    df_res = _predizer(pkg_pred, X_pred_raw, wn_pred)
                    # Re-attach metadata (sample names) if available
                    meta_df = st.session_state.get("pred_amostras")
                    if meta_df is not None and len(meta_df) == len(df_res):
                        df_res = pd.concat(
                            [meta_df.reset_index(drop=True), df_res], axis=1)
                st.session_state["pred_resultados"] = df_res
                st.success(f"Prediction complete: {len(df_res)} samples.")
            except Exception as e_pred:
                st.error(f"Prediction error: {e_pred}")

    # ---- Results display ----------------------------------------
    df_show = st.session_state.get("pred_resultados")
    if df_show is not None:
        st.divider()
        st.markdown("**Prediction results**")

        # Color accepted/rejected
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
            "⬇️ Download results (.csv)",
            data=csv_bytes,
            file_name="predicao_resultados.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Quick summary
        if "aceito" in df_show.columns:
            n_ac = int(df_show["aceito"].sum())
            n_tot = len(df_show)
            st.metric("Accepted samples (T² ≤ UCL and Q ≤ UCL)",
                      f"{n_ac} / {n_tot}",
                      delta=f"{n_ac/n_tot*100:.1f}%")

def _gerar_pptx_relatorio(pasta: str, projeto: Dict,
                           max_figuras: int = 12) -> io.BytesIO:
    """
    Generates a PowerPoint presentation with professional scientific design.
    Palette: Navy #0F172A + Accent #0369A1 (UI Pro Max — B2B Professional).
    Structure: Cover | Methodology | Metrics | Figures | Benchmark | Conclusions
    """
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    import re as _re

    # ── UI Pro Max — B2B Professional palette ──────────────────────────────
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
        "Optimal LVs":            _ex(r"LVs?\s+otim.*?[:=]\s*(\d+)"),
        "Preprocessing":          _ex(r"[Pp]re.?[Pp]rocess.*?[:=]\s*([A-Za-z0-9_+]+)"),
        "N samples":              _ex(r"[Nn]\s+treino.*?[:=]\s*(\d+)"),
        "N classes":              _ex(r"[Nn]\.?\s*[Cc]lasses.*?[:=]\s*(\d+)"),
        "p permutation":          _ex(r"p.?value.*?[:=]\s*([\d.E+-]+)"),
    }

    # ── Layout helpers ─────────────────────────────────────────────────────
    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)
    W = prs.slide_width   # Emu — assigned above, never None
    H = prs.slide_height  # Emu — assigned above, never None
    assert W is not None and H is not None  # satisfies Pylance Optional[Emu] stub

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
        data_str = time.strftime("%Y-%m-%d")
        inst = projeto.get("inst", "GEAAp / UFPA")
        _txt(slide,
             f"{inst}  •  Chemometrics Platform  •  {data_str}",
             int(Inches(0.3)), int(H - Inches(0.35)),
             int(W - Inches(0.6)), int(Inches(0.35)),
             size=9, color=_LIGHT, align=PP_ALIGN.CENTER)

    # ── SLIDE 1: Cover ─────────────────────────────────────────────────────
    slide1 = _slide_novo()
    _fundo(slide1, _NAVY)
    # Horizontal accent band
    _rect(slide1, 0, int(Inches(3.5)), int(W), int(Inches(0.12)), _ACCENT)
    # Main title
    _txt(slide1, projeto.get("nome", "Chemometrics Platform"),
         int(Inches(1.0)), int(Inches(1.2)),
         int(W - Inches(2.0)), int(Inches(1.5)),
         bold=True, size=36, color=_WHITE)
    # Subtitle
    _txt(slide1, projeto.get("tipo", "Chemometric Analysis FT-NIR"),
         int(Inches(1.0)), int(Inches(2.8)),
         int(W - Inches(2.0)), int(Inches(0.8)),
         size=20, color=RGBColor(0xCB, 0xD5, 0xE1))
    # Metadata
    autor  = projeto.get("autor", "")
    inst   = projeto.get("inst", "GEAAp / UFPA")
    data_s = time.strftime("%Y-%m-%d")
    _txt(slide1, f"{autor}\n{inst}\n{data_s}",
         int(Inches(1.0)), int(Inches(4.0)),
         int(Inches(6.0)), int(Inches(2.0)),
         size=14, color=RGBColor(0x94, 0xA3, 0xB8))

    # ── SLIDE 2: Methodology ──────────────────────────────────────────────
    slide2 = _slide_novo()
    _fundo(slide2)
    _barra_topo(slide2, "Methodology")
    _rodape(slide2)

    metod_items = [
        f"Preprocessing: {metricas['Preprocessing']}",
        f"Main classifier: PLS-DA ({metricas['Optimal LVs']} optimal LVs)",
        f"Cross-validation: GroupKFold anti-leakage of replicates (mae_id)",
        f"Statistical tests: Y-permutation, Wold (R2Y/Q2Y), CV-ANOVA",
        f"Variable selection: iPLS, VIP >= 1, SR top-20%, sPLS-DA",
        f"External validation: stratified holdout (pure samples always in training)",
        f"Benchmark: PLS-DA vs SVM RBF vs RF vs GBM vs XGBoost",
        f"Data: {metricas['N samples']} samples  |  {metricas['N classes']} classes",
    ]
    for i, item in enumerate(metod_items):
        _txt(slide2, f"• {item}",
             int(Inches(0.7)), int(Inches(1.3) + i * Inches(0.62)),
             int(W - Inches(1.4)), int(Inches(0.6)),
             size=13, color=_SLATE)

    # ── SLIDE 3: Performance Metrics ───────────────────────────────────────
    slide3 = _slide_novo()
    _fundo(slide3)
    _barra_topo(slide3, "Results — Performance Metrics")
    _rodape(slide3)

    met_exibir = [
        ("Balanced Accuracy (CV)", metricas["Balanced Accuracy (CV)"], _ACCENT),
        ("AUC macro OvR",          metricas["AUC macro OvR"],          _NAVY),
        ("R2Y",                    metricas["R2Y"],                    _SLATE),
        ("Q2Y",                    metricas["Q2Y"],                    _SLATE),
        ("Optimal LVs",            metricas["Optimal LVs"],            _MUTED),
        ("p permutation",          metricas["p permutation"],          _MUTED),
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

    # ── SLIDES 4+: Figures ────────────────────────────────────────────────
    imgs = _listar_figuras(pasta)
    # Prioritize relevant figures
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
        _barra_topo(slide_f, "Results — Figures")
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

    # ── SLIDE Benchmark (if CSV exists) ──────────────────────────────────
    bench_csv = os.path.join(pasta, "dados", "benchmark_classificadores.csv")
    if os.path.exists(bench_csv):
        try:
            df_b = pd.read_csv(bench_csv, sep=";", decimal=",")
            slide_b = _slide_novo()
            _fundo(slide_b)
            _barra_topo(slide_b, "Classifier Benchmark")
            _rodape(slide_b)
            # Table
            cols_b = list(df_b.columns)
            n_rows = min(len(df_b) + 1, 8)
            col_w  = int((W - Inches(1.0)) / len(cols_b))
            row_h  = int(Inches(0.52))
            # Header
            for ci, col_n in enumerate(cols_b):
                _rect(slide_b, int(Inches(0.5) + ci * col_w),
                      int(Inches(1.3)), col_w - 2, row_h, _NAVY)
                _txt(slide_b, str(col_n),
                     int(Inches(0.5) + ci * col_w + Inches(0.05)),
                     int(Inches(1.35)), col_w - int(Inches(0.1)),
                     int(Inches(0.45)), bold=True, size=10, color=_WHITE)
            # Data rows
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

    # ── Final SLIDE: Conclusions ───────────────────────────────────────────
    slide_c = _slide_novo()
    _fundo(slide_c, _NAVY)
    _rect(slide_c, 0, int(Inches(3.2)), int(W), int(Inches(0.1)), _ACCENT)
    _txt(slide_c, "Conclusions",
         int(Inches(1.0)), int(Inches(1.0)),
         int(W - Inches(2.0)), int(Inches(0.9)),
         bold=True, size=30, color=_WHITE)
    ba   = metricas.get("Balanced Accuracy (CV)", "—")
    auc  = metricas.get("AUC macro OvR", "—")
    r2y  = metricas.get("R2Y", "—"); q2y = metricas.get("Q2Y", "—")
    conc = [
        f"• MSC-SG-MC pipeline achieved Balanced Accuracy = {ba} (GroupKFold anti-leakage)",
        f"• AUC macro OvR = {auc}  |  R2Y = {r2y}  |  Q2Y = {q2y}",
        f"• {metricas['N samples']} samples, {metricas['N classes']} classes "
          f"of Amazonian vegetable oils — FT-NIR spectroscopy",
        f"• Validation: Y-permutation (p = {metricas['p permutation']}), "
          f"Wold, CV-ANOVA and external holdout",
        "• Platform exports .joblib models for prediction of new samples",
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
#  Report cache — avoids regenerating on every Streamlit rerun.
#  Wrappers return bytes (immutable, cacheable); BytesIO is created
#  at download_button time to guarantee cursor at position 0.
# ==========================================================================
@st.cache_data(show_spinner=False)
def _pdf_bytes(pasta: str, proj_items: tuple) -> bytes:
    return _gerar_pdf_relatorio(pasta, dict(proj_items)).read()

@st.cache_data(show_spinner=False)
def _word_bytes(pasta: str, proj_items: tuple) -> bytes:
    return _gerar_word_relatorio(pasta, dict(proj_items)).read()

@st.cache_data(show_spinner=False)
def _excel_bytes(pasta: str) -> bytes:
    return _gerar_excel_relatorio(pasta).read()

@st.cache_data(show_spinner=False)
def _latex_bytes(pasta: str, proj_items: tuple) -> bytes:
    return _gerar_latex_template(pasta, dict(proj_items))

@st.cache_data(show_spinner=False)
def _pptx_bytes(pasta: str, proj_items: tuple) -> bytes:
    return _gerar_pptx_relatorio(pasta, dict(proj_items)).read()


# ==========================================================================
#  TAB 7 — REPORTS
# ==========================================================================
with tab_rel:
    st.subheader("Reports and Downloads")
    pasta_r = st.session_state.get("ultima_pasta")

    if not pasta_r or not os.path.isdir(pasta_r):
        st.info("Run the pipeline (Model tab) to generate reports.")
    else:
        st.caption(f"Results folder: `{os.path.abspath(pasta_r)}`")

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
        # Sorted tuple used as cache key (Dict is not hashable)
        _proj_items = tuple(sorted(_projeto_info.items()))

        # Row 1: ZIP + PDF
        col_a, col_b = st.columns(2)
        with col_a:
            try:
                st.download_button(
                    "📦 Full results (.zip)",
                    data=_zip_da_pasta(pasta_r),
                    file_name=_nome_base + ".zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            except Exception as e_zip:
                st.warning(f"ZIP: {e_zip}")

        with col_b:
            try:
                st.download_button(
                    "📄 PDF Report",
                    data=_pdf_bytes(pasta_r, _proj_items),
                    file_name=_nome_base + "_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                )
            except Exception as e_pdf:
                st.error(f"PDF: {e_pdf}")

        # Row 2: Word + Excel
        col_c, col_d = st.columns(2)
        with col_c:
            try:
                st.download_button(
                    "📝 Word Report (.docx)",
                    data=_word_bytes(pasta_r, _proj_items),
                    file_name=_nome_base + "_report.docx",
                    mime="application/vnd.openxmlformats-officedocument"
                         ".wordprocessingml.document",
                    use_container_width=True,
                )
            except Exception as e_word:
                st.error(f"Word: {e_word}")

        with col_d:
            try:
                st.download_button(
                    "📊 Data in Excel (.xlsx)",
                    data=_excel_bytes(pasta_r),
                    file_name=_nome_base + "_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument"
                         ".spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e_xlsx:
                st.error(f"Excel: {e_xlsx}")

        # Row 3: LaTeX + PowerPoint
        col_e, col_f = st.columns(2)
        with col_e:
            try:
                st.download_button(
                    "🔬 LaTeX Template (Talanta / Food Chemistry / J. Chemom.)",
                    data=_latex_bytes(pasta_r, _proj_items),
                    file_name=_nome_base + "_template.tex",
                    mime="text/plain",
                    use_container_width=True,
                )
            except Exception as e_tex:
                st.error(f"LaTeX: {e_tex}")

        with col_f:
            try:
                from pptx import Presentation as _PPTXCheck  # noqa: F401
                st.download_button(
                    "🎯 PowerPoint Presentation (.pptx)",
                    data=_pptx_bytes(pasta_r, _proj_items),
                    file_name=_nome_base + "_presentation.pptx",
                    mime="application/vnd.openxmlformats-officedocument"
                         ".presentationml.presentation",
                    use_container_width=True,
                )
            except ImportError:
                st.warning(
                    "python-pptx not installed. "
                    "Run: `pip install python-pptx>=1.1`",
                    icon="⚠️",
                )
            except Exception as e_pptx:
                st.error(f"PowerPoint: {e_pptx}")

        st.divider()

        # ── Clean up old results ─────────────────────────────────────────
        with st.expander("🗑️ Free space — Clean up old results",
                         expanded=False):
            _pasta_base_lim = os.path.dirname(pasta_r)
            _pastas_exist = sorted(
                [p for p in os.scandir(_pasta_base_lim) if p.is_dir()],
                key=lambda p: p.stat().st_mtime, reverse=True)
            n_pastas = len(_pastas_exist)
            # Calculate total size
            def _tamanho_pasta_mb(pasta_p: str) -> float:
                tot = 0
                for raiz, _, arqs in os.walk(pasta_p):
                    for a in arqs:
                        try: tot += os.path.getsize(os.path.join(raiz, a))
                        except Exception: pass
                return round(tot / (1024 * 1024), 1)

            st.caption(f"Results folder: `{_pasta_base_lim}`  "
                       f"({n_pastas} runs stored)")
            if n_pastas > 1:
                _manter = st.slider(
                    "Keep N most recent runs",
                    min_value=1, max_value=max(1, n_pastas - 1),
                    value=min(3, n_pastas - 1), key="lim_manter")
                _n_remover = n_pastas - _manter
                _tam_est = sum(
                    _tamanho_pasta_mb(p.path)
                    for p in _pastas_exist[_manter:])
                st.info(
                    f"**{_n_remover}** old run(s) will be removed "
                    f"(~{_tam_est:.0f} MB freed). "
                    f"The current run **will not be affected**.")
                if st.button("🗑️ Confirm cleanup",
                             key="btn_limpar_resultados",
                             type="secondary"):
                    _res = pq.limpar_resultados_antigos(
                        _pasta_base_lim, _manter)
                    if _res["removidas"]:
                        st.success(
                            f"Removed {len(_res['removidas'])} folder(s), "
                            f"freed {_res['liberado_mb']:.0f} MB.")
                    else:
                        st.info("No folders removed.")
                    if _res["erro"]:
                        st.warning(f"Errors: {_res['erro']}")
            else:
                st.info("Only one run stored. Nothing to clean.")

        st.divider()

        # Model summary
        st.markdown("### 📋 Model summary")
        resumo_r = _ler_resumo(pasta_r)
        if resumo_r:
            st.text_area("resumo_modelo.txt", resumo_r, height=400)
        else:
            st.info("File resumo_modelo.txt not found.")

        st.divider()

        # Full gallery with filter
        st.markdown("### 🖼️ Figure gallery")
        imgs_r = _listar_figuras(pasta_r)
        if imgs_r:
            _CATS_R = {
                "All":              "",
                "PCA":              "pca",
                "PLS-DA":           "plsda",
                "Outliers":         "outlier",
                "Confusion":        "confus",
                "ROC / AUC":        "roc",
                "VIP / SR":         "vip",
                "Loading":          "loading",
                "HCA":              "hca",
                "OPLS-DA":          "opls",
                "DD-SIMCA":         "ddsimca",
                "Cooman's Plot":    "cooman",
                "S-Plot":           "splot",
                "Permutation":      "permut",
                "Wold/ANOVA":       "wold",
                "Regression":       "regressao",
                "Benchmark":        "benchmark",
                "Monte Carlo CV":   "monte_carlo",
                "DET curves":       "fig_det",
                "SHAP":             "fig_shap",
            }
            filtro_r = st.selectbox("Filter figures",
                                    list(_CATS_R.keys()), key="filtro_rel")
            token_r  = _CATS_R[filtro_r].lower()
            imgs_filt_r = [im for im in imgs_r
                           if token_r in os.path.basename(im).lower()] \
                          if token_r else imgs_r
            st.caption(f"{len(imgs_filt_r)} figure(s) displayed.")
            n_cols_r = st.slider("Columns", 1, 3, 2, key="slider_cols_rel")
            cols_r   = st.columns(n_cols_r)
            for j, img in enumerate(imgs_filt_r):
                with cols_r[j % n_cols_r]:
                    st.image(img, caption=os.path.basename(img),
                             use_container_width=True)
        else:
            st.info("No PNG/JPG images found in the results folder.")

        st.divider()

        # Execution log
        if st.session_state.get("ultimo_log"):
            with st.expander("📜 Execution log (terminal output)"):
                st.code(st.session_state.ultimo_log, language="text")
