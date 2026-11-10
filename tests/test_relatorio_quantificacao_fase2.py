"""Relatorio (PDF/DOCX/XLSX/PPTX/TEX) de uma execucao de QUANTIFICACAO: mostra as metricas de regressao
(RMSEP, RMSECV, R2cv...) e o pre-processamento completo, e nao mostra o molde de classificacao PLS-DA
(R2Y/Q2Y/R2X, balanced accuracy). O inverso vale para uma execucao de classificacao.

Fase 2 do checkup: antes, o PDF/DOCX trazia R2Y=0, Q2Y=0, R2X=0, "Classes 1", "Balanced accuracy -" e o
pre-processamento cortado em "SNV" (o resto da cadeia sumia)."""
import io
import os

import numpy as np
import pytest

from guaraci import reports
from guaraci.config import NOME_RELATORIOS
from guaraci.resumo_parse import (MODO_CLASSIFICACAO, MODO_QUANTIFICACAO, modo_do_resumo, parse_regressao)

_CAB = """============================================================
  PLS-DA Model Summary
============================================================

  Input mode                      : csv
  Grouping guarantee              : medium
  Total samples                   : 48
  Total variables                 : 200
  Total classes                   : {n_classes}
  Method                          : PLS-DA
  Pre-processamento               : SNV -> mean-centering
  Faixa espectral                 : [4000, 10000]
  Optimal LVs                     : 4
  Validation                      : StableStratifiedGroupKFold n_splits=5
  Nivel                           : {nivel}
  Accuracy (CV)                   : {acc}
  Balanced accuracy               : {acc}
  F1 (macro)                      : {acc}
  R2X                             : {r2x}
  R2Y                             : {r2y}
  Q2                              : {q2}
"""

_REGRESSAO = """
============================================================
  PLS Regression — Analytical Figures of Merit
  (Valderrama, Braga & Poppi, 2009, Quim. Nova 32(5):1278-1287)
============================================================

  Pooled metrics:
    R2cal .....: 0.9998
    R2val .....: 0.9940
    RMSEC .....: 0.070
    RMSECV ....: 0.408
    R2cv ......: 0.9948
    RMSEP .....: 0.473
    Bias ......: 0.1984
    (RMSEP acima = validacao INTERNA: split 70/30 dos dados de
     treino, n pequeno; nao e' o numero publicavel.)
    RMSEP (holdout externo) : 0.829   n=12  R2=0.973  bias=-0.061   <- numero publicavel
    DModY critico (SIMCA): 2.145
    Amostras fora do DModY: 0

  Per-species figures of merit:
    Species            LVs   RMSEP  R2val     LOD     LOQ     SEN    SEL
    --------------------------------------------------------------------
    composto_a           3    0.48  0.990    0.06    0.17   0.008  0.670
      IC(95%): LOD [0.04-0.10]  LOQ [0.12-0.31]  |  validacao: [1.84, 12.50]% DP=5.05%
      Intervalo de predicao: NAO_VALIDADO (3 grupo(s) de validacao, minimo 19 p/ alpha=0.05)
    composto_b           2    0.46  0.996    0.05    0.16   0.008  0.681
      IC(95%): LOD [0.04-0.09]  LOQ [0.11-0.28]  |  validacao: [3.38, 19.50]% DP=7.58%
      Intervalo de predicao: NAO_VALIDADO (3 grupo(s) de validacao, minimo 19 p/ alpha=0.05)
"""

RESUMO_QUANT = _CAB.format(n_classes=1, nivel="N3 (Quantificacao de teor)", acc="0.0000",
                           r2x="0.0000", r2y="0.0000", q2="0.0000") + _REGRESSAO
RESUMO_CLASSIF = _CAB.format(n_classes=3, nivel="N1 (Classificacao por especie)", acc="0.9120",
                             r2x="0.8100", r2y="0.8700", q2="0.8100") + \
    "\n  Permutation p-value             : 0.0099\n"


def _pasta(tmp_path, resumo):
    rel = tmp_path / NOME_RELATORIOS
    rel.mkdir()
    (rel / "resumo_modelo.txt").write_text(resumo, encoding="utf-8")
    return str(tmp_path)


_PROJ = {"nome": "Proj", "autor": "A", "instituicao": "I", "tipo": "T"}


def _texto_pdf(dados: bytes) -> str:
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(dados)).pages)


def _texto_docx(dados: bytes) -> str:
    import docx
    d = docx.Document(io.BytesIO(dados))
    partes = [p.text for p in d.paragraphs]
    for t in d.tables:
        for lin in t.rows:
            partes += [c.text for c in lin.cells]
    return "\n".join(partes)


def _texto_xlsx(dados: bytes) -> str:
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(dados))
    partes = []
    for ws in wb.worksheets:
        if ws.title.lower().startswith(("raw", "resumo_bruto", "bruto")):
            continue
        for lin in ws.iter_rows(values_only=True):
            partes += [str(c) for c in lin if c is not None]
    return "\n".join(partes)


def _texto_pptx(dados: bytes) -> str:
    from pptx import Presentation
    partes = []
    for s in Presentation(io.BytesIO(dados)).slides:
        for sh in s.shapes:
            if sh.has_text_frame:
                partes.append(sh.text_frame.text)
            if getattr(sh, "has_table", False) and sh.has_table:
                for lin in sh.table.rows:
                    partes += [c.text for c in lin.cells]
    return "\n".join(partes)


_MOLDE_CLASSIF = ("R2Y", "Q2Y", "R2X", "Balanced accuracy", "Balanced Accuracy", "Acurácia balanceada",
                  "Acuracia balanceada", "AUC")


# ---------------------------------------------------------------- deteccao e parser
def test_modo_do_resumo_distingue_quantificacao_e_classificacao():
    assert modo_do_resumo(RESUMO_QUANT) == MODO_QUANTIFICACAO
    assert modo_do_resumo(RESUMO_CLASSIF) == MODO_CLASSIFICACAO


def test_parse_regressao_le_as_metricas_do_resumo():
    r = parse_regressao(RESUMO_QUANT)
    assert r["pooled"]["rmsecv"] == pytest.approx(0.408)
    assert r["pooled"]["r2cv"] == pytest.approx(0.9948)
    assert r["holdout"]["rmsep"] == pytest.approx(0.829)
    assert r["holdout"]["n"] == 12
    assert [e["especie"] for e in r["especies"]] == ["composto_a", "composto_b"]


# ---------------------------------------------------------------- PDF / DOCX / XLSX / PPTX / TEX
@pytest.fixture
def pasta_quant(tmp_path):
    return _pasta(tmp_path, RESUMO_QUANT)


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_pdf_quantificacao_mostra_regressao_e_nao_o_molde_plsda(pasta_quant, lang, monkeypatch):
    monkeypatch.setenv("GUARACI_IDIOMA", lang)
    t = _texto_pdf(reports.generate_pdf_report(pasta_quant, _PROJ).getvalue())
    centragem = "centragem" if lang == "pt" else "mean-centering"
    for esperado in ("RMSEP", "RMSECV", "R2cv", "0.829", "0.408", "SNV", centragem):
        assert esperado in t, esperado
    for proibido in _MOLDE_CLASSIF:
        assert proibido not in t, proibido


def test_docx_xlsx_pptx_tex_de_quantificacao(pasta_quant):
    docx_t = _texto_docx(reports.generate_word_report(pasta_quant, _PROJ).getvalue())
    xlsx_t = _texto_xlsx(reports.generate_excel_report(pasta_quant).getvalue())
    pptx_t = _texto_pptx(reports.generate_pptx_report(pasta_quant, _PROJ).getvalue())
    tex = reports.generate_latex_template(pasta_quant, _PROJ).decode("utf-8")
    for nome, t in (("docx", docx_t), ("xlsx", xlsx_t), ("pptx", pptx_t), ("tex", tex)):
        for esperado in ("RMSEP", "RMSECV"):
            assert esperado in t, (nome, esperado)
        assert "0.829" in t or "0,829" in t, nome
        for proibido in ("R2Y", "Q2Y", "R2X"):
            assert proibido not in t, (nome, proibido)
    # a cadeia inteira de pre-processamento, nao so' o primeiro passo
    assert "centragem" in docx_t.lower() or "mean-centering" in docx_t


def test_pdf_de_classificacao_nao_traz_metricas_de_regressao(tmp_path):
    pasta = _pasta(tmp_path, RESUMO_CLASSIF)
    t = _texto_pdf(reports.generate_pdf_report(pasta, _PROJ).getvalue())
    assert "RMSECV" not in t and "RMSEP" not in t
    assert "R2Y" in t or "R2Y".lower() in t.lower()
    # pre-processamento completo tambem na classificacao
    assert "mean-centering" in t or "centragem" in t.lower()


def test_quantificacao_uma_classe_nao_mostra_classes_nem_acuracia_balanceada(pasta_quant):
    t = _texto_pdf(reports.generate_pdf_report(pasta_quant, _PROJ).getvalue())
    assert "Classes: 1" not in t and "Classes 1" not in t


# ---------------------------------------------------------------- R2cv com verdade conhecida
def _dados_uma_especie(seed=0, n_pontos=10, n_replicas=3, p=30):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=p)
    X, conc, mae = [], [], []
    for i, c in enumerate(np.linspace(0, 40, n_pontos)):
        for _ in range(n_replicas):
            X.append(w * (c / 40.0) * 3.0 + rng.normal(scale=0.10, size=p))
            conc.append(c)
            mae.append(f"U{i:02d}")
    return np.array(X), np.array(conc), np.array(["composto_a"] * len(X)), np.array(mae)


def test_pooled_devolve_r2cv_coerente_com_o_rmsecv(pq, tmp_path):
    """R2cv = 1 - PRESS/SS_tot sobre as mesmas predicoes de CV do RMSECV. Verdade conhecida: y uniforme em
    [0, 40] (variancia ~133), ruido pequeno -> R2cv ~ 1 - RMSECV^2/var(y) (a variancia de y da calibracao e'
    a de um subconjunto, dai a tolerancia)."""
    X, conc, rot, mae = _dados_uma_especie()
    cfg = pq.Config(seed=0, max_lvs=5, frac_cal=0.7, default_preprocessing="autoscaling")
    pasta = str(tmp_path)
    for d in (pq.NOME_TABELAS, pq.NOME_GRAFICOS, pq.NOME_RELATORIOS):
        os.makedirs(os.path.join(pasta, d), exist_ok=True)
    r = pq.pls_regressao_pooled(X, conc, rot, mae, cfg, pasta, os.path.join(pasta, pq.NOME_RELATORIOS), n_splits=3)
    assert "r2cv" in r and np.isfinite(r["r2cv"])
    assert 0.9 < r["r2cv"] <= 1.0   # sinal forte: o R2cv tem que ser alto (com MSC o sinal escalar sumiria)
    esperado = 1.0 - r["rmsecv"] ** 2 / np.var(conc)
    assert r["r2cv"] == pytest.approx(esperado, abs=0.05)


def test_resumo_grava_a_linha_r2cv(tmp_path):
    from guaraci.resultados_io import append_regression_summary
    (tmp_path / "resumo_modelo.txt").write_text("", encoding="utf-8")
    append_regression_summary(str(tmp_path), pooled={"r2c": 0.99, "r2v": 0.98, "rmsec": 0.1, "rmsecv": 0.4,
                                                     "r2cv": 0.97, "rmsep": 0.5, "bias": 0.0},
                              tabela_especie=[])
    txt = (tmp_path / "resumo_modelo.txt").read_text(encoding="utf-8")
    assert "R2cv" in txt and "0.9700" in txt
