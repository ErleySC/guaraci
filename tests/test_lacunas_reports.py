"""Geradores de relatorio (`guaraci.reports`) com figuras corrompidas, tabelas ilegiveis e numero impar de figuras."""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from guaraci import reports
from guaraci.config import NOME_TABELAS

_PROJETO = {"nome": "P", "autor": "A", "inst": "I", "tipo": "Classificacao", "objetivo": "O"}


def _base(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / "resumo_modelo.txt").write_text("Balanced Accuracy (CV): 0.9\nR2Y: 0.8\nQ2Y: 0.7\nN treino: 12\nN. Classes: 3\n", encoding="utf-8")
    (tmp_path / "Graficos").mkdir()
    (tmp_path / NOME_TABELAS).mkdir()
    return tmp_path


def _figuras_validas(pasta, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for i in range(n):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1])
        fig.savefig(str(pasta / "Graficos" / f"fig{i + 1}_pca_scores.png"))
        plt.close(fig)


@pytest.fixture
def pasta_com_defeitos(tmp_path):
    base = _base(tmp_path / "com_defeitos")
    _figuras_validas(base, 2)
    (base / "Graficos" / "fig9_corrompida.png").write_bytes(b"isto nao e uma imagem")
    for nome in ("etapa4_selecao_variaveis.csv", "benchmark_classifiers.csv", "monte_carlo_cv.csv"):
        (base / NOME_TABELAS / nome).write_bytes(b"\xff\xfe\x00\x80\x81")
    return str(base)


@pytest.fixture
def pasta_grande(tmp_path):
    base = _base(tmp_path / "grande")
    _figuras_validas(base, 3)
    pd.DataFrame({"Classifier": [f"M{i}" for i in range(9)], "Bal.Acc mean": [0.8 + i / 100 for i in range(9)]}).to_csv(
        base / NOME_TABELAS / "benchmark_classifiers.csv", sep=";", decimal=",", index=False)
    return str(base)


def test_pdf_com_figura_corrompida_registra_aviso_e_termina(pasta_com_defeitos, caplog):
    with caplog.at_level(logging.WARNING):
        buf = reports.generate_pdf_report(pasta_com_defeitos, _PROJETO, max_figuras=14)
    assert len(buf.getvalue()) > 0 and "figura nao embutida" in caplog.text


def test_pdf_com_numero_impar_de_figuras_deixa_a_ultima_pagina_com_uma_so(pasta_grande):
    assert len(reports.generate_pdf_report(pasta_grande, _PROJETO, max_figuras=14).getvalue()) > 0


def test_word_com_figura_corrompida_mostra_o_marcador_de_imagem_indisponivel(pasta_com_defeitos):
    buf = reports.generate_word_report(pasta_com_defeitos, _PROJETO, max_figuras=14)
    import docx
    texto = "\n".join(p.text for p in docx.Document(buf).paragraphs)
    assert "imagem indisponível" in texto   # relatorio em portugues (padrao)


def test_excel_com_tabelas_ilegiveis_escreve_a_mensagem_de_erro_na_planilha(pasta_com_defeitos):
    buf = reports.generate_excel_report(pasta_com_defeitos)
    import openpyxl
    wb = openpyxl.load_workbook(buf)
    celulas = [str(c.value) for ws in wb.worksheets for row in ws.iter_rows() for c in row if c.value]
    for trecho in ("Erro ao ler etapa4_selecao_variaveis.csv", "Erro ao ler benchmark_classifiers.csv", "Erro ao ler monte_carlo_cv.csv"):
        assert trecho in celulas


def test_pptx_cobre_figura_corrompida_benchmark_longo_e_benchmark_ilegivel(pasta_com_defeitos, pasta_grande, caplog):
    with caplog.at_level(logging.WARNING):
        assert len(reports.generate_pptx_report(pasta_com_defeitos, _PROJETO).getvalue()) > 0
    assert "slide de benchmark nao gerado" in caplog.text
    assert len(reports.generate_pptx_report(pasta_grande, _PROJETO).getvalue()) > 0
