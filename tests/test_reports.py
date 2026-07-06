"""Testes de guaraci.reports (item 18: geradores de relatório extraídos de
app_quimiometria.py para módulo de serviço).

Regressão: gerar_pdf_relatorio usava um literal Unicode (em-dash "—") em duas
strings passadas direto ao fpdf2 sem passar pelo normalizador `_a()` (que
remove acentos/Unicode para a fonte Helvetica, latin-1-only). Isso derrubava
a geração de PDF em QUALQUER projeto (achado ao exercitar o gerador contra
dados reais após a extração do item 18). Os testes abaixo rodam os 5
geradores contra uma pasta mínima e confirmam que nenhum lança exceção.
"""
import pandas as pd
import pytest

from guaraci import reports
from guaraci.config import NOME_TABELAS


@pytest.fixture
def pasta_resultados(tmp_path):
    """Pasta mínima: sem figuras, com um resumo_modelo.txt simples."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "resumo_modelo.txt").write_text(
        "Balanced Accuracy (CV): 0.912\nR2Y: 0.87\nQ2Y: 0.81\n"
        "Preprocessamento: msc_sg_mc\nN treino: 120\nN. Classes: 5\n",
        encoding="utf-8",
    )
    return str(tmp_path)


@pytest.fixture
def pasta_resultados_completa(tmp_path):
    """Pasta RICA: resumo + 2 figuras PNG reais + as tabelas CSV que os
    geradores de relatorio procuram (identificadores, selecao de variaveis,
    benchmark, Monte Carlo). Exercita os ramos "if os.path.exists(...)" dos
    5 geradores que a fixture minima nunca alcancava (reforço de cobertura
    de CI, auditoria jul/2026 — piso 60%, margem apertada)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "resumo_modelo.txt").write_text(
        "Balanced Accuracy (CV): 0.912\nR2Y: 0.87\nQ2Y: 0.81\n"
        "Preprocessamento: msc_sg_mc\nN treino: 120\nN. Classes: 5\n",
        encoding="utf-8",
    )

    figuras = tmp_path / "Graficos"
    figuras.mkdir()
    for nome in ("fig1_pca_scores", "fig2_plsda_scores"):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(str(figuras / f"{nome}.png"))
        plt.close(fig)

    tabelas = tmp_path / NOME_TABELAS
    tabelas.mkdir()
    pd.DataFrame({"amostra": ["a1", "a2"], "classe": ["X", "Y"]}).to_csv(
        tabelas / "amostras_identificadores.csv", sep=";", decimal=",", index=False)
    pd.DataFrame({"wavenumber": [4000, 4010], "vip": [1.2, 0.8]}).to_csv(
        tabelas / "etapa4_selecao_variaveis.csv", sep=";", decimal=",", index=False)
    pd.DataFrame({"Classificador": ["PLS-DA", "SVM"],
                  "Bal.Acc media": [0.90, 0.88]}).to_csv(
        tabelas / "benchmark_classificadores.csv", sep=";", decimal=",", index=False)
    pd.DataFrame({"Classificador": ["PLS-DA"], "IC95% inf": [0.85],
                  "IC95% sup": [0.93]}).to_csv(
        tabelas / "monte_carlo_cv.csv", sep=";", decimal=",", index=False)

    return str(tmp_path)


@pytest.fixture
def projeto():
    return {
        "nome": "Projeto de teste",
        "autor": "Autor Teste",
        "inst": "GEAAp/UFPA",
        "tipo": "Classificacao",
        "objetivo": "Verificar geracao de relatorio sem erro de encoding.",
    }


def test_gerar_pdf_relatorio_nao_lanca_erro_de_encoding(pasta_resultados, projeto):
    """Regressão: em-dash fora de _a() derrubava fpdf2 com FPDFUnicodeEncodingException."""
    buf = reports.gerar_pdf_relatorio(pasta_resultados, projeto, max_figuras=0)
    assert len(buf.getvalue()) > 0


def test_gerar_word_relatorio_ok(pasta_resultados, projeto):
    buf = reports.gerar_word_relatorio(pasta_resultados, projeto, max_figuras=0)
    assert len(buf.getvalue()) > 0


def test_gerar_excel_relatorio_ok(pasta_resultados):
    buf = reports.gerar_excel_relatorio(pasta_resultados)
    assert len(buf.getvalue()) > 0


def test_gerar_latex_template_ok(pasta_resultados, projeto):
    tex = reports.gerar_latex_template(pasta_resultados, projeto)
    assert isinstance(tex, bytes) and len(tex) > 0


def test_gerar_pptx_relatorio_ok(pasta_resultados, projeto):
    buf = reports.gerar_pptx_relatorio(pasta_resultados, projeto, max_figuras=0)
    assert len(buf.getvalue()) > 0


def test_versao_no_relatorio_pdf_usa_pipeline_version(pasta_resultados, projeto):
    """_APP_VERSION do modulo deve refletir guaraci.pipeline.__version__ (single-source)."""
    import guaraci.pipeline as pq
    assert reports._APP_VERSION == f"v{pq.__version__}"


# ── Geradores com pasta RICA (figuras + tabelas reais) ──────────────────────
# Exercita os ramos "arquivo existe" que a pasta minima acima nunca alcança
# (secao de figuras do PDF/Word/PPTX, abas Identifiers/VIP/Benchmark do Excel).

def test_gerar_pdf_relatorio_com_figuras_reais(pasta_resultados_completa, projeto):
    buf = reports.gerar_pdf_relatorio(pasta_resultados_completa, projeto,
                                        max_figuras=2)
    assert len(buf.getvalue()) > 0


def test_gerar_word_relatorio_com_figuras_reais(pasta_resultados_completa, projeto):
    buf = reports.gerar_word_relatorio(pasta_resultados_completa, projeto,
                                         max_figuras=2)
    assert len(buf.getvalue()) > 0


def test_gerar_excel_relatorio_com_tabelas_reais(pasta_resultados_completa):
    buf = reports.gerar_excel_relatorio(pasta_resultados_completa)
    assert len(buf.getvalue()) > 0


def test_gerar_pptx_relatorio_com_figuras_e_benchmark(pasta_resultados_completa, projeto):
    buf = reports.gerar_pptx_relatorio(pasta_resultados_completa, projeto,
                                         max_figuras=2)
    assert len(buf.getvalue()) > 0


def test_gerar_excel_relatorio_csv_corrompido_nao_quebra(tmp_path):
    """Se um CSV esperado existir mas estiver corrompido/ilegivel, a aba
    correspondente mostra uma mensagem de erro em vez de propagar excecao."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "resumo_modelo.txt").write_text("Balanced Accuracy (CV): 0.9\n",
                                              encoding="utf-8")
    tabelas = tmp_path / NOME_TABELAS
    tabelas.mkdir()
    (tabelas / "amostras_identificadores.csv").write_bytes(b"\xff\xfe\x00\x01lixo binario")
    buf = reports.gerar_excel_relatorio(str(tmp_path))
    assert len(buf.getvalue()) > 0
