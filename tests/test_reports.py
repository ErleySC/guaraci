"""Testes de guaraci.reports (item 18: geradores de relatório extraídos de
app_quimiometria.py para módulo de serviço).

Regressão: gerar_pdf_relatorio usava um literal Unicode (em-dash "—") em duas
strings passadas direto ao fpdf2 sem passar pelo normalizador `_a()` (que
remove acentos/Unicode para a fonte Helvetica, latin-1-only). Isso derrubava
a geração de PDF em QUALQUER projeto (achado ao exercitar o gerador contra
dados reais após a extração do item 18). Os testes abaixo rodam os 5
geradores contra uma pasta mínima e confirmam que nenhum lança exceção.
"""
import pytest

from guaraci import reports


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
