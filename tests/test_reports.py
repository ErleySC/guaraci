"""Testes de guaraci.reports (item 18: geradores de relatório extraídos de
app_quimiometria.py para módulo de serviço).

Regressão: generate_pdf_report usava um literal Unicode (em-dash "—") em duas
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
    pd.DataFrame({"Classifier": ["PLS-DA", "SVM"],
                  "Bal.Acc mean": [0.90, 0.88]}).to_csv(
        tabelas / "benchmark_classifiers.csv", sep=";", decimal=",", index=False)
    pd.DataFrame({"Classifier": ["PLS-DA"], "CI95% inf": [0.85],
                  "CI95% sup": [0.93]}).to_csv(
        tabelas / "monte_carlo_cv.csv", sep=";", decimal=",", index=False)

    return str(tmp_path)


@pytest.fixture
def projeto():
    return {
        "nome": "Projeto de teste",
        "autor": "Autor Teste",
        "inst": "Laboratorio de Teste",
        "tipo": "Classificacao",
        "objetivo": "Verificar geracao de relatorio sem erro de encoding.",
    }


def test_gerar_pdf_relatorio_nao_lanca_erro_de_encoding(pasta_resultados, projeto):
    """Regressão: em-dash fora de _a() derrubava fpdf2 com FPDFUnicodeEncodingException."""
    buf = reports.generate_pdf_report(pasta_resultados, projeto, max_figuras=0)
    assert len(buf.getvalue()) > 0


def test_gerar_word_relatorio_ok(pasta_resultados, projeto):
    buf = reports.generate_word_report(pasta_resultados, projeto, max_figuras=0)
    assert len(buf.getvalue()) > 0


def test_gerar_excel_relatorio_ok(pasta_resultados):
    buf = reports.generate_excel_report(pasta_resultados)
    assert len(buf.getvalue()) > 0


def test_gerar_latex_template_ok(pasta_resultados, projeto):
    tex = reports.generate_latex_template(pasta_resultados, projeto)
    assert isinstance(tex, bytes) and len(tex) > 0


def test_gerar_latex_template_afirma_group_aware_so_quando_realmente_usado(
        tmp_path, projeto):
    """Regressao do achado B3-1 (auditoria 2026-08): o template LaTeX
    tinha a frase "group-aware cross-validation (GroupKFold...)" CRAVADA,
    mesmo quando o pipeline caiu para StratifiedKFold (mae_id indisponivel
    -- ex.: modo_entrada="imagem"). Um manuscrito gerado nesse caso
    afirmava metodologia que nao foi aplicada naquela execucao.

    Este teste roda o gerador duas vezes com resumos que diferem SO no
    campo "Group-aware (mae_id)" e confirma que o texto do LaTeX muda de
    acordo -- nao so' que compila."""
    def _pasta(group_aware: str, cv_label: str) -> str:
        p = tmp_path / group_aware
        logs = p / "logs"
        logs.mkdir(parents=True)
        (logs / "resumo_modelo.txt").write_text(
            "Balanced Accuracy (CV): 0.912\nR2Y: 0.87\nQ2Y: 0.81\n"
            "Preprocessamento: msc_sg_mc\nN treino: 120\nN. Classes: 5\n"
            "Optimal LVs: 6\n"
            f"Validation: {cv_label}\n"
            f"Group-aware (mae_id): {group_aware}\n"
            "Faixa espectral (cm-1): [4000, 10000]\n"
            "Permutation n_validos: 200\n",
            encoding="utf-8",
        )
        return str(p)

    tex_sim = reports.generate_latex_template(
        _pasta("sim", "StableStratifiedGroupKFold n_splits=5"), projeto
    ).decode("utf-8")
    tex_nao = reports.generate_latex_template(
        _pasta("nao", "RepeatedStratifiedKFold n_splits=5 repeats=3"), projeto
    ).decode("utf-8")

    # Caso "sim": afirma group-aware, cita o cv_label real.
    assert "group-aware cross-validation" in tex_sim
    assert "StableStratifiedGroupKFold" in tex_sim
    assert "NOT applied" not in tex_sim

    # Caso "nao": NAO afirma group-aware, avisa explicitamente, cita o
    # cv_label real (nao GroupKFold, que nunca rodou nesse caso).
    assert "group-aware cross-validation" not in tex_nao
    assert "NOT applied" in tex_nao
    assert "RepeatedStratifiedKFold" in tex_nao

    # Os dois textos devem divergir de fato (nao so' passar por acidente).
    assert tex_sim != tex_nao


def _pasta_com_garantia(tmp_path, grouping_guarantee: str) -> str:
    """Pasta de resultados minima cujo resumo declara `Grouping guarantee`."""
    p = tmp_path / f"garantia_{grouping_guarantee}"
    logs = p / "logs"
    logs.mkdir(parents=True)
    (logs / "resumo_modelo.txt").write_text(
        "Balanced Accuracy (CV): 0.912\nR2Y: 0.87\nQ2Y: 0.81\n"
        "Preprocessamento: msc_sg_mc\nN treino: 120\nN. Classes: 5\n"
        "Optimal LVs: 6\nValidation: RepeatedStratifiedKFold n_splits=5\n"
        "Group-aware (mae_id): nao\n"
        "Faixa espectral (cm-1): [4000, 10000]\n"
        "Permutation n_validos: 200\n"
        "Input mode: imagem\n"
        f"Grouping guarantee: {grouping_guarantee}\n",
        encoding="utf-8",
    )
    return str(p)


def test_relatorios_carimbam_prototipo_so_sem_garantia_de_agrupamento(
        tmp_path, projeto):
    """Regressao do achado B4-1 (auditoria 2026-08), redesenhada no Bloco 8
    (2026-08-25): o carimbo NAO e' mais por `mode == "imagem"` -- e' por
    `Grouping guarantee == "none"`. Ate' 2026-08-25 TODO mode="imagem"
    carimbava, mesmo quando o usuario organizou a pasta por amostra fisica
    (nivel "high") ou forneceu o CSV de associacao (nivel "medium"), os
    dois com a MESMA garantia contra vazamento que dx/sintetico tem --
    carimbar esses dois como "nao validado" seria informacao FALSA, nao
    conservadora. Este teste confirma que so' "none" carimba; "high" e
    "medium" (mesmo vindo de mode="imagem") nao."""
    pasta_none   = _pasta_com_garantia(tmp_path, "none")
    pasta_high   = _pasta_com_garantia(tmp_path, "high")
    pasta_medium = _pasta_com_garantia(tmp_path, "medium")

    tex_none   = reports.generate_latex_template(pasta_none, projeto).decode("utf-8")
    tex_high   = reports.generate_latex_template(pasta_high, projeto).decode("utf-8")
    tex_medium = reports.generate_latex_template(pasta_medium, projeto).decode("utf-8")

    assert "PROTOTYPE OUTPUT" in tex_none
    assert "PROTOTYPE OUTPUT" not in tex_high
    assert "PROTOTYPE OUTPUT" not in tex_medium

    # PDF e Word: nao da' p/ inspecionar o texto renderizado facilmente,
    # entao verifica-se que o carimbo muda o TAMANHO da saida (conteudo a
    # mais) e que os dois geradores rodam sem erro nos dois casos.
    pdf_none = reports.generate_pdf_report(pasta_none, projeto, max_figuras=0)
    pdf_high = reports.generate_pdf_report(pasta_high, projeto, max_figuras=0)
    assert len(pdf_none.getvalue()) > len(pdf_high.getvalue())

    doc_none = reports.generate_word_report(pasta_none, projeto, max_figuras=0)
    doc_high = reports.generate_word_report(pasta_high, projeto, max_figuras=0)
    assert len(doc_none.getvalue()) > len(doc_high.getvalue())


def test_gerar_pptx_relatorio_ok(pasta_resultados, projeto):
    buf = reports.generate_pptx_report(pasta_resultados, projeto, max_figuras=0)
    assert len(buf.getvalue()) > 0


def test_versao_no_relatorio_pdf_usa_pipeline_version(pasta_resultados, projeto):
    """_APP_VERSION do modulo deve refletir guaraci.pipeline.__version__ (single-source)."""
    import guaraci.pipeline as pq
    assert reports._APP_VERSION == f"v{pq.__version__}"


# ── Geradores com pasta RICA (figuras + tabelas reais) ──────────────────────
# Exercita os ramos "arquivo existe" que a pasta minima acima nunca alcança
# (secao de figuras do PDF/Word/PPTX, abas Identifiers/VIP/Benchmark do Excel).

def test_gerar_pdf_relatorio_com_figuras_reais(pasta_resultados_completa, projeto):
    buf = reports.generate_pdf_report(pasta_resultados_completa, projeto,
                                        max_figuras=2)
    assert len(buf.getvalue()) > 0


def test_gerar_word_relatorio_com_figuras_reais(pasta_resultados_completa, projeto):
    buf = reports.generate_word_report(pasta_resultados_completa, projeto,
                                         max_figuras=2)
    assert len(buf.getvalue()) > 0


def test_gerar_excel_relatorio_com_tabelas_reais(pasta_resultados_completa):
    buf = reports.generate_excel_report(pasta_resultados_completa)
    assert len(buf.getvalue()) > 0


def test_gerar_pptx_relatorio_com_figuras_e_benchmark(pasta_resultados_completa, projeto):
    buf = reports.generate_pptx_report(pasta_resultados_completa, projeto,
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
    buf = reports.generate_excel_report(str(tmp_path))
    assert len(buf.getvalue()) > 0
