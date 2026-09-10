"""Testes de unidade da camada de objetivo científico (modos_analise.py).

Cobre a nova função que decide QUAIS figuras cada mode gera: resolução do
objetivo (auto <- nível, override explícito, entradas inválidas), o gate
`should_generate` (pertinência + fail-open para chaves de overview/desconhecidas),
a regra das exploratórias e o plano de figuras exibido ao usuário.
"""
import pytest

from guaraci import modos_analise as m
from guaraci.config import Config


# ---- resolve_objective --------------------------------------------------
@pytest.mark.parametrize("nivel,esperado", [
    ("N1", m.CLASSIFICACAO),
    ("N2", m.CLASSIFICACAO),
    ("N3", m.QUANTIFICACAO),
])
def test_auto_deriva_do_nivel(nivel, esperado):
    cfg = Config(level=nivel, objective="auto")
    assert m.resolve_objective(cfg) == esperado


@pytest.mark.parametrize("obj", [m.EXPLORATORIO, m.CLASSIFICACAO, m.QUANTIFICACAO])
def test_objetivo_explicito_sobrepoe_nivel(obj):
    # Mesmo com level=N1 (que derivaria classificacao), o explícito vence.
    cfg = Config(level="N1", objective=obj)
    assert m.resolve_objective(cfg) == obj


def test_objetivo_invalido_cai_para_derivacao_do_nivel():
    cfg = Config(level="N3", objective="banana")
    assert m.resolve_objective(cfg) == m.QUANTIFICACAO


def test_objetivo_normaliza_caixa_e_espaco():
    cfg = Config(level="N1", objective="  ExPlOrAtOrIo  ")
    assert m.resolve_objective(cfg) == m.EXPLORATORIO


def test_nivel_desconhecido_default_classificacao():
    cfg = Config(level="ZZ", objective="auto")
    assert m.resolve_objective(cfg) == m.CLASSIFICACAO


# ---- should_generate ---------------------------------------------------------
def test_deve_gerar_pertinencia_por_objetivo():
    cfg = Config(objective=m.QUANTIFICACAO)
    assert m.should_generate(cfg, "regressao") is True
    assert m.should_generate(cfg, "confusao") is False
    assert m.should_generate(cfg, "ddsimca") is False

    cfg = Config(objective=m.CLASSIFICACAO)
    assert m.should_generate(cfg, "confusao") is True
    assert m.should_generate(cfg, "roc") is True
    assert m.should_generate(cfg, "regressao") is False

    cfg = Config(objective=m.EXPLORATORIO)
    assert m.should_generate(cfg, "hca") is True
    assert m.should_generate(cfg, "plsda_scores") is False
    assert m.should_generate(cfg, "regressao") is False


def test_deve_gerar_fail_open_para_chave_desconhecida():
    """Chaves não mapeadas (overview PCA/outliers, ou futuras) nunca são
    silenciosamente suprimidas."""
    for obj in (m.EXPLORATORIO, m.CLASSIFICACAO, m.QUANTIFICACAO):
        cfg = Config(objective=obj)
        assert m.should_generate(cfg, "fig1_pca_scores_overview") is True
        assert m.should_generate(cfg, "chave_inexistente_qualquer") is True


# ---- exploratory_figures_enabled -------------------------------------
def test_exploratorias_ligadas_no_modo_exploratorio():
    cfg = Config(objective=m.EXPLORATORIO, detailed_figures=False)
    assert m.exploratory_figures_enabled(cfg) is True


def test_exploratorias_escotilha_em_classificacao_com_detalhadas():
    cfg = Config(objective=m.CLASSIFICACAO, detailed_figures=True)
    assert m.exploratory_figures_enabled(cfg) is True
    cfg = Config(objective=m.CLASSIFICACAO, detailed_figures=False)
    assert m.exploratory_figures_enabled(cfg) is False


def test_exploratorias_desligadas_em_quantificacao():
    cfg = Config(objective=m.QUANTIFICACAO, detailed_figures=True)
    assert m.exploratory_figures_enabled(cfg) is False


# ---- figure_plan / describe_plan --------------------------------
def test_plano_de_figuras_por_objetivo():
    cfg = Config(objective=m.QUANTIFICACAO)
    assert m.figure_plan(cfg) == ["regressao"]

    cfg = Config(objective=m.EXPLORATORIO)
    assert set(m.figure_plan(cfg)) == {
        "hca", "loadings", "biplot", "preprocessamento"}

    cfg = Config(objective=m.CLASSIFICACAO)
    plano = set(m.figure_plan(cfg))
    assert {"plsda_scores", "confusao", "roc"} <= plano
    assert "regressao" not in plano
    # level=N1 (default) + toggle desligado (default): ddsimca nao entra.
    assert "ddsimca" not in plano

    # DD-SIMCA no preview segue a mesma regra de pipeline.executar(): N2
    # forca ligado (mesmo sem tocar no toggle); N1 sempre ignora, mesmo com
    # o toggle ligado manualmente.
    assert "ddsimca" in set(m.figure_plan(Config(level="N2")))
    assert "ddsimca" not in set(
        m.figure_plan(Config(level="N1", run_ddsimca=True)))


def test_descrever_plano_retorna_texto_legivel():
    cfg = Config(objective=m.QUANTIFICACAO)
    desc = m.describe_plan(cfg)
    assert desc and all(isinstance(s, str) and s for s in desc)
    # Não devolve a chave crua quando há descrição cadastrada.
    assert "regressao" not in desc
