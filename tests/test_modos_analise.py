"""Testes de unidade da camada de objetivo científico (modos_analise.py).

Cobre a nova função que decide QUAIS figuras cada modo gera: resolução do
objetivo (auto <- nível, override explícito, entradas inválidas), o gate
`deve_gerar` (pertinência + fail-open para chaves de overview/desconhecidas),
a regra das exploratórias e o plano de figuras exibido ao usuário.
"""
import pytest

from guaraci import modos_analise as m
from guaraci.config import Config


# ---- resolver_objetivo --------------------------------------------------
@pytest.mark.parametrize("nivel,esperado", [
    ("N1", m.CLASSIFICACAO),
    ("N2", m.CLASSIFICACAO),
    ("N3", m.QUANTIFICACAO),
])
def test_auto_deriva_do_nivel(nivel, esperado):
    cfg = Config(nivel=nivel, objetivo="auto")
    assert m.resolver_objetivo(cfg) == esperado


@pytest.mark.parametrize("obj", [m.EXPLORATORIO, m.CLASSIFICACAO, m.QUANTIFICACAO])
def test_objetivo_explicito_sobrepoe_nivel(obj):
    # Mesmo com nivel=N1 (que derivaria classificacao), o explícito vence.
    cfg = Config(nivel="N1", objetivo=obj)
    assert m.resolver_objetivo(cfg) == obj


def test_objetivo_invalido_cai_para_derivacao_do_nivel():
    cfg = Config(nivel="N3", objetivo="banana")
    assert m.resolver_objetivo(cfg) == m.QUANTIFICACAO


def test_objetivo_normaliza_caixa_e_espaco():
    cfg = Config(nivel="N1", objetivo="  ExPlOrAtOrIo  ")
    assert m.resolver_objetivo(cfg) == m.EXPLORATORIO


def test_nivel_desconhecido_default_classificacao():
    cfg = Config(nivel="ZZ", objetivo="auto")
    assert m.resolver_objetivo(cfg) == m.CLASSIFICACAO


# ---- deve_gerar ---------------------------------------------------------
def test_deve_gerar_pertinencia_por_objetivo():
    cfg = Config(objetivo=m.QUANTIFICACAO)
    assert m.deve_gerar(cfg, "regressao") is True
    assert m.deve_gerar(cfg, "confusao") is False
    assert m.deve_gerar(cfg, "ddsimca") is False

    cfg = Config(objetivo=m.CLASSIFICACAO)
    assert m.deve_gerar(cfg, "confusao") is True
    assert m.deve_gerar(cfg, "roc") is True
    assert m.deve_gerar(cfg, "regressao") is False

    cfg = Config(objetivo=m.EXPLORATORIO)
    assert m.deve_gerar(cfg, "hca") is True
    assert m.deve_gerar(cfg, "plsda_scores") is False
    assert m.deve_gerar(cfg, "regressao") is False


def test_deve_gerar_fail_open_para_chave_desconhecida():
    """Chaves não mapeadas (overview PCA/outliers, ou futuras) nunca são
    silenciosamente suprimidas."""
    for obj in (m.EXPLORATORIO, m.CLASSIFICACAO, m.QUANTIFICACAO):
        cfg = Config(objetivo=obj)
        assert m.deve_gerar(cfg, "fig1_pca_scores_overview") is True
        assert m.deve_gerar(cfg, "chave_inexistente_qualquer") is True


# ---- figuras_exploratorias_ligadas -------------------------------------
def test_exploratorias_ligadas_no_modo_exploratorio():
    cfg = Config(objetivo=m.EXPLORATORIO, figuras_detalhadas=False)
    assert m.figuras_exploratorias_ligadas(cfg) is True


def test_exploratorias_escotilha_em_classificacao_com_detalhadas():
    cfg = Config(objetivo=m.CLASSIFICACAO, figuras_detalhadas=True)
    assert m.figuras_exploratorias_ligadas(cfg) is True
    cfg = Config(objetivo=m.CLASSIFICACAO, figuras_detalhadas=False)
    assert m.figuras_exploratorias_ligadas(cfg) is False


def test_exploratorias_desligadas_em_quantificacao():
    cfg = Config(objetivo=m.QUANTIFICACAO, figuras_detalhadas=True)
    assert m.figuras_exploratorias_ligadas(cfg) is False


# ---- plano_de_figuras / descrever_plano --------------------------------
def test_plano_de_figuras_por_objetivo():
    cfg = Config(objetivo=m.QUANTIFICACAO)
    assert m.plano_de_figuras(cfg) == ["regressao"]

    cfg = Config(objetivo=m.EXPLORATORIO)
    assert set(m.plano_de_figuras(cfg)) == {"hca", "loadings", "preprocessamento"}

    cfg = Config(objetivo=m.CLASSIFICACAO)
    plano = set(m.plano_de_figuras(cfg))
    assert {"plsda_scores", "confusao", "roc", "ddsimca"} <= plano
    assert "regressao" not in plano


def test_descrever_plano_retorna_texto_legivel():
    cfg = Config(objetivo=m.QUANTIFICACAO)
    desc = m.descrever_plano(cfg)
    assert desc and all(isinstance(s, str) and s for s in desc)
    # Não devolve a chave crua quando há descrição cadastrada.
    assert "regressao" not in desc
