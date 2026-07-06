"""Testes de guaraci.resumo_parse (item 19: parsing puro do resumo_modelo.txt).

Antes da extração, este parsing vivia duplicado — `_ex` em 5 cópias nos
geradores de relatório e o dicionário de 12 métricas repetido em PDF/Word, mais
o parse de acurácia-por-classe inline na aba Validation. Agora é um módulo só,
testável sem UI.
"""
import textwrap

from guaraci.resumo_parse import (
    extrair_metrica, parse_metricas_modelo, parse_acuracia_por_classe,
)

_RESUMO_EXEMPLO = textwrap.dedent("""\
    Modelo PLS-DA — GUARACI
    Pre-processamento: msc_sg_mc
    Balanced Accuracy (CV): 0.912
    ROC AUC macro OvR: 0.98
    R2Y: 0.87
    Q2: 0.81
    R2X: 0.95
    LVs otimo: 14
    p-value: 0.004
    Hotelling T2 UCL: 21.3
    Q-residual UCL: 0.0123
    N treino: 120
    N. Classes: 14
    Acc Andiroba: 0.95
    Acc Copaiba: 0.88
    Acc Babacu: 0.42
""")


# ── extrair_metrica ──────────────────────────────────────────────────────────
def test_extrair_metrica_casa():
    assert extrair_metrica(_RESUMO_EXEMPLO, r"R2Y.*?[:=]\s*([\d.]+)") == "0.87"


def test_extrair_metrica_default_quando_nao_casa():
    assert extrair_metrica(_RESUMO_EXEMPLO, r"INEXISTENTE.*?([\d]+)") == "-"
    assert extrair_metrica(_RESUMO_EXEMPLO, r"INEXISTENTE.*?([\d]+)", "—") == "—"


def test_extrair_metrica_resumo_vazio():
    assert extrair_metrica("", r"R2Y.*?([\d.]+)", "n/a") == "n/a"
    assert extrair_metrica(None, r"R2Y.*?([\d.]+)", "n/a") == "n/a"


def test_extrair_metrica_ignora_case():
    assert extrair_metrica("balanced accuracy: 0.5", r"Balanced Accuracy.*?([\d.]+)") == "0.5"


# ── parse_metricas_modelo ────────────────────────────────────────────────────
def test_parse_metricas_tem_12_chaves():
    m = parse_metricas_modelo(_RESUMO_EXEMPLO)
    assert len(m) == 12


def test_parse_metricas_valores_esperados():
    m = parse_metricas_modelo(_RESUMO_EXEMPLO)
    assert m["Balanced Accuracy (CV)"] == "0.912"
    assert m["R2Y"] == "0.87"
    assert m["Q2Y"] == "0.81"
    assert m["Optimal LVs"] == "14"
    assert m["Preprocessing"] == "msc_sg_mc"
    assert m["n classes"] == "14"


def test_parse_metricas_ausentes_viram_default():
    m = parse_metricas_modelo("resumo sem nenhuma metrica reconhecivel")
    assert all(v == "-" for v in m.values())


def test_parse_metricas_equivale_ao_ex_manual():
    """Equivalência: parse_metricas_modelo deve dar o MESMO resultado que
    aplicar extrair_metrica manualmente aos padrões (garante que a
    consolidação dos 5 geradores não mudou o parsing)."""
    from guaraci.resumo_parse import _PADROES_METRICAS
    m = parse_metricas_modelo(_RESUMO_EXEMPLO)
    for nome, padrao in _PADROES_METRICAS.items():
        assert m[nome] == extrair_metrica(_RESUMO_EXEMPLO, padrao)


# ── parse_acuracia_por_classe ────────────────────────────────────────────────
def test_parse_acuracia_extrai_todas_as_classes():
    acc = parse_acuracia_por_classe(_RESUMO_EXEMPLO)
    assert acc == {"Andiroba": 0.95, "Copaiba": 0.88, "Babacu": 0.42}


def test_parse_acuracia_vazio_sem_linhas_acc():
    assert parse_acuracia_por_classe("resumo sem linhas de acuracia") == {}
    assert parse_acuracia_por_classe("") == {}
    assert parse_acuracia_por_classe(None) == {}


def test_parse_acuracia_aceita_igual_ou_doispontos():
    acc = parse_acuracia_por_classe("Acc X = 0.7\nAcc Y: 0.8")
    assert acc == {"X": 0.7, "Y": 0.8}
