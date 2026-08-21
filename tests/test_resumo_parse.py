"""Testes de guaraci.resumo_parse (item 19: parsing puro do resumo_modelo.txt).

Antes da extração, este parsing vivia duplicado — `_ex` em 5 cópias nos
geradores de relatório e o dicionário de 12 métricas repetido em PDF/Word, mais
o parse de acurácia-por-classe inline na aba Validation. Agora é um módulo só,
testável sem UI.
"""
import textwrap

from guaraci.resumo_parse import (
    extract_metric, parse_model_metrics, parse_accuracy_by_class,
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


# ── extract_metric ──────────────────────────────────────────────────────────
def test_extrair_metrica_casa():
    assert extract_metric(_RESUMO_EXEMPLO, r"R2Y.*?[:=]\s*([\d.]+)") == "0.87"


def test_extrair_metrica_default_quando_nao_casa():
    assert extract_metric(_RESUMO_EXEMPLO, r"INEXISTENTE.*?([\d]+)") == "-"
    assert extract_metric(_RESUMO_EXEMPLO, r"INEXISTENTE.*?([\d]+)", "—") == "—"


def test_extrair_metrica_resumo_vazio():
    assert extract_metric("", r"R2Y.*?([\d.]+)", "n/a") == "n/a"
    assert extract_metric(None, r"R2Y.*?([\d.]+)", "n/a") == "n/a"


def test_extrair_metrica_ignora_case():
    assert extract_metric("balanced accuracy: 0.5", r"Balanced Accuracy.*?([\d.]+)") == "0.5"


# ── parse_model_metrics ────────────────────────────────────────────────────
def test_parse_metricas_tem_12_chaves():
    m = parse_model_metrics(_RESUMO_EXEMPLO)
    assert len(m) == 12


def test_parse_metricas_valores_esperados():
    m = parse_model_metrics(_RESUMO_EXEMPLO)
    assert m["Balanced Accuracy (CV)"] == "0.912"
    assert m["R2Y"] == "0.87"
    assert m["Q2Y"] == "0.81"
    assert m["Optimal LVs"] == "14"
    assert m["Preprocessing"] == "msc_sg_mc"
    assert m["n classes"] == "14"


def test_parse_metricas_ausentes_viram_default():
    m = parse_model_metrics("resumo sem nenhuma metrica reconhecivel")
    assert all(v == "-" for v in m.values())


def test_parse_metricas_equivale_ao_ex_manual():
    """Equivalência: parse_model_metrics deve dar o MESMO resultado que
    aplicar extract_metric manualmente aos padrões (garante que a
    consolidação dos 5 geradores não mudou o parsing)."""
    from guaraci.resumo_parse import _PADROES_METRICAS
    m = parse_model_metrics(_RESUMO_EXEMPLO)
    for nome, padrao in _PADROES_METRICAS.items():
        assert m[nome] == extract_metric(_RESUMO_EXEMPLO, padrao)


# ── parse_accuracy_by_class ────────────────────────────────────────────────
def test_parse_acuracia_extrai_todas_as_classes():
    acc = parse_accuracy_by_class(_RESUMO_EXEMPLO)
    assert acc == {"Andiroba": 0.95, "Copaiba": 0.88, "Babacu": 0.42}


def test_parse_acuracia_vazio_sem_linhas_acc():
    assert parse_accuracy_by_class("resumo sem linhas de acuracia") == {}
    assert parse_accuracy_by_class("") == {}
    assert parse_accuracy_by_class(None) == {}


def test_parse_acuracia_aceita_igual_ou_doispontos():
    acc = parse_accuracy_by_class("Acc X = 0.7\nAcc Y: 0.8")
    assert acc == {"X": 0.7, "Y": 0.8}
