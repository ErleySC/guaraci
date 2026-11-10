"""Testes de hsi_uncertainty.py (Passo 107) -- nota de confianca formal
a partir da heterogeneidade de pixel + decisao registrada sobre
Bonferroni (ver docstring do modulo)."""
from __future__ import annotations

from guaraci.hsi_classification import ObjectAggregationResult
from guaraci.hsi_uncertainty import confidence_note, enrich_object_results


def test_confidence_note_unanimidade():
    nota = confidence_note(0.0)
    assert "unanimidade" in nota.lower()


def test_confidence_note_alta_concordancia():
    nota = confidence_note(0.05)
    assert "alta concordancia" in nota.lower()


def test_confidence_note_moderada():
    nota = confidence_note(0.20)
    assert "moderada" in nota.lower()


def test_confidence_note_baixa():
    nota = confidence_note(0.50)
    assert "baixa concordancia" in nota.lower()
    assert "questionavel" in nota.lower()


def test_confidence_note_e_monotonica_na_fronteira():
    """Fronteiras dos limiares (10%/30%) devem cair no lado documentado
    (<=), nao no lado oposto por erro de off-by-one."""
    assert "alta" in confidence_note(0.10).lower()
    assert "moderada" in confidence_note(0.30).lower()
    assert "baixa" in confidence_note(0.30001).lower()


def test_enrich_object_results_preserva_dados_e_anexa_nota():
    predicoes = {
        "obj1": ObjectAggregationResult(
            classe_predita="perfect", heterogeneidade=0.0, n_pixels=100),
        "obj2": ObjectAggregationResult(
            classe_predita="overripe", heterogeneidade=0.45, n_pixels=50),
    }
    relatorio = enrich_object_results(predicoes)
    assert relatorio["obj1"].classe_predita == "perfect"
    assert relatorio["obj1"].n_pixels == 100
    assert "unanimidade" in relatorio["obj1"].nota_confianca.lower()
    assert "baixa concordancia" in relatorio["obj2"].nota_confianca.lower()
