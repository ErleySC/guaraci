# -*- coding: utf-8 -*-
"""Testes de guaraci.plano_amostral (P1 do Bloco 10, reformulado).

O invariante central (achado de 2026-08-26, ver `scripts/medicoes/
medir_ddsimca_cobertura_vs_n.py` e docstring do modulo): DD-SIMCA nao
converge para cobertura nominal so' com mais `n` -- estanca num plato de
~0,94-0,945. Este arquivo prova que `ddsimca_sample_size_guidance`
NUNCA promete uma cobertura-alvo acima desse plato, qualquer que seja o
`n` sugerido -- essa e' a contra-prova que a instrucao pediu
explicitamente.
"""
from __future__ import annotations

import pytest

from guaraci.conformal import n_minimum_for_alpha
from guaraci.plano_amostral import (
    PLATO_COBERTURA_DDSIMCA,
    MEASURED_DDSIMCA_COVERAGE_TABLE,
    n_minimum_conformal,
    ddsimca_sample_size_guidance,
)


# ── Conformal: reaproveita n_minimum_for_alpha, nao reimplementa ──────────

def test_n_minimo_conformal_reaproveita_n_minimum_for_alpha():
    for alpha in (0.05, 0.10, 0.25, 0.01):
        assert n_minimum_conformal(alpha) == n_minimum_for_alpha(alpha)


def test_n_minimo_conformal_valores_conhecidos():
    assert n_minimum_conformal(0.05) == 19
    assert n_minimum_conformal(0.10) == 9
    assert n_minimum_conformal(0.25) == 3


# ── DD-SIMCA: contra-prova central -- nunca promete acima do plato ───────

@pytest.mark.parametrize("cobertura_alvo", [0.95, 0.97, 0.99, 0.999])
def test_nunca_promete_cobertura_acima_do_plato(cobertura_alvo):
    r = ddsimca_sample_size_guidance(cobertura_alvo)
    assert r.alcancavel is False
    assert r.n_sugerido is None
    assert r.cobertura_no_n_sugerido is None
    assert r.recomendar_conformal is True
    assert "conformal" in r.ressalva.lower()
    assert "acima do plato" in r.ressalva.lower()


def test_alvo_exatamente_no_plato_ainda_nao_promete():
    """`>` estrito no gate: o plato e' o TETO do que foi observado, nao
    uma garantia -- pedir exatamente o teto ainda deve ser tratado com a
    mesma cautela (nao alcancavel), nao um caso de borda favoravel."""
    r = ddsimca_sample_size_guidance(PLATO_COBERTURA_DDSIMCA + 1e-9)
    assert r.alcancavel is False


@pytest.mark.parametrize("cobertura_alvo,n_esperado_no_maximo", [
    (0.80, 5), (0.90, 40), (0.93, 80),
])
def test_alvo_abaixo_do_plato_sugere_n_da_tabela_medida(
        cobertura_alvo, n_esperado_no_maximo):
    r = ddsimca_sample_size_guidance(cobertura_alvo)
    assert r.alcancavel is True
    assert r.recomendar_conformal is False
    assert r.n_sugerido is not None
    assert r.n_sugerido <= n_esperado_no_maximo
    # o n sugerido tem que vir literalmente da tabela medida -- nunca um
    # numero interpolado/inventado fora dos pontos realmente medidos.
    assert r.n_sugerido in dict(MEASURED_DDSIMCA_COVERAGE_TABLE)
    assert r.cobertura_no_n_sugerido >= cobertura_alvo


def test_n_sugerido_e_o_menor_que_atinge_o_alvo():
    """Nunca sugere um n maior que o necessario -- o objetivo e' o menor
    numero de amostras fisicas que atinge a cobertura pedida."""
    r = ddsimca_sample_size_guidance(0.90)
    # n=20 (cobertura 0.9038) ja' atinge 0.90 -- nao pode sugerir 40/80/etc.
    assert r.n_sugerido == 20


def test_ressalva_sempre_menciona_dado_sintetico_e_faixa_medida():
    for alvo in (0.5, 0.90, 0.99):
        r = ddsimca_sample_size_guidance(alvo)
        assert "sintetico" in r.ressalva.lower()
        assert "extrapola" in r.ressalva.lower() or "faixa medida" in r.ressalva.lower()


def test_cobertura_alvo_fora_do_intervalo_valido_falha_alto():
    with pytest.raises(ValueError, match="cobertura_alvo"):
        ddsimca_sample_size_guidance(0.0)
    with pytest.raises(ValueError, match="cobertura_alvo"):
        ddsimca_sample_size_guidance(1.0)
    with pytest.raises(ValueError, match="cobertura_alvo"):
        ddsimca_sample_size_guidance(-0.1)


def test_tabela_medida_e_monotona_crescente_em_n():
    """Contrato de forma da tabela -- os pontos estao em ordem crescente
    de n (a logica de busca do menor n depende disso)."""
    ns = [n for n, _ in MEASURED_DDSIMCA_COVERAGE_TABLE]
    assert ns == sorted(ns)


def test_plato_e_conservador_frente_a_tabela_medida():
    """PLATO_COBERTURA_DDSIMCA tem que ser <= todo ponto medido em n
    grande (n>=150) -- e' o teto conservador, nunca otimista frente ao
    que foi de fato observado."""
    pontos_grandes = [cov for n, cov in MEASURED_DDSIMCA_COVERAGE_TABLE
                       if n >= 150]
    assert all(PLATO_COBERTURA_DDSIMCA <= cov for cov in pontos_grandes)
