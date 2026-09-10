# -*- coding: utf-8 -*-
"""Testes de linearity.py (Bloco 13d, Frente 1, L1-L3)."""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.linearity import lack_of_fit_test


def _curva_linear(seed=0, n_niveis=6, n_replicas=3, ruido=0.02):
    rng = np.random.default_rng(seed)
    niveis = np.linspace(0, 10, n_niveis)
    y_ref, y_pred, grupos = [], [], []
    for i, nivel in enumerate(niveis):
        for r in range(n_replicas):
            y_ref.append(nivel)
            y_pred.append(nivel + rng.normal(scale=ruido))   # y = x + ruido: reta perfeita
            grupos.append(f"G{i}")
    return np.array(y_ref), np.array(y_pred), np.array(grupos)


def _curva_com_curvatura(seed=0, n_niveis=6, n_replicas=3, ruido=0.02, curvatura=0.6):
    rng = np.random.default_rng(seed)
    niveis = np.linspace(0, 10, n_niveis)
    y_ref, y_pred, grupos = [], [], []
    for i, nivel in enumerate(niveis):
        # termo quadratico -- desvio sistematico da reta, mesma escala em
        # todos os niveis (nao e' so' ruido de um nivel especifico)
        resposta_curva = nivel + curvatura * (nivel - 5.0) ** 2 / 10.0
        for r in range(n_replicas):
            y_ref.append(nivel)
            y_pred.append(resposta_curva + rng.normal(scale=ruido))
            grupos.append(f"G{i}")
    return np.array(y_ref), np.array(y_pred), np.array(grupos)


# =========================================================================
#  Contrato / casos degenerados (L2)
# =========================================================================

def test_dados_lineares_nao_detectam_falta_de_ajuste():
    y_ref, y_pred, grupos = _curva_linear()
    r = lack_of_fit_test(y_ref, y_pred, grupos)
    assert r.computavel
    assert r.linear is True
    assert r.p_value > 0.05
    assert r.n_niveis == 6
    assert r.n_niveis_com_replica == 6


def test_niveis_insuficientes_nao_e_computavel():
    y_ref = np.array([1.0, 1.0, 2.0, 2.0])
    y_pred = np.array([1.1, 1.05, 2.1, 2.05])
    grupos = np.array(["G0", "G0", "G1", "G1"])
    r = lack_of_fit_test(y_ref, y_pred, grupos)
    assert not r.computavel
    assert r.linear is None
    assert "niveis insuficientes" in r.motivo


def test_sem_nenhuma_replica_verdadeira_nao_e_computavel():
    """Nenhum grupo repetido -- 1 amostra por nivel -- sem como separar
    erro puro de falta de ajuste (L2)."""
    y_ref = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    y_pred = np.array([0.1, 2.2, 3.9, 6.1, 7.8])
    grupos = np.array(["G0", "G1", "G2", "G3", "G4"])
    r = lack_of_fit_test(y_ref, y_pred, grupos)
    assert not r.computavel
    assert "replica" in r.motivo


def test_faixa_de_trabalho_filtra_amostras_fora(caplog=None):
    """L3: uma faixa de trabalho estreita reduz os niveis usados no teste
    -- confirma que a faixa e' de fato aplicada, nao ignorada."""
    y_ref, y_pred, grupos = _curva_linear(n_niveis=8)
    r_sem_faixa = lack_of_fit_test(y_ref, y_pred, grupos)
    r_com_faixa = lack_of_fit_test(y_ref, y_pred, grupos, faixa_trabalho=(2.0, 6.0))
    assert r_sem_faixa.computavel and r_com_faixa.computavel
    assert r_com_faixa.n_niveis < r_sem_faixa.n_niveis


def test_comprimentos_incompativeis_falha_alto():
    with pytest.raises(ValueError, match="comprimento"):
        lack_of_fit_test(np.array([1.0, 2.0]), np.array([1.0]), np.array(["G0", "G1"]))


# =========================================================================
#  Contra-prova (L1): curvatura deliberada -> F significativo
# =========================================================================

def test_contraprova_curvatura_deliberada_produz_falta_de_ajuste_significativa():
    y_ref, y_pred, grupos = _curva_com_curvatura()
    r = lack_of_fit_test(y_ref, y_pred, grupos)
    assert r.computavel
    assert r.linear is False, (
        f"curvatura deliberada deveria produzir falta de ajuste "
        f"significativa (p<0.05); obtido p={r.p_value}")
    assert r.p_value < 0.01
    assert r.F > 10, (
        "F deveria ser claramente grande com curvatura desta magnitude -- "
        "se nao for, o teste pode nao estar sensivel a curvatura real")


def test_contraprova_curvatura_maior_produz_F_maior():
    """Reforca que o teste MEDE curvatura (nao so' acerta um limiar fixo):
    dobrar a curvatura tem que aumentar F, nao dar o mesmo valor."""
    _, _, grupos = _curva_com_curvatura()
    y_ref1, y_pred1, _ = _curva_com_curvatura(curvatura=0.3)
    y_ref2, y_pred2, _ = _curva_com_curvatura(curvatura=1.2)
    r1 = lack_of_fit_test(y_ref1, y_pred1, grupos)
    r2 = lack_of_fit_test(y_ref2, y_pred2, grupos)
    assert r1.computavel and r2.computavel
    assert r2.F > r1.F, (
        f"curvatura maior deveria produzir F maior (F1={r1.F}, F2={r2.F}) "
        f"-- se nao aumentou, o teste pode nao estar de fato medindo a "
        f"magnitude da curvatura")
