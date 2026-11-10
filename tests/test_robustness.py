# -*- coding: utf-8 -*-
"""Testes de robustness.py (Bloco 13d, Frente 1, R1-R3)."""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.robustness import (
    RobustnessResult,
    avaliar_bal_acc_plsda,
    avaliar_rmsep_plsr,
    baseline_drift_variants,
    gaussian_noise_variants,
    preprocessing_config_variants,
    run_robustness_protocol,
)


def _espectros_com_teor(seed=0, n_grupos=20, n_replicas=3, p=40):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=p)
    X, y, mae = [], [], []
    for g in range(n_grupos):
        teor = rng.uniform(0, 10)
        base = rng.normal(size=p) * 0.1
        for _r in range(n_replicas):
            X.append(base + w * (teor / 10.0) + rng.normal(scale=0.01, size=p))
            y.append(teor)
            mae.append(f"G{g}")
    return np.array(X), np.array(y), np.array(mae)


def _espectros_com_classe(seed=0, n_grupos_por_classe=8, n_replicas=3, p=30, n_classes=3):
    rng = np.random.default_rng(seed)
    X, y, mae = [], [], []
    for c in range(n_classes):
        centro = rng.normal(loc=c * 4.0, size=p)
        for g in range(n_grupos_por_classe):
            base = centro + rng.normal(scale=0.2, size=p)
            for _r in range(n_replicas):
                X.append(base + rng.normal(scale=0.03, size=p))
                y.append(f"classe_{c}")
                mae.append(f"C{c}G{g}")
    return np.array(X), np.array(y), np.array(mae)


# =========================================================================
#  Geradores de perturbacao (R1)
# =========================================================================

def test_gaussian_noise_variants_gera_n_replicas_por_nivel():
    X, _y, _mae = _espectros_com_teor()
    variantes = gaussian_noise_variants(X, niveis=(0.001, 0.01), n_replicas=4)
    assert set(variantes) == {"ruido_gaussiano_0.001", "ruido_gaussiano_0.01"}
    for lista in variantes.values():
        assert len(lista) == 4
        for Xv in lista:
            assert Xv.shape == X.shape
            assert not np.array_equal(Xv, X)


def test_baseline_drift_variants_gera_n_replicas_por_nivel():
    X, _y, _mae = _espectros_com_teor()
    variantes = baseline_drift_variants(X, niveis=(0.005,), n_replicas=3)
    assert len(variantes) == 1
    lista = next(iter(variantes.values()))
    assert len(lista) == 3
    for Xv in lista:
        assert Xv.shape == X.shape


def test_preprocessing_config_variants_produz_configs_distintas(pq):
    cfg_base = pq.Config(sg_window=25, sg_polyorder=2,
                         default_preprocessing="msc_sg_mc")
    variantes = preprocessing_config_variants(cfg_base)
    assert len(variantes) > 0
    for nome, cfg_v in variantes.items():
        assert cfg_v != cfg_base or nome  # so' existe se realmente distinta
        algo_mudou = (cfg_v.sg_window != cfg_base.sg_window
                      or cfg_v.sg_polyorder != cfg_base.sg_polyorder
                      or cfg_v.default_preprocessing != cfg_base.default_preprocessing)
        assert algo_mudou, f"variante '{nome}' nao difere do cfg_base"


# =========================================================================
#  Avaliadores PLS-R / PLS-DA (R3)
# =========================================================================

def test_avaliar_rmsep_plsr_retorna_numero_finito():
    X, y, mae = _espectros_com_teor()
    rmsep = avaliar_rmsep_plsr(X, y, mae)
    assert np.isfinite(rmsep)
    assert rmsep >= 0


def test_avaliar_bal_acc_plsda_retorna_numero_no_intervalo_valido():
    X, y, mae = _espectros_com_classe()
    bal_acc = avaliar_bal_acc_plsda(X, y, mae)
    assert np.isfinite(bal_acc)
    assert 0.0 <= bal_acc <= 1.0


# =========================================================================
#  Protocolo (R2): reporta INTERVALO, nunca binario
# =========================================================================

def test_run_robustness_protocol_reporta_intervalo_nao_binario():
    X, y, mae = _espectros_com_teor()
    variantes_dados = gaussian_noise_variants(X, niveis=(0.005,), n_replicas=5)

    variantes_callables = {
        nome: [lambda Xv=Xv: avaliar_rmsep_plsr(Xv, y, mae) for Xv in lista]
        for nome, lista in variantes_dados.items()
    }
    resultados = run_robustness_protocol(
        lambda: avaliar_rmsep_plsr(X, y, mae), variantes_callables)

    assert set(resultados) == set(variantes_dados)
    r = resultados["ruido_gaussiano_0.005"]
    assert isinstance(r, RobustnessResult)
    assert r.n_replicas == 5
    assert len(r.valores) == 5
    assert r.minimo <= r.mediana <= r.maximo
    assert r.variacao_absoluta == pytest.approx(r.maximo - r.minimo)
    # R2: nao existe campo booleano de aprovado/reprovado no resultado --
    # confere via introspeccao dos campos do dataclass, nao so' por leitura.
    campos = {f.name for f in __import__("dataclasses").fields(RobustnessResult)}
    assert not any("aprov" in c or "reprov" in c or "pass" in c for c in campos), (
        "RobustnessResult nao deveria ter campo de veredito binario (R2)")


# =========================================================================
#  Contra-prova (R2/R3): perturbacao maior -> variacao maior. Se a
#  variacao fosse constante independente da magnitude, o protocolo nao
#  estaria medindo nada de verdade.
# =========================================================================

def test_contraprova_perturbacao_maior_produz_variacao_maior():
    X, y, mae = _espectros_com_teor(n_grupos=25)
    variantes_dados = gaussian_noise_variants(X, niveis=(0.0005, 0.05), n_replicas=6)

    variantes_callables = {
        nome: [lambda Xv=Xv: avaliar_rmsep_plsr(Xv, y, mae) for Xv in lista]
        for nome, lista in variantes_dados.items()
    }
    resultados = run_robustness_protocol(
        lambda: avaliar_rmsep_plsr(X, y, mae), variantes_callables)

    r_pequena = resultados["ruido_gaussiano_0.0005"]
    r_grande = resultados["ruido_gaussiano_0.05"]
    assert r_grande.variacao_absoluta > r_pequena.variacao_absoluta, (
        f"ruido 100x maior deveria produzir variacao de RMSEP maior "
        f"(pequena={r_pequena.variacao_absoluta:.4f}, "
        f"grande={r_grande.variacao_absoluta:.4f}) -- se nao aumentou, o "
        f"protocolo pode nao estar medindo robustez de verdade")
