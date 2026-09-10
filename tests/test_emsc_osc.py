# -*- coding: utf-8 -*-
"""Testes de EMSC/OSC em preprocessamento.py (Bloco 16) -- espectros
SINTETICOS com espalhamento multiplicativo/aditivo conhecido (EMSC) e alvo
de classe conhecido (OSC).
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline

from guaraci.config import Config
from guaraci.preprocessamento import EMSC, OSC, MSC, SNV, build_preprocessor


def _espectros_com_espalhamento(seed=0, n=40, p=50):
    """Espectros com 2 picos gaussianos + espalhamento MULTIPLICATIVO
    (ganho por amostra) e ADITIVO (deslocamento de linha de base LINEAR
    por amostra) -- o tipo de distorcao que EMSC foi desenhado para tratar
    alem do que MSC ja trata (so' o termo multiplicativo)."""
    rng = np.random.default_rng(seed)
    eixo = np.arange(p, dtype=float)
    base = (np.exp(-0.5 * ((eixo - 15) / 3) ** 2)
            + 0.7 * np.exp(-0.5 * ((eixo - 35) / 4) ** 2))
    ganho = rng.uniform(0.7, 1.4, size=n)
    inclinacao_baseline = rng.uniform(-0.01, 0.01, size=n)
    ruido = rng.normal(0, 0.005, size=(n, p))
    X = ganho[:, None] * base[None, :] + inclinacao_baseline[:, None] * eixo[None, :] + ruido
    return X, eixo


def test_emsc_produz_espectro_diferente_de_msc_e_snv():
    X, eixo = _espectros_com_espalhamento()
    Xc_emsc = EMSC(eixo=eixo, ordem_polinomial=2).fit_transform(X)
    Xc_msc = MSC().fit_transform(X)
    Xc_snv = SNV().fit_transform(X)

    assert Xc_emsc.shape == X.shape
    assert not np.allclose(Xc_emsc, Xc_msc, atol=1e-6)
    assert not np.allclose(Xc_emsc, Xc_snv, atol=1e-6)
    assert np.all(np.isfinite(Xc_emsc))


def test_emsc_remove_baseline_linear_melhor_que_msc():
    # EMSC (ordem>=1) modela a linha de base LINEAR explicitamente; MSC nao
    # -- o residuo pos-correcao contra o espectro "puro" (sem ganho/baseline)
    # deve ser menor com EMSC nesse caso construido para favorece-lo.
    rng = np.random.default_rng(1)
    n, p = 30, 40
    eixo = np.arange(p, dtype=float)
    base_pura = np.exp(-0.5 * ((eixo - 20) / 3) ** 2)
    ganho = rng.uniform(0.8, 1.3, size=n)
    inclinacao = rng.uniform(-0.02, 0.02, size=n)
    X = ganho[:, None] * base_pura[None, :] + inclinacao[:, None] * eixo[None, :]

    Xc_emsc = EMSC(eixo=eixo, ordem_polinomial=1).fit_transform(X)
    Xc_msc = MSC().fit_transform(X)

    ref_norm = base_pura / np.linalg.norm(base_pura)
    def _erro_medio(Xc):
        Xc_norm = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
        return np.mean(np.linalg.norm(Xc_norm - ref_norm[None, :], axis=1))

    assert _erro_medio(Xc_emsc) < _erro_medio(Xc_msc)


def test_emsc_ordem_zero_sem_interferentes_aproxima_msc():
    X, eixo = _espectros_com_espalhamento(seed=2)
    Xc_emsc0 = EMSC(eixo=eixo, ordem_polinomial=0).fit_transform(X)
    Xc_msc = MSC().fit_transform(X)
    # Mesma base de regressao (so' [1, ref]) -- diferem no maximo por
    # resolucao numerica (lstsq vs formula fechada), nao estruturalmente.
    np.testing.assert_allclose(Xc_emsc0, Xc_msc, atol=1e-6)


def test_emsc_com_interferente_conhecido_muda_resultado():
    X, eixo = _espectros_com_espalhamento(seed=3)
    rng = np.random.default_rng(99)
    interferente = rng.normal(0, 1, size=len(eixo))
    Xc_sem = EMSC(eixo=eixo, ordem_polinomial=1).fit_transform(X)
    Xc_com = EMSC(eixo=eixo, ordem_polinomial=1,
                  interferentes=interferente).fit_transform(X)
    assert not np.allclose(Xc_sem, Xc_com, atol=1e-6)


def test_emsc_rejeita_eixo_ausente_usa_indice():
    X, _eixo = _espectros_com_espalhamento(seed=4, p=30)
    # Sem eixo explicito: usa indice do canal -- so' precisa nao quebrar e
    # ser numericamente estavel.
    Xc = EMSC(ordem_polinomial=2).fit_transform(X)
    assert np.all(np.isfinite(Xc))


# ---------------------------------------------------------------------------
#  OSC
# ---------------------------------------------------------------------------

def _dataset_classificacao(seed=0, n=60, p=30):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    eixo = np.arange(p, dtype=float)
    sinal_classe = np.exp(-0.5 * ((eixo - 15) / 3) ** 2)
    X = rng.normal(0, 1, size=(n, p)) + y[:, None] * 2.0 * sinal_classe[None, :]
    Y_bin = np.eye(2)[y]
    return X, Y_bin, y


def test_osc_produz_espectro_diferente_de_so_centrar():
    X, Y_bin, _y = _dataset_classificacao()
    osc = OSC(n_componentes=1)
    Xc_osc = osc.fit_transform(X, Y_bin)
    Xc_mc = X - X.mean(axis=0)
    assert not np.allclose(Xc_osc, Xc_mc, atol=1e-6)
    assert np.all(np.isfinite(Xc_osc))


def test_osc_reduz_variancia_sem_destruir_correlacao_com_y():
    """OSC remove variacao ORTOGONAL a y -- a norma total de X deve cair
    (parte da variancia foi removida), mas um PLS treinado no X corrigido
    ainda precisa separar as classes razoavelmente bem (o sinal
    correlacionado com y foi preservado por construcao do metodo)."""
    X, Y_bin, y = _dataset_classificacao(seed=5, n=80)
    osc = OSC(n_componentes=1)
    Xc = osc.fit_transform(X, Y_bin)

    variancia_antes = np.var(X)
    variancia_depois = np.var(Xc)
    assert variancia_depois < variancia_antes

    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(Xc - Xc.mean(axis=0), Y_bin)
    y_hat = np.argmax(pls.predict(Xc - Xc.mean(axis=0)), axis=1)
    acc = float(np.mean(y_hat == y))
    assert acc > 0.8, f"OSC destruiu o sinal de classe (acc={acc:.3f})"


def test_osc_fit_exige_y():
    X, _Y_bin, _y = _dataset_classificacao(seed=6)
    with pytest.raises(TypeError):
        OSC().fit(X)  # falta y -- assinatura exige


def test_osc_transform_de_dado_novo_usa_pesos_do_treino():
    X, Y_bin, _y = _dataset_classificacao(seed=7, n=80)
    X_treino, X_teste = X[:60], X[60:]
    Y_treino = Y_bin[:60]
    osc = OSC(n_componentes=1).fit(X_treino, Y_treino)
    Xc_teste = osc.transform(X_teste)
    assert Xc_teste.shape == X_teste.shape
    assert np.all(np.isfinite(Xc_teste))


# ---------------------------------------------------------------------------
#  Integracao com build_preprocessor (leque configuravel, cfg.apply_emsc/
#  cfg.apply_osc -- mesmo padrao de apply_snv/apply_sg/apply_mc)
# ---------------------------------------------------------------------------

def test_build_preprocessor_aplica_emsc_quando_ligado():
    cfg = Config()
    cfg.default_preprocessing = "custom"
    cfg.apply_snv = False
    cfg.apply_sg = False
    cfg.apply_mc = True
    cfg.apply_emsc = True
    pre = build_preprocessor(cfg)
    nomes = [nome for nome, _ in pre.steps]
    assert "emsc" in nomes


def test_build_preprocessor_aplica_osc_quando_ligado_fim_a_fim():
    X, Y_bin, y = _dataset_classificacao(seed=8, n=60)
    cfg = Config()
    cfg.default_preprocessing = "custom"
    cfg.apply_snv = True
    cfg.apply_sg = False
    cfg.apply_mc = True
    cfg.apply_osc = True
    cfg.osc_n_componentes = 1

    pipe = Pipeline([
        ("preproc", build_preprocessor(cfg)),
        ("pls", PLSRegression(n_components=2, scale=False)),
    ])
    pipe.fit(X, Y_bin)  # y precisa chegar ate' a etapa OSC dentro do Pipeline
    nomes = [nome for nome, _ in pipe.named_steps["preproc"].steps]
    assert nomes[-1] == "osc"
    y_hat = np.argmax(pipe.predict(X), axis=1)
    assert np.mean(y_hat == y) > 0.7


def test_build_preprocessor_sem_emsc_osc_nao_muda_pipeline_default():
    cfg = Config()
    pre = build_preprocessor(cfg)
    nomes = [nome for nome, _ in pre.steps]
    assert "emsc" not in nomes
    assert "osc" not in nomes
