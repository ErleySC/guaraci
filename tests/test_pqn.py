# -*- coding: utf-8 -*-
"""Testes de PQN (Probabilistic Quotient Normalization) em
preprocessamento.py -- proposta T6 da rodada multiagente de 2026-09-10.

Referência: Dieterle et al. (2006), Anal. Chem. 78:4281-4290,
DOI 10.1021/ac051632c.
"""
from __future__ import annotations

import numpy as np

from guaraci.config import Config
from guaraci.preprocessamento import PQN, build_preprocessor


def _espectros_com_diluicao(seed=0, n=30, p=60):
    """Um padrão espectral fixo, cada amostra é o padrão vezes um fator de
    DILUIÇÃO aleatório (o cenário exato que PQN foi desenhado para
    corrigir), mais um pico "analito" que varia de tamanho de forma
    INDEPENDENTE da diluição -- para testar que a mediana não deixa o
    analito dominar a estimativa do fator."""
    rng = np.random.default_rng(seed)
    eixo = np.arange(p, dtype=float)
    padrao = (np.exp(-0.5 * ((eixo - 20) / 3) ** 2)
              + 0.5 * np.exp(-0.5 * ((eixo - 45) / 4) ** 2)) + 1.0
    diluicao = rng.uniform(0.3, 3.0, size=n)
    analito = np.exp(-0.5 * ((eixo - 10) / 1.5) ** 2)
    tamanho_analito = rng.uniform(0, 5.0, size=n)
    X = diluicao[:, None] * (padrao[None, :] + tamanho_analito[:, None] * analito[None, :])
    return X, diluicao


def test_pqn_recupera_espectros_proporcionais_ao_mesmo_fator():
    """Contra-prova central de PQN: amostras que só diferem por um fator
    de diluição escalar devem sair (quase) IDÊNTICAS depois da correção."""
    p = 50
    eixo = np.arange(p, dtype=float)
    padrao = np.exp(-0.5 * ((eixo - 25) / 4) ** 2) + 1.0
    fatores = np.array([0.5, 1.0, 2.0, 3.5])
    X = fatores[:, None] * padrao[None, :]

    Xc = PQN().fit_transform(X)
    for i in range(1, len(fatores)):
        assert np.allclose(Xc[0], Xc[i], atol=1e-8)


def test_pqn_robusto_a_pico_de_analito_variavel():
    """A mediana das razões não deve deixar 1-2 canais com analito forte
    distorcer a estimativa de diluição do espectro inteiro."""
    X, diluicao = _espectros_com_diluicao(seed=2)
    Xc = PQN().fit_transform(X)
    # Apos a correcao, a amplitude do "corpo" do espectro (fora do pico do
    # analito) deve ficar proxima entre as amostras -- nao mais escalada
    # pela diluicao original.
    canais_corpo = slice(30, 50)   # longe do pico do analito (canal ~10)
    amplitudes = Xc[:, canais_corpo].mean(axis=1)
    cv_antes = diluicao.std() / diluicao.mean()
    cv_depois = amplitudes.std() / abs(amplitudes.mean())
    assert cv_depois < cv_antes


def test_pqn_referencia_e_do_treino_nao_vaza_do_teste():
    """Mesmo padrão de MSC/EMSC: a referência é ajustada só no treino
    (fit) e reaplicada no teste (transform), sem reajustar."""
    X_treino, _ = _espectros_com_diluicao(seed=3, n=20)
    X_teste, _ = _espectros_com_diluicao(seed=4, n=5)
    pqn = PQN().fit(X_treino)
    ref_antes = pqn.ref_.copy()
    pqn.transform(X_teste)
    assert np.array_equal(pqn.ref_, ref_antes)   # transform nao reajusta


def test_pqn_referencia_degenerada_nao_gera_nan_nem_crasha():
    X = np.zeros((5, 10))
    Xc = PQN().fit_transform(X)
    assert np.all(np.isfinite(Xc))
    assert Xc.shape == X.shape


def test_pqn_entra_no_preset_custom_via_config():
    cfg = Config(default_preprocessing="custom", apply_snv=False, apply_sg=False,
                 apply_mc=False, apply_pqn=True)
    pipe = build_preprocessor(cfg)
    assert "pqn" in dict(pipe.steps)


def test_pqn_desligado_por_padrao_no_preset_custom():
    cfg = Config(default_preprocessing="custom", apply_snv=False, apply_sg=False,
                 apply_mc=False)
    pipe = build_preprocessor(cfg)
    assert "pqn" not in dict(pipe.steps)
