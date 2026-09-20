# -*- coding: utf-8 -*-
"""test_integracao_vrm_dual_spls.py -- Parte C da varredura final
(2026-09-20): confirma que VRM (`aumento_dados.py`) e Dual-sPLS
(`dual_spls.py`), implementados em paralelo por agentes separados, NAO
conflitam entre si nem com a garantia group-aware quando usados juntos no
mesmo pipeline (VRM aumenta o fold de treino, Dual-sPLS ajusta sobre o
treino aumentado, predicao no fold de validacao original).

Isto e' um teste de INTEGRACAO, nao de mutation/portao -- cada tecnica ja
tem seu proprio portao de aceite isolado (`scripts/medicoes/
portao_vrm_tecator.py`, `scripts/benchmark_dual_spls_tecator.py`). Aqui a
pergunta e' estrutural: "os dois juntos quebram alguma garantia?", nao
"os dois juntos melhoram a metrica?"."""
from __future__ import annotations

import numpy as np

from guaraci.aumento_dados import AumentoVRM
from guaraci.dual_spls import DualSPLS
from guaraci.validacao_estatistica import StableStratifiedGroupKFold


def _dataset_sintetico(rng: np.random.Generator, n: int = 40, p: int = 30):
    """Espectro sintetico simples (nao precisa ser realista -- so' testa
    que o encadeamento VRM -> Dual-sPLS nao lanca excecao nem vaza grupo)."""
    eixo = np.linspace(0, 1, p)
    coefs = rng.normal(0.0, 1.0, size=n)
    X = coefs[:, None] * np.sin(eixo * 3.0)[None, :] + rng.normal(0.0, 0.02, size=(n, p))
    y = coefs * 2.0 + rng.normal(0.0, 0.05, size=n)
    grupos = np.arange(n)  # sem replica fisica -- 1 grupo por amostra original
    return X, y, grupos


def test_vrm_seguido_de_dual_spls_nao_lanca_excecao_e_produz_forma_correta():
    rng = np.random.default_rng(7)
    X, y, grupos = _dataset_sintetico(rng)

    splitter = StableStratifiedGroupKFold(n_splits=4, seed=7)
    folds = list(splitter.split(np.zeros(len(X)), np.zeros(len(X)), groups=grupos))
    idx_tr, idx_va = folds[0]

    X_tr, y_tr, grupos_tr = X[idx_tr], y[idx_tr], grupos[idx_tr]
    X_va, y_va = X[idx_va], y[idx_va]

    # VRM so' no fold de treino -- validacao nunca vê amostra aumentada.
    X_tr_aug, y_tr_aug, grupos_tr_aug = AumentoVRM(
        n_aumentos=3, seed=11).aumentar(X_tr, y_tr, grupos_tr)

    modelo = DualSPLS(n_components=3, sparsity=0.5)
    modelo.fit(X_tr_aug, y_tr_aug)
    y_hat = modelo.predict(X_va)

    assert y_hat.shape == y_va.shape
    assert np.all(np.isfinite(y_hat))


def test_augmentar_treino_nao_introduz_grupo_que_vaze_para_validacao():
    """Garantia central da Parte C: mesmo com VRM rodando ANTES do fit do
    Dual-sPLS, nenhum grupo (original ou herdado por amostra aumentada)
    aparece tanto no treino quanto na validacao -- a augmentacao acontece
    estritamente DEPOIS do split, nunca antes."""
    rng = np.random.default_rng(3)
    X, y, grupos = _dataset_sintetico(rng, n=48, p=25)

    splitter = StableStratifiedGroupKFold(n_splits=4, seed=3)
    for idx_tr, idx_va in splitter.split(np.zeros(len(X)), np.zeros(len(X)), groups=grupos):
        grupos_tr = grupos[idx_tr]
        grupos_va = grupos[idx_va]

        # Grupos originais: split e' particao, nenhuma sobreposicao.
        assert set(grupos_tr).isdisjoint(set(grupos_va))

        X_tr_aug, y_tr_aug, grupos_tr_aug = AumentoVRM(
            n_aumentos=2, seed=5).aumentar(X[idx_tr], y[idx_tr], grupos_tr)

        # Todo grupo do treino aumentado (original + herdado) continua
        # fora do conjunto de grupos de validacao -- a augmentacao nunca
        # CRIA um grupo que colida com validacao, porque so' herda ids
        # que ja' estavam do lado do treino.
        assert set(grupos_tr_aug).isdisjoint(set(grupos_va))
        # E o augmento nao perdeu nenhum grupo original do treino.
        assert set(grupos_tr).issubset(set(grupos_tr_aug))

        modelo = DualSPLS(n_components=2, sparsity=0.7)
        modelo.fit(X_tr_aug, y_tr_aug)
        y_hat = modelo.predict(X[idx_va])
        assert np.all(np.isfinite(y_hat))


def test_dual_spls_aceita_x_com_linhas_repetidas_sem_erro_de_posto():
    """VRM produz linhas quase-identicas (perturbacao pequena) -- confirma
    que o ajuste do Dual-sPLS nao quebra com essa estrutura de posto
    proximo de deficiente (situacao que NAO ocorre com dado original sem
    augmentacao)."""
    rng = np.random.default_rng(1)
    X, y, grupos = _dataset_sintetico(rng, n=20, p=15)

    X_aug, y_aug, grupos_aug = AumentoVRM(
        n_aumentos=5, amplitude_multiplicativa=0.001,
        amplitude_baseline=0.001, seed=1).aumentar(X, y, grupos)

    modelo = DualSPLS(n_components=4, sparsity=0.5)
    modelo.fit(X_aug, y_aug)
    y_hat = modelo.predict(X_aug)
    assert np.all(np.isfinite(y_hat))


def test_config_nao_habilita_vrm_por_padrao():
    """Confirma que a nota adicionada em config.py pela Parte A nao virou
    acidentalmente uma flag default=True -- VRM tem que continuar opt-in
    explicito via `AumentoVRM`/`aumentar_vrm`, nunca no caminho default do
    pipeline."""
    from guaraci.config import Config

    cfg = Config()
    assert not hasattr(cfg, "apply_vrm") or getattr(cfg, "apply_vrm") is False
