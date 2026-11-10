# -*- coding: utf-8 -*-
"""iPLS: a validacao cruzada INTERNA que escolhe o melhor intervalo respeita os grupos (replicas fisicas).

Antes, `_mask_melhor_intervalo` chamava `_cv_local` sem grupos: replicas do mesmo objeto caiam em dobras de treino e de teste
diferentes, e o intervalo escolhido premiava a similaridade entre replicas. As buscas SPA/AG/CARS ja eram group-aware (achado B1-3).
"""
from __future__ import annotations

import numpy as np

import guaraci.selecao_variaveis as sv
from guaraci.selecao_variaveis import _avaliar_subset_nested_cv, _mask_melhor_intervalo


def _dados(n_grupos=20, replicas=4, p=40, seed=3):
    rng = np.random.default_rng(seed)
    y_grupo = np.tile([0, 1], n_grupos // 2)
    y = np.repeat(y_grupo, replicas)
    grupos = np.repeat(np.arange(n_grupos), replicas)
    # cada objeto tem uma assinatura propria (replicas muito parecidas entre si): o terreno onde o vazamento aparece
    base = rng.normal(0, 1, size=(n_grupos, p))
    X = np.repeat(base, replicas, axis=0) + rng.normal(0, 0.05, size=(n_grupos * replicas, p))
    return X, np.eye(2)[y], y, grupos


def _espiar_cv(monkeypatch):
    capturadas = []
    original = sv._avaliar_subset_cv

    def espia(X_sel, Y_bin, y_int, cv_indices, n_lv):
        capturadas.append(cv_indices)
        return original(X_sel, Y_bin, y_int, cv_indices, n_lv)

    monkeypatch.setattr(sv, "_avaliar_subset_cv", espia)
    return capturadas


def _grupos_cruzam_dobras(cv, grupos):
    return any(set(grupos[tr]) & set(grupos[va]) for tr, va in cv)


def test_cv_interna_do_ipls_nunca_separa_replicas_do_mesmo_grupo(monkeypatch):
    X, Y, _y, grupos = _dados()
    capturadas = _espiar_cv(monkeypatch)
    _mask_melhor_intervalo(X, Y, 2, 4, seed=0, grupos=grupos)
    assert capturadas, "a CV interna nao foi usada"
    for cv in capturadas:
        assert not _grupos_cruzam_dobras(cv, grupos)


def test_sem_grupos_o_comportamento_antigo_continua_sendo_o_estratificado(monkeypatch):
    X, Y, _y, grupos = _dados()
    capturadas = _espiar_cv(monkeypatch)
    _mask_melhor_intervalo(X, Y, 2, 4, seed=0)
    assert any(_grupos_cruzam_dobras(cv, grupos) for cv in capturadas), "o controle deveria vazar replicas (sem grupos)"


def test_avaliacao_aninhada_entrega_os_grupos_do_treino_de_cada_dobra_externa():
    X, Y, y, grupos = _dados()
    from guaraci.validacao_estatistica import StableStratifiedGroupKFold
    externo = list(StableStratifiedGroupKFold(n_splits=4, seed=0).split(np.zeros(len(y)), y, groups=grupos))
    vistos = []

    def selecionar(Xtr, Ytr, nlv, gtr=None):
        vistos.append(gtr)
        return np.ones(Xtr.shape[1], dtype=bool)

    _avaliar_subset_nested_cv(X, Y, y, externo, 2, selecionar, mae_id=grupos, passar_grupos=True)
    assert len(vistos) == len(externo)
    for (tr, _va), g in zip(externo, vistos):
        assert np.array_equal(g, grupos[tr])


def test_sem_mae_id_a_selecao_continua_com_tres_argumentos():
    X, Y, y, _grupos = _dados()
    from guaraci.validacao_estatistica import StableStratifiedGroupKFold
    externo = list(StableStratifiedGroupKFold(n_splits=4, seed=0).split(np.zeros(len(y)), y, groups=np.repeat(np.arange(20), 4)))
    chamadas = []

    def selecionar(Xtr, Ytr, nlv, gtr=None):
        chamadas.append(gtr)
        return np.ones(Xtr.shape[1], dtype=bool)

    _avaliar_subset_nested_cv(X, Y, y, externo, 2, selecionar, passar_grupos=True)
    assert chamadas and all(g is None for g in chamadas)
