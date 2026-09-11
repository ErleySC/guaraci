# -*- coding: utf-8 -*-
"""Testes de epo_glsw.py -- proposta T2 da rodada multiagente de
2026-09-10.

Referências: Roger, Chauchard & Bellon-Maurel (2003), Chemom. Intell.
Lab. Syst. 66:191-204, DOI 10.1016/S0169-7439(03)00051-0 (EPO);
Martens, Høy, Wise, Bro & Brockhoff (2003), J. Chemometrics 17:153-165,
DOI 10.1002/cem.780 (GLSW).
"""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.epo_glsw import EPO, GLSW, build_difference_matrix


def _dataset_com_fator_de_perturbacao(seed=0, n_grupos=8, p=30):
    """Cada `grupo_interesse` (o "sinal" a preservar) é medido em 2
    "hospedeiras" (o fator de perturbação a remover) -- a hospedeira
    desloca o espectro por um vetor FIXO, independente do sinal de
    interesse."""
    rng = np.random.default_rng(seed)
    eixo = np.arange(p, dtype=float)
    sinal_base = np.exp(-0.5 * ((eixo - 15) / 3) ** 2)
    deslocamento_hospedeira = 3.0 * np.exp(-0.5 * ((eixo - 8) / 2) ** 2)

    grupo_interesse, fator_incomodo, X = [], [], []
    for g in range(n_grupos):
        amplitude = rng.uniform(0.5, 2.0)   # varia por grupo -- o "sinal"
        for hospedeira in ("h1", "h2"):
            ruido = rng.normal(scale=0.01, size=p)
            espectro = amplitude * sinal_base + ruido
            if hospedeira == "h2":
                espectro = espectro + deslocamento_hospedeira
            X.append(espectro)
            grupo_interesse.append(f"g{g}")
            fator_incomodo.append(hospedeira)
    return (np.array(X), np.array(grupo_interesse), np.array(fator_incomodo),
            deslocamento_hospedeira)


def test_build_difference_matrix_recupera_a_direcao_do_fator():
    X, grupo, fator, deslocamento = _dataset_com_fator_de_perturbacao()
    D = build_difference_matrix(X, grupo, fator)
    assert D.shape == (8, X.shape[1])   # 1 diferenca por grupo (2 niveis)
    # a media das diferencas tem que apontar na direcao do deslocamento
    direcao_media = D.mean(axis=0)
    cos = (direcao_media @ deslocamento) / (
        np.linalg.norm(direcao_media) * np.linalg.norm(deslocamento))
    assert cos > 0.95


def test_build_difference_matrix_levanta_erro_sem_par_formavel():
    X = np.zeros((5, 10))
    grupo = np.array(["g1", "g2", "g3", "g4", "g5"])
    fator = np.array(["h1", "h1", "h1", "h1", "h1"])   # so' 1 nivel em todo grupo
    with pytest.raises(ValueError, match="par formável|par formavel"):
        build_difference_matrix(X, grupo, fator)


def test_build_difference_matrix_exige_mesmo_comprimento():
    with pytest.raises(ValueError, match="comprimento"):
        build_difference_matrix(np.zeros((5, 10)), np.zeros(5), np.zeros(3))


def test_epo_remove_a_direcao_do_fator_de_perturbacao():
    """Contra-prova central de EPO: apos a correcao, o efeito da
    hospedeira quase desaparece -- amostras da mesma condicao de
    interesse, hospedeiras diferentes, ficam muito mais parecidas."""
    X, grupo, fator, _ = _dataset_com_fator_de_perturbacao()
    D = build_difference_matrix(X, grupo, fator)
    epo = EPO(n_componentes=1).fit(D)
    Xc = epo.transform(X)

    # Diferenca media intra-grupo (h1 vs h2 do MESMO grupo de interesse)
    dif_antes, dif_depois = [], []
    for g in np.unique(grupo):
        idx = np.where(grupo == g)[0]
        h1 = idx[fator[idx] == "h1"][0]
        h2 = idx[fator[idx] == "h2"][0]
        dif_antes.append(np.linalg.norm(X[h1] - X[h2]))
        dif_depois.append(np.linalg.norm(Xc[h1] - Xc[h2]))
    assert np.mean(dif_depois) < 0.1 * np.mean(dif_antes)


def test_epo_preserva_a_diferenca_de_amplitude_entre_grupos():
    """EPO nao pode apagar o SINAL de interesse -- so' a direcao do
    incomodo. Grupos com amplitude diferente continuam distinguiveis."""
    X, grupo, fator, _ = _dataset_com_fator_de_perturbacao(seed=3)
    D = build_difference_matrix(X, grupo, fator)
    epo = EPO(n_componentes=1).fit(D)
    Xc = epo.transform(X)
    normas = np.array([np.linalg.norm(Xc[i]) for i in range(len(Xc))])
    assert normas.std() > 0.05 * normas.mean()   # ainda ha' variacao real entre grupos


def test_glsw_atenua_mas_nao_zera_como_epo():
    X, grupo, fator, _ = _dataset_com_fator_de_perturbacao(seed=5)
    D = build_difference_matrix(X, grupo, fator)
    glsw = GLSW(alpha=1e-3).fit(D)
    Xc = glsw.transform(X)

    dif_antes, dif_depois = [], []
    for g in np.unique(grupo):
        idx = np.where(grupo == g)[0]
        h1 = idx[fator[idx] == "h1"][0]
        h2 = idx[fator[idx] == "h2"][0]
        dif_antes.append(np.linalg.norm(X[h1] - X[h2]))
        dif_depois.append(np.linalg.norm(Xc[h1] - Xc[h2]))
    # atenua bastante, mas GLSW com alpha pequeno se aproxima de EPO --
    # o que importa e' que atenue (mesma direcao de efeito de EPO).
    assert np.mean(dif_depois) < 0.5 * np.mean(dif_antes)


def test_glsw_com_alpha_grande_tende_a_nao_alterar_o_espectro():
    X, grupo, fator, _ = _dataset_com_fator_de_perturbacao(seed=6)
    D = build_difference_matrix(X, grupo, fator)
    glsw = GLSW(alpha=1e6).fit(D)
    Xc = glsw.transform(X)
    assert np.allclose(Xc, X, atol=1e-3)


def test_epo_e_glsw_recusam_matriz_de_diferenca_vazia():
    with pytest.raises(ValueError):
        EPO().fit(np.zeros((0, 10)))
    with pytest.raises(ValueError):
        GLSW().fit(np.zeros((0, 10)))
