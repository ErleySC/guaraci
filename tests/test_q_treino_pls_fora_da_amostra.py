"""Lote 3.4 da vistoria (achado A4-03 / V57): o Q de treino do espaco PLS e'
calculado fora da amostra (leave-one-group-out) e com X centrado pela media do
PLS, e a predicao centra igual. Cada teste fixa um comportamento que o codigo
anterior nao tinha (o Q do proprio ajuste, sem centrar)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2
from sklearn.cross_decomposition import PLSRegression

import guaraci.predicao as pr
from guaraci.chemometric_stats import (
    combined_distance, hotelling_t2, mean_and_dof_moments, q_residuals,
    q_residuals_limit, q_residuals_pls_loo)

_P = 200
_X = np.linspace(0, 1, _P)


def _mundo(seed: int):
    rng = np.random.default_rng(seed)
    base = {c: np.stack([np.exp(-0.5 * ((_X - rng.uniform(.1, .9)) / .03) ** 2)
                         for _ in range(4)]) for c in "ABC"}
    media = {c: rng.uniform(0.5, 1.5, 4) for c in "ABC"}

    def amostras(c, n_grupos, rep=3):
        z = np.repeat(media[c] + rng.normal(0, 0.15, (n_grupos, 4)), rep, 0)
        z = z + rng.normal(0, 0.02, (n_grupos * rep, 4))
        X = z @ base[c] + rng.normal(0, 0.005, (n_grupos * rep, _P))
        return X, np.repeat([f"{c}{i}" for i in range(n_grupos)], rep)
    return amostras


def _treino(amostras, n_grupos):
    Xs, ys, gs = [], [], []
    for i, c in enumerate("ABC"):
        X, g = amostras(c, n_grupos)
        Xs.append(X)
        ys += [i] * len(X)
        gs += list(g)
    return np.vstack(Xs), np.eye(3)[ys], np.array(gs)


def _taxa_de_rejeicao_de_q(n_grupos, k, convencao, repeticoes=30):
    """Fracao de amostras NOVAS, da mesma distribuicao, acima do limite de Q de
    95 %. `antiga`: Q do proprio ajuste, sem centrar (codigo anterior).
    `nova`: Q fora da amostra, X centrado."""
    amostras = _mundo(7)
    taxas = []
    for _ in range(repeticoes):
        X, Y, g = _treino(amostras, n_grupos)
        pls = PLSRegression(k, scale=False).fit(X, Y)
        T, P = pls.x_scores_, pls.x_loadings_.T
        Xn = np.vstack([amostras(c, 20)[0] for c in "ABC"])
        Tn = pls.transform(Xn)
        if convencao == "antiga":
            q_tr = q_residuals(X, T, P)
            q_novo = np.sum((Xn - Tn @ P) ** 2, axis=1)
        else:
            q_tr = q_residuals_pls_loo(X, Y, k, groups=g)
            q_novo = np.sum((Xn - pls._x_mean - Tn @ P) ** 2, axis=1)
        taxas.append(float(np.mean(q_novo > q_residuals_limit(q_tr))))
    return float(np.mean(taxas))


def test_rejeicao_em_controle_de_q_com_10_grupos_e_3_lvs_fica_ate_010():
    nova = _taxa_de_rejeicao_de_q(10, 3, "nova")
    antiga = _taxa_de_rejeicao_de_q(10, 3, "antiga")
    assert nova <= 0.10, nova
    assert nova < antiga, (nova, antiga)


def test_com_5_grupos_a_convencao_antiga_estourava_e_a_nova_nao():
    """Medido: 0,176 (antiga) contra 0,081 (nova), alvo nominal 0,05."""
    antiga = _taxa_de_rejeicao_de_q(5, 3, "antiga")
    nova = _taxa_de_rejeicao_de_q(5, 3, "nova")
    assert antiga > 0.10, antiga
    assert nova <= 0.10, nova


def test_distancia_combinada_pls_em_controle_com_10_grupos_fica_ate_010():
    """A coluna `aceito` (T2+Q combinados): rejeicao em controle <= 0,10."""
    amostras = _mundo(11)
    taxas = []
    for _ in range(30):
        X, Y, g = _treino(amostras, 10)
        pls = PLSRegression(3, scale=False).fit(X, Y)
        T, P = pls.x_scores_, pls.x_loadings_.T
        T2 = hotelling_t2(T)
        Q = q_residuals_pls_loo(X, Y, 3, groups=g)
        h0, Nh = mean_and_dof_moments(T2)
        q0, Nq = mean_and_dof_moments(Q)
        Xn = np.vstack([amostras(c, 20)[0] for c in "ABC"])
        Tn = pls.transform(Xn)
        T2n = np.sum(Tn ** 2 / T.var(0, ddof=1), axis=1)
        Qn = np.sum((Xn - pls._x_mean - Tn @ P) ** 2, axis=1)
        f = combined_distance(T2n, Qn, h0, q0, Nh, Nq)
        taxas.append(float(np.mean(f > chi2.ppf(0.95, Nh + Nq))))
    assert np.mean(taxas) <= 0.10, np.mean(taxas)


def test_q_de_treino_fora_da_amostra_nao_e_otimista():
    amostras = _mundo(3)
    X, Y, g = _treino(amostras, 6)
    pls = PLSRegression(3, scale=False).fit(X, Y)
    q_in = q_residuals(X - pls._x_mean, pls.x_scores_, pls.x_loadings_.T)
    q_out = q_residuals_pls_loo(X, Y, 3, groups=g)
    assert q_out.shape == q_in.shape and np.all(q_out >= 0)
    assert q_out.mean() > q_in.mean()


def test_q_pls_loo_deixa_o_grupo_inteiro_de_fora_e_aceita_sem_grupos():
    amostras = _mundo(5)
    X, Y, g = _treino(amostras, 5)
    por_grupo = q_residuals_pls_loo(X, Y, 3, groups=g)
    por_amostra = q_residuals_pls_loo(X, Y, 3)
    # sem grupos cada replica ajuda a reconstruir as irmas: Q menor (otimista)
    assert por_amostra.mean() < por_grupo.mean()
    # blocos: com poucos ajustes permitidos o resultado continua finito/estavel
    blocos = q_residuals_pls_loo(X, Y, 3, groups=g, max_ajustes=4)
    assert np.all(np.isfinite(blocos))
    assert np.array_equal(blocos, q_residuals_pls_loo(X, Y, 3, groups=g, max_ajustes=4))


def test_pacote_real_marca_q_centrado_e_predicao_centra_igual(modelo_e_dados):
    pkg, X_novos, wn = modelo_e_dados
    assert pkg.get("pls_q_centrado") is True
    df = pr.predict_samples(pkg, X_novos, wn)
    pls = pkg["pls_final"]
    Xp = pkg["preprocessador"].transform(pr._interpolate_to_reference(pkg, X_novos, wn))
    T = np.asarray(pls.transform(Xp), dtype=float)
    P = np.asarray(pls.x_loadings_, dtype=float).T
    esperado = np.sum((Xp - np.asarray(pls._x_mean) - T @ P) ** 2, axis=1)
    np.testing.assert_allclose(df["Q"].to_numpy(), np.round(esperado, 6), rtol=1e-9, atol=1e-6)


def test_pacote_antigo_sem_marca_e_recusado_com_pedido_de_retreino(modelo_e_dados):
    """Pacotes salvos antes da correcao do limite de Q (sem a marca) nao sao carregados em silencio: a predicao e' recusada
    com uma mensagem que pede o retreino, e nada e' predito."""
    pkg, X_novos, wn = modelo_e_dados
    antigo = {k: v for k, v in pkg.items() if k != "pls_q_centrado"}
    with pytest.raises(ValueError, match="Retreine o modelo"):
        pr.predict_samples(antigo, X_novos, wn)
    with pytest.raises(ValueError, match="Retreine o modelo"):
        pr.predict_blind(antigo, X_novos, wn)
