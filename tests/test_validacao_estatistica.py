"""Rede de segurança da validação estatística (guaraci.validacao_estatistica).

BCa (IC de confiança) e CV-ANOVA (significância do modelo) são o que torna os
resultados "publication-grade" — uma regressão silenciosa aqui corromperia
intervalos de confiança e p-valores reportados em monografia/artigo. Testes
das duas funções numéricas PURAS (as demais — teste_wold/permutação — exigem
pipeline+CV e são cobertas pelos testes end-to-end 'slow').
"""
import numpy as np
from sklearn.metrics import accuracy_score

from guaraci.validacao_estatistica import bootstrap_bca_ci, cv_anova_eriksson


# ── bootstrap_bca_ci ─────────────────────────────────────────────────────────
def test_bca_predicao_perfeita_ic_em_um():
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    low, high, obs = bootstrap_bca_ci(y, y.copy(), accuracy_score, n_boot=200, seed=1)
    assert obs == 1.0
    assert low == 1.0 and high == 1.0


def test_bca_observed_bate_com_metrica():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=40)
    yp = y.copy()
    yp[:10] = (yp[:10] + 1) % 3  # 25% de erro
    _, _, obs = bootstrap_bca_ci(y, yp, accuracy_score, n_boot=200, seed=1)
    assert obs == accuracy_score(y, yp)


def test_bca_intervalo_contem_observado_e_dentro_de_0_1():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=60)
    yp = y.copy()
    yp[::5] = 1 - yp[::5]  # ~20% erro
    low, high, obs = bootstrap_bca_ci(y, yp, accuracy_score, n_boot=300, seed=7)
    assert 0.0 <= low <= obs <= high <= 1.0


def test_bca_reprodutivel_com_mesma_seed():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, size=50)
    yp = y.copy(); yp[:8] = (yp[:8] + 1) % 3
    r1 = bootstrap_bca_ci(y, yp, accuracy_score, n_boot=200, seed=99)
    r2 = bootstrap_bca_ci(y, yp, accuracy_score, n_boot=200, seed=99)
    assert r1 == r2


def test_bca_n_boot_baixo_retorna_nan():
    y = np.array([0, 0, 1, 1, 0, 1])
    low, high, obs = bootstrap_bca_ci(y, y.copy(), accuracy_score, n_boot=5, seed=1)
    assert np.isnan(low) and np.isnan(high)
    assert obs == 1.0


# ── cv_anova_eriksson ────────────────────────────────────────────────────────
def test_cv_anova_predicao_perfeita_q2_alto_p_baixo():
    Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    r = cv_anova_eriksson(Y, Y.copy(), n_components=2)
    assert r["Q2"] > 0.999
    assert r["p_value"] < 0.05


def test_cv_anova_predicao_ruim_q2_baixo_p_alto():
    Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    Y_cv = np.full_like(Y, Y.mean())  # prever a média = PRESS ~ SS_total
    r = cv_anova_eriksson(Y, Y_cv, n_components=2)
    assert r["Q2"] <= 0.0
    assert r["p_value"] == 1.0
    assert r["F"] == 0.0


def test_cv_anova_y_constante_retorna_nan():
    Y = np.full(10, 3.0)          # variância zero
    r = cv_anova_eriksson(Y, Y.copy(), n_components=1)
    assert np.isnan(r["F"])
    assert r["p_value"] == 1.0
    assert r["Q2"] == 0.0


def test_cv_anova_aceita_y_2d_onehot():
    # Y one-hot (m=K classes); a função reduz para univariado (m_eff=1)
    Y = np.eye(3)[np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])]
    Y_cv = Y * 0.9 + 0.033  # quase perfeito
    r = cv_anova_eriksson(Y, Y_cv, n_components=2)
    assert 0.0 < r["Q2"] <= 1.0
    assert 0.0 <= r["p_value"] <= 1.0


def test_cv_anova_q2_formula():
    # Q2 = 1 - PRESS/SS_total; monta um caso com valor conhecido
    Y = np.array([0.0, 2.0, 4.0, 6.0])           # média 3, SS_total = 9+1+1+9 = 20
    Y_cv = np.array([1.0, 2.0, 4.0, 5.0])        # PRESS = 1+0+0+1 = 2
    r = cv_anova_eriksson(Y, Y_cv, n_components=1)
    assert abs(r["Q2"] - (1.0 - 2.0 / 20.0)) < 1e-9
