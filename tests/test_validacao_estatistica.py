"""Rede de segurança da validação estatística (guaraci.validacao_estatistica).

BCa (IC de confiança) e CV-ANOVA (significância do modelo) são o que torna os
resultados "publication-grade" — uma regressão silenciosa aqui corromperia
intervalos de confiança e p-valores reportados em monografia/artigo. Testes
das duas funções numéricas PURAS (as demais — teste_wold/permutação — exigem
pipeline+CV e são cobertas pelos testes end-to-end 'slow').
"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from guaraci.validacao_estatistica import bootstrap_bca_ci, cv_anova_eriksson
# Alias com prefixo _ para o pytest NAO coletar a funcao importada como teste
# (o nome 'teste_permutacao' casa com o padrao de coleta 'test*').
from guaraci.validacao_estatistica import teste_permutacao as _teste_permutacao


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


# ── teste_permutacao (Y-randomization) ──────────────────────────────────────
def _dados_perm(separavel: bool, seed: int):
    """Dataset binario de 2x20 amostras, 10 variaveis. `separavel=True` cria
    duas nuvens bem afastadas (sinal real); False = ruido puro (rotulos sem
    relacao com X)."""
    rng = np.random.default_rng(seed)
    n = 40
    y_int = np.array([0] * (n // 2) + [1] * (n // 2))
    if separavel:
        X = np.vstack([rng.normal(-3.0, 0.5, size=(n // 2, 10)),
                       rng.normal(+3.0, 0.5, size=(n // 2, 10))])
    else:
        X = rng.normal(0.0, 1.0, size=(n, 10))
    Y_bin = np.zeros((n, 2)); Y_bin[np.arange(n), y_int] = 1.0
    return X, Y_bin, y_int


def _factory_pls():
    return Pipeline([("pls", PLSRegression(n_components=2, scale=False))])


def test_permutacao_da_p_baixo_com_sinal_real():
    """VALIDACAO: com classes bem separadas, a acuracia observada deve superar
    quase todas as permutacoes -> p pequeno. Se o teste desse p alto aqui,
    estaria mascarando sinal real como se fosse acaso."""
    X, Y_bin, y_int = _dados_perm(separavel=True, seed=1)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    res = _teste_permutacao(_factory_pls, X, Y_bin, y_int, cv,
                           n_perm=40, seed=1)
    assert res["p_value"] < 0.05
    assert res["acc_observada"] > 0.9


def test_permutacao_da_p_alto_com_rotulos_aleatorios():
    """VALIDACAO: sob H0 (rotulos sem relacao com X) a acuracia observada e'
    apenas mais uma amostra da distribuicao permutada -> p NAO deve ser
    pequeno. Se desse p baixo aqui, seria um falso positivo (acha estrutura
    onde so ha ruido)."""
    X, Y_bin, y_int = _dados_perm(separavel=False, seed=2)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    res = _teste_permutacao(_factory_pls, X, Y_bin, y_int, cv,
                           n_perm=40, seed=2)
    assert res["p_value"] > 0.10
