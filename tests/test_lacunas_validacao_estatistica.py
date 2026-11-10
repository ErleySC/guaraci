"""Ramos de borda de guaraci.validacao_estatistica (BCa, CV-ANOVA, Wold, permutacao): um teste por ramo, dados minusculos."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from guaraci import validacao_estatistica as vs


def _acuracia(yt, yp):
    return float(np.mean(yt == yp))


def _classificacao(n=24, erro_a_cada=4):
    y = np.array([0, 1] * (n // 2))
    p = y.copy()
    p[::erro_a_cada] = 1 - p[::erro_a_cada]
    return y, p


# ── bootstrap_bca_ci ────────────────────────────────────────────────────────
@pytest.mark.parametrize("com_grupos", [False, True])
def test_bca_descarta_reamostra_degenerada_e_aproxima_o_jackknife_pelo_observado(com_grupos):
    y, p = _classificacao()
    grupos = np.repeat(np.arange(12), 2) if com_grupos else None
    chamadas = {"n": 0}

    def metrica_instavel(a, b):
        chamadas["n"] += 1
        if chamadas["n"] % 7 == 3:                           # reamostras e dobras do jackknife que "dao erro"
            raise ValueError("classe ausente")
        return _acuracia(a, b)
    lo, hi, obs = vs.bootstrap_bca_ci(y, p, metrica_instavel, n_boot=60, groups=grupos)
    assert obs == pytest.approx(_acuracia(y, p)) and np.isfinite(lo) and np.isfinite(hi) and lo <= hi


def test_bca_denominador_zero_cai_para_o_percentil_simples(monkeypatch):
    y, p = _classificacao()
    n = len(y)
    jack = np.array([_acuracia(np.delete(y, i), np.delete(p, i)) for i in range(n)])
    diffs = jack.mean() - jack
    a = float(np.sum(diffs ** 3)) / (6.0 * float(np.sum(diffs ** 2)) ** 1.5)
    if a == 0.0:
        pytest.fail("o cenario precisa de aceleracao != 0")
    s = 1.0 / a
    vizinhos = [s]                                            # procura, entre os doubles vizinhos de 1/a, um com 1 - a*s exatamente 0
    for direcao in (np.inf, -np.inf):
        v = s
        for _ in range(20):
            v = float(np.nextafter(v, direcao))
            vizinhos.append(v)
    s = next(v for v in vizinhos if 1.0 - a * v == 0.0)

    class _Normal:
        valores = iter([s, 0.0, 0.0])                         # z0, z_alfa/2, z_1-alfa/2

        def ppf(self, q):
            return next(self.valores)

    monkeypatch.setattr(vs, "_norm_dist", _Normal())
    lo, hi, obs = vs.bootstrap_bca_ci(y, p, _acuracia, n_boot=80)
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi


# ── CV-ANOVA ────────────────────────────────────────────────────────────────
def test_cv_anova_sem_graus_de_liberdade_residuais_devolve_nan_e_p_um():
    Y = np.array([[0.0], [1.0], [2.0]])
    r = vs.cv_anova_eriksson(Y, Y + np.array([[0.1], [-0.1], [0.2]]), n_components=3)
    assert np.isnan(r["F"]) and r["p_value"] == 1.0 and r["df_resid"] <= 0


# ── Wold e permutacao ───────────────────────────────────────────────────────
def _dados(n=24, k=4):
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, k)) + y[:, None]
    return X, np.eye(2)[y], y


def _fabrica():
    return make_pipeline(StandardScaler(), PLSRegression(n_components=1, scale=False))


def test_iteracao_wold_com_y_permutado_constante_e_pulada_e_com_predicao_nao_finita_e_falha():
    X, Y, y = _dados()
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    y_const = np.zeros_like(y)
    assert vs._iter_wold(_fabrica, X, np.eye(2)[y_const], y_const, y, cv, None)[0] == "skip"

    class _Nan:
        def fit(self, X, Y):
            return self

        def predict(self, X):
            return np.full((len(X), 2), np.nan)
    assert vs._iter_wold(_Nan, X, Y, y, y, cv, None)[0] == "fail"


def test_wold_sequencial_ignora_iteracao_pulada_conta_a_falha_e_so_loga_de_tempos_em_tempos(monkeypatch):
    X, Y, y = _dados()
    ciclo = iter(["ok", "skip", "fail"] * 40)

    def iteracao(fabrica, X_, Yp, yp, y_, cv, g):
        s = next(ciclo)
        return (s, 0.5, 0.2, 0.1) if s == "ok" else (s, None, None, None)
    monkeypatch.setattr(vs, "_iter_wold", iteracao)
    r = vs.wold_test(_fabrica, X, Y, y, StratifiedKFold(3, shuffle=True, random_state=0), n_perm=60, seed=1)
    assert r["n_validos"] == 20 and r["n_falhos"] == 20           # 60 iteracoes: 1/3 ok, 1/3 pulada, 1/3 falha


class _ParaleloSequencial:
    """Substitui joblib.Parallel: executa na hora, sem processos (o ramo paralelo e' o que se quer cobrir, nao o loky)."""
    def __init__(self, **_):
        pass

    def __call__(self, tarefas):
        return [f(*a, **k) for f, a, k in tarefas]


def test_wold_em_paralelo_agrega_ok_pulado_e_falha(monkeypatch):
    import joblib
    X, Y, y = _dados()
    ciclo = iter(["ok", "skip", "fail"] * 10)
    monkeypatch.setattr(joblib, "Parallel", _ParaleloSequencial)
    monkeypatch.setattr(vs, "_iter_wold", lambda *a: (lambda s: (s, 0.5, 0.2, 0.1) if s == "ok" else (s, None, None, None))(next(ciclo)))
    r = vs.wold_test(_fabrica, X, Y, y, StratifiedKFold(3, shuffle=True, random_state=0), n_perm=9, seed=1, n_jobs=2)
    assert r["n_validos"] == 3 and r["n_falhos"] == 3


def test_permutacao_em_paralelo_conta_as_falhas(monkeypatch):
    import joblib
    X, Y, y = _dados()
    ciclo = iter(["ok", "fail"] * 10)
    monkeypatch.setattr(joblib, "Parallel", _ParaleloSequencial)
    monkeypatch.setattr(vs, "_iter_permutacao", lambda *a: ("ok", 0.6) if next(ciclo) == "ok" else ("fail", None))
    r = vs.permutation_test(_fabrica, X, Y, y, StratifiedKFold(3, shuffle=True, random_state=0), n_perm=8, seed=1, n_jobs=2)
    assert r["n_validos"] == 4 and r["n_falhos"] == 4 and r["failure_rate"] == pytest.approx(0.5)


def test_permutacao_sequencial_conta_a_falha(monkeypatch):
    X, Y, y = _dados()
    ciclo = iter(["ok", "fail"] * 10)
    monkeypatch.setattr(vs, "_iter_permutacao", lambda *a: ("ok", 0.6) if next(ciclo) == "ok" else ("fail", None))
    r = vs.permutation_test(_fabrica, X, Y, y, StratifiedKFold(3, shuffle=True, random_state=0), n_perm=6, seed=1)
    assert r["n_falhos"] == 3
