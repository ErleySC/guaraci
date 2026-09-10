"""A1b. O SR baseado em w1 muda o RANKING de variaveis (isto e', a selecao),
ou so' a magnitude? Testa em cenario espectral mais dificil: varias bandas
alvo + varios interferentes sobrepostos, que e' quando LVs>1 importam."""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import sys
sys.path.insert(0, "src")
from guaraci.chemometric_stats import compute_selectivity_ratio


def sr_referencia(modelo, X):
    b = np.asarray(modelo.coef_, dtype=float).reshape(-1)
    w_tp = b / np.linalg.norm(b)
    t = X @ w_tp
    p_tp = (t @ X) / float(t @ t)
    Xtp = np.outer(t, p_tp)
    vr = (X - Xtp).var(axis=0, ddof=1); vr[vr < 1e-12] = 1e-12
    return Xtp.var(axis=0, ddof=1) / vr


def gera(n, p, seed):
    rng = np.random.default_rng(seed)
    wn = np.linspace(0, 1, p)
    def banda(c, w): return np.exp(-((wn - c) ** 2) / (2 * w ** 2))
    # analito: 2 bandas; 3 interferentes correlacionados entre si
    a = rng.uniform(.1, 1, n)
    i1, i2, i3 = (rng.uniform(0, 1, n) for _ in range(3))
    i2 = .7 * i1 + .3 * i2          # interferentes correlacionados
    X = (a[:, None] * (banda(.20, .025) + .6 * banda(.55, .03))[None, :]
         + i1[:, None] * banda(.35, .04)[None, :]
         + i2[:, None] * banda(.62, .05)[None, :]
         + i3[:, None] * banda(.80, .06)[None, :]
         + rng.uniform(0, .4, n)[:, None] * wn[None, :]
         + rng.normal(0, .01, (n, p)))
    return X - X.mean(0), a - a.mean()


print("=" * 74)
print("A1b. SR: impacto no RANKING (Jaccard top-k) em cenario multi-interferente")
print("=" * 74)
print(f"{'LV':>3} {'Jac@20':>8} {'Jac@50':>8} {'rho Spearman':>14} {'maxSR ref':>11} {'maxSR impl':>11}")
from scipy.stats import spearmanr
for n_lv in (1, 2, 3, 4, 6, 8):
    jac20, jac50, rhos, mr, mi = [], [], [], [], []
    for seed in range(15):
        X, y = gera(80, 300, seed)
        m = PLSRegression(n_components=n_lv, scale=False).fit(X, y)
        si, sr = compute_selectivity_ratio(m, X), sr_referencia(m, X)
        for k, acc in ((20, jac20), (50, jac50)):
            A, B = set(np.argsort(si)[-k:]), set(np.argsort(sr)[-k:])
            acc.append(len(A & B) / len(A | B))
        rhos.append(spearmanr(si, sr).statistic)
        mr.append(sr.max()); mi.append(si.max())
    print(f"{n_lv:>3} {np.mean(jac20):>8.3f} {np.mean(jac50):>8.3f} "
          f"{np.mean(rhos):>14.4f} {np.mean(mr):>11.1f} {np.mean(mi):>11.1f}")
