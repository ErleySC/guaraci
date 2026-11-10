"""m-06 do parecer de 2026-09-25: a sensibilidade LOGO do DD-SIMCA reajusta o
pre-processamento (MSC, autoscaling) so' no treino de cada dobra quando recebe os
espectros brutos -- ajusta-lo com o grupo retido deixava o numero otimista."""
import numpy as np

import guaraci.pipeline as pq
from guaraci.classificadores import DDSimca, ddsimca_logo_sensitivity


def _dados(seed=1, G=8, p=80):
    rng = np.random.default_rng(seed)
    wn = np.linspace(4000, 9000, p)
    base = np.exp(-((wn - 6000) / 600) ** 2)
    L = rng.standard_normal((3, p)) * 0.05
    X, g = [], []
    for i in range(G):
        mu = rng.standard_normal(3)
        for _ in range(3):
            esc = 1 + 0.15 * rng.standard_normal()
            X.append(esc * (base + (mu + 0.3 * rng.standard_normal(3)) @ L)
                     + 0.005 * rng.standard_normal(p))
            g.append(f"g{i}")
    return np.array(X), np.array(g)


def test_logo_com_fabrica_reajusta_o_preprocessamento_em_cada_dobra():
    X, g = _dados()
    cfg = pq.Config(default_preprocessing="msc_sg_mc", sg_window=11)
    fab = lambda: pq.build_preprocessor(cfg)  # noqa: E731
    Xp = fab().fit(X).transform(X)
    r = ddsimca_logo_sensitivity(Xp, g, n_components=3, X_brutos=X, fabrica_preproc=fab)
    esperado = []
    for h in np.unique(g):
        tr, te = g != h, g == h
        pp = fab().fit(X[tr])
        m = DDSimca(n_components=3).fit(pp.transform(X[tr]), np.array(["_c"] * tr.sum()),
                                        mae_id=g[tr])
        s = m.score_matrix(pp.transform(X[te]))["_c"]
        esperado.extend(np.asarray(s["f"]) <= s["f_crit"])
    assert r["sensibilidade"] == float(np.mean(esperado))
    # sem os brutos, comportamento anterior (preprocessamento global)
    r_global = ddsimca_logo_sensitivity(Xp, g, n_components=3)
    assert np.isfinite(r_global["sensibilidade"])
