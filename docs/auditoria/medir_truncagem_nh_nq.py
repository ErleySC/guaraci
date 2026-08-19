# -*- coding: utf-8 -*-
"""Nh/Nq inteiros (referencia) x Nh/Nq continuos (implementacao atual).

Kucheryavskiy, Rodionova & Pomerantsev (2024), J. Chemometrics 38(7):e3556,
p.4, definem Nh e Nq como GRAUS DE LIBERDADE de uma chi-quadrado, obtidos
pelo metodo dos momentos e ARREDONDADOS PARA INTEIRO. A implementacao do
GUARACI (chemometric_stats.media_e_dof_momentos) devolve o valor continuo.

Este script mede o efeito: cobertura empirica (fracao de amostras da
propria distribuicao de treino aceitas) com alpha nominal 0,05, nos dois
criterios, variando n e p, com repeticoes independentes.

Uso: python docs/auditoria/medir_truncagem_nh_nq.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from guaraci.chemometric_stats import (distancia_combinada,  # noqa: E402
                                       media_e_dof_momentos)
from sklearn.decomposition import PCA  # noqa: E402

ALPHA = 0.05
N_REP = 200


def cobertura(n_treino, p, n_comp, rng, arredondar):
    X = rng.normal(size=(n_treino, p))
    Xt = rng.normal(size=(400, p))          # mesma distribuicao do treino
    pca = PCA(n_components=n_comp).fit(X)
    T = pca.transform(X)
    var_t = T.var(axis=0, ddof=1)
    var_t[var_t == 0] = 1.0
    T2 = np.sum(T ** 2 / var_t, axis=1)
    Q = np.sum((X - pca.inverse_transform(T)) ** 2, axis=1)
    h0, Nh = media_e_dof_momentos(T2)
    q0, Nq = media_e_dof_momentos(Q)
    if arredondar:
        Nh = max(1.0, float(round(Nh)))
        Nq = max(1.0, float(round(Nq)))
    f_crit = float(stats.chi2.ppf(1 - ALPHA, Nh + Nq))

    Tt = pca.transform(Xt)
    T2t = np.sum(Tt ** 2 / var_t, axis=1)
    Qt = np.sum((Xt - pca.inverse_transform(Tt)) ** 2, axis=1)
    f = distancia_combinada(T2t, Qt, h0, q0, Nh, Nq)
    return float(np.mean(f <= f_crit)), Nh, Nq


def main():
    print("cobertura empirica com alpha nominal %.2f "
          "(alvo = %.3f de aceitacao)\n" % (ALPHA, 1 - ALPHA))
    print("%6s %6s %6s | %-22s | %-22s | %8s"
          % ("n", "p", "ncomp", "continuo (atual)", "inteiro (referencia)",
             "delta"))
    for (n, p, k) in [(30, 50, 3), (50, 200, 3), (80, 500, 5),
                      (150, 1000, 5), (300, 1200, 10), (30, 1200, 3)]:
        rng_a = np.random.default_rng(20260817)
        rng_b = np.random.default_rng(20260817)
        ca, na_h, na_q = [], [], []
        cb, nb_h, nb_q = [], [], []
        for _ in range(N_REP):
            c, h, q = cobertura(n, p, k, rng_a, False)
            ca.append(c); na_h.append(h); na_q.append(q)
        for _ in range(N_REP):
            c, h, q = cobertura(n, p, k, rng_b, True)
            cb.append(c); nb_h.append(h); nb_q.append(q)
        ma, mb = float(np.mean(ca)), float(np.mean(cb))
        print("%6d %6d %6d | %.4f (Nh=%.2f Nq=%.2f) | "
              "%.4f (Nh=%.1f Nq=%.1f) | %+.4f"
              % (n, p, k, ma, np.mean(na_h), np.mean(na_q),
                 mb, np.mean(nb_h), np.mean(nb_q), mb - ma))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
