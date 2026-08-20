# -*- coding: utf-8 -*-
"""O dominio de aplicabilidade (AD) rejeita amostras da PROPRIA distribuicao
de treino? Mede o vies in-sample de q0/Nq -- com as funcoes de PRODUCAO.

CONTEXTO: `CLAUDE.md` P1 (2026-07-19) ja diagnosticou e corrigiu esta mesma
classe de defeito no DD-SIMCA (`DDSimca._q_residuals_loo`): com n << p, a
PCA reconstroi o proprio treino quase exatamente, entao o Q in-sample e'
otimista (baixo demais), o limite derivado dele fica apertado demais, e o
modelo rejeita amostras legitimas. A correcao la' foi calcular Q por
leave-one-out.

`dominio_aplicabilidade_treino` (chemometric_stats.py:587) NAO recebeu a
mesma correcao: `q_train` continua in-sample. E' o caminho que roda em
producao em `predicao.py` (colunas AD_*), isto e', a decisao
"esta amostra nova esta dentro do dominio de calibracao?".

Uso: python docs/auditoria/medir_ad_vies_insample.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from guaraci.chemometric_stats import (  # noqa: E402
    dominio_aplicabilidade_amostras_novas, dominio_aplicabilidade_treino,
    media_e_dof_momentos)

ALPHA = 0.05
N_REP = 100


def _q_loo(X, k):
    """Q de treino por leave-one-out -- mesma ideia de
    DDSimca._q_residuals_loo, aplicada ao AD."""
    n = X.shape[0]
    q = np.empty(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        p = PCA(n_components=min(k, mask.sum() - 1)).fit(X[mask])
        t = p.transform(X[i:i + 1])
        q[i] = float(np.sum((X[i:i + 1] - p.inverse_transform(t)) ** 2))
    return q


def uma_rodada(n, p, k, rng, corrigido):
    X = rng.normal(size=(n, p))
    X_novo = rng.normal(size=(300, p))       # MESMA distribuicao do treino
    pca = PCA(n_components=k).fit(X)

    if not corrigido:
        art = dominio_aplicabilidade_treino(pca, X, alpha=ALPHA)
    else:
        T = np.asarray(pca.transform(X), dtype=float)
        var_t = T.var(axis=0, ddof=1)
        var_t[var_t == 0] = 1.0
        T2 = np.sum((T ** 2) / var_t, axis=1)
        h0, Nh = media_e_dof_momentos(T2)
        q0, Nq = media_e_dof_momentos(_q_loo(X, k))
        art = {"var_t": var_t, "h0": h0, "q0": q0, "Nh": Nh, "Nq": Nq,
               "f_crit": float(chi2.ppf(1 - ALPHA, Nh + Nq))}

    r = dominio_aplicabilidade_amostras_novas(
        pca, X_novo, art["var_t"], art["h0"], art["q0"],
        art["Nh"], art["Nq"], art["f_crit"])
    return float(np.mean(r["dentro_dominio"]))


def main():
    print("Fracao de amostras da PROPRIA distribuicao de treino aceitas "
          "pelo AD.")
    print("alpha=%.2f -> o correto e' ~%.2f. Abaixo disso, o software diz "
          "'fora do dominio' para amostra legitima.\n" % (ALPHA, 1 - ALPHA))
    print("%5s %6s %5s | %-12s | %-12s"
          % ("n", "p", "k", "atual", "com Q por LOO"))
    for (n, p, k) in [(20, 1200, 3), (30, 1200, 3), (50, 1200, 5),
                      (100, 1200, 5), (300, 1200, 10), (50, 100, 5)]:
        a = np.mean([uma_rodada(n, p, k, np.random.default_rng(1000 + i),
                                False) for i in range(N_REP)])
        b = np.mean([uma_rodada(n, p, k, np.random.default_rng(1000 + i),
                                True) for i in range(max(20, N_REP // 5))])
        print("%5d %6d %5d | %.4f       | %.4f" % (n, p, k, a, b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
