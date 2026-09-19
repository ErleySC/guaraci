# -*- coding: utf-8 -*-
"""Camada 1b da revalidacao do MSPC com R2 (Passo 220): poder sob deriva
SUTIL, com o codigo de PRODUCAO, no cenario sintetico de
`tests/test_mspc_validacao_deriva.py` (1 banda gaussiana, ruido 0,02,
n_cal=60, 5 PCs), MAS com uma calibracao NOVA por repeticao -- o que o
teste de 1 calibracao fixa nao faz.

Para cada delta (deslocamento linear da deriva; 0 = controle) mede a fracao
de repeticoes com alarme em algum look (n=20/40/60, sessoes de 20), para o
teste LEGADO (binomial vs 5%) e para R2 (`ad_rejection_rate_cv` + Fisher),
sobre a MESMA sequencia. delta=0 e' o falso alarme; delta>0 e' o poder.
Reproduz o experimento exploratorio do Passo 219 (que usou 10-fold manual e
150 repeticoes) agora com o codigo entregue.

Uso: python scripts/medicoes/medir_mspc_r2_deriva_sutil.py [N_REPS=200]
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.decomposition import PCA

from guaraci.chemometric_stats import (
    ad_rejection_rate_cv,
    applicability_domain_new_samples,
    training_applicability_domain,
)
from guaraci.sentinela_deriva import SentinelState, check_drift

P, NCAL, K = 50, 60, 5
DELTAS = (0.0, 0.010, 0.015, 0.020, 0.030, 0.050)
LOOKS = (20, 40, 60)
_X = np.linspace(0.0, 1.0, P)
BASE = np.exp(-((_X - 0.5) ** 2) / (2 * 0.05 ** 2))


def _alarme(dentro, ref):
    for n in LOOKS:
        e = SentinelState(alpha_nominal=0.05)
        if ref is not None:
            e.ref_rejeitadas, e.ref_n = ref
        e.historico = [bool(v) for v in dentro[:n]]
        if check_drift(e, n_minimo=19).alerta:
            return True
    return False


def main(n_reps=200):
    leg = {d: 0 for d in DELTAS}
    r2 = {d: 0 for d in DELTAS}
    for s in range(n_reps):
        rng = np.random.default_rng(s)
        Xc = BASE[None, :] + rng.normal(scale=0.02, size=(NCAL, P))
        pca = PCA(n_components=K).fit(Xc)
        a = training_applicability_domain(pca, Xc, alpha=0.05)
        ref = ad_rejection_rate_cv(Xc, K)
        for d in DELTAS:
            rr = np.random.default_rng(10_000 + s)
            Z = np.vstack([BASE[None, :] + d * _X[None, :]
                           + rr.normal(scale=0.02, size=(20, P))
                           for _ in range(3)])
            dentro = applicability_domain_new_samples(
                pca, Z, a["var_t"], a["h0"], a["q0"], a["Nh"], a["Nq"],
                a["f_crit"])["dentro_dominio"]
            leg[d] += _alarme(dentro, None)
            r2[d] += _alarme(dentro, ref)
    print(f"N reps={n_reps}  n_cal={NCAL}  k={K}  looks={LOOKS}  "
          "(fracao com alarme em algum look)")
    print("delta   legado   R2      (delta=0 e' o falso alarme)")
    for d in DELTAS:
        print(f"{d:5.3f}  {leg[d] / n_reps:6.3f}  {r2[d] / n_reps:6.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
