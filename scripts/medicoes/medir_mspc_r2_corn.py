# -*- coding: utf-8 -*-
"""Camada 2 da revalidacao do MSPC com R2 (Passo 220): falso alarme e
deteccao no Corn REAL usando o codigo de PRODUCAO
(`chemometric_stats.ad_rejection_rate_cv` + `sentinela_deriva.check_drift`),
nao a reimplementacao de `medir_mspc_falso_alarme_corn.py`.

Mesmo protocolo dos testes (`tests/test_mspc_validacao_corn.py`): calibra o
Dominio de Aplicabilidade em 40 amostras m5, processa as 40 restantes em
lotes de 8 (fase em controle: m5; fase deriva real: as MESMAS amostras
fisicas medidas em mp5), com `check_drift` a cada lote a partir de n>=19
(looks efetivos n=24/32/40). Compara o teste LEGADO (binomial vs 5%, sem
referencia) com R2 na MESMA sequencia.

Uso:
    GUARACI_DATASETS_DIR=<pasta com corn.mat> python \
        scripts/medicoes/medir_mspc_r2_corn.py [N_SPLITS=300] [SEED0=0] [r2|ambos]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA

from guaraci.chemometric_stats import (
    ad_rejection_rate_cv,
    applicability_domain_new_samples,
    training_applicability_domain,
)
from guaraci.sentinela_deriva import SentinelState, check_drift

N_CAL, TAM_LOTE, K = 40, 8, 2


def _cenario(X_m5, X_mp5, seed, com_referencia):
    idx = np.random.default_rng(seed).permutation(80)
    cal, resto = idx[:N_CAL], idx[N_CAL:]
    pca = PCA(n_components=K).fit(X_m5[cal])
    art = training_applicability_domain(pca, X_m5[cal], alpha=0.05)

    def dentro(X, ii):
        return applicability_domain_new_samples(
            pca, X[ii], art["var_t"], art["h0"], art["q0"], art["Nh"],
            art["Nq"], art["f_crit"])["dentro_dominio"]

    e = SentinelState(alpha_nominal=0.05)
    if com_referencia:
        e.ref_rejeitadas, e.ref_n = ad_rejection_rate_cv(X_m5[cal], K)
    falso = False
    for i in range(0, len(resto), TAM_LOTE):
        for b in dentro(X_m5, resto[i:i + TAM_LOTE]):
            e.registrar(bool(b))
        if e.n >= 19 and check_drift(e).alerta:
            falso = True
    lote_det = None
    for j, i in enumerate(range(0, len(resto), TAM_LOTE)):
        for b in dentro(X_mp5, resto[i:i + TAM_LOTE]):
            e.registrar(bool(b))
        if check_drift(e).alerta and lote_det is None:
            lote_det = j
    return falso, lote_det


def _ic(k, n):
    """Wilson e Clopper-Pearson (95%) para k sucessos em n."""
    from scipy.stats import binomtest
    r = binomtest(k, n)
    w = r.proportion_ci(confidence_level=0.95, method="wilson")
    c = r.proportion_ci(confidence_level=0.95, method="exact")
    return (w.low, w.high), (c.low, c.high)


def _linha(nome, res):
    n = len(res)
    k_fa = sum(bool(r[0]) for r in res)
    k_det = sum(r[1] is not None for r in res)
    atrasos = [r[1] for r in res if r[1] is not None]
    dist = {j: atrasos.count(j) for j in sorted(set(atrasos))}
    for rot, k in (("falso alarme", k_fa), ("deteccao   ", k_det)):
        (wl, wh), (cl, ch) = _ic(k, n)
        print(f"{nome:34s} {rot} {k}/{n} = {k / n:.3f}  "
              f"Wilson95 [{wl:.3f}, {wh:.3f}]  Clopper-Pearson95 "
              f"[{cl:.3f}, {ch:.3f}]")
    print(f"{'':34s} atraso(lotes) medio={np.mean(atrasos):.2f} "
          f"max={max(atrasos)} distribuicao={dist}")


def main(n_splits=300, seed0=0, com_legado=True):
    m = sio.loadmat(os.path.join(os.environ["GUARACI_DATASETS_DIR"], "corn.mat"))
    X_m5 = np.asarray(m["m5spec"]["data"][0, 0], dtype=float)
    X_mp5 = np.asarray(m["mp5spec"]["data"][0, 0], dtype=float)
    print(f"N splits={n_splits}  sementes {seed0}..{seed0 + n_splits - 1}  "
          f"n_cal={N_CAL}  lote={TAM_LOTE}")
    braços = [("R2 (Fisher vs CV da calibracao)", True)]
    if com_legado:
        braços.insert(0, ("legado (binomial vs 5%)", False))
    for nome, ref in braços:
        res = [_cenario(X_m5, X_mp5, s, ref)
               for s in range(seed0, seed0 + n_splits)]
        _linha(nome, res)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300,
         int(sys.argv[2]) if len(sys.argv) > 2 else 0,
         (sys.argv[3] != "r2") if len(sys.argv) > 3 else True)
