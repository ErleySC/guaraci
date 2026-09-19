# -*- coding: utf-8 -*-
"""Camada 1 da revalidacao do MSPC com R2 (Passo 220): falso alarme e poder
do `sentinela_deriva.check_drift` em cenario SINTETICO onde o Dominio de
Aplicabilidade e' mal calibrado como nos espectros reais.

Diferenca para `tests/test_mspc_validacao_deriva.py`: la' UMA calibracao
gaussiana bem comportada e' reutilizada em todas as repeticoes -- o que
nao expoe o problema. Aqui CADA repeticao sorteia uma calibracao NOVA
(n_cal=40, escores t-Student df=4 -> caudas pesadas, 2 PCs sobre 3 fatores
latentes), que e' exatamente onde a variancia de calibracao finita (Fase I,
Jensen et al. 2006) faz o teste contra o nominal falhar.

Compara o teste LEGADO (binomial vs 5%, `SentinelState` sem referencia)
com R2 (`SentinelState` com a referencia de CV de
`chemometric_stats.ad_rejection_rate_cv`), sobre a MESMA sequencia de
amostras, nos looks n=24/32/40. Usa o `check_drift` de producao.

Uso: python scripts/medicoes/medir_mspc_r2_sintetico.py [N_REPS=200]
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

P = 60
LOOKS = (24, 32, 40)
N_CAL = 40
DERIVAS = (0.10, 0.20, 0.40)          # deslocamento linear ao longo do eixo


def _cenario():
    r0 = np.random.default_rng(123)
    L = r0.normal(size=(3, P))
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    return L, np.array([3.0, 1.0, 0.4])


L, ESC = _cenario()


def gera(rng, n, shift=0.0):
    T = rng.standard_t(4, size=(n, 3)) * ESC
    return T @ L + rng.normal(scale=0.02, size=(n, P)) \
        + shift * np.linspace(0.0, 1.0, P)


SIGS_LEGADO = (0.05, 0.005, 0.001)   # niveis do teste legado (pareamento de tamanho)


def _alarme(dentro, ref, sig=0.05):
    """Alarme em algum look, para o teste legado (ref=None) ou R2."""
    for n in LOOKS:
        e = SentinelState(alpha_nominal=0.05)
        if ref is not None:
            e.ref_rejeitadas, e.ref_n = ref
        e.historico = [bool(v) for v in dentro[:n]]
        if check_drift(e, significancia=sig).alerta:
            return True
    return False


def medir(n_reps: int = 200):
    fa_leg = fa_r2 = 0
    pw_leg = {d: 0 for d in DERIVAS}
    pw_r2 = {d: 0 for d in DERIVAS}
    # legado com significancia MENOR (pareamento de tamanho com R2)
    fa_lp = {g: 0 for g in SIGS_LEGADO}
    pw_lp = {g: {d: 0 for d in DERIVAS} for g in SIGS_LEGADO}
    taxa_ctrl, taxa_cv = [], []
    for s in range(n_reps):
        rng = np.random.default_rng(s)
        Xc = gera(rng, N_CAL)
        pca = PCA(2).fit(Xc)
        a = training_applicability_domain(pca, Xc)
        ref = ad_rejection_rate_cv(Xc, 2)

        def julga(X):
            return applicability_domain_new_samples(
                pca, X, a["var_t"], a["h0"], a["q0"], a["Nh"], a["Nq"],
                a["f_crit"])["dentro_dominio"]

        ctrl = julga(gera(rng, max(LOOKS)))
        taxa_ctrl.append(1.0 - float(np.mean(ctrl)))
        taxa_cv.append(ref[0] / ref[1])
        fa_leg += _alarme(ctrl, None)
        fa_r2 += _alarme(ctrl, ref)
        for g in SIGS_LEGADO:
            fa_lp[g] += _alarme(ctrl, None, g)
        for d in DERIVAS:
            dr = julga(gera(rng, max(LOOKS), shift=d))
            pw_leg[d] += _alarme(dr, None)
            pw_r2[d] += _alarme(dr, ref)
            for g in SIGS_LEGADO:
                pw_lp[g][d] += _alarme(dr, None, g)
    print(f"N reps={n_reps}  n_cal={N_CAL}  looks={LOOKS}")
    print(f"rejeicao por amostra em controle: media={np.mean(taxa_ctrl):.4f} "
          f"| taxa em CV da calibracao: media={np.mean(taxa_cv):.4f}")
    print(f"falso alarme (algum look, controle): legado={fa_leg / n_reps:.3f}  "
          f"R2={fa_r2 / n_reps:.3f}")
    for d in DERIVAS:
        print(f"poder deriva delta={d:.2f}: legado={pw_leg[d] / n_reps:.3f}  "
              f"R2={pw_r2[d] / n_reps:.3f}")
    print("legado com significancia menor (pareamento de tamanho com R2):")
    for g in SIGS_LEGADO:
        pw = "  ".join(f"d={d:.2f}:{pw_lp[g][d] / n_reps:.3f}" for d in DERIVAS)
        print(f"  sig={g:<6} falso_alarme={fa_lp[g] / n_reps:.3f}  poder {pw}")


if __name__ == "__main__":
    medir(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
