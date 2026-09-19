# -*- coding: utf-8 -*-
"""Decompoe a taxa de falso alarme do MSPC (`sentinela_deriva`) no Corn.

Achado (2026-09-19): a taxa de falso alarme "em controle" reportada como
~7-10% (30 seeds) era subestimada -- com 1500 splits ela e' ~21%. Este script
reproduz a decomposicao que localizou a CAUSA RAIZ e avalia remedios
candidatos, sem alterar nenhum codigo de producao:

  1. taxa de rejeicao POR AMOSTRA do Dominio de Aplicabilidade em controle
     (nominal 5%) e sua dispersao entre calibracoes;
  2. falso alarme POR LOOK e ao longo da sequencia de looks;
  3. nulo teorico Bernoulli(0.05) com a MESMA agenda de looks (quanto do
     excesso e' teste sequencial repetido);
  4. remedios: (R1) alpha-spending sobre os looks, (R3) tolerancia
     p1=10%, (R2) teste de 2 amostras contra a taxa de rejeicao em
     VALIDACAO CRUZADA da propria calibracao.

Uso:
    GUARACI_DATASETS_DIR=<pasta com corn.mat> python scripts/medicoes/
        medir_mspc_falso_alarme_corn.py [N_SPLITS=300] [N_CAL=40]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.io as sio
from scipy.stats import binomtest, fisher_exact
from sklearn.decomposition import PCA

from guaraci.chemometric_stats import (
    applicability_domain_new_samples, training_applicability_domain)

K = 2                      # PCs: 99,86% da variancia do m5 (ver test_mspc_validacao_corn)
LOOKS_BASE = (24, 32, 40)  # n acumulado nos looks (n>=19 e' o minimo do sentinela)


def _carregar():
    m = sio.loadmat(os.path.join(os.environ["GUARACI_DATASETS_DIR"], "corn.mat"))
    return (np.asarray(m["m5spec"]["data"][0, 0], dtype=float),
            np.asarray(m["mp5spec"]["data"][0, 0], dtype=float))


def _ajustar(X, cal):
    pca = PCA(n_components=K).fit(X[cal])
    return pca, training_applicability_domain(pca, X[cal], alpha=0.05)


def _rejeita(pca, art, Z):
    return ~applicability_domain_new_samples(
        pca, Z, art["var_t"], art["h0"], art["q0"], art["Nh"], art["Nq"],
        art["f_crit"])["dentro_dominio"]


def _crit(n, p0, sig):
    ks = [k for k in range(n + 1)
          if binomtest(k, n, p0, alternative="greater").pvalue < sig]
    return ks[0] if ks else n + 1


def main(n_splits: int = 300, n_cal: int = 40) -> None:
    X, Xb = _carregar()
    looks = [n for n in LOOKS_BASE if n <= len(X) - n_cal]
    linhas = []
    for s in range(n_splits):
        idx = np.random.default_rng(s).permutation(len(X))
        cal, ho = idx[:n_cal], idx[n_cal:]
        pca, art = _ajustar(X, cal)
        rA = _rejeita(pca, art, X[ho]).astype(int)
        rB = _rejeita(pca, art, Xb[ho]).astype(int)
        k0 = 0                                     # rejeicao LOO na calibracao
        for i in range(n_cal):
            p2, a2 = _ajustar(X, np.delete(cal, i))
            k0 += int(_rejeita(p2, a2, X[cal[i]][None, :])[0])
        linhas.append((rA, rB, k0))

    taxa = np.array([r[0].mean() for r in linhas])
    print(f"N splits={n_splits}  n_cal={n_cal}  looks={looks}")
    print(f"1) rejeicao por amostra em controle: media={taxa.mean():.4f} "
          f"(nominal 0.05)  sd entre calibracoes={taxa.std():.4f}")

    def alarmes(r, k0, modo, sig):
        out = []
        for n in looks:
            k = int(r[:n].sum())
            if modo == "binom05":
                out.append(k >= _crit(n, 0.05, sig))
            elif modo == "tol10":
                out.append(k >= _crit(n, 0.10, sig))
            else:  # fisher
                out.append(fisher_exact([[k, n - k], [k0, n_cal - k0]],
                                        alternative="greater")[1] < sig)
        return out

    rng = np.random.default_rng(1)
    B = (rng.random((100000, max(looks))) < 0.05).astype(int)
    anyt = np.zeros(len(B), bool)
    for n in looks:
        anyt |= B[:, :n].sum(1) >= _crit(n, 0.05, 0.05)
    print(f"3) NULO Bernoulli(0.05), mesma agenda de looks: falso alarme "
          f"(qualquer look)={anyt.mean():.3f}")

    def resumo(nome, modo, sig):
        A = np.array([alarmes(r[0], r[2], modo, sig) for r in linhas])
        P = np.array([alarmes(r[1], r[2], modo, sig) for r in linhas])
        print(f"   {nome:34s} falso_alarme={A.any(1).mean():.3f} "
              f"(por look {A.mean(0).round(3)}) | poder={P.any(1).mean():.3f}")

    print("2/4) falso alarme em controle x poder (troca m5->mp5):")
    resumo("R0 binomial vs 5% (atual)", "binom05", 0.05)
    resumo("R1 idem, alpha-spending (~0.048)", "binom05", 0.048)
    resumo("R3 binomial vs tolerancia 10%", "tol10", 0.05)
    resumo("R2 Fisher vs CV da calibracao, .05", "fisher", 0.05)
    resumo("R2b Fisher vs CV da calibracao, .01", "fisher", 0.01)
    k0 = np.array([r[2] for r in linhas]) / n_cal
    print(f"   taxa de rejeicao em CV na calibracao: media={k0.mean():.4f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300,
         int(sys.argv[2]) if len(sys.argv) > 2 else 40)
