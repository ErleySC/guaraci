"""FASE A / A1. O Q do treino plotado na figura de aceitacao esta na MESMA
escala do f_crit que desenha a fronteira?

`DDSimca.fit()` calibra q0/Nq/f_crit a partir de Q_train LEAVE-ONE-OUT
(_q_residuals_loo). `score_matrix()` -> `_t2_q()` recalcula Q IN-SAMPLE
(a propria amostra ajudou a ajustar a PCA que depois a reconstroi).
A figura (fig_sprint3_ddsimca_acceptance) plota pontos vindos de
score_matrix contra a fronteira derivada de q0/f_crit.

Se Q_in-sample << Q_loo, os pontos de TREINO aparecem sistematicamente
mais perto da origem do que a fronteira pressupoe -> a figura mostra o
treino confortavelmente dentro da regiao de aceitacao por construcao.

MEDE:
  (1) razao mediana Q_score_matrix / Q_loo nas amostras de treino;
  (2) fracao de amostras de treino que MUDAM DE LADO da fronteira
      (f<=f_crit com um Q, f>f_crit com o outro).

Regime testado = regime tipico de FT-NIR: p=8192 canais,
nc = 3,4,6,10 amostras puras por especie.
"""
import sys

import numpy as np

sys.path.insert(0, "src")
from guaraci.classificadores import DDSimca  # noqa: E402


def gera_espectros(nc: int, p: int, seed: int) -> np.ndarray:
    """Espectros suaves correlacionados + ruido — estrutura de banda tipica
    de FT-NIR, nao ruido branco (ruido branco nao tem o regime n<<p que
    importa aqui)."""
    rng = np.random.default_rng(seed)
    wn = np.linspace(0, 1, p)

    def banda(c, w):
        return np.exp(-((wn - c) ** 2) / (2 * w ** 2))

    base = 1.0 * banda(0.25, 0.05) + 0.7 * banda(0.55, 0.08) + 0.4 * banda(0.8, 0.04)
    # variacao entre replicas: escala multiplicativa + deslocamento de linha
    # de base + ruido — os 3 efeitos reais de replica fisica em NIR
    esc = rng.normal(1.0, 0.03, nc)[:, None]
    base_off = rng.normal(0.0, 0.01, nc)[:, None]
    X = esc * base[None, :] + base_off + rng.normal(0, 0.002, (nc, p))
    return X


def mede(nc: int, p: int, n_components: int, seeds: range):
    razoes, fracoes_virada, f_in_med, f_loo_med = [], [], [], []
    for s in seeds:
        X = gera_espectros(nc, p, s)
        y = np.array(["A"] * nc)
        dd = DDSimca(n_components=n_components, alpha=0.05,
                     ucl_method="empirical").fit(X, y)
        if "A" not in dd._modelos:
            continue
        m = dd._modelos["A"]
        q_loo = np.asarray(m["Q_train"], dtype=float)

        sm = dd.score_matrix(X)["A"]
        q_in = np.asarray(sm["Q"], dtype=float)
        t2 = np.asarray(sm["T2"], dtype=float)

        pos = q_loo > 0
        if pos.any():
            razoes.append(float(np.median(q_in[pos] / q_loo[pos])))

        f_in = DDSimca._f_distance(t2, q_in, m)
        f_loo = DDSimca._f_distance(t2, q_loo, m)
        fc = float(m["f_crit"])
        virou = (f_in <= fc) != (f_loo <= fc)
        fracoes_virada.append(float(np.mean(virou)))
        f_in_med.append(float(np.median(f_in)))
        f_loo_med.append(float(np.median(f_loo)))

    return (np.array(razoes), np.array(fracoes_virada),
            np.array(f_in_med), np.array(f_loo_med))


if __name__ == "__main__":
    P = 8192
    SEEDS = range(40)
    print(f"p={P}, {len(SEEDS)} seeds por celula, alpha=0.05\n")
    print(f"{'nc':>3} {'n_comp_cfg':>10} {'Q_in/Q_loo (mediana)':>22} "
          f"{'f_in med':>10} {'f_loo med':>10} {'f_crit-cross':>13}")
    print("-" * 76)
    for nc in (3, 4, 6, 10):
        for ncp in (2, 3):
            r, v, fi, fl = mede(nc, P, ncp, SEEDS)
            if r.size == 0:
                print(f"{nc:>3} {ncp:>10}  (modelo degenerado — pulado)")
                continue
            print(f"{nc:>3} {ncp:>10} {np.median(r):>22.4g} "
                  f"{np.median(fi):>10.3g} {np.median(fl):>10.3g} "
                  f"{100*np.mean(v):>12.1f}%")
