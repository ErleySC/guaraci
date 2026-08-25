"""A4. O teste de permutacao respeita os grupos mae_id?

Cenario: H0 VERDADEIRO (rotulo sorteado por GRUPO, sem relacao com X).
Um teste calibrado deve rejeitar H0 em ~5% das vezes com alpha=0.05.

Compara:
  Null A = permutacao por AMOSTRA  (o que validacao_estatistica.py faz hoje)
  Null B = permutacao por GRUPO    (unidade de troca correta em dado agrupado)
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score

import sys
sys.path.insert(0, "src")
from guaraci.validacao_estatistica import StableStratifiedGroupKFold

G, R, K, P, NLV = 12, 3, 3, 40, 2
N = G * R
N_PERM, N_REP = 100, 120


def cv_bal_acc(X, y_int, groups, cv):
    """Balanced accuracy por CV group-aware (PLS-DA, argmax)."""
    Yb = np.zeros((len(y_int), K)); Yb[np.arange(len(y_int)), y_int] = 1
    yhat = np.zeros((len(y_int), K))
    for tr, va in cv.split(X, y_int, groups=groups):
        if len(np.unique(y_int[tr])) < 2:
            return np.nan
        m = PLSRegression(n_components=NLV, scale=False).fit(X[tr], Yb[tr])
        yhat[va] = m.predict(X[va])
    return balanced_accuracy_score(y_int, np.argmax(yhat, axis=1))


def uma_replica(seed):
    rng = np.random.default_rng(seed)
    gid = np.repeat(np.arange(G), R)
    # X com forte estrutura de grupo (replicas quase identicas), como FT-NIR real
    lat_grupo = rng.normal(0, 1, (G, P))
    X = lat_grupo[gid] + rng.normal(0, 0.15, (N, P))
    # H0: rotulo sorteado POR GRUPO, independente de X
    lab_grupo = np.array([i % K for i in range(G)])
    rng.shuffle(lab_grupo)
    y_int = lab_grupo[gid]
    cv = StableStratifiedGroupKFold(n_splits=4, seed=42)

    obs = cv_bal_acc(X, y_int, gid, cv)
    if not np.isfinite(obs):
        return None

    acc_amostra, acc_grupo = [], []
    for _ in range(N_PERM):
        # Null A: permuta rotulos por AMOSTRA (implementacao atual)
        ya = y_int[rng.permutation(N)]
        a = cv_bal_acc(X, ya, gid, cv)
        if np.isfinite(a):
            acc_amostra.append(a)
        # Null B: permuta rotulos por GRUPO (correto p/ dado agrupado)
        yg = lab_grupo[rng.permutation(G)][gid]
        b = cv_bal_acc(X, yg, gid, cv)
        if np.isfinite(b):
            acc_grupo.append(b)

    pa = (np.sum(np.array(acc_amostra) >= obs) + 1) / (len(acc_amostra) + 1)
    pb = (np.sum(np.array(acc_grupo) >= obs) + 1) / (len(acc_grupo) + 1)
    return obs, pa, pb, float(np.mean(acc_amostra)), float(np.mean(acc_grupo)), \
        float(np.std(acc_amostra)), float(np.std(acc_grupo))


res = Parallel(n_jobs=-1, backend="loky")(
    delayed(uma_replica)(s) for s in range(N_REP))
res = [r for r in res if r is not None]
obs, pa, pb, ma, mb, sa, sb = map(np.array, zip(*res))

print("=" * 72)
print("A4. TESTE DE PERMUTACAO: unidade de troca (amostra vs grupo)")
print("=" * 72)
print(f"  Cenario: H0 VERDADEIRO, {G} grupos x {R} replicas, {K} classes, "
      f"{len(res)} repeticoes, {N_PERM} permutacoes cada")
print(f"  Acuracia balanceada observada (media): {obs.mean():.4f}  "
      f"(acaso = {1/K:.4f})")
print()
print("  Null A - permuta por AMOSTRA (implementado hoje):")
print(f"      media da distribuicao nula = {ma.mean():.4f}   sd = {sa.mean():.4f}")
print(f"      TAXA DE FALSO POSITIVO (p<0.05) = {np.mean(pa < 0.05):.3f}")
print(f"      p-valor mediano                 = {np.median(pa):.4f}")
print()
print("  Null B - permuta por GRUPO (correto):")
print(f"      media da distribuicao nula = {mb.mean():.4f}   sd = {sb.mean():.4f}")
print(f"      TAXA DE FALSO POSITIVO (p<0.05) = {np.mean(pb < 0.05):.3f}")
print(f"      p-valor mediano                 = {np.median(pb):.4f}")
print()
print("  Alvo para um teste calibrado: falso positivo ~= 0.050")
