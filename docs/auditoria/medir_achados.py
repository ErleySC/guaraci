"""Mede empiricamente o impacto dos achados da auditoria. Nenhum numero
deste script e' suposto -- todos sao medidos."""
import numpy as np
from scipy.stats import beta as beta_dist, f as f_dist, chi2
from sklearn.cross_decomposition import PLSRegression

import sys
sys.path.insert(0, "src")
from guaraci.chemometric_stats import (calcular_selectivity_ratio,
                                       hotelling_t2_limite)

rng_global = np.random.default_rng(0)

print("=" * 72)
print("A1. SELECTIVITY RATIO: w1 (implementado) vs b/||b|| (Rajalahti/Kvalheim)")
print("=" * 72)


def sr_referencia(modelo, X):
    """SR conforme a literatura: alvo = vetor de regressao normalizado."""
    b = np.asarray(modelo.coef_, dtype=float).reshape(-1)
    nb = np.linalg.norm(b)
    w_tp = b / nb
    t_tp = X @ w_tp
    tt = float(t_tp @ t_tp)
    p_tp = (t_tp @ X) / tt
    X_tp = np.outer(t_tp, p_tp)
    X_res = X - X_tp
    vr = X_res.var(axis=0, ddof=1)
    vr[vr < 1e-12] = 1e-12
    return X_tp.var(axis=0, ddof=1) / vr


# Dado espectral sintetico realista: bandas gaussianas sobrepostas + ruido
def gera_espectros(n, p, seed):
    rng = np.random.default_rng(seed)
    wn = np.linspace(0, 1, p)
    conc = rng.uniform(0.1, 1.0, n)
    interf = rng.uniform(0.0, 1.0, n)
    banda_alvo = np.exp(-((wn - 0.30) ** 2) / (2 * 0.03 ** 2))
    banda_int = np.exp(-((wn - 0.65) ** 2) / (2 * 0.05 ** 2))
    base = rng.uniform(0, 0.3, n)[:, None] * wn[None, :]
    X = (conc[:, None] * banda_alvo[None, :]
         + interf[:, None] * banda_int[None, :]
         + base + rng.normal(0, 0.005, (n, p)))
    return X - X.mean(0), conc - conc.mean()


for n_lv in (1, 2, 3, 5):
    X, y = gera_espectros(60, 200, seed=1)
    m = PLSRegression(n_components=n_lv, scale=False).fit(X, y)
    sr_impl = calcular_selectivity_ratio(m, X)
    sr_ref = sr_referencia(m, X)
    # propriedade definidora: t_tp deve ser proporcional a y_hat
    b = np.asarray(m.coef_).reshape(-1)
    t_ref = X @ (b / np.linalg.norm(b))
    t_impl = X @ (m.x_weights_[:, 0] / np.linalg.norm(m.x_weights_[:, 0]))
    yhat = m.predict(X).ravel()
    cor_ref = abs(np.corrcoef(t_ref, yhat)[0, 1])
    cor_impl = abs(np.corrcoef(t_impl, yhat)[0, 1])
    # ranking das 20 variaveis mais importantes
    top_impl = set(np.argsort(sr_impl)[-20:])
    top_ref = set(np.argsort(sr_ref)[-20:])
    jac = len(top_impl & top_ref) / len(top_impl | top_ref)
    print(f"  LV={n_lv}: corr(t_tp, yhat) ref={cor_ref:.6f} impl={cor_impl:.6f} | "
          f"max SR ref={sr_ref.max():8.2f} impl={sr_impl.max():8.2f} | "
          f"Jaccard top-20 = {jac:.3f}")

print()
print("=" * 72)
print("A2. HOTELLING T2: limite Fase II (F, implementado) vs Fase I (Beta, TYM1992)")
print("=" * 72)
print("  Aplicado a AMOSTRAS DE TREINO (dominio_aplicabilidade_treino) o correto")
print("  e' Fase I. Razao > 1 => limite implementado alto demais (sub-deteccao).")
for n in (10, 20, 30, 50, 100, 300):
    for k in (2, 3):
        if n - k <= 1:
            continue
        lim_f = hotelling_t2_limite(n, k, 0.05)
        # Fase I exato (Tracy, Young & Mason 1992): T2 ~ ((n-1)^2/n) Beta(k/2,(n-k-1)/2)
        lim_beta = ((n - 1) ** 2 / n) * beta_dist.ppf(0.95, k / 2, (n - k - 1) / 2)
        print(f"  n={n:4d} k={k}: FaseII(F)={lim_f:8.3f}  FaseI(Beta)={lim_beta:8.3f}  "
              f"razao={lim_f / lim_beta:5.2f}x")

print()
print("=" * 72)
print("A3. DOMINIO DE APLICABILIDADE: regra retangular (T2 E Q independentes)")
print("=" * 72)
print("  Mesma classe do bug corrigido no DD-SIMCA, ainda presente em")
print("  dominio_aplicabilidade_amostras_novas (usado em predicao.py).")
# Simula: quantas amostras genuinas (mesma distribuicao do treino) sao
# rejeitadas pela regra retangular com alpha=0.05 em cada eixo?
from sklearn.decomposition import PCA
from guaraci.chemometric_stats import (dominio_aplicabilidade_treino,
                                       dominio_aplicabilidade_amostras_novas)
taxas = []
for seed in range(40):
    rng = np.random.default_rng(100 + seed)
    Xtr = rng.normal(0, 1, (200, 30))
    Xnew = rng.normal(0, 1, (2000, 30))   # MESMA distribuicao => H0 verdadeiro
    pca = PCA(n_components=3).fit(Xtr)
    tr = dominio_aplicabilidade_treino(pca, Xtr, alpha=0.05)
    r = dominio_aplicabilidade_amostras_novas(pca, Xnew, tr["var_t"],
                                              tr["t2_limite"], tr["q_limite"])
    taxas.append(1.0 - float(r["fracao_dentro"]))
taxas = np.array(taxas)
print(f"  alpha nominal declarado          : 0.050")
print(f"  taxa de rejeicao MEDIDA (n=40 sim): {taxas.mean():.4f} "
      f"(sd={taxas.std():.4f})")
print(f"  1-(1-alpha)^2 esperado p/ retangular: {1 - 0.95 ** 2:.4f}")
