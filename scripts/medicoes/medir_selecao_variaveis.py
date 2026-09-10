"""FASE B / B1. Duas divergencias medidas em selecao_variaveis.py.

B1-1. O bal.acc reportado do iPLS e' o MAXIMO de n_intervalos avaliacoes
      feitas na MESMA particao de CV que depois reporta o numero
      (selecao_ipls linha 260 escolhe o melhor intervalo por
      balanced_accuracy sobre `cv_indices`; etapa4 linha 567 reavalia o
      vencedor na MESMA `cv_indices` e poe na tabela). Isso e' viés de
      maximo-de-N, e o numero entra numa tabela onde VIP/SR/sPLS-DA/SPA/AG
      ja' passaram por nested-CV. MEDE o tamanho do viés: iPLS "como esta'"
      contra iPLS com a escolha do intervalo refeita dentro de cada fold.

B1-2. `sparse_plsda_mask` faz truncamento DURO (mantem as keep_por_comp
      variaveis de maior |w|, sem encolher as sobreviventes). Le Cao et al.
      (2008) definem sPLS por penalizacao com SOFT-thresholding
      (w_j <- sign(w_j)*(|w_j|-lambda)_+). Com 1 componente os dois dao o
      MESMO conjunto; a partir de 2 as direcoes normalizadas diferem, a
      deflacao diverge, e o conjunto selecionado muda. MEDE Jaccard entre
      os dois conjuntos em funcao de n_comp.
"""
import sys

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "src")
from guaraci.selecao_variaveis import (  # noqa: E402
    _avaliar_subset_cv,
    _avaliar_subset_nested_cv,
    selecao_ipls,
    sparse_plsda_mask,
)


# ---------------------------------------------------------------- dados
def gera(n_grupos_por_classe=10, n_classes=4, n_rep=3, p=200, seed=0):
    """Espectros com estrutura de replica: cada grupo (mae_id) tem n_rep
    replicas quase identicas. Sinal de classe concentrado numa faixa
    estreita — o resto do espectro e' ruido correlacionado."""
    rng = np.random.default_rng(seed)
    wn = np.linspace(0, 1, p)
    X, y, g = [], [], []
    gid = 0
    for c in range(n_classes):
        centro = 0.2 + 0.15 * c
        for _ in range(n_grupos_por_classe):
            nivel = rng.normal(1.0, 0.25)
            base = nivel * np.exp(-((wn - centro) ** 2) / (2 * 0.02 ** 2))
            fundo = rng.normal(0, 1, p).cumsum() * 0.01   # deriva suave
            for _r in range(n_rep):
                X.append(base + fundo + rng.normal(0, 0.05, p))
                y.append(c)
                g.append(gid)
            gid += 1
    return np.asarray(X), np.asarray(y), np.asarray(g), wn


def y_binario(y_int, n_classes):
    Y = np.zeros((len(y_int), n_classes))
    Y[np.arange(len(y_int)), y_int] = 1.0
    return Y


# ------------------------------------------------------------- B1-1
def mask_melhor_intervalo(X_tr, Y_tr, n_lv, n_intervalos):
    """Escolhe o melhor intervalo usando SO' o treino do fold — a versao
    aninhada correta do que selecao_ipls faz no dataset inteiro."""
    p = X_tr.shape[1]
    bordas = np.linspace(0, p, n_intervalos + 1).astype(int)
    y_tr_int = np.argmax(Y_tr, axis=1)
    # CV interna simples sobre o treino do fold (sem grupos: e' so' para
    # ESCOLHER o intervalo; o numero reportado vem do fold externo)
    from sklearn.model_selection import StratifiedKFold
    _c, cont = np.unique(y_tr_int, return_counts=True)
    k = max(2, min(3, int(cont.min())))
    cv_in = list(StratifiedKFold(n_splits=k, shuffle=True,
                                 random_state=0).split(X_tr, y_tr_int))
    melhor_bal, melhor_ab = -1.0, (0, p)
    for i in range(n_intervalos):
        a, b = bordas[i], bordas[i + 1]
        if b - a < 2:
            continue
        m = _avaliar_subset_cv(X_tr[:, a:b], Y_tr, y_tr_int, cv_in, n_lv)
        if m["balanced_accuracy"] > melhor_bal:
            melhor_bal, melhor_ab = m["balanced_accuracy"], (a, b)
    mask = np.zeros(p, dtype=bool)
    mask[melhor_ab[0]:melhor_ab[1]] = True
    return mask


def b1_1(seeds=range(12), n_intervalos=10, n_lv=4):
    print("B1-1. iPLS: viés de maximo-de-N na particao que reporta\n")
    print(f"{'seed':>5} {'iPLS como esta':>15} {'iPLS aninhado':>14} "
          f"{'viés':>8}")
    print("-" * 46)
    gaps = []
    for s in seeds:
        X, y, g, wn = gera(seed=s)
        n_cls = len(np.unique(y))
        Y = y_binario(y, n_cls)
        cv = list(GroupKFold(n_splits=5).split(X, y, groups=g))

        # como esta' hoje: escolhe o melhor intervalo NA MESMA cv e reporta
        _res, mask = selecao_ipls(X, Y, y, wn, cv, n_lv, n_intervalos)
        atual = _avaliar_subset_cv(X[:, mask], Y, y, cv, n_lv)["balanced_accuracy"]

        # aninhado: escolha do intervalo refeita dentro de cada fold
        aninhado = _avaliar_subset_nested_cv(
            X, Y, y, cv, n_lv,
            lambda Xtr, Ytr, nlv: mask_melhor_intervalo(
                Xtr, Ytr, nlv, n_intervalos))["balanced_accuracy"]

        gaps.append(atual - aninhado)
        print(f"{s:>5} {atual:>15.4f} {aninhado:>14.4f} {atual-aninhado:>8.4f}")
    gaps = np.array(gaps)
    print(f"\n  viés medio  : {gaps.mean():+.4f} pontos de bal.acc")
    print(f"  viés mediano: {np.median(gaps):+.4f}")
    print(f"  positivo em : {int((gaps > 0).sum())}/{len(gaps)} seeds\n")


# ------------------------------------------------------------- B1-2
def splsda_mask_soft(X_proc, Y_bin, n_comp, keep_por_comp):
    """Mesma rotina de sparse_plsda_mask, trocando o truncamento DURO pelo
    SOFT-thresholding de Le Cao et al. (2008): lambda = o (keep+1)-esimo
    maior |w|, e as sobreviventes sao ENCOLHIDAS por lambda."""
    X = np.asarray(X_proc, dtype=float)
    Y = np.asarray(Y_bin, dtype=float)
    Xr = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    p = X.shape[1]
    selecionadas: set = set()
    n_comp = int(max(1, min(n_comp, p)))
    for _ in range(n_comp):
        M = Xr.T @ Yc
        try:
            U, _S, _Vt = np.linalg.svd(M, full_matrices=False)
            w = U[:, 0]
        except np.linalg.LinAlgError:
            break
        aw = np.abs(w)
        ordem = np.sort(aw)[::-1]
        lam = ordem[keep_por_comp] if keep_por_comp < p else 0.0
        w_sp = np.sign(w) * np.maximum(aw - lam, 0.0)     # soft-threshold
        nw = float(np.linalg.norm(w_sp))
        if nw < 1e-12:
            break
        w_sp /= nw
        t = Xr @ w_sp
        tt = float(t @ t)
        if tt < 1e-12:
            break
        pld = Xr.T @ t / tt
        Xr = Xr - np.outer(t, pld)
        c = Yc.T @ t / tt
        Yc = Yc - np.outer(t, c)
        selecionadas.update(int(i) for i in np.where(w_sp != 0)[0])
    mask = np.zeros(p, dtype=bool)
    if selecionadas:
        mask[list(selecionadas)] = True
    return mask


def b1_2(seeds=range(12), keep=15):
    print("B1-2. sPLS-DA: truncamento duro (codigo) vs soft-thresholding "
          "(Le Cao 2008)\n")
    print(f"{'n_comp':>7} {'Jaccard mediano':>17} {'|dif| mediana de vars':>23}")
    print("-" * 50)
    for n_comp in (1, 2, 3, 4, 5):
        jac, dif = [], []
        for s in seeds:
            X, y, g, _wn = gera(seed=s)
            Y = y_binario(y, len(np.unique(y)))
            m_duro = sparse_plsda_mask(X, Y, n_comp, keep)
            m_soft = splsda_mask_soft(X, Y, n_comp, keep)
            a, b = set(np.where(m_duro)[0]), set(np.where(m_soft)[0])
            uni = len(a | b)
            jac.append(len(a & b) / uni if uni else 1.0)
            dif.append(len(a ^ b))
        print(f"{n_comp:>7} {np.median(jac):>17.3f} {np.median(dif):>23.0f}")
    print()


if __name__ == "__main__":
    b1_1()
    b1_2()
