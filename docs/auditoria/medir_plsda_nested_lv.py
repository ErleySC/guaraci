"""BLOCO F -- bal.acc do PLS-DA discriminante com o numero de LVs escolhido
por CV ANINHADA, nao por varredura na particao que reporta.

O 0,9167 do BLOCO C foi obtido varrendo n_LV em (10,20,30) e reportando o
melhor -- exatamente o vies de maximo-de-N que o achado B1-1 corrigiu no
iPLS. Aqui a escolha do n_LV acontece numa particao INTERNA ao fold de
treino, e a avaliacao no fold externo que a busca nunca viu.

Reporta os dois lado a lado para a diferenca ficar explicita.
"""
import sys

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from guaraci.dados_io import carregar_dx  # noqa: E402
from guaraci.pipeline import Config  # noqa: E402
from guaraci.preprocessamento import construir_preprocessador  # noqa: E402
from guaraci.validacao_estatistica import StratifiedGroupKFoldEstavel  # noqa: E402

GRADE_LV = (5, 10, 15, 20, 25, 30, 35, 40)


def _ajusta_prediz(X_tr, Y_tr, X_te, n_lv):
    n_lv = int(max(1, min(n_lv, X_tr.shape[1], X_tr.shape[0] - 1)))
    pipe = Pipeline([("mc", StandardScaler(with_std=False)),
                     ("pls", PLSRegression(n_components=n_lv, scale=False))])
    pipe.fit(X_tr, Y_tr)
    return pipe.predict(X_te)


def bal_acc_varredura(X, Y, y_int, grupos, n_splits, seed):
    """COMO FOI FEITO NO BLOCO C: avalia cada n_LV na MESMA particao e
    devolve o melhor -- o numero enviesado, reproduzido aqui de proposito
    para a comparacao ser direta."""
    cv = StratifiedGroupKFoldEstavel(n_splits=n_splits, seed=seed)
    folds = list(cv.split(X, y_int, groups=grupos))
    melhor, melhor_lv = -1.0, None
    for n_lv in GRADE_LV:
        y_hat = np.zeros_like(Y)
        for tr, te in folds:
            y_hat[te] = _ajusta_prediz(X[tr], Y[tr], X[te], n_lv)
        b = balanced_accuracy_score(y_int, np.argmax(y_hat, 1))
        if b > melhor:
            melhor, melhor_lv = b, n_lv
    return float(melhor), melhor_lv


def bal_acc_nested(X, Y, y_int, grupos, n_splits, seed, n_splits_int=3):
    """n_LV escolhido DENTRO de cada fold de treino (particao interna,
    group-aware), avaliado no fold externo que a escolha nunca viu."""
    cv = StratifiedGroupKFoldEstavel(n_splits=n_splits, seed=seed)
    y_hat = np.zeros_like(Y)
    lvs_escolhidos = []
    for tr, te in cv.split(X, y_int, groups=grupos):
        X_tr, Y_tr, y_tr, g_tr = X[tr], Y[tr], y_int[tr], grupos[tr]
        cv_int = StratifiedGroupKFoldEstavel(n_splits=n_splits_int, seed=seed)
        folds_int = list(cv_int.split(X_tr, y_tr, groups=g_tr))
        melhor, melhor_lv = -1.0, GRADE_LV[0]
        for n_lv in GRADE_LV:
            y_int_hat = np.zeros_like(Y_tr)
            for tri, tei in folds_int:
                y_int_hat[tei] = _ajusta_prediz(
                    X_tr[tri], Y_tr[tri], X_tr[tei], n_lv)
            b = balanced_accuracy_score(y_tr, np.argmax(y_int_hat, 1))
            if b > melhor:
                melhor, melhor_lv = b, n_lv
        lvs_escolhidos.append(melhor_lv)
        y_hat[te] = _ajusta_prediz(X_tr, Y_tr, X[te], melhor_lv)
    return float(balanced_accuracy_score(y_int, np.argmax(y_hat, 1))), lvs_escolhidos


def curva_lv(X, Y, y_int, grupos, n_splits, seed):
    """bal.acc por n_LV fixo -- para ver se satura ou se 40 ainda sobe."""
    cv = StratifiedGroupKFoldEstavel(n_splits=n_splits, seed=seed)
    folds = list(cv.split(X, y_int, groups=grupos))
    out = {}
    for n_lv in GRADE_LV:
        y_hat = np.zeros_like(Y)
        for tr, te in folds:
            y_hat[te] = _ajusta_prediz(X[tr], Y[tr], X[te], n_lv)
        out[n_lv] = float(balanced_accuracy_score(y_int, np.argmax(y_hat, 1)))
    return out


if __name__ == "__main__":
    pasta = sys.argv[1]
    wn, X, rot, _conc, mae_id, _meta = carregar_dx(pasta)
    rot = np.asarray(rot, dtype=str)
    grupos = np.asarray(mae_id, dtype=str)

    m = (wn >= 4000) & (wn <= 10000)
    cfg = Config()
    cfg.preprocessamento_padrao = "msc_sg_mc"
    Xp = construir_preprocessador(cfg).fit_transform(X[:, m])

    classes = np.unique(rot)
    y_int = np.searchsorted(classes, rot)
    Y = np.zeros((len(rot), len(classes)))
    Y[np.arange(len(rot)), y_int] = 1.0

    SEEDS = range(5)
    print(f"\nclasses={len(classes)}  grupos={len(np.unique(grupos))}  "
          f"variaveis={Xp.shape[1]}  seeds={len(SEEDS)}\n")

    print("--- curva bal.acc por n_LV fixo (satura?) ---")
    c = curva_lv(Xp, Y, y_int, grupos, 5, 0)
    for k, v in c.items():
        print(f"  n_LV={k:>3} -> {v:.4f}")

    print("\n--- VARREDURA (enviesado, como no BLOCO C) vs NESTED-CV ---")
    print(f"{'seed':>5} {'varredura':>11} {'LV*':>5} {'nested':>9} "
          f"{'LVs internos':>26} {'vies':>8}")
    print("-" * 72)
    v_all, n_all = [], []
    for s in SEEDS:
        bv, lv = bal_acc_varredura(Xp, Y, y_int, grupos, 5, s)
        bn, lvs = bal_acc_nested(Xp, Y, y_int, grupos, 5, s)
        v_all.append(bv); n_all.append(bn)
        print(f"{s:>5} {bv:>11.4f} {lv:>5} {bn:>9.4f} "
              f"{str(lvs):>26} {bv - bn:>8.4f}")
    v, n = np.array(v_all), np.array(n_all)
    print("-" * 72)
    print(f"  varredura : media {v.mean():.4f}  IC95% "
          f"[{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]")
    print(f"  nested-CV : media {n.mean():.4f}  IC95% "
          f"[{np.percentile(n, 2.5):.4f}, {np.percentile(n, 97.5):.4f}]")
    print(f"  VIES MEDIO: {np.mean(v - n):+.4f}")
