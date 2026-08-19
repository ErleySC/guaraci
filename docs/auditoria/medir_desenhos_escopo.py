"""BLOCO C -- os dois numeros para a reuniao de escopo, nos DADOS REAIS.

(A) PLS-DA discriminante nas 13 especies com GroupKFold real (por mae_id):
    bal.acc com IC95%. Validacao preliminar, NAO o pipeline de publicacao.

(B) Conformal one-class AGRUPADO (as 13 especies como classe unica "oleo
    puro"): confirma o alpha alcancavel com o n real de amostras fisicas
    puras.

Uso:  python medir_desenhos_escopo.py "<pasta com .dx>"
"""
import sys

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from guaraci.conformal import alpha_alcancavel, limiar_conformal  # noqa: E402
from guaraci.dados_io import carregar_dx  # noqa: E402
from guaraci.pipeline import Config  # noqa: E402
from guaraci.preprocessamento import construir_preprocessador  # noqa: E402
from guaraci.validacao_estatistica import StratifiedGroupKFoldEstavel  # noqa: E402


def carregar(pasta):
    _wn, X, rot, conc, mae_id, _meta = carregar_dx(pasta)
    conc_f = np.asarray(conc, dtype=float)
    puros = np.isnan(conc_f) | (conc_f == 0.0)
    return (X, np.asarray(rot, dtype=str), np.asarray(mae_id, dtype=str), puros)


def desenho_a(X, rot, mae_id, n_lv=10, n_splits=5, seeds=range(10)):
    """PLS-DA discriminante de especie, GroupKFold real por mae_id."""
    cfg = Config()
    cfg.preprocessamento_padrao = "msc_sg_mc"
    Xp = construir_preprocessador(cfg).fit_transform(X)

    classes = np.unique(rot)
    y_int = np.searchsorted(classes, rot)
    Y = np.zeros((len(rot), len(classes)))
    Y[np.arange(len(rot)), y_int] = 1.0

    bals = []
    for s in seeds:
        cv = StratifiedGroupKFoldEstavel(n_splits=n_splits, seed=s)
        y_hat = np.zeros_like(Y)
        for tr, te in cv.split(Xp, y_int, groups=mae_id):
            pipe = Pipeline([("mc", StandardScaler(with_std=False)),
                             ("pls", PLSRegression(n_components=n_lv,
                                                   scale=False))])
            pipe.fit(Xp[tr], Y[tr])
            y_hat[te] = pipe.predict(Xp[te])
        bals.append(float(balanced_accuracy_score(y_int, np.argmax(y_hat, 1))))
    b = np.array(bals)
    return {"media": float(b.mean()), "dp": float(b.std(ddof=1)),
            "ic_lo": float(np.percentile(b, 2.5)),
            "ic_hi": float(np.percentile(b, 97.5)),
            "n_classes": len(classes), "n_grupos": int(len(np.unique(mae_id))),
            "n_seeds": len(bals)}


def desenho_b(mae_id, puros):
    """Conformal AGRUPADO: as 13 especies como uma classe unica."""
    n_fis = int(len(np.unique(mae_id[puros])))
    out = {"n_amostras_fisicas": n_fis,
           "alpha_min": alpha_alcancavel(n_fis)}
    for a in (0.05, 0.10, 0.15, 0.20):
        r = limiar_conformal(np.arange(float(n_fis)), alpha=a)
        out[f"alpha_{a:.2f}"] = r["alcancavel"]
    return out


if __name__ == "__main__":
    X, rot, mae_id, puros = carregar(sys.argv[1])

    print("\n" + "=" * 70)
    print("(B) CONFORMAL ONE-CLASS AGRUPADO -- dados reais")
    print("=" * 70)
    b = desenho_b(mae_id, puros)
    print(f"  amostras fisicas puras (13 especies juntas) : "
          f"{b['n_amostras_fisicas']}")
    print(f"  alpha minimo garantivel = 1/(n+1) ......... : {b['alpha_min']:.4f}")
    for a in (0.05, 0.10, 0.15, 0.20):
        print(f"  alpha={a:.2f} alcancavel? ................... : "
              f"{b[f'alpha_{a:.2f}']}")

    print("\n" + "=" * 70)
    print("(A) PLS-DA DISCRIMINANTE DE ESPECIE -- GroupKFold real por mae_id")
    print("=" * 70)
    a = desenho_a(X, rot, mae_id)
    print(f"  classes .................. {a['n_classes']}")
    print(f"  grupos mae_id ............ {a['n_grupos']}")
    print(f"  seeds .................... {a['n_seeds']}")
    print(f"  balanced accuracy ........ {a['media']:.4f}  (dp {a['dp']:.4f})")
    print(f"  IC95% (percentil) ........ [{a['ic_lo']:.4f}, {a['ic_hi']:.4f}]")
