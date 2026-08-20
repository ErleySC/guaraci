"""FASE B / B2-1. Quantos splits o Monte Carlo CV descarta, e o IC95%
reportado sobre os sobreviventes e' otimista?

`monte_carlo_cv` (avaliacao_modelos.py:406) pula a iteracao quando o TREINO
nao contem todas as classes:

    if len(np.unique(y_tr)) < len(lb.classes_): continue
    if len(np.unique(y_te)) < 2: continue

Com split no nivel de GRUPO e classes com poucos grupos, as iteracoes
descartadas sao as MAIS DIFICEIS (o modelo nao conseguiria prever a classe
ausente do treino -> balanced accuracy baixa). Descarta-las remove a cauda
inferior, entao Media BA e sobretudo o IC95% ficam mais altos e mais
estreitos que a distribuicao real do procedimento.

MEDE, num regime tipico de classificacao multiclasse com poucos grupos
por classe entre os puros -- o caso em que o descarte de iteracao morde
mais:
  (1) fracao de iteracoes descartadas;
  (2) BA das descartadas vs BA das sobreviventes -- se as descartadas
      forem sistematicamente piores, o vies e' confirmado e quantificado;
  (3) IC95% sobre sobreviventes vs IC95% sobre TODAS as iteracoes.

Para (2)/(3) e' preciso AVALIAR as iteracoes que o codigo descarta, o que
o codigo de producao nao faz -- por isso este script reimplementa o laco
de forma minima (mesma logica de split, mesmo classificador) em vez de
chamar monte_carlo_cv() direto.
"""
import sys
import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, "src")
from guaraci.avaliacao_modelos import _stratified_group_shuffle_splits  # noqa: E402

warnings.filterwarnings("ignore")


def gera_regime(n_classes: int, grupos_por_classe: int, n_rep: int,
                p: int, seed: int):
    """Estrutura de replica realista: cada classe tem `grupos_por_classe`
    amostras fisicas, cada uma com `n_rep` replicas tecnicas."""
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    gid = 0
    # Classes deliberadamente POUCO separadas (separacao 0.25 contra ruido
    # entre-amostras 1.0): com classes bem separadas a BA satura em 1.000 e
    # a comparacao "descartadas vs sobreviventes" nao diz nada.
    for c in range(n_classes):
        centro = rng.normal(scale=1.0, size=p) + c * 0.25
        for _ in range(grupos_por_classe):
            base = centro + rng.normal(scale=1.0, size=p)
            for _r in range(n_rep):
                X.append(base + rng.normal(scale=0.25, size=p))
                y.append(c)
                g.append(gid)
            gid += 1
    return np.asarray(X), np.asarray(y), np.asarray(g)


def mede(n_classes, grupos_por_classe, n_rep=3, p=40, n_iter=200,
         test_size=0.25, seed=0):
    X, y, g = gera_regime(n_classes, grupos_por_classe, n_rep, p, seed)
    try:
        splits = _stratified_group_shuffle_splits(y, g, n_iter, test_size, seed)
    except ValueError as e:
        # StratifiedShuffleSplit exige n_grupos_teste >= n_classes. Com
        # poucos grupos por classe o splitter nem chega a rodar -- e' um
        # modo de falha DIFERENTE do descarte silencioso (este e' ruidoso,
        # levanta excecao), registrado aqui para nao ser confundido com ele.
        return None, str(e)

    ba_sobreviventes, ba_descartadas = [], []
    for tr, te in splits:
        y_tr, y_te = y[tr], y[te]
        descartada = (len(np.unique(y_tr)) < n_classes
                      or len(np.unique(y_te)) < 2)
        try:
            clf = LinearDiscriminantAnalysis()
            clf.fit(X[tr], y_tr)
            ba = float(balanced_accuracy_score(y_te, clf.predict(X[te])))
        except Exception:                      # noqa: BLE001 -- medicao
            continue
        (ba_descartadas if descartada else ba_sobreviventes).append(ba)

    return np.array(ba_sobreviventes), np.array(ba_descartadas)  # type: ignore[return-value]


def ic95(a):
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


if __name__ == "__main__":
    print("Monte Carlo CV: iteracoes descartadas por classe ausente no treino")
    print("(split group-aware, LDA, 200 iteracoes por celula)\n")
    cab = (f"{'classes':>8} {'grupos/cl':>10} {'descarte':>9} "
           f"{'BA sobrev.':>11} {'BA descart.':>12} "
           f"{'IC95 sobrev.':>20} {'IC95 todas':>20}")
    print(cab)
    print("-" * len(cab))
    # Varredura de regimes, do mais folgado (muitos grupos por classe) ao
    # mais estressado (poucos grupos por classe) -- nao ha' uma celula unica
    # "real": o ponto e' mostrar o descarte de 0% em toda a faixa
    # plausivel, nao amarrar o numero de grupos do acervo a uma celula.
    for n_classes, gpc in ((10, 50), (10, 20), (10, 10), (10, 6), (10, 4),
                           (10, 3), (5, 3)):
        s, d = mede(n_classes, gpc)
        if s is None:
            print(f"{n_classes:>8} {gpc:>10}   splitter falhou: {d[:52]}")
            continue
        total = len(s) + len(d)
        if total == 0:
            continue
        frac = 100.0 * len(d) / total
        todas = np.concatenate([s, d]) if len(d) else s
        ic_s = ic95(s) if len(s) else (float("nan"),) * 2
        ic_t = ic95(todas)
        ba_d = f"{d.mean():.4f}" if len(d) else "     -"
        print(f"{n_classes:>8} {gpc:>10} {frac:>8.1f}% "
              f"{s.mean() if len(s) else float('nan'):>11.4f} {ba_d:>12} "
              f"{f'[{ic_s[0]:.3f},{ic_s[1]:.3f}]':>20} "
              f"{f'[{ic_t[0]:.3f},{ic_t[1]:.3f}]':>20}")
