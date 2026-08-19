"""BLOCO I -- a discriminacao de especie e' quimica ou assinatura de
sessao de medicao?

`mae_id = cod + data` impede que replicas da MESMA amostra se separem
entre treino e teste, mas NAO impede que a mesma DATA apareca dos dois
lados. Se cada especie foi medida em datas proprias, o modelo pode estar
lendo deriva de instrumento/ambiente daquela sessao em vez de composicao.

Tres medicoes, da mais barata para a mais cara:
  I.1  tabela especie x datas unicas
  I.2  classificador TRIVIAL: preve especie so' a partir da data.
       Acuracia alta => o confundimento existe e e' explorAvel.
  I.3  PLS-DA com GroupKFold POR DATA (nenhuma data em treino e teste ao
       mesmo tempo), comparado com o 0,9203 obtido agrupando por mae_id.

Uso: python medir_confundimento_data.py "<pasta .dx>"
"""
import sys
from collections import defaultdict

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from guaraci.dados_io import carregar_dx  # noqa: E402
from guaraci.pipeline import Config  # noqa: E402
from guaraci.preprocessamento import construir_preprocessador  # noqa: E402


def data_de(mae_id: str) -> str:
    """mae_id = 'COD-DD-MM-AAAA[-adulterante+teor]' -> 'DD-MM-AAAA'."""
    partes = str(mae_id).split("-")
    return "-".join(partes[1:4]) if len(partes) >= 4 else "?"


def i1_tabela(rot, datas, puros):
    print("=" * 74)
    print("I.1 -- ESPECIE x DATAS UNICAS DE MEDICAO")
    print("=" * 74)
    for nome, mask in (("PUROS", puros), ("TODOS", np.ones(len(rot), bool))):
        print(f"\n--- modo {nome} ---")
        print(f"{'especie':<20} {'n_espectros':>12} {'datas unicas':>13}")
        print("-" * 48)
        por_data = defaultdict(set)
        for esp in sorted(set(rot[mask])):
            idx = (rot == esp) & mask
            ds = sorted(set(datas[idx]))
            for d in ds:
                por_data[d].add(esp)
            print(f"{esp:<20} {int(idx.sum()):>12} {len(ds):>13}")
        compart = {d: e for d, e in por_data.items() if len(e) > 1}
        print(f"\n  datas totais ................. {len(por_data)}")
        print(f"  datas com >1 especie ......... {len(compart)}")
        print(f"  datas EXCLUSIVAS de 1 especie  {len(por_data) - len(compart)}"
              f"  ({100*(len(por_data)-len(compart))/max(len(por_data),1):.0f}%)")
        if compart:
            ex = list(compart.items())[:5]
            print("  exemplos de datas compartilhadas (n_especies):")
            for d, e in ex:
                print(f"    {d}: {len(e)} especies")


def i2_classificador_trivial(rot, datas):
    """So' a DATA prediz a especie? Regra: cada data vota na especie
    majoritaria daquela data; avalia por leave-one-DATE-out para nao ser
    trivialmente 100%."""
    print("\n" + "=" * 74)
    print("I.2 -- CLASSIFICADOR TRIVIAL: PREVER ESPECIE SO' PELA DATA")
    print("=" * 74)
    datas_u = np.unique(datas)
    acertos, total = 0, 0
    y_true, y_pred = [], []
    for d in datas_u:
        # treina nas OUTRAS datas, prediz esta
        outras = datas != d
        alvo = datas == d
        # especie majoritaria entre as outras datas -> chute base
        vals, cnt = np.unique(rot[outras], return_counts=True)
        chute_global = vals[np.argmax(cnt)]
        # se esta data existe em outro lugar (nao existe, por construcao),
        # usaria a especie dela; como nao existe, cai no chute global
        for r in rot[alvo]:
            y_true.append(r)
            y_pred.append(chute_global)
            acertos += int(r == chute_global)
            total += 1
    print(f"  datas unicas ................. {len(datas_u)}")
    print(f"  leave-one-DATE-out, acuracia . {acertos/max(total,1):.4f}")
    print(f"  bal.acc ...................... "
          f"{balanced_accuracy_score(y_true, y_pred):.4f}")
    print("  (baixo aqui NAO absolve: significa so' que a data isolada nao")
    print("   preve a especie fora da propria data. O teste que importa e' o I.3.)")

    # O teste direto: dentro do dataset, a data determina a especie?
    det = sum(1 for d in datas_u if len(set(rot[datas == d])) == 1)
    print(f"\n  datas que determinam UNICAMENTE a especie: {det}/{len(datas_u)}"
          f"  ({100*det/max(len(datas_u),1):.0f}%)")
    print("  ^ ESTE e' o indicador de confundimento: se ~100%, saber a data")
    print("    equivale a saber a especie, e o modelo pode aprender a sessao.")


def i3_groupkfold_por_data(X, rot, datas, n_lv_grade=(20, 40), seeds=range(3)):
    print("\n" + "=" * 74)
    print("I.3 -- PLS-DA COM GroupKFold POR DATA (vs. por mae_id)")
    print("=" * 74)
    classes = np.unique(rot)
    y_int = np.searchsorted(classes, rot)
    Y = np.zeros((len(rot), len(classes)))
    Y[np.arange(len(rot)), y_int] = 1.0

    n_datas = len(np.unique(datas))
    print(f"  grupos = datas unicas: {n_datas}")
    # e' possivel particionar? precisa de >= n_splits datas COM todas as classes
    n_splits = min(5, n_datas)
    if n_datas < 2:
        print("  IMPOSSIVEL particionar por data (menos de 2 datas).")
        return

    for n_lv in n_lv_grade:
        bals = []
        for _s in seeds:
            gkf = GroupKFold(n_splits=n_splits)
            y_hat = np.zeros_like(Y)
            visto = np.zeros(len(Y), bool)
            for tr, te in gkf.split(X, y_int, groups=datas):
                # classes ausentes do treino tornam o fold nao-avaliavel
                if len(np.unique(y_int[tr])) < len(classes):
                    continue
                nlv = int(min(n_lv, X.shape[1], len(tr) - 1))
                pipe = Pipeline([("mc", StandardScaler(with_std=False)),
                                 ("pls", PLSRegression(n_components=nlv,
                                                       scale=False))])
                pipe.fit(X[tr], Y[tr])
                y_hat[te] = pipe.predict(X[te])
                visto[te] = True
            if visto.sum() == 0:
                print(f"  n_LV={n_lv}: NENHUM fold avaliavel -- toda particao "
                      f"por data deixa classe fora do treino.")
                break
            bals.append(balanced_accuracy_score(
                y_int[visto], np.argmax(y_hat[visto], 1)))
        if bals:
            b = np.array(bals)
            print(f"  n_LV={n_lv:>3}: bal.acc {b.mean():.4f} "
                  f"(cobertura {100*visto.mean():.0f}% das amostras)")


if __name__ == "__main__":
    wn, X, rot, conc, mae_id, _meta = carregar_dx(sys.argv[1])
    rot = np.asarray(rot, dtype=str)
    mae_id = np.asarray(mae_id, dtype=str)
    datas = np.array([data_de(m) for m in mae_id], dtype=str)
    conc_f = np.asarray(conc, dtype=float)
    puros = np.isnan(conc_f) | (conc_f == 0.0)

    i1_tabela(rot, datas, puros)
    i2_classificador_trivial(rot, datas)

    m = (wn >= 4000) & (wn <= 10000)
    cfg = Config()
    cfg.preprocessamento_padrao = "msc_sg_mc"
    Xp = construir_preprocessador(cfg).fit_transform(X[:, m])
    i3_groupkfold_por_data(Xp, rot, datas)
