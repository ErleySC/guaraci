"""BLOCO 1.1/1.2 -- n efetivo do DD-SIMCA por especie, e o que exatamente
h0/q0/Nh/Nq estimam quando ha' 1 unica amostra fisica.

1.1: tabela especie | n_espectros | n_amostras_fisicas (mae_id unicos) |
     n usado em h0/q0/Nh/Nq -- para os PUROS (modo one-class do N2) e
     para o modo `todos`.

1.2: execucao passo a passo do calculo do limiar numa classe concreta,
     com os numeros reais, mostrando os intermediarios -- nao a formula
     em abstrato.

Uso:
    python medir_n_efetivo_ddsimca.py "<pasta com .dx>"
"""
import sys

import numpy as np

sys.path.insert(0, "src")
from guaraci.chemometric_stats import media_e_dof_momentos  # noqa: E402
from guaraci.classificadores import DDSimca  # noqa: E402
from guaraci.dados_io import carregar_dx  # noqa: E402
from guaraci.pipeline import Config  # noqa: E402
from guaraci.preprocessamento import construir_preprocessador  # noqa: E402


def carregar(pasta):
    wn, X, rot, conc, mae_id, meta = carregar_dx(pasta)
    conc_f = np.asarray(conc, dtype=float)
    # Mesma regra do pipeline: NaN ou 0.0 == amostra pura
    mask_puros = np.isnan(conc_f) | (conc_f == 0.0)
    return X, np.asarray(rot, dtype=str), np.asarray(mae_id, dtype=str), mask_puros


def tabela_n(rot, mae_id, mask_puros):
    print("1.1 -- n EFETIVO POR ESPECIE\n")
    print("MODO 'puros' (one-class do N2: DD-SIMCA treina SO' nas puras)")
    print(f"{'especie':<20} {'n_espectros':>12} {'n_amostras_fisicas':>19} "
          f"{'n usado p/ h0,q0,Nh,Nq':>24}")
    print("-" * 78)
    especies = sorted(set(rot))
    tot_e = tot_a = 0
    for esp in especies:
        idx = (rot == esp) & mask_puros
        n_esp = int(idx.sum())
        if n_esp == 0:
            continue
        n_fis = int(len(np.unique(mae_id[idx])))
        tot_e += n_esp
        tot_a += n_fis
        alerta = "  <-- NAO ESTIMAVEL" if n_fis < 2 else ""
        print(f"{esp:<20} {n_esp:>12} {n_fis:>19} {n_fis:>24}{alerta}")
    print("-" * 78)
    print(f"{'TOTAL':<20} {tot_e:>12} {tot_a:>19} {tot_a:>24}\n")

    print("MODO 'todos' (treina em TODAS as amostras da classe)")
    print(f"{'especie':<20} {'n_espectros':>12} {'n_amostras_fisicas':>19} "
          f"{'n usado p/ h0,q0,Nh,Nq':>24}")
    print("-" * 78)
    for esp in especies:
        idx = rot == esp
        n_esp = int(idx.sum())
        n_fis = int(len(np.unique(mae_id[idx])))
        print(f"{esp:<20} {n_esp:>12} {n_fis:>19} {n_fis:>24}")
    print()


def passo_a_passo(X, rot, mae_id, mask_puros, especie=None):
    """1.2 -- executa o calculo do limiar numa classe real, imprimindo os
    intermediarios."""
    especies = sorted(set(rot[mask_puros]))
    if especie is None:
        especie = especies[0]
    idx = (rot == especie) & mask_puros
    Xc = X[idx]
    grupos = mae_id[idx]

    print(f"1.2 -- CALCULO DO LIMIAR, PASSO A PASSO ({especie})\n")
    print(f"  espectros de treino .......... {Xc.shape[0]}")
    print(f"  variaveis .................... {Xc.shape[1]}")
    print(f"  mae_id distintos ............. {len(np.unique(grupos))}  "
          f"{sorted(set(grupos))}")

    # Pre-processamento igual ao do pipeline (MSC+SG+MC, o preset usado nas
    # execucoes reais do TCC)
    _cfg = Config()
    _cfg.preprocessamento_padrao = "msc_sg_mc"
    Xp = construir_preprocessador(_cfg).fit_transform(Xc)

    dd = DDSimca(n_components=3, alpha=0.05, ucl_method="empirical")
    dd.fit(Xp, np.array([especie] * len(Xp)), mae_id=grupos)
    if especie not in dd._modelos:
        print("  modelo PULADO (amostras insuficientes)")
        return
    m = dd._modelos[especie]

    print(f"\n  n_comp efetivo ............... {m['n_comp']}")
    print(f"  T2_train (por espectro) ...... "
          f"{np.array2string(m['T2_train'], precision=4)}")
    print(f"  Q_train  (LOO, por espectro) . "
          f"{np.array2string(m['Q_train'], precision=6)}")

    T2_g = DDSimca._media_por_grupo(m["T2_train"], grupos)
    Q_g = DDSimca._media_por_grupo(m["Q_train"], grupos)
    print("\n  --> colapso por amostra fisica (media por mae_id):")
    print(f"  T2 por AMOSTRA ............... {np.array2string(T2_g, precision=4)}")
    print(f"  Q  por AMOSTRA ............... {np.array2string(Q_g, precision=6)}")
    print(f"  ou seja: o metodo dos momentos recebe {T2_g.size} valor(es).")

    h0, Nh = media_e_dof_momentos(T2_g)
    q0, Nq = media_e_dof_momentos(Q_g)
    print(f"\n  media_e_dof_momentos(T2) -> h0={h0:.6g}, Nh={Nh:.6g}")
    print(f"  media_e_dof_momentos(Q)  -> q0={q0:.6g}, Nq={Nq:.6g}")

    # Reproduz a logica interna para deixar explicito de onde vem o N
    for nome, v in (("T2", T2_g), ("Q", Q_g)):
        media = float(v.mean())
        desvio = float(v.std(ddof=1)) if v.size > 1 else 0.0
        print(f"\n  {nome}: media={media:.6g}  desvio(ddof=1)="
              f"{'indefinido (n=1)' if v.size < 2 else f'{desvio:.6g}'}")
        if v.size < 2 or desvio <= 0:
            print("      -> desvio nao estimavel: N cai no PISO 1.0 "
                  "(nao e' um grau de liberdade medido dos dados)")
        else:
            print(f"      -> N = 2*(media/desvio)^2 = {2.0*(media/desvio)**2:.6g}")

    from scipy.stats import chi2
    f_crit = float(chi2.ppf(1 - 0.05, Nh + Nq))
    print(f"\n  f_crit = chi2.ppf(0.95, Nh+Nq={Nh + Nq:.6g}) = {f_crit:.6g}")
    print(f"  regiao de aceitacao: (T2/h0)*Nh + (Q/q0)*Nq <= {f_crit:.6g}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    X, rot, mae_id, mask_puros = carregar(sys.argv[1])
    tabela_n(rot, mae_id, mask_puros)
    passo_a_passo(X, rot, mae_id, mask_puros,
                  especie=sys.argv[2] if len(sys.argv) > 2 else None)
