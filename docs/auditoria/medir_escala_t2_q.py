"""BLOCO A.1 -- onde exatamente T2/Q e h0/q0 divergem de escala?

Hipotese do prompt: Q estaria em escala LOO (valores maiores) enquanto q0
teria sido calibrado noutra base, e T2 continuaria in-sample com h0
compativel. Este script NAO assume a hipotese: imprime as quatro series
lado a lado e mede a razao entre o que o modelo VE no treino e o que
amostras NOVAS produzem, eixo por eixo.

O diagnostico que importa e' comparar, para cada eixo:
    escala usada para calibrar o limiar   (h0 / q0)
    valores que amostras NOVAS produzem   (T2_novo / Q_novo)
Se a razao mediana T2_novo/h0 (ou Q_novo/q0) for >> 1, aquele eixo esta'
inflado e empurra `f` para cima artificialmente -> super-rejeicao.
"""
import sys

import numpy as np

sys.path.insert(0, "src")
from guaraci.classificadores import DDSimca  # noqa: E402


def gera_classe(n_amostras, n_rep, p, seed):
    """Mesmo gerador da medicao de cobertura: variacao ENTRE amostras
    (biologica) >> variacao ENTRE replicas (instrumental)."""
    rng = np.random.default_rng(seed)
    wn = np.linspace(0, 1, p)
    base = np.exp(-((wn - 0.35) ** 2) / (2 * 0.05 ** 2))
    X, grupos = [], []
    for i in range(n_amostras):
        amostra = base * rng.normal(1.0, 0.15) + rng.normal(0, 0.02, p)
        for _ in range(n_rep):
            X.append(amostra + rng.normal(0, 0.004, p))
            grupos.append(f"g{i}")
    return np.asarray(X), np.asarray(grupos, dtype=str)


def diagnostico(n_treino=50, n_teste=50, n_rep=3, p=400, n_comp=2, seed=0):
    X, g = gera_classe(n_treino + n_teste, n_rep, p, seed)
    gs = np.unique(g)
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(gs)
    m_tr = np.isin(g, gs[:n_treino])
    m_te = np.isin(g, gs[n_treino:])

    dd = DDSimca(n_components=n_comp, alpha=0.05, ucl_method="empirical")
    dd.fit(X[m_tr], np.array(["_c"] * int(m_tr.sum())), mae_id=g[m_tr])
    m = dd._modelos["_c"]

    T2_tr_ins = np.asarray(m["T2_train"], dtype=float)   # IN-SAMPLE
    Q_tr_loo = np.asarray(m["Q_train"], dtype=float)     # LEAVE-ONE-OUT
    h0, q0 = float(m["h0"]), float(m["q0"])
    Nh, Nq = float(m["Nh"]), float(m["Nq"])

    # O que o mesmo modelo produz para amostras NOVAS (out-of-sample real)
    T2_novo, Q_novo = dd._t2_q(X[m_te], "_c")

    # Q in-sample do treino, para isolar o efeito do LOO
    _T2_ins, Q_tr_ins = dd._t2_q(X[m_tr], "_c")

    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print(f"=== DIAGNOSTICO DE ESCALA (n_treino={n_treino} amostras fisicas, "
          f"n_comp={m['n_comp']}) ===\n")

    print("--- EIXO T2 ---")
    print(f"T2_train (in-sample, 1as 12 de {T2_tr_ins.size}):\n  {T2_tr_ins[:12]}")
    print(f"  mediana treino ......... {np.median(T2_tr_ins):.4f}")
    print(f"h0 (escala calibrada) .... {h0:.4f}   Nh={Nh:.4f}")
    print(f"T2_novo (out-of-sample, 1as 12):\n  {np.asarray(T2_novo)[:12]}")
    print(f"  mediana novas .......... {np.median(T2_novo):.4f}")
    print(f"  >> razao mediana T2_novo/h0 = {np.median(T2_novo)/h0:.4f}")

    print("\n--- EIXO Q ---")
    print(f"Q_train (LOO, 1as 12 de {Q_tr_loo.size}):\n  {Q_tr_loo[:12]}")
    print(f"  mediana treino LOO ..... {np.median(Q_tr_loo):.6f}")
    print(f"Q_train (IN-SAMPLE, 1as 12):\n  {np.asarray(Q_tr_ins)[:12]}")
    print(f"  mediana treino in-sample {np.median(Q_tr_ins):.6f}")
    print(f"  razao LOO/in-sample .... {np.median(Q_tr_loo)/np.median(Q_tr_ins):.4f}")
    print(f"q0 (escala calibrada) .... {q0:.6f}   Nq={Nq:.4f}")
    print(f"Q_novo (out-of-sample, 1as 12):\n  {np.asarray(Q_novo)[:12]}")
    print(f"  mediana novas .......... {np.median(Q_novo):.6f}")
    print(f"  >> razao mediana Q_novo/q0 = {np.median(Q_novo)/q0:.4f}")

    # Contribuicao de cada eixo para f, e onde f_crit corta
    f_novo = (T2_novo / h0) * Nh + (Q_novo / q0) * Nq
    contrib_t2 = (T2_novo / h0) * Nh
    contrib_q = (Q_novo / q0) * Nq
    f_crit = float(m["f_crit"])
    print("\n--- DECOMPOSICAO DE f NAS AMOSTRAS NOVAS ---")
    print(f"contribuicao T2 (mediana) . {np.median(contrib_t2):.4f}")
    print(f"contribuicao Q  (mediana) . {np.median(contrib_q):.4f}")
    print(f"f (mediana) ............... {np.median(f_novo):.4f}")
    print(f"f_crit = chi2(0.95, {Nh + Nq:.2f}) = {f_crit:.4f}")
    print(f"cobertura (fracao aceita) . {np.mean(f_novo <= f_crit):.4f}")
    print(f"\n  f_crit/(Nh+Nq) = {f_crit/(Nh+Nq):.4f}  <- margem relativa que a")
    print("  aproximacao chi2 concede: uma amostra nova so' e' aceita se a")
    print("  media ponderada de T2/h0 e Q/q0 ficar dentro dessa fracao de 1.")
    return {"razao_t2": np.median(T2_novo) / h0,
            "razao_q": np.median(Q_novo) / q0,
            "cobertura": float(np.mean(f_novo <= f_crit))}


if __name__ == "__main__":
    diagnostico()
    print("\n\n=== O MESMO, VARIANDO n_treino (o eixo que a medicao do "
          "BLOCO 1 NAO variava para o DD-SIMCA) ===")
    print(f"{'n_treino':>9} {'T2_novo/h0':>12} {'Q_novo/q0':>12} "
          f"{'f_crit/(Nh+Nq)':>16} {'cobertura':>11}")
    print("-" * 64)
    for n in (10, 19, 30, 50, 100):
        rs = [diagnostico.__wrapped__(n) if hasattr(diagnostico, "__wrapped__")
              else None]
        # roda silenciosamente
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = diagnostico(n_treino=n, seed=1)
        print(f"{n:>9} {r['razao_t2']:>12.4f} {r['razao_q']:>12.4f} "
              f"{'':>16} {r['cobertura']:>11.4f}")
