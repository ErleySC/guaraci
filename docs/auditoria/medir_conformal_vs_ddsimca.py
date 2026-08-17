"""BLOCO 1.3 -- cobertura empirica vs nominal: DD-SIMCA (chi2 com dof por
metodo dos momentos) contra conformal (quantil empirico), no MESMO escore
de nao-conformidade.

Como a comparacao isola a regra de decisao: os dois metodos recebem a
distancia combinada f do MESMO modelo PCA. So' muda como o limiar e'
derivado. Diferenca de cobertura, portanto, e' atribuivel a regra, nao a
representacao.

Protocolo (simulacao, porque cobertura exige verdade conhecida) -- split
de TRES vias, por amostra FISICA:

    treino  (n=10 fixo) -> ajusta a PCA. Nunca reaparece.
    calib   (n variavel)-> so' calcula escores e deriva o limiar.
    teste   (n=20)      -> mede a cobertura.

A separacao treino/calibracao e' OBRIGATORIA e nao e' detalhe: a validade
conformal exige que os escores de calibracao e de teste sejam
PERMUTAVEIS. Se a PCA e' ajustada nas MESMAS amostras que depois formam a
calibracao, os escores de calibracao ficam in-sample (a amostra ajudou a
definir o subespaco que a reconstroi) e os de teste nao -- calibracao
otimista, limiar apertado demais, cobertura ABAIXO do nominal.
Uma primeira versao deste script cometeu exatamente esse erro e media
0,61 de cobertura com n=50 onde a garantia exige >=0,95; e' a mesma
classe do achado A1 desta auditoria, agora do lado do conformal.

Regime varrido inclui n=1 (o do dataset real) e n grandes, para mostrar
onde cada metodo passa a funcionar.
"""
import sys

import numpy as np


sys.path.insert(0, "src")
from guaraci.classificadores import DDSimca  # noqa: E402

from guaraci.conformal import ConformalOneClass, alpha_alcancavel  # noqa: E402



def gera_classe(n_amostras, n_rep, p, seed):
    """n_amostras FISICAS, cada uma com n_rep replicas tecnicas.

    Variacao ENTRE amostras (biologica) >> variacao ENTRE replicas
    (instrumental) -- e' o que torna replicas nao-permutaveis.
    """
    rng = np.random.default_rng(seed)
    wn = np.linspace(0, 1, p)
    base = np.exp(-((wn - 0.35) ** 2) / (2 * 0.05 ** 2))
    X, grupos = [], []
    for i in range(n_amostras):
        # amostra fisica: perfil proprio
        amostra = base * rng.normal(1.0, 0.15) + rng.normal(0, 0.02, p)
        for _ in range(n_rep):
            X.append(amostra + rng.normal(0, 0.004, p))   # replica tecnica
            grupos.append(f"g{i}")
    return np.asarray(X), np.asarray(grupos, dtype=str)


def escores_f(X_treino, mae_treino, X_alvo, n_comp, alpha):
    """Escores f e limiar chi2 usando a classe DDSimca DE PRODUCAO.

    Deliberadamente NAO reimplementado a mao: o DD-SIMCA do GUARACI ja
    inclui as correcoes desta auditoria (Q_train por leave-one-out,
    calibracao de h0/q0/Nh/Nq por amostra fisica via mae_id). Compara-lo
    contra o conformal usando uma versao caseira SEM essas correcoes
    seria um espantalho -- a comparacao tem que ser com o codigo que roda.
    """
    dd = DDSimca(n_components=n_comp, alpha=alpha, ucl_method="empirical")
    dd.fit(X_treino, np.array(["_c"] * len(X_treino)), mae_id=mae_treino)
    if "_c" not in dd._modelos:
        return None, None, float("nan")
    res = dd.score_matrix(X_alvo)["_c"]
    return np.asarray(res["f"], dtype=float), float(res["f_crit"]), float(
        dd._modelos["_c"]["n_grupos_calibracao"])


N_TESTE = 50


def uma_rodada(n_amostras, n_rep, alpha, p, seed):
    """Orcamento IGUAL de `n_amostras` fisicas para os dois metodos.

    CORRIGIDO em 2026-08-17 (BLOCO A). A primeira versao desta medicao
    dava ao DD-SIMCA um treino FIXO de 10 amostras em todas as linhas e
    variava so' o conjunto de calibracao do conformal -- entao a coluna
    `n_amostras` nao significava a mesma coisa nos dois lados, e a
    conclusao "o DD-SIMCA nao melhora com n" era artefato do protocolo,
    nao propriedade do metodo. Retratado no GATE.

    Protocolo justo: cada metodo recebe as MESMAS `n_amostras` fisicas e
    as gasta como o seu proprio desenho exige --
      DD-SIMCA: ajusta modelo E deriva limiar nas n (e' assim que ele e'
                definido: o limiar vem do treino);
      conformal: parte as n ao meio -- metade ajusta o modelo, metade
                calibra o limiar (a separacao e' o que da' a garantia).
    O conformal, portanto, opera com METADE do n efetivo: e' o preco da
    garantia, e a comparacao tem que mostra-lo, nao esconde-lo.
    """
    total = n_amostras + N_TESTE
    X, g = gera_classe(total, n_rep, p, seed)
    gs = np.unique(g)
    rng = np.random.default_rng(seed + 9999)
    rng.shuffle(gs)
    g_orc = gs[:n_amostras]          # orcamento comum
    g_te = gs[n_amostras:]
    m_te = np.isin(g, list(g_te))

    # --- DD-SIMCA: usa as n inteiras (modelo + limiar) ---
    m_dd = np.isin(g, list(g_orc))
    f_te_dd, f_crit, _n = escores_f(
        X[m_dd], g[m_dd], X[m_te], n_comp=2, alpha=alpha)
    cob_dd = (float("nan") if f_te_dd is None
              else float(np.mean(f_te_dd <= f_crit)))

    # --- Conformal: parte o mesmo orcamento em modelo + calibracao ---
    n_mod = max(1, n_amostras // 2)
    g_mod, g_cal = g_orc[:n_mod], g_orc[n_mod:]
    m_mod = np.isin(g, list(g_mod))
    m_cal = np.isin(g, list(g_cal))
    if m_cal.sum() == 0:
        return cob_dd, float("nan"), False
    f_alvo, _fc, _n2 = escores_f(
        X[m_mod], g[m_mod], X[m_cal | m_te], n_comp=2, alpha=alpha)
    if f_alvo is None:
        return cob_dd, float("nan"), False
    idx_alvo = np.where(m_cal | m_te)[0]
    e_cal = np.isin(idx_alvo, np.where(m_cal)[0])
    f_cal, f_tes = f_alvo[e_cal], f_alvo[~e_cal]

    cc = ConformalOneClass(alpha=alpha).fit(f_cal, mae_id=g[m_cal])
    cob_cf = (float(np.mean(cc.predict(f_tes)))
              if cc.info_["alcancavel"] else float("nan"))
    return cob_dd, cob_cf, cc.info_["alcancavel"]


def main():
    P, N_REP, SEEDS = 400, 3, range(60)
    print("Cobertura empirica vs nominal (60 seeds por celula)")
    print("Escore identico nos dois metodos; muda so' a regra de decisao.\n")
    for alpha in (0.05, 0.10):
        nominal = 1 - alpha
        print(f"### alpha = {alpha:.2f}   (cobertura nominal = {nominal:.2f})")
        print(f"{'n_amostras':>11} {'alpha_min':>10} {'DD-SIMCA':>18} "
              f"{'Conformal':>18} {'conformal definido?':>20}")
        print("-" * 82)
        for n_am in (1, 3, 10, 19, 30, 40, 50, 80):
            dd, cf, ok = [], [], []
            for s in SEEDS:
                a, b, c = uma_rodada(n_am, N_REP, alpha, P, s)
                dd.append(a); cf.append(b); ok.append(c)
            cf_arr = np.array(cf, dtype=float)
            cf_txt = ("nao estimavel" if np.all(np.isnan(cf_arr))
                      else f"{np.nanmean(cf_arr):.3f}")
            frac_ok = 100.0 * float(np.mean(ok))
            dd_m = float(np.mean(dd))
            marca = "  <-- dataset real" if n_am == 1 else ""
            print(f"{n_am:>11} {alpha_alcancavel(n_am):>10.3f} "
                  f"{dd_m:>18.3f} {cf_txt:>18} {frac_ok:>19.0f}%{marca}")
        print()


if __name__ == "__main__":
    main()
