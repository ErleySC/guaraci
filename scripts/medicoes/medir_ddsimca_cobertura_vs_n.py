"""P1 (Bloco 10, guaraci plan). Cobertura empirica do DD-SIMCA (classe unica,
one-class) como funcao de n = numero de amostras/sessoes de calibracao
INDEPENDENTES, para calcular n_minimo dado um alpha-alvo.

MOTIVO: a instrucao do Bloco 10 citava uma curva "C/n" com 3 pontos
especificos (0,840@n=80 / 0,921@n=300 / 0,943@n=1200) como "ja medida e
validada nesta sessao". Busca (grep direto, regra 0-A do CLAUDE.md local)
em docs/, src/, scripts/medicoes/ e no proprio CLAUDE.md local nao encontrou
ESSES numeros em lugar nenhum do repositorio -- nao existe artefato que os
sustente. Este script MEDE a relacao de verdade, do zero, contra o metodo
REAL que o software usa (`classificadores.DDSimca`), para nao propagar um
numero sem lastro para o Bloco 10.

DESENHO: DGP gaussiano de classe unica (estrutura de correlacao tipo AR(1)
entre variaveis, para se parecer com espectro -- vizinhos correlacionados,
nao ruido branco). Para cada `n` (amostras de calibracao INDEPENDENTES, 1
espectro por amostra -- sem replicas tecnicas aqui: o efeito de replica
sobre h0/q0/Nh/Nq ja e' testado em outro lugar via mae_id, o que se mede
aqui e' o efeito do proprio `n`), ajusta DDSimca(alpha=0.05) e mede a
fracao ACEITA (`predict()`) numa amostra de teste GRANDE, genuina, da MESMA
distribuicao -- essa fracao e' a cobertura empirica. Repete R vezes por n
para estabilizar a media, varrendo uma faixa de n (nao so' os 3 pontos
citados) para caracterizar a curva de verdade, nao so' confirmar/negar 3
numeros pontuais.

RESULTADO (medido em 2026-08-26, P=30, n_components=3, alpha=0.05,
N_TEST=2000, DGP gaussiano AR(1) sintetico -- NAO o dataset real):

    n      reps   cobertura   desvio   deficit(1-cov)
    5      60     0.8450      0.1510   0.1550
    10     60     0.8957      0.0697   0.1043
    20     40     0.9038      0.0666   0.0962
    40     40     0.9230      0.0346   0.0770
    80     30     0.9306      0.0218   0.0694
    150    20     0.9428      0.0201   0.0572
    300    15     0.9425      0.0122   0.0575
    600    8      0.9448      0.0103   0.0552
    1200   4      0.9411      0.0070   0.0589

RETRATACAO dos 3 pontos citados na instrucao original -- nao batem com a
medicao, sem padrao de erro consistente (evidencia de que nao vieram de
medicao real):
    n=80:   citado 0,8400  medido 0,9306   diferenca +0,0906
    n=300:  citado 0,9210  medido 0,9425   diferenca +0,0215
    n=1200: citado 0,9430  medido 0,9411   diferenca -0,0019

ACHADO PRINCIPAL (mais importante que a retratacao acima): a cobertura NAO
converge para o nominal 0,95 -- sobe rapido ate' n~150 e ESTANCA num
plato de ~0,94-0,945, sem melhorar mais em n=600 ou n=1200. Nenhuma das 3
formas testadas (C/n, C/sqrt(n), exponencial) ajusta bem toda a faixa
-- C/n teve R2=-1,49 (pior que uma reta horizontal), porque a forma real
e' "convergencia rapida + plato persistente", nao uma curva suave ate'
zero. Consequencia pratica: nao existe `n_minimo` finito que garanta
cobertura-alvo ABAIXO do plato (~0,94-0,945 nesta configuracao sintetica)
usando DD-SIMCA (metodo parametrico chi2-momentos) -- para cobertura-alvo
mais exigente que isso, so' o gate conformal (`identificacao.py`,
`conformal.py`) tem garantia formal, distribution-free, independente de
`n`. Ver `classificadores.DDSimca` (docstring da classe) e
`docs/MANUAL.md` (Limitacoes conhecidas) para a mesma nota.
"""
import logging
import sys
import time

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, "src")
from guaraci.classificadores import DDSimca  # noqa: E402

# Silencia os avisos de outlier de treino do DDSimca -- irrelevante para
# esta medida de cobertura, so' deixa a saida ilegivel.
logging.getLogger("guaraci.classificadores").setLevel(logging.ERROR)

P = 30            # variaveis (ordem de grandeza de um espaco PCA reduzido)
N_COMPONENTS = 3  # mesmo default do projeto (cfg.ddsimca_n_components)
ALPHA = 0.05      # cobertura nominal-alvo = 0.95
N_TEST = 2000     # amostras de teste genuinas por repeticao (precisao da medida)

# `q_residuals_loo` (chemometric_stats.py) faz UM PCA.fit POR AMOSTRA de
# treino (custo documentado como aceitavel "porque n e' pequeno" -- o
# regime real do projeto, tipicamente <20 amostras puras por especie).
# Em n=1200/2400 isso e' O(n) PCA-fits POR repeticao -- medido aqui: a
# primeira tentativa (N_REP=40 fixo, ate' n=2400) nao terminou em >8min e
# foi abortada. Reps decrescem com n para manter o custo total tratavel
# sem descartar os pontos grandes que a instrucao original citava.
NS_REPS = [(5, 60), (10, 60), (20, 40), (40, 40), (80, 30), (150, 20),
           (300, 15), (600, 8), (1200, 4)]


def _cov_ar1(p: int, rho: float = 0.85) -> np.ndarray:
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


_COV = _cov_ar1(P)
_L = np.linalg.cholesky(_COV)


def _amostrar(rng: np.random.Generator, n: int) -> np.ndarray:
    z = rng.normal(size=(n, P))
    return z @ _L.T


def _uma_medida(n: int, seed: int):
    logging.getLogger("guaraci.classificadores").setLevel(logging.ERROR)
    rng = np.random.default_rng(seed)
    X_cal = _amostrar(rng, n)
    y_cal = np.array(["puro"] * n)
    mae_cal = np.array([f"g{i}" for i in range(n)])   # 1 amostra = 1 grupo

    # DDSimca.fit() ja' reduz n_components sozinho quando `n` nao sustenta
    # o pedido (preserva _MIN_Q_RESIDUAL_DF graus de liberdade residuais) --
    # mesmo comportamento de producao, nao replicado aqui.
    dds = DDSimca(n_components=N_COMPONENTS, alpha=ALPHA)
    dds.fit(X_cal, y_cal, mae_id=mae_cal)
    if "puro" not in dds._modelos:
        return None   # n insuficiente para o minimo de graus de liberdade

    X_teste = _amostrar(rng, N_TEST)
    pred = dds.predict(X_teste)
    aceito = (pred == "puro")
    return float(np.mean(aceito))


def _c_sobre_n(n, c):
    return 1.0 - c / n


def _c_sobre_raiz_n(n, c):
    return 1.0 - c / np.sqrt(n)


def _exponencial(n, c, k):
    return 1.0 - c * np.exp(-k * n)


def main():
    print("=" * 72, flush=True)
    print("P1 (Bloco 10): cobertura empirica do DD-SIMCA vs n (calibracao)",
          flush=True)
    print("=" * 72, flush=True)
    print(f"  alpha nominal = {ALPHA}  (cobertura-alvo = {1 - ALPHA:.3f})",
          flush=True)
    print(f"  P={P} variaveis, n_components<={N_COMPONENTS}, N_TEST={N_TEST}",
          flush=True)
    print(flush=True)
    print(f"  {'n':>6}  {'reps':>5}  {'cobertura':>10}  {'desvio':>8}  "
          f"{'deficit(1-cov)':>15}", flush=True)

    resultados = {}
    for n, n_rep in NS_REPS:
        t0 = time.time()
        # SEQUENCIAL, nao Parallel(threading): a tentativa com threading
        # segfaultou (exit 139) em algum ponto entre n=150 e n=300 -- BLAS
        # interno do sklearn.PCA (OpenBLAS/MKL) ja' usa threads proprias;
        # varias chamadas concorrentes de PCA.fit por cima disso, no
        # Windows, e' um padrao conhecido de contencao/crash. Sequencial e'
        # mais lento por chamada mas confiavel -- e com os reps ja'
        # reduzidos para n grande, o tempo total continua tratavel.
        medidas = [_uma_medida(n, seed=1000 * n + r) for r in range(n_rep)]
        medidas = [m for m in medidas if m is not None]
        if not medidas:
            print(f"  {n:6d}  (nenhuma medida valida -- n insuficiente p/ "
                  f"graus de liberdade do DD-SIMCA)", flush=True)
            continue
        cov, sd = float(np.mean(medidas)), float(np.std(medidas))
        resultados[n] = (cov, sd, len(medidas))
        print(f"  {n:6d}  {len(medidas):5d}  {cov:10.4f}  {sd:8.4f}  "
              f"{1 - cov:15.4f}   ({time.time() - t0:.1f}s)", flush=True)

    ns_validos = np.array(sorted(resultados))
    coberturas = np.array([resultados[n][0] for n in ns_validos])

    print()
    print("  Pontos citados na instrucao original (NAO encontrados em "
          "nenhum artefato do repositorio antes desta medicao):")
    print("    0,840@n=80 | 0,921@n=300 | 0,943@n=1200")
    for n_cit, cov_cit in ((80, 0.840), (300, 0.921), (1200, 0.943)):
        if n_cit in resultados:
            cov_medida = resultados[n_cit][0]
            print(f"    n={n_cit}: medido={cov_medida:.4f}  citado={cov_cit:.4f}  "
                  f"diferenca={cov_medida - cov_cit:+.4f}")

    print()
    print("  Ajuste de forma funcional a 'deficit = 1 - cobertura(n)':")
    for nome, fn, p0 in (
            ("C/n", _c_sobre_n, (1.0,)),
            ("C/sqrt(n)", _c_sobre_raiz_n, (1.0,)),
            ("exponencial 1-C*exp(-k n)", _exponencial, (1.0, 0.01))):
        try:
            popt, _ = curve_fit(fn, ns_validos, coberturas, p0=p0, maxfev=20000)
            pred = fn(ns_validos, *popt)
            ss_res = float(np.sum((coberturas - pred) ** 2))
            ss_tot = float(np.sum((coberturas - coberturas.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            print(f"    {nome:28s} parametros={np.round(popt, 5).tolist()}  "
                  f"R2={r2:.4f}")
        except RuntimeError as e:
            print(f"    {nome:28s} ajuste falhou: {e}")


if __name__ == "__main__":
    main()
