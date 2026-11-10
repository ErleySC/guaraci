"""pcv_regressao.py -- Procrustes cross-validation (PCV) para regressao PLS, modulo isolado.

Referencias:
  * S. Kucheryavskiy, O. Rodionova, A. Pomerantsev, "Procrustes cross-validation of multivariate regression
    models", Anal. Chim. Acta 1255 (2023), https://doi.org/10.1016/j.aca.2023.341096 (metodo de regressao;
    titulo, autores e DOI conferidos no `man/pcvpls.Rd` do pacote R `pcv`; o DOI redireciona para a Elsevier,
    mas a pagina do artigo nao pode ser lida aqui).
  * Ideia original, para PCA: Anal. Chem. 92(17):11842-11850 (2020)
    (referencia do `NEWS.md` do mdatools; DOI nao conferido no Crossref, por isso nao e' citado).

O que faz: para cada segmento de validacao cruzada, ajusta um PLS local, alinha-o ao PLS global (sinais e
escala dos componentes) e transforma os pontos do segmento em pontos "pseudo-validacao" que se comportam como
um conjunto de validacao independente. O conjunto completo (`Xpv`) serve para predizer `Y` com o modelo
global e calcular um RMSEP-PV por numero de componentes (`rmsep_pv`), de onde sai uma escolha de componentes
sem sobrar amostras para um conjunto de teste.

STATUS DE VALIDACAO:
  * Referencia independente: pacote R `pcv` 1.2.0 (espelho github.com/cran/pcv; os arquivos R foram lidos
    com `source()`, sem instalar). `tests/oraculos_metodos/gen_oraculo_pcv.R` gera, para 7 conjuntos, a parte
    explicada do pseudo-conjunto, as matrizes D, as distancias ortogonais q_k, R, R_k e o RMSEP-PV. O teste
    `tests/test_pcv_regressao.py` exige igualdade numerica (1e-8 a 1e-10), a menos do sinal global dos componentes.
  * A parte ORTOGONAL do pseudo-conjunto usa numeros aleatorios, tanto no pacote quanto aqui. Ela nao pode
    coincidir entre implementacoes; o que se confere e' a sua definicao (cada linha tem norma^2 = q_k e e'
    ortogonal ao espaco de R) e que o RMSEP-PV nao depende dela (o gerador do fixture afirma isso no R com
    duas sementes; o teste afirma o mesmo aqui).
  * Alem da reproducao do pacote, ha' um teste com verdade conhecida (dados simulados com conjunto de teste
    independente grande): ele mostra o comportamento do RMSEP-PV em UM cenario, nao e' garantia geral. Como a
    propria literatura do metodo adverte, o PCV nao cria variacao que os dados nao tem: com replicas da mesma
    amostra fisica ou lotes confundidos ele repete o otimismo da validacao cruzada comum.
  * Nao implementado: PCR (`pcvpcr` do pacote), PCA (`pcvpca`, usado pelo DD-SIMCA no Guaraci via `prcv`),
    e a escolha por regra de um desvio-padrao.

Nao ha' ligacao com o pipeline: nada em guaraci importa este modulo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["pcv_pls", "PcvPlsResultado", "rmsep_pv", "escolher_componentes", "simpls"]

_EPS100 = 100 * np.finfo(float).eps

Cv = Union[Sequence, np.ndarray]


@dataclass
class PcvPlsResultado:
    Xpv: np.ndarray               # n x p, na escala original
    Xpv_explicada: np.ndarray     # parte explicada (deterministica), escala original menos a parte ortogonal
    D: np.ndarray                 # nseg x ncomp, diagonais das matrizes D
    q: np.ndarray                 # n, distancia ortogonal^2 de cada ponto ao modelo local do seu segmento
    R: np.ndarray                 # p x ncomp, pesos do modelo global (sinal arbitrario)
    Rk: np.ndarray                # nseg x p x ncomp, pesos locais alinhados ao global
    C: np.ndarray                 # nresp x ncomp, cargas de Y do modelo global
    ncomp: int                    # numero de componentes efetivamente usado
    segmentos: np.ndarray         # n, 1..nseg
    centro_x: np.ndarray
    escala_x: np.ndarray
    centro_y: np.ndarray
    escala_y: np.ndarray


def simpls(X: np.ndarray, Y: np.ndarray, ncomp: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SIMPLS como no pacote pcv (R, P, C). Componentes que degeneram ficam com colunas zero."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    nobj, npred = X.shape
    nresp = Y.shape[1]
    S = X.T @ Y
    C = np.zeros((nresp, ncomp))
    R = np.zeros((npred, ncomp))
    V = np.zeros((npred, ncomp))
    P = np.zeros((npred, ncomp))
    TT = np.zeros((nobj, ncomp))
    for a in range(ncomp):
        r = np.linalg.svd(S, full_matrices=False)[0][:, 0]
        t = X @ r
        tn = np.sqrt(t @ t)
        if tn < _EPS100:
            break
        t = t / tn
        r = r / tn
        p = X.T @ t
        c = Y.T @ t
        v = p.copy()
        if a > 0:
            v = v - V[:, :a] @ (V[:, :a].T @ p)
        vn = np.sqrt(v @ v)
        if vn < _EPS100:
            break
        v = v / vn
        R[:, a], V[:, a], P[:, a], TT[:, a], C[:, a] = r, v, p, t, c
        S = S - np.outer(v, v @ S)
    return R, P, C


def _normalize_cols(M: np.ndarray) -> np.ndarray:
    n = np.sqrt((M ** 2).sum(axis=0))
    n[n < _EPS100] = 1.0
    return M / n


def _flip_signs(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    cos = np.clip((V1 * V2).sum(axis=0), -1.0, 1.0)
    return np.where(np.arccos(cos) < np.pi / 2, 1.0, -1.0)


def _segmentos(cv: Cv, n: int, y1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """pcv:::pcvcrossval. `cv` = ("ven", nseg) | ("loo",) | ("rand", nseg) | vetor de segmentos 1..nseg."""
    if not isinstance(cv, (tuple, list)) or (len(cv) == n and not isinstance(cv[0], str)):
        v = np.asarray(cv)
        if v.shape == (n,):
            return v.astype(int)
        raise ValueError("cv numerico deve ter um segmento por amostra")
    tipo = cv[0]
    if tipo == "loo":
        return np.arange(1, n + 1)
    nseg = int(cv[1])
    if nseg < 2 or nseg > n:
        raise ValueError("numero de segmentos deve estar entre 2 e o numero de amostras")
    if tipo == "ven":
        # rank(resp, ties.method = "first")
        rk = np.empty(n, dtype=int)
        rk[np.argsort(y1, kind="stable")] = np.arange(1, n + 1)
        return (np.resize(np.arange(1, nseg + 1), n))[rk - 1]
    if tipo == "rand":
        return np.resize(np.arange(1, nseg + 1), n)[rng.permutation(n)]
    raise ValueError("cv desconhecido: %r" % (tipo,))


def _orth(q: np.ndarray, Xk: np.ndarray, Pi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    m = len(q)
    Xo = rng.uniform(-1.0, 1.0, size=(m, m)) @ Xk
    Xo = _normalize_cols(Xo)
    Xo = Xo @ Pi
    n2 = (Xo ** 2).sum(axis=1)
    n2[n2 < _EPS100] = 1.0
    return np.sqrt(q / n2)[:, None] * Xo


def pcv_pls(X, Y, ncomp: Optional[int] = None, cv: Cv = ("ven", 4), center: bool = True, scale: bool = False,
            cv_scope: str = "global", seed: int = 0) -> PcvPlsResultado:
    """Pseudo-conjunto de validacao para PLS (`pcv::pcvpls`). `seed` so' afeta a parte ortogonal e o `rand`."""
    if cv_scope not in ("global", "local"):
        raise ValueError("cv_scope deve ser 'global' ou 'local'")
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("X e Y nao podem ter NaN/Inf")
    n, p = X.shape
    if Y.shape[0] != n:
        raise ValueError("X e Y devem ter o mesmo numero de linhas")
    nresp = Y.shape[1]
    rng = np.random.default_rng(seed)
    ind = _segmentos(cv, n, Y[:, 0], rng)
    nseg = int(ind.max())
    if ncomp is None:
        ncomp = min(n - 1, p, 30)
    maxc = min(n - int(np.ceil(n / nseg)) - 1, p)
    if maxc < 1:
        raise ValueError("amostras insuficientes para esta validacao cruzada")
    ncomp = min(int(ncomp), maxc)

    mX = X.mean(axis=0) if center else np.zeros(p)
    mY = Y.mean(axis=0) if center else np.zeros(nresp)
    sX, sY = np.ones(p), np.ones(nresp)
    if scale:
        sX = X.std(axis=0, ddof=1)
        sY = Y.std(axis=0, ddof=1)
        sX[sX < _EPS100] = 1.0
        sY[sY < _EPS100] = 1.0
    Xs = (X - mX) / sX
    Ys = (Y - mY) / sY
    R, P, C = simpls(Xs, Ys, ncomp)
    Pi = np.eye(p) - R @ P.T

    Xpv_hat = np.zeros((n, p))
    Xpv_orth = np.zeros((n, p))
    D = np.zeros((nseg, ncomp))
    Rk_all = np.zeros((nseg, p, ncomp))
    q_all = np.zeros(n)
    for k in range(1, nseg + 1):
        ik = ind == k
        ic = ~ik
        Xk, Xc, Yc = Xs[ik], Xs[ic], Ys[ic]
        if cv_scope == "local":
            mXl = Xc.mean(axis=0) if center else np.zeros(p)
            mYl = Yc.mean(axis=0) if center else np.zeros(nresp)
            sXl, sYl = np.ones(p), np.ones(nresp)
            if scale:
                sXl = Xc.std(axis=0, ddof=1)
                sYl = Yc.std(axis=0, ddof=1)
                sXl[sXl < _EPS100] = 1.0
                sYl[sYl < _EPS100] = 1.0
            Xc = (Xc - mXl) / sXl
            Yc = (Yc - mYl) / sYl
            Xk = (Xk - mXl) / sXl
        Rl, Pl, Cl = simpls(Xc, Yc, ncomp)
        S = np.diag(_flip_signs(_normalize_cols(R), _normalize_cols(Rl)))
        Rl, Pl, Cl = Rl @ S, Pl @ S, Cl @ S
        Dk = np.eye(ncomp)
        for a in range(ncomp):
            den = float(C[:, a] @ C[:, a])
            Dk[a, a] = 0.0 if den < 100 * np.finfo(float).eps else float(Cl[:, a] @ C[:, a]) / den
        Tk = Xk @ Rl
        Xpv_hat[ik] = (Tk @ Dk) @ P.T
        D[k - 1] = np.diag(Dk)
        Rk_all[k - 1] = Rl
        q = ((Xk - Tk @ Pl.T) ** 2).sum(axis=1)
        q_all[ik] = q
        Xpv_orth[ik] = _orth(q, Xk, Pi, rng)
    Xpv = (Xpv_hat + Xpv_orth) * sX + mX
    Xhat = Xpv_hat * sX + mX
    return PcvPlsResultado(Xpv=Xpv, Xpv_explicada=Xhat, D=D, q=q_all, R=R, Rk=Rk_all, C=C, ncomp=ncomp,
                           segmentos=ind, centro_x=mX, escala_x=sX, centro_y=mY, escala_y=sY)


def rmsep_pv(res: PcvPlsResultado, Y, Xpv: Optional[np.ndarray] = None) -> np.ndarray:
    """RMSEP do modelo PLS global (n_comp = 1..res.ncomp) ao predizer `Y` a partir do pseudo-conjunto.
    Forma (ncomp, nresp). Nao depende da parte ortogonal do pseudo-conjunto."""
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    Xp = res.Xpv if Xpv is None else Xpv
    Z = (Xp - res.centro_x) / res.escala_x
    out = np.zeros((res.ncomp, Y.shape[1]))
    for a in range(1, res.ncomp + 1):
        Yh = (Z @ res.R[:, :a] @ res.C[:, :a].T) * res.escala_y + res.centro_y
        out[a - 1] = np.sqrt(((Y - Yh) ** 2).mean(axis=0))
    return out


def escolher_componentes(res: PcvPlsResultado, Y, resposta: int = 0) -> Tuple[int, np.ndarray]:
    """Numero de componentes com menor RMSEP-PV para a resposta `resposta` (1-based no retorno) e a curva."""
    curva = rmsep_pv(res, Y)[:, resposta]
    return int(np.argmin(curva)) + 1, curva
