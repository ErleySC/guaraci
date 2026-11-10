"""robpca.py -- ROBPCA (PCA robusta de Hubert, Rousseeuw & Vanden Branden 2005), modulo isolado.

Referencia: M. Hubert, P. J. Rousseeuw, K. Vanden Branden, "ROBPCA: a new approach to robust principal
component analysis", Technometrics 47:64-79, 2005 (volume e paginas conferidos no `man/PcaHubert.Rd` do rrcov; o DOI nao foi
conferido porque o Crossref estava bloqueado, entao nao e' citado aqui).

STATUS DE VALIDACAO (o que foi e o que nao foi conferido):

  * Conferido contra uma referencia independente: este modulo e' uma reimplementacao do algoritmo de
    `rrcov::PcaHubert` (rrcov 1.7-5, robustbase 0.99-2, R 4.3), comparada numero a numero (autovalores,
    cargas a menos do sinal, escores, distancias de escore e ortogonais, cortes, bandeiras e centro) em
    conjuntos pequenos gerados por `tests/oraculos_metodos/gen_oraculo_robpca.R` e gravados em
    `tests/oraculos_metodos/*.json`. O teste e' `tests/test_robpca.py`.
  * Conferencia por reimplementacao NAO e' prova de que o R esteja certo: ela mostra que este codigo
    faz o mesmo que o rrcov nos casos testados. A correcao do metodo em si vem do artigo e do uso do rrcov.
  * Sem referencia independente: dados com p grande e n grande (a busca de direcoes aleatorias usa o mesmo
    gerador congruencial do rrcov, entao deve coincidir, mas so' foi conferida nos tamanhos dos fixtures),
    `alpha = 0.5` (o robustbase usa ali uma tabela simulada que nao foi portada: aqui vale a formula
    ajustada, a diferenca e' de segunda ordem), e o modo `skew=TRUE` do rrcov (nao implementado).
  * Conferencia mais larga, fora do repositorio (2026-10-16): 60 conjuntos aleatorios (n de 15 a 150, p de 3 a
    60, 0 a 20% de contaminacao, k e alpha variados). 56 coincidem com o rrcov (autovalores, distancias
    ortogonais e bandeiras, a 1e-8). Nos 4 restantes o proprio rrcov devolve resultados diferentes para os
    mesmos dados conforme a semente (medido: ex. primeiro autovalor 170.75 ou 164.38 num caso; 36.59 ou 36.38
    em outro), isto e', o MCD final tem otimos quase empatados e nao existe uma resposta unica de referencia.
    Aqui o resultado nesses casos tambem depende da semente (`seed`).
  * O MCD final (`mcd`) e' uma busca por C-steps com muitos pontos de partida, nao o FAST-MCD em C do
    robustbase. Coincide quando o otimo e' unico; nos casos acima os dois sao aproximacoes de um problema
    sem solucao exata barata.

Quirk do rrcov reproduzido de proposito: em `projectMCD` o rrcov compara `crit` (log do determinante) com o
determinante do C-step, e por isso o ramo "covariancia reweighted" nunca e' tomado; aqui tambem nao e'.

Nao ha' ligacao com o pipeline: nada em guaraci importa este modulo.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.special import gammainc
from scipy.stats import chi2, norm

__all__ = ["robpca", "RobpcaResultado", "mcd", "McdResultado", "h_alpha_n"]

_EPS = np.finfo(float).eps


@dataclass
class RobpcaResultado:
    loadings: np.ndarray          # p x k, colunas ortonormais (sinal arbitrario)
    eigenvalues: np.ndarray       # k, variancias robustas ao longo das colunas
    center: np.ndarray            # p, centro robusto na escala original
    scale: np.ndarray             # p, escala usada (1 se nao padronizou)
    scores: np.ndarray            # n x k
    k: int
    h: int
    rank: int
    sd: np.ndarray                # distancia de escore
    od: np.ndarray                # distancia ortogonal
    cutoff_sd: float
    cutoff_od: float              # 0 se k == posto (ortogonal nao se aplica)
    flag: np.ndarray              # True = regular, False = outlier
    eig0: np.ndarray = field(default_factory=lambda: np.zeros(0))  # autovalores do passo 1 (escolha de k)


@dataclass
class McdResultado:
    raw_center: np.ndarray
    raw_cov: np.ndarray           # ja multiplicada pelos fatores de consistencia e de amostra pequena
    center: np.ndarray            # reweighted
    cov: np.ndarray
    best: np.ndarray              # indices (0-based, ordenados) do melhor subconjunto de h pontos
    crit: float                   # log det da covariancia (nao corrigida) do melhor subconjunto
    h: int
    weights: np.ndarray           # pesos 0/1 do reweighting


# ---------------------------------------------------------------- utilidades (rrcov / robustbase)
def h_alpha_n(alpha: float, n: int, p: int) -> int:
    """robustbase::h.alpha.n"""
    n2 = (n + p + 1) // 2
    return int(np.floor(2 * n2 - n + 2 * (n - n2) * alpha))


def _rank_mm(sv: np.ndarray, shape: Tuple[int, int]) -> int:
    tol = max(shape) * _EPS * abs(sv[0])
    return int(np.sum(sv >= tol))


def _signflip(L: np.ndarray) -> np.ndarray:
    L = np.array(L, dtype=float, copy=True)
    for j in range(L.shape[1]):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] = -L[:, j]
    return L


def _class_pc(x: np.ndarray, scale: bool = False, signflip: bool = True):
    """rrcov:::.classPC com center = media."""
    n, p = x.shape
    if n <= 1:
        raise ValueError("o tamanho da amostra deve ser maior que 1")
    center = x.mean(axis=0)
    sc = x.std(axis=0, ddof=1) if scale else np.ones(p)
    xs = (x - center) / sc
    if n > p:
        _, d, vt = np.linalg.svd(xs, full_matrices=False)
        rank = _rank_mm(d, xs.shape)
        loadings = vt.T[:, :rank]
        eig = d[:rank] ** 2 / (n - 1)
    else:
        e, v = np.linalg.eigh(xs @ xs.T)
        o = np.argsort(e)[::-1]
        e, v = e[o], v[:, o]
        tol = n * e.max() * _EPS
        rank = int(np.sum(e > tol))
        e, v = e[:rank], v[:, :rank]
        eig = e / (n - 1)
        loadings = (xs.T @ v) / np.sqrt(e)
    if signflip:
        loadings = _signflip(loadings)
    return rank, eig, loadings, xs @ loadings, center, sc


def _mad(x: np.ndarray) -> float:
    m = np.median(x)
    return 1.4826 * float(np.median(np.abs(x - m)))


def unimcd(y: np.ndarray, quan: int) -> Tuple[float, float, np.ndarray]:
    """rrcov:::unimcd: MCD univariado exato (media, desvio, pesos na ordem original)."""
    y = np.asarray(y, dtype=float)
    ncas = len(y)
    ln = ncas - quan + 1
    if ln == 1:
        return float(y.mean()), float(np.sqrt(y.var(ddof=1))), np.ones(ncas, dtype=bool)
    I = np.argsort(y, kind="stable")
    ys = y[I]
    ay = np.zeros(ln)
    ay[0] = ys[:quan].sum()
    for s in range(1, ln):
        ay[s] = ay[s - 1] - ys[s - 1] + ys[s + quan - 1]
    ay2 = ay ** 2 / quan
    sq = np.zeros(ln)
    sq[0] = (ys[:quan] ** 2).sum() - ay2[0]
    for s in range(1, ln):
        sq[s] = sq[s - 1] - ys[s - 1] ** 2 + ys[s + quan - 1] ** 2 - ay2[s] + ay2[s - 1]
    sqmin = sq.min()
    isq = np.argsort(sq, kind="stable")
    ndup = int(np.sum(sq == sqmin))
    ii = isq[:ndup]
    initmean = ay[ii][(ndup + 1) // 2 - 1] / quan
    initcov = sqmin / (quan - 1)
    res = (ys - initmean) ** 2 / initcov
    factor = np.sort(res)[quan - 1] / chi2.ppf(quan / ncas, 1)
    initcov = factor * initcov
    res = (ys - initmean) ** 2 / initcov
    w = res < chi2.ppf(0.975, 1)
    tm = float((ys * w).sum() / w.sum())
    sm = float(np.sqrt(((ys - tm) ** 2 * w).sum() / (w.sum() - 1)))
    inv = np.argsort(I, kind="stable")
    return tm, sm, w[inv]


def _mcd_cons(p: int, alpha: float) -> float:
    qalpha = chi2.ppf(alpha, p)
    return 1.0 / (gammainc(p / 2.0 + 1.0, qalpha / 2.0) / alpha)


def _cnp2(p: int, n: int, alpha: float, rew: bool) -> float:
    """robustbase:::.MCDcnp2 (rew=False) e .MCDcnp2.rew (rew=True). Sem a tabela simulada de alpha = 0.5."""
    if p > 2:
        if not rew:
            c875 = np.array([[-0.455179464070565, -0.294241208320834],
                             [1.11192541278794, 1.09649329149811], [2.0, 3.0]])
            c500 = np.array([[-1.42764571687802, -1.06141115981725],
                             [1.26263336932151, 1.28907991440387], [2.0, 3.0]])
        else:
            c875 = np.array([[-0.544482443573914, -0.343791072183285],
                             [1.25994483222292, 1.25159004257133], [2.0, 3.0]])
            c500 = np.array([[-1.02842572724793, -0.26800273450853],
                             [1.67659883081926, 1.35968562893582], [2.0, 3.0]])

        def _fp(c):
            y = np.log(-c[0] / p ** c[1])
            A = np.column_stack([np.ones(2), -np.log(c[2] * p ** 2)])
            co = np.linalg.solve(A, y)
            return 1 - np.exp(co[0]) / n ** co[1]
        fp500, fp875 = _fp(c500), _fp(c875)
    elif p == 2:
        if not rew:
            fp500 = 1 - np.exp(0.673292623522027) / n ** 0.691365864961895
            fp875 = 1 - np.exp(0.446537815635445) / n ** 1.06690782995919
        else:
            fp500 = 1 - np.exp(3.11101712909049) / n ** 1.91401056721863
            fp875 = 1 - np.exp(0.79473550581058) / n ** 1.10081930350091
    else:
        if not rew:
            fp500 = 1 - np.exp(0.262024211897096) / n ** 0.604756680630497
            fp875 = 1 - np.exp(-0.351584646688712) / n ** 1.01646567502486
        else:
            fp500 = 1 - np.exp(1.11098143415027) / n ** 1.5182890270453
            fp875 = 1 - np.exp(-0.66046776772861) / n ** 0.88939595831888
    if alpha <= 0.875:
        fp = fp500 + (fp875 - fp500) / 0.375 * (alpha - 0.5)
    else:
        fp = fp875 + (1 - fp875) / 0.125 * (alpha - 0.875)
    return 1.0 / fp


def _mahal2(X: np.ndarray, mu: np.ndarray, S: np.ndarray) -> np.ndarray:
    d = X - mu
    return np.einsum("ij,ij->i", d @ np.linalg.inv(S), d)


def _logdet(S: np.ndarray) -> float:
    s, ld = np.linalg.slogdet(S)
    return ld if s > 0 else -np.inf


def _csteps(X: np.ndarray, idx: np.ndarray, h: int, maxit: int) -> Tuple[np.ndarray, float]:
    cur = np.sort(idx)
    obj = np.inf
    for _ in range(maxit):
        sub = X[cur]
        S = np.atleast_2d(np.cov(sub, rowvar=False))
        ld = _logdet(S)
        if not np.isfinite(ld):
            return cur, -np.inf
        d = _mahal2(X, sub.mean(axis=0), S)
        nxt = np.sort(np.argsort(d, kind="stable")[:h])
        if np.array_equal(nxt, cur) or ld >= obj - 1e-14:
            return cur, ld
        obj, cur = ld, nxt
    return cur, _logdet(np.atleast_2d(np.cov(X[cur], rowvar=False)))


def mcd(X: np.ndarray, alpha: float = 0.5, nsamp: int = 1500, seed: int = 0) -> McdResultado:
    """MCD (Rousseeuw 1984) com correcoes de consistencia e de amostra pequena do robustbase (Pison et al. 2002)
    e reweighting a 97.5%. Busca por C-steps a partir de `nsamp` subconjuntos de p+1 pontos (mais um inicio
    na mediana), nao o FAST-MCD em C. Para p == 1 o MCD e' exato."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 1 and X.shape[1] > 1:
        X = X.T
    n, p = X.shape
    h = h_alpha_n(alpha, n, p)
    if n <= p + 1:
        raise ValueError("n <= p + 1: amostra pequena demais para o MCD")
    if h >= n:
        raise ValueError("alpha = 1 (h = n) nao e' suportado: use a covariancia classica")
    calpha = _mcd_cons(p, h / n)
    corr = _cnp2(p, n, alpha, rew=False)
    if p == 1:
        y = X[:, 0]
        ys = np.sort(y)
        # janela contigua de menor variancia (nao de menor amplitude)
        v = np.array([np.var(ys[i:i + h], ddof=1) for i in range(n - h + 1)])
        i0 = int(np.argmin(v))
        order = np.argsort(y, kind="stable")
        best = np.sort(order[i0:i0 + h])
        mu = np.array([ys[i0:i0 + h].mean()])
        S = np.array([[v[i0]]])
    else:
        rng = np.random.default_rng(seed)
        cand: List[Tuple[float, np.ndarray]] = []
        starts = [rng.choice(n, size=p + 1, replace=False) for _ in range(nsamp)]
        for st in starts:
            sub = X[st]
            S0 = np.atleast_2d(np.cov(sub, rowvar=False))
            m = p + 1
            while not np.isfinite(_logdet(S0)) and m < n:
                m += 1
                extra = rng.choice(np.setdiff1d(np.arange(n), st), size=m - len(st), replace=False) \
                    if m - len(st) > 0 else np.array([], dtype=int)
                S0 = np.atleast_2d(np.cov(X[np.concatenate([st, extra]).astype(int)], rowvar=False))
                st = np.concatenate([st, extra]).astype(int)
            if not np.isfinite(_logdet(S0)):
                continue
            d = _mahal2(X, X[st].mean(axis=0), S0)
            idx = np.argsort(d, kind="stable")[:h]
            idx, ld = _csteps(X, idx, h, maxit=2)
            cand.append((ld, idx))
        cand.sort(key=lambda t: t[0])
        seen, top = set(), []
        for ld, idx in cand:
            key = tuple(idx.tolist())
            if key in seen:
                continue
            seen.add(key)
            top.append(idx)
            if len(top) == 20:
                break
        best_ld, best = np.inf, None
        for idx in top:
            idx, ld = _csteps(X, idx, h, maxit=200)
            if ld < best_ld:
                best_ld, best = ld, idx
        if best is None:
            raise ValueError("MCD singular: os dados estao sobre um hiperplano")
        mu = X[best].mean(axis=0)
        S = np.atleast_2d(np.cov(X[best], rowvar=False))
    crit = _logdet(S)
    raw_S = S * calpha * corr
    if p == 1:
        d = (X[:, 0] - mu[0]) ** 2 / raw_S[0, 0]
    else:
        d = _mahal2(X, mu, raw_S)
    w = d < chi2.ppf(0.975, p)
    sw = int(w.sum())
    Xw = X[w]
    c2 = Xw.mean(axis=0)
    S2 = np.atleast_2d(np.cov(Xw, rowvar=False))
    if sw != n:
        S2 = S2 * _mcd_cons(p, 0.975) * _cnp2(p, n, alpha, rew=True)
    return McdResultado(raw_center=mu, raw_cov=raw_S, center=c2, cov=S2, best=np.sort(best), crit=float(crit),
                        h=h, weights=w)


# ---------------------------------------------------------------- ROBPCA
def _uniran(seed: int) -> Tuple[int, float]:
    seed = int(np.floor(seed * 5761)) + 999
    quot = seed // 65536
    seed = seed - quot * 65536
    return seed, seed / 65536.0


def _direcoes(X: np.ndarray, ndir: int, todas: bool) -> np.ndarray:
    n = X.shape[0]
    if todas:
        cc = np.array(list(itertools.combinations(range(n), 2)))
        return X[cc[:, 0]] - X[cc[:, 1]]
    B = np.zeros((ndir, X.shape[1]))
    seed = 0
    for r in range(ndir):
        ran: List[int] = []
        for j in range(2):
            seed, u = _uniran(seed)
            num = int(np.floor(u * n)) + 1
            if j > 0:
                while num in ran:
                    seed, u = _uniran(seed)
                    num = int(np.floor(u * n)) + 1
            ran.append(num)
        B[r] = X[ran[0] - 1] - X[ran[1] - 1]
    return B


def _outl(X: np.ndarray, maxdir: int, h: int) -> np.ndarray:
    n = X.shape[0]
    alldir = n * (n - 1) // 2
    ndir = min(maxdir, alldir)
    B = _direcoes(X, ndir, ndir == alldir)
    nb = np.linalg.norm(B, axis=1)
    ok = nb > 1e-12
    A = B[ok] / nb[ok, None]
    Y = X @ A.T
    Z = np.zeros((n, A.shape[0]))
    for i in range(A.shape[0]):
        tm, sm, w = unimcd(Y[:, i], h)
        if sm < 1e-12:
            sv = np.linalg.svd(X[w], compute_uv=False)
            if _rank_mm(sv, X[w].shape) == 1:
                raise ValueError("pelo menos %d observacoes sao identicas" % int(w.sum()))
        else:
            Z[:, i] = np.abs(Y[:, i] - tm) / sm
    return Z.max(axis=1)


def _crit_od(od: np.ndarray, crit: float, metodo: str, quan: Optional[int] = None) -> float:
    o = od ** (2.0 / 3.0)
    if metodo == "umcd":
        t, s, _ = unimcd(o, quan)
    elif metodo == "medmad":
        t, s = float(np.median(o)), _mad(o)
    else:
        raise ValueError(metodo)
    return float((t + s * norm.ppf(crit)) ** 1.5)


def robpca(X, k: int = 0, kmax: int = 10, alpha: Optional[float] = None, mcd_se_p_pequeno: bool = True,
           maxdir: int = 250, scale: bool = False, crit: float = 0.975, seed: int = 0) -> RobpcaResultado:
    """ROBPCA como em `rrcov::PcaHubert(x, k, kmax, alpha, mcd, maxdir, scale, crit.pca.distances)`.

    `alpha=None` reproduz o padrao do rrcov (alpha = 0.75 "ausente": h = h.alpha.n(0.75, n, kmax) e piso
    (n+k+1)/2). Com `alpha` informado vale 0.5 <= alpha <= 1. `mcd_se_p_pequeno` e' o argumento `mcd` do rrcov
    (usa MCD direto quando o posto reduzido e' <= min(n/5, kmax)). `seed` so' afeta a busca do MCD."""
    data = np.asarray(X, dtype=float)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("X deve ser uma matriz numerica sem NaN/Inf")
    n, p = data.shape
    rank, eig_x, load_x, Xs, center0, myscale = _class_pc(data, scale=scale, signflip=True)
    if rank == 0:
        raise ValueError("todos os pontos colapsam")
    kmax = max(min(int(np.floor(kmax)), rank), 1)
    k = int(np.floor(k))
    if k < 0:
        k = 0
    elif k > kmax:
        k = kmax
    if alpha is None:
        alpha = 0.75
        h = min(h_alpha_n(alpha, n, kmax), n)
        a_ef = h / n
        piso = (n + (kmax if k == 0 else k) + 1) // 2
        if h < piso:
            h = piso
            a_ef = h / n
        if h > n:
            h = h_alpha_n(alpha, n, kmax if k == 0 else k)
            a_ef = alpha
        alpha = a_ef
    else:
        if alpha < 0.5 or alpha > 1:
            raise ValueError("alpha fora do intervalo: entre 1/2 e 1")
        h = h_alpha_n(alpha, n, kmax if k == 0 else k)

    Xr = Xs
    center = center0.copy()
    rot = load_x
    eig0 = np.zeros(0)

    def escolher_k(ev, kmax_, rank_):
        teste = np.where(ev / ev[0] <= 1e-3)[0]
        kk = min(min(rank_, int(teste[0]) + 1), kmax_) if len(teste) else min(rank_, kmax_)
        cum = np.cumsum(ev[:kk]) / ev.sum()
        if cum[kk - 1] > 0.8:
            kk = int(np.where(cum >= 0.8)[0][0]) + 1
        return kk

    if Xr.shape[1] <= min(n // 5, kmax) and mcd_se_p_pequeno:
        m = mcd(Xr, alpha=alpha, seed=seed)
        ev_all, U = np.linalg.eigh(m.cov)
        o = np.argsort(ev_all)[::-1]
        ev_all, U = ev_all[o], U[:, o]
        rk = Xr.shape[1]
        eig0 = ev_all
        if k != 0:
            k = min(k, Xr.shape[1])
        else:
            k = escolher_k(ev_all, kmax, rk)
        scores = (Xr - m.center) @ U
        center = center0 + (m.center @ rot.T) * myscale
        eigenvalues = ev_all[:k]
        loadings = rot @ U[:, :k]
        scores = scores[:, :k]
        h_out = m.h
    else:
        outl = _outl(Xr, maxdir, h)
        H0 = np.argsort(outl, kind="stable")
        Xh = Xr[H0[:h]]
        r0, e0, L0, _, c0, _ = _class_pc(Xh)
        kmax = min(r0, kmax)
        eig0 = e0
        if k == 0:
            k = escolher_k(e0, kmax, r0)
        if k != rank:
            k = min(r0, k)
            XRc = Xr - c0
            Xt = XRc @ L0[:, :k] @ L0[:, :k].T
            odh = np.linalg.norm(XRc - Xt, axis=1)
            cut = _crit_od(odh, 0.975, "umcd", h)
            idxset = odh <= cut
            r0, e0, L0, _, c0, _ = _class_pc(Xr[idxset])
            k = min(r0, k)
        center = center + (c0 @ rot.T) * myscale
        rot = rot @ L0
        X2 = (Xr - c0) @ L0
        X2 = X2[:, :k]
        rot = rot[:, :k]
        # projectMCD
        ev = e0
        mah = np.einsum("ij,j,ij->i", X2, 1.0 / ev[:k], X2)
        oldobj = float(np.prod(ev[:k]))
        obj = oldobj
        kk = k
        for _ in range(100):
            Xh2 = X2[np.argsort(mah, kind="stable")[:h]]
            r1, e1, L1, _, c1, _ = _class_pc(Xh2)
            obj = float(np.prod(e1))
            X2 = (X2 - c1) @ L1
            center = center + (c1 @ rot.T) * myscale
            rot = rot @ L1
            mah = np.einsum("ij,j,ij->i", X2, 1.0 / e1, X2)
            if r1 == kk and abs(oldobj - obj) < 1e-12:
                break
            oldobj = obj
            if r1 < kk:
                kk = r1
        k = X2.shape[1]
        m = mcd(X2, alpha=alpha, seed=seed)
        ev2, P6 = np.linalg.eigh(m.cov)
        o = np.argsort(ev2)[::-1]
        ev2, P6 = ev2[o], P6[:, o]
        center = center + (m.center @ rot.T) * myscale
        eigenvalues = ev2
        loadings = rot @ P6
        scores = (X2 - m.center) @ P6
        h_out = h
        k = scores.shape[1]

    # distancias (rrcov:::.distances)
    sdist = np.sqrt(np.einsum("ij,j,ij->i", scores, 1.0 / eigenvalues, scores))
    cutoff_sd = float(np.sqrt(chi2.ppf(crit, k)))
    od = np.linalg.norm((data - center) / myscale - scores @ loadings.T, axis=1)
    cutoff_od = 0.0
    if k != rank:
        cutoff_od = _crit_od(od, crit, "medmad")
    flag = sdist <= cutoff_sd
    if cutoff_od > 0:
        flag = flag & (od <= cutoff_od)
    return RobpcaResultado(loadings=loadings, eigenvalues=np.asarray(eigenvalues), center=center, scale=myscale,
                           scores=scores, k=k, h=h_out, rank=rank, sd=sdist, od=od, cutoff_sd=cutoff_sd,
                           cutoff_od=cutoff_od, flag=flag, eig0=np.asarray(eig0))
