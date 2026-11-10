"""
figuras_extras.py -- graficos quimiometricos que faltavam (auditoria de
figuras, secao D): scree da PCA, vetor de coeficientes b do PLS, grafico de
Williams, predito x medido com faixa +-RMSEP, perfis C/S do MCR-ALS, loadings
do PARAFAC por modo, efeitos do ASCA, mapa EEM medio por classe e
transferencia de calibracao (mestre x escravo antes/depois de DS/PDS).

So' DESENHA: nenhum numero publicado e' recalculado de outro jeito. Quando
uma figura precisa de uma grandeza derivada (alavancagem, residuo
padronizado, loadings do efeito do ASCA, correcao deixando cada amostra de
transferencia de fora) ela vem da mesma formula/funcao que o motor usa, e os
testes em tests/test_figuras_extras.py conferem essa igualdade.

Toda figura passa por `figuras.save()` (dpi/formato do cfg e subpasta de
grupo quando `figuras.ORGANIZAR_POR_GRUPO` esta ligado; ver
grupos_figuras.py).
"""
from __future__ import annotations

import glob
import os

from guaraci.config import nome_pasta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.layout_engine import ConstrainedLayoutEngine

from guaraci.eixo_espectral import ROTULO_EIXO_NEUTRO, rotulo_do_eixo
from guaraci.paleta_cores import color, get_edge_color, map_class_colors
from guaraci.figuras import GRAFICOS_POR_ITEM, nome_pagina, paginas

__all__ = [
    "variancia_acumulada",
    "limiar_williams",
    "alavancagem_scores",
    "residuos_padronizados",
    "dados_calibracao_pls",
    "grade_eem_da_pasta",
    "rotulo_eixo_cromatograma",
    "fig_pca_scree",
    "fig_predito_medido_faixa",
    "fig_coeficientes_regressao",
    "fig_williams",
    "figuras_regressao_do_modelo",
    "fig_mcr_als",
    "fig_parafac_eem",
    "fig_asca",
    "fig_eem_media_classe",
    "fig_transferencia_calibracao",
    "figuras_transferencia_de_argv",
]


# =========================================================================
#  Helpers numericos (testados em tests/test_figuras_extras.py)
# =========================================================================

def variancia_acumulada(var_pct: Sequence[float]) -> np.ndarray:
    """Soma acumulada da variancia explicada (%) das componentes calculadas."""
    return np.cumsum(np.asarray(var_pct, dtype=float))


def limiar_williams(n_lv: int, n: int) -> float:
    """Alavancagem de alerta do grafico de Williams: h* = 3(A+1)/n."""
    if n <= 0:
        raise ValueError("n precisa ser > 0")
    return 3.0 * (int(n_lv) + 1) / float(n)


def alavancagem_scores(T: np.ndarray, T_ref: Optional[np.ndarray] = None) -> np.ndarray:
    """h_i = 1/n + t_i (T'T)^-1 t_i' (modelo centrado na media).

    `T_ref` sao os scores da CALIBRACAO (que definem T'T e n); sem ele, as
    proprias linhas de `T`. Serve para amostras novas projetadas no modelo."""
    T = np.atleast_2d(np.asarray(T, dtype=float))
    R = T if T_ref is None else np.atleast_2d(np.asarray(T_ref, dtype=float))
    n = R.shape[0]
    G = np.linalg.pinv(R.T @ R)
    return 1.0 / n + np.einsum("ij,jk,ik->i", T, G, T)


def residuos_padronizados(y: np.ndarray, y_hat: np.ndarray, n_lv: int,
                          s: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """(residuo/s, s), residuo = predito - referencia (mesmo sinal do Bias da
    figS2) e s = sqrt(SSE/(n-A-1)) da CALIBRACAO (ou o `s` dado, para
    padronizar validacao/holdout com o desvio da calibracao)."""
    y = np.asarray(y, dtype=float).ravel()
    e = np.asarray(y_hat, dtype=float).ravel() - y
    if s is None:
        gl = len(e) - int(n_lv) - 1
        s = float(np.sqrt(np.sum(e ** 2) / (gl if gl > 0 else len(e))))
    s = float(s)
    return (e / s if s > 0 else np.zeros_like(e)), s


def dados_calibracao_pls(pls) -> Dict[str, np.ndarray]:
    """y, y_hat e scores T da CALIBRACAO de um `PLSRegression` ja ajustado
    (resposta unica), recuperados do proprio modelo salvo -- sem precisar dos
    espectros de calibracao. y_hat = T q' (+ media) e' exatamente
    `pls.predict(X_cal)`; com uma unica resposta o NIPALS faz u = y c / (c c),
    entao y = u c (+ media). Conferido contra o ajuste direto nos testes."""
    T = np.asarray(pls.x_scores_, dtype=float)
    Q = np.asarray(pls.y_loadings_, dtype=float)
    W_y = np.asarray(pls.y_weights_, dtype=float)
    if Q.shape[0] != 1:
        raise ValueError("dados_calibracao_pls: so' para uma resposta (y unico).")
    y_mean = float(np.ravel(getattr(pls, "_y_mean"))[0])
    y_std = float(np.ravel(getattr(pls, "_y_std", np.ones(1)))[0])
    y_hat = (T @ Q.T).ravel() * y_std + y_mean
    c1 = float(W_y[0, 0])
    if abs(c1) < 1e-12:
        raise ValueError("peso de y degenerado; nao da' para recuperar y.")
    y = np.asarray(pls.y_scores_, dtype=float)[:, 0] * c1 * y_std + y_mean
    return {"y": y, "y_hat": y_hat, "T": T}


def grade_eem_da_pasta(pasta: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(excitacao, emissao) do primeiro .dat legivel da pasta do modo `eem`
    (mesma busca e ordem do leitor `leitores_avancados._leitor_eem`, que usa
    a grade do primeiro arquivo). None se nao ha' grade."""
    from guaraci.eem_io import parse_eem_dat
    for arq in sorted(glob.glob(os.path.join(str(pasta or ""), "**", "*.dat"),
                                recursive=True)):
        try:
            ex, em, _m, _r = parse_eem_dat(arq)
        except Exception:  # noqa: BLE001 -- mesmo criterio do leitor: pula ilegivel
            continue
        return np.asarray(ex, dtype=float), np.asarray(em, dtype=float)
    return None


def rotulo_eixo_cromatograma(cfg) -> str:
    """Rotulo do eixo para gcms/hplc: o do perfil, se houver; senao tempo de
    retencao em minutos (os leitores gcms_io/parse_cromatograma_hplc devolvem
    o eixo sempre em minutos)."""
    r = rotulo_do_eixo(cfg)
    return r if r != ROTULO_EIXO_NEUTRO else "Retention time (min)"


# =========================================================================
#  Internos
# =========================================================================

def _cfg_ou_padrao(cfg):
    from guaraci.figuras import setup_matplotlib
    if cfg is None:
        from guaraci.config import Config
        cfg = Config()
    setup_matplotlib(cfg)
    return cfg


def _save(fig, nome: str, pasta: str, cfg) -> str:
    from guaraci import figuras
    return figuras.save(fig, nome, pasta, cfg)       # caminho real (grupo e serie inclusos)


def _orientar(ax, cfg) -> None:
    from guaraci.figuras import _orientar_eixo_espectral
    _orientar_eixo_espectral(ax, cfg)


def _legenda_fora(ax_leg, ax_dados, titulo: str = "") -> None:
    from guaraci.figuras import _legenda_lateral
    _legenda_lateral(ax_leg, ax_dados, titulo=titulo)


def _ordem_por_classe(rotulos) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Indices das amostras agrupadas por classe (ordem alfabetica, estavel)."""
    rot = np.asarray(rotulos, dtype=str)
    classes = sorted(set(rot.tolist()))
    ordem = np.concatenate([np.flatnonzero(rot == c) for c in classes])
    return ordem, rot[ordem], classes


def _faixas_de_classe(ax, rot_ordenado: np.ndarray, classes: List[str]) -> None:
    """Separadores verticais e nome da classe no topo de cada bloco de amostras."""
    cores = map_class_colors(classes)
    ini = 0
    for c in classes:
        n = int(np.sum(rot_ordenado == c))
        ax.axvspan(ini - 0.5, ini + n - 0.5, color=cores[c], alpha=0.08, lw=0)
        if ini > 0:
            ax.axvline(ini - 0.5, color="0.6", lw=0.6, ls="--")
        ax.text(ini + (n - 1) / 2.0, 1.01, c, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.5, color=cores[c])
        ini += n


# =========================================================================
#  1. Scree / variancia explicada (PCA)
# =========================================================================

def fig_pca_scree(var_pct: Sequence[float], cfg, pasta: str,
                  nome: str = "fig1b_pca_scree") -> str:
    """Variancia explicada (%) por componente (barras) e acumulada (linha),
    das componentes que a PCA exploratoria calculou (`n_pcs_pca`)."""
    cfg = _cfg_ou_padrao(cfg)
    v = np.asarray(var_pct, dtype=float).ravel()
    k = len(v)
    acum = variancia_acumulada(v.tolist())
    pcs = np.arange(1, k + 1)
    fig = plt.figure(figsize=(8.0, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[5.0, 1.3])
    ax, ax_leg = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax.bar(pcs, v, color=color(0), width=0.7, label="Per component", zorder=2)
    ax.plot(pcs, acum, "o-", color=color(1), ms=4, lw=1.3, label="Cumulative", zorder=3)
    if k <= 15:
        for x, val in zip(pcs, v):
            ax.text(float(x), val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(pcs)
    ax.set_xticklabels([f"PC{i}" for i in pcs], rotation=0 if k <= 10 else 45)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"PCA scree — {k} components computed, "
                 f"cumulative {acum[-1]:.1f}%", loc="left")
    ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
    _legenda_fora(ax_leg, ax)
    return _save(fig, nome, pasta, cfg)


# =========================================================================
#  2-4. Regressao: predito x medido com faixa, coeficientes b, Williams
# =========================================================================

def fig_predito_medido_faixa(y_cal, yhat_cal, y_val, yhat_val, rmsep: float,
                             cfg, pasta: str, *, y_holdout=None, yhat_holdout=None,
                             rmsep_holdout: Optional[float] = None,
                             nome: str = "figS2b_predito_medido_faixa") -> str:
    """Predito x referencia com a faixa y = x +- RMSEP (validacao interna) e,
    quando informados, os pontos do HOLDOUT externo com a sua faixa +-RMSEP."""
    cfg = _cfg_ou_padrao(cfg)
    yc, yhc = np.ravel(y_cal).astype(float), np.ravel(yhat_cal).astype(float)
    yv, yhv = np.ravel(y_val).astype(float), np.ravel(yhat_val).astype(float)
    tem_ho = y_holdout is not None and yhat_holdout is not None and len(np.ravel(y_holdout)) > 0
    partes = [yc, yhc, yv, yhv]
    if tem_ho:
        yh_, yhh_ = np.ravel(y_holdout).astype(float), np.ravel(yhat_holdout).astype(float)
        partes += [yh_, yhh_]
    todos = np.concatenate(partes)
    lo, hi = float(np.nanmin(todos)), float(np.nanmax(todos))
    pad = 0.06 * (hi - lo if hi > lo else 1.0) + float(rmsep)
    lim = (lo - pad, hi + pad)
    xx = np.array(lim)

    fig = plt.figure(figsize=(8.2, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[5.0, 1.6])
    ax, ax_leg = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax.fill_between(xx, xx - rmsep, xx + rmsep, color="0.55", alpha=0.18, lw=0,
                    label=f"y = x ± RMSEP ({rmsep:.3g})", zorder=1)
    ax.plot(xx, xx, "k--", lw=0.8, label="y = x", zorder=2)
    ax.scatter(yc, yhc, color=color(0), s=30, edgecolors=get_edge_color(color(0)),
               linewidths=0.5, alpha=0.85, label=f"Calibration (n={len(yc)})", zorder=3)
    ax.scatter(yv, yhv, color=color(1), s=40, marker="^",
               edgecolors=get_edge_color(color(1)), linewidths=0.5, alpha=0.9,
               label=f"Validation (n={len(yv)})", zorder=4)
    dentro = int(np.sum(np.abs(yhv - yv) <= rmsep))
    subt = f"{dentro}/{len(yv)} validation points inside ±RMSEP"
    if tem_ho:
        ax.scatter(yh_, yhh_, color=color(2), s=42, marker="s",
                   edgecolors=get_edge_color(color(2)), linewidths=0.5, alpha=0.95,
                   label=f"External holdout (n={len(yh_)})", zorder=5)
        if rmsep_holdout is not None and np.isfinite(rmsep_holdout):
            for sgn in (-1, 1):
                ax.plot(xx, xx + sgn * rmsep_holdout, color=color(2), lw=0.8, ls=":",
                        label=(f"± RMSEP holdout ({rmsep_holdout:.3g})" if sgn > 0 else None),
                        zorder=2)
        subt += f"; {int(np.sum(np.abs(yhh_ - yh_) <= rmsep))}/{len(yh_)} holdout"
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reference value")
    ax.set_ylabel("Predicted value")
    ax.set_title("Predicted vs reference with ±RMSEP band\n" + subt, loc="left")
    ax.grid(color="0.94", lw=0.5, zorder=0)
    _legenda_fora(ax_leg, ax)
    return _save(fig, nome, pasta, cfg)


def fig_coeficientes_regressao(eixo, coeficientes: Mapping[str, Any], cfg, pasta: str,
                               n_lv: Optional[Mapping[str, int]] = None,
                               nome: str = "figS2c_coeficientes_regressao") -> str:
    """Vetor de regressao b do PLS contra o eixo espectral, um painel por modelo
    (especie). b se aplica ao espectro PRE-PROCESSADO (o do modelo)."""
    cfg = _cfg_ou_padrao(cfg)
    eixo = np.asarray(eixo, dtype=float)
    blocos = paginas(list(coeficientes), GRAFICOS_POR_ITEM)
    caminhos = [_coeficientes_pagina(eixo, coeficientes, bloco, cfg, pasta, n_lv,
                                     nome_pagina(nome, i, len(blocos)))
                for i, bloco in enumerate(blocos, 1)]
    return caminhos[0]


def _coeficientes_pagina(eixo, coeficientes, nomes, cfg, pasta: str, n_lv, nome: str) -> str:
    """Uma imagem com ate' 3 vetores b (um por modelo/especie)."""
    n = len(nomes)
    fig, axes = plt.subplots(n, 1, figsize=(10.5, 2.7 * n + 0.8),
                             sharex=True, constrained_layout=True, squeeze=False)
    for i, (ax, nm) in enumerate(zip(axes[:, 0], nomes)):
        b = np.ravel(np.asarray(coeficientes[nm], dtype=float))
        ax.fill_between(eixo, 0, b, where=b >= 0, color=color(0), alpha=0.25, lw=0)
        ax.fill_between(eixo, 0, b, where=b < 0, color=color(1), alpha=0.25, lw=0)
        ax.plot(eixo, b, color="0.2", lw=0.9)
        ax.axhline(0, color="0.5", lw=0.6, ls="--")
        a = f" — {int(n_lv[nm])} LVs" if n_lv and nm in n_lv else ""
        ax.set_title(f"{nm}{a}", loc="left")
        ax.set_ylabel("Coefficient b")
        ax.grid(axis="y", color="0.94", lw=0.5)
        _orientar(ax, cfg)
    axes[-1, 0].set_xlabel(rotulo_do_eixo(cfg))
    fig.suptitle("PLS regression coefficients (b) on the pre-processed signal  "
                 "| fill: positive / negative", fontsize=10, fontweight="bold")
    return _save(fig, nome, pasta, cfg)


def fig_williams(modelos: Mapping[str, Mapping[str, Any]], cfg, pasta: str,
                 nome: str = "figS2d_williams") -> str:
    """Grafico de Williams: residuo padronizado x alavancagem, com h* = 3(A+1)/n
    e +-3. Cada item de `modelos`: h_cal, r_cal, n_lv (e opcionais h_val, r_val
    e h_holdout, r_holdout, projetados no mesmo modelo e padronizados pelo s
    da calibracao)."""
    cfg = _cfg_ou_padrao(cfg)
    blocos = paginas(list(modelos), GRAFICOS_POR_ITEM)
    caminhos = [_williams_pagina(modelos, bloco, cfg, pasta, nome_pagina(nome, i, len(blocos)))
                for i, bloco in enumerate(blocos, 1)]
    return caminhos[0]


def _williams_pagina(modelos, nomes, cfg, pasta: str, nome: str) -> str:
    """Uma imagem do grafico de Williams com ate' 3 modelos lado a lado."""
    n = len(nomes)
    fig = plt.figure(figsize=(4.6 * n + 1.8, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, n + 1, width_ratios=[4.6] * n + [1.6])
    ax0 = None
    for i, nm in enumerate(nomes):
        d = modelos[nm]
        h = np.ravel(d["h_cal"]).astype(float)
        r = np.ravel(d["r_cal"]).astype(float)
        A = int(d["n_lv"])
        hs = limiar_williams(A, len(h))
        ax = fig.add_subplot(gs[i])
        ax0 = ax0 or ax
        ax.scatter(h, r, color=color(0), s=30, edgecolors=get_edge_color(color(0)),
                   linewidths=0.5, alpha=0.85, label="Calibration", zorder=3)
        hv = rv = None
        if d.get("h_val") is not None and d.get("r_val") is not None:
            hv, rv = np.ravel(d["h_val"]).astype(float), np.ravel(d["r_val"]).astype(float)
            ax.scatter(hv, rv, color=color(1), s=38, marker="^",
                       edgecolors=get_edge_color(color(1)), linewidths=0.5,
                       alpha=0.9, label="Validation", zorder=4)
        hho = rho = None
        if d.get("h_holdout") is not None and d.get("r_holdout") is not None:
            hho = np.ravel(d["h_holdout"]).astype(float)
            rho = np.ravel(d["r_holdout"]).astype(float)
            ax.scatter(hho, rho, color=color(2), s=38, marker="s",
                       edgecolors=get_edge_color(color(2)), linewidths=0.5,
                       alpha=0.95, label="External holdout", zorder=5)
        ax.axvline(hs, color=color(3), ls="--", lw=1.0, label="h* = 3(A+1)/n")
        for s_ in (-3, 3):
            ax.axhline(s_, color="0.35", ls=":", lw=0.9, label=("±3 s" if s_ > 0 else None))
        ax.axhline(0, color="0.8", lw=0.5)
        hh = np.concatenate([h] + ([hv] if hv is not None else [])
                            + ([hho] if hho is not None else []))
        rr = np.concatenate([r] + ([rv] if rv is not None else [])
                            + ([rho] if rho is not None else []))
        ax.set_xlim(0, max(float(hh.max()), hs) * 1.12)
        ym = max(3.6, float(np.abs(rr).max()) * 1.12)
        ax.set_ylim(-ym, ym)
        n_h = int(np.sum(hh > hs)); n_r = int(np.sum(np.abs(rr) > 3))
        ax.set_title(f"{nm}: h* = {hs:.3f} (A={A}, n={len(h)})\n"
                     f"{n_h} above h*, {n_r} with |r| > 3", loc="left")
        ax.set_xlabel("Leverage h")
        if i == 0:
            ax.set_ylabel("Standardized residual (pred − ref) / s")
        ax.grid(color="0.94", lw=0.5, zorder=0)
    ax_leg = fig.add_subplot(gs[n])
    _legenda_fora(ax_leg, ax0)
    fig.suptitle("Williams plot — applicability domain of the PLS regression",
                 fontsize=10, fontweight="bold")
    return _save(fig, nome, pasta, cfg)


def figuras_regressao_do_modelo(pasta_run: str, cfg) -> List[str]:
    """Depois de uma corrida de Quantificacao: abre o modelo salvo em
    Modelos/ e desenha coeficientes b e Williams (calibracao) de cada
    regressao por especie. Nada e' reajustado: tudo vem dos PLS salvos.
    Devolve os caminhos gravados ([] se o modelo nao tem regressao)."""
    from guaraci.config import NOME_MODELOS, pasta_execucao
    if not pasta_run or not os.path.isdir(pasta_execucao(pasta_run, NOME_MODELOS)):
        return []
    cams = sorted(glob.glob(os.path.join(pasta_execucao(pasta_run, NOME_MODELOS), "*.joblib")))
    if not cams:
        return []
    if cfg is not None:
        from guaraci.modos_analise import QUANTIFICACAO, resolve_objective
        if resolve_objective(cfg) != QUANTIFICACAO:
            return []
    from guaraci.predicao import load_model
    # confiar=True: e' o modelo que ESTA corrida acabou de gravar; o
    # manifesto assinado (sha256 + HMAC local) continua sendo conferido.
    pkg = load_model(cams[0], confiar=True)
    reg = pkg.get("regressao_por_especie") or {}
    if not reg:
        return []
    eixo = np.asarray(pkg.get("wavenumbers"), dtype=float)
    coefs: Dict[str, np.ndarray] = {}
    lvs: Dict[str, int] = {}
    will: Dict[str, Dict[str, Any]] = {}
    for esp, info in reg.items():
        pls = info["pipeline"].named_steps["pls"]
        A = int(pls.n_components)
        coefs[str(esp)] = np.ravel(pls.coef_)
        lvs[str(esp)] = A
        d = dados_calibracao_pls(pls)
        r, _s = residuos_padronizados(d["y"], d["y_hat"], A)
        will[str(esp)] = {"h_cal": alavancagem_scores(d["T"]), "r_cal": r, "n_lv": A}
    saidas = []
    if eixo.size == len(next(iter(coefs.values()))):
        saidas.append(fig_coeficientes_regressao(eixo, coefs, cfg, pasta_run, n_lv=lvs))
    saidas.append(fig_williams(will, cfg, pasta_run))
    return saidas


# =========================================================================
#  5. MCR-ALS
# =========================================================================

def fig_mcr_als(C: np.ndarray, S: np.ndarray, eixo, rotulos, cfg, pasta: str,
                lof_percent: Optional[float] = None,
                nome: str = "fig_mcr_als_perfis_C_S") -> str:
    """(a) perfis de concentracao C por amostra (agrupadas por classe) e
    (b) espectros puros S contra o eixo espectral; mesma cor por componente."""
    cfg = _cfg_ou_padrao(cfg)
    C = np.asarray(C, dtype=float); S = np.asarray(S, dtype=float)
    eixo = np.asarray(eixo, dtype=float)
    ordem, rot_ord, classes = _ordem_por_classe(rotulos)
    k = C.shape[1]
    fig = plt.figure(figsize=(11.0, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[6.0, 1.2])
    ax_c, ax_s = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_leg = fig.add_subplot(gs[:, 1])
    pos = np.arange(len(ordem))
    for j in range(k):
        ax_c.plot(pos, C[ordem, j], "o-", color=color(j), ms=3.2, lw=1.0,
                  label=f"Component {j + 1}")
        ax_s.plot(eixo, S[j], color=color(j), lw=1.3)
    _faixas_de_classe(ax_c, rot_ord, classes)
    ax_c.set_xlim(-0.5, len(ordem) - 0.5)
    ax_c.set_xlabel("Sample (grouped by class)")
    ax_c.set_ylabel("Concentration profile C (a.u.)")
    ax_c.set_title("(a) Concentration profiles C", loc="left", pad=16)
    ax_s.set_xlabel(rotulo_do_eixo(cfg))
    ax_s.set_ylabel("Pure spectrum S (a.u.)")
    ax_s.set_title("(b) Resolved pure spectra S", loc="left")
    _orientar(ax_s, cfg)
    for ax in (ax_c, ax_s):
        ax.grid(color="0.94", lw=0.5); ax.set_axisbelow(True)
    _legenda_fora(ax_leg, ax_c, titulo="MCR-ALS")
    lof = f" — lack of fit {lof_percent:.2f}%" if lof_percent is not None else ""
    fig.suptitle(f"MCR-ALS resolution{lof}  (C·S scale is ambiguous: compare "
                 "shapes, not absolute heights)", fontsize=10, fontweight="bold")
    return _save(fig, nome, pasta, cfg)


# =========================================================================
#  6. PARAFAC (EEM)
# =========================================================================

def fig_parafac_eem(resultado, grade_excitacao, grade_emissao, rotulos, cfg,
                    pasta: str, nome: str = "fig_parafac_loadings") -> str:
    """Loadings por modo do PARAFAC: excitacao, emissao e scores das amostras."""
    cfg = _cfg_ou_padrao(cfg)
    A = np.asarray(resultado.fator_amostra, dtype=float)
    B = np.asarray(resultado.fator_excitacao, dtype=float)
    Cm = np.asarray(resultado.fator_emissao, dtype=float)
    ex = np.asarray(grade_excitacao, dtype=float)
    em = np.asarray(grade_emissao, dtype=float)
    R = A.shape[1]
    ordem, rot_ord, classes = _ordem_por_classe(rotulos)
    fig = plt.figure(figsize=(13.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[4.0, 4.0, 4.6, 1.3])
    ax_ex, ax_em, ax_a = (fig.add_subplot(gs[i]) for i in range(3))
    ax_leg = fig.add_subplot(gs[3])
    o_ex, o_em = np.argsort(ex), np.argsort(em)
    pos = np.arange(len(ordem))
    for r in range(R):
        ax_ex.plot(ex[o_ex], B[o_ex, r], "-", color=color(r), lw=1.4,
                   label=f"Component {r + 1}")
        ax_em.plot(em[o_em], Cm[o_em, r], "-", color=color(r), lw=1.4)
        ax_a.plot(pos, A[ordem, r], "o-", color=color(r), ms=3.2, lw=0.9)
    ax_ex.set_xlabel("Excitation wavelength (nm)"); ax_ex.set_ylabel("Loading")
    ax_ex.set_title("(a) Excitation mode", loc="left")
    ax_em.set_xlabel("Emission wavelength (nm)"); ax_em.set_ylabel("Loading")
    ax_em.set_title("(b) Emission mode", loc="left")
    _faixas_de_classe(ax_a, rot_ord, classes)
    ax_a.set_xlim(-0.5, len(ordem) - 0.5)
    ax_a.set_xlabel("Sample (grouped by class)"); ax_a.set_ylabel("Score")
    ax_a.set_title("(c) Sample mode (scores)", loc="left", pad=16)
    for ax in (ax_ex, ax_em, ax_a):
        ax.grid(color="0.94", lw=0.5); ax.set_axisbelow(True)
    _legenda_fora(ax_leg, ax_ex, titulo="PARAFAC")
    fig.suptitle(f"PARAFAC loadings per mode — {R} components, relative "
                 f"reconstruction error {float(resultado.erro_reconstrucao_relativo):.3f}",
                 fontsize=10, fontweight="bold")
    restricao = getattr(resultado, "restricao", "")
    if restricao:
        motor = fig.get_layout_engine()
        if isinstance(motor, ConstrainedLayoutEngine):          # figura criada com constrained_layout=True
            motor.set(rect=(0, 0.04, 1, 0.96))                  # reserva a base para a nota
        fig.text(0.005, 0.005, ("PARAFAC with non-negativity constraint (HALS) since v1.0.0; earlier versions or unconstrained "
                                "software may give different signs/orders of the factors."
                                if restricao == "nao-negativo" else
                                "PARAFAC without constraint (free ALS), by user choice; the default since v1.0.0 is non-negative (HALS)."),
                 fontsize=6.5, color="0.35", va="bottom", ha="left")
    return _save(fig, nome, pasta, cfg)


# =========================================================================
#  7. ASCA
# =========================================================================

def _svd_efeito(efeito: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """(scores, loadings) do efeito, com o sinal de cada componente alinhado aos
    `scores` que `asca.asca_decompose` devolveu (mesma SVD da mesma matriz)."""
    Xf = np.asarray(efeito["matriz"], dtype=float)
    ref = np.asarray(efeito["scores"], dtype=float)
    k = ref.shape[1]
    U, S, Vt = np.linalg.svd(Xf, full_matrices=False)
    P = Vt[:k].copy()
    T = U[:, :k] * S[:k]
    for j in range(k):
        if float(np.dot(T[:, j], ref[:, j])) < 0:
            P[j] *= -1; T[:, j] *= -1
    return T, P


def fig_asca(decomposicao: Mapping[str, Any], fatores: Mapping[str, Any], eixo, cfg,
             pasta: str, permutacao: Optional[Mapping[str, Any]] = None,
             nome: str = "fig_asca_efeitos") -> str:
    """Por fator: scores do efeito (nivel = marcador grande; nuvem = efeito +
    residuo projetado) e loadings do efeito contra o eixo espectral."""
    cfg = _cfg_ou_padrao(cfg)
    eixo = np.asarray(eixo, dtype=float)
    efeitos = decomposicao["efeitos"]
    residuo = np.asarray(decomposicao["residuo"], dtype=float)
    nomes = list(efeitos)
    caminhos: List[str] = []
    # Uma imagem por fator (scores + loadings do efeito = 2 paineis): com 2+ fatores a grade
    # unica passava de MAX_PAINEIS_POR_FIGURA.
    for i_f, nm in enumerate(nomes, 1):
        fig = plt.figure(figsize=(13.0, 4.6), constrained_layout=True)
        gs = fig.add_gridspec(1, 3, width_ratios=[4.2, 1.3, 6.0])
        i = 0
        ef = efeitos[nm]
        T, P = _svd_efeito(ef)
        # componente com variancia ~0 (efeito de posto menor que k, ex.: fator
        # de 2 niveis tem posto 1) e' ruido numerico: nao vira eixo nem loading
        k = max(1, int(np.sum(np.asarray(ef["var_explicada"], dtype=float) > 1e-9)))
        T, P = T[:, :k], P[:k]
        proj = (np.asarray(ef["matriz"], dtype=float) + residuo) @ P.T
        niveis = np.asarray(fatores[nm]).astype(str)
        cores = map_class_colors(niveis)
        ve = np.asarray(ef["var_explicada"], dtype=float) * 100
        ax_s, ax_leg, ax_p = (fig.add_subplot(gs[i, j]) for j in range(3))
        for j_n, nv in enumerate(sorted(cores)):
            m = niveis == nv
            c = cores[nv]
            if k >= 2:
                ax_s.scatter(proj[m, 0], proj[m, 1], color=c, s=14, alpha=0.45, lw=0)
                ax_s.scatter(T[m, 0][:1], T[m, 1][:1], color=c, s=90, marker="D",
                             edgecolors="black", linewidths=0.8, zorder=4,
                             label=f"{nv} (n={int(m.sum())})")
            else:
                jit = (np.random.default_rng(j_n).random(int(m.sum())) - 0.5) * 0.3
                ax_s.scatter(np.full(int(m.sum()), j_n) + jit, proj[m, 0], color=c,
                             s=14, alpha=0.45, lw=0)
                ax_s.scatter([j_n], T[m, 0][:1], color=c, s=90, marker="D",
                             edgecolors="black", linewidths=0.8, zorder=4,
                             label=f"{nv} (n={int(m.sum())})")
        if k >= 2:
            ax_s.set_xlabel(f"Effect PC1 ({ve[0]:.1f}%)")
            ax_s.set_ylabel(f"Effect PC2 ({ve[1]:.1f}%)")
            ax_s.axvline(0, color="0.85", lw=0.5)
        else:
            ax_s.set_xticks(range(len(cores)))
            ax_s.set_xticklabels(sorted(cores), rotation=30, ha="right")
            ax_s.set_ylabel(f"Effect PC1 ({ve[0]:.1f}%)")
        ax_s.axhline(0, color="0.85", lw=0.5)
        p_txt = ""
        if permutacao is not None and nm in permutacao:
            p_txt = f", permutation p = {float(permutacao[nm]['p_value']):.3g}"
        ax_s.set_title(f"{nm}: {float(ef['fracao_ss_total']) * 100:.1f}% of total SS"
                       f"{p_txt}", loc="left")
        ax_s.grid(color="0.94", lw=0.5); ax_s.set_axisbelow(True)
        _legenda_fora(ax_leg, ax_s, titulo=str(nm))
        for j in range(k):
            ax_p.plot(eixo, P[j], color="0.15" if j == 0 else "0.5", lw=1.2,
                      ls="-" if j == 0 else "--", label=f"PC{j + 1}")
        ax_p.axhline(0, color="0.6", lw=0.6, ls=":")
        ax_p.set_xlabel(rotulo_do_eixo(cfg)); ax_p.set_ylabel("Effect loading")
        ax_p.set_title(f"{nm}: effect loadings" + (" (solid PC1, dashed PC2)" if k > 1 else " (PC1)"),
                       loc="left")
        ax_p.grid(axis="y", color="0.94", lw=0.5)
        _orientar(ax_p, cfg)
        fig.suptitle(f"ASCA — factor {nm}: effect scores (diamond = level, cloud = level + "
                     "residual) and loadings", fontsize=10, fontweight="bold")
        caminhos.append(_save(fig, nome_pagina(nome, i_f, len(nomes)), pasta, cfg))
    return caminhos[0] if caminhos else ""


# =========================================================================
#  8. EEM media por classe (mapa de contorno)
# =========================================================================

def fig_eem_media_classe(grade_excitacao, grade_emissao, X, rotulos, cfg, pasta: str,
                         nome: str = "fig0_espectros_medios_classe") -> str:
    """Mapa de contorno da EEM MEDIA de cada classe (escala de cor comum).
    Cada linha de X e' a matriz emissao x excitacao achatada pelo leitor."""
    cfg = _cfg_ou_padrao(cfg)
    ex = np.asarray(grade_excitacao, dtype=float)
    em = np.asarray(grade_emissao, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.shape[1] != len(ex) * len(em):
        raise ValueError(f"X tem {X.shape[1]} colunas; a grade e' {len(em)} x {len(ex)}.")
    rot = np.asarray(rotulos, dtype=str)
    classes = sorted(set(rot.tolist()))
    cores = map_class_colors(classes)
    medias = {c: X[rot == c].mean(axis=0).reshape(len(em), len(ex)) for c in classes}
    vmin = min(float(m.min()) for m in medias.values())
    vmax = max(float(m.max()) for m in medias.values())
    niveis = np.linspace(vmin, vmax if vmax > vmin else vmin + 1.0, 21)
    blocos = paginas(classes, GRAFICOS_POR_ITEM)
    caminhos = [_eem_media_pagina(ex, em, medias, bloco, rot, cores, niveis, cfg, pasta,
                                  nome_pagina(nome, i, len(blocos)))
                for i, bloco in enumerate(blocos, 1)]
    return caminhos[0]


def _eem_media_pagina(ex, em, medias, classes, rot, cores, niveis, cfg, pasta: str,
                      nome: str) -> str:
    """Uma imagem com ate' 3 EEM medias; `niveis` e' o mesmo em todas as paginas (escala comum)."""
    nc = len(classes); nr = 1
    fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc + 1.0, 3.9 * nr),
                             constrained_layout=True, squeeze=False)
    o_ex, o_em = np.argsort(ex), np.argsort(em)
    cs = None
    for ax, c in zip(axes.ravel(), classes):
        M = medias[c][np.ix_(o_em, o_ex)]
        cs = ax.contourf(ex[o_ex], em[o_em], M, levels=niveis, cmap="viridis")
        ax.contour(ex[o_ex], em[o_em], M, levels=niveis[::4], colors="white",
                   linewidths=0.35, alpha=0.6)
        ax.set_title(f"{c} (n={int(np.sum(rot == c))})", loc="left", color="0.1")
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(cores[c]); sp.set_linewidth(2.0)
        ax.set_xlabel("Excitation wavelength (nm)")
        ax.set_ylabel("Emission wavelength (nm)")
    for ax in axes.ravel()[len(classes):]:
        ax.axis("off")
    if cs is not None:
        from matplotlib.ticker import MaxNLocator
        cb = fig.colorbar(cs, ax=axes.ravel().tolist(), shrink=0.9,
                          label="Mean intensity (raw)")
        cb.locator = MaxNLocator(nbins=6)
        cb.update_ticks()
    fig.suptitle("Mean excitation–emission matrix per class (raw data, common colour scale)",
                 fontsize=10, fontweight="bold")
    return _save(fig, nome, pasta, cfg)


# =========================================================================
#  9. Transferencia de calibracao (DS/PDS)
# =========================================================================

def _ajustar_transferencia(metodo: str, Xm, Xs, janela: int, alpha: float):
    from guaraci.transferencia_calibracao import (
        direct_standardization, piecewise_direct_standardization)
    if metodo == "ds":
        return direct_standardization(Xm, Xs, alpha=alpha)
    return piecewise_direct_standardization(Xm, Xs, janela=janela, alpha=alpha)


def fig_transferencia_calibracao(eixo, Xm, Xs, metodo: str, cfg, pasta: str, *,
                                 janela: int = 5, alpha: float = 1.0,
                                 nome: str = "fig_transferencia_mestre_escravo"
                                 ) -> Tuple[str, float, float]:
    """(a) mestre x escravo (media das amostras de transferencia) sem correcao;
    (b) mestre x escravo corrigido, cada amostra corrigida por um ajuste que
    NAO a viu (deixa-uma-fora, a mesma conta de `transferir_cli.erro_loo`);
    (c) erro RMS escravo-mestre por amostra, antes e depois.
    Devolve (caminho, rms_antes, rms_depois) -- iguais aos do resumo do CLI."""
    from guaraci.transferencia_calibracao import apply_standardization
    orientar = cfg is not None      # sem cfg (subcomando) o eixo e' desconhecido: fica crescente
    cfg = _cfg_ou_padrao(cfg)
    Xm = np.asarray(Xm, dtype=float); Xs = np.asarray(Xs, dtype=float)
    eixo = np.asarray(eixo, dtype=float)
    n = Xm.shape[0]
    Xc = np.empty_like(Xs)
    for i in range(n):
        dentro = np.arange(n) != i
        t = _ajustar_transferencia(metodo, Xm[dentro], Xs[dentro], janela, alpha)
        Xc[i] = apply_standardization(Xs[i:i + 1], t)[0]
    mse_antes = np.mean((Xs - Xm) ** 2, axis=1)
    mse_depois = np.mean((Xc - Xm) ** 2, axis=1)
    rms_antes = float(np.sqrt(np.mean(mse_antes)))
    rms_depois = float(np.sqrt(np.mean(mse_depois)))

    fig = plt.figure(figsize=(12.0, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[5.0, 5.0, 1.5], height_ratios=[1.0, 0.85])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[1, 0:2]); ax_leg = fig.add_subplot(gs[:, 2])
    mm = Xm.mean(axis=0)
    ax_a.plot(eixo, mm, color=color(0), lw=1.4, label="Master")
    ax_a.plot(eixo, Xs.mean(axis=0), color=color(1), lw=1.2, ls="--", label="Slave (raw)")
    ax_a.set_title(f"(a) Before: RMS slave − master = {rms_antes:.4g}", loc="left")
    ax_b.plot(eixo, mm, color=color(0), lw=1.4)
    ax_b.plot(eixo, Xc.mean(axis=0), color=color(2), lw=1.2, ls="-.",
              label=f"Slave after {metodo.upper()} (LOO)")
    ax_b.set_title(f"(b) After {metodo.upper()}: RMS = {rms_depois:.4g}", loc="left")
    for ax in (ax_a, ax_b):
        ax.set_xlabel(rotulo_do_eixo(cfg))
        ax.grid(color="0.94", lw=0.5); ax.set_axisbelow(True)
        if orientar:
            _orientar(ax, cfg)
    ax_a.set_ylabel("Mean signal of the transfer samples")
    pos = np.arange(n); w = 0.38
    ax_c.bar(pos - w / 2, np.sqrt(mse_antes), w, color=color(1), label="RMS before")
    ax_c.bar(pos + w / 2, np.sqrt(mse_depois), w, color=color(2), label="RMS after (LOO)")
    ax_c.set_xticks(pos); ax_c.set_xticklabels([str(i + 1) for i in pos])
    ax_c.set_xlabel("Transfer sample (each corrected by a fit that left it out)")
    ax_c.set_ylabel("RMS slave − master")
    ax_c.set_title("(c) Error per transfer sample, before and after", loc="left")
    ax_c.grid(axis="y", color="0.94", lw=0.5); ax_c.set_axisbelow(True)
    hs_, ls_ = [], []
    for ax in (ax_a, ax_b, ax_c):
        h, l_ = ax.get_legend_handles_labels(); hs_ += h; ls_ += l_
    ax_leg.axis("off")
    ax_leg.legend(hs_, ls_, loc="center left", frameon=False, fontsize=8.5)
    jan = f", window {janela}" if metodo == "pds" else ""
    fig.suptitle(f"Calibration transfer {metodo.upper()}{jan}, alpha {alpha:g} — "
                 f"{n} transfer samples", fontsize=10, fontweight="bold")
    return _save(fig, nome, pasta, cfg), rms_antes, rms_depois


def _ler_csv_numerico(caminho: str) -> Tuple[np.ndarray, np.ndarray]:
    """(eixo, X) com as colunas numericas do CSV, como `transferir_cli._ler`;
    eixo = nome da coluna se todos forem numeros, senao 1..p."""
    import pandas as pd
    df = pd.read_csv(caminho, sep=None, engine="python", encoding="utf-8-sig")
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    try:
        eixo = np.array([float(c) for c in num])
    except (TypeError, ValueError):
        eixo = np.arange(1, len(num) + 1, dtype=float)
    return eixo, df[num].to_numpy(dtype=float)


def figuras_transferencia_de_argv(argv: Sequence[str], cfg=None) -> Optional[str]:
    """Desenha a figura de transferencia para os MESMOS argumentos de
    `guaraci transferir` (chamar depois que ele terminou com 0). None se os
    argumentos nao bastam ou a ajuda foi pedida."""
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    for a in ("--mestre", "--escravo", "--novos"):
        p.add_argument(a)
    p.add_argument("--metodo", default="pds")
    p.add_argument("--janela", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--saida", default=nome_pasta("resultados_transferencia"))
    p.add_argument("--ajuda", "--help", "-h", action="store_true", dest="ajuda")
    args, _resto = p.parse_known_args(list(argv))
    if args.ajuda or not (args.mestre and args.escravo):
        return None
    eixo, Xm = _ler_csv_numerico(args.mestre)
    _e, Xs = _ler_csv_numerico(args.escravo)
    caminho, _a, _d = fig_transferencia_calibracao(
        eixo, Xm, Xs, str(args.metodo).lower(), cfg, args.saida,
        janela=args.janela, alpha=args.alpha)
    return caminho
