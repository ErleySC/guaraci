"""vip_estabilidade.py -- VIP do PLS com estabilidade nas dobras da validacao.

O VIP (Variable Importance in Projection) resume quanto cada variavel pesa nas
componentes do PLS. Formula (Wold, Sjostrom & Eriksson 2001,
doi:10.1016/S0169-7439(01)00155-1; equivalente a de Chong & Jun 2005,
doi:10.1016/j.chemolab.2004.12.011):

    VIP_j = sqrt( p * sum_k [ SS_k * (w_jk / ||w_k||)^2 ] / sum_k SS_k ),
    SS_k = (t_k . t_k) * (q_k . q_k)

com w_k os pesos, t_k os escores e q_k os carregamentos de Y da componente k
(`PLSRegression.x_weights_`, `x_scores_`, `y_loadings_`). Escala: a media de
VIP^2 sobre as p variaveis e' 1, isto e', sum_j VIP_j^2 = p (propriedade
conferida em teste); por isso o corte 1 separa "acima" de "abaixo da media".

Estabilidade: o VIP e' recalculado em cada dobra de treino de uma validacao
cruzada AGRUPADA (`StableStratifiedGroupKFold`: replicas da mesma amostra ficam
juntas), repetida com sementes diferentes. Para cada variavel sai a frequencia
com que VIP > limiar e o intervalo (percentis 2,5-97,5) do VIP nas dobras.

O que o VIP NAO e': nao e' teste de significancia; VIP > 1 e' regra pratica.
Escolher variaveis pelo VIP e depois validar o modelo com os MESMOS dados da
escolha da resultado otimista. Reduzir variaveis pelo VIP so' vale dentro de
uma validacao externa (aninhada): este comando so' mostra a estabilidade e nao
reduz variaveis; a selecao aninhada e' a do `guaraci run` (opcao 'Selecao de
variaveis', selecao_variaveis_etapa4).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from guaraci.chemometric_stats import vip_scores
from guaraci.validacao_estatistica import StableStratifiedGroupKFold

log = logging.getLogger(__name__)

__all__ = ["vip_estabilidade", "figura_vip_estabilidade", "TEXTO_VIP", "LIMIAR_PADRAO",
           "FREQ_ESTAVEL", "FREQ_INSTAVEL"]

LIMIAR_PADRAO = 1.0
#: Frequencia de VIP > limiar a partir da qual a variavel e' "estavel acima" e
#: ate' a qual e' "estavel abaixo" (regra pratica; entre os dois, instavel).
FREQ_ESTAVEL = 0.8
FREQ_INSTAVEL = 0.2

TEXTO_VIP = (
    "VIP > 1 e' uma regra pratica, nao um teste de significancia. Escolher "
    "variaveis olhando o VIP e depois validar o modelo com os mesmos dados da "
    "escolha da resultado otimista; reduzir variaveis pelo VIP so' vale dentro "
    "de uma validacao externa (aninhada): este comando nao reduz variaveis; "
    "a selecao aninhada e' a do `guaraci run`, com a opcao 'Selecao de variaveis' "
    "ligada (selecao_variaveis_etapa4). "
    "A estabilidade mostrada vem de dobras de validacao agrupadas: variavel "
    "com VIP > 1 em quase todas as dobras e' mais confiavel que a que passa "
    "do limite so' em algumas. Referencia: Wold, Sjostrom & Eriksson 2001, "
    "doi:10.1016/S0169-7439(01)00155-1.")


def _alvo(y: Sequence[Any], tarefa: str):
    """(Y para o PLS, rotulos de estratificacao). Classificacao: Y one-hot.
    Regressao: Y numerico; estratifica por terco do valor."""
    arr = np.asarray(y)
    if tarefa == "classificacao":
        classes, idx = np.unique(arr.astype(str), return_inverse=True)
        if len(classes) < 2:
            raise ValueError("VIP recusado: a coluna de classe tem menos de 2 classes")
        Y = np.zeros((len(arr), len(classes)))
        Y[np.arange(len(arr)), idx] = 1.0
        return Y, idx
    yv = arr.astype(float)
    if not np.all(np.isfinite(yv)):
        raise ValueError("VIP recusado: a resposta numerica tem NaN/inf; remova ou impute antes")
    if np.ptp(yv) == 0:
        raise ValueError("VIP recusado: a resposta numerica e' constante")
    cortes = np.quantile(yv, [1 / 3, 2 / 3])
    return yv.reshape(-1, 1), np.searchsorted(cortes, yv, side="right")


def _autoescalar(Xtr: np.ndarray, autoescalar: bool) -> np.ndarray:
    media = Xtr.mean(axis=0)
    if not autoescalar:
        return Xtr - media
    dp = Xtr.std(axis=0, ddof=1)
    dp[dp == 0] = 1.0
    return (Xtr - media) / dp


def vip_estabilidade(X: np.ndarray, y: Sequence[Any], n_componentes: int,
                     grupos: Optional[Sequence[Any]], *, tarefa: str = "classificacao",
                     n_divisoes: int = 5, n_repeticoes: int = 10,
                     limiar: float = LIMIAR_PADRAO, seed: int = 0,
                     grouping_guarantee: str = "high",
                     autoescalar: bool = True) -> Dict[str, Any]:
    """VIP do modelo com todos os dados + estabilidade nas dobras agrupadas.

    `tarefa`: 'classificacao' (PLS-DA, Y one-hot) ou 'regressao' (Y numerico).
    `grupos`: id da amostra fisica/mae por linha (replicas com o MESMO id). Sem
    grupos, cada linha vira um grupo e o resultado sai com AVISO (replicas
    vazariam entre treino e teste e o VIP ficaria otimista)."""
    if tarefa not in ("classificacao", "regressao"):
        raise ValueError(f"tarefa deve ser 'classificacao' ou 'regressao' (recebido {tarefa!r})")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 4:
        raise ValueError("VIP recusado: X precisa ser uma matriz amostras x variaveis com >= 4 linhas")
    if not np.all(np.isfinite(X)):
        raise ValueError("VIP recusado: X tem NaN/inf; remova ou impute antes")
    n, p = X.shape
    if len(y) != n:
        raise ValueError(f"VIP recusado: y tem {len(y)} linhas e X tem {n}")
    if not 1 <= n_componentes <= min(n - 1, p):
        raise ValueError(f"VIP recusado: n_componentes={n_componentes} fora de 1..{min(n - 1, p)}")
    if n_divisoes < 2 or n_repeticoes < 1:
        raise ValueError("VIP recusado: n_divisoes >= 2 e n_repeticoes >= 1")
    Y, estrato = _alvo(y, tarefa)
    avisos: List[str] = []
    if grupos is None:
        avisos.append("sem coluna de grupo: cada linha foi tratada como grupo; se ha replicas da "
                      "mesma amostra, elas vazam entre treino e teste e o VIP/estabilidade fica otimista")
        g = np.arange(n).astype(str)
    else:
        g = np.asarray(grupos).astype(str)
        if len(g) != n:
            raise ValueError(f"VIP recusado: grupos tem {len(g)} linhas e X tem {n}")
    if grouping_guarantee in ("none", "medium"):
        avisos.append(f"garantia de agrupamento '{grouping_guarantee}': as dobras podem nao separar "
                      "todas as replicas; leia a estabilidade com cautela")
    n_grupos = len(set(g))
    div = n_divisoes
    if n_grupos < div:
        if n_grupos < 2:
            raise ValueError("VIP recusado: menos de 2 grupos independentes; a estabilidade nao e' definida")
        avisos.append(f"so' {n_grupos} grupos: n_divisoes reduzido de {div} para {n_grupos}")
        div = n_grupos

    def _vip(idx: np.ndarray) -> np.ndarray:
        Xc = _autoescalar(X[idx], autoescalar)
        pls = PLSRegression(n_components=n_componentes, scale=False)
        pls.fit(Xc, Y[idx])
        return vip_scores(pls)

    completo = _vip(np.arange(n))
    vips: List[np.ndarray] = []
    n_falhos = 0
    for r in range(n_repeticoes):
        cv = StableStratifiedGroupKFold(n_splits=div, seed=seed + r)
        for tr, _ in cv.split(X, estrato, groups=g):
            try:
                v = _vip(tr)
            except (ValueError, np.linalg.LinAlgError):
                n_falhos += 1
                continue
            if np.all(np.isfinite(v)):
                vips.append(v)
            else:
                n_falhos += 1
    if not vips:
        raise ValueError("VIP recusado: nenhum ajuste das dobras convergiu; reduza n_componentes")
    if n_falhos:
        avisos.append(f"{n_falhos} ajuste(s) de dobra falharam e ficaram de fora (n = {len(vips)})")
    arr = np.asarray(vips)
    freq = (arr > limiar).mean(axis=0)
    classe = np.where(freq >= FREQ_ESTAVEL, "estavel_acima",
                      np.where(freq <= FREQ_INSTAVEL, "estavel_abaixo", "instavel"))
    return {"vip": completo, "media": arr.mean(axis=0),
            "dp": arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros(p),
            "ic95_inf": np.percentile(arr, 2.5, axis=0), "ic95_sup": np.percentile(arr, 97.5, axis=0),
            "frequencia": freq, "classe": classe, "limiar": float(limiar),
            "n_ajustes": int(len(arr)), "n_falhos": int(n_falhos), "n_variaveis": int(p),
            "n_divisoes": int(div), "n_repeticoes": int(n_repeticoes), "n_grupos": int(n_grupos),
            "soma_vip2": float((completo ** 2).sum()), "tarefa": tarefa,
            "avisos": avisos, "texto": TEXTO_VIP}


def figura_vip_estabilidade(res: Dict[str, Any], caminho: str,
                            nomes: Optional[Sequence[str]] = None, *, max_barras: int = 60) -> str:
    """Barras do VIP medio das dobras (intervalo 95%) coloridas pela frequencia de
    VIP > limiar, com a linha do limiar. Mais de `max_barras` variaveis: barras
    finas, sem rotulos (espectro). Figure/Agg direto, sem estado global do pyplot."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    p = res["n_variaveis"]
    fig = Figure(figsize=(max(7.0, min(14.0, 0.28 * p + 3)) if p <= max_barras else 10.0, 4.2))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(p)
    cores = np.where(res["classe"] == "estavel_acima", "#4C72B0",
                     np.where(res["classe"] == "instavel", "#DD8452", "#BBBBBB"))
    ax.bar(x, res["media"], width=0.8 if p <= max_barras else 1.0, color=cores, zorder=2)
    if p <= max_barras:
        # clip em 0: media e percentil iguais em ponto flutuante podem diferir por -1e-17 e o matplotlib recusa yerr negativo
        ax.errorbar(x, res["media"], yerr=[np.maximum(res["media"] - res["ic95_inf"], 0.0),
                                           np.maximum(res["ic95_sup"] - res["media"], 0.0)],
                    fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(list(nomes) if nomes is not None else [str(i) for i in x], rotation=60,
                           ha="right", fontsize=7)
    ax.axhline(res["limiar"], color="#C44E52", lw=1, ls="--", zorder=4)
    ax.set_ylabel("VIP (media das dobras)", fontsize=9)
    ax.set_title(f"VIP com estabilidade ({res['n_ajustes']} ajustes; azul = VIP > {res['limiar']:g} "
                 f"em >= {int(FREQ_ESTAVEL * 100)}% das dobras; laranja = instavel)", fontsize=9)
    from guaraci.idioma_figuras import traduzir_figura   # texto da figura no idioma da sessao
    traduzir_figura(fig)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    return caminho
