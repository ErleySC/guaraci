# -*- coding: utf-8 -*-
"""alinhamento_retencao.py -- COW (Correlation Optimized Warping) para
cromatogramas (GC-MS/GC-FID/HPLC), Passo 150 da auditoria das 11
tecnicas analiticas (Fase D, 2026-09-04).

Referencia (confirmada no Crossref em 2026-09-04): Nielsen, N.-P. V.,
Carstensen, J. M. & Smedsgaard, J. (1998). "Aligning of single and
multiple wavelength chromatographic profiles for chemometric data
analysis using correlation optimised warping." Journal of
Chromatography A, 805(1-2), 17-35. DOI: 10.1016/S0021-9673(98)00021-1.

O QUE O ALGORITMO FAZ: alinha um cromatograma AMOSTRA a um cromatograma
REFERENCIA por deformacao linear POR PARTES (piecewise-linear warping).
Os dois sao divididos em `n_segmentos` (a referencia em segmentos de
tamanho FIXO; a amostra em segmentos cujo limite pode se deslocar
dentro de uma folga `slack`). Para cada escolha de limites, cada
segmento da amostra e' esticado/comprimido (interpolacao linear) para o
mesmo tamanho do segmento correspondente da referencia, e a correlacao
de Pearson entre os dois e' calculada. Programacao dinamica escolhe os
limites que MAXIMIZAM a soma das correlacoes segmento-a-segmento,
respeitando continuidade (o fim de um segmento e' o inicio do proximo).

LIMITACAO HONESTA: implementacao da formulacao classica (limites
inteiros na grade de amostragem, slack simetrico, correlacao de
Pearson como criterio) -- nao implementa as extensões multi-lambda
(multiplos comprimentos de onda simultaneos) nem pesos por segmento do
artigo original, que este projeto nao precisa (alinhamento de TIC de
GC-MS, um unico canal por cromatograma)."""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["ResultadoCOW", "cow", "AlinhadorCOW", "alinhar_com_referencia", "limpar_cache_cow"]


@dataclass
class ResultadoCOW:
    """`amostra_alinhada` tem o MESMO comprimento da referencia (cada
    segmento da amostra foi reamostrado para o tamanho do segmento
    correspondente da referencia -- e' assim que COW produz saida numa
    grade comum, pre-requisito para empilhar amostras numa tabela).
    `nos_amostra`/`nos_referencia` sao os indices de fronteira de
    segmento escolhidos (mesmo comprimento, `n_segmentos + 1`);
    `correlacao_media` e' a media das correlacoes por segmento no
    alinhamento otimo -- diagnostico de qualidade (1.0 = perfeito)."""
    amostra_alinhada: np.ndarray
    nos_amostra: np.ndarray
    nos_referencia: np.ndarray
    correlacao_media: float


def _correlacao_segura(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson, mas 0.0 (nem NaN) se um dos dois for constante --
    segmento sem variacao nao pode contribuir sinal de alinhamento, mas
    tambem nao pode quebrar o max() da DP com NaN."""
    if a.std() < 1e-300 or b.std() < 1e-300:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def _interpolar_para(segmento: np.ndarray, novo_tamanho: int) -> np.ndarray:
    if len(segmento) == novo_tamanho:
        return segmento
    x_orig = np.linspace(0.0, 1.0, len(segmento))
    x_novo = np.linspace(0.0, 1.0, novo_tamanho)
    return np.interp(x_novo, x_orig, segmento)


def _correlacoes_do_segmento(amostra: np.ndarray, seg_ref: np.ndarray,
                             p_prev: np.ndarray, p_atual: np.ndarray) -> np.ndarray:
    """Matriz C[k_atual, k_prev] com a correlacao de Pearson entre `seg_ref` e o trecho
    `amostra[p_prev:p_atual+1]` reamostrado (linear) para `len(seg_ref)` pontos. `-inf` onde o
    trecho teria menos de 2 pontos (p_atual - p_prev < 1). Mesmo calculo que a versao em laco
    (`_interpolar_para` + `_correlacao_segura`), feito de uma vez para todos os pares da janela:
    de ~1 s para ~5 ms por cromatograma de 2000 pontos, o que torna o COW viavel dentro da CV."""
    tam = len(seg_ref)
    x_novo = np.linspace(0.0, 1.0, tam)
    P = p_prev.astype(float)[None, :, None]
    A = p_atual.astype(float)[:, None, None]
    pos = P + (A - P) * x_novo[None, None, :]
    # interpolacao linear por indice (equivale a np.interp na grade inteira, ~3x mais rapida aqui);
    # os pares invalidos so' precisam de indices dentro do vetor (sao mascarados no fim)
    pos = np.clip(pos, 0.0, len(amostra) - 1.0)
    i0 = np.minimum(pos.astype(np.intp), len(amostra) - 2)
    fr = pos - i0
    valores = amostra[i0] * (1.0 - fr) + amostra[i0 + 1] * fr
    vc = valores - valores.mean(axis=2, keepdims=True)
    rc = seg_ref - seg_ref.mean()
    sv = np.sqrt((vc ** 2).sum(axis=2))
    sr = float(np.sqrt((rc ** 2).sum()))
    # std < 1e-300 (como em `_correlacao_segura`) equivale a soma de quadrados ~ 0
    limite = 1e-300 * np.sqrt(tam)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (vc @ rc) / (sv * sr)
    corr = np.where((sv < limite) | (sr < limite) | ~np.isfinite(corr), 0.0, corr)
    invalido = (p_atual[:, None] - p_prev[None, :]) < 1
    return np.where(invalido, -np.inf, corr)


def cow(referencia: np.ndarray, amostra: np.ndarray, n_segmentos: int,
        slack: int) -> ResultadoCOW:
    """Alinha `amostra` a `referencia` por COW. `n_segmentos` >= 2;
    `slack` >= 0 (folga em NUMERO DE PONTOS da grade da amostra ao
    redor da posicao nominal de cada limite intermediario).

    Levanta `ValueError` se `referencia`/`amostra` tiverem menos de
    `n_segmentos + 1` pontos (nao ha' como formar segmentos de
    comprimento >= 1)."""
    referencia = np.asarray(referencia, dtype=float)
    amostra = np.asarray(amostra, dtype=float)
    n_ref, n_am = len(referencia), len(amostra)
    if n_segmentos < 2:
        raise ValueError(f"n_segmentos={n_segmentos} -- precisa ser >= 2")
    if n_ref < n_segmentos + 1 or n_am < n_segmentos + 1:
        raise ValueError(
            f"referencia ({n_ref} pontos) ou amostra ({n_am} pontos) "
            f"curta demais para {n_segmentos} segmentos")

    nos_ref = np.round(np.linspace(0, n_ref - 1, n_segmentos + 1)).astype(int)
    nos_nominais_am = np.round(np.linspace(0, n_am - 1, n_segmentos + 1)).astype(int)

    # Janela de posicoes candidatas por no intermediario (0 e N sao fixos).
    janelas: List[np.ndarray] = [np.array([0])]
    for i in range(1, n_segmentos):
        lo = max(1, nos_nominais_am[i] - slack)
        hi = min(n_am - 2, nos_nominais_am[i] + slack)
        janelas.append(np.arange(lo, hi + 1))
    janelas.append(np.array([n_am - 1]))

    # dp[i] indexado pela POSICAO na janela do no i (nao pelo valor bruto).
    dp: List[np.ndarray] = [np.zeros(1)]
    voltar: List[np.ndarray] = [np.zeros(1, dtype=int)]

    for i in range(1, n_segmentos + 1):
        seg_ref = referencia[nos_ref[i - 1]:nos_ref[i] + 1]
        corr = _correlacoes_do_segmento(amostra, seg_ref, janelas[i - 1], janelas[i])
        total = dp[i - 1][None, :] + corr           # [k_atual, k_prev]
        origem = np.argmax(total, axis=1)           # 1o maximo, como o laco com ">"
        atual = total[np.arange(len(janelas[i])), origem]
        dp.append(atual)
        voltar.append(origem)

    # Backtrack a partir do (unico) no final.
    nos_escolhidos = [0] * (n_segmentos + 1)
    k = 0   # janelas[n_segmentos] tem 1 elemento so'
    nos_escolhidos[n_segmentos] = janelas[n_segmentos][0]
    for i in range(n_segmentos, 0, -1):
        k = voltar[i][k]
        nos_escolhidos[i - 1] = janelas[i - 1][k]

    nos_amostra = np.asarray(nos_escolhidos, dtype=int)
    correlacoes: List[float] = []
    pedacos: List[np.ndarray] = []
    for i in range(1, n_segmentos + 1):
        tam_ref_seg = nos_ref[i] - nos_ref[i - 1] + 1
        seg_am = amostra[nos_amostra[i - 1]:nos_amostra[i] + 1]
        seg_am_interp = _interpolar_para(seg_am, tam_ref_seg)
        seg_ref = referencia[nos_ref[i - 1]:nos_ref[i] + 1]
        correlacoes.append(_correlacao_segura(seg_am_interp, seg_ref))
        # Evita duplicar o ponto de fronteira compartilhado entre segmentos.
        pedacos.append(seg_am_interp[:-1] if i < n_segmentos else seg_am_interp)

    amostra_alinhada = np.concatenate(pedacos)
    return ResultadoCOW(
        amostra_alinhada=amostra_alinhada,
        nos_amostra=nos_amostra, nos_referencia=nos_ref,
        correlacao_media=float(np.mean(correlacoes)))


def _correlacao_media_pares(X: np.ndarray) -> float:
    """Media das correlacoes de Pearson entre todos os pares de linhas (diagnostico de alinhamento)."""
    n = X.shape[0]
    if n < 2:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.nan_to_num(np.corrcoef(X), nan=0.0)
    return float((c.sum() - np.trace(c)) / (n * (n - 1)))


# Cache do alinhamento de UMA amostra a UMA referencia. O resultado so' depende da referencia, da
# amostra e dos parametros (nao de y, de dobra ou de semente), mas a validacao repete o mesmo par
# muitas vezes: cada permutacao do teste de permutacao, cada repeticao da CV aninhada, cada
# bootstrap refaz as mesmas dobras. Sem o cache o COW dentro da CV multiplicava o tempo por ~10;
# com ele cada par e' calculado uma vez. Chave = hash do conteudo (nao o id do objeto), logo uma
# referencia diferente (outro treino) nunca reaproveita o alinhamento de outra.
_CACHE_COW: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_CACHE_COW_BYTES = [0]
_CACHE_COW_LIMITE_BYTES = 256 * 1024 * 1024
_CACHE_COW_LOCK = threading.Lock()


def limpar_cache_cow() -> None:
    """Esvazia o cache de alinhamentos (testes; ou para liberar memoria)."""
    with _CACHE_COW_LOCK:
        _CACHE_COW.clear()
        _CACHE_COW_BYTES[0] = 0


def _resumo_hash(v: np.ndarray) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(v, dtype=float).tobytes(), digest_size=16).digest()


def alinhar_com_referencia(referencia: np.ndarray, X: np.ndarray, n_segmentos: int,
                           folga: int) -> np.ndarray:
    """Alinha cada linha de `X` (mesma grade da `referencia`) por COW. A linha identica a
    referencia passa sem mudanca (o alinhamento dela consigo mesma e' a identidade). O alinhamento
    de cada par (referencia, linha) e' calculado uma vez e reaproveitado (ver `_CACHE_COW`)."""
    referencia = np.asarray(referencia, dtype=float)
    X = np.asarray(X, dtype=float)
    saida = np.empty_like(X)
    h_ref = _resumo_hash(referencia)
    for i in range(X.shape[0]):
        if np.array_equal(X[i], referencia):
            saida[i] = X[i]
            continue
        chave = (h_ref, _resumo_hash(X[i]), int(n_segmentos), int(folga))
        with _CACHE_COW_LOCK:
            achado = _CACHE_COW.get(chave)
            if achado is not None:
                _CACHE_COW.move_to_end(chave)
        if achado is None:
            achado = cow(referencia, X[i], n_segmentos, folga).amostra_alinhada
            achado.setflags(write=False)
            with _CACHE_COW_LOCK:
                if chave not in _CACHE_COW:
                    _CACHE_COW[chave] = achado
                    _CACHE_COW_BYTES[0] += achado.nbytes
                    while _CACHE_COW_BYTES[0] > _CACHE_COW_LIMITE_BYTES and _CACHE_COW:
                        _k, velho = _CACHE_COW.popitem(last=False)
                        _CACHE_COW_BYTES[0] -= velho.nbytes
        saida[i] = achado
    return saida


class AlinhadorCOW(BaseEstimator, TransformerMixin):
    """Alinhamento de retencao COW como etapa do pre-processamento (sklearn).

    `fit` escolhe a REFERENCIA so' com as linhas que recebe (a de maior correlacao media com as
    demais, a mesma regra de `tecnicas_avancadas.rodar_cow`) e fixa os segmentos dela
    (`nos_referencia_`). `transform` alinha qualquer linha a ESSA referencia, com a mesma grade:
    dentro de uma Pipeline, cada dobra da validacao cruzada escolhe a referencia so' no treino
    da dobra e a aplica as amostras de teste da dobra, e o modelo exportado leva a referencia
    para alinhar amostras novas do mesmo jeito. Nada do teste entra na escolha da referencia.

    Ordem das colunas = ordem do tempo de retencao, com passo constante (cromatograma na
    grade comum); a saida tem o mesmo numero de colunas da entrada.

    Atributos apos `fit`: `referencia_`, `indice_referencia_` (linha dentro dos dados do fit),
    `nos_referencia_`; apos `fit_transform` tambem `correlacao_antes_`/`correlacao_depois_`
    (correlacao media par a par dos dados do fit, antes e depois do alinhamento)."""

    def __init__(self, n_segmentos: int = 30, folga: int = 10):
        self.n_segmentos = n_segmentos
        self.folga = folga

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("COW precisa de pelo menos 2 cromatogramas (linhas) para escolher a referencia.")
        ns, fo = int(self.n_segmentos), int(self.folga)
        if ns < 2:
            raise ValueError(f"alinhamento_segmentos={ns}: precisa ser >= 2")
        if fo < 0:
            raise ValueError(f"alinhamento_folga={fo}: precisa ser >= 0")
        if X.shape[1] < ns + 1:
            raise ValueError(f"o cromatograma tem {X.shape[1]} pontos, poucos para {ns} segmentos "
                             f"(reduza alinhamento_segmentos)")
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.nan_to_num(np.corrcoef(X), nan=0.0)
        self.indice_referencia_ = int(np.argmax(c.sum(axis=1)))
        self.referencia_ = X[self.indice_referencia_].copy()
        self.nos_referencia_ = np.round(np.linspace(0, X.shape[1] - 1, ns + 1)).astype(int)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"o alinhador foi ajustado com {self.n_features_in_} pontos por cromatograma, "
                             f"mas recebeu {X.shape[1]}")
        return alinhar_com_referencia(self.referencia_, X, int(self.n_segmentos), int(self.folga))

    def fit_transform(self, X, y=None, **fit_params):
        X = np.asarray(X, dtype=float)
        self.fit(X, y)
        Xa = self.transform(X)
        self.correlacao_antes_ = _correlacao_media_pares(X)
        self.correlacao_depois_ = _correlacao_media_pares(Xa)
        return Xa
