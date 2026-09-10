# -*- coding: utf-8 -*-
"""hsi_multiway.py -- Decomposicao multiway do cubo hiperespectral: PARAFAC
(nao-supervisionado) e N-PLS (supervisionado), Bloco 15.

REFERENCIAS (verificadas no Crossref em 2026-09-04):
  - PARAFAC: Bro, R. (1997). "PARAFAC. Tutorial and applications."
    Chemometrics and Intelligent Laboratory Systems, 38(2), 149-171.
    DOI: 10.1016/S0169-7439(97)00032-4.
  - N-PLS: Bro, R. (1996). "Multiway calibration. Multilinear PLS."
    Journal of Chemometrics, 10(1), 47-61.
    DOI: 10.1002/(SICI)1099-128X(199601)10:1<47::AID-CEM400>3.0.CO;2-C.

PARAFAC usa `tensorly` (BSD -- compativel com GPL-3.0-or-later deste
projeto, avaliada ANTES de escrever decomposicao tensorial do zero, mesmo
principio do Bloco 18): biblioteca madura e dedicada a decomposicao
tensorial, reimplementar ALS de N-way a mao teria risco de bug muito
maior que reusar uma implementacao testada por terceiros para essa parte.
N-PLS NAO tem biblioteca Python madura equivalente (nao esta' em
`tensorly.regression`, que nao cobre PLS multiway supervisionado) --
implementado aqui seguindo o algoritmo NIPALS generalizado do artigo
original.

PROBLEMA DE ENGENHARIA RESOLVIDO ANTES DA DECOMPOSICAO EM SI: PARAFAC/
N-PLS exigem um array N-way REGULAR (mesma forma em todo modo) -- mas
gravacoes HSI reais de objetos fisicos DIFERENTES quase nunca tem a MESMA
resolucao espacial (ver `hsi_pixels.py`: Kaki 64x64, Avocado/VIS
~286x294). `construir_tensor_amostras` resolve isso reduzindo a ROI de
CADA gravacao a uma grade espacial FIXA (media de bloco, nao redimensiona
por interpolacao) antes de empilhar -- assim o modo "espacial" tem o
MESMO significado relativo (ex.: bloco [0,0] = canto superior-esquerdo da
bounding box da ROI) em toda amostra, premissa necessaria para o modo
espacial do PARAFAC/N-PLS fazer sentido fisico entre objetos.

GROUP-AWARE: cada LINHA do tensor de amostras e' UMA GRAVACAO (nao um
pixel) -- o mesmo `group_id`/objeto fisico pode aparecer em mais de uma
linha (ex. frente + costas da mesma fruta, ver `hsi_pixels.py`). Nenhuma
funcao deste modulo faz split; quem monta o split (`comparar_npls_vs_
pixelwise` ou o chamador) DEVE usar `StableStratifiedGroupKFold` sobre o
MESMO array de `grupos` alinhado as linhas do tensor -- nunca amostragem
aleatoria simples. Testado por propriedade dedicada.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelBinarizer

from guaraci.chemometric_stats import expandir_binario_um_quente

__all__ = [
    "construir_tensor_amostras",
    "ParafacHSIResultado",
    "parafac_hsi",
    "NPLS",
    "NPLSClassifier",
    "comparar_npls_vs_pixelwise",
]


def _grade_bloco_medio(cubo: np.ndarray, mascara: np.ndarray,
                        n_linhas: int, n_colunas: int) -> np.ndarray:
    """Reduz a ROI (bounding box da `mascara`) de UM cubo a uma grade
    espacial fixa `(n_linhas, n_colunas, n_bandas)` por media de bloco dos
    pixels da ROI dentro de cada celula. Bloco sem nenhum pixel de ROI
    (bounding box irregular) cai para a media global da ROI -- nunca
    NaN/zero artificial."""
    cubo = np.asarray(cubo, dtype=float)
    mascara = np.asarray(mascara, dtype=bool)
    if cubo.ndim != 3:
        raise ValueError(f"cubo precisa ser 3D (altura, largura, bandas), "
                          f"recebeu shape {cubo.shape}")
    if mascara.shape != cubo.shape[:2]:
        raise ValueError(f"mascara {mascara.shape} nao bate com o cubo "
                          f"{cubo.shape[:2]}")
    if not mascara.any():
        raise ValueError("mascara vazia -- nenhum pixel de ROI")

    linhas_idx, colunas_idx = np.nonzero(mascara)
    r0, r1 = int(linhas_idx.min()), int(linhas_idx.max()) + 1
    c0, c1 = int(colunas_idx.min()), int(colunas_idx.max()) + 1
    sub_cubo = cubo[r0:r1, c0:c1, :]
    sub_mask = mascara[r0:r1, c0:c1]
    n_bandas = cubo.shape[2]

    bordas_l = np.linspace(0, sub_cubo.shape[0], n_linhas + 1).astype(int)
    bordas_c = np.linspace(0, sub_cubo.shape[1], n_colunas + 1).astype(int)
    media_global_roi = sub_cubo[sub_mask].mean(axis=0)

    grade = np.empty((n_linhas, n_colunas, n_bandas))
    for i in range(n_linhas):
        for j in range(n_colunas):
            bloco = sub_cubo[bordas_l[i]:bordas_l[i + 1], bordas_c[j]:bordas_c[j + 1], :]
            bloco_mask = sub_mask[bordas_l[i]:bordas_l[i + 1], bordas_c[j]:bordas_c[j + 1]]
            grade[i, j, :] = (bloco[bloco_mask].mean(axis=0) if bloco_mask.any()
                              else media_global_roi)
    return grade


def construir_tensor_amostras(cubos: Sequence[np.ndarray],
                               mascaras: Sequence[np.ndarray], *,
                               n_linhas: int = 6, n_colunas: int = 6
                               ) -> np.ndarray:
    """Constroi um tensor 3-way REGULAR `(n_amostras, n_linhas*n_colunas,
    n_bandas)` a partir de N gravacoes (cubo + mascara de ROI) de
    resolucao espacial possivelmente DIFERENTE entre si -- uma "amostra"
    (linha do tensor) por gravacao, via `_grade_bloco_medio`. Pre-
    requisito de forma regular para `parafac_hsi`/`NPLS`."""
    if len(cubos) != len(mascaras):
        raise ValueError(f"cubos ({len(cubos)}) e mascaras ({len(mascaras)}) "
                          f"com comprimentos diferentes")
    if len(cubos) == 0:
        raise ValueError("nenhuma gravacao fornecida")
    grades = [_grade_bloco_medio(c, m, n_linhas, n_colunas)
              .reshape(n_linhas * n_colunas, -1)
              for c, m in zip(cubos, mascaras)]
    return np.stack(grades, axis=0)


@dataclass
class ParafacHSIResultado:
    """`fator_amostra` (n_amostras, R), `fator_espacial` (n_posicoes, R),
    `fator_espectral` (n_bandas, R) -- R = numero de componentes.
    `erro_reconstrucao_relativo` = ||tensor - reconstrucao|| / ||tensor||
    (Frobenius) -- nunca reportar os fatores sem esse numero junto."""
    fator_amostra: np.ndarray
    fator_espacial: np.ndarray
    fator_espectral: np.ndarray
    erro_reconstrucao_relativo: float


def parafac_hsi(tensor: np.ndarray, n_componentes: int, *,
                 max_iter: int = 200, seed: int = 0) -> ParafacHSIResultado:
    """PARAFAC/CP (Bro, 1997) do tensor `(n_amostras, n_posicoes,
    n_bandas)` (ver `construir_tensor_amostras`) via `tensorly`. Decompoe
    em 3 fatores -- amostra, espacial, espectral -- cada componente `r`
    e' um "source" caracterizado por um padrao espacial E um espectro
    puros, expresso em graus variaveis em cada amostra (fator_amostra)."""
    from tensorly.decomposition import parafac
    from tensorly.cp_tensor import cp_to_tensor

    tensor = np.asarray(tensor, dtype=float)
    if tensor.ndim != 3:
        raise ValueError(f"tensor precisa ser 3-way, recebeu shape {tensor.shape}")
    norma = float(np.linalg.norm(tensor))
    if norma < 1e-300:
        raise ValueError("tensor e' (quase) todo zero -- nada para decompor")

    cp = parafac(tensor, rank=int(n_componentes), n_iter_max=max_iter,
                 init="svd", random_state=seed)
    reconstrucao = cp_to_tensor(cp)
    erro = float(np.linalg.norm(tensor - reconstrucao) / norma)
    return ParafacHSIResultado(
        fator_amostra=cp.factors[0], fator_espacial=cp.factors[1],
        fator_espectral=cp.factors[2], erro_reconstrucao_relativo=erro)


class NPLS:
    """N-PLS / Multilinear PLS (Bro, 1996) para um tensor de entrada
    3-way `X` (n_amostras, n_posicoes, n_bandas) e alvo `Y` (n_amostras,
    n_targets). Cada componente encontra um par de pesos rank-1 (peso
    espacial `w_J`, peso espectral `w_K`) que, combinados, maximizam a
    covariancia do escore de amostra resultante `t` com `Y` -- via SVD do
    tensor ponderado por `u` (o "truque de Kroonenberg", generalizacao
    direta de como o PLS 2-way encontra seu peso `w = X^T u`). Deflaciona
    X (rank-1 3-way) e Y (bilinear, como no PLS2 comum) e repete."""

    def __init__(self, n_componentes: int = 2, max_iter: int = 200, tol: float = 1e-8):
        self.n_componentes = n_componentes
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "NPLS":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if X.ndim != 3:
            raise ValueError(f"X precisa ser 3-way, recebeu shape {X.shape}")
        I, J, K = X.shape
        if Y.shape[0] != I:
            raise ValueError(f"X tem {I} amostras, Y tem {Y.shape[0]}")

        self.mean_X_ = X.mean(axis=0)
        self.mean_Y_ = Y.mean(axis=0)
        Xr = X - self.mean_X_[None, :, :]
        Yr = Y - self.mean_Y_[None, :]

        n_comp = int(max(1, min(self.n_componentes, J, K, I - 1)))
        ws_J: List[np.ndarray] = []
        ws_K: List[np.ndarray] = []
        ts: List[np.ndarray] = []
        qs: List[np.ndarray] = []

        for _ in range(n_comp):
            u = Yr[:, int(np.argmax(np.var(Yr, axis=0)))].copy()
            if np.linalg.norm(u) < 1e-300:
                u = Yr[:, 0].copy()
            t_ant = None
            w_J = w_K = None
            for _it in range(self.max_iter):
                Z = np.tensordot(u, Xr, axes=(0, 0))          # (J, K)
                Uz, _Sz, Vtz = np.linalg.svd(Z, full_matrices=False)
                w_J, w_K = Uz[:, 0], Vtz[0, :]
                nJ, nK = np.linalg.norm(w_J), np.linalg.norm(w_K)
                w_J = w_J / nJ if nJ > 1e-300 else w_J
                w_K = w_K / nK if nK > 1e-300 else w_K
                t = np.einsum("ijk,j,k->i", Xr, w_J, w_K)
                tt = float(t @ t)
                q = (Yr.T @ t) / tt if tt > 1e-300 else np.zeros(Yr.shape[1])
                qq = float(q @ q)
                u_novo = (Yr @ q) / qq if qq > 1e-300 else u
                if t_ant is not None and np.linalg.norm(t - t_ant) < self.tol:
                    u = u_novo
                    break
                t_ant, u = t, u_novo

            ws_J.append(w_J); ws_K.append(w_K); ts.append(t); qs.append(q)
            Xr = Xr - np.einsum("i,j,k->ijk", t, w_J, w_K)
            Yr = Yr - np.outer(t, q)

        self.pesos_espaciais_ = np.column_stack(ws_J)     # (J, n_comp)
        self.pesos_espectrais_ = np.column_stack(ws_K)    # (K, n_comp)
        self.T_ = np.column_stack(ts)                      # (I, n_comp)
        self.q_ = np.column_stack(qs)                       # (n_targets, n_comp)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projeta amostras NOVAS nos componentes ja' ajustados --
        deflaciona sequencialmente com os pesos ARMAZENADOS do treino
        (nunca reajusta), mesmo principio de `PLSRegression.transform`
        generalizado para 3-way."""
        X = np.asarray(X, dtype=float)
        Xr = X - self.mean_X_[None, :, :]
        n_comp = self.pesos_espaciais_.shape[1]
        T = np.empty((X.shape[0], n_comp))
        for c in range(n_comp):
            w_J, w_K = self.pesos_espaciais_[:, c], self.pesos_espectrais_[:, c]
            t = np.einsum("ijk,j,k->i", Xr, w_J, w_K)
            T[:, c] = t
            Xr = Xr - np.einsum("i,j,k->ijk", t, w_J, w_K)
        return T

    def predict(self, X: np.ndarray) -> np.ndarray:
        T = self.transform(X)
        return T @ self.q_.T + self.mean_Y_[None, :]


class NPLSClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper de classificacao para `NPLS`, mesmo padrao de
    `avaliacao_modelos.PLSDAClassifier` (Y one-hot via `LabelBinarizer`,
    predicao por argmax) -- so' que sobre um tensor 3-way de entrada em
    vez de uma matriz 2D."""

    def __init__(self, n_componentes: int = 2):
        self.n_componentes = n_componentes

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NPLSClassifier":
        self._lb = LabelBinarizer()
        Y_bin = np.asarray(self._lb.fit_transform(y))
        Y_bin = expandir_binario_um_quente(Y_bin)
        self._npls = NPLS(n_componentes=self.n_componentes).fit(X, Y_bin)
        self.classes_ = self._lb.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Y_hat = self._npls.predict(X)
        return self.classes_[np.argmax(Y_hat, axis=1)]


def comparar_npls_vs_pixelwise(
        cubos: Sequence[np.ndarray], mascaras: Sequence[np.ndarray],
        rotulos: Sequence[str], grupos: Sequence[str], *,
        n_linhas_grade: int = 6, n_colunas_grade: int = 6,
        n_componentes: int = 2, n_splits: int = 3, seed: int = 42,
        max_pixels_por_gravacao: Optional[int] = 2000,
        ) -> Dict[str, object]:
    """Compara N-PLS multiway (feature = grade espacial fixa por
    gravacao) contra o PLS-DA por-pixel ja' existente
    (`hsi_classification.fit_predict_pixel_plsda`), sob o MESMO split
    group-aware por objeto fisico (`StableStratifiedGroupKFold` sobre
    `grupos`, alinhado 1:1 as gravacoes -- nunca reamostragem simples).

    Ambos avaliados no NIVEL DE OBJETO (voto majoritario quando um
    group_id tem mais de uma gravacao no fold de teste, ex. frente +
    costas) -- comparacao no MESMO nivel de agregacao, honesta.

    Retorna dict com `balanced_accuracy_npls`, `balanced_accuracy_pixelwise`
    (media entre folds) e `por_fold` (lista de dicts por fold)."""
    from guaraci.hsi_classification import (fit_predict_pixel_plsda,
                                            aggregate_predictions_by_object,
                                            ObjectAggregationResult)
    from guaraci.hsi_pixels import build_pixel_dataset
    from guaraci.validacao_estatistica import StableStratifiedGroupKFold

    rotulos_arr = np.asarray(rotulos, dtype=str)
    grupos_arr = np.asarray(grupos, dtype=str)
    n = len(cubos)
    if not (len(mascaras) == len(rotulos_arr) == len(grupos_arr) == n):
        raise ValueError("cubos/mascaras/rotulos/grupos com comprimentos diferentes")

    tensor = construir_tensor_amostras(cubos, mascaras,
                                        n_linhas=n_linhas_grade, n_colunas=n_colunas_grade)

    splitter = StableStratifiedGroupKFold(n_splits=n_splits, seed=seed)
    folds = list(splitter.split(np.zeros(n), rotulos_arr, groups=grupos_arr))

    resultados_fold: List[Dict[str, object]] = []
    for idx_treino, idx_teste in folds:
        grupos_treino_set = set(grupos_arr[idx_treino])
        grupos_teste_set = set(grupos_arr[idx_teste])
        if grupos_treino_set & grupos_teste_set:
            raise RuntimeError(
                "vazamento de group_id entre treino/teste -- nunca deveria "
                "acontecer com StableStratifiedGroupKFold")

        # --- N-PLS: 1 predicao por gravacao, agregada por objeto -----
        clf_npls = NPLSClassifier(n_componentes=n_componentes)
        clf_npls.fit(tensor[idx_treino], rotulos_arr[idx_treino])
        pred_npls_gravacao = clf_npls.predict(tensor[idx_teste])
        agregados_npls = aggregate_predictions_by_object(
            pred_npls_gravacao, grupos_arr[idx_teste])
        y_real_objeto: Dict[str, str] = {}
        for gid in grupos_arr[idx_teste]:
            if gid not in y_real_objeto:
                y_real_objeto[gid] = rotulos_arr[idx_teste][
                    list(grupos_arr[idx_teste]).index(gid)]
        objetos_teste = sorted(agregados_npls)
        y_true_npls = [y_real_objeto[g] for g in objetos_teste]
        y_pred_npls = [agregados_npls[g].classe_predita for g in objetos_teste]
        bal_npls = balanced_accuracy_score(y_true_npls, y_pred_npls)

        # --- PLS-DA por pixel (ja existente) --------------------------
        X_treino, y_treino_px, g_treino_px = build_pixel_dataset(
            [cubos[i] for i in idx_treino], [mascaras[i] for i in idx_treino],
            [grupos[i] for i in idx_treino], [rotulos[i] for i in idx_treino],
            max_pixels_por_gravacao=max_pixels_por_gravacao, seed=seed)
        X_teste, y_teste_px, g_teste_px = build_pixel_dataset(
            [cubos[i] for i in idx_teste], [mascaras[i] for i in idx_teste],
            [grupos[i] for i in idx_teste], [rotulos[i] for i in idx_teste],
            max_pixels_por_gravacao=max_pixels_por_gravacao, seed=seed)
        resultado_px = fit_predict_pixel_plsda(
            X_treino, y_treino_px, g_treino_px, X_teste, g_teste_px, seed=seed)
        predicoes_objeto_px: Dict[str, ObjectAggregationResult] = \
            resultado_px["predicoes_objeto"]  # type: ignore[assignment]
        y_real_objeto_px = {gid: y_teste_px[list(g_teste_px).index(gid)]
                            for gid in np.unique(g_teste_px)}
        objetos_teste_px = sorted(predicoes_objeto_px)
        y_true_px = [y_real_objeto_px[g] for g in objetos_teste_px]
        y_pred_px = [predicoes_objeto_px[g].classe_predita
                     for g in objetos_teste_px]
        bal_px = balanced_accuracy_score(y_true_px, y_pred_px)

        resultados_fold.append({
            "balanced_accuracy_npls": float(bal_npls),
            "balanced_accuracy_pixelwise": float(bal_px),
            "n_objetos_teste": len(objetos_teste),
        })

    return {
        "balanced_accuracy_npls": float(np.mean(
            [r["balanced_accuracy_npls"] for r in resultados_fold])),
        "balanced_accuracy_pixelwise": float(np.mean(
            [r["balanced_accuracy_pixelwise"] for r in resultados_fold])),
        "por_fold": resultados_fold,
    }
