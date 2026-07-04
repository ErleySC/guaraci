# -*- coding: utf-8 -*-
"""Predicao em amostras desconhecidas a partir de um modelo salvo (.joblib).

Modulo PURO (numpy/pandas apenas) extraido de app_quimiometria.py para que
o app web e o CLI (guaraci.py) usem exatamente a mesma logica cientifica --
mesmo padrao de extracao da Fase H (chemometric_stats.py, dados_io.py etc.):
mover codigo coeso + reexportar por nome, nunca duplicar.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_CHAVES_PACOTE_REQUERIDAS = {
    "preprocessador", "pls_final", "label_binarizer", "wavenumbers"}


def validar_pacote_modelo(pkg: Dict) -> None:
    """Validacao minima de estrutura do pacote .joblib carregado.

    Nao valida CONTEUDO (nao ha como, com pickle) -- so' confirma que as
    chaves esperadas existem, para dar um erro claro em vez de deixar um
    AttributeError/KeyError confuso estourar mais adiante. Evita tambem
    tentar "usar" um pickle qualquer como se fosse um pacote de modelo.
    """
    if not _CHAVES_PACOTE_REQUERIDAS.issubset(pkg.keys()):
        raise ValueError(
            f"Modelo invalido: esperado as chaves {_CHAVES_PACOTE_REQUERIDAS}, "
            f"encontrado {set(pkg.keys())}")


def carregar_csv_predicao(caminho_ou_buffer) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Le um CSV de espectros novos (colunas=numeros de onda, sem coluna de
    classe) e separa as colunas numericas (espectro) das colunas de
    metadados (ex.: nome da amostra), se houver.

    Retorna (X, wavenumbers, metadados_df). `metadados_df` pode ter 0
    colunas (nenhuma coluna nao-numerica encontrada).
    """
    df = pd.read_csv(caminho_ou_buffer, sep=None, engine="python")
    num_cols = []
    for c in df.columns:
        try:
            float(c)
            num_cols.append(c)
        except ValueError:
            pass
    if not num_cols:
        raise ValueError(
            "Nenhuma coluna com nome numerico (numero de onda) encontrada. "
            "Garanta que os cabecalhos das colunas espectrais sejam numeros "
            "de onda (ex.: 4000.5, 4001.0...).")
    wn = np.array([float(c) for c in num_cols])
    X = df[num_cols].values.astype(float)
    meta_df = df.drop(columns=num_cols, errors="ignore")
    return X, wn, meta_df


def predizer_amostras(pkg: Dict, X_new_raw: np.ndarray,
                       wn_new: Optional[np.ndarray]) -> pd.DataFrame:
    """Aplica o pacote de modelo salvo a espectros novos.

    Interpola para o eixo de referencia do treino, aplica o pre-processador
    ajustado, calcula classe predita (scores PLS softmax-normalizados),
    e residuos T2/Q para diagnostico de dominio de aplicabilidade.
    Retorna um DataFrame com o diagnostico por amostra.
    """
    preproc = pkg["preprocessador"]
    pls     = pkg["pls_final"]
    lb      = pkg["label_binarizer"]
    wn_train = np.asarray(pkg["wavenumbers"], dtype=float)
    if wn_new is None:
        raise ValueError("wn_new nao pode ser None")
    wn_min   = float(pkg.get("wn_min", wn_train.min()))
    wn_max   = float(pkg.get("wn_max", wn_train.max()))

    # Eixo de referencia do treino (faixa usada durante o treinamento)
    mask_ref = (wn_train >= wn_min) & (wn_train <= wn_max)
    wn_ref   = wn_train[mask_ref]

    # Interpola espectros novos para o eixo de treino
    X_interp = np.zeros((X_new_raw.shape[0], len(wn_ref)))
    wn_new_f = wn_new.astype(float)
    for i in range(X_new_raw.shape[0]):
        X_interp[i] = np.interp(wn_ref, wn_new_f, X_new_raw[i].astype(float))

    # Aplica o pre-processamento do treino
    X_proc = preproc.transform(X_interp)

    # Scores PLS (aplica a centragem interna do modelo)
    T_new  = np.asarray(pls.transform(X_proc), dtype=float)
    P      = np.asarray(pls.x_loadings_, dtype=float)   # (p, k)
    P_T    = P.T                                          # (k, p)

    # Hotelling T2 -- mesma formula do pipeline (escalado pela variancia de treino)
    T_train = np.asarray(pls.x_scores_, dtype=float)
    var_t   = T_train.var(axis=0, ddof=1)
    var_t[var_t == 0] = 1.0
    T2_new  = np.sum((T_new ** 2) / var_t, axis=1)

    # Q-residuos -- mesma convencao do pipeline (X_proc nao subtraido)
    X_rec  = T_new @ P_T                                  # (n_new, p)
    Q_new  = np.sum((X_proc - X_rec) ** 2, axis=1)

    # UCL do pacote (gerado pelo pipeline v25+) ou fallback conservador
    t2_ucl = float(pkg.get("t2_ucl", np.percentile(
        np.sum((T_train ** 2) / var_t, axis=1), 95)))
    q_ucl  = float(pkg.get("q_ucl", np.percentile(Q_new, 99) * 1.5
                            if len(Q_new) > 0 else 1e6))

    # Classe predita via scores PLS softmax-normalizados
    Y_soft  = np.asarray(pls.predict(X_proc), dtype=float)
    Y_clip  = np.clip(Y_soft, 0.0, 1.0)
    totais  = Y_clip.sum(axis=1, keepdims=True)
    totais[totais < 1e-12] = 1.0
    Y_norm  = Y_clip / totais

    classes    = list(lb.classes_)
    idx_pred   = Y_norm.argmax(axis=1)
    classe_pred = [classes[i] if i < len(classes) else "?" for i in idx_pred]
    confianca  = Y_norm.max(axis=1)

    n = X_new_raw.shape[0]
    return pd.DataFrame({
        "amostra":    [f"S{i+1:03d}" for i in range(n)],
        "classe_pred": classe_pred,
        "confianca_%": np.round(confianca * 100, 1),
        "T2":          np.round(T2_new, 3),
        "T2_ucl":      round(t2_ucl, 3),
        "Q":           np.round(Q_new, 6),
        "Q_ucl":       round(q_ucl, 6),
        "T2_ok":       T2_new <= t2_ucl,
        "Q_ok":        Q_new  <= q_ucl,
        "aceito":      (T2_new <= t2_ucl) & (Q_new <= q_ucl),
    })
