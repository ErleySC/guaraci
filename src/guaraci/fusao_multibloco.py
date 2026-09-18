# -*- coding: utf-8 -*-
"""fusao_multibloco.py -- Fusao multibloco de baixo nivel (Grupo 2, prova
de conceito).

POR QUE ESTE MODULO EXISTE
---------------------------
Hoje cada tecnica espectroscopica roda como um bloco isolado: FT-NIR,
MIR, EEM etc. cada uma produz sua propria matriz `X` e seu proprio
modelo. "Fusao multibloco" significa combinar >=2 blocos medidos na
MESMA amostra fisica num unico modelo. Este modulo implementa o nivel 1
(o unico que corresponde a' literatura de "fusao multibloco"
propriamente dita -- PLS multibloco/MB-PLS, SO-PLS, ComDim; ver
`docs/ESCOPO_FUSAO_MULTIBLOCO_E_MSPC.md` secao 1.1): concatenacao de
baixo nivel, cada bloco pre-processado SEPARADAMENTE antes de
concatenar.

Par de prova de conceito: dataset publico Mendeley `10.17632/ctgg7k4m5g.2`
(Ottaway et al. 2021), NIR 8mm + MIR -- as MESMAS 100 garrafas de oleo
medidas nas duas tecnicas, alinhadas linha-a-linha (ver
`tests/test_fusao_multibloco.py` para a confirmacao por inspecao direta
do dado real).

POR QUE PRE-PROCESSAR CADA BLOCO SEPARADO, NUNCA A MATRIZ JA CONCATENADA
-------------------------------------------------------------------------
SG (Savitzky-Golay) e' uma convolucao sobre canais ADJACENTES -- assume
um eixo espectral continuo. Concatenar NIR (3899-7498 cm-1) com MIR
(699-3999 cm-1) produz um eixo com um SALTO na fronteira entre blocos;
aplicar SG/MSC sobre a matriz ja concatenada passaria a janela de
convolucao POR CIMA dessa fronteira, corrompendo os ultimos canais de um
bloco e os primeiros do outro com informacao da tecnica errada. Por isso
`build_multiblock_dataset` sempre ajusta e aplica um pre-processador POR
BLOCO, e so' concatena depois.

GUARDA CONTRA DESALINHAMENTO SILENCIOSO
-----------------------------------------
`build_multiblock_dataset` exige que todos os blocos tenham o MESMO
`n_amostras` -- levanta `BlockMismatchError` (nunca concatena
silenciosamente blocos de tamanhos diferentes, que produziria um
`hstack` com linhas de amostras FISICAS diferentes emparelhadas por
acidente de indice).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict

from guaraci.chemometric_stats import rmse_flat
from guaraci.config import Config
from guaraci.preprocessamento import build_preprocessor

__all__ = [
    "BlockMismatchError",
    "build_multiblock_dataset",
    "RegressionFitResult",
    "fit_evaluate_pls_regression",
]


class BlockMismatchError(ValueError):
    """Blocos com `n_amostras` diferente -- nunca concatenados
    silenciosamente (ver docstring do modulo)."""


def build_multiblock_dataset(
    blocos: Dict[str, np.ndarray],
    cfg_por_bloco: Optional[Dict[str, Config]] = None,
    indices_ajuste: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Tuple[int, int]]]:
    """Concatena blocos pre-processados SEPARADAMENTE (fusao de nivel 1).

    Parameters
    ----------
    blocos: nome_da_tecnica -> X bruto (n_amostras x p_tecnica). MESMA
        ordem de linha entre todos os blocos (mesma amostra fisica na
        mesma posicao) -- responsabilidade do CHAMADOR garantir isso
        (ver docstring do modulo).
    cfg_por_bloco: `Config` por tecnica -- so' os campos de
        pre-processamento importam (`default_preprocessing`/`sg_*`/
        `airpls_lam`). Bloco sem entrada usa `Config()` default
        (msc_sg_mc).
    indices_ajuste: indices usados para AJUSTAR (fit) cada
        pre-processador -- devem ser os indices de CALIBRACAO, nunca os
        de validacao/teste (MSC/StandardScaler sao stateful; ajustar com
        dado de validacao vazaria informacao do teste para o treino).
        `None` ajusta com todas as amostras passadas.

    Returns
    -------
    (X_fundida, preprocessadores, fatias): `X_fundida` e' o hstack dos
    blocos ja processados; `preprocessadores[nome]` e' o `Pipeline`
    ajustado daquele bloco (reutilizavel para transformar amostras
    novas); `fatias[nome]` e' o par `(inicio, fim)` de colunas de
    `X_fundida` que pertencem aquele bloco.
    """
    if not blocos:
        raise ValueError("build_multiblock_dataset precisa de >=1 bloco")
    n_por_bloco = {nome: int(np.asarray(X).shape[0])
                   for nome, X in blocos.items()}
    if len(set(n_por_bloco.values())) > 1:
        raise BlockMismatchError(
            "Blocos com n_amostras diferente -- nunca concatenados "
            f"silenciosamente (guarda explicita): {n_por_bloco}")
    n = next(iter(n_por_bloco.values()))
    idx_fit = (np.asarray(indices_ajuste) if indices_ajuste is not None
               else np.arange(n))

    blocos_proc = []
    preprocessadores: Dict[str, Any] = {}
    fatias: Dict[str, Tuple[int, int]] = {}
    col_atual = 0
    for nome, X_bruto in blocos.items():
        X = np.asarray(X_bruto, dtype=float)
        cfg_bloco = (cfg_por_bloco or {}).get(nome) or Config()
        prep = build_preprocessor(cfg_bloco)
        prep.fit(X[idx_fit])
        X_proc = np.asarray(prep.transform(X), dtype=float)
        blocos_proc.append(X_proc)
        preprocessadores[nome] = prep
        fatias[nome] = (col_atual, col_atual + X_proc.shape[1])
        col_atual += X_proc.shape[1]

    X_fundida = np.hstack(blocos_proc)
    return X_fundida, preprocessadores, fatias


@dataclass
class RegressionFitResult:
    """Resultado de `fit_evaluate_pls_regression` -- mesmas grandezas
    (RMSEP/R2cal/R2val) ja reportadas para NIR/MIR/Raman individuais em
    `docs/VALIDACAO_PUBLICA.md` §2/2b, para comparacao lado a lado
    honesta (nunca so' o numero da fusao)."""

    n_lv: int
    rmsep: float
    r2cal: float
    r2val: float
    n_cal: int
    n_val: int


def fit_evaluate_pls_regression(
    X_cal: np.ndarray, y_cal: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    max_lvs: int = 15, n_splits_cv: int = 5, seed: int = 0,
) -> RegressionFitResult:
    """PLS de regressao com selecao de LVs por CV (RMSECV minimo),
    ajustado em `(X_cal, y_cal)` (ja pre-processado -- ver
    `build_multiblock_dataset`), avaliado em `(X_val, y_val)` nunca visto
    pelo ajuste. Protocolo deliberadamente simples (KFold comum, sem
    group-aware): o dataset de prova de conceito (Mendeley oleos) nao tem
    replicas fisicas/`mae_id` (cada linha e' uma garrafa distinta,
    confirmado em `tests/test_validacao_publica_mendeley.py`/
    `..._mir_raman.py`, que tambem usam `group_by_mae_id=False` para este
    mesmo dataset) -- nao ha' vazamento de replica a evitar aqui.
    """
    X_cal = np.asarray(X_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float).reshape(-1, 1)
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float).reshape(-1, 1)

    lv_max = min(max_lvs, max(2, X_cal.shape[0] // 5))
    cv = KFold(n_splits=n_splits_cv, shuffle=True, random_state=seed)
    erros = []
    for n_lv in range(1, lv_max + 1):
        y_hat_cv = cross_val_predict(
            PLSRegression(n_components=n_lv, scale=False), X_cal, y_cal, cv=cv)
        erros.append(rmse_flat(y_cal, y_hat_cv))
    n_opt = int(np.argmin(erros)) + 1

    pls = PLSRegression(n_components=n_opt, scale=False).fit(X_cal, y_cal)
    y_cal_hat = pls.predict(X_cal)
    y_val_hat = pls.predict(X_val)

    from sklearn.metrics import r2_score
    r2cal = float(r2_score(y_cal, y_cal_hat))
    r2val = float(r2_score(y_val, y_val_hat))
    rmsep = rmse_flat(y_val, y_val_hat)
    return RegressionFitResult(
        n_lv=n_opt, rmsep=rmsep, r2cal=r2cal, r2val=r2val,
        n_cal=int(X_cal.shape[0]), n_val=int(X_val.shape[0]))
