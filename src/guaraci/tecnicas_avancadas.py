# -*- coding: utf-8 -*-
"""tecnicas_avancadas.py -- orquestracao testavel (sem Rich/Streamlit) para
as tecnicas cientificas que existiam implementadas e testadas em isolamento
mas sem NENHUM caminho de execucao real em CLI/web (achado da auditoria de
acessibilidade, preparacao para o primeiro usuario externo): ASCA, EPO/GLSW,
MCR-ALS (com e sem restricao de correlacao), fusao multibloco.

Cada funcao aqui: recebe o dado JA CARREGADO (mesmo contrato de
`pipeline.load_data`/`validate_input` -- nunca reimplementa leitura de
arquivo), monta os parametros que a funcao cientifica de baixo nivel exige a
partir do que o dataset ja' tem (rotulos, metadados_df, mae_id, conc) e
devolve um dict/dataclass pronto pra' exibicao (CLI `_menu_tecnicas_
avancadas` em guaraci.py) ou pra' `app_tabs/tecnicas.py` (web) -- fonte
UNICA de logica entre as duas interfaces, mesmo padrao de `auditoria_
delineamento.run_audit`/`sentinela_deriva.hook_apos_predicao`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from guaraci.asca import asca_decompose, asca_permutation_test
from guaraci.epo_glsw import EPO, GLSW, build_difference_matrix
from guaraci.fusao_multibloco import (
    BlockMismatchError,
    RegressionFitResult,
    build_multiblock_dataset,
    fit_evaluate_pls_regression,
)
from guaraci.mcr_als import (
    MCRALSResultado,
    MCRALSResultadoSupervisionado,
    mcr_als,
    mcr_als_com_restricao_correlacao,
)

__all__ = [
    "fatores_categoricos_disponiveis",
    "rodar_asca",
    "AscaRelatorio",
    "rodar_epo_glsw",
    "EpoGlswRelatorio",
    "rodar_mcr_als",
    "rodar_fusao_multibloco",
]

#: Cardinalidade maxima p/ uma coluna de metadados contar como "fator
#: categorico" candidato (acima disso e' mais provavel ser um identificador
#: ou uma medida continua, nao um nivel de delineamento).
_CARDINALIDADE_MAX_FATOR = 12
#: Colunas de metadados que sao identificacao de amostra, nunca fator
#: cientifico -- mesma lista de `dados_io._COLUNAS_IDENTIFICADORAS`, mas
#: reafirmada aqui (nao importada) porque e' privada do outro modulo e o
#: criterio de uso e' distinto (elegibilidade p/ ASCA/EPO, nao privacidade).
_COLUNAS_NAO_FATOR = {"title_original", "arquivo", "cod", "data", "mae_id",
                       "subpasta", "teor", "cod_conhecido"}


def fatores_categoricos_disponiveis(
        rotulos: np.ndarray,
        metadados_df: Optional[pd.DataFrame]) -> Dict[str, np.ndarray]:
    """Fatores candidatos p/ ASCA/EPO-GLSW: `especie` (rotulos, sempre
    disponivel) + qualquer coluna categorica de baixa cardinalidade do
    `metadados_df` (se existir -- `None` em mode csv/opus/spc/sp/rmn/hplc/
    gcms/eem, que nao tem metadado rico). Nunca inclui coluna continua
    (`teor`) nem identificadora (mae_id/arquivo/...) -- ver `_COLUNAS_NAO_
    FATOR`/`_CARDINALIDADE_MAX_FATOR`."""
    rotulos = np.asarray(rotulos)
    n = len(rotulos)
    fatores: Dict[str, np.ndarray] = {"especie": rotulos}
    if metadados_df is None:
        return fatores
    for col in metadados_df.columns:
        if col in _COLUNAS_NAO_FATOR or col == "especie":
            continue
        serie = metadados_df[col]
        if len(serie) != n:
            continue
        try:
            n_unicos = serie.nunique(dropna=True)
        except TypeError:
            continue
        if 1 < n_unicos <= _CARDINALIDADE_MAX_FATOR:
            fatores[col] = serie.to_numpy()
    return fatores


@dataclass
class AscaRelatorio:
    decomposicao: Dict[str, Any]
    permutacao: Optional[Dict[str, Any]]
    fatores_usados: List[str]
    group_aware: bool


def rodar_asca(X: np.ndarray, fatores: Dict[str, np.ndarray],
                n_componentes: int = 2,
                mae_id: Optional[np.ndarray] = None,
                n_perm: int = 200, seed: int = 42,
                testar_significancia: bool = True) -> AscaRelatorio:
    """Decompoe `X` pelos `fatores` escolhidos (subconjunto do retorno de
    `fatores_categoricos_disponiveis`) e, se `testar_significancia`, testa
    cada um por permutacao -- group-aware (por `mae_id`) quando disponivel,
    nunca por amostra individual se houver replica fisica conhecida (ver
    docstring de `asca.asca_permutation_test`)."""
    if not fatores:
        raise ValueError("Escolha pelo menos 1 fator para ASCA.")
    decomposicao = asca_decompose(X, fatores, n_componentes=n_componentes)
    permutacao = None
    if testar_significancia:
        permutacao = asca_permutation_test(
            X, fatores, unidade_permutacao=mae_id, n_perm=n_perm, seed=seed)
    return AscaRelatorio(decomposicao=decomposicao, permutacao=permutacao,
                          fatores_usados=list(fatores.keys()),
                          group_aware=mae_id is not None)


@dataclass
class EpoGlswRelatorio:
    metodo: str
    X_corrigido: np.ndarray
    n_pares_diferenca: int
    variancia_removida_fracao: float
    grupo_interesse_nome: str
    fator_incomodo_nome: str


def rodar_epo_glsw(X: np.ndarray, grupo_interesse: np.ndarray,
                    fator_incomodo: np.ndarray,
                    grupo_interesse_nome: str, fator_incomodo_nome: str,
                    metodo: str = "EPO",
                    n_componentes: int = 1, alpha: float = 1e-3
                    ) -> EpoGlswRelatorio:
    """Monta a matriz de diferencas (`grupo_interesse` = o que preservar,
    `fator_incomodo` = o que remover/atenuar) e ajusta `EPO`
    (`metodo="EPO"`) ou `GLSW` (`metodo="GLSW"`) sobre ela, devolvendo X
    corrigido + fracao de variancia total removida (diagnostico, nao um
    modelo treinado p/ reuso automatico -- ver limite de escopo no
    docstring de `epo_glsw.py`: NAO se aplica quando o fator de perturbacao
    e' colinear com o alvo cientifico)."""
    if metodo not in ("EPO", "GLSW"):
        raise ValueError(f"metodo precisa ser 'EPO' ou 'GLSW', recebido {metodo!r}")
    D = build_difference_matrix(X, grupo_interesse, fator_incomodo)
    transformador = EPO(n_componentes=n_componentes) if metodo == "EPO" \
        else GLSW(alpha=alpha)
    transformador.fit(D)
    X_corrigido = transformador.transform(X)

    var_antes = float(np.sum((X - X.mean(axis=0)) ** 2))
    var_depois = float(np.sum((X_corrigido - X_corrigido.mean(axis=0)) ** 2))
    frac_removida = (1.0 - var_depois / var_antes) if var_antes > 1e-300 else float("nan")

    return EpoGlswRelatorio(
        metodo=metodo, X_corrigido=X_corrigido, n_pares_diferenca=D.shape[0],
        variancia_removida_fracao=frac_removida,
        grupo_interesse_nome=grupo_interesse_nome,
        fator_incomodo_nome=fator_incomodo_nome)


def rodar_mcr_als(
        X: np.ndarray, n_componentes: int, *,
        conc: Optional[np.ndarray] = None,
        indice_componente_alvo: Optional[int] = None,
        nao_negativo_c: bool = True, nao_negativo_s: bool = True,
        max_iter: int = 200
) -> "MCRALSResultado | MCRALSResultadoSupervisionado":
    """MCR-ALS puro (sem `conc`/`indice_componente_alvo`) ou com restricao
    de correlacao (Bayat et al. 2020 -- quando os dois sao dados): usa
    todas as amostras com `conc` NAO-nulo como `indices_calibracao`, ancora
    o componente `indice_componente_alvo` a esses valores de referencia."""
    if conc is not None and indice_componente_alvo is not None:
        conc = np.asarray(conc, dtype=float)
        indices_calibracao = np.where(~np.isnan(conc))[0]
        if indices_calibracao.size < 3:
            raise ValueError(
                "MCR-ALS com restricao de correlacao precisa de pelo menos "
                "3 amostras com teor/concentracao de referencia conhecido "
                f"(encontradas: {indices_calibracao.size}).")
        return mcr_als_com_restricao_correlacao(
            X, n_componentes,
            indice_componente_alvo=indice_componente_alvo,
            y_referencia=conc[indices_calibracao],
            indices_calibracao=indices_calibracao,
            nao_negativo_c=nao_negativo_c, nao_negativo_s=nao_negativo_s,
            max_iter=max_iter)
    return mcr_als(X, n_componentes, nao_negativo_c=nao_negativo_c,
                    nao_negativo_s=nao_negativo_s, max_iter=max_iter)


def rodar_fusao_multibloco(
        blocos: Dict[str, np.ndarray], y: np.ndarray, *,
        frac_cal: float = 0.7, seed: int = 0, max_lvs: int = 15,
) -> Tuple[RegressionFitResult, Dict[str, Tuple[int, int]]]:
    """Funde `blocos` (>=2 tecnicas na MESMA amostra fisica, MESMA ordem de
    linha -- responsabilidade do chamador, ver `BlockMismatchError`) e
    avalia PLS-R contra `y` com split aleatorio simples calibracao/
    validacao (mesmo protocolo, deliberadamente simples, de `fit_evaluate_
    pls_regression` -- ver docstring: sem group-aware pq o par de prova de
    conceito nao tem replica fisica conhecida; se o dataset do usuario
    tiver mae_id, o split deveria ser refeito group-aware antes de
    confiar no numero, ressalva exibida pelo chamador CLI/web)."""
    if len(blocos) < 2:
        raise ValueError(
            f"Fusao multibloco precisa de >=2 blocos, recebido {len(blocos)}.")
    y = np.asarray(y, dtype=float)
    n = len(y)
    for nome, X in blocos.items():
        if len(X) != n:
            raise BlockMismatchError(
                f"bloco {nome!r} tem {len(X)} amostras, y tem {n} -- "
                f"precisam bater (mesma amostra fisica, mesma ordem).")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_cal = max(2, int(round(frac_cal * n)))
    n_cal = min(n_cal, n - 2) if n > 3 else max(1, n - 1)
    idx_cal, idx_val = idx[:n_cal], idx[n_cal:]
    if len(idx_val) < 1:
        raise ValueError(
            f"n={n} amostras insuficiente para separar calibracao/validacao "
            f"(frac_cal={frac_cal}).")

    X_fundida, _preprocessadores, fatias = build_multiblock_dataset(
        blocos, indices_ajuste=idx_cal)
    resultado = fit_evaluate_pls_regression(
        X_fundida[idx_cal], y[idx_cal], X_fundida[idx_val], y[idx_val],
        max_lvs=max_lvs, seed=seed)
    return resultado, fatias
