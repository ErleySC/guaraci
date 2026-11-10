# -*- coding: utf-8 -*-
"""eem_multiway.py -- Decomposicao PARAFAC de matrizes de fluorescencia
excitacao-emissao (EEM), Passo 144/145 da auditoria das 11 tecnicas
analiticas (2026-09-04).

POR QUE UM MODULO SEPARADO DE `hsi_multiway.py` (nao so' reusar o dele
com outro nome): `hsi_multiway.construir_tensor_amostras` e' acoplado
ao problema ESPECIFICO de imageamento hiperespectral -- reduz a ROI de
cada gravacao a uma GRADE ESPACIAL FIXA por media de bloco, porque
objetos fisicos diferentes fotografados por uma HSI quase nunca tem a
MESMA resolucao espacial (ver docstring daquele modulo). EEM nao tem
esse problema: todas as amostras de UMA campanha de medicao
compartilham a MESMA grade de comprimentos de onda de excitacao e
emissao (propriedade do INSTRUMENTO, nao da amostra) -- nao ha' nada
para "reduzir", so' empilhar. Forcar EEM pelo caminho da HSI
contaminaria o vocabulario do resultado (`fator_espacial`/
`fator_espectral` nao fazem sentido pra' excitacao/emissao -- mesmo
problema que `perfil_matriz.py` existe para evitar, ver Passo 141 da
auditoria) e carregaria a logica de bounding-box/ROI que EEM nunca
precisa.

DIFERENCA PARA A HSI: `parafac_eem` usa, por padrao, PARAFAC com
NAO-NEGATIVIDADE (HALS; fluorescencia nao tem intensidade negativa; sem a
restricao os fatores saem com sinal arbitrario) e reporta a consistencia do
nucleo (CORCONDIA) para escolher o numero de componentes. `nao_negativo=False`
devolve o ALS livre (sem restricao), para reproduzir versoes anteriores ou
outros programas.

DADOS FALTANTES: a regiao de espalhamento Rayleigh/Raman removida (ver
`eem_io.tratar_espalhamento`) entra como `mascara_faltantes`: as celulas
mascaradas nao contam no ajuste (EM: preenche a celula com a reconstrucao do
modelo a cada rodada e reajusta, Tomasi & Bro, 2005, Chemom. Intell. Lab.
Syst. 75:163; Andersen & Bro, 2003, J. Chemometrics 17:200).

REFERENCIA (verificada no Crossref em 2026-09-04, mesma citada em
`hsi_multiway.py`): Bro, R. (1997). "PARAFAC. Tutorial and
applications." Chemometrics and Intelligent Laboratory Systems, 38(2),
149-171. DOI: 10.1016/S0169-7439(97)00032-4 -- PARAFAC de EEM e' a
aplicacao ORIGINAL que motivou o metodo (fluorescencia multi-amostra),
mais direta ainda que o uso em cubo hiperespectral.

LIMITACAO HONESTA: `construir_tensor_eem` exige que todas as EEMs de
entrada tenham EXATAMENTE a mesma forma (mesma grade de excitacao/
emissao) -- amostras medidas em campanhas/instrumentos diferentes (com
grades diferentes) precisam ser interpoladas para uma grade comum
ANTES de chamar esta funcao; nenhuma interpolacao e' feita aqui (regra
de nunca inventar dado que nao foi medido).

Formatos brutos de exportacao de instrumento podem ser um CSV IRREGULAR
por amostra/excitacao (cada bloco de coluna excitacao->emissao com um
numero DIFERENTE de linhas validas, sem preenchimento consistente).
Escrever um parser robusto para esse formato especifico e' trabalho de escopo proprio (fora do que foi
aprovado nesta rodada: "PARAFAC generalizado", nao "parser do formato
Horiba/instrumento X") -- registrado como pendencia honesta, nao
escondida. Este modulo e' provado pela CONTRA-PROVA SINTETICA em
`tests/test_eem_multiway.py` (recupera 2 componentes puros conhecidos
de uma mistura simulada), mesma disciplina de seguranca usada para
MCR-ALS antes de tocar dado real (Bloco 14)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

__all__ = [
    "construir_tensor_eem",
    "ParafacEEMResultado",
    "parafac_eem",
    "consistencia_nucleo",
    "nota_parafac",
    "NOTA_PARAFAC",
]


def construir_tensor_eem(matrizes: Dict[str, np.ndarray]) -> "tuple[np.ndarray, List[str]]":
    """Empilha um dicionario `{id_amostra: matriz_eem}` (cada matriz
    `(n_excitacao, n_emissao)`, MESMA forma em todas -- ver LIMITACAO
    HONESTA no docstring do modulo) num tensor `(n_amostras,
    n_excitacao, n_emissao)`, na ordem de insercao do dict. Devolve
    `(tensor, ids_na_ordem)` -- os ids sao necessarios pra' interpretar
    `fator_amostra` do resultado do PARAFAC depois.

    Levanta `ValueError` se `matrizes` estiver vazio, se alguma matriz
    nao for 2D, ou se as formas nao baterem entre si -- nunca
    completa/trunca silenciosamente pra' forcar um tensor regular."""
    if not matrizes:
        raise ValueError("matrizes vazio -- nada para empilhar")

    ids = list(matrizes.keys())
    primeira = np.asarray(matrizes[ids[0]], dtype=float)
    if primeira.ndim != 2:
        raise ValueError(
            f"matriz EEM de '{ids[0]}' precisa ser 2D (excitacao, emissao), "
            f"recebeu shape {primeira.shape}")
    forma = primeira.shape

    camadas = [primeira]
    for id_amostra in ids[1:]:
        m = np.asarray(matrizes[id_amostra], dtype=float)
        if m.shape != forma:
            raise ValueError(
                f"matriz EEM de '{id_amostra}' tem forma {m.shape}, "
                f"esperado {forma} (mesma grade excitacao/emissao de "
                f"'{ids[0]}') -- interpole para uma grade comum antes "
                f"de chamar construir_tensor_eem.")
        camadas.append(m)

    tensor = np.stack(camadas, axis=0)
    return tensor, ids


NOTA_PARAFAC = {
    "nao-negativo": (
        "PARAFAC com restricao de nao-negatividade (HALS) desde a v1.0.0; versoes anteriores ou outros softwares sem "
        "restricao podem dar sinais/ordens diferentes dos fatores; CORCONDIA reportado.",
        "PARAFAC with a non-negativity constraint (HALS) since v1.0.0; earlier versions or other unconstrained software "
        "may give different signs/orders of the factors; CORCONDIA reported."),
    "nenhuma": (
        "PARAFAC sem restricao (ALS livre), por escolha do usuario (parafac_nao_negativo: false); o padrao desde a v1.0.0 "
        "e' nao-negativo (HALS), e os fatores podem ter sinais/ordens diferentes dos dele; CORCONDIA reportado.",
        "PARAFAC without constraint (free ALS), by user choice (parafac_nao_negativo: false); the default since v1.0.0 is "
        "non-negative (HALS), and its factors may have different signs/orders; CORCONDIA reported."),
}


def nota_parafac(restricao: str, idioma: str = "PT") -> str:
    """Nota curta, no idioma pedido (`PT` ou `EN`), sobre a restricao do PARAFAC usado (`nao-negativo` ou `nenhuma`)."""
    pt, en = NOTA_PARAFAC.get(restricao, NOTA_PARAFAC["nao-negativo"])
    return en if str(idioma).upper().startswith("EN") else pt


@dataclass
class ParafacEEMResultado:
    """`fator_amostra` (n_amostras, R), `fator_excitacao` (n_excitacao, R),
    `fator_emissao` (n_emissao, R) -- R = numero de componentes.
    `erro_reconstrucao_relativo` = ||tensor - reconstrucao|| / ||tensor||
    (Frobenius, so' nas celulas observadas) -- nunca reportar os fatores sem esse
    numero junto (mesma disciplina de `hsi_multiway.ParafacHSIResultado`).
    `consistencia_nucleo` = CORCONDIA em % (Bro & Kiers, 2003, J. Chemometrics
    17:274-286): perto de 100 = modelo trilinear adequado; caindo muito ao
    acrescentar um componente = componentes demais (com dados faltantes e'
    calculada sobre o tensor com as celulas faltantes preenchidas pelo modelo).
    `restricao` ('nao-negativo' ou 'nenhuma') e `algoritmo` dizem como foi ajustado;
    `n_faltantes`/`n_iter_em` descrevem o tratamento de dados faltantes (0 = nenhum)."""
    fator_amostra: np.ndarray
    fator_excitacao: np.ndarray
    fator_emissao: np.ndarray
    erro_reconstrucao_relativo: float
    consistencia_nucleo: float = float("nan")
    restricao: str = "nao-negativo"
    algoritmo: str = "HALS (tensorly.non_negative_parafac_hals)"
    n_faltantes: int = 0
    n_iter_em: int = 0

    @property
    def nota(self) -> str:
        return nota_parafac(self.restricao, "PT")

    @property
    def nota_en(self) -> str:
        return nota_parafac(self.restricao, "EN")


def consistencia_nucleo(tensor: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """CORCONDIA (%): ajusta por minimos quadrados o nucleo de Tucker G com os
    fatores PARAFAC fixos e compara com a superdiagonal unitaria T:
    100 * (1 - sum((G - T)^2) / R). Escala dos fatores absorvida em A."""
    R = A.shape[1]
    G = np.einsum("ai,bj,ck,ijk->abc", np.linalg.pinv(A), np.linalg.pinv(B),
                  np.linalg.pinv(C), np.asarray(tensor, dtype=float))
    T = np.zeros((R, R, R))
    T[np.arange(R), np.arange(R), np.arange(R)] = 1.0
    return float(100.0 * (1.0 - np.sum((G - T) ** 2) / R))


def _pre_preencher(tensor: np.ndarray, faltante: np.ndarray) -> np.ndarray:
    """Valor inicial das celulas faltantes para o EM: interpolacao linear ao longo da emissao (ultimo eixo) entre os vizinhos
    observados de cada (amostra, excitacao); sem vizinho, 0."""
    out = np.where(faltante, 0.0, tensor)
    x = np.arange(tensor.shape[2], dtype=float)
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            f = faltante[i, j]
            if f.any() and (~f).sum() >= 2:
                out[i, j, f] = np.interp(x[f], x[~f], tensor[i, j, ~f])
    return out


def parafac_eem(tensor: np.ndarray, n_componentes: int, *,
                 max_iter: int = 200, seed: int = 0, nao_negativo: bool = True,
                 mascara_faltantes: "np.ndarray | None" = None,
                 max_iter_em: int = 60, tol_em: float = 1e-4) -> ParafacEEMResultado:
    """PARAFAC/CP (Bro, 1997) do tensor `(n_amostras, n_excitacao,
    n_emissao)` (ver `construir_tensor_eem`), via `tensorly`.
    Por padrao (`nao_negativo=True`) o ajuste e' PARAFAC com restricao de
    nao-negatividade nos 3 modos (HALS; pratica padrao de EEM; sem ela escores e
    perfis saem com sinal arbitrario, ex.: escore negativo para um fluoroforo
    presente). `nao_negativo=False` usa o ALS livre (sem restricao; os sinais dos
    perfis sao normalizados para soma positiva, so' por convencao). Perfis de
    excitacao/emissao normalizados (norma 1); a escala vai para `fator_amostra`,
    que fica proporcional a concentracao.

    `mascara_faltantes`: booleana, do mesmo formato do tensor ou `(n_excitacao,
    n_emissao)` (vale para todas as amostras); True = celula faltante (ex.: banda de
    espalhamento removida). Tambem sao tratadas como faltantes as celulas NaN do
    tensor. Ajuste por EM: as celulas faltantes sao preenchidas com a reconstrucao
    do modelo a cada rodada e o modelo e' reajustado ate' a mudanca relativa do erro
    nas celulas observadas ficar abaixo de `tol_em`."""
    from tensorly.cp_tensor import cp_to_tensor
    from tensorly.decomposition import non_negative_parafac_hals, parafac

    tensor = np.asarray(tensor, dtype=float)
    if tensor.ndim != 3:
        raise ValueError(f"tensor precisa ser 3-way, recebeu shape {tensor.shape}")
    falt = ~np.isfinite(tensor)
    if mascara_faltantes is not None:
        m = np.asarray(mascara_faltantes, dtype=bool)
        if m.shape == tensor.shape[1:]:
            m = np.broadcast_to(m, tensor.shape)
        elif m.shape != tensor.shape:
            raise ValueError(f"mascara_faltantes tem shape {m.shape}, esperado {tensor.shape} ou {tensor.shape[1:]}")
        falt = falt | m
    n_falt = int(falt.sum())
    obs = ~falt
    if obs.sum() == 0:
        raise ValueError("todas as celulas do tensor sao faltantes -- nada para decompor")
    base = np.where(falt, 0.0, tensor)
    norma = float(np.linalg.norm(base))
    if norma < 1e-300:
        raise ValueError("tensor e' (quase) todo zero -- nada para decompor")
    rank = int(n_componentes)

    def ajustar(X, n_iter, init):
        if nao_negativo:
            return non_negative_parafac_hals(X, rank=rank, n_iter_max=n_iter, init=init, random_state=seed)
        return parafac(X, rank=rank, n_iter_max=n_iter, init=init, random_state=seed)

    n_em = 0
    if n_falt == 0:
        cp = ajustar(tensor, max_iter, "svd")
        X_final = tensor
    else:
        X_final = _pre_preencher(np.where(falt, np.nan, tensor), falt)
        cp = ajustar(X_final, max_iter, "svd")
        erro_ant = float("inf")
        for n_em in range(1, int(max_iter_em) + 1):
            rec = cp_to_tensor(cp)
            X_final = np.where(falt, rec, tensor)
            e = float(np.linalg.norm((base - rec)[obs]))
            if abs(erro_ant - e) <= tol_em * max(e, 1e-300):
                break
            erro_ant = e
            cp = ajustar(X_final, 25, cp)
    rec = cp_to_tensor(cp)
    erro = float(np.linalg.norm((base - rec)[obs]) / norma)
    A, B, C = (np.asarray(f, dtype=float) for f in cp.factors)
    A = A * np.asarray(cp.weights, dtype=float)[None, :]
    if not nao_negativo:
        sb = np.where(B.sum(axis=0) < 0, -1.0, 1.0)
        sc = np.where(C.sum(axis=0) < 0, -1.0, 1.0)
        A, B, C = A * sb * sc, B * sb, C * sc
    nb = np.linalg.norm(B, axis=0); nb[nb < 1e-300] = 1.0
    nc = np.linalg.norm(C, axis=0); nc[nc < 1e-300] = 1.0
    A, B, C = A * nb * nc, B / nb, C / nc
    return ParafacEEMResultado(
        fator_amostra=A, fator_excitacao=B, fator_emissao=C,
        erro_reconstrucao_relativo=erro,
        consistencia_nucleo=consistencia_nucleo(X_final, A, B, C),
        restricao="nao-negativo" if nao_negativo else "nenhuma",
        algoritmo=("HALS (tensorly.non_negative_parafac_hals)" if nao_negativo else "ALS (tensorly.parafac)")
                  + (", EM para dados faltantes" if n_falt else ""),
        n_faltantes=n_falt, n_iter_em=int(n_em))
