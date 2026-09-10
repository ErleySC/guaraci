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

O QUE E' REUSADO: a chamada PARAFAC de baixo nivel
(`hsi_multiway.parafac_hsi`) e' matematicamente generica (qualquer
tensor 3-way) -- so' os NOMES dos campos do resultado sao especificos
de HSI. `parafac_eem` abaixo chama `parafac_hsi` e RENOMEIA os fatores
para o vocabulario correto de EEM antes de devolver, sem duplicar a
decomposicao em si.

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

DATASET PUBLICO REAL identificado nesta auditoria (Mendeley
`10.17632/g6y69g8gwm.1`, Venturini et al. 2023, CC BY 4.0 -- 24
azeites de oliva, EEMs reais em 10 etapas de envelhecimento): baixado e
inspecionado por leitura direta em 2026-09-04, mas o formato bruto de
exportacao do instrumento e' um CSV IRREGULAR por amostra/excitacao
(cada bloco de coluna excitacao->emissao tem um numero DIFERENTE de
linhas validas, sem preenchimento consistente -- confirmado por
tokenizacao linha-a-linha, `csv.reader` sobre a linha bruta varia de
281 a 1 campo dependendo da linha). Escrever um parser robusto para
esse formato especifico e' trabalho de escopo proprio (fora do que foi
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


@dataclass
class ParafacEEMResultado:
    """`fator_amostra` (n_amostras, R), `fator_excitacao` (n_excitacao, R),
    `fator_emissao` (n_emissao, R) -- R = numero de componentes.
    `erro_reconstrucao_relativo` = ||tensor - reconstrucao|| / ||tensor||
    (Frobenius) -- nunca reportar os fatores sem esse numero junto
    (mesma disciplina de `hsi_multiway.ParafacHSIResultado`)."""
    fator_amostra: np.ndarray
    fator_excitacao: np.ndarray
    fator_emissao: np.ndarray
    erro_reconstrucao_relativo: float


def parafac_eem(tensor: np.ndarray, n_componentes: int, *,
                 max_iter: int = 200, seed: int = 0) -> ParafacEEMResultado:
    """PARAFAC/CP (Bro, 1997) do tensor `(n_amostras, n_excitacao,
    n_emissao)` (ver `construir_tensor_eem`). Reusa a decomposicao de
    `hsi_multiway.parafac_hsi` (matematicamente generica p/ qualquer
    tensor 3-way) e RENOMEIA os fatores pro vocabulario correto de EEM
    -- ver docstring do modulo pra' o motivo de nao reusar os NOMES."""
    from guaraci.hsi_multiway import parafac_hsi

    bruto = parafac_hsi(tensor, n_componentes, max_iter=max_iter, seed=seed)
    return ParafacEEMResultado(
        fator_amostra=bruto.fator_amostra,
        fator_excitacao=bruto.fator_espacial,
        fator_emissao=bruto.fator_espectral,
        erro_reconstrucao_relativo=bruto.erro_reconstrucao_relativo)
