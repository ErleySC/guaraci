"""hsi_chemistry.py — Explicabilidade cruzada: VIP das bandas mais
importantes do PLS-DA por pixel x tabela de atribuicao quimica
conhecida (Passo 100 da `INSTRUCAO_HSI_MINIMO_VIAVEL.md`).

Reaproveita `chemometric_stats.vip_scores` (ja' implementado, ver
`hsi_classification.py`) -- este modulo so' adiciona o cruzamento com
uma tabela de atribuicao. A tabela abaixo e' ESPECIFICA da faixa/matriz
do dataset real usado nesta integracao (VIS, 397-1004nm, fruta -- ver
`docs/PROGRESSO.md` Passo 93): atribuicoes de OUTRA matriz/faixa (ex.
FT-NIR de oleo) NAO se aplicam aqui, por isso a tabela e' passada como
parametro em vez de fixa no codigo -- outro dataset precisa da sua
propria tabela.

Cada entrada e' baseada em literatura real de espectroscopia VIS de
fruta, citada individualmente (nao um "exemplo" inventado):

  - 660-680 nm: banda de absorcao da clorofila-a (perda de clorofila
    durante o amadurecimento). Merzlyak, Solovchenko & Gitelson (2003),
    "Reflectance spectral features and non-destructive estimation of
    chlorophyll, carotenoid and anthocyanin content in apple fruit",
    Postharvest Biology and Technology 27(2):197-211.
  - 500-550 nm: regiao de carotenoides/antocianinas (pigmentos que
    aumentam durante o amadurecimento, a medida que a clorofila
    degrada) -- mesma referencia acima.
  - 960-980 nm: sobretom de estiramento O-H da agua (2a harmonica) --
    banda classica de NIR, Osborne, Fearn & Hindle (1993), "Practical
    NIR Spectroscopy with Applications in Food and Beverage Analysis",
    2a ed., Longman Scientific & Technical, cap. 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple

import numpy as np

__all__ = [
    "BandAttribution",
    "BandFinding",
    "ATRIBUICAO_QUIMICA_VIS_FRUTA",
    "cross_reference_vip_with_chemistry",
]


class BandAttribution(NamedTuple):
    banda_min_nm: float
    banda_max_nm: float
    atribuicao: str
    referencia: str


ATRIBUICAO_QUIMICA_VIS_FRUTA: List[BandAttribution] = [
    BandAttribution(660.0, 680.0, "absorcao de clorofila-a",
                    "Merzlyak, Solovchenko & Gitelson (2003), "
                    "Postharvest Biology and Technology 27(2):197-211"),
    BandAttribution(500.0, 550.0, "regiao de carotenoides/antocianinas",
                    "Merzlyak, Solovchenko & Gitelson (2003), "
                    "Postharvest Biology and Technology 27(2):197-211"),
    BandAttribution(960.0, 980.0, "sobretom O-H da agua (2a harmonica)",
                    "Osborne, Fearn & Hindle (1993), Practical NIR "
                    "Spectroscopy, 2a ed., cap. 2"),
]


@dataclass
class BandFinding:
    wavelength_nm: float
    vip: float
    atribuicao: str          # texto da atribuicao OU a frase de "sem atribuicao"
    referencia: str          # referencia da tabela, ou "" se sem atribuicao


_SEM_ATRIBUICAO = ("sem atribuicao obvia -- possivel artefato de "
                   "instrumento/lote")


def cross_reference_vip_with_chemistry(
        wavelengths: np.ndarray, vip: np.ndarray,
        tabela: List[BandAttribution], *, top_n: int = 5,
        ) -> List[BandFinding]:
    """Cruza as `top_n` bandas de MAIOR VIP com `tabela` -- devolve UMA
    entrada por banda, nunca afirmando causalidade sem essa checagem
    (Passo 100): "consistente com [atribuicao] ([referencia])" quando o
    comprimento de onda cai dentro de alguma faixa da tabela, ou a frase
    padrao de "sem atribuicao" caso contrario."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    vip = np.asarray(vip, dtype=float)
    if wavelengths.shape != vip.shape:
        raise ValueError(
            f"cross_reference_vip_with_chemistry: wavelengths "
            f"{wavelengths.shape} e vip {vip.shape} devem ter a mesma forma.")
    if top_n < 1:
        raise ValueError("top_n deve ser >= 1.")

    top_n = min(top_n, len(vip))
    indices_top = np.argsort(-vip)[:top_n]

    achados: List[BandFinding] = []
    for i in indices_top:
        wl = float(wavelengths[i])
        entrada_bate = next(
            (e for e in tabela if e.banda_min_nm <= wl <= e.banda_max_nm), None)
        if entrada_bate is not None:
            achados.append(BandFinding(
                wavelength_nm=wl, vip=float(vip[i]),
                atribuicao=f"consistente com {entrada_bate.atribuicao}",
                referencia=entrada_bate.referencia))
        else:
            achados.append(BandFinding(
                wavelength_nm=wl, vip=float(vip[i]),
                atribuicao=_SEM_ATRIBUICAO, referencia=""))
    return achados
