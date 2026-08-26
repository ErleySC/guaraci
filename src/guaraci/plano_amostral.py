# -*- coding: utf-8 -*-
"""plano_amostral.py -- Orientacao de tamanho amostral (P1, Bloco 10 --
`guaraci plan`). Duas fontes de garantia bem diferentes, nunca misturadas:

- CONFORMAL (Identificar/agrupado, `identificacao.py`/`conformal.py`):
  garantia distribution-free real. `n_minimo = ceil(1/alpha) - 1`, ja'
  implementado em `conformal.n_minimum_for_alpha` -- REAPROVEITADO aqui
  diretamente, nunca reimplementado.
- DD-SIMCA por especie (`classificadores.DDSimca`): metodo PARAMETRICO
  (aproximacao chi2-momentos). Medido em 2026-08-26
  (`scripts/medicoes/medir_ddsimca_cobertura_vs_n.py`, ver tambem a
  docstring de `classificadores.DDSimca` e `docs/MANUAL.md` secao 9): a
  cobertura empirica NAO converge para o nominal so' com mais `n` -- sobe
  rapido ate' n~150 e ESTANCA num plato de ~0,94-0,945, mesmo em n=1200.
  Por isso este modulo NUNCA promete uma cobertura-alvo acima do plato
  observado, por maior que seja o `n` sugerido -- isso seria uma alegacao
  falsa embutida no `guaraci plan`. Acima do plato, a orientacao e'
  recomendar o gate conformal (garantia real) em vez de DD-SIMCA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from guaraci.conformal import n_minimum_for_alpha

__all__ = [
    "OrientacaoDDSimca",
    "n_minimo_conformal",
    "orientacao_tamanho_amostral_ddsimca",
]

# Tabela medida (2026-08-26): DGP gaussiano AR(1) sintetico, DD-SIMCA REAL
# do projeto, alpha=0.05, n_components=3, N_TEST=2000, ver script para o
# desenho completo. NAO e' o dataset real do usuario -- ver `ressalva` em
# `orientacao_tamanho_amostral_ddsimca`.
TABELA_COBERTURA_DDSIMCA_MEDIDA: List[Tuple[int, float]] = [
    (5, 0.8450), (10, 0.8957), (20, 0.9038), (40, 0.9230),
    (80, 0.9306), (150, 0.9428), (300, 0.9425), (600, 0.9448),
    (1200, 0.9411),
]

# Piso conservador do plato observado (a faixa medida em n>=150 oscila
# entre 0,9411 e 0,9448 -- ruido de amostragem da propria medicao, nao
# uma tendencia real de subida). Usar o MENOR valor da faixa, nao a media,
# como teto do que se promete: errar para o lado conservador aqui.
PLATO_COBERTURA_DDSIMCA = 0.94

_FAIXA_MEDIDA_N = (
    TABELA_COBERTURA_DDSIMCA_MEDIDA[0][0],
    TABELA_COBERTURA_DDSIMCA_MEDIDA[-1][0],
)


def n_minimo_conformal(alpha: float) -> int:
    """`n_minimo` para o gate conformal (Identificar/agrupado) -- fina
    reexportacao de `conformal.n_minimum_for_alpha`, garantia
    distribution-free real (Vovk, Gammerman & Shafer 2005): nenhum `n`
    menor sustenta o `alpha` pedido, qualquer `n` maior sustenta.
    """
    return n_minimum_for_alpha(alpha)


@dataclass
class OrientacaoDDSimca:
    """Resultado de `orientacao_tamanho_amostral_ddsimca` -- nunca promete
    uma cobertura-alvo que a medicao nao sustenta (ver docstring do
    modulo)."""

    cobertura_alvo: float
    alcancavel: bool
    n_sugerido: Optional[int]
    cobertura_no_n_sugerido: Optional[float]
    plato_observado: float
    recomendar_conformal: bool
    ressalva: str


def orientacao_tamanho_amostral_ddsimca(cobertura_alvo: float
                                         ) -> OrientacaoDDSimca:
    """Orientacao de `n` para DD-SIMCA por especie, dada uma cobertura-
    alvo (ex.: 0.90 para 90% de cobertura). NUNCA devolve `alcancavel=True`
    com uma cobertura-alvo acima de `PLATO_COBERTURA_DDSIMCA` -- essa e' a
    contra-prova central deste modulo (D2 do Bloco 9b, mesmo espirito:
    responsabilidade do usuario decidir se prossegue, mas o numero
    devolvido tem que ser honesto sobre o que e' alcancavel).

    Acima do plato: `alcancavel=False`, `n_sugerido=None`,
    `recomendar_conformal=True` -- nenhum `n`, por maior que seja, resolve
    via DD-SIMCA (metodo parametrico, sem garantia distribution-free).

    Na faixa medida ou abaixo dela: devolve o MENOR `n` da tabela medida
    cuja cobertura observada ja' alcanca o alvo (nunca extrapola para um
    `n` fora de `TABELA_COBERTURA_DDSIMCA_MEDIDA` sem dizer isso na
    ressalva).
    """
    if not (0.0 < cobertura_alvo < 1.0):
        raise ValueError(
            f"cobertura_alvo deve estar em (0,1), recebido {cobertura_alvo}")

    ressalva_base = (
        "Curva de cobertura vs n medida em dado SINTETICO gaussiano de "
        "referencia (scripts/medicoes/medir_ddsimca_cobertura_vs_n.py), "
        "NAO no dataset real do usuario -- a estrutura de correlacao "
        f"espectral real pode diferir. Verificada apenas na faixa medida "
        f"n=[{_FAIXA_MEDIDA_N[0]}, {_FAIXA_MEDIDA_N[1]}]; fora dela e' "
        "extrapolacao nao verificada.")

    if cobertura_alvo > PLATO_COBERTURA_DDSIMCA:
        return OrientacaoDDSimca(
            cobertura_alvo=cobertura_alvo, alcancavel=False,
            n_sugerido=None, cobertura_no_n_sugerido=None,
            plato_observado=PLATO_COBERTURA_DDSIMCA,
            recomendar_conformal=True,
            ressalva=(
                f"{ressalva_base} Cobertura-alvo ({cobertura_alvo:.3f}) "
                f"esta' ACIMA do plato observado "
                f"(~{PLATO_COBERTURA_DDSIMCA:.2f}) -- nenhum n, por maior "
                "que seja, garante isso via DD-SIMCA (metodo parametrico "
                "chi2-momentos, sem garantia distribution-free; medido: "
                "cobertura estanca em ~0,94-0,945 mesmo em n=1200). Use o "
                "gate conformal (identificacao.py/conformal.py, "
                "ConformalOneClass) para uma garantia formal nesse nivel "
                "de exigencia."))

    for n, cov in TABELA_COBERTURA_DDSIMCA_MEDIDA:
        if cov >= cobertura_alvo:
            return OrientacaoDDSimca(
                cobertura_alvo=cobertura_alvo, alcancavel=True,
                n_sugerido=n, cobertura_no_n_sugerido=cov,
                plato_observado=PLATO_COBERTURA_DDSIMCA,
                recomendar_conformal=False,
                ressalva=(
                    f"{ressalva_base} n={n} atingiu cobertura media "
                    f"{cov:.4f} nesta medicao (desvio entre repeticoes "
                    "existe -- ver tabela completa); nao e' garantia "
                    "formal, e' referencia empirica."))

    # Nao deveria ser alcancavel aqui (o teto acima ja' cobriu o caso),
    # mas sem forcar: se a tabela mudar no futuro e nenhum ponto alcancar
    # o alvo, devolve nao-alcancavel em vez de estourar.
    return OrientacaoDDSimca(
        cobertura_alvo=cobertura_alvo, alcancavel=False, n_sugerido=None,
        cobertura_no_n_sugerido=None, plato_observado=PLATO_COBERTURA_DDSIMCA,
        recomendar_conformal=True,
        ressalva=(f"{ressalva_base} Nenhum ponto da tabela medida alcanca "
                  f"a cobertura-alvo ({cobertura_alvo:.3f})."))
