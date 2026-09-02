"""hsi_uncertainty.py — Propagacao de incerteza formal para resultados
HSI (Passo 107 da `INSTRUCAO_HSI_ROBUSTEZ_E_VALIDACAO.md`).

A heterogeneidade de pixel dentro do objeto (fracao de pixels da ROI
em DESACORDO com a classe majoritaria, ja' calculada em
`hsi_classification.ObjectAggregationResult.heterogeneidade` desde o
Passo 98) vira aqui um relatorio FORMAL -- nota de confianca textual +
os numeros crus -- em vez de so' um numero que existe no objeto de
retorno sem interpretacao nenhuma anexada.

DECISAO REGISTRADA (Passo 107 exige reportar a decisao ANTES de
implementar, nao so' depois): NAO combinar alpha por Bonferroni entre
etapas do fluxo HSI, ao contrario do fluxo tabular Detectar->
Identificar->Quantificar. Motivo: a combinacao de Bonferroni
(`identificacao.combine_alpha_bonferroni`) so' faz sentido entre
etapas SEQUENCIAIS que cada uma tem seu PROPRIO alpha (taxa de erro
Tipo I) formalmente calibrado -- e' isso que Detectar (DD-SIMCA/AD),
Identificar (conformal) e Quantificar (intervalo de predicao) tem em
comum na tabela tabular. No fluxo HSI atual:
  - Quality gate (Passo 95): limiar deterministico (SNR/saturacao),
    NAO um teste estatistico com alpha.
  - Classificacao por pixel + agregacao (Passo 98): decisao pontual
    (voto majoritario) + heterogeneidade DESCRITIVA -- nao e' uma
    garantia de cobertura com alpha proprio.
  - Identificacao de conjunto aberto (Passo 106): esta SIM tem um
    alpha formal (`ConformalOneClass`).
So' ha' UMA etapa alpha-calibrada no fluxo HSI hoje -- Bonferroni de 1
alpha so' e' o proprio alpha, nada a combinar. Se um dia o HSI ganhar
uma etapa de QUANTIFICACAO formal com intervalo de predicao proprio
(ex.: prever teor de acucar por pixel/objeto com um alpha proprio),
Bonferroni entre Identificacao e essa nova etapa passaria a fazer
sentido, espelhando o fluxo tabular -- nao antes disso.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from guaraci.hsi_classification import ObjectAggregationResult

__all__ = ["ConfidenceReport", "confidence_note", "enrich_object_results"]

#: Limiares de heterogeneidade -> nota de confianca -- HEURISTICOS
#: (engenharia, nao garantia estatistica calibrada como alpha conformal),
#: documentados explicitamente como tal para nao serem confundidos com
#: os outros mecanismos de garantia formal do projeto (DD-SIMCA,
#: conformal). Escolhidos por leitura simples: 0 = unanimidade; ate' 10%
#: de pixels em desacordo ainda e' "alta concordancia" tipica de ruido de
#: borda de segmentacao; acima de 30% o voto majoritario deixa de
#: representar bem o objeto (quase 1 em cada 3 pixels discorda).
_LIMIAR_ALTA = 0.10
_LIMIAR_MODERADA = 0.30


@dataclass
class ConfidenceReport:
    classe_predita: str
    heterogeneidade: float
    n_pixels: int
    nota_confianca: str


def confidence_note(heterogeneidade: float) -> str:
    """Nota textual HEURISTICA (nao um alpha calibrado -- ver docstring
    do modulo) a partir da fracao de pixels em desacordo com a classe
    majoritaria do objeto."""
    if heterogeneidade <= 0.0:
        return "unanimidade -- todos os pixels da ROI concordam"
    if heterogeneidade <= _LIMIAR_ALTA:
        return f"alta concordancia ({heterogeneidade:.1%} dos pixels em desacordo)"
    if heterogeneidade <= _LIMIAR_MODERADA:
        return (f"concordancia moderada ({heterogeneidade:.1%} dos pixels em "
                f"desacordo) -- revisar antes de usar isoladamente")
    return (f"baixa concordancia ({heterogeneidade:.1%} dos pixels em "
            f"desacordo) -- resultado questionavel, a classe majoritaria "
            f"pode nao representar bem o objeto")


def enrich_object_results(
        predicoes_objeto: Dict[str, "ObjectAggregationResult"],
        ) -> Dict[str, ConfidenceReport]:
    """Converte o dict `group_id -> ObjectAggregationResult` (Passo 98)
    num relatorio FORMAL `group_id -> ConfidenceReport`, com a nota de
    confianca anexada -- parte do resultado reportado ao usuario, nao
    so' um numero interno de log."""
    return {
        gid: ConfidenceReport(
            classe_predita=r.classe_predita,
            heterogeneidade=r.heterogeneidade,
            n_pixels=r.n_pixels,
            nota_confianca=confidence_note(r.heterogeneidade))
        for gid, r in predicoes_objeto.items()
    }
