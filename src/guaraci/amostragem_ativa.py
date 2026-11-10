# -*- coding: utf-8 -*-
"""amostragem_ativa.py -- Amostragem ativa orientada por incerteza
(Bloco 25): prioriza ONDE coletar mais amostra fisica, reaproveitando o
que o pipeline ja calcula (nunca recalculado do zero aqui):

  - `identificacao.train_identification_ensemble` (cobertura_status por
    combinacao especie x adulterante, ja' calibrado com `ConformalOneClass`)
  - `conformal.n_minimum_for_alpha`/`achievable_alpha` (quantas sessoes
    de coleta faltam para uma combinacao alcancar o alpha desejado)
  - erro por especie ja' medido (RMSEP/balanced_accuracy, quando o
    chamador ja tem esse numero -- este modulo NAO re-treina modelo
    nenhum so' pra' obter erro, isso duplicaria trabalho pesado que o
    pipeline ja' fez)

ESCOPO -- `applicability_domain` NAO entra diretamente: AD e' uma
propriedade POR AMOSTRA NOVA (essa amostra especifica esta dentro do
que o treino ja cobre?), nao um agregado natural "onde investir a
proxima coleta" por combinacao -- o sinal que de fato responde essa
pergunta e' cobertura estatistica (quantas sessoes independentes faltam
pra validar), que e' o que este modulo usa.

PRIORIZACAO: combinacoes JA validadas tem prioridade 0 (nada a fazer).
Entre as NAO validadas, prioridade = 1/(1+sessoes_faltantes) -- quanto
MENOS sessoes faltam pra' cruzar o limiar de validacao, MAIOR a
prioridade (maior "impacto esperado" por sessao investida: uma
combinacao a 1 sessao de distancia da validacao e' um alvo muito mais
eficiente que uma a 15 sessoes). Quando `erro_por_especie` e' fornecido,
multiplica a prioridade pelo erro normalizado da especie (especie com
erro pior ganha peso extra -- mais a ganhar com dado novo)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from guaraci.conformal import n_minimum_for_alpha
from guaraci.identificacao import CoverageStatus

__all__ = [
    "PrioridadeAmostragem",
    "priorizar_amostragem",
]


@dataclass
class PrioridadeAmostragem:
    """Uma linha da lista priorizada -- uma combinacao especie x
    adulterante. `prioridade` e' um score RELATIVO (maior = mais
    urgente), nao uma probabilidade nem um numero absoluto."""
    especie: str
    adulterante: str
    n_sessoes_atual: int
    n_sessoes_necessario: int
    sessoes_faltantes: int
    cobertura_status: Optional[CoverageStatus]
    prioridade: float


def priorizar_amostragem(
        ensemble: Dict[Tuple[str, str], Dict[str, Any]],
        alpha: float = 0.05,
        erro_por_especie: Optional[Dict[str, float]] = None,
        ) -> List[PrioridadeAmostragem]:
    """Constroi a lista priorizada a partir de um `ensemble` JA calibrado
    (mesmo dict devolvido por
    `identificacao.train_identification_ensemble` -- este modulo nunca
    re-treina/recalibra, so' consome).

    `erro_por_especie` (opcional): dict especie->erro (RMSEP ou
    1-balanced_accuracy, quanto MAIOR pior) ja' medido em outro lugar do
    pipeline. Quando fornecido, especies com erro pior recebem peso
    extra na prioridade (normalizado pelo maior erro do dict, 1.0 =
    pior especie do lote). Sem ele, todas as especies pesam igual.

    Retorna lista ordenada por prioridade DECRESCENTE (primeiro = mais
    urgente). Combinacoes ja `CoverageStatus.VALIDATED` sempre ficam por
    ultimo (prioridade 0.0) -- ja' tem garantia estatistica, nao
    precisam de mais coleta pra' esse fim."""
    n_necessario = n_minimum_for_alpha(alpha)

    erro_max = (max(erro_por_especie.values())
                if erro_por_especie else None)

    linhas: List[PrioridadeAmostragem] = []
    for (esp, adult), info in ensemble.items():
        n_atual = int(info.get("n_grupos", 0))
        status = info.get("cobertura_status")
        faltam = max(0, n_necessario - n_atual)

        if status == CoverageStatus.VALIDATED:
            prioridade = 0.0
        else:
            prioridade = 1.0 / (1.0 + faltam)
            if erro_por_especie and erro_max and erro_max > 1e-300:
                peso_erro = erro_por_especie.get(esp, 0.0) / erro_max
                # peso em [0.5, 1.5]: erro pior empurra prioridade pra
                # cima sem jamais zerar a prioridade de uma especie com
                # erro nao medido (peso_erro=0 -> multiplicador 0.5, nao 0)
                prioridade *= (0.5 + peso_erro)

        linhas.append(PrioridadeAmostragem(
            especie=esp, adulterante=adult, n_sessoes_atual=n_atual,
            n_sessoes_necessario=n_necessario, sessoes_faltantes=faltam,
            cobertura_status=status, prioridade=prioridade))

    linhas.sort(key=lambda r: -r.prioridade)
    return linhas
