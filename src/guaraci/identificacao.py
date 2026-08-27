# -*- coding: utf-8 -*-
"""identificacao.py -- Identificacao de combinacao especie x adulterante (Bloco 9b).

POR QUE ESTE MODULO EXISTE
---------------------------
Medido e RE-VERIFICADO com script reprodutivel em 2026-08-26 (ver
docs/MANUAL.md, secao "Limitacoes conhecidas", e
scripts/medicoes/medir_especie_vs_adulterante_permanova.py): especie
explica de 21x a 175x mais variancia do desvio espectral de uma amostra
adulterada que o TIPO de adulterante (algodao/milho/soja) -- retratacao de
uma estimativa anterior ("6 a 13x") que nunca teve script reprodutivel e
cuja aritmetica publicada nao fechava; a conclusao qualitativa (especie
domina) nao muda, fica mais forte. Essa relacao se FORTALECE em teor mais
alto -- a matriz-hospedeira domina o sinal. Agregar
"soja em Andiroba" e "soja em Castanha do Para" na mesma classe "soja"
violaria exchangeability (nao sao a mesma populacao estatistica). Por isso o
identificador e' calibrado por COMBINACAO especie x adulterante (ate 38 no
dataset atual: 13 especies x 3 adulterantes, menos combinacoes ausentes),
nao por adulterante agregado.

ESCORE DE NAO-CONFORMIDADE
---------------------------
Reaproveita o MESMO espaco PCA do dominio de aplicabilidade ja salvo no
pacote de modelo (`chemometric_stats.training_applicability_domain`), em vez
de ajustar um espaco novo por combinacao -- a maioria das 38 combinacoes tem
menos de 15 espectros vindos de 1-2 sessoes, insuficiente para uma PCA
propria sem overfitting severo. O escore de uma amostra e' a distancia
T2-like (normalizada pela variancia dos scores de TREINO, `var_t`, a mesma
usada no AD) entre seus scores PCA e o CENTROIDE PCA da combinacao:

    score = sum( (T_amostra - T_centroide_combinacao)^2 / var_t )

Isso mede "quao tipica esta amostra e' da assinatura conhecida daquela
combinacao", na mesma escala/normalizacao ja usada no restante do projeto
(T2 do DD-SIMCA/AD), sem inventar uma metrica nova.

COBERTURA: POR QUE A MAIORIA NAO E' VALIDAVEL
------------------------------------------------
A validade conformal (ver conformal.py) exige `n` GRUPOS independentes de
calibracao, nao espectros -- e, medido contra o dataset real em 2026-08-25,
NAO E' o `mae_id` bruto que representa essa independencia aqui: uma amostra
adulterada tem UM `mae_id` POR NIVEL DE TEOR (ex.: 'AND-10-06-2099-A1.05',
'AND-10-06-2099-A2.11', ... -- 15 diluicoes da MESMA sessao '10-06-2099').
Contar `mae_id` bruto infla o `n` de 1 sessao real para ate' 15 -- a mesma
pseudo-replicacao que o restante do projeto evita com GroupKFold, aqui no
passo de calibracao conformal. Por isso o "grupo" usado abaixo e' a SESSAO
DE COLETA (`dados_io.session_from_mae_id`, que remove o token de teor do
final do `mae_id`), nao o `mae_id` em si. Com essa correcao, 36 das 38
combinacoes tem exatamente 1 sessao de coleta independente e 2 tem
exatamente 2 (Andiroba x soja, Maracuja x algodao) -- nenhuma chega perto
do minimo pratico (n>=19 para alpha=0.05). A distincao entre os dois grupos
importa:

    n_grupos == 1  -> NAO_VALIDADO_N1: nem um limiar conformal minimamente
                       informativo pode ser calculado (variabilidade zero
                       dentro da propria calibracao). alpha_alcancavel=None
                       -- nao dignificamos esse caso com um numero.
    n_grupos == 2  -> NAO_VALIDADO_N2_FRACO: um limiar EXISTE, mas o alpha
                       garantivel e' 1/(2+1)=0,333 -- reportado explicito,
                       com a MESMA linguagem de ressalva forte usada no
                       gate DD-SIMCA para n pequeno (nunca "n/a" generico).
    n_grupos >= 3 mas ainda < 19 -> mesmo bucket NAO_VALIDADO_N2_FRACO,
                       generalizado: o nome reflete os dois casos atuais do
                       dataset, mas a logica cobre qualquer n intermediario
                       que ja produz um alpha_alcancavel real, so' que fraco.
    conformal alcancavel no alpha nominal (0.05) -> VALIDADO.

NUNCA FORCA CLASSE
-------------------
`identify_sample` so' devolve `classe_identificada` quando EXATAMENTE uma
combinacao e' aceita sob cobertura VALIDADA. Fora disso (a realidade do
dataset atual, em TODAS as 38 combinacoes), `classe_identificada=None` --
as combinacoes mais proximas aparecem so' em `candidatos_ambiguos`,
explicitamente rotuladas como palpite informacional, nunca como resultado
com garantia estatistica.

Referencias:
    Vovk V., Gammerman A. & Shafer G. (2005). Algorithmic Learning in a
    Random World. Springer. (limite de uniao/Bonferroni sobre eventos de
    cobertura conformal sequenciais)
    Angelopoulos A. N. & Bates S. (2022). A Gentle Introduction to
    Conformal Prediction and Distribution-Free Uncertainty Quantification.
    arXiv:2107.07511. (composicao de garantias conformal em pipelines
    multi-etapa via limite de uniao)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from guaraci.conformal import ConformalOneClass, achievable_alpha
from guaraci.dados_io import adulterant_from_mae_id, session_from_mae_id

log = logging.getLogger(__name__)

__all__ = [
    "CoverageStatus",
    "IdentificationResult",
    "combine_alpha_bonferroni",
    "train_identification_ensemble",
    "identify_sample",
]


class CoverageStatus(str, Enum):
    """Status de cobertura estatistica de uma identificacao (ver docstring
    do modulo). Enum explicito, nunca string magica solta -- testavel por
    igualdade/pertencimento."""

    VALIDATED = "validado"
    NOT_VALIDATED_N1 = "nao_validado_n1"
    NOT_VALIDATED_N2_WEAK = "nao_validado_n2_fraco"


@dataclass
class IdentificationResult:
    """Resultado estruturado de `identify_sample` (D3, Bloco 9b)."""

    classe_identificada: Optional[str]
    candidatos_ambiguos: List[str] = field(default_factory=list)
    cobertura_status: Optional[CoverageStatus] = None
    alpha_alcancavel: Optional[float] = None
    escores: Dict[str, float] = field(default_factory=dict)


def combine_alpha_bonferroni(*alphas: Optional[float]) -> Optional[float]:
    """Limite de uniao (Bonferroni) sobre os alpha de etapas SEQUENCIAIS
    (Detectar -> Identificar -> Quantificar), nao soma-de-variancias GUM --
    cada etapa e' um PORTAO independente (a amostra pode falhar em qualquer
    uma), nao uma fonte de erro da MESMA grandeza fisica somando-se as
    outras. Sob o limite de uniao,

        P(falha em pelo menos uma etapa) <= sum(alpha_i)

    (Vovk, Gammerman & Shafer 2005; Angelopoulos & Bates 2022, secao sobre
    composicao de garantias). Retorna `None` se QUALQUER alpha de entrada
    for `None` (etapa sem alpha computavel -- nao ha' base para bounded o
    total; ver `CoverageStatus.NOT_VALIDADO_N1`). O resultado nunca excede
    1.0 (probabilidade); se a soma estourar, o limite deixa de ser
    informativo (equivale a "sem garantia util") e um aviso e' logado.
    """
    if any(a is None for a in alphas):
        return None
    total = float(sum(float(a) for a in alphas))
    if total >= 1.0:
        log.warning(
            "[Bonferroni] soma dos alpha (%.4g) >= 1.0 -- o limite de uniao "
            "deixou de ser informativo (equivale a nenhuma garantia util). "
            "Reportado como 1.0.", total)
        return 1.0
    return total


def _pca_scores(pca, X_proc: np.ndarray) -> np.ndarray:
    return np.asarray(pca.transform(np.asarray(X_proc, dtype=float)), dtype=float)


def _score_to_centroid(T: np.ndarray, centroide: np.ndarray,
                        var_t: np.ndarray) -> np.ndarray:
    """Distancia T2-like (normalizada por `var_t`, o mesmo usado no AD) de
    cada linha de `T` (scores PCA) ao `centroide` (scores PCA) de uma
    combinacao especie x adulterante."""
    d = T - centroide.reshape(1, -1)
    return np.sum((d ** 2) / var_t.reshape(1, -1), axis=1)


def train_identification_ensemble(
        pca, var_t: np.ndarray, X_proc: np.ndarray, rotulos: np.ndarray,
        conc: np.ndarray, mae_id: Optional[np.ndarray],
        alpha_nominal: float = 0.05) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Calibra um `ConformalOneClass` por combinacao especie x adulterante
    presente nos dados de treino. Ver docstring do modulo para o escore e
    a classificacao de cobertura.

    `mae_id` e' OBRIGATORIO (sem ele nao ha' como contar grupos/sessoes
    independentes, e o resultado seria sempre NAO_VALIDADO_N1 por falta de
    informacao, nao por falta real de dados) -- retorna vazio se ausente.
    O "grupo" de calibracao e' a SESSAO DE COLETA (`dados_io.
    session_from_mae_id`), nao o `mae_id` bruto -- ver docstring do modulo.

    Retorna dict {(especie, adulterante): {"centroide", "conformal"
    (ConformalOneClass|None), "n_grupos", "n_amostras", "cobertura_status",
    "alpha_alcancavel"}}.
    """
    ensemble: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if mae_id is None:
        log.warning(
            "[Identificar] mae_id ausente -- ensemble de identificacao NAO "
            "pode ser calibrado (sem ele nao ha' como contar sessoes "
            "independentes). Retornando ensemble vazio.")
        return ensemble

    rotulos = np.asarray(rotulos, dtype=str)
    mae_id = np.asarray(mae_id, dtype=str)
    sessao_por_amostra = np.array(
        [session_from_mae_id(m) for m in mae_id], dtype=str)
    conc_v = np.asarray(conc, dtype=float)
    conc_v = np.where(np.isnan(conc_v), 0.0, conc_v)
    adult_por_amostra = np.array(
        [adulterant_from_mae_id(m) for m in mae_id], dtype=object)

    T_all = _pca_scores(pca, X_proc)
    var_t = np.asarray(var_t, dtype=float)

    especies = sorted({str(r) for r in rotulos})
    adulterantes = sorted({a for a in adult_por_amostra if a})

    for esp in especies:
        mask_esp = (rotulos == esp)
        for adult in adulterantes:
            mask = mask_esp & (adult_por_amostra == adult) & (conc_v > 0.0)
            n_amostras = int(mask.sum())
            if n_amostras == 0:
                continue

            T_combo = T_all[mask]
            grupos = sessao_por_amostra[mask]
            n_grupos = int(len(np.unique(grupos)))
            centroide = T_combo.mean(axis=0)
            scores = _score_to_centroid(T_combo, centroide, var_t)

            conformal: Optional[ConformalOneClass] = None
            if n_grupos <= 1:
                status = CoverageStatus.NOT_VALIDATED_N1
                alpha_alcancavel: Optional[float] = None
            else:
                conformal = ConformalOneClass(alpha=alpha_nominal).fit(
                    scores, mae_id=grupos)
                if conformal.info_["alcancavel"]:
                    status = CoverageStatus.VALIDATED
                    alpha_alcancavel = float(alpha_nominal)
                else:
                    status = CoverageStatus.NOT_VALIDATED_N2_WEAK
                    alpha_alcancavel = achievable_alpha(n_grupos)

            ensemble[(esp, adult)] = {
                "centroide": centroide,
                "conformal": conformal,
                "n_grupos": n_grupos,
                "n_amostras": n_amostras,
                "cobertura_status": status,
                "alpha_alcancavel": alpha_alcancavel,
            }

    return ensemble


def identify_sample(
        ensemble: Dict[Tuple[str, str], Dict[str, Any]],
        pca, var_t: np.ndarray, X_proc_amostra: np.ndarray,
        n_candidatos: int = 3) -> IdentificationResult:
    """Identifica a combinacao especie x adulterante de UMA amostra nova
    (espectro ja pre-processado, 1 linha ou vetor 1D) contra o `ensemble`
    calibrado por `train_identification_ensemble`.

    Nunca forca uma classe: `classe_identificada` so' e' preenchido quando
    EXATAMENTE uma combinacao e' aceita sob cobertura VALIDADA. Caso
    contrario (a realidade do dataset atual, em toda combinacao),
    `classe_identificada=None` e as combinacoes mais proximas aparecem em
    `candidatos_ambiguos` como palpite informacional, com o
    `cobertura_status`/`alpha_alcancavel` da combinacao mais proxima
    (o "teto" de garantia que se poderia esperar, mesmo sem alcanca-lo).
    """
    if not ensemble:
        return IdentificationResult(
            classe_identificada=None, candidatos_ambiguos=[],
            cobertura_status=None, alpha_alcancavel=None, escores={})

    X = np.asarray(X_proc_amostra, dtype=float).reshape(1, -1)
    T = _pca_scores(pca, X)
    var_t = np.asarray(var_t, dtype=float)

    escores: Dict[str, float] = {}
    aceitos: List[str] = []
    for (esp, adult), info in ensemble.items():
        rotulo = f"{esp}|{adult}"
        score = float(_score_to_centroid(T, info["centroide"], var_t)[0])
        escores[rotulo] = score
        conformal = info["conformal"]
        if (info["cobertura_status"] == CoverageStatus.VALIDATED
                and conformal is not None
                and bool(conformal.predict(np.array([score]))[0])):
            aceitos.append(rotulo)

    ordenados = sorted(escores.items(), key=lambda kv: kv[1])
    candidatos_proximos = [rotulo for rotulo, _ in ordenados[:n_candidatos]]

    if len(aceitos) == 1:
        rotulo_ok = aceitos[0]
        esp_ok, adult_ok = rotulo_ok.split("|", 1)
        return IdentificationResult(
            classe_identificada=rotulo_ok,
            candidatos_ambiguos=[],
            cobertura_status=CoverageStatus.VALIDATED,
            alpha_alcancavel=ensemble[(esp_ok, adult_ok)]["alpha_alcancavel"],
            escores=escores)

    if len(aceitos) > 1:
        # Mais de uma combinacao aceita sob cobertura validada -- ambiguo
        # por construcao (garantia estatistica nao decide ENTRE elas).
        return IdentificationResult(
            classe_identificada=None,
            candidatos_ambiguos=sorted(aceitos),
            cobertura_status=CoverageStatus.VALIDATED,
            alpha_alcancavel=min(
                ensemble[(_esp_r, _adult_r)]["alpha_alcancavel"]
                for _esp_r, _adult_r in (r.split("|", 1) for r in aceitos)),
            escores=escores)

    # Nenhuma combinacao aceita sob cobertura validada -- reporta a mais
    # proxima como referencia do "teto" de garantia, sem forcar classe.
    melhor_rotulo = candidatos_proximos[0] if candidatos_proximos else None
    if melhor_rotulo is None:
        return IdentificationResult(
            classe_identificada=None, candidatos_ambiguos=[],
            cobertura_status=None, alpha_alcancavel=None, escores=escores)
    esp_m, adult_m = melhor_rotulo.split("|", 1)
    info_m = ensemble[(esp_m, adult_m)]
    return IdentificationResult(
        classe_identificada=None,
        candidatos_ambiguos=candidatos_proximos,
        cobertura_status=info_m["cobertura_status"],
        alpha_alcancavel=info_m["alpha_alcancavel"],
        escores=escores)
