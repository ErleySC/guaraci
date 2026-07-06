"""modos_analise.py — Camada de OBJETIVO CIENTIFICO da analise.

Fonte UNICA de verdade sobre QUAIS figuras/relatorios pertencem a cada
objetivo cientifico (Exploratorio / Classificacao / Quantificacao). O motor
(`pipeline.executar`) consulta `deve_gerar(cfg, chave)` em cada ponto de
geracao de figura, de modo que cada modo produza EXCLUSIVAMENTE os
resultados pertinentes ao seu objetivo — sem duplicacoes nem graficos de
outros modos (o defeito auditado: N2 e N3 geravam o mesmo conjunto de
figuras, misturando classificacao e quantificacao).

Preserva os termos N1/N2/N3 intactos (decisao do usuario): quando
`cfg.objetivo == "auto"` (default) o objetivo e' DERIVADO do nivel,
mantendo 100% do comportamento historico:

    N1 (Classificacao por especie)     -> classificacao
    N2 (Discriminacao puro/adulterado) -> classificacao
    N3 (Quantificacao de teor)         -> quantificacao

`cfg.objetivo` pode ser fixado explicitamente em "exploratorio" |
"classificacao" | "quantificacao" para SOBREPOR essa derivacao. O Modo
Exploratorio e' a funcao NOVA: um objetivo que gera apenas as analises
nao-supervisionadas (PCA/HCA/loadings/pre-processamento), sem as figuras
supervisionadas de PLS-DA nem de regressao.

Modulo PURO: sem numpy/sklearn/matplotlib, sem dependencia de pipeline —
importavel por qualquer camada (motor, CLI, app) sem risco de ciclo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Set

if TYPE_CHECKING:                      # evita import circular em runtime
    from guaraci.config import Config

# ---- Objetivos cientificos -------------------------------------------------
EXPLORATORIO = "exploratorio"
CLASSIFICACAO = "classificacao"
QUANTIFICACAO = "quantificacao"

OBJETIVOS_VALIDOS: Set[str] = {EXPLORATORIO, CLASSIFICACAO, QUANTIFICACAO}

# Rotulo amigavel (UI/terminal). Valor interno inalterado.
OBJETIVO_ROTULO: Dict[str, str] = {
    EXPLORATORIO: "Exploratorio",
    CLASSIFICACAO: "Classificacao",
    QUANTIFICACAO: "Quantificacao",
}

# Derivacao objetivo <- nivel quando cfg.objetivo == "auto" (preserva
# comportamento historico; NAO renomeia N1/N2/N3).
_OBJETIVO_POR_NIVEL: Dict[str, str] = {
    "N1": CLASSIFICACAO,
    "N2": CLASSIFICACAO,
    "N3": QUANTIFICACAO,
}

# ---- Pertinencia figura -> objetivo(s) -------------------------------------
# Chaves consultadas por pipeline.executar() em cada ponto de geracao. Uma
# figura so' e' salva se o objetivo resolvido pertencer a este conjunto.
# Figuras de OVERVIEW (fig1_pca_scores, fig3_outliers T2/Q) NAO aparecem aqui
# de proposito: sao contexto valido em qualquer modo e alimentam o resumo,
# entao `deve_gerar` retorna True para elas (fail-open p/ chave desconhecida).
_FIG_OBJETIVOS: Dict[str, Set[str]] = {
    # --- Exploratorias (nao-supervisionadas) ---
    "hca":                {EXPLORATORIO},
    "loadings":           {EXPLORATORIO},
    "preprocessamento":   {EXPLORATORIO},
    # --- Classificacao (supervisionada PLS-DA e derivados) ---
    "plsda_scores":       {CLASSIFICACAO},
    "confusao":           {CLASSIFICACAO},
    "roc":                {CLASSIFICACAO},
    "vip":                {CLASSIFICACAO},
    "selecao_lvs":        {CLASSIFICACAO},
    "sr_vip":             {CLASSIFICACAO},
    "score_contribution": {CLASSIFICACAO},
    "ddsimca":            {CLASSIFICACAO},
    "opls":               {CLASSIFICACAO},
    "etapa4":             {CLASSIFICACAO},
    "comparar_pipelines": {CLASSIFICACAO},
    "wold":               {CLASSIFICACAO},
    "holdout":            {CLASSIFICACAO},
    "martens":            {CLASSIFICACAO},
    "benchmark":          {CLASSIFICACAO},
    "monte_carlo":        {CLASSIFICACAO},
    "shap":               {CLASSIFICACAO},
    # --- Quantificacao (regressao PLS) ---
    "regressao":          {QUANTIFICACAO},
}

# Descricao curta de cada chave (para o painel de terminal / UI).
_FIG_DESCRICAO: Dict[str, str] = {
    "hca": "Dendrograma HCA",
    "loadings": "Loadings PCA",
    "preprocessamento": "Efeito do pre-processamento",
    "plsda_scores": "Scores PLS-DA",
    "confusao": "Matriz de confusao + metricas por classe",
    "roc": "Curvas ROC/AUC multiclasse",
    "vip": "VIP scores (bootstrap)",
    "selecao_lvs": "Selecao de variaveis latentes",
    "sr_vip": "Selectivity Ratio + VIP",
    "score_contribution": "Contribuicao de score",
    "ddsimca": "DD-SIMCA (autenticacao de pureza)",
    "opls": "OPLS-DA (scores + S-plot)",
    "etapa4": "Selecao de variaveis (iPLS/sPLS-DA/SPA/AG)",
    "comparar_pipelines": "Comparacao de pipelines de pre-proc",
    "wold": "Teste de permutacao (Wold)",
    "holdout": "Avaliacao em holdout externo",
    "martens": "Teste de incerteza de Martens",
    "benchmark": "Auto-Benchmark (SVM/RF/GBM/XGBoost)",
    "monte_carlo": "Monte Carlo CV",
    "shap": "SHAP values",
    "regressao": "Regressao PLS + figuras de merito analiticas",
}


def resolver_objetivo(cfg: "Config") -> str:
    """Retorna o objetivo cientifico efetivo do run.

    Prioridade: `cfg.objetivo` explicito (exploratorio/classificacao/
    quantificacao) sobrepoe; caso contrario ('auto' ou ausente/invalido)
    deriva do `cfg.nivel` preservando o comportamento historico.
    """
    obj = (getattr(cfg, "objetivo", "auto") or "auto").strip().lower()
    if obj in OBJETIVOS_VALIDOS:
        return obj
    return _OBJETIVO_POR_NIVEL.get(getattr(cfg, "nivel", "N1"), CLASSIFICACAO)


def deve_gerar(cfg: "Config", chave: str) -> bool:
    """True se a figura `chave` pertence ao objetivo resolvido de `cfg`.

    Chaves nao mapeadas (overview PCA/outliers, ou futuras) retornam True
    (fail-open): o gating so' SUPRIME o que sabidamente pertence a outro
    modo; nunca esconde silenciosamente uma figura desconhecida.
    """
    objetivos = _FIG_OBJETIVOS.get(chave)
    if objetivos is None:
        return True
    return resolver_objetivo(cfg) in objetivos


def figuras_exploratorias_ligadas(cfg: "Config") -> bool:
    """Regra unica p/ as exploratorias opcionais (HCA/loadings/pre-proc).

    Ligadas quando o objetivo e' Exploratorio (nucleo do modo) OU quando o
    usuario pediu detalhe extra dentro da Classificacao (escotilha de
    compatibilidade: `figuras_detalhadas=True` nunca perde funcionalidade).
    Em Quantificacao ficam desligadas (filtradas).
    """
    obj = resolver_objetivo(cfg)
    if obj == EXPLORATORIO:
        return True
    if obj == CLASSIFICACAO and getattr(cfg, "figuras_detalhadas", False):
        return True
    return False


def plano_de_figuras(cfg: "Config") -> List[str]:
    """Lista ordenada das chaves de figura pertinentes ao objetivo do run.

    Base para o painel de terminal (secao 5 da auditoria) e para a UI
    informar 'quais graficos serao produzidos' antes de executar.
    """
    obj = resolver_objetivo(cfg)
    return [chave for chave, objs in _FIG_OBJETIVOS.items() if obj in objs]


def descrever_plano(cfg: "Config") -> List[str]:
    """Descricoes legiveis das figuras pertinentes (para exibir ao usuario)."""
    return [_FIG_DESCRICAO.get(k, k) for k in plano_de_figuras(cfg)]
