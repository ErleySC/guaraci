"""
cli_assistente.py — Assistente CLI hierarquico para o Pipeline Quimiometrico FT-NIR
AmaNIR — Plataforma Quimiometrica FT-NIR
GEAAp / UFPA — Plataforma de autenticacao de oleos vegetais amazonicos.

Uso:
    python cli_assistente.py
    python pineline_quimiometria_14.py   (chama este modulo automaticamente)

Requer: pineline_quimiometria_14.py no mesmo diretorio.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Integracao com o pipeline (sem modificar nenhuma funcao analitica)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pineline_quimiometria_14 as pq

# Atalhos
Config = pq.Config
_CONFIG_SPEC = pq._CONFIG_SPEC
executar = pq.executar
salvar_config = pq.salvar_config
carregar_config = pq.carregar_config
_attr_para_yaml = pq._attr_para_yaml
_fmt_yaml = pq._fmt_yaml
_coagir_valor = pq._coagir_valor

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
_BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_CFG_PATH = _BASE_DIR / "config.yaml"
_PERFIS_DIR = _BASE_DIR / "perfis"
_WIZARD_FLAG = _BASE_DIR / ".cli_wizard_done"

# ---------------------------------------------------------------------------
# Estado global mutavel — idioma persiste em tempo real
# ---------------------------------------------------------------------------
_STATE: Dict[str, Any] = {"lang": "PT"}


def _lang() -> str:
    """Retorna o idioma atual do estado global."""
    return _STATE["lang"]


def _set_lang(l: str) -> None:
    """Define o idioma no estado global e persiste no arquivo de flag."""
    _STATE["lang"] = l
    try:
        _WIZARD_FLAG.write_text(l, encoding="utf-8")
    except OSError:
        pass


def _toggle_idioma() -> str:
    """Alterna entre PT e EN e retorna o novo idioma."""
    novo = "EN" if _lang() == "PT" else "PT"
    _set_lang(novo)
    return novo


# ---------------------------------------------------------------------------
# Internacionalizacao
# ---------------------------------------------------------------------------
I18N: Dict[str, Dict[str, str]] = {
    "PT": {
        "titulo": "AmaNIR — Plataforma Quimiometrica FT-NIR",
        "subtitulo": "GEAAp / UFPA  |  Oleos Vegetais Amazonicos",
        "menu_projeto": "Projeto",
        "menu_dados": "Dados",
        "menu_preproc": "Pre-processamento",
        "menu_modelo": "Modelagem",
        "menu_valid": "Validacao",
        "menu_avancado": "Metodos Avancados",
        "menu_viz": "Visualizacao",
        "menu_ajuda": "Ajuda",
        "salvar": "Salvar Perfil",
        "carregar": "Carregar Perfil",
        "rodar": "Rodar Pipeline",
        "sair": "Sair",
        "perfis": "Perfis Prontos",
        "aviso_analitico": "Alteracao ANALITICA — pode modificar resultados. Confirma? (s/n): ",
        "aviso_avancado": "Parametro AVANCADO — aumenta tempo de processamento.",
        "status_ok": "Dados OK",
        "status_erro": "Pasta nao encontrada",
        "idioma": "Idioma",
        "opcao": "Opcao",
        "campo_atualizado": "[{campo}] atualizado: {valor}",
        "cancelado": "Operacao cancelada.",
        "invalido": "Opcao invalida.",
        "confirmar_sn": "s",
        "nao": "n",
        "voltar": "Voltar",
        "continuar": "Enter para continuar",
        "ajuda_campo": "Ajuda sobre campo",
        "novo_valor": "Novo valor (Enter = manter, ? = ajuda completa): ",
        "mantido": "Mantido.",
        "atual": "Atual",
        "padrao": "Padrao",
        "faixa": "Faixa",
        "impacto": "Impacto",
        "exemplos": "Exemplos",
        "descricao": "Descricao",
        "listar_todos": "Listar todos",
    },
    "EN": {
        "titulo": "AmaNIR — FT-NIR Chemometrics Platform",
        "subtitulo": "GEAAp / UFPA  |  Amazonian Vegetable Oils",
        "menu_projeto": "Project",
        "menu_dados": "Data",
        "menu_preproc": "Preprocessing",
        "menu_modelo": "Modelling",
        "menu_valid": "Validation",
        "menu_avancado": "Advanced Methods",
        "menu_viz": "Visualization",
        "menu_ajuda": "Help",
        "salvar": "Save Profile",
        "carregar": "Load Profile",
        "rodar": "Run Pipeline",
        "sair": "Exit",
        "perfis": "Preset Profiles",
        "aviso_analitico": "ANALYTICAL change — may affect results. Confirm? (y/n): ",
        "aviso_avancado": "ADVANCED parameter — increases processing time.",
        "status_ok": "Data OK",
        "status_erro": "Folder not found",
        "idioma": "Language",
        "opcao": "Option",
        "campo_atualizado": "[{campo}] updated: {valor}",
        "cancelado": "Operation cancelled.",
        "invalido": "Invalid option.",
        "confirmar_sn": "y",
        "nao": "n",
        "voltar": "Back",
        "continuar": "Enter to continue",
        "ajuda_campo": "Help on field",
        "novo_valor": "New value (Enter = keep, ? = full help): ",
        "mantido": "Kept.",
        "atual": "Current",
        "padrao": "Default",
        "faixa": "Range",
        "impacto": "Impact",
        "exemplos": "Examples",
        "descricao": "Description",
        "listar_todos": "List all",
    },
}

# ---------------------------------------------------------------------------
# Classificacao de risco
# ---------------------------------------------------------------------------
RISK_CLASS: Dict[str, str] = {
    # VISUAL
    "dpi": "VISUAL", "formato_figura": "VISUAL",
    "figuras_mostrar_marcadores": "VISUAL", "figuras_mostrar_elipses": "VISUAL",
    "abrir_figuras_na_tela": "VISUAL",
    # ANALITICO
    "pre_processamento": "ANALITICO", "max_lvs": "ANALITICO",
    "n_permutacoes": "ANALITICO", "holdout_fracao": "ANALITICO",
    "nivel": "ANALITICO", "excluir_classes": "ANALITICO",
    "faixa_min_cm": "ANALITICO", "faixa_max_cm": "ANALITICO",
    "modo_ddsimca": "ANALITICO", "ddsimca": "ANALITICO",
    "opls_da": "ANALITICO", "selecao_variaveis_etapa4": "ANALITICO",
    "comparar_pre_processamentos": "ANALITICO",
    "validacao_group_aware": "ANALITICO", "teste_wold": "ANALITICO",
    "teste_cv_anova": "ANALITICO", "pasta_dados": "ANALITICO",
    "pasta_saida": "ANALITICO", "modo_entrada": "ANALITICO",
    "arquivo_csv": "ANALITICO", "coluna_classe": "ANALITICO",
    "coluna_concentracao": "ANALITICO",
    # AVANCADO
    "benchmark": "AVANCADO", "monte_carlo": "AVANCADO",
    "shap_benchmark": "AVANCADO", "n_monte_carlo": "AVANCADO",
    "shap_max_amostras": "AVANCADO", "monte_carlo_incluir_todos": "AVANCADO",
}

RISK_COLOR: Dict[str, str] = {
    "VISUAL": "\033[92m",     # verde
    "ANALITICO": "\033[93m",  # amarelo
    "AVANCADO": "\033[91m",   # vermelho
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "CYAN": "\033[96m",
}

# ---------------------------------------------------------------------------
# Nomes traduzidos dos campos (chave tecnica → nome legível por idioma)
# ---------------------------------------------------------------------------
FIELD_NAMES: Dict[str, Dict[str, str]] = {
    "pasta_dados":                  {"PT": "Pasta de entrada",        "EN": "Input folder"},
    "pasta_saida":                  {"PT": "Pasta de saida",          "EN": "Output folder"},
    "modo_entrada":                 {"PT": "Modo de entrada",         "EN": "Input mode"},
    "arquivo_csv":                  {"PT": "Arquivo CSV",             "EN": "CSV file"},
    "coluna_classe":                {"PT": "Coluna de classe",        "EN": "Class column"},
    "coluna_concentracao":          {"PT": "Coluna concentracao",     "EN": "Concentration column"},
    "faixa_min_cm":                 {"PT": "Faixa minima (cm-1)",     "EN": "Min range (cm-1)"},
    "faixa_max_cm":                 {"PT": "Faixa maxima (cm-1)",     "EN": "Max range (cm-1)"},
    "excluir_classes":              {"PT": "Excluir classes",         "EN": "Exclude classes"},
    "pre_processamento":            {"PT": "Pre-processamento",       "EN": "Preprocessing"},
    "comparar_pre_processamentos":  {"PT": "Comparar pre-proc.",      "EN": "Compare preproc."},
    "nivel":                        {"PT": "Nivel de analise",        "EN": "Analysis level"},
    "max_lvs":                      {"PT": "Maximo de LVs",           "EN": "Max LVs"},
    "opls_da":                      {"PT": "OPLS-DA",                 "EN": "OPLS-DA"},
    "ddsimca":                      {"PT": "DD-SIMCA",                "EN": "DD-SIMCA"},
    "modo_ddsimca":                 {"PT": "Modo DD-SIMCA",           "EN": "DD-SIMCA mode"},
    "selecao_variaveis_etapa4":     {"PT": "Selecao de variaveis",    "EN": "Variable selection"},
    "holdout_fracao":               {"PT": "Fracao holdout",          "EN": "Holdout fraction"},
    "validacao_group_aware":        {"PT": "Validacao group-aware",   "EN": "Group-aware CV"},
    "n_permutacoes":                {"PT": "N. permutacoes",          "EN": "N permutations"},
    "teste_wold":                   {"PT": "Teste de Wold",           "EN": "Wold test"},
    "teste_cv_anova":               {"PT": "CV-ANOVA",                "EN": "CV-ANOVA"},
    "benchmark":                    {"PT": "Benchmark",               "EN": "Benchmark"},
    "monte_carlo":                  {"PT": "Monte Carlo CV",          "EN": "Monte Carlo CV"},
    "n_monte_carlo":                {"PT": "N. repeticoes MC",        "EN": "N MC repetitions"},
    "monte_carlo_incluir_todos":    {"PT": "MC incluir todos",        "EN": "MC include all"},
    "shap_benchmark":               {"PT": "SHAP values",             "EN": "SHAP values"},
    "shap_max_amostras":            {"PT": "SHAP max. amostras",      "EN": "SHAP max samples"},
    "figuras_mostrar_marcadores":   {"PT": "Marcadores por classe",   "EN": "Class markers"},
    "figuras_mostrar_elipses":      {"PT": "Elipses de confianca",    "EN": "Confidence ellipses"},
    "formato_figura":               {"PT": "Formato das figuras",     "EN": "Figure format"},
    "dpi":                          {"PT": "Resolucao (DPI)",         "EN": "Resolution (DPI)"},
    "abrir_figuras_na_tela":        {"PT": "Abrir figuras na tela",   "EN": "Open figures on screen"},
}


def _nome_campo(key: str) -> str:
    """Retorna o nome traduzido do campo para o idioma atual. Fallback: key."""
    return FIELD_NAMES.get(key, {}).get(_lang(), key)


# ---------------------------------------------------------------------------
# Aliases de exibicao para modo_ddsimca
# ---------------------------------------------------------------------------
_DDSIMCA_DISPLAY: Dict[str, Dict[str, str]] = {
    "PT": {"puros": "autenticacao", "todos": "exploratorio"},
    "EN": {"puros": "authentication", "todos": "exploratory"},
}

_DDSIMCA_INPUT: Dict[str, str] = {
    "autenticacao": "puros", "authentication": "puros", "puros": "puros",
    "exploratorio": "todos", "exploratory": "todos", "todos": "todos",
}

# ---------------------------------------------------------------------------
# Descricoes de secao (bilinguais)
# ---------------------------------------------------------------------------
SECTION_DESC: Dict[str, Dict[str, str]] = {
    "projeto": {
        "PT": "Pastas de entrada e saida dos dados e resultados.",
        "EN": "Input and output folders for data and results.",
    },
    "dados": {
        "PT": "Configuracao dos dados espectrais: formato, faixa, classes excluidas.",
        "EN": "Spectral data configuration: format, range, excluded classes.",
    },
    "preproc": {
        "PT": "Pre-processamento espectral: MSC corrige espalhamento; SG suaviza e deriva; MC centraliza.",
        "EN": "Spectral preprocessing: MSC corrects scattering; SG smooths and derives; MC mean-centers.",
    },
    "modelo": {
        "PT": "Configuracao dos modelos PLS-DA, OPLS-DA e DD-SIMCA.",
        "EN": "PLS-DA, OPLS-DA and DD-SIMCA model configuration.",
    },
    "validacao": {
        "PT": "Estrategia de validacao cruzada e testes estatisticos do modelo.",
        "EN": "Cross-validation strategy and statistical model tests.",
    },
    "avancado": {
        "PT": "Modulos computacionalmente intensivos: benchmark, Monte Carlo, SHAP.",
        "EN": "Computationally intensive modules: benchmark, Monte Carlo, SHAP.",
    },
    "visualizacao": {
        "PT": "Configuracoes visuais das figuras. Nao afetam resultados analiticos.",
        "EN": "Figure visual settings. Do not affect analytical results.",
    },
}

# ---------------------------------------------------------------------------
# Base de ajuda bilinguie (HELP_DB)
# ---------------------------------------------------------------------------
HELP_DB: Dict[str, Dict[str, Any]] = {
    "dpi": {
        "PT": {
            "desc": "Resolucao das figuras em pontos por polegada.",
            "impacto": "VISUAL — nao altera resultados analiticos.",
            "exemplos": {"300": "Apresentacoes/slides", "600": "Artigos cientificos", "1200": "Impressao profissional"},
        },
        "EN": {
            "desc": "Figure resolution in dots per inch.",
            "impacto": "VISUAL — does not affect analytical results.",
            "exemplos": {"300": "Presentations/slides", "600": "Scientific articles", "1200": "Professional printing"},
        },
        "default": 600, "range": "150-1200",
    },
    "max_lvs": {
        "PT": {
            "desc": "Numero maximo de Variaveis Latentes testadas na selecao automatica.",
            "impacto": "ANALITICO — afeta diretamente o modelo PLS-DA.",
            "exemplos": {"10": "Datasets pequenos", "40": "Recomendado", "80": "Datasets grandes"},
        },
        "EN": {
            "desc": "Maximum number of Latent Variables tested in automatic selection.",
            "impacto": "ANALYTICAL — directly affects the PLS-DA model.",
            "exemplos": {"10": "Small datasets", "40": "Recommended", "80": "Large datasets"},
        },
        "default": 40, "range": "5-80",
    },
    "n_permutacoes": {
        "PT": {
            "desc": "Numero de permutacoes para o teste Y-randomization.",
            "impacto": "ANALITICO — afeta o p-value do teste de permutacao.",
            "exemplos": {"50": "Diagnostico rapido", "200": "Publicacao", "1000": "Alta precisao"},
        },
        "EN": {
            "desc": "Number of permutations for the Y-randomization test.",
            "impacto": "ANALYTICAL — affects the permutation test p-value.",
            "exemplos": {"50": "Quick diagnosis", "200": "Publication", "1000": "High precision"},
        },
        "default": 200, "range": "50-1000",
    },
    "pre_processamento": {
        "PT": {
            "desc": "Pipeline de pre-processamento espectral aplicado antes do PLS-DA.",
            "impacto": "ANALITICO — MSC+SG+MC obteve Bal.Acc=0.923 no dataset atual.",
            "exemplos": {"msc_sg_mc": "Melhor resultado (recomendado)", "snv_sg_mc": "Alternativa robusta", "autoscaling": "Baseline simples"},
        },
        "EN": {
            "desc": "Spectral preprocessing pipeline applied before PLS-DA.",
            "impacto": "ANALYTICAL — MSC+SG+MC achieved Bal.Acc=0.923 on current dataset.",
            "exemplos": {"msc_sg_mc": "Best result (recommended)", "snv_sg_mc": "Robust alternative", "autoscaling": "Simple baseline"},
        },
        "default": "MSC+SG+MC", "range": "Escolha na lista / Choose from list",
    },
    "modo_ddsimca": {
        "PT": {
            "desc": "Modo de treino do DD-SIMCA para autenticacao de amostras.",
            "impacto": "ANALITICO — 'autenticacao' usa so amostras de referencia pura; 'exploratorio' usa todas.",
            "exemplos": {"autenticacao": "Validacao de autenticidade (recomendado para publicacao)", "exploratorio": "Analise exploratoria inicial"},
        },
        "EN": {
            "desc": "DD-SIMCA training mode for sample authentication.",
            "impacto": "ANALYTICAL — 'authentication' uses only pure reference samples; 'exploratory' uses all.",
            "exemplos": {"authentication": "Authenticity validation (recommended for publication)", "exploratory": "Initial exploratory analysis"},
        },
        "default": "autenticacao", "range": "autenticacao | exploratorio  (EN: authentication | exploratory)",
    },
    "nivel": {
        "PT": {
            "desc": "Nivel de analise: N1 (exploratorio), N2 (classificacao completa), N3 (regressao).",
            "impacto": "ANALITICO — define quais modulos sao executados.",
            "exemplos": {"N1": "So PCA/HCA", "N2": "Classificacao completa (PLS-DA + DD-SIMCA)", "N3": "Quantificacao (PLS-R)"},
        },
        "EN": {
            "desc": "Analysis level: N1 (exploratory), N2 (full classification), N3 (regression).",
            "impacto": "ANALYTICAL — defines which modules are executed.",
            "exemplos": {"N1": "PCA/HCA only", "N2": "Full classification (PLS-DA + DD-SIMCA)", "N3": "Quantification (PLS-R)"},
        },
        "default": "N2", "range": "N1 | N2 | N3",
    },
    "holdout_fracao": {
        "PT": {
            "desc": "Fracao do dataset reservada para validacao externa independente (holdout).",
            "impacto": "ANALITICO — afeta a estimativa de generalizacao do modelo.",
            "exemplos": {"0.1": "Poucos dados disponiveis", "0.2": "Recomendado (padrao)", "0.3": "Dataset grande"},
        },
        "EN": {
            "desc": "Fraction of the dataset held out for independent external validation.",
            "impacto": "ANALYTICAL — affects the model generalization estimate.",
            "exemplos": {"0.1": "Limited data available", "0.2": "Recommended (default)", "0.3": "Large dataset"},
        },
        "default": 0.2, "range": "0.1-0.3",
    },
    "ddsimca": {
        "PT": {
            "desc": "Ativa o DD-SIMCA para autenticacao e deteccao de amostras desconhecidas por classe.",
            "impacto": "ANALITICO — adiciona figuras de Cooman's Plot e metricas de autenticacao.",
            "exemplos": {"true": "Recomendado para publicacao", "false": "Economiza tempo em exploracao"},
        },
        "EN": {
            "desc": "Enables DD-SIMCA for per-class sample authentication and unknown detection.",
            "impacto": "ANALYTICAL — adds Cooman's Plot figures and authentication metrics.",
            "exemplos": {"true": "Recommended for publication", "false": "Saves time in exploration"},
        },
        "default": True, "range": "true | false",
    },
    "benchmark": {
        "PT": {
            "desc": "Compara PLS-DA contra SVM RBF, Random Forest e XGBoost com mesma CV group-aware.",
            "impacto": "AVANCADO — adiciona ~30-60 minutos de processamento.",
            "exemplos": {"false": "TCC/producao (recomendado)", "true": "Artigo com comparacao de classificadores"},
        },
        "EN": {
            "desc": "Compares PLS-DA against SVM RBF, Random Forest and XGBoost with same group-aware CV.",
            "impacto": "ADVANCED — adds ~30-60 minutes of processing time.",
            "exemplos": {"false": "TCC/production (recommended)", "true": "Article with classifier comparison"},
        },
        "default": False, "range": "true | false",
    },
    "formato_figura": {
        "PT": {
            "desc": "Formato de saida das figuras geradas pelo pipeline.",
            "impacto": "VISUAL — nao altera analise ou resultados.",
            "exemplos": {"png": "Apresentacoes/TCC (menor tamanho)", "pdf": "Artigos (vetorial, editavel)", "svg": "Edicao pos-processamento"},
        },
        "EN": {
            "desc": "Output format for pipeline-generated figures.",
            "impacto": "VISUAL — does not affect analysis or results.",
            "exemplos": {"png": "Presentations/TCC (smaller size)", "pdf": "Articles (vector, editable)", "svg": "Post-processing editing"},
        },
        "default": "png", "range": "png | pdf | svg",
    },
    "monte_carlo": {
        "PT": {
            "desc": "Ativa Monte Carlo CV: calcula IC95% das metricas por reamostragem estratificada por grupo.",
            "impacto": "AVANCADO — aumenta significativamente o tempo de processamento.",
            "exemplos": {"false": "Producao/TCC (recomendado)", "true": "Dissertacao/Tese"},
        },
        "EN": {
            "desc": "Enables Monte Carlo CV: computes 95% CI for metrics via stratified group resampling.",
            "impacto": "ADVANCED — significantly increases processing time.",
            "exemplos": {"false": "Production/TCC (recommended)", "true": "Dissertation/Thesis"},
        },
        "default": False, "range": "true | false",
    },
    "n_monte_carlo": {
        "PT": {
            "desc": "Numero de repeticoes do Monte Carlo CV.",
            "impacto": "AVANCADO — mais repeticoes = IC mais estreito, mas mais tempo.",
            "exemplos": {"50": "Teste rapido", "100": "TCC", "200": "Dissertacao/Tese"},
        },
        "EN": {
            "desc": "Number of Monte Carlo CV repetitions.",
            "impacto": "ADVANCED — more repetitions = narrower CI, but more time.",
            "exemplos": {"50": "Quick test", "100": "TCC", "200": "Dissertation/Thesis"},
        },
        "default": 100, "range": "50-500",
    },
    "shap_benchmark": {
        "PT": {
            "desc": "Calcula SHAP values (TreeExplainer) para RF/XGBoost — interpretabilidade espectral.",
            "impacto": "AVANCADO — requer benchmark=true; adiciona ~10-20 minutos.",
            "exemplos": {"false": "Sem SHAP (producao)", "true": "Artigo com interpretabilidade espectral"},
        },
        "EN": {
            "desc": "Computes SHAP values (TreeExplainer) for RF/XGBoost — spectral interpretability.",
            "impacto": "ADVANCED — requires benchmark=true; adds ~10-20 minutes.",
            "exemplos": {"false": "No SHAP (production)", "true": "Article with spectral interpretability"},
        },
        "default": False, "range": "true | false",
    },
    "shap_max_amostras": {
        "PT": {
            "desc": "Limite de amostras para calculo de SHAP (controla uso de memoria RAM).",
            "impacto": "AVANCADO — valores maiores aumentam tempo e uso de RAM.",
            "exemplos": {"200": "RAM limitada (<8 GB)", "500": "Recomendado (padrao)", "1000": "RAM abundante (>32 GB)"},
        },
        "EN": {
            "desc": "Sample limit for SHAP computation (controls RAM usage).",
            "impacto": "ADVANCED — larger values increase time and RAM usage.",
            "exemplos": {"200": "Limited RAM (<8 GB)", "500": "Recommended (default)", "1000": "Abundant RAM (>32 GB)"},
        },
        "default": 500, "range": "100-1000",
    },
    "validacao_group_aware": {
        "PT": {
            "desc": "Manter replicas (T1/T2/T3) sempre juntas nos folds de validacao cruzada.",
            "impacto": "ANALITICO — false pode inflar artificialmente as metricas (data leakage).",
            "exemplos": {"true": "Obrigatorio para dados com replicas fisicas", "false": "Apenas para amostras 100% independentes"},
        },
        "EN": {
            "desc": "Keep replicates (T1/T2/T3) together in all cross-validation folds.",
            "impacto": "ANALYTICAL — false may artificially inflate metrics (data leakage).",
            "exemplos": {"true": "Mandatory for data with physical replicates", "false": "Only for fully independent samples"},
        },
        "default": True, "range": "true | false",
    },
    "opls_da": {
        "PT": {
            "desc": "Executa OPLS-DA (Analise Discriminante por Minimos Quadrados Parciais Ortogonais).",
            "impacto": "ANALITICO — gera S-Plot e separa variacao ortogonal da preditiva.",
            "exemplos": {"true": "Recomendado para publicacao", "false": "Analise exploratoria rapida"},
        },
        "EN": {
            "desc": "Runs OPLS-DA (Orthogonal Partial Least Squares Discriminant Analysis).",
            "impacto": "ANALYTICAL — generates S-Plot and separates orthogonal from predictive variation.",
            "exemplos": {"true": "Recommended for publication", "false": "Quick exploratory analysis"},
        },
        "default": True, "range": "true | false",
    },
    "selecao_variaveis_etapa4": {
        "PT": {
            "desc": "Executa selecao de variaveis: iPLS, VIP>=1, SR top 20% e sPLS-DA.",
            "impacto": "ANALITICO — identifica regioes espectrais mais relevantes para o modelo.",
            "exemplos": {"true": "Publicacao (recomendado)", "false": "Exploracao rapida"},
        },
        "EN": {
            "desc": "Runs variable selection: iPLS, VIP>=1, SR top 20% and sPLS-DA.",
            "impacto": "ANALYTICAL — identifies the most relevant spectral regions for the model.",
            "exemplos": {"true": "Publication (recommended)", "false": "Quick exploration"},
        },
        "default": True, "range": "true | false",
    },
    "comparar_pre_processamentos": {
        "PT": {
            "desc": "Compara automaticamente varios pipelines de pre-processamento e reporta o melhor.",
            "impacto": "ANALITICO — aumenta o tempo de execucao; util para otimizar o pipeline.",
            "exemplos": {"false": "Usar pre-processamento padrao", "true": "Otimizacao do pipeline espectral"},
        },
        "EN": {
            "desc": "Automatically compares several preprocessing pipelines and reports the best.",
            "impacto": "ANALYTICAL — increases execution time; useful to optimize the pipeline.",
            "exemplos": {"false": "Use default preprocessing", "true": "Spectral pipeline optimization"},
        },
        "default": False, "range": "true | false",
    },
    "pasta_dados": {
        "PT": {
            "desc": "Pasta com os arquivos .dx de espectros FT-NIR (uma subpasta por classe/especie).",
            "impacto": "ANALITICO — define os dados de entrada do pipeline.",
            "exemplos": {"dados": "Pasta padrao do projeto", r"C:\meus_dados\oleos": "Caminho absoluto personalizado"},
        },
        "EN": {
            "desc": "Folder with .dx FT-NIR spectral files (one subfolder per class/species).",
            "impacto": "ANALYTICAL — defines the pipeline input data.",
            "exemplos": {"dados": "Default project folder", r"C:\meus_dados\oleos": "Custom absolute path"},
        },
        "default": "dados", "range": "Valid system path",
    },
    "pasta_saida": {
        "PT": {
            "desc": "Pasta onde os resultados (figuras, metricas, relatorios) serao gravados.",
            "impacto": "ANALITICO — define onde as saidas do pipeline sao salvas.",
            "exemplos": {"resultados": "Pasta padrao", r"C:\experimento_01": "Experimento com nome especifico"},
        },
        "EN": {
            "desc": "Folder where results (figures, metrics, reports) will be saved.",
            "impacto": "ANALYTICAL — defines where pipeline outputs are saved.",
            "exemplos": {"resultados": "Default folder", r"C:\experimento_01": "Specifically named experiment"},
        },
        "default": "resultados", "range": "Valid system path",
    },
    "modo_entrada": {
        "PT": {
            "desc": "Origem dos dados de entrada: dx (espectros JCAMP-DX) | csv | sintetico (testes).",
            "impacto": "ANALITICO — define o formato de leitura e parsing dos dados.",
            "exemplos": {"dx": "Espectros FT-NIR (padrao GEAAp)", "csv": "Tabela generica com colunas espectrais", "sintetico": "Dados simulados para teste do pipeline"},
        },
        "EN": {
            "desc": "Input data source: dx (JCAMP-DX spectra) | csv | synthetic (for testing).",
            "impacto": "ANALYTICAL — defines the data reading and parsing format.",
            "exemplos": {"dx": "FT-NIR spectra (GEAAp standard)", "csv": "Generic table with spectral columns", "sintetico": "Simulated data for pipeline testing"},
        },
        "default": "dx", "range": "dx | csv | sintetico",
    },
    "faixa_min_cm": {
        "PT": {
            "desc": "Limite inferior da faixa espectral analisada (numero de onda minimo, cm-1).",
            "impacto": "ANALITICO — regioes abaixo deste valor sao descartadas.",
            "exemplos": {"4000": "Padrao NIR (recomendado)", "5500": "Regiao de combinacoes apenas", "6000": "Sobretons superiores"},
        },
        "EN": {
            "desc": "Lower bound of the analyzed spectral range (minimum wavenumber, cm-1).",
            "impacto": "ANALYTICAL — regions below this value are discarded.",
            "exemplos": {"4000": "Standard NIR (recommended)", "5500": "Combination region only", "6000": "Upper overtones"},
        },
        "default": 4000.0, "range": "400-12000",
    },
    "faixa_max_cm": {
        "PT": {
            "desc": "Limite superior da faixa espectral analisada (numero de onda maximo, cm-1).",
            "impacto": "ANALITICO — regioes acima deste valor sao descartadas.",
            "exemplos": {"10000": "Padrao NIR (recomendado)", "7500": "NIR curto", "12000": "NIR completo"},
        },
        "EN": {
            "desc": "Upper bound of the analyzed spectral range (maximum wavenumber, cm-1).",
            "impacto": "ANALYTICAL — regions above this value are discarded.",
            "exemplos": {"10000": "Standard NIR (recommended)", "7500": "Short NIR", "12000": "Full NIR"},
        },
        "default": 10000.0, "range": "400-12000",
    },
    "excluir_classes": {
        "PT": {
            "desc": "Lista de especies/classes a remover da analise (por nome exato).",
            "impacto": "ANALITICO — altera o conjunto de treinamento e o numero de classes.",
            "exemplos": {"[]": "Usar todas as classes disponiveis", "[Copaiba]": "Remover Copaiba (outlier composicional)", "[Copaiba, Palmiste]": "Remover multiplas classes"},
        },
        "EN": {
            "desc": "List of species/classes to remove from analysis (by exact name).",
            "impacto": "ANALYTICAL — changes the training set and number of classes.",
            "exemplos": {"[]": "Use all available classes", "[Copaiba]": "Remove Copaiba (compositional outlier)", "[Copaiba, Palmiste]": "Remove multiple classes"},
        },
        "default": [], "range": "List of class names",
    },
    "teste_wold": {
        "PT": {
            "desc": "Executa o criterio de parcimonia de Wold para selecao automatica do numero otimo de LVs.",
            "impacto": "ANALITICO — define o numero de LVs usando tolerancia de 2% no RMSECV.",
            "exemplos": {"true": "Recomendado (evita overfitting)", "false": "Usar max_lvs diretamente"},
        },
        "EN": {
            "desc": "Runs Wold's parsimony criterion for automatic optimal number of LVs selection.",
            "impacto": "ANALYTICAL — defines LV count using 2% RMSECV tolerance.",
            "exemplos": {"true": "Recommended (prevents overfitting)", "false": "Use max_lvs directly"},
        },
        "default": True, "range": "true | false",
    },
    "teste_cv_anova": {
        "PT": {
            "desc": "Executa CV-ANOVA (Eriksson et al. 2008) para testar a significancia estatistica do modelo.",
            "impacto": "ANALITICO — gera F-statistic e p-value de significancia.",
            "exemplos": {"true": "Publicacao (obrigatorio)", "false": "Exploracao rapida"},
        },
        "EN": {
            "desc": "Runs CV-ANOVA (Eriksson et al. 2008) to test statistical significance of the model.",
            "impacto": "ANALYTICAL — generates F-statistic and significance p-value.",
            "exemplos": {"true": "Publication (mandatory)", "false": "Quick exploration"},
        },
        "default": True, "range": "true | false",
    },
    "figuras_mostrar_marcadores": {
        "PT": {
            "desc": "Usar formas diferentes (circulo, triangulo, quadrado) por classe nos graficos de score.",
            "impacto": "VISUAL — nao altera resultados analiticos.",
            "exemplos": {"true": "Graficos mais informativos e acessiveis", "false": "Graficos mais limpos e simples"},
        },
        "EN": {
            "desc": "Use different shapes (circle, triangle, square) per class in score plots.",
            "impacto": "VISUAL — does not affect analytical results.",
            "exemplos": {"true": "More informative and accessible plots", "false": "Cleaner, simpler plots"},
        },
        "default": True, "range": "true | false",
    },
    "figuras_mostrar_elipses": {
        "PT": {
            "desc": "Desenhar elipses de confianca de Hotelling T2 por grupo nos graficos de score.",
            "impacto": "VISUAL — nao altera resultados analiticos.",
            "exemplos": {"true": "Padrao para publicacao cientifica", "false": "Graficos mais simples"},
        },
        "EN": {
            "desc": "Draw Hotelling T2 confidence ellipses per group in score plots.",
            "impacto": "VISUAL — does not affect analytical results.",
            "exemplos": {"true": "Standard for scientific publication", "false": "Simpler plots"},
        },
        "default": True, "range": "true | false",
    },
    "abrir_figuras_na_tela": {
        "PT": {
            "desc": "Abrir cada figura na tela ao gerar (alem de salvar automaticamente em arquivo).",
            "impacto": "VISUAL — nao altera analise.",
            "exemplos": {"false": "Execucao automatizada sem interrupcoes", "true": "Revisao interativa das figuras"},
        },
        "EN": {
            "desc": "Open each figure on screen when generated (in addition to saving to file).",
            "impacto": "VISUAL — does not affect analysis.",
            "exemplos": {"false": "Automated execution without interruptions", "true": "Interactive figure review"},
        },
        "default": False, "range": "true | false",
    },
    "arquivo_csv": {
        "PT": {
            "desc": "Caminho do arquivo CSV (no modo csv): deve conter colunas espectrais + 1 coluna de classe.",
            "impacto": "ANALITICO — define os dados de entrada no modo CSV.",
            "exemplos": {"dados.csv": "Arquivo na pasta atual", r"C:\dados\amostras.csv": "Caminho absoluto"},
        },
        "EN": {
            "desc": "CSV file path (csv mode): must contain spectral columns + 1 class column.",
            "impacto": "ANALYTICAL — defines input data in CSV mode.",
            "exemplos": {"dados.csv": "File in current folder", r"C:\dados\amostras.csv": "Absolute path"},
        },
        "default": "", "range": "Valid .csv path",
    },
    "coluna_classe": {
        "PT": {
            "desc": "Nome exato da coluna de classe/rotulo no arquivo CSV.",
            "impacto": "ANALITICO — define o alvo de classificacao.",
            "exemplos": {"classe": "Nome padrao", "especie": "Nome alternativo", "label": "Nome em ingles"},
        },
        "EN": {
            "desc": "Exact name of the class/label column in the CSV file.",
            "impacto": "ANALYTICAL — defines the classification target.",
            "exemplos": {"classe": "Default name (Portuguese)", "especie": "Alternative name", "label": "English name"},
        },
        "default": "classe", "range": "Existing column name in CSV",
    },
    "coluna_concentracao": {
        "PT": {
            "desc": "Nome da coluna de concentracao no CSV (deixe vazio se nao houver).",
            "impacto": "ANALITICO — usado no nivel N3 (regressao de adulterantes).",
            "exemplos": {"": "Sem concentracao — use para N1/N2", "conc": "Com concentracao — use para N3"},
        },
        "EN": {
            "desc": "Name of the concentration column in CSV (leave empty if not present).",
            "impacto": "ANALYTICAL — used at level N3 (adulterant regression).",
            "exemplos": {"": "No concentration — use for N1/N2", "conc": "With concentration — use for N3"},
        },
        "default": "", "range": "Column name or empty",
    },
    "monte_carlo_incluir_todos": {
        "PT": {
            "desc": "MC CV: incluir SVM RBF / RF / XGBoost alem do PLS-DA (muito mais lento).",
            "impacto": "AVANCADO — aumenta consideravelmente o tempo do Monte Carlo CV.",
            "exemplos": {"false": "Apenas PLS-DA (recomendado)", "true": "Comparacao completa de classificadores"},
        },
        "EN": {
            "desc": "MC CV: include SVM RBF / RF / XGBoost in addition to PLS-DA (much slower).",
            "impacto": "ADVANCED — considerably increases Monte Carlo CV time.",
            "exemplos": {"false": "PLS-DA only (recommended)", "true": "Full classifier comparison"},
        },
        "default": False, "range": "true | false",
    },
}

# ---------------------------------------------------------------------------
# Perfis prontos
# ---------------------------------------------------------------------------
PROFILES: Dict[str, Dict[str, Any]] = {
    "Exploracao Rapida": {
        "max_lvs": 20, "n_permutacoes": 50, "ddsimca": False,
        "opls_da": False, "benchmark": False, "monte_carlo": False,
        "shap_benchmark": False, "selecao_variaveis_etapa4": False,
        "comparar_pre_processamentos": False, "dpi": 150,
    },
    "TCC": {
        "max_lvs": 40, "n_permutacoes": 200, "ddsimca": True,
        "modo_ddsimca": "puros", "opls_da": True, "benchmark": False,
        "monte_carlo": False, "shap_benchmark": False,
        "selecao_variaveis_etapa4": True, "dpi": 300,
        "figuras_mostrar_elipses": True,
    },
    "Artigo Cientifico": {
        "max_lvs": 40, "n_permutacoes": 200, "ddsimca": True,
        "modo_ddsimca": "puros", "opls_da": True, "benchmark": True,
        "monte_carlo": False, "shap_benchmark": True,
        "selecao_variaveis_etapa4": True, "dpi": 600,
        "figuras_mostrar_elipses": True,
    },
    "Dissertacao / Tese": {
        "max_lvs": 40, "n_permutacoes": 500, "ddsimca": True,
        "modo_ddsimca": "puros", "opls_da": True, "benchmark": True,
        "monte_carlo": True, "n_monte_carlo": 200,
        "shap_benchmark": True, "selecao_variaveis_etapa4": True,
        "dpi": 600, "formato_figura": "pdf",
    },
}

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Descricoes dos perfis prontos (bilíngue)
# ---------------------------------------------------------------------------
PROFILE_DESC: Dict[str, Dict[str, str]] = {
    "Exploracao Rapida": {
        "PT": ("Analise exploratoria rapida. Modulos pesados desativados.\n"
               "    Ideal para: primeiro contato com os dados, teste do pipeline.\n"
               "    Tempo estimado: ~2-5 min."),
        "EN": ("Quick exploratory analysis. Heavy modules disabled.\n"
               "    Ideal for: first contact with data, pipeline testing.\n"
               "    Estimated time: ~2-5 min."),
    },
    "TCC": {
        "PT": ("Configuracao balanceada para Trabalho de Conclusao de Curso.\n"
               "    PLS-DA + OPLS-DA + DD-SIMCA (autenticacao) + selecao de variaveis.\n"
               "    DPI 300. Sem benchmark (economiza tempo).\n"
               "    Tempo estimado: ~15-30 min."),
        "EN": ("Balanced config for undergraduate thesis (TCC).\n"
               "    PLS-DA + OPLS-DA + DD-SIMCA (authentication) + variable selection.\n"
               "    DPI 300. No benchmark (saves time).\n"
               "    Estimated time: ~15-30 min."),
    },
    "Artigo Cientifico": {
        "PT": ("Configuracao completa para publicacao em revista cientifica.\n"
               "    Benchmark (SVM/RF/XGBoost) + SHAP para interpretabilidade espectral.\n"
               "    200 permutacoes, DPI 600, elipses ativadas.\n"
               "    Tempo estimado: ~60-120 min."),
        "EN": ("Full config for scientific journal publication.\n"
               "    Benchmark (SVM/RF/XGBoost) + SHAP for spectral interpretability.\n"
               "    200 permutations, DPI 600, ellipses enabled.\n"
               "    Estimated time: ~60-120 min."),
    },
    "Dissertacao / Tese": {
        "PT": ("Configuracao premium para dissertacao ou tese.\n"
               "    Monte Carlo CV (200 rep.) + benchmark + SHAP + 500 permutacoes.\n"
               "    Figuras em PDF (vetorial). Alta rigorosidade estatistica.\n"
               "    Tempo estimado: ~3-6 horas."),
        "EN": ("Premium config for MSc dissertation or PhD thesis.\n"
               "    Monte Carlo CV (200 rep.) + benchmark + SHAP + 500 permutations.\n"
               "    PDF figures (vector). High statistical rigor.\n"
               "    Estimated time: ~3-6 hours."),
    },
}

PROFILE_KEY_SUMMARY: Dict[str, Dict[str, str]] = {
    "Exploracao Rapida": {
        "PT": "LVs=20 | perm=50 | DD-SIMCA=OFF | OPLS=OFF | DPI=150",
        "EN": "LVs=20 | perm=50 | DD-SIMCA=OFF | OPLS=OFF | DPI=150",
    },
    "TCC": {
        "PT": "LVs=40 | perm=200 | DD-SIMCA=autenticacao | OPLS=ON | DPI=300",
        "EN": "LVs=40 | perm=200 | DD-SIMCA=authentication | OPLS=ON | DPI=300",
    },
    "Artigo Cientifico": {
        "PT": "LVs=40 | perm=200 | Benchmark=ON | SHAP=ON | DPI=600",
        "EN": "LVs=40 | perm=200 | Benchmark=ON | SHAP=ON | DPI=600",
    },
    "Dissertacao / Tese": {
        "PT": "perm=500 | MonteCarlo=200rep | Benchmark=ON | SHAP=ON | PDF",
        "EN": "perm=500 | MonteCarlo=200rep | Benchmark=ON | SHAP=ON | PDF",
    },
}

# Mapeamento de menus -> campos (chaves do _CONFIG_SPEC)
# ---------------------------------------------------------------------------
MENU_FIELDS: Dict[str, list] = {
    "projeto": ["pasta_dados", "pasta_saida"],
    "dados": ["modo_entrada", "arquivo_csv", "coluna_classe", "coluna_concentracao",
              "faixa_min_cm", "faixa_max_cm", "excluir_classes"],
    "preproc": ["pre_processamento", "comparar_pre_processamentos"],
    "modelo": ["nivel", "max_lvs", "opls_da", "ddsimca", "modo_ddsimca",
               "selecao_variaveis_etapa4"],
    "validacao": ["holdout_fracao", "validacao_group_aware", "n_permutacoes",
                  "teste_wold", "teste_cv_anova"],
    "avancado": ["benchmark", "monte_carlo", "n_monte_carlo",
                 "monte_carlo_incluir_todos", "shap_benchmark", "shap_max_amostras"],
    "visualizacao": ["figuras_mostrar_marcadores", "figuras_mostrar_elipses",
                     "formato_figura", "dpi", "abrir_figuras_na_tela"],
}

# Indice inverso: chave -> especificacao do _CONFIG_SPEC
_SPEC_BY_KEY: Dict[str, Dict[str, Any]] = {s["key"]: s for s in _CONFIG_SPEC}


# ===========================================================================
# Utilitarios de terminal
# ===========================================================================

def cls() -> None:
    """Limpa a tela de forma cross-platform."""
    os.system("cls" if os.name == "nt" else "clear")


def _c(color_key: str, text: str) -> str:
    """Envolve texto com codigo ANSI de cor."""
    return f"{RISK_COLOR.get(color_key, '')}{text}{RISK_COLOR['RESET']}"


def _risk_label(key: str) -> str:
    """Retorna a label colorida de risco para um campo."""
    risk = RISK_CLASS.get(key, "ANALITICO")
    return _c(risk, f"[{risk}]")


def _get_val(cfg: Config, key: str) -> Any:
    """Le o valor atual do campo na Config, com alias de exibicao para modo_ddsimca."""
    spec = _SPEC_BY_KEY.get(key)
    if spec is None:
        return "?"
    raw = _attr_para_yaml(spec, cfg)
    if key == "modo_ddsimca":
        lang = _lang()
        return _DDSIMCA_DISPLAY.get(lang, {}).get(str(raw), raw)
    return raw


def _set_val(cfg: Config, key: str, raw: str) -> None:
    """Converte raw string e seta no atributo da Config, com alias de entrada para modo_ddsimca."""
    spec = _SPEC_BY_KEY[key]
    if key == "modo_ddsimca":
        interno = _DDSIMCA_INPUT.get(raw.lower().strip())
        if interno is None:
            raise ValueError(
                f"Valor invalido para modo_ddsimca: '{raw}'. "
                "Use: autenticacao | exploratorio (ou authentication | exploratory)"
            )
        raw = interno
    valor = _coagir_valor(spec, raw)
    setattr(cfg, spec["attr"], valor)


def _status_dados(cfg: Config) -> str:
    """Retorna string de status da pasta de dados."""
    lang = _lang()
    t = I18N[lang]
    pasta = getattr(cfg, "pasta_entrada", "dados")
    if pasta and os.path.isdir(str(pasta)):
        return _c("VISUAL", t["status_ok"]) + f" ({pasta})"
    return _c("AVANCADO", t["status_erro"]) + f" ({pasta})"


def _largura() -> int:
    try:
        return min(os.get_terminal_size().columns, 62)
    except OSError:
        return 62


def _wrap_line(text: str, width: int, indent: int = 4) -> list:
    """Quebra texto em linhas de no maximo `width` caracteres com indentacao."""
    words = text.split()
    lines = []
    current = " " * indent
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current += ("" if current.strip() == "" else " ") + word
        else:
            lines.append(current)
            current = " " * indent + word
    if current.strip():
        lines.append(current)
    return lines or [" " * indent]


# ===========================================================================
# Cabecalho
# ===========================================================================

def print_header(cfg: Config) -> None:
    """Imprime o cabecalho AmaNIR com titulo, idioma atual e status dos dados."""
    lang = _lang()
    w = 60
    linha1 = "AmaNIR — Plataforma Quimiometrica FT-NIR"
    linha2 = "GEAAp / UFPA  |  Oleos Vegetais Amazonicos"
    if lang == "EN":
        linha1 = "AmaNIR — FT-NIR Chemometrics Platform"
        linha2 = "GEAAp / UFPA  |  Amazonian Vegetable Oils"
    status = _status_dados(cfg)
    idioma_str = f"[{lang}]"
    print("\n" + "╔" + "═" * (w - 2) + "╗")
    print("║" + linha1.center(w - 2) + "║")
    print("║" + linha2.center(w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")
    status_raw = f"  {status}"
    # Linha com status e idioma, considerando escapes ANSI
    print("║" + status_raw.ljust(w - 3 + 14) + idioma_str + " ║")
    print("╚" + "═" * (w - 2) + "╝")


# ===========================================================================
# Menu principal
# ===========================================================================

def print_main_menu() -> None:
    """Imprime o menu principal hierarquico com borda ASCII."""
    lang = _lang()
    t = I18N[lang]
    w = 60
    print("╔" + "═" * (w - 2) + "╗")
    print("║" + "  MENU PRINCIPAL".ljust(w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")
    linha1 = f"  [1] {t['menu_projeto']:<18s}  [5] {t['menu_valid']}"
    linha2 = f"  [2] {t['menu_dados']:<18s}  [6] {t['menu_avancado']}"
    linha3 = f"  [3] {t['menu_preproc']:<18s}  [7] {t['menu_viz']}"
    linha4 = f"  [4] {t['menu_modelo']:<18s}  [8] {t['menu_ajuda']}"
    for l in [linha1, linha2, linha3, linha4]:
        print("║" + l.ljust(w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")
    linha_p = f"  [P] {t['perfis']}"
    linha_i = f"  [I] {t['idioma']}"
    for l in [linha_p, linha_i]:
        print("║" + l.ljust(w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")
    barra = (f"  [S] {t['salvar']}   [L] {t['carregar']}   "
             f"[R] {t['rodar']}   [Q] {t['sair']}")
    print("║" + barra.ljust(w - 2) + "║")
    print("╚" + "═" * (w - 2) + "╝")


# ===========================================================================
# Painel de ajuda expandido
# ===========================================================================

def _mostrar_painel_ajuda(key: str) -> None:
    """Exibe painel de ajuda completo com borda ASCII usando toda a largura do terminal."""
    lang = _lang()
    t = I18N[lang]
    h = HELP_DB.get(key)
    if h is None:
        print(f"  Sem ajuda disponivel para '{key}'.")
        return

    h_lang = h.get(lang, h.get("PT", {}))
    desc = h_lang.get("desc", "—")
    impacto = h_lang.get("impacto", "—")
    exemplos = h_lang.get("exemplos", {})
    default_val = h.get("default", "—")
    range_val = h.get("range", "—")
    risk = RISK_CLASS.get(key, "ANALITICO")

    w = _largura()
    inner = w - 4  # espaco interior da caixa (sem bordas e espacos)

    risk_lbl = f"[{risk}]"
    risk_cor = RISK_COLOR.get(risk, "")
    rst = RISK_COLOR["RESET"]

    titulo_linha = f"  {key}"
    risco_alinhado = f"{risk_cor}{risk_lbl}{rst}"
    # Linha titulo
    print("  " + "┌" + "─" * (w - 4) + "┐")
    # Titulo com label de risco a direita
    espaco = w - 4 - len(key) - 2 - len(risk_lbl) - 2
    if espaco < 1:
        espaco = 1
    linha_titulo = f"  {key}" + " " * espaco + risk_cor + risk_lbl + rst + "  "
    print("  │" + linha_titulo + "│")
    print("  ├" + "─" * (w - 4) + "┤")

    # Descricao
    desc_label = f"  {t['descricao']}:"
    print("  │" + desc_label.ljust(w - 4) + "│")
    for dline in _wrap_line(desc, w - 6, indent=4):
        print("  │" + dline.ljust(w - 4) + "│")

    print("  │" + " " * (w - 4) + "│")

    # Padrao e faixa
    pad_fai = f"    {t['padrao']}: {default_val}   |   {t['faixa']}: {range_val}"
    print("  │" + pad_fai.ljust(w - 4) + "│")

    print("  │" + " " * (w - 4) + "│")

    # Impacto
    imp_label = f"  {t['impacto']}:"
    print("  │" + imp_label.ljust(w - 4) + "│")
    for iline in _wrap_line(impacto, w - 6, indent=4):
        print("  │" + iline.ljust(w - 4) + "│")

    if exemplos:
        print("  │" + " " * (w - 4) + "│")
        ex_label = f"  {t['exemplos']}:"
        print("  │" + ex_label.ljust(w - 4) + "│")
        for val_ex, desc_ex in exemplos.items():
            ex_linha = f"    {val_ex:>12}  ->  {desc_ex}"
            # Truncar se necessario
            if len(ex_linha) > w - 4:
                ex_linha = ex_linha[: w - 7] + "..."
            print("  │" + ex_linha.ljust(w - 4) + "│")

    print("  " + "└" + "─" * (w - 4) + "┘")


def _mostrar_help_campo(key: str) -> None:
    """Exibe o HELP_DB completo para um campo (interface legada compativel)."""
    _mostrar_painel_ajuda(key)
    print()


# ===========================================================================
# Edicao generica de campo (com confirmacao de risco)
# ===========================================================================

def _editar_campo_cli(cfg: Config, key: str) -> bool:
    """
    Mostra painel de ajuda + valor atual, pede novo valor, aplica confirmacao de risco.
    Retorna True se o campo foi atualizado, False se cancelado.
    """
    lang = _lang()
    t = I18N[lang]
    spec = _SPEC_BY_KEY.get(key)
    if spec is None:
        print(f"  Campo '{key}' nao encontrado.")
        return False

    atual = _get_val(cfg, key)
    risk = RISK_CLASS.get(key, "ANALITICO")

    # Mostrar painel de ajuda expandido antes do prompt
    print()
    _mostrar_painel_ajuda(key)

    if spec.get("opcoes"):
        # Para modo_ddsimca, mostrar aliases amigaveis
        if key == "modo_ddsimca":
            disp = _DDSIMCA_DISPLAY.get(lang, {})
            opcoes_str = " | ".join(disp.get(o, o) for o in spec["opcoes"])
        else:
            opcoes_str = " | ".join(str(o) for o in spec["opcoes"])
        print(f"  Opcoes: {opcoes_str}")

    print(f"  {t['atual']}: {_fmt_yaml(atual)}")

    try:
        novo_raw = input(f"  {t['novo_valor']}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    if novo_raw == "":
        print(f"  {t['mantido']}")
        return False

    # Ajuda inline
    if novo_raw.lower() in ("?", "help"):
        _mostrar_painel_ajuda(key)
        return False

    # Confirmacao para ANALITICO / AVANCADO
    if risk == "AVANCADO":
        print(_c("AVANCADO", f"\n  {t['aviso_avancado']}"))
    if risk in ("ANALITICO", "AVANCADO"):
        try:
            conf = input(f"  {t['aviso_analitico']}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if conf != t["confirmar_sn"]:
            print(f"  {t['cancelado']}")
            return False

    try:
        _set_val(cfg, key, novo_raw)
        novo_val = _get_val(cfg, key)
        msg = t["campo_atualizado"].format(campo=key, valor=_fmt_yaml(novo_val))
        print(f"  {_c('VISUAL', msg)}")
        return True
    except (ValueError, TypeError) as e:
        print(f"  Erro: {e}")
        return False


def _submenu_campos(cfg: Config, titulo: str, campos: list, secao_key: str = "") -> None:
    """Loop generico para exibir e editar um grupo de campos.
    Inclui [I] Idioma na barra de rodape de cada submenu.
    """
    while True:
        lang = _lang()
        t = I18N[lang]
        cls()
        print_header(cfg)
        w = 60
        print("╔" + "═" * (w - 2) + "╗")
        print("║" + f"  {titulo}".ljust(w - 2) + "║")
        # Exibir descricao da secao, se disponivel
        if secao_key and secao_key in SECTION_DESC:
            desc_sec = SECTION_DESC[secao_key].get(lang, "")
            if desc_sec:
                # Quebrar descricao em linhas se necessario
                for dline in _wrap_line(desc_sec, w - 6, indent=2):
                    print("║" + _c("DIM", dline).ljust(w + 7) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        for i, key in enumerate(campos, 1):
            val_raw = _fmt_yaml(_get_val(cfg, key))
            # Truncar valor longo para caber na caixa (max 22 chars)
            val_str = (val_raw[:20] + "…") if len(str(val_raw)) > 22 else val_raw
            nome = _nome_campo(key)
            risk = RISK_CLASS.get(key, "ANALITICO")
            cor = RISK_COLOR.get(risk, "")
            rst = RISK_COLOR["RESET"]
            lbl = f"{cor}●{rst}"
            linha = f"  [{i:2d}] {lbl} {nome:<28s}: {val_str}"
            print("║" + linha.ljust(w + 10) + "║")  # +10 para escapes ANSI
        print("╠" + "═" * (w - 2) + "╣")
        rodape = f"  [?] {t['ajuda_campo']}   [I] {t['idioma']}   [0] {t['voltar']}"
        print("║" + rodape.ljust(w - 2) + "║")
        print("╚" + "═" * (w - 2) + "╝")
        print()
        try:
            escolha = input(f"  {t['opcao']}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if escolha == "0" or escolha.lower() == "q":
            break

        # Troca de idioma dentro do submenu
        if escolha.upper() == "I":
            _toggle_idioma()
            continue  # redesenha com novo idioma

        # Ajuda por nome de campo
        if escolha.lower().startswith("?") or escolha.lower().startswith("help"):
            partes = escolha.split(maxsplit=1)
            campo_help = partes[1] if len(partes) > 1 else ""
            if campo_help:
                _mostrar_help_campo(campo_help.strip())
            else:
                # Listar campos disponiveis
                print(f"\n  Campos neste menu: {', '.join(campos)}")
                print("  Ex: ? dpi")
            input(f"  [{t['continuar']}]")
            continue

        if escolha.isdigit() and 1 <= int(escolha) <= len(campos):
            key = campos[int(escolha) - 1]
            _editar_campo_cli(cfg, key)
            input(f"  [{t['continuar']}]")
        else:
            print(f"  {t['invalido']}")
            input(f"  [{t['continuar']}]")


# ===========================================================================
# Submenus tematicos
# ===========================================================================

def menu_projeto(cfg: Config) -> None:
    """Menu 1 — Projeto: pasta_dados e pasta_saida."""
    lang = _lang()
    _submenu_campos(cfg, I18N[lang]["menu_projeto"], MENU_FIELDS["projeto"], "projeto")


def menu_dados(cfg: Config) -> None:
    """Menu 2 — Dados: modo, CSV, colunas, faixa espectral, exclusoes."""
    lang = _lang()
    _submenu_campos(cfg, I18N[lang]["menu_dados"], MENU_FIELDS["dados"], "dados")


def menu_preproc(cfg: Config) -> None:
    """Menu 3 — Pre-processamento: pipeline + comparacao."""
    while True:
        lang = _lang()
        t = I18N[lang]
        cls()
        print_header(cfg)
        w = 60
        titulo = t["menu_preproc"]
        campos = MENU_FIELDS["preproc"]
        print("╔" + "═" * (w - 2) + "╗")
        print("║" + f"  {titulo}".ljust(w - 2) + "║")
        # Descricao da secao
        desc_sec = SECTION_DESC["preproc"].get(lang, "")
        if desc_sec:
            for dline in _wrap_line(desc_sec, w - 6, indent=2):
                print("║" + _c("DIM", dline).ljust(w + 7) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        # Mostrar preview do pipeline atual
        pp_atual = _fmt_yaml(_get_val(cfg, "pre_processamento"))
        preview = f"  Pipeline atual: {_c('ANALITICO', pp_atual)}"
        print("║" + preview.ljust(w + 10) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        for i, key in enumerate(campos, 1):
            val_str = _fmt_yaml(_get_val(cfg, key))
            risk = RISK_CLASS.get(key, "ANALITICO")
            cor = RISK_COLOR.get(risk, "")
            rst = RISK_COLOR["RESET"]
            lbl = f"{cor}●{rst}"
            nome_tr = _nome_campo(key)
            val_trunc = (val_str[:20] + "…") if len(str(val_str)) > 22 else val_str
            linha = f"  [{i:2d}] {lbl} {nome_tr:<28s}: {val_trunc}"
            print("║" + linha.ljust(w + 10) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        rodape = f"  [I] {t['idioma']}   [0] {t['voltar']}"
        print("║" + rodape.ljust(w - 2) + "║")
        print("╚" + "═" * (w - 2) + "╝")
        print()
        try:
            escolha = input(f"  {I18N[_lang()]["opcao"]}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if escolha == "0" or escolha.lower() == "q":
            break
        if escolha.upper() == "I":
            _toggle_idioma()
            continue
        if escolha.isdigit() and 1 <= int(escolha) <= len(campos):
            key = campos[int(escolha) - 1]
            _editar_campo_cli(cfg, key)
            input(f"  [{t['continuar']}]")
        else:
            print(f"  {t['invalido']}")
            input(f"  [{t['continuar']}]")


def menu_modelagem(cfg: Config) -> None:
    """Menu 4 — Modelagem."""
    lang = _lang()
    _submenu_campos(cfg, I18N[lang]["menu_modelo"], MENU_FIELDS["modelo"], "modelo")


def menu_validacao(cfg: Config) -> None:
    """Menu 5 — Validacao."""
    lang = _lang()
    _submenu_campos(cfg, I18N[lang]["menu_valid"], MENU_FIELDS["validacao"], "validacao")


def menu_avancado(cfg: Config) -> None:
    """Menu 6 — Metodos Avancados (benchmark, MC, SHAP)."""
    lang = _lang()
    _submenu_campos(cfg, I18N[lang]["menu_avancado"], MENU_FIELDS["avancado"], "avancado")


def menu_visualizacao(cfg: Config) -> None:
    """Menu 7 — Visualizacao (apenas campos VISUAL)."""
    lang = _lang()
    _submenu_campos(cfg, I18N[lang]["menu_viz"], MENU_FIELDS["visualizacao"], "visualizacao")


# ===========================================================================
# Menu de Ajuda
# ===========================================================================

def menu_ajuda() -> None:
    """Menu 8 — Sistema de ajuda. Suporta 'help <topico>' ou '?' para listar."""
    while True:
        lang = _lang()
        t = I18N[lang]
        cls()
        w = 60
        print("╔" + "═" * (w - 2) + "╗")
        print("║" + f"  {t['menu_ajuda']}".ljust(w - 2) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        if lang == "PT":
            print("║" + "  Digite o nome do parametro para ver detalhes.".ljust(w - 2) + "║")
            print("║" + "  Exemplos: dpi  |  benchmark  |  pre_processamento".ljust(w - 2) + "║")
        else:
            print("║" + "  Type the parameter name to see details.".ljust(w - 2) + "║")
            print("║" + "  Examples: dpi  |  benchmark  |  pre_processamento".ljust(w - 2) + "║")
        rodape = f"  [L] {t['listar_todos']}   [I] {t['idioma']}   [0] {t['voltar']}"
        print("║" + rodape.ljust(w - 2) + "║")
        print("╚" + "═" * (w - 2) + "╝")
        print()
        try:
            entrada = input("  ? ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if entrada == "0" or entrada.lower() == "q":
            break
        if entrada.upper() == "I":
            _toggle_idioma()
            continue
        if entrada.lower() == "l":
            cls()
            keys_sorted = sorted(HELP_DB.keys())
            hdr = (f"  Parametros disponíveis ({len(keys_sorted)}) — digite o número ou nome:"
                   if lang == "PT" else
                   f"  Available parameters ({len(keys_sorted)}) — type number or name:")
            print(f"\n{hdr}\n")
            for idx_k, k in enumerate(keys_sorted, 1):
                risk = RISK_CLASS.get(k, "ANALITICO")
                cor = RISK_COLOR.get(risk, "")
                rst = RISK_COLOR["RESET"]
                h_lang = HELP_DB[k].get(lang, HELP_DB[k].get("PT", {}))
                desc_short = h_lang.get("desc", "")[:36] if isinstance(h_lang, dict) else ""
                nome_tr = FIELD_NAMES.get(k, {}).get(lang, k)
                print(f"  {idx_k:>2}. {cor}{nome_tr:<26s}{rst}  {desc_short}")
            tip = ("\n  Digite numero ou nome do parametro (Enter = voltar):"
                   if lang == "PT" else
                   "\n  Type number or parameter name (Enter = back):")
            print(tip)
            try:
                escolha_l = input(f"  {t['opcao']}: ").strip()
            except (EOFError, KeyboardInterrupt):
                escolha_l = ""
            if escolha_l:
                if escolha_l.isdigit() and 1 <= int(escolha_l) <= len(keys_sorted):
                    _mostrar_help_campo(keys_sorted[int(escolha_l) - 1])
                    input(f"  [{t['continuar']}]")
                else:
                    topico_l = escolha_l.replace("help", "").replace("?", "").strip()
                    if topico_l:
                        _mostrar_help_campo(topico_l)
                        input(f"  [{t['continuar']}]")
            continue
        # Permite "help benchmark" ou apenas "benchmark"
        topico = entrada.lower().replace("help", "").replace("?", "").strip()
        if topico:
            _mostrar_help_campo(topico)
        else:
            if lang == "PT":
                print("  Digite um nome de parametro. Ex: dpi")
            else:
                print("  Type a parameter name. Ex: dpi")
        input(f"  [{t['continuar']}]")


# ===========================================================================
# Menu de Perfis
# ===========================================================================

def menu_perfis(cfg: Config) -> None:
    """Lista PROFILES com descricoes e permite carregar ou criar novo perfil."""
    nomes = list(PROFILES.keys())
    while True:
        lang = _lang()
        t = I18N[lang]
        cls()
        w = 72
        titulo_perfil = "Perfis Prontos" if lang == "PT" else "Preset Profiles"
        instrucao = ("  Selecione um numero para aplicar o perfil e ver detalhes:"
                     if lang == "PT" else
                     "  Select a number to apply the profile and see details:")
        print("\u2554" + "\u2550" * (w - 2) + "\u2557")
        print("\u2551" + f"  {titulo_perfil}".ljust(w - 2) + "\u2551")
        print("\u2551" + instrucao.ljust(w - 2) + "\u2551")
        print("\u2560" + "\u2550" * (w - 2) + "\u2563")
        for i, nome in enumerate(nomes, 1):
            print("\u2551" + f"  [{i}] {nome}".ljust(w - 2) + "\u2551")
            summary = PROFILE_KEY_SUMMARY.get(nome, {}).get(lang, "")
            if summary:
                print("\u2551" + f"      {_c('DIM', summary)}".ljust(w + 7) + "\u2551")
            desc_lines = PROFILE_DESC.get(nome, {}).get(lang, "").split("\n")
            for dl in desc_lines:
                print("\u2551" + f"      {_c('DIM', dl.strip())}".ljust(w + 7) + "\u2551")
            print("\u2551" + "".ljust(w - 2) + "\u2551")
        perfis_usuario = _listar_perfis_salvos()
        if perfis_usuario:
            print("\u2560" + "\u2550" * (w - 2) + "\u2563")
            label_salvos = "  Perfis salvos pelo usuario:" if lang == "PT" else "  User saved profiles:"
            print("\u2551" + label_salvos.ljust(w - 2) + "\u2551")
            base = len(nomes)
            for j, nome_u in enumerate(perfis_usuario, base + 1):
                print("\u2551" + f"  [{j}] {nome_u}".ljust(w - 2) + "\u2551")
        print("\u2560" + "\u2550" * (w - 2) + "\u2563")
        novo_label = "[N] Criar novo perfil" if lang == "PT" else "[N] Create new profile"
        como = ("  Como criar: configure os menus 1-7 e pressione [S] no menu principal."
                if lang == "PT" else
                "  How to: configure menus 1-7, then press [S] in main menu.")
        print("\u2551" + f"  {novo_label}".ljust(w - 2) + "\u2551")
        print("\u2551" + f"  {_c('DIM', como.strip())}".ljust(w + 7) + "\u2551")
        print("\u2560" + "\u2550" * (w - 2) + "\u2563")
        rodape = f"  [I] {t['idioma']}   [0] {t['voltar']}"
        print("\u2551" + rodape.ljust(w - 2) + "\u2551")
        print("\u255a" + "\u2550" * (w - 2) + "\u255d")
        print()
        try:
            escolha = input(f"  {t['opcao']}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if escolha == "0" or escolha.lower() == "q":
            break
        if escolha.upper() == "I":
            _toggle_idioma()
            continue
        if escolha.upper() == "N":
            msg = ("  Configure os parametros nos menus 1-7 e pressione [S] no menu principal para salvar."
                   if lang == "PT" else
                   "  Configure parameters in menus 1-7 and press [S] in main menu to save.")
            print(f"\n{msg}")
            input(f"  [{t['continuar']}]")
            break
        if escolha.isdigit():
            n = int(escolha)
            if 1 <= n <= len(nomes):
                nome_perfil = nomes[n - 1]
                cls()
                print(f"\n  Aplicando perfil: {nome_perfil}" if lang == "PT" else f"\n  Applying profile: {nome_perfil}")
                desc = PROFILE_DESC.get(nome_perfil, {}).get(lang, "")
                for dl in desc.split("\n"):
                    print(f"  {dl.strip()}")
                _aplicar_perfil(cfg, PROFILES[nome_perfil])
                ok = "\n  Perfil aplicado! Os parametros foram atualizados." if lang == "PT" else "\n  Profile applied! Parameters updated."
                print(ok)
                input(f"  [{t['continuar']}]")
            elif perfis_usuario and len(nomes) < n <= len(nomes) + len(perfis_usuario):
                nome_u = perfis_usuario[n - len(nomes) - 1]
                carregar_perfil(nome_u, cfg)
                input(f"  [{t['continuar']}]")
            else:
                print(f"  {t['invalido']}")
                input(f"  [{t['continuar']}]")
        else:
            print(f"  {t['invalido']}")
            input(f"  [{t['continuar']}]")

def _aplicar_perfil(cfg: Config, perfil: Dict[str, Any]) -> None:
    """Aplica os valores de um perfil na Config."""
    for key, val in perfil.items():
        spec = _SPEC_BY_KEY.get(key)
        if spec is None:
            continue
        try:
            valor = _coagir_valor(spec, val)
            setattr(cfg, spec["attr"], valor)
        except (ValueError, TypeError):
            pass
    lang = _lang()
    msg = "  Perfil aplicado com sucesso." if lang == "PT" else "  Profile applied successfully."
    print(msg)


def _listar_perfis_salvos() -> list:
    """Lista nomes dos perfis JSON salvos na pasta perfis/."""
    if not _PERFIS_DIR.exists():
        return []
    return [p.stem for p in _PERFIS_DIR.glob("*.json")]


# ===========================================================================
# Salvar / Carregar perfil do usuario
# ===========================================================================

def salvar_perfil(cfg: Config) -> None:
    """Salva a Config atual como JSON em perfis/<nome>.json."""
    lang = _lang()
    _PERFIS_DIR.mkdir(parents=True, exist_ok=True)
    print()
    prompt = "  Nome do perfil (sem espacos): " if lang == "PT" else "  Profile name (no spaces): "
    try:
        nome = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not nome:
        msg = "  Nome vazio. Operacao cancelada." if lang == "PT" else "  Empty name. Operation cancelled."
        print(msg)
        return
    # Sanitizar nome
    nome_arquivo = "".join(c if c.isalnum() or c in "-_" else "_" for c in nome)
    dados: Dict[str, Any] = {}
    for s in _CONFIG_SPEC:
        dados[s["key"]] = _attr_para_yaml(s, cfg)
    caminho = _PERFIS_DIR / f"{nome_arquivo}.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    msg = f"  Perfil salvo em: {caminho}" if lang == "PT" else f"  Profile saved to: {caminho}"
    print(msg)


def carregar_perfil(nome: str, cfg: Config) -> None:
    """Carrega um perfil JSON de perfis/<nome>.json para a Config."""
    lang = _lang()
    caminho = _PERFIS_DIR / f"{nome}.json"
    if not caminho.exists():
        if lang == "PT":
            print(f"  Perfil '{nome}' nao encontrado em {_PERFIS_DIR}.")
        else:
            print(f"  Profile '{nome}' not found in {_PERFIS_DIR}.")
        return
    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)
    _aplicar_perfil(cfg, dados)
    msg = f"  Perfil '{nome}' carregado." if lang == "PT" else f"  Profile '{nome}' loaded."
    print(msg)


# ===========================================================================
# Salvar / Carregar config YAML (integrado ao pipeline)
# ===========================================================================

def _salvar_yaml(cfg: Config) -> None:
    """Salva config.yaml via salvar_config do pipeline."""
    lang = _lang()
    salvar_config(cfg, str(_CFG_PATH))
    msg = f"  Configuracao salva em: {_CFG_PATH}" if lang == "PT" else f"  Configuration saved to: {_CFG_PATH}"
    print(msg)
    t = I18N[lang]
    input(f"  [{t['continuar']}]")


def _carregar_yaml(cfg: Config) -> None:
    """Carrega config.yaml via carregar_config do pipeline e atualiza cfg."""
    lang = _lang()
    t = I18N[lang]
    if not _CFG_PATH.exists():
        if lang == "PT":
            print(f"  Arquivo {_CFG_PATH} nao encontrado. Salve primeiro.")
        else:
            print(f"  File {_CFG_PATH} not found. Save first.")
        input(f"  [{t['continuar']}]")
        return
    cfg_novo = carregar_config(str(_CFG_PATH))
    # Copiar atributos do Config carregado para o atual (em-place)
    for s in _CONFIG_SPEC:
        try:
            setattr(cfg, s["attr"], getattr(cfg_novo, s["attr"]))
        except AttributeError:
            pass
    msg = f"  Configuracao carregada de: {_CFG_PATH}" if lang == "PT" else f"  Configuration loaded from: {_CFG_PATH}"
    print(msg)
    input(f"  [{t['continuar']}]")


# ===========================================================================
# Wizard inicial
# ===========================================================================

def wizard_inicial() -> None:
    """
    Wizard de boas-vindas para primeira execucao (quando config.yaml nao existe).
    Define o idioma escolhido pelo usuario no estado global.
    """
    cls()
    w = 60
    print("\n" + "╔" + "═" * (w - 2) + "╗")
    print("║" + "  AmaNIR — Plataforma Quimiometrica FT-NIR".ljust(w - 2) + "║")
    print("║" + "  GEAAp / UFPA  |  Oleos Vegetais Amazonicos".ljust(w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")
    print("║" + "  Qual e o seu idioma? / What is your language?".ljust(w - 2) + "║")
    print("║" + "  [1] Portugues (PT)   [2] English (EN)".ljust(w - 2) + "║")
    print("╚" + "═" * (w - 2) + "╝")
    try:
        resp = input(f"  {I18N[_lang()]["opcao"]}: ").strip()
    except (EOFError, KeyboardInterrupt):
        resp = "1"

    if resp == "2":
        _set_lang("EN")
    else:
        _set_lang("PT")

    lang = _lang()
    cls()
    print("\n" + "╔" + "═" * (w - 2) + "╗")
    titulo_wiz = "  Bem-vindo!" if lang == "PT" else "  Welcome!"
    print("║" + titulo_wiz.ljust(w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")
    q_perfil = "  Qual e o seu perfil?" if lang == "PT" else "  What is your profile?"
    print("║" + q_perfil.ljust(w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")
    if lang == "PT":
        linhas_wiz = [
            "  [1] Iniciante    — configuracoes essenciais, pre-proc padrao",
            "  [2] Pesquisador  — opcoes intermediarias, validacao completa",
            "  [3] Especialista — acesso a todos os parametros",
        ]
    else:
        linhas_wiz = [
            "  [1] Beginner    — essential settings, default preprocessing",
            "  [2] Researcher  — intermediate options, full validation",
            "  [3] Expert      — access to all parameters",
        ]
    for l in linhas_wiz:
        print("║" + l.ljust(w - 2) + "║")
    print("╚" + "═" * (w - 2) + "╝")
    print()
    try:
        resp = input(f"  {I18N[_lang()]["opcao"]}: ").strip()
    except (EOFError, KeyboardInterrupt):
        resp = "1"

    # Marcar wizard como concluido (salva apenas o idioma)
    try:
        _WIZARD_FLAG.write_text(lang, encoding="utf-8")
    except OSError:
        pass


# ===========================================================================
# Rodar pipeline
# ===========================================================================

def _rodar_pipeline(cfg: Config) -> None:
    """Salva config.yaml e dispara executar(cfg)."""
    lang = _lang()
    t = I18N[lang]
    # Verificar pasta de dados
    pasta = getattr(cfg, "pasta_entrada", "")
    modo = getattr(cfg, "modo", "dx")
    if modo != "sintetico" and (not pasta or not os.path.isdir(str(pasta))):
        print(f"\n  {t['status_erro']}")
        if lang == "PT":
            print("  Corrija a pasta_dados antes de rodar.")
        else:
            print("  Fix pasta_dados before running.")
        input(f"  [{t['continuar']}]")
        return

    if lang == "PT":
        print(f"\n  Salvando configuracao em {_CFG_PATH}...")
        print("  Iniciando pipeline...\n")
    else:
        print(f"\n  Saving configuration to {_CFG_PATH}...")
        print("  Starting pipeline...\n")

    salvar_config(cfg, str(_CFG_PATH))
    try:
        executar(cfg)
    except KeyboardInterrupt:
        msg = "\n  Pipeline interrompido pelo usuario." if lang == "PT" else "\n  Pipeline interrupted by user."
        print(msg)
    except Exception as e:  # noqa: BLE001
        msg = f"\n  Erro no pipeline: {e}" if lang == "PT" else f"\n  Pipeline error: {e}"
        print(msg)
    input(f"\n  [{t['continuar']}]")


# ===========================================================================
# Loop principal
# ===========================================================================

def main() -> None:
    """Ponto de entrada do assistente CLI hierarquico."""
    # Carrega config existente ou cria padrao
    if _CFG_PATH.exists():
        try:
            cfg = carregar_config(str(_CFG_PATH))
        except Exception:  # noqa: BLE001
            cfg = Config()
    else:
        cfg = Config()

    # Tentar recuperar idioma salvo antes do wizard
    try:
        idioma_salvo = _WIZARD_FLAG.read_text(encoding="utf-8").strip()
        if idioma_salvo in ("EN", "2"):
            _set_lang("EN")
        elif idioma_salvo == "PT":
            _set_lang("PT")
    except OSError:
        pass

    # Wizard na primeira vez
    if not _WIZARD_FLAG.exists() and not _CFG_PATH.exists():
        wizard_inicial()

    # Loop principal
    while True:
        lang = _lang()
        t = I18N[lang]
        cls()
        print_header(cfg)
        print_main_menu()
        print()
        try:
            escolha = input(f"  {I18N[_lang()]["opcao"]}: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {t['sair']}.")
            break

        if escolha == "1":
            menu_projeto(cfg)
        elif escolha == "2":
            menu_dados(cfg)
        elif escolha == "3":
            menu_preproc(cfg)
        elif escolha == "4":
            menu_modelagem(cfg)
        elif escolha == "5":
            menu_validacao(cfg)
        elif escolha == "6":
            menu_avancado(cfg)
        elif escolha == "7":
            menu_visualizacao(cfg)
        elif escolha == "8":
            menu_ajuda()
        elif escolha == "P":
            menu_perfis(cfg)
        elif escolha == "I":
            _toggle_idioma()
        elif escolha == "S":
            _salvar_yaml(cfg)
        elif escolha == "L":
            _carregar_yaml(cfg)
        elif escolha == "R":
            _rodar_pipeline(cfg)
        elif escolha == "Q":
            print(f"\n  {t['sair']}.")
            break
        else:
            print(f"  {t['invalido']}")
            import time
            time.sleep(0.8)


if __name__ == "__main__":
    main()
