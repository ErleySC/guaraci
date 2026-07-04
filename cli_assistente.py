"""
cli_assistente.py — Assistente CLI hierarquico para o Pipeline Quimiometrico FT-NIR
AmaNIR — Plataforma Quimiometrica FT-NIR
GEAAp / UFPA — Plataforma de autenticacao de oleos vegetais amazonicos.

Uso:
    python cli_assistente.py
    python pipeline.py   (chama este modulo automaticamente)

Requer: pipeline.py no mesmo diretorio.
"""

from __future__ import annotations

import json
import os
import re as _re
import sys
import textwrap as _textwrap
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Integracao com o pipeline (sem modificar nenhuma funcao analitica)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as pq

# Atalhos
Config = pq.Config
_CONFIG_SPEC = pq._CONFIG_SPEC
executar = pq.executar
salvar_config = pq.salvar_config
carregar_config = pq.carregar_config
_attr_para_yaml = pq._attr_para_yaml
_fmt_yaml = pq._fmt_yaml
_coagir_valor = pq._coagir_valor

# Tema visual compartilhado com guaraci.py (mesma paleta e Console Rich).
from guaraci_theme import (
    console as _console, ansi as _ansi_tom, _W as _theme_W,
    ANSI_RESET as _RESET, ANSI_BOLD as _BOLD, ANSI_DIM as _DIM,
    err as _err,
)
from rich.panel import Panel
from rich.text import Text
from rich.console import Group
from rich import box as _rbox

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
_BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_CFG_PATH = _BASE_DIR / "config.yaml"
_PERFIS_DIR = _BASE_DIR / "perfis"
_WIZARD_FLAG = _BASE_DIR / ".cli_wizard_done"
_CODIGOS_PATH = _BASE_DIR / "codigos_usuario.json"

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
        "titulo": "GUARACI — Inteligencia Quimiometrica para Matrizes Amazonicas",
        "subtitulo": "GEAAp / UFPA  |  Quimiometria multitecnica",
        "menu_projeto": "Projeto",
        "menu_dados": "Dados",
        "menu_preproc": "Pre-processamento",
        "menu_modelo": "Modelagem",
        "menu_valid": "Validacao",
        "menu_avancado": "Metodos Avancados",
        "menu_viz": "Visualizacao",
        "menu_ajuda": "Ajuda",
        "menu_tecnica": "Tecnica Analitica",
        "grp_analise": "Configuracao da Analise",
        "grp_sistema": "Sistema e Visualizacao",
        "grp_perfis": "Perfis e Idioma",
        "ajuda_curta": "Ajuda",
        "rodar_pipeline": "Rodar Pipeline",
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
        "hardware": "Verificar Hardware",
        "menu_codificacao": "Codificacao de Arquivos",
        "nome_saida": "Nome saida",
    },
    "EN": {
        "titulo": "GUARACI — Chemometric Intelligence for Amazonian Matrices",
        "subtitulo": "GEAAp / UFPA  |  Multi-technique chemometrics",
        "menu_projeto": "Project",
        "menu_dados": "Data",
        "menu_preproc": "Preprocessing",
        "menu_modelo": "Modelling",
        "menu_valid": "Validation",
        "menu_avancado": "Advanced Methods",
        "menu_viz": "Visualization",
        "menu_ajuda": "Help",
        "menu_tecnica": "Analytical Technique",
        "grp_analise": "Analysis Configuration",
        "grp_sistema": "System and Visualization",
        "grp_perfis": "Profiles and Language",
        "ajuda_curta": "Help",
        "rodar_pipeline": "Run Pipeline",
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
        "hardware": "Check Hardware",
        "menu_codificacao": "File Encoding",
        "nome_saida": "Output name",
    },
}

# ---------------------------------------------------------------------------
# Classificacao de risco
# ---------------------------------------------------------------------------
RISK_CLASS: Dict[str, str] = {
    # VISUAL
    "dpi": "VISUAL", "formato_figura": "VISUAL", "figuras_detalhadas": "VISUAL",
    "figuras_mostrar_marcadores": "VISUAL", "figuras_mostrar_elipses": "VISUAL",
    "abrir_figuras_na_tela": "VISUAL", "tag": "VISUAL", "nome_execucao": "VISUAL",
    # ANALITICO
    "pre_processamento": "ANALITICO", "max_lvs": "ANALITICO",
    "n_permutacoes": "ANALITICO", "n_jobs_permutacao": "ANALITICO",
    "holdout_fracao": "ANALITICO",
    "nivel": "ANALITICO", "excluir_classes": "ANALITICO",
    "faixa_min_cm": "ANALITICO", "faixa_max_cm": "ANALITICO",
    "modo_ddsimca": "ANALITICO", "ddsimca": "ANALITICO",
    "opls_da": "ANALITICO", "selecao_variaveis_etapa4": "ANALITICO",
    "selecao_spa": "ANALITICO", "selecao_ag": "ANALITICO",
    "comparar_pre_processamentos": "ANALITICO",
    "validacao_group_aware": "ANALITICO", "teste_wold": "ANALITICO",
    "teste_cv_anova": "ANALITICO", "teste_martens": "ANALITICO",
    "pasta_dados": "ANALITICO",
    "pasta_saida": "ANALITICO", "modo_entrada": "ANALITICO",
    "arquivo_csv": "ANALITICO", "coluna_classe": "ANALITICO",
    "coluna_concentracao": "ANALITICO", "imagem_incluir_textura": "ANALITICO",
    # AVANCADO
    "benchmark": "AVANCADO", "monte_carlo": "AVANCADO",
    "shap_benchmark": "AVANCADO", "n_monte_carlo": "AVANCADO",
    "shap_max_amostras": "AVANCADO", "monte_carlo_incluir_todos": "AVANCADO",
    "benchmark_regressao": "AVANCADO",
}

# Cores agora derivadas da paleta compartilhada (guaraci_theme): mesmos tons
# discretos do guaraci.py, no lugar dos ANSI berrantes (92/93/91/96) que
# destoavam. Um unico ponto recolore todas as ~40 chamadas de _c/RISK_COLOR.
RISK_COLOR: Dict[str, str] = {
    "VISUAL": _ansi_tom("PG"),      # verde discreto (sucesso/leve)
    "ANALITICO": _ansi_tom("PA"),   # ambar (atencao moderada)
    "AVANCADO": _ansi_tom("PR"),    # rust (custo/avancado)
    "RESET": _RESET,
    "BOLD": _BOLD,
    "DIM": _DIM,
    "CYAN": _ansi_tom("PS"),        # sage — substitui o ciano berrante
}

# ---------------------------------------------------------------------------
# Utilitario ANSI: ljust que ignora caracteres invisiveis no calculo da largura
# ---------------------------------------------------------------------------
_ANSI_RE = _re.compile(r'\x1b\[[0-9;]*[mGKHF]')


def _ansi_len(s: str) -> int:
    """Comprimento visivel da string (ignora codigos ANSI)."""
    return len(_ANSI_RE.sub('', s))


def _ansi_ljust(s: str, width: int, fillchar: str = ' ') -> str:
    """ljust que calcula o padding pelo comprimento VISIVEL."""
    pad = max(0, width - _ansi_len(s))
    return s + fillchar * pad


# ---------------------------------------------------------------------------
# Nomes traduzidos dos campos (chave tecnica → nome legível por idioma)
# ---------------------------------------------------------------------------
FIELD_NAMES: Dict[str, Dict[str, str]] = {
    "pasta_dados":                  {"PT": "Pasta de entrada",        "EN": "Input folder"},
    "pasta_saida":                  {"PT": "Pasta de saida",          "EN": "Output folder"},
    "tag":                          {"PT": "Sufixo da pasta saida",   "EN": "Output folder tag"},
    "modo_entrada":                 {"PT": "Modo de entrada",         "EN": "Input mode"},
    "arquivo_csv":                  {"PT": "Arquivo CSV",             "EN": "CSV file"},
    "coluna_classe":                {"PT": "Coluna de classe",        "EN": "Class column"},
    "coluna_concentracao":          {"PT": "Coluna concentracao",     "EN": "Concentration column"},
    "imagem_incluir_textura":       {"PT": "Textura (imagem)",        "EN": "Texture (image)"},
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
    "selecao_spa":                  {"PT": "SPA (APS)",               "EN": "SPA (successive proj.)"},
    "selecao_ag":                   {"PT": "AG (Genetico)",           "EN": "GA (genetic algorithm)"},
    "holdout_fracao":               {"PT": "Fracao holdout",          "EN": "Holdout fraction"},
    "validacao_group_aware":        {"PT": "Validacao group-aware",   "EN": "Group-aware CV"},
    "n_permutacoes":                {"PT": "N. permutacoes",          "EN": "N permutations"},
    "n_jobs_permutacao":            {"PT": "Processos paralelos",     "EN": "Parallel processes"},
    "teste_wold":                   {"PT": "Teste de Wold",           "EN": "Wold test"},
    "teste_cv_anova":               {"PT": "CV-ANOVA",                "EN": "CV-ANOVA"},
    "teste_martens":                {"PT": "Teste de Martens",        "EN": "Martens test"},
    "benchmark":                    {"PT": "Benchmark",               "EN": "Benchmark"},
    "benchmark_regressao":          {"PT": "Benchmark de regressao",  "EN": "Regression benchmark"},
    "monte_carlo":                  {"PT": "Monte Carlo CV",          "EN": "Monte Carlo CV"},
    "n_monte_carlo":                {"PT": "N. repeticoes MC",        "EN": "N MC repetitions"},
    "monte_carlo_incluir_todos":    {"PT": "MC incluir todos",        "EN": "MC include all"},
    "shap_benchmark":               {"PT": "SHAP values",             "EN": "SHAP values"},
    "shap_max_amostras":            {"PT": "SHAP max. amostras",      "EN": "SHAP max samples"},
    "figuras_detalhadas":           {"PT": "Figuras detalhadas",      "EN": "Detailed figures"},
    "figuras_mostrar_marcadores":   {"PT": "Marcadores por classe",   "EN": "Class markers"},
    "figuras_mostrar_elipses":      {"PT": "Elipses de confianca",    "EN": "Confidence ellipses"},
    "formato_figura":               {"PT": "Formato das figuras",     "EN": "Figure format"},
    "dpi":                          {"PT": "Resolucao (DPI)",         "EN": "Resolution (DPI)"},
    "abrir_figuras_na_tela":        {"PT": "Abrir figuras na tela",   "EN": "Open figures on screen"},
    "nome_execucao":                {"PT": "Nome da execucao",         "EN": "Run name"},
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
# Paletas de cores para figuras
# ---------------------------------------------------------------------------
PALETAS_COR: Dict[str, Dict[str, Any]] = {
    "qualitativo": {
        "PT": {
            "nome": "Qualitativo (padrao matplotlib)",
            "desc": "Paleta padrao do matplotlib. Boa para apresentacoes gerais.",
        },
        "EN": {
            "nome": "Qualitative (matplotlib default)",
            "desc": "Default matplotlib palette. Good for general presentations.",
        },
        "cores": None,
    },
    "daltonismo_safe": {
        "PT": {
            "nome": "Seguro para Daltonismo (Wong 2011)",
            "desc": "8 cores distinguiveis por pessoas com deuteranopia, protanopia e tritanopia.",
        },
        "EN": {
            "nome": "Colorblind Safe (Wong 2011)",
            "desc": "8 colors distinguishable for people with deuteranopia, protanopia and tritanopia.",
        },
        "cores": ["#000000", "#E69F00", "#56B4E9", "#009E73",
                  "#F0E442", "#0072B2", "#D55E00", "#CC79A7"],
    },
    "cinza": {
        "PT": {
            "nome": "Escala de Cinza",
            "desc": "Para publicacoes em preto e branco. Distingue por intensidade.",
        },
        "EN": {
            "nome": "Grayscale",
            "desc": "For black and white publications. Distinguishes by intensity.",
        },
        "cores": ["#000000", "#333333", "#555555", "#777777",
                  "#999999", "#bbbbbb", "#dddddd", "#eeeeee"],
    },
    "viridis": {
        "PT": {
            "nome": "Viridis (perceptualmente uniforme)",
            "desc": "Paleta sequencial perceptualmente uniforme. Robusta para daltonismo.",
        },
        "EN": {
            "nome": "Viridis (perceptually uniform)",
            "desc": "Perceptually uniform sequential palette. Robust for colorblindness.",
        },
        "cores": None,
        "cmap": "viridis",
    },
    "publicacao": {
        "PT": {
            "nome": "Publicacao Cientifica (Nature/Elsevier)",
            "desc": "Cores usadas em revistas como Nature, Food Chemistry e Talanta.",
        },
        "EN": {
            "nome": "Scientific Publication (Nature/Elsevier)",
            "desc": "Colors used in journals like Nature, Food Chemistry and Talanta.",
        },
        "cores": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
    },
    "escuro": {
        "PT": {
            "nome": "Tema Escuro (fundo preto)",
            "desc": "Otimizado para fundos escuros. Ideal para apresentacoes em tela.",
        },
        "EN": {
            "nome": "Dark Theme (black background)",
            "desc": "Optimized for dark backgrounds. Ideal for screen presentations.",
        },
        "cores": ["#00bfff", "#ff6347", "#7fff00", "#ffd700",
                  "#da70d6", "#ff8c00", "#40e0d0", "#ff69b4"],
        "style": "dark_background",
    },
}

# ---------------------------------------------------------------------------
# Presets de tamanho de fonte para matplotlib (A1)
# ---------------------------------------------------------------------------
FONT_PRESETS: Dict[str, Dict[str, Any]] = {
    "xs": {"font.size": 8,  "axes.titlesize": 9,  "axes.labelsize": 8,
           "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7},
    "s":  {"font.size": 9,  "axes.titlesize": 10, "axes.labelsize": 9,
           "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8},
    "m":  {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
           "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9},
    "l":  {"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
           "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10},
    "xl": {"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
           "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11},
}

_VISUAL_CFG_PATH = _BASE_DIR / "visual_config.json"
_VISUAL_CFG_CACHE: Dict[str, Any] = {}


def _carregar_visual_cfg() -> Dict[str, Any]:
    """Carrega visual_config.json com cache em memoria."""
    global _VISUAL_CFG_CACHE
    if _VISUAL_CFG_CACHE:
        return _VISUAL_CFG_CACHE
    if not _VISUAL_CFG_PATH.exists():
        _VISUAL_CFG_CACHE = {"paleta": "qualitativo", "estilo_matplotlib": "default",
                              "tamanho_fonte": "m", "grid_major": True, "grid_minor": False,
                              "grid_style": "dotted", "grid_alpha": 0.4, "alpha_pontos": "medio"}
        return _VISUAL_CFG_CACHE
    try:
        with open(_VISUAL_CFG_PATH, "r", encoding="utf-8") as f:
            _VISUAL_CFG_CACHE = json.load(f)
    except (json.JSONDecodeError, OSError):
        _VISUAL_CFG_CACHE = {"paleta": "qualitativo", "tamanho_fonte": "m"}
    return _VISUAL_CFG_CACHE


def _salvar_visual_cfg(cfg_v: Dict[str, Any]) -> None:
    """Salva visual_config.json e invalida o cache."""
    global _VISUAL_CFG_CACHE
    _VISUAL_CFG_CACHE = cfg_v.copy()  # atualiza cache ao mesmo tempo
    try:
        with open(_VISUAL_CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg_v, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"  [AVISO] Nao foi possivel salvar visual_config.json: {e}")


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
            "desc": "Pipeline de pre-processamento espectral aplicado antes da modelagem quimiometrica.",
            "impacto": "ANALITICO — o pre-processamento correto e essencial para resultados validos em FT-NIR.",
            "exemplos": {
                "msc_sg_mc": "MSC + Savitzky-Golay + Mean-Centering (recomendado para oleos vegetais)",
                "snv_sg_mc": "SNV + Savitzky-Golay + Mean-Centering (alternativa robusta)",
                "autoscaling": "Autoscaling — linha de base simples, nao recomendado para FT-NIR",
            },
        },
        "EN": {
            "desc": "Spectral preprocessing pipeline applied before chemometric modelling.",
            "impacto": "ANALYTICAL — correct preprocessing is essential for valid FT-NIR results.",
            "exemplos": {
                "msc_sg_mc": "MSC + Savitzky-Golay + Mean-Centering (recommended for vegetable oils)",
                "snv_sg_mc": "SNV + Savitzky-Golay + Mean-Centering (robust alternative)",
                "autoscaling": "Autoscaling — simple baseline, not recommended for FT-NIR",
            },
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
            "desc": "Modo de analise: N1=Classificacao (identifica a especie), "
                    "N2=Discriminacao (puro vs. adulterado, com DD-SIMCA), "
                    "N3=Quantificacao (estima o teor de adulterante, PLS-R).",
            "impacto": "ANALITICO — define quais modulos sao executados.",
            "exemplos": {
                "N1": "Classificacao por especie (PLS-DA/OPLS-DA)",
                "N2": "Puro vs. adulterado (PLS-DA + DD-SIMCA one-class)",
                "N3": "Quantificacao do teor de adulterante (PLS-R)"},
        },
        "EN": {
            "desc": "Analysis mode: N1=Classification (identify species), "
                    "N2=Discrimination (pure vs. adulterated, with DD-SIMCA), "
                    "N3=Quantification (estimate adulterant content, PLS-R).",
            "impacto": "ANALYTICAL — defines which modules are executed.",
            "exemplos": {
                "N1": "Species classification (PLS-DA/OPLS-DA)",
                "N2": "Pure vs. adulterated (PLS-DA + DD-SIMCA one-class)",
                "N3": "Adulterant content quantification (PLS-R)"},
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
    "benchmark_regressao": {
        "PT": {
            "desc": "Compara PLS-R (ja calibrado por especie) contra Ridge, Lasso, Elastic Net, "
                    "SVR e Random Forest -- mesmo split cal/val e pre-processamento, por especie.",
            "impacto": "AVANCADO — so' roda em N2/N3 com regressao multi-especie ja calculada.",
            "exemplos": {"false": "TCC/producao (recomendado)", "true": "Artigo com comparacao de regressores"},
        },
        "EN": {
            "desc": "Compares PLS-R (already calibrated per species) against Ridge, Lasso, Elastic Net, "
                    "SVR and Random Forest -- same cal/val split and preprocessing, per species.",
            "impacto": "ADVANCED — only runs on N2/N3 with multi-species regression already computed.",
            "exemplos": {"false": "TCC/production (recommended)", "true": "Article with regressor comparison"},
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
    "selecao_spa": {
        "PT": {
            "desc": "Alem dos metodos acima, roda tambem SPA/APS (Algoritmo das "
                    "Projecoes Sucessivas, Araujo et al. 2001) — constroi cadeias "
                    "de variaveis com baixa colinearidade a partir de varios "
                    "pontos de partida do espectro.",
            "impacto": "ANALITICO — mais lento que iPLS/VIP/SR/sPLS-DA (varias "
                       "avaliacoes de CV por ponto de partida).",
            "exemplos": {"false": "Exploracao rapida (recomendado)", "true": "Publicacao"},
        },
        "EN": {
            "desc": "In addition to the methods above, also runs SPA (Successive "
                    "Projections Algorithm, Araujo et al. 2001) — builds "
                    "low-collinearity variable chains from several spectral "
                    "starting points.",
            "impacto": "ANALYTICAL — slower than iPLS/VIP/SR/sPLS-DA (several CV "
                       "evaluations per starting point).",
            "exemplos": {"false": "Quick exploration (recommended)", "true": "Publication"},
        },
        "default": False, "range": "true | false",
    },
    "selecao_ag": {
        "PT": {
            "desc": "Alem dos metodos acima, roda tambem AG (Algoritmo Genetico, "
                    "GA-PLS) — populacao de subconjuntos de variaveis evoluida "
                    "por selecao/crossover/mutacao, fitness = acuracia via CV.",
            "impacto": "ANALITICO — o mais lento dos metodos de selecao "
                       "(populacao x geracoes avaliacoes de CV).",
            "exemplos": {"false": "Exploracao rapida (recomendado)", "true": "Publicacao"},
        },
        "EN": {
            "desc": "In addition to the methods above, also runs GA (Genetic "
                    "Algorithm, GA-PLS) — a population of variable subsets "
                    "evolved via selection/crossover/mutation, fitness = CV "
                    "accuracy.",
            "impacto": "ANALYTICAL — the slowest selection method (population x "
                       "generations CV evaluations).",
            "exemplos": {"false": "Quick exploration (recommended)", "true": "Publication"},
        },
        "default": False, "range": "true | false",
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
            "desc": "Pasta com arquivos .dx (JCAMP-DX). Suporta FT-NIR, NIR, MIR, Raman.",
            "impacto": "ANALITICO — define os dados de entrada do pipeline.",
            "exemplos": {"dados": "Pasta padrao do projeto", r"C:\meus_dados\oleos": "Caminho absoluto personalizado"},
        },
        "EN": {
            "desc": "Folder with .dx (JCAMP-DX) files. Supports FT-NIR, NIR, MIR, Raman.",
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
            "desc": "Origem dos dados de entrada: dx (espectros JCAMP-DX) | csv | "
                    "imagem (colorimetria digital, prototipo) | sintetico (testes).",
            "impacto": "ANALITICO — define o formato de leitura e parsing dos dados.",
            "exemplos": {"dx": "Espectros JCAMP-DX (FT-NIR, Raman, MIR)", "csv": "Tabela generica com colunas espectrais",
                         "imagem": "Fotos (1 subpasta por classe) -> features de cor RGB/HSV/Lab",
                         "sintetico": "Dados simulados para teste do pipeline"},
        },
        "EN": {
            "desc": "Input data source: dx (JCAMP-DX spectra) | csv | "
                    "imagem (digital colorimetry, prototype) | synthetic (for testing).",
            "impacto": "ANALYTICAL — defines the data reading and parsing format.",
            "exemplos": {"dx": "JCAMP-DX spectra (FT-NIR, Raman, MIR)", "csv": "Generic table with spectral columns",
                         "imagem": "Photos (1 subfolder per class) -> RGB/HSV/Lab color features",
                         "sintetico": "Simulated data for pipeline testing"},
        },
        "default": "dx", "range": "dx | csv | imagem | sintetico",
    },
    "imagem_incluir_textura": {
        "PT": {
            "desc": "Modo imagem: alem das features de cor (media/desvio RGB+HSV+Lab), "
                    "inclui features de textura via GLCM (contraste, homogeneidade, "
                    "energia, correlacao). Requer 'pip install scikit-image'.",
            "impacto": "ANALITICO — mais variaveis, so tem efeito no modo_entrada='imagem'.",
            "exemplos": {"false": "So cor (recomendado, sem dependencia extra)", "true": "Cor + textura"},
        },
        "EN": {
            "desc": "Image mode: besides color features (RGB+HSV+Lab mean/std), "
                    "also includes GLCM texture features (contrast, homogeneity, "
                    "energy, correlation). Requires 'pip install scikit-image'.",
            "impacto": "ANALYTICAL — more variables, only affects modo_entrada='imagem'.",
            "exemplos": {"false": "Color only (recommended, no extra dependency)", "true": "Color + texture"},
        },
        "default": False, "range": "true | false",
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
    "teste_martens": {
        "PT": {
            "desc": "Jackknifing group-aware dos coeficientes PLS (Martens & Martens, 2000) -- "
                    "teste de hipotese formal (p-valor) de significancia por variavel, mais "
                    "rigoroso que VIP/Selectivity Ratio (medidas de magnitude).",
            "impacto": "ANALITICO — gera teste_martens.csv com t/p-valor por comprimento de onda.",
            "exemplos": {"true": "Artigo com selecao de variaveis rigorosa", "false": "Exploracao rapida"},
        },
        "EN": {
            "desc": "Group-aware jackknifing of PLS coefficients (Martens & Martens, 2000) -- "
                    "formal hypothesis test (p-value) of per-variable significance, more "
                    "rigorous than VIP/Selectivity Ratio (magnitude measures).",
            "impacto": "ANALYTICAL — generates teste_martens.csv with t/p-value per wavenumber.",
            "exemplos": {"true": "Article with rigorous variable selection", "false": "Quick exploration"},
        },
        "default": False, "range": "true | false",
    },
    "figuras_detalhadas": {
        "PT": {
            "desc": "Gerar tambem as figuras exploratorias/detalhadas (HCA, loadings PCA, "
                    "pre-processamento, contribuicao de score, DD-SIMCA por classe, Cooman).",
            "impacto": "VISUAL — mais arquivos e tempo de execucao; nao altera o modelo.",
            "exemplos": {"false": "So o essencial (mais rapido, recomendado)", "true": "Analise exploratoria completa"},
        },
        "EN": {
            "desc": "Also generate exploratory/detailed figures (HCA, PCA loadings, "
                    "preprocessing, score contribution, per-class DD-SIMCA, Cooman's plot).",
            "impacto": "VISUAL — more files and runtime; does not change the model.",
            "exemplos": {"false": "Essentials only (faster, recommended)", "true": "Full exploratory analysis"},
        },
        "default": False, "range": "true | false",
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
    "tag": {
        "PT": {
            "desc": "Sufixo personalizado para o nome da pasta de resultados.",
            "impacto": "VISUAL — nao altera analise, apenas organiza as saidas.",
            "exemplos": {
                "": "Pasta automatica: PLSDA_N2_MSC-SG1-MC_YYYYMMDD_HHMMSS",
                "experimento01": "Pasta: PLSDA_N2_..._experimento01",
                "artigo_final": "Pasta: PLSDA_N2_..._artigo_final",
            },
        },
        "EN": {
            "desc": "Custom suffix for the results folder name.",
            "impacto": "VISUAL — does not affect analysis, only organizes outputs.",
            "exemplos": {
                "": "Auto folder: PLSDA_N2_MSC-SG1-MC_YYYYMMDD_HHMMSS",
                "experiment01": "Folder: PLSDA_N2_..._experiment01",
                "final_paper": "Folder: PLSDA_N2_..._final_paper",
            },
        },
        "default": "", "range": "Texto livre (sem espacos recomendado)",
    },
    "nome_execucao": {
        "PT": {
            "desc": "Nome personalizado para a execucao atual. Aparece no nome da pasta de resultados.",
            "impacto": "VISUAL — nao afeta calculos. Organiza diferentes rodadas.",
            "exemplos": {
                "": "Nome automatico: PLSDA_N2_MSC-SG1-MC_YYYYMMDD_HHMMSS",
                "artigo_2026": "Pasta: ...artigo_2026",
                "dataset_completo": "Pasta: ...dataset_completo",
            },
        },
        "EN": {
            "desc": "Custom name for the current run. Appears in the results folder name.",
            "impacto": "VISUAL — does not affect calculations. Organizes different runs.",
            "exemplos": {
                "": "Auto name: PLSDA_N2_MSC-SG1-MC_YYYYMMDD_HHMMSS",
                "paper_2026": "Folder: ...paper_2026",
                "full_dataset": "Folder: ...full_dataset",
            },
        },
        "default": "", "range": "Texto livre (sem espacos)",
    },
    "paleta_cores": {
        "PT": {
            "desc": "Paleta de cores para as figuras geradas pelo pipeline.",
            "impacto": "VISUAL — nao afeta calculos. Aplica antes de rodar.",
            "exemplos": {
                "qualitativo": "Cores matplotlib padrao (multiuso)",
                "daltonismo_safe": "Seguro para daltonismo (Wong 2011)",
                "publicacao": "Cores de revistas cientificas",
                "escuro": "Tema escuro para apresentacoes",
            },
        },
        "EN": {
            "desc": "Color palette for pipeline-generated figures.",
            "impacto": "VISUAL — does not affect calculations. Applied before running.",
            "exemplos": {
                "qualitativo": "Default matplotlib colors (multipurpose)",
                "daltonismo_safe": "Colorblind safe (Wong 2011)",
                "publicacao": "Scientific journal colors",
                "escuro": "Dark theme for presentations",
            },
        },
        "default": "qualitativo", "range": "qualitativo | daltonismo_safe | cinza | viridis | publicacao | escuro",
    },
    "codificacao_arquivos": {
        "PT": {
            "desc": "Formato de nomenclatura dos arquivos DX para leitura pelo pipeline.",
            "impacto": "ANALITICO — arquivos com nomenclatura incorreta serao ignorados.",
            "exemplos": {
                "AND-10-06-2020_T1.dx": "Andiroba pura, triplicata 1",
                "BCB-03-03-2020_AD-S-20_T1.dx": "Bacaba adulterada com 20% de soja",
            },
        },
        "EN": {
            "desc": "DX file naming format for pipeline reading.",
            "impacto": "ANALYTICAL — files with incorrect naming will be ignored.",
            "exemplos": {
                "AND-10-06-2020_T1.dx": "Pure Andiroba, replicate 1",
                "BCB-03-03-2020_AD-S-20_T1.dx": "Bacaba adulterated with 20% soy",
            },
        },
        "default": "COD-DD-MM-AAAA_Tn.dx", "range": "Ver menu Codificacao",
    },
    "tamanho_fonte": {
        "PT": {
            "desc": "Tamanho das fontes nas figuras geradas (titulo, eixos, legenda).",
            "impacto": "VISUAL — nao afeta calculos. Essencial para posters e apresentacoes.",
            "exemplos": {"xs": "Muito pequeno", "m": "Medio (padrao)", "xl": "Muito grande"},
        },
        "EN": {
            "desc": "Font size in generated figures (title, axes, legend).",
            "impacto": "VISUAL — no effect on calculations. Essential for posters and presentations.",
            "exemplos": {"xs": "Very small", "m": "Medium (default)", "xl": "Very large"},
        },
        "default": "m", "range": "xs | s | m | l | xl",
    },
    "grid_major": {
        "PT": {
            "desc": "Ativa grid principal nas figuras. Melhora leitura de valores.",
            "impacto": "VISUAL — padrao cientifico para publicacoes.",
            "exemplos": {"true": "Grid principal ativado (recomendado)", "false": "Sem grid"},
        },
        "EN": {
            "desc": "Enable major grid in figures. Improves value reading.",
            "impacto": "VISUAL — scientific standard for publications.",
            "exemplos": {"true": "Major grid enabled (recommended)", "false": "No grid"},
        },
        "default": "true", "range": "true | false",
    },
    "alpha_pontos": {
        "PT": {
            "desc": "Transparencia dos pontos nos graficos de dispersao (scatter).",
            "impacto": "VISUAL — alfa alto revela densidade de pontos sobrepostos.",
            "exemplos": {"baixo": "0.9 — pontos opacos", "medio": "0.65 — equilibrado", "alto": "0.35 — translucido"},
        },
        "EN": {
            "desc": "Transparency of points in scatter plots.",
            "impacto": "VISUAL — high alpha reveals density of overlapping points.",
            "exemplos": {"baixo": "0.9 — opaque points", "medio": "0.65 — balanced", "alto": "0.35 — translucent"},
        },
        "default": "medio", "range": "baixo | medio | alto",
    },
    "pca_biplot": {
        "PT": {
            "desc": "Gera PCA Biplot 2D com elipse de confianca 95% (Hotelling T2) por classe.",
            "impacto": "ANALITICO — mostra separacao espectral e regioes de sobreposicao entre classes.",
            "exemplos": {
                "PC1 vs PC2": "Componentes com maior variancia explicada",
                "Elipse 95%": "Regiao de confianca por Hotelling T2 (chi2 2 graus)",
                "Setas loadings": "Top-8 variaveis com maior contribuicao nos PCs",
            },
        },
        "EN": {
            "desc": "Generates 2D PCA Biplot with 95% confidence ellipse (Hotelling T2) per class.",
            "impacto": "ANALYTICAL — shows spectral separation and overlap regions between classes.",
            "exemplos": {
                "PC1 vs PC2": "Components with highest explained variance",
                "95% ellipse": "Confidence region by Hotelling T2 (chi2 2 degrees)",
                "Loading arrows": "Top-8 variables with highest PC contribution",
            },
        },
        "default": "N/A", "range": "menu [7] > [B]",
    },
    "wavelength_importance": {
        "PT": {
            "desc": "Importancia espectral por loadings PCA ponderados pela variancia explicada.",
            "impacto": "ANALITICO — identifica regioes espectrais informativas antes da modelagem.",
            "exemplos": {
                "Pico em 5200 cm-1": "Alta importancia = regiao discriminante para as classes",
                "Top-10 regioes": "Marcados em vermelho no grafico",
            },
        },
        "EN": {
            "desc": "Spectral importance via PCA loadings weighted by explained variance.",
            "impacto": "ANALYTICAL — identifies informative spectral regions before modelling.",
            "exemplos": {
                "Peak at 5200 cm-1": "High importance = discriminating region for classes",
                "Top-10 regions": "Marked in red on the figure",
            },
        },
        "default": "N/A", "range": "menu [7] > [V]",
    },
}

# ---------------------------------------------------------------------------
# Perfis prontos
# ---------------------------------------------------------------------------
PROFILES: Dict[str, Dict[str, Any]] = {
    # 1 — primeiro contato com os dados, sem esperar
    "Exploracao Rapida": {
        "max_lvs": 20, "n_permutacoes": 50,
        "ddsimca": False, "opls_da": False,
        "benchmark": False, "monte_carlo": False, "shap_benchmark": False,
        "selecao_variaveis_etapa4": False, "comparar_pre_processamentos": False,
        "dpi": 150,
    },
    # 2 — análise completa balanceada, neutro, para uso geral
    "Analise Padrao": {
        "max_lvs": 30, "n_permutacoes": 100,
        "ddsimca": True, "modo_ddsimca": "puros", "opls_da": True,
        "benchmark": False, "monte_carlo": False, "shap_benchmark": False,
        "selecao_variaveis_etapa4": True, "comparar_pre_processamentos": False,
        "dpi": 300, "figuras_mostrar_elipses": True,
        "_paleta": "publicacao",
    },
    # 3 — validação estatística mais rigorosa para trabalho científico
    "Pesquisa Academica": {
        "max_lvs": 40, "n_permutacoes": 200,
        "ddsimca": True, "modo_ddsimca": "puros", "opls_da": True,
        "benchmark": False, "monte_carlo": False, "shap_benchmark": False,
        "selecao_variaveis_etapa4": True, "comparar_pre_processamentos": False,
        "dpi": 300, "figuras_mostrar_elipses": True,
        "_paleta": "publicacao",
    },
    # 4 — pronto para submissão em periódico indexado
    "Publicacao Cientifica": {
        "max_lvs": 40, "n_permutacoes": 200,
        "ddsimca": True, "modo_ddsimca": "puros", "opls_da": True,
        "benchmark": True, "monte_carlo": False, "shap_benchmark": True,
        "selecao_variaveis_etapa4": True, "comparar_pre_processamentos": False,
        "dpi": 600, "figuras_mostrar_elipses": True,
        "_paleta": "publicacao",
    },
    # 5 — máxima rigorosidade estatística (dissertação / tese)
    "Alta Rigorosidade": {
        "max_lvs": 40, "n_permutacoes": 500,
        "ddsimca": True, "modo_ddsimca": "puros", "opls_da": True,
        "benchmark": True, "monte_carlo": True, "n_monte_carlo": 200,
        "shap_benchmark": True, "selecao_variaveis_etapa4": True,
        "comparar_pre_processamentos": False,
        "dpi": 600, "formato_figura": "pdf",
        "_paleta": "publicacao",
    },
    # 6 — foco exclusivo em comparar pipelines de pré-processamento
    "Benchmark Preprocessamento": {
        "max_lvs": 30, "n_permutacoes": 100,
        "ddsimca": False, "opls_da": False,
        "benchmark": False, "monte_carlo": False, "shap_benchmark": False,
        "selecao_variaveis_etapa4": False, "comparar_pre_processamentos": True,
        "dpi": 300,
    },
    # 7 — acessibilidade para daltonismo (único com daltonismo_safe)
    "Acessibilidade": {
        "max_lvs": 30, "n_permutacoes": 100,
        "ddsimca": True, "modo_ddsimca": "puros", "opls_da": True,
        "benchmark": False, "monte_carlo": False, "shap_benchmark": False,
        "selecao_variaveis_etapa4": True, "comparar_pre_processamentos": False,
        "dpi": 300, "figuras_mostrar_elipses": True,
        "figuras_mostrar_marcadores": True,
        "_paleta": "daltonismo_safe",
    },
}

# ---------------------------------------------------------------------------
# Descricoes dos perfis prontos (bilíngue)
# ---------------------------------------------------------------------------
PROFILE_DESC: Dict[str, Dict[str, str]] = {
    "Exploracao Rapida": {
        "PT": ("Primeiro contato com os dados. Somente PLS-DA basico, sem modulos pesados.\n"
               "    Ideal para: verificar se os dados carregam, testar o pipeline.\n"
               "    Tempo estimado: ~5 min."),
        "EN": ("First contact with data. Basic PLS-DA only, no heavy modules.\n"
               "    Ideal for: verifying data loads, testing the pipeline.\n"
               "    Estimated time: ~5 min."),
    },
    "Analise Padrao": {
        "PT": ("Analise completa equilibrada. Uso geral, neutro, sem exageros.\n"
               "    PLS-DA + OPLS-DA + DD-SIMCA + selecao de variaveis. DPI 300.\n"
               "    Paleta Nature/Elsevier. Sem benchmark nem Monte Carlo.\n"
               "    Tempo estimado: ~15-30 min."),
        "EN": ("Balanced complete analysis. General use, neutral, no excess.\n"
               "    PLS-DA + OPLS-DA + DD-SIMCA + variable selection. DPI 300.\n"
               "    Nature/Elsevier palette. No benchmark or Monte Carlo.\n"
               "    Estimated time: ~15-30 min."),
    },
    "Pesquisa Academica": {
        "PT": ("Validacao estatistica mais rigorosa para trabalho cientifico.\n"
               "    Analise completa com 200 permutacoes. Sem benchmark (economiza tempo).\n"
               "    Recomendado para relatorios de pesquisa e comunicacoes em eventos.\n"
               "    Tempo estimado: ~30-45 min."),
        "EN": ("Stricter statistical validation for scientific work.\n"
               "    Full analysis with 200 permutations. No benchmark (saves time).\n"
               "    Recommended for research reports and conference communications.\n"
               "    Estimated time: ~30-45 min."),
    },
    "Publicacao Cientifica": {
        "PT": ("Pronto para submissao em revista indexada. Foco: comparacao de modelos.\n"
               "    Benchmark SVM/RF/XGBoost vs PLS-DA + SHAP + 200 perm + DPI 600.\n"
               "    Recomendado: Talanta, Food Chemistry, J. Chemometrics.\n"
               "    Tempo estimado: ~60-120 min."),
        "EN": ("Ready for indexed journal submission. Focus: model comparison.\n"
               "    Benchmark SVM/RF/XGBoost vs PLS-DA + SHAP + 200 perm + DPI 600.\n"
               "    Recommended: Talanta, Food Chemistry, J. Chemometrics.\n"
               "    Estimated time: ~60-120 min."),
    },
    "Alta Rigorosidade": {
        "PT": ("Maxima rigorosidade estatistica. Foco: incerteza e validacao multipla.\n"
               "    Monte Carlo CV IC95% (200 rep.) + benchmark + SHAP + 500 perm.\n"
               "    Figuras PDF vetorial. Para dissertacao, tese ou revisao externa.\n"
               "    Tempo estimado: ~3-6 horas."),
        "EN": ("Maximum statistical rigor. Focus: uncertainty and multiple validation.\n"
               "    Monte Carlo CV CI95% (200 rep.) + benchmark + SHAP + 500 perm.\n"
               "    Vector PDF figures. For dissertation, thesis or external review.\n"
               "    Estimated time: ~3-6 hours."),
    },
    "Benchmark Preprocessamento": {
        "PT": ("Foco exclusivo: qual pipeline de pre-processamento e o melhor?\n"
               "    Compara raw, MSC, SNV, SG1, SG2, MSC+SG, SNV+SG com metricas.\n"
               "    Sem analise completa — apenas comparacao de pipelines espectrais.\n"
               "    Tempo estimado: ~20-40 min."),
        "EN": ("Exclusive focus: which pre-processing pipeline is best?\n"
               "    Compares raw, MSC, SNV, SG1, SG2, MSC+SG, SNV+SG with metrics.\n"
               "    No full analysis — only spectral pipeline comparison.\n"
               "    Estimated time: ~20-40 min."),
    },
    "Acessibilidade": {
        "PT": ("Foco: figuras acessiveis para daltonismo. Analise padrao.\n"
               "    Paleta Wong 2011 (daltonismo_safe): 8 cores para deuteranopia/protanopia.\n"
               "    Marcadores de forma por classe alem da cor. DPI 300.\n"
               "    Tempo estimado: ~15-30 min."),
        "EN": ("Focus: accessible figures for color blindness. Standard analysis.\n"
               "    Wong 2011 palette (colorblind_safe): 8 colors for deuteranopia/protanopia.\n"
               "    Shape markers per class in addition to color. DPI 300.\n"
               "    Estimated time: ~15-30 min."),
    },
}

PROFILE_KEY_SUMMARY: Dict[str, Dict[str, str]] = {
    "Exploracao Rapida": {
        "PT": "LVs=20 | perm=50 | DD-SIMCA=OFF | OPLS=OFF | DPI=150",
        "EN": "LVs=20 | perm=50 | DD-SIMCA=OFF | OPLS=OFF | DPI=150",
    },
    "Analise Padrao": {
        "PT": "LVs=30 | perm=100 | OPLS=ON | DD-SIMCA=ON | DPI=300 | paleta=publicacao",
        "EN": "LVs=30 | perm=100 | OPLS=ON | DD-SIMCA=ON | DPI=300 | palette=publication",
    },
    "Pesquisa Academica": {
        "PT": "LVs=40 | perm=200 | OPLS=ON | DD-SIMCA=ON | DPI=300 | paleta=publicacao",
        "EN": "LVs=40 | perm=200 | OPLS=ON | DD-SIMCA=ON | DPI=300 | palette=publication",
    },
    "Publicacao Cientifica": {
        "PT": "LVs=40 | perm=200 | Benchmark=ON | SHAP=ON | DPI=600 | paleta=publicacao",
        "EN": "LVs=40 | perm=200 | Benchmark=ON | SHAP=ON | DPI=600 | palette=publication",
    },
    "Alta Rigorosidade": {
        "PT": "perm=500 | MonteCarlo=200rep | Benchmark=ON | SHAP=ON | PDF | paleta=publicacao",
        "EN": "perm=500 | MonteCarlo=200rep | Benchmark=ON | SHAP=ON | PDF | palette=publication",
    },
    "Benchmark Preprocessamento": {
        "PT": "comparar_pipelines=ON | LVs=30 | perm=100 | OPLS=OFF | DD-SIMCA=OFF",
        "EN": "compare_pipelines=ON | LVs=30 | perm=100 | OPLS=OFF | DD-SIMCA=OFF",
    },
    "Acessibilidade": {
        "PT": "LVs=30 | perm=100 | OPLS=ON | DD-SIMCA=ON | marcadores=ON | paleta=daltonismo_safe",
        "EN": "LVs=30 | perm=100 | OPLS=ON | DD-SIMCA=ON | markers=ON | palette=colorblind_safe",
    },
}

# ---------------------------------------------------------------------------
# Referencias bibliograficas verificadas (usadas no help [?] e no Assistente)
# ---------------------------------------------------------------------------
REFERENCIAS_GUARACI: Dict[str, Dict[str, str]] = {
    "pls_da_brereton": {
        "cit": ("Brereton, R. G.; Lloyd, G. R. (2014). Partial least squares discriminant "
                "analysis: taking the magic away. Journal of Chemometrics, 28(4), 213-225. "
                "doi:10.1002/cem.2609"),
        "contexto": "PLS-DA — fundamentos e armadilhas comuns",
    },
    "pls_geladi_1986": {
        "cit": ("Geladi, P.; Kowalski, B. R. (1986). Partial least-squares regression: "
                "a tutorial. Analytica Chimica Acta, 185, 1-17. "
                "doi:10.1016/0003-2670(85)85121-2"),
        "contexto": "PLS — referencia fundamental",
    },
    "opls_da_trygg_2002": {
        "cit": ("Trygg, J.; Wold, S. (2002). Orthogonal projections to latent structures (O-PLS). "
                "Journal of Chemometrics, 16(3), 119-128. doi:10.1002/cem.695"),
        "contexto": "OPLS-DA — metodo base",
    },
    "dd_simca_pomerantsev": {
        "cit": ("Pomerantsev, A. L.; Rodionova, O. Y. (2014). Concept and role of extreme objects "
                "in PCA/SIMCA. Journal of Chemometrics, 28(5), 429-438. doi:10.1002/cem.2506"),
        "contexto": "DD-SIMCA — fundamentos",
    },
    "msc_geladi_1985": {
        "cit": ("Geladi, P. et al. (1985). Linearization and scatter-correction for "
                "near-infrared reflectance spectra of meat. Applied Spectroscopy, 39(3), 491-500. "
                "doi:10.1366/0003702854248684"),
        "contexto": "MSC — artigo original",
    },
    "snv_barnes_1989": {
        "cit": ("Barnes, R. J. et al. (1989). Standard Normal Variate Transformation and "
                "De-trending of Near-Infrared Diffuse Reflectance Spectra. "
                "Applied Spectroscopy, 43(5), 772-777. doi:10.1366/0003702894202201"),
        "contexto": "SNV — artigo original",
    },
    "savitzky_golay_1964": {
        "cit": ("Savitzky, A.; Golay, M. J. E. (1964). Smoothing and Differentiation of Data "
                "by Simplified Least Squares Procedures. Analytical Chemistry, 36(8), 1627-1639. "
                "doi:10.1021/ac60214a047"),
        "contexto": "Savitzky-Golay — artigo original",
    },
    "oleos_vegetais_nir_review": {
        "cit": ("Sherazi, S. T. H. et al. (2023). Application of near-infrared spectroscopy "
                "for authentication of edible oils: A review. Food Chemistry, 408, 135181. "
                "doi:10.1016/j.foodchem.2022.135181"),
        "contexto": "Revisao NIR para oleos vegetais — atual",
    },
    "shap_lundberg_2017": {
        "cit": ("Lundberg, S. M.; Lee, S.-I. (2017). A Unified Approach to Interpreting "
                "Model Predictions. Advances in Neural Information Processing Systems (NeurIPS), "
                "30. arXiv:1705.07874"),
        "contexto": "SHAP values — metodo base",
    },
    "monte_carlo_cv_xu": {
        "cit": ("Xu, Q.-S.; Liang, Y.-Z. (2001). Monte Carlo cross validation. "
                "Chemometrics and Intelligent Laboratory Systems, 56(1), 1-11. "
                "doi:10.1016/S0169-7439(00)00122-2"),
        "contexto": "Monte Carlo CV — metodo",
    },
}

# Mapeamento de menus -> campos (chaves do _CONFIG_SPEC)
# ---------------------------------------------------------------------------
MENU_FIELDS: Dict[str, list] = {
    "projeto": ["pasta_dados", "pasta_saida", "nome_execucao"],
    "dados": ["modo_entrada", "arquivo_csv", "coluna_classe", "coluna_concentracao",
              "faixa_min_cm", "faixa_max_cm", "excluir_classes",
              "imagem_incluir_textura"],
    "preproc": ["pre_processamento", "comparar_pre_processamentos"],
    "modelo": ["nivel", "max_lvs", "opls_da", "ddsimca", "modo_ddsimca",
               "selecao_variaveis_etapa4", "selecao_spa", "selecao_ag"],
    "validacao": ["holdout_fracao", "validacao_group_aware", "n_permutacoes",
                  "teste_wold", "teste_cv_anova", "teste_martens", "n_jobs_permutacao"],
    "avancado": ["benchmark", "benchmark_regressao", "monte_carlo", "n_monte_carlo",
                 "monte_carlo_incluir_todos", "shap_benchmark", "shap_max_amostras"],
    "visualizacao": ["figuras_detalhadas", "figuras_mostrar_marcadores",
                     "figuras_mostrar_elipses",
                     "formato_figura", "dpi", "abrir_figuras_na_tela"],
}

# Indice inverso: chave -> especificacao do _CONFIG_SPEC
_SPEC_BY_KEY: Dict[str, Dict[str, Any]] = {s["key"]: s for s in _CONFIG_SPEC}

# Campos extras presentes em Config mas ausentes do _CONFIG_SPEC (nao editaveis pelo pipeline CLI)
_SPEC_EXTRAS: Dict[str, Dict[str, Any]] = {
    "tag": {"key": "tag", "attr": "tag", "tipo": "str", "desc": "Sufixo da pasta de saida", "opcoes": None},
    "nome_execucao": {"key": "nome_execucao", "attr": "tag", "tipo": "str",
                      "desc": "Nome da execucao atual (alias de tag)", "opcoes": None},
}
_SPEC_BY_KEY.update(_SPEC_EXTRAS)


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
    if key == "nivel":
        # Mostra o codigo + nome amigavel (Classificacao/Discriminacao/
        # Quantificacao); o valor gravado continua N1/N2/N3.
        nome = pq._NIVEL_NOME.get(str(raw), "")
        return f"{raw} — {nome}" if nome else raw
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
    pasta = getattr(cfg, "pasta_dados", "dados")
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


def _wrap_box(text: str, width: int, indent: str = "  ") -> list:
    """Quebra texto em linhas que cabem dentro da caixa, sem cortar palavras."""
    wrapped = _textwrap.fill(
        text,
        width=width - len(indent),
        break_long_words=True,
        break_on_hyphens=True,
    )
    return [indent + line for line in wrapped.split("\n")]


def _prompt(msg: str, default: str = "") -> str:
    """Le input do usuario com tratamento seguro de EOF e Ctrl+C.

    Retorna `default` se o usuario pressionar Ctrl+C ou fechar o stdin.
    """
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        return default


def _ler_dx_pasta(pasta: str, max_files: int = 300, ler_x: bool = False):
    """Le arquivos .dx de uma pasta e retorna (spectra, labels, wavenumbers).

    Returns:
        spectra: list of list[float] — valores Y de cada espectro
        labels: list[str] — codigo de especie (ex: 'AND', 'BCB')
        wavenumbers: list[float] — eixo X do primeiro arquivo (se ler_x=True)
    """
    import re as _re_dx
    from pathlib import Path as _Path_dx
    dx_files = sorted(_Path_dx(pasta).rglob("*.dx"))[:max_files]
    spectra, labels, wavenumbers = [], [], []
    n_falhos = 0
    for f in dx_files:
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            xvals, yvals = [], []
            in_data = False
            for line in lines:
                if "##XYDATA" in line or "##DATA TABLE" in line:
                    in_data = True; continue
                if in_data and line.startswith("##"):
                    in_data = False
                if in_data:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            if ler_x:
                                xvals.append(float(parts[0]))
                            yvals.append(float(parts[1]))
                        except ValueError:
                            pass
            if len(yvals) > 10:
                spectra.append(yvals)
                cod = _re_dx.match(r'^([A-Za-z]+)', f.stem)
                labels.append(cod.group(1).upper() if cod else f.stem[:4])
                if ler_x and not wavenumbers and xvals:
                    wavenumbers = xvals
            else:
                n_falhos += 1   # arquivo sem dados espectrais utilizaveis
        except Exception:
            n_falhos += 1
            continue
    if n_falhos:
        # Sem este aviso, arquivos ilegiveis eram descartados em silencio e a
        # previa/estatistica era calculada so nos que sobraram, sem o usuario saber.
        print(f"  [AVISO] {n_falhos} de {len(dx_files)} arquivo(s) .dx nao "
              f"puderam ser lidos e foram ignorados.")
    return spectra, labels, wavenumbers


# ===========================================================================
# Cabecalho
# ===========================================================================

def print_header(cfg: Config) -> None:
    """Cabecalho GUARACI (Rich Panel): titulo, subtitulo, idioma e status dados."""
    lang = _lang()
    if lang == "EN":
        linha2 = "Chemometric Intelligence  ·  GEAAp / UFPA"
        linha3 = "Amazonian Matrices"
    else:
        linha2 = "Inteligencia Quimiometrica  ·  GEAAp / UFPA"
        linha3 = "Matrizes Amazonicas"
    # Status dos dados: caminho truncado + icone semantico
    pasta = getattr(cfg, "pasta_dados", "dados")
    pasta_str = str(pasta)
    if len(pasta_str) > 44:
        pasta_str = "..." + pasta_str[-41:]
    dados_ok = os.path.isdir(str(pasta))

    titulo = Text("●●●  GUARACI  ●●●", style="hdr", justify="center")
    sub = Text(f"{linha2}\n{linha3}", style="m", justify="center")
    if dados_ok:
        status = Text.assemble(("✓ ", "ok"), (I18N[lang]["status_ok"], "g"),
                               ("  —  ", "d"), (pasta_str, "w"))
    else:
        status = Text.assemble(("✗ ", "err"), (I18N[lang]["status_erro"], "r"),
                               ("  —  ", "d"), (pasta_str, "m"))

    corpo = Group(titulo, sub, Text(""), status)
    _console.print()
    _console.print(Panel(corpo, box=_rbox.ROUNDED, border_style="f",
                         width=_theme_W(), padding=(0, 2),
                         subtitle=f"[m]\\[{lang}][/m]", subtitle_align="right"))


# ===========================================================================
# Menu principal
# ===========================================================================

def _opt(key: str, label: str) -> str:
    """Celula de opcao de menu: tecla em ambar + rotulo em branco."""
    return f"[a]\\[{key}][/a] [w]{label}[/w]"


def _secao_menu(titulo: str) -> Text:
    """Divisor de subgrupo em verde-floresta."""
    return Text(f"── {titulo} ", style="f")


def print_main_menu() -> None:
    """Menu principal (Rich Panel): subgrupos, teclas destacadas, barra de acoes."""
    from rich.table import Table as _Tbl
    lang = _lang()
    t = I18N[lang]

    def _grade(pares):
        tbl = _Tbl.grid(padding=(0, 3))
        tbl.add_column(); tbl.add_column()
        for esq, dir_ in pares:
            tbl.add_row(esq, dir_ or "")
        return tbl

    corpo = Group(
        _secao_menu(t["grp_analise"]),
        _grade([
            (_opt("1", t["menu_projeto"]), _opt("2", t["menu_dados"])),
            (_opt("3", t["menu_preproc"]), _opt("4", t["menu_modelo"])),
            (_opt("5", t["menu_valid"]),   _opt("6", t["menu_avancado"])),
        ]),
        Text(""),
        _secao_menu(t["grp_sistema"]),
        _grade([
            (_opt("7", t["menu_viz"]),          _opt("8", t["menu_tecnica"])),
            (_opt("9", t["menu_codificacao"]),  _opt("H", t["hardware"])),
        ]),
        Text(""),
        _secao_menu(t["grp_perfis"]),
        _grade([(_opt("P", t["perfis"]), _opt("I", t["idioma"]))]),
    )
    _console.print(Panel(corpo, box=_rbox.ROUNDED, border_style="d",
                         width=_theme_W(), padding=(0, 2),
                         title="[hdr]MENU PRINCIPAL[/hdr]", title_align="left"))
    # Barra de acoes (destaque para Rodar Pipeline em verde de sucesso).
    acoes = Text.assemble(
        ("  [R] ", "ok"), ("► " + t["rodar_pipeline"], "g"), ("    ", ""),
        (f"[N] {t['nome_saida']}   ", "w"),
        (f"[S] {t['salvar']}   ", "w"),
        (f"[L] {t['carregar']}   ", "w"),
        (f"[?] {t['ajuda_curta']}   ", "m"),
        (f"[Q] {t['sair']}", "m"))
    _console.print(acoes)


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
    print("  │" + _ansi_ljust(desc_label, w - 4) + "│")
    for dline in _wrap_line(desc, w - 6, indent=4):
        print("  │" + _ansi_ljust(dline, w - 4) + "│")

    print("  │" + " " * (w - 4) + "│")

    # Padrao e faixa
    pad_fai = f"    {t['padrao']}: {default_val}   |   {t['faixa']}: {range_val}"
    print("  │" + _ansi_ljust(pad_fai, w - 4) + "│")

    print("  │" + " " * (w - 4) + "│")

    # Impacto
    imp_label = f"  {t['impacto']}:"
    print("  │" + _ansi_ljust(imp_label, w - 4) + "│")
    for iline in _wrap_line(impacto, w - 6, indent=4):
        print("  │" + _ansi_ljust(iline, w - 4) + "│")

    if exemplos:
        print("  │" + " " * (w - 4) + "│")
        ex_label = f"  {t['exemplos']}:"
        print("  │" + _ansi_ljust(ex_label, w - 4) + "│")
        for val_ex, desc_ex in exemplos.items():
            ex_linha = f"    {val_ex:>12}  ->  {desc_ex}"
            # Truncar se necessario
            if _ansi_len(ex_linha) > w - 4:
                ex_linha = ex_linha[: w - 7] + "..."
            print("  │" + _ansi_ljust(ex_linha, w - 4) + "│")

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

    novo_raw = _prompt(f"  {t['novo_valor']}")

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
        conf = _prompt(f"  {t['aviso_analitico']}").lower()
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


def _submenu_campos(cfg: Config, titulo_key: str, campos: list, secao_key: str = "") -> None:
    """Loop generico para exibir e editar um grupo de campos.
    titulo_key e uma chave de I18N — re-avaliada a cada iteracao para suporte a troca de idioma.
    Inclui [I] Idioma na barra de rodape de cada submenu.
    """
    while True:
        lang = _lang()
        t = I18N[lang]
        titulo = I18N[lang].get(titulo_key, titulo_key)  # re-avalia cada iteracao
        cls()
        print_header(cfg)
        w = 68
        print("╔" + "═" * (w - 2) + "╗")
        print("║" + _ansi_ljust(f"  {titulo}", w - 2) + "║")
        # Exibir descricao da secao, se disponivel
        if secao_key and secao_key in SECTION_DESC:
            desc_sec = SECTION_DESC[secao_key].get(lang, "")
            if desc_sec:
                for dline in _wrap_box(desc_sec, w - 4, "  "):
                    print("║" + _ansi_ljust(_c("DIM", dline), w - 2) + "║")
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
            print("║" + _ansi_ljust(linha, w - 2) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        rodape = f"  [?] {t['ajuda_campo']}   [I] {t['idioma']}   [0] {t['voltar']}"
        print("║" + _ansi_ljust(rodape, w - 2) + "║")
        print("╚" + "═" * (w - 2) + "╝")
        print()
        escolha_raw = _prompt(f"  {t['opcao']}: ")
        if not escolha_raw:
            break
        escolha = escolha_raw

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
            _prompt(f"  [{t['continuar']}]")
            continue

        if escolha.isdigit() and 1 <= int(escolha) <= len(campos):
            key = campos[int(escolha) - 1]
            _editar_campo_cli(cfg, key)
            _prompt(f"  [{t['continuar']}]")
        else:
            print(f"  {t['invalido']}")
            _prompt(f"  [{t['continuar']}]")


# ===========================================================================
# Submenus tematicos
# ===========================================================================

def menu_projeto(cfg: Config) -> None:
    """Menu 1 — Projeto: pasta_dados e pasta_saida."""
    _submenu_campos(cfg, "menu_projeto", MENU_FIELDS["projeto"], "projeto")


def menu_dados(cfg: Config) -> None:
    """Menu 2 — Dados: modo, CSV, colunas, faixa espectral, exclusoes."""
    _submenu_campos(cfg, "menu_dados", MENU_FIELDS["dados"], "dados")


def menu_preproc(cfg: Config) -> None:
    """Menu 3 — Pre-processamento: pipeline + comparacao."""
    while True:
        lang = _lang()
        t = I18N[lang]
        cls()
        print_header(cfg)
        w = 68
        titulo = t["menu_preproc"]
        campos = MENU_FIELDS["preproc"]
        print("╔" + "═" * (w - 2) + "╗")
        print("║" + _ansi_ljust(f"  {titulo}", w - 2) + "║")
        # Descricao da secao
        desc_sec = SECTION_DESC["preproc"].get(lang, "")
        if desc_sec:
            for dline in _wrap_box(desc_sec, w - 4, "  "):
                print("║" + _ansi_ljust(_c("DIM", dline), w - 2) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        # Mostrar preview do pipeline atual
        pp_atual = _fmt_yaml(_get_val(cfg, "pre_processamento"))
        preview = f"  Pipeline atual: {_c('ANALITICO', pp_atual)}"
        print("║" + _ansi_ljust(preview, w - 2) + "║")
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
            print("║" + _ansi_ljust(linha, w - 2) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        rodape = f"  [I] {t['idioma']}   [0] {t['voltar']}"
        print("║" + _ansi_ljust(rodape, w - 2) + "║")
        print("╚" + "═" * (w - 2) + "╝")
        print()
        escolha = _prompt(f"  {I18N[_lang()]['opcao']}: ")
        if not escolha or escolha == "0" or escolha.lower() == "q":
            break
        if escolha.upper() == "I":
            _toggle_idioma()
            continue
        if escolha.isdigit() and 1 <= int(escolha) <= len(campos):
            key = campos[int(escolha) - 1]
            _editar_campo_cli(cfg, key)
            _prompt(f"  [{t['continuar']}]")
        else:
            print(f"  {t['invalido']}")
            _prompt(f"  [{t['continuar']}]")


def menu_modelagem(cfg: Config) -> None:
    """Menu 4 — Modelagem."""
    _submenu_campos(cfg, "menu_modelo", MENU_FIELDS["modelo"], "modelo")


def menu_validacao(cfg: Config) -> None:
    """Menu 5 — Validacao."""
    _submenu_campos(cfg, "menu_valid", MENU_FIELDS["validacao"], "validacao")


def menu_avancado(cfg: Config) -> None:
    """Menu 6 — Metodos Avancados (benchmark, MC, SHAP)."""
    _submenu_campos(cfg, "menu_avancado", MENU_FIELDS["avancado"], "avancado")


def menu_paletas() -> None:
    """Submenu de selecao de paleta de cores para as figuras."""
    while True:
        lang = _lang()
        t = I18N[lang]
        cls()
        w = 68
        vcfg = _carregar_visual_cfg()
        paleta_atual = vcfg.get("paleta", "qualitativo")
        titulo = "Paletas de Cor" if lang == "PT" else "Color Palettes"
        print("╔" + "═" * (w - 2) + "╗")
        print("║" + _ansi_ljust(f"  {titulo}", w - 2) + "║")
        atual_label = f"  Paleta atual: {paleta_atual}" if lang == "PT" else f"  Current palette: {paleta_atual}"
        print("║" + _ansi_ljust(atual_label, w - 2) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        nomes_paleta = list(PALETAS_COR.keys())
        for i, key in enumerate(nomes_paleta, 1):
            pal = PALETAS_COR[key]
            pal_lang = pal.get(lang, pal.get("PT", {}))
            nome_p = pal_lang.get("nome", key)
            desc_p = pal_lang.get("desc", "")
            ativo = " ◄" if key == paleta_atual else ""
            print("║" + _ansi_ljust(f"  [{i}] {nome_p}{ativo}", w - 2) + "║")
            print("║" + _ansi_ljust(_c("DIM", f"      {desc_p}"), w - 2) + "║")
            cores = pal.get("cores")
            if cores:
                amostra = "      " + " ".join(f"[{c}]" for c in cores[:6])
            else:
                cmap_name = pal.get("cmap", "")
                amostra = f"      (paleta matplotlib: {cmap_name or 'default'})"
            print("║" + _ansi_ljust(amostra, w - 2) + "║")
            print("║" + _ansi_ljust("", w - 2) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        rodape = f"  [I] {t['idioma']}   [0] {t['voltar']}"
        print("║" + _ansi_ljust(rodape, w - 2) + "║")
        print("╚" + "═" * (w - 2) + "╝")
        print()
        escolha = _prompt(f"  {t['opcao']}: ")
        if not escolha or escolha == "0" or escolha.lower() == "q":
            break
        if escolha.upper() == "I":
            _toggle_idioma()
            continue
        if escolha.isdigit() and 1 <= int(escolha) <= len(nomes_paleta):
            key_sel = nomes_paleta[int(escolha) - 1]
            vcfg["paleta"] = key_sel
            _salvar_visual_cfg(vcfg)
            pal_lang = PALETAS_COR[key_sel].get(lang, PALETAS_COR[key_sel].get("PT", {}))
            nome_sel = pal_lang.get("nome", key_sel)
            if lang == "PT":
                print(f"  Paleta '{nome_sel}' selecionada e salva em visual_config.json.")
            else:
                print(f"  Palette '{nome_sel}' selected and saved to visual_config.json.")
            _prompt(f"  [{t['continuar']}]")
        else:
            print(f"  {t['invalido']}")
            _prompt(f"  [{t['continuar']}]")


def _gerar_heatmap_espectros(cfg: Config) -> None:
    """Gera heatmap de espectros colorido por classe — pre-visualizacao dos dados."""
    lang = _lang()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path as _Path
    except ImportError:
        print("  [ERRO] matplotlib ou numpy nao instalado." if lang == "PT" else "  [ERROR] matplotlib or numpy not installed.")
        return

    pasta = getattr(cfg, "pasta_dados", "dados")
    msg_lendo = "  Lendo arquivos DX..." if lang == "PT" else "  Reading DX files..."
    print(msg_lendo)

    spectra, labels, _ = _ler_dx_pasta(pasta, max_files=200, ler_x=False)
    if not spectra:
        print("  Nao foi possivel ler espectros." if lang == "PT" else "  Could not read spectra.")
        return

    # Truncar ao mesmo comprimento
    min_len = min(len(s) for s in spectra)
    arr = np.array([s[:min_len] for s in spectra], dtype=float)

    # Normalizar por linha (SNV simplificado)
    arr_norm = (arr - arr.mean(axis=1, keepdims=True)) / (arr.std(axis=1, keepdims=True) + 1e-8)

    # Ordenar por classe
    order = np.argsort(labels)
    arr_sorted = arr_norm[order]
    labels_sorted = [labels[i] for i in order]

    # Colormap por classe
    classes_unicas = sorted(set(labels_sorted))
    classe_pos: Dict[str, list] = {c: [] for c in classes_unicas}
    for i, lb in enumerate(labels_sorted):
        classe_pos[lb].append(i)

    # Aplicar visual_config
    vcfg = _carregar_visual_cfg()
    fonte_key = vcfg.get("tamanho_fonte", "m")
    fonte_preset = FONT_PRESETS.get(fonte_key, FONT_PRESETS["m"])
    _rc_extras = {**fonte_preset}
    if vcfg.get("grid_major", True):
        _rc_extras["axes.grid"] = True
        _rc_extras["grid.linestyle"] = vcfg.get("grid_style", "dotted")
        _rc_extras["grid.alpha"] = float(vcfg.get("grid_alpha", 0.4))

    pasta_saida = getattr(cfg, "pasta_saida", "resultados")
    saida_path = _Path(pasta_saida)
    saida_path.mkdir(parents=True, exist_ok=True)
    fname = saida_path / "heatmap_espectros.png"
    dpi = int(getattr(cfg, "dpi_figuras", vcfg.get("dpi", 150)))

    with plt.rc_context(_rc_extras):
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(spectra) * 0.12 + 2)),
                                  gridspec_kw={"width_ratios": [20, 1]},
                                  constrained_layout=True)
        ax, ax_cbar = axes

        im = ax.imshow(arr_sorted, aspect="auto", cmap="RdYlGn",
                       interpolation="nearest", vmin=-2, vmax=2)
        ax.set_xlabel("Variavel espectral (index)" if lang == "PT" else "Spectral variable (index)")
        ax.set_ylabel("Amostra" if lang == "PT" else "Sample")
        titulo_hm = "Heatmap de Espectros (SNV normalizado)" if lang == "PT" else "Spectral Heatmap (SNV normalized)"
        ax.set_title(titulo_hm)

        # Linhas divisorias entre classes
        tick_pos, tick_labels = [], []
        pos = 0
        for c in classes_unicas:
            n = len(classe_pos[c])
            ax.axhline(pos - 0.5, color="white", linewidth=1.5)
            tick_pos.append(pos + n // 2)
            tick_labels.append(c)
            pos += n

        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labels, fontsize=8)

        fig.colorbar(im, cax=ax_cbar, label="Intensidade norm." if lang == "PT" else "Norm. intensity")

        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    ok = f"  Heatmap salvo em: {fname}" if lang == "PT" else f"  Heatmap saved to: {fname}"
    print(ok)


def _gerar_confusion_matrix(pasta_saida: str) -> None:
    """Gera confusion matrix a partir dos resultados CSV do pipeline."""
    lang = _lang()
    from pathlib import Path as _Path

    pasta = _Path(pasta_saida)
    # Procura subpastas mais recentes
    subdirs = sorted(pasta.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if pasta.exists() else []
    csv_found = None
    for sd in subdirs[:5]:
        if sd.is_dir():
            csvs = list(sd.glob("*predicoes*")) + list(sd.glob("*cv_oof*")) + list(sd.glob("*scores*"))
            if csvs:
                csv_found = csvs[0]
                break
        elif sd.suffix == ".csv" and ("predicoes" in sd.name or "scores" in sd.name):
            csv_found = sd
            break

    if csv_found is None:
        msg = ("  Nenhum CSV de predicoes encontrado. Execute o pipeline primeiro."
               if lang == "PT" else
               "  No predictions CSV found. Run the pipeline first.")
        print(msg)
        return

    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        df = pd.read_csv(csv_found)

        # Detectar colunas y_true e y_pred
        col_true = next((c for c in df.columns if "true" in c.lower() or "real" in c.lower() or "classe" in c.lower()), None)
        col_pred = next((c for c in df.columns if "pred" in c.lower()), None)

        if col_true is None or col_pred is None:
            print(f"  CSV: {csv_found.name}")
            print(f"  Colunas: {list(df.columns)}")
            msg = ("  Colunas 'true'/'pred' nao encontradas no CSV."
                   if lang == "PT" else
                   "  Could not find 'true'/'pred' columns in CSV.")
            print(msg)
            return

        y_true = df[col_true].values
        y_pred = df[col_pred].values
        classes = sorted(set(y_true) | set(y_pred))

        # Matriz
        n = len(classes)
        cm = np.zeros((n, n), dtype=int)
        class_idx = {c: i for i, c in enumerate(classes)}
        for yt, yp in zip(y_true, y_pred):
            if yt in class_idx and yp in class_idx:
                cm[class_idx[yt]][class_idx[yp]] += 1

        # Normalizar por linha (recall)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0)

        vcfg = _carregar_visual_cfg()
        fonte_key = vcfg.get("tamanho_fonte", "m")
        _rc_extras_cm = {**FONT_PRESETS.get(fonte_key, FONT_PRESETS["m"])}
        if vcfg.get("grid_major", True):
            _rc_extras_cm["axes.grid"] = True
            _rc_extras_cm["grid.linestyle"] = vcfg.get("grid_style", "dotted")
            _rc_extras_cm["grid.alpha"] = float(vcfg.get("grid_alpha", 0.4))

        saida = _Path(csv_found).parent / "confusion_matrix.png"
        dpi = int(vcfg.get("dpi", 150))

        with plt.rc_context(_rc_extras_cm):
            fig_size = max(6, n * 0.8 + 2)
            fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2.2, fig_size), constrained_layout=True)

            for ax, mat, titulo in zip(axes,
                                        [cm, cm_norm],
                                        ["Confusion Matrix (N amostras)" if lang == "PT" else "Confusion Matrix (N samples)",
                                         "Confusion Matrix (Recall %" + (" por classe)" if lang == "PT" else " per class)")]):
                im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
                ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=max(6, 9 - n // 4))
                ax.set_yticklabels(classes, fontsize=max(6, 9 - n // 4))
                ax.set_xlabel("Predito" if lang == "PT" else "Predicted")
                ax.set_ylabel("Real" if lang == "PT" else "Actual")
                ax.set_title(titulo)
                is_norm = (mat is cm_norm)
                def fmt_val(v, _is_norm=is_norm): return f"{v:.0%}" if _is_norm else f"{int(v)}"
                for i in range(n):
                    for j in range(n):
                        val = mat[i, j]
                        txt = fmt_val(val)
                        color = "white" if val > mat.max() * 0.6 else "black"
                        ax.text(j, i, txt, ha="center", va="center", color=color,
                                fontsize=max(5, 8 - n // 4))
                fig.colorbar(im, ax=ax, shrink=0.8)

            fig.savefig(saida, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        ok = f"  Confusion Matrix salva em: {saida}" if lang == "PT" else f"  Confusion Matrix saved to: {saida}"
        print(ok)

    except Exception as e:
        print(f"  [ERRO] {e}" if lang == "PT" else f"  [ERROR] {e}")


def _gerar_pca_biplot(cfg: Config) -> None:
    """Gera PCA Biplot 2D com elipse de confianca 95% (Hotelling T2) por classe."""
    lang = _lang()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        from pathlib import Path as _Path
        from sklearn.decomposition import PCA
    except ImportError as e:
        print(f"  [ERRO] Dependencia ausente: {e}")
        return

    pasta = getattr(cfg, "pasta_dados", "dados")
    msg = ("  Lendo arquivos para PCA Biplot..."
           if lang == "PT" else
           "  Reading files for PCA Biplot...")
    print(msg)

    spectra, labels, _ = _ler_dx_pasta(pasta, max_files=300, ler_x=False)
    if len(spectra) < 3:
        print("  Espectros insuficientes para PCA." if lang == "PT" else "  Insufficient spectra for PCA.")
        return

    min_len = min(len(s) for s in spectra)
    X = np.array([s[:min_len] for s in spectra], dtype=float)

    # SNV preprocessing
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    # PCA
    n_pcs = min(10, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_pcs)
    T = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    vcfg = _carregar_visual_cfg()
    paleta_key = vcfg.get("paleta", "qualitativo")
    paleta = PALETAS_COR.get(paleta_key, PALETAS_COR["qualitativo"])
    cores_p = paleta.get("cores")
    fonte_key = vcfg.get("tamanho_fonte", "m")
    _rc_extras_bp = {**FONT_PRESETS.get(fonte_key, FONT_PRESETS["m"])}
    if vcfg.get("grid_major", True):
        _rc_extras_bp["axes.grid"] = True
        _rc_extras_bp["grid.linestyle"] = vcfg.get("grid_style", "dotted")
        _rc_extras_bp["grid.alpha"] = float(vcfg.get("grid_alpha", 0.4))
    alpha_map_b = {"baixo": 0.9, "medio": 0.65, "alto": 0.35}
    alpha_v = alpha_map_b.get(vcfg.get("alpha_pontos", "medio"), 0.65)

    classes_u = sorted(set(labels))
    if cores_p:
        cmap_list = [cores_p[i % len(cores_p)] for i in range(len(classes_u))]
    else:
        cmap_tab = plt.get_cmap("tab20")
        cmap_list = [cmap_tab(i / max(1, len(classes_u) - 1)) for i in range(len(classes_u))]
    cor_por_classe = dict(zip(classes_u, cmap_list))

    pasta_saida = getattr(cfg, "pasta_saida", "resultados")
    saida = _Path(pasta_saida) / "pca_biplot_elipse.png"
    _Path(pasta_saida).mkdir(parents=True, exist_ok=True)
    dpi = int(getattr(cfg, "dpi_figuras", 150))

    with plt.rc_context(_rc_extras_bp):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

        for ax_idx, (pc_x, pc_y) in enumerate([(0, 1), (0, 2)]):
            if pc_y >= n_pcs:
                axes[ax_idx].set_visible(False)
                continue
            ax = axes[ax_idx]

            # Elipses de confianca 95% por classe (Hotelling T2)
            for classe in classes_u:
                idx_c = [i for i, lb in enumerate(labels) if lb == classe]
                if len(idx_c) < 3:
                    continue
                Tc = T[idx_c][:, [pc_x, pc_y]]
                mean_c = Tc.mean(axis=0)
                cov_c = np.cov(Tc.T)
                try:
                    vals, vecs = np.linalg.eigh(cov_c)
                    vals = np.maximum(vals, 0)
                    chi2_95 = 5.991  # chi2 95%, 2 graus de liberdade
                    w_e, h_e = 2 * np.sqrt(vals * chi2_95)
                    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
                    from matplotlib.patches import Ellipse
                    ell = Ellipse(xy=mean_c, width=w_e, height=h_e, angle=angle,
                                  facecolor=cor_por_classe[classe], alpha=0.12,
                                  edgecolor=cor_por_classe[classe], linewidth=1.5,
                                  linestyle="--")
                    ax.add_patch(ell)
                except Exception:
                    pass

                # Scatter
                ax.scatter(Tc[:, 0], Tc[:, 1],
                           c=[cor_por_classe[classe]] * len(Tc),
                           alpha=alpha_v, s=35, label=classe, edgecolors="none")
                ax.scatter(*mean_c, c=[cor_por_classe[classe]], s=120,
                           marker="*", edgecolors="black", linewidths=0.5, zorder=5)

            # Loadings (setas) — top 8 variaveis mais importantes
            P = pca.components_[[pc_x, pc_y]].T  # (n_vars, 2)
            loading_mag = np.sqrt(P[:, 0]**2 + P[:, 1]**2)
            top_idx = np.argsort(loading_mag)[-8:]
            scale = (np.abs(T[:, [pc_x, pc_y]]).max() * 0.8) / (loading_mag[top_idx].max() + 1e-8)
            for vi in top_idx:
                ax.annotate("", xy=(P[vi, 0] * scale, P[vi, 1] * scale),
                            xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2, alpha=0.5))
                ax.text(P[vi, 0] * scale * 1.08, P[vi, 1] * scale * 1.08,
                        f"v{vi}", fontsize=6, color="gray", alpha=0.7)

            xlabel = f"PC{pc_x+1} ({var[pc_x]:.1f}%)"
            ylabel = f"PC{pc_y+1} ({var[pc_y]:.1f}%)"
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
            ax.axvline(0, color="gray", lw=0.5, alpha=0.4)
            titulo_ax = (f"PCA Biplot — PC{pc_x+1} vs PC{pc_y+1} (Elipse 95% Hotelling T2)"
                         if lang == "PT" else
                         f"PCA Biplot — PC{pc_x+1} vs PC{pc_y+1} (95% Hotelling T2 Ellipse)")
            ax.set_title(titulo_ax)
            if ax_idx == 0:
                handles = [mpatches.Patch(color=cor_por_classe[c], label=c) for c in classes_u]
                ax.legend(handles=handles, fontsize=7, ncol=max(1, len(classes_u)//8 + 1),
                          loc="best", framealpha=0.7)
            if vcfg.get("grid_major", True):
                ax.grid(True, linestyle=vcfg.get("grid_style", "dotted"),
                        alpha=float(vcfg.get("grid_alpha", 0.4)))

        fig.savefig(saida, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    ok = (f"  PCA Biplot salvo em: {saida}"
          if lang == "PT" else f"  PCA Biplot saved to: {saida}")
    print(ok)


def _gerar_variancia_wavelength(cfg: Config) -> None:
    """Gera painel: variancia acumulada PCA (esq) + Wavelength Importance (dir)."""
    lang = _lang()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path as _Path
        from sklearn.decomposition import PCA
    except ImportError as e:
        print(f"  [ERRO] {e}")
        return

    pasta = getattr(cfg, "pasta_dados", "dados")
    msg = ("  Calculando variancia acumulada + importancia espectral..."
           if lang == "PT" else
           "  Computing cumulative variance + spectral importance...")
    print(msg)

    spectra, labels_vw, wavenumbers = _ler_dx_pasta(pasta, max_files=300, ler_x=True)
    if len(spectra) < 3:
        print("  Espectros insuficientes." if lang == "PT" else "  Insufficient spectra.")
        return

    min_len = min(len(s) for s in spectra)
    X = np.array([s[:min_len] for s in spectra], dtype=float)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    n_pcs = min(20, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_pcs)
    pca.fit(X)
    var_ratio = pca.explained_variance_ratio_ * 100
    var_cum = np.cumsum(var_ratio)

    # Wavelength Importance = soma ponderada dos loadings absolutos pelos autovalores
    importance = np.sum(np.abs(pca.components_) * var_ratio[:, None], axis=0)
    importance = importance / importance.max()

    wn = np.array(wavenumbers[:min_len]) if wavenumbers else np.arange(min_len)

    vcfg = _carregar_visual_cfg()
    paleta_key = vcfg.get("paleta", "qualitativo")
    paleta = PALETAS_COR.get(paleta_key, PALETAS_COR["qualitativo"])
    cores_p = paleta.get("cores", None)
    cor_bar = cores_p[0] if cores_p else "#1f77b4"
    cor_line = cores_p[1] if (cores_p and len(cores_p) > 1) else "#ff7f0e"
    cor_wl = cores_p[2] if (cores_p and len(cores_p) > 2) else "#2ca02c"

    fonte_key = vcfg.get("tamanho_fonte", "m")
    _rc_extras_vw = {**FONT_PRESETS.get(fonte_key, FONT_PRESETS["m"])}
    if vcfg.get("grid_major", True):
        _rc_extras_vw["axes.grid"] = True
        _rc_extras_vw["grid.linestyle"] = vcfg.get("grid_style", "dotted")
        _rc_extras_vw["grid.alpha"] = float(vcfg.get("grid_alpha", 0.4))

    pasta_saida = getattr(cfg, "pasta_saida", "resultados")
    _Path(pasta_saida).mkdir(parents=True, exist_ok=True)
    saida = _Path(pasta_saida) / "variancia_wavelength_importance.png"
    dpi = int(getattr(cfg, "dpi_figuras", 150))

    with plt.rc_context(_rc_extras_vw):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

        # Esquerdo: variancia por componente + acumulada
        pcs = np.arange(1, n_pcs + 1)
        ax1.bar(pcs, var_ratio, color=cor_bar, alpha=0.75,
                label="Variancia por PC" if lang == "PT" else "Variance per PC")
        ax1_r = ax1.twinx()
        ax1_r.plot(pcs, var_cum, color=cor_line, marker="o", ms=5, lw=2,
                   label="Acumulada" if lang == "PT" else "Cumulative")
        ax1_r.axhline(95, color="red", lw=1.2, linestyle="--", alpha=0.6)
        ax1_r.text(n_pcs * 0.98, 95.5, "95%", color="red", fontsize=8, ha="right")
        pc_95 = int(np.searchsorted(var_cum, 95)) + 1
        if pc_95 <= n_pcs:
            ax1.axvline(pc_95, color="red", lw=1.2, linestyle=":", alpha=0.6)
            ax1.text(pc_95 + 0.1, var_ratio.max() * 0.9,
                     f"PC{pc_95}", color="red", fontsize=8)

        ax1.set_xlabel("Componente Principal" if lang == "PT" else "Principal Component")
        ax1.set_ylabel("Variancia Explicada (%)" if lang == "PT" else "Explained Variance (%)")
        ax1_r.set_ylabel("Variancia Acumulada (%)" if lang == "PT" else "Cumulative Variance (%)")
        ax1_r.set_ylim(0, 105)
        ax1.set_title("Variancia Acumulada — PCA" if lang == "PT" else "Cumulative Variance — PCA")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_r.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
        if vcfg.get("grid_major", True):
            ax1.grid(True, linestyle=vcfg.get("grid_style", "dotted"),
                     alpha=float(vcfg.get("grid_alpha", 0.4)))

        # Direito: Wavelength Importance
        ax2.fill_between(wn, importance, alpha=0.35, color=cor_wl)
        ax2.plot(wn, importance, color=cor_wl, lw=1.2)

        top10_idx = np.argsort(importance)[-10:]
        ax2.scatter(wn[top10_idx], importance[top10_idx],
                    color="red", s=30, zorder=5, alpha=0.8,
                    label="Top 10" if lang == "PT" else "Top 10")

        ax2.set_xlabel("Numero de Onda (cm-1)" if lang == "PT" else "Wavenumber (cm-1)")
        ax2.set_ylabel("Importancia Relativa" if lang == "PT" else "Relative Importance")
        ax2.set_title("Importancia Espectral (Loadings PCA ponderados)"
                      if lang == "PT" else "Spectral Importance (Weighted PCA Loadings)")
        ax2.legend(fontsize=8)
        if vcfg.get("grid_major", True):
            ax2.grid(True, linestyle=vcfg.get("grid_style", "dotted"),
                     alpha=float(vcfg.get("grid_alpha", 0.4)))

        fig.savefig(saida, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    ok = (f"  Grafico salvo em: {saida}"
          if lang == "PT" else f"  Figure saved to: {saida}")
    print(ok)


def menu_visualizacao(cfg: Config) -> None:
    """Menu 7 — Visualizacao (campos VISUAL + paletas + fonte + grid + heatmap + CM)."""
    while True:
        lang = _lang()
        t = I18N[lang]
        cls()
        print_header(cfg)
        w = 68
        campos = MENU_FIELDS["visualizacao"]
        titulo = t["menu_viz"]
        print("╔" + "═" * (w - 2) + "╗")
        print("║" + _ansi_ljust(f"  {titulo}", w - 2) + "║")
        desc_sec = SECTION_DESC.get("visualizacao", {}).get(lang, "")
        if desc_sec:
            for dline in _wrap_box(desc_sec, w - 4, "  "):
                print("║" + _ansi_ljust(_c("DIM", dline), w - 2) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        for i, key in enumerate(campos, 1):
            val_raw = _fmt_yaml(_get_val(cfg, key))
            val_str = (val_raw[:20] + "…") if len(str(val_raw)) > 22 else val_raw
            nome = _nome_campo(key)
            risk = RISK_CLASS.get(key, "VISUAL")
            cor = RISK_COLOR.get(risk, "")
            rst = RISK_COLOR["RESET"]
            lbl = f"{cor}●{rst}"
            linha = f"  [{i:2d}] {lbl} {nome:<28s}: {val_str}"
            print("║" + _ansi_ljust(linha, w - 2) + "║")
        # Item extra: Paletas de Cor
        vcfg = _carregar_visual_cfg()
        paleta_ativa = vcfg.get("paleta", "qualitativo")
        pal_data = PALETAS_COR.get(paleta_ativa, PALETAS_COR["qualitativo"])
        pal_nome = pal_data.get(lang, pal_data.get("PT", {})).get("nome", paleta_ativa)
        paleta_label = "Paleta de Cores" if lang == "PT" else "Color Palette"
        lbl_v = f"{RISK_COLOR['VISUAL']}●{RISK_COLOR['RESET']}"
        linha_p = f"  [ P] {lbl_v} {paleta_label:<28s}: {pal_nome}"
        print("║" + _ansi_ljust(linha_p, w - 2) + "║")
        # Exibir configuracoes de fonte, grid e alpha salvas
        fonte_key = vcfg.get("tamanho_fonte", "m")
        grid_major = vcfg.get("grid_major", True)
        grid_style = vcfg.get("grid_style", "dotted")
        alpha_key_disp = vcfg.get("alpha_pontos", "medio")
        fonte_label = "Tamanho de Fonte" if lang == "PT" else "Font Size"
        grid_label = "Grid" if lang == "PT" else "Grid"
        alpha_label = "Transparencia Pontos" if lang == "PT" else "Point Transparency"
        grid_status = ("ON" if grid_major else "OFF") + f" ({grid_style})"
        linha_f = f"  [ F] {lbl_v} {fonte_label:<28s}: {fonte_key.upper()}"
        linha_g = f"  [ G] {lbl_v} {grid_label:<28s}: {grid_status}"
        linha_a = f"  [ A] {lbl_v} {alpha_label:<28s}: {alpha_key_disp}"
        print("║" + _ansi_ljust(linha_f, w - 2) + "║")
        print("║" + _ansi_ljust(linha_g, w - 2) + "║")
        print("║" + _ansi_ljust(linha_a, w - 2) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        # Rodape com 2 colunas
        if lang == "PT":
            rodape1 = "  [F] Tamanho Fonte    [G] Grid"
            rodape2 = "  [A] Transparencia    [P] Paletas"
            rodape3 = "  [H] Heatmap          [M] Confusion Matrix"
            rodape4 = "  [B] PCA Biplot        [V] Variancia + Wavelength"
            rodape5 = "  [I] Idioma            [0] Voltar"
        else:
            rodape1 = "  [F] Font Size        [G] Grid"
            rodape2 = "  [A] Transparency     [P] Palettes"
            rodape3 = "  [H] Heatmap          [M] Confusion Matrix"
            rodape4 = "  [B] PCA Biplot        [V] Variance + Wavelength"
            rodape5 = "  [I] Language          [0] Back"
        for rline in [rodape1, rodape2, rodape3, rodape4, rodape5]:
            print("║" + _ansi_ljust(rline, w - 2) + "║")
        print("╚" + "═" * (w - 2) + "╝")
        print()
        escolha = _prompt(f"  {t['opcao']}: ")
        if not escolha or escolha == "0" or escolha.lower() == "q":
            break
        if escolha.upper() == "I":
            _toggle_idioma()
            continue
        if escolha.upper() == "P":
            menu_paletas()
            continue

        # Handler [F] — Tamanho de Fonte
        if escolha.upper() == "F":
            cls()
            w2 = 68
            opcoes_fonte = [
                ("XS", "xs", "Muito pequeno — max. DPI, figuras compactas" if lang == "PT" else "Very small — max DPI, compact figures"),
                ("S",  "s",  "Pequeno — relatorios condensados" if lang == "PT" else "Small — condensed reports"),
                ("M",  "m",  "Medio (padrao) — equilibrado" if lang == "PT" else "Medium (default) — balanced"),
                ("L",  "l",  "Grande — apresentacoes, posters" if lang == "PT" else "Large — presentations, posters"),
                ("XL", "xl", "Muito grande — conferencia, sala" if lang == "PT" else "Very large — conference, room"),
            ]
            vcfg2 = _carregar_visual_cfg()
            atual_fonte = vcfg2.get("tamanho_fonte", "m")
            titulo_f = "Tamanho de Fonte das Figuras" if lang == "PT" else "Figure Font Size"
            print("╔" + "═" * (w2 - 2) + "╗")
            print("║" + _ansi_ljust(f"  {titulo_f}", w2 - 2) + "║")
            print("╠" + "═" * (w2 - 2) + "╣")
            for idx_f, (label, key_f, desc_f) in enumerate(opcoes_fonte, 1):
                marcador = "►" if key_f == atual_fonte else " "
                linha_ff = f"  [{idx_f}] {marcador} {label:<4} — {desc_f}"
                print("║" + _ansi_ljust(linha_ff, w2 - 2) + "║")
            print("╠" + "═" * (w2 - 2) + "╣")
            print("║" + _ansi_ljust(f"  [0] {t['voltar']}", w2 - 2) + "║")
            print("╚" + "═" * (w2 - 2) + "╝")
            ef = _prompt(f"\n  {t.get('opcao','Option')}: ")
            if ef.isdigit() and 1 <= int(ef) <= len(opcoes_fonte):
                nova_fonte = opcoes_fonte[int(ef) - 1][1]
                vcfg2["tamanho_fonte"] = nova_fonte
                _salvar_visual_cfg(vcfg2)
                ok_f = (f"  Fonte '{opcoes_fonte[int(ef)-1][0]}' salva."
                        if lang == "PT" else
                        f"  Font '{opcoes_fonte[int(ef)-1][0]}' saved.")
                print(ok_f)
            _prompt(f"  [{t['continuar']}]")
            continue

        # Handler [G] — Grid
        if escolha.upper() == "G":
            vcfg3 = _carregar_visual_cfg()
            cls()
            w3 = 68
            titulo_g = "Configuracao de Grid" if lang == "PT" else "Grid Configuration"
            estilos = ["solid", "dotted", "dashed"]
            est_atual = vcfg3.get("grid_style", "dotted")
            print("╔" + "═" * (w3 - 2) + "╗")
            print("║" + _ansi_ljust(f"  {titulo_g}", w3 - 2) + "║")
            print("╠" + "═" * (w3 - 2) + "╣")
            opcoes_grid = [
                ("1", "grid_major", "Grid principal (major)", "Major grid"),
                ("2", "grid_minor", "Grid secundario (minor)", "Minor grid"),
            ]
            for cod_g, key_vc, lp, le in opcoes_grid:
                val_g = vcfg3.get(key_vc, True)
                status_g = "ON ✓" if val_g else "OFF"
                label_g = lp if lang == "PT" else le
                print("║" + _ansi_ljust(f"  [{cod_g}] {label_g}: {status_g}", w3 - 2) + "║")
            print("║" + _ansi_ljust(f"  [3] {'Estilo' if lang=='PT' else 'Style'}: {est_atual}  (solid/dotted/dashed)", w3 - 2) + "║")
            alpha_atual = vcfg3.get("grid_alpha", 0.4)
            print("║" + _ansi_ljust(f"  [4] {'Transparencia grid' if lang=='PT' else 'Grid alpha'}: {alpha_atual:.1f}  (0.1 - 0.9)", w3 - 2) + "║")
            print("╠" + "═" * (w3 - 2) + "╣")
            print("║" + _ansi_ljust(f"  [0] {t['voltar']}", w3 - 2) + "║")
            print("╚" + "═" * (w3 - 2) + "╝")
            eg = _prompt(f"\n  {t.get('opcao','Option')}: ")
            if eg == "1":
                vcfg3["grid_major"] = not vcfg3.get("grid_major", True)
            elif eg == "2":
                vcfg3["grid_minor"] = not vcfg3.get("grid_minor", False)
            elif eg == "3":
                idx_e = (estilos.index(est_atual) + 1) % len(estilos)
                vcfg3["grid_style"] = estilos[idx_e]
            elif eg == "4":
                try:
                    novo_a = float(input("  " + ("Novo valor (0.1-0.9): " if lang == "PT" else "New value (0.1-0.9): ")))
                    vcfg3["grid_alpha"] = max(0.1, min(0.9, novo_a))
                except (ValueError, EOFError):
                    pass
            _salvar_visual_cfg(vcfg3)
            _prompt(f"  [{t['continuar']}]")
            continue

        # Handler [H] — Heatmap de Espectros
        if escolha.upper() == "H":
            print()
            _gerar_heatmap_espectros(cfg)
            _prompt(f"  [{t['continuar']}]")
            continue

        # Handler [M] — Confusion Matrix
        if escolha.upper() == "M":
            print()
            pasta_s = getattr(cfg, "pasta_saida", "resultados")
            _gerar_confusion_matrix(pasta_s)
            try:
                input(f"  [{t['continuar']}]")
            except (EOFError, KeyboardInterrupt):
                pass
            continue

        # Handler [A] — Transparencia dos pontos
        if escolha.upper() == "A":
            vcfg_a = _carregar_visual_cfg()
            cls()
            w_a = 68
            titulo_a = "Transparencia dos Pontos" if lang == "PT" else "Point Transparency"
            print("╔" + "═" * (w_a - 2) + "╗")
            print("║" + _ansi_ljust(f"  {titulo_a}", w_a - 2) + "║")
            print("╠" + "═" * (w_a - 2) + "╣")
            opcoes_alpha = [
                ("1", "baixo",  "0.9 — Pontos opacos (poucos dados, sem sobreposicao)",
                                 "0.9 — Opaque points (few data, no overlap)"),
                ("2", "medio",  "0.65 — Equilibrado (padrao recomendado)",
                                 "0.65 — Balanced (recommended default)"),
                ("3", "alto",   "0.35 — Translucido (muitos dados, sobrepostos)",
                                 "0.35 — Translucent (many overlapping data points)"),
            ]
            atual_a = vcfg_a.get("alpha_pontos", "medio")
            for cod, key_a, desc_pt, desc_en in opcoes_alpha:
                marcador = "►" if key_a == atual_a else " "
                desc = desc_pt if lang == "PT" else desc_en
                linha = f"  [{cod}] {marcador} {desc}"
                print("║" + _ansi_ljust(linha, w_a - 2) + "║")
            print("╠" + "═" * (w_a - 2) + "╣")
            print("║" + _ansi_ljust(f"  [0] {t['voltar']}", w_a - 2) + "║")
            print("╚" + "═" * (w_a - 2) + "╝")
            ea = _prompt(f"\n  {t.get('opcao', 'Option')}: ")
            if ea in ("1", "2", "3"):
                nova_a = opcoes_alpha[int(ea) - 1][1]
                vcfg_a["alpha_pontos"] = nova_a
                _salvar_visual_cfg(vcfg_a)
                ok_a = (f"  Transparencia '{nova_a}' salva."
                        if lang == "PT" else f"  Transparency '{nova_a}' saved.")
                print(ok_a)
            _prompt(f"  [{t['continuar']}]")
            continue

        # Handler [B] — PCA Biplot com elipse de confianca 95%
        if escolha.upper() == "B":
            print()
            _gerar_pca_biplot(cfg)
            _prompt(f"  [{t['continuar']}]")
            continue

        # Handler [V] — Variancia acumulada + Wavelength Importance
        if escolha.upper() == "V":
            print()
            _gerar_variancia_wavelength(cfg)
            _prompt(f"  [{t['continuar']}]")
            continue

        if escolha.lower().startswith("?") or escolha.lower().startswith("help"):
            partes = escolha.split(maxsplit=1)
            campo_help = partes[1] if len(partes) > 1 else ""
            if campo_help:
                _mostrar_help_campo(campo_help.strip())
            else:
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
# Menu de Ajuda
# ===========================================================================

def menu_ajuda() -> None:
    """Menu 8 — Sistema de ajuda. Suporta 'help <topico>' ou '?' para listar."""
    while True:
        lang = _lang()
        t = I18N[lang]
        cls()
        w = 68
        print("╔" + "═" * (w - 2) + "╗")
        print("║" + _ansi_ljust(f"  {t['menu_ajuda']}", w - 2) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        if lang == "PT":
            print("║" + _ansi_ljust("  Digite o nome do parametro para ver detalhes.", w - 2) + "║")
            print("║" + _ansi_ljust("  Exemplos: dpi  |  benchmark  |  pre_processamento", w - 2) + "║")
        else:
            print("║" + _ansi_ljust("  Type the parameter name to see details.", w - 2) + "║")
            print("║" + _ansi_ljust("  Examples: dpi  |  benchmark  |  pre_processamento", w - 2) + "║")
        rodape = f"  [L] {t['listar_todos']}   [I] {t['idioma']}   [0] {t['voltar']}"
        print("║" + _ansi_ljust(rodape, w - 2) + "║")
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
            escolha_l = _prompt(f"  {t['opcao']}: ")
            if escolha_l:
                if escolha_l.isdigit() and 1 <= int(escolha_l) <= len(keys_sorted):
                    _mostrar_help_campo(keys_sorted[int(escolha_l) - 1])
                    _prompt(f"  [{t['continuar']}]")
                else:
                    topico_l = escolha_l.replace("help", "").replace("?", "").strip()
                    if topico_l:
                        _mostrar_help_campo(topico_l)
                        _prompt(f"  [{t['continuar']}]")
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
# Menu de Hardware
# ===========================================================================

def menu_hardware() -> None:
    """Menu H — Verificacao de hardware e recomendacoes de perfil."""
    lang = _lang()
    cls()
    w = 68
    titulo_hw = "Verificacao de Hardware" if lang == "PT" else "Hardware Check"

    # Chamar hardware_probe() do pipeline
    try:
        hw = pq.hardware_probe()
        ram_gb    = hw.get("ram_total_gb", 0)
        ram_disp  = hw.get("ram_livre_gb", 0)
        cpu_log   = hw.get("cpu_logicos", 1)
        cpu_fis   = hw.get("cpu_fisicos", 1)
        disco_gb  = hw.get("disco_livre_gb", 0)
        psutil_ok = hw.get("psutil_ok", False)
    except Exception:
        hw = {}
        ram_gb    = 0
        ram_disp  = 0
        cpu_log   = 1
        cpu_fis   = 1
        disco_gb  = 0
        psutil_ok = False

    print("╔" + "═" * (w - 2) + "╗")
    print("║" + _ansi_ljust(f"  {titulo_hw}", w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")

    # Mostrar specs
    if lang == "PT":
        psutil_str = "Instalado" if psutil_ok else "Nao instalado (estimativa conservadora)"
    else:
        psutil_str = "Installed" if psutil_ok else "Not installed (conservative estimate)"
    ram_line   = f"  RAM total    : {ram_gb:.1f} GB"
    disp_line  = f"  RAM livre    : {ram_disp:.1f} GB"
    cpu_fis_ln = f"  CPU fisicos  : {cpu_fis}"
    cpu_log_ln = f"  CPU logicos  : {cpu_log}"
    disco_line = f"  Disco livre  : {disco_gb:.1f} GB"
    psutil_ln  = f"  psutil       : {psutil_str}"
    for ln in [ram_line, disp_line, cpu_fis_ln, cpu_log_ln, disco_line, psutil_ln]:
        print("║" + _ansi_ljust(ln, w - 2) + "║")

    print("╠" + "═" * (w - 2) + "╣")

    # Tier e recomendacao baseados na RAM
    if ram_gb >= 16:
        tier = "ALTO" if lang == "PT" else "HIGH"
        rec_perfil = "Alta Rigorosidade" if lang == "PT" else "Alta Rigorosidade (High Rigor)"
        cor_tier = RISK_COLOR["VISUAL"]  # verde
        limitacoes = (["Sem limitacoes significativas.", "Todos os perfis sao compativeis."]
                      if lang == "PT" else
                      ["No significant limitations.", "All profiles are compatible."])
    elif ram_gb >= 8:
        tier = "MEDIO" if lang == "PT" else "MEDIUM"
        rec_perfil = "Publicacao Cientifica"
        cor_tier = RISK_COLOR["ANALITICO"]  # amarelo
        limitacoes = (["SHAP com datasets grandes pode ser lento (>500 amostras).",
                       "Monte Carlo CV com N>100 pode demorar >2h.",
                       "Recomendado: shap_max_amostras <= 300."]
                      if lang == "PT" else
                      ["SHAP with large datasets may be slow (>500 samples).",
                       "Monte Carlo CV with N>100 may take >2h.",
                       "Recommended: shap_max_amostras <= 300."])
    elif ram_gb >= 4:
        tier = "BASICO" if lang == "PT" else "BASIC"
        rec_perfil = "Pesquisa Academica"
        cor_tier = RISK_COLOR["AVANCADO"]  # vermelho
        limitacoes = (["Benchmark pode causar lentidao (SVM em datasets grandes).",
                       "SHAP DESATIVADO recomendado.",
                       "Monte Carlo CV DESATIVADO recomendado.",
                       "Recomendado: shap_max_amostras <= 200, max_lvs <= 30."]
                      if lang == "PT" else
                      ["Benchmark may be slow (SVM on large datasets).",
                       "SHAP DISABLED recommended.",
                       "Monte Carlo CV DISABLED recommended.",
                       "Recommended: shap_max_amostras <= 200, max_lvs <= 30."])
    else:
        tier = "LIMITADO" if lang == "PT" else "LIMITED"
        rec_perfil = "Exploracao Rapida"
        cor_tier = RISK_COLOR["AVANCADO"]
        limitacoes = (["RAM insuficiente para analises completas.",
                       "Use apenas: Exploracao Rapida ou Analise Padrao sem benchmark/SHAP.",
                       "Desative: Benchmark, SHAP, Monte Carlo, OPLS-DA.",
                       "Reduza max_lvs para 20 e shap_max_amostras para 100."]
                      if lang == "PT" else
                      ["Insufficient RAM for full analyses.",
                       "Use only: Exploracao Rapida or Analise Padrao without benchmark/SHAP.",
                       "Disable: Benchmark, SHAP, Monte Carlo, OPLS-DA.",
                       "Reduce max_lvs to 20 and shap_max_amostras to 100."])

    tier_label = f"  Capacidade: {cor_tier}{tier}{RISK_COLOR['RESET']}"
    rec_label  = (f"  Perfil recomendado: {rec_perfil}" if lang == "PT"
                  else f"  Recommended profile: {rec_perfil}")
    print("║" + _ansi_ljust(tier_label, w - 2) + "║")
    print("║" + _ansi_ljust(rec_label, w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")
    lim_titulo = "  Limitacoes e Recomendacoes:" if lang == "PT" else "  Limitations and Recommendations:"
    print("║" + _ansi_ljust(lim_titulo, w - 2) + "║")
    for lim in limitacoes:
        print("║" + _ansi_ljust(f"    • {lim}", w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")
    voltar_txt = "  [0] Voltar" if lang == "PT" else "  [0] Back"
    print("║" + _ansi_ljust(voltar_txt, w - 2) + "║")
    print("╚" + "═" * (w - 2) + "╝")
    print()
    _prompt(f"  {I18N[lang].get('opcao', 'Option')}: ")


# ===========================================================================
# Tecnicas analiticas — dict de defaults por tecnica
# ===========================================================================

TECNICAS: Dict[str, Dict[str, Any]] = {
    "ft-nir": {
        "PT": {
            "nome": "FT-NIR (Infravermelho Proximo por Transformada de Fourier)",
            "desc": "Espectroscopia NIR de alta resolucao. Ideal para oleos, alimentos, farmacos.",
            "preproc_rec": "MSC+SG+MC",
            "faixa": "4000-10000 cm-1 (tipico) ou 4000-12000 cm-1",
        },
        "EN": {
            "nome": "FT-NIR (Fourier Transform Near Infrared Spectroscopy)",
            "desc": "High-resolution NIR spectroscopy. Ideal for oils, food, pharmaceuticals.",
            "preproc_rec": "MSC+SG+MC",
            "faixa": "4000-10000 cm-1 (typical) or 4000-12000 cm-1",
        },
        "faixa_min": 4000.0, "faixa_max": 10000.0,
        "preproc": "msc_sg_mc", "modo": "dx",
    },
    "nir": {
        "PT": {
            "nome": "NIR Dispersivo (Infravermelho Proximo)",
            "desc": "NIR convencional com detector dispersivo. Faixa tipica 700-2500 nm.",
            "preproc_rec": "SNV+SG+MC ou MSC+SG+MC",
            "faixa": "4000-14000 cm-1 (700-2500 nm)",
        },
        "EN": {
            "nome": "Dispersive NIR (Near Infrared Spectroscopy)",
            "desc": "Conventional NIR with dispersive detector. Typical range 700-2500 nm.",
            "preproc_rec": "SNV+SG+MC or MSC+SG+MC",
            "faixa": "4000-14000 cm-1 (700-2500 nm)",
        },
        "faixa_min": 4000.0, "faixa_max": 14000.0,
        "preproc": "snv_sg_mc", "modo": "dx",
    },
    "mir": {
        "PT": {
            "nome": "MIR/FTIR (Infravermelho Medio)",
            "desc": "Infravermelho medio (4000-400 cm-1). Bandas fundamentais de absorcao molecular.",
            "preproc_rec": "SNV+MC ou SG+MC (MSC menos comum em MIR)",
            "faixa": "400-4000 cm-1",
        },
        "EN": {
            "nome": "MIR/FTIR (Mid-Infrared Spectroscopy)",
            "desc": "Mid-infrared (4000-400 cm-1). Fundamental molecular absorption bands.",
            "preproc_rec": "SNV+MC or SG+MC (MSC less common in MIR)",
            "faixa": "400-4000 cm-1",
        },
        "faixa_min": 400.0, "faixa_max": 4000.0,
        "preproc": "snv_sg_mc", "modo": "dx",
    },
    "raman": {
        "PT": {
            "nome": "Raman (Espectroscopia Raman)",
            "desc": "Espectroscopia vibracional por espalhamento Raman. Complementar ao IR.",
            "preproc_rec": "SG+MC (sem MSC — baseline Raman diferente do NIR)",
            "faixa": "50-4000 cm-1 (Raman shift)",
        },
        "EN": {
            "nome": "Raman Spectroscopy",
            "desc": "Vibrational spectroscopy by Raman scattering. Complementary to IR.",
            "preproc_rec": "SG+MC (no MSC — Raman baseline differs from NIR)",
            "faixa": "50-4000 cm-1 (Raman shift)",
        },
        "faixa_min": 50.0, "faixa_max": 4000.0,
        "preproc": "sg_mc", "modo": "dx",
    },
    "uv-vis": {
        "PT": {
            "nome": "UV-Vis (Ultravioleta-Visivel)",
            "desc": "Espectroscopia de absorbancia UV-Vis. Use modo CSV com colunas de comprimento de onda.",
            "preproc_rec": "SNV+MC ou Mean-centering (dados UV geralmente ja normalizados)",
            "faixa": "190-900 nm (use CSV — wavelength em nm como colunas)",
        },
        "EN": {
            "nome": "UV-Vis (Ultraviolet-Visible Spectroscopy)",
            "desc": "UV-Vis absorbance spectroscopy. Use CSV mode with wavelength columns.",
            "preproc_rec": "SNV+MC or Mean-centering (UV data is often already normalized)",
            "faixa": "190-900 nm (use CSV — wavelength in nm as columns)",
        },
        "faixa_min": 190.0, "faixa_max": 900.0,
        "preproc": "snv_mc", "modo": "csv",
    },
    "fluorescencia": {
        "PT": {
            "nome": "Fluorescencia Molecular",
            "desc": "Espectrofluorimetria. Dados tipicamente em formato CSV (comprimento de onda em nm).",
            "preproc_rec": "SNV+MC ou apenas MC (fluorescencia varia muito entre instrumentos)",
            "faixa": "200-800 nm (use CSV)",
        },
        "EN": {
            "nome": "Molecular Fluorescence",
            "desc": "Spectrofluorimetry. Data typically in CSV format (wavelength in nm).",
            "preproc_rec": "SNV+MC or MC only (fluorescence varies widely between instruments)",
            "faixa": "200-800 nm (use CSV)",
        },
        "faixa_min": 200.0, "faixa_max": 800.0,
        "preproc": "snv_mc", "modo": "csv",
    },
    "hplc": {
        "PT": {
            "nome": "HPLC (Cromatografia Liquida de Alta Performance)",
            "desc": "Dados cromatograficos em formato CSV. Colunas = tempo de retencao ou compostos.",
            "preproc_rec": "MC ou autoscaling (dados HPLC sao areas/alturas de pico)",
            "faixa": "Nao aplicavel — use CSV com colunas de compostos",
        },
        "EN": {
            "nome": "HPLC (High Performance Liquid Chromatography)",
            "desc": "Chromatographic data in CSV format. Columns = retention times or compounds.",
            "preproc_rec": "MC or autoscaling (HPLC data are peak areas/heights)",
            "faixa": "Not applicable — use CSV with compound columns",
        },
        "faixa_min": 0.0, "faixa_max": 60.0,
        "preproc": "autoscaling", "modo": "csv",
    },
    "gc-ms": {
        "PT": {
            "nome": "GC-MS (Cromatografia Gasosa / Massas)",
            "desc": "Compostos volateis. Dados em CSV: TIC ou tabela de fragmentos m/z.",
            "preproc_rec": "MC ou autoscaling (apos alinhamento de picos)",
            "faixa": "Nao aplicavel — use CSV (m/z ou tempo de retencao)",
        },
        "EN": {
            "nome": "GC-MS (Gas Chromatography / Mass Spectrometry)",
            "desc": "Volatile compounds. CSV data: TIC or m/z fragment table.",
            "preproc_rec": "MC or autoscaling (after peak alignment)",
            "faixa": "Not applicable — use CSV (m/z or retention time)",
        },
        "faixa_min": 0.0, "faixa_max": 90.0,
        "preproc": "autoscaling", "modo": "csv",
    },
    "nmr": {
        "PT": {
            "nome": "RMN / NMR (Ressonancia Magnetica Nuclear)",
            "desc": "Fingerprint molecular. Bucketing por janelas de ppm antes de PCA/PLS.",
            "preproc_rec": "SNV ou PQN + MC (apos binning)",
            "faixa": "0-12 ppm (deslocamento quimico)",
        },
        "EN": {
            "nome": "NMR (Nuclear Magnetic Resonance)",
            "desc": "Molecular fingerprint. Bucketing by ppm windows before PCA/PLS.",
            "preproc_rec": "SNV or PQN + MC (after binning)",
            "faixa": "0-12 ppm (chemical shift)",
        },
        "faixa_min": 0.0, "faixa_max": 12.0,
        "preproc": "snv_mc", "modo": "csv",
    },
    "ims": {
        "PT": {
            "nome": "IMS (Espectrometria de Mobilidade Ionica)",
            "desc": "Separacao de ions por mobilidade. Normalizar pelo pico do solvente (RIP).",
            "preproc_rec": "SNV+SG+MC (apos alinhamento por tempo de deriva)",
            "faixa": "5-50 ms (tempo de deriva)",
        },
        "EN": {
            "nome": "IMS (Ion Mobility Spectrometry)",
            "desc": "Ion separation by mobility. Normalize by solvent peak (RIP).",
            "preproc_rec": "SNV+SG+MC (after drift-time alignment)",
            "faixa": "5-50 ms (drift time)",
        },
        "faixa_min": 5.0, "faixa_max": 50.0,
        "preproc": "snv_sg_mc", "modo": "csv",
    },
    "generico": {
        "PT": {
            "nome": "Generico / Personalizado",
            "desc": "Qualquer tipo de dado multivariado. Configure faixa e pre-processamento manualmente.",
            "preproc_rec": "Depende dos dados — configure manualmente no menu Pre-processamento",
            "faixa": "Configurar manualmente no menu Dados",
        },
        "EN": {
            "nome": "Generic / Custom",
            "desc": "Any type of multivariate data. Configure range and preprocessing manually.",
            "preproc_rec": "Depends on data — configure manually in Preprocessing menu",
            "faixa": "Configure manually in Data menu",
        },
        "faixa_min": 0.0, "faixa_max": 999999.0,
        "preproc": "msc_sg_mc", "modo": "dx",
    },
}


def menu_tecnica(cfg: Config) -> None:
    """Menu 8 — Selecao de tecnica analitica com defaults automaticos."""
    nomes = list(TECNICAS.keys())
    while True:
        lang = _lang()
        cls()
        w = 68
        titulo = I18N[lang].get("menu_tecnica", "Tecnica Analitica")
        print("╔" + "═" * (w - 2) + "╗")
        print("║" + _ansi_ljust(f"  {titulo}", w - 2) + "║")
        if lang == "PT":
            instr = "  Selecione a tecnica. Ajusta faixa espectral e pre-processamento sugeridos:"
        else:
            instr = "  Select the technique. Adjusts spectral range and suggested preprocessing:"
        print("║" + _ansi_ljust(instr, w - 2) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        for i, key in enumerate(nomes, 1):
            tec = TECNICAS[key]
            tec_lang = tec.get(lang, tec.get("PT", {}))
            nome_tec = tec_lang.get("nome", key)
            desc_tec = tec_lang.get("desc", "")
            rec = tec_lang.get("preproc_rec", "")
            faixa = tec_lang.get("faixa", "")
            print("║" + _ansi_ljust(f"  [{i}] {nome_tec}", w - 2) + "║")
            for dl in _wrap_box(desc_tec, w - 2, "      "):
                print("║" + _ansi_ljust(_c("DIM", dl), w - 2) + "║")
            if lang == "PT":
                linha_rec = f"      Pre-proc. rec.: {rec}"
            else:
                linha_rec = f"      Rec. preproc.: {rec}"
            print("║" + _ansi_ljust(_c("DIM", linha_rec), w - 2) + "║")
            if lang == "PT":
                linha_fx = f"      Faixa: {faixa}"
            else:
                linha_fx = f"      Range: {faixa}"
            print("║" + _ansi_ljust(_c("DIM", linha_fx), w - 2) + "║")
            print("║" + _ansi_ljust("", w - 2) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        if lang == "PT":
            aviso = "  ANALITICO: aplicar uma tecnica ajusta faixa e pre-processamento automaticamente."
        else:
            aviso = "  ANALYTICAL: applying a technique auto-adjusts range and preprocessing."
        print("║" + _ansi_ljust(_c("ANALITICO", aviso), w - 2) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        rodape = f"  [I] {I18N[lang]['idioma']}   [0] {I18N[lang]['voltar']}"
        print("║" + _ansi_ljust(rodape, w - 2) + "║")
        print("╚" + "═" * (w - 2) + "╝")
        print()
        try:
            escolha = input(f"  {I18N[lang].get('opcao', 'Option')}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if escolha == "0" or escolha.lower() == "q":
            break
        if escolha.upper() == "I":
            _toggle_idioma()
            continue
        if escolha.isdigit() and 1 <= int(escolha) <= len(nomes):
            key = nomes[int(escolha) - 1]
            tec = TECNICAS[key]
            aviso_conf = I18N[lang]["aviso_analitico"]
            conf = _prompt(f"  {aviso_conf}").lower()
            if conf != I18N[lang]["confirmar_sn"]:
                continue
            # Aplicar defaults de faixa espectral
            spec_min = _SPEC_BY_KEY.get("faixa_min_cm")
            if spec_min:
                try:
                    setattr(cfg, spec_min["attr"], _coagir_valor(spec_min, tec["faixa_min"]))
                except Exception:
                    pass
            spec_max = _SPEC_BY_KEY.get("faixa_max_cm")
            if spec_max:
                try:
                    setattr(cfg, spec_max["attr"], _coagir_valor(spec_max, tec["faixa_max"]))
                except Exception:
                    pass
            # Aplicar pre-processamento recomendado
            spec_pp = _SPEC_BY_KEY.get("pre_processamento")
            if spec_pp:
                try:
                    setattr(cfg, spec_pp["attr"], _coagir_valor(spec_pp, tec["preproc"]))
                except Exception:
                    pass
            # Aplicar modo de entrada
            spec_modo = _SPEC_BY_KEY.get("modo_entrada")
            if spec_modo:
                try:
                    setattr(cfg, spec_modo["attr"], _coagir_valor(spec_modo, tec["modo"]))
                except Exception:
                    pass
            tec_lang = tec.get(lang, tec.get("PT", {}))
            if lang == "PT":
                ok = f"  Tecnica '{tec_lang.get('nome', key)}' aplicada."
            else:
                ok = f"  Technique '{tec_lang.get('nome', key)}' applied."
            print(ok)
            _prompt(f"  [{I18N[lang]['continuar']}]")
        else:
            print(f"  {I18N[lang]['invalido']}")
            _prompt(f"  [{I18N[lang]['continuar']}]")


# ===========================================================================
# Menu de Codificacao de Arquivos
# ===========================================================================

def _carregar_codigos_usuario() -> Dict[str, str]:
    """Carrega codigos de especie personalizados de codigos_usuario.json."""
    if not _CODIGOS_PATH.exists():
        return {}
    try:
        with open(_CODIGOS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _salvar_codigos_usuario(codigos: Dict[str, str]) -> None:
    """Salva codigos de especie personalizados em codigos_usuario.json."""
    with open(_CODIGOS_PATH, "w", encoding="utf-8") as f:
        json.dump(codigos, f, ensure_ascii=False, indent=2)


def menu_codificacao(cfg: Config) -> None:
    """Menu 9 — Explicacao do formato de nomenclatura dos arquivos DX."""
    lang = _lang()
    cls()
    w = 68

    titulo = "Codificacao de Arquivos DX" if lang == "PT" else "DX File Encoding"
    print("╔" + "═" * (w - 2) + "╗")
    print("║" + _ansi_ljust(f"  {titulo}", w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")

    if lang == "PT":
        secoes = [
            ("Formato de nomenclatura esperado:",
             ["COD-DD-MM-AAAA_Tn.dx",
              "COD-DD-MM-AAAA_AD-X-PP_Tn.dx  (adulterado)",
              "",
              "Onde:",
              "  COD    = Codigo da especie (2-4 letras maiusculas, ex: AND, BCB)",
              "  DD-MM-AAAA = Data de coleta",
              "  Tn     = Triplicata (T1, T2 ou T3)",
              "  AD-X-PP = Adulterante: X=tipo (S=soja,M=milho,A=algodao), PP=porcentagem"]),
            ("Exemplos validos:",
             ["AND-10-06-2020_T1.dx       (Andiroba pura, triplicata 1)",
              "BCB-03-03-2020_T2.dx       (Bacaba pura, triplicata 2)",
              "AND-10-06-2020_AD-S-20_T1.dx  (Andiroba + 20% soja)"]),
            ("Codigos de especies pre-cadastrados:",
             ["AND=Andiroba  ACE=Acai  BCB=Bacaba  BRT=Buriti",
              "BAB=Babacu    CAP=Castanha  COC=Coco   GOI=Goiaba",
              "GRV=Graviola  MAR=Maracuja  PAL=Palmiste",
              "PAT=Pataua    PRA=Pracaxi"]),
            ("Como usar arquivos com nomenclatura diferente:",
             ["Opcao 1: Renomeie os arquivos para o formato acima.",
              "Opcao 2: Use modo CSV (menu Dados > Modo de entrada = csv)",
              "         com coluna 'classe' para rotular cada amostra.",
              "Opcao 3: Use [C] abaixo para cadastrar novos codigos de especie.",
              "         Os codigos sao salvos em codigos_usuario.json."]),
        ]
    else:
        secoes = [
            ("Expected naming format:",
             ["COD-DD-MM-YYYY_Tn.dx",
              "COD-DD-MM-YYYY_AD-X-PP_Tn.dx  (adulterated)",
              "",
              "Where:",
              "  COD    = Species code (2-4 uppercase letters, e.g. AND, BCB)",
              "  DD-MM-YYYY = Collection date",
              "  Tn     = Replicate (T1, T2 or T3)",
              "  AD-X-PP = Adulterant: X=type (S=soy,M=corn,A=cotton), PP=percentage"]),
            ("Valid examples:",
             ["AND-10-06-2020_T1.dx       (Pure Andiroba, replicate 1)",
              "BCB-03-03-2020_T2.dx       (Pure Bacaba, replicate 2)",
              "AND-10-06-2020_AD-S-20_T1.dx  (Andiroba + 20% soy)"]),
            ("Pre-registered species codes:",
             ["AND=Andiroba  ACE=Acai  BCB=Bacaba  BRT=Buriti",
              "BAB=Babacu    CAP=Brazil nut  COC=Coconut  GOI=Guava",
              "GRV=Soursop   MAR=Passion fruit  PAL=Palm kernel",
              "PAT=Pataua    PRA=Pracaxi"]),
            ("Using files with different naming:",
             ["Option 1: Rename files to the format above.",
              "Option 2: Use CSV mode (Data menu > Input mode = csv)",
              "          with a 'classe' column to label each sample.",
              "Option 3: Use [C] below to register new species codes.",
              "          Codes are saved in codigos_usuario.json."]),
        ]

    for titulo_sec, linhas in secoes:
        print("║" + _ansi_ljust(f"  {_c('BOLD', titulo_sec)}", w - 2) + "║")
        for linha in linhas:
            if linha == "":
                print("║" + _ansi_ljust("", w - 2) + "║")
            else:
                for wl in _wrap_box(linha, w - 4, "    "):
                    print("║" + _ansi_ljust(wl, w - 2) + "║")
        print("║" + _ansi_ljust("", w - 2) + "║")

    print("╠" + "═" * (w - 2) + "╣")

    # Verificar se pasta de dados existe e contar arquivos
    try:
        pasta = getattr(cfg, "pasta_dados", "dados")
        if pasta and os.path.isdir(str(pasta)):
            n_dx = sum(1 for _ in Path(pasta).rglob("*.dx"))
            status_arq = (f"  {n_dx} arquivos .dx encontrados em: {pasta}"
                          if lang == "PT" else
                          f"  {n_dx} .dx files found in: {pasta}")
        else:
            status_arq = ("  Pasta de dados nao configurada."
                          if lang == "PT" else
                          "  Data folder not configured.")
    except Exception:
        status_arq = ""
    if status_arq:
        print("║" + _ansi_ljust(status_arq, w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")

    if lang == "PT":
        rodape_c = "  [C] Cadastrar novo codigo     [D] Ver todos os codigos"
    else:
        rodape_c = "  [C] Register new code         [D] View all codes"
    print("║" + _ansi_ljust(rodape_c, w - 2) + "║")
    rodape_i = f"  [I] {I18N[lang]['idioma']}   [0] {I18N[lang]['voltar']}"
    print("║" + _ansi_ljust(rodape_i, w - 2) + "║")
    print("╚" + "═" * (w - 2) + "╝")
    print()
    try:
        resp = input(f"  {I18N[lang].get('opcao', 'Option')}: ").strip()
    except (EOFError, KeyboardInterrupt):
        resp = "0"

    if resp.upper() == "I":
        _toggle_idioma()
    elif resp.upper() == "C":
        # Cadastrar novo codigo de especie
        codigos_u = _carregar_codigos_usuario()
        print()
        try:
            if lang == "PT":
                cod_raw = input("  Codigo (2-4 letras, ex: MAN): ").strip().upper()
                nome_raw = input("  Nome da especie (ex: Manga):  ").strip()
            else:
                cod_raw = input("  Code (2-4 letters, e.g. MAN): ").strip().upper()
                nome_raw = input("  Species name (e.g. Mango):    ").strip()
        except (EOFError, KeyboardInterrupt):
            cod_raw, nome_raw = "", ""
        if cod_raw and nome_raw:
            codigos_u[cod_raw] = nome_raw
            _salvar_codigos_usuario(codigos_u)
            if lang == "PT":
                print(f"  Codigo '{cod_raw}' -> '{nome_raw}' cadastrado e salvo em codigos_usuario.json")
            else:
                print(f"  Code '{cod_raw}' -> '{nome_raw}' registered and saved to codigos_usuario.json")
        else:
            print(f"  {I18N[lang]['cancelado']}")
        _prompt(f"  [{I18N[lang]['continuar']}]")
    elif resp.upper() == "D":
        # Exibir todos os codigos
        codigos_u = _carregar_codigos_usuario()
        cls()
        print()
        if lang == "PT":
            print("  === Codigos de especies cadastrados ===\n")
            print("  --- Pre-cadastrados (pipeline) ---")
        else:
            print("  === Registered species codes ===\n")
            print("  --- Pre-registered (pipeline) ---")
        try:
            for cod, nome in sorted(pq.CODIGO_ESPECIE.items()):
                print(f"    {cod:<6s} = {nome}")
        except Exception:
            pass
        if codigos_u:
            print()
            if lang == "PT":
                print("  --- Cadastrados pelo usuario (codigos_usuario.json) ---")
            else:
                print("  --- User-registered (codigos_usuario.json) ---")
            for cod, nome in sorted(codigos_u.items()):
                print(f"    {cod:<6s} = {nome}")
        else:
            if lang == "PT":
                print("\n  (Nenhum codigo cadastrado pelo usuario ainda.)")
            else:
                print("\n  (No user-registered codes yet.)")
        print()
        _prompt(f"  [{I18N[lang]['continuar']}]")


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
        w = 68
        titulo_perfil = "Perfis Prontos" if lang == "PT" else "Preset Profiles"
        instrucao = ("  Selecione um numero para aplicar o perfil e ver detalhes:"
                     if lang == "PT" else
                     "  Select a number to apply the profile and see details:")
        print("\u2554" + "\u2550" * (w - 2) + "\u2557")
        print("\u2551" + _ansi_ljust(f"  {titulo_perfil}", w - 2) + "\u2551")
        print("\u2551" + _ansi_ljust(instrucao, w - 2) + "\u2551")
        print("\u2560" + "\u2550" * (w - 2) + "\u2563")
        for i, nome in enumerate(nomes, 1):
            print("\u2551" + _ansi_ljust(f"  [{i}] {nome}", w - 2) + "\u2551")
            summary = PROFILE_KEY_SUMMARY.get(nome, {}).get(lang, "")
            if summary:
                print("\u2551" + _ansi_ljust(f"      {_c('DIM', summary)}", w - 2) + "\u2551")
            desc_lines = PROFILE_DESC.get(nome, {}).get(lang, "").split("\n")
            for dl in desc_lines:
                print("\u2551" + _ansi_ljust(f"      {_c('DIM', dl.strip())}", w - 2) + "\u2551")
            print("\u2551" + _ansi_ljust("", w - 2) + "\u2551")
        perfis_usuario = _listar_perfis_salvos()
        if perfis_usuario:
            print("\u2560" + "\u2550" * (w - 2) + "\u2563")
            label_salvos = "  Perfis salvos pelo usuario:" if lang == "PT" else "  User saved profiles:"
            print("\u2551" + _ansi_ljust(label_salvos, w - 2) + "\u2551")
            base = len(nomes)
            for j, nome_u in enumerate(perfis_usuario, base + 1):
                print("\u2551" + _ansi_ljust(f"  [{j}] {nome_u}", w - 2) + "\u2551")
        print("\u2560" + "\u2550" * (w - 2) + "\u2563")
        novo_label = "[N] Criar novo perfil" if lang == "PT" else "[N] Create new profile"
        como = ("  Como criar: configure os menus 1-7 e pressione [S] no menu principal."
                if lang == "PT" else
                "  How to: configure menus 1-7, then press [S] in main menu.")
        print("\u2551" + _ansi_ljust(f"  {novo_label}", w - 2) + "\u2551")
        print("\u2551" + _ansi_ljust(f"  {_c('DIM', como.strip())}", w - 2) + "\u2551")
        print("\u2560" + "\u2550" * (w - 2) + "\u2563")
        rodape = f"  [I] {t['idioma']}   [0] {t['voltar']}"
        print("\u2551" + _ansi_ljust(rodape, w - 2) + "\u2551")
        print("\u255a" + "\u2550" * (w - 2) + "\u255d")
        print()
        escolha = _prompt(f"  {t['opcao']}: ")
        if not escolha or escolha == "0" or escolha.lower() == "q":
            break
        if escolha.upper() == "I":
            _toggle_idioma()
            continue
        if escolha.upper() == "N":
            msg = ("  Configure os parametros nos menus 1-7 e pressione [S] no menu principal para salvar."
                   if lang == "PT" else
                   "  Configure parameters in menus 1-7 and press [S] in main menu to save.")
            print(f"\n{msg}")
            _prompt(f"  [{t['continuar']}]")
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
                _prompt(f"  [{t['continuar']}]")
            elif perfis_usuario and len(nomes) < n <= len(nomes) + len(perfis_usuario):
                nome_u = perfis_usuario[n - len(nomes) - 1]
                carregar_perfil(nome_u, cfg)
                _prompt(f"  [{t['continuar']}]")
            else:
                print(f"  {t['invalido']}")
                _prompt(f"  [{t['continuar']}]")
        else:
            print(f"  {t['invalido']}")
            _prompt(f"  [{t['continuar']}]")

def _aplicar_perfil(cfg: Config, perfil: Dict[str, Any]) -> None:
    """Aplica os valores de um perfil na Config e, se presente, atualiza a paleta visual."""
    for key, val in perfil.items():
        if key.startswith("_"):
            continue
        spec = _SPEC_BY_KEY.get(key)
        if spec is None:
            continue
        try:
            valor = _coagir_valor(spec, val)
            setattr(cfg, spec["attr"], valor)
        except (ValueError, TypeError):
            pass
    # Aplicar configurações visuais do perfil (não fazem parte da Config analítica)
    paleta = perfil.get("_paleta")
    if paleta and paleta in PALETAS_COR:
        vcfg = _carregar_visual_cfg()
        vcfg["paleta"] = paleta
        _salvar_visual_cfg(vcfg)
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
    nome = _prompt(prompt)
    if nome == "":
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
    _prompt(f"  [{t['continuar']}]")


def _carregar_yaml(cfg: Config) -> None:
    """Carrega config.yaml via carregar_config do pipeline e atualiza cfg."""
    lang = _lang()
    t = I18N[lang]
    if not _CFG_PATH.exists():
        if lang == "PT":
            print(f"  Arquivo {_CFG_PATH} nao encontrado. Salve primeiro.")
        else:
            print(f"  File {_CFG_PATH} not found. Save first.")
        _prompt(f"  [{t['continuar']}]")
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
    _prompt(f"  [{t['continuar']}]")


# ===========================================================================
# Wizard inicial
# ===========================================================================

def wizard_inicial() -> None:
    """
    Wizard de boas-vindas para primeira execucao (quando config.yaml nao existe).
    Define o idioma escolhido pelo usuario no estado global.
    """
    cls()
    w = 68
    print("\n" + "╔" + "═" * (w - 2) + "╗")
    print("║" + _ansi_ljust("  AmaNIR — Plataforma Quimiometrica FT-NIR", w - 2) + "║")
    print("║" + _ansi_ljust("  GEAAp / UFPA  |  Oleos Vegetais Amazonicos", w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")
    print("║" + _ansi_ljust("  Qual e o seu idioma? / What is your language?", w - 2) + "║")
    print("║" + _ansi_ljust("  [1] Portugues (PT)   [2] English (EN)", w - 2) + "║")
    print("╚" + "═" * (w - 2) + "╝")
    try:
        resp = input(f"  {I18N[_lang()]['opcao']}: ").strip()
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
    print("║" + _ansi_ljust(titulo_wiz, w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")
    q_perfil = "  Qual e o seu perfil?" if lang == "PT" else "  What is your profile?"
    print("║" + _ansi_ljust(q_perfil, w - 2) + "║")
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
        print("║" + _ansi_ljust(l, w - 2) + "║")
    print("╚" + "═" * (w - 2) + "╝")
    print()
    try:
        resp = input(f"  {I18N[_lang()]['opcao']}: ").strip()
    except (EOFError, KeyboardInterrupt):
        resp = "1"

    # Marcar wizard como concluido (salva apenas o idioma)
    try:
        _WIZARD_FLAG.write_text(lang, encoding="utf-8")
    except OSError:
        pass


# ===========================================================================
# Preview de configuracao antes de rodar
# ===========================================================================

def _mostrar_resumo_e_confirmar(cfg: Config) -> bool:
    """
    Exibe um resumo visual da configuracao atual e pede confirmacao.
    Retorna True se o usuario confirmar, False caso contrario.
    """
    lang = _lang()
    t = I18N[lang]
    w = 50

    # Coletar valores relevantes
    pasta = str(getattr(cfg, "pasta_dados", "?"))
    # Contagem de arquivos na pasta
    try:
        if os.path.isdir(pasta):
            n_arq = sum(1 for _ in Path(pasta).rglob("*") if Path(_).is_file())
            pasta_str = f"{pasta} ({n_arq} arquivos)"
        else:
            pasta_str = pasta
    except Exception:
        pasta_str = pasta

    # Tecnica — detectar pelo preproc/faixa atual
    pp_val = str(_fmt_yaml(_get_val(cfg, "pre_processamento")))
    fmin_val = _get_val(cfg, "faixa_min_cm")
    fmax_val = _get_val(cfg, "faixa_max_cm")
    # Inferir tecnica pela faixa
    tecnica_str = "?"
    for tec_key, tec in TECNICAS.items():
        if abs(float(tec["faixa_min"]) - float(fmin_val)) < 1 and abs(float(tec["faixa_max"]) - float(fmax_val)) < 1:
            tec_lang = tec.get(lang, tec.get("PT", {}))
            tecnica_str = tec_lang.get("nome", tec_key).split("(")[0].strip()
            break

    nivel_val = str(_fmt_yaml(_get_val(cfg, "nivel")))
    max_lvs_val = str(_fmt_yaml(_get_val(cfg, "max_lvs")))
    ddsimca_val = _fmt_yaml(_get_val(cfg, "ddsimca"))
    modo_dds = _fmt_yaml(_get_val(cfg, "modo_ddsimca"))
    opls_val = _fmt_yaml(_get_val(cfg, "opls_da"))
    bench_val = _fmt_yaml(_get_val(cfg, "benchmark"))
    dpi_val = str(_fmt_yaml(_get_val(cfg, "dpi")))

    def _on_off(v: Any) -> str:
        sv = str(v).lower()
        if sv in ("true", "sim", "yes", "on"):
            return "ON"
        return "OFF"

    ddsimca_str = f"{_on_off(ddsimca_val)} ({modo_dds})" if _on_off(ddsimca_val) == "ON" else "OFF"

    cls()
    print()
    print("╔" + "═" * (w - 2) + "╗")
    if lang == "PT":
        print("║" + _ansi_ljust("  ► Resumo da Configuracao", w - 2) + "║")
    else:
        print("║" + _ansi_ljust("  ► Configuration Summary", w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")

    def _row(label: str, val: str) -> None:
        linha = f"  {label:<12s}: {val}"
        if len(linha) > w - 2:
            linha = linha[:w - 5] + "..."
        print("║" + _ansi_ljust(linha, w - 2) + "║")

    if lang == "PT":
        _row("Dados", pasta_str)
        _row("Tecnica", tecnica_str)
        _row("Pre-proc.", pp_val)
        _row("Nivel", f"{nivel_val}  |  Max LVs: {max_lvs_val}")
        _row("DD-SIMCA", ddsimca_str)
        _row("OPLS-DA", _on_off(opls_val))
        _row("Benchmark", _on_off(bench_val))
        _row("DPI", dpi_val)
    else:
        _row("Data", pasta_str)
        _row("Technique", tecnica_str)
        _row("Preproc.", pp_val)
        _row("Level", f"{nivel_val}  |  Max LVs: {max_lvs_val}")
        _row("DD-SIMCA", ddsimca_str)
        _row("OPLS-DA", _on_off(opls_val))
        _row("Benchmark", _on_off(bench_val))
        _row("DPI", dpi_val)

    # Opcao "Salvar como" — nome personalizado da pasta de saida
    tag_atual = getattr(cfg, "tag", "") or ""
    auto_label = "(automatico)" if lang == "PT" else "(automatic)"
    nome_label = "Nome da pasta de saida" if lang == "PT" else "Output folder name"
    atual_label = f"  {nome_label}: {tag_atual if tag_atual else auto_label}"
    print("╠" + "═" * (w - 2) + "╣")
    print("║" + _ansi_ljust(atual_label, w - 2) + "║")
    instr_salvar = (f"  Novo nome (Enter = manter {auto_label if not tag_atual else repr(tag_atual)}, ? = limpar):"
                    if lang == "PT" else
                    f"  New name (Enter = keep {auto_label if not tag_atual else repr(tag_atual)}, ? = clear):")
    print("║" + _ansi_ljust(instr_salvar, w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")
    novo_nome = _prompt(f"  {t.get('opcao', 'Option')}: ")
    if novo_nome == "?":
        cfg.tag = ""
        limpo = "  Nome limpo — pasta automatica." if lang == "PT" else "  Name cleared — automatic folder."
        print(limpo)
    elif novo_nome:
        import re as _re_tag
        novo_nome_san = _re_tag.sub(r'[^\w\-_]', '_', novo_nome)
        cfg.tag = novo_nome_san
        if novo_nome_san != novo_nome:
            aviso = f"  Nome ajustado para: {novo_nome_san}" if lang == "PT" else f"  Name adjusted to: {novo_nome_san}"
            print(aviso)
        else:
            ok_nome = f"  Pasta: ..._{novo_nome_san}/" if lang == "PT" else f"  Folder: ..._{novo_nome_san}/"
            print(ok_nome)
    # else: mantém o tag atual
    print("╠" + "═" * (w - 2) + "╣")
    if lang == "PT":
        print("║" + _ansi_ljust("  Confirmar e rodar? (s/n):", w - 2) + "║")
    else:
        print("║" + _ansi_ljust("  Confirm and run? (y/n):", w - 2) + "║")
    print("╚" + "═" * (w - 2) + "╝")
    print()
    resp = _prompt(f"  {t['opcao']}: ").lower()
    return resp == t["confirmar_sn"]


# ===========================================================================
# Rodar pipeline
# ===========================================================================

def _rodar_pipeline(cfg: Config) -> None:
    """Salva config.yaml e dispara executar(cfg)."""
    lang = _lang()
    t = I18N[lang]
    # Verificar pasta de dados
    pasta = getattr(cfg, "pasta_dados", "")
    modo = getattr(cfg, "modo_entrada", "dx")
    if modo != "sintetico" and (not pasta or not os.path.isdir(str(pasta))):
        print(f"\n  {t['status_erro']}")
        if lang == "PT":
            print("  Corrija a pasta_dados antes de rodar.")
        else:
            print("  Fix pasta_dados before running.")
        _prompt(f"  [{t['continuar']}]")
        return

    # Mostrar resumo e pedir confirmacao
    if not _mostrar_resumo_e_confirmar(cfg):
        print(f"  {t['cancelado']}")
        _prompt(f"  [{t['continuar']}]")
        return

    # Mesclar codigos do usuario com o pipeline
    codigos_u = _carregar_codigos_usuario()
    if codigos_u:
        try:
            pq.CODIGO_ESPECIE.update(codigos_u)
        except Exception:
            pass

    if lang == "PT":
        print(f"\n  Salvando configuracao em {_CFG_PATH}...")
        print("  Iniciando pipeline...\n")
    else:
        print(f"\n  Saving configuration to {_CFG_PATH}...")
        print("  Starting pipeline...\n")

    salvar_config(cfg, str(_CFG_PATH))

    # Aplicar paleta de cores antes de rodar
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vcfg_pal = _carregar_visual_cfg()
        paleta_key = vcfg_pal.get("paleta", "qualitativo")
        paleta = PALETAS_COR.get(paleta_key, PALETAS_COR["qualitativo"])
        estilo = paleta.get("style", "default")
        try:
            plt.style.use(estilo)
        except Exception:
            pass
        cores = paleta.get("cores")
        if cores:
            plt.rcParams["axes.prop_cycle"] = plt.cycler(color=cores)  # type: ignore[attr-defined]
        cmap = paleta.get("cmap")
        if cmap:
            plt.rcParams["image.cmap"] = cmap
        # A1 — Tamanho de Fonte
        fonte_key = vcfg_pal.get("tamanho_fonte", "m")
        fonte_preset = FONT_PRESETS.get(fonte_key, FONT_PRESETS["m"])
        for k, v in fonte_preset.items():
            plt.rcParams[k] = v
        # A2 — Grid
        if vcfg_pal.get("grid_major", True):
            plt.rcParams["axes.grid"] = True
            plt.rcParams["grid.linestyle"] = vcfg_pal.get("grid_style", "dotted")
            plt.rcParams["grid.alpha"] = float(vcfg_pal.get("grid_alpha", 0.4))
            plt.rcParams["axes.grid.which"] = "both" if vcfg_pal.get("grid_minor", False) else "major"
        else:
            plt.rcParams["axes.grid"] = False
        # A3 — Transparencia (alpha) dos elementos graficos
        alpha_key = vcfg_pal.get("alpha_pontos", "medio")
        alpha_map = {"baixo": 0.9, "medio": 0.65, "alto": 0.35}
        alpha_val = alpha_map.get(alpha_key, 0.65)
        plt.rcParams["scatter.marker"] = "o"
        plt.rcParams["lines.alpha"] = alpha_val  # hint global
    except Exception:
        pass

    try:
        executar(cfg)
    except KeyboardInterrupt:
        msg = "\n  Pipeline interrompido pelo usuario." if lang == "PT" else "\n  Pipeline interrupted by user."
        print(msg)
    except Exception as e:  # noqa: BLE001
        msg = f"\n  Erro no pipeline: {e}" if lang == "PT" else f"\n  Pipeline error: {e}"
        print(msg)
    _prompt(f"\n  [{t['continuar']}]")


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
            _raw = input(f"  {I18N[_lang()]['opcao']}: ").strip()
            escolha = "?" if _raw == "?" else _raw.upper()
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
            menu_tecnica(cfg)
        elif escolha == "9":
            menu_codificacao(cfg)
        elif escolha == "?":
            menu_ajuda()
        elif escolha == "P":
            menu_perfis(cfg)
        elif escolha == "H":
            menu_hardware()
        elif escolha == "I":
            _toggle_idioma()
        elif escolha == "S":
            _salvar_yaml(cfg)
        elif escolha == "L":
            _carregar_yaml(cfg)
        elif escolha == "R":
            _rodar_pipeline(cfg)
        elif escolha == "N":
            lang = _lang()
            tag_atual = getattr(cfg, "tag", "") or ""
            auto_l = "(automatico)" if lang == "PT" else "(automatic)"
            print(f"\n  Nome atual: {tag_atual if tag_atual else auto_l}")
            instr = ("  Novo nome da pasta de saida (Enter = manter, ? = limpar): "
                     if lang == "PT" else
                     "  New output folder name (Enter = keep, ? = clear): ")
            novo = _prompt(instr)
            import re as _re_n
            if novo == "?":
                cfg.tag = ""
                print("  Nome limpo." if lang == "PT" else "  Name cleared.")
            elif novo:
                cfg.tag = _re_n.sub(r'[^\w\-_]', '_', novo)
                print(f"  Definido: {cfg.tag}" if lang == "PT" else f"  Set to: {cfg.tag}")
            _prompt(f"  [{I18N[lang]['continuar']}]")
        elif escolha == "Q":
            print(f"\n  {t['sair']}.")
            break
        else:
            _err(t['invalido'])
            import time
            time.sleep(0.8)


if __name__ == "__main__":
    main()
