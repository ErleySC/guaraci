"""
cli_assistente.py — Assistente CLI hierárquico para o Pipeline Quimiométrico FT-NIR
GEAAp / UFPA — Plataforma de autenticação de óleos vegetais amazônicos.

Uso:
    python cli_assistente.py
    python pineline_quimiometria_14.py   (chama este módulo automaticamente)

Requer: pineline_quimiometria_14.py no mesmo diretório.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Integração com o pipeline (sem modificar nenhuma função analítica)
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
# Internacionalização
# ---------------------------------------------------------------------------
I18N: Dict[str, Dict[str, str]] = {
    "PT": {
        "titulo": "PIPELINE QUIMIOMETRICO FT-NIR",
        "subtitulo": "GEAAp / UFPA",
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
        "aviso_analitico": "⚠  Alteracao ANALITICA — pode modificar resultados. Confirma? (s/n): ",
        "aviso_avancado": "⚠  Parametro AVANCADO — aumenta tempo de processamento.",
        "status_ok": "✅ Dados OK",
        "status_erro": "❌ Pasta nao encontrada",
        "idioma": "Idioma",
        "campo_atualizado": "✅ [{campo}] atualizado: {valor}",
        "cancelado": "Operacao cancelada.",
        "invalido": "Opcao invalida.",
        "confirmar_sn": "s",
        "nao": "n",
    },
    "EN": {
        "titulo": "FT-NIR CHEMOMETRICS PIPELINE",
        "subtitulo": "GEAAp / UFPA",
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
        "aviso_analitico": "⚠  ANALYTICAL change — may affect results. Confirm? (y/n): ",
        "aviso_avancado": "⚠  ADVANCED parameter — increases processing time.",
        "status_ok": "✅ Data OK",
        "status_erro": "❌ Folder not found",
        "idioma": "Language",
        "campo_atualizado": "✅ [{campo}] updated: {valor}",
        "cancelado": "Operation cancelled.",
        "invalido": "Invalid option.",
        "confirmar_sn": "y",
        "nao": "n",
    },
}

# ---------------------------------------------------------------------------
# Classificação de risco
# ---------------------------------------------------------------------------
RISK_CLASS: Dict[str, str] = {
    # VISUAL
    "dpi": "VISUAL", "formato_figura": "VISUAL",
    "figuras_mostrar_marcadores": "VISUAL", "figuras_mostrar_elipses": "VISUAL",
    "abrir_figuras_na_tela": "VISUAL",
    # ANALÍTICO
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
    # AVANÇADO
    "benchmark": "AVANCADO", "monte_carlo": "AVANCADO",
    "shap_benchmark": "AVANCADO", "n_monte_carlo": "AVANCADO",
    "shap_max_amostras": "AVANCADO", "monte_carlo_incluir_todos": "AVANCADO",
}

RISK_COLOR: Dict[str, str] = {
    "VISUAL": "\033[92m",    # verde
    "ANALITICO": "\033[93m", # amarelo
    "AVANCADO": "\033[91m",  # vermelho
    "RESET": "\033[0m",
}

# ---------------------------------------------------------------------------
# Base de ajuda (HELP_DB)
# ---------------------------------------------------------------------------
HELP_DB: Dict[str, Dict[str, Any]] = {
    "dpi": {
        "desc": "Resolucao das figuras em pontos por polegada.",
        "default": 600,
        "range": "150-1200",
        "impacto": "VISUAL — nao altera resultados analiticos.",
        "exemplos": {"300": "Apresentacoes/slides", "600": "Artigos cientificos", "1200": "Impressao profissional"},
    },
    "max_lvs": {
        "desc": "Numero maximo de Variaveis Latentes testadas na selecao automatica.",
        "default": 40,
        "range": "5-80",
        "impacto": "ANALITICO — afeta diretamente o modelo PLS-DA.",
        "exemplos": {"10": "Datasets pequenos", "40": "Recomendado", "80": "Datasets grandes"},
    },
    "n_permutacoes": {
        "desc": "Numero de permutacoes para o teste Y-randomization.",
        "default": 200,
        "range": "50-1000",
        "impacto": "ANALITICO — afeta o p-value do teste de permutacao.",
        "exemplos": {"50": "Diagnostico rapido", "200": "Publicacao", "1000": "Alta precisao"},
    },
    "pre_processamento": {
        "desc": "Pipeline de pre-processamento espectral aplicado antes do PLS-DA.",
        "default": "MSC+SG+MC",
        "range": "Escolha na lista",
        "impacto": "ANALITICO — MSC+SG+MC deu Bal.Acc=0.923 no dataset atual.",
        "exemplos": {"msc_sg_mc": "Melhor resultado (recomendado)", "snv_sg_mc": "Alternativa robusta", "autoscaling": "Baseline simples"},
    },
    "modo_ddsimca": {
        "desc": "Modo de treino do DD-SIMCA.",
        "default": "puros",
        "range": "puros | todos",
        "impacto": "ANALITICO — 'puros' valida autenticacao; 'todos' e exploratorio.",
        "exemplos": {"puros": "Autenticacao (recomendado para publicacao)", "todos": "Analise exploratoria"},
    },
    "nivel": {
        "desc": "Nivel de analise: N1 (basico), N2 (PLS-DA + DD-SIMCA), N3 (regressao).",
        "default": "N2",
        "range": "N1 | N2 | N3",
        "impacto": "ANALITICO — define quais modulos sao executados.",
        "exemplos": {"N1": "So PCA/HCA", "N2": "Classificacao completa", "N3": "Quantificacao"},
    },
    "holdout_fracao": {
        "desc": "Fracao do dataset reservada para validacao externa (holdout).",
        "default": 0.2,
        "range": "0.1-0.3",
        "impacto": "ANALITICO — afeta a estimativa de generalizacao do modelo.",
        "exemplos": {"0.1": "Poucos dados", "0.2": "Recomendado", "0.3": "Dataset grande"},
    },
    "ddsimca": {
        "desc": "Ativa o DD-SIMCA para autenticacao de amostras por classe.",
        "default": True,
        "range": "true | false",
        "impacto": "ANALITICO — adiciona figuras e metricas de autenticacao.",
        "exemplos": {"true": "Recomendado para publicacao", "false": "Economiza tempo em exploracao"},
    },
    "benchmark": {
        "desc": "Compara PLS-DA contra SVM, RF, XGBoost com mesma CV group-aware.",
        "default": False,
        "range": "true | false",
        "impacto": "ANALITICO — adiciona ~30-60 min de processamento.",
        "exemplos": {"false": "Producao/TCC", "true": "Artigo com comparacao de modelos"},
    },
    "formato_figura": {
        "desc": "Formato de saida das figuras geradas.",
        "default": "png",
        "range": "png | pdf | svg",
        "impacto": "VISUAL — nao altera analise.",
        "exemplos": {"png": "Apresentacoes/TCC", "pdf": "Artigos (vetorial)", "svg": "Edicao pos-processamento"},
    },
    "monte_carlo": {
        "desc": "Ativa Monte Carlo CV com IC95% por percentil.",
        "default": False,
        "range": "true | false",
        "impacto": "AVANCADO — aumenta significativamente o tempo de processamento.",
        "exemplos": {"false": "Producao rapida", "true": "Dissertacao/Tese"},
    },
    "n_monte_carlo": {
        "desc": "Numero de repeticoes do Monte Carlo CV.",
        "default": 100,
        "range": "50-500",
        "impacto": "AVANCADO — mais repeticoes = IC mais estreito, mais tempo.",
        "exemplos": {"50": "Teste rapido", "100": "TCC", "200": "Dissertacao"},
    },
    "shap_benchmark": {
        "desc": "Calcula SHAP values (TreeExplainer) para RF/XGBoost/GBM.",
        "default": False,
        "range": "true | false",
        "impacto": "AVANCADO — requer benchmark=true; +10-20 min.",
        "exemplos": {"false": "Producao", "true": "Artigo com interpretabilidade espectral"},
    },
    "shap_max_amostras": {
        "desc": "Limite de amostras para calculo de SHAP (controle de memoria).",
        "default": 500,
        "range": "100-1000",
        "impacto": "AVANCADO — valores maiores aumentam tempo e uso de RAM.",
        "exemplos": {"200": "RAM limitada", "500": "Recomendado"},
    },
    "validacao_group_aware": {
        "desc": "Manter replicas (T1/T2/T3) juntas na validacao (evita vazamento de dados).",
        "default": True,
        "range": "true | false",
        "impacto": "ANALITICO — false pode inflar artificialmente as metricas.",
        "exemplos": {"true": "Correto para dados com replicas", "false": "Apenas para datasets independentes"},
    },
    "opls_da": {
        "desc": "Executa OPLS-DA (Orthogonal PLS Discriminant Analysis).",
        "default": True,
        "range": "true | false",
        "impacto": "ANALITICO — gera S-Plot e separa variacao ortogonal.",
        "exemplos": {"true": "Recomendado para publicacao", "false": "Analise exploratoria rapida"},
    },
    "selecao_variaveis_etapa4": {
        "desc": "Executa iPLS/VIP/SR/sPLS-DA para selecao de variaveis.",
        "default": True,
        "range": "true | false",
        "impacto": "ANALITICO — identifica regioes espectrais mais relevantes.",
        "exemplos": {"true": "Publicacao", "false": "Exploracao rapida"},
    },
    "comparar_pre_processamentos": {
        "desc": "Compara varios pipelines de pre-processamento automaticamente.",
        "default": False,
        "range": "true | false",
        "impacto": "ANALITICO — aumenta o tempo; util para selecionar o melhor pipeline.",
        "exemplos": {"false": "Usar pre-processamento padrao", "true": "Otimizacao do pipeline"},
    },
    "pasta_dados": {
        "desc": "Pasta com os arquivos .dx (modo dx; uma subpasta por classe).",
        "default": "dados",
        "range": "Caminho valido no sistema",
        "impacto": "ANALITICO — define os dados de entrada.",
        "exemplos": {"dados": "Pasta padrao do projeto", "C:/meus_dados": "Pasta personalizada"},
    },
    "pasta_saida": {
        "desc": "Pasta onde os resultados serao gravados.",
        "default": "resultados",
        "range": "Caminho valido no sistema",
        "impacto": "ANALITICO — define onde as figuras e metricas sao salvas.",
        "exemplos": {"resultados": "Pasta padrao", "C:/experimento_01": "Pasta de experimento especifico"},
    },
    "modo_entrada": {
        "desc": "Origem dos dados: dx (JCAMP-DX, FT-NIR) | csv | sintetico (teste).",
        "default": "dx",
        "range": "dx | csv | sintetico",
        "impacto": "ANALITICO — define o formato de leitura dos dados.",
        "exemplos": {"dx": "Espectros FT-NIR", "csv": "Tabela generica", "sintetico": "Dados de teste"},
    },
    "faixa_min_cm": {
        "desc": "Inicio da faixa espectral util (cm-1).",
        "default": 4000.0,
        "range": "400-12000",
        "impacto": "ANALITICO — restringe a regiao espectral analisada.",
        "exemplos": {"4000": "Padrao NIR", "5500": "Regiao de combinacoes"},
    },
    "faixa_max_cm": {
        "desc": "Fim da faixa espectral util (cm-1).",
        "default": 10000.0,
        "range": "400-12000",
        "impacto": "ANALITICO — restringe a regiao espectral analisada.",
        "exemplos": {"10000": "Padrao NIR", "7500": "NIR curto"},
    },
    "excluir_classes": {
        "desc": "Especies a remover da analise.",
        "default": [],
        "range": "Lista de nomes de classes",
        "impacto": "ANALITICO — altera o conjunto de treinamento.",
        "exemplos": {"[]": "Usar todas as classes", "[Copaiba]": "Remover Copaiba"},
    },
    "teste_wold": {
        "desc": "Rodar teste de Wold (intercepts R2Y/Q2Y para parsimonia).",
        "default": True,
        "range": "true | false",
        "impacto": "ANALITICO — define o numero otimo de LVs (criterio Wold).",
        "exemplos": {"true": "Recomendado", "false": "Economiza tempo"},
    },
    "teste_cv_anova": {
        "desc": "Rodar CV-ANOVA (Eriksson) para significancia estatistica.",
        "default": True,
        "range": "true | false",
        "impacto": "ANALITICO — testa significancia do modelo por ANOVA.",
        "exemplos": {"true": "Publicacao", "false": "Exploracao"},
    },
    "figuras_mostrar_marcadores": {
        "desc": "Usar formas diferentes por classe nos graficos de score.",
        "default": True,
        "range": "true | false",
        "impacto": "VISUAL — nao altera resultados analiticos.",
        "exemplos": {"true": "Graficos mais informativos", "false": "Graficos mais limpos"},
    },
    "figuras_mostrar_elipses": {
        "desc": "Desenhar elipses de confianca por grupo.",
        "default": True,
        "range": "true | false",
        "impacto": "VISUAL — nao altera resultados analiticos.",
        "exemplos": {"true": "Padrao para publicacao", "false": "Graficos mais simples"},
    },
    "abrir_figuras_na_tela": {
        "desc": "Abrir cada figura na tela ao gerar (alem de salvar em arquivo).",
        "default": False,
        "range": "true | false",
        "impacto": "VISUAL — nao altera analise.",
        "exemplos": {"false": "Execucao automatizada", "true": "Revisao interativa"},
    },
    "arquivo_csv": {
        "desc": "Caminho do CSV (modo csv): colunas espectrais + 1 coluna de classe.",
        "default": "",
        "range": "Caminho valido (.csv)",
        "impacto": "ANALITICO — define os dados de entrada no modo CSV.",
        "exemplos": {"dados.csv": "Arquivo na pasta atual", "C:/dados/amostras.csv": "Caminho completo"},
    },
    "coluna_classe": {
        "desc": "Nome da coluna de classe/rotulo no CSV.",
        "default": "classe",
        "range": "Nome de coluna existente no CSV",
        "impacto": "ANALITICO — define o alvo de classificacao.",
        "exemplos": {"classe": "Nome padrao", "especie": "Nome alternativo"},
    },
    "coluna_concentracao": {
        "desc": "Nome da coluna de concentracao no CSV (vazio se nao houver).",
        "default": "",
        "range": "Nome de coluna ou vazio",
        "impacto": "ANALITICO — usado no nivel N3 (regressao).",
        "exemplos": {"": "Sem concentracao (N1/N2)", "conc": "Com concentracao (N3)"},
    },
    "monte_carlo_incluir_todos": {
        "desc": "MC CV: incluir SVM RBF / RF / XGBoost alem do PLS-DA (mais lento).",
        "default": False,
        "range": "true | false",
        "impacto": "AVANCADO — aumenta muito o tempo de MC CV.",
        "exemplos": {"false": "Apenas PLS-DA", "true": "Comparacao completa"},
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
# Mapeamento de menus → campos (chaves do _CONFIG_SPEC)
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

# Índice inverso: chave -> especificação do _CONFIG_SPEC
_SPEC_BY_KEY: Dict[str, Dict[str, Any]] = {s["key"]: s for s in _CONFIG_SPEC}


# ===========================================================================
# Utilitários de terminal
# ===========================================================================

def cls() -> None:
    """Limpa a tela de forma cross-platform."""
    os.system("cls" if os.name == "nt" else "clear")


def _c(color_key: str, text: str) -> str:
    """Envolve texto com código ANSI de cor."""
    return f"{RISK_COLOR.get(color_key, '')}{text}{RISK_COLOR['RESET']}"


def _risk_label(key: str) -> str:
    """Retorna a label colorida de risco para um campo."""
    risk = RISK_CLASS.get(key, "ANALITICO")
    return _c(risk, f"[{risk}]")


def _get_val(cfg: Config, key: str) -> Any:
    """Lê o valor atual do campo na Config."""
    spec = _SPEC_BY_KEY.get(key)
    if spec is None:
        return "?"
    return _attr_para_yaml(spec, cfg)


def _set_val(cfg: Config, key: str, raw: str) -> None:
    """Converte raw string e seta no atributo da Config."""
    spec = _SPEC_BY_KEY[key]
    valor = _coagir_valor(spec, raw)
    setattr(cfg, spec["attr"], valor)


def _status_dados(cfg: Config, lang: str) -> str:
    """Retorna string de status da pasta de dados."""
    pasta = getattr(cfg, "pasta_entrada", "dados")
    if pasta and os.path.isdir(str(pasta)):
        return I18N[lang]["status_ok"] + f" ({pasta})"
    return I18N[lang]["status_erro"] + f" ({pasta})"


def _largura() -> int:
    try:
        return min(os.get_terminal_size().columns, 60)
    except OSError:
        return 60


# ===========================================================================
# Cabeçalho
# ===========================================================================

def print_header(cfg: Config, lang: str) -> None:
    """Imprime o cabeçalho com título, idioma atual e status dos dados."""
    w = 60
    titulo = I18N[lang]["titulo"]
    subtitulo = I18N[lang]["subtitulo"]
    status = _status_dados(cfg, lang)
    idioma_str = f"[{lang}]"
    print("\n" + "╔" + "═" * (w - 2) + "╗")
    print("║" + titulo.center(w - 2) + "║")
    print("║" + subtitulo.center(w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")
    print("║" + f"  {status}".ljust(w - 5) + idioma_str + "  ║")
    print("╚" + "═" * (w - 2) + "╝")


# ===========================================================================
# Menu principal
# ===========================================================================

def print_main_menu(lang: str) -> None:
    """Imprime o menu principal hierárquico com borda ASCII."""
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
# Edição genérica de campo (com confirmação de risco)
# ===========================================================================

def _editar_campo_cli(cfg: Config, key: str, lang: str) -> bool:
    """
    Mostra valor atual, pede novo valor, aplica confirmação de risco.
    Retorna True se o campo foi atualizado, False se cancelado.
    """
    spec = _SPEC_BY_KEY.get(key)
    if spec is None:
        print(f"  Campo '{key}' nao encontrado.")
        return False

    atual = _get_val(cfg, key)
    risk = RISK_CLASS.get(key, "ANALITICO")
    risk_lbl = _risk_label(key)

    print(f"\n  {risk_lbl} {key}")
    if key in HELP_DB:
        print(f"  Desc: {HELP_DB[key]['desc']}")
        if HELP_DB[key].get("range"):
            print(f"  Faixa: {HELP_DB[key]['range']}")
    if spec.get("opcoes"):
        print(f"  Opcoes: {' | '.join(str(o) for o in spec['opcoes'])}")
    print(f"  Atual: {_fmt_yaml(atual)}")

    try:
        novo_raw = input("  Novo valor (Enter = manter): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    if novo_raw == "":
        print("  Mantido.")
        return False

    # Ajuda inline
    if novo_raw.lower() in ("?", "help"):
        _mostrar_help_campo(key, lang)
        return False

    # Confirmação para ANALITICO / AVANCADO
    if risk == "AVANCADO":
        print(_c("AVANCADO", f"\n  {I18N[lang]['aviso_avancado']}"))
    if risk in ("ANALITICO", "AVANCADO"):
        try:
            conf = input(f"  {I18N[lang]['aviso_analitico']}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if conf != I18N[lang]["confirmar_sn"]:
            print(f"  {I18N[lang]['cancelado']}")
            return False

    try:
        _set_val(cfg, key, novo_raw)
        novo_val = _get_val(cfg, key)
        msg = I18N[lang]["campo_atualizado"].format(campo=key, valor=_fmt_yaml(novo_val))
        print(f"  {msg}")
        return True
    except (ValueError, TypeError) as e:
        print(f"  Erro: {e}")
        return False


def _mostrar_help_campo(key: str, lang: str) -> None:  # noqa: ARG001
    """Exibe o HELP_DB completo para um campo."""
    h = HELP_DB.get(key)
    if h is None:
        print(f"  Sem ajuda disponivel para '{key}'.")
        return
    print(f"\n  === Ajuda: {key} ===")
    print(f"  Descricao : {h['desc']}")
    print(f"  Padrao    : {h['default']}")
    print(f"  Faixa     : {h.get('range', 'N/A')}")
    print(f"  Impacto   : {h['impacto']}")
    if h.get("exemplos"):
        print("  Exemplos  :")
        for val, desc in h["exemplos"].items():
            print(f"    {val:>8} → {desc}")
    print()


def _submenu_campos(cfg: Config, lang: str, titulo: str, campos: list) -> None:
    """Loop genérico para exibir e editar um grupo de campos."""
    while True:
        cls()
        print_header(cfg, lang)
        w = 60
        print("╔" + "═" * (w - 2) + "╗")
        print("║" + f"  {titulo}".ljust(w - 2) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        for i, key in enumerate(campos, 1):
            val_str = _fmt_yaml(_get_val(cfg, key))
            risk = RISK_CLASS.get(key, "ANALITICO")
            cor = RISK_COLOR.get(risk, "")
            rst = RISK_COLOR["RESET"]
            lbl = f"{cor}●{rst}"
            linha = f"  [{i:2d}] {lbl} {key:<32s}: {val_str}"
            print("║" + linha.ljust(w + 10) + "║")  # +10 para escapes ANSI
        print("╠" + "═" * (w - 2) + "╣")
        print("║" + "  [?] Ajuda sobre campo   [0] Voltar".ljust(w - 2) + "║")
        print("╚" + "═" * (w - 2) + "╝")
        print()
        try:
            escolha = input("  Opcao: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if escolha == "0" or escolha.lower() == "q":
            break
        # Ajuda por nome de campo
        if escolha.lower().startswith("?") or escolha.lower().startswith("help"):
            partes = escolha.split(maxsplit=1)
            campo_help = partes[1] if len(partes) > 1 else ""
            if campo_help:
                _mostrar_help_campo(campo_help.strip(), lang)
            else:
                _mostrar_help_campo("", lang)
            input("  [Enter para continuar]")
            continue
        if escolha.isdigit() and 1 <= int(escolha) <= len(campos):
            key = campos[int(escolha) - 1]
            _editar_campo_cli(cfg, key, lang)
            input("  [Enter para continuar]")
        else:
            print(f"  {I18N[lang]['invalido']}")
            input("  [Enter para continuar]")


# ===========================================================================
# Submenus temáticos
# ===========================================================================

def menu_projeto(cfg: Config, lang: str) -> None:
    """Menu 1 — Projeto: pasta_dados e pasta_saida."""
    _submenu_campos(cfg, lang, I18N[lang]["menu_projeto"],
                    MENU_FIELDS["projeto"])


def menu_dados(cfg: Config, lang: str) -> None:
    """Menu 2 — Dados: modo, CSV, colunas, faixa espectral, exclusões."""
    _submenu_campos(cfg, lang, I18N[lang]["menu_dados"],
                    MENU_FIELDS["dados"])


def menu_preproc(cfg: Config, lang: str) -> None:
    """Menu 3 — Pré-processamento: pipeline + comparação."""
    while True:
        cls()
        print_header(cfg, lang)
        w = 60
        titulo = I18N[lang]["menu_preproc"]
        campos = MENU_FIELDS["preproc"]
        print("╔" + "═" * (w - 2) + "╗")
        print("║" + f"  {titulo}".ljust(w - 2) + "║")
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
            linha = f"  [{i:2d}] {lbl} {key:<32s}: {val_str}"
            print("║" + linha.ljust(w + 10) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        print("║" + "  [0] Voltar".ljust(w - 2) + "║")
        print("╚" + "═" * (w - 2) + "╝")
        print()
        try:
            escolha = input("  Opcao: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if escolha == "0" or escolha.lower() == "q":
            break
        if escolha.isdigit() and 1 <= int(escolha) <= len(campos):
            key = campos[int(escolha) - 1]
            _editar_campo_cli(cfg, key, lang)
            input("  [Enter para continuar]")
        else:
            print(f"  {I18N[lang]['invalido']}")
            input("  [Enter para continuar]")


def menu_modelagem(cfg: Config, lang: str) -> None:
    """Menu 4 — Modelagem."""
    _submenu_campos(cfg, lang, I18N[lang]["menu_modelo"],
                    MENU_FIELDS["modelo"])


def menu_validacao(cfg: Config, lang: str) -> None:
    """Menu 5 — Validação."""
    _submenu_campos(cfg, lang, I18N[lang]["menu_valid"],
                    MENU_FIELDS["validacao"])


def menu_avancado(cfg: Config, lang: str) -> None:
    """Menu 6 — Métodos Avançados (benchmark, MC, SHAP)."""
    _submenu_campos(cfg, lang, I18N[lang]["menu_avancado"],
                    MENU_FIELDS["avancado"])


def menu_visualizacao(cfg: Config, lang: str) -> None:
    """Menu 7 — Visualização (apenas campos VISUAL)."""
    _submenu_campos(cfg, lang, I18N[lang]["menu_viz"],
                    MENU_FIELDS["visualizacao"])


# ===========================================================================
# Menu de Ajuda
# ===========================================================================

def menu_ajuda(lang: str) -> None:
    """Menu 8 — Sistema de ajuda. Suporta 'help <topico>' ou '?' para listar."""
    while True:
        cls()
        w = 60
        print("╔" + "═" * (w - 2) + "╗")
        print("║" + f"  {I18N[lang]['menu_ajuda']}".ljust(w - 2) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        print("║" + "  Digite o nome do parametro para ver detalhes.".ljust(w - 2) + "║")
        print("║" + "  Exemplos: dpi  |  benchmark  |  pre_processamento".ljust(w - 2) + "║")
        print("║" + "  [L] Listar todos   [0] Voltar".ljust(w - 2) + "║")
        print("╚" + "═" * (w - 2) + "╝")
        print()
        try:
            entrada = input("  ? ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if entrada == "0" or entrada.lower() == "q":
            break
        if entrada.lower() == "l":
            cls()
            print(f"\n  Parametros com ajuda disponivel ({len(HELP_DB)} total):\n")
            for k in sorted(HELP_DB.keys()):
                risk = RISK_CLASS.get(k, "ANALITICO")
                cor = RISK_COLOR.get(risk, "")
                rst = RISK_COLOR["RESET"]
                print(f"    {cor}{k:<36s}{rst}  {HELP_DB[k]['desc'][:40]}")
            print()
            input("  [Enter para continuar]")
            continue
        # Permite "help benchmark" ou apenas "benchmark"
        topico = entrada.lower().replace("help", "").replace("?", "").strip()
        if topico:
            _mostrar_help_campo(topico, lang)
        else:
            print("  Digite um nome de parametro. Ex: dpi")
        input("  [Enter para continuar]")


# ===========================================================================
# Menu de Perfis
# ===========================================================================

def menu_perfis(cfg: Config, lang: str) -> None:
    """Lista PROFILES e permite carregar um deles na Config."""
    nomes = list(PROFILES.keys())
    while True:
        cls()
        w = 60
        print("╔" + "═" * (w - 2) + "╗")
        print("║" + f"  {I18N[lang]['perfis']}".ljust(w - 2) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        for i, nome in enumerate(nomes, 1):
            print("║" + f"  [{i}] {nome}".ljust(w - 2) + "║")
        # Perfis salvos pelo usuário
        perfis_usuario = _listar_perfis_salvos()
        if perfis_usuario:
            print("╠" + "═" * (w - 2) + "╣")
            print("║" + "  Perfis salvos pelo usuario:".ljust(w - 2) + "║")
            base = len(nomes)
            for j, nome_u in enumerate(perfis_usuario, base + 1):
                print("║" + f"  [{j}] {nome_u} (usuario)".ljust(w - 2) + "║")
        print("╠" + "═" * (w - 2) + "╣")
        print("║" + "  [0] Voltar".ljust(w - 2) + "║")
        print("╚" + "═" * (w - 2) + "╝")
        print()
        try:
            escolha = input("  Opcao: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if escolha == "0" or escolha.lower() == "q":
            break
        if escolha.isdigit():
            n = int(escolha)
            if 1 <= n <= len(nomes):
                nome_perfil = nomes[n - 1]
                _aplicar_perfil(cfg, PROFILES[nome_perfil], lang)
                input("  [Enter para continuar]")
            elif perfis_usuario and len(nomes) < n <= len(nomes) + len(perfis_usuario):
                nome_u = perfis_usuario[n - len(nomes) - 1]
                carregar_perfil(nome_u, cfg, lang)
                input("  [Enter para continuar]")
            else:
                print(f"  {I18N[lang]['invalido']}")
                input("  [Enter para continuar]")
        else:
            print(f"  {I18N[lang]['invalido']}")
            input("  [Enter para continuar]")


def _aplicar_perfil(cfg: Config, perfil: Dict[str, Any], lang: str) -> None:
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
    print("  Perfil aplicado com sucesso.")


def _listar_perfis_salvos() -> list:
    """Lista nomes dos perfis JSON salvos na pasta perfis/."""
    if not _PERFIS_DIR.exists():
        return []
    return [p.stem for p in _PERFIS_DIR.glob("*.json")]


# ===========================================================================
# Salvar / Carregar perfil do usuário
# ===========================================================================

def salvar_perfil(cfg: Config, lang: str) -> None:
    """Salva a Config atual como JSON em perfis/<nome>.json."""
    _PERFIS_DIR.mkdir(parents=True, exist_ok=True)
    print()
    try:
        nome = input("  Nome do perfil (sem espacos): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not nome:
        print("  Nome vazio. Operacao cancelada.")
        return
    # Sanitizar nome
    nome_arquivo = "".join(c if c.isalnum() or c in "-_" else "_" for c in nome)
    dados: Dict[str, Any] = {}
    for s in _CONFIG_SPEC:
        dados[s["key"]] = _attr_para_yaml(s, cfg)
    caminho = _PERFIS_DIR / f"{nome_arquivo}.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    print(f"  Perfil salvo em: {caminho}")


def carregar_perfil(nome: str, cfg: Config, lang: str) -> None:
    """Carrega um perfil JSON de perfis/<nome>.json para a Config."""
    caminho = _PERFIS_DIR / f"{nome}.json"
    if not caminho.exists():
        print(f"  Perfil '{nome}' nao encontrado em {_PERFIS_DIR}.")
        return
    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)
    _aplicar_perfil(cfg, dados, lang)
    print(f"  Perfil '{nome}' carregado.")


# ===========================================================================
# Salvar / Carregar config YAML (integrado ao pipeline)
# ===========================================================================

def _salvar_yaml(cfg: Config, lang: str) -> None:
    """Salva config.yaml via salvar_config do pipeline."""
    salvar_config(cfg, str(_CFG_PATH))
    print(f"  Configuracao salva em: {_CFG_PATH}")
    input("  [Enter para continuar]")


def _carregar_yaml(cfg: Config, lang: str) -> None:
    """Carrega config.yaml via carregar_config do pipeline e atualiza cfg."""
    if not _CFG_PATH.exists():
        print(f"  Arquivo {_CFG_PATH} nao encontrado. Salve primeiro.")
        input("  [Enter para continuar]")
        return
    cfg_novo = carregar_config(str(_CFG_PATH))
    # Copiar atributos do Config carregado para o atual (em-place)
    for s in _CONFIG_SPEC:
        try:
            setattr(cfg, s["attr"], getattr(cfg_novo, s["attr"]))
        except AttributeError:
            pass
    print(f"  Configuracao carregada de: {_CFG_PATH}")
    input("  [Enter para continuar]")


# ===========================================================================
# Trocar idioma
# ===========================================================================

def _toggle_idioma(lang: str) -> str:
    """Alterna entre PT e EN."""
    return "EN" if lang == "PT" else "PT"


# ===========================================================================
# Wizard inicial
# ===========================================================================

def wizard_inicial(lang: str) -> str:
    """
    Wizard de boas-vindas para primeira execução (quando config.yaml não existe).
    Retorna o idioma escolhido pelo usuário.
    """
    cls()
    w = 60
    print("\n" + "╔" + "═" * (w - 2) + "╗")
    print("║" + "  Bem-vindo ao Pipeline Quimiometrico FT-NIR".ljust(w - 2) + "║")
    print("║" + "  GEAAp / UFPA".ljust(w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")
    print("║" + "  Qual e o seu idioma? / What is your language?".ljust(w - 2) + "║")
    print("║" + "  [1] Portugues (PT)   [2] English (EN)".ljust(w - 2) + "║")
    print("╚" + "═" * (w - 2) + "╝")
    try:
        resp = input("  Opcao: ").strip()
    except (EOFError, KeyboardInterrupt):
        resp = "1"
    if resp == "2":
        lang = "EN"

    cls()
    t = I18N[lang]
    print("\n" + "╔" + "═" * (w - 2) + "╗")
    titulo_wiz = "  Bem-vindo!" if lang == "PT" else "  Welcome!"
    print("║" + titulo_wiz.ljust(w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")
    q_perfil = "  Qual e o seu perfil?" if lang == "PT" else "  What is your profile?"
    print("║" + q_perfil.ljust(w - 2) + "║")
    print("╠" + "═" * (w - 2) + "╣")
    linhas_wiz = [
        "  [1] Iniciante    — configuracoes essenciais, pre-proc padrao",
        "  [2] Pesquisador  — opcoes intermediarias, validacao completa",
        "  [3] Especialista — acesso a todos os parametros",
    ] if lang == "PT" else [
        "  [1] Beginner    — essential settings, default preprocessing",
        "  [2] Researcher  — intermediate options, full validation",
        "  [3] Expert      — access to all parameters",
    ]
    for l in linhas_wiz:
        print("║" + l.ljust(w - 2) + "║")
    print("╚" + "═" * (w - 2) + "╝")
    print()
    try:
        resp = input("  Opcao: ").strip()
    except (EOFError, KeyboardInterrupt):
        resp = "1"

    # Marcar wizard como concluído
    try:
        _WIZARD_FLAG.write_text(resp, encoding="utf-8")
    except OSError:
        pass
    return lang


# ===========================================================================
# Rodar pipeline
# ===========================================================================

def _rodar_pipeline(cfg: Config, lang: str) -> None:
    """Salva config.yaml e dispara executar(cfg)."""
    # Verificar pasta de dados
    pasta = getattr(cfg, "pasta_entrada", "")
    modo = getattr(cfg, "modo", "dx")
    if modo != "sintetico" and (not pasta or not os.path.isdir(str(pasta))):
        print(f"\n  {I18N[lang]['status_erro']}")
        print("  Corrija a pasta_dados antes de rodar.")
        input("  [Enter para continuar]")
        return

    print(f"\n  Salvando configuracao em {_CFG_PATH}...")
    salvar_config(cfg, str(_CFG_PATH))
    print("  Iniciando pipeline...\n")
    try:
        executar(cfg)
    except KeyboardInterrupt:
        print("\n  Pipeline interrompido pelo usuario.")
    except Exception as e:  # noqa: BLE001
        print(f"\n  Erro no pipeline: {e}")
    input("\n  [Enter para voltar ao menu]")


# ===========================================================================
# Loop principal
# ===========================================================================

def main() -> None:
    """Ponto de entrada do assistente CLI hierárquico."""
    # Carrega config existente ou cria padrão
    if _CFG_PATH.exists():
        try:
            cfg = carregar_config(str(_CFG_PATH))
        except Exception:  # noqa: BLE001
            cfg = Config()
    else:
        cfg = Config()

    # Wizard na primeira vez
    lang = "PT"
    if not _WIZARD_FLAG.exists() and not _CFG_PATH.exists():
        lang = wizard_inicial(lang)
    else:
        # Tentar recuperar idioma salvo
        try:
            idioma_salvo = _WIZARD_FLAG.read_text(encoding="utf-8").strip()
            if idioma_salvo in ("EN", "2"):
                lang = "EN"
        except OSError:
            pass

    # Loop principal
    while True:
        cls()
        print_header(cfg, lang)
        print_main_menu(lang)
        print()
        try:
            escolha = input("  Opcao: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {I18N[lang]['sair']}.")
            break

        if escolha == "1":
            menu_projeto(cfg, lang)
        elif escolha == "2":
            menu_dados(cfg, lang)
        elif escolha == "3":
            menu_preproc(cfg, lang)
        elif escolha == "4":
            menu_modelagem(cfg, lang)
        elif escolha == "5":
            menu_validacao(cfg, lang)
        elif escolha == "6":
            menu_avancado(cfg, lang)
        elif escolha == "7":
            menu_visualizacao(cfg, lang)
        elif escolha == "8":
            menu_ajuda(lang)
        elif escolha == "P":
            menu_perfis(cfg, lang)
        elif escolha == "I":
            lang = _toggle_idioma(lang)
            # Atualizar flag de idioma
            try:
                _WIZARD_FLAG.write_text(lang, encoding="utf-8")
            except OSError:
                pass
        elif escolha == "S":
            _salvar_yaml(cfg, lang)
        elif escolha == "L":
            _carregar_yaml(cfg, lang)
        elif escolha == "R":
            _rodar_pipeline(cfg, lang)
        elif escolha == "Q":
            print(f"\n  {I18N[lang]['sair']}.")
            break
        else:
            print(f"  {I18N[lang]['invalido']}")
            import time
            time.sleep(0.8)


if __name__ == "__main__":
    main()
