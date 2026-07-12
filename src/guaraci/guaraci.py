"""
guaraci.py v31.3.0 — Interface profissional GUARACI para o pipeline quimiometrico
☀  GUARACI — Inteligencia Quimiometrica para Matrizes Amazonicas
GEAAp / UFPA  |  Quimiometria • Machine Learning • Espectroscopia multitecnica

Uso:
    python guaraci.py

Requer: pipeline.py e cli_assistente.py no mesmo diretorio.
Rich 15.0+ necessario (pip install rich).
"""

from __future__ import annotations

import contextlib
import logging
import json
import os
import re as _re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# UTF-8 no Windows antes de qualquer import rich
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (OSError, ValueError):
            pass   # stdout/stderr redirecionado p/ algo sem reconfigure util
    try:
        import subprocess
        subprocess.run(["chcp", "65001"], capture_output=True, shell=True)
    except OSError:
        pass   # chcp indisponivel neste shell
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ---------------------------------------------------------------------------
# Rich
# ---------------------------------------------------------------------------
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich.text import Text
from rich.rule import Rule
from rich.live import Live
from rich.markup import escape
from rich import box as rbox

# ---------------------------------------------------------------------------
# Pipeline — ZERO modificacoes analiticas
# ---------------------------------------------------------------------------
import guaraci.pipeline as pq
from guaraci.app_logic import (
    LogThreadSafe as _LogThreadSafe,
    figuras_concluidas as _figuras_concluidas,
    avisos_do_log as _avisos_do_log,
    progresso_do_log as _progresso_do_log,
    fmt_tempo as _fmt_tempo,
)

Config        = pq.Config
executar      = pq.executar
salvar_config = pq.salvar_config
carregar_config = pq.carregar_config

# Dicionarios de i18n/perfis do cli_assistente. Import de pacote normal (mesmo
# pacote): importar o modulo NAO dispara main() (guardado por __main__), entao
# nao ha efeito colateral — o antigo carregamento por caminho (spec_from_file)
# so existia porque os modulos eram scripts soltos na raiz.
import guaraci.cli_assistente as _cli

def _try(name, fallback=None):
    return getattr(_cli, name, fallback if fallback is not None else {})

FIELD_NAMES          = _try("FIELD_NAMES")
HELP_DB              = _try("HELP_DB")
RISK_CLASS           = _try("RISK_CLASS")
PROFILES             = _try("PROFILES")
PROFILE_DESC         = _try("PROFILE_DESC")
PROFILE_KEY_SUMMARY  = _try("PROFILE_KEY_SUMMARY")
PALETAS_COR          = _try("PALETAS_COR")
FONT_PRESETS         = _try("FONT_PRESETS")
TECNICAS             = _try("TECNICAS")
REFERENCIAS_GUARACI  = _try("REFERENCIAS_GUARACI")
_CONFIG_SPEC         = _try("_CONFIG_SPEC", [])
_SPEC_BY_KEY         = _try("_SPEC_BY_KEY")
_DDSIMCA_DISPLAY     = _try("_DDSIMCA_DISPLAY")
_DDSIMCA_INPUT       = _try("_DDSIMCA_INPUT")
_coagir_valor        = _try("_coagir_valor", lambda s, r: r)
_attr_para_yaml      = _try("_attr_para_yaml", lambda s, c: "")
_fmt_yaml            = _try("_fmt_yaml", str)
salvar_config        = _try("salvar_config", pq.salvar_config)
carregar_config      = _try("carregar_config", pq.carregar_config)

def _carregar_visual_cfg() -> dict:
    fn = getattr(_cli, "_carregar_visual_cfg", None)
    return fn() if callable(fn) else {}

def _salvar_visual_cfg(d: dict) -> None:
    fn = getattr(_cli, "_salvar_visual_cfg", None)
    if callable(fn):
        fn(d)

def _carregar_codigos_usuario() -> dict:
    fn = getattr(_cli, "_carregar_codigos_usuario", None)
    return fn() if callable(fn) else {}

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
_BASE_DIR    = Path(os.path.dirname(os.path.abspath(__file__)))
_CFG_PATH    = _BASE_DIR / "config.yaml"
_PERFIS_DIR  = _BASE_DIR / "perfis"
_LANG_FLAG   = _BASE_DIR / ".cli_wizard_done"
_CODIGOS_PATH= _BASE_DIR / "codigos_usuario.json"

# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
_STATE: Dict[str, Any] = {"lang": "PT"}

def _lang() -> str:
    return _STATE["lang"]

def _set_lang(l: str) -> None:
    _STATE["lang"] = l
    try:
        _LANG_FLAG.write_text(l, encoding="utf-8")
    except OSError:
        pass

def _toggle_idioma() -> str:
    novo = "EN" if _lang() == "PT" else "PT"
    _set_lang(novo)
    return novo

# ---------------------------------------------------------------------------
# PALETA + tema + console: fonte unica em guaraci_theme (compartilhada com o
# cli_assistente para visual identico). Nomes reexportados sem alteracao.
# ---------------------------------------------------------------------------
from guaraci.guaraci_theme import (  # noqa: E402
    PA, PF, PS, PR, PW, PM, PD, PG, console, _W,
)
# Lógica pura extraída da CLI (item 19): testável sem Rich/console. Ver cli_logic.py.
from guaraci.cli_logic import (  # noqa: E402
    trunc as _trunc,
    truncar_desc_por_frase as _truncar_desc_por_frase,
    fmt_bool as _fmt_bool_puro,
    validar_faixas as _validar_faixas_puro,
    contar_dx as _contar_dx,
)

# Estado global da tecnica selecionada (persiste entre menus)
_TECNICA_SELECIONADA: Dict[str, str] = {"key": "ft-nir", "nome": "FT-NIR"}

# ---------------------------------------------------------------------------
# Internacionalizacao — sem repeticao entre idiomas
# ---------------------------------------------------------------------------
I18N: Dict[str, Dict[str, str]] = {
    "PT": {
        # Titulos de menu
        "t_projeto":    "Projeto",
        "t_dados":      "Dados",
        "t_preproc":    "Pre-processamento",
        "t_modelagem":  "Modelagem",
        "t_validacao":  "Validacao",
        "t_avancado":   "Metodos Avancados",
        "t_viz":        "Visualizacao",
        "t_tecnica":    "Tecnica Analitica",
        "t_codigos":    "Codificacao DX",
        "t_hardware":   "Hardware",
        "t_perfis":     "Perfis Prontos",
        "t_predicao":   "Predicao em Lote",
        "t_idioma":     "Idioma",
        "t_ajuda":      "Ajuda",
        # Descricoes de secao (curtas)
        "d_projeto":    "Pastas de entrada e saida.",
        "d_dados":      "Formato, faixa espectral e classes.",
        "d_preproc":    "MSC / SNV, Savitzky-Golay, centragem.",
        "d_modelagem":  "PLS-DA, OPLS-DA e DD-SIMCA.",
        "d_validacao":  "GroupKFold, holdout e permutacoes.",
        "d_avancado":   "Benchmark, Monte Carlo e SHAP.",
        "d_viz":        "DPI, formato, paleta e graficos extras.",
        "d_tecnica":    "Tecnica especifica com faixas automaticas.",
        "d_codigos":    "Nomenclatura JCAMP-DX e especies.",
        "d_hardware":   "Capacidade e perfil recomendado.",
        "d_perfis":     "Configuracoes prontas para uso.",
        "d_ajuda":      "Documentacao interativa por campo.",
        # Grupos do menu principal
        "grp_config":   "Configuracao da Analise",
        "grp_analise":  "Analise e Visualizacao",
        "grp_sistema":  "Sistema",
        "grp_execucao": "Execucao",
        # Acoes
        "rodar":        "Rodar Pipeline",
        "salvar":       "Salvar Perfil",
        "carregar":     "Carregar Perfil",
        "nome_saida":   "Nome da Saida",
        "sair":         "Sair",
        # Status
        "status_ok":    "Pronto",
        "status_erro":  "Configurar dados",
        "dados_ok":     "{n} arquivos .dx",
        "dados_err":    "Pasta invalida",
        # Interacao
        "opcao":        "Opcao",
        "voltar":       "Voltar",
        "continuar":    "Enter para continuar",
        "cancelado":    "Cancelado.",
        "invalido":     "Opcao invalida.",
        "mantido":      "Mantido.",
        "novo_valor":   "Novo valor (Enter=manter, ?=ajuda): ",
        "confirmar":    "Confirmar? (s/n): ",
        "conf_anal":    "Alteracao afeta resultados — confirmar? (s/n): ",
        "atualizado":   "Atualizado: {campo} = {valor}",
        # Checklist
        "chk_dados":    "Dados carregados",
        "chk_csv":      "CSV localizado",
        "chk_leak":     "Anti-leakage ativo",
        "chk_saida":    "Pasta de saida definida",
        "chk_hw":       "Hardware compativel",
        "chk_preproc":  "Pre-processamento definido",
        "chk_err_dados":"Pasta de dados nao encontrada",
        "chk_err_csv":  "Arquivo CSV nao encontrado",
        "chk_err_leak": "GroupKFold DESATIVADO — risco de leakage",
        "chk_warn_hw":  "RAM baixa com modulos pesados ativos",
        # Hardware
        "hw_alto":      "Alto Desempenho",
        "hw_medio":     "Desempenho Medio",
        "hw_basico":    "Desempenho Basico",
        "hw_limitado":  "Limitado",
        "hw_rec":       "Perfil recomendado",
        "hw_cpu":       "CPU fisicos",
        "hw_threads":   "Threads logicos",
        "hw_ram_total": "RAM total",
        "hw_ram_livre": "RAM livre",
        "hw_disco":     "Disco livre",
        # Execucao
        "exec_inicio":  "Iniciando analise...",
        "exec_leitura": "Leitura dos espectros",
        "exec_preproc": "Pre-processamento",
        "exec_pca":     "PCA + HCA",
        "exec_plsda":   "PLS-DA",
        "exec_opls":    "OPLS-DA",
        "exec_dds":     "DD-SIMCA",
        "exec_valid":   "Validacao estatistica",
        "exec_relat":   "Relatorios e figuras",
        "exec_bench":   "Benchmark (SVM/RF/XGB)",
        "exec_mc":      "Monte Carlo CV",
        "exec_concluido": "Analise concluida",
        "exec_erro":    "Erro na execucao",
        "exec_saida":   "Resultados salvos em",
        "exec_interrompido": "Interrompido pelo usuario",
        "exec_objetivo": "Objetivo",
        "exec_eta":      "Tempo estimado restante",
        "exec_figuras":  "Figuras geradas",
        "exec_avisos":   "Avisos",
        "exec_sem_avisos": "Nenhum aviso ate agora",
        # Resumo cientifico
        "res_tecnica":  "Tecnica",
        "res_preproc":  "Pre-processamento",
        "res_modelo":   "Modelo principal",
        "res_lvs":      "Max. variaveis latentes",
        "res_valid":    "Validacao",
        "res_perm":     "Permutacoes",
        "res_opls":     "OPLS-DA",
        "res_dds":      "DD-SIMCA",
        "res_bench":    "Benchmark",
        "res_mc":       "Monte Carlo",
        "res_shap":     "SHAP",
        "res_dpi":      "Resolucao (DPI)",
        "res_fmt":      "Formato figura",
        "res_nivel":    "Nivel",
        "res_tag":      "Identificador",
        # Nome de pasta
        "tag_atual":    "Identificador atual",
        "tag_novo":     "Novo identificador (Enter=manter, ?=limpar): ",
        "tag_limpo":    "Identificador removido — proximo run usa timestamp.",
        # Perfis
        "perf_tempo":   "Tempo",
        "perf_uso":     "Indicado para",
        # Codigos
        "cod_cadastrar":"Cadastrar novo codigo",
        "cod_listar":   "Listar todos os codigos",
        "cod_novo_cod": "Codigo (2-4 letras maiusculas, ex: MAN): ",
        "cod_novo_esp": "Nome da especie para '{cod}': ",
        "cod_salvo":    "Cadastrado: {cod} = {esp}",
        "cod_invalido": "Codigo invalido. Use 2-4 letras maiusculas.",
        # Visualizacao
        "viz_paleta":   "Paleta de Cores",
        "viz_fonte":    "Tamanho de Fonte",
        "viz_grid":     "Grade",
        "viz_alpha":    "Transparencia dos Pontos",
        # Pipeline info
        "pip_sem_dados": "Corrija a pasta de dados antes de rodar.",
        # Guaraci fala
        "g_prefixo":    "Guaraci:",
    },
    "EN": {
        "t_projeto":    "Project",
        "t_dados":      "Data",
        "t_preproc":    "Preprocessing",
        "t_modelagem":  "Modelling",
        "t_validacao":  "Validation",
        "t_avancado":   "Advanced Methods",
        "t_viz":        "Visualization",
        "t_tecnica":    "Analytical Technique",
        "t_codigos":    "DX Encoding",
        "t_hardware":   "Hardware",
        "t_perfis":     "Ready Profiles",
        "t_predicao":   "Batch Prediction",
        "t_idioma":     "Language",
        "t_ajuda":      "Help",
        "d_projeto":    "Input and output folders.",
        "d_dados":      "Format, spectral range and classes.",
        "d_preproc":    "MSC / SNV, Savitzky-Golay, centering.",
        "d_modelagem":  "PLS-DA, OPLS-DA and DD-SIMCA.",
        "d_validacao":  "GroupKFold, holdout and permutations.",
        "d_avancado":   "Benchmark, Monte Carlo and SHAP.",
        "d_viz":        "DPI, format, palette and extra plots.",
        "d_tecnica":    "Specific technique with automatic ranges.",
        "d_codigos":    "JCAMP-DX naming and species codes.",
        "d_hardware":   "Capacity and recommended profile.",
        "d_perfis":     "Ready-to-use configurations.",
        "d_ajuda":      "Interactive field documentation.",
        "grp_config":   "Analysis Configuration",
        "grp_analise":  "Analysis & Visualization",
        "grp_sistema":  "System",
        "grp_execucao": "Execution",
        "rodar":        "Run Pipeline",
        "salvar":       "Save Profile",
        "carregar":     "Load Profile",
        "nome_saida":   "Output Name",
        "sair":         "Exit",
        "status_ok":    "Ready",
        "status_erro":  "Configure data",
        "dados_ok":     "{n} .dx files",
        "dados_err":    "Invalid folder",
        "opcao":        "Option",
        "voltar":       "Back",
        "continuar":    "Press Enter to continue",
        "cancelado":    "Cancelled.",
        "invalido":     "Invalid option.",
        "mantido":      "Kept.",
        "novo_valor":   "New value (Enter=keep, ?=help): ",
        "confirmar":    "Confirm? (y/n): ",
        "conf_anal":    "This changes results — confirm? (y/n): ",
        "atualizado":   "Updated: {campo} = {valor}",
        "chk_dados":    "Data loaded",
        "chk_csv":      "CSV located",
        "chk_leak":     "Anti-leakage active",
        "chk_saida":    "Output folder defined",
        "chk_hw":       "Compatible hardware",
        "chk_preproc":  "Preprocessing defined",
        "chk_err_dados":"Data folder not found",
        "chk_err_csv":  "CSV file not found",
        "chk_err_leak": "GroupKFold DISABLED — leakage risk",
        "chk_warn_hw":  "Low RAM with heavy modules active",
        "hw_alto":      "High Performance",
        "hw_medio":     "Medium Performance",
        "hw_basico":    "Basic Performance",
        "hw_limitado":  "Limited",
        "hw_rec":       "Recommended profile",
        "hw_cpu":       "Physical CPUs",
        "hw_threads":   "Logical threads",
        "hw_ram_total": "Total RAM",
        "hw_ram_livre": "Free RAM",
        "hw_disco":     "Free disk",
        "exec_inicio":  "Starting analysis...",
        "exec_leitura": "Reading spectra",
        "exec_preproc": "Preprocessing",
        "exec_pca":     "PCA + HCA",
        "exec_plsda":   "PLS-DA",
        "exec_opls":    "OPLS-DA",
        "exec_dds":     "DD-SIMCA",
        "exec_valid":   "Statistical validation",
        "exec_relat":   "Reports and figures",
        "exec_bench":   "Benchmark (SVM/RF/XGB)",
        "exec_mc":      "Monte Carlo CV",
        "exec_concluido": "Analysis completed",
        "exec_erro":    "Pipeline error",
        "exec_saida":   "Results saved in",
        "exec_interrompido": "Interrupted by user",
        "exec_objetivo": "Objective",
        "exec_eta":      "Estimated time remaining",
        "exec_figuras":  "Figures generated",
        "exec_avisos":   "Warnings",
        "exec_sem_avisos": "No warnings so far",
        "res_tecnica":  "Technique",
        "res_preproc":  "Preprocessing",
        "res_modelo":   "Main model",
        "res_lvs":      "Max. latent variables",
        "res_valid":    "Validation",
        "res_perm":     "Permutations",
        "res_opls":     "OPLS-DA",
        "res_dds":      "DD-SIMCA",
        "res_bench":    "Benchmark",
        "res_mc":       "Monte Carlo",
        "res_shap":     "SHAP",
        "res_dpi":      "Resolution (DPI)",
        "res_fmt":      "Figure format",
        "res_nivel":    "Level",
        "res_tag":      "Run ID",
        "tag_atual":    "Current ID",
        "tag_novo":     "New ID (Enter=keep, ?=clear): ",
        "tag_limpo":    "ID cleared — next run uses timestamp.",
        "perf_tempo":   "Time",
        "perf_uso":     "Best for",
        "cod_cadastrar":"Register new code",
        "cod_listar":   "List all codes",
        "cod_novo_cod": "Code (2-4 uppercase letters, e.g. MAN): ",
        "cod_novo_esp": "Species name for '{cod}': ",
        "cod_salvo":    "Registered: {cod} = {esp}",
        "cod_invalido": "Invalid code. Use 2-4 uppercase letters.",
        "viz_paleta":   "Color Palette",
        "viz_fonte":    "Font Size",
        "viz_grid":     "Grid",
        "viz_alpha":    "Point Transparency",
        "pip_sem_dados": "Fix data folder before running.",
        "g_prefixo":    "Guaraci:",
    },
}

def _t(key: str, **kw) -> str:
    s = I18N[_lang()].get(key, key)
    if kw:
        try:
            s = s.format(**kw)
        except (KeyError, IndexError):
            pass
    return s

# ---------------------------------------------------------------------------
# GUARACI TIPS — dicas unicas, diferentes das descricoes do HELP_DB
# ---------------------------------------------------------------------------
GUARACI_TIPS: Dict[str, Dict[str, str]] = {
    "pasta_dados": {
        "PT": "Use caminho absoluto para evitar problemas. Verifique se os .dx estao na raiz da pasta, nao em subpastas.",
        "EN": "Use absolute paths to avoid issues. Check that .dx files are at the folder root, not in subfolders.",
    },
    "pasta_saida": {
        "PT": "Se a pasta nao existir ela sera criada. Use nomes sem espacos para compatibilidade com LaTeX.",
        "EN": "The folder will be created if it does not exist. Avoid spaces for LaTeX compatibility.",
    },
    "tag": {
        "PT": "Identifique rodadas diferentes com tags como 'artigo_v2' ou 'tcc_final'. Facilita comparar resultados.",
        "EN": "Tag runs like 'paper_v2' or 'thesis_final'. Makes it easy to compare results across runs.",
    },
    "modo_entrada": {
        "PT": "Para óleos amazônicos com arquivos do espectrômetro: use 'dx'. CSV é para dados tabelados de outras fontes.",
        "EN": "For Amazonian oils from the spectrometer: use 'dx'. CSV is for tabular data from other sources.",
    },
    "pre_processamento": {
        "PT": "Para FT-NIR de óleos vegetais, MSC+SG+MC deu Bal.Acc=0.92 no benchmark. Autoscaling isolado caiu para 0.47.",
        "EN": "For vegetable oil FT-NIR, MSC+SG+MC achieved Bal.Acc=0.92 in benchmark. Autoscaling alone dropped to 0.47.",
    },
    "comparar_pre_processamentos": {
        "PT": "Ativa teste de todos os 6 pipelines. Use apenas uma vez para descobrir o melhor — depois fixe e desative.",
        "EN": "Tests all 6 pipelines. Use once to find the best one — then fix it and disable this.",
    },
    "nivel": {
        # Nome amigavel primeiro, codigo interno (N1/N2/N3) entre parenteses
        # so' como referencia tecnica -- mesma convencao de _rotulo_opcao()
        # (P8). Corrigido em 2026-07-13: antes o codigo vinha colado ao
        # nome sem essa hierarquia, unico ponto do assistente "G" que
        # ainda vazava N1/N2/N3 como termo primario.
        "PT": "Classificação por espécie (N1) para uma checagem rápida, Discriminação puro/adulterado (N2) para TCC/publicação (autenticação completa), Quantificação de teor (N3) só se tiver tempo (pode levar horas).",
        "EN": "Species classification (N1) for a quick check, Pure/adulterated discrimination (N2) for papers/thesis (full authentication), Content quantification (N3) only if you have time (may take hours).",
    },
    "max_lvs": {
        "PT": "O criterio de Wold para automaticamente antes do maximo. Comece com 40; suba se o modelo nao convergir.",
        "EN": "Wold's criterion stops automatically before the max. Start with 40; increase if the model does not converge.",
    },
    "opls_da": {
        "PT": "Gera o S-plot, essencial para publicacão em Talanta e Food Chemistry. Adiciona ~2 min.",
        "EN": "Generates the S-plot, essential for Talanta and Food Chemistry publications. Adds ~2 min.",
    },
    "ddsimca": {
        "PT": "Cria modelo de autenticacao por especie. Rejeita amostras suspeitas com elipse UCL 95%. Essencial para fraude.",
        "EN": "Creates per-species authentication model. Rejects suspicious samples with 95% UCL ellipse. Essential for fraud detection.",
    },
    "holdout_fracao": {
        "PT": "0.2 é o padrão seguro. Se o dataset for pequeno (<200 amostras), use 0.15 para ter mais dados de treino.",
        "EN": "0.2 is the safe default. For small datasets (<200 samples), use 0.15 to keep more training data.",
    },
    "validacao_group_aware": {
        "PT": "NUNCA desative. Com triplicatas (T1,T2,T3), o KFold simples vaza informacao. GroupKFold evita isso.",
        "EN": "NEVER disable. With triplicates (T1,T2,T3), plain KFold leaks information. GroupKFold prevents this.",
    },
    "n_permutacoes": {
        "PT": "200 é suficiente para TCC. 500 para artigo, 1000 para tese. Dobrar o N demora 2x mais.",
        "EN": "200 is enough for undergraduate thesis. 500 for papers, 1000 for dissertations. Doubling N doubles time.",
    },
    "benchmark": {
        "PT": "Compara PLS-DA com SVM, RF e XGBoost. Se o PLS-DA ganhar, o argumento de interpretabilidade é mais forte.",
        "EN": "Compares PLS-DA with SVM, RF and XGBoost. If PLS-DA wins, the interpretability argument is stronger.",
    },
    "monte_carlo": {
        "PT": "Calcula IC95% real para cada metrica. Muito mais robusto que uma unica divisao treino/teste.",
        "EN": "Calculates real 95% CI for each metric. Far more robust than a single train/test split.",
    },
    "shap_benchmark": {
        "PT": "Mostra quais regioes espectrais o Random Forest usa. Compare com o VIP do PLS-DA para validacao cruzada.",
        "EN": "Shows which spectral regions Random Forest uses. Cross-check with PLS-DA VIP for validation.",
    },
    "dpi": {
        "PT": "300 para TCC, 600 para revista cientifica (Nature exige 300-600 DPI). Maior DPI = arquivo maior.",
        "EN": "300 for thesis, 600 for journals (Nature requires 300-600 DPI). Higher DPI = larger file.",
    },
    "formato_figura": {
        "PT": "PDF para LaTeX (vetorial), PNG para Word/PowerPoint, SVG para editar no Inkscape.",
        "EN": "PDF for LaTeX (vector), PNG for Word/PowerPoint, SVG to edit in Inkscape.",
    },
    "faixa_min_cm": {
        "PT": "Para FT-NIR de óleos: 4000 cm-1. Se cortar regiao ruidosa, aumente o minimo para 4500.",
        "EN": "For vegetable oil FT-NIR: 4000 cm-1. To cut noisy regions, raise the minimum to 4500.",
    },
    "faixa_max_cm": {
        "PT": "Para FT-NIR de óleos: 10000 cm-1. Diminua para 9000 se a regiao acima for ruidosa.",
        "EN": "For vegetable oil FT-NIR: 10000 cm-1. Lower to 9000 if the region above is noisy.",
    },
    "n_monte_carlo": {
        "PT": "100 ja e representativo. Acima de 300, o ganho de precisao e minimo mas o tempo triplica.",
        "EN": "100 is already representative. Above 300, precision gain is minimal but time triples.",
    },
    "shap_max_amostras": {
        "PT": "Mantenha em 500. Acima disso, Random Forest com 14 classes pode consumir mais de 4GB de RAM.",
        "EN": "Keep at 500. Above that, Random Forest with 14 classes may consume over 4GB of RAM.",
    },
}

# ---------------------------------------------------------------------------
# Identidade visual — icones e cores por risco
# ---------------------------------------------------------------------------
_RISK_HEX  = {"VISUAL": PF, "ANALITICO": PA, "AVANCADO": PR}
_RISK_ICON = {"VISUAL": "●", "ANALITICO": "◆", "AVANCADO": "▲"}
_RISK_MARK = {"VISUAL": "○", "ANALITICO": "◆", "AVANCADO": "▲"}  # icon inline

def _risco_hex(key: str) -> str:
    return _RISK_HEX.get(RISK_CLASS.get(key, "ANALITICO"), PA)

def _risco_icon(key: str) -> str:
    return _RISK_ICON.get(RISK_CLASS.get(key, "ANALITICO"), "◆")

# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------
def cls() -> None:
    os.system("cls" if os.name == "nt" else "clear")

def _input(msg: str = "", default: str = "") -> str:
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        return default

def _ask(prompt_markup: str = "", default: str = "") -> str:
    """Le entrada exibindo um prompt COM markup Rich (cores) na mesma linha.

    Use esta funcao sempre que o prompt tiver cores/teclas coloridas — o
    input() puro nao interpreta markup e o exibiria literal (ex.: [#B8963E]).
    """
    try:
        if prompt_markup:
            console.print(prompt_markup, end="")
        return input().strip()
    except (EOFError, KeyboardInterrupt):
        return default

def _pause(msg: str = "") -> None:
    lbl = msg or _t("continuar")
    console.print(f"  [{PM}][{escape(lbl)}][/{PM}]", end="")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass

def _nome_campo(key: str) -> str:
    return FIELD_NAMES.get(key, {}).get(_lang(), key)

def _rotulo_opcao(key: str, op: Any) -> str:
    """Nome amigavel de um valor de opcao (so exibicao; o valor gravado no
    config continua o codigo interno, ex.: N1/N2/N3, puros/todos)."""
    if key == "nivel":
        # Lidera com o nome amigavel (P8: aposentar N1/N2/N3 como termo
        # PRIMARIO na UI); o codigo interno fica entre parenteses so' como
        # referencia tecnica, nunca como o rotulo principal exibido ao usuario.
        nome = pq._NIVEL_NOME.get(str(op), "")
        return f"{nome} ({op})" if nome else str(op)
    if key == "modo_ddsimca":
        return _DDSIMCA_DISPLAY.get(_lang(), {}).get(str(op), str(op))
    return str(op)


def _get_val(cfg: Config, key: str) -> Any:
    spec = _SPEC_BY_KEY.get(key)
    if spec is None:
        return getattr(cfg, key, "?")
    raw = _attr_para_yaml(spec, cfg)
    if key == "modo_ddsimca":
        return _DDSIMCA_DISPLAY.get(_lang(), {}).get(str(raw), raw)
    if key == "nivel":
        return _rotulo_opcao(key, raw)
    return raw

def _set_val(cfg: Config, key: str, raw: str) -> None:
    spec = _SPEC_BY_KEY[key]
    if key == "modo_ddsimca":
        interno = _DDSIMCA_INPUT.get(raw.lower().strip())
        if interno is None:
            raise ValueError(f"Valor invalido: '{raw}'")
        raw = interno
    valor = _coagir_valor(spec, raw)
    setattr(cfg, spec["attr"], valor)

def _cfgv(cfg: Config, key: str, default: Any = None) -> Any:
    """Le um valor do Config pela KEY do _CONFIG_SPEC, resolvendo o atributo real.

    Evita o erro comum de usar `getattr(cfg, "benchmark")` quando o atributo
    real e `executar_benchmark`. Sempre use esta funcao para ler config por key.
    """
    spec = _SPEC_BY_KEY.get(key)
    attr = spec["attr"] if spec else key
    return getattr(cfg, attr, default)


# _contar_dx importada de guaraci.cli_logic (item 19) — ver topo do arquivo.

def _fmt_bool(v: Any, lang: str = "") -> str:
    """Wrapper fino: resolve o idioma ativo (ou usa o passado) e delega o
    formato Sim/Não a `guaraci.cli_logic.fmt_bool` (função pura, testada)."""
    l = lang or _lang()
    if isinstance(v, bool):
        return _fmt_bool_puro(v, l)
    return escape(str(v))

# ---------------------------------------------------------------------------
# MENSAGENS DINAMICAS — boas-vindas, despedida, pausa para cafe
# ---------------------------------------------------------------------------
_BOAS_VINDAS = {
    "PT": [
        "Bem-vindo de volta ao GUARACI. Os seus dados aguardam analise. ☀",
        "GUARACI inicializado. Ciencia comeca agora. ☀",
        "Ola, pesquisador. O GUARACI esta pronto para revelar padroes. ☀",
        "Sistema iniciado. Que a analise seja precisa e os resultados, claros. ☀",
    ],
    "EN": [
        "Welcome back to GUARACI. Your data awaits analysis. ☀",
        "GUARACI initialized. Science starts now. ☀",
        "Hello, researcher. GUARACI is ready to reveal patterns. ☀",
        "System started. May the analysis be precise and results clear. ☀",
    ],
}

_DESPEDIDAS = {
    "PT": [
        "Ate logo! Que seus modelos tenham alta acuracia. ☀",
        "Encerrando GUARACI. Boas analises, pesquisador!",
        "Saindo. Os resultados estao salvos e a sua disposicao. ☀",
        "Ate a proxima sessao. A ciencia continua! ☀",
    ],
    "EN": [
        "Goodbye! May your models have high accuracy. ☀",
        "Closing GUARACI. Good analyses, researcher!",
        "Exiting. Results are saved and ready for you. ☀",
        "Until next session. Science continues! ☀",
    ],
}

_PAUSAS_CAFE = {
    "PT": [
        "Analise em andamento — otimo momento para um cafe. ☕",
        "Processando espectros... aproveite para um intervalo. ☕",
        "O pipeline esta trabalhando. Voce merece uma pausa. ☕",
        "Modelos sendo calculados. Cafe quentinho esperando? ☕",
    ],
    "EN": [
        "Analysis in progress — great time for a coffee break. ☕",
        "Processing spectra... enjoy a short break. ☕",
        "Pipeline running. You deserve a pause. ☕",
        "Models being computed. Coffee time? ☕",
    ],
}

def _exibir_boas_vindas() -> None:
    """Exibe mensagem aleatoria de boas-vindas (apenas na inicializacao)."""
    import random
    import time as _time
    msg = random.choice(_BOAS_VINDAS[_lang()])
    console.print()
    console.print(Panel(
        Align.center(Text(f"\n  {msg}\n", style=f"italic {PA}")),
        border_style=PF, box=rbox.ROUNDED, padding=(0, 2), width=_W()
    ))
    _time.sleep(1.0)

def _exibir_despedida() -> None:
    """Exibe mensagem de despedida ao sair do programa."""
    import random
    import time as _time
    msg = random.choice(_DESPEDIDAS[_lang()])
    console.print()
    console.print(Panel(
        Align.center(Text(f"\n  {msg}\n", style=f"italic {PS}")),
        border_style=PF, box=rbox.ROUNDED, padding=(0, 2), width=_W()
    ))
    _time.sleep(0.6)

def _sugerir_cafe() -> None:
    """Exibe sugestao de cafe durante execucoes longas."""
    import random
    msg = random.choice(_PAUSAS_CAFE[_lang()])
    console.print(f"\n  [{PA}]{msg}[/{PA}]")

# ---------------------------------------------------------------------------
# VALIDACAO DE INTEGRIDADE — faixas e paleta antes de rodar
# ---------------------------------------------------------------------------
def _validar_faixas(cfg: Config) -> list:
    """Wrapper fino: le faixa_min/max do Config e delega a validacao pura
    a `guaraci.cli_logic.validar_faixas` (testada)."""
    f_min = _cfgv(cfg, "faixa_min_cm", 400)
    f_max = _cfgv(cfg, "faixa_max_cm", 4000)
    return _validar_faixas_puro(f_min, f_max)

def _sincronizar_dpi(cfg: Config) -> None:
    """Garante que cfg.dpi reflita o visual_config.json antes de rodar."""
    vcfg = _carregar_visual_cfg()
    dpi_v = vcfg.get("dpi")
    if dpi_v:
        try:
            setattr(cfg, "dpi", int(dpi_v))
        except (TypeError, ValueError):
            pass

# ---------------------------------------------------------------------------
# ASSISTENTE GUARACI — tecla G em qualquer tela
# ---------------------------------------------------------------------------
def _guaraci_revisar_config(cfg: Config) -> None:
    """Exibe revisao da configuracao atual em linguagem natural."""
    lang = _lang()
    linhas = [f"[{PA}]Configuracao atual detectada:[/{PA}]", ""]

    preproc  = _cfgv(cfg, "pre_processamento",     "msc_sg_mc")
    max_lvs  = _cfgv(cfg, "max_lvs",               40)
    mc       = _cfgv(cfg, "monte_carlo",           False)
    shap     = _cfgv(cfg, "shap_benchmark",        False)
    n_perm   = _cfgv(cfg, "n_permutacoes",         200)
    opls     = _cfgv(cfg, "opls_da",               True)
    dds      = _cfgv(cfg, "ddsimca",               True)
    holdout  = _cfgv(cfg, "holdout_fracao",        0.20)
    bench    = _cfgv(cfg, "benchmark",             False)

    ok  = f"[{PG}]✓[/{PG}]"
    av  = f"[{PA}]⚑[/{PA}]"
    inf = f"[{PS}]ℹ[/{PS}]"

    linhas.append(f"  {ok if preproc != 'raw' else av} Pre-proc: {preproc}" +
                  (f"  [{PM}](Para FT-NIR, msc+sg+mc = Bal.Acc 0.923)[/{PM}]" if preproc == "raw" else ""))
    linhas.append(f"  {ok if max_lvs <= 40 else av} max_lvs = {max_lvs}" +
                  (f"  [{PM}](>40 aumenta risco de overfitting)[/{PM}]" if max_lvs > 40 else ""))
    linhas.append(f"  {ok if 100 <= n_perm else av} n_permutacoes = {n_perm}" +
                  (f"  [{PM}](<100 e fraco para publicacao)[/{PM}]" if n_perm < 100 else ""))
    holdout_pct = int(holdout * 100)
    ok_h = 0.15 <= holdout <= 0.35
    linhas.append(f"  {ok if ok_h else av} holdout = {holdout_pct}%" +
                  ("" if ok_h else f"  [{PM}](ideal: 15-35%)[/{PM}]"))
    if mc:
        linhas.append(f"  {inf} Monte Carlo ativo — analise mais robusta, tempo maior.")
    if shap:
        linhas.append(f"  {inf} SHAP ativo — interpretabilidade aumentada, tempo maior.")
    if bench:
        linhas.append(f"  {inf} Benchmark SVM/RF/XGB ativo.")
    if not opls:
        linhas.append(f"  [{PM}]ℹ OPLS-DA desativado.[/{PM}]")
    if not dds:
        linhas.append(f"  [{PM}]ℹ DD-SIMCA desativado.[/{PM}]")

    avisos_faixa = _validar_faixas(cfg)
    for av_msg in avisos_faixa:
        linhas.append(f"  [{PR}]✖ {av_msg}[/{PR}]")

    _titulo_rev = ("Revisao da Configuracao" if _lang() == "PT"
                   else "Configuration Review")
    console.print(Panel(
        "\n".join(linhas),
        title=f"[{PA}]☀ {_titulo_rev}[/{PA}]",
        border_style=PA, box=rbox.ROUNDED, padding=(1, 2), width=_W()
    ))
    _pause()

def _guaraci_navegar_secoes(cfg: Config) -> None:
    """Lista as secoes e exibe descricao quando selecionada."""
    lang = _lang()
    secoes = {
        "1": ("Projeto",           "Pastas de entrada/saida e nome da execucao."),
        "2": ("Dados",             "Formato espectral, faixa de comprimento de onda e classes."),
        "3": ("Pre-processamento", "Pipeline espectral — recomendado msc+sg+mc para FT-NIR."),
        "4": ("Modelagem",         "PLS-DA, OPLS-DA, DD-SIMCA e selecao de variaveis."),
        "5": ("Validacao",         "Holdout, permutacoes, Wold e CV-ANOVA."),
        "6": ("Avancado",          "Benchmark, Monte Carlo CV e SHAP (aumentam tempo de execucao)."),
        "7": ("Visualizacao",      "DPI, paleta, formato de figura e grid."),
        "8": ("Tecnica Analitica", "Selecione a tecnica espectroscopica e ajuste faixas."),
    }
    console.print()
    t = Table(show_header=False, box=rbox.SIMPLE, padding=(0, 1))
    t.add_column("N", style=PA, width=4)
    t.add_column("Secao", style=PW, width=22)
    t.add_column("Descricao", style=PM)
    for k, (nome, desc) in secoes.items():
        t.add_row(k, nome, desc)
    console.print(t)
    raw = _ask(f"  [{PA}]Selecione ([0] voltar): [/{PA}]").strip()
    if raw in secoes:
        nome, desc = secoes[raw]
        console.print(Panel(
            f"[{PW}]{desc}[/{PW}]",
            title=f"[{PA}]{nome}[/{PA}]",
            border_style=PF, box=rbox.ROUNDED, padding=(0, 2), width=_W()
        ))
        _pause()

def _abrir_assistente(contexto: str = "", cfg: Optional[Config] = None) -> None:
    """Abre o Assistente Guaraci (tecla G em qualquer tela)."""
    lang = _lang()
    cls(); _print_header()
    console.print()

    opcoes = [
        ("1", "Revisar configuracao atual" if lang=="PT" else "Review current configuration"),
        ("2", "Informacoes sobre uma secao" if lang=="PT" else "Information about a section"),
        ("Q", "Fechar assistente"           if lang=="PT" else "Close assistant"),
    ]
    t = Table(show_header=False, box=rbox.SIMPLE, padding=(0, 1))
    t.add_column("Tecla", style=PA, width=6)
    t.add_column("Opcao", style=PW)
    for k, v in opcoes:
        t.add_row(f"[{k}]", v)

    titulo_ctx = (f"  Chamado de: {contexto}" if contexto else "")
    console.print(Panel(
        Group(
            Text(f"\n  Ola, pesquisador! Como posso ajudar?\n{titulo_ctx}\n",
                 style=f"italic {PS}"),
            t,
        ),
        title=f"[bold {PA}]☀ GUARACI — Assistente Cientifico[/bold {PA}]",
        border_style=PA, box=rbox.ROUNDED, padding=(1, 2), width=_W()
    ))

    raw = _ask(f"  [{PA}]Opcao: [/{PA}]").strip().upper()
    if raw == "1" and cfg is not None:
        _guaraci_revisar_config(cfg)
    elif raw == "2":
        _guaraci_navegar_secoes(cfg or Config())

# ---------------------------------------------------------------------------
# CABECALHO COMPACTO
# ---------------------------------------------------------------------------
def _print_header() -> None:
    # Titulo com icone solar flanqueando GUARACI
    titulo = Text(justify="center")
    titulo.append("  ", style=f"{PA}")
    titulo.append("GUARACI", style=f"bold {PA}")
    titulo.append("  ", style=f"{PA}")

    # Tecnica ativa (atualizada dinamicamente)
    tec_nome = _TECNICA_SELECIONADA.get("nome", "FT-NIR")
    tec_str  = f"Tecnica: {tec_nome}" if _lang() == "PT" else f"Technique: {tec_nome}"

    sub = Text(
        "Inteligencia Quimiometrica para Matrizes Amazonicas"
        if _lang() == "PT" else
        "Chemometric Intelligence for Amazonian Matrices",
        style=PS, justify="center"
    )
    rod_txt = f"Quimiometria  |  Machine Learning  |  GEAAp / UFPA  |  {tec_str}"
    rod = Text(rod_txt, style=PM, justify="center")
    console.print(Panel(
        Align.center(Group(titulo, sub, rod)),
        border_style=PA, box=rbox.DOUBLE, padding=(0, 2)
    ))

# ---------------------------------------------------------------------------
# BARRA DE STATUS — compacta, 2 linhas
# ---------------------------------------------------------------------------
def _print_status(cfg: Config) -> None:
    lang = _lang()
    pasta = _cfgv(cfg, "pasta_dados", "dados")
    pasta_ok = bool(pasta) and os.path.isdir(str(pasta))
    n_dx = _contar_dx(pasta) if pasta_ok else 0

    if pasta_ok and n_dx > 0:
        dados_str = f"[g]{_t('dados_ok', n=n_dx)}[/g]"
    elif pasta_ok:
        dados_str = f"[warn]{escape(str(pasta))}[/warn]"
    else:
        dados_str = f"[err]{_t('dados_err')}[/err]"

    preproc  = escape(str(_cfgv(cfg, "pre_processamento", "msc_sg_mc")))
    # Barra de status compacta: usa so a palavra-chave do modo (Classificacao/
    # Discriminacao/Quantificacao) — o rotulo completo estouraria as colunas.
    _niv_raw  = str(_cfgv(cfg, "nivel", "N1"))
    _niv_nome = pq._NIVEL_NOME.get(_niv_raw, "")
    nivel    = escape(_niv_nome.split()[0] if _niv_nome else _niv_raw)
    tag      = escape(str(getattr(cfg, "tag", "") or ""))
    pasta_s  = escape(str(pasta))

    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        hw_str = (
            f"[g]{_t('hw_alto')}[/g]" if ram_gb >= 16 else
            f"[warn]{_t('hw_medio')}[/warn]" if ram_gb >= 8 else
            f"[warn]{_t('hw_basico')}[/warn]" if ram_gb >= 4 else
            f"[err]{_t('hw_limitado')}[/err]"
        )
    except ImportError:
        hw_str = "[m]N/A[/m]"

    status_str = (
        f"[g]{_t('status_ok')} ({_t('dados_ok', n=n_dx)})[/g]" if pasta_ok and n_dx > 0
        else f"[err]{_t('status_erro')}[/err]"
    )

    tec_nome = _TECNICA_SELECIONADA.get("nome", "FT-NIR")

    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column("L1", style=PM, width=10, no_wrap=True)
    t.add_column("V1", no_wrap=True, min_width=14)
    t.add_column("L2", style=PM, width=10, no_wrap=True)
    t.add_column("V2", no_wrap=True, min_width=14)
    t.add_column("L3", style=PM, width=10, no_wrap=True)
    t.add_column("V3", no_wrap=True)

    proj_lbl  = "Dados" if lang == "PT" else "Data"
    preproc_l = "Preproc."
    hw_lbl    = "Hardware"
    nivel_l   = "Nivel" if lang == "PT" else "Level"
    tec_l     = "Tecnica" if lang == "PT" else "Technique"
    status_l  = "Status"

    t.add_row(
        proj_lbl,  dados_str,
        preproc_l, f"[info]{preproc}[/info]",
        hw_lbl,    hw_str,
    )
    t.add_row(
        tec_l,     f"[a]{escape(tec_nome)}[/a]",
        nivel_l,   f"[a]{nivel}[/a]",
        status_l,  status_str,
    )

    tit = "Status do Projeto" if lang == "PT" else "Project Status"
    console.print(Panel(
        t, title=f"[info]{tit}[/info]",
        border_style=PS, box=rbox.ROUNDED, padding=(0, 1)
    ))

# ---------------------------------------------------------------------------
# MENU PRINCIPAL — compacto, numeros adjacentes ao texto
# ---------------------------------------------------------------------------
def _print_main_menu() -> None:
    lang = _lang()
    is_pt = lang == "PT"
    G, F, S, M = PA, PF, PS, PM

    def _grp(label: str, cor: str = PA) -> str:
        return f"[{cor}]{'─' * 3} {label} {'─' * 3}[/{cor}]"

    # Cada par de opcoes em uma linha, sem separacao de colunas
    t = Table(box=None, show_header=False, padding=(0, 0), expand=False)
    t.add_column("a", no_wrap=True, min_width=26)
    t.add_column("b", no_wrap=True, min_width=26)

    G_k = f"{G}"  # amber para teclas

    def row(k1, lbl1, k2="", lbl2="", style1=G_k, style2=G_k):
        c1 = f"  [{style1}][{k1}][/{style1}] {lbl1}" if k1 else f"  {lbl1}"
        c2 = f"  [{style2}][{k2}][/{style2}] {lbl2}" if k2 else (f"  {lbl2}" if lbl2 else "")
        return Text.from_markup(c1), Text.from_markup(c2)

    # Grupos
    t.add_row(Text.from_markup(_grp(_t("grp_config"))), Text.from_markup(""))
    t.add_row(*row("1", _t("t_projeto"),    "2", _t("t_dados")))
    t.add_row(*row("3", _t("t_preproc"),    "4", _t("t_modelagem")))
    t.add_row(*row("5", _t("t_validacao"),  "6", _t("t_avancado")))

    t.add_row(Text.from_markup(""), Text.from_markup(""))
    t.add_row(Text.from_markup(_grp(_t("grp_analise"))), Text.from_markup(""))
    t.add_row(*row("7", _t("t_viz"),        "8", _t("t_tecnica")))
    t.add_row(*row("9", _t("t_codigos"),    "H", _t("t_hardware"), style2=S))
    t.add_row(*row("B", _t("t_predicao"), style1=S))

    t.add_row(Text.from_markup(""), Text.from_markup(""))
    t.add_row(Text.from_markup(_grp(_t("grp_sistema"), cor=S)), Text.from_markup(""))
    t.add_row(*row("P", _t("t_perfis"),  "I", _t("t_idioma"), style1=S, style2=S))
    t.add_row(*row("G", "Guaraci ☀",    "?", _t("t_ajuda"),  style1=PA, style2=M))
    sobre_lbl = "Sobre" if lang == "PT" else "About"
    t.add_row(*row("A", sobre_lbl,       "Q", _t("sair"),     style1=S, style2=M))

    tit_menu = "GUARACI — MENU PRINCIPAL" if lang == "PT" else "GUARACI — MAIN MENU"
    console.print(Panel(
        t,
        title=f"[bold {PA}]  {tit_menu}  [/bold {PA}]",
        border_style=PA,
        box=rbox.DOUBLE,
        padding=(0, 1),
    ))


# ---------------------------------------------------------------------------
# CAIXA DE EXECUCAO — call-to-action destacado para iniciar a analise (R)
# ---------------------------------------------------------------------------
def _print_run_box(cfg: Config) -> None:
    """Caixa de destaque para iniciar a analise.

    Muda de cor conforme a prontidao (verde = pronto, cinza = falta config)
    e exibe info complementar ao Status do Projeto: RAM livre (indicador de
    3 niveis) e os modulos pesados ativos (Benchmark / Monte Carlo / SHAP).
    """
    lang = _lang()
    is_pt = lang == "PT"

    pasta = _cfgv(cfg, "pasta_dados", "dados")
    pasta_ok = bool(pasta) and os.path.isdir(str(pasta))
    n_dx = _contar_dx(pasta) if pasta_ok else 0
    pronto = pasta_ok and n_dx > 0

    # RAM livre — indicador visual de 3 niveis
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_livre = vm.available / (1024 ** 3)
        ram_total = vm.total / (1024 ** 3)
        if ram_livre >= 8:
            ram_cor, ram_ico = PG, "●●●"
        elif ram_livre >= 4:
            ram_cor, ram_ico = PA, "●●○"
        else:
            ram_cor, ram_ico = PR, "●○○"
        ram_txt = f"[{ram_cor}]{ram_ico}  RAM {ram_livre:.1f}/{ram_total:.0f} GB[/{ram_cor}]"
    except Exception:  # noqa: BLE001 -- indicador visual best-effort (psutil
        # ausente ou probe falha por variacao de SO); "N/A" e' o fallback
        # documentado, nunca impede a tela de renderizar.
        ram_txt = f"[{PM}]RAM N/A[/{PM}]"

    # Prontidao + cor da chamada
    if pronto:
        cta_cor = PG
        check = (f"[{PG}]✔ Pronto para executar[/{PG}]" if is_pt
                 else f"[{PG}]✔ Ready to run[/{PG}]")
    else:
        cta_cor = PM
        check = (f"[{PR}]✖ {_t('status_erro')}[/{PR}]")

    # Modulos pesados ativos — definem o tempo de execucao e NAO aparecem
    # no Status do Projeto (info complementar, nao redundante).
    extras = []
    if _cfgv(cfg, "benchmark", False):      extras.append("Benchmark")
    if _cfgv(cfg, "monte_carlo", False):    extras.append("Monte Carlo")
    if _cfgv(cfg, "shap_benchmark", False): extras.append("SHAP")
    if extras:
        extras_txt = f"[{PA}]" + " · ".join(extras) + f"[/{PA}]"
    else:
        extras_txt = f"[{PM}]" + ("nenhum" if is_pt else "none") + f"[/{PM}]"
    ext_lbl = "Extras" if is_pt else "Extras"

    big = "RODAR PIPELINE" if is_pt else "RUN PIPELINE"
    sub = ("Pressione  [R]  e Enter para comecar a analise"
           if is_pt else "Press  [R]  then Enter to start the analysis")

    inner = Table(box=None, show_header=False, padding=(0, 1), expand=True)
    inner.add_column("c", justify="center")
    inner.add_row(Text.from_markup(
        f"[bold {cta_cor}]▶   [{cta_cor}]\\[R][/{cta_cor}]   {big}   ◀[/bold {cta_cor}]"))
    inner.add_row(Text.from_markup(f"[{PM}]{sub}[/{PM}]"))
    inner.add_row(Text.from_markup(""))
    inner.add_row(Text.from_markup(
        f"{check}     {ram_txt}     "
        f"[{PM}]{ext_lbl}:[/{PM}] {extras_txt}"))

    tit = "  ☀  EXECUCAO  ☀  " if is_pt else "  ☀  EXECUTION  ☀  "
    console.print(Panel(
        Align.center(inner),
        title=f"[bold {cta_cor}]{tit}[/bold {cta_cor}]",
        border_style=cta_cor,
        box=rbox.HEAVY,
        padding=(0, 1),
    ))

# ---------------------------------------------------------------------------
# SUBMENU COMPACTO — campos com valor na mesma linha
# ---------------------------------------------------------------------------
# _trunc importada de guaraci.cli_logic (item 19) — ver topo do arquivo.


def _desc_curta(key: str, max_c: int = 42) -> str:
    """Retorna descricao resumida do campo (max_c chars) para exibicao inline.

    Resolve idioma/HELP_DB/_SPEC_BY_KEY (estado desta tela) e delega o
    truncamento a `guaraci.cli_logic.truncar_desc_por_frase` (funcao pura,
    testada).
    """
    lang = _lang()
    h = HELP_DB.get(key, {})
    desc = h.get(lang, h.get("PT", {})).get("desc", "")
    if not desc:
        desc = _SPEC_BY_KEY.get(key, {}).get("desc", "")
    return _truncar_desc_por_frase(desc, max_c)


def _print_submenu_compact(
    title: str, desc: str, fields: List[str], cfg: Config,
    extras: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """
    Submenu compacto: [N] ICON Nome  Valor  Descricao-breve
    Exibe o valor atual e uma descricao curta na mesma linha.
    """
    lang = _lang()
    t = Table(box=None, show_header=False, padding=(0, 0), expand=True)
    t.add_column("N",    no_wrap=True, width=5)
    t.add_column("Ico",  no_wrap=True, width=2)
    t.add_column("Nome", no_wrap=True, min_width=20, max_width=26)
    t.add_column("Val",  no_wrap=True, min_width=10, max_width=18)
    t.add_column("Desc", no_wrap=True, style=PM)

    for i, key in enumerate(fields, 1):
        nome  = _nome_campo(key)
        val   = _get_val(cfg, key)
        r_hex = _risco_hex(key)
        r_ico = _risco_icon(key)
        breve = _desc_curta(key, 44)

        # Formatar valor — sem hifenizacao
        if isinstance(val, bool):
            val_txt = Text("Sim" if lang == "PT" else "Yes", style=PG) if val \
                      else Text("Nao" if lang == "PT" else "No", style=PM)
        elif val is None or str(val) == "":
            val_txt = Text("—", style=PM)
        else:
            val_txt = Text(_trunc(str(val), 16), style=PS)

        t.add_row(
            Text.from_markup(f"  [{r_hex}][{i}][/{r_hex}]"),
            Text(r_ico, style=r_hex),
            Text(f" {nome}", style=PW),
            val_txt,
            Text(breve, style=PM) if breve else Text(""),
        )

    if extras:
        t.add_row(Text(""), Text(""), Text(""), Text(""), Text(""))
        for ek, ed in extras:
            t.add_row(
                Text.from_markup(f"  [{PA}][{ek}][/{PA}]"),
                Text(""),
                Text(f" {ed}", style=PW),
                Text(""),
                Text(""),
            )

    # Rodape: legenda de risco + comandos
    rodape = Text.from_markup(
        f"  [{PF}]●[/{PF}] Visual  "
        f"[{PA}]◆[/{PA}] Analitico  "
        f"[{PR}]▲[/{PR}] Avancado  "
        f"   [{PM}][0][/{PM}] {_t('voltar')}"
        f"  [{PM}][?][/{PM}] Ajuda do campo"
        f"  [{PA}][G][/{PA}] Guaraci"
        f"  [{PM}][I][/{PM}] Idioma"
    )

    console.print(Panel(
        Group(t, Rule(style=PD), rodape),
        title=f"[bold {PA}]{escape(title)}[/bold {PA}]",
        subtitle=f"[{PM}]{escape(desc)}[/{PM}]",
        border_style=PA,
        box=rbox.ROUNDED,
        padding=(0, 1),
    ))

# ---------------------------------------------------------------------------
# EDICAO DE CAMPO
# ---------------------------------------------------------------------------
def _editar_campo(cfg: Config, key: str) -> bool:
    lang = _lang()
    spec = _SPEC_BY_KEY.get(key)
    if spec is None:
        _msg = (f"Campo '{key}' nao encontrado." if lang == "PT"
                else f"Field '{key}' not found.")
        console.print(f"  [err]{_msg}[/err]")
        return False

    nome     = _nome_campo(key)
    val_atual = _get_val(cfg, key)
    # Valor interno cru (ex.: "N1") para comparar com as opcoes — val_atual
    # pode ser um rotulo amigavel de exibicao ("Classificacao... (N1)").
    val_cru   = _attr_para_yaml(spec, cfg)
    tipo      = spec.get("tipo", "str")
    opcoes    = spec.get("opcoes")
    risk      = RISK_CLASS.get(key, "ANALITICO")
    r_hex     = _risco_hex(key)

    # Painel de edicao minimalista
    info = Table(box=None, show_header=False, padding=(0, 1))
    info.add_column("L", style=PM, width=10, no_wrap=True)
    info.add_column("V", no_wrap=False)
    info.add_row("Campo:", Text(nome, style=f"bold {r_hex}"))
    info.add_row("Atual:", Text(str(val_atual), style=PS))
    info.add_row("Tipo:", Text(tipo, style=PM))
    if opcoes:
        info.add_row("Opcoes:", Text(
            " | ".join(_rotulo_opcao(key, o) for o in opcoes), style=PM))

    console.print(Panel(
        info,
        title=f"[{r_hex}]Editar: {escape(nome)}[/{r_hex}]",
        border_style=r_hex, box=rbox.ROUNDED, padding=(0, 1)
    ))

    if opcoes:
        console.print()
        for j, op in enumerate(opcoes, 1):
            mk = f"[{PA}]►[/{PA}]" if str(op) == str(val_cru) else " "
            console.print(f"  {mk} [{PA}][{j}][/{PA}] "
                          f"{escape(_rotulo_opcao(key, op))}")
        console.print()
        raw = _input(f"  [{1}-{len(opcoes)}] ou Enter=manter: ")
        if not raw:
            console.print(f"  [{PM}]{_t('mantido')}[/{PM}]"); return False
        if raw == "?":
            _mostrar_ajuda(key); return False
        if raw.isdigit() and 1 <= int(raw) <= len(opcoes):
            raw = str(opcoes[int(raw) - 1])
    else:
        console.print()
        raw = _input(f"  {_t('novo_valor')}")
        if not raw:
            console.print(f"  [{PM}]{_t('mantido')}[/{PM}]"); return False
        if raw == "?":
            _mostrar_ajuda(key); return False

    # Confirmacao para campos analiticos
    if risk == "ANALITICO":
        conf = _ask(f"  [{PA}]{_t('conf_anal')}[/{PA}] ")
        if conf.lower() not in ("s", "y", "sim", "yes"):
            console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); return False

    try:
        _set_val(cfg, key, raw)
        msg = _t("atualizado", campo=nome, valor=raw)
        console.print(f"  [g]✓ {escape(msg)}[/g]")
        return True
    except ValueError as e:   # _set_val/_coagir_valor: valor digitado invalido
        console.print(f"  [err]Erro: {escape(str(e))}[/err]")
        return False

# ---------------------------------------------------------------------------
# AJUDA POR CAMPO — descricao completa + dica unica do Guaraci
# ---------------------------------------------------------------------------
def _mostrar_ajuda(key: str) -> None:
    lang   = _lang()
    h      = HELP_DB.get(key, {})
    h_lang = h.get(lang, h.get("PT", {}))
    nome   = _nome_campo(key)
    r_hex  = _risco_hex(key)
    spec   = _SPEC_BY_KEY.get(key, {})

    # Fallback: se HELP_DB nao cobre o campo, usa a descricao do _CONFIG_SPEC
    desc    = h_lang.get("desc") or spec.get("desc") or (
        "Sem descricao detalhada para este campo." if lang == "PT"
        else "No detailed description for this field.")
    impacto = h_lang.get("impacto", "—")
    exemplos = h_lang.get("exemplos", {})
    default  = h.get("default", spec.get("default", "—"))
    opcoes   = spec.get("opcoes")
    faixa    = h.get("range") or (" | ".join(str(o) for o in opcoes) if opcoes else
                                  spec.get("tipo", "—"))
    tip      = GUARACI_TIPS.get(key, {}).get(lang, "")

    info = Table(box=None, show_header=False, padding=(0, 1))
    info.add_column("L", style=PM, width=12, no_wrap=True)
    info.add_column("V")
    info.add_row("Campo:", Text(nome, style=f"bold {r_hex}"))
    info.add_row("Descricao:" if lang == "PT" else "Description:", Text(desc, style=PW))
    info.add_row("Impacto:" if lang == "PT" else "Impact:", Text(impacto, style=r_hex))
    info.add_row("Padrao:" if lang == "PT" else "Default:", Text(str(default), style=PS))
    info.add_row("Faixa:" if lang == "PT" else "Range:", Text(str(faixa), style=PM))

    if exemplos:
        info.add_row("", Text(""))
        ex_lbl = "Exemplos:" if lang == "PT" else "Examples:"
        info.add_row(ex_lbl, Text(""))
        for ek, ev in list(exemplos.items())[:4]:
            info.add_row(f"  {ek}", Text(str(ev), style=PM))

    parts = [info]

    # Dica unica do Guaraci (diferente da descricao)
    if tip:
        tip_panel = Panel(
            Text.from_markup(f"[{PA}]  {escape(tip)}[/{PA}]"),
            title=f"[bold {PA}]Guaraci diz:[/bold {PA}]"
            if lang == "PT" else f"[bold {PA}]Guaraci says:[/bold {PA}]",
            border_style=PA, box=rbox.SIMPLE, padding=(0, 1)
        )
        parts.append(tip_panel)

    console.print(Panel(
        Group(*parts),
        title=f"[{r_hex}] {escape(nome)} [/{r_hex}]",
        border_style=r_hex, box=rbox.ROUNDED, padding=(0, 1)
    ))
    _pause()

# ===========================================================================
# MENUS DE CONFIGURACAO
# ===========================================================================

def _loop_menu(title: str, desc: str, fields: List[str], cfg: Config,
               extras: Optional[List[Tuple[str, str]]] = None,
               on_extra: Optional[Dict[str, Any]] = None) -> None:
    """Loop generico para submenus de configuracao."""
    while True:
        cls()
        _print_header()
        _print_submenu_compact(title, desc, fields, cfg, extras)
        raw = _input(f"\n  {_t('opcao')}: ").upper()

        if raw in ("0", "Q", ""):
            break
        elif raw == "I":
            _toggle_idioma()
        elif raw == "G":
            _abrir_assistente(title, cfg)
        elif raw == "?":
            r2 = _input("  Campo (N ou nome): ").strip()
            if r2.isdigit() and 1 <= int(r2) <= len(fields):
                _mostrar_ajuda(fields[int(r2) - 1])
            elif r2 in HELP_DB:
                _mostrar_ajuda(r2)
            else:
                found = [k for k in HELP_DB if r2.lower() in k.lower() or r2.lower() in _nome_campo(k).lower()]
                _mostrar_ajuda(found[0]) if found else console.print(f"  [{PM}]{_t('invalido')}[/{PM}]")
        elif raw.isdigit() and 1 <= int(raw) <= len(fields):
            _editar_campo(cfg, fields[int(raw) - 1])
            _pause()
        elif on_extra and raw in on_extra:
            on_extra[raw]()
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]")
            _pause()


def menu_projeto(cfg: Config) -> None:
    _loop_menu(_t("t_projeto"), _t("d_projeto"), ["pasta_dados", "pasta_saida", "tag"], cfg)


def menu_dados(cfg: Config) -> None:
    _loop_menu(_t("t_dados"), _t("d_dados"),
               ["modo_entrada", "arquivo_csv", "coluna_classe",
                "coluna_concentracao", "faixa_min_cm", "faixa_max_cm", "excluir_classes"], cfg)


def menu_preproc(cfg: Config) -> None:
    def _show_pipeline():
        preproc = str(_cfgv(cfg, "pre_processamento", "msc_sg_mc"))
        comps = {
            "msc": "MSC — Multiplicative Scatter Correction",
            "snv": "SNV — Standard Normal Variate",
            "sg":  "SG  — Savitzky-Golay",
            "mc":  "MC  — Mean-Centering",
        }
        partes = [c for c in ["msc","snv","sg","mc"] if c in preproc.lower()]
        t = Table(box=None, show_header=False, padding=(0, 1))
        t.add_column("I", style=PG, width=2)
        t.add_column("D", style=PW)
        for p in partes:
            t.add_row("✓", comps.get(p, p))
        lbl = "Pipeline ativo" if _lang() == "PT" else "Active pipeline"
        console.print(Panel(t, title=f"[{PS}]{lbl}: [{PA}]{escape(preproc)}[/{PA}][/{PS}]",
                            border_style=PS, box=rbox.SIMPLE, padding=(0, 1)))

    fields = ["pre_processamento", "comparar_pre_processamentos"]
    while True:
        cls(); _print_header(); _show_pipeline()
        _print_submenu_compact(_t("t_preproc"), _t("d_preproc"), fields, cfg)
        raw = _input(f"\n  {_t('opcao')}: ").upper()
        if raw in ("0", "Q"):
            break
        elif raw == "I":
            _toggle_idioma()
        elif raw == "G":
            _abrir_assistente(_t("t_preproc"), cfg)
        elif raw.isdigit() and 1 <= int(raw) <= len(fields):
            _editar_campo(cfg, fields[int(raw) - 1])
            _pause()
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()


def menu_modelagem(cfg: Config) -> None:
    _loop_menu(_t("t_modelagem"), _t("d_modelagem"),
               ["nivel", "max_lvs", "opls_da", "ddsimca",
                "modo_ddsimca", "selecao_variaveis_etapa4"], cfg)


def menu_validacao(cfg: Config) -> None:
    fields = ["holdout_fracao", "validacao_group_aware",
              "n_permutacoes", "teste_wold", "teste_cv_anova"]
    while True:
        cls(); _print_header()
        ga = _cfgv(cfg, "validacao_group_aware", True)
        if not ga:
            console.print(Panel(
                "[err]  GroupKFold DESATIVADO — risco de data leakage![/err]\n"
                "[err]  Ative o campo [2] imediatamente.[/err]",
                border_style=PR, box=rbox.HEAVY, padding=(0, 1)
            ))
        _print_submenu_compact(_t("t_validacao"), _t("d_validacao"), fields, cfg)
        raw = _input(f"\n  {_t('opcao')}: ").upper()
        if raw in ("0", "Q"):
            break
        elif raw == "I":
            _toggle_idioma()
        elif raw == "G":
            _abrir_assistente(_t("t_validacao"), cfg)
        elif raw.isdigit() and 1 <= int(raw) <= len(fields):
            _editar_campo(cfg, fields[int(raw) - 1]); _pause()
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()


def menu_avancado(cfg: Config) -> None:
    fields = ["benchmark", "monte_carlo", "n_monte_carlo",
              "monte_carlo_incluir_todos", "shap_benchmark", "shap_max_amostras"]
    while True:
        cls(); _print_header()
        console.print(Panel(
            f"[{PR}]  ▲ Modulos pesados — verificar hardware em [H] antes de ativar.[/{PR}]",
            border_style=PR, box=rbox.SIMPLE, padding=(0, 1)
        ))
        _print_submenu_compact(_t("t_avancado"), _t("d_avancado"), fields, cfg)
        raw = _input(f"\n  {_t('opcao')}: ").upper()
        if raw in ("0", "Q"):
            break
        elif raw == "I":
            _toggle_idioma()
        elif raw == "G":
            _abrir_assistente(_t("t_avancado"), cfg)
        elif raw.isdigit() and 1 <= int(raw) <= len(fields):
            _editar_campo(cfg, fields[int(raw) - 1]); _pause()
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()


# ---------------------------------------------------------------------------
# VISUALIZACAO — submenu especial com sub-handlers
# ---------------------------------------------------------------------------
def menu_visualizacao(cfg: Config) -> None:
    fields = ["figuras_mostrar_marcadores", "figuras_mostrar_elipses",
              "formato_figura", "dpi", "abrir_figuras_na_tela"]
    # H/M/B/V (heatmap espectral, matriz de confusao, biplot PCA, variancia
    # x wavelength) removidos daqui: auditoria de 2026-07-12 encontrou que
    # apontavam para _gerar_heatmap_espectros/_gerar_confusion_matrix/
    # _gerar_pca_biplot/_gerar_variancia_wavelength, funcoes que nunca
    # existiram em nenhum modulo do projeto -- as 4 opcoes sempre falhavam
    # com "Funcao nao disponivel", mascarado por um except Exception generico.
    # Gerar essas figuras fora de uma execucao completa exigiria carregar
    # dados + ajustar modelo aqui (feature nova, nao um bugfix) -- por isso
    # as opcoes foram retiradas em vez de remendadas. Ver CLAUDE.md secao 13.
    extras_pt = [
        ("P", _t("viz_paleta")), ("F", _t("viz_fonte")),
        ("D", _t("viz_grid")),   ("A", _t("viz_alpha")),
    ]

    def _pal():
        vcfg = _carregar_visual_cfg()
        t = Table(box=None, show_header=False, padding=(0, 1))
        t.add_column("N", no_wrap=True, width=4)
        t.add_column("Nome", no_wrap=True, width=30)
        t.add_column("Desc", style=PM)
        atual = vcfg.get("paleta", "qualitativo")
        for i, (pk, pd) in enumerate(PALETAS_COR.items(), 1):
            nm = pd.get("nome", {}).get(_lang(), pk) if isinstance(pd.get("nome"), dict) else pk
            dsc = pd.get("desc", {}).get(_lang(), "") if isinstance(pd.get("desc"), dict) else ""
            mk = f"[{PA}]►[/{PA}]" if pk == atual else " "
            t.add_row(f"  [{PA}][{i}][/{PA}]", f"{mk} {escape(nm)}", escape(_trunc(dsc, 35)))
        console.print(Panel(t, title=f"[bold {PA}]{_t('viz_paleta')}[/bold {PA}]",
                            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)))
        r = _input(f"  [1-{len(PALETAS_COR)}] ou Enter: ")
        if r.isdigit():
            idx = int(r) - 1
            if 0 <= idx < len(PALETAS_COR):
                vcfg["paleta"] = list(PALETAS_COR.keys())[idx]
                _salvar_visual_cfg(vcfg)
                _lbl = "Paleta" if _lang() == "PT" else "Palette"
                console.print(f"  [g]✓ {_lbl}: {vcfg['paleta']}[/g]")

    def _fonte():
        vcfg = _carregar_visual_cfg()
        presets = [
            ("xs", "XS — Muito pequeno" if _lang()=="PT" else "XS — Very small"),
            ("s",  "S  — Pequeno" if _lang()=="PT" else "S  — Small"),
            ("m",  "M  — Medio (padrao)" if _lang()=="PT" else "M  — Medium (default)"),
            ("l",  "L  — Grande (apresentacoes)" if _lang()=="PT" else "L  — Large (presentations)"),
            ("xl", "XL — Muito grande (conferencias)" if _lang()=="PT" else "XL — Extra large (conferences)"),
        ]
        t = Table(box=None, show_header=False, padding=(0, 1))
        t.add_column("N", no_wrap=True, width=4)
        t.add_column("Desc", style=PW)
        atual = vcfg.get("tamanho_fonte", "m")
        for i, (fk, fd) in enumerate(presets, 1):
            mk = f"[{PA}]►[/{PA}]" if fk == atual else " "
            t.add_row(f"  [{PA}][{i}][/{PA}]", f"{mk} {escape(fd)}")
        console.print(Panel(t, title=f"[bold {PA}]{_t('viz_fonte')}[/bold {PA}]",
                            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)))
        r = _input("  [1-5] ou Enter: ")
        if r.isdigit() and 1 <= int(r) <= 5:
            vcfg["tamanho_fonte"] = presets[int(r)-1][0]
            _salvar_visual_cfg(vcfg)
            _lbl = "Fonte" if _lang() == "PT" else "Font"
            console.print(f"  [g]✓ {_lbl}: {vcfg['tamanho_fonte']}[/g]")

    def _grid():
        vcfg = _carregar_visual_cfg()
        gm = vcfg.get("grid_major", True)
        gmi = vcfg.get("grid_minor", False)
        gs = vcfg.get("grid_style", "dotted")
        ga = vcfg.get("grid_alpha", 0.4)
        t = Table(box=None, show_header=False, padding=(0, 1))
        t.add_column("N", no_wrap=True, width=4)
        t.add_column("Opcao", no_wrap=True, width=26)
        t.add_column("Valor", style=PS)
        t.add_row(f"  [{PA}][1][/{PA}]", "Grid principal", "[g]ON[/g]" if gm else "[m]OFF[/m]")
        t.add_row(f"  [{PA}][2][/{PA}]", "Grid secundario", "[g]ON[/g]" if gmi else "[m]OFF[/m]")
        t.add_row(f"  [{PA}][3][/{PA}]", "Estilo", escape(gs))
        t.add_row(f"  [{PA}][4][/{PA}]", "Transparencia", str(ga))
        console.print(Panel(t, title=f"[bold {PA}]{_t('viz_grid')}[/bold {PA}]",
                            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)))
        r = _input("  [1-4] ou Enter: ")
        if r == "1": vcfg["grid_major"] = not gm
        elif r == "2": vcfg["grid_minor"] = not gmi
        elif r == "3":
            ests = ["solid","dotted","dashed"]
            vcfg["grid_style"] = ests[(ests.index(gs)+1)%3] if gs in ests else "dotted"
        elif r == "4":
            try: vcfg["grid_alpha"] = float(_input("  Valor [0.1-0.9]: "))
            except ValueError:
                pass   # entrada nao-numerica -- mantem o valor anterior
        _salvar_visual_cfg(vcfg)

    def _alpha():
        vcfg = _carregar_visual_cfg()
        ops = [
            ("baixo", "0.9 — Opacos" if _lang()=="PT" else "0.9 — Opaque"),
            ("medio", "0.65 — Equilibrado (padrao)" if _lang()=="PT" else "0.65 — Balanced (default)"),
            ("alto",  "0.35 — Translucido" if _lang()=="PT" else "0.35 — Translucent"),
        ]
        t = Table(box=None, show_header=False, padding=(0, 1))
        t.add_column("N", no_wrap=True, width=4)
        t.add_column("Desc", style=PW)
        atual = vcfg.get("alpha_pontos", "medio")
        for i, (ak, ad) in enumerate(ops, 1):
            mk = f"[{PA}]►[/{PA}]" if ak == atual else " "
            t.add_row(f"  [{PA}][{i}][/{PA}]", f"{mk} {escape(ad)}")
        console.print(Panel(t, title=f"[bold {PA}]{_t('viz_alpha')}[/bold {PA}]",
                            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)))
        r = _input("  [1-3] ou Enter: ")
        if r in ("1","2","3"):
            vcfg["alpha_pontos"] = ops[int(r)-1][0]
            _salvar_visual_cfg(vcfg)

    while True:
        cls(); _print_header()
        _print_submenu_compact(_t("t_viz"), _t("d_viz"), fields, cfg, extras=extras_pt)
        raw = _input(f"\n  {_t('opcao')}: ").upper()
        if raw in ("0","Q"): break
        elif raw == "I": _toggle_idioma()
        elif raw == "G": _abrir_assistente(_t("t_viz"), cfg)
        elif raw == "P": _pal(); _pause()
        elif raw == "F": _fonte(); _pause()
        elif raw == "D": _grid(); _pause()
        elif raw == "A": _alpha(); _pause()
        elif raw.isdigit() and 1 <= int(raw) <= len(fields):
            _editar_campo(cfg, fields[int(raw)-1]); _pause()
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()


# ---------------------------------------------------------------------------
# TECNICA ANALITICA
# ---------------------------------------------------------------------------
# Agrupamento das tecnicas por categoria (segue o modelo do prompt GUARACI).
# So inclui chaves presentes em TECNICAS; chaves ausentes sao ignoradas.
_TECNICA_CATEGORIAS = [
    ("Vibracional",          "Vibrational",        ["ft-nir", "nir", "mir", "raman", "uv-vis"]),
    ("Luminescencia",        "Luminescence",       ["fluorescencia"]),
    ("Cromatografia",        "Chromatography",     ["hplc", "gc-ms"]),
    ("Ressonancia / Outras", "Resonance / Others", ["nmr", "ims", "generico"]),
]


def _tecnica_ordem() -> list:
    """Retorna a lista achatada de chaves de tecnica na ordem das categorias."""
    ordem = []
    vistos = set()
    for _pt, _en, keys in _TECNICA_CATEGORIAS:
        for k in keys:
            if k in TECNICAS and k not in vistos:
                ordem.append(k); vistos.add(k)
    # Acrescenta quaisquer tecnicas nao categorizadas, ao final
    for k in TECNICAS:
        if k not in vistos:
            ordem.append(k); vistos.add(k)
    return ordem


def _tecnica_detalhe(tk: str, lang: str) -> None:
    """Painel com detalhes completos de uma tecnica."""
    td = TECNICAS.get(tk, {})
    tdl = td.get(lang, td.get("PT", {}))
    linhas = [
        f"[{PW}]{escape(tdl.get('desc',''))}[/{PW}]", "",
        f"[{PA}]{'Faixa tipica' if lang=='PT' else 'Typical range'}:[/{PA}] "
        f"[{PW}]{escape(str(tdl.get('faixa','—')))}[/{PW}]",
        f"[{PA}]{'Pre-proc. recomendado' if lang=='PT' else 'Recommended preproc'}:[/{PA}] "
        f"[{PW}]{escape(str(tdl.get('preproc_rec', td.get('preproc','—'))))}[/{PW}]",
        f"[{PA}]{'Modo de entrada' if lang=='PT' else 'Input mode'}:[/{PA}] "
        f"[{PW}]{escape(str(td.get('modo','dx')))}[/{PW}]",
    ]
    console.print(Panel(
        Text.from_markup("\n".join(linhas)),
        title=f"[bold {PA}]{escape(tdl.get('nome', tk))}[/bold {PA}]",
        border_style=PF, box=rbox.ROUNDED, padding=(1, 2), width=_W()
    ))
    _pause()


def menu_tecnica(cfg: Config) -> None:
    """Tecnica analitica — agrupada por categoria (modelo GUARACI)."""
    lang = _lang()

    def _aplicar(tk_sel: str) -> None:
        td_sel = TECNICAS.get(tk_sel, {})
        tdl = td_sel.get(lang, td_sel.get("PT", {}))
        try:
            nm_sel = tdl.get("nome", tk_sel)
            _TECNICA_SELECIONADA["key"]  = tk_sel
            _TECNICA_SELECIONADA["nome"] = tk_sel.upper()
            fmin = td_sel.get("faixa_min"); fmax = td_sel.get("faixa_max")
            prep = td_sel.get("preproc", ""); modo = td_sel.get("modo", "dx")
            if fmin is not None: _set_val(cfg, "faixa_min_cm", str(fmin))
            if fmax is not None: _set_val(cfg, "faixa_max_cm", str(fmax))
            if prep: _set_val(cfg, "pre_processamento", prep)
            if modo: _set_val(cfg, "modo_entrada", modo)
            fa_str = tdl.get("faixa", f"{fmin}-{fmax}")
            console.print(f"  [g]✓ {escape(_trunc(nm_sel, 44))} {'selecionado' if lang=='PT' else 'selected'}.[/g]")
            console.print(f"  [info]  {'Faixa' if lang=='PT' else 'Range'}: {escape(_trunc(str(fa_str), 44))}[/info]")
            console.print(f"  [info]  Preproc.: {escape(str(prep))}  |  {'Modo' if lang=='PT' else 'Mode'}: {modo}[/info]")
        except ValueError as e:   # _set_val: faixa/preproc/modo invalido p/ a tecnica
            console.print(f"  [err]{escape(str(e))}[/err]")

    while True:
        cls(); _print_header()
        lang = _lang()
        ordem = _tecnica_ordem()
        num = {tk: i for i, tk in enumerate(ordem, 1)}  # chave -> numero
        tec_atual = _TECNICA_SELECIONADA.get("key", "ft-nir")

        t = Table(box=None, show_header=True, header_style=PM, padding=(0, 1), expand=True)
        t.add_column("N",      style=PA, width=4, no_wrap=True)
        t.add_column("Tecnica" if lang=="PT" else "Technique", width=34, no_wrap=True)
        t.add_column("Faixa" if lang=="PT" else "Range", style=PS, width=22, no_wrap=True)
        t.add_column("Preproc.", style=PM, no_wrap=True)

        for cat_pt, cat_en, keys in _TECNICA_CATEGORIAS:
            keys_presentes = [k for k in keys if k in TECNICAS]
            if not keys_presentes:
                continue
            cat = cat_pt if lang == "PT" else cat_en
            t.add_row("", Text.from_markup(f"[{PF}]── {escape(cat)} ──[/{PF}]"), "", "")
            for tk in keys_presentes:
                td = TECNICAS.get(tk, {})
                tdl = td.get(lang, td.get("PT", {}))
                nm = tdl.get("nome", tk)
                fa = tdl.get("faixa", "—")
                pr = td.get("preproc", "—")
                mk = f"[{PA}]►[/{PA}] " if tk == tec_atual else "  "
                t.add_row(
                    f"[{PA}][{num[tk]}][/{PA}]",
                    Text.from_markup(f"{mk}{escape(_trunc(nm, 30))}"),
                    escape(_trunc(str(fa), 20)),
                    escape(_trunc(str(pr), 12)),
                )

        sub = ("Selecione o numero para aplicar. [?] N = detalhes."
               if lang=="PT" else "Select the number to apply. [?] N = details.")
        console.print(Panel(
            t,
            title=f"[bold {PA}]{_t('t_tecnica')}[/bold {PA}]",
            subtitle=f"[{PM}]{sub}[/{PM}]",
            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)
        ))
        if lang == "PT":
            console.print(f"  [{PA}][?][/{PA}] N detalhes   [{PA}][G][/{PA}] Guaraci"
                          f"   [{PM}][I][/{PM}] Idioma   [{PM}][0][/{PM}] Voltar")
        else:
            console.print(f"  [{PA}][?][/{PA}] N details   [{PA}][G][/{PA}] Guaraci"
                          f"   [{PM}][I][/{PM}] Language   [{PM}][0][/{PM}] Back")

        raw = _input(f"\n  {_t('opcao')}: ").strip().upper()
        if raw in ("0","Q",""): break
        elif raw == "I": _toggle_idioma(); lang = _lang()
        elif raw == "G": _abrir_assistente(_t("t_tecnica"), cfg)
        elif raw == "?":
            r2 = _input("  N: ").strip()
            if r2.isdigit() and 1 <= int(r2) <= len(ordem):
                _tecnica_detalhe(ordem[int(r2)-1], lang)
        elif raw.isdigit() and 1 <= int(raw) <= len(ordem):
            _aplicar(ordem[int(raw)-1])
            _pause(); break
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()


# ---------------------------------------------------------------------------
# CODIFICACAO DX
# ---------------------------------------------------------------------------
def menu_codificacao(cfg: Config) -> None:
    """Codificacao DX — explica o conceito e so lista os codigos sob demanda."""
    lang = _lang()
    CODIGOS_BASE = getattr(pq, "CODIGO_ESPECIE", {
        "AND":"Andiroba","ACE":"Acai","BCB":"Bacaba","BRT":"Buriti",
        "BAB":"Babacu","CAP":"Castanha-do-Para","COC":"Coco","GOI":"Goiaba",
        "GRV":"Graviola","MAR":"Maracuja","PAL":"Palmiste","PAT":"Pataua",
        "PRA":"Pracaxi",
    })

    def _cod_usr() -> dict:
        try:
            p = _CODIGOS_PATH
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except (OSError, json.JSONDecodeError):
            return {}   # arquivo ausente/corrompido -- sem codigos extras do usuario

    def _salvar_cod(d: dict) -> bool:
        try:
            _CODIGOS_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
            return True
        except OSError as e:
            # Antes engolia o erro: o usuario "salvava" um codigo, nao via aviso,
            # e ele sumia no proximo inicio se a gravacao tivesse falhado.
            console.print(f"[err]✗ Falha ao salvar codigos: {e}[/err]")
            return False

    def _listar() -> None:
        cod_usr = _cod_usr()
        tc = Table(show_header=True, header_style=PM, box=rbox.SIMPLE, padding=(0, 1))
        tc.add_column("COD", style=PS, width=6, no_wrap=True)
        tc.add_column("Especie" if lang=="PT" else "Species", width=26)
        tc.add_column("Origem" if lang=="PT" else "Source", style=PM, width=10, no_wrap=True)
        for cod, esp in CODIGOS_BASE.items():
            tc.add_row(cod, escape(str(esp)), "Pipeline")
        for cod, esp in cod_usr.items():
            tc.add_row(f"[{PA}]{cod}[/{PA}]", escape(str(esp)),
                       f"[{PA}]{'Usuario' if lang=='PT' else 'User'}[/{PA}]")
        n = len(CODIGOS_BASE) + len(cod_usr)
        src_lbl = (f"Codigos cadastrados ({n})" if lang=="PT"
                   else f"Registered codes ({n})")
        console.print(Panel(tc, title=f"[{PS}]{src_lbl}[/{PS}]",
                            border_style=PS, box=rbox.ROUNDED, padding=(0, 1)))
        _pause()

    def _adicionar() -> None:
        cod_usr = _cod_usr()
        console.print()
        cod_n = _input(f"  {_t('cod_novo_cod')}").upper().strip()
        if not cod_n or not _re.match(r'^[A-Z]{2,4}$', cod_n):
            console.print(f"  [{PR}]{_t('cod_invalido')}[/{PR}]")
        else:
            esp_n = _input(f"  {_t('cod_novo_esp', cod=cod_n)}").strip()
            if esp_n:
                cod_usr[cod_n] = esp_n
                _salvar_cod(cod_usr)
                console.print(f"  [g]✓ {_t('cod_salvo', cod=cod_n, esp=esp_n)}[/g]")
        _pause()

    def _importar_csv() -> None:
        import csv as _csv
        console.print()
        prompt = ("  Caminho do CSV (colunas: codigo,especie): " if lang=="PT"
                  else "  CSV path (columns: code,species): ")
        caminho = _input(prompt).strip().strip('"')
        if not caminho:
            return
        if not os.path.isfile(caminho):
            console.print(f"  [{PR}]{'Arquivo nao encontrado.' if lang=='PT' else 'File not found.'}[/{PR}]")
            _pause(); return
        cod_usr = _cod_usr()
        n_add = 0
        try:
            with open(caminho, newline="", encoding="utf-8-sig") as fh:
                # Detecta separador (',' ou ';')
                amostra = fh.read(2048); fh.seek(0)
                sep = ";" if amostra.count(";") > amostra.count(",") else ","
                leitor = _csv.reader(fh, delimiter=sep)
                for linha in leitor:
                    if len(linha) < 2:
                        continue
                    cod = str(linha[0]).strip().upper()
                    esp = str(linha[1]).strip()
                    # Pula cabecalho comum
                    if cod.lower() in ("codigo", "cod", "code") or not cod:
                        continue
                    if not _re.match(r'^[A-Z]{2,4}$', cod) or not esp:
                        continue
                    cod_usr[cod] = esp
                    n_add += 1
            _salvar_cod(cod_usr)
            msg = (f"{n_add} codigo(s) importado(s)." if lang=="PT"
                   else f"{n_add} code(s) imported.")
            console.print(f"  [g]✓ {msg}[/g]")
        except (OSError, UnicodeDecodeError, _csv.Error) as e:
            console.print(f"  [{PR}]{escape(str(e))}[/{PR}]")
        _pause()

    def _exportar_csv() -> None:
        import csv as _csv
        cod_usr = _cod_usr()
        destino = str(_BASE_DIR / "codigos_exportados.csv")
        try:
            with open(destino, "w", newline="", encoding="utf-8-sig") as fh:
                w = _csv.writer(fh)
                w.writerow(["codigo", "especie", "origem"])
                for cod, esp in CODIGOS_BASE.items():
                    w.writerow([cod, esp, "pipeline"])
                for cod, esp in cod_usr.items():
                    w.writerow([cod, esp, "usuario"])
            msg = (f"Exportado para: {destino}" if lang=="PT"
                   else f"Exported to: {destino}")
            console.print(f"  [g]✓ {escape(msg)}[/g]")
        except OSError as e:
            console.print(f"  [{PR}]{escape(str(e))}[/{PR}]")
        _pause()

    while True:
        cls(); _print_header()

        # Painel explicativo — o que e, padrao de nome, como cadastrar/importar
        if lang == "PT":
            explicacao = (
                "[a]O que e:[/a] cada arquivo .dx comeca com um codigo de 2-4 letras\n"
                "que identifica a especie do oleo. A codificacao mapeia esse\n"
                "codigo para o nome legivel da especie usado nos resultados.\n\n"
                "[a]Padrao de nome dos arquivos:[/a]\n"
                "  COD-DD-MM-AAAA_Tn.dx            (especie pura)\n"
                "  COD-DD-MM-AAAA_AD-X-PP_Tn.dx    (adulterada)\n"
                "  Ex.: AND-10-06-2020_T1.dx  ->  Andiroba pura, triplicata 1\n\n"
                "[a]Como cadastrar:[/a]\n"
                "  [A] um codigo por vez, ou [M] importar um CSV pronto\n"
                "  (CSV com 2 colunas: codigo,especie — separador , ou ;)."
            )
        else:
            explicacao = (
                "[a]What it is:[/a] each .dx file starts with a 2-4 letter code\n"
                "identifying the oil species. The coding maps that code to the\n"
                "readable species name used in the results.\n\n"
                "[a]File name pattern:[/a]\n"
                "  COD-DD-MM-YYYY_Tn.dx            (pure species)\n"
                "  COD-DD-MM-YYYY_AD-X-PP_Tn.dx    (adulterated)\n"
                "  E.g.: AND-10-06-2020_T1.dx  ->  Andiroba pure, replicate 1\n\n"
                "[a]How to register:[/a]\n"
                "  [A] one code at a time, or [M] import a ready CSV\n"
                "  (CSV with 2 columns: code,species — separator , or ;)."
            )
        console.print(Panel(
            Text.from_markup(explicacao),
            title=f"[bold {PA}]{_t('t_codigos')}[/bold {PA}]",
            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)
        ))

        # Botoes de acao (a lista so aparece ao pressionar [L])
        n_usr = len(_cod_usr())
        if lang == "PT":
            acoes = (
                f"  [{PA}][L][/{PA}] Listar codigos cadastrados"
                f"   [{PA}][A][/{PA}] Adicionar codigo\n"
                f"  [{PA}][M][/{PA}] Importar de CSV"
                f"            [{PA}][X][/{PA}] Exportar para CSV\n"
                f"  [{PA}][G][/{PA}] Guaraci   [{PM}][I][/{PM}] Idioma"
                f"   [{PM}][0][/{PM}] Voltar"
                f"   [{PM}]({n_usr} do usuario)[/{PM}]"
            )
        else:
            acoes = (
                f"  [{PA}][L][/{PA}] List registered codes"
                f"   [{PA}][A][/{PA}] Add code\n"
                f"  [{PA}][M][/{PA}] Import from CSV"
                f"           [{PA}][X][/{PA}] Export to CSV\n"
                f"  [{PA}][G][/{PA}] Guaraci   [{PM}][I][/{PM}] Language"
                f"  [{PM}][0][/{PM}] Back"
                f"   [{PM}]({n_usr} user)[/{PM}]"
            )
        console.print(Text.from_markup(acoes))

        raw = _input(f"\n  {_t('opcao')}: ").upper()
        if raw in ("0","Q"): break
        elif raw == "I": _toggle_idioma(); lang = _lang()
        elif raw == "G": _abrir_assistente(_t("t_codigos"), cfg)
        elif raw == "L": _listar()
        elif raw in ("A", "C"): _adicionar()
        elif raw == "M": _importar_csv()
        elif raw == "X": _exportar_csv()
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()


# ---------------------------------------------------------------------------
# HARDWARE — dashboard compacto com barras
# ---------------------------------------------------------------------------
def menu_hardware(cfg: Optional[Config] = None) -> None:
    """Dashboard de hardware com diagnostico e recomendacoes por tier."""
    lang = _lang()
    try:
        import psutil
        ram  = psutil.virtual_memory()
        cpu_f = psutil.cpu_count(logical=False) or 1
        cpu_l = psutil.cpu_count(logical=True) or 1
        disk  = psutil.disk_usage(".")
        rt = ram.total / 1024**3
        rl = ram.available / 1024**3
        rp = ram.percent
        df = disk.free / 1024**3
        dp = disk.percent
        ok_psutil = True
    except ImportError:
        rt = rl = rp = cpu_f = cpu_l = df = dp = 0.0
        ok_psutil = False

    # Tiers com recomendacoes especificas por modulo
    if rt >= 16:
        tier, tcor = ("Alto Desempenho" if lang=="PT" else "High Performance"), PG
        tier_perfil = ("Rigor Maximo / Publicacao em Periodicos" if lang=="PT"
                       else "Maximum Rigor / Journal Publication")
        tier_mods = [
            ("Benchmark SVM/RF/XGB", "[g]Ativar[/g]"    if lang=="PT" else "[g]Enable[/g]"),
            ("Monte Carlo CV",        "[g]Ativar[/g]"    if lang=="PT" else "[g]Enable[/g]"),
            ("SHAP TreeExplainer",    "[g]Ativar[/g]"    if lang=="PT" else "[g]Enable[/g]"),
            ("SHAP max. amostras",    "[g]500+[/g]"),
            ("Permutacoes",           "[g]500 para publicacao[/g]" if lang=="PT" else "[g]500 for publication[/g]"),
        ]
    elif rt >= 8:
        tier, tcor = ("Desempenho Medio" if lang=="PT" else "Medium Performance"), PA
        tier_perfil = ("Indexacao Cientifica" if lang=="PT" else "Scientific Publication")
        tier_mods = [
            ("Benchmark SVM/RF/XGB", "[g]Ativar[/g]"    if lang=="PT" else "[g]Enable[/g]"),
            ("Monte Carlo CV",        "[warn]Opcional (lento)[/warn]" if lang=="PT" else "[warn]Optional (slow)[/warn]"),
            ("SHAP TreeExplainer",    "[warn]max 300 amostras[/warn]" if lang=="PT" else "[warn]max 300 samples[/warn]"),
            ("Permutacoes",           "[g]200 suficiente[/g]" if lang=="PT" else "[g]200 sufficient[/g]"),
        ]
    elif rt >= 4:
        tier, tcor = ("Desempenho Basico" if lang=="PT" else "Basic Performance"), PA
        tier_perfil = ("Pesquisa Exploratoria / Controle de Qualidade" if lang=="PT"
                       else "Exploratory Research / Quality Control")
        tier_mods = [
            ("Benchmark",  "[err]Desativar em [6][/err]" if lang=="PT" else "[err]Disable in [6][/err]"),
            ("Monte Carlo","[err]Desativar em [6][/err]" if lang=="PT" else "[err]Disable in [6][/err]"),
            ("SHAP",       "[err]Desativar em [6][/err]" if lang=="PT" else "[err]Disable in [6][/err]"),
            ("max_lvs",    "[warn]Reduzir para 20[/warn]" if lang=="PT" else "[warn]Reduce to 20[/warn]"),
        ]
    else:
        tier, tcor = ("Limitado" if lang=="PT" else "Limited"), PR
        tier_perfil = ("Apenas Exploracao de Dados" if lang=="PT" else "Data Exploration Only")
        tier_mods = [
            ("Todos em [6]", "[err]Desativar tudo[/err]" if lang=="PT" else "[err]Disable all[/err]"),
            ("Modo entrada",  "[warn]Usar modo sintetico[/warn]" if lang=="PT" else "[warn]Use synthetic mode[/warn]"),
            ("max_lvs",       "[warn]Reduzir para 15[/warn]" if lang=="PT" else "[warn]Reduce to 15[/warn]"),
        ]

    def _bar(pct: float, n: int = 22) -> Text:
        filled = int(pct / 100 * n)
        bar = Text()
        col = PG if pct < 70 else PA if pct < 85 else PR
        bar.append("█" * filled, style=col)
        bar.append("░" * max(0, n - filled), style=PD)
        bar.append(f" {pct:.0f}%", style=PM)
        return bar

    # Tabela principal: recursos
    hw = Table(show_header=True, header_style=PM, box=rbox.SIMPLE, padding=(0, 1))
    col_r = "Recurso" if lang=="PT" else "Resource"
    col_v = "Valor"   if lang=="PT" else "Value"
    col_u = "Uso"     if lang=="PT" else "Usage"
    hw.add_column(col_r, style=PM, width=16, no_wrap=True)
    hw.add_column(col_v, width=14, no_wrap=True)
    hw.add_column(col_u, no_wrap=True, min_width=26)

    if ok_psutil:
        hw.add_row(
            "RAM total",
            Text(f"{rt:.1f} GB", style=f"bold {PA}"),
            _bar(rp),
        )
        hw.add_row(
            "RAM disponivel" if lang=="PT" else "Available RAM",
            Text(f"{rl:.1f} GB", style=PG),
            Text(f"({100-rp:.0f}% livre)" if lang=="PT" else f"({100-rp:.0f}% free)", style=PM),
        )
        hw.add_row(
            "CPU fisicos"   if lang=="PT" else "Physical CPUs",
            Text(f"{cpu_f} cores", style=PS),
            Text(f"{cpu_l} threads logicos" if lang=="PT" else f"{cpu_l} logical threads", style=PM),
        )
        hw.add_row(
            "Disco livre"   if lang=="PT" else "Free disk",
            Text(f"{df:.1f} GB", style=PG),
            _bar(dp),
        )
    else:
        hw.add_row("psutil", Text("Nao instalado" if lang=="PT" else "Not installed", style=PR),
                   Text("pip install psutil", style=PM))

    # Tabela de recomendacoes por modulo
    rec = Table(show_header=True, header_style=PM, box=rbox.SIMPLE, padding=(0, 1))
    col_m = "Modulo" if lang=="PT" else "Module"
    col_r2 = "Recomendacao" if lang=="PT" else "Recommendation"
    rec.add_column(col_m, style=PW, width=24, no_wrap=True)
    rec.add_column(col_r2, no_wrap=True)
    for modulo, rec_str in tier_mods:
        rec.add_row(escape(modulo), Text.from_markup(rec_str))

    cap_lbl = "Capacidade" if lang=="PT" else "Capacity"
    per_lbl = "Perfil indicado" if lang=="PT" else "Recommended profile"
    cap_txt = Text.from_markup(f"  [{tcor}]{tier}[/{tcor}]  |  [{PA}]{escape(tier_perfil)}[/{PA}]")

    tit_hw   = "Recursos do Sistema" if lang=="PT" else "System Resources"
    tit_rec  = "Recomendacoes por Modulo" if lang=="PT" else "Per-Module Recommendations"

    console.print(Panel(
        Group(
            Text.from_markup(f"  [{PM}]{cap_lbl}:[/{PM}] {cap_txt.markup}"),
            Text(""),
            hw,
            Rule(style=PD),
            Text.from_markup(f"  [{PM}]{tit_rec}:[/{PM}]"),
            rec,
        ),
        title=f"[bold {PS}]{_t('t_hardware')}[/bold {PS}]",
        border_style=PS, box=rbox.ROUNDED, padding=(0, 1)
    ))
    console.print()
    raw = _ask(f"  [{PA}][G][/{PA}] Guaraci   [{PM}][0][/{PM}] {_t('voltar')}: ").strip().upper()
    if raw == "G":
        _abrir_assistente(_t("t_hardware"), cfg)


# ---------------------------------------------------------------------------
# PREDICAO EM LOTE — aplica modelo salvo (.joblib) a espectros novos (CSV)
# ---------------------------------------------------------------------------
def menu_predicao(cfg: Optional[Config] = None) -> None:
    """Predicao em lote via terminal: aplica um modelo .joblib salvo a um
    CSV de espectros novos (colunas=numero de onda, sem coluna de classe).

    Mesma logica cientifica do app web (aba Prediction) via predicao.py --
    zero duplicacao entre as duas interfaces (mesmo padrao da Fase H).
    Abre porta pra integracao com scripts/LIMS sem precisar de navegador.
    """
    lang = _lang()
    is_pt = lang == "PT"
    cls(); _print_header()

    intro = (
        "Aplica um modelo treinado (.joblib) a espectros novos e reporta "
        "classe predita + diagnostico T2/Q (dominio de aplicabilidade)."
        if is_pt else
        "Applies a trained model (.joblib) to new spectra and reports "
        "predicted class + T2/Q diagnostics (applicability domain)."
    )
    console.print(Panel(
        Text.from_markup(f"  {intro}"),
        title=f"[bold {PS}]{_t('t_predicao')}[/bold {PS}]",
        border_style=PS, box=rbox.ROUNDED, padding=(1, 2),
    ))
    console.print()

    lbl_modelo = "Caminho do modelo (.joblib)" if is_pt else "Model path (.joblib)"
    lbl_csv    = "Caminho do CSV de espectros novos" if is_pt else "New spectra CSV path"
    lbl_saida  = "Caminho de saida do CSV de resultados" if is_pt else "Output results CSV path"
    nao_encontrado = "Arquivo nao encontrado" if is_pt else "File not found"

    cam_modelo = _ask(f"  [{PA}]{lbl_modelo}:[/{PA}] ").strip().strip('"')
    if not cam_modelo:
        return
    if not os.path.isfile(cam_modelo):
        console.print(f"  [{PR}]{nao_encontrado}: {escape(cam_modelo)}[/{PR}]")
        _pause(); return

    # Aviso de seguranca (P5): .joblib e' pickle -- executa codigo arbitrario
    # NO CARREGAMENTO, antes de qualquer validacao ser possivel. Confirmacao
    # explicita do operador e' a unica protecao real (ver docs/SECURITY.md).
    aviso_pickle = (
        f"  [{PR}]⚠ '.joblib' executa codigo ao ser carregado (formato "
        f"pickle). So confirme se voce mesmo treinou este modelo ou confia "
        f"plenamente na origem.[/{PR}]" if is_pt else
        f"  [{PR}]⚠ '.joblib' runs code when loaded (pickle format). Only "
        f"confirm if you trained this model yourself or fully trust its "
        f"source.[/{PR}]")
    console.print(aviso_pickle)
    conf_lbl = "Confirma o carregamento? (s/n)" if is_pt else "Confirm loading? (y/n)"
    if _ask(f"  [{PA}]{conf_lbl}[/{PA}] ").strip().lower() not in ("s", "y", "sim", "yes"):
        console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return

    cam_csv = _ask(f"  [{PA}]{lbl_csv}:[/{PA}] ").strip().strip('"')
    if not cam_csv:
        return
    if not os.path.isfile(cam_csv):
        console.print(f"  [{PR}]{nao_encontrado}: {escape(cam_csv)}[/{PR}]")
        _pause(); return

    padrao_saida = str(Path(cam_csv).with_name(Path(cam_csv).stem + "_predicao.csv"))
    cam_saida = _ask(
        f"  [{PA}]{lbl_saida}[/{PA}] [{PM}](Enter = {escape(padrao_saida)})[/{PM}]: "
    ).strip().strip('"')
    if not cam_saida:
        cam_saida = padrao_saida

    try:
        import pandas as pd
        import guaraci.predicao as _pred
        status_msg = ("Carregando modelo e aplicando..." if is_pt
                       else "Loading model and applying...")
        with console.status(f"[{PA}]{status_msg}[/{PA}]"):
            # confiar=True: o operador ja confirmou explicitamente acima.
            pkg = _pred.carregar_modelo(cam_modelo, confiar=True)
            _pred.validar_pacote_modelo(pkg)
            X_new, wn_new, meta_df = _pred.carregar_csv_predicao(cam_csv)
            df_res = _pred.predizer_amostras(pkg, X_new, wn_new)
            if len(meta_df.columns) > 0 and len(meta_df) == len(df_res):
                df_res = pd.concat([meta_df.reset_index(drop=True), df_res], axis=1)
            df_res.to_csv(cam_saida, index=False, sep=";", decimal=",")
    except Exception as e:  # noqa: BLE001 -- multi-etapa (joblib.load +
        # validacao do pacote + parsing CSV + predicao + escrita); erro
        # exibido ao usuario, tela volta ao menu sem crashar a CLI.
        console.print(f"  [{PR}]{'Erro' if is_pt else 'Error'}: {escape(str(e))}[/{PR}]")
        _pause(); return

    # Resumo
    n_tot = len(df_res)
    n_ac  = int(df_res["aceito"].sum()) if "aceito" in df_res.columns else n_tot
    t_res = Table(show_header=True, header_style=PM, box=rbox.SIMPLE, padding=(0, 1))
    t_res.add_column("Classe" if is_pt else "Class", style=PW)
    t_res.add_column("N")
    if "classe_pred" in df_res.columns:
        for cls_pred, n in df_res["classe_pred"].value_counts().items():
            t_res.add_row(escape(str(cls_pred)), str(n))

    resumo_txt = (
        f"  [{PG}]✔ {n_tot} amostras processadas[/{PG}]  |  "
        f"[{PG}]{n_ac}[/{PG}] aceitas (ajuste ao modelo PLS-DA, T2/Q) / "
        f"[{PR}]{n_tot - n_ac}[/{PR}] rejeitadas"
        if is_pt else
        f"  [{PG}]✔ {n_tot} samples processed[/{PG}]  |  "
        f"[{PG}]{n_ac}[/{PG}] accepted (PLS-DA model fit, T2/Q) / "
        f"[{PR}]{n_tot - n_ac}[/{PR}] rejected"
    )
    # Dominio de Aplicabilidade (PCA exploratorio, Jaworska et al. 2005) --
    # so' aparece se o pacote .joblib foi salvo por uma versao do pipeline
    # que exporta os artefatos leves (retrocompativel com pacotes antigos).
    if "AD_dentro_dominio" in df_res.columns:
        n_ad_dentro = int(df_res["AD_dentro_dominio"].sum())
        ad_txt = (
            f"  [{PG}]🔎 Dominio de aplicabilidade:[/{PG}] "
            f"[{PG}]{n_ad_dentro}[/{PG}] dentro / "
            f"[{PR}]{n_tot - n_ad_dentro}[/{PR}] fora "
            "(espectro atipico frente a calibracao)"
            if is_pt else
            f"  [{PG}]🔎 Applicability domain:[/{PG}] "
            f"[{PG}]{n_ad_dentro}[/{PG}] within / "
            f"[{PR}]{n_tot - n_ad_dentro}[/{PR}] outside "
            "(atypical spectrum vs. calibration)"
        )
        resumo_txt += "\n" + ad_txt
    console.print()
    console.print(Panel(
        Group(Text.from_markup(resumo_txt), Text(""), t_res),
        title=f"[bold {PG}]{'Resultado' if is_pt else 'Result'}[/bold {PG}]",
        border_style=PG, box=rbox.ROUNDED, padding=(1, 2),
    ))
    console.print(f"  [{PM}]{'Salvo em' if is_pt else 'Saved to'}:[/{PM}] "
                  f"{escape(cam_saida)}")
    _pause()


# ---------------------------------------------------------------------------
# PERFIS — cartoes compactos (2 por linha)
# ---------------------------------------------------------------------------
def menu_perfis(cfg: Config) -> None:
    """Perfis prontos — lista enxuta de 1 linha; detalhes so com [?]."""
    lang = _lang()
    # (nome_chave, tempo, cor, foco_curto). Foco curto = 1 linha, sem cortar.
    # Os 3 primeiros escolhem O QUE analisar (objetivo cientifico); os
    # demais escolhem QUAO A FUNDO (rigor). Cor PA nos 3 primeiros = "comece
    # aqui" (CLAUDE.md secao 6: presets Autenticar/Explorar/Quantificar).
    perfis = [
        ("Explorar Dados",            "~5-10 min",  PA,
         "Primeiro olhar: PCA/HCA, sem forcar classificacao" if lang=="PT"
         else "First look: PCA/HCA, no forced classification"),
        ("Autenticar Pureza",         "~10-20 min", PA,
         "Puro vs. adulterado por especie (DD-SIMCA)" if lang=="PT"
         else "Pure vs. adulterated per species (DD-SIMCA)"),
        ("Quantificar Teor",          "~15-30 min", PA,
         "Teor de adulterante por regressao PLS" if lang=="PT"
         else "Adulterant content via PLS regression"),
        ("Exploracao Rapida",         "~5 min",     PW,
         "Teste rapido do pipeline" if lang=="PT" else "Quick pipeline test"),
        ("Analise Padrao",            "~15-30 min", PF,
         "Uso geral equilibrado (recomendado)" if lang=="PT" else "Balanced general use (recommended)"),
        ("Pesquisa Academica",        "~30-45 min", PS,
         "Validacao estatistica reforcada" if lang=="PT" else "Reinforced statistical validation"),
        ("Publicacao Cientifica",     "~1-2 horas", PG,
         "Benchmark + SHAP para periodico" if lang=="PT" else "Benchmark + SHAP for journals"),
        ("Alta Rigorosidade",         "~3-6 horas", PR,
         "Monte Carlo + tudo (tese/dissertacao)" if lang=="PT" else "Monte Carlo + all (thesis)"),
        ("Benchmark Preprocessamento","~20-40 min", PS,
         "Comparar pre-processamentos" if lang=="PT" else "Compare preprocessings"),
        ("Acessibilidade",            "~15-30 min", PM,
         "Cores seguras p/ daltonismo" if lang=="PT" else "Colorblind-safe palette"),
    ]

    def _aplicar(pname: str) -> int:
        pdata = PROFILES.get(pname, {})
        n = 0
        for k, v in pdata.items():
            if k.startswith("_"):
                continue
            sp = _SPEC_BY_KEY.get(k)
            if sp:
                try:
                    setattr(cfg, sp["attr"], v); n += 1
                except (AttributeError, TypeError) as _e_prof:
                    # Campo de perfil desalinhado com o Config atual (dado
                    # constante do PROFILES, nao input do usuario) -- pulado.
                    logging.getLogger(__name__).debug(
                        "perfil '%s': campo '%s' nao aplicado: %s",
                        pname, k, _e_prof)
        paleta = pdata.get("_paleta")
        if paleta and PALETAS_COR and paleta in PALETAS_COR:
            vcfg = _carregar_visual_cfg()
            vcfg["paleta"] = paleta
            _salvar_visual_cfg(vcfg); n += 1
        return n

    def _detalhe(pname: str) -> None:
        desc = PROFILE_DESC.get(pname, {}).get(lang, "")
        summ = PROFILE_KEY_SUMMARY.get(pname, {}).get(lang, "")
        corpo = Text()
        if desc:
            corpo.append(desc.strip() + "\n", style=PW)
        if summ:
            corpo.append("\n" + summ, style=PM)
        console.print(Panel(
            corpo if (desc or summ) else Text("—", style=PM),
            title=f"[bold {PA}]{escape(pname)}[/bold {PA}]",
            border_style=PA, box=rbox.ROUNDED, padding=(0, 2), width=_W()
        ))
        _pause()

    while True:
        cls(); _print_header()

        t = Table(box=None, show_header=True, header_style=PM, padding=(0, 1))
        t.add_column("N",      style=PA, width=4, no_wrap=True)
        t.add_column("Perfil" if lang=="PT" else "Profile", width=26, no_wrap=True)
        t.add_column("Tempo" if lang=="PT" else "Time", style=PS, width=12, no_wrap=True)
        t.add_column("Foco" if lang=="PT" else "Focus", style=PW)
        for i, (pname, tempo, cor, foco) in enumerate(perfis, 1):
            estrela = " ★" if pname == "Analise Padrao" else ""
            t.add_row(f"[{i}]", Text.from_markup(f"[{cor}]{escape(pname)}[/{cor}]{estrela}"),
                      tempo, escape(foco))

        sub = ("Selecione para aplicar. [?] N = detalhes." if lang=="PT"
               else "Select to apply. [?] N = details.")
        console.print(Panel(
            t,
            title=f"[bold {PA}]{_t('t_perfis')}[/bold {PA}]",
            subtitle=f"[{PM}]{sub}[/{PM}]",
            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)
        ))

        # Perfis salvos pelo usuário
        _PERFIS_DIR.mkdir(parents=True, exist_ok=True)
        salvos = sorted(_PERFIS_DIR.glob("*.yaml"))
        if salvos:
            sl = (f"  [{PM}]Salvos:[/{PM}] " if lang=="PT" else f"  [{PM}]Saved:[/{PM}] ")
            sl += ", ".join(f.stem for f in salvos[:5])
            console.print(sl)

        if lang == "PT":
            rod = (f"  [{PA}][?][/{PA}] N detalhes   [{PA}][S][/{PA}] salvar config atual"
                   f"   [{PA}][L][/{PA}] carregar salvo\n"
                   f"  [{PA}][G][/{PA}] Guaraci   [{PM}][I][/{PM}] Idioma   [{PM}][0][/{PM}] Voltar")
        else:
            rod = (f"  [{PA}][?][/{PA}] N details   [{PA}][S][/{PA}] save current config"
                   f"   [{PA}][L][/{PA}] load saved\n"
                   f"  [{PA}][G][/{PA}] Guaraci   [{PM}][I][/{PM}] Language   [{PM}][0][/{PM}] Back")
        console.print(Text.from_markup(rod))

        raw = _input(f"\n  [1-{len(perfis)}] / [?] / [S] / [L] / [0]: ").strip().upper()
        if raw in ("0","Q",""): break
        elif raw == "I": _toggle_idioma(); lang = _lang()
        elif raw == "G": _abrir_assistente(_t("t_perfis"), cfg)
        elif raw == "S":
            _salvar_yaml(cfg)
        elif raw == "L":
            _carregar_yaml(cfg)
        elif raw == "?":
            r2 = _input("  N: ").strip()
            if r2.isdigit() and 1 <= int(r2) <= len(perfis):
                _detalhe(perfis[int(r2)-1][0])
        elif raw.isdigit() and 1 <= int(raw) <= len(perfis):
            pname = perfis[int(raw)-1][0]
            n = _aplicar(pname)
            console.print(f"  [g]✓ {'Perfil' if lang=='PT' else 'Profile'} "
                          f"'{escape(pname)}' {'aplicado' if lang=='PT' else 'applied'} "
                          f"({n} {'campos' if lang=='PT' else 'fields'})[/g]")
            # "Rodar analise recomendada" (CLAUDE.md secao 6): aplicar +
            # rodar num so' fluxo, sem precisar voltar ao menu principal e
            # digitar R separadamente. Continua exigindo confirmacao (nunca
            # roda sem o usuario decidir).
            pergunta = ("  Rodar agora com essa configuracao? [S/n]: " if lang=="PT"
                        else "  Run now with this configuration? [Y/n]: ")
            resp = _input(pergunta).strip().lower()
            if resp in ("", "s", "y", "sim", "yes"):
                _rodar_pipeline(cfg)
            else:
                _pause()
            break
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()


# ---------------------------------------------------------------------------
# SOBRE — identidade, citacao e referencias (para publicacao)
# ---------------------------------------------------------------------------
def _ler_citation() -> dict:
    """Le campos basicos do CITATION.cff (parser leve, sem dependencia YAML)."""
    info: Dict[str, str] = {}
    p = _BASE_DIR / "CITATION.cff"
    if not p.exists():
        return info
    try:
        for linha in p.read_text(encoding="utf-8").splitlines():
            if ":" in linha and not linha.lstrip().startswith("-"):
                chave, _, val = linha.partition(":")
                val = val.strip().strip('"').strip()
                if val:
                    info[chave.strip()] = val
    except OSError:
        pass
    return info


def menu_sobre(cfg: Optional[Config] = None) -> None:
    """Secao Sobre — proposito, autor, citacao em multiplos formatos e referencias."""
    # Dados fixos do projeto e do autor
    _AUTOR_NOME    = "Erley S. da Costa"
    _AUTOR_TAG     = "Pesquisador / Desenvolvedor  |  GEAAp / UFPA"
    _AUTOR_LATTES  = "http://lattes.cnpq.br/5755582193284309"
    _AUTOR_GITHUB  = "https://github.com/ErleySC"
    _AUTOR_EMAIL   = "erleysdacosta@gmail.com"
    _REPO          = "https://github.com/ErleySC/guaraci"
    _VERSAO        = pq.__version__
    _ANO           = "2026"
    _TITULO_CURTO  = "GUARACI"
    _TITULO_LONGO  = ("Inteligencia Quimiometrica para Matrizes Amazonicas"
                      if True else "")  # resolvido abaixo por lang
    _INST          = "GEAAp/UFPA"
    _LIC           = "GPL-3.0-or-later"

    def _titulo(lang: str) -> str:
        return ("Inteligencia Quimiometrica para Matrizes Amazonicas"
                if lang == "PT" else
                "Chemometric Intelligence for Amazonian Matrices")

    def _painel_identidade(lang: str) -> None:
        """Painel principal: nome, proposito e links rapidos."""
        t = Table(box=None, show_header=False, padding=(0, 1), expand=True)
        t.add_column("L", style=PA, width=14, no_wrap=True)
        t.add_column("V", style=PW, overflow="fold")

        # Titulo
        t.add_row(
            "",
            Text.from_markup(f"[bold {PA}]{_TITULO_CURTO}[/bold {PA}]"
                             f"[{PW}] — {escape(_titulo(lang))}[/{PW}]"),
        )
        t.add_row("", Text(""))

        # Proposito
        if lang == "PT":
            p1 = ("Democratizar o acesso a analises quimiometricas de alta qualidade"
                  " para pesquisadores que nao dominam programacao.")
            p2 = ("Oferece um ambiente confiavel, reproducivel e bilingue (PT/EN)"
                  " para classificacao, autenticacao e exploracao de matrizes complexas"
                  " — do FT-NIR ao GC-MS, sem escrever uma linha de codigo.")
            p3 = ("Desenvolvido no ambito de uma pesquisa PIBIC/UFPA sobre oleos"
                  " vegetais amazonicos, com metodologia generalizavel para"
                  " qualquer tecnica analitica com dados multivariados.")
        else:
            p1 = ("Democratize access to high-quality chemometric analyses"
                  " for researchers without a programming background.")
            p2 = ("Provides a reliable, reproducible and bilingual (PT/EN)"
                  " environment for classification, authentication and exploration"
                  " of complex matrices — from FT-NIR to GC-MS, without writing code.")
            p3 = ("Developed within a PIBIC/UFPA research project on Amazonian"
                  " vegetable oils, with a methodology generalized to any"
                  " analytical technique with multivariate data.")
        prop_lbl = "Proposito" if lang == "PT" else "Purpose"
        t.add_row(f"{prop_lbl}:", Text(p1, style=PW))
        t.add_row("", Text(p2, style=PM))
        t.add_row("", Text(p3, style=PM))
        t.add_row("", Text(""))

        # Tecnicas suportadas
        tec_lbl = "Tecnicas" if lang == "PT" else "Techniques"
        tec_val = ("FT-NIR · NIR · MIR/FTIR · Raman · UV-Vis · Fluorescencia"
                   " · HPLC · GC-MS · NMR · IMS · Generica")
        t.add_row(f"{tec_lbl}:", Text(tec_val, style=PW))
        t.add_row("", Text(""))

        # Metadados
        ver_lbl = "Versao" if lang == "PT" else "Version"
        lic_lbl = "Licenca" if lang == "PT" else "License"
        t.add_row(f"{ver_lbl}:", Text(f"{_VERSAO}  ({_ANO})", style=PS))
        t.add_row(f"{lic_lbl}:", Text(_LIC, style=PS))
        t.add_row("Repo:", Text(_REPO, style=PS))

        titulo_p = "Sobre" if lang == "PT" else "About"
        console.print(Panel(t,
            title=f"[bold {PA}]{titulo_p}[/bold {PA}]",
            border_style=PA, box=rbox.ROUNDED, padding=(1, 2), width=_W()))

    def _painel_autor(lang: str) -> None:
        """Painel do autor com Lattes, GitHub e tag."""
        t = Table(box=None, show_header=False, padding=(0, 1), expand=True)
        t.add_column("L", style=PA, width=14, no_wrap=True)
        t.add_column("V", style=PW, overflow="fold")

        nome_lbl  = "Nome" if lang == "PT" else "Name"
        cargo_lbl = "Cargo" if lang == "PT" else "Role"
        t.add_row(f"{nome_lbl}:",  Text(_AUTOR_NOME,   style=f"bold {PW}"))
        t.add_row(f"{cargo_lbl}:", Text(_AUTOR_TAG,    style=PW))
        t.add_row("", Text(""))
        t.add_row("Lattes:",  Text(_AUTOR_LATTES, style=PS))
        t.add_row("GitHub:",  Text(_AUTOR_GITHUB, style=PS))
        t.add_row("E-mail:",  Text(_AUTOR_EMAIL,  style=PS))
        t.add_row("", Text(""))
        t.add_row("Projeto:", Text(_REPO, style=PS))

        titulo_a = "Autor" if lang == "PT" else "Author"
        console.print(Panel(t,
            title=f"[bold {PA}]{titulo_a}[/bold {PA}]",
            border_style=PF, box=rbox.ROUNDED, padding=(1, 2), width=_W()))
        _pause()

    def _painel_citar(lang: str) -> None:
        """Formatos de citacao: APA, ABNT, BibTeX."""
        tit_full = f"{_TITULO_CURTO}: {_titulo(lang)}"
        autor_abnt = "COSTA, E. S. da"

        # APA
        apa = (f"Costa, E. S. da. ({_ANO}). {tit_full} (v{_VERSAO})"
               f" [Software]. {_INST}. {_REPO}")
        # ABNT (NBR 6023:2018 — software)
        abnt = (f"{autor_abnt}. {_TITULO_CURTO}: {_titulo('PT')}."
                f" Versao {_VERSAO}. {_INST}, {_ANO}."
                f" Disponivel em: <{_REPO}>.")
        # BibTeX
        bibtex = (
            f"@software{{guaraci_{_ANO},\n"
            f"  author    = {{Costa, Erley S. da}},\n"
            f"  title     = {{{{{_TITULO_CURTO}: {_titulo('PT')}}}}},\n"
            f"  version   = {{{_VERSAO}}},\n"
            f"  year      = {{{_ANO}}},\n"
            f"  institution = {{{_INST}}},\n"
            f"  url       = {{{_REPO}}},\n"
            f"  license   = {{{_LIC}}}\n"
            f"}}"
        )

        corpo = Text()
        corpo.append("APA\n", style=f"bold {PA}")
        corpo.append(apa + "\n\n", style=PW)
        corpo.append("ABNT (NBR 6023:2018)\n", style=f"bold {PA}")
        corpo.append(abnt + "\n\n", style=PW)
        corpo.append("BibTeX\n", style=f"bold {PA}")
        corpo.append(bibtex + "\n\n", style=PS)
        nota = ("Detalhes completos em CITATION.cff (raiz do projeto)."
                if lang == "PT" else
                "Full details in CITATION.cff (project root).")
        corpo.append(nota, style=PM)

        titulo_c = "Como Citar" if lang == "PT" else "How to Cite"
        console.print(Panel(corpo,
            title=f"[bold {PA}]{titulo_c}[/bold {PA}]",
            border_style=PF, box=rbox.ROUNDED, padding=(1, 2), width=_W()))
        _pause()

    def _painel_diferenciais(lang: str) -> None:
        """Comparativo com softwares pagos — posicionamento do projeto."""
        t = Table(show_header=True, header_style=f"bold {PA}", box=rbox.SIMPLE,
                  padding=(0, 1), expand=True)
        crit = "Criterio" if lang == "PT" else "Criterion"
        pagos = "Pagos*" if lang == "PT" else "Paid*"
        t.add_column(crit, style=PW, width=30)
        t.add_column("GUARACI", style=PG, justify="center", width=10)
        t.add_column(pagos, style=PM, justify="center", width=10)

        if lang == "PT":
            linhas = [
                ("Custo de licenca", "Gratuito", "Alto"),
                ("Codigo aberto / auditavel", "Sim", "Nao"),
                ("Validacao anti-vazamento (group-aware)", "Padrao", "Manual"),
                ("Reprodutibilidade (seeds, versionado)", "Sim", "Parcial"),
                ("Uso sem programar", "Sim", "Sim (GUI)"),
                ("Bilingue PT / EN", "Sim", "Raro"),
                ("Multitecnica (NIR a GC-MS)", "Sim", "Sim"),
                ("Relatorios prontos (PDF/Word/PPTX)", "Sim", "Parcial"),
                ("Roda offline, sem nuvem obrigatoria", "Sim", "Varia"),
            ]
            nota = ("* Refere-se a softwares comerciais como MATLAB/PLS_Toolbox,\n"
                    "  The Unscrambler, SIMCA e similares. Comparativo informativo.")
            intro = ("GUARACI nasce para democratizar a quimiometria de alto nivel:\n"
                     "o rigor de um software pago, sem o custo e sem travar voce\n"
                     "em um formato fechado. Ciencia aberta, reproduzivel e acessivel.")
        else:
            linhas = [
                ("License cost", "Free", "High"),
                ("Open source / auditable", "Yes", "No"),
                ("Leakage-safe validation (group-aware)", "Default", "Manual"),
                ("Reproducibility (seeds, versioned)", "Yes", "Partial"),
                ("Usable without coding", "Yes", "Yes (GUI)"),
                ("Bilingual PT / EN", "Yes", "Rare"),
                ("Multi-technique (NIR to GC-MS)", "Yes", "Yes"),
                ("Ready-made reports (PDF/Word/PPTX)", "Yes", "Partial"),
                ("Runs offline, no mandatory cloud", "Yes", "Varies"),
            ]
            nota = ("* Refers to commercial software such as MATLAB/PLS_Toolbox,\n"
                    "  The Unscrambler, SIMCA and similar. Informative comparison.")
            intro = ("GUARACI exists to democratize high-end chemometrics:\n"
                     "the rigor of paid software, without the cost and without\n"
                     "locking you into a closed format. Open, reproducible, accessible.")
        for c, a, b in linhas:
            t.add_row(c, a, b)

        tit_d = ("Por que o GUARACI?" if lang == "PT" else "Why GUARACI?")
        console.print(Panel(
            Group(Text(intro, style=PW), Text(""), t, Text(""), Text(nota, style=PM)),
            title=f"[bold {PA}]{tit_d}[/bold {PA}]",
            border_style=PF, box=rbox.ROUNDED, padding=(1, 2), width=_W()))
        _pause()

    def _painel_referencias(lang: str) -> None:
        """Referencias metodologicas fundamentais (max 5)."""
        fundamentais = [
            "pls_da_brereton",
            "opls_da_trygg_2002",
            "dd_simca_pomerantsev",
            "savitzky_golay_1964",
            "monte_carlo_cv_xu",
        ]
        t = Table(show_header=False, box=rbox.SIMPLE, padding=(0, 1), expand=True)
        t.add_column("•", style=PA, width=2, no_wrap=True)
        t.add_column("Ref", style=PW, overflow="fold")
        achou = False
        for rk in fundamentais:
            ref = (REFERENCIAS_GUARACI or {}).get(rk, {})
            cit_txt = ref.get("cit")
            ctx     = ref.get("contexto", "")
            if cit_txt:
                t.add_row("•", Text.from_markup(
                    f"[{PM}]{escape(ctx)}[/{PM}]\n[{PW}]{escape(cit_txt)}[/{PW}]"))
                achou = True
        tit_r = ("Referencias Fundamentais" if lang == "PT"
                 else "Key Methodological References")
        sub_r = ("Metodologias implementadas no pipeline."
                 if lang == "PT" else "Methodologies implemented in the pipeline.")
        console.print(Panel(
            t if achou else Text("—", style=PM),
            title=f"[bold {PA}]{tit_r}[/bold {PA}]",
            subtitle=f"[{PM}]{sub_r}[/{PM}]",
            border_style=PF, box=rbox.ROUNDED, padding=(1, 2), width=_W()))
        _pause()

    # Loop principal da secao Sobre
    while True:
        lang = _lang()
        cls(); _print_header()
        _painel_identidade(lang)

        if lang == "PT":
            console.print(
                f"  [{PA}][D][/{PA}] Por que o GUARACI?"
                f"   [{PA}][A][/{PA}] Autor / Contato"
                f"   [{PA}][C][/{PA}] Como Citar"
                f"   [{PA}][R][/{PA}] Referencias\n"
                f"  [{PA}][G][/{PA}] Guaraci"
                f"   [{PM}][I][/{PM}] Idioma"
                f"   [{PM}][0][/{PM}] Voltar"
            )
        else:
            console.print(
                f"  [{PA}][D][/{PA}] Why GUARACI?"
                f"   [{PA}][A][/{PA}] Author / Contact"
                f"   [{PA}][C][/{PA}] How to Cite"
                f"   [{PA}][R][/{PA}] References\n"
                f"  [{PA}][G][/{PA}] Guaraci"
                f"   [{PM}][I][/{PM}] Language"
                f"   [{PM}][0][/{PM}] Back"
            )

        raw = _input(f"\n  {_t('opcao')}: ").strip().upper()
        if raw in ("0", "Q", ""): break
        elif raw == "I": _toggle_idioma()
        elif raw == "G": _abrir_assistente("Sobre", cfg)
        elif raw == "D": _painel_diferenciais(lang)
        elif raw == "A": _painel_autor(lang)
        elif raw == "C": _painel_citar(lang)
        elif raw == "R": _painel_referencias(lang)
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()


# ---------------------------------------------------------------------------
# AJUDA INTERATIVA
# ---------------------------------------------------------------------------
def menu_ajuda(cfg: Optional[Config] = None) -> None:
    """Ajuda navegavel — lista todos os campos de cara; numero abre a ajuda."""
    lang = _lang()
    # Lista unificada a partir do _CONFIG_SPEC (todos os campos editaveis).
    keys = [s["key"] for s in _CONFIG_SPEC] if _CONFIG_SPEC else list(HELP_DB.keys())
    # Remove duplicatas mantendo ordem
    seen: set = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]

    while True:
        cls(); _print_header()

        t = Table(show_header=True, header_style=PM, box=rbox.SIMPLE, padding=(0, 1))
        t.add_column("N", style=PA, width=4, no_wrap=True)
        t.add_column("Campo" if lang=="PT" else "Field", width=24, no_wrap=True)
        t.add_column("Tipo" if lang=="PT" else "Risk", width=11, no_wrap=True)
        t.add_column("Descricao" if lang=="PT" else "Description", style=PM)
        for i, key in enumerate(keys, 1):
            r_hex = _risco_hex(key)
            t.add_row(
                str(i),
                escape(_nome_campo(key)),
                Text(_risco_icon(key) + " " + RISK_CLASS.get(key, "—"), style=r_hex),
                escape(_desc_curta(key, 40)),
            )

        sub = ("Digite o numero do campo para ver a ajuda completa, ou busque por nome."
               if lang=="PT" else
               "Type the field number for full help, or search by name.")
        console.print(Panel(
            t,
            title=f"[bold {PA}]{_t('t_ajuda')}[/bold {PA}]",
            subtitle=f"[{PM}]{sub}[/{PM}]",
            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)
        ))
        if lang == "PT":
            console.print(f"  [{PA}][G][/{PA}] Guaraci   [{PM}][I][/{PM}] Idioma   [{PM}][0][/{PM}] Voltar")
        else:
            console.print(f"  [{PA}][G][/{PA}] Guaraci   [{PM}][I][/{PM}] Language   [{PM}][0][/{PM}] Back")

        raw = _input(f"\n  {_t('opcao')}: ").strip()
        if raw in ("0","Q","q",""):
            break
        elif raw.upper() == "I":
            _toggle_idioma(); lang = _lang()
        elif raw.upper() == "G":
            _abrir_assistente(_t("t_ajuda"), cfg)
        elif raw.lower().startswith("help "):
            campo = raw[5:].strip()
            found = [k for k in keys if campo.lower() in k.lower() or campo.lower() in _nome_campo(k).lower()]
            (_mostrar_ajuda(found[0]) if found
             else (console.print(f"  [{PM}]{'Nao encontrado.' if lang=='PT' else 'Not found.'}[/{PM}]"), _pause()))
        elif raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(keys):
                _mostrar_ajuda(keys[idx])
            else:
                console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()
        elif raw in keys:
            _mostrar_ajuda(raw)
        else:
            found = [k for k in keys if raw.lower() in k.lower() or raw.lower() in _nome_campo(k).lower()]
            if found:
                _mostrar_ajuda(found[0])
            else:
                console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()


# ---------------------------------------------------------------------------
# CHECKLIST PRE-EXECUCAO
# ---------------------------------------------------------------------------
def _checklist(cfg: Config) -> Tuple[bool, List]:
    lang = _lang()
    checks = []; erros = []

    pasta = _cfgv(cfg, "pasta_dados", "dados")
    pasta_ok = bool(pasta) and os.path.isdir(str(pasta))
    n_dx = _contar_dx(pasta) if pasta_ok else 0

    if pasta_ok and n_dx > 0:
        checks.append((True,  _t("chk_dados") + f" ({n_dx} .dx)"))
    elif pasta_ok:
        checks.append((None,  _t("chk_dados") + " (0 .dx)"))
        erros.append("pasta_dados vazia")
    else:
        checks.append((False, _t("chk_err_dados")))
        erros.append("pasta_dados")

    modo = _cfgv(cfg, "modo_entrada", "dx")
    if modo == "csv":
        arq = _cfgv(cfg, "arquivo_csv", "")
        if arq and os.path.isfile(str(arq)):
            checks.append((True, _t("chk_csv")))
        else:
            checks.append((False, _t("chk_err_csv")))
            erros.append("arquivo_csv")
    else:
        checks.append((True, f"Modo: {modo}"))

    ga = _cfgv(cfg, "validacao_group_aware", True)
    if ga:
        checks.append((True,  _t("chk_leak")))
    else:
        checks.append((False, _t("chk_err_leak")))
        erros.append("validacao_group_aware")

    ps = _cfgv(cfg, "pasta_saida", "")
    checks.append((True if ps else None, f"{_t('chk_saida')}: {ps or '(padrao)'}"))

    try:
        import psutil
        rt = psutil.virtual_memory().total / 1024**3
        bench = _cfgv(cfg, "benchmark", False)
        mc = _cfgv(cfg, "monte_carlo", False)
        shap = _cfgv(cfg, "shap_benchmark", False)
        if rt < 4 and (bench or mc or shap):
            checks.append((None, _t("chk_warn_hw") + f" ({rt:.1f}GB)"))
        else:
            checks.append((True, _t("chk_hw") + f" ({rt:.1f}GB)"))
    except ImportError:
        checks.append((None, "psutil N/A"))

    pp = _cfgv(cfg, "pre_processamento", "")
    checks.append((True if pp else None, f"{_t('chk_preproc')}: {pp or '—'}"))

    return (len(erros) == 0), erros, checks


def _print_checklist(cfg: Config) -> bool:
    ok, erros, checks = _checklist(cfg)
    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column("S", width=3, no_wrap=True)
    t.add_column("Item", style=PW)
    for estado, msg in checks:
        if estado is True:
            t.add_row(Text("✓", style=PG), escape(msg))
        elif estado is False:
            t.add_row(Text("✖", style=PR), escape(msg))
        else:
            t.add_row(Text("—", style=PM), escape(msg))
    b_hex = PG if ok else PR
    lbl = "Checklist Pre-Execucao" if _lang()=="PT" else "Pre-Execution Checklist"
    console.print(Panel(t, title=f"[{PA}]{lbl}[/{PA}]",
                        border_style=b_hex, box=rbox.ROUNDED, padding=(0, 1)))
    return ok


# ---------------------------------------------------------------------------
# RESUMO CIENTIFICO
# ---------------------------------------------------------------------------
def _print_resumo(cfg: Config) -> None:
    lang = _lang()
    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column("P", style=PM, width=22, no_wrap=True)
    t.add_column("V", style=PS, no_wrap=True)

    def row(lbl, val, fmt="s"):
        if isinstance(val, bool):
            v_txt = Text("[g]Sim[/g]" if lang=="PT" else "[g]Yes[/g]") if val \
                    else Text("[m]Nao[/m]" if lang=="PT" else "[m]No[/m]")
            t.add_row(lbl, v_txt)
        else:
            t.add_row(lbl, Text(str(val), style=PS))

    row(_t("res_tecnica"),  _TECNICA_SELECIONADA.get("nome", "FT-NIR"))
    row(_t("res_preproc"),  _cfgv(cfg, "pre_processamento", "—"))
    row(_t("res_modelo"),   "PLS-DA")
    row(_t("res_lvs"),      _cfgv(cfg, "max_lvs", "—"))
    row(_t("res_valid"),    "GroupKFold" if _cfgv(cfg, "validacao_group_aware", True) else "[err]KFold[/err]")
    row(_t("res_perm"),     _cfgv(cfg, "n_permutacoes", "—"))
    row(_t("res_opls"),     _cfgv(cfg, "opls_da", True))
    row(_t("res_dds"),      _cfgv(cfg, "ddsimca", True))
    row(_t("res_bench"),    _cfgv(cfg, "benchmark", False))
    row(_t("res_mc"),       _cfgv(cfg, "monte_carlo", False))
    row(_t("res_shap"),     _cfgv(cfg, "shap_benchmark", False))
    row(_t("res_dpi"),      _cfgv(cfg, "dpi", 300))
    row(_t("res_fmt"),      _cfgv(cfg, "formato_figura", "png"))
    row(_t("res_nivel"),    _rotulo_opcao("nivel", _cfgv(cfg, "nivel", "N1")))
    tag = getattr(cfg, "tag", "") or ""
    if tag: row(_t("res_tag"), tag)

    lbl = "Configuracao Cientifica" if lang=="PT" else "Scientific Configuration"
    console.print(Panel(t, title=f"[{PA}]{lbl}[/{PA}]",
                        border_style=PA, box=rbox.ROUNDED, padding=(0, 1)))


# ---------------------------------------------------------------------------
# EXECUCAO DO PIPELINE
# ---------------------------------------------------------------------------
def _montar_painel_execucao(texto_log: str, elapsed: float,
                             objetivo_rotulo: str,
                             plano_figuras: List[str]) -> Panel:
    """Monta o painel de acompanhamento ao vivo (auditoria jul/2026, item 5):
    objetivo cientifico, barra de progresso + rotulo da analise em
    andamento (via app_logic.progresso_do_log), figuras ja concluidas
    (app_logic.figuras_concluidas) contra o plano do modo (modos_analise.
    descrever_plano), tempo decorrido/estimado restante e avisos nao-fatais
    (app_logic.avisos_do_log). Extraida de _rodar_pipeline como funcao de
    modulo para ser testavel isoladamente (ver test_guaraci_cli.py) sem
    precisar rodar o pipeline de verdade nem simular entrada interativa."""
    frac, label = _progresso_do_log(texto_log)
    figs = _figuras_concluidas(texto_log)
    avisos = _avisos_do_log(texto_log)

    bar_w = 32
    preenchido = int(bar_w * frac)
    barra = "█" * preenchido + "░" * (bar_w - preenchido)
    eta_txt = "…"
    if frac > 0.05:
        eta_txt = _fmt_tempo(max(0.0, elapsed / frac - elapsed))

    partes = [
        Text.assemble(
            (f"{_t('exec_objetivo')}: ", PM), (objetivo_rotulo, f"bold {PA}")),
        Text(f"[{barra}] {frac * 100:5.1f}%  {label}", style=PA),
        Text(f"{_t('exec_eta')}: {eta_txt}   |   "
             f"{_fmt_tempo(elapsed)}", style=PW),
        Rule(style=PD),
        Text(f"{_t('exec_figuras')} ({len(figs)}/{len(plano_figuras)} "
             "planejadas):", style=f"bold {PM}"),
        Text("  ".join(f"✓ {f}" for f in figs) if figs else "…",
             style=PW),
    ]
    if avisos:
        partes += [
            Rule(style=PD),
            Text(f"{_t('exec_avisos')} ({len(avisos)}):", style=f"bold {PR}"),
            Text("\n".join(f"⚠ {a}" for a in avisos), style=PR),
        ]
    return Panel(Group(*partes), border_style=PA, box=rbox.ROUNDED,
                 padding=(0, 1), title=f"  {_t('exec_inicio')}  ")


def _rodar_pipeline(cfg: Config) -> None:
    lang = _lang()
    cls(); _print_header()

    pode = _print_checklist(cfg)
    if not pode:
        console.print(f"  [{PR}]{_t('pip_sem_dados')}[/{PR}]")
        _pause(); return

    console.print(); _print_resumo(cfg)

    # Nome da execucao
    console.print()
    tag_atual = getattr(cfg, "tag", "") or ""
    console.print(f"  [{PM}]{_t('tag_atual')}:[/{PM}] [{PA}]{escape(tag_atual) or '(automatico)'}[/{PA}]")
    novo = _input(f"  {_t('tag_novo')}")
    if novo == "?":
        cfg.tag = ""; console.print(f"  [{PM}]{_t('tag_limpo')}[/{PM}]")
    elif novo:
        san = _re.sub(r"[^\w\-_]", "_", novo)
        cfg.tag = san; console.print(f"  [g]✓ ID: {escape(san)}[/g]")

    console.print()
    conf_str = _t("confirmar").replace("(s/n)", "(s/n)").replace("(y/n)","(y/n)")
    iniciar = _ask(f"  [{PA}]► {_t('rodar')}?[/{PA}] (s/n) ")
    if iniciar.lower() not in ("s","y","sim","yes"):
        console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return

    # Mesclar codigos usuario
    try:
        cod_u = _carregar_codigos_usuario()
        if cod_u: pq.CODIGO_ESPECIE.update(cod_u)
    except (OSError, json.JSONDecodeError) as _e_cod:
        logging.getLogger(__name__).debug(
            "codigos de usuario nao mesclados: %s", _e_cod)

    # Aplicar configuracoes visuais
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vcfg = _carregar_visual_cfg()
        paleta = PALETAS_COR.get(vcfg.get("paleta", "qualitativo"), {})
        try: plt.style.use(paleta.get("style", "default"))
        except OSError:
            pass   # nome de estilo matplotlib desconhecido -- mantem o default
        cores = paleta.get("cores")
        if cores: plt.rcParams["axes.prop_cycle"] = plt.cycler(color=cores)
        cmap = paleta.get("cmap")
        if cmap: plt.rcParams["image.cmap"] = cmap
        fp = FONT_PRESETS.get(vcfg.get("tamanho_fonte","m"), {})
        for k, v in fp.items(): plt.rcParams[k] = v
        if vcfg.get("grid_major", True):
            plt.rcParams["axes.grid"] = True
            plt.rcParams["grid.linestyle"] = vcfg.get("grid_style","dotted")
            plt.rcParams["grid.alpha"] = float(vcfg.get("grid_alpha", 0.4))
        else:
            plt.rcParams["axes.grid"] = False
        alpha_map = {"baixo":0.9,"medio":0.65,"alto":0.35}
        plt.rcParams["lines.alpha"] = alpha_map.get(vcfg.get("alpha_pontos","medio"), 0.65)
    except Exception as _e_vis:  # noqa: BLE001 -- configuracao visual
        # cosmetica (paleta/fonte/grid); um erro aqui nunca deve impedir a
        # corrida de acontecer, so' os defaults do matplotlib ficam em uso.
        logging.getLogger(__name__).debug(
            "configuracao visual nao aplicada: %s", _e_vis)

    # Sincronizar DPI do visual_config antes de salvar
    _sincronizar_dpi(cfg)
    salvar_config(cfg, str(_CFG_PATH))

    # Sugestao de cafe em execucoes longas
    if (_cfgv(cfg, "monte_carlo", False)
            or _cfgv(cfg, "shap_benchmark", False)
            or _cfgv(cfg, "benchmark", False)
            or _cfgv(cfg, "comparar_pre_processamentos", False)):
        _sugerir_cafe()

    # Painel de acompanhamento (auditoria jul/2026, item 5): o terminal deve
    # mostrar quais figuras serao calculadas, quais ja foram concluidas,
    # analise em andamento, % de progresso, tempo estimado e avisos — em vez
    # da simulacao por tempo decorrido usada antes (etapas fixas avancando a
    # cada 15s, sem relacao real com o que o pipeline estava fazendo).
    objetivo_rotulo = pq.OBJETIVO_ROTULO.get(
        pq.resolver_objetivo(cfg), cfg.nivel)
    plano_figuras = pq.descrever_plano(cfg)

    _done: Dict[str, object] = {"ok": False, "error": None}
    _logger = _LogThreadSafe()

    def _run():
        try:
            with contextlib.redirect_stdout(_logger), \
                 contextlib.redirect_stderr(_logger):
                executar(cfg)
        except KeyboardInterrupt:
            _done["error"] = _t("exec_interrompido")
        except Exception as e:  # noqa: BLE001 -- boundary de topo da thread de
            # execucao: QUALQUER excecao do pipeline (1269 linhas de
            # executar()) precisa ser capturada aqui para nao matar a
            # thread silenciosamente -- e' reportada em _done["error"] e
            # exibida ao usuario, nunca perdida.
            _done["error"] = str(e)
        finally:
            _done["ok"] = True

    def _render_painel(elapsed: float) -> Panel:
        return _montar_painel_execucao(
            texto_log=_logger.text(), elapsed=elapsed,
            objetivo_rotulo=objetivo_rotulo, plano_figuras=plano_figuras)

    console.print()
    t_ini = time.time()
    thr = threading.Thread(target=_run, daemon=True)
    thr.start()

    with Live(console=console, refresh_per_second=3) as live:
        while not _done["ok"]:
            live.update(_render_painel(time.time() - t_ini))
            time.sleep(0.3)
        live.update(_render_painel(time.time() - t_ini))

    thr.join()
    console.print()

    if _done.get("error"):
        console.print(Panel(
            Text(f"✖ {_t('exec_erro')}\n{escape(str(_done['error']))}", style=PR),
            border_style=PR, box=rbox.ROUNDED, padding=(0, 1)
        ))
    else:
        pasta_s = _cfgv(cfg, "pasta_saida", "resultados")
        tag     = getattr(cfg, "tag", "") or ""
        destino = f"{pasta_s}/{tag}" if tag else pasta_s
        lbl_concluido = _t("exec_concluido").upper()
        lbl_saida     = _t("exec_saida")
        console.print(Panel(
            Align.center(Group(
                Text(f"\n  ✓ {lbl_concluido}\n", style=f"bold {PG}"),
                Text(f"  {lbl_saida}:\n  {destino}/\n", style=PW),
            )),
            border_style=PG, box=rbox.ROUNDED, padding=(0, 2)
        ))

    _pause()


# ---------------------------------------------------------------------------
# SALVAR / CARREGAR PERFIL
# ---------------------------------------------------------------------------
def _salvar_yaml(cfg: Config) -> None:
    lang = _lang()
    console.print()
    lbl = "Nome do perfil: " if lang=="PT" else "Profile name: "
    nome = _input(f"  {lbl}").strip()
    if not nome:
        console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return
    san = _re.sub(r"[^\w\-_]", "_", nome)
    _PERFIS_DIR.mkdir(parents=True, exist_ok=True)
    path = _PERFIS_DIR / f"{san}.yaml"
    try:
        salvar_config(cfg, str(path))
        _lbl = "Salvo" if _lang() == "PT" else "Saved"
        console.print(f"  [g]✓ {_lbl}: {escape(str(path))}[/g]")
    except OSError as e:
        console.print(f"  [err]{escape(str(e))}[/err]")
    _pause()


def _carregar_yaml(cfg: Config) -> None:
    _PERFIS_DIR.mkdir(parents=True, exist_ok=True)
    arquivos = sorted(_PERFIS_DIR.glob("*.yaml"))
    if not arquivos:
        lbl = "Nenhum perfil salvo." if _lang()=="PT" else "No saved profiles."
        console.print(f"  [{PM}]{lbl}[/{PM}]"); _pause(); return

    t = Table(show_header=False, box=rbox.SIMPLE, padding=(0, 1))
    t.add_column("N", style=PA, width=4)
    t.add_column("Nome", style=PW)
    for i, f in enumerate(arquivos, 1):
        t.add_row(str(i), escape(f.stem))
    titulo = "Perfis Salvos" if _lang()=="PT" else "Saved Profiles"
    console.print(Panel(t, title=f"[bold {PA}]{titulo}[/bold {PA}]",
                        border_style=PA, box=rbox.ROUNDED, padding=(0,1)))
    raw = _input("  N: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(arquivos):
        path = arquivos[int(raw) - 1]
        try:
            cfg2 = carregar_config(str(path))
            for k, v in vars(cfg2).items():
                try: setattr(cfg, k, v)
                except (AttributeError, TypeError) as _e_attr:
                    # Campo do perfil salvo desalinhado com o Config atual
                    # (ex.: perfil antigo de uma versao com schema diferente).
                    logging.getLogger(__name__).debug(
                        "perfil '%s': campo '%s' nao aplicado: %s",
                        path.stem, k, _e_attr)
            _lbl = "Carregado" if _lang() == "PT" else "Loaded"
            console.print(f"  [g]✓ {_lbl}: {escape(path.stem)}[/g]")
        except (RuntimeError, FileNotFoundError, ValueError) as e:
            console.print(f"  [err]{escape(str(e))}[/err]")
    else:
        console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]")
    _pause()


# ===========================================================================
# COMANDOS DE LINHA DE COMANDO (--version, demo, doctor) — P7 (CLAUDE.md)
# ===========================================================================
def _comando_versao() -> None:
    print(f"GUARACI v{pq.__version__}")


def _comando_doctor() -> None:
    """Diagnostico de ambiente: Python, dependencias, RAM/CPU/disco.

    Nao lanca excecao — cada checagem e best-effort, pensada para rodar
    em qualquer maquina antes do usuario abrir um issue de instalacao.
    """
    import importlib.util
    import platform

    linhas: List[str] = []
    linhas.append(f"GUARACI v{pq.__version__} — diagnostico de ambiente")
    linhas.append(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    linhas.append("")
    linhas.append(f"Python:   {platform.python_version()} ({platform.python_implementation()})")
    linhas.append(f"SO:       {platform.system()} {platform.release()} ({platform.machine()})")

    hw = pq.hardware_probe()
    linhas.append(f"RAM:      {hw['ram_livre_gb']:.1f} GB livre / {hw['ram_total_gb']:.1f} GB total")
    linhas.append(f"CPU:      {hw['cpu_fisicos']} fisicos / {hw['cpu_logicos']} logicos")
    linhas.append(f"Disco:    {hw['disco_livre_gb']:.1f} GB livre (pasta atual)")
    if not hw["psutil_ok"]:
        linhas.append("          (psutil ausente — valores conservadores/estimados)")

    linhas.append("")
    linhas.append("Dependencias obrigatorias:")
    obrigatorias = ["numpy", "pandas", "scipy", "sklearn", "matplotlib",
                     "joblib", "threadpoolctl", "yaml", "rich", "PIL"]
    faltando_obrig = []
    for mod in obrigatorias:
        spec = importlib.util.find_spec(mod)
        if spec is None:
            linhas.append(f"  [FALTA] {mod}")
            faltando_obrig.append(mod)
        else:
            try:
                m = importlib.import_module(mod)
                ver = getattr(m, "__version__", "?")
            except ImportError as _e_imp:
                ver = f"erro ao importar: {_e_imp}"
            linhas.append(f"  [ok]    {mod} {ver}")

    linhas.append("")
    linhas.append("Extras opcionais ([web], [reports], [benchmark], [imagem]):")
    opcionais = {
        "streamlit": "web", "psutil": "web",
        "fpdf2": "reports", "docx": "reports (python-docx)",
        "openpyxl": "reports", "pptx": "reports (python-pptx)",
        "xgboost": "benchmark", "shap": "benchmark",
        "skimage": "imagem (scikit-image)",
    }
    for mod, extra in opcionais.items():
        spec = importlib.util.find_spec(mod)
        status = "ok" if spec is not None else "ausente"
        linhas.append(f"  [{status:7s}] {mod:12s} ({extra})")

    console.print()
    for l in linhas:
        console.print(f"  {escape(l)}")
    console.print()

    if faltando_obrig:
        console.print(f"  [err]✗ Dependencias obrigatorias faltando: "
                       f"{', '.join(faltando_obrig)}. Rode: pip install guaraci-chemometrics[/err]")
    else:
        console.print("  [g]✓ Todas as dependencias obrigatorias estao instaladas.[/g]")

    destino = Path.cwd() / "guaraci_doctor.txt"
    try:
        destino.write_text("\n".join(linhas), encoding="utf-8")
        console.print(f"  [{PM}]Relatorio salvo em: {escape(str(destino))}[/{PM}]")
    except OSError as e:
        console.print(f"  [err]Nao foi possivel salvar o relatorio: {escape(str(e))}[/err]")


def _comando_demo() -> None:
    """Roda o pipeline completo com dados sinteticos, sem exigir dado do
    usuario. Fluxo dos 5 minutos: pip install -> guaraci demo -> figuras."""
    console.print()
    console.print(f"  [{PA}]GUARACI demo — gerando espectros sinteticos e rodando o pipeline...[/{PA}]")
    console.print(f"  [{PM}](nenhum dado seu e usado; tudo aqui e gerado artificialmente)[/{PM}]")
    console.print()

    saida = Path.cwd() / "GUARACI_Demo"
    cfg = Config(
        pasta_entrada=str(saida / "dados_dummy"),  # modo sintetico ignora isto
        pasta_saida_raiz=str(saida),
        modo="sintetico",
        tag="demo",
        nivel="N2",       # DD-SIMCA (sensibilidade LOGO) — diferencial central do projeto
        objetivo="auto",
        n_por_classe=15,
        n_pontos_sint=300,
        wn_min=400.0,
        wn_max=4001.0,
        sint_adulterantes=("S", "M", "A"),
        max_lvs=15,
        n_splits_cv=3,
        n_repeats_cv=1,
        n_permutacoes=50,
        n_permutacoes_wold=50,
        n_bootstrap_vip=10,
        n_bootstrap_bca=100,
        n_monte_carlo=20,
        executar_benchmark=False,
        executar_monte_carlo=False,
        executar_shap=False,
        executar_wold=False,
        executar_cv_anova=False,
        executar_opls=False,
        executar_etapa4=False,
        comparar_pipelines=False,
        comparar_hca_pipelines=False,
    )
    os.makedirs(cfg.pasta_entrada, exist_ok=True)

    try:
        pq.executar(cfg)
    except Exception as e:  # noqa: BLE001 -- demo deve reportar erro legivel, nao stack trace cru
        console.print(f"  [err]✗ A demo falhou: {escape(str(e))}[/err]")
        console.print(f"  [{PM}]Rode 'guaraci doctor' para checar o ambiente.[/{PM}]")
        raise SystemExit(1) from e

    runs: List[str] = []
    if saida.exists():
        for raiz, dirs, _arqs in os.walk(str(saida)):
            for d in dirs:
                if d.startswith("PLSDA_OE_"):
                    runs.append(os.path.join(raiz, d))
    if not runs:
        console.print(f"  [err]✗ Pipeline rodou mas nenhuma pasta de saida foi encontrada em {escape(str(saida))}[/err]")
        raise SystemExit(1)

    pasta_run = Path(sorted(runs)[-1])
    console.print()
    console.print(f"  [g]✓ Demo concluida.[/g] Resultados em: [{PA}]{escape(str(pasta_run))}[/{PA}]")
    console.print(f"  [{PM}]Veja {pq.NOME_GRAFICOS}/ para as figuras e {pq.NOME_RELATORIOS}/resumo_modelo.txt "
                  f"para o resumo numerico.[/{PM}]")

    try:
        if sys.platform == "win32":
            os.startfile(str(pasta_run))  # noqa: S606 -- abre o explorador, caminho e nosso proprio output
        elif sys.platform == "darwin":
            os.system(f'open "{pasta_run}"')  # noqa: S605
        else:
            os.system(f'xdg-open "{pasta_run}"')  # noqa: S605
    except OSError as _e_open:
        logging.getLogger(__name__).debug("nao foi possivel abrir a pasta de saida: %s", _e_open)


# ===========================================================================
# MAIN LOOP
# ===========================================================================
def main() -> None:
    """Ponto de entrada GUARACI (versao unica em _VERSAO)."""
    if len(sys.argv) > 1:
        comando = sys.argv[1].strip().lower().lstrip("-")
        if comando in ("version", "v"):
            _comando_versao(); return
        if comando == "demo":
            _comando_demo(); return
        if comando == "doctor":
            _comando_doctor(); return
        if comando in ("help", "h"):
            print("Uso: guaraci [demo|doctor|--version]\n"
                  "  (sem argumentos)  abre o assistente interativo\n"
                  "  demo              roda o pipeline com dados sinteticos\n"
                  "  doctor            diagnostica o ambiente (deps, RAM, CPU)\n"
                  "  --version         mostra a versao instalada")
            return

    # Carregar config
    cfg = Config()
    if _CFG_PATH.exists():
        try:
            cfg = carregar_config(str(_CFG_PATH))
        except (RuntimeError, FileNotFoundError, ValueError) as _e_cfg:
            logging.getLogger(__name__).debug(
                "config.yaml nao carregado no boot, usando defaults: %s", _e_cfg)

    # Recuperar idioma salvo
    try:
        saved_lang = _LANG_FLAG.read_text(encoding="utf-8").strip()
        if saved_lang in ("EN", "PT"):
            _set_lang(saved_lang)
    except OSError:
        pass

    # Boas-vindas uma vez por sessao
    _exibir_boas_vindas()

    while True:
        cls()
        _print_header()
        _print_status(cfg)
        console.print()
        _print_main_menu()
        console.print()
        _print_run_box(cfg)
        console.print()

        try:
            raw = _input(f"  {_t('opcao')}: ")
            escolha = "?" if raw == "?" else raw.upper().strip()
        except (EOFError, KeyboardInterrupt):
            _exibir_despedida()
            break

        if escolha == "1": menu_projeto(cfg)
        elif escolha == "2": menu_dados(cfg)
        elif escolha == "3": menu_preproc(cfg)
        elif escolha == "4": menu_modelagem(cfg)
        elif escolha == "5": menu_validacao(cfg)
        elif escolha == "6": menu_avancado(cfg)
        elif escolha == "7": menu_visualizacao(cfg)
        elif escolha == "8": menu_tecnica(cfg)
        elif escolha == "9": menu_codificacao(cfg)
        elif escolha == "H":
            cls(); _print_header(); menu_hardware(cfg)
        elif escolha == "B":
            menu_predicao(cfg)
        elif escolha == "P":
            menu_perfis(cfg)
        elif escolha == "G":
            _abrir_assistente("menu principal", cfg)
        elif escolha == "I":
            _toggle_idioma()
        elif escolha == "S":
            _salvar_yaml(cfg)
        elif escolha == "L":
            _carregar_yaml(cfg)
        elif escolha == "R":
            _rodar_pipeline(cfg)
        elif escolha == "N":
            console.print()
            tag_atual = getattr(cfg, "tag", "") or ""
            console.print(f"  [{PM}]{_t('tag_atual')}:[/{PM}] [{PA}]{escape(tag_atual) or '(automatico)'}[/{PA}]")
            novo = _input(f"  {_t('tag_novo')}")
            if novo == "?":
                cfg.tag = ""; console.print(f"  [{PM}]{_t('tag_limpo')}[/{PM}]")
            elif novo:
                san = _re.sub(r"[^\w\-_]", "_", novo)
                cfg.tag = san; console.print(f"  [g]✓ ID: {escape(san)}[/g]")
            _pause()
        elif escolha == "A":
            menu_sobre(cfg)
        elif escolha == "?":
            menu_ajuda(cfg)
        elif escolha == "Q":
            _exibir_despedida()
            break
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()


if __name__ == "__main__":
    main()
