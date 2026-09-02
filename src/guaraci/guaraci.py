"""
guaraci.py v31.9.0 — Interface profissional GUARACI para o pipeline quimiometrico
☀  GUARACI — Plataforma quimiometrica com validacao anti-vazamento por padrao
Quimiometria • Machine Learning • Espectroscopia multitecnica

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
import shutil
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
        # shell=True e' necessario aqui: `chcp` e' builtin do cmd.exe, nao um
        # executavel. Os argumentos sao constantes -- nada vindo do usuario
        # chega nesta linha, entao nao ha superficie de injecao.
        subprocess.run("chcp 65001", capture_output=True, shell=True)  # noqa: S602
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
    figures_completed as _figures_completed,
    log_warnings as _avisos_do_log,
    log_progress as _progresso_do_log,
    fmt_time as _fmt_time,
)

Config        = pq.Config
_executar      = pq.executar
_save_config = pq.save_config
_load_config = pq.load_config

# Dicionarios de i18n/perfis do cli_assistente. Import de pacote normal (mesmo
# pacote): importar o modulo NAO dispara main() (guardado por __main__), entao
# nao ha efeito colateral — o antigo carregamento por caminho (spec_from_file)
# so existia porque os modulos eram scripts soltos na raiz.
import guaraci.cli_assistente as _cli

def _try(name, fallback=None):
    return getattr(_cli, name, fallback if fallback is not None else {})

_FIELD_NAMES          = _try("FIELD_NAMES")
_HELP_DB              = _try("HELP_DB")
_RISK_CLASS           = _try("RISK_CLASS")
PROFILES             = _try("PROFILES")
PROFILE_DESC         = _try("PROFILE_DESC")
PROFILE_KEY_SUMMARY  = _try("PROFILE_KEY_SUMMARY")
_PALETAS_COR          = _try("PALETAS_COR")
_FONT_PRESETS         = _try("FONT_PRESETS")
_TECNICAS             = _try("TECNICAS")
_REFERENCIAS_GUARACI  = _try("REFERENCIAS_GUARACI")
_CONFIG_SPEC         = _try("_CONFIG_SPEC", [])
_SPEC_BY_KEY         = _try("_SPEC_BY_KEY")
_DDSIMCA_DISPLAY     = _try("_DDSIMCA_DISPLAY")
_DDSIMCA_INPUT       = _try("_DDSIMCA_INPUT")
_coagir_valor        = _try("_coagir_valor", lambda s, r: r)
_attr_para_yaml      = _try("_attr_para_yaml", lambda s, c: "")
_fmt_yaml            = _try("_fmt_yaml", str)
_save_config        = _try("save_config", pq.save_config)
_load_config      = _try("load_config", pq.load_config)

# __all__ so' cobre a superficie realmente consumida de fora deste arquivo
# (confirmado por grep + suite de testes): `main` e' o ponto de entrada da
# CLI; `Config` e' usado extensivamente pela suite via `guaraci_mod.Config`;
# PROFILES/PROFILE_DESC/PROFILE_KEY_SUMMARY sao testados diretamente por
# `test_guaraci_cli.py` via `guaraci_mod.X` (achado ao aplicar este __all__:
# pareciam mirrors so'-internos de cli_assistente.py, mas ha um teste que os
# consome por este caminho especifico -- mantidos publicos aqui por isso).
# O resto (menu_*, cls, I18N, GUARACI_TIPS, FIELD_NAMES/HELP_DB/RISK_CLASS/
# PALETAS_COR/FONT_PRESETS/TECNICAS/REFERENCIAS_GUARACI, os aliases locais
# _executar/_save_config/_load_config) e' wiring interno da CLI -- nunca
# consumido fora deste arquivo, renomeado para _privado nesta auditoria.
__all__ = [
    "main",
    "Config",
    "PROFILES",
    "PROFILE_DESC",
    "PROFILE_KEY_SUMMARY",
]

# Persistencia de estado do CLI.
#
# CORRIGIDO em 2026-08-16 (varredura de bugs): estas tres funcoes eram
# wrappers que procuravam implementacoes em `cli_assistente` -- que NUNCA
# existiram la'. `getattr(..., None)` devolvia None, entao `_carregar_*`
# retornava sempre {} e `_salvar_visual_cfg` era um no-op SILENCIOSO. O
# comentario do proprio codigo (secao do Modo Iniciante/Avancado) ja
# registrava "esse esta quebrado ... fora do escopo desta feature
# consertar isso" desde 2026-07-13.
#
# Consequencias reais, ambas do tipo que o projeto mais combate (o software
# mente sem travar):
#   1. As 4 opcoes do menu Visualizacao (Paleta/Fonte/Grid/Alpha) gravavam
#      no dicionario, chamavam _salvar_visual_cfg(), imprimiam "OK Paleta:
#      X" e NAO persistiam nada -- a confirmacao era falsa. Na proxima
#      abertura o valor voltava ao default. O mesmo valia para o DPI
#      (`_sincronizar_dpi`) e para toda a aplicacao de estilo em
#      `_rodar_pipeline` (paleta/fonte/grid/alpha nos rcParams do
#      matplotlib), que le de `_carregar_visual_cfg()`.
#   2. Codigos de especie cadastrados pelo usuario eram gravados
#      corretamente pelo menu (`_salvar_cod`, que tem implementacao propria
#      e funciona), apareciam listados no proprio menu (`_cod_usr`, idem),
#      mas NAO eram aplicados a analise: a unica linha que injeta os
#      codigos no pipeline (`pq.CODIGO_ESPECIE.update(cod_u)`) usava o
#      wrapper quebrado e recebia {} sempre.
#
# Agora as tres leem/gravam direto em _USER_DIR, no mesmo padrao ja usado
# por `_cod_usr`/`_salvar_cod` (que sempre funcionaram) e pelos demais
# arquivos de estado (_CFG_PATH, _LANG_FLAG, _MODO_FLAG).
def _carregar_visual_cfg() -> dict:
    """Config visual (paleta/fonte/grid/alpha/dpi) de _VISUAL_PATH.

    Arquivo ausente ou corrompido -> {} (defaults do matplotlib), nunca
    excecao: configuracao cosmetica nao pode impedir uma analise de rodar.
    """
    try:
        p = _VISUAL_PATH
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}

def _salvar_visual_cfg(d: dict) -> None:
    """Grava a config visual. Falha de escrita AVISA em vez de sumir em
    silencio -- mesma licao de `_salvar_cod`: o usuario nao pode ver um
    'salvo' que nao aconteceu."""
    try:
        _USER_DIR.mkdir(parents=True, exist_ok=True)
        _VISUAL_PATH.write_text(
            json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        console.print(f"[err]✗ Falha ao salvar config visual: {e}[/err]")

def _carregar_codigos_usuario() -> dict:
    """Codigos de especie cadastrados pelo usuario, de _CODIGOS_PATH.

    Mesma leitura de `_cod_usr()` (menu de codificacao) -- e' de proposito
    que as duas leiam o MESMO arquivo: o que o menu lista tem de ser o que
    a analise aplica.
    """
    try:
        p = _CODIGOS_PATH
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
# _BASE_DIR: diretorio de INSTALACAO do pacote -- so' para recursos
# somente-leitura que o pacote ja traz consigo (ex.: CITATION.cff).
#
# _USER_DIR: onde o CLI grava ESTADO do usuario (config.yaml, perfis
# salvos, flags de idioma/mode, codigos customizados). CORRIGIDO em
# 2026-08-07 (achado do "checkup geral" de interface -- ver
# ): ate' entao esses arquivos eram gravados dentro de
# _BASE_DIR, ou seja, DENTRO do diretorio de instalacao do pacote. Isso
# quebra em qualquer instalacao read-only (pip de sistema, imagem Docker,
# alguns `pip install --user`) -- `save_config()` logo antes de rodar o
# pipeline (ver `_rodar_pipeline`) nao tinha nenhuma guarda contra isso e
# derrubava o CLI com um PermissionError bem na hora de rodar a analise.
# Home do usuario e' gravavel em praticamente qualquer instalacao.
_BASE_DIR    = Path(os.path.dirname(os.path.abspath(__file__)))
_USER_DIR    = Path.home() / ".guaraci"
_CFG_PATH    = _USER_DIR / "config.yaml"
_PERFIS_DIR  = _USER_DIR / "perfis"
_LANG_FLAG   = _USER_DIR / ".cli_wizard_done"
_CODIGOS_PATH= _USER_DIR / "codigos_usuario.json"
_VISUAL_PATH = _USER_DIR / "visual_config.json"


def _migrar_estado_legado() -> None:
    """Copia, uma vez, o estado gravado pela versao anterior (dentro de
    _BASE_DIR) para o novo local (_USER_DIR), se o novo local ainda nao
    tiver esse arquivo. NUNCA sobrescreve nem apaga o arquivo antigo --
    so' copia o que falta, best-effort (falha de permissao aqui nao pode
    impedir o CLI de abrir). Chamada uma vez no inicio de `main()`, nao na
    importacao do modulo (importar `guaraci.guaraci` -- em testes, por
    exemplo -- nao deve escrever no HOME de quem esta rodando os testes).
    """
    try:
        _USER_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    for nome, alvo in (
        ("config.yaml", _CFG_PATH),
        (".cli_wizard_done", _LANG_FLAG),
        ("codigos_usuario.json", _CODIGOS_PATH),
        (".cli_modo_usuario", _MODO_FLAG),
        ("visual_config.json", _VISUAL_PATH),
    ):
        origem = _BASE_DIR / nome
        if origem.exists() and not alvo.exists():
            try:
                shutil.copy2(origem, alvo)
            except OSError:
                pass
    # `visual_config.json` tinha uma origem legada A MAIS: a versao antiga
    # gravava por caminho RELATIVO, entao o arquivo ficava no diretorio de
    # onde o CLI foi chamado (tipicamente a raiz do repositorio), nao em
    # _BASE_DIR. Recuperado aqui para nao perder a paleta que o usuario ja
    # tinha escolhido -- so' se _USER_DIR ainda nao tiver a sua.
    if not _VISUAL_PATH.exists():
        try:
            origem_cwd = Path.cwd() / "visual_config.json"
            if origem_cwd.is_file():
                shutil.copy2(origem_cwd, _VISUAL_PATH)
        except OSError:
            pass
    origem_perfis = _BASE_DIR / "perfis"
    if origem_perfis.is_dir() and not _PERFIS_DIR.exists():
        try:
            shutil.copytree(origem_perfis, _PERFIS_DIR)
        except OSError:
            pass

# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
_STATE: Dict[str, Any] = {"lang": "PT", "modo_usuario": "iniciante"}

def _lang() -> str:
    return _STATE["lang"]

def _set_lang(l: str) -> None:
    _STATE["lang"] = l
    try:
        _USER_DIR.mkdir(parents=True, exist_ok=True)
        _LANG_FLAG.write_text(l, encoding="utf-8")
    except OSError:
        pass

def _toggle_idioma() -> str:
    novo = "EN" if _lang() == "PT" else "PT"
    _set_lang(novo)
    return novo

# Modo Iniciante/Avancado (CLAUDE.md secao 6 / auditoria 2026-07-12): alterna
# GLOBALMENTE se os submenus escondem campos avancados por padrao. Usa
# arquivo-flag proprio (mesmo padrao de persistencia do idioma), nao o
# visual_config.json -- que guarda aparencia de FIGURA (paleta/fonte/grid),
# nao estado de navegacao do menu; sao coisas diferentes.
# (Este comentario dizia, ate 2026-08-16, que o mecanismo de
# visual_config.json estava quebrado e que consertar estava "fora do
# escopo". Estava mesmo quebrado desde 2026-07-13 e foi CORRIGIDO na
# varredura de bugs de 2026-08-16 -- ver as funcoes de persistencia no topo
# do modulo.)
_MODO_FLAG = _USER_DIR / ".cli_modo_usuario"

def _modo_usuario() -> str:
    return _STATE["modo_usuario"]

def _set_modo_usuario(m: str) -> None:
    _STATE["modo_usuario"] = m
    try:
        _USER_DIR.mkdir(parents=True, exist_ok=True)
        _MODO_FLAG.write_text(m, encoding="utf-8")
    except OSError:
        pass

def _toggle_modo_usuario() -> str:
    novo = "avancado" if _modo_usuario() == "iniciante" else "iniciante"
    _set_modo_usuario(novo)
    return novo

if _MODO_FLAG.exists():
    try:
        _v = _MODO_FLAG.read_text(encoding="utf-8").strip()
        if _v in ("iniciante", "avancado"):
            _STATE["modo_usuario"] = _v
    except OSError:
        pass

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
    truncate_desc_by_sentence as _truncar_desc_por_frase,
    fmt_bool as _fmt_bool_puro,
    validate_ranges as _validar_faixas_puro,
    count_dx as _count_dx,
)

# Estado global da tecnica selecionada (persiste entre menus)
_TECNICA_SELECIONADA: Dict[str, str] = {"key": "ft-nir", "nome": "FT-NIR"}

# ---------------------------------------------------------------------------
# Internacionalizacao — sem repeticao entre idiomas
# ---------------------------------------------------------------------------
_I18N: Dict[str, Dict[str, str]] = {
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
        "t_hsi":        "Imageamento Hiperespectral",
        "t_planejamento": "Planejamento de Coleta",
        "t_selecao_amostras": "Selecao de Amostras",
        "t_auditoria":  "Auditoria de Delineamento",
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
        # Faltavam pra' _guaraci_navegar_secoes cobrir as 18 abas reais
        # (achado do Agente 6, docs/DESIGN.md): so' existiam d_ das 12
        # abas com campo de _CONFIG_SPEC associado.
        "d_predicao":       "Aplica modelo .joblib a espectros novos.",
        "d_hsi":            "Cubo hiperespectral: quality gate, segmentacao, classificacao por pixel.",
        "d_planejamento":   "Tamanho amostral e plano de coleta.",
        "d_auditoria":      "Checagens de delineamento anti-vazamento.",
        "d_selecao_amostras": "Divide CSV em calibracao/validacao.",
        "d_sobre":          "Versao, licenca e creditos.",
        # Grupos do menu principal (docs/DESIGN.md secao 4 -- Agente 5.2,
        # aprovado 2026-09-01: substitui os 3 grupos antigos config/analise/
        # sistema por 6 grupos alinhados ao fluxo real de trabalho)
        "grp_preparar": "Preparar",
        "grp_planejar": "Planejar",
        "grp_modelar":  "Modelar",
        "grp_validar":  "Validar",
        "grp_prever":   "Prever",
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
        "opt_sim":      "Sim",
        "opt_nao":      "Nao",
        "escolha_num":  "Numero ou Enter=manter: ",
        "ajuste_nivel": "Ajustado automaticamente (nao pertinente a este nivel): {campos}",
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
        "chk_tempo":    "Tempo estimado",
        "chk_descarte": "{n} espectros serao DESCARTADOS (faixa espectral incompativel)",
        "chk_orfaos":   "{n} amostras sem mae_id — entram SEM protecao anti-leakage",
        "chk_grupos":   "{n} grupos de replica (mae_id)",
        "chk_dica_jobs": "(n_jobs_permutation=1 — subir p/ 4 reduz o tempo sem mudar o resultado)",
        "chk_modo":     "Modo",
        "chk_prescan_erro": "pre-varredura indisponivel ({erro})",
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
        "tag_auto":     "(automatico)",
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
        "t_hsi":        "Hyperspectral Imaging",
        "t_planejamento": "Collection Planning",
        "t_selecao_amostras": "Sample Selection",
        "t_auditoria":  "Design Audit",
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
        "d_predicao":       "Applies a .joblib model to new spectra.",
        "d_hsi":            "Hyperspectral cube: quality gate, segmentation, per-pixel classification.",
        "d_planejamento":   "Sample size and collection plan.",
        "d_auditoria":      "Anti-leakage design checks.",
        "d_selecao_amostras": "Splits a CSV into calibration/validation.",
        "d_sobre":          "Version, license and credits.",
        "grp_preparar": "Prepare",
        "grp_planejar": "Plan",
        "grp_modelar":  "Model",
        "grp_validar":  "Validate",
        "grp_prever":   "Predict",
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
        "opt_sim":      "Yes",
        "opt_nao":      "No",
        "escolha_num":  "Number or Enter=keep: ",
        "ajuste_nivel": "Automatically adjusted (not relevant to this level): {campos}",
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
        "chk_tempo":    "Estimated time",
        "chk_descarte": "{n} spectra will be DISCARDED (incompatible spectral range)",
        "chk_orfaos":   "{n} samples without mae_id — enter WITHOUT anti-leakage protection",
        "chk_grupos":   "{n} replicate groups (mae_id)",
        "chk_dica_jobs": "(n_jobs_permutation=1 — raising it to 4 cuts the time without changing results)",
        "chk_modo":     "Mode",
        "chk_prescan_erro": "pre-scan unavailable ({erro})",
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
        "tag_auto":     "(automatic)",
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
    s = _I18N[_lang()].get(key, key)
    if kw:
        try:
            s = s.format(**kw)
        except (KeyError, IndexError):
            pass
    return s

# ---------------------------------------------------------------------------
# GUARACI TIPS — dicas unicas, diferentes das descricoes do _HELP_DB
# ---------------------------------------------------------------------------
_GUARACI_TIPS: Dict[str, Dict[str, str]] = {
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
        "PT": "Arquivos do espectrômetro: use 'dx'. CSV é para dados tabelados de outras fontes.",
        "EN": "Spectrometer files: use 'dx'. CSV is for tabular data from other sources.",
    },
    "pre_processamento": {
        "PT": "MSC+SG+MC costuma ser um bom padrão para FT-NIR/NIR. Autoscaling isolado tende a perder desempenho em espectros com espalhamento — compare os dois no Auto-Benchmark.",
        "EN": "MSC+SG+MC tends to be a strong default for FT-NIR/NIR. Autoscaling alone tends to lose performance on spectra with scattering — compare both in the Auto-Benchmark.",
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
    return _RISK_HEX.get(_RISK_CLASS.get(key, "ANALITICO"), PA)

def _risco_icon(key: str) -> str:
    return _RISK_ICON.get(_RISK_CLASS.get(key, "ANALITICO"), "◆")

# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------
def _cls() -> None:
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
    return _FIELD_NAMES.get(key, {}).get(_lang(), key)

# Traducao dos nomes de nivel APENAS para exibicao. `pq._NIVEL_NOME` (em
# config.py) segue sendo a fonte unica em portugues, porque tambem alimenta o
# `resumo_modelo.txt` — traduzir la' mudaria um ARQUIVO DE SAIDA conforme o
# idioma da interface, o que quebraria a comparabilidade entre execucoes.
# Mesmo padrao ja' usado por _DDSIMCA_DISPLAY.
_NIVEL_NOME_EN = {
    "N1": "Species classification",
    "N2": "Pure vs. adulterated discrimination",
    "N3": "Adulterant content quantification",
}


def _rotulo_opcao(key: str, op: Any) -> str:
    """Nome amigavel de um valor de opcao (so exibicao; o valor gravado no
    config continua o codigo interno, ex.: N1/N2/N3, puros/todos)."""
    if key == "nivel":
        # Lidera com o nome amigavel (P8: aposentar N1/N2/N3 como termo
        # PRIMARIO na UI); o codigo interno fica entre parenteses so' como
        # referencia tecnica, nunca como o rotulo principal exibido ao usuario.
        nome = (_NIVEL_NOME_EN.get(str(op), "") if _lang() == "EN"
                else pq._NIVEL_NOME.get(str(op), ""))
        return f"{nome} ({op})" if nome else str(op)
    if key == "modo_ddsimca":
        return _DDSIMCA_DISPLAY.get(_lang(), {}).get(str(op), str(op))
    if key in ("perfil_matriz", "perfil_tecnica"):
        # Agente 5B: expoe o que hoje e' dado morto (descricao do perfil, e
        # para perfil de tecnica tambem a garantia tipica de agrupamento) --
        # sem isso o usuario via' so' o nome de arquivo (ex. "bancada"), sem
        # saber o que ele significa, tendo que sair da sessao e rodar
        # `guaraci perfis` num terminal separado so' pra ler a descricao.
        if not str(op):
            return "(nao declarado)" if _lang() == "PT" else "(not declared)"
        try:
            from guaraci.perfil_matriz import load_profile
            p = load_profile(str(op))
        except Exception:  # noqa: BLE001 -- rotulo e' so' exibicao; um
            # perfil quebrado/removido nao pode travar a tela de edicao,
            # so' cai pro nome cru (a validacao real acontece ao aplicar).
            return str(op)
        desc = (p.descricao or "").split(" (")[0].strip()
        # Indicador de cobertura validada (Agente 5B, item pendente):
        # `referencia` ja' e' o dado que distingue perfil validado com dado
        # PUBLICO real (milho_nir/oleos_comestiveis_nir tem paper/dataset
        # citado) de perfil so' declarado (mel_vis_nir/oleo_nir/bancada/
        # celular/scanner -- referencia vazia, nenhum tem validacao
        # publicada ainda). Nao inventa um campo novo -- so' expoe o que
        # ja' existia sem aparecer em lugar nenhum da UI. "generico" fica
        # de fora do selo -- e' um placeholder neutro, nao uma alegacao.
        selo = ""
        if str(op) != "generico":
            if p.referencia:
                selo = "  ✅" if _lang() == "PT" else "  ✅"
            else:
                selo = ("  ⚠ nao validado" if _lang() == "PT"
                        else "  ⚠ not validated")
        if key == "perfil_tecnica" and p.nivel_agrupamento_tipico:
            garantia = ("garantia tipica" if _lang() == "PT" else "typical guarantee")
            return f"{op} — {desc} [{garantia}: {p.nivel_agrupamento_tipico}]{selo}"
        return (f"{op} — {desc}{selo}" if desc else f"{op}{selo}")
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
    real e `run_benchmark`. Sempre use esta funcao para ler config por key.
    """
    spec = _SPEC_BY_KEY.get(key)
    attr = spec["attr"] if spec else key
    return getattr(cfg, attr, default)


# _count_dx importada de guaraci.cli_logic (item 19) — ver topo do arquivo.

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
def _validate_ranges(cfg: Config) -> list:
    """Wrapper fino: le faixa_min/max do Config e delega a validacao pura
    a `guaraci.cli_logic.validate_ranges` (testada)."""
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
                  (f"  [{PM}](Para FT-NIR difuso, comece por msc+sg+mc)[/{PM}]" if preproc == "raw" else ""))
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

    avisos_faixa = _validate_ranges(cfg)
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

#: As 18 abas reais do CLI (confirmado por auditoria funcional, Agente 1 --
#: nao existe "0"), tecla -> (chave t_, chave d_). "G" fica de fora: e' o
#: proprio assistente, nao uma secao pra' navegar ATE (circular).
_SECOES_NAVEGAVEIS: List[Tuple[str, str, str]] = [
    ("1", "t_projeto", "d_projeto"),
    ("2", "t_dados", "d_dados"),
    ("3", "t_preproc", "d_preproc"),
    ("4", "t_modelagem", "d_modelagem"),
    ("5", "t_validacao", "d_validacao"),
    ("6", "t_avancado", "d_avancado"),
    ("7", "t_viz", "d_viz"),
    ("8", "t_tecnica", "d_tecnica"),
    ("9", "t_codigos", "d_codigos"),
    ("H", "t_hardware", "d_hardware"),
    ("B", "t_predicao", "d_predicao"),
    ("X", "t_hsi", "d_hsi"),
    ("J", "t_planejamento", "d_planejamento"),
    ("U", "t_auditoria", "d_auditoria"),
    ("K", "t_selecao_amostras", "d_selecao_amostras"),
    ("P", "t_perfis", "d_perfis"),
    ("?", "t_ajuda", "d_ajuda"),
]


def _guaraci_navegar_secoes(cfg: Config) -> None:
    """Lista as 18 abas reais do CLI e exibe descricao quando selecionada.

    Antes (achado do Agente 6, docs/DESIGN.md): dict estatico escrito a
    mao com so' 8 secoes, faltando 10 das 18 abas reais (H/B/J/U/K/P/?/A
    + a lista supunha existir uma aba "0" que nunca existiu). Agora deriva
    de `_t()`/`d_*`, a MESMA fonte que ja alimenta o rodape/ajuda de cada
    aba individual -- uma secao nova precisa so' de um par t_/d_ na tabela
    de traducao pra' aparecer aqui tambem, nao de editar este dict.
    """
    lang = _lang()
    secoes = {k: (_t(t_key), _t(d_key)) for k, t_key, d_key in _SECOES_NAVEGAVEIS}
    secoes["A"] = ("Sobre" if lang == "PT" else "About", _t("d_sobre"))
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


def _guaraci_diagnosticar(cfg: Config) -> None:
    """Diagnostica o dataset carregado -- reaproveita `run_audit`, o MESMO
    motor da aba U (Auditoria de delineamento), so' apresentado dentro do
    assistente. So' roda sob demanda (nao em toda abertura do assistente,
    que seria lento e a maioria das aberturas e' so' pra ajuda pontual).

    Regra dura (Agente 6): nunca inventa numero, nunca esconde ressalva --
    todo achado abaixo vem de uma funcao ja testada (`run_audit`,
    `achievable_alpha`, `n_minimum_for_alpha`), nunca de um calculo novo
    so' pra esta tela.
    """
    lang = _lang(); is_pt = lang == "PT"
    status_msg = ("Carregando dados e diagnosticando..." if is_pt
                  else "Loading data and diagnosing...")
    try:
        with console.status(f"[{PA}]{status_msg}[/{PA}]"):
            wavenumbers, X_raw, rotulos, conc, mae_id, _metadados = pq.load_data(cfg)
            X_raw, wavenumbers, rotulos, conc, mae_id, _relatorio = pq.validate_input(
                X_raw, wavenumbers, rotulos, conc, mae_id)
            from guaraci.auditoria_delineamento import run_audit
            achados = run_audit(X_raw, wavenumbers, rotulos, cfg, conc, mae_id)
    except Exception as e:  # noqa: BLE001 -- dado externo pode falhar de
        # varias formas (parsing, faixa espectral vazia, etc.) -- reportar
        # a mensagem, nunca stack trace cru numa ferramenta interativa
        # (mesmo tratamento de _menu_audit).
        console.print(f"  [{PR}]{'Erro ao carregar dados' if is_pt else 'Error loading data'}: "
                      f"{escape(str(e))}[/{PR}]")
        _pause(); return

    cores = {"ok": PG, "aviso": PA, "critico": PR, "silenciado": PM}
    console.print()
    for a in achados:
        cor = cores.get(a.severidade, PW)
        console.print(f"  [{cor}]{a.severidade.upper():>10}[/{cor}]  "
                      f"[{PW}]{escape(a.nome)}[/{PW}]: {escape(a.mensagem)}")

    n_criticos = sum(1 for a in achados if a.severidade == "critico")
    n_avisos = sum(1 for a in achados if a.severidade == "aviso")
    resumo_lbl = (f"{n_criticos} critico(s), {n_avisos} aviso(s) de {len(achados)} checagem(ns)."
                  if is_pt else
                  f"{n_criticos} critical, {n_avisos} warning(s) out of {len(achados)} check(s).")
    cor_resumo = PR if n_criticos else (PA if n_avisos else PG)
    console.print()
    console.print(Panel(
        Text.from_markup(f"  {resumo_lbl}"),
        border_style=cor_resumo, box=rbox.ROUNDED, padding=(0, 2),
    ))

    # Sugerir+executar (Agente 6, Fase 1 -- so' este 1 caso, como prova do
    # padrao; os demais exemplos do pedido original ficam para uma rodada
    # futura, ver docs/DESIGN.md).
    achado_n = next((a for a in achados
                      if a.nome == "n_insuficiente" and a.severidade == "aviso"),
                     None)
    if achado_n is not None and mae_id is not None:
        info = _sugestao_alpha_classe_fraca(rotulos, mae_id)
        if info is not None:
            sugestao = (
                f"💡 Sua classe mais fraca ('{info['classe']}') tem "
                f"{info['n']} sessao(oes) independente(s) → alpha minimo "
                f"alcancavel = {info['alpha_alcancavel']:.2f}. Para "
                f"alpha={info['alpha_ref']} de referencia (gate conformal "
                f"padrao), seriam necessarias {info['n_para_ref']} sessoes "
                "independentes dessa classe."
                if is_pt else
                f"💡 Your weakest class ('{info['classe']}') has "
                f"{info['n']} independent session(s) → minimum achievable "
                f"alpha = {info['alpha_alcancavel']:.2f}. For the "
                f"reference alpha={info['alpha_ref']} (standard conformal "
                f"gate), {info['n_para_ref']} independent sessions of that "
                "class would be needed.")
            console.print()
            console.print(Panel(
                Text.from_markup(f"  [{PA}]{escape(sugestao)}[/{PA}]"),
                border_style=PA, box=rbox.ROUNDED, padding=(0, 2),
            ))
            # Fase 2 (Agente 6): "sugerir" nao para no texto -- oferece a
            # ACAO. Reaproveita plan_from_statistical_target (mesma funcao
            # que _menu_plan chama), nao inventa um calculo novo aqui.
            pergunta = ("Quer ver o plano de coleta para atingir esse alpha? (s/n) "
                        if is_pt else
                        "Want to see the collection plan to reach that alpha? (y/n) ")
            if _ask(f"  [{PA}]{pergunta}[/{PA}]").strip().lower() in ("s", "y", "sim", "yes"):
                from guaraci.plano_coleta import plan_from_statistical_target
                try:
                    classes_unicas = sorted({str(r) for r in rotulos})
                    plano, meta = plan_from_statistical_target(
                        classes_unicas, n_sessoes=2,
                        alpha_conformal=info["alpha_ref"])
                    console.print()
                    console.print(f"  [{PS}]{'n por classe' if is_pt else 'n per class'}: "
                                  f"{meta['n_por_classe']} ({meta['origem']})[/{PS}]")
                    for alerta in plano.alertas:
                        console.print(f"  [{PM}]• {escape(str(alerta))}[/{PM}]")
                except ValueError as e_plano:
                    console.print(f"  [{PR}]{escape(str(e_plano))}[/{PR}]")
    _pause()


def _sugestao_alpha_classe_fraca(rotulos, mae_id) -> Optional[Dict[str, Any]]:
    """Sessoes independentes por classe (mesma funcao `session_from_mae_id`
    que `check_insufficient_n` usa) e o alpha conformal minimo alcancavel
    para a classe mais fraca. Funcao PURA (sem console) para ser testavel
    direto -- usada por `_guaraci_diagnosticar` (Agente 6, sugerir+executar).
    Nao duplica o VEREDITO da auditoria (isso continua em
    `auditoria_delineamento.check_insufficient_n`), so' os numeros
    descritivos que `AuditFinding` nao expoe (so' a mensagem ja formatada).
    """
    import numpy as np
    from guaraci.conformal import achievable_alpha, n_minimum_for_alpha
    from guaraci.dados_io import session_from_mae_id

    rotulos_arr = np.asarray(rotulos, dtype=str)
    sessao = np.array([session_from_mae_id(m) for m in mae_id], dtype=str)
    contagens = {
        classe: len({s for s, r in zip(sessao, rotulos_arr) if r == classe})
        for classe in sorted(set(rotulos_arr))
    }
    if not contagens:
        return None
    classe_fraca, n_fraco = min(contagens.items(), key=lambda kv: kv[1])
    alpha_ref = 0.05
    return {
        "classe": classe_fraca,
        "n": n_fraco,
        "alpha_alcancavel": achievable_alpha(n_fraco),
        "alpha_ref": alpha_ref,
        "n_para_ref": n_minimum_for_alpha(alpha_ref),
    }


def _guaraci_tecnicas() -> None:
    """Lista o catalogo de tecnicas cientificas do GUARACI (Agente 6, item
    d) -- gerado a partir de `technique_registry.REGISTRY`, fonte unica de
    verdade (nunca uma lista escrita a mao so' pra esta tela)."""
    lang = _lang(); is_pt = lang == "PT"
    from guaraci.technique_registry import REGISTRY

    rotulos_categoria = {
        "classificacao_deteccao": ("Classificacao / deteccao", "Classification / detection"),
        "quantificacao": ("Quantificacao", "Quantification"),
        "identificacao_conjunto_aberto": ("Identificacao (conjunto aberto)", "Identification (open set)"),
        "selecao_amostras": ("Selecao de amostras", "Sample selection"),
        "transferencia_calibracao": ("Transferencia de calibracao", "Calibration transfer"),
        "figuras_de_merito": ("Figuras de merito", "Figures of merit"),
        "robustez_linearidade": ("Robustez / linearidade", "Robustness / linearity"),
        "perfis": ("Perfis (matriz / tecnica de aquisicao)", "Profiles (matrix / acquisition technique)"),
    }
    console.print()
    for cat_id, (nome_pt, nome_en) in rotulos_categoria.items():
        entradas = [e for e in REGISTRY if e.categoria == cat_id]
        if not entradas:
            continue
        console.print(f"  [bold {PA}]{nome_pt if is_pt else nome_en}[/bold {PA}]")
        for e in entradas:
            console.print(f"    [{PW}]▸ {escape(e.nome)}[/{PW}]")
            console.print(f"      [{PM}]{escape(e.quando_usar)}[/{PM}]")
        console.print()
    console.print(f"  [{PM}]{len(REGISTRY)} tecnicas cadastradas.[/{PM}]" if is_pt
                  else f"  [{PM}]{len(REGISTRY)} techniques registered.[/{PM}]")
    _pause()


def _faq_metodo_recomendado(cfg: Config, is_pt: bool) -> str:
    """Grounded no cfg REAL da sessao (nivel/objetivo/mode) -- nunca um
    conselho generico solto, regra dura do Agente 6."""
    nivel = str(getattr(cfg, "level", "N1"))
    mode = str(getattr(cfg, "mode", "dx"))
    nome_nivel = pq._NIVEL_NOME.get(nivel, nivel)
    if is_pt:
        base = f"Sua sessao esta configurada para '{nome_nivel}' ({nivel}), modo de entrada '{mode}'."
        if nivel == "N1":
            return (base + " N1 = classificacao multiclasse -> PLS-DA e o "
                    "metodo padrao do pipeline. Se quiser autenticar pureza "
                    "(puro vs. adulterado) em vez de identificar a especie, "
                    "troque pra N2 na aba Modelagem.")
        if nivel == "N2":
            return (base + " N2 = autenticacao one-class -> DD-SIMCA e "
                    "conformal sao os dois metodos disponiveis (aba "
                    "Modelagem, campo 'modo_ddsimca'/similar). DD-SIMCA da' "
                    "um score continuo; conformal da' garantia de cobertura "
                    "explicita, mas exige mais sessoes independentes pra "
                    "alpha baixo (ver [3] Diagnosticar).")
        return (base + " N3 = quantificacao -> PLS-R (pooled ou por "
                "especie, aba Modelagem). Se tambem quiser identificar QUAL "
                "adulterante antes de quantificar, o fluxo cego "
                "(Detectar->Identificar->Quantificar) ja roda automatico na "
                "predicao em lote quando o modelo tem o ensemble de "
                "identificacao treinado.")
    base = f"Your session is set to '{nome_nivel}' ({nivel}), input mode '{mode}'."
    if nivel == "N1":
        return (base + " N1 = multiclass classification -> PLS-DA is the "
                "pipeline's default method. For purity authentication "
                "(pure vs. adulterated) instead of species ID, switch to "
                "N2 in the Model tab.")
    if nivel == "N2":
        return (base + " N2 = one-class authentication -> DD-SIMCA and "
                "conformal are the two available methods (Model tab). "
                "DD-SIMCA gives a continuous score; conformal gives an "
                "explicit coverage guarantee but needs more independent "
                "sessions for a low alpha (see [3] Diagnose).")
    return (base + " N3 = quantification -> PLS-R (pooled or per-species, "
            "Model tab). If you also want to identify WHICH adulterant "
            "before quantifying, the blind flow (Detect->Identify->"
            "Quantify) already runs automatically in batch prediction when "
            "the model has a trained identification ensemble.")


_FAQ: List[Tuple[str, str, Any]] = [
    ("O que o GUARACI sabe fazer?", "What can GUARACI do?",
     lambda cfg, is_pt: (
         "Veja [4] Tecnicas disponiveis no menu do assistente -- lista "
         "gerada do catalogo real do projeto, nunca fica desatualizada."
         if is_pt else
         "See [4] Available techniques in the assistant menu -- generated "
         "from the project's real catalog, never goes stale.")),
    ("Qual metodo devo usar?", "Which method should I use?",
     _faq_metodo_recomendado),
    ("Por que uma quantificacao pode ficar bloqueada?",
     "Why can quantification be blocked?",
     lambda cfg, is_pt: (
         "No fluxo cego (Detectar->Identificar->Quantificar), a "
         "quantificacao SO' roda quando o adulterante foi identificado com "
         "garantia estatistica validada (identificacao_cobertura="
         "'validado', >=2 sessoes de coleta independentes por combinacao "
         "especie x adulterante). Sem essa garantia, a coluna "
         "'quantificacao_motivo_bloqueio' explica o motivo especifico "
         "('identificacao_desconhecida' ou 'identificacao_ambigua') -- "
         "nunca um numero sem base."
         if is_pt else
         "In the blind flow (Detect->Identify->Quantify), quantification "
         "ONLY runs when the adulterant was identified with a validated "
         "statistical guarantee (identificacao_cobertura='validado', >=2 "
         "independent collection sessions per species x adulterant "
         "combination). Without that guarantee, the "
         "'quantificacao_motivo_bloqueio' column explains the specific "
         "reason ('identificacao_desconhecida' or 'identificacao_ambigua') "
         "-- never a number without basis.")),
    ("O que significa um perfil 'nao validado'?",
     "What does an 'not validated' profile mean?",
     lambda cfg, is_pt: (
         "O selo ✅/⚠ no seletor de perfil (aba Dados) vem do campo "
         "'referencia' do perfil: nao-vazio = tem paper/dataset publico "
         "citado (ex.: milho_nir cita o dataset Corn); vazio = o perfil so' "
         "foi DECLARADO (faixa/vocabulario definidos), sem nenhuma "
         "validacao publicada ainda. Nao impede de usar -- so' nao alegue "
         "que o resultado foi validado com dado real."
         if is_pt else
         "The ✅/⚠ badge on the profile selector (Data tab) comes from the "
         "profile's 'referencia' field: non-empty = cites a public paper/"
         "dataset (e.g. milho_nir cites the Corn dataset); empty = the "
         "profile was only DECLARED (range/vocabulary set), with no "
         "published validation yet. It doesn't block usage -- just don't "
         "claim the result was validated against real data.")),
]


def _guaraci_faq(cfg: Config) -> None:
    """Perguntas frequentes curadas (Agente 6, Fase 2) -- casadas por
    numero, resposta ancorada no `cfg` real quando aplicavel (regra dura:
    nunca inventa numero). Nao e' um chat de linguagem livre -- o projeto
    nao tem dependencia de LLM/NLP, um FAQ curado e' o que da' pra' fazer
    sem mudar a natureza determinista do software (ver docs/DESIGN.md,
    secao do Agente 6, Fase 2)."""
    lang = _lang(); is_pt = lang == "PT"
    console.print()
    t = Table(show_header=False, box=rbox.SIMPLE, padding=(0, 1))
    t.add_column("N", style=PA, width=4)
    t.add_column("Pergunta" if is_pt else "Question", style=PW)
    for i, (q_pt, q_en, _resp) in enumerate(_FAQ, 1):
        t.add_row(f"[{i}]", q_pt if is_pt else q_en)
    console.print(t)
    raw = _ask(f"\n  [1-{len(_FAQ)}] ou Enter=voltar: " if is_pt
               else f"\n  [1-{len(_FAQ)}] or Enter=back: ")
    if raw.isdigit() and 1 <= int(raw) <= len(_FAQ):
        _q_pt, _q_en, resposta_fn = _FAQ[int(raw) - 1]
        resposta = resposta_fn(cfg, is_pt)
        console.print()
        console.print(Panel(
            Text(resposta, style=PW),
            title=f"[bold {PA}]{_q_pt if is_pt else _q_en}[/bold {PA}]",
            border_style=PA, box=rbox.ROUNDED, padding=(0, 2), width=_W(),
        ))
    _pause()


def _abrir_assistente(contexto: str = "", cfg: Optional[Config] = None) -> None:
    """Abre o Assistente Guaraci (tecla G em qualquer tela)."""
    lang = _lang()
    _cls(); _print_header(cfg)
    console.print()

    opcoes = [
        ("1", "Revisar configuracao atual" if lang=="PT" else "Review current configuration"),
        ("2", "Informacoes sobre uma secao" if lang=="PT" else "Information about a section"),
        ("3", "Diagnosticar dados carregados" if lang=="PT" else "Diagnose loaded data"),
        ("4", "Tecnicas disponiveis" if lang=="PT" else "Available techniques"),
        ("5", "Perguntas frequentes" if lang=="PT" else "Frequently asked questions"),
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
    elif raw == "3":
        _guaraci_diagnosticar(cfg or Config())
    elif raw == "4":
        _guaraci_tecnicas()
    elif raw == "5":
        _guaraci_faq(cfg or Config())

def _rotulo_tecnica_efetivo(cfg: Optional[Config]) -> str:
    """Nome de 'tecnica' a exibir nos cabecalhos, ajustado ao `cfg.mode`
    real -- achado do Passo 103 (INSTRUCAO_HSI_ROBUSTEZ_E_VALIDACAO.md):
    `_TECNICA_SELECIONADA` (escolhida em [8] Tecnica Analitica) so' faz
    sentido para mode dx/csv/sintetico (espectros vibracionais). Modes
    "imagem" (colorimetria) e "hsi" (imageamento hiperespectral) NAO sao
    "uma tecnica vibracional escolhida" -- mostrar o default global
    'FT-NIR' nessas telas era herdado do template generico, incorreto
    (a tela HSI mostrando "Tecnica: FT-NIR" foi o achado que motivou
    esta correcao). Usado tanto por `_print_header` quanto por
    `_print_status` -- fonte unica, nao duas heuristicas divergentes."""
    modo = getattr(cfg, "mode", None) if cfg is not None else None
    if modo == "hsi":
        return "HSI"
    if modo == "imagem":
        return "Colorimetria digital" if _lang() == "PT" else "Digital colorimetry"
    return _TECNICA_SELECIONADA.get("nome", "FT-NIR")


# ---------------------------------------------------------------------------
# CABECALHO COMPACTO
# ---------------------------------------------------------------------------
def _print_header(cfg: Optional[Config] = None) -> None:
    # Titulo com icone solar flanqueando GUARACI
    titulo = Text(justify="center")
    titulo.append("  ", style=f"{PA}")
    titulo.append("GUARACI", style=f"bold {PA}")
    titulo.append("  ", style=f"{PA}")

    # Tecnica ativa (atualizada dinamicamente, ajustada ao mode -- ver
    # _rotulo_tecnica_efetivo)
    tec_nome = _rotulo_tecnica_efetivo(cfg)
    tec_str  = f"Tecnica: {tec_nome}" if _lang() == "PT" else f"Technique: {tec_nome}"

    sub = Text(
        "Plataforma quimiometrica com validacao anti-vazamento por padrao"
        if _lang() == "PT" else
        "Chemometrics platform with leakage-safe validation by default",
        style=PS, justify="center"
    )
    rod_txt = f"Quimiometria  |  Machine Learning  |  {tec_str}"
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
    n_dx = _count_dx(pasta) if pasta_ok else 0

    if pasta_ok and n_dx > 0:
        dados_str = f"[g]{_t('dados_ok', n=n_dx)}[/g]"
    elif pasta_ok:
        dados_str = f"[warn]{escape(str(pasta))}[/warn]"
    else:
        dados_str = f"[err]{_t('dados_err')}[/err]"

    preproc  = escape(str(_cfgv(cfg, "pre_processamento", "msc_sg_mc")))
    # Barra de status compacta: usa so a palavra-chave do mode (Classificacao/
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

    tec_nome = _rotulo_tecnica_efetivo(cfg)

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

    # Grupos (docs/DESIGN.md secao 4 -- Agente 5.2, aprovado 2026-09-01):
    # 6 grupos alinhados ao fluxo real de trabalho, substituindo os 3 grupos
    # antigos (config/analise/sistema). Atalho direto de cada tecla continua
    # funcionando igual, digitado de qualquer lugar deste menu -- so' a
    # organizacao visual mudou, o dispatch em main() nao foi tocado.
    t.add_row(Text.from_markup(_grp(_t("grp_preparar"))), Text.from_markup(""))
    t.add_row(*row("2", _t("t_dados"),      "3", _t("t_preproc")))
    t.add_row(*row("9", _t("t_codigos"),    "P", _t("t_perfis")))

    t.add_row(Text.from_markup(""), Text.from_markup(""))
    t.add_row(Text.from_markup(_grp(_t("grp_planejar"))), Text.from_markup(""))
    t.add_row(*row("J", _t("t_planejamento"), "K", _t("t_selecao_amostras")))
    t.add_row(*row("U", _t("t_auditoria")))

    t.add_row(Text.from_markup(""), Text.from_markup(""))
    t.add_row(Text.from_markup(_grp(_t("grp_modelar"))), Text.from_markup(""))
    t.add_row(*row("4", _t("t_modelagem"),  "6", _t("t_avancado")))
    t.add_row(*row("8", _t("t_tecnica")))

    t.add_row(Text.from_markup(""), Text.from_markup(""))
    t.add_row(Text.from_markup(_grp(_t("grp_validar"), cor=S)), Text.from_markup(""))
    t.add_row(*row("5", _t("t_validacao"),  "7", _t("t_viz"), style1=S, style2=S))

    t.add_row(Text.from_markup(""), Text.from_markup(""))
    t.add_row(Text.from_markup(_grp(_t("grp_prever"), cor=S)), Text.from_markup(""))
    t.add_row(*row("B", _t("t_predicao"), "X", _t("t_hsi"), style1=S, style2=S))

    t.add_row(Text.from_markup(""), Text.from_markup(""))
    t.add_row(Text.from_markup(_grp(_t("grp_sistema"), cor=S)), Text.from_markup(""))
    modo_lbl = (f"Modo: {'Iniciante' if _modo_usuario()=='iniciante' else 'Avancado'}"
                if is_pt else
                f"Mode: {'Beginner' if _modo_usuario()=='iniciante' else 'Advanced'}")
    t.add_row(*row("1", _t("t_projeto"), "H", _t("t_hardware"), style1=S, style2=S))
    t.add_row(*row("G", "Guaraci ☀", "M", modo_lbl, style1=PA, style2=S))
    t.add_row(*row("I", _t("t_idioma"),  "?", _t("t_ajuda"),  style1=S, style2=M))
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
    n_dx = _count_dx(pasta) if pasta_ok else 0
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

    Resolve idioma/_HELP_DB/_SPEC_BY_KEY (estado desta tela) e delega o
    truncamento a `guaraci.cli_logic.truncate_desc_by_sentence` (funcao pura,
    testada).
    """
    lang = _lang()
    h = _HELP_DB.get(key, {})
    desc = h.get(lang, h.get("PT", {})).get("desc", "")
    if not desc:
        desc = _SPEC_BY_KEY.get(key, {}).get("desc", "")
    return _truncar_desc_por_frase(desc, max_c)


def _print_submenu_compact(
    title: str, desc: str, fields: List[str], cfg: Config,
    extras: Optional[List[Tuple[str, str]]] = None,
    campos_avancados: Optional[set] = None,
    mostrar_avancado: bool = True,
) -> List[str]:
    """
    Submenu compacto: [N] ICON Nome  Valor  Descricao-breve
    Exibe o valor atual e uma descricao curta na mesma linha.

    `campos_avancados`: subconjunto de `fields` a ESCONDER quando o mode do
    usuario e' Iniciante e `mostrar_avancado=False` (CLAUDE.md secao 6 /
    auditoria 2026-07-12: reduzir a densidade de configuracao p/ quem so'
    quer usar os defaults). Quando `None`, nenhum campo e' escondido --
    comportamento identico ao de antes desta feature (compatibilidade com
    chamadores que nao passam esse argumento).

    Retorna a lista de campos REALMENTE exibidos (numerados 1..N na tela),
    que o chamador deve usar para indexar a escolha do usuario -- os
    numeros na tela sempre correspondem a esta lista, nunca ao `fields`
    original quando ha' campos escondidos.
    """
    lang = _lang()
    ocultos = (campos_avancados or set()) if (
        campos_avancados and _modo_usuario() == "iniciante" and not mostrar_avancado
    ) else set()
    fields_visiveis = [f for f in fields if f not in ocultos]

    t = Table(box=None, show_header=False, padding=(0, 0), expand=True)
    t.add_column("N",    no_wrap=True, width=5)
    t.add_column("Ico",  no_wrap=True, width=2)
    t.add_column("Nome", no_wrap=True, min_width=20, max_width=26)
    t.add_column("Val",  no_wrap=True, min_width=10, max_width=18)
    t.add_column("Desc", no_wrap=True, style=PM)

    for i, key in enumerate(fields_visiveis, 1):
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
    n_ocultos = len(fields) - len(fields_visiveis)
    aviso_ocultos = (
        (f"  [{PR}][V][/{PR}] Mostrar opcoes avancadas ({n_ocultos} ocultas)\n"
         if lang == "PT" else
         f"  [{PR}][V][/{PR}] Show advanced options ({n_ocultos} hidden)\n")
        if n_ocultos > 0 else ""
    )
    rodape = Text.from_markup(
        f"{aviso_ocultos}"
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
    return fields_visiveis

# Toggles cuja utilidade depende do OBJETIVO cientifico resolvido (nao do
# nivel em si) -- espelha 1:1 modos_analise._FIG_OBJETIVOS: fora do objetivo
# listado, o motor nem computa (gated por should_generate() em pipeline.executar(),
# ou, no caso de teste_wold/teste_cv_anova, pelo guard adicionado no achado
# de 2026-08-06 -- antes rodavam incondicionalmente e escreviam metrica de
# CLASSIFICACAO sem sentido no resumo de um run de Quantificacao).
_TOGGLES_SO_CLASSIFICACAO = (
    "opls_da", "selecao_variaveis_etapa4", "comparar_pre_processamentos",
    "teste_wold", "teste_cv_anova", "teste_martens",
    "benchmark", "monte_carlo", "shap_benchmark",
)
_TOGGLES_SO_QUANTIFICACAO = ("benchmark_regressao",)


def _ajustar_toggles_por_nivel(cfg: Config) -> List[str]:
    """Desliga toggles que ficam INERTES no nivel/objetivo atual de `cfg`.

    Chamado apos o campo "nivel" mudar de valor (pedido do usuario,
    2026-08-06: "quando mudo de mode... continua ativado a dd simca e
    semelhantes... gostaria que ao mudar o N, mudasse as opcoes de modos
    como esse que nao agrega a analise"). Antes disso, o toggle DD-SIMCA
    permanecia visualmente "ligado" no menu mesmo em N1 -- funcionalmente
    inerte (pipeline.executar() ignora com aviso), mas confuso na tela.

    A logica espelha EXATAMENTE o que pipeline.executar()/modos_analise.py
    decidem em runtime -- nao e' uma aproximacao da UI, e' a mesma regra:
      - DD-SIMCA: N1 sempre ignora (forca False); N2 sempre forca ligado
        internamente (forca True, refletindo o que vai acontecer de
        qualquer forma); demais niveis respeitam should_generate (objetivo).
      - Demais toggles (ver `_TOGGLES_SO_CLASSIFICACAO`): forcados False
        fora de objective=Classificacao; `benchmark_regressao` fora de
        Quantificacao.

    Retorna a lista de CHAVES (_CONFIG_SPEC) efetivamente alteradas, para o
    chamador avisar o usuario -- um reset silencioso seria tao confuso
    quanto o problema original.
    """
    mudou: List[str] = []

    def _forcar(key: str, valor: bool) -> None:
        spec = _SPEC_BY_KEY.get(key)
        if spec is None:
            return
        attr = spec["attr"]
        if bool(getattr(cfg, attr, False)) != valor:
            setattr(cfg, attr, valor)
            mudou.append(key)

    if cfg.level == "N1":
        _forcar("ddsimca", False)
    elif cfg.level == "N2":
        _forcar("ddsimca", True)
    else:
        _forcar("ddsimca", pq.should_generate(cfg, "ddsimca"))

    objetivo = pq.resolve_objective(cfg)
    if objetivo != pq.CLASSIFICACAO:
        for key in _TOGGLES_SO_CLASSIFICACAO:
            _forcar(key, False)
    if objetivo != pq.QUANTIFICACAO:
        for key in _TOGGLES_SO_QUANTIFICACAO:
            _forcar(key, False)

    return mudou


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
    risk      = _RISK_CLASS.get(key, "ANALITICO")
    r_hex     = _risco_hex(key)

    # Painel de edicao minimalista
    info = Table(box=None, show_header=False, padding=(0, 1))
    info.add_column("L", style=PM, width=10, no_wrap=True)
    info.add_column("V", no_wrap=False)
    # Campos booleanos SEMPRE viram escolha numerada [1]/[2], nunca texto
    # livre. Antes disso, os 20 campos bool do _CONFIG_SPEC caiam no ramo
    # else (texto livre): _coagir_valor aceitava so' um punhado de palavras
    # magicas ("true"/"sim"/"1"/"yes"/"s"/"v") como True e tratava QUALQUER
    # outra coisa como False SEM AVISO — digitar "y" (comum em outros
    # softwares) virava False silenciosamente, e um erro de digitacao nunca
    # dava mensagem de erro. Unificado com o MESMO padrao numerado que todo
    # campo de escolha ja usa (nivel, modo_ddsimca, etc.) — "transformar tudo
    # em modelo padrao" (pedido do usuario, 2026-08-06).
    eh_bool = (tipo == "bool")
    if eh_bool:
        val_txt_atual = _t("opt_sim") if bool(val_cru) else _t("opt_nao")
    else:
        val_txt_atual = str(val_atual)

    info.add_row("Campo:", Text(nome, style=f"bold {r_hex}"))
    info.add_row("Atual:", Text(val_txt_atual, style=PS))
    info.add_row("Tipo:", Text(tipo, style=PM))
    if opcoes:
        info.add_row("Opcoes:", Text(
            " | ".join(_rotulo_opcao(key, o) for o in opcoes), style=PM))

    console.print(Panel(
        info,
        title=f"[{r_hex}]Editar: {escape(nome)}[/{r_hex}]",
        border_style=r_hex, box=rbox.ROUNDED, padding=(0, 1)
    ))

    if eh_bool:
        opcoes_bool = (True, False)   # [1]=Sim/Yes, [2]=Nao/No, sempre nesta ordem
        console.print()
        for j, op in enumerate(opcoes_bool, 1):
            mk = f"[{PA}]►[/{PA}]" if bool(op) == bool(val_cru) else " "
            rotulo = _t("opt_sim") if op else _t("opt_nao")
            console.print(f"  {mk} [{PA}][{j}][/{PA}] {rotulo}")
        console.print()
        raw = _input(f"  {_t('escolha_num')}")
        if not raw:
            console.print(f"  [{PM}]{_t('mantido')}[/{PM}]"); return False
        if raw == "?":
            _mostrar_ajuda(key); return False
        if raw.isdigit() and 1 <= int(raw) <= 2:
            # Sempre "true"/"false" em ingles: e' o vocabulario que
            # _coagir_valor reconhece de forma inequivoca, independente do
            # idioma da interface (mesmo padrao do resto do _CONFIG_SPEC,
            # que grava sempre em ingles/codigo interno no YAML).
            raw = "true" if opcoes_bool[int(raw) - 1] else "false"
        else:
            _msg = ("Digite 1 ou 2." if lang == "PT" else "Enter 1 or 2.")
            console.print(f"  [err]{_msg}[/err]"); return False
    elif opcoes:
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
        nivel_mudou = (key == "nivel" and raw != str(val_cru))
        _set_val(cfg, key, raw)
        # Bool sempre mostra o rotulo localizado (Sim/Nao ou Yes/No), nunca
        # o "true"/"false" cru gravado internamente.
        valor_msg = (_t("opt_sim") if raw == "true" else _t("opt_nao")) \
                    if eh_bool else raw
        msg = _t("atualizado", campo=nome, valor=valor_msg)
        console.print(f"  [g]✓ {escape(msg)}[/g]")
        if nivel_mudou:
            ajustados = _ajustar_toggles_por_nivel(cfg)
            if ajustados:
                nomes = ", ".join(_nome_campo(k) for k in ajustados)
                console.print(f"  [{PM}]{_t('ajuste_nivel', campos=nomes)}[/{PM}]")
        return True
    except ValueError as e:   # _set_val/_coagir_valor: valor digitado invalido
        console.print(f"  [err]Erro: {escape(str(e))}[/err]")
        return False

# ---------------------------------------------------------------------------
# AJUDA POR CAMPO — descricao completa + dica unica do Guaraci
# ---------------------------------------------------------------------------
def _mostrar_ajuda(key: str) -> None:
    lang   = _lang()
    h      = _HELP_DB.get(key, {})
    h_lang = h.get(lang, h.get("PT", {}))
    nome   = _nome_campo(key)
    r_hex  = _risco_hex(key)
    spec   = _SPEC_BY_KEY.get(key, {})

    # Fallback: se _HELP_DB nao cobre o campo, usa a descricao do _CONFIG_SPEC
    desc    = h_lang.get("desc") or spec.get("desc") or (
        "Sem descricao detalhada para este campo." if lang == "PT"
        else "No detailed description for this field.")
    impacto = h_lang.get("impacto", "—")
    exemplos = h_lang.get("exemplos", {})
    default  = h.get("default", spec.get("default", "—"))
    opcoes   = spec.get("opcoes")
    faixa    = h.get("range") or (" | ".join(str(o) for o in opcoes) if opcoes else
                                  spec.get("tipo", "—"))
    tip      = _GUARACI_TIPS.get(key, {}).get(lang, "")

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
               on_extra: Optional[Dict[str, Any]] = None,
               campos_avancados: Optional[set] = None) -> None:
    """Loop generico para submenus de configuracao.

    `campos_avancados`: ver `_print_submenu_compact`. O reveal ("V") e'
    local a esta visita ao menu -- sai e volta a entrar reseta p/ escondido
    de novo quando o mode do usuario e' Iniciante (design: expandir um
    submenu especifico nao muda o mode da sessao inteira)."""
    mostrar_avancado = False
    while True:
        _cls()
        _print_header(cfg)
        fields_visiveis = _print_submenu_compact(
            title, desc, fields, cfg, extras,
            campos_avancados=campos_avancados, mostrar_avancado=mostrar_avancado)
        raw = _input(f"\n  {_t('opcao')}: ").upper()

        if raw in ("0", "Q", ""):
            break
        elif raw == "V" and campos_avancados:
            mostrar_avancado = not mostrar_avancado
        elif raw == "I":
            _toggle_idioma()
        elif raw == "G":
            _abrir_assistente(title, cfg)
        elif raw == "?":
            r2 = _input("  Campo (N ou nome): ").strip()
            if r2.isdigit() and 1 <= int(r2) <= len(fields_visiveis):
                _mostrar_ajuda(fields_visiveis[int(r2) - 1])
            elif r2 in _HELP_DB:
                _mostrar_ajuda(r2)
            else:
                found = [k for k in _HELP_DB if r2.lower() in k.lower() or r2.lower() in _nome_campo(k).lower()]
                _mostrar_ajuda(found[0]) if found else console.print(f"  [{PM}]{_t('invalido')}[/{PM}]")
        elif raw.isdigit() and 1 <= int(raw) <= len(fields_visiveis):
            _editar_campo(cfg, fields_visiveis[int(raw) - 1])
            _pause()
        elif on_extra and raw in on_extra:
            on_extra[raw]()
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]")
            _pause()


def _menu_project(cfg: Config) -> None:
    _loop_menu(_t("t_projeto"), _t("d_projeto"), ["pasta_dados", "pasta_saida", "tag"], cfg)


_DIR_PERFIS_COMBINADOS = _USER_DIR / "perfis_matriz"


def _salvar_perfil_combinado(cfg: Config) -> None:
    """Funde `perfil_matriz` + `perfil_tecnica` (Agente 5B) e salva como
    YAML de usuario, pronto pra reusar digitando o caminho no campo
    `perfil_matriz` de uma proxima sessao."""
    lang = _lang(); is_pt = lang == "PT"
    nome_matriz = str(getattr(cfg, "matrix_profile", "") or "")
    nome_tecnica = str(getattr(cfg, "acquisition_profile", "") or "")
    if not nome_matriz:
        console.print(f"  [{PR}]{'Defina o Perfil de matriz antes de combinar.' if is_pt else 'Set the matrix profile before combining.'}[/{PR}]")
        _pause(); return
    if not nome_tecnica:
        console.print(f"  [{PR}]{'Defina o Perfil de tecnica de aquisicao antes de combinar.' if is_pt else 'Set the acquisition technique profile before combining.'}[/{PR}]")
        _pause(); return

    from guaraci.perfil_matriz import (UnknownProfileError, combine_profiles,
                                        load_profile, save_profile)
    try:
        matriz = load_profile(nome_matriz)
        tecnica = load_profile(nome_tecnica)
    except UnknownProfileError as e:
        console.print(f"  [{PR}]{escape(str(e))}[/{PR}]"); _pause(); return

    nome_novo = _ask(
        f"  [{PA}]{'Nome do perfil combinado (ex.: mel_celular)' if is_pt else 'Combined profile name (e.g. honey_phone)'}: [/{PA}]"
    ).strip()
    if not nome_novo:
        console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return

    combinado = combine_profiles(nome_novo, matriz, tecnica)
    caminho = _DIR_PERFIS_COMBINADOS / f"{nome_novo}.yaml"
    try:
        save_profile(combinado, str(caminho))
    except OSError as e:
        console.print(f"  [{PR}]{'Erro ao salvar' if is_pt else 'Error saving'}: {escape(str(e))}[/{PR}]")
        _pause(); return

    msg = (f"✓ Perfil combinado salvo em {caminho}\n"
           f"  Use digitando o caminho no campo 'Perfil de matriz' numa proxima sessao."
           if is_pt else
           f"✓ Combined profile saved to {caminho}\n"
           f"  Use it by typing the path in the 'Matrix profile' field in a future session.")
    console.print(f"  [g]{escape(msg)}[/g]")
    _pause()


def _menu_data(cfg: Config) -> None:
    # imagem_incluir_textura adicionado 2026-08-06: mesma classe de bug de
    # n_jobs_permutacao. So' relevante quando modo_entrada="imagem"
    # (prototipo de colorimetria digital, CLAUDE.md) -- vai em
    # campos_avancados por ser niche, nao por risco.
    extra_lbl = ("Salvar perfil combinado (matriz + tecnica)" if _lang() == "PT"
                 else "Save combined profile (matrix + technique)")
    _loop_menu(_t("t_dados"), _t("d_dados"),
               ["modo_entrada", "perfil_matriz", "perfil_tecnica", "arquivo_csv",
                "coluna_classe", "coluna_concentracao", "faixa_min_cm", "faixa_max_cm",
                "excluir_classes", "imagem_incluir_textura"], cfg,
               extras=[("C", extra_lbl)],
               on_extra={"C": lambda: _salvar_perfil_combinado(cfg)},
               campos_avancados={"perfil_tecnica", "imagem_incluir_textura"})


def _menu_preprocessing(cfg: Config) -> None:
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
        _cls(); _print_header(cfg); _show_pipeline()
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


def _menu_modeling(cfg: Config) -> None:
    # Essenciais p/ Iniciante: nivel (o que estou fazendo) + max_lvs (unico
    # numero que costuma precisar ajustar). Avancados: DD-SIMCA/OPLS-DA/
    # selecao de variaveis sao metodos extras, nao o caminho basico.
    # objetivo/selecao_ag/selecao_spa adicionados 2026-08-06: mesma classe de
    # bug de n_jobs_permutacao. "objetivo" sobrepoe a derivacao automatica
    # nivel->objetivo (uso avancado: a maioria usa "auto" via PROFILES);
    # selecao_ag/selecao_spa sao sub-opcoes de selecao_variaveis_etapa4, so'
    # fazem sentido com ele ligado -- todos em campos_avancados.
    _loop_menu(_t("t_modelagem"), _t("d_modelagem"),
               ["nivel", "objetivo", "max_lvs", "opls_da", "ddsimca",
                "modo_ddsimca", "ddsimca_pcv", "selecao_variaveis_etapa4",
                "selecao_spa", "selecao_ag"], cfg,
               campos_avancados={"objetivo", "opls_da", "ddsimca", "modo_ddsimca",
                                  "ddsimca_pcv", "selecao_variaveis_etapa4",
                                  "selecao_spa", "selecao_ag"})


def _menu_validation(cfg: Config) -> None:
    # n_jobs_permutacao/teste_martens adicionados 2026-08-06: mesma classe de
    # bug -- existiam no Config/_CONFIG_SPEC/_HELP_DB, mas nunca tinham sido
    # colocados em NENHUM menu (so' editaveis a mao no YAML).
    fields = ["holdout_fracao", "validacao_group_aware",
              "n_permutacoes", "n_jobs_permutacao", "teste_wold",
              "teste_cv_anova", "teste_martens"]
    # Essenciais p/ Iniciante: holdout_fracao (facil de entender: quanto fica
    # de fora p/ teste) + validacao_group_aware (o diferencial central do
    # projeto -- fica visivel mesmo p/ quem nao vai mexer nele). Avancados:
    # testes estatisticos extras (Wold/CV-ANOVA/Martens), permutacoes sao
    # tuning fino.
    #
    # n_jobs_permutacao FORA de campos_avancados de proposito: nao muda
    # nenhum resultado (so' o tempo), e o proprio checklist de pre-execucao
    # (_checklist) sugere subir esse valor quando ha' muitas permutacoes
    # sequenciais -- esconder atras do mode Avancado criaria uma dica que o
    # usuario Iniciante nao consegue seguir.
    campos_avancados = {"n_permutacoes", "teste_wold", "teste_cv_anova", "teste_martens"}
    mostrar_avancado = False
    while True:
        _cls(); _print_header(cfg)
        ga = _cfgv(cfg, "validacao_group_aware", True)
        if not ga:
            console.print(Panel(
                "[err]  GroupKFold DESATIVADO — risco de data leakage![/err]\n"
                "[err]  Ative o campo [2] imediatamente.[/err]",
                border_style=PR, box=rbox.HEAVY, padding=(0, 1)
            ))
        fields_visiveis = _print_submenu_compact(
            _t("t_validacao"), _t("d_validacao"), fields, cfg,
            campos_avancados=campos_avancados, mostrar_avancado=mostrar_avancado)
        raw = _input(f"\n  {_t('opcao')}: ").upper()
        if raw in ("0", "Q"):
            break
        elif raw == "V":
            mostrar_avancado = not mostrar_avancado
        elif raw == "I":
            _toggle_idioma()
        elif raw == "G":
            _abrir_assistente(_t("t_validacao"), cfg)
        elif raw.isdigit() and 1 <= int(raw) <= len(fields_visiveis):
            _editar_campo(cfg, fields_visiveis[int(raw) - 1]); _pause()
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()


def _menu_advanced(cfg: Config) -> None:
    # benchmark_regressao adicionado 2026-08-06: existia no Config/
    # _CONFIG_SPEC/_HELP_DB, mas nunca tinha sido colocado em NENHUM menu.
    fields = ["benchmark", "benchmark_regressao", "monte_carlo", "n_monte_carlo",
              "monte_carlo_incluir_todos", "shap_benchmark", "shap_max_amostras"]
    while True:
        _cls(); _print_header(cfg)
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
def _menu_visualization(cfg: Config) -> None:
    # figuras_detalhadas adicionado 2026-08-06: mesma classe de bug de
    # n_jobs_permutacao -- existia no Config/_CONFIG_SPEC/_HELP_DB, mas nunca
    # tinha sido colocado em NENHUM menu.
    fields = ["figuras_detalhadas", "figuras_mostrar_marcadores",
              "figuras_mostrar_elipses", "formato_figura", "dpi",
              "abrir_figuras_na_tela"]
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
        for i, (pk, pd) in enumerate(_PALETAS_COR.items(), 1):
            nm = pd.get("nome", {}).get(_lang(), pk) if isinstance(pd.get("nome"), dict) else pk
            dsc = pd.get("desc", {}).get(_lang(), "") if isinstance(pd.get("desc"), dict) else ""
            mk = f"[{PA}]►[/{PA}]" if pk == atual else " "
            t.add_row(f"  [{PA}][{i}][/{PA}]", f"{mk} {escape(nm)}", escape(_trunc(dsc, 35)))
        console.print(Panel(t, title=f"[bold {PA}]{_t('viz_paleta')}[/bold {PA}]",
                            border_style=PA, box=rbox.ROUNDED, padding=(0, 1)))
        r = _input(f"  [1-{len(_PALETAS_COR)}] ou Enter: ")
        if r.isdigit():
            idx = int(r) - 1
            if 0 <= idx < len(_PALETAS_COR):
                vcfg["paleta"] = list(_PALETAS_COR.keys())[idx]
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
        _cls(); _print_header(cfg)
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
# So inclui chaves presentes em _TECNICAS; chaves ausentes sao ignoradas.
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
            if k in _TECNICAS and k not in vistos:
                ordem.append(k); vistos.add(k)
    # Acrescenta quaisquer tecnicas nao categorizadas, ao final
    for k in _TECNICAS:
        if k not in vistos:
            ordem.append(k); vistos.add(k)
    return ordem


def _tecnica_detalhe(tk: str, lang: str) -> None:
    """Painel com detalhes completos de uma tecnica."""
    td = _TECNICAS.get(tk, {})
    tdl = td.get(lang, td.get("PT", {}))
    linhas = [
        f"[{PW}]{escape(tdl.get('desc',''))}[/{PW}]", "",
        f"[{PA}]{'Faixa tipica' if lang=='PT' else 'Typical range'}:[/{PA}] "
        f"[{PW}]{escape(str(tdl.get('faixa','—')))}[/{PW}]",
        f"[{PA}]{'Pre-proc. recomendado' if lang=='PT' else 'Recommended preproc'}:[/{PA}] "
        f"[{PW}]{escape(str(tdl.get('preproc_rec', td.get('preproc','—'))))}[/{PW}]",
        f"[{PA}]{'Modo de entrada' if lang=='PT' else 'Input mode'}:[/{PA}] "
        f"[{PW}]{escape(str(td.get('mode','dx')))}[/{PW}]",
    ]
    console.print(Panel(
        Text.from_markup("\n".join(linhas)),
        title=f"[bold {PA}]{escape(tdl.get('nome', tk))}[/bold {PA}]",
        border_style=PF, box=rbox.ROUNDED, padding=(1, 2), width=_W()
    ))
    _pause()


def _menu_technique(cfg: Config) -> None:
    """Tecnica analitica — agrupada por categoria (modelo GUARACI)."""
    lang = _lang()

    def _aplicar(tk_sel: str) -> None:
        td_sel = _TECNICAS.get(tk_sel, {})
        tdl = td_sel.get(lang, td_sel.get("PT", {}))
        try:
            nm_sel = tdl.get("nome", tk_sel)
            _TECNICA_SELECIONADA["key"]  = tk_sel
            _TECNICA_SELECIONADA["nome"] = tk_sel.upper()
            fmin = td_sel.get("faixa_min"); fmax = td_sel.get("faixa_max")
            prep = td_sel.get("preproc", ""); mode = td_sel.get("mode", "dx")
            if fmin is not None: _set_val(cfg, "faixa_min_cm", str(fmin))
            if fmax is not None: _set_val(cfg, "faixa_max_cm", str(fmax))
            if prep: _set_val(cfg, "pre_processamento", prep)
            if mode: _set_val(cfg, "modo_entrada", mode)
            fa_str = tdl.get("faixa", f"{fmin}-{fmax}")
            console.print(f"  [g]✓ {escape(_trunc(nm_sel, 44))} {'selecionado' if lang=='PT' else 'selected'}.[/g]")
            console.print(f"  [info]  {'Faixa' if lang=='PT' else 'Range'}: {escape(_trunc(str(fa_str), 44))}[/info]")
            console.print(f"  [info]  Preproc.: {escape(str(prep))}  |  {'Modo' if lang=='PT' else 'Mode'}: {mode}[/info]")
        except ValueError as e:   # _set_val: faixa/preproc/mode invalido p/ a tecnica
            console.print(f"  [err]{escape(str(e))}[/err]")

    while True:
        _cls(); _print_header(cfg)
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
            keys_presentes = [k for k in keys if k in _TECNICAS]
            if not keys_presentes:
                continue
            cat = cat_pt if lang == "PT" else cat_en
            t.add_row("", Text.from_markup(f"[{PF}]── {escape(cat)} ──[/{PF}]"), "", "")
            for tk in keys_presentes:
                td = _TECNICAS.get(tk, {})
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
def _menu_encoding(cfg: Config) -> None:
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
            _USER_DIR.mkdir(parents=True, exist_ok=True)
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
        destino = str(_USER_DIR / "codigos_exportados.csv")
        try:
            _USER_DIR.mkdir(parents=True, exist_ok=True)
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
        _cls(); _print_header(cfg)

        # Painel explicativo — o que e, padrao de nome, como cadastrar/importar
        if lang == "PT":
            explicacao = (
                "[a]O que e:[/a] cada arquivo .dx comeca com um codigo de 2-4 letras\n"
                "que identifica a especie do oleo. A codificacao mapeia esse\n"
                "codigo para o nome legivel da especie usado nos resultados.\n\n"
                "[a]Padrao de nome dos arquivos:[/a]\n"
                "  COD-DD-MM-AAAA_Tn.dx            (especie pura)\n"
                "  COD-DD-MM-AAAA_AD-X-PP_Tn.dx    (adulterada)\n"
                "  Ex.: AND-10-06-2099_T1.dx  ->  Andiroba pura, triplicata 1\n\n"
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
                "  E.g.: AND-10-06-2099_T1.dx  ->  Andiroba pure, replicate 1\n\n"
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
def _menu_hardware(cfg: Optional[Config] = None) -> None:
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
            ("Modo entrada",  "[warn]Usar mode sintetico[/warn]" if lang=="PT" else "[warn]Use synthetic mode[/warn]"),
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
def _menu_prediction(cfg: Optional[Config] = None) -> None:
    """Predicao em lote via terminal: aplica um modelo .joblib salvo a um
    CSV de espectros novos (colunas=numero de onda, sem coluna de classe).

    Mesma logica cientifica do app web (aba Prediction) via predicao.py --
    zero duplicacao entre as duas interfaces (mesmo padrao da Fase H).
    Abre porta pra integracao com scripts/LIMS sem precisar de navegador.
    """
    lang = _lang()
    is_pt = lang == "PT"

    intro = (
        "Aplica um modelo treinado (.joblib) a espectros novos e reporta "
        "classe predita + diagnostico T2/Q (dominio de aplicabilidade)."
        if is_pt else
        "Applies a trained model (.joblib) to new spectra and reports "
        "predicted class + T2/Q diagnostics (applicability domain)."
    )

    # Redesenha cabecalho + painel de contexto -- extraido p/ funcao local
    # porque precisa ser chamado de novo apos fechar o assistente [G] (que
    # limpa a tela), mantendo o mesmo padrao visual usado em todo o resto
    # do CLI (rodape [G]/[0] logo abaixo do painel de intro).
    def _intro() -> None:
        _cls(); _print_header(cfg)
        console.print(Panel(
            Text.from_markup(f"  {intro}"),
            title=f"[bold {PS}]{_t('t_predicao')}[/bold {PS}]",
            border_style=PS, box=rbox.ROUNDED, padding=(1, 2),
        ))
        console.print(f"  [{PA}][G][/{PA}] Guaraci   [{PM}][0][/{PM}] {_t('voltar')}")
        console.print()

    _intro()

    lbl_modelo = "Caminho do modelo (.joblib)" if is_pt else "Model path (.joblib)"
    lbl_csv    = "Caminho do CSV de espectros novos" if is_pt else "New spectra CSV path"
    lbl_saida  = "Caminho de saida do CSV de resultados" if is_pt else "Output results CSV path"
    nao_encontrado = "Arquivo nao encontrado" if is_pt else "File not found"

    while True:
        cam_modelo = _ask(f"  [{PA}]{lbl_modelo}:[/{PA}] ").strip().strip('"')
        if cam_modelo.upper() == "G":
            _abrir_assistente(_t("t_predicao"), cfg)
            _intro(); continue
        if not cam_modelo or cam_modelo == "0":
            return
        break
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

    # Confirmacao antes de sobrescrever -- mesmo idioma s/n ja usado acima
    # para o aviso de pickle (nao introduz um mecanismo novo).
    if os.path.exists(cam_saida):
        conf_sobre = ("Arquivo ja existe. Sobrescrever? (s/n)" if is_pt
                      else "File already exists. Overwrite? (y/n)")
        if _ask(f"  [{PA}]{conf_sobre}[/{PA}] ").strip().lower() not in ("s", "y", "sim", "yes"):
            console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return

    try:
        import pandas as pd
        import guaraci.predicao as _pred
        status_msg = ("Carregando modelo e aplicando..." if is_pt
                       else "Loading model and applying...")
        with console.status(f"[{PA}]{status_msg}[/{PA}]"):
            # confiar=True: o operador ja confirmou explicitamente acima.
            pkg = _pred.load_model(cam_modelo, confiar=True)
            _pred.validate_model_package(pkg)
            X_new, wn_new, meta_df = _pred.load_prediction_csv(cam_csv)
            # Bloco 9b (D6): estende a predicao existente com o fluxo
            # completo Detectar -> Identificar -> Quantificar quando o
            # pacote traz o ensemble de identificacao (modelos exportados
            # antes do Bloco 9b nao tem essa chave -- cai no caminho
            # anterior, sem quebrar).
            if pkg.get("identification_ensemble"):
                df_res, resultados_cego = _pred.predict_blind(pkg, X_new, wn_new)
                df_res["detectado_puro_especie"] = [
                    r.pureza.aceito for r in resultados_cego]
                df_res["pureza_confiavel"] = [
                    r.pureza.confiavel for r in resultados_cego]
                df_res["classe_identificada"] = [
                    r.identificacao.classe_identificada for r in resultados_cego]
                df_res["identificacao_cobertura"] = [
                    (r.identificacao.cobertura_status.value
                     if r.identificacao.cobertura_status else None)
                    for r in resultados_cego]
                df_res["identificacao_alpha_alcancavel"] = [
                    r.identificacao.alpha_alcancavel for r in resultados_cego]
                df_res["identificacao_candidatos"] = [
                    ", ".join(r.identificacao.candidatos_ambiguos)
                    for r in resultados_cego]
                df_res["teor_estimado"] = [
                    r.quantificacao.teor_estimado for r in resultados_cego]
                df_res["quantificacao_motivo_bloqueio"] = [
                    r.quantificacao.motivo_bloqueio for r in resultados_cego]
                df_res["alpha_total"] = [
                    r.alpha_total for r in resultados_cego]
            else:
                df_res = _pred.predict_samples(pkg, X_new, wn_new)
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
        f"[{PG}]{n_ac}[/{PG}] amostras dentro do ajuste do modelo PLS-DA "
        f"(T2 <= limite e resíduo Q <= limite -- ver 'criterio' no CSV) / "
        f"[{PR}]{n_tot - n_ac}[/{PR}] fora do ajuste (espectro atipico, "
        f"tratar com cautela)"
        if is_pt else
        f"  [{PG}]✔ {n_tot} samples processed[/{PG}]  |  "
        f"[{PG}]{n_ac}[/{PG}] samples within the PLS-DA model fit "
        f"(Hotelling T2 <= limit and Q-residual <= limit -- see 'criterio' "
        f"in the CSV) / "
        f"[{PR}]{n_tot - n_ac}[/{PR}] outside the fit (atypical spectrum, "
        f"treat with caution)"
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
            "(espectro parecido/nao-parecido com o que o modelo viu no "
            "treino -- 'fora' NAO significa 'adulterado', so' 'diferente do "
            "que foi calibrado')"
            if is_pt else
            f"  [{PG}]🔎 Applicability domain:[/{PG}] "
            f"[{PG}]{n_ad_dentro}[/{PG}] within / "
            f"[{PR}]{n_tot - n_ad_dentro}[/{PR}] outside "
            "(similar/dissimilar to what the model saw during training -- "
            "'outside' does NOT mean 'adulterated', only 'different from "
            "what was calibrated')"
        )
        resumo_txt += "\n" + ad_txt

        # Bloco 13b: sentinela de deriva -- persistida ao lado do MODELO
        # (nao do CSV de saida), porque e' o modelo que fica fixo entre
        # varias rodadas de predicao ao longo do tempo -- exatamente o
        # uso continuo que a sentinela existe para acompanhar. Depende de
        # AD_dentro_dominio (nao de identification_ensemble), por isso fica
        # sob este if, nao sob o do fluxo cego abaixo.
        try:
            import guaraci.sentinela_deriva as _sent
            cam_sentinela = cam_modelo + ".sentinela.json"
            estado_sent = (_sent.load_state(cam_sentinela)
                           if os.path.isfile(cam_sentinela)
                           else _sent.SentinelState(alpha_nominal=0.05))
            _sent.update_with_predictions(estado_sent, df_res)
            _sent.save_state(estado_sent, cam_sentinela)
            alerta_sent = _sent.check_drift(estado_sent)
            cor_sent = PR if alerta_sent.alerta else PM
            sent_txt = (
                f"  [{cor_sent}]🛰 Sentinela de deriva (n={alerta_sent.n}):"
                f"[/{cor_sent}] {escape(alerta_sent.mensagem)}"
            )
            resumo_txt += "\n" + sent_txt
        except Exception as _e_sent:  # noqa: BLE001 -- diagnostico
            # opcional; erro impresso, nao afeta a predicao ja gravada.
            resumo_txt += (
                f"\n  [{PM}]{'Sentinela de deriva indisponivel' if is_pt else 'Drift sentinel unavailable'}"
                f": {escape(str(_e_sent))}[/{PM}]")

    # Bloco 9b: fluxo cego completo (Detectar -> Identificar -> Quantificar)
    # -- so' presente quando o pacote .joblib traz o ensemble de
    # identificacao. Sem este bloco a saida seria so' uma tabela de colunas
    # tecnicas (identificacao_cobertura, motivo_bloqueio, alpha_total) sem
    # explicacao nenhuma -- o motivo original desta auditoria de veracidade
    # (Agente 3, Passo 92).
    if "classe_identificada" in df_res.columns:
        n_pura = int(sum(1 for r in resultados_cego if r.pureza.aceito is True))
        n_adulterada = int(sum(1 for r in resultados_cego if r.pureza.aceito is False))
        n_pureza_indet = n_tot - n_pura - n_adulterada
        n_identificado = int(sum(
            1 for r in resultados_cego
            if r.identificacao.classe_identificada is not None))
        n_desconhecido = n_tot - n_identificado
        n_quantificado = int(sum(
            1 for r in resultados_cego if r.quantificacao.teor_estimado is not None))
        n_bloqueado = n_tot - n_quantificado

        if is_pt:
            fluxo_txt = (
                "\n  [bold]Fluxo cego -- Detectar → Identificar → Quantificar "
                "(Bloco 9b):[/bold]\n"
                f"  🧪 Pureza (DD-SIMCA da espécie prevista): "
                f"[{PG}]{n_pura}[/{PG}] detectada como pura / "
                f"[{PR}]{n_adulterada}[/{PR}] detectada como adulterada"
                + (f" / [{PM}]{n_pureza_indet}[/{PM}] indeterminada "
                   "(sem modelo de pureza calibrado para a espécie prevista)"
                   if n_pureza_indet else "") + "\n"
                f"  🏷 Adulterante: "
                f"[{PG}]{n_identificado}[/{PG}] identificado / "
                f"[{PR}]{n_desconhecido}[/{PR}] DESCONHECIDO (nenhuma "
                "combinação espécie×adulterante teve garantia estatística "
                "suficiente E exclusiva para rotular esta amostra -- por "
                "falta de garantia, ou por 2+ combinações validadas "
                "empatando, o que também bloqueia o rótulo)\n"
                f"  ⚖ Quantificação (teor de adulterante estimado -- mesma "
                "unidade da coluna de referência usada no treino do modelo, "
                f"tipicamente %m/m): [{PG}]{n_quantificado}[/{PG}] com "
                f"número / [{PR}]{n_bloqueado}[/{PR}] BLOQUEADA "
                "(quantificação recusada por não haver identificação "
                "confiável do adulterante -- ver coluna "
                "'quantificacao_motivo_bloqueio' no CSV)\n"
                f"  [{PM}]⚠ Um rótulo em 'classe_identificada' só existe "
                "quando 'identificacao_cobertura'='validado' (garantia "
                "estatística formal, calibrada com >=2 sessões de coleta "
                "independentes) -- nunca há rótulo 'informativo' sem essa "
                "garantia: nesse caso a amostra é DESCONHECIDA de propósito, "
                "e a coluna 'identificacao_candidatos' traz só o palpite "
                "mais próximo (SEM garantia nenhuma) para referência, nunca "
                "como resultado a usar numa decisão de controle de "
                f"qualidade sem confirmar por método de referência.[/{PM}]"
            )
        else:
            fluxo_txt = (
                "\n  [bold]Blind flow -- Detect → Identify → Quantify "
                "(Bloco 9b):[/bold]\n"
                f"  🧪 Purity (DD-SIMCA for the predicted species): "
                f"[{PG}]{n_pura}[/{PG}] detected as pure / "
                f"[{PR}]{n_adulterada}[/{PR}] detected as adulterated"
                + (f" / [{PM}]{n_pureza_indet}[/{PM}] undetermined "
                   "(no purity model calibrated for the predicted species)"
                   if n_pureza_indet else "") + "\n"
                f"  🏷 Adulterant: "
                f"[{PG}]{n_identificado}[/{PG}] identified / "
                f"[{PR}]{n_desconhecido}[/{PR}] UNKNOWN (no species x "
                "adulterant combination had a statistical guarantee that "
                "was both sufficient AND exclusive for this sample -- "
                "either no guarantee, or 2+ validated combinations tied, "
                "which also blocks the label)\n"
                f"  ⚖ Quantification (estimated adulterant content -- same "
                "unit as the reference column used to train the model, "
                f"typically %w/w): [{PG}]{n_quantificado}[/{PG}] with a "
                f"number / [{PR}]{n_bloqueado}[/{PR}] BLOCKED "
                "(quantification refused because the adulterant was not "
                "reliably identified -- see the 'quantificacao_motivo_"
                "bloqueio' column in the CSV)\n"
                f"  [{PM}]⚠ A 'classe_identificada' label only ever exists "
                "when 'identificacao_cobertura'='validado' (formal "
                "statistical guarantee, calibrated with >=2 independent "
                "collection sessions) -- there is no 'informational' label "
                "without that guarantee: in that case the sample is "
                "UNKNOWN on purpose, and the 'identificacao_candidatos' "
                "column carries only the closest guess (with NO guarantee "
                "at all) for reference, never as a result to act on for a "
                "quality decision without confirming by a reference "
                f"method.[/{PM}]"
            )
        resumo_txt += fluxo_txt
    console.print()
    console.print(Panel(
        Group(Text.from_markup(resumo_txt), Text(""), t_res),
        title=f"[bold {PG}]{'Resultado' if is_pt else 'Result'}[/bold {PG}]",
        border_style=PG, box=rbox.ROUNDED, padding=(1, 2),
    ))
    console.print(f"  [{PM}]{'Salvo em' if is_pt else 'Saved to'}:[/{PM}] "
                  f"{escape(cam_saida)}")
    _pause()


# Aviso de maturidade da tela HSI (Passo 103 da
# INSTRUCAO_HSI_ROBUSTEZ_E_VALIDACAO.md) -- fonte UNICA (nao frase solta
# duplicada em varios lugares, mesmo padrao de _AVISO_PROTOTIPO_TITULO/
# _AVISO_PROTOTIPO_CORPO em reports.py), descrevendo a limitacao REAL e
# especifica, nao um rotulo generico "prototipo". Escolhido em vez do
# carimbo formal "PROTOTYPE OUTPUT" (reports.py) porque aquele carimbo
# tem um criterio objetivo DIFERENTE (ausencia de garantia de
# agrupamento anti-vazamento) que NAO se aplica aqui -- o HSI TEM
# garantia de agrupamento real (group_id por objeto fisico, Passo 97,
# validada por teste de propriedade Hypothesis). A limitacao real do
# HSI e' outra: cobertura de validacao ainda pequena. Atualizar este
# texto sempre que a cobertura de validacao mudar (ver Passo 104).
_AVISO_MATURIDADE_HSI_PT = (
    "Validado em 1 fruta (Kaki) e 1 camera (VIS) do dataset publico "
    "DeepHS Fruit, com desbalanceamento de classe severo e nao corrigido "
    "(overripe n=12, unripe n=2) -- ver docs/VALIDACAO_PUBLICA.md secao 7 "
    "para os numeros completos antes de usar para resultado publicavel."
)
_AVISO_MATURIDADE_HSI_EN = (
    "Validated on 1 fruit (Kaki) and 1 camera (VIS) from the public "
    "DeepHS Fruit dataset, with severe and uncorrected class imbalance "
    "(overripe n=12, unripe n=2) -- see docs/VALIDACAO_PUBLICA.md "
    "section 7 for the full numbers before using for a publishable result."
)


def _menu_hsi(cfg: Optional[Config] = None) -> None:
    """Imageamento hiperespectral (HSI, mode='hsi', prototipo "minimo
    viavel" -- Passos 92-102 da INSTRUCAO_HSI_MINIMO_VIAVEL.md).

    DISTINTO do mode "imagem" (colorimetria de foto comum, tecla [K] da
    lista de fields de _menu_data): HSI opera POR PIXEL de um cubo
    hiperespectral (ENVI .hdr+.bin), com quality gate, segmentacao,
    classificacao por pixel + agregacao por objeto, explicabilidade
    cruzada com banda quimica e validacao externa por dia de medicao --
    fluxo orquestrado por `hsi_pipeline.run_hsi_pipeline`, nao pelo
    `pipeline.executar()` usado pelos outros modes (forma de dado
    diferente, ver docstring de hsi_pipeline.py).
    """
    if cfg is None:
        cfg = Config()
    # Setado JA' aqui (nao so' apos validar a pasta) -- e' o que faz
    # _print_header/_print_status mostrarem "Tecnica: HSI" em vez do
    # default global errado assim que a tela abre, nao so' depois de
    # rodar o pipeline (achado do Passo 103).
    cfg.mode = "hsi"
    lang = _lang()
    is_pt = lang == "PT"

    intro = (
        "Roda o pipeline HSI (leitura -> quality gate -> segmentacao -> "
        "classificacao por pixel -> mapa espacial -> validacao) sobre "
        "seus proprios cubos hiperespectrais (ENVI .hdr/.bin, 1 subpasta "
        "por classe). Datasets publicos (DeepHS Fruit) sao usados so' "
        "para os testes de validacao do projeto, nao sao necessarios "
        "para uso normal. "
        + _AVISO_MATURIDADE_HSI_PT
        if is_pt else
        "Runs the HSI pipeline (reading -> quality gate -> segmentation -> "
        "per-pixel classification -> spatial map -> validation) over "
        "your own hyperspectral cubes (ENVI .hdr/.bin, 1 subfolder per "
        "class). Public datasets (DeepHS Fruit) are used only for the "
        "project's own validation tests, not required for normal use. "
        + _AVISO_MATURIDADE_HSI_EN
    )

    _cls(); _print_header(cfg)
    console.print(Panel(
        Text.from_markup(f"  {intro}"),
        title=f"[bold {PS}]{_t('t_hsi')}[/bold {PS}]",
        border_style=PS, box=rbox.ROUNDED, padding=(1, 2),
    ))
    console.print(f"  [{PM}][0][/{PM}] {_t('voltar')}")
    console.print()

    lbl_pasta = ("Pasta com seus cubos hiperespectrais (ou dataset publico "
                "de validacao)" if is_pt else
                "Folder with your hyperspectral cubes (or public "
                "validation dataset)")
    default_pasta = getattr(cfg, "hsi_dataset_folder", "") or ""
    sufixo = f" [{default_pasta}]" if default_pasta else ""
    pasta = _ask(f"  [{PA}]{lbl_pasta}{sufixo}:[/{PA}] ").strip().strip('"')
    if pasta == "0":
        return
    if not pasta:
        pasta = default_pasta
    if not pasta or not os.path.isdir(pasta):
        msg = (f"Pasta invalida: {pasta}" if is_pt
              else f"Invalid folder: {pasta}")
        console.print(f"  [{PR}]{msg}[/{PR}]")
        _pause(); return

    cfg.hsi_dataset_folder = pasta

    from guaraci.hsi_pipeline import run_hsi_pipeline
    console.print(f"  [{PM}]{'Rodando pipeline HSI...' if is_pt else 'Running HSI pipeline...'}[/{PM}]")
    try:
        resumo = run_hsi_pipeline(cfg)
    except Exception as e:  # noqa: BLE001 -- reporta erro completo, nunca engole
        console.print(f"  [{PR}]{'Erro' if is_pt else 'Error'}: {e}[/{PR}]")
        _pause(); return

    linhas = [
        f"  {'Gravacoes aceitas' if is_pt else 'Accepted recordings'}: "
        f"{resumo['n_gravacoes_aceitas']}/{resumo['n_gravacoes_total']} "
        f"({'rejeitadas pelo quality gate' if is_pt else 'rejected by quality gate'}: "
        f"{resumo['n_gravacoes_rejeitadas']})",
        f"  {'Variaveis latentes (Wold)' if is_pt else 'Latent variables (Wold)'}: "
        f"{resumo['n_components']}",
    ]
    # Passo 111: dataset generico (sem manifest.json) so' tem validacao
    # INTERNA (n_objetos_teste_externo=0, dicts *_externa vazios -- ver
    # hsi_validation.run_internal_validation_group_aware) -- declarado
    # aqui explicitamente, nunca escondido atras de um "0" sem explicacao.
    val = resumo["validacao_externa"]
    tem_externa = val.n_objetos_teste_externo > 0
    if tem_externa:
        linhas.append(
            f"  {'Validacao externa' if is_pt else 'External validation'}: "
            f"n_interno={val.n_objetos_teste_interno}, "
            f"n_externo={val.n_objetos_teste_externo}")
        for classe in val.classes:
            linhas.append(
                f"    {classe}: sens(int/ext)="
                f"{val.sensibilidade_interna[classe]:.2f}/"
                f"{val.sensibilidade_externa[classe]:.2f}")
    else:
        rotulo_val = ("Validacao (so interna -- sem particao externa "
                      "neste dataset)" if is_pt else
                      "Validation (internal only -- no external "
                      "partition for this dataset)")
        linhas.append(
            f"  {rotulo_val}: n_interno={val.n_objetos_teste_interno}")
        for classe in val.classes:
            linhas.append(
                f"    {classe}: sens(int)="
                f"{val.sensibilidade_interna[classe]:.2f}")

    # Confianca por objeto (Passo 107): heterogeneidade de pixel deixa de
    # ser so' um numero interno -- resumo por faixa + objetos de baixa
    # concordancia listados explicitamente (nunca escondidos).
    conf = resumo.get("confianca_por_objeto", {})
    if conf:
        baixa = [(gid, r) for gid, r in conf.items()
                if r.heterogeneidade > 0.30]
        linhas.append("")
        linhas.append(
            f"  {'Confianca por objeto' if is_pt else 'Per-object confidence'}: "
            f"{len(conf) - len(baixa)}/{len(conf)} "
            f"{'com concordancia alta/moderada' if is_pt else 'high/moderate agreement'}")
        if baixa:
            linhas.append(
                f"    [{PR}]{'baixa concordancia' if is_pt else 'low agreement'} "
                f"({len(baixa)}): " +
                ", ".join(f"{gid} ({r.heterogeneidade:.0%})" for gid, r in baixa[:5]) +
                ("..." if len(baixa) > 5 else "") + f"[/{PR}]")

    console.print(Panel(
        "\n".join(linhas),
        title=f"[bold {PG}]{'Resultado' if is_pt else 'Result'}[/bold {PG}]",
        border_style=PG, box=rbox.ROUNDED, padding=(1, 2),
    ))
    console.print(f"  [{PM}]{'Salvo em' if is_pt else 'Saved to'}:[/{PM}] "
                  f"{escape(cfg.output_folder)}")
    _pause()


def _menu_plan(cfg: Optional[Config] = None) -> None:
    """Planejamento de coleta (Bloco 10): quantas amostras por classe
    (`plano_amostral.py`) + como distribui-las entre sessoes e em que
    ordem le-las (`plano_coleta.py`), evitando os dois confundimentos ja
    documentados no projeto (classe x sessao, ordem de leitura x teor).
    """
    lang = _lang()
    is_pt = lang == "PT"

    intro = (
        "Calcula quantas amostras coletar (gate conformal ou DD-SIMCA) e "
        "gera um plano de sessoes + ordem de leitura aleatorizada, com "
        "alertas de replica/branco."
        if is_pt else
        "Calculates how many samples to collect (conformal or DD-SIMCA "
        "gate) and generates a session plan + randomized reading order, "
        "with replicate/blank alerts."
    )

    # Ver _menu_prediction: mesmo padrao de redesenho apos fechar [G].
    def _intro() -> None:
        _cls(); _print_header(cfg)
        console.print(Panel(
            Text.from_markup(f"  {intro}"),
            title=f"[bold {PS}]{_t('t_planejamento')}[/bold {PS}]",
            border_style=PS, box=rbox.ROUNDED, padding=(1, 2),
        ))
        console.print(f"  [{PA}][G][/{PA}] Guaraci   [{PM}][0][/{PM}] {_t('voltar')}")
        console.print()

    _intro()

    lbl_classes = "Classes (separadas por virgula)" if is_pt else "Classes (comma-separated)"
    while True:
        classes_raw = _ask(f"  [{PA}]{lbl_classes}:[/{PA}] ").strip()
        if classes_raw.upper() == "G":
            _abrir_assistente(_t("t_planejamento"), cfg)
            _intro(); continue
        break
    if not classes_raw or classes_raw == "0":
        return
    classes = [c.strip() for c in classes_raw.split(",") if c.strip()]
    if not classes:
        return

    lbl_sessoes = "Numero de sessoes de coleta" if is_pt else "Number of collection sessions"
    sessoes_raw = _ask(f"  [{PA}]{lbl_sessoes}:[/{PA}] [{PM}](Enter = 2)[/{PM}] ")
    try:
        n_sessoes = int(sessoes_raw) if sessoes_raw else 2
    except ValueError:
        console.print(f"  [{PR}]{'Numero invalido' if is_pt else 'Invalid number'}[/{PR}]")
        _pause(); return
    if n_sessoes < 1:
        console.print(f"  [{PR}]{'Precisa de pelo menos 1 sessao' if is_pt else 'Needs at least 1 session'}[/{PR}]")
        _pause(); return

    lbl_alvo = (
        "Alvo: (C)onformal (Identificar/agrupado) ou (D)D-SIMCA (pureza por especie)?"
        if is_pt else
        "Target: (C)onformal (Identify/pooled) or (D)D-SIMCA (per-species purity)?"
    )
    alvo = _ask(f"  [{PA}]{lbl_alvo}[/{PA}] ").strip().upper()
    if alvo not in ("C", "D"):
        console.print(f"  [{PR}]{'Opcao invalida' if is_pt else 'Invalid option'}[/{PR}]")
        _pause(); return

    if alvo == "C":
        lbl_valor = "Alpha desejado (ex.: 0.05)" if is_pt else "Desired alpha (e.g., 0.05)"
    else:
        lbl_valor = "Cobertura-alvo (ex.: 0.90)" if is_pt else "Target coverage (e.g., 0.90)"
    valor_raw = _ask(f"  [{PA}]{lbl_valor}:[/{PA}] ")
    try:
        valor = float(valor_raw)
    except ValueError:
        console.print(f"  [{PR}]{'Numero invalido' if is_pt else 'Invalid number'}[/{PR}]")
        _pause(); return

    try:
        import guaraci.plano_coleta as _plano
        if alvo == "C":
            plano, meta = _plano.plan_from_statistical_target(
                classes, n_sessoes, alpha_conformal=valor)
        else:
            plano, meta = _plano.plan_from_statistical_target(
                classes, n_sessoes, cobertura_ddsimca=valor)
    except ValueError as e:
        console.print(f"  [{PR}]{'Erro' if is_pt else 'Error'}: {escape(str(e))}[/{PR}]")
        _pause(); return

    padrao_saida = str(Path.cwd() / "plano_coleta")
    lbl_saida = "Prefixo dos arquivos de saida" if is_pt else "Output file prefix"
    cam_saida = _ask(
        f"  [{PA}]{lbl_saida}[/{PA}] [{PM}](Enter = {escape(padrao_saida)})[/{PM}]: "
    ).strip().strip('"')
    if not cam_saida:
        cam_saida = padrao_saida

    lbl_pdf = ("Gerar tambem PDF (opcional, alem de Markdown+Excel)? (s/N)"
               if is_pt else
               "Also generate PDF (optional, in addition to Markdown+Excel)? (y/N)")
    quer_pdf = _ask(f"  [{PA}]{lbl_pdf}[/{PA}] ").strip().lower() in ("s", "y", "sim", "yes")

    # Confirmacao antes de sobrescrever -- mesmo idioma s/n de _menu_prediction.
    _existentes = [p for p in (cam_saida + ".md", cam_saida + ".xlsx",
                                cam_saida + ".pdf" if quer_pdf else None)
                   if p and os.path.exists(p)]
    if _existentes:
        conf_sobre = ("Arquivo(s) ja existem. Sobrescrever? (s/n)" if is_pt
                      else "File(s) already exist. Overwrite? (y/n)")
        if _ask(f"  [{PA}]{conf_sobre}[/{PA}] ").strip().lower() not in ("s", "y", "sim", "yes"):
            console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return

    try:
        md = _plano.export_markdown(plano)
        cam_md = cam_saida + ".md"
        with open(cam_md, "w", encoding="utf-8") as f:
            f.write(md)
        cam_xlsx = cam_saida + ".xlsx"
        _plano.export_excel(plano, cam_xlsx)
        cam_pdf = None
        if quer_pdf:
            cam_pdf = cam_saida + ".pdf"
            _plano.export_pdf(plano, cam_pdf)
    except OSError as e:
        console.print(f"  [{PR}]{'Erro ao salvar' if is_pt else 'Error saving'}: {escape(str(e))}[/{PR}]")
        _pause(); return

    resumo_txt = (
        f"  [{PG}]✔ n por classe:[/{PG}] [{PG}]{meta['n_por_classe']}[/{PG}]  |  "
        f"[{PG}]origem:[/{PG}] {escape(meta['origem'])}  |  "
        f"[{PG}]total de amostras:[/{PG}] {len(plano.itens)}"
    )
    console.print()
    console.print(Panel(
        Text.from_markup(resumo_txt),
        title=f"[bold {PG}]{'Plano gerado' if is_pt else 'Plan generated'}[/bold {PG}]",
        border_style=PG, box=rbox.ROUNDED, padding=(1, 2),
    ))
    console.print(f"  [{PM}]{'Alertas' if is_pt else 'Alerts'}:[/{PM}]")
    for a in plano.alertas:
        console.print(f"    [{PA}]•[/{PA}] {escape(a)}")
    console.print()
    caminhos_salvos = [cam_md, cam_xlsx] + ([cam_pdf] if cam_pdf else [])
    console.print(f"  [{PM}]{'Salvo em' if is_pt else 'Saved to'}:[/{PM}] "
                  f"{', '.join(escape(c) for c in caminhos_salvos)}")
    _pause()


def _menu_selecao_amostras(cfg: Optional[Config] = None) -> None:
    """Selecao de amostras de calibracao/validacao (Bloco 10, Passo 87):
    dado um CSV com espectros JA medidos (e opcionalmente uma coluna de
    referencia/teor), escolhe QUAIS amostras vao para calibracao via
    Kennard-Stone, Duplex ou SPXY (`dados_io.py`) -- mesmo fluxo de
    planejamento experimental de `_menu_plan` (que decide QUANTAS
    coletar), agora atuando sobre dados que ja existem.
    """
    lang = _lang(); is_pt = lang == "PT"

    intro = (
        "Escolhe quais amostras de um CSV vao para calibracao (cobertura "
        "representativa do espaco espectral e/ou do teor) via Kennard-"
        "Stone, Duplex ou SPXY -- so' separa/marca, nao altera o CSV "
        "original."
        if is_pt else
        "Chooses which samples from a CSV go into the calibration set "
        "(representative coverage of spectral space and/or target range) "
        "via Kennard-Stone, Duplex or SPXY -- only splits/labels, never "
        "alters the original CSV."
    )

    # Ver _menu_prediction: mesmo padrao de redesenho apos fechar [G].
    def _intro() -> None:
        _cls(); _print_header(cfg)
        console.print(Panel(
            Text.from_markup(f"  {intro}"),
            title=f"[bold {PS}]{_t('t_selecao_amostras')}[/bold {PS}]",
            border_style=PS, box=rbox.ROUNDED, padding=(1, 2),
        ))
        console.print(f"  [{PA}][G][/{PA}] Guaraci   [{PM}][0][/{PM}] {_t('voltar')}")
        console.print()

    _intro()

    lbl_csv = ("Caminho do CSV com os espectros (1 amostra por linha)"
               if is_pt else
               "Path to the CSV with spectra (1 sample per row)")
    while True:
        caminho_csv = _ask(f"  [{PA}]{lbl_csv}:[/{PA}] ").strip().strip('"')
        if caminho_csv.upper() == "G":
            _abrir_assistente(_t("t_selecao_amostras"), cfg)
            _intro(); continue
        break
    if caminho_csv == "0":
        return
    if not caminho_csv or not os.path.isfile(caminho_csv):
        console.print(f"  [{PR}]{'Arquivo nao encontrado' if is_pt else 'File not found'}[/{PR}]")
        _pause(); return

    import numpy as _np
    import pandas as _pd
    try:
        df = _pd.read_csv(caminho_csv)
    except (OSError, ValueError, UnicodeDecodeError) as e:
        console.print(f"  [{PR}]{'Erro ao ler CSV' if is_pt else 'Error reading CSV'}: {escape(str(e))}[/{PR}]")
        _pause(); return

    lbl_alvo = ("Coluna de referencia/teor (Enter = nenhuma -- so' habilita "
                "Kennard-Stone/Duplex, nao SPXY)"
                if is_pt else
                "Reference/target column (Enter = none -- only enables "
                "Kennard-Stone/Duplex, not SPXY)")
    col_alvo = _ask(f"  [{PA}]{lbl_alvo}:[/{PA}] ").strip()
    if col_alvo and col_alvo not in df.columns:
        console.print(f"  [{PR}]{'Coluna nao encontrada' if is_pt else 'Column not found'}: {escape(col_alvo)}[/{PR}]")
        _pause(); return

    colunas_x = [c for c in df.columns if c != col_alvo]
    df_x = df[colunas_x].select_dtypes(include=[_np.number])
    if df_x.shape[1] == 0 or df_x.shape[0] < 2:
        console.print(f"  [{PR}]{'CSV sem colunas numericas suficientes (pelo menos 2 amostras, 1 variavel)' if is_pt else 'CSV without enough numeric columns (at least 2 samples, 1 variable)'}[/{PR}]")
        _pause(); return
    X = df_x.to_numpy(dtype=float)

    metodos = ["Kennard-Stone", "Duplex"] + (["SPXY"] if col_alvo else [])
    lbl_metodo = "Metodo" if is_pt else "Method"
    console.print(f"  [{PA}]{lbl_metodo}:[/{PA}]")
    for i, m in enumerate(metodos, 1):
        console.print(f"    ({i}) {m}")
    escolha_m = _ask("  > ").strip()
    if not escolha_m.isdigit() or not (1 <= int(escolha_m) <= len(metodos)):
        console.print(f"  [{PR}]{'Opcao invalida' if is_pt else 'Invalid option'}[/{PR}]")
        _pause(); return
    metodo = metodos[int(escolha_m) - 1]

    lbl_frac = "Fracao para calibracao (Enter = 0.7)" if is_pt else "Calibration fraction (Enter = 0.7)"
    frac_raw = _ask(f"  [{PA}]{lbl_frac}:[/{PA}] ").strip()
    try:
        frac_cal = float(frac_raw) if frac_raw else 0.7
    except ValueError:
        console.print(f"  [{PR}]{'Numero invalido' if is_pt else 'Invalid number'}[/{PR}]")
        _pause(); return
    if not (0.0 < frac_cal < 1.0):
        console.print(f"  [{PR}]{'Fracao precisa estar entre 0 e 1' if is_pt else 'Fraction must be between 0 and 1'}[/{PR}]")
        _pause(); return

    from guaraci.dados_io import duplex_split, kennard_stone_split, spxy_split
    # console.status: Kennard-Stone/Duplex/SPXY sao O(n^2) em distancias --
    # sem isso a tela ficava parada sem nenhum sinal para datasets maiores
    # (mesmo padrao ja usado em _menu_prediction para o carregamento do modelo).
    status_msg = "Calculando particao..." if is_pt else "Computing split..."
    with console.status(f"[{PA}]{status_msg}[/{PA}]"):
        if metodo == "Kennard-Stone":
            idx_cal, idx_val = kennard_stone_split(X, frac_treino=frac_cal)
        elif metodo == "Duplex":
            idx_cal, idx_val = duplex_split(X, frac_treino=frac_cal)
        else:
            y = df[col_alvo].to_numpy(dtype=float)
            idx_cal, idx_val = spxy_split(X, y, frac_treino=frac_cal)

    df_saida = df.copy()
    col_conjunto = "conjunto" if is_pt else "set"
    df_saida[col_conjunto] = ""
    df_saida.iloc[idx_cal, df_saida.columns.get_loc(col_conjunto)] = (
        "calibracao" if is_pt else "calibration")
    df_saida.iloc[idx_val, df_saida.columns.get_loc(col_conjunto)] = (
        "validacao" if is_pt else "validation")

    padrao_saida = str(Path.cwd() / "selecao_amostras.csv")
    lbl_saida = "Arquivo de saida" if is_pt else "Output file"
    cam_saida = _ask(
        f"  [{PA}]{lbl_saida}[/{PA}] [{PM}](Enter = {escape(padrao_saida)})[/{PM}]: "
    ).strip().strip('"')
    if not cam_saida:
        cam_saida = padrao_saida

    # Confirmacao antes de sobrescrever -- mesmo idioma s/n de _menu_prediction.
    if os.path.exists(cam_saida):
        conf_sobre = ("Arquivo ja existe. Sobrescrever? (s/n)" if is_pt
                      else "File already exists. Overwrite? (y/n)")
        if _ask(f"  [{PA}]{conf_sobre}[/{PA}] ").strip().lower() not in ("s", "y", "sim", "yes"):
            console.print(f"  [{PM}]{_t('cancelado')}[/{PM}]"); _pause(); return

    try:
        df_saida.to_csv(cam_saida, index=False)
    except OSError as e:
        console.print(f"  [{PR}]{'Erro ao salvar' if is_pt else 'Error saving'}: {escape(str(e))}[/{PR}]")
        _pause(); return

    resumo_txt = (
        f"  [{PG}]✔ {escape(metodo)}:[/{PG}] {len(idx_cal)} "
        f"{'calibracao' if is_pt else 'calibration'} / {len(idx_val)} "
        f"{'validacao' if is_pt else 'validation'} "
        f"({'de' if is_pt else 'of'} {len(df)})"
    )
    console.print()
    console.print(Panel(
        Text.from_markup(resumo_txt),
        title=f"[bold {PG}]{'Selecao gerada' if is_pt else 'Selection generated'}[/bold {PG}]",
        border_style=PG, box=rbox.ROUNDED, padding=(1, 2),
    ))
    console.print(f"  [{PM}]{'Salvo em' if is_pt else 'Saved to'}:[/{PM}] {escape(cam_saida)}")
    _pause()


# ---------------------------------------------------------------------------
# AUDITORIA DE DELINEAMENTO — comando dedicado (Bloco 11)
# ---------------------------------------------------------------------------
def _menu_audit(cfg: Optional[Config] = None) -> None:
    """Auditoria de delineamento (Bloco 11) isolada -- roda
    `auditoria_delineamento.run_audit` sobre o dataset configurado (as
    MESMAS checagens que ja rodam automaticamente em toda execucao e
    aparecem no model card), sem exigir rodar classificacao/quantificacao
    inteira. Reaproveita `load_data`/`validate_input` -- mesmo caminho de
    dados que `pipeline.executar()` usa antes de chamar `run_audit`, nao
    duplica logica."""
    cfg = cfg or Config()
    lang = _lang()
    is_pt = lang == "PT"
    _cls(); _print_header(cfg)

    intro = (
        "Roda so' a auditoria de delineamento (agrupamento, confundimento "
        "classe x sessao, duplicatas, N insuficiente, faixa de validacao) "
        "sobre o dataset configurado em [2] Dados -- sem rodar o pipeline "
        "de classificacao/quantificacao inteiro."
        if is_pt else
        "Runs only the design audit (grouping, class x session "
        "confounding, duplicates, insufficient N, validation range) over "
        "the dataset configured in [2] Data -- without running the full "
        "classification/quantification pipeline."
    )
    console.print(Panel(
        Text.from_markup(f"  {intro}"),
        title=f"[bold {PS}]{_t('t_auditoria')}[/bold {PS}]",
        border_style=PS, box=rbox.ROUNDED, padding=(1, 2),
    ))
    console.print()

    from guaraci.config_io import _validar_pasta_dados
    ok, msg = _validar_pasta_dados(cfg)
    if not ok:
        console.print(f"  [{PR}]{escape(msg)}[/{PR}]")
        console.print(f"  [{PM}]{'Configure a fonte de dados em' if is_pt else 'Configure the data source in'} "
                      f"[{PA}][2] {_t('t_dados')}[/{PA}].[/{PM}]")
        _pause(); return
    console.print(f"  [{PM}]{msg}[/{PM}]")

    # Gate antes de carregar dados + rodar a auditoria (pode demorar em
    # datasets grandes): unica forma desta tela de oferecer [G] Guaraci e
    # [0] Voltar sem sair -- ate aqui a tela so' rodava direto, sem nenhum
    # ponto de escape ou ajuda contextual (mesmo padrao [G]/[0] usado em
    # _menu_hardware/_menu_prediction).
    gate_lbl = "[Enter] Rodar auditoria" if is_pt else "[Enter] Run audit"
    raw_gate = _ask(f"  [{PA}][G][/{PA}] Guaraci   [{PM}][0][/{PM}] {_t('voltar')}"
                     f"   [{PM}]{gate_lbl}[/{PM}]: ").strip().upper()
    if raw_gate in ("0", "Q"):
        return
    if raw_gate == "G":
        _abrir_assistente(_t("t_auditoria"), cfg)
        return

    status_msg = "Carregando dados e auditando..." if is_pt else "Loading data and auditing..."
    try:
        with console.status(f"[{PA}]{status_msg}[/{PA}]"):
            wavenumbers, X_raw, rotulos, conc, mae_id, _metadados = pq.load_data(cfg)
            X_raw, wavenumbers, rotulos, conc, mae_id, _relatorio = pq.validate_input(
                X_raw, wavenumbers, rotulos, conc, mae_id)
            from guaraci.auditoria_delineamento import run_audit
            achados = run_audit(X_raw, wavenumbers, rotulos, cfg, conc, mae_id)
    except Exception as e:  # noqa: BLE001 -- dado externo pode falhar de
        # varias formas (parsing, faixa espectral vazia, etc.) -- reportar
        # a mensagem, nunca stack trace cru numa ferramenta interativa.
        console.print(f"  [{PR}]{'Erro ao carregar dados' if is_pt else 'Error loading data'}: "
                      f"{escape(str(e))}[/{PR}]")
        _pause(); return

    cores = {"ok": PG, "aviso": PA, "critico": PR, "silenciado": PM}
    console.print()
    for a in achados:
        cor = cores.get(a.severidade, PW)
        console.print(f"  [{cor}]{a.severidade.upper():>10}[/{cor}]  "
                      f"[{PW}]{escape(a.nome)}[/{PW}]: {escape(a.mensagem)}")

    n_criticos = sum(1 for a in achados if a.severidade == "critico")
    n_avisos = sum(1 for a in achados if a.severidade == "aviso")
    resumo_lbl = (f"{n_criticos} critico(s), {n_avisos} aviso(s) de {len(achados)} checagem(ns)."
                  if is_pt else
                  f"{n_criticos} critical, {n_avisos} warning(s) out of {len(achados)} check(s).")
    cor_resumo = PR if n_criticos else (PA if n_avisos else PG)
    console.print()
    console.print(Panel(
        Text.from_markup(f"  {resumo_lbl}"),
        border_style=cor_resumo, box=rbox.ROUNDED, padding=(0, 2),
    ))
    _pause()


# ---------------------------------------------------------------------------
# PERFIS — cartoes compactos (2 por linha)
# ---------------------------------------------------------------------------
def _menu_profiles(cfg: Config) -> None:
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
        if paleta and _PALETAS_COR and paleta in _PALETAS_COR:
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
        _cls(); _print_header(cfg)

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


def _menu_about(cfg: Optional[Config] = None) -> None:
    """Secao Sobre — proposito, autor, citacao em multiplos formatos e referencias."""
    # Dados fixos do projeto e do autor
    _AUTOR_NOME    = "Erley S. da Costa"
    _AUTOR_TAG     = "Pesquisador / Desenvolvedor"
    _AUTOR_LATTES  = "http://lattes.cnpq.br/5755582193284309"
    _AUTOR_GITHUB  = "https://github.com/ErleySC"
    _AUTOR_EMAIL   = "erleysdacosta@gmail.com"
    _REPO          = "https://github.com/ErleySC/guaraci"
    _VERSAO        = pq.__version__
    _ANO           = "2026"
    _TITULO_CURTO  = "GUARACI"
    _LIC           = "GPL-3.0-or-later"

    def _titulo(lang: str) -> str:
        return ("Plataforma quimiometrica com validacao anti-vazamento por padrao"
                if lang == "PT" else
                "Chemometrics platform with leakage-safe validation by default")

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
            p3 = ("Desenvolvido no ambito de uma pesquisa sobre oleos"
                  " vegetais amazonicos, com metodologia generalizavel para"
                  " qualquer tecnica analitica com dados multivariados.")
        else:
            p1 = ("Democratize access to high-quality chemometric analyses"
                  " for researchers without a programming background.")
            p2 = ("Provides a reliable, reproducible and bilingual (PT/EN)"
                  " environment for classification, authentication and exploration"
                  " of complex matrices — from FT-NIR to GC-MS, without writing code.")
            p3 = ("Developed within a research project on Amazonian"
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
               f" [Software]. {_REPO}")
        # ABNT (NBR 6023:2018 — software)
        abnt = (f"{autor_abnt}. {_TITULO_CURTO}: {_titulo('PT')}."
                f" Versao {_VERSAO}. {_ANO}."
                f" Disponivel em: <{_REPO}>.")
        # BibTeX
        bibtex = (
            f"@software{{guaraci_{_ANO},\n"
            f"  author    = {{Costa, Erley S. da}},\n"
            f"  title     = {{{{{_TITULO_CURTO}: {_titulo('PT')}}}}},\n"
            f"  version   = {{{_VERSAO}}},\n"
            f"  year      = {{{_ANO}}},\n"
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
            ref = (_REFERENCIAS_GUARACI or {}).get(rk, {})
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
        _cls(); _print_header(cfg)
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
def _menu_help(cfg: Optional[Config] = None) -> None:
    """Ajuda navegavel — lista todos os campos de cara; numero abre a ajuda."""
    lang = _lang()
    # Lista unificada a partir do _CONFIG_SPEC (todos os campos editaveis).
    keys = [s["key"] for s in _CONFIG_SPEC] if _CONFIG_SPEC else list(_HELP_DB.keys())
    # Remove duplicatas mantendo ordem
    seen: set = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]

    while True:
        _cls(); _print_header(cfg)

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
                Text(_risco_icon(key) + " " + _RISK_CLASS.get(key, "—"), style=r_hex),
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
def _estimar_tempo(cfg: Config, n_amostras: int) -> Optional[str]:
    """Estimativa de ORDEM DE GRANDEZA do tempo de execucao, em texto.

    Existe porque o usuario apertava [R] sem nenhuma nocao de estar
    comprometendo 5 minutos ou 3 horas -- e a diferenca entre esses dois
    casos e' so' um campo de configuracao (n_jobs_permutacao).

    Calibracao (medida em 2026-08-05 num acervo de referencia interno
    x 2000 variaveis, CPython 3.12, 8 nucleos fisicos):
        - 1 ajuste PLS com 40 LVs .......... 3,5 s
        - carga de alguns milhares de .dx ... ~20 s
    O custo de um ajuste escala aproximadamente com (n_amostras x n_LVs),
    o que da a constante `_S_POR_AMOSTRA_LV` abaixo.

    E' ESTIMATIVA, nao promessa: ignora I/O de figuras, variacao de hardware
    e o custo dos modulos opcionais mais pesados (SHAP em especial). Por isso
    o texto devolvido usa faixa ("~15-25 min"), nunca um numero exato.
    Devolve None quando nao ha' base para estimar (n_amostras desconhecido).
    """
    if not n_amostras or n_amostras <= 0:
        return None

    # Segundos por amostra por LV, da calibracao acima.
    _S_POR_AMOSTRA_LV = 5.2e-5

    max_lvs = int(_cfgv(cfg, "max_lvs", 40) or 40)
    n_splits = int(_cfgv(cfg, "n_splits_cv", 5) or 5)
    frac_treino = 1.0 - 1.0 / max(n_splits, 2)

    # Selecao de LVs: para cada n=1..max_lvs, um ajuste por fold. O custo
    # cresce com n, entao a soma equivale a max_lvs*(max_lvs+1)/2 "LVs-ajuste".
    lvs_ajuste = max_lvs * (max_lvs + 1) / 2.0
    t_cv = lvs_ajuste * n_splits * n_amostras * frac_treino * _S_POR_AMOSTRA_LV

    # Permutacao/Wold: n_perm repeticoes de uma CV completa no LV escolhido
    # (aproximado por max_lvs/2, o valor tipico do criterio de parcimonia).
    t_perm = 0.0
    n_perm = int(_cfgv(cfg, "n_permutacoes", 0) or 0)
    if _cfgv(cfg, "teste_wold", False):
        n_perm += int(_cfgv(cfg, "n_permutations_wold", 0) or 0)
    if n_perm:
        n_jobs = max(1, int(_cfgv(cfg, "n_jobs_permutacao", 1) or 1))
        t_1perm = (max_lvs / 2.0) * n_splits * n_amostras * frac_treino * _S_POR_AMOSTRA_LV
        t_perm = n_perm * t_1perm / n_jobs

    # Modulos opcionais pesados, como multiplicadores grosseiros do custo de CV.
    t_extra = 0.0
    if _cfgv(cfg, "benchmark", False):      t_extra += 3.0 * t_cv
    if _cfgv(cfg, "monte_carlo", False):
        t_extra += int(_cfgv(cfg, "n_monte_carlo", 60) or 60) / max(n_splits, 2) * (t_cv / max_lvs)
    if _cfgv(cfg, "shap_benchmark", False): t_extra += 2.0 * t_cv
    if _cfgv(cfg, "selecao_variaveis_etapa4", False): t_extra += 2.0 * t_cv
    if _cfgv(cfg, "selecao_ag", False):     t_extra += 10.0 * t_cv
    if _cfgv(cfg, "selecao_spa", False):    t_extra += 4.0 * t_cv

    total_s = t_cv + t_perm + t_extra
    if total_s < 90:
        return "< 2 min"
    minutos = total_s / 60.0
    # Faixa de +-40%: a incerteza real e' dessa ordem (hardware, figuras, I/O).
    lo, hi = max(1, int(minutos * 0.6)), int(minutos * 1.4) + 1
    if hi < 60:
        return f"~{lo}-{hi} min"
    return f"~{lo / 60.0:.1f}-{hi / 60.0:.1f} h"


def _checklist(cfg: Config) -> Tuple[bool, List]:
    lang = _lang()
    checks = []; erros = []

    pasta = _cfgv(cfg, "pasta_dados", "dados")
    pasta_ok = bool(pasta) and os.path.isdir(str(pasta))
    n_dx = _count_dx(pasta) if pasta_ok else 0

    n_para_estimar = n_dx
    if pasta_ok and n_dx > 0:
        checks.append((True,  _t("chk_dados") + f" ({n_dx} .dx)"))
        # Varredura barata dos cabecalhos (decimos de segundo): antecipa
        # os dois efeitos que MUDAM O N da analise e que antes so' apareciam
        # no meio do log, depois de o usuario ja' ter iniciado a rodada.
        if _cfgv(cfg, "modo_entrada", "dx") == "dx":
            try:
                from guaraci.dados_io import prescan_dx
                pre = prescan_dx(str(pasta))
            except (OSError, ValueError) as _e_pre:  # diagnostico e' best-effort
                checks.append((None, _t("chk_prescan_erro", erro=_e_pre)))
            else:
                n_para_estimar = pre["n_apos_descarte"] or n_dx
                if pre["n_fora_da_faixa"]:
                    detalhe = ", ".join(
                        f"{esp} {n}" for esp, n in
                        sorted(pre["fora_por_especie"].items(), key=lambda kv: -kv[1]))
                    checks.append((None, _t("chk_descarte", n=pre["n_fora_da_faixa"])
                                          + f" — {detalhe}"))
                if pre["n_sem_mae_id"]:
                    checks.append((None, _t("chk_orfaos", n=pre["n_sem_mae_id"])))
                if pre["n_grupos"]:
                    checks.append((True, _t("chk_grupos", n=pre["n_grupos"])))
    elif pasta_ok:
        checks.append((None,  _t("chk_dados") + " (0 .dx)"))
        erros.append("pasta_dados vazia")
    else:
        checks.append((False, _t("chk_err_dados")))
        erros.append("pasta_dados")

    mode = _cfgv(cfg, "modo_entrada", "dx")
    if mode == "csv":
        arq = _cfgv(cfg, "arquivo_csv", "")
        if arq and os.path.isfile(str(arq)):
            checks.append((True, _t("chk_csv")))
        else:
            checks.append((False, _t("chk_err_csv")))
            erros.append("arquivo_csv")
    else:
        checks.append((True, f"{_t('chk_modo')}: {mode}"))

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

    est = _estimar_tempo(cfg, n_para_estimar)
    if est:
        # Dica acionavel junto da estimativa: o unico campo que muda a ordem
        # de grandeza sem mudar nenhum resultado e' o paralelismo do teste de
        # permutacao (resultado identico, so' o tempo muda).
        dica = ""
        n_jobs = int(_cfgv(cfg, "n_jobs_permutacao", 1) or 1)
        if n_jobs == 1 and int(_cfgv(cfg, "n_permutacoes", 0) or 0) >= 50:
            dica = "  " + _t("chk_dica_jobs")
        checks.append((True, f"{_t('chk_tempo')}: {est}{dica}"))

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

    def row(lbl, val):
        if isinstance(val, bool):
            # Text.from_markup, NAO Text(): o construtor trata a string como
            # LITERAL e imprimia "[g]Yes[/g]" cru na tela (em PT e EN).
            if val:
                v_txt = Text.from_markup("[g]Sim[/g]" if lang == "PT" else "[g]Yes[/g]")
            else:
                v_txt = Text.from_markup("[m]Nao[/m]" if lang == "PT" else "[m]No[/m]")
            t.add_row(lbl, v_txt)
        else:
            # Mesmo motivo: alguns valores (ex.: "[err]KFold[/err]") carregam
            # markup e precisam ser interpretados, nao exibidos literalmente.
            texto = str(val)
            t.add_row(lbl, Text.from_markup(texto) if "[" in texto
                            else Text(texto, style=PS))

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
    andamento (via app_logic.log_progress), figuras ja concluidas
    (app_logic.figures_completed) contra o plano do mode (modos_analise.
    describe_plan), tempo decorrido/estimado restante e avisos nao-fatais
    (app_logic.log_warnings). Extraida de _rodar_pipeline como funcao de
    modulo para ser testavel isoladamente (ver test_guaraci_cli.py) sem
    precisar rodar o pipeline de verdade nem simular entrada interativa."""
    frac, label = _progresso_do_log(texto_log, len(plano_figuras) or None)
    figs = _figures_completed(texto_log)
    avisos = _avisos_do_log(texto_log)

    # LIMITE DE ALTURA (bug real, 2026-08-07: "tela preta").
    # `figs` e `avisos` crescem sem teto durante a execucao. Numa corrida
    # completa (26 figuras + varios avisos distintos) o painel passava de 35
    # linhas num terminal de 24. O `Live` do Rich reposiciona o cursor para
    # redesenhar; quando o bloco nao cabe na janela ele nao consegue e a tela
    # fica preta com so' o cursor piscando -- exatamente o sintoma relatado.
    # O calculo seguia rodando por baixo, so' o painel morria.
    # Solucao: manter o painel com altura LIMITADA, mostrando os itens mais
    # recentes (que e' o que interessa acompanhar) + um contador do resto.
    MAX_AVISOS  = 4
    MAX_FIG_LIN = 3      # linhas gastas com a lista de figuras
    n_ocultos = max(0, len(avisos) - MAX_AVISOS)
    avisos_vis = avisos[-MAX_AVISOS:]

    bar_w = 32
    preenchido = int(bar_w * frac)
    barra = "█" * preenchido + "░" * (bar_w - preenchido)
    eta_txt = "…"
    if frac > 0.05:
        eta_txt = _fmt_time(max(0.0, elapsed / frac - elapsed))

    partes = [
        Text.assemble(
            (f"{_t('exec_objetivo')}: ", PM), (objetivo_rotulo, f"bold {PA}")),
        Text(f"[{barra}] {frac * 100:5.1f}%  {label}", style=PA),
        Text(f"{_t('exec_eta')}: {eta_txt}   |   "
             f"{_fmt_time(elapsed)}", style=PW),
        Rule(style=PD),
        Text(f"{_t('exec_figuras')} ({len(figs)}/{len(plano_figuras)} "
             "planejadas):", style=f"bold {PM}"),
        # overflow="ellipsis" + no_wrap corta na largura; as figuras mais
        # recentes ficam visiveis e a linha nunca cresce em altura.
        Text("  ".join(f"✓ {f}" for f in figs[-MAX_FIG_LIN * 3:])
             if figs else "…",
             style=PW, overflow="ellipsis", no_wrap=False),
    ]
    if avisos:
        cab = f"{_t('exec_avisos')} ({len(avisos)})"
        if n_ocultos:
            cab += f" — mostrando os {MAX_AVISOS} ultimos, +{n_ocultos} antes"
        partes += [
            Rule(style=PD),
            Text(f"{cab}:", style=f"bold {PR}"),
            Text("\n".join(f"⚠ {a[:110]}" for a in avisos_vis), style=PR,
                 overflow="ellipsis"),
        ]
    return Panel(Group(*partes), border_style=PA, box=rbox.ROUNDED,
                 padding=(0, 1), title=f"  {_t('exec_inicio')}  ")


def _rodar_pipeline(cfg: Config) -> None:
    lang = _lang()
    _cls(); _print_header(cfg)

    pode = _print_checklist(cfg)
    if not pode:
        console.print(f"  [{PR}]{_t('pip_sem_dados')}[/{PR}]")
        _pause(); return

    console.print(); _print_resumo(cfg)

    # Nome da execucao
    console.print()
    tag_atual = getattr(cfg, "tag", "") or ""
    console.print(f"  [{PM}]{_t('tag_atual')}:[/{PM}] [{PA}]{escape(tag_atual) or _t('tag_auto')}[/{PA}]")
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
        paleta = _PALETAS_COR.get(vcfg.get("paleta", "qualitativo"), {})
        try: plt.style.use(paleta.get("style", "default"))
        except OSError:
            pass   # nome de estilo matplotlib desconhecido -- mantem o default
        cores = paleta.get("cores")
        if cores: plt.rcParams["axes.prop_cycle"] = plt.cycler(color=cores)
        cmap = paleta.get("cmap")
        if cmap: plt.rcParams["image.cmap"] = cmap
        fp = _FONT_PRESETS.get(vcfg.get("tamanho_fonte","m"), {})
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
    try:
        _USER_DIR.mkdir(parents=True, exist_ok=True)
        _save_config(cfg, str(_CFG_PATH))
    except OSError as _e_cfg_save:
        # Achado do "checkup geral" de interface (2026-08-07): esta chamada
        # nao tinha NENHUMA guarda -- um PermissionError aqui (HOME
        # read-only, disco cheio) derrubava o CLI com traceback bem na hora
        # de rodar a analise, no meio de uma sessao interativa. Config nao
        # persistida so' significa que as escolhas desta sessao nao vao
        # sobreviver ao proximo start -- nao pode impedir a corrida atual.
        _msg = (f"[AVISO] config.yaml nao pode ser salvo ({_e_cfg_save}); "
               f"as preferencias desta sessao nao serao lembradas."
               if lang == "PT" else
               f"[WARNING] config.yaml could not be saved ({_e_cfg_save}); "
               f"this session's preferences will not be remembered.")
        console.print(f"  [{PM}]{escape(_msg)}[/{PM}]")

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
        pq.resolve_objective(cfg), cfg.level)
    plano_figuras = pq.describe_plan(cfg)

    _done: Dict[str, object] = {"ok": False, "error": None}
    _logger = _LogThreadSafe()

    def _run():
        try:
            with contextlib.redirect_stdout(_logger), \
                 contextlib.redirect_stderr(_logger):
                _executar(cfg)
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

    # CAUSA RAIZ do bug da "tela preta" (2026-08-07/08): `console` (definido
    # em guaraci_theme.py) e' construido SEM `file=`, entao `Console.file` e'
    # uma property que resolve `sys.stdout` DINAMICAMENTE a cada escrita
    # (rich/console.py: `self._file or sys.stdout`). `contextlib.redirect_
    # stdout` troca `sys.stdout` GLOBALMENTE no processo -- nao por thread.
    # Enquanto `_run()` (rodando em background) segura esse redirect durante
    # TODA a execucao do pipeline, o `Live` deste thread principal tambem
    # passa a escrever no MESMO buffer (`_logger`), nao no terminal de
    # verdade. Medido isolado: 0 bytes chegavam ao "terminal", 100% ia pro
    # buffer engolido. O painel nao travava nem estourava altura -- ele
    # simplesmente escrevia no lugar errado o tempo todo, daí a tela ficar
    # preta com so' o cursor. A correcao anterior (limitar altura do painel,
    # vertical_overflow="crop") ficou valida mas nao atacava esta causa.
    #
    # Fix: capturar a referencia REAL de stdout/stderr ANTES do redirect
    # comecar, e fixar `console._file` nela pela duracao do Live -- assim
    # o Console para de resolver `sys.stdout` dinamicamente e continua
    # escrevendo no terminal de verdade mesmo com o redirect global ativo
    # na outra thread.
    _stdout_real = sys.stdout
    _file_original = console._file
    console._file = _stdout_real

    thr = threading.Thread(target=_run, daemon=True)
    thr.start()

    try:
        # vertical_overflow="crop": rede de seguranca complementar. Mesmo
        # que o painel volte a crescer alem da janela, o Rich corta o
        # excesso em vez de perder o controle do cursor.
        with Live(console=console, refresh_per_second=3,
                  vertical_overflow="crop") as live:
            while not _done["ok"]:
                live.update(_render_painel(time.time() - t_ini))
                time.sleep(0.3)
            live.update(_render_painel(time.time() - t_ini))
    finally:
        # Restaura a resolucao dinamica de sys.stdout assim que o Live
        # termina -- nao deixar o pin permanente afetaria qualquer outro
        # uso de `console` depois desta tela (ex.: redirecionamento em
        # outro comando da mesma sessao do CLI).
        console._file = _file_original

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
        _save_config(cfg, str(path))
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
            cfg2 = _load_config(str(path))
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


# Modulos cujo nome de import difere do nome da distribuicao no PyPI. Sem este
# mapa, buscar os metadados por `<nome do modulo>` falha e a versao sai como "?".
_DIST_DE_MODULO = {"sklearn": "scikit-learn", "yaml": "PyYAML", "PIL": "pillow",
                   "skimage": "scikit-image", "fpdf": "fpdf2",
                   "docx": "python-docx", "pptx": "python-pptx"}


def _versao_por_metadados(modulo: str) -> str:
    """Versao pelos metadados da distribuicao, para pacotes sem __version__."""
    from importlib import metadata
    try:
        return metadata.version(_DIST_DE_MODULO.get(modulo, modulo))
    except metadata.PackageNotFoundError:
        return "?"


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
                ver = getattr(m, "__version__", None)
                if ver is None:
                    # Nem todo pacote expoe __version__ (rich, por exemplo) --
                    # os metadados da distribuicao sempre tem. Sem este fallback
                    # o doctor imprimia "rich ?" com o rich instalado e sao.
                    ver = _versao_por_metadados(mod)
            except ImportError as _e_imp:
                ver = f"erro ao importar: {_e_imp}"
            linhas.append(f"  [ok]    {mod} {ver}")

    linhas.append("")
    linhas.append("Extras opcionais ([web], [reports], [benchmark], [imagem]):")
    # CHAVE = nome do MODULO a importar; valor = (rotulo exibido, extra).
    # Nao confundir com o nome do PACOTE no pip: `pip install fpdf2` instala o
    # modulo `fpdf`, e `find_spec("fpdf2")` sempre devolve None -- o doctor
    # reportava "ausente" com o pacote instalado (bug achado em 2026-08-05
    # rodando o doctor de verdade num ambiente completo).
    opcionais = {
        "streamlit": ("streamlit", "web"),
        "psutil":    ("psutil", "web"),
        "fpdf":      ("fpdf2", "reports"),
        "docx":      ("python-docx", "reports"),
        "openpyxl":  ("openpyxl", "reports"),
        "pptx":      ("python-pptx", "reports"),
        "xgboost":   ("xgboost", "benchmark"),
        "shap":      ("shap", "benchmark"),
        "skimage":   ("scikit-image", "imagem"),
    }
    for mod, (rotulo, extra) in opcionais.items():
        spec = importlib.util.find_spec(mod)
        status = "ok" if spec is not None else "ausente"
        linhas.append(f"  [{status:7s}] {rotulo:13s} ({extra})")

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
        input_folder=str(saida / "dados_dummy"),  # mode sintetico ignora isto
        output_root_folder=str(saida),
        mode="sintetico",
        tag="demo",
        level="N2",       # DD-SIMCA (sensibilidade LOGO) — diferencial central do projeto
        objective="auto",
        n_per_class=15,
        n_synthetic_points=300,
        wn_min=400.0,
        wn_max=4001.0,
        synthetic_adulterants=("S", "M", "A"),
        max_lvs=15,
        n_splits_cv=3,
        n_repeats_cv=1,
        n_permutations=50,
        n_permutations_wold=50,
        n_bootstrap_vip=10,
        n_bootstrap_bca=100,
        n_monte_carlo=20,
        run_benchmark=False,
        run_monte_carlo=False,
        run_shap=False,
        run_wold=False,
        run_cv_anova=False,
        run_opls=False,
        executar_etapa4=False,
        comparar_pipelines=False,
        compare_hca_pipelines=False,
    )
    os.makedirs(cfg.input_folder, exist_ok=True)

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
        else:
            # subprocess com lista de argumentos (achado de auditoria de
            # seguranca, 2026-08-07): a versao anterior interpolava
            # pasta_run direto numa string de shell (os.system(f'open
            # "{pasta_run}"')) -- pasta_run e' sempre gerado internamente
            # neste caminho (guaraci demo), entao nao era explora'vel HOJE,
            # mas e' o mesmo PADRAO que seria uma injecao de comando real
            # se algum dia alimentado por um caminho influenciado pelo
            # usuario. Lista de argumentos nunca passa por um shell --
            # elimina a classe de vulnerabilidade por completo, nao so'
            # o caso de uso atual.
            import subprocess
            cmd = ["open"] if sys.platform == "darwin" else ["xdg-open"]
            subprocess.run(cmd + [str(pasta_run)], check=False)
    except OSError as _e_open:
        logging.getLogger(__name__).debug("nao foi possivel abrir a pasta de saida: %s", _e_open)


# ===========================================================================
# MAIN LOOP
# ===========================================================================
#: Texto de `--help`. Fonte unica: `--help`, `help` e a mensagem de
#: argumento desconhecido leem daqui, para nunca divergirem entre si.
_TEXTO_AJUDA = """Uso: guaraci [COMANDO] [OPCOES]

Comandos:
  (sem argumentos)  abre o assistente interativo
  demo              roda o pipeline com dados sinteticos (nao precisa de dado)
  doctor            diagnostica o ambiente (dependencias, RAM, CPU)
  perfis            lista os perfis de matriz disponiveis
  --version         mostra a versao instalada
  --help            mostra esta ajuda

Opcoes:
  --perfil=NOME     perfil da matriz analisada (padrao: generico). Define
                    faixa espectral, pre-processamento e o vocabulario da
                    saida. `guaraci perfis` lista os nomes; tambem aceita o
                    caminho de um YAML proprio.
  --mode=MODO       'cego' (PADRAO) ou 'controle'.
                    cego     : a quantificacao usa a classe PREDITA pelo
                               classificador -- o unico mode que corresponde
                               ao uso real, em que a classe da amostra e'
                               desconhecida.
                    controle : usa a classe VERDADEIRA. So' para diagnostico
                               interno (separar erro de quantificacao de erro
                               de classificacao). Os numeros obtidos assim
                               NAO representam desempenho de uso, e a saida
                               marca isso explicitamente.

Codigos de saida: 0 sucesso | 1 erro de execucao | 2 uso incorreto."""

#: Opcoes de linha de comando aplicadas ao Config depois de carrega-lo.
_OPCOES_CLI: Dict[str, str] = {}


def _extrair_opcoes(argv: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Separa `--chave=valor` dos argumentos posicionais.

    Aceita as opcoes em qualquer posicao (`guaraci --mode=controle demo` e
    `guaraci demo --mode=controle` sao equivalentes) -- comportamento
    previsivel importa mais aqui do que economizar codigo.
    """
    opcoes: Dict[str, str] = {}
    restantes: List[str] = []
    for arg in argv:
        if arg.startswith("--") and "=" in arg:
            chave, valor = arg[2:].split("=", 1)
            opcoes[chave.strip().lower()] = valor.strip()
        else:
            restantes.append(arg)
    return opcoes, restantes


def _validar_opcoes(opcoes: Dict[str, str]) -> None:
    """Valida ANTES de processar qualquer dado. Sai com codigo 2 (uso
    incorreto), nunca 1 (erro de execucao) -- sao coisas diferentes para
    quem chama o programa de um script."""
    for chave in ("perfil", "mode"):
        if chave in opcoes and not opcoes[chave]:
            print(f"Erro: --{chave} exige um valor (ex.: --mode=cego).",
                  file=sys.stderr)
            raise SystemExit(2)
    mode = opcoes.get("mode")
    if mode is not None and mode not in ("cego", "controle"):
        print(f"Erro: --mode={mode} invalido. Use 'cego' (padrao, usa a "
              f"classe predita) ou 'controle' (usa a classe verdadeira, "
              f"so' para diagnostico interno).", file=sys.stderr)
        raise SystemExit(2)
    if "perfil" in opcoes:
        from guaraci.perfil_matriz import (UnknownProfileError,
                                           load_profile)
        try:
            load_profile(opcoes["perfil"])
        except UnknownProfileError as e:
            print(f"Erro: {e}", file=sys.stderr)
            raise SystemExit(2) from None


def _listar_perfis() -> None:
    from guaraci.perfil_matriz import DIR_PERFIS, load_profile
    print("Perfis de matriz disponiveis:")
    for arquivo in sorted(DIR_PERFIS.glob("*.yaml")):
        perfil = load_profile(arquivo.stem)
        print(f"  {perfil.nome:<14} {perfil.descricao}")
    print("\nUse: guaraci --perfil=NOME  (ou o caminho de um YAML proprio)")


def main(argv: Optional[List[str]] = None) -> None:
    """Ponto de entrada GUARACI (versao unica em _VERSAO).

    `argv` sao os argumentos SEM o nome do programa. O default lê de
    `sys.argv`, que e' o caminho do entry point instalado. Chamadores
    programaticos (testes, embutimento em outro app) devem passar a lista
    explicitamente -- caso contrario herdam o `sys.argv` de quem os
    hospeda, e desde que argumento desconhecido virou erro de uso isso
    significaria sair com codigo 2 por causa das flags do pytest.
    """
    opcoes, restantes = _extrair_opcoes(
        list(sys.argv[1:]) if argv is None else list(argv))
    _validar_opcoes(opcoes)
    _OPCOES_CLI.clear()
    _OPCOES_CLI.update(opcoes)
    sys.argv = [sys.argv[0]] + restantes

    if restantes:
        comando = restantes[0].strip().lower().lstrip("-")
        if comando in ("version", "v"):
            _comando_versao(); return
        if comando == "demo":
            _comando_demo(); return
        if comando == "doctor":
            _comando_doctor(); return
        if comando == "perfis":
            _listar_perfis(); return
        if comando in ("help", "h"):
            print(_TEXTO_AJUDA)
            return
        print(f"Erro: comando desconhecido '{sys.argv[1]}'.\n",
              file=sys.stderr)
        print(_TEXTO_AJUDA, file=sys.stderr)
        raise SystemExit(2)

    # Migra estado gravado pela versao anterior (dentro do pacote instalado)
    # para _USER_DIR, se aplicavel -- ver docstring de _migrar_estado_legado.
    _migrar_estado_legado()

    # Carregar config
    cfg = Config()
    if _CFG_PATH.exists():
        try:
            cfg = _load_config(str(_CFG_PATH))
        except (RuntimeError, FileNotFoundError, ValueError) as _e_cfg:
            logging.getLogger(__name__).debug(
                "config.yaml nao carregado no boot, usando defaults: %s", _e_cfg)

    # Opcoes de linha de comando vencem o config.yaml: quem digitou a flag
    # agora quer ela agora. Ja' validadas em _validar_opcoes (saida 2).
    if "perfil" in _OPCOES_CLI:
        cfg.matrix_profile = _OPCOES_CLI["perfil"]
    if "mode" in _OPCOES_CLI:
        cfg.label_mode = _OPCOES_CLI["mode"]

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
        _cls()
        _print_header(cfg)
        _print_status(cfg)
        console.print()
        _print_main_menu()
        console.print()
        _print_run_box(cfg)
        console.print()

        try:
            # input() direto (nao _input()): _input() engole EOFError/
            # KeyboardInterrupt internamente e devolve "" -- com isso o
            # except abaixo NUNCA disparava em EOF real (stdin fechado/
            # redirecionado de arquivo vazio/pipe encerrado). "" nao bate
            # com nenhuma opcao do menu, cai no ramo "invalida" + _pause()
            # (tambem EOF-safe), e o loop volta a chamar _cls() (spawna
            # subprocesso via os.system) e ler de novo -- SEMPRE "" de novo
            # em EOF permanente -- girando para sempre, sem sair, gastando
            # CPU (achado 2026-08-07, "checkup geral" de interface:
            # reproduzido com stdin vazio, >350 redesenhos em 8s sem
            # terminar). input() aqui deixa o EOFError propagar ate o
            # except que ja existe para tratar exatamente este caso.
            raw = input(f"  {_t('opcao')}: ").strip()
            escolha = "?" if raw == "?" else raw.upper().strip()
        except (EOFError, KeyboardInterrupt):
            _exibir_despedida()
            break

        if escolha == "1": _menu_project(cfg)
        elif escolha == "2": _menu_data(cfg)
        elif escolha == "3": _menu_preprocessing(cfg)
        elif escolha == "4": _menu_modeling(cfg)
        elif escolha == "5": _menu_validation(cfg)
        elif escolha == "6": _menu_advanced(cfg)
        elif escolha == "7": _menu_visualization(cfg)
        elif escolha == "8": _menu_technique(cfg)
        elif escolha == "9": _menu_encoding(cfg)
        elif escolha == "H":
            _cls(); _print_header(cfg); _menu_hardware(cfg)
        elif escolha == "B":
            _menu_prediction(cfg)
        elif escolha == "X":
            _menu_hsi(cfg)
        elif escolha == "J":
            _menu_plan(cfg)
        elif escolha == "U":
            _menu_audit(cfg)
        elif escolha == "K":
            _menu_selecao_amostras(cfg)
        elif escolha == "P":
            _menu_profiles(cfg)
        elif escolha == "G":
            _abrir_assistente("menu principal", cfg)
        elif escolha == "M":
            novo_modo = _toggle_modo_usuario()
            _lbl = ("Iniciante" if novo_modo == "iniciante" else "Avancado") \
                if _lang() == "PT" else \
                ("Beginner" if novo_modo == "iniciante" else "Advanced")
            console.print(f"  [g]✓ Modo: {_lbl}[/g]" if _lang() == "PT"
                         else f"  [g]✓ Mode: {_lbl}[/g]")
            _pause()
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
            console.print(f"  [{PM}]{_t('tag_atual')}:[/{PM}] [{PA}]{escape(tag_atual) or _t('tag_auto')}[/{PA}]")
            novo = _input(f"  {_t('tag_novo')}")
            if novo == "?":
                cfg.tag = ""; console.print(f"  [{PM}]{_t('tag_limpo')}[/{PM}]")
            elif novo:
                san = _re.sub(r"[^\w\-_]", "_", novo)
                cfg.tag = san; console.print(f"  [g]✓ ID: {escape(san)}[/g]")
            _pause()
        elif escolha == "A":
            _menu_about(cfg)
        elif escolha == "?":
            _menu_help(cfg)
        elif escolha == "Q":
            _exibir_despedida()
            break
        else:
            console.print(f"  [{PM}]{_t('invalido')}[/{PM}]"); _pause()


if __name__ == "__main__":
    main()
