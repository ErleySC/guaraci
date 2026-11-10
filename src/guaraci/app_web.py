# -*- coding: utf-8 -*-
"""
============================================================================
 Chemometrics Platform — Streamlit Interface · 7 tabs
 (version is sourced from pipeline.__version__ at runtime — see _APP_VERSION)
============================================================================
Organization:
   1. Project      — study identification and objective
   2. Data         — input (FT-NIR .dx, local CSV, CSV upload)
   3. Preprocessing — spectral preset + before/after visualization
   4. Model        — advanced parameters + execution with live progress
   5. Validation   — figures and metrics from the last run
   6. Prediction   — apply saved model to unknown samples
   7. Reports      — download ZIP, summary, figure gallery, log

Engine: pipeline.py (dynamically imported).
No code editing required: configure, run, download.
============================================================================
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
from PIL import Image
matplotlib.use("Agg")
import streamlit as st

# `streamlit run .../guaraci/app_web.py` poe a pasta do pacote em sys.path[0], e
# la' `guaraci/guaraci.py` passaria a sombrear o pacote `guaraci` (ImportError
# circular num wheel instalado). Tira essa pasta do caminho antes dos imports.
_PASTA_DO_PACOTE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [_p for _p in sys.path
               if os.path.abspath(_p or os.curdir) != _PASTA_DO_PACOTE]

from guaraci import catalogo as _catalogo
from guaraci.design_tokens import tokens as _theme_tokens
# Lógica pura extraída da UI (item 19): testável sem Streamlit. Ver app_logic.py.
from guaraci.app_logic import (
    list_figures as _listar_figuras_pura,
    load_summary as _ler_resumo_pura,
    load_model_card as _ler_model_card_pura,
)
# Geração de relatórios (PDF/Word/Excel/LaTeX/PPTX) extraída para módulo de
# serviço próprio (item 18): app_quimiometria.py só cacheia e serve o download.
import guaraci.reports as reports
# Abas do app extraídas para módulos próprios (item 18): cada `with tab_x:`
# abaixo delega a app_tabs.<nome>.render(...); app_quimiometria.py fica só
# com a orquestração (setup compartilhado + chamada de cada aba).
from guaraci.app_tabs import projeto as _tab_projeto
from guaraci.app_tabs import dados as _tab_dados
from guaraci.app_tabs import preprocessamento as _tab_preprocessamento
from guaraci.app_tabs import modelo as _tab_modelo
from guaraci.app_tabs import validacao as _tab_validacao
from guaraci.app_tabs import predicao as _tab_predicao
from guaraci.app_tabs import tecnicas as _tab_tecnicas
from guaraci.app_tabs import relatorios as _tab_relatorios
from guaraci.app_tabs import sobre as _tab_sobre
# Telas novas da reestruturacao de 2026-09-08 (mockup seguido a risca):
# painel de status e personalizacao de cor, ambas fora dos 4 grupos.
from guaraci.app_tabs import inicio as _tab_inicio
from guaraci.app_tabs import visualizacao as _tab_visualizacao
from guaraci import app_nav


def _active_theme() -> str:
    """Tema ativo do Streamlit ('light' ou 'dark'), lido da API nativa.

    Usa `st.context.theme` (Streamlit >= 1.44) — fonte de verdade oficial, em
    vez de um estado paralelo. Faz fallback seguro para 'light' em versoes
    antigas ou quando o tema ainda nao esta disponivel no primeiro render.
    """
    try:
        t = getattr(st.context, "theme", None)
        if t is not None and getattr(t, "type", None) in ("light", "dark"):
            return t.type
    except (AttributeError, RuntimeError):
        # Streamlit antigo sem st.context, ou contexto indisponivel fora de
        # uma sessao ativa (ex.: import/teste) -- fallback claro documentado.
        pass
    return "light"


def _tok() -> Dict[str, str]:
    """Tokens de cor do tema atualmente ativo (dict semantico)."""
    return _theme_tokens(_active_theme())

# ──────────────────────────────────────────────────────────────────────────
# Page config (must be the first Streamlit command)
# ──────────────────────────────────────────────────────────────────────────
_icon_path = Path(__file__).resolve().parent / "assets" / "guaraci_icon.png"
_page_icon = Image.open(_icon_path) if _icon_path.exists() else "🧪"

st.set_page_config(
    page_title="GUARACI — Chemometrics Platform",
    page_icon=_page_icon,
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _logo_data_uri() -> str:
    """Logo embutido como data URI (mesmo arquivo do favicon) — usado no
    cabeçalho para não depender de um segundo `st.image` fora do layout."""
    import base64
    try:
        return "data:image/png;base64," + base64.b64encode(
            _icon_path.read_bytes()).decode("ascii")
    except OSError:
        return ""

# ──────────────────────────────────────────────────────────────────────────
# Pipeline engine
# ──────────────────────────────────────────────────────────────────────────
# config.yaml do modo local: fica ao lado do `app_quimiometria.py` que iniciou o
# app (a raiz da copia do repositorio). Instalado por pip nao ha' esse arquivo,
# entao vale a pasta onde o comando foi aberto (se tiver config.yaml; se nao,
# a configuracao padrao).
def _pasta_do_config_local() -> str:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        script = Path(ctx.main_script_path).resolve() if ctx else None
    except (ImportError, AttributeError, OSError, RuntimeError):  # sem contexto: pasta atual
        script = None
    if script is not None and (script.parent / "app_quimiometria.py").is_file():
        return str(script.parent)
    return str(Path.cwd())


_AQUI = _pasta_do_config_local()
_CFG_PATH_LOCAL = os.path.join(_AQUI, "config.yaml")


@st.cache_resource(show_spinner="Loading pipeline engine...")
def _carregar_motor():
    """Load the chemometrics pipeline module (guaraci.pipeline).

    Import de pacote normal — o bootstrap de ./src no topo do arquivo garante
    que `guaraci` esteja no path (inclusive no Streamlit Cloud). O sentinela
    `_CONFIG_SPEC` mantém uma mensagem de erro clara se algum import de nível
    de módulo falhar e deixar o pacote parcialmente carregado.
    """
    try:
        import guaraci.pipeline as _pq
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import pipeline engine:\n"
            f"  {type(exc).__name__}: {exc}\n\n"
            f"Check that all dependencies in requirements.txt are installed "
            f"and compatible with Python {sys.version.split()[0]}."
        ) from exc

    # Sentinel: ensure the module executed completely
    if not hasattr(_pq, "_CONFIG_SPEC"):
        attrs = sorted(a for a in dir(_pq) if not a.startswith("__"))
        raise RuntimeError(
            f"Pipeline module loaded but _CONFIG_SPEC is missing.\n"
            f"Attributes present ({len(attrs)}): {attrs[:30]}"
        )
    return _pq


pq = _carregar_motor()

# Fonte UNICA de versao: derivada de pipeline.__version__ (evita version drift).
# Toda string exibida ao usuario (relatorios PDF/DOCX, template LaTeX, rodapes)
# deve usar esta constante, nunca um literal "vXX.Y" hardcoded.
_APP_VERSION = f"v{getattr(pq, '__version__', '?')}"

# ── Segurança: modo servidor (PADRÃO) × modo local ───────────────────────────
# Parecer de revisão por pares de 2026-09-25: o app web nasceu para uso
# single-user na máquina do pesquisador e ganhou deploy público sem mudar o
# padrão -- carga de .joblib (pickle = execução de código), caminhos livres do
# servidor e um config.yaml compartilhado entre visitantes. Agora o PADRÃO é
# seguro para rede (modo servidor): só upload, tudo numa pasta de sessão
# isolada e apagada ao fim, modelo só em JSON portátil, sem config.yaml
# compartilhado. O uso local liga GUARACI_LOCAL_MODE=1 pelo comando `guaraci web`
# (o iniciar_guaraci.bat abre so' a CLI e nao liga o modo local). Política e justificativa: guaraci.web_seguranca.
from guaraci import web_seguranca as _ws

_MODO_LOCAL = _ws.modo_local()
_MODO_SERVIDOR = not _MODO_LOCAL
# Nome mantido: a aba Predição recebe "carga de pickle bloqueada?".
_UPLOAD_MODELO_BLOQUEADO = not _ws.pickle_permitido()

# Campos de CAMINHO do servidor. Em modo servidor nenhum deles é editável: um
# text_input livre deixa qualquer visitante enumerar diretórios, ler arquivos
# de texto (o conteúdo vaza na mensagem de erro) ou escrever fora da sessão.
_CAMPOS_CAMINHO_SERVIDOR = {"pasta_dados", "arquivo_csv", "pasta_saida",
                            "hsi_pasta_dataset", "arquivo_grupos",
                            "hsi_referencia_branco", "hsi_referencia_escuro"}
# Modos de entrada que dependem de pasta/arquivo do servidor: só locais.
_MODOS_ENTRADA_SERVIDOR = _ws.MODOS_ENTRADA_SERVIDOR   # fonte unica: web_seguranca

# ── Language state ──────────────────────────────────────────────────────────
# Light/dark é gerido pelo TEMA NATIVO do Streamlit (menu ⋮ → Settings → Theme),
# lido via _active_theme(). Não há mais estado paralelo `dark_mode` nem CSS
# `!important` pintando widgets à mão (origem do bug de cor ao trocar tema).
if "lang" not in st.session_state:
    st.session_state.lang = "EN"

_TR: Dict[str, Dict[str, str]] = {
    # Tabs
    "Project":              {"PT": "Projeto",              "EN": "Project"},
    "Data":                 {"PT": "Dados",                "EN": "Data"},
    "Preprocessing":        {"PT": "Pré-processamento",    "EN": "Preprocessing"},
    "Model":                {"PT": "Modelo",               "EN": "Model"},
    "Validation":           {"PT": "Validação",            "EN": "Validation"},
    "Prediction":           {"PT": "Predição",             "EN": "Prediction"},
    "Reports":              {"PT": "Relatórios",           "EN": "Reports"},
    "About":                {"PT": "Sobre",                "EN": "About"},
    # Sidebar
    "Language":             {"PT": "Idioma",               "EN": "Language"},
    "Dark mode":            {"PT": "Modo noturno",         "EN": "Dark mode"},
    # Data tab
    "Upload spectra":       {"PT": "Upload espectros",     "EN": "Upload spectra"},
    "Data preview":         {"PT": "Prévia dos dados",     "EN": "Data preview"},
    "Load spectra preview": {"PT": "Carregar prévia",      "EN": "Load spectra preview"},
    "Spectra folder":       {"PT": "Pasta de espectros",   "EN": "Spectra folder"},
    "Upload CSV":           {"PT": "Upload CSV",           "EN": "Upload CSV"},
    # Model tab
    "Run pipeline":         {"PT": "Executar pipeline",    "EN": "Run pipeline"},
    "Stop":                 {"PT": "Parar",                "EN": "Stop"},
    "Analysis and partitioning": {"PT": "Análise e particionamento", "EN": "Analysis and partitioning"},
    "Validation settings":  {"PT": "Configurações de validação", "EN": "Validation settings"},
    "Advanced options":     {"PT": "Opções avançadas",     "EN": "Advanced options"},
    "Figures":              {"PT": "Figuras",              "EN": "Figures"},
    # Validation tab
    "Model summary":        {"PT": "Resumo do modelo",     "EN": "Model summary"},
    "Accuracy by class":    {"PT": "Acurácia por classe",  "EN": "Accuracy by class"},
    "Benchmark results":    {"PT": "Resultados benchmark", "EN": "Benchmark results"},
    "Figure gallery":       {"PT": "Galeria de figuras",   "EN": "Figure gallery"},
    # Prediction tab
    "Upload model":         {"PT": "Carregar modelo",      "EN": "Upload model"},
    "Upload samples":       {"PT": "Upload amostras",      "EN": "Upload samples"},
    "Predict":              {"PT": "Predizer",             "EN": "Predict"},
    "Results":              {"PT": "Resultados",           "EN": "Results"},
    # Reports tab
    "Download ZIP":         {"PT": "Baixar ZIP",           "EN": "Download ZIP"},
    "Generate PDF":         {"PT": "Gerar PDF",            "EN": "Generate PDF"},
    "Generate Word":        {"PT": "Gerar Word",           "EN": "Generate Word"},
    "Generate Excel":       {"PT": "Gerar Excel",          "EN": "Generate Excel"},
    "Generate LaTeX":       {"PT": "Gerar LaTeX",          "EN": "Generate LaTeX"},
    "Generate PowerPoint":  {"PT": "Gerar PowerPoint",     "EN": "Generate PowerPoint"},
    "Clean old runs":       {"PT": "Limpar execuções antigas", "EN": "Clean old runs"},
    # Messages
    "No results yet":       {"PT": "Sem resultados ainda", "EN": "No results yet"},
    "Pipeline running":     {"PT": "Pipeline em execução", "EN": "Pipeline running"},
    "Pipeline complete":    {"PT": "Pipeline concluído",   "EN": "Pipeline complete"},
    "Error":                {"PT": "Erro",                 "EN": "Error"},
    "Warning":              {"PT": "Aviso",                "EN": "Warning"},
    "Success":              {"PT": "Sucesso",              "EN": "Success"},
    "Hardware status":      {"PT": "Status de hardware",   "EN": "Hardware status"},
    "Total RAM":            {"PT": "RAM total",            "EN": "Total RAM"},
    "Free RAM":             {"PT": "RAM livre",            "EN": "Free RAM"},
    "CPU cores":            {"PT": "Núcleos CPU",          "EN": "CPU cores"},
    # Onboarding / section headers
    "Project Identification": {"PT": "Identificação do Projeto", "EN": "Project Identification"},
    "Data Input":             {"PT": "Entrada de Dados",         "EN": "Data Input"},
    "Model Parameters and Execution": {"PT": "Parâmetros e Execução do Modelo", "EN": "Model Parameters and Execution"},
    "Validation Results":     {"PT": "Resultados de Validação",  "EN": "Validation Results"},
    "Run":                    {"PT": "Executar",                 "EN": "Run"},
    "Save to session":        {"PT": "Salvar na sessão",         "EN": "Save to session"},
    "No results yet. Run the pipeline in the Model tab.": {"PT": "Sem resultados ainda. Execute o pipeline na aba Modelo.", "EN": "No results yet. Run the pipeline in the Model tab."},
    "Fix the data input (Data tab) to enable.": {"PT": "Corrija os dados de entrada (aba Dados) para habilitar.", "EN": "Fix the data input (Data tab) to enable."},
    "What will be generated": {"PT": "O que será gerado", "EN": "What will be generated"},
    "Preview of the analyses/figures this run will produce, based on the scientific objective above.": {"PT": "Prévia das análises/figuras que esta execução vai produzir, com base no objetivo científico acima.", "EN": "Preview of the analyses/figures this run will produce, based on the scientific objective above."},
    "No mode-specific figures for this objective (only the always-on overview figures).": {"PT": "Nenhuma figura específica de modo para este objetivo (só as figuras de visão geral, sempre geradas).", "EN": "No mode-specific figures for this objective (only the always-on overview figures)."},
    "Step 1: Fill project info": {"PT": "Etapa 1: Preencha as informações do projeto", "EN": "Step 1: Fill project info"},
    "Step 2: Upload or select spectra folder": {"PT": "Etapa 2: Faça upload ou selecione a pasta de espectros", "EN": "Step 2: Upload or select spectra folder"},
    "Step 3: Configure parameters and run": {"PT": "Etapa 3: Configure os parâmetros e execute", "EN": "Step 3: Configure parameters and run"},
    # ── Data tab (Agente 6.2 pos-sessao: as 4 abas abaixo -- Data/
    # Preprocessing/Prediction/Reports -- nao passavam por T() nenhum,
    # ficavam sempre em ingles mesmo com idioma=PT selecionado. Fechado
    # aqui, 2026-09-01.) ──────────────────────────────────────────────
    "📂 Step 2: Upload or select spectra folder → then go to **Model** tab.": {"PT": "📂 Etapa 2: Faça upload ou selecione a pasta de espectros → depois vá para a aba **Modelo**.", "EN": "📂 Step 2: Upload or select spectra folder → then go to **Model** tab."},
    "**🎯 Recommended analysis** *(optional shortcut)*": {"PT": "**🎯 Análise recomendada** *(atalho opcional)*", "EN": "**🎯 Recommended analysis** *(optional shortcut)*"},
    "Pick what you want to do — sets sensible defaults across all tabs; fine-tune afterwards if needed.": {"PT": "Escolha o que você quer fazer — define valores padrão sensatos em todas as abas; ajuste depois se precisar.", "EN": "Pick what you want to do — sets sensible defaults across all tabs; fine-tune afterwards if needed."},
    "**Upload CSV** *(alternative to the local path below)*": {"PT": "**Upload de CSV** *(alternativa ao caminho local abaixo)*", "EN": "**Upload CSV** *(alternative to the local path below)*"},
    "Do not upload personal or confidential data.": {"PT": "Não envie dados pessoais nem confidenciais.", "EN": "Do not upload personal or confidential data."},
    "Drag or select a CSV file": {"PT": "Arraste ou selecione um arquivo CSV", "EN": "Drag or select a CSV file"},
    "**Data preview**": {"PT": "**Prévia dos dados**", "EN": "**Data preview**"},
    "The file will be saved to a temporary folder and the path adjusted automatically.": {"PT": "O arquivo será salvo numa pasta temporária e o caminho ajustado automaticamente.", "EN": "The file will be saved to a temporary folder and the path adjusted automatically."},
    "Mode automatically set to 'csv'. The path above will be overridden when running.": {"PT": "Modo ajustado automaticamente para 'csv'. O caminho acima será sobrescrito ao executar.", "EN": "Mode automatically set to 'csv'. The path above will be overridden when running."},
    "🔍 Load spectra preview": {"PT": "🔍 Carregar prévia dos espectros", "EN": "🔍 Load spectra preview"},
    "💾 Save config.yaml": {"PT": "💾 Salvar config.yaml", "EN": "💾 Save config.yaml"},
    "↺ Reload config.yaml": {"PT": "↺ Recarregar config.yaml", "EN": "↺ Reload config.yaml"},
    "Status: {msg_dados}": {"PT": "Status: {msg_dados}", "EN": "Status: {msg_dados}"},
    "Loading spectra sample...": {"PT": "Carregando amostra de espectros...", "EN": "Loading spectra sample..."},
    "Could not load spectra for preview. Check the path/mode.": {"PT": "Não foi possível carregar espectros para a prévia. Verifique o caminho/modo.", "EN": "Could not load spectra for preview. Check the path/mode."},
    "Fields with errors:\n- ": {"PT": "Campos com erro:\n- ", "EN": "Fields with errors:\n- "},
    "File saved: `{tmp_path}`": {"PT": "Arquivo salvo: `{tmp_path}`", "EN": "File saved: `{tmp_path}`"},
    "Raw spectra (sample)": {"PT": "Espectros brutos (amostra)", "EN": "Raw spectra (sample)"},
    "{n} variables inside `faixas_excluir` are not shown (the model does not use them).": {
        "PT": "{n} variáveis dentro de `faixas_excluir` não aparecem (o modelo não as usa).",
        "EN": "{n} variables inside `faixas_excluir` are not shown (the model does not use them)."},
    "Fix the fields before saving.": {"PT": "Corrija os campos antes de salvar.", "EN": "Fix the fields before saving."},
    "Config reloaded.": {"PT": "Config recarregado.", "EN": "Config reloaded."},
    "**{n} spectra** · {k} classes: `{amostra}`{reticencias}": {"PT": "**{n} espectros** · {k} classes: `{amostra}`{reticencias}", "EN": "**{n} spectra** · {k} classes: `{amostra}`{reticencias}"},
    "Preset '{pname}' applied — check the Model tab.": {"PT": "Preset '{pname}' aplicado — confira a aba Modelo.", "EN": "Preset '{pname}' applied — check the Model tab."},
    "Saved to {cfg_path}": {"PT": "Salvo em {cfg_path}", "EN": "Saved to {cfg_path}"},
    "Error: {e}": {"PT": "Erro: {e}", "EN": "Error: {e}"},
    # ── Preprocessing tab ────────────────────────────────────────────
    "Spectral Preprocessing": {"PT": "Pré-processamento Espectral", "EN": "Spectral Preprocessing"},
    "⚗️ Choose the spectral preprocessing preset and preview before/after → then go to **Model** tab.": {"PT": "⚗️ Escolha o preset de pré-processamento espectral e visualize antes/depois → depois vá para a aba **Modelo**.", "EN": "⚗️ Choose the spectral preprocessing preset and preview before/after → then go to **Model** tab."},
    "**Before / after preprocessing visualization**": {"PT": "**Visualização antes / depois do pré-processamento**", "EN": "**Before / after preprocessing visualization**"},
    "Configure and validate data input (Data tab) to enable the preview.": {"PT": "Configure e valide a entrada de dados (aba Dados) para habilitar a prévia.", "EN": "Configure and validate data input (Data tab) to enable the preview."},
    "⚗️ Generate before/after preview": {"PT": "⚗️ Gerar prévia antes/depois", "EN": "⚗️ Generate before/after preview"},
    "Loading and processing spectra...": {"PT": "Carregando e processando espectros...", "EN": "Loading and processing spectra..."},
    "Could not load spectra. Check the Data tab.": {"PT": "Não foi possível carregar espectros. Verifique a aba Dados.", "EN": "Could not load spectra. Check the Data tab."},
    "Before preprocessing": {"PT": "Antes do pré-processamento", "EN": "Before preprocessing"},
    "After: {preset}": {"PT": "Depois: {preset}", "EN": "After: {preset}"},
    "Error applying preprocessing: {e}": {"PT": "Erro ao aplicar pré-processamento: {e}", "EN": "Error applying preprocessing: {e}"},
    "MSC (scatter correction) → 1st derivative SG (Savitzky-Golay) → Mean-Centering. **Best for FT-NIR with pronounced scatter.** Compare presets on your own data by turning on 'Compare preprocessing' in the Model tab.": {"PT": "MSC (correção de espalhamento) → 1ª derivada SG (Savitzky-Golay) → Centralização pela média. **Melhor para FT-NIR com espalhamento pronunciado.** Compare presets nos seus próprios dados ligando 'Comparar pré-processamentos' na aba Modelo.", "EN": "MSC (scatter correction) → 1st derivative SG (Savitzky-Golay) → Mean-Centering. **Best for FT-NIR with pronounced scatter.** Compare presets on your own data with `comparar_pre_processamentos`."},
    "SNV (variance normalization) → SG → Mean-Centering. Robust alternative to MSC when global reference is not stable.": {"PT": "SNV (normalização por variância) → SG → Centralização pela média. Alternativa robusta ao MSC quando a referência global não é estável.", "EN": "SNV (variance normalization) → SG → Mean-Centering. Robust alternative to MSC when global reference is not stable."},
    "Mean-Centering + division by standard deviation. **Caution**: collapses spectral noise when SG is not applied first.": {"PT": "Centralização pela média + divisão pelo desvio padrão. **Cuidado**: colapsa o ruído espectral quando o SG não é aplicado antes.", "EN": "Mean-Centering + division by standard deviation. **Caution**: collapses spectral noise when SG is not applied first."},
    "Mean centering only. Recommended as a comparative baseline.": {"PT": "Só centralização pela média. Recomendado como referência comparativa.", "EN": "Mean centering only. Recommended as a comparative baseline."},
    # ── Prediction tab ───────────────────────────────────────────────
    "Prediction on Unknown Samples": {"PT": "Predição em Amostras Desconhecidas", "EN": "Prediction on Unknown Samples"},
    "Upload a `.joblib` model generated by the pipeline and a CSV with new spectra (columns = wavenumbers, no class column).": {"PT": "Faça upload de um modelo `.joblib` gerado pelo pipeline e um CSV com espectros novos (colunas = números de onda, sem coluna de classe).", "EN": "Upload a `.joblib` model generated by the pipeline and a CSV with new spectra (columns = wavenumbers, no class column)."},
    "🔮 Predict": {"PT": "🔮 Predizer", "EN": "🔮 Predict"},
    "**1. Trained model (.joblib)**": {"PT": "**1. Modelo treinado (.joblib)**", "EN": "**1. Trained model (.joblib)**"},
    "**2. New spectra (CSV)**": {"PT": "**2. Espectros novos (CSV)**", "EN": "**2. New spectra (CSV)**"},
    "Upload CSV with new spectra": {"PT": "Upload de CSV com espectros novos", "EN": "Upload CSV with new spectra"},
    "First column to use as wavenumber (leave empty = auto)": {"PT": "Primeira coluna a usar como número de onda (deixe vazio = automático)", "EN": "First column to use as wavenumber (leave empty = auto)"},
    "**Prediction results**": {"PT": "**Resultados da predição**", "EN": "**Prediction results**"},
    "⬇️ Download results (.csv)": {"PT": "⬇️ Baixar resultados (.csv)", "EN": "⬇️ Download results (.csv)"},
    "🔒 Model loading (upload and local path) is disabled on this public deployment — a `.joblib`/pickle can execute arbitrary code when loaded, from ANY path on the server, not just uploaded files. Run the CLI or app locally to use the Prediction tab.": {"PT": "🔒 O carregamento de modelo (upload e caminho local) está desabilitado neste deploy público — um `.joblib`/pickle pode executar código arbitrário ao ser carregado, de QUALQUER caminho do servidor, não só de arquivos enviados. Rode a CLI ou o app localmente para usar a aba Predição.", "EN": "🔒 Model loading (upload and local path) is disabled on this public deployment — a `.joblib`/pickle can execute arbitrary code when loaded, from ANY path on the server, not just uploaded files. Run the CLI or app locally to use the Prediction tab."},
    "⚠️ Only upload `.joblib` models you generated yourself. A model file is a pickle and **runs code when loaded** — never load one from an untrusted source.": {"PT": "⚠️ Só faça upload de modelos `.joblib` que você mesmo gerou. Um arquivo de modelo é um pickle e **executa código ao ser carregado** — nunca carregue um de origem não confiável.", "EN": "⚠️ Only upload `.joblib` models you generated yourself. A model file is a pickle and **runs code when loaded** — never load one from an untrusted source."},
    "Upload the .joblib model": {"PT": "Upload do modelo .joblib", "EN": "Upload the .joblib model"},
    "Or local path to model": {"PT": "Ou caminho local do modelo", "EN": "Or local path to model"},
    "🔒 Local path to CSV is disabled on this public deployment — a free-text server-side path would let any visitor read arbitrary files on the server. Use the upload above instead.": {"PT": "🔒 O caminho local do CSV está desabilitado neste deploy público — um caminho de servidor em texto livre permitiria que qualquer visitante lesse arquivos arbitrários do servidor. Use o upload acima.", "EN": "🔒 Local path to CSV is disabled on this public deployment — a free-text server-side path would let any visitor read arbitrary files on the server. Use the upload above instead."},
    "Or local path to CSV": {"PT": "Ou caminho local do CSV", "EN": "Or local path to CSV"},
    "**Blind flow — Detect → Identify → Quantify**": {"PT": "**Fluxo cego — Detectar → Identificar → Quantificar**", "EN": "**Blind flow — Detect → Identify → Quantify**"},
    "⚠ 'classe_identificada' only ever exists when 'identificacao_cobertura'='validado' (formal statistical guarantee, calibrated with ≥2 independent collection sessions) — there is no 'informational' label without that guarantee. 'identificacao_candidatos' carries only the closest guess (with NO guarantee at all) for reference, never as a result to act on for a quality decision without confirming by a reference method.": {"PT": "⚠ 'classe_identificada' só existe quando 'identificacao_cobertura'='validado' (garantia estatística formal, calibrada com ≥2 sessões de coleta independentes) — não há rótulo 'informativo' sem essa garantia. 'identificacao_candidatos' traz só o palpite mais próximo (SEM garantia nenhuma) para referência, nunca como resultado a usar numa decisão de qualidade sem confirmar por método de referência.", "EN": "⚠ 'classe_identificada' only ever exists when 'identificacao_cobertura'='validado' (formal statistical guarantee, calibrated with ≥2 independent collection sessions) — there is no 'informational' label without that guarantee. 'identificacao_candidatos' carries only the closest guess (with NO guarantee at all) for reference, never as a result to act on for a quality decision without confirming by a reference method."},
    "No spectra CSV provided.": {"PT": "Nenhum CSV de espectros fornecido.", "EN": "No spectra CSV provided."},
    "Within PLS-DA model fit": {"PT": "Dentro do ajuste do modelo PLS-DA", "EN": "Within PLS-DA model fit"},
    "Combined distance of Hotelling T² (distance along the model's main directions) and Q-residual (unexplained variation) within the 95% limit — see the 'criterio' column for the exact rule applied; the T2_ok and Q_ok columns are per-axis diagnostics only. A sample outside this fit is an atypical spectrum for the calibration, not necessarily 'adulterated'.": {"PT": "Distância combinada de Hotelling T² (distância ao longo das direções principais do modelo) e resíduo Q (variação não explicada) dentro do limite de 95% — veja a coluna 'criterio' para a regra exata aplicada; as colunas T2_ok e Q_ok são só diagnóstico por eixo. Uma amostra fora deste ajuste é um espectro atípico para a calibração, não necessariamente 'adulterada'.", "EN": "Combined distance of Hotelling T² (distance along the model's main directions) and Q-residual (unexplained variation) within the 95% limit — see the 'criterio' column for the exact rule applied; the T2_ok and Q_ok columns are per-axis diagnostics only. A sample outside this fit is an atypical spectrum for the calibration, not necessarily 'adulterated'."},
    "Within applicability domain": {"PT": "Dentro do domínio de aplicabilidade", "EN": "Within applicability domain"},
    "Exploratory PCA check (Hotelling T²/Q-residual, Jaworska et al. 2005) of similarity to the training set as a whole — a broader, less strict screen than the PLS-DA fit above. 'Outside' flags a spectrum unlike anything the model was calibrated on.": {"PT": "Checagem exploratória por PCA (Hotelling T²/resíduo Q, Jaworska et al. 2005) de similaridade com o conjunto de treino como um todo — uma triagem mais ampla e menos estrita que o ajuste PLS-DA acima. 'Fora' sinaliza um espectro diferente de tudo que o modelo foi calibrado para reconhecer.", "EN": "Exploratory PCA check (Hotelling T²/Q-residual, Jaworska et al. 2005) of similarity to the training set as a whole — a broader, less strict screen than the PLS-DA fit above. 'Outside' flags a spectrum unlike anything the model was calibrated on."},
    "'confianca_%' is a relative score (normalized PLS-DA), not a calibrated probability.": {"PT": "'confianca_%' é um escore relativo (PLS-DA normalizado), não uma probabilidade calibrada.", "EN": "'confianca_%' is a relative score (normalized PLS-DA), not a calibrated probability."},
    "Expected alarm rate on legitimate samples (cross-validated rejection of the calibration): {taxa} (nominal 5%). With few physical samples per class it can be much higher.": {"PT": "Taxa esperada de alarmes em amostras legítimas (rejeição em validação cruzada da calibração): {taxa} (nominal 5%). Com poucas amostras físicas por classe ela pode ser muito maior.", "EN": "Expected alarm rate on legitimate samples (cross-validated rejection of the calibration): {taxa} (nominal 5%). With few physical samples per class it can be much higher."},
    # ── Sentinela de deriva + conjunto conforme de classificacao (Grupo 2,
    # fechamento 2026-09-18, paridade CLI/web) ──────────────────────────
    "🛰 Drift sentinel (n={n}): {msg}": {"PT": "🛰 Sentinela de deriva (n={n}): {msg}", "EN": "🛰 Drift sentinel (n={n}): {msg}"},
    "Drift sentinel unavailable: {e}": {"PT": "Sentinela de deriva indisponível: {e}", "EN": "Drift sentinel unavailable: {e}"},
    "🎯 Plausible class set (conformal, {cov:.0%} coverage): {amb} sample(s) with more than 1 plausible class, {vaz} with none — see 'classes_plausiveis' column.": {"PT": "🎯 Conjunto de classes plausíveis (conforme, {cov:.0%} de cobertura): {amb} amostra(s) com mais de 1 classe plausível, {vaz} com nenhuma — veja a coluna 'classes_plausiveis'.", "EN": "🎯 Plausible class set (conformal, {cov:.0%} coverage): {amb} sample(s) with more than 1 plausible class, {vaz} with none — see 'classes_plausiveis' column."},
    "🎯 Plausible class set: NOT VALIDATED (holdout lacked enough independent groups for alpha=0.05).": {"PT": "🎯 Conjunto de classes plausíveis: NÃO VALIDADO (holdout sem grupos independentes suficientes p/ alpha=0,05).", "EN": "🎯 Plausible class set: NOT VALIDATED (holdout lacked enough independent groups for alpha=0.05)."},
    "🧪 Purity (predicted species)": {"PT": "🧪 Pureza (espécie prevista)", "EN": "🧪 Purity (predicted species)"},
    "🏷 Adulterant identified": {"PT": "🏷 Adulterante identificado", "EN": "🏷 Adulterant identified"},
    "UNKNOWN means no species×adulterant combination had a statistical guarantee both sufficient AND exclusive for this sample — either no guarantee, or 2+ validated combinations tied (also blocks the label).": {"PT": "DESCONHECIDO significa que nenhuma combinação espécie×adulterante teve garantia estatística ao mesmo tempo suficiente E exclusiva para esta amostra — ou não há garantia, ou 2+ combinações validadas empataram (o que também bloqueia o rótulo).", "EN": "UNKNOWN means no species×adulterant combination had a statistical guarantee both sufficient AND exclusive for this sample — either no guarantee, or 2+ validated combinations tied (also blocks the label)."},
    "⚖ Quantified": {"PT": "⚖ Quantificado", "EN": "⚖ Quantified"},
    "Blocked = quantification refused because the adulterant was not reliably identified — see 'quantificacao_motivo_bloqueio' in the table above.": {"PT": "Bloqueada = quantificação recusada porque o adulterante não foi identificado com confiabilidade — veja 'quantificacao_motivo_bloqueio' na tabela acima.", "EN": "Blocked = quantification refused because the adulterant was not reliably identified — see 'quantificacao_motivo_bloqueio' in the table above."},
    "No valid model provided (upload or path).": {"PT": "Nenhum modelo válido fornecido (upload ou caminho).", "EN": "No valid model provided (upload or path)."},
    # ── Barra superior e navegação (reestruturação de 2026-09-08) ────
    "Switch in ⋮ → Settings → Theme": {"PT": "Trocar em ⋮ → Settings → Theme", "EN": "Switch in ⋮ → Settings → Theme"},
    "Export report": {"PT": "Exportar relatório", "EN": "Export report"},
    "Go to Model →": {"PT": "Ir para Modelo →", "EN": "Go to Model →"},
    "Go there": {"PT": "Ver", "EN": "Go there"},
    # ── Tela Início ──────────────────────────────────────────────────
    "DATA LOADED": {"PT": "DADOS CARREGADOS", "EN": "DATA LOADED"},
    "MEAN SPECTRA BY CLASS": {"PT": "ESPECTROS MÉDIOS POR CLASSE", "EN": "MEAN SPECTRA BY CLASS"},
    "PREDICTION RESULT": {"PT": "RESULTADO DA PREDIÇÃO", "EN": "PREDICTION RESULT"},
    "DECISION RANGE · ESTIMATED CONTENT": {"PT": "FAIXA DE DECISÃO · TEOR ESTIMADO", "EN": "DECISION RANGE · ESTIMATED CONTENT"},
    "physical samples": {"PT": "amostras físicas", "EN": "physical samples"},
    "Technique": {"PT": "Técnica", "EN": "Technique"},
    "Grouping": {"PT": "Agrupamento", "EN": "Grouping"},
    "By physical sample": {"PT": "Por amostra física", "EN": "By physical sample"},
    "protected": {"PT": "protegido", "EN": "protected"},
    "unprotected": {"PT": "desprotegido", "EN": "unprotected"},
    "Guarantee: {g}": {"PT": "Garantia: {g}", "EN": "Guarantee: {g}"},
    "No data loaded yet — start on the **Data** screen: point to a spectra folder or upload a CSV.": {"PT": "Nenhum dado carregado ainda — comece pela tela **Dados**: aponte uma pasta de espectros ou envie um CSV.", "EN": "No data loaded yet — start on the **Data** screen: point to a spectra folder or upload a CSV."},
    "Load the spectra preview on the **Data** screen to see the mean spectrum of each class here.": {"PT": "Carregue a prévia dos espectros na tela **Dados** para ver aqui o espectro médio de cada classe.", "EN": "Load the spectra preview on the **Data** screen to see the mean spectrum of each class here."},
    "Wavenumber (cm⁻¹) — decreasing": {"PT": "Número de onda (cm⁻¹) — decrescente", "EN": "Wavenumber (cm⁻¹) — decreasing"},
    "No prediction yet — the **Prediction** screen applies a saved model to new samples.": {"PT": "Nenhuma predição ainda — a tela **Predição** aplica um modelo salvo a amostras novas.", "EN": "No prediction yet — the **Prediction** screen applies a saved model to new samples."},
    "Detection": {"PT": "Detecção", "EN": "Detection"},
    "Identification": {"PT": "Identificação", "EN": "Identification"},
    "Quantification": {"PT": "Quantificação", "EN": "Quantification"},
    "{n} of {t} adulterated": {"PT": "{n} de {t} adulteradas", "EN": "{n} of {t} adulterated"},
    "{n} of {t} identified": {"PT": "{n} de {t} identificadas", "EN": "{n} of {t} identified"},
    "{n} of {t} quantified": {"PT": "{n} de {t} quantificadas", "EN": "{n} of {t} quantified"},
    "Unknown adulterant": {"PT": "Adulterante desconhecido", "EN": "Unknown adulterant"},
    "Not performed": {"PT": "Não realizada", "EN": "Not performed"},
    "Quantification was blocked because identification found no match with a known adulterant. A number here would have no statistical backing.": {"PT": "A quantificação foi bloqueada porque a identificação não encontrou correspondência com nenhum adulterante conhecido. Um número aqui não teria suporte estatístico.", "EN": "Quantification was blocked because identification found no match with a known adulterant. A number here would have no statistical backing."},
    "Samples": {"PT": "Amostras", "EN": "Samples"},
    "This model predates the blind flow, so there is no identification/quantification to show.": {"PT": "Este modelo é anterior ao fluxo cego, então não há identificação/quantificação para mostrar.", "EN": "This model predates the blind flow, so there is no identification/quantification to show."},
    "No quantified sample yet — the decision range appears once a content is estimated.": {"PT": "Nenhuma amostra quantificada ainda — a faixa de decisão aparece quando houver teor estimado.", "EN": "No quantified sample yet — the decision range appears once a content is estimated."},
    "species used": {"PT": "espécie usada", "EN": "species used"},
    "No LOD/LOQ persisted for this species model, so no decision range is drawn — a bar without limits would suggest a confidence the data does not support.": {"PT": "Sem LOD/LOQ persistido para o modelo desta espécie, então nenhuma faixa é desenhada — uma barra sem limites sugeriria uma confiança que os dados não sustentam.", "EN": "No LOD/LOQ persisted for this species model, so no decision range is drawn — a bar without limits would suggest a confidence the data does not support."},
    "< LOD": {"PT": "< LOD", "EN": "< LOD"},
    "grey zone": {"PT": "zona cinzenta", "EN": "grey zone"},
    "quantifiable": {"PT": "quantificável", "EN": "quantifiable"},
    "Coverage of this combination: {c}.": {"PT": "Cobertura desta combinação: {c}.", "EN": "Coverage of this combination: {c}."},
    "not declared": {"PT": "não declarada", "EN": "not declared"},
    # ── Tela Visualização ────────────────────────────────────────────
    "Colour scheme": {"PT": "Esquema de cor", "EN": "Colour scheme"},
    "Colour per class (customise)": {"PT": "Cor por classe (personalizar)", "EN": "Colour per class (customise)"},
    "Load the spectra preview on the **Data** screen to customise the colour of each real class.": {"PT": "Carregue a prévia dos espectros na tela **Dados** para personalizar a cor de cada classe real.", "EN": "Load the spectra preview on the **Data** screen to customise the colour of each real class."},
    "Reset to the scheme": {"PT": "Voltar ao esquema", "EN": "Reset to the scheme"},
    "This is the same palette catalogue the CLI offers (Visualisation menu). The choice is saved in `~/.guaraci/visual_config.json`, so terminal and web stay in sync, and it applies to **every figure of the next run**, not only to the preview below.": {"PT": "Este é o mesmo catálogo de paletas que a CLI oferece (menu Visualização). A escolha é gravada em `~/.guaraci/visual_config.json`, então terminal e web ficam em sincronia, e vale para **todas as figuras da próxima execução**, não só para a prévia ao lado.", "EN": "This is the same palette catalogue the CLI offers (Visualisation menu). The choice is saved in `~/.guaraci/visual_config.json`, so terminal and web stay in sync, and it applies to **every figure of the next run**, not only to the preview below."},
    "Preview · mean spectrum by class": {"PT": "Pré-visualização · espectro médio por classe", "EN": "Preview · mean spectrum by class"},
    "Figure colour palette: **{p}** — change it on the Visualisation screen.": {"PT": "Paleta de cores das figuras: **{p}** — troque na tela Visualização.", "EN": "Figure colour palette: **{p}** — change it on the Visualisation screen."},
    "No spectra loaded yet, so there is nothing real to draw here. The swatches on the left already show the exact colours this scheme will use.": {"PT": "Nenhum espectro carregado ainda, então não há nada real para desenhar aqui. As amostras à esquerda já mostram as cores exatas que este esquema vai usar.", "EN": "No spectra loaded yet, so there is nothing real to draw here. The swatches on the left already show the exact colours this scheme will use."},
    # ── Painel de status do projeto (topo da aba Projeto) ────────────
    "No data loaded yet — start on the **Data** tab: point to a spectra folder or upload a CSV. This panel fills in with the real numbers as soon as there is data.": {"PT": "Nenhum dado carregado ainda — comece pela aba **Dados**: aponte uma pasta de espectros ou envie um CSV. Este painel se preenche com os números reais assim que houver dado.", "EN": "No data loaded yet — start on the **Data** tab: point to a spectra folder or upload a CSV. This panel fills in with the real numbers as soon as there is data."},
    "### 📌 Project status": {"PT": "### 📌 Status do projeto", "EN": "### 📌 Project status"},
    "last completed run": {"PT": "última execução concluída", "EN": "last completed run"},
    "data preview (no run yet)": {"PT": "prévia dos dados (sem execução ainda)", "EN": "data preview (no run yet)"},
    "spectra": {"PT": "espectros", "EN": "spectra"},
    "classes": {"PT": "classes", "EN": "classes"},
    "spectral variables": {"PT": "variáveis espectrais", "EN": "spectral variables"},
    "Source: {origem}.": {"PT": "Origem: {origem}.", "EN": "Source: {origem}."},
    "**Matrix / technique:** {matriz}": {"PT": "**Matriz / técnica:** {matriz}", "EN": "**Matrix / technique:** {matriz}"},
    "Design audit not available for this run (run produced before this record existed).": {"PT": "Auditoria de delineamento indisponível para esta execução (execução anterior a este registro).", "EN": "Design audit not available for this run (run produced before this record existed)."},
    "{crit} critical · {aviso} warning(s) · {n} checks": {"PT": "{crit} crítico(s) · {aviso} aviso(s) · {n} checagens", "EN": "{crit} critical · {aviso} warning(s) · {n} checks"},
    "Design audit: {resumo}": {"PT": "Auditoria de delineamento: {resumo}", "EN": "Design audit: {resumo}"},
    "Design audit findings": {"PT": "Achados da auditoria de delineamento", "EN": "Design audit findings"},
    # ── Paleta de cores das figuras (aba Modelo) ─────────────────────
    "Figure color palette": {"PT": "Paleta de cores das figuras", "EN": "Figure color palette"},
    "Applies to the figures of the NEXT run — figures already saved on disk are not recolored.": {"PT": "Vale para as figuras da PRÓXIMA execução — figuras já gravadas em disco não são recoloridas.", "EN": "Applies to the figures of the NEXT run — figures already saved on disk are not recolored."},
    "A palette with fewer colors than the number of classes is refused at figure time (it would give two classes the same color) — the maximum-distinctiveness palette is used instead, and the run log says so.": {"PT": "Uma paleta com menos cores que o número de classes é recusada na hora da figura (daria a mesma cor a duas classes) — a paleta de máxima distintividade é usada no lugar, e o log da execução avisa.", "EN": "A palette with fewer colors than the number of classes is refused at figure time (it would give two classes the same color) — the maximum-distinctiveness palette is used instead, and the run log says so."},
    "Uses the default color sequence — no fixed color list.": {"PT": "Usa a sequência de cores padrão — sem lista fixa de cores.", "EN": "Uses the default color sequence — no fixed color list."},
    "Could not save the palette choice: {e}": {"PT": "Não foi possível salvar a escolha de paleta: {e}", "EN": "Could not save the palette choice: {e}"},
    # ── Faixa de decisão (LOD/LOQ, Bloco 24) na aba Predição ─────────
    "**Decision range (LOD / LOQ)**": {"PT": "**Faixa de decisão (LOD / LOQ)**", "EN": "**Decision range (LOD / LOQ)**"},
    "No LOD/LOQ available for the species models used — either the package predates the decision-range feature, or the limits are not computable (not enough physical replicates to estimate instrument noise). No decision range is shown rather than one without backing.": {"PT": "Sem LOD/LOQ disponível para os modelos de espécie usados — ou o pacote é anterior à faixa de decisão, ou os limites não são computáveis (sem réplicas físicas suficientes para estimar o ruído instrumental). Nenhuma faixa é mostrada, em vez de uma faixa sem lastro.", "EN": "No LOD/LOQ available for the species models used — either the package predates the decision-range feature, or the limits are not computable (not enough physical replicates to estimate instrument noise). No decision range is shown rather than one without backing."},
    "Below LOD (not detectable)": {"PT": "Abaixo do LOD (não detectável)", "EN": "Below LOD (not detectable)"},
    "Estimated content below the detection limit — the method cannot distinguish it from noise.": {"PT": "Teor estimado abaixo do limite de detecção — o método não consegue distingui-lo do ruído.", "EN": "Estimated content below the detection limit — the method cannot distinguish it from noise."},
    "Grey zone (LOD–LOQ)": {"PT": "Zona cinzenta (LOD–LOQ)", "EN": "Grey zone (LOD–LOQ)"},
    "Detection is possible, but quantification is not reliable in this range — report as 'detected, not quantifiable'.": {"PT": "A detecção é possível, mas a quantificação não é confiável nesta faixa — relate como 'detectado, não quantificável'.", "EN": "Detection is possible, but quantification is not reliable in this range — report as 'detected, not quantifiable'."},
    "Quantified with confidence (≥ LOQ)": {"PT": "Quantificado com confiança (≥ LOQ)", "EN": "Quantified with confidence (≥ LOQ)"},
    "At or above the quantification limit — the numeric value can be reported as a measurement.": {"PT": "No limite de quantificação ou acima dele — o valor numérico pode ser relatado como medida.", "EN": "At or above the quantification limit — the numeric value can be reported as a measurement."},
    "{n} quantified sample(s) have no decision range: the species model used has no persisted LOD/LOQ.": {"PT": "{n} amostra(s) quantificada(s) sem faixa de decisão: o modelo de espécie usado não tem LOD/LOQ persistido.", "EN": "{n} quantified sample(s) have no decision range: the species model used has no persisted LOD/LOQ."},
    "Limits from the analytical figures of merit of the per-species regression (Valderrama, Braga & Poppi 2009), computed during the run — see `lod`/`loq` in the table above.": {"PT": "Limites vindos das figuras de mérito analíticas da regressão por espécie (Valderrama, Braga & Poppi 2009), calculadas durante a execução — ver `lod`/`loq` na tabela acima.", "EN": "Limits from the analytical figures of merit of the per-species regression (Valderrama, Braga & Poppi 2009), computed during the run — see `lod`/`loq` in the table above."},
    "pure": {"PT": "pura", "EN": "pure"},
    "Error loading model: {e}": {"PT": "Erro ao carregar modelo: {e}", "EN": "Error loading model: {e}"},
    "Error reading CSV: {e}": {"PT": "Erro ao ler CSV: {e}", "EN": "Error reading CSV: {e}"},
    "Applying model...": {"PT": "Aplicando modelo...", "EN": "Applying model..."},
    "DD-SIMCA for the species predicted above. {n} detected as adulterated.": {"PT": "DD-SIMCA para a espécie prevista acima. {n} detectada(s) como adulterada(s).", "EN": "DD-SIMCA for the species predicted above. {n} detected as adulterated."},
    "Portable model: no applicability-domain check. 'confidence' is a softmax score, not a calibrated probability, and is high even for spectra unlike the training data (noise, absent bands). Do not trust it for samples outside the training profile; use the local mode with the full model for a domain check.": {"PT": "Modelo portátil: sem verificação de domínio de aplicabilidade. A 'confiança' é um escore softmax, não uma probabilidade calibrada, e fica alta até para espectros diferentes do treino (ruído, bandas ausentes). Não confie nela para amostras fora do perfil de treino; use o modo local com o modelo completo para checar o domínio.", "EN": "Portable model: no applicability-domain check. 'confidence' is a softmax score, not a calibrated probability, and is high even for spectra unlike the training data (noise, absent bands). Do not trust it for samples outside the training profile; use the local mode with the full model for a domain check."},
    "Synthetic mode has no file to preview: the spectra are generated when you run.": {"PT": "O modo sintético não tem arquivo para pré-visualizar: os espectros são gerados ao executar.", "EN": "Synthetic mode has no file to preview: the spectra are generated when you run."},
    "{n} sample(s) outside the model's applicability domain: the predicted class is not reliable for them.": {"PT": "{n} amostra(s) fora do domínio de aplicabilidade do modelo: a classe prevista não é confiável para elas.", "EN": "{n} sample(s) outside the model's applicability domain: the predicted class is not reliable for them."},
    "Decomposes the loaded spectra by design factors (species, and any low-cardinality metadata column) and tests each factor's significance by permutation — group-aware when mae_id is available.": {"PT": "Decompõe os espectros carregados pelos fatores do delineamento (espécie e qualquer coluna de metadados com poucos níveis) e testa a significância de cada fator por permutação — agrupado por mae_id quando disponível.", "EN": "Decomposes the loaded spectra by design factors (species, and any low-cardinality metadata column) and tests each factor's significance by permutation — group-aware when mae_id is available."},
    "Load dataset": {"PT": "Carregar dados", "EN": "Load dataset"},
    "Factors": {"PT": "Fatores", "EN": "Factors"},
    "Run ASCA": {"PT": "Executar ASCA", "EN": "Run ASCA"},
    "Removes/attenuates the variation of a KNOWN nuisance factor (session, host species, instrument), estimated from difference spectra between samples that share the condition to preserve but differ in the nuisance factor. Exploratory diagnostic, not wired into the automatic training pipeline.": {"PT": "Remove/atenua a variação de um fator de incômodo CONHECIDO (sessão, espécie hospedeira, instrumento), estimada por espectros de diferença entre amostras que compartilham a condição a preservar mas diferem no fator de incômodo. Diagnóstico exploratório, fora do pipeline automático de treino.", "EN": "Removes/attenuates the variation of a KNOWN nuisance factor (session, host species, instrument), estimated from difference spectra between samples that share the condition to preserve but differ in the nuisance factor. Exploratory diagnostic, not wired into the automatic training pipeline."},
    "Method": {"PT": "Método", "EN": "Method"},
    "Resolves the spectral matrix into pure concentration/spectrum profiles (unsupervised), or anchors one component to known reference values when content/concentration is available (correlation-constrained variant). Interpretive tool, does NOT replace PLS-R for quantification.": {"PT": "Resolve a matriz espectral em perfis puros de concentração/espectro (não supervisionado), ou ancora um componente em valores de referência conhecidos quando há teor/concentração (variante com restrição de correlação). Ferramenta interpretativa, NÃO substitui o PLS-R na quantificação.", "EN": "Resolves the spectral matrix into pure concentration/spectrum profiles (unsupervised), or anchors one component to known reference values when content/concentration is available (correlation-constrained variant). Interpretive tool, does NOT replace PLS-R for quantification."},
    "Number of components": {"PT": "Número de componentes", "EN": "Number of components"},
    "Run MCR-ALS": {"PT": "Executar MCR-ALS", "EN": "Run MCR-ALS"},
    "Fuses >=2 blocks measured on the SAME physical sample (e.g. NIR + MIR) into a single PLS-R regression. Needs 2 folders/files with the same samples in the same row order. Fusion does not always beat the best single block: compare against each block alone.": {"PT": "Funde ≥2 blocos medidos na MESMA amostra física (ex.: NIR + MIR) numa única regressão PLS-R. Precisa de 2 pastas/arquivos com as mesmas amostras na mesma ordem de linhas. A fusão nem sempre supera o melhor bloco isolado: compare com cada bloco sozinho.", "EN": "Fuses >=2 blocks measured on the SAME physical sample (e.g. NIR + MIR) into a single PLS-R regression. Needs 2 folders/files with the same samples in the same row order. Fusion does not always beat the best single block: compare against each block alone."},
    "Run multiblock fusion": {"PT": "Executar fusão multibloco", "EN": "Run multiblock fusion"},
    "Advanced Techniques": {"PT": "Técnicas Avançadas", "EN": "Advanced Techniques"},
    "🧪 ASCA, EPO/GLSW, MCR-ALS and multiblock fusion — exploratory/diagnostic techniques, not wired into the automatic training pipeline. ASCA/EPO-GLSW/MCR-ALS use the dataset configured in the Data tab; multiblock fusion asks for 2 folders on the spot.": {"PT": "🧪 ASCA, EPO/GLSW, MCR-ALS e fusão multibloco — técnicas exploratórias/diagnósticas, fora do pipeline automático de treino. ASCA/EPO-GLSW/MCR-ALS usam os dados configurados na tela Dados; a fusão multibloco pede 2 pastas na hora.", "EN": "🧪 ASCA, EPO/GLSW, MCR-ALS and multiblock fusion — exploratory/diagnostic techniques, not wired into the automatic training pipeline. ASCA/EPO-GLSW/MCR-ALS use the dataset configured in the Data tab; multiblock fusion asks for 2 folders on the spot."},
    "Choose at least 1 factor.": {"PT": "Escolha pelo menos 1 fator.", "EN": "Choose at least 1 factor."},
    "grupo_interesse (preserve)": {"PT": "grupo de interesse (preservar)", "EN": "grupo_interesse (preserve)"},
    "fator_incomodo (remove)": {"PT": "fator de incômodo (remover)", "EN": "fator_incomodo (remove)"},
    "The two factors must be different.": {"PT": "Os dois fatores precisam ser diferentes.", "EN": "The two factors must be different."},
    "Use correlation restriction with known content": {"PT": "Usar restrição de correlação com o teor conhecido", "EN": "Use correlation restriction with known content"},
    "Multiblock fusion": {"PT": "Fusão multibloco", "EN": "Multiblock fusion"},
    "Group-aware (mae_id)": {"PT": "Agrupado (mae_id)", "EN": "Group-aware (mae_id)"},
    "large = correlated factors, see module scope limit": {"PT": "grande = fatores correlacionados; ver limite de escopo do módulo", "EN": "large = correlated factors, see module scope limit"},
    "Target component index (0-indexed)": {"PT": "Índice do componente-alvo (começa em 0)", "EN": "Target component index (0-indexed)"},
    "Block name": {"PT": "Nome do bloco", "EN": "Block name"},
    "Neither block has content/concentration (y) — multiblock fusion here only supports regression.": {"PT": "Nenhum bloco tem teor/concentração (y) — aqui a fusão multibloco só faz regressão.", "EN": "Neither block has content/concentration (y) — multiblock fusion here only supports regression."},
    "ss_desbalanco": {"PT": "SS de desbalanço", "EN": "ss_desbalanco"},
    "difference pairs": {"PT": "pares de diferença", "EN": "difference pairs"},
    "Total variance fraction removed": {"PT": "Fração da variância total removida", "EN": "Total variance fraction removed"},
    "iterations": {"PT": "iterações", "EN": "iterations"},
    "CSV file": {"PT": "Arquivo CSV", "EN": "CSV file"},
    "Input mode": {"PT": "Modo de entrada", "EN": "Input mode"},
    "Folder (or CSV file)": {"PT": "Pasta (ou arquivo CSV)", "EN": "Folder (or CSV file)"},
    "Fill in both folders/files.": {"PT": "Preencha as duas pastas/arquivos.", "EN": "Fill in both folders/files."},
    "Error loading data": {"PT": "Erro ao carregar os dados", "EN": "Error loading data"},
    "of total variance": {"PT": "da variância total", "EN": "of total variance"},
    "Target component x reference correlation": {"PT": "Correlação componente-alvo × referência", "EN": "Target component x reference correlation"},
    "Block": {"PT": "Bloco", "EN": "Block"},
    "samples": {"PT": "amostras", "EN": "samples"},
    "variables": {"PT": "variáveis", "EN": "variables"},
    "Error loading": {"PT": "Erro ao carregar", "EN": "Error loading"},
    "Project name": {"PT": "Nome do projeto", "EN": "Project name"},
    "Author(s)": {"PT": "Autor(es)", "EN": "Author(s)"},
    "Institution / Laboratory": {"PT": "Instituição / Laboratório", "EN": "Institution / Laboratory"},
    "Objective": {"PT": "Objetivo", "EN": "Objective"},
    "💻 Hardware Status": {"PT": "💻 Estado do hardware", "EN": "💻 Hardware Status"},
    "CPU": {"PT": "CPU", "EN": "CPU"},
    "Free disk": {"PT": "Disco livre", "EN": "Free disk"},
    "⚠️ psutil not available — approximate readings. Install with `pip install psutil`.": {"PT": "⚠️ psutil indisponível — leituras aproximadas. Instale com `pip install psutil`.", "EN": "⚠️ psutil not available — approximate readings. Install with `pip install psutil`."},
    "Hardware: could not detect hardware specifications.": {"PT": "Hardware: não foi possível detectar as especificações.", "EN": "Hardware: could not detect hardware specifications."},
    "ℹ️ Configure the options below, then click **▶️ Run pipeline**.": {"PT": "ℹ️ Configure as opções abaixo e clique em **▶️ Executar pipeline**.", "EN": "ℹ️ Configure the options below, then click **▶️ Run pipeline**."},
    "**Input:**": {"PT": "**Entrada:**", "EN": "**Input:**"},
    "🧮 Analysis & partitioning": {"PT": "🧮 Análise e partição", "EN": "🧮 Analysis & partitioning"},
    "📊 Statistical validation": {"PT": "📊 Validação estatística", "EN": "📊 Statistical validation"},
    "🧩 Extra modules": {"PT": "🧩 Módulos extras", "EN": "🧩 Extra modules"},
    "⚙️ Advanced parameters": {"PT": "⚙️ Parâmetros avançados", "EN": "⚙️ Advanced parameters"},
    "🖼️ Figures": {"PT": "🖼️ Figuras", "EN": "🖼️ Figures"},
    "Analysis mode": {"PT": "Modo de análise", "EN": "Analysis mode"},
    "A previous analysis is still running in the background. Wait for it to finish (or reload the page) before running again.": {"PT": "Uma análise anterior ainda está em execução em segundo plano. Aguarde ela terminar (ou recarregue a página) antes de rodar novamente.", "EN": "A previous analysis is still running in the background. Wait for it to finish (or reload the page) before running again."},
    "✅ Completed! View results in the Validation and Reports tabs.": {"PT": "✅ Concluído! Veja os resultados nas telas Validação e Relatórios.", "EN": "✅ Completed! View results in the Validation and Reports tabs."},
    "Error traceback": {"PT": "Detalhes técnicos do erro", "EN": "Error traceback"},
    "📊 Figures and metrics generated by the last pipeline run (Model tab).": {"PT": "📊 Figuras e métricas da última execução (tela Modelo).", "EN": "📊 Figures and metrics generated by the last pipeline run (Model tab)."},
    "Filter by category": {"PT": "Filtrar por categoria", "EN": "Filter by category"},
    "Figures saved in non-displayable format (PDF/SVG). Use the download button in the Reports tab.": {"PT": "Figuras salvas em formato não exibível (PDF/SVG). Use o botão de download na tela Relatórios.", "EN": "Figures saved in non-displayable format (PDF/SVG). Use the download button in the Reports tab."},
    "📋 Model summary (resumo_modelo.txt)": {"PT": "📋 Resumo do modelo (resumo_modelo.txt)", "EN": "📋 Model summary (resumo_modelo.txt)"},
    "🏅 Classifier Benchmark": {"PT": "🏅 Comparação de classificadores", "EN": "🏅 Classifier Benchmark"},
    "📊 Accuracy by Class": {"PT": "📊 Acurácia por classe", "EN": "📊 Accuracy by Class"},
    "**Auto-Benchmark — Balanced Accuracy by model (GroupKFold)**": {"PT": "**Comparação automática — acurácia balanceada por modelo (GroupKFold)**", "EN": "**Auto-Benchmark — Balanced Accuracy by model (GroupKFold)**"},
    "**Monte Carlo CV — 95% CI by percentile**": {"PT": "**CV Monte Carlo — IC 95% por percentil**", "EN": "**Monte Carlo CV — 95% CI by percentile**"},
    "📋 Methodological Notes (for peer review)": {"PT": "📋 Notas metodológicas (para revisão por pares)", "EN": "📋 Methodological Notes (for peer review)"},
    "🌐 Language": {"PT": "🌐 Idioma", "EN": "🌐 Language"},
    "⚠️ Completed, but with CRITICAL design problems — do not cite the metrics before reading them:": {"PT": "⚠️ Concluído, mas com problemas CRÍTICOS de delineamento — não cite as métricas antes de lê-los:", "EN": "⚠️ Completed, but with CRITICAL design problems — do not cite the metrics before reading them:"},
    "Free RAM: {ram} GB — Benchmark and SHAP will be switched off automatically to avoid freezing. The scientific results do not change.": {"PT": "RAM livre: {ram} GB — Benchmark e SHAP serão desligados automaticamente para evitar travamento. Os resultados científicos não mudam.", "EN": "Free RAM: {ram} GB — Benchmark and SHAP will be switched off automatically to avoid freezing. The scientific results do not change."},
    "Free RAM: {ram} GB — SHAP and Monte Carlo CV will be reduced automatically. The scientific results do not change.": {"PT": "RAM livre: {ram} GB — SHAP e CV Monte Carlo serão reduzidos automaticamente. Os resultados científicos não mudam.", "EN": "Free RAM: {ram} GB — SHAP and Monte Carlo CV will be reduced automatically. The scientific results do not change."},
    "No data loaded yet — on the **Data** screen, upload a CSV, or just run the synthetic demo in **Model**.": {"PT": "Nenhum dado carregado ainda — na tela **Dados**, envie um CSV, ou apenas rode a demonstração sintética em **Modelo**.", "EN": "No data loaded yet — on the **Data** screen, upload a CSV, or just run the synthetic demo in **Model**."},
    "No data loaded yet — on the **Data** tab, upload a CSV (or run the synthetic demo in **Model**). This panel fills in with the real numbers as soon as there is data.": {"PT": "Nenhum dado carregado ainda — na aba **Dados**, envie um CSV (ou rode a demonstração sintética em **Modelo**). Este painel mostra os números reais assim que houver dados.", "EN": "No data loaded yet — on the **Data** tab, upload a CSV (or run the synthetic demo in **Model**). This panel fills in with the real numbers as soon as there is data."},
    "🔒 Online version: only the portable model (`modelo_plsda.portatil.json`, saved next to every trained model) is accepted, because it cannot run code. The full model (`.joblib`: blind flow, drift sentinel) needs the version installed on your own computer (`guaraci web`).": {"PT": "🔒 Versão online: só o modelo portátil (`modelo_plsda.portatil.json`, salvo ao lado de todo modelo treinado) é aceito, porque ele não executa código. O modelo completo (`.joblib`: fluxo cego, sentinela de deriva) precisa da versão instalada no seu computador (`guaraci web`).", "EN": "🔒 Online version: only the portable model (`modelo_plsda.portatil.json`, saved next to every trained model) is accepted, because it cannot run code. The full model (`.joblib`: blind flow, drift sentinel) needs the version installed on your own computer (`guaraci web`)."},
    "Model loaded: classes {cls}; spectral range {a:.0f}–{b:.0f}; applicability domain: {ad}.": {"PT": "Modelo carregado: classes {cls}; faixa espectral {a:.0f}–{b:.0f}; domínio de aplicabilidade: {ad}.", "EN": "Model loaded: classes {cls}; spectral range {a:.0f}–{b:.0f}; applicability domain: {ad}."},
    "yes": {"PT": "sim", "EN": "yes"},
    "no (old file)": {"PT": "não (arquivo antigo)", "EN": "no (old file)"},
    "No figures found for '{f}'.": {"PT": "Nenhuma figura encontrada para '{f}'.", "EN": "No figures found for '{f}'."},
    "Prediction complete: {n} samples.": {"PT": "Predição concluída: {n} amostras.", "EN": "Prediction complete: {n} samples."},
    "Prediction error: {e}": {"PT": "Erro de predição: {e}", "EN": "Prediction error: {e}"},
    # ── Reports tab ──────────────────────────────────────────────────
    "Reports and Downloads": {"PT": "Relatórios e Downloads", "EN": "Reports and Downloads"},
    "📄 Download reports (ZIP/PDF/Word/Excel/LaTeX/PowerPoint), browse the figure gallery, and clean up old result folders.": {"PT": "📄 Baixe relatórios (ZIP/PDF/Word/Excel/LaTeX/PowerPoint), navegue pela galeria de figuras e limpe pastas de resultados antigas.", "EN": "📄 Download reports (ZIP/PDF/Word/Excel/LaTeX/PowerPoint), browse the figure gallery, and clean up old result folders."},
    "### ⬇️ Downloads": {"PT": "### ⬇️ Downloads", "EN": "### ⬇️ Downloads"},
    "### 🪪 Model Card": {"PT": "### 🪪 Model Card", "EN": "### 🪪 Model Card"},
    "### 📋 Model summary": {"PT": "### 📋 Resumo do modelo", "EN": "### 📋 Model summary"},
    "### 🖼️ Figure gallery": {"PT": "### 🖼️ Galeria de figuras", "EN": "### 🖼️ Figure gallery"},
    "Run the pipeline (Model tab) to generate reports.": {"PT": "Execute o pipeline (aba Modelo) para gerar relatórios.", "EN": "Run the pipeline (Model tab) to generate reports."},
    "Preparing report files (cached after the first time)...": {"PT": "Preparando arquivos de relatório (fica em cache depois da primeira vez)...", "EN": "Preparing report files (cached after the first time)..."},
    "🗑️ Free space — Clean up old results": {"PT": "🗑️ Liberar espaço — Limpar resultados antigos", "EN": "🗑️ Free space — Clean up old results"},
    "⬇️ Model Card (.md)": {"PT": "⬇️ Model Card (.md)", "EN": "⬇️ Model Card (.md)"},
    "File model_card.md not found.": {"PT": "Arquivo model_card.md não encontrado.", "EN": "File model_card.md not found."},
    "File resumo_modelo.txt not found.": {"PT": "Arquivo resumo_modelo.txt não encontrado.", "EN": "File resumo_modelo.txt not found."},
    "Filter figures": {"PT": "Filtrar figuras", "EN": "Filter figures"},
    "Columns": {"PT": "Colunas", "EN": "Columns"},
    "No PNG/JPG images found in the results folder.": {"PT": "Nenhuma imagem PNG/JPG encontrada na pasta de resultados.", "EN": "No PNG/JPG images found in the results folder."},
    "Results folder: `{pasta}`": {"PT": "Pasta de resultados: `{pasta}`", "EN": "Results folder: `{pasta}`"},
    "Keep N most recent runs": {"PT": "Manter N execuções mais recentes", "EN": "Keep N most recent runs"},
    "🗑️ Confirm cleanup": {"PT": "🗑️ Confirmar limpeza", "EN": "🗑️ Confirm cleanup"},
    "Only one run stored. Nothing to clean.": {"PT": "Só há uma execução armazenada. Nada para limpar.", "EN": "Only one run stored. Nothing to clean."},
    "View Model Card": {"PT": "Ver Model Card", "EN": "View Model Card"},
    "Show only figures of one analysis type (e.g. PCA scores, confusion matrix, DD-SIMCA acceptance). 'All' shows every figure generated by the run.": {"PT": "Mostra só figuras de um tipo de análise (ex.: scores de PCA, matriz de confusão, aceitação DD-SIMCA). 'All' mostra todas as figuras geradas pela execução.", "EN": "Show only figures of one analysis type (e.g. PCA scores, confusion matrix, DD-SIMCA acceptance). 'All' shows every figure generated by the run."},
    "📜 Execution log (terminal output)": {"PT": "📜 Log de execução (saída do terminal)", "EN": "📜 Execution log (terminal output)"},
    "📦 Full results (.zip)": {"PT": "📦 Resultados completos (.zip)", "EN": "📦 Full results (.zip)"},
    "📄 PDF Report": {"PT": "📄 Relatório PDF", "EN": "📄 PDF Report"},
    "📝 Word Report (.docx)": {"PT": "📝 Relatório Word (.docx)", "EN": "📝 Word Report (.docx)"},
    "📊 Data in Excel (.xlsx)": {"PT": "📊 Dados em Excel (.xlsx)", "EN": "📊 Data in Excel (.xlsx)"},
    "🔬 LaTeX Template (Talanta / Food Chemistry / J. Chemom.)": {"PT": "🔬 Template LaTeX (Talanta / Food Chemistry / J. Chemom.)", "EN": "🔬 LaTeX Template (Talanta / Food Chemistry / J. Chemom.)"},
    "🎯 PowerPoint Presentation (.pptx)": {"PT": "🎯 Apresentação PowerPoint (.pptx)", "EN": "🎯 PowerPoint Presentation (.pptx)"},
    "Results folder: `{pasta}`  ({n} runs stored)": {"PT": "Pasta de resultados: `{pasta}`  ({n} execuções armazenadas)", "EN": "Results folder: `{pasta}`  ({n} runs stored)"},
    "{n} figure(s) displayed.": {"PT": "{n} figura(s) exibida(s).", "EN": "{n} figure(s) displayed."},
    "python-pptx not installed. Run: `pip install python-pptx>=1.1`": {"PT": "python-pptx não instalado. Rode: `pip install python-pptx>=1.1`", "EN": "python-pptx not installed. Run: `pip install python-pptx>=1.1`"},
    "**{n_remover}** old run(s) will be removed (~{tam_est:.0f} MB freed). The current run **will not be affected**.": {"PT": "**{n_remover}** execução(ões) antiga(s) será(ão) removida(s) (~{tam_est:.0f} MB liberados). A execução atual **não será afetada**.", "EN": "**{n_remover}** old run(s) will be removed (~{tam_est:.0f} MB freed). The current run **will not be affected**."},
    "No folders removed.": {"PT": "Nenhuma pasta removida.", "EN": "No folders removed."},
    "ZIP: {e}": {"PT": "ZIP: {e}", "EN": "ZIP: {e}"},
    "PDF: {e}": {"PT": "PDF: {e}", "EN": "PDF: {e}"},
    "Word: {e}": {"PT": "Word: {e}", "EN": "Word: {e}"},
    "Excel: {e}": {"PT": "Excel: {e}", "EN": "Excel: {e}"},
    "LaTeX: {e}": {"PT": "LaTeX: {e}", "EN": "LaTeX: {e}"},
    "PowerPoint: {e}": {"PT": "PowerPoint: {e}", "EN": "PowerPoint: {e}"},
    "Removed {n} folder(s), freed {mb:.0f} MB.": {"PT": "Removida(s) {n} pasta(s), {mb:.0f} MB liberados.", "EN": "Removed {n} folder(s), freed {mb:.0f} MB."},
    "Errors: {erro}": {"PT": "Erros: {erro}", "EN": "Errors: {erro}"},
    "⚠️ No group key: validation will NOT be grouped (StratifiedKFold). Replicates of the same sample may fall on both sides of a split, so treat the metrics as exploratory. Fix: in the Data screen, set 'group column' or upload a group file.": {"PT": "⚠️ Sem chave de grupo: a validação NÃO será agrupada (StratifiedKFold). Réplicas da mesma amostra podem cair dos dois lados de uma divisão; trate as métricas como exploratórias. Solução: na tela Dados, defina a 'coluna de grupo' ou envie um arquivo de grupos.", "EN": "⚠️ No group key: validation will NOT be grouped (StratifiedKFold). Replicates of the same sample may fall on both sides of a split, so treat the metrics as exploratory. Fix: in the Data screen, set 'group column' or upload a group file."},
    'Validation grouped by the group you declared — guarantee MEDIUM (the software cannot verify that groups are independent).': {"PT": 'Validação agrupada pelo grupo que você declarou — garantia MÉDIA (o software não consegue verificar que os grupos são independentes).', "EN": 'Validation grouped by the group you declared — guarantee MEDIUM (the software cannot verify that groups are independent).'},
    'Group file (optional) — CSV: sample id, group': {"PT": 'Arquivo de grupos (opcional) — CSV: id da amostra, grupo', "EN": 'Group file (optional) — CSV: sample id, group'},
    'Replicates of the same physical sample must share the group; it keeps them on the same side of every split.': {"PT": 'Réplicas da mesma amostra física devem compartilhar o grupo; ele mantém todas do mesmo lado de cada divisão.', "EN": 'Replicates of the same physical sample must share the group; it keeps them on the same side of every split.'},
    '**1. Portable model (.json)**': {"PT": '**1. Modelo portátil (.json)**', "EN": '**1. Portable model (.json)**'},
    "Check 'I trust the source of this model file' above before loading — required (see `guaraci manual seguranca`).": {"PT": "Marque 'Confio na origem deste arquivo de modelo' acima antes de carregar — obrigatório (ver `guaraci manual seguranca`).", "EN": "Check 'I trust the source of this model file' above before loading — required (see `guaraci manual seguranca`)."},
    'File received: `{nome}`': {"PT": 'Arquivo recebido: `{nome}`', "EN": 'File received: `{nome}`'},
    'I trust the source of this model file (required to load it — `.joblib` executes code when loaded, see `guaraci manual seguranca`)': {"PT": 'Confio na origem deste arquivo de modelo (obrigatório para carregar — `.joblib` executa código ao ser carregado, ver `guaraci manual seguranca`)', "EN": 'I trust the source of this model file (required to load it — `.joblib` executes code when loaded, see `guaraci manual seguranca`)'},
    'Load a model generated by the pipeline and a CSV with new spectra (columns = wavenumbers, no class column).': {"PT": 'Carregue um modelo gerado pelo pipeline e um CSV com espectros novos (colunas = números de onda, sem coluna de classe).', "EN": 'Load a model generated by the pipeline and a CSV with new spectra (columns = wavenumbers, no class column).'},
    'Local path to model': {"PT": 'Caminho local do modelo', "EN": 'Local path to model'},
    "Mode automatically set to 'csv'.": {"PT": "Modo ajustado automaticamente para 'csv'.", "EN": "Mode automatically set to 'csv'."},
    'Public server limit applied (run locally with `guaraci web` for larger values):': {"PT": 'Limite do servidor público aplicado (rode localmente com `guaraci web` para valores maiores):', "EN": 'Public server limit applied (run locally with `guaraci web` for larger values):'},
    'No valid model provided (upload).': {"PT": 'Nenhum modelo válido enviado.', "EN": 'No valid model provided (upload).'},
    'The file is not a GUARACI portable model.': {"PT": 'O arquivo não é um modelo portátil do GUARACI.', "EN": 'The file is not a GUARACI portable model.'},
    'The file is not a valid portable model (JSON).': {"PT": 'O arquivo não é um modelo portátil válido (JSON).', "EN": 'The file is not a valid portable model (JSON).'},
    'Upload the portable model (.portatil.json)': {"PT": 'Envie o modelo portátil (.portatil.json)', "EN": 'Upload the portable model (.portatil.json)'},
    '⚠️ Only load `.joblib` models you generated yourself. A model file is a pickle and **runs code when loaded** — never load one from an untrusted source. The manifest signature is checked first; a model from another machine must be signed explicitly.': {"PT": '⚠️ Carregue só modelos `.joblib` que você mesmo gerou. Um arquivo de modelo é um pickle e **executa código ao ser carregado** — nunca carregue um de origem não confiável. A assinatura do manifesto é conferida primeiro; um modelo de outra máquina precisa ser assinado explicitamente.', "EN": '⚠️ Only load `.joblib` models you generated yourself. A model file is a pickle and **runs code when loaded** — never load one from an untrusted source. The manifest signature is checked first; a model from another machine must be signed explicitly.'},
    '⬇️ Download config.yaml': {"PT": '⬇️ Baixar config.yaml', "EN": '⬇️ Download config.yaml'},
    '🔒 Server mode: only the portable JSON model is accepted (it cannot execute code). It is written next to every trained model as `modelo_plsda.portatil.json`. `.joblib` models (blind flow, applicability domain, drift sentinel) need the local app: `GUARACI_LOCAL_MODE=1`.': {"PT": '🔒 Modo servidor: só o modelo portátil em JSON é aceito (ele não executa código). Ele é gravado ao lado de todo modelo treinado como `modelo_plsda.portatil.json`. Modelos `.joblib` (fluxo cego, domínio de aplicabilidade, sentinela de deriva) exigem o app local: `GUARACI_LOCAL_MODE=1`.', "EN": '🔒 Server mode: only the portable JSON model is accepted (it cannot execute code). It is written next to every trained model as `modelo_plsda.portatil.json`. `.joblib` models (blind flow, applicability domain, drift sentinel) need the local app: `GUARACI_LOCAL_MODE=1`.'},
}

def _T(key: str) -> str:
    lang = st.session_state.get("lang", "EN")
    return _TR.get(key, {}).get(lang, key)


# Rótulos amigáveis para o "nivel" de análise (N1/N2/N3). O valor interno
# gravado continua sendo N1/N2/N3 — isto só troca o que o usuário vê/escolhe.
_MODO_ANALISE_ROTULO = {
    "N1": "Classificação (por espécie)",
    "N2": "Discriminação (puro vs. adulterado)",
    "N3": "Quantificação (% de adulterante)",
}
_MODO_ANALISE_AJUDA = {
    "N1": "Identifica a qual espécie/classe cada amostra pertence "
          "(ex.: 14 classes).",
    "N2": "Separa amostras puras de adulteradas (autenticação).",
    "N3": "Estima o teor de adulterante (% ) por regressão.",
}

# ──────────────────────────────────────────────────────────────────────────
# Config helpers (_CONFIG_SPEC as single source of truth)
# ──────────────────────────────────────────────────────────────────────────

def _spec_por_key() -> Dict:
    cfg_spec = getattr(pq, "_CONFIG_SPEC", None)
    if cfg_spec is None:
        raise RuntimeError(
            "pq.CONFIG_SPEC not found — pipeline module did not load fully. "
            "Restart the app or check the Streamlit Cloud logs."
        )
    return {s["key"]: s for s in cfg_spec}


# Rótulos amigáveis para campos "choice" onde o valor interno gravado no
# config (ex.: "puros"/"todos") não é autoexplicativo por si só — só troca
# o que aparece no selectbox, o valor salvo continua o código interno.
_ROTULOS_OPCAO: Dict[str, Dict[str, str]] = {
    "modo_ddsimca": {
        "puros": "Somente puras (autenticação — resto = contaminante)",
        "todos": "Todas as amostras (exploratório)",
    },
}


def _rotulo_perfil(op: str) -> str:
    """Rotulo dinamico p/ perfil_matriz/perfil_tecnica -- mesma logica de
    `_rotulo_opcao` da CLI (guaraci.py): descricao + selo de cobertura
    (`referencia` nao-vazia = perfil com referencia declarada).
    Duplicado deliberadamente (nao importa guaraci.py aqui -- CLI e web sao
    interfaces separadas por design, ver Fase H) mas usa a MESMA fonte de
    dado (`perfil_matriz.load_profile`), nunca reinventa o julgamento.
    """
    if not op:
        return "(não declarado)"
    try:
        from guaraci.perfil_matriz import load_profile
        p = load_profile(op)
    except Exception:  # noqa: BLE001 -- rotulo e' so' exibicao.
        return op
    desc = (p.descricao or "").split(" (")[0].strip()
    selo = ""
    if op != "generico":
        selo = "  ✅" if p.referencia else "  ⚠ não validado"
    extra = ""
    if p.nivel_agrupamento_tipico:
        extra = f" [garantia típica: {p.nivel_agrupamento_tipico}]"
    return f"{op} — {desc}{extra}{selo}" if desc else f"{op}{selo}"


def _widget_para_campo(s: Dict, valor_atual, prefixo: str = "w_"):
    """Renders ONE widget according to field type and returns current value."""
    chave = prefixo + s["key"]
    # Use a short label from desc if available, otherwise humanize the key
    # Rotulo e ajuda vem do catalogo (guaraci.catalogo): nome curto + gabarito
    # Faz / Use quando / Nao use quando; fallback = chave humanizada.
    _lang = "PT" if st.session_state.get("lang") != "EN" else "EN"
    if s["key"] in _catalogo.OPCOES:
        rotulo = _catalogo.opcao(s["key"])["nome"][1 if _lang == "EN" else 0]
        ajuda = _catalogo.ajuda_texto(s["key"], _lang)
    else:
        rotulo = s["key"].replace("_", " ").capitalize()
        ajuda = s.get("desc", "")
    t = s["tipo"]
    if s["key"] in _CAMPOS_CAMINHO_SERVIDOR and _MODO_SERVIDOR:
        st.caption(
            f"🔒 {rotulo}: "
            + ("gerenciado pelo servidor — seus arquivos ficam numa pasta "
               "isolada desta sessão (envie o CSV acima)."
               if st.session_state.get("lang") == "PT" else
               "managed by the server — your files live in an isolated "
               "folder of this session (upload the CSV above)."))
        return valor_atual
    if t == "bool":
        return st.checkbox(rotulo, value=bool(valor_atual), help=ajuda, key=chave)
    if t in ("choice", "preproc"):
        ops = list(s.get("opcoes") or [])
        if s["key"] == "modo_entrada" and _MODO_SERVIDOR:
            ops = [o for o in ops if o in _MODOS_ENTRADA_SERVIDOR]
        idx = ops.index(valor_atual) if valor_atual in ops else 0
        if s["key"] in ("perfil_matriz", "perfil_tecnica"):
            return st.selectbox(rotulo, ops, index=idx, help=ajuda, key=chave,
                                format_func=_rotulo_perfil)
        _rot = _ROTULOS_OPCAO.get(s["key"])
        if _rot:
            return st.selectbox(rotulo, ops, index=idx, help=ajuda, key=chave,
                                format_func=lambda v: _rot.get(v, v))
        return st.selectbox(rotulo, ops, index=idx, help=ajuda, key=chave)
    if t == "int":
        _lo = s.get("min"); _hi = s.get("max")
        # Clampa o valor inicial p/ dentro de [min,max]: um config.yaml antigo
        # com valor fora da faixa faria o st.number_input LANÇAR exceção.
        _v = int(valor_atual)
        if _lo is not None: _v = max(_v, int(_lo))
        if _hi is not None: _v = min(_v, int(_hi))
        return st.number_input(
            rotulo, value=_v, step=1, help=ajuda, key=chave,
            min_value=int(_lo) if _lo is not None else None,
            max_value=int(_hi) if _hi is not None else None)
    if t == "float":
        _flo = s.get("min"); _fhi = s.get("max")
        _fv = float(valor_atual)
        if _flo is not None: _fv = max(_fv, float(_flo))
        if _fhi is not None: _fv = min(_fv, float(_fhi))
        return st.number_input(
            rotulo, value=_fv, help=ajuda, key=chave, format="%.4f",
            min_value=float(_flo) if _flo is not None else None,
            max_value=float(_fhi) if _fhi is not None else None)
    if t in ("list", "recorte"):
        txt = ", ".join(str(x) for x in (valor_atual or ()))
        return st.text_input(rotulo + " (comma-separated)", value=txt,
                             help=ajuda, key=chave)
    return st.text_input(rotulo, value=str(valor_atual), help=ajuda, key=chave)


# _coletar_config foi movida para guaraci.app_logic (item 19); cada aba a
# importa diretamente de lá agora (item 18), sem passar por app_quimiometria.py.

# ──────────────────────────────────────────────────────────────────────────
# File helpers
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _zip_da_pasta(pasta: str) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for raiz, _dirs, arqs in os.walk(pasta):
            for a in arqs:
                cam = os.path.join(raiz, a)
                z.write(cam, os.path.relpath(cam, os.path.dirname(pasta)))
    buf.seek(0)
    return buf


# Wrappers finos: logica pura em guaraci.app_logic (item 19), cache Streamlit aqui.
@st.cache_data(show_spinner=False, ttl=120)
def _listar_figuras(pasta: str) -> List[str]:
    return _listar_figuras_pura(pasta)


@st.cache_data(show_spinner=False, ttl=120)
def _ler_resumo(pasta: str) -> Optional[str]:
    return _ler_resumo_pura(pasta)


@st.cache_data(show_spinner=False, ttl=120)
def _ler_model_card(pasta: str) -> Optional[str]:
    return _ler_model_card_pura(pasta)


# _RE_ETAPA/_ETAPA_NOMES/_ETAPA_SUBSTEP/_progresso_do_log/_fmt_tempo (item 19)
# e _LogThreadSafe/_ram_mb/_rodar_worker (item 18) moraram em
# guaraci.app_tabs.modelo, que os importa/redefine diretamente — usados
# apenas pela aba Model.


# _preview_espectros_dx/_csv + _plot_espectros_media movidos para
# guaraci.spectra_preview (item 18: usados pelas abas Data e Preprocessing).

# ──────────────────────────────────────────────────────────────────────────
# Prediction on unknown samples
# ──────────────────────────────────────────────────────────────────────────
# Extracted to predicao.py (shared with the CLI's batch prediction menu) —
# same Fase H pattern: move once, reexport by name, never duplicate.


# ──────────────────────────────────────────────────────────────────────────
# Initial state
# ──────────────────────────────────────────────────────────────────────────

# Sessão: id aleatório + guarda que apaga a pasta da sessão quando o estado do
# Streamlit é descartado (fim da sessão). Varredura por idade cobre reinícios.
if "_sessao_id" not in st.session_state:
    st.session_state["_sessao_id"] = _ws.nova_sessao()
    st.session_state["_sessao_guarda"] = _ws.GuardaSessao(
        st.session_state["_sessao_id"])
    if _MODO_SERVIDOR:
        _ws.limpar_sessoes_antigas()
_SESSAO_ID: str = st.session_state["_sessao_id"]

# Só existe config.yaml compartilhado no modo local. No modo servidor a
# configuração é estado DA SESSÃO (nunca lida de/gravada em disco compartilhado).
_CFG_PATH: Optional[str] = _CFG_PATH_LOCAL if _MODO_LOCAL else None
_IS_PUBLIC_DEMO = _MODO_SERVIDOR


def _config_inicial():
    if _CFG_PATH and os.path.exists(_CFG_PATH):
        try:
            return pq.load_config(_CFG_PATH)
        except (RuntimeError, FileNotFoundError, ValueError):
            # load_config so' lanca esses 3 tipos (PyYAML ausente, arquivo
            # ausente, chaves invalidas) -- config.yaml quebrado nunca impede o
            # primeiro carregamento do app, cai para o modo demo sintetico.
            pass
    cfg = pq.Config(mode="sintetico")
    if _MODO_SERVIDOR:
        # Resultados numa pasta isolada da sessão (nunca relativa ao cwd do
        # servidor, nunca compartilhada entre visitantes).
        cfg.output_root_folder = str(_ws.pasta_sessao(_SESSAO_ID, "resultados"))
    return cfg


if "cfg_base" not in st.session_state:
    st.session_state.cfg_base = _config_inicial()

cfg_base = st.session_state.cfg_base

# Reset cfg_base if it's missing any field from the current Config
_fresh_cfg = pq.Config()
for _s in pq.CONFIG_SPEC:
    if not hasattr(cfg_base, _s["attr"]):
        cfg_base = pq.Config()
        st.session_state.cfg_base = cfg_base
        break
del _fresh_cfg

if _MODO_SERVIDOR:
    _ws.impor_politica_servidor(cfg_base, _SESSAO_ID)


specs    = _spec_por_key()

# ── Navegação: estado da tela atual ─────────────────────────────────────────
# A interface deixou de ser 8 abas horizontais e passou a barra lateral com
# grupos (mockup de 2026-09-08). Cada tela é renderizada sozinha, em vez de
# todas as abas serem montadas a cada rerun — o que também deixa a página
# mais leve, já que só o conteúdo visível é calculado.
if "pagina" not in st.session_state:
    st.session_state.pagina = app_nav.PAGINA_INICIAL


def _ir_para(chave: str) -> None:
    """Navega para outra tela. Usado pela barra lateral, pelos botões da
    barra superior e pelo botão de 'próxima ação sugerida' da tela Início."""
    st.session_state.pagina = chave
    st.rerun()

# ── Polimento visual (design tokens, à prova de tema) ────────────────────────
# NÃO pinta widgets internos do Streamlit (isso é papel do tema nativo, que
# garante consistência ao trocar claro/escuro). Só adiciona "chrome" de cartão
# neutro (cinza translúcido, válido nos dois temas) e realce de marca no header.
_tk = _tok()
# M-18b: o texto da barra lateral e' branco; cada ponto do degrade e' escurecido
# (mesmo matiz) ate' dar 4,5:1 (WCAG AA) -- o dourado puro dava 1,63:1.
from guaraci.design_tokens import escurecer_para_contraste as _esc
_grad = {k: _esc(_tk[k]) for k in ('primary', 'accent', 'success')}
st.markdown(f"""
<style>
:root {{ --gua-primary: {_tk['primary']}; --gua-accent: {_tk['accent']}; }}
/* 3,4rem e não 2,2rem: a barra superior própria (título + ações) precisa
   começar ABAIXO da barra fixa do Streamlit (Deploy / ⋮), senão o primeiro
   elemento da linha fica cortado -- visto no tema claro em 2026-09-08. */
.block-container {{ padding-top: 3.4rem; max-width: 1400px; }}
/* KPIs / métricas como cartões */
[data-testid="stMetric"] {{
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 12px; padding: 14px 18px;
    background: rgba(128,128,128,.045);
}}
/* Figuras científicas = "papel" branco emoldurado (intencional em qualquer tema) */
[data-testid="stImage"] img {{
    background: #ffffff; padding: 10px; border-radius: 10px;
    border: 1px solid rgba(128,128,128,.22);
}}
/* Header / hero */
.gua-hero {{ display:flex; align-items:center; gap:14px; margin-bottom:.15rem; }}
.gua-hero .gua-logo {{ font-size: 3.4rem; line-height:1; }}
/* Fundo claro FIXO na moldura (não é widget do Streamlit, é elemento nosso —
   sem CSS sobre componente nativo, que foi o que quebrou ao trocar tema
   antes). Medido em 2026-09-08: a tinta mais escura da logo (#0E3724, base
   do frasco) tem contraste 1,40:1 contra o fundo do tema escuro (#0F1613) —
   some. Sobre este fundo claro dá 12,4:1 nos dois temas. Mesma convenção já
   usada nas figuras científicas ("papel" branco, intencional em qualquer
   tema). */
.gua-hero .gua-logo-frame {{
    width: 96px; height: 96px; min-width: 96px; flex-shrink: 0;
    padding: 8px; box-sizing: border-box;
    background: #FDFDFD;
    border-radius: 20px; overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,.18);
    display: flex; align-items: center; justify-content: center;
}}
.gua-hero .gua-logo-img {{
    width: 100%; height: 100%; object-fit: contain; display: block;
}}
.gua-hero .gua-title {{
    font-size: 1.95rem; font-weight: 800; letter-spacing:-.02em; line-height:1.1;
    background: linear-gradient(90deg, var(--gua-primary), var(--gua-accent));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.gua-sub {{ color: {_tk['text_muted']}; font-size:.95rem; margin:.15rem 0 0; }}
.gua-badges {{ display:flex; gap:6px; margin:.35rem 0 0; flex-wrap:wrap; }}
.gua-badge {{
    font-size:.72rem; font-weight:600; padding:.15rem .55rem; border-radius:999px;
    border:1px solid rgba(128,128,128,.3); color: {_tk['text_muted']};
}}

/* ── Barra lateral com degradê (mockup de 2026-09-08) ────────────────────
   ATENÇÃO, decisão revertida: até 2026-09-07 a regra do projeto era "não
   pintar widget nativo do Streamlit", justamente porque uma tentativa
   anterior quebrou ao trocar de tema. O mockup novo foi pedido "à risca",
   então o degradê voltou — mas com duas salvaguardas que a tentativa
   antiga não tinha: (1) as cores vêm de `design_tokens` do tema ATIVO,
   lido de `st.context.theme`, então o degradê se redesenha ao trocar
   claro/escuro em vez de ficar preso à paleta de um tema só; (2) nenhum
   `!important` — o CSS só complementa o widget, não briga com ele. */
section[data-testid="stSidebar"] > div:first-child {{
    background: linear-gradient(180deg,
        {_grad['primary']} 0%, {_grad['accent']} 42%, {_grad['success']} 100%);
}}
section[data-testid="stSidebar"] * {{ color: #FFFFFF; }}
section[data-testid="stSidebar"] .stButton > button {{
    background: transparent; border: none; color: #FFFFFF;
    text-align: left; justify-content: flex-start;
    font-weight: 600; padding: .35rem .6rem;
    border-left: 3px solid transparent; border-radius: 6px;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255,255,255,.13);
}}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background: rgba(255,255,255,.24); border-left-color: #FFFFFF;
}}
/* O `summary` também precisa ficar transparente, não só o `details`: no tema
   CLARO o cabeçalho do expander vem branco por padrão e o rótulo (branco,
   para contrastar com o degradê) sumia dentro dele -- o grupo ① aparecia
   como uma caixa branca vazia. Visto na verificação de 2026-09-08. */
section[data-testid="stSidebar"] [data-testid="stExpander"] details,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {{
    border: none; background: transparent; background-color: transparent;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
    font-size: .72rem; font-weight: 800; letter-spacing: .08em;
    text-transform: uppercase; color: #FFFFFF;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {{
    background: rgba(255,255,255,.10);
}}
.gua-marca {{ text-align:center; padding:.2rem 0 .6rem; }}
.gua-marca .n {{ font-weight:800; font-size:.86rem; letter-spacing:.06em; }}
.gua-marca .v {{ font-size:.62rem; letter-spacing:.03em; }}

/* ── Barra superior (título da tela + ações) ─────────────────────────── */
.gua-topo h1 {{ font-size:1.15rem; font-weight:800; margin:0; }}
.gua-topo .sub {{ font-size:.78rem; color:{_tk['text_muted']}; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────

_logo_uri = _logo_data_uri()
_logo_html = (f'<div class="gua-logo-frame"><img class="gua-logo-img" '
              f'src="{_logo_uri}" alt="GUARACI"></div>'
              if _logo_uri else '<span class="gua-logo">🧪</span>')

_pt = st.session_state.lang == "PT"
_pagina_atual = st.session_state.pagina

# ──────────────────────────────────────────────────────────────────────────
# Barra lateral: marca + navegação agrupada
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div class="gua-hero" style="justify-content:center">{_logo_html}</div>'
        f'<div class="gua-marca"><div class="n">GUARACI</div>'
        f'<div class="v">v{pq.__version__} · local-first</div></div>',
        unsafe_allow_html=True)

    for _pag in app_nav.PAGINAS_FIXAS:
        if st.button(f"{_pag.icone}  {_pag.rotulo(_pt)}",
                     key=f"nav_{_pag.chave}", use_container_width=True,
                     type="primary" if _pagina_atual == _pag.chave else "secondary"):
            _ir_para(_pag.chave)

    # Selos com números REAIS ao lado do item (o mockup mostra "Dados 934";
    # aqui o número só aparece quando existe de fato).
    _previa = st.session_state.get("previa_dados") or {}
    _selos: Dict[str, str] = {}
    if _previa.get("n_espectros"):
        _selos["dados"] = str(_previa["n_espectros"])
    if st.session_state.get("pred_resultados_cego"):
        _selos["predicao"] = "cego" if _pt else "blind"
    if st.session_state.get("ultima_pasta"):
        _selos["relatorios"] = "✓"

    _grupo_atual = app_nav.grupo_da_pagina(_pagina_atual)
    for _grupo in app_nav.GRUPOS:
        _num = "①②③④⑤⑥⑦⑧⑨⑩"[_grupo.numero - 1]
        # Numa tela fixa (Início/Visualização) nenhum grupo contém a tela
        # atual; abre-se o primeiro, como no mockup, para o fluxo ficar à
        # vista em vez de tudo fechado.
        _aberto = (_grupo.numero == _grupo_atual.numero if _grupo_atual
                   else _grupo.numero == 1)
        with st.expander(f"{_num}  {_grupo.rotulo(_pt)}", expanded=_aberto):
            for _pag in _grupo.paginas:
                _selo = _selos.get(_pag.chave)
                _rot = f"{_pag.icone}  {_pag.rotulo(_pt)}"
                if _selo:
                    _rot += f"  ·  {_selo}"
                if st.button(_rot, key=f"nav_{_pag.chave}",
                             use_container_width=True,
                             type=("primary" if _pagina_atual == _pag.chave
                                   else "secondary")):
                    _ir_para(_pag.chave)

    st.markdown("---")
    _lang_choice = st.radio(
        _T("🌐 Language"), ["EN", "PT"],
        index=0 if st.session_state.lang == "EN" else 1,
        key="_sidebar_lang", horizontal=True)
    if _lang_choice != st.session_state.lang:
        st.session_state.lang = _lang_choice
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────
# Barra superior: título da tela, tema e ações rápidas
# ──────────────────────────────────────────────────────────────────────────
_pag_obj = app_nav.pagina_por_chave(_pagina_atual)
_titulo = _pag_obj.rotulo(_pt) if _pag_obj else ""
_subtitulo = app_nav.subtitulo(_pagina_atual, _pt)

_c_tit, _c_tema, _c_rel, _c_mod = st.columns([6, 2, 1.6, 1.6])
with _c_tit:
    st.markdown(
        f'<div class="gua-topo"><h1>{_titulo}</h1>'
        f'<div class="sub">{_subtitulo}</div></div>', unsafe_allow_html=True)
with _c_tema:
    # O mockup traz um interruptor Claro/Escuro. O Streamlit NÃO expõe API
    # para trocar o tema por código (`st.context.theme` é somente leitura) —
    # forçar por CSS foi exatamente o que quebrou antes. Então aqui fica o
    # estado real do tema e o caminho para trocá-lo, sem fingir um botão que
    # não funcionaria.
    _tema_ativo = _active_theme()
    st.caption(
        ("🌗 Tema: **claro**" if _tema_ativo == "light" else "🌗 Tema: **escuro**")
        if _pt else
        ("🌗 Theme: **light**" if _tema_ativo == "light" else "🌗 Theme: **dark**"))
    st.caption(_T("Switch in ⋮ → Settings → Theme"))
with _c_rel:
    if st.button(_T("Export report"), use_container_width=True,
                 key="btn_topo_relatorio"):
        _ir_para("relatorios")
with _c_mod:
    if st.button(_T("Go to Model →"), type="primary",
                 use_container_width=True, key="btn_topo_modelo"):
        _ir_para("modelo")

st.divider()

if _MODO_SERVIDOR and _pagina_atual == app_nav.PAGINA_INICIAL:
    st.info(
        "🔬 **Modo servidor** — o app abre com **espectros sintéticos** "
        "(nenhuma amostra real) e você pode enviar o seu CSV na tela Dados. "
        "Seus arquivos ficam numa pasta isolada desta sessão e são apagados "
        "quando ela termina; não há acesso a pastas do servidor nem "
        "carregamento de modelos `.joblib` (na predição só o modelo portátil "
        ".json). Para usar pastas locais, `.dx`, HSI e modelos `.joblib`, "
        "rode na sua máquina com `guaraci web` — ver "
        f"[repositório]({_tab_sobre._REPO})."
        if st.session_state.lang == "PT" else
        "🔬 **Server mode** — the app opens with **synthetic spectra** (no "
        "real samples) and you can upload your own CSV on the Data screen. "
        "Your files live in a folder isolated to this session and are deleted "
        "when it ends; there is no access to server folders and `.joblib` "
        "models cannot be loaded (prediction takes only the portable .json "
        "model). To use local folders, `.dx`, HSI and `.joblib` models, run "
        "on your own machine with `guaraci web` — see the "
        f"[repository]({_tab_sobre._REPO})."
    )

valores: Dict = {}  # accumulated by widgets from the Data/Preproc/Model screens


def _coletar_config_persistente(cfg, vals):
    """Aplica os valores dos widgets a `cfg` (copia); campos invalidos mantem o
    valor anterior (a propria tela ja exibe o erro)."""
    from guaraci.app_logic import collect_config
    novo, _erros = collect_config(cfg, vals)
    if _MODO_SERVIDOR:
        _ws.impor_politica_servidor(novo, _SESSAO_ID)
    return novo

# ==========================================================================
#  Report cache — avoids regenerating on every Streamlit rerun.
#  Wrappers return bytes (immutable, cacheable); BytesIO is created
#  at download_button time to guarantee cursor at position 0.
#  Geração real delegada a guaraci.reports (item 18: serviço extraído da UI).
# ==========================================================================
@st.cache_data(show_spinner=False)
def _pdf_bytes(pasta: str, proj_items: tuple) -> bytes:
    return reports.generate_pdf_report(pasta, dict(proj_items)).read()

@st.cache_data(show_spinner=False)
def _word_bytes(pasta: str, proj_items: tuple) -> bytes:
    return reports.generate_word_report(pasta, dict(proj_items)).read()

@st.cache_data(show_spinner=False)
def _excel_bytes(pasta: str) -> bytes:
    return reports.generate_excel_report(pasta).read()

@st.cache_data(show_spinner=False)
def _latex_bytes(pasta: str, proj_items: tuple) -> bytes:
    return reports.generate_latex_template(pasta, dict(proj_items))

@st.cache_data(show_spinner=False)
def _pptx_bytes(pasta: str, proj_items: tuple) -> bytes:
    return reports.generate_pptx_report(pasta, dict(proj_items)).read()


# ==========================================================================
#  Roteamento das telas
# ==========================================================================
# Só a tela escolhida é desenhada. Antes, `st.tabs` montava as 8 abas em
# todo rerun (inclusive as caras, como galeria de figuras e geração de
# relatório), mesmo com o usuário olhando uma só.
_pagina = st.session_state.pagina


def _desenhar_tela(_pagina: str) -> None:
    if _pagina == "inicio":
        _tab_inicio.render(pq, _T, _ir_para, _tok)
    elif _pagina == "visualizacao":
        _tab_visualizacao.render(_T, _tok)
    elif _pagina == "projeto":
        _tab_projeto.render(pq, _T, is_public_demo=_IS_PUBLIC_DEMO)
    elif _pagina == "dados":
        _tab_dados.render(pq, cfg_base, specs, valores, _widget_para_campo,
                          _CFG_PATH, _T)
    elif _pagina == "preprocessamento":
        _tab_preprocessamento.render(pq, cfg_base, specs, valores,
                                     _widget_para_campo, _T)
    elif _pagina == "modelo":
        _tab_modelo.render(pq, cfg_base, specs, valores, _T, _widget_para_campo,
                           _MODO_ANALISE_ROTULO, _MODO_ANALISE_AJUDA, _CFG_PATH)
    elif _pagina == "validacao":
        _tab_validacao.render(_T, _tok, _ler_resumo, _listar_figuras)
    elif _pagina == "predicao":
        _tab_predicao.render(_UPLOAD_MODELO_BLOQUEADO, _tok, _T)
    elif _pagina == "tecnicas":
        _tab_tecnicas.render(pq, cfg_base, _T)
    elif _pagina == "relatorios":
        _tab_relatorios.render(pq, _MODO_ANALISE_ROTULO, _zip_da_pasta,
                              _pdf_bytes, _word_bytes, _excel_bytes,
                              _latex_bytes, _pptx_bytes,
                              _ler_resumo, _ler_model_card, _listar_figuras, _T)
    elif _pagina == "sobre":
        _tab_sobre.render(pq, _T)
    else:
        # Chave desconhecida (ex.: sessão antiga com estado salvo): volta ao
        # Início em vez de renderizar uma tela em branco.
        st.session_state.pagina = app_nav.PAGINA_INICIAL
        st.rerun()


# Modo servidor (A1-03): exceção não tratada de qualquer tela vira mensagem
# amigável + id de erro; o traceback vai só para o log do servidor
# (`mensagem_visitante`). st.rerun/st.stop não são capturados (no Streamlit
# >= 1.54 herdam de BaseException). No modo local o erro segue como antes.
try:
    _desenhar_tela(_pagina)
except Exception as _exc:  # noqa: BLE001 -- fronteira da tela: nada vaza ao visitante
    if _MODO_LOCAL:
        raise
    _msg_erro, _ = _ws.mensagem_visitante(
        _exc, f"tela {_pagina}" if st.session_state.get("lang") == "PT" else f"screen {_pagina}",
        pt=st.session_state.get("lang") == "PT")
    st.error(_msg_erro)

# ── Persistência da configuração entre telas ──
# O Streamlit APAGA do session_state os widgets que não foram desenhados no
# rerun. Como só a tela atual é desenhada, o que o usuário escolheu em Dados ou
# Pré-processamento sumia ao ir para Modelo, e a execução usava a config
# antiga -- sem erro. Fonte da verdade agora é `cfg_base`: ao fim de cada tela
# de configuração, os valores dos widgets são aplicados nele. Os widgets são
# só uma vista; a Config que roda é exatamente a que as telas mostram.
if _pagina in ("dados", "preprocessamento", "modelo") and valores:
    st.session_state.cfg_base = _coletar_config_persistente(
        st.session_state.cfg_base, valores)
