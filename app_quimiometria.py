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

# Bootstrap do pacote: este arquivo e o ENTRY POINT do Streamlit (roda como
# script solto, `streamlit run app_quimiometria.py`), entao o pacote `guaraci`
# em ./src precisa entrar no path antes de qualquer `import guaraci.*`. Esta e
# a unica insercao de sys.path que resta no projeto — justificada por ser o
# ponto de entrada. Módulos internos do pacote usam imports absolutos limpos.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

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
from guaraci.app_tabs import relatorios as _tab_relatorios
from guaraci.app_tabs import sobre as _tab_sobre


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
_icon_path = Path(__file__).parent / "assets" / "guaraci_icon.png"
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
_AQUI = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_PATH = os.path.join(_AQUI, "pipeline.py")
_CFG_PATH = os.path.join(_AQUI, "config.yaml")


@st.cache_resource(show_spinner="Loading pipeline engine...")
def _carregar_motor():
    """Load the chemometrics pipeline module (guaraci.pipeline).

    Import de pacote normal — o bootstrap de ./src no topo do arquivo garante
    que `guaraci` esteja no path (inclusive no Streamlit Cloud). O sentinela
    `_CONFIG_SPEC` mantém uma mensagem de erro clara se algum import de nível
    de módulo falhar e deixar o pacote parcialmente carregado.
    """
    try:
        import guaraci.pipeline as _pq  # type: ignore[import]
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

# ── Segurança: upload de modelo (.joblib) ────────────────────────────────────
# joblib.load usa pickle, que EXECUTA código arbitrário DURANTE o load (RCE).
# A validação de estrutura (_validar_pacote_modelo) só roda DEPOIS de carregar,
# tarde demais para impedir a execução. Logo, aceitar upload de .joblib de
# origem desconhecida é um vetor de RCE — inaceitável num demo hospedado
# público. O operador do deploy público deve exportar
# GUARACI_DISABLE_MODEL_UPLOAD=1 para esconder o upload e aceitar apenas
# caminhos locais (controlados pelo próprio operador). O uso local single-user
# (máquina do pesquisador) mantém o upload habilitado por padrão.
_UPLOAD_MODELO_BLOQUEADO = os.getenv(
    "GUARACI_DISABLE_MODEL_UPLOAD", "").strip().lower() in ("1", "true", "yes", "on")

# Mesmo gate protege os campos de CAMINHO DE SERVIDOR (achado S-NOVO-1 da
# auditoria de seguranca de 2026-09-01): um text_input livre com caminho de
# pasta/arquivo, num app publico sem autenticacao, deixa QUALQUER visitante
# remoto enumerar diretorios do servidor (glob) ou ler o conteudo de
# qualquer arquivo de texto acessivel ao processo -- nao so' o operador que
# "sabe o caminho certo". Nao e' RCE (como o pickle), mas e' leitura
# arbitraria de arquivo/divulgacao de informacao real em deploy publico.
_CAMPOS_CAMINHO_SERVIDOR = {"pasta_dados", "arquivo_csv"}

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
    "Step 1: Fill project info": {"PT": "Passo 1: Preencha as informações do projeto", "EN": "Step 1: Fill project info"},
    "Step 2: Upload or select spectra folder": {"PT": "Passo 2: Faça upload ou selecione a pasta de espectros", "EN": "Step 2: Upload or select spectra folder"},
    "Step 3: Configure parameters and run": {"PT": "Passo 3: Configure os parâmetros e execute", "EN": "Step 3: Configure parameters and run"},
    # ── Data tab (Agente 6.2 pos-sessao: as 4 abas abaixo -- Data/
    # Preprocessing/Prediction/Reports -- nao passavam por T() nenhum,
    # ficavam sempre em ingles mesmo com idioma=PT selecionado. Fechado
    # aqui, 2026-09-01.) ──────────────────────────────────────────────
    "📂 Step 2: Upload or select spectra folder → then go to **Model** tab.": {"PT": "📂 Passo 2: Faça upload ou selecione a pasta de espectros → depois vá para a aba **Modelo**.", "EN": "📂 Step 2: Upload or select spectra folder → then go to **Model** tab."},
    "**🎯 Recommended analysis** *(optional shortcut)*": {"PT": "**🎯 Análise recomendada** *(atalho opcional)*", "EN": "**🎯 Recommended analysis** *(optional shortcut)*"},
    "Pick what you want to do — sets sensible defaults across all tabs; fine-tune afterwards if needed.": {"PT": "Escolha o que você quer fazer — define valores padrão sensatos em todas as abas; ajuste depois se precisar.", "EN": "Pick what you want to do — sets sensible defaults across all tabs; fine-tune afterwards if needed."},
    "**Upload CSV** *(alternative to the local path below)*": {"PT": "**Upload de CSV** *(alternativa ao caminho local abaixo)*", "EN": "**Upload CSV** *(alternative to the local path below)*"},
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
    "MSC (scatter correction) → 1st derivative SG (Savitzky-Golay) → Mean-Centering. **Best for FT-NIR with pronounced scatter.** Compare presets on your own data with `comparar_pre_processamentos`.": {"PT": "MSC (correção de espalhamento) → 1ª derivada SG (Savitzky-Golay) → Centralização pela média. **Melhor para FT-NIR com espalhamento pronunciado.** Compare presets nos seus próprios dados com `comparar_pre_processamentos`.", "EN": "MSC (scatter correction) → 1st derivative SG (Savitzky-Golay) → Mean-Centering. **Best for FT-NIR with pronounced scatter.** Compare presets on your own data with `comparar_pre_processamentos`."},
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
    "I trust the source of this model file (required to load it — `.joblib` executes code when loaded, see docs/SECURITY.md)": {"PT": "Eu confio na origem deste arquivo de modelo (obrigatório para carregar — `.joblib` executa código ao ser carregado, ver docs/SECURITY.md)", "EN": "I trust the source of this model file (required to load it — `.joblib` executes code when loaded, see docs/SECURITY.md)"},
    "🔒 Local path to CSV is disabled on this public deployment — a free-text server-side path would let any visitor read arbitrary files on the server. Use the upload above instead.": {"PT": "🔒 O caminho local do CSV está desabilitado neste deploy público — um caminho de servidor em texto livre permitiria que qualquer visitante lesse arquivos arbitrários do servidor. Use o upload acima.", "EN": "🔒 Local path to CSV is disabled on this public deployment — a free-text server-side path would let any visitor read arbitrary files on the server. Use the upload above instead."},
    "Or local path to CSV": {"PT": "Ou caminho local do CSV", "EN": "Or local path to CSV"},
    "**Blind flow — Detect → Identify → Quantify**": {"PT": "**Fluxo cego — Detectar → Identificar → Quantificar**", "EN": "**Blind flow — Detect → Identify → Quantify**"},
    "⚠ 'classe_identificada' only ever exists when 'identificacao_cobertura'='validado' (formal statistical guarantee, calibrated with ≥2 independent collection sessions) — there is no 'informational' label without that guarantee. 'identificacao_candidatos' carries only the closest guess (with NO guarantee at all) for reference, never as a result to act on for a quality decision without confirming by a reference method.": {"PT": "⚠ 'classe_identificada' só existe quando 'identificacao_cobertura'='validado' (garantia estatística formal, calibrada com ≥2 sessões de coleta independentes) — não há rótulo 'informativo' sem essa garantia. 'identificacao_candidatos' traz só o palpite mais próximo (SEM garantia nenhuma) para referência, nunca como resultado a usar numa decisão de qualidade sem confirmar por método de referência.", "EN": "⚠ 'classe_identificada' only ever exists when 'identificacao_cobertura'='validado' (formal statistical guarantee, calibrated with ≥2 independent collection sessions) — there is no 'informational' label without that guarantee. 'identificacao_candidatos' carries only the closest guess (with NO guarantee at all) for reference, never as a result to act on for a quality decision without confirming by a reference method."},
    "Check 'I trust the source of this model file' above before loading — required (see docs/SECURITY.md).": {"PT": "Marque 'Eu confio na origem deste arquivo de modelo' acima antes de carregar — obrigatório (ver docs/SECURITY.md).", "EN": "Check 'I trust the source of this model file' above before loading — required (see docs/SECURITY.md)."},
    "No spectra CSV provided.": {"PT": "Nenhum CSV de espectros fornecido.", "EN": "No spectra CSV provided."},
    "Within PLS-DA model fit": {"PT": "Dentro do ajuste do modelo PLS-DA", "EN": "Within PLS-DA model fit"},
    "Hotelling T² (distance along the model's main directions) and Q-residual (unexplained variation) both within their statistical limit — see the 'criterio' column for the exact rule applied. A sample outside this fit is an atypical spectrum for the calibration, not necessarily 'adulterated'.": {"PT": "Hotelling T² (distância ao longo das direções principais do modelo) e resíduo Q (variação não explicada), ambos dentro do limite estatístico — veja a coluna 'criterio' para a regra exata aplicada. Uma amostra fora deste ajuste é um espectro atípico para a calibração, não necessariamente 'adulterada'.", "EN": "Hotelling T² (distance along the model's main directions) and Q-residual (unexplained variation) both within their statistical limit — see the 'criterio' column for the exact rule applied. A sample outside this fit is an atypical spectrum for the calibration, not necessarily 'adulterated'."},
    "Within applicability domain": {"PT": "Dentro do domínio de aplicabilidade", "EN": "Within applicability domain"},
    "Exploratory PCA check (Hotelling T²/Q-residual, Jaworska et al. 2005) of similarity to the training set as a whole — a broader, less strict screen than the PLS-DA fit above. 'Outside' flags a spectrum unlike anything the model was calibrated on.": {"PT": "Checagem exploratória por PCA (Hotelling T²/resíduo Q, Jaworska et al. 2005) de similaridade com o conjunto de treino como um todo — uma triagem mais ampla e menos estrita que o ajuste PLS-DA acima. 'Fora' sinaliza um espectro diferente de tudo que o modelo foi calibrado para reconhecer.", "EN": "Exploratory PCA check (Hotelling T²/Q-residual, Jaworska et al. 2005) of similarity to the training set as a whole — a broader, less strict screen than the PLS-DA fit above. 'Outside' flags a spectrum unlike anything the model was calibrated on."},
    "🧪 Purity (predicted species)": {"PT": "🧪 Pureza (espécie prevista)", "EN": "🧪 Purity (predicted species)"},
    "🏷 Adulterant identified": {"PT": "🏷 Adulterante identificado", "EN": "🏷 Adulterant identified"},
    "UNKNOWN means no species×adulterant combination had a statistical guarantee both sufficient AND exclusive for this sample — either no guarantee, or 2+ validated combinations tied (also blocks the label).": {"PT": "DESCONHECIDO significa que nenhuma combinação espécie×adulterante teve garantia estatística ao mesmo tempo suficiente E exclusiva para esta amostra — ou não há garantia, ou 2+ combinações validadas empataram (o que também bloqueia o rótulo).", "EN": "UNKNOWN means no species×adulterant combination had a statistical guarantee both sufficient AND exclusive for this sample — either no guarantee, or 2+ validated combinations tied (also blocks the label)."},
    "⚖ Quantified": {"PT": "⚖ Quantificado", "EN": "⚖ Quantified"},
    "Blocked = quantification refused because the adulterant was not reliably identified — see 'quantificacao_motivo_bloqueio' in the table above.": {"PT": "Bloqueada = quantificação recusada porque o adulterante não foi identificado com confiabilidade — veja 'quantificacao_motivo_bloqueio' na tabela acima.", "EN": "Blocked = quantification refused because the adulterant was not reliably identified — see 'quantificacao_motivo_bloqueio' in the table above."},
    "No valid model provided (upload or path).": {"PT": "Nenhum modelo válido fornecido (upload ou caminho).", "EN": "No valid model provided (upload or path)."},
    "pure": {"PT": "pura", "EN": "pure"},
    "Error loading model: {e}": {"PT": "Erro ao carregar modelo: {e}", "EN": "Error loading model: {e}"},
    "Error reading CSV: {e}": {"PT": "Erro ao ler CSV: {e}", "EN": "Error reading CSV: {e}"},
    "Applying model...": {"PT": "Aplicando modelo...", "EN": "Applying model..."},
    "DD-SIMCA for the species predicted above. {n} detected as adulterated.": {"PT": "DD-SIMCA para a espécie prevista acima. {n} detectada(s) como adulterada(s).", "EN": "DD-SIMCA for the species predicted above. {n} detected as adulterated."},
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
          "(ex.: 14 óleos amazônicos).",
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
            "pq._CONFIG_SPEC not found — pipeline module did not load fully. "
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


def _widget_para_campo(s: Dict, valor_atual, prefixo: str = "w_"):
    """Renders ONE widget according to field type and returns current value."""
    chave = prefixo + s["key"]
    # Use a short label from desc if available, otherwise humanize the key
    _desc = s.get("desc", "") or ""
    # Take first sentence of desc (up to 50 chars) as label hint
    _short = _desc.split(".")[0][:50].strip() if _desc else ""
    rotulo = _short if len(_short) > 4 else s["key"].replace("_", " ").capitalize()
    ajuda = s.get("desc", "")
    t = s["tipo"]
    if s["key"] in _CAMPOS_CAMINHO_SERVIDOR and _UPLOAD_MODELO_BLOQUEADO:
        st.caption(
            f"🔒 {rotulo}: disabled on this public deployment — a free-text "
            "server-side path would let any visitor enumerate directories "
            "or read arbitrary files on the server. Use the CSV upload "
            "above instead."
        )
        return valor_atual
    if t == "bool":
        return st.checkbox(rotulo, value=bool(valor_atual), help=ajuda, key=chave)
    if t in ("choice", "preproc"):
        ops = list(s.get("opcoes") or [])
        idx = ops.index(valor_atual) if valor_atual in ops else 0
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
        _lo = s.get("min"); _hi = s.get("max")
        _v = float(valor_atual)
        if _lo is not None: _v = max(_v, float(_lo))
        if _hi is not None: _v = min(_v, float(_hi))
        return st.number_input(
            rotulo, value=_v, help=ajuda, key=chave, format="%.4f",
            min_value=float(_lo) if _lo is not None else None,
            max_value=float(_hi) if _hi is not None else None)
    if t == "list":
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

_IS_PUBLIC_DEMO = not os.path.exists(_CFG_PATH)

if "cfg_base" not in st.session_state:
    try:
        st.session_state.cfg_base = (
            pq.load_config(_CFG_PATH) if os.path.exists(_CFG_PATH)
            # No local config.yaml (e.g. public demo deploy): default to
            # synthetic data so first-time visitors get a working demo
            # instead of an empty "dados/" folder error.
            else pq.Config(mode="sintetico"))
    except (RuntimeError, FileNotFoundError, ValueError):
        # load_config so' lanca esses 3 tipos (PyYAML ausente, arquivo
        # ausente, chaves invalidas) -- config.yaml quebrado nunca impede o
        # primeiro carregamento do app, cai para o modo demo sintetico.
        st.session_state.cfg_base = pq.Config(mode="sintetico")

cfg_base = st.session_state.cfg_base

# Reset cfg_base if it's missing any field from the current Config
_fresh_cfg = pq.Config()
for _s in pq._CONFIG_SPEC:
    if not hasattr(cfg_base, _s["attr"]):
        cfg_base = pq.Config()
        st.session_state.cfg_base = cfg_base
        break
del _fresh_cfg

specs    = _spec_por_key()

# ── Sidebar: Language ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    _lang_choice = st.radio(
        "🌐 Language", ["EN", "PT"],
        index=0 if st.session_state.lang == "EN" else 1,
        key="_sidebar_lang", horizontal=True
    )
    if _lang_choice != st.session_state.lang:
        st.session_state.lang = _lang_choice
        st.rerun()
    st.caption(
        "🌗 " + ("Tema claro/escuro: menu ⋮ → Settings → Theme"
                 if st.session_state.lang == "PT"
                 else "Light/dark theme: ⋮ menu → Settings → Theme")
    )
    st.markdown("---")

# ── Polimento visual (design tokens, à prova de tema) ────────────────────────
# NÃO pinta widgets internos do Streamlit (isso é papel do tema nativo, que
# garante consistência ao trocar claro/escuro). Só adiciona "chrome" de cartão
# neutro (cinza translúcido, válido nos dois temas) e realce de marca no header.
_tk = _tok()
st.markdown(f"""
<style>
:root {{ --gua-primary: {_tk['primary']}; --gua-accent: {_tk['accent']}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
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
.gua-hero .gua-logo-frame {{
    width: 96px; height: 96px; min-width: 96px; flex-shrink: 0;
    padding: 8px; box-sizing: border-box;
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
.gua-sub {{ color: rgba(128,128,128,1); font-size:.95rem; margin:.15rem 0 0; }}
.gua-badges {{ display:flex; gap:6px; margin:.35rem 0 0; flex-wrap:wrap; }}
.gua-badge {{
    font-size:.72rem; font-weight:600; padding:.15rem .55rem; border-radius:999px;
    border:1px solid rgba(128,128,128,.3); color: rgba(128,128,128,1);
}}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────

_logo_uri = _logo_data_uri()
_logo_html = (f'<div class="gua-logo-frame"><img class="gua-logo-img" '
              f'src="{_logo_uri}" alt="GUARACI"></div>'
              if _logo_uri else '<span class="gua-logo">🧪</span>')

st.markdown(
    f"""
    <div class="gua-hero">
      {_logo_html}
      <div>
        <div class="gua-title">GUARACI · Chemometrics Platform</div>
      </div>
    </div>
    <p class="gua-sub">
      PLS-DA · PCA · OPLS-DA · DD-SIMCA · variable selection ·
      group-aware validation (anti-leakage of replicates).
      FT-NIR (.dx) or CSV table (Raman, UV-Vis, FTIR, chromatography…).
    </p>
    <div class="gua-badges">
      <span class="gua-badge">v{pq.__version__}</span>
      <span class="gua-badge">GPL-3.0-or-later</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if _IS_PUBLIC_DEMO:
    st.info(
        "🔬 **Modo demonstração pública** — sem dados reais configurados "
        "neste servidor, então o pipeline roda com **espectros sintéticos** "
        "gerados automaticamente (nenhuma amostra real de óleo). Explore "
        "livremente: todas as análises, gráficos e relatórios funcionam de "
        "verdade. Para usar com seus próprios dados, envie um CSV na aba "
        "Data ou rode o projeto localmente — ver "
        f"[repositório]({_tab_sobre._REPO})."
        if st.session_state.lang == "PT" else
        "🔬 **Public demo mode** — no real data configured on this server, "
        "so the pipeline runs on **synthetic spectra** generated "
        "automatically (no real oil samples). Feel free to explore: every "
        "analysis, figure and report is fully functional. To use your own "
        "data, upload a CSV in the Data tab or run the project locally — "
        f"see the [repository]({_tab_sobre._REPO})."
    )

# ──────────────────────────────────────────────────────────────────────────
# 7 Tabs
# ──────────────────────────────────────────────────────────────────────────

(tab_proj, tab_dados, tab_preproc, tab_modelo,
 tab_valid, tab_pred, tab_rel, tab_sobre) = st.tabs([
    "📋 " + _T("Project"),
    "📂 " + _T("Data"),
    "⚗️ " + _T("Preprocessing"),
    "🧮 " + _T("Model"),
    "📊 " + _T("Validation"),
    "🔮 " + _T("Prediction"),
    "📄 " + _T("Reports"),
    "ℹ️ " + _T("About"),
])

valores: Dict = {}  # accumulated by widgets from each tab

# ==========================================================================
#  TAB 1 — PROJECT (guaraci.app_tabs.projeto — item 18)
# ==========================================================================
with tab_proj:
    _tab_projeto.render(pq, _T, is_public_demo=_IS_PUBLIC_DEMO)

# ==========================================================================
#  TAB 2 — DATA
# ==========================================================================
with tab_dados:
    _tab_dados.render(pq, cfg_base, specs, valores, _widget_para_campo, _CFG_PATH, _T)


# ==========================================================================
#  TAB 3 — PREPROCESSING
# ==========================================================================
with tab_preproc:
    _tab_preprocessamento.render(pq, cfg_base, specs, valores, _widget_para_campo, _T)


# ==========================================================================
#  TAB 4 — MODEL (advanced parameters + execution)
# ==========================================================================
with tab_modelo:
    _tab_modelo.render(pq, cfg_base, specs, valores, _T, _widget_para_campo,
                       _MODO_ANALISE_ROTULO, _MODO_ANALISE_AJUDA, _CFG_PATH)


# ==========================================================================
#  TAB 5 — VALIDATION
# ==========================================================================
with tab_valid:
    _tab_validacao.render(_T, _tok, _ler_resumo, _listar_figuras)


# ==========================================================================
#  TAB 6 — PREDICTION
# ==========================================================================
with tab_pred:
    _tab_predicao.render(_UPLOAD_MODELO_BLOQUEADO, _tok, _T)




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
#  TAB 7 — REPORTS
# ==========================================================================
with tab_rel:
    _tab_relatorios.render(pq, _MODO_ANALISE_ROTULO, _zip_da_pasta,
                          _pdf_bytes, _word_bytes, _excel_bytes,
                          _latex_bytes, _pptx_bytes,
                          _ler_resumo, _ler_model_card, _listar_figuras, _T)


# ==========================================================================
#  TAB 8 — ABOUT
# ==========================================================================
with tab_sobre:
    _tab_sobre.render(pq, _T)
