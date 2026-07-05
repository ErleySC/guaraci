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

import logging
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
    listar_figuras as _listar_figuras_pura,
    ler_resumo as _ler_resumo_pura,
    ler_model_card as _ler_model_card_pura,
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
    except Exception:
        logging.getLogger(__name__).debug("suppressed non-critical exception", exc_info=True)
    return "light"


def _tok() -> Dict[str, str]:
    """Tokens de cor do tema atualmente ativo (dict semantico)."""
    return _theme_tokens(_active_theme())

# ──────────────────────────────────────────────────────────────────────────
# Page config (must be the first Streamlit command)
# ──────────────────────────────────────────────────────────────────────────
_icon_path = Path(__file__).parent / "guaraci_icon.png"
_page_icon = Image.open(_icon_path) if _icon_path.exists() else "🧪"

st.set_page_config(
    page_title="Chemometrics Platform",
    page_icon=_page_icon,
    layout="wide",
)

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
    "Step 1: Fill project info": {"PT": "Passo 1: Preencha as informações do projeto", "EN": "Step 1: Fill project info"},
    "Step 2: Upload or select spectra folder": {"PT": "Passo 2: Faça upload ou selecione a pasta de espectros", "EN": "Step 2: Upload or select spectra folder"},
    "Step 3: Configure parameters and run": {"PT": "Passo 3: Configure os parâmetros e execute", "EN": "Step 3: Configure parameters and run"},
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

if "cfg_base" not in st.session_state:
    try:
        st.session_state.cfg_base = (
            pq.carregar_config(_CFG_PATH) if os.path.exists(_CFG_PATH)
            # No local config.yaml (e.g. public demo deploy): default to
            # synthetic data so first-time visitors get a working demo
            # instead of an empty "dados/" folder error.
            else pq.Config(modo="sintetico"))
    except Exception:
        st.session_state.cfg_base = pq.Config(modo="sintetico")

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
.gua-hero .gua-logo {{ font-size: 2.4rem; line-height:1; }}
.gua-hero .gua-title {{
    font-size: 1.95rem; font-weight: 800; letter-spacing:-.02em; line-height:1.1;
    background: linear-gradient(90deg, var(--gua-primary), var(--gua-accent));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.gua-sub {{ color: rgba(128,128,128,1); font-size:.95rem; margin:.15rem 0 0; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="gua-hero">
      <span class="gua-logo">🧪</span>
      <div>
        <div class="gua-title">GUARACI · Chemometrics Platform</div>
      </div>
    </div>
    <p class="gua-sub">
      PLS-DA · PCA · OPLS-DA · DD-SIMCA · variable selection ·
      group-aware validation (anti-leakage of replicates).
      FT-NIR (.dx) or CSV table (Raman, UV-Vis, FTIR, chromatography…).
    </p>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# 7 Tabs
# ──────────────────────────────────────────────────────────────────────────

(tab_proj, tab_dados, tab_preproc, tab_modelo,
 tab_valid, tab_pred, tab_rel) = st.tabs([
    "📋 " + _T("Project"),
    "📂 " + _T("Data"),
    "⚗️ " + _T("Preprocessing"),
    "🧮 " + _T("Model"),
    "📊 " + _T("Validation"),
    "🔮 " + _T("Prediction"),
    "📄 " + _T("Reports"),
])

valores: Dict = {}  # accumulated by widgets from each tab

# ==========================================================================
#  TAB 1 — PROJECT (guaraci.app_tabs.projeto — item 18)
# ==========================================================================
with tab_proj:
    _tab_projeto.render(pq, _T)

# ==========================================================================
#  TAB 2 — DATA
# ==========================================================================
with tab_dados:
    _tab_dados.render(pq, cfg_base, specs, valores, _widget_para_campo, _CFG_PATH)


# ==========================================================================
#  TAB 3 — PREPROCESSING
# ==========================================================================
with tab_preproc:
    _tab_preprocessamento.render(pq, cfg_base, specs, valores, _widget_para_campo)


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
    _tab_predicao.render(_UPLOAD_MODELO_BLOQUEADO, _tok)




# ==========================================================================
#  Report cache — avoids regenerating on every Streamlit rerun.
#  Wrappers return bytes (immutable, cacheable); BytesIO is created
#  at download_button time to guarantee cursor at position 0.
#  Geração real delegada a guaraci.reports (item 18: serviço extraído da UI).
# ==========================================================================
@st.cache_data(show_spinner=False)
def _pdf_bytes(pasta: str, proj_items: tuple) -> bytes:
    return reports.gerar_pdf_relatorio(pasta, dict(proj_items)).read()

@st.cache_data(show_spinner=False)
def _word_bytes(pasta: str, proj_items: tuple) -> bytes:
    return reports.gerar_word_relatorio(pasta, dict(proj_items)).read()

@st.cache_data(show_spinner=False)
def _excel_bytes(pasta: str) -> bytes:
    return reports.gerar_excel_relatorio(pasta).read()

@st.cache_data(show_spinner=False)
def _latex_bytes(pasta: str, proj_items: tuple) -> bytes:
    return reports.gerar_latex_template(pasta, dict(proj_items))

@st.cache_data(show_spinner=False)
def _pptx_bytes(pasta: str, proj_items: tuple) -> bytes:
    return reports.gerar_pptx_relatorio(pasta, dict(proj_items)).read()


# ==========================================================================
#  TAB 7 — REPORTS
# ==========================================================================
with tab_rel:
    _tab_relatorios.render(pq, _MODO_ANALISE_ROTULO, _zip_da_pasta,
                          _pdf_bytes, _word_bytes, _excel_bytes,
                          _latex_bytes, _pptx_bytes,
                          _ler_resumo, _ler_model_card, _listar_figuras)
