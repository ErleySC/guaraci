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
import time
import tempfile
import zipfile
import threading
import contextlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
from PIL import Image
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    coletar_config as _coletar_config,
    fmt_tempo as _fmt_tempo,
    progresso_do_log as _progresso_do_log,
    listar_figuras as _listar_figuras_pura,
    ler_resumo as _ler_resumo_pura,
    ler_model_card as _ler_model_card_pura,
)
# Geração de relatórios (PDF/Word/Excel/LaTeX/PPTX) extraída para módulo de
# serviço próprio (item 18): app_quimiometria.py só cacheia e serve o download.
import guaraci.reports as reports


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


# _coletar_config foi movida para guaraci.app_logic (item 19) e importada acima.

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


class _LogThreadSafe:
    """Captures pipeline stdout/stderr into a lock-protected list."""
    def __init__(self, tee=None):
        self._buf: List[str] = []
        self._lock = threading.Lock()
        self._tee = tee

    def write(self, s: str):
        with self._lock:
            self._buf.append(s)
        if self._tee is not None:
            try: self._tee.write(s)
            except Exception:
                logging.getLogger(__name__).debug("suppressed non-critical exception", exc_info=True)
        return len(s)

    def flush(self):
        if self._tee is not None:
            try: self._tee.flush()
            except Exception:
                logging.getLogger(__name__).debug("suppressed non-critical exception", exc_info=True)

    def text(self) -> str:
        with self._lock:
            return "".join(self._buf)


# _RE_ETAPA/_ETAPA_NOMES/_ETAPA_SUBSTEP + _progresso_do_log + _fmt_tempo foram
# movidos para guaraci.app_logic (item 19) e importados no topo do arquivo.


def _ram_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _rodar_worker(cfg, logger: _LogThreadSafe, estado: Dict):
    try:
        with contextlib.redirect_stdout(logger), \
             contextlib.redirect_stderr(logger):
            pq.executar(cfg)
        estado["pasta"] = getattr(cfg, "pasta_saida", None)
        estado["erro"] = None
    except Exception:
        import traceback
        estado["erro"] = traceback.format_exc()
    finally:
        estado["fim"] = True


# ──────────────────────────────────────────────────────────────────────────
# Spectra preview (cached by path)
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=120)
def _preview_espectros_dx(pasta: str, wn_min: float, wn_max: float,
                           max_por_classe: int = 5):
    """Loads up to max_por_classe samples per subfolder for visualization."""
    try:
        subpastas = sorted(
            p for p in Path(pasta).iterdir()
            if p.is_dir() and list(p.glob("*.dx"))
        )
        if not subpastas:
            # flat folder with .dx files
            subpastas = [Path(pasta)]
        wn_ref, specs, labs = None, [], []
        for sp in subpastas:
            arqs = sorted(sp.glob("*.dx"))[:max_por_classe]
            for arq in arqs:
                try:
                    wn_a, sp_a = pq.parse_dx(str(arq))
                    mask = (wn_a >= wn_min) & (wn_a <= wn_max)
                    wn_a, sp_a = wn_a[mask], sp_a[mask]
                    if wn_ref is None:
                        wn_ref = wn_a
                    else:
                        # np.interp replaces deprecated scipy.interpolate.interp1d
                        sp_a = np.interp(wn_ref, wn_a, sp_a)
                    specs.append(sp_a)
                    labs.append(sp.name)
                except Exception:
                    continue
        if not specs or wn_ref is None:
            return None, None, None
        return wn_ref, np.array(specs), np.array(labs)
    except Exception:
        return None, None, None


@st.cache_data(show_spinner=False, ttl=120)
def _preview_espectros_csv(caminho: str, col_cls: str,
                            wn_min: float, wn_max: float,
                            max_n: int = 50):
    """Loads up to max_n CSV rows for visualization."""
    try:
        df = pd.read_csv(caminho, sep=None, engine="python", nrows=max_n)
        meta = {col_cls}
        num_cols = [c for c in df.columns if c not in meta]
        try:
            wn = np.array([float(c) for c in num_cols])
        except ValueError:
            return None, None, None
        mask = (wn >= wn_min) & (wn <= wn_max)
        X = df[num_cols].values[:, mask].astype(float)
        labs = df[col_cls].astype(str).values if col_cls in df.columns else \
               np.array(["?" ] * len(df))
        return wn[mask], X, labs
    except Exception:
        return None, None, None


def _plot_espectros_media(wn: np.ndarray, X: np.ndarray,
                           rotulos: np.ndarray, titulo: str = ""):  # -> matplotlib.figure.Figure
    """Plots mean ± std per class."""
    classes = np.unique(np.asarray(rotulos))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    for i, cls in enumerate(classes[:10]):
        cor = cmap(i / 10)
        mask = rotulos == cls
        med = X[mask].mean(axis=0)
        std = X[mask].std(axis=0)
        ax.plot(wn, med, color=cor, lw=1.3, label=f"{cls} (n={mask.sum()})")
        ax.fill_between(wn, med - std, med + std, color=cor, alpha=0.15)
    if len(wn) > 1 and wn[0] > wn[-1]:
        ax.invert_xaxis()
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorbance")
    if titulo:
        ax.set_title(titulo, fontsize=9)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(color="0.93", lw=0.5)
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Prediction on unknown samples
# ──────────────────────────────────────────────────────────────────────────
# Extracted to predicao.py (shared with the CLI's batch prediction menu) —
# same Fase H pattern: move once, reexport by name, never duplicate.
from guaraci.predicao import (  # noqa: E402
    predizer_amostras as _predizer,
    validar_pacote_modelo as _validar_pacote_modelo,
    carregar_csv_predicao as _carregar_csv_predicao,
)


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
#  TAB 1 — PROJECT
# ==========================================================================
def _hardware_status_widget():
    """Displays hardware panel with compatibility alerts."""
    try:
        hw = pq.hardware_probe()
        ram_t = hw["ram_total_gb"]
        ram_l = hw["ram_livre_gb"]
        cpu_f = hw["cpu_fisicos"]
        cpu_l = hw["cpu_logicos"]
        disco = hw["disco_livre_gb"]
        psutil_ok = hw["psutil_ok"]

        # RAM traffic-light color
        if ram_l < 2.0:
            cor_ram = "🔴"
            dica = "Critical RAM. Disable Benchmark, SHAP and MC CV."
        elif ram_l < 4.0:
            cor_ram = "🟠"
            dica = "Low RAM. Benchmark and SHAP will be disabled automatically."
        elif ram_l < 8.0:
            cor_ram = "🟡"
            dica = "Moderate RAM. Limits will be adjusted automatically."
        else:
            cor_ram = "🟢"
            dica = "Sufficient RAM for all operations."

        _ram_note = " ⚠️ (Cloud container)" if ram_t > 64 else ""
        c_hw1, c_hw2, c_hw3 = st.columns(3)
        with c_hw1:
            st.metric("Total RAM", f"{ram_t:.1f} GB{_ram_note}",
                      delta=f"{cor_ram} {ram_l:.1f} GB free",
                      delta_color="off")
        with c_hw2:
            st.metric("CPU", f"{cpu_f} cores",
                      delta=f"{cpu_l} logical threads",
                      delta_color="off")
        with c_hw3:
            st.metric("Free disk", f"{disco:.0f} GB",
                      delta="working folder",
                      delta_color="off")

        if ram_l < 8.0:
            st.warning(f"**Limited hardware detected.** {dica}")
        if not psutil_ok:
            st.caption("⚠️ psutil not available — approximate readings. "
                       "Install with `pip install psutil`.")
    except Exception:
        st.caption("Hardware: could not detect hardware specifications.")


with tab_proj:
    st.subheader(_T("Project Identification"))
    st.caption(
        "Descriptive only — used in the report cover and saved automatically in "
        "this session. What the pipeline actually runs is set by the "
        "**Analysis mode** (Model tab)."
        if st.session_state.get("lang") != "PT" else
        "Apenas descritivo — vai na capa dos relatórios e é salvo automaticamente "
        "nesta sessão. O que o pipeline executa é definido pelo **Modo de "
        "análise** (aba Modelo).")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Project name", key="proj_nome",
                      placeholder="e.g.: Authentication of Amazonian Oils FT-NIR")
        st.text_input("Author(s)", key="proj_autor",
                      placeholder="e.g.: Silva, J.A.; Costa, M.B.")
        st.text_input("Institution / Laboratory", key="proj_inst",
                      placeholder="e.g.: GEAAp / UFPA")
    with c2:
        st.text_area("Objective", key="proj_objetivo", height=182,
                     placeholder="Describe the objective of the chemometric analysis...")

    with st.expander("💻 Hardware Status", expanded=False):
        _hardware_status_widget()

    run_proj = st.session_state.get("proj_nome", "")
    if run_proj:
        st.caption(f"✅ Active project: **{run_proj}**")

# ==========================================================================
#  TAB 2 — DATA
# ==========================================================================
with tab_dados:
    st.subheader(_T("Data Input"))
    st.caption("📂 " + _T("Step 2: Upload or select spectra folder") + " → then go to **Model** tab.")

    # ---- CSV Upload (at top for easy access) ---------------------------------
    st.markdown("**Upload CSV** *(alternative to the local path below)*")
    upld = st.file_uploader(
        "Drag or select a CSV file",
        type=["csv", "txt"],
        key="csv_upload_widget",
        help="The file will be saved to a temporary folder and the path adjusted automatically.",
    )
    if upld is not None:
        tmp_dir = Path(tempfile.gettempdir()) / "pq_uploads"
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = str(tmp_dir / Path(upld.name).name)  # basename only — blocks path traversal
        if (st.session_state.get("_csv_upload_name") != upld.name
                or not os.path.exists(tmp_path)):
            with open(tmp_path, "wb") as f:
                f.write(upld.getvalue())
            st.session_state["_csv_upload_name"] = upld.name
            st.session_state["_csv_upload_path"] = tmp_path
        st.success(f"File saved: `{tmp_path}`")
        st.info("Mode automatically set to 'csv'. "
                "The path above will be overridden when running.")

    # ---- Path / config fields -----------------------------------------------
    st.divider()
    _DADOS_KEYS = ["modo_entrada", "pasta_dados", "arquivo_csv",
                   "coluna_classe", "coluna_concentracao",
                   "pasta_saida", "excluir_classes", "imagem_incluir_textura"]

    col_d1, col_d2 = st.columns(2)
    for i, k in enumerate(_DADOS_KEYS):
        s = specs.get(k)
        if s is None:
            continue
        with (col_d1 if i % 2 == 0 else col_d2):
            valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    # ---- Data statistics preview -----------------------------------------
    st.divider()
    st.markdown("**Data preview**")
    cfg_prev, _ = _coletar_config(cfg_base, valores)
    ok_dados, msg_dados = pq._validar_pasta_dados(cfg_prev)
    (st.success if ok_dados else st.warning)(f"Status: {msg_dados}")

    if ok_dados and st.button("🔍 Load spectra preview", key="btn_prev_dados"):
        modo = cfg_prev.modo
        wn_mn = float(cfg_prev.wn_min)
        wn_mx = float(cfg_prev.wn_max)
        with st.spinner("Loading spectra sample..."):
            if modo == "dx":
                wn_p, X_p, labs_p = _preview_espectros_dx(
                    cfg_prev.pasta_entrada, wn_mn, wn_mx)
            elif modo == "csv":
                csv_cam = st.session_state.get("_csv_upload_path",
                                               cfg_prev.arquivo_csv)
                wn_p, X_p, labs_p = _preview_espectros_csv(
                    csv_cam, cfg_prev.coluna_classe, wn_mn, wn_mx)
            else:
                wn_p, X_p, labs_p = None, None, None

        if wn_p is not None and X_p is not None:
            cls_u = np.unique(np.asarray(labs_p))
            st.markdown(f"**{len(X_p)} spectra** · {len(cls_u)} classes: "
                        f"`{'`, `'.join(cls_u[:8])}`"
                        + (" ..." if len(cls_u) > 8 else ""))
            fig_p = _plot_espectros_media(wn_p, X_p, np.asarray(labs_p), titulo="Raw spectra (sample)")
            st.pyplot(fig_p, use_container_width=True)
            plt.close(fig_p)
        else:
            st.warning("Could not load spectra for preview. "
                       "Check the path/mode.")

    # ---- Save / Reload config.yaml ---------------------------------
    st.divider()
    cfg_dados, erros_dados = _coletar_config(cfg_base, valores)
    if erros_dados:
        st.warning("Fields with errors:\n- " + "\n- ".join(erros_dados))
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        if st.button("💾 Save config.yaml", key="btn_salvar_cfg_dados",
                     use_container_width=True):
            if erros_dados:
                st.error("Fix the fields before saving.")
            else:
                pq.salvar_config(cfg_dados, _CFG_PATH)
                st.session_state.cfg_base = cfg_dados
                st.success(f"Saved to {_CFG_PATH}")
    with c_s2:
        if st.button("↺ Reload config.yaml", key="btn_reload_cfg_dados",
                     use_container_width=True):
            try:
                st.session_state.cfg_base = pq.carregar_config(_CFG_PATH)
                cfg_base = st.session_state.cfg_base
                st.success("Config reloaded.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================================================
#  TAB 3 — PREPROCESSING
# ==========================================================================
with tab_preproc:
    st.subheader("Spectral Preprocessing")

    _PREPROC_KEYS = ["pre_processamento", "faixa_min_cm", "faixa_max_cm"]

    col_p1, col_p2 = st.columns(2)
    for i, k in enumerate(_PREPROC_KEYS):
        s = specs.get(k)
        if s is None:
            continue
        with (col_p1 if i % 2 == 0 else col_p2):
            valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    # Information about each preset
    preset_selecionado = valores.get("pre_processamento", "")
    _PRESET_INFO = {
        "MSC+SG+MC":      "MSC (scatter correction) → 1st derivative SG (Savitzky-Golay) → Mean-Centering. **Best for FT-NIR with pronounced scatter.** Acc=0.923 on 1807 Amazonian oil samples.",
        "SNV+SG+MC":      "SNV (variance normalization) → SG → Mean-Centering. Robust alternative to MSC when global reference is not stable.",
        "Autoscaling":    "Mean-Centering + division by standard deviation. **Caution**: collapses spectral noise when SG is not applied first.",
        "Mean-centering": "Mean centering only. Recommended as a comparative baseline.",
    }
    if preset_selecionado in _PRESET_INFO:
        st.info(_PRESET_INFO[preset_selecionado])

    # ---- Before/after preview -------------------------------------------
    st.divider()
    st.markdown("**Before / after preprocessing visualization**")
    cfg_pp, _ = _coletar_config(cfg_base, valores)
    ok_pp, _ = pq._validar_pasta_dados(cfg_pp)

    if not ok_pp:
        st.info("Configure and validate data input (Data tab) to enable the preview.")
    elif st.button("⚗️ Generate before/after preview", key="btn_prev_preproc"):
        with st.spinner("Loading and processing spectra..."):
            modo_pp = cfg_pp.modo
            wn_mn_pp = float(cfg_pp.wn_min)
            wn_mx_pp = float(cfg_pp.wn_max)
            if modo_pp == "dx":
                wn_raw, X_raw, labs_raw = _preview_espectros_dx(
                    cfg_pp.pasta_entrada, wn_mn_pp, wn_mx_pp)
            elif modo_pp == "csv":
                csv_cam_pp = st.session_state.get("_csv_upload_path",
                                                   cfg_pp.arquivo_csv)
                wn_raw, X_raw, labs_raw = _preview_espectros_csv(
                    csv_cam_pp, cfg_pp.coluna_classe, wn_mn_pp, wn_mx_pp)
            else:
                wn_raw, X_raw, labs_raw = None, None, None

        if wn_raw is not None and X_raw is not None:
            try:
                preproc_pp = pq.construir_preprocessador(cfg_pp)
                preproc_pp.fit(X_raw)
                X_proc_pp = preproc_pp.transform(X_raw)
                labs_raw_arr = np.asarray(labs_raw)
                fig_antes = _plot_espectros_media(
                    wn_raw, X_raw, labs_raw_arr, "Before preprocessing")
                fig_depois = _plot_espectros_media(
                    wn_raw, X_proc_pp, labs_raw_arr,
                    f"After: {preset_selecionado}")
                col_ant, col_dep = st.columns(2)
                with col_ant:
                    st.pyplot(fig_antes, use_container_width=True)
                    plt.close(fig_antes)
                with col_dep:
                    st.pyplot(fig_depois, use_container_width=True)
                    plt.close(fig_depois)
            except Exception as e_pp:
                st.error(f"Error applying preprocessing: {e_pp}")
        else:
            st.warning("Could not load spectra. Check the Data tab.")

# ==========================================================================
#  TAB 4 — MODEL (advanced parameters + execution)
# ==========================================================================
with tab_modelo:
    st.subheader(_T("Model Parameters and Execution"))

    # ---- Quick-access Run button at the very top of the Model tab ----------
    # Rebuild enabled state from session_state widget keys (populated on reruns)
    _valores_top = {
        k: st.session_state[f"w_{k}"]
        for k in specs
        if f"w_{k}" in st.session_state
    }
    _cfg_top, _erros_top = _coletar_config(cfg_base, _valores_top)
    _erros_top = _erros_top + pq._validar_semantico(_cfg_top)
    _ok_top = (not _erros_top) and pq._validar_pasta_dados(_cfg_top)[0]
    _rodar_top = st.button(
        "▶️ " + _T("Run pipeline"), type="primary",
        disabled=not _ok_top,
        use_container_width=True,
        key="btn_run_top",
    )
    st.caption("ℹ️ Configure the options below, then click **▶️ Run pipeline**.")

    _MODELO_KEYS_ANALISE  = ["nivel", "max_lvs", "holdout_fracao",
                              "validacao_group_aware"]
    _MODELO_KEYS_VALID    = ["n_permutacoes", "teste_wold", "teste_cv_anova",
                              "teste_martens", "n_jobs_permutacao"]
    _MODELO_KEYS_EXTRAS   = ["selecao_variaveis_etapa4", "selecao_spa", "selecao_ag",
                              "ddsimca", "modo_ddsimca", "opls_da",
                              "comparar_pre_processamentos", "benchmark",
                              "benchmark_regressao",
                              "monte_carlo", "n_monte_carlo",
                              "monte_carlo_incluir_todos",
                              "shap_benchmark", "shap_max_amostras"]
    _MODELO_KEYS_FIGURAS  = ["figuras_detalhadas",
                              "figuras_mostrar_marcadores",
                              "figuras_mostrar_elipses",
                              "formato_figura", "dpi",
                              "abrir_figuras_na_tela"]

    # ---- Key options at top (analysis mode + n_lvs shown inline) ----------
    _KEY_TOP = ["nivel", "max_lvs"]
    _cols_top = st.columns(len(_KEY_TOP))
    with _cols_top[0]:
        # "nivel" é mostrado com rótulos amigáveis; o valor gravado continua
        # sendo N1/N2/N3 (não quebra config.yaml nem o pipeline).
        _s_niv = specs.get("nivel")
        if _s_niv is not None:
            _niv_atual = pq._attr_para_yaml(_s_niv, cfg_base)
            _ops_niv = list(_s_niv.get("opcoes") or ["N1", "N2", "N3"])
            _idx_niv = _ops_niv.index(_niv_atual) if _niv_atual in _ops_niv else 0
            valores["nivel"] = st.selectbox(
                "Modo de análise", _ops_niv, index=_idx_niv, key="w_nivel",
                format_func=lambda v: _MODO_ANALISE_ROTULO.get(v, v))
            st.caption(_MODO_ANALISE_AJUDA.get(valores["nivel"], ""))
    with _cols_top[1]:
        _s_lvs = specs.get("max_lvs")
        if _s_lvs is not None:
            valores["max_lvs"] = _widget_para_campo(
                _s_lvs, pq._attr_para_yaml(_s_lvs, cfg_base))

    # Preprocessing is chosen in the Preprocessing tab (single source of truth).
    # Mirror that choice here read-only. Previously a second editable widget with
    # a separate key rendered here and, running later, silently overrode the
    # user's Preprocessing-tab choice with its own default.
    _s_pre = specs.get("pre_processamento")
    if _s_pre is not None:
        _pre_escolhido = st.session_state.get(
            "w_pre_processamento", pq._attr_para_yaml(_s_pre, cfg_base))
        valores["pre_processamento"] = _pre_escolhido
        st.caption(
            f"⚗️ Pré-processamento: **{_pre_escolhido}** "
            "— definido no separador *Pré-processamento*.")

    st.divider()

    with st.expander("🧮 Analysis & partitioning", expanded=True):
        cols_a = st.columns(2)
        _MODELO_KEYS_ANALISE_EXP = [k for k in _MODELO_KEYS_ANALISE if k not in _KEY_TOP]
        for i, k in enumerate(_MODELO_KEYS_ANALISE_EXP):
            s = specs.get(k)
            if s is None: continue
            with cols_a[i % 2]:
                valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    with st.expander("📊 Statistical validation", expanded=False):
        cols_v = st.columns(2)
        for i, k in enumerate(_MODELO_KEYS_VALID):
            s = specs.get(k)
            if s is None: continue
            with cols_v[i % 2]:
                valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    with st.expander("🧩 Extra modules", expanded=False):
        # Hardware warning for heavy operations
        try:
            _hw_mod = pq.hardware_probe()
            _ram_l  = _hw_mod.get("ram_livre_gb", 16.0)
            if _ram_l < 4.0:
                st.error(
                    f"⚠️ Free RAM: **{_ram_l:.1f} GB** — "
                    "Benchmark and SHAP will be **automatically disabled** "
                    "by the pipeline to prevent freezing.")
            elif _ram_l < 8.0:
                st.warning(
                    f"⚠️ Free RAM: **{_ram_l:.1f} GB** — "
                    "SHAP and Monte Carlo CV limits will be adjusted "
                    "automatically. Closing other programs is recommended.")
        except Exception:
            logging.getLogger(__name__).debug("suppressed non-critical exception", exc_info=True)

        cols_e = st.columns(2)
        for i, k in enumerate(_MODELO_KEYS_EXTRAS):
            s = specs.get(k)
            if s is None: continue
            with cols_e[i % 2]:
                valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    with st.expander("🖼️ Figures", expanded=False):
        cols_f = st.columns(2)
        for i, k in enumerate(_MODELO_KEYS_FIGURAS):
            s = specs.get(k)
            if s is None: continue
            with cols_f[i % 2]:
                valores[k] = _widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    st.divider()

    # ---- Final Config assembly and execution ----------------------------
    cfg_run, erros_run = _coletar_config(cfg_base, valores)

    # If user uploaded a CSV, override the path
    if st.session_state.get("_csv_upload_path"):
        csv_upld_path = st.session_state["_csv_upload_path"]
        if os.path.exists(csv_upld_path):
            cfg_run.modo = "csv"
            cfg_run.arquivo_csv = csv_upld_path

    ok_run, msg_run = pq._validar_pasta_dados(cfg_run)
    erros_run = erros_run + pq._validar_semantico(cfg_run)
    st.write("**Input:**", msg_run)
    if erros_run:
        st.error("Invalid fields in configuration:\n- " + "\n- ".join(erros_run))

    pode_rodar = ok_run and not erros_run
    rodar = st.button("▶️ " + _T("Run pipeline"), type="primary",
                      disabled=not pode_rodar, use_container_width=True,
                      key="btn_rodar")
    if not pode_rodar:
        st.info(_T("Fix the data input (Data tab) to enable."))

    if rodar or _rodar_top:
        # Trava anti-concorrência: se um worker anterior ainda estiver vivo
        # (ex.: análise que estourou o timeout de 2 h mas segue rodando em
        # segundo plano), não inicia outro. Antes, dois pq.executar() podiam
        # rodar em paralelo disputando config.yaml/pastas e dobrando a RAM.
        _prev_worker = st.session_state.get("_worker_ativo")
        if _prev_worker is not None and _prev_worker.is_alive():
            st.warning(
                "Uma análise anterior ainda está em execução em segundo plano. "
                "Aguarde ela terminar (ou recarregue a página) antes de rodar "
                "novamente.")
            st.stop()
        try:
            pq.salvar_config(cfg_run, _CFG_PATH)
        except Exception as _e_cfg:
            # Não impede a execução (a config vai em memória para o worker),
            # mas avisa: sem isso, um filesystem só-leitura (comum no Cloud)
            # falhava em silêncio e "Reload config.yaml" restaurava config antiga.
            st.warning(
                "Não foi possível gravar config.yaml — a análise continua com "
                f"os valores atuais, mas 'Reload config.yaml' pode restaurar uma "
                f"versão antiga. Detalhe: {_e_cfg}")

        logger = _LogThreadSafe(tee=sys.__stdout__)
        estado: Dict = {"fim": False, "erro": None, "pasta": None}
        worker = threading.Thread(
            target=_rodar_worker, args=(cfg_run, logger, estado), daemon=True)
        st.session_state["_worker_ativo"] = worker

        t0 = time.monotonic()
        eta_best: Optional[float] = None
        # Max runtime guard: 2 hours (prevents infinite blocking on Cloud)
        _MAX_RUNTIME = 7200

        worker.start()

        # Use st.status() + single placeholder to avoid simultaneous DOM
        # mutations that cause React's removeChild error on Streamlit Cloud.
        with st.status("⚙️ Running pipeline...", expanded=True) as _run_status:
            ph = st.empty()
            while not estado["fim"]:
                # Safety timeout
                if time.monotonic() - t0 > _MAX_RUNTIME:
                    estado["fim"] = True
                    estado["erro"] = "Pipeline exceeded maximum runtime (2 h)."
                    break
                txt = logger.text()
                frac, nome = _progresso_do_log(txt)
                elapsed = time.monotonic() - t0
                if frac >= 0.10:
                    eta = elapsed / frac - elapsed
                    eta_best = eta if eta_best is None else min(eta_best, eta)
                ram = _ram_mb()
                linhas = txt.strip().splitlines()
                log_tail = "\n".join(linhas[-10:]) if linhas else "..."
                # Single atomic DOM update per iteration
                ph.markdown(
                    f"**[{int(frac * 100)}%] {nome}**\n\n"
                    f"⏱️ `{_fmt_tempo(elapsed)}` elapsed  |  "
                    f"⏳ `{_fmt_tempo(eta_best) if eta_best else 'calculating…'}` remaining  |  "
                    f"💾 `{f'{ram:.0f} MB' if ram else 'n/a'}`\n\n"
                    f"```text\n{log_tail}\n```"
                )
                time.sleep(0.5)

            # Final state update
            txt     = logger.text()
            elapsed = time.monotonic() - t0
            ph.empty()
            if estado["erro"]:
                _run_status.update(
                    label=f"❌ Pipeline failed after {_fmt_tempo(elapsed)}.",
                    state="error", expanded=True)
            else:
                _run_status.update(
                    label=f"✅ Completed in {_fmt_tempo(elapsed)}!",
                    state="complete", expanded=False)

        # Libera a trava só se o worker de fato terminou; se ficou órfão por
        # timeout (ainda vivo), mantém a referência para bloquear novo run.
        if not worker.is_alive():
            st.session_state["_worker_ativo"] = None

        st.session_state.ultimo_log  = txt
        if estado["erro"]:
            st.session_state.erro_run  = estado["erro"]
            st.session_state.ultima_pasta = None
            st.error(f"Pipeline failed after {_fmt_tempo(elapsed)}.")
        else:
            st.session_state.erro_run  = None
            st.session_state.ultima_pasta = estado["pasta"]
            st.success("✅ Completed! View results in the Validation and Reports tabs.")

    if st.session_state.get("erro_run"):
        st.subheader("Error traceback")
        st.code(st.session_state.erro_run, language="text")

# ==========================================================================
#  TAB 5 — VALIDATION
# ==========================================================================
with tab_valid:
    st.subheader(_T("Validation Results"))
    pasta_v = st.session_state.get("ultima_pasta")

    if not pasta_v or not os.path.isdir(pasta_v):
        st.info(_T("No results yet. Run the pipeline in the Model tab."))
    else:
        st.caption(f"Folder: `{os.path.abspath(pasta_v)}`")

        # Numeric summary
        resumo_txt = _ler_resumo(pasta_v)
        if resumo_txt:
            with st.expander("📋 Model summary (resumo_modelo.txt)", expanded=True):
                # Split into main section and notes section for cleaner display
                if "Methodological Notes" in resumo_txt:
                    parts = resumo_txt.split("---")
                    main_part = parts[0].strip() if parts else resumo_txt
                    notes_part = resumo_txt[resumo_txt.find("Methodological Notes"):].strip() if "Methodological Notes" in resumo_txt else ""
                    st.code(main_part, language="text")
                    if notes_part:
                        with st.expander("📋 Methodological Notes (for peer review)", expanded=False):
                            st.code(notes_part, language="text")
                else:
                    st.code(resumo_txt, language="text")

        # ── Per-Class Accuracy table (extracted from summary) ───────────────
        if resumo_txt:
            import re as _re_acc
            acc_map: Dict[str, float] = {}
            for _linha in resumo_txt.splitlines():
                _m = _re_acc.match(r"\s*Acc\s+(.+?)\s*[:=]\s*([\d.]+)", _linha)
                if _m:
                    acc_map[_m.group(1).strip()] = float(_m.group(2))
            if acc_map:
                with st.expander("📊 Accuracy by Class", expanded=True):
                    _df_acc = pd.DataFrame(
                        list(acc_map.items()), columns=["Class", "Accuracy (recall)"]
                    ).sort_values("Accuracy (recall)")
                    # Color: red < 0.7, yellow 0.7-0.9, green >= 0.9
                    # Cores vindas dos design tokens do tema ativo → contraste
                    # correto em claro E escuro (não mais hex claros fixos).
                    _tkc = _tok()
                    def _cor_acc(v: object) -> str | None:  # Scalar compat
                        try:
                            fv = float(v)  # type: ignore[arg-type]
                        except (TypeError, ValueError):
                            return None
                        if fv >= 0.90:
                            return (f"background-color:{_tkc['success_bg']};"
                                    f"color:{_tkc['success']}")
                        if fv >= 0.70:
                            return (f"background-color:{_tkc['warn_bg']};"
                                    f"color:{_tkc['warn']}")
                        return (f"background-color:{_tkc['error_bg']};"
                                f"color:{_tkc['error']}")
                    st.dataframe(
                        _df_acc.style.map(
                            _cor_acc, subset=["Accuracy (recall)"]),
                        use_container_width=True, height=min(400, 35 * len(_df_acc) + 38))

        # ── Benchmark and MC CV tables (if they exist) ───────────────────
        _bench_csv_v = os.path.join(pasta_v, "dados",
                                    "benchmark_classificadores.csv")
        _mc_csv_v    = os.path.join(pasta_v, "dados", "monte_carlo_cv.csv")
        if os.path.exists(_bench_csv_v) or os.path.exists(_mc_csv_v):
            with st.expander("🏅 Classifier Benchmark", expanded=False):
                if os.path.exists(_bench_csv_v):
                    try:
                        _df_b = pd.read_csv(_bench_csv_v, sep=";", decimal=",")
                        st.markdown("**Auto-Benchmark — Balanced Accuracy by model (GroupKFold)**")
                        st.dataframe(_df_b, use_container_width=True)
                    except Exception as _e_b:
                        st.warning(f"Error reading benchmark CSV: {_e_b}")
                if os.path.exists(_mc_csv_v):
                    try:
                        _df_mc = pd.read_csv(_mc_csv_v, sep=";", decimal=",")
                        st.markdown("**Monte Carlo CV — 95% CI by percentile**")
                        st.dataframe(_df_mc, use_container_width=True)
                    except Exception as _e_mc:
                        st.warning(f"Error reading MC CV CSV: {_e_mc}")

        # Gallery filtered by figure category
        imgs_v = _listar_figuras(pasta_v)
        if imgs_v:
            st.markdown(f"**{len(imgs_v)} figures generated**")
            _CATS = {
                "All":              "",
                "PCA":              "pca",
                "PLS-DA scores":    "plsda",
                "Outliers (T²/Q)":  "outlier",
                "Confusion":        "confus",
                "ROC / AUC":        "roc",
                "VIP / SR":         "vip",
                "Loading":          "loading",
                "HCA":              "hca",
                "OPLS-DA":          "opls",
                "DD-SIMCA":         "ddsimca",
                "Cooman's Plot":    "cooman",
                "S-Plot":           "splot",
                "Permutation":      "permut",
                "Wold":             "wold",
                "Benchmark":        "benchmark",
                "Monte Carlo CV":   "monte_carlo",
                "DET curves":       "fig_det",
                "SHAP":             "fig_shap",
            }
            filtro_v = st.selectbox("Filter by category",
                                    list(_CATS.keys()), key="filtro_valid")
            token_v  = _CATS[filtro_v].lower()
            imgs_filt = [im for im in imgs_v
                         if token_v in os.path.basename(im).lower()] \
                        if token_v else imgs_v

            if imgs_filt:
                cols_v = st.columns(2)
                for j, img in enumerate(imgs_filt):
                    with cols_v[j % 2]:
                        st.image(img, caption=os.path.basename(img),
                                 use_container_width=True)
            else:
                st.info(f"No figures found for '{filtro_v}'.")
        else:
            st.info("Figures saved in non-displayable format (PDF/SVG). "
                    "Use the download button in the Reports tab.")

# ==========================================================================
#  TAB 6 — PREDICTION
# ==========================================================================
with tab_pred:
    st.subheader("Prediction on Unknown Samples")
    st.markdown(
        "Upload a `.joblib` model generated by the pipeline and a CSV "
        "with new spectra (columns = wavenumbers, no class column)."
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**1. Trained model (.joblib)**")
        if _UPLOAD_MODELO_BLOQUEADO:
            upld_jbl = None
            st.info(
                "🔒 Model upload is disabled on this public deployment "
                "(a `.joblib`/pickle can execute arbitrary code when loaded). "
                "Provide a local path to a model file below instead.")
        else:
            st.caption(
                "⚠️ Only upload `.joblib` models you generated yourself. "
                "A model file is a pickle and **runs code when loaded** — "
                "never load one from an untrusted source.")
            upld_jbl = st.file_uploader("Upload the .joblib model",
                                        type=["joblib", "pkl"],
                                        key="pred_model_upload")
        cam_jbl  = st.text_input("Or local path to model",
                                  key="pred_model_path",
                                  placeholder="C:/results/model_pls.joblib")

    with col_m2:
        st.markdown("**2. New spectra (CSV)**")
        upld_csv_pred = st.file_uploader("Upload CSV with new spectra",
                                          type=["csv", "txt"],
                                          key="pred_csv_upload")
        cam_csv_pred  = st.text_input("Or local path to CSV",
                                       key="pred_csv_path",
                                       placeholder="C:/data/new_spectra.csv")
        col_wn_pred   = st.text_input(
            "First column to use as wavenumber (leave empty = auto)",
            key="pred_col_wn",
            placeholder="e.g.: 4000.0  (name or index of first spectral column)")

    # ---- Predict button -------------------------------------------------
    st.divider()
    if st.button("🔮 Predict", type="primary", key="btn_predizer",
                 use_container_width=True):
        erros_pred: List[str] = []
        pkg_pred = None

        # Load model
        try:
            import joblib
            if upld_jbl is not None and not _UPLOAD_MODELO_BLOQUEADO:
                tmp_jbl = Path(tempfile.gettempdir()) / "pq_pred_model.joblib"
                with open(tmp_jbl, "wb") as f:
                    f.write(upld_jbl.getvalue())
                pkg_pred = joblib.load(str(tmp_jbl))
            elif cam_jbl and os.path.exists(cam_jbl):
                pkg_pred = joblib.load(cam_jbl)
            # NOTE: this is STRUCTURE validation only — it runs AFTER joblib.load,
            # so it does NOT prevent RCE from a malicious pickle (the code already
            # ran during load). The real mitigation is upstream: only trusted
            # sources reach here (upload gated by _UPLOAD_MODELO_BLOQUEADO on
            # public deployments; local paths are operator-controlled).
            if pkg_pred is not None:
                _validar_pacote_modelo(pkg_pred)
            else:
                erros_pred.append("No valid model provided (upload or path).")
        except Exception as e_jbl:
            erros_pred.append(f"Error loading model: {e_jbl}")

        # Load prediction CSV
        X_pred_raw = None
        wn_pred    = None
        try:
            fonte_csv = (upld_csv_pred if upld_csv_pred is not None
                         else cam_csv_pred if cam_csv_pred and os.path.exists(cam_csv_pred)
                         else None)
            if fonte_csv is None:
                erros_pred.append("No spectra CSV provided.")
            else:
                X_pred_raw, wn_pred, meta_pred = _carregar_csv_predicao(fonte_csv)
                st.session_state["pred_amostras"] = meta_pred
        except Exception as e_csv:
            erros_pred.append(f"Error reading CSV: {e_csv}")

        if erros_pred:
            for e in erros_pred:
                st.error(e)
        elif pkg_pred is not None and X_pred_raw is not None:
            try:
                with st.spinner("Applying model..."):
                    df_res = _predizer(pkg_pred, X_pred_raw, wn_pred)
                    # Re-attach metadata (sample names) if available
                    meta_df = st.session_state.get("pred_amostras")
                    if meta_df is not None and len(meta_df) == len(df_res):
                        df_res = pd.concat(
                            [meta_df.reset_index(drop=True), df_res], axis=1)
                st.session_state["pred_resultados"] = df_res
                st.success(f"Prediction complete: {len(df_res)} samples.")
            except Exception as e_pred:
                st.error(f"Prediction error: {e_pred}")

    # ---- Results display ----------------------------------------
    df_show = st.session_state.get("pred_resultados")
    if df_show is not None:
        st.divider()
        st.markdown("**Prediction results**")

        # Color accepted/rejected — via design tokens (contraste correto em
        # tema claro e escuro).
        _tkp = _tok()
        def _colorir_aceito(val):
            if val is True:
                return (f"background-color:{_tkp['success_bg']};"
                        f" color:{_tkp['success']}")
            if val is False:
                return (f"background-color:{_tkp['error_bg']};"
                        f" color:{_tkp['error']}")
            return ""

        cols_bool_colorir = [c for c in ("aceito", "AD_dentro_dominio")
                             if c in df_show.columns]
        if cols_bool_colorir:
            st.dataframe(
                df_show.style.map(_colorir_aceito, subset=cols_bool_colorir),
                use_container_width=True,
            )
        else:
            st.dataframe(df_show, use_container_width=True)

        # Download CSV
        csv_bytes = df_show.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
        st.download_button(
            "⬇️ Download results (.csv)",
            data=csv_bytes,
            file_name="predicao_resultados.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Quick summary
        col_m1, col_m2 = st.columns(2)
        if "aceito" in df_show.columns:
            n_ac = int(df_show["aceito"].sum())
            n_tot = len(df_show)
            with col_m1:
                st.metric("Accepted samples (PLS-DA fit, T² ≤ UCL and Q ≤ UCL)",
                          f"{n_ac} / {n_tot}",
                          delta=f"{n_ac/n_tot*100:.1f}%")
        # Applicability Domain (PCA-based, Jaworska et al. 2005) — only
        # present if the .joblib package carries the AD artifacts (pipeline
        # versions exporting pca/ad_var_t/ad_t2_limite/ad_q_limite).
        if "AD_dentro_dominio" in df_show.columns:
            n_ad = int(df_show["AD_dentro_dominio"].sum())
            n_tot = len(df_show)
            with col_m2:
                st.metric("Within applicability domain (PCA T²/Q)",
                          f"{n_ad} / {n_tot}",
                          delta=f"{n_ad/n_tot*100:.1f}%")



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
    st.subheader("Reports and Downloads")
    pasta_r = st.session_state.get("ultima_pasta")

    if not pasta_r or not os.path.isdir(pasta_r):
        st.info("Run the pipeline (Model tab) to generate reports.")
    else:
        st.caption(f"Results folder: `{os.path.abspath(pasta_r)}`")

        # ── Downloads ─────────────────────────────────────────────────────
        st.markdown("### ⬇️ Downloads")

        _nome_base = os.path.basename(pasta_r)
        # "Study type" na capa do relatório é DERIVADO do Modo de análise
        # escolhido (N1/N2/N3) — sem campo duplicado na aba Project.
        _tipo_estudo = _MODO_ANALISE_ROTULO.get(
            st.session_state.get("w_nivel", ""), "")
        _projeto_info = {
            "nome":     st.session_state.get("proj_nome", ""),
            "autor":    st.session_state.get("proj_autor", ""),
            "inst":     st.session_state.get("proj_inst", "GEAAp / UFPA"),
            "tipo":     _tipo_estudo,
            "objetivo": st.session_state.get("proj_objetivo", ""),
        }
        # Sorted tuple used as cache key (Dict is not hashable)
        _proj_items = tuple(sorted(_projeto_info.items()))

        # Row 1: ZIP + PDF
        col_a, col_b = st.columns(2)
        with col_a:
            try:
                st.download_button(
                    "📦 Full results (.zip)",
                    data=_zip_da_pasta(pasta_r),
                    file_name=_nome_base + ".zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            except Exception as e_zip:
                st.warning(f"ZIP: {e_zip}")

        with col_b:
            try:
                st.download_button(
                    "📄 PDF Report",
                    data=_pdf_bytes(pasta_r, _proj_items),
                    file_name=_nome_base + "_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                )
            except Exception as e_pdf:
                st.error(f"PDF: {e_pdf}")

        # Row 2: Word + Excel
        col_c, col_d = st.columns(2)
        with col_c:
            try:
                st.download_button(
                    "📝 Word Report (.docx)",
                    data=_word_bytes(pasta_r, _proj_items),
                    file_name=_nome_base + "_report.docx",
                    mime="application/vnd.openxmlformats-officedocument"
                         ".wordprocessingml.document",
                    use_container_width=True,
                )
            except Exception as e_word:
                st.error(f"Word: {e_word}")

        with col_d:
            try:
                st.download_button(
                    "📊 Data in Excel (.xlsx)",
                    data=_excel_bytes(pasta_r),
                    file_name=_nome_base + "_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument"
                         ".spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e_xlsx:
                st.error(f"Excel: {e_xlsx}")

        # Row 3: LaTeX + PowerPoint
        col_e, col_f = st.columns(2)
        with col_e:
            try:
                st.download_button(
                    "🔬 LaTeX Template (Talanta / Food Chemistry / J. Chemom.)",
                    data=_latex_bytes(pasta_r, _proj_items),
                    file_name=_nome_base + "_template.tex",
                    mime="text/plain",
                    use_container_width=True,
                )
            except Exception as e_tex:
                st.error(f"LaTeX: {e_tex}")

        with col_f:
            try:
                from pptx import Presentation as _PPTXCheck  # noqa: F401
                st.download_button(
                    "🎯 PowerPoint Presentation (.pptx)",
                    data=_pptx_bytes(pasta_r, _proj_items),
                    file_name=_nome_base + "_presentation.pptx",
                    mime="application/vnd.openxmlformats-officedocument"
                         ".presentationml.presentation",
                    use_container_width=True,
                )
            except ImportError:
                st.warning(
                    "python-pptx not installed. "
                    "Run: `pip install python-pptx>=1.1`",
                    icon="⚠️",
                )
            except Exception as e_pptx:
                st.error(f"PowerPoint: {e_pptx}")

        st.divider()

        # ── Clean up old results ─────────────────────────────────────────
        with st.expander("🗑️ Free space — Clean up old results",
                         expanded=False):
            _pasta_base_lim = os.path.dirname(pasta_r)
            _pastas_exist = sorted(
                [p for p in os.scandir(_pasta_base_lim) if p.is_dir()],
                key=lambda p: p.stat().st_mtime, reverse=True)
            n_pastas = len(_pastas_exist)
            # Calculate total size
            def _tamanho_pasta_mb(pasta_p: str) -> float:
                tot = 0
                for raiz, _, arqs in os.walk(pasta_p):
                    for a in arqs:
                        try: tot += os.path.getsize(os.path.join(raiz, a))
                        except Exception:
                            logging.getLogger(__name__).debug("suppressed non-critical exception", exc_info=True)
                return round(tot / (1024 * 1024), 1)

            st.caption(f"Results folder: `{_pasta_base_lim}`  "
                       f"({n_pastas} runs stored)")
            if n_pastas > 1:
                _manter = st.slider(
                    "Keep N most recent runs",
                    min_value=1, max_value=max(1, n_pastas - 1),
                    value=min(3, n_pastas - 1), key="lim_manter")
                _n_remover = n_pastas - _manter
                _tam_est = sum(
                    _tamanho_pasta_mb(p.path)
                    for p in _pastas_exist[_manter:])
                st.info(
                    f"**{_n_remover}** old run(s) will be removed "
                    f"(~{_tam_est:.0f} MB freed). "
                    f"The current run **will not be affected**.")
                if st.button("🗑️ Confirm cleanup",
                             key="btn_limpar_resultados",
                             type="secondary"):
                    _res = pq.limpar_resultados_antigos(
                        _pasta_base_lim, _manter)
                    if _res["removidas"]:
                        st.success(
                            f"Removed {len(_res['removidas'])} folder(s), "
                            f"freed {_res['liberado_mb']:.0f} MB.")
                    else:
                        st.info("No folders removed.")
                    if _res["erro"]:
                        st.warning(f"Errors: {_res['erro']}")
            else:
                st.info("Only one run stored. Nothing to clean.")

        st.divider()

        # Model Card (Mitchell et al. 2019) — intended use, data, metrics,
        # caveats in one shareable document. Rendered as Markdown (tables
        # display nicely), with its own download button.
        st.markdown("### 🪪 Model Card")
        model_card_r = _ler_model_card(pasta_r)
        if model_card_r:
            with st.expander("View Model Card", expanded=False):
                st.markdown(model_card_r)
            st.download_button(
                "⬇️ Model Card (.md)",
                data=model_card_r.encode("utf-8"),
                file_name=_nome_base + "_model_card.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.info("File model_card.md not found.")

        st.divider()

        # Model summary
        st.markdown("### 📋 Model summary")
        resumo_r = _ler_resumo(pasta_r)
        if resumo_r:
            st.text_area("resumo_modelo.txt", resumo_r, height=400)
        else:
            st.info("File resumo_modelo.txt not found.")

        st.divider()

        # Full gallery with filter
        st.markdown("### 🖼️ Figure gallery")
        imgs_r = _listar_figuras(pasta_r)
        if imgs_r:
            _CATS_R = {
                "All":              "",
                "PCA":              "pca",
                "PLS-DA":           "plsda",
                "Outliers":         "outlier",
                "Confusion":        "confus",
                "ROC / AUC":        "roc",
                "VIP / SR":         "vip",
                "Loading":          "loading",
                "HCA":              "hca",
                "OPLS-DA":          "opls",
                "DD-SIMCA":         "ddsimca",
                "Cooman's Plot":    "cooman",
                "S-Plot":           "splot",
                "Permutation":      "permut",
                "Wold/ANOVA":       "wold",
                "Regression":       "regressao",
                "Benchmark":        "benchmark",
                "Monte Carlo CV":   "monte_carlo",
                "DET curves":       "fig_det",
                "SHAP":             "fig_shap",
            }
            filtro_r = st.selectbox("Filter figures",
                                    list(_CATS_R.keys()), key="filtro_rel")
            token_r  = _CATS_R[filtro_r].lower()
            imgs_filt_r = [im for im in imgs_r
                           if token_r in os.path.basename(im).lower()] \
                          if token_r else imgs_r
            st.caption(f"{len(imgs_filt_r)} figure(s) displayed.")
            n_cols_r = st.slider("Columns", 1, 3, 2, key="slider_cols_rel")
            cols_r   = st.columns(n_cols_r)
            for j, img in enumerate(imgs_filt_r):
                with cols_r[j % n_cols_r]:
                    st.image(img, caption=os.path.basename(img),
                             use_container_width=True)
        else:
            st.info("No PNG/JPG images found in the results folder.")

        st.divider()

        # Execution log
        if st.session_state.get("ultimo_log"):
            with st.expander("📜 Execution log (terminal output)"):
                st.code(st.session_state.ultimo_log, language="text")
