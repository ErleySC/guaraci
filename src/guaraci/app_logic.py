"""app_logic.py — Lógica PURA da interface web (Streamlit), sem dependência do
próprio Streamlit.

Extraído de app_quimiometria.py (item 19 da auditoria: separar lógica testável
da camada de UI). Nada aqui importa `streamlit`, então cada função é testável
em isolamento (ver tests/test_app_logic.py). A UI apenas importa e usa.
"""
from __future__ import annotations

import copy
import os
import re
from typing import Dict, List, Optional, Tuple

# ── Parsing do log de progresso do pipeline ──────────────────────────────────
# O pipeline emite marcadores "[N/7]" (e sub-passos "[7b/7]", "[7c/7]") no
# stdout; a UI converte isso numa barra de progresso + rótulo legível.
_RE_ETAPA = re.compile(r"\[(\d+)[a-z]?/7\]")
_ETAPA_NOMES: Dict[int, str] = {
    0: "Validating input",
    1: "Spectral preprocessing",
    2: "Latent variable (LV) selection",
    3: "Exploratory PCA",
    4: "Validation tests (permutation / Wold / CV-ANOVA)",
    5: "Final metrics + bootstrap CI",
    6: "Figures, DD-SIMCA, OPLS-DA, holdout",
    7: "Regression / finalization and model saved",
}
# Sub-steps after step 7 (benchmark / MC CV)
_ETAPA_SUBSTEP: Dict[str, str] = {
    "[7b/7]": "Auto-Benchmark (SVM / RF / XGBoost vs PLS-DA)...",
    "[7c/7]": "Monte Carlo CV (95% CI by percentile)...",
}


def progresso_do_log(txt: str) -> Tuple[float, str]:
    """Deriva (fração 0..0.99, rótulo) do log acumulado do pipeline.

    Usa o MAIOR marcador "[N/7]" visto — o progresso nunca regride mesmo que o
    log traga linhas antigas. Retorna (0.0, "Starting...") se nada casou ainda.
    """
    achados = _RE_ETAPA.findall(txt)
    if not achados:
        return 0.0, "Starting..."
    n = max(int(a) for a in achados)
    nome = _ETAPA_NOMES.get(n, f"Step {n}/7")
    # Heavy sub-steps: show specific name for benchmark and MC CV
    if n >= 7:
        for tag, descricao in _ETAPA_SUBSTEP.items():
            if tag in txt:
                nome = descricao
                break
    return min(0.99, n / 7.0), nome


def fmt_tempo(seg) -> str:
    """Formata uma duração em segundos como string compacta (d/h/min/s).

    Robusto a None, não-numérico, NaN e negativo (retorna "—"/"0s").
    """
    if seg is None:
        return "—"
    try:
        seg = float(seg)
    except (TypeError, ValueError):
        return "—"
    if seg != seg or seg < 0:   # NaN ou negativo
        return "0s"
    seg = int(round(seg))
    d, r = divmod(seg, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    if d: return f"{d}d {h}h"
    if h: return f"{h}h {m:02d}min"
    if m: return f"{m}min {s:02d}s"
    return f"{s}s"


def coletar_config(cfg_base, valores: Dict):
    """Aplica os valores dos widgets a uma cópia profunda de Config.

    Percorre o _CONFIG_SPEC do pipeline (fonte única), coagindo cada valor com
    a mesma função do núcleo. Retorna (cfg, erros) — `erros` lista campos que
    falharam a coerção, sem interromper os demais.
    """
    import guaraci.pipeline as pq
    cfg = copy.deepcopy(cfg_base)
    erros: List[str] = []
    for s in pq._CONFIG_SPEC:
        if s["key"] not in valores:
            continue
        try:
            setattr(cfg, s["attr"], pq._coagir_valor(s, valores[s["key"]]))
        except Exception as e:
            erros.append(f"{s['key']}: {e}")
    return cfg, erros


# ── Leitura de artefatos de uma pasta de resultados ──────────────────────────
# Puro I/O de arquivo; a UI envolve com @st.cache_data (ver app_quimiometria.py)
# e guaraci.reports as usa diretamente (sem cache — geração é one-shot).
def listar_figuras(pasta: str) -> List[str]:
    """Lista os caminhos de figuras (.png/.jpg/.jpeg) em `pasta`, recursivo."""
    imgs: List[str] = []
    for raiz, _dirs, arqs in os.walk(pasta):
        for a in sorted(arqs):
            if a.lower().endswith((".png", ".jpg", ".jpeg")):
                imgs.append(os.path.join(raiz, a))
    return sorted(imgs)


def ler_resumo(pasta: str) -> Optional[str]:
    """Lê logs/resumo_modelo.txt (ou resumo_modelo.txt na raiz), se existir."""
    for candidato in [
        os.path.join(pasta, "logs", "resumo_modelo.txt"),
        os.path.join(pasta, "resumo_modelo.txt"),
    ]:
        if os.path.exists(candidato):
            with open(candidato, encoding="utf-8", errors="replace") as f:
                return f.read()
    return None


def ler_model_card(pasta: str) -> Optional[str]:
    """Lê logs/model_card.md (ou model_card.md na raiz), se existir."""
    for candidato in [
        os.path.join(pasta, "logs", "model_card.md"),
        os.path.join(pasta, "model_card.md"),
    ]:
        if os.path.exists(candidato):
            with open(candidato, encoding="utf-8", errors="replace") as f:
                return f.read()
    return None


__all__ = ["progresso_do_log", "fmt_tempo", "coletar_config",
           "listar_figuras", "ler_resumo", "ler_model_card",
           "_RE_ETAPA", "_ETAPA_NOMES", "_ETAPA_SUBSTEP"]
