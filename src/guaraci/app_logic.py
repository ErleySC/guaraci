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
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from guaraci.config import NOME_RELATORIOS


class LogThreadSafe:
    """Captura stdout/stderr de executar() num buffer protegido por lock.

    Extraido do app web (item 18/auditoria jul/2026, item 5: painel de
    acompanhamento tambem no terminal) para ser reutilizado pelo CLI
    (guaraci.py) e pelo app (app_tabs/modelo.py) a partir da MESMA classe —
    antes cada interface tinha sua propria copia. `tee`, se fornecido,
    recebe uma copia de cada escrita (usado pelo CLI para nao perder o
    log em disco/console caso o usuario redirecione stdout externamente).
    """
    def __init__(self, tee=None):
        self._buf: List[str] = []
        self._lock = threading.Lock()
        self._tee = tee

    def write(self, s: str):
        with self._lock:
            self._buf.append(s)
        if self._tee is not None:
            try:
                self._tee.write(s)
            except (OSError, ValueError):
                # tee quebrado (ex.: stdout fechado/pipe partido) -- o buffer
                # acima (fonte do painel) ja recebeu a linha; logar aqui
                # spamaria a cada escrita (chamado por linha de output do
                # pipeline inteiro), entao so' para de espelhar, sem crashar.
                pass
        return len(s)

    def flush(self):
        if self._tee is not None:
            try:
                self._tee.flush()
            except (OSError, ValueError):
                pass   # mesmo motivo de write() acima

    def text(self) -> str:
        with self._lock:
            return "".join(self._buf)


# ── Parsing de figuras concluidas e avisos (painel de acompanhamento) ───────
# O pipeline imprime "  -> <caminho>.<ext>" apos salvar cada figura/tabela/
# relatorio, e "[AVISO] <texto>" para inconsistencias nao-fatais. Ambos os
# padroes ja existiam no log; aqui so' extraimos estrutura deles para
# alimentar o painel ao vivo (CLI) e a barra de progresso (app web).
_RE_ARQUIVO_SALVO = re.compile(
    r"^\s*->\s*.*[\\/]([\w\-]+)\.(?:png|jpg|jpeg|pdf|svg)\s*$", re.MULTILINE)
_RE_AVISO = re.compile(r"^\s*\[AVISO\]\s*(.+)$", re.MULTILINE)


def figures_completed(txt: str) -> List[str]:
    """Nomes (sem extensao) das figuras já salvas em disco, na ordem em que
    apareceram no log, sem duplicatas."""
    vistos: List[str] = []
    for nome in _RE_ARQUIVO_SALVO.findall(txt or ""):
        if nome not in vistos:
            vistos.append(nome)
    return vistos


def log_warnings(txt: str) -> List[str]:
    """Textos de aviso ([AVISO] ...) já emitidos, na ordem, sem duplicatas."""
    vistos: List[str] = []
    for aviso in _RE_AVISO.findall(txt or ""):
        aviso = aviso.strip()
        if aviso and aviso not in vistos:
            vistos.append(aviso)
    return vistos

# ── Parsing do log de progresso do pipeline ──────────────────────────────────
# O pipeline emite marcadores "[N/7]" (e sub-passos "[6b/7]", "[6c/7]",
# "[7b/7]", "[7c/7]") no stdout; a UI converte isso numa barra de progresso +
# rótulo legível.
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
# Sub-passos de uma etapa (sufixo de letra em "[Nb/7]"/"[Nc/7]"): cada
# entrada mapeia a tag para (etapa numerica N a que pertence, rotulo
# legivel). "[6b/7]"/"[6c/7]" adicionados em 2026-08-07 -- ja existiam no
# log do pipeline (pipeline.py:1866,1884) mas nao eram reconhecidos aqui
# (so' o `if n >= 7` cobria sub-passos), entao o rotulo ficava generico
# durante eles. Ver `log_progress` para o bug de fundo que isso ajuda
# a mitigar.
_ETAPA_SUBSTEP: Dict[str, Tuple[int, str]] = {
    "[6b/7]": (6, "Comparing preprocessing pipelines..."),
    "[6c/7]": (6, "External holdout evaluation..."),
    "[7b/7]": (7, "Auto-Benchmark (SVM / RF / XGBoost vs PLS-DA)..."),
    "[7c/7]": (7, "Monte Carlo CV (95% CI by percentile)..."),
}


def log_progress(txt: str,
                      total_figuras_planejadas: Optional[int] = None
                      ) -> Tuple[float, str]:
    """Deriva (fração 0..0.99, rótulo) do log acumulado do pipeline.

    Usa o MAIOR marcador "[N/7]" visto — o progresso nunca regride mesmo que o
    log traga linhas antigas. Retorna (0.0, "Starting...") se nada casou ainda.

    CORRIGIDO em 2026-08-07 ("bug do progresso" relatado no CLI): a etapa
    "[6/7]" (geração de figuras + DD-SIMCA + OPLS-DA + holdout) concentra a
    maior parte do tempo real de execução, mas só tinha 2 marcadores de
    texto OPCIONAIS ("[6b/7]"/"[6c/7]", nem sempre emitidos) entre o início
    da etapa e o fim — sem eles, o progresso ficava CRAVADO em 6/7≈85,7%
    durante toda essa fase. Medido reproduzindo o mecanismo exato do painel
    (thread em background + `contextlib.redirect_stdout`, ver
    `scripts/medicoes/medir_bug_progresso_cli.py`) num run sintético pequeno:
    96,1% das amostras de progresso ficaram cravadas em 0,857, mesmo com
    figuras sendo salvas visivelmente no log. Com a correção, essa mesma
    fração cai para 31,1% (e o que resta é o platô legítimo perto do fim
    da etapa, não mais um travamento).

    `total_figuras_planejadas` (opcional, retrocompatível — sem ele o
    comportamento é IDÊNTICO ao anterior): quando fornecido e a etapa atual
    é a 6, soma um bônus fracionário proporcional a
    `len(figures_completed(txt)) / total_figuras_planejadas` — o progresso
    passa a avançar suavemente conforme cada figura é salva, em vez de só
    saltar nos 2 marcadores de texto esparsos. Nunca regride e nunca atinge
    o próximo número inteiro de etapa (capado abaixo de 7/7).
    """
    achados = _RE_ETAPA.findall(txt)
    if not achados:
        return 0.0, "Starting..."
    n = max(int(a) for a in achados)
    nome = _ETAPA_NOMES.get(n, f"Step {n}/7")
    for tag, (n_tag, descricao) in _ETAPA_SUBSTEP.items():
        if n_tag == n and tag in txt:
            nome = descricao
            break

    n_efetivo = float(n)
    if n == 6 and total_figuras_planejadas:
        n_feitas = len(figures_completed(txt))
        bonus = min(0.99, n_feitas / total_figuras_planejadas)
        n_efetivo = n + bonus

    return min(0.99, n_efetivo / 7.0), nome


def fmt_time(seg) -> str:
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


def collect_config(cfg_base, valores: Dict):
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
        except ValueError as e:   # _coagir_valor: valor do widget invalido
            erros.append(f"{s['key']}: {e}")
    return cfg, erros


# ── Caminho seguro p/ arquivo temporario de upload (achado de auditoria de
#    seguranca, 2026-08-07) ──────────────────────────────────────────────────
def temp_upload_path(nome_original: str, session_id: str, *,
                        base: Optional[Path] = None,
                        subpasta: str = "pq_uploads") -> Path:
    """Caminho seguro para salvar um arquivo temporario recebido via upload
    da UI web (CSV de dados, modelo .joblib).

    Duas protecoes:

    1. So' o BASENAME de `nome_original` e' usado (`Path(...).name`) -- um
       nome de arquivo como "../../etc/passwd" nao consegue escapar do
       diretorio de destino (bloqueia path traversal).
    2. Isolado numa subpasta por `session_id`. Sem isso, uploads de
       sessoes/visitantes DIFERENTES caem no MESMO caminho previsivel
       (`{tempdir}/pq_uploads/<nome do arquivo>`), o que (a) e' uma
       condicao de corrida real entre sessoes concorrentes e (b)
       participava do bypass de RCE via pickle registrado como achado S1
       da auditoria de 2026-08-07 (ver
       AUDITORIA_SEGURANCA_2026-08-07.md). `session_id` deve ser um valor
       aleatorio gerado uma vez por sessao (nunca exposto ao cliente,
       ex.: `uuid.uuid4().hex` guardado em `st.session_state`), nunca
       previsivel/derivado de dado do usuario.

    `base` (opcional, default `tempfile.gettempdir()`) existe so' para
    tornar a funcao testavel sem depender do diretorio temp real do SO.
    """
    raiz = base if base is not None else Path(tempfile.gettempdir())
    destino = raiz / subpasta / session_id
    return destino / Path(nome_original).name


# ── Leitura de artefatos de uma pasta de resultados ──────────────────────────
# Puro I/O de arquivo; a UI envolve com @st.cache_data (ver app_quimiometria.py)
# e guaraci.reports as usa diretamente (sem cache — geração é one-shot).
def list_figures(pasta: str) -> List[str]:
    """Lista os caminhos de figuras (.png/.jpg/.jpeg) em `pasta`, recursivo."""
    imgs: List[str] = []
    for raiz, _dirs, arqs in os.walk(pasta):
        for a in sorted(arqs):
            if a.lower().endswith((".png", ".jpg", ".jpeg")):
                imgs.append(os.path.join(raiz, a))
    return sorted(imgs)


def load_summary(pasta: str) -> Optional[str]:
    """Lê Relatorios/resumo_modelo.txt (ou variantes antigas: logs/ da
    estrutura pre-jul/2026, ou resumo_modelo.txt na raiz), se existir."""
    for candidato in [
        os.path.join(pasta, NOME_RELATORIOS, "resumo_modelo.txt"),
        os.path.join(pasta, "logs", "resumo_modelo.txt"),  # runs antigos
        os.path.join(pasta, "resumo_modelo.txt"),
    ]:
        if os.path.exists(candidato):
            with open(candidato, encoding="utf-8", errors="replace") as f:
                return f.read()
    return None


def load_model_card(pasta: str) -> Optional[str]:
    """Lê Relatorios/model_card.md (ou variantes antigas: logs/ da estrutura
    pre-jul/2026, ou model_card.md na raiz), se existir."""
    for candidato in [
        os.path.join(pasta, NOME_RELATORIOS, "model_card.md"),
        os.path.join(pasta, "logs", "model_card.md"),   # runs antigos
        os.path.join(pasta, "model_card.md"),
    ]:
        if os.path.exists(candidato):
            with open(candidato, encoding="utf-8", errors="replace") as f:
                return f.read()
    return None


__all__ = ["log_progress", "fmt_time", "collect_config",
           "list_figures", "load_summary", "load_model_card",
           "LogThreadSafe", "figures_completed", "log_warnings",
           "temp_upload_path"]
