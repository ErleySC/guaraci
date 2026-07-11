"""app_tabs/modelo.py — Aba 4 (Model): parâmetros avançados + execução do
pipeline com progresso ao vivo. Extraído de app_quimiometria.py (item 18).
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import time
from typing import Callable, Dict, Optional

import streamlit as st

from guaraci.app_logic import coletar_config, fmt_tempo, progresso_do_log
from guaraci.app_logic import LogThreadSafe as _LogThreadSafe


def _ram_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _rodar_worker(pq, cfg, logger: _LogThreadSafe, estado: Dict):
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


def render(pq, cfg_base, specs: Dict, valores: Dict, T: Callable[[str], str],
           widget_para_campo: Callable, modo_analise_rotulo: Dict[str, str],
           modo_analise_ajuda: Dict[str, str], cfg_path: str) -> None:
    """Renderiza a aba Model. `valores` é o dict compartilhado com
    Data/Preprocessing (mesmo objeto, mutado em sequência)."""
    st.subheader(T("Model Parameters and Execution"))

    # ---- Quick-access Run button at the very top of the Model tab ----------
    # Rebuild enabled state from session_state widget keys (populated on reruns)
    _valores_top = {
        k: st.session_state[f"w_{k}"]
        for k in specs
        if f"w_{k}" in st.session_state
    }
    _cfg_top, _erros_top = coletar_config(cfg_base, _valores_top)
    _erros_top = _erros_top + pq._validar_semantico(_cfg_top)
    _ok_top = (not _erros_top) and pq._validar_pasta_dados(_cfg_top)[0]
    _rodar_top = st.button(
        "▶️ " + T("Run pipeline"), type="primary",
        disabled=not _ok_top,
        use_container_width=True,
        key="btn_run_top",
    )
    st.caption("ℹ️ Configure the options below, then click **▶️ Run pipeline**.")

    _MODELO_KEYS_ANALISE  = ["nivel", "objetivo", "max_lvs", "holdout_fracao",
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
                format_func=lambda v: modo_analise_rotulo.get(v, v))
            st.caption(modo_analise_ajuda.get(valores["nivel"], ""))
    with _cols_top[1]:
        _s_lvs = specs.get("max_lvs")
        if _s_lvs is not None:
            valores["max_lvs"] = widget_para_campo(
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
                valores[k] = widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    with st.expander("📊 Statistical validation", expanded=False):
        cols_v = st.columns(2)
        for i, k in enumerate(_MODELO_KEYS_VALID):
            s = specs.get(k)
            if s is None: continue
            with cols_v[i % 2]:
                valores[k] = widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

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
                valores[k] = widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    with st.expander("🖼️ Figures", expanded=False):
        cols_f = st.columns(2)
        for i, k in enumerate(_MODELO_KEYS_FIGURAS):
            s = specs.get(k)
            if s is None: continue
            with cols_f[i % 2]:
                valores[k] = widget_para_campo(s, pq._attr_para_yaml(s, cfg_base))

    st.divider()

    # ---- Final Config assembly and execution ----------------------------
    cfg_run, erros_run = coletar_config(cfg_base, valores)

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

    _objetivo_run = pq.resolver_objetivo(cfg_run)
    _plano_run = pq.descrever_plano(cfg_run)
    with st.expander(
        f"📋 {T('What will be generated')} — "
        f"{pq.OBJETIVO_ROTULO.get(_objetivo_run, _objetivo_run.capitalize())}",
        expanded=False,
    ):
        st.caption(
            T("Preview of the analyses/figures this run will produce, "
              "based on the scientific objective above."))
        if _plano_run:
            st.markdown("\n".join(f"- {item}" for item in _plano_run))
        else:
            st.caption(T("No mode-specific figures for this objective "
                          "(only the always-on overview figures)."))

    pode_rodar = ok_run and not erros_run
    rodar = st.button("▶️ " + T("Run pipeline"), type="primary",
                      disabled=not pode_rodar, use_container_width=True,
                      key="btn_rodar")
    if not pode_rodar:
        st.info(T("Fix the data input (Data tab) to enable."))

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
            pq.salvar_config(cfg_run, cfg_path)
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
            target=_rodar_worker, args=(pq, cfg_run, logger, estado), daemon=True)
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
                frac, nome = progresso_do_log(txt)
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
                    f"⏱️ `{fmt_tempo(elapsed)}` elapsed  |  "
                    f"⏳ `{fmt_tempo(eta_best) if eta_best else 'calculating…'}` remaining  |  "
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
                    label=f"❌ Pipeline failed after {fmt_tempo(elapsed)}.",
                    state="error", expanded=True)
            else:
                _run_status.update(
                    label=f"✅ Completed in {fmt_tempo(elapsed)}!",
                    state="complete", expanded=False)

        # Libera a trava só se o worker de fato terminou; se ficou órfão por
        # timeout (ainda vivo), mantém a referência para bloquear novo run.
        if not worker.is_alive():
            st.session_state["_worker_ativo"] = None

        st.session_state.ultimo_log  = txt
        if estado["erro"]:
            st.session_state.erro_run  = estado["erro"]
            st.session_state.ultima_pasta = None
            st.error(f"Pipeline failed after {fmt_tempo(elapsed)}.")
        else:
            st.session_state.erro_run  = None
            st.session_state.ultima_pasta = estado["pasta"]
            st.success("✅ Completed! View results in the Validation and Reports tabs.")

    if st.session_state.get("erro_run"):
        st.subheader("Error traceback")
        st.code(st.session_state.erro_run, language="text")
