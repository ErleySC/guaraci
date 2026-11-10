"""app_tabs/modelo.py — Tela (Model): parâmetros avançados + execução do
pipeline com progresso ao vivo. Extraído de app_quimiometria.py (item 18).
"""
from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from typing import Callable, Dict, Optional

import streamlit as st

from guaraci import web_seguranca as _ws
from guaraci.app_logic import collect_config, fmt_time, log_progress
from guaraci.app_logic import LogThreadSafe as _LogThreadSafe
from guaraci.cli_assistente import apply_palette
from guaraci.preferencias_visuais import load_visual_config


#: Guarda de tempo maximo da execucao (2 h): evita travar a sessao para sempre na nuvem. Constante de modulo para o teste
#: poder reduzi-la.
MAX_RUNTIME_S = 7200


def _ram_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001 -- indicador cosmetico best-effort
        # (psutil ausente ou probe falha por variacao de SO/permissao);
        # None e' o fallback documentado pelo tipo de retorno Optional.
        return None


def _paleta_escolhida() -> str:
    """Paleta ativa do usuario, vinda do MESMO arquivo de preferencias que a
    CLI grava (`~/.guaraci/visual_config.json`) — sem estado de cor paralelo
    só da web."""
    return str(load_visual_config().get("paleta", "qualitativo"))


def _rodar_worker(pq, cfg, logger: _LogThreadSafe, estado: Dict):
    try:
        with contextlib.redirect_stdout(logger), \
             contextlib.redirect_stderr(logger):
            pq.executar(cfg)
        estado["pasta"] = getattr(cfg, "output_folder", None)
        estado["erro"] = None
    except Exception:  # noqa: BLE001 -- boundary de topo da thread de
        # execucao (mesmo padrao de guaraci.py): qualquer excecao do
        # pipeline precisa ser capturada aqui p/ nao morrer silenciosamente
        # numa thread -- traceback completo salvo em estado["erro"] e
        # exibido na UI.
        import traceback
        estado["erro"] = traceback.format_exc()
    finally:
        estado["fim"] = True


def render(pq, cfg_base, specs: Dict, valores: Dict, T: Callable[[str], str],
           widget_para_campo: Callable, modo_analise_rotulo: Dict[str, str],
           modo_analise_ajuda: Dict[str, str],
           cfg_path: Optional[str] = None) -> None:
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
    _cfg_top, _erros_top = collect_config(cfg_base, _valores_top)
    _erros_top = _erros_top + pq.validar_semantico(_cfg_top)
    _ok_top = (not _erros_top) and pq.validar_fonte_de_dados(_cfg_top)[0]
    _rodar_top = st.button(
        "▶️ " + T("Run pipeline"), type="primary",
        disabled=not _ok_top,
        use_container_width=True,
        key="btn_run_top",
    )
    st.caption(T("ℹ️ Configure the options below, then click **▶️ Run pipeline**."))

    _MODELO_KEYS_ANALISE  = ["nivel", "objetivo", "max_lvs", "holdout_fracao",
                              "holdout_estratificado", "validacao_group_aware"]
    _MODELO_KEYS_VALID    = ["n_permutacoes", "teste_wold", "teste_cv_anova",
                              "teste_martens", "n_jobs_permutacao",
                              "selecao_lv_cv_aninhada"]
    _MODELO_KEYS_EXTRAS   = ["selecao_variaveis_etapa4", "selecao_spa", "selecao_ag",
                              "selecao_cars", "selecao_uve",
                              "ddsimca", "modo_ddsimca", "ddsimca_pcv", "opls_da",
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
        # Os campos desta tela existem sempre em `_CONFIG_SPEC` (teste `test_campos_das_abas_existem_no_spec`).
        _s_niv = specs["nivel"]
        _niv_atual = pq.attr_para_yaml(_s_niv, cfg_base)
        _ops_niv = list(_s_niv.get("opcoes") or ["N1", "N2", "N3"])
        _idx_niv = _ops_niv.index(_niv_atual) if _niv_atual in _ops_niv else 0
        valores["nivel"] = st.selectbox(
            T("Analysis mode"), _ops_niv, index=_idx_niv, key="w_nivel",
            format_func=lambda v: modo_analise_rotulo.get(v, v))
        st.caption(modo_analise_ajuda.get(valores["nivel"], ""))
    with _cols_top[1]:
        _s_lvs = specs["max_lvs"]
        valores["max_lvs"] = widget_para_campo(
            _s_lvs, pq.attr_para_yaml(_s_lvs, cfg_base))

    # Preprocessing is chosen in the Preprocessing tab (single source of truth).
    # Mirror that choice here read-only. Previously a second editable widget with
    # a separate key rendered here and, running later, silently overrode the
    # user's Preprocessing-tab choice with its own default.
    _s_pre = specs["pre_processamento"]
    _pre_escolhido = st.session_state.get(
        "w_pre_processamento", pq.attr_para_yaml(_s_pre, cfg_base))
    valores["pre_processamento"] = _pre_escolhido
    st.caption(
        f"⚗️ Pré-processamento: **{_pre_escolhido}** "
        "— definido no separador *Pré-processamento*.")

    st.divider()

    with st.expander(T("🧮 Analysis & partitioning"), expanded=True):
        cols_a = st.columns(2)
        _MODELO_KEYS_ANALISE_EXP = [k for k in _MODELO_KEYS_ANALISE if k not in _KEY_TOP]
        for i, k in enumerate(_MODELO_KEYS_ANALISE_EXP):
            s = specs[k]
            with cols_a[i % 2]:
                valores[k] = widget_para_campo(s, pq.attr_para_yaml(s, cfg_base))

    with st.expander(T("📊 Statistical validation"), expanded=False):
        cols_v = st.columns(2)
        for i, k in enumerate(_MODELO_KEYS_VALID):
            s = specs[k]
            with cols_v[i % 2]:
                valores[k] = widget_para_campo(s, pq.attr_para_yaml(s, cfg_base))

    with st.expander(T("🧩 Extra modules"), expanded=False):
        # Hardware warning for heavy operations
        try:
            _hw_mod = pq.hardware_probe()
            _ram_l  = _hw_mod.get("ram_livre_gb", 16.0)
            # R5-19: informativo (extras desligados), nao erro -- e traduzido.
            # RAM nao medida (None): sem aviso, nada sera desligado.
            if _ram_l is None:
                pass
            elif _ram_l < 4.0:
                st.warning(T("Free RAM: {ram} GB — Benchmark and SHAP will be switched off automatically to avoid freezing. The scientific results do not change.").format(ram=f"{_ram_l:.1f}"))
            elif _ram_l < 8.0:
                st.info(T("Free RAM: {ram} GB — SHAP and Monte Carlo CV will be reduced automatically. The scientific results do not change.").format(ram=f"{_ram_l:.1f}"))
        except Exception:  # noqa: BLE001 -- aviso cosmetico best-effort
            # (hardware_probe ja e' fail-safe por contrato, mas psutil pode
            # variar entre plataformas); sem o aviso, o pipeline ainda roda.
            logging.getLogger(__name__).debug(
                "aviso de RAM nao exibido (hardware_probe falhou)")

        cols_e = st.columns(2)
        for i, k in enumerate(_MODELO_KEYS_EXTRAS):
            s = specs[k]
            with cols_e[i % 2]:
                valores[k] = widget_para_campo(s, pq.attr_para_yaml(s, cfg_base))

    with st.expander(T("🖼️ Figures"), expanded=False):
        cols_f = st.columns(2)
        for i, k in enumerate(_MODELO_KEYS_FIGURAS):
            s = specs[k]
            with cols_f[i % 2]:
                valores[k] = widget_para_campo(s, pq.attr_para_yaml(s, cfg_base))
        st.divider()
        # A escolha de paleta mora na tela Visualização (mockup de
        # 2026-09-08) — aqui fica só o lembrete de qual está ativa, para
        # não haver dois lugares disputando a mesma preferência.
        st.caption(T("Figure colour palette: **{p}** — change it on the "
                     "Visualisation screen.").format(p=_paleta_escolhida()))

    # ---- Parametros avancados: o que nenhuma aba lista a mao, derivado do catalogo
    from guaraci import catalogo as _cat
    from guaraci.app_tabs.dados import _DADOS_KEYS
    from guaraci.app_tabs.preprocessamento import _PREPROC_KEYS
    _explicitas = (_DADOS_KEYS + _PREPROC_KEYS + _MODELO_KEYS_ANALISE + _MODELO_KEYS_VALID
                   + _MODELO_KEYS_EXTRAS + _MODELO_KEYS_FIGURAS)
    _restantes = _cat.campos_restantes_web(_explicitas, specs)
    if _restantes:
        _L = 1 if st.session_state.get("lang") == "EN" else 0
        with st.expander(T("⚙️ Advanced parameters"), expanded=False):
            for tid in dict.fromkeys(_cat.OPCOES[k]["tela"] for k in _restantes):
                st.caption(f"**{_cat.TELAS[tid]['nome'][_L]}**")
                ks = [k for k in _restantes if _cat.OPCOES[k]["tela"] == tid]
                cols_r = st.columns(2)
                for i, k in enumerate(ks):
                    with cols_r[i % 2]:
                        valores[k] = widget_para_campo(specs[k], pq.attr_para_yaml(specs[k], cfg_base))

    st.divider()

    # ---- Final Config assembly and execution ----------------------------
    cfg_run, erros_run = collect_config(cfg_base, valores)

    # Modo servidor: defesa em profundidade -- a Config que vai rodar SEMPRE
    # respeita a política (saída na pasta da sessão, entrada só sintética/CSV
    # da sessão), mesmo que a UI já esconda os campos.
    if _ws.modo_servidor():
        _reduzidos = _ws.tetos_excedidos(cfg_run)
        _ws.impor_politica_servidor(cfg_run, st.session_state["_sessao_id"])
        if _reduzidos:
            st.info(T("Public server limit applied (run locally with `guaraci web` for larger values):")
                    + " " + "; ".join(f"{a}: {v} → {t}" for a, v, t in _reduzidos))

    ok_run, msg_run = pq.validar_fonte_de_dados(cfg_run)
    erros_run = erros_run + pq.validar_semantico(cfg_run)
    st.write(T("**Input:**"), _ws.ocultar_caminhos(msg_run) if _ws.modo_servidor() else msg_run)
    if erros_run:
        st.error("Invalid fields in configuration:\n- " + "\n- ".join(erros_run))

    # B4b: nunca "anti-vazamento ativo" so' porque a flag esta ligada -- o
    # nivel de garantia depende de haver chave de grupo.
    from guaraci.dados_io import garantia_agrupamento_prevista
    _nivel_g = garantia_agrupamento_prevista(cfg_run)
    if _nivel_g == "none":
        st.warning(T(
            "⚠️ No group key: validation will NOT be grouped (StratifiedKFold). "
            "Replicates of the same sample may fall on both sides of a split, "
            "so treat the metrics as exploratory. Fix: in the Data screen, set "
            "'group column' or upload a group file."))
    elif _nivel_g == "medium":
        st.info(T(
            "Validation grouped by the group you declared — guarantee MEDIUM "
            "(the software cannot verify that groups are independent)."))

    _objetivo_run = pq.resolve_objective(cfg_run)
    _plano_run = pq.describe_plan(cfg_run)
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
        # Clicar Run NAO grava config.yaml:
        # antes, um Run gravava config.yaml e a pasta de resultados na pasta do app
        # e, com isso, desligava o modo demo. A configuracao e' estado da
        # sessao; "Save config.yaml" (Dados) e' a unica gravacao, explicita.

        # Paleta escolhida vale a partir daqui: aplicada ANTES de executar()
        # (mesmo ponto e mesma funcao que a CLI usa em `_rodar_pipeline`), o
        # que faz as figuras desta execucao sairem com a cor escolhida.
        apply_palette(_paleta_escolhida())

        logger = _LogThreadSafe(tee=sys.__stdout__)
        estado: Dict = {"fim": False, "erro": None, "pasta": None}
        worker = threading.Thread(
            target=_rodar_worker, args=(pq, cfg_run, logger, estado), daemon=True)
        st.session_state["_worker_ativo"] = worker

        t0 = time.monotonic()
        eta_best: Optional[float] = None
        worker.start()

        # Use st.status() + single placeholder to avoid simultaneous DOM
        # mutations that cause React's removeChild error on Streamlit Cloud.
        with st.status("⚙️ Running pipeline...", expanded=True) as _run_status:
            ph = st.empty()
            while not estado["fim"]:
                # Safety timeout
                if time.monotonic() - t0 > MAX_RUNTIME_S:
                    estado["fim"] = True
                    estado["erro"] = "Pipeline exceeded maximum runtime (2 h)."
                    break
                txt = logger.text()
                frac, nome = log_progress(txt, len(_plano_run) or None)
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
                    f"⏱️ `{fmt_time(elapsed)}` elapsed  |  "
                    f"⏳ `{fmt_time(eta_best) if eta_best else 'calculating…'}` remaining  |  "
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
                    label=f"❌ Pipeline failed after {fmt_time(elapsed)}.",
                    state="error", expanded=True)
            else:
                _run_status.update(
                    label=f"✅ Completed in {fmt_time(elapsed)}!",
                    state="complete", expanded=False)

        # Libera a trava só se o worker de fato terminou; se ficou órfão por
        # timeout (ainda vivo), mantém a referência para bloquear novo run.
        if not worker.is_alive():
            st.session_state["_worker_ativo"] = None

        st.session_state.ultimo_log  = txt
        st.session_state.pop("_erro_run_id", None)
        if estado["erro"]:
            st.session_state.erro_run  = estado["erro"]
            st.session_state.ultima_pasta = None
            st.error(f"Pipeline failed after {fmt_time(elapsed)}.")
        else:
            st.session_state.erro_run  = None
            st.session_state.ultima_pasta = estado["pasta"]
            # R2-10: CRITICO na auditoria (ex.: uma so' classe) nao pode vir so'
            # com um "Concluido!" verde -- o aviso aparece aqui, na hora.
            _crit = [l for l in txt.splitlines() if "[AUDITORIA][CRITICO]" in l]
            if _crit:
                st.warning(T("⚠️ Completed, but with CRITICAL design problems — do not cite the metrics before reading them:"))
                for _l in _crit[:6]:
                    st.markdown("- " + _ws.msg_segura(_l.split("[AUDITORIA][CRITICO]", 1)[1].strip()))
            else:
                st.success(T("✅ Completed! View results in the Validation and Reports tabs."))

    if st.session_state.get("erro_run"):
        if _ws.modo_servidor():
            # Nunca traceback nem caminho ao visitante: id de erro + mensagem
            # curta; o traceback completo vai para o log do servidor.
            if not st.session_state.get("_erro_run_id"):
                st.session_state["_erro_run_id"] = _ws.registrar_erro(
                    st.session_state.erro_run, "execução do pipeline")
            _ultima = st.session_state.erro_run.strip().splitlines()[-1]
            st.error(
                "Não foi possível concluir a análise. Motivo: "
                + _ws.ocultar_caminhos(_ultima)[:300]
                + f" (código do erro: {st.session_state['_erro_run_id']})")
        else:
            st.subheader(T("Error traceback"))
            st.code(st.session_state.erro_run, language="text")
    else:
        st.session_state.pop("_erro_run_id", None)
