"""app_tabs/dados.py — Tela (Data): entrada de dados (.dx, CSV local/upload)
+ prévia dos espectros. Extraído de app_quimiometria.py (item 18).
"""
from __future__ import annotations

import copy
import os
from typing import Callable, Dict

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from guaraci.spectra_preview import (preview_spectra_dx, preview_spectra_csv, plot_mean_spectra,
                                     remover_faixas_excluidas)
from guaraci import web_seguranca as _ws
from guaraci.app_logic import collect_config


#: Campos desta aba (nivel de modulo: a aba Modelo usa para achar os que sobram).
_DADOS_KEYS = ["modo_entrada", "perfil_matriz", "perfil_tecnica", "eixo_unidade", "tecnica_analitica", "pasta_dados",
               "arquivo_csv", "coluna_classe", "coluna_concentracao",
               "coluna_grupo", "arquivo_grupos", "unidade_agrupamento",
               "aplicar_correcoes_locais",
               "pasta_saida", "ajuste_automatico", "saturacao", "faixas_excluir", "excluir_classes", "imagem_incluir_textura",
               "hsi_pasta_dataset", "hsi_calibracao", "hsi_referencia_branco",
               "hsi_referencia_escuro", "hsi_eixo_bandas", "hsi_validacao", "hsi_cv_dobras",
               "imagem_referencia", "imagem_segmentacao", "hplc_detector", "cromatografia_canal",
               "espectro_max_pontos", "cromatograma_max_pontos",
               "eem_espalhamento", "eem_espalhamento_largura_nm", "eem_raman", "parafac_nao_negativo",
               "icp_unidade", "icp_coluna_amostra", "icp_coluna_tipo",
               "icp_coluna_sessao", "icp_padrao_interno", "icp_politica_censura",
               "icp_limiar_censura", "icp_log"]


def _sincronizar_widgets_com_cfg(pq, cfg, apenas_chaves=None) -> None:
    """Escreve o valor de cada campo (via `_CONFIG_SPEC`) direto em
    `st.session_state["w_"+key]`, forcando os widgets a refletir `cfg` no
    proximo render.

    Necessario porque widgets com `key=` estatico (todo `_widget_para_campo`
    em app_quimiometria.py, e o selectbox "nivel" em app_tabs/modelo.py) so'
    re-sincronizam o valor EXIBIDO a partir de uma escrita explicita em
    session_state -- mudar `value=`/`index=` ou apagar a chave nao basta
    (verificado empiricamente: um combobox ficou mostrando o rotulo antigo
    enquanto o valor usado ja' era o novo por baixo -- corrigido 2026-07-13).

    `apenas_chaves`: se dado, sincroniza so' esses campos (aplicacao PARCIAL
    de um preset, sem mexer no resto da config atual do usuario). Se `None`,
    sincroniza TODOS os campos do spec (reload completo de um config.yaml).
    """
    for s in pq.CONFIG_SPEC:
        if apenas_chaves is not None and s["key"] not in apenas_chaves:
            continue
        valor = pq.attr_para_yaml(s, cfg)
        # Cada tipo de widget so' aceita o seu tipo de valor (mesma conversao de `_widget_para_campo`):
        # lista vira texto separado por virgula e ausente vira vazio; antes, recarregar um config.yaml
        # com campo opcional vazio quebrava o text_input com TypeError.
        if s["tipo"] in ("list", "recorte"):
            valor = ", ".join(str(x) for x in (valor or ()))
        elif s["tipo"] not in ("bool", "int", "float", "choice", "preproc"):
            valor = "" if valor is None else str(valor)
        st.session_state[f"w_{s['key']}"] = valor


def render(pq, cfg_base, specs: Dict, valores: Dict,
           widget_para_campo: Callable, cfg_path: str,
           T: Callable[[str], str] = lambda s: s) -> None:
    """Renderiza a aba Data.

    `valores` é o dict COMPARTILHADO entre Data/Preprocessing/Model (o mesmo
    objeto é mutado por cada aba — mesma semântica de antes da extração,
    já que o Config final só é montado depois que as 3 abas rodaram).

    `T`: traducao PT/EN (mesma funcao `_T` de app_quimiometria.py, ver
    `_TR`). Default no-op (`lambda s: s`) so' para chamada direta em teste
    isolado -- toda chamada real do app passa `_T`.
    """
    st.subheader(T("Data Input"))
    _recarregada = st.session_state.pop("_cfg_recarregada", None)
    if _recarregada is not None:
        # Sem isso, widgets (ex.: "Modo de analise") continuam mostrando o valor ANTIGO mesmo
        # com cfg_base atualizado. Aqui, antes de qualquer widget ser criado, a troca e' permitida.
        _sincronizar_widgets_com_cfg(pq, _recarregada)
        cfg_base = _recarregada
        st.success(T("Config reloaded."))
    _preset_msg = st.session_state.pop("_preset_aplicado", None)
    if _preset_msg:
        st.success(T("Preset '{pname}' applied — check the Model tab.").format(pname=_preset_msg))
    st.caption(T("📂 Step 2: Upload or select spectra folder → then go to **Model** tab."))

    # ---- Recommended analysis presets: 3 presets (Autenticar/Explorar/Quantificar)
    # so a new user doesn't have to understand nivel/objetivo before picking
    # something reasonable.
    # Same PROFILES dict as the CLI's [P] menu — one source of truth, no
    # duplicated preset definitions between interfaces.
    st.markdown(T("**🎯 Recommended analysis** *(optional shortcut)*"))
    st.caption(T("Pick what you want to do — sets sensible defaults across all "
               "tabs; fine-tune afterwards if needed."))
    from guaraci import catalogo as _cat
    _L = 1 if st.session_state.get("lang") == "EN" else 0
    _PRESETS_OBJETIVO = [k for k, v in _cat.PERFIS_OBJETIVO.items() if v["cfg"] and k != "imagem"]
    cols_preset = st.columns(len(_PRESETS_OBJETIVO))
    for pname, col in zip(_PRESETS_OBJETIVO, cols_preset):
        with col:
            _po = _cat.PERFIS_OBJETIVO[pname]
            if st.button(_po["nome"][_L], key=f"btn_preset_{pname}",
                         use_container_width=True, help=_po["faz"][_L]):
                # mesmo perfil do CLI ([P]): objetivo + rigor padrao
                pdata = dict(_po["cfg"], **_cat.PERFIS_RIGOR["padrao"]["cfg"])
                cfg_novo = copy.deepcopy(cfg_base)
                for s in pq.CONFIG_SPEC:
                    if s["key"] in pdata:
                        setattr(cfg_novo, s["attr"], pdata[s["key"]])
                _sincronizar_widgets_com_cfg(pq, cfg_novo, apenas_chaves=pdata.keys())
                st.session_state.cfg_base = cfg_novo
                # R5-18: a mensagem sumia no rerun; fica guardada para a proxima
                # renderizacao.
                st.session_state["_preset_aplicado"] = _po["nome"][_L]
                st.rerun()

    # ---- CSV Upload (at top for easy access) -----------------------------
    st.markdown(T("**Upload CSV** *(alternative to the local path below)*"))
    upld = st.file_uploader(
        T("Drag or select a CSV file"),
        type=["csv", "txt"],
        key="csv_upload_widget",
        help=T("The file will be saved to a temporary folder and the path adjusted automatically."),
    )
    if _ws.modo_servidor():
        st.caption(T("Do not upload personal or confidential data."))
    if upld is not None and st.session_state.get("_csv_upload_name") != f"{upld.name}:{upld.size}":
        # Pasta POR SESSAO (parecer 2026-09-25, M-01): id aleatório, nunca
        # exposto; só o basename do arquivo é usado; a pasta é apagada ao fim
        # da sessão (ver web_seguranca.GuardaSessao).
        destino = _ws.salvar_upload(
            st.session_state["_sessao_id"], upld.name, upld.getvalue())
        st.session_state["_csv_upload_name"] = f"{upld.name}:{upld.size}"
        # O upload passa a SER a configuração (não um override escondido no
        # Run): modo csv + caminho do arquivo entram em cfg_base e nos widgets.
        cfg_up = copy.deepcopy(st.session_state.cfg_base)
        # Tabela ICP (multielementar) tambem chega por CSV: o upload so' muda o
        # modo para 'csv' se o modo escolhido nao for o multielementar.
        cfg_up.mode = "multielementar" if cfg_up.mode == "multielementar" else "csv"
        cfg_up.csv_file = str(destino)
        _sincronizar_widgets_com_cfg(pq, cfg_up, apenas_chaves={"modo_entrada", "arquivo_csv"})
        st.session_state.cfg_base = cfg_up
        cfg_base = cfg_up
    if upld is not None:
        st.success(T("File received: `{nome}`").format(nome=os.path.basename(upld.name)))
        if getattr(st.session_state.cfg_base, "mode", "") != "multielementar":
            st.info(T("Mode automatically set to 'csv'."))

    if _ws.modo_servidor():
        # Arquivo de grupos (amostra -> grupo) por upload: sem ele a validacao
        # nao e' agrupada (a menos que o CSV tenha coluna de grupo).
        upl_g = st.file_uploader(
            T("Group file (optional) — CSV: sample id, group"),
            type=["csv", "txt", "tsv"], key="grupos_upload_widget",
            help=T("Replicates of the same physical sample must share the "
                   "group; it keeps them on the same side of every split."))
        if upl_g is not None:
            _chave_g = f"{upl_g.name}:{upl_g.size}"
            if st.session_state.get("_grupos_upload_name") != _chave_g:
                destino_g = _ws.salvar_upload(
                    st.session_state["_sessao_id"], "grupos_" + upl_g.name,
                    upl_g.getvalue())
                st.session_state["_grupos_upload_name"] = _chave_g
                cfg_g = copy.deepcopy(st.session_state.cfg_base)
                cfg_g.group_file = str(destino_g)
                st.session_state.cfg_base = cfg_g
                cfg_base = cfg_g
            st.success(T("File received: `{nome}`").format(
                nome=os.path.basename(upl_g.name)))

    # ---- Path / config fields ---------------------------------------------
    st.divider()
    col_d1, col_d2 = st.columns(2)
    for i, k in enumerate(_DADOS_KEYS):
        s = specs[k]
        with (col_d1 if i % 2 == 0 else col_d2):
            valores[k] = widget_para_campo(s, pq.attr_para_yaml(s, cfg_base))

    # ---- Data statistics preview -------------------------------------------
    st.divider()
    st.markdown(T("**Data preview**"))
    cfg_prev, _ = collect_config(cfg_base, valores)
    ok_dados, msg_dados = pq.validar_fonte_de_dados(cfg_prev)
    if _ws.modo_servidor():
        msg_dados = _ws.ocultar_caminhos(msg_dados)
    (st.success if ok_dados else st.warning)(
        T("Status: {msg_dados}").format(msg_dados=msg_dados))

    if ok_dados and st.button(T("🔍 Load spectra preview"), key="btn_prev_dados"):
        mode = cfg_prev.mode
        wn_mn = float(cfg_prev.wn_min)
        wn_mx = float(cfg_prev.wn_max)
        with st.spinner(T("Loading spectra sample...")):
            if mode == "dx":
                wn_p, X_p, labs_p = preview_spectra_dx(
                    cfg_prev.input_folder, wn_mn, wn_mx)
            elif mode == "csv":
                wn_p, X_p, labs_p = preview_spectra_csv(
                    cfg_prev.csv_file, cfg_prev.class_column, wn_mn, wn_mx)
            else:
                wn_p, X_p, labs_p = None, None, None

        n_excl_prev = 0
        if wn_p is not None and X_p is not None:
            # a previa mostra o eixo que o modelo ve: sem a agua/TSP da RMN, por exemplo (faixas_excluir)
            wn_p, X_p, n_excl_prev = remover_faixas_excluidas(wn_p, X_p, getattr(cfg_prev, "exclude_ranges", None))
            cls_u = np.unique(np.asarray(labs_p))
            # Guardado na sessao para o painel de status da aba Projeto poder
            # mostrar contagem REAL antes de existir execucao -- os mesmos
            # numeros exibidos logo abaixo, nao uma segunda contagem.
            st.session_state["previa_dados"] = {
                "n_espectros": int(len(X_p)),
                "n_classes": int(len(cls_u)),
            }
            # Espectros guardados (nao a figura): as telas Inicio e
            # Visualizacao redesenham com a paleta ATIVA, o que faz a
            # pre-visualizacao de cor ser a figura de verdade, nao uma
            # simulacao. Amostra pequena (previa), nao o dataset inteiro.
            st.session_state["previa_espectros"] = {
                "wn": np.asarray(wn_p), "X": np.asarray(X_p),
                "labels": np.asarray(labs_p),
            }
            st.markdown(T("**{n} spectra** · {k} classes: `{amostra}`{reticencias}").format(
                n=len(X_p), k=len(cls_u),
                amostra="`, `".join(cls_u[:8]),
                reticencias=" ..." if len(cls_u) > 8 else ""))
            if n_excl_prev:
                st.caption(T("{n} variables inside `faixas_excluir` are not shown (the model does not use them).").format(n=n_excl_prev))
            fig_p = plot_mean_spectra(wn_p, X_p, np.asarray(labs_p),
                                       titulo=T("Raw spectra (sample)"))
            st.pyplot(fig_p, use_container_width=True)
            plt.close(fig_p)
        elif mode == "sintetico":
            st.info(T("Synthetic mode has no file to preview: the spectra are generated when you run."))
        else:
            st.warning(T("Could not load spectra for preview. "
                       "Check the path/mode."))

    # ---- Save / Reload config.yaml ---------------------------------
    st.divider()
    cfg_dados, erros_dados = collect_config(cfg_base, valores)
    if erros_dados:
        st.warning(T("Fields with errors:\n- ") + "\n- ".join(erros_dados))
    if _ws.modo_servidor():
        # Sem config.yaml compartilhado no servidor: a configuração é da
        # sessão. O visitante pode BAIXAR o YAML para reutilizar localmente.
        if not erros_dados:
            import tempfile
            with tempfile.TemporaryDirectory() as _td:
                _yaml_tmp = os.path.join(_td, "config.yaml")
                pq.save_config(cfg_dados, _yaml_tmp)
                with open(_yaml_tmp, "rb") as _f:
                    _yaml_bytes = _f.read()
            st.download_button(
                T("⬇️ Download config.yaml"), data=_yaml_bytes,
                file_name="config.yaml", mime="text/yaml",
                use_container_width=True, key="btn_baixar_cfg_dados")
        return
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        if st.button(T("💾 Save config.yaml"), key="btn_salvar_cfg_dados",
                     use_container_width=True):
            if erros_dados:
                st.error(T("Fix the fields before saving."))
            else:
                pq.save_config(cfg_dados, cfg_path)
                st.session_state.cfg_base = cfg_dados
                st.success(T("Saved to {cfg_path}").format(cfg_path=cfg_path))
    with c_s2:
        if st.button(T("↺ Reload config.yaml"), key="btn_reload_cfg_dados",
                     use_container_width=True):
            try:
                cfg_recarregado = pq.load_config(cfg_path)
                st.session_state.cfg_base = cfg_recarregado
                # Os widgets desta tela JA foram criados nesta execucao, e o Streamlit recusa
                # alterar o valor de um widget depois de criado (levantava
                # StreamlitWidgetAlreadyInstantiatedError ao clicar em recarregar). Entao o
                # sincronismo fica para a proxima execucao, antes de os widgets nascerem.
                st.session_state["_cfg_recarregada"] = cfg_recarregado
                st.rerun()
            except (RuntimeError, FileNotFoundError, ValueError) as e:
                st.error(T("Error: {e}").format(e=_ws.msg_segura(e)))
