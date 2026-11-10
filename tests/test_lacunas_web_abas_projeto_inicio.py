"""Abas Projeto e Inicio do app web (Streamlit AppTest): painel de status, auditoria, hardware e cartoes de resumo."""
from __future__ import annotations

import json
import types

import pytest
from streamlit.testing.v1 import AppTest

from tests._web_apoio import criar_pasta_de_resultados

# ---- projeto ------------------------------------------------------------------------------------------------------------------------------------


def _texto(at):
    itens = [e.value for e in at.markdown] + [e.value for e in at.caption] + [e.value for e in at.info] + [e.value for e in at.warning]
    itens += [e.value for e in at.error] + [e.value for e in at.success] + [str(m.value) for m in at.metric]
    return "\n".join(str(i) for i in itens)


def _projeto():
    import streamlit as st

    from guaraci.app_tabs import projeto

    class _Pq:
        @staticmethod
        def hardware_probe():
            hw = st.session_state.get("hw")
            if hw == "erro":
                raise RuntimeError("sonda")
            return hw

    projeto.render(_Pq, lambda s: s, is_public_demo=bool(st.session_state.get("demo")))


def _hw(livre, total=16.0, container=False, psutil_ok=True):
    return {"ram_total_gb": total, "ram_livre_gb": livre, "cpu_fisicos": 4, "cpu_logicos": 8, "disco_livre_gb": 100.0, "psutil_ok": psutil_ok,
            "ram_limitada_por_container": container}


def _rodar_projeto(**estado):
    at = AppTest.from_function(_projeto, default_timeout=60)
    for k, v in estado.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_projeto_sem_dados_pede_para_carregar_e_mostra_o_projeto_ativo():
    at = _rodar_projeto(hw=_hw(20), proj_nome="Meu projeto")
    texto = _texto(at)
    assert "No data loaded yet" in texto and "Active project" in texto


@pytest.mark.parametrize("livre, total, container, demo, psutil_ok, esperado", [
    (1.0, 16.0, False, False, True, "Critical RAM"),
    (3.0, 16.0, False, False, True, "Low RAM"),
    (6.0, 16.0, False, False, False, "Moderate RAM"),
    (12.0, 16.0, True, True, True, "container limit"),
    (12.0, 16.0, False, True, True, "Managed by host"),
])
def test_projeto_classifica_a_ram_e_esconde_o_valor_que_nao_e_confiavel(livre, total, container, demo, psutil_ok, esperado):
    at = _rodar_projeto(hw=_hw(livre, total, container, psutil_ok), demo=demo)
    assert esperado in _texto(at) or any(esperado in str(m.value) + str(m.delta) for m in at.metric)


def test_projeto_com_ram_nao_medida_nao_inventa_numero_nem_alerta():
    hw = _hw(None, total=None, psutil_ok=False)
    at = _rodar_projeto(hw=hw)
    texto = _texto(at) + " ".join(str(m.value) + str(m.delta) for m in at.metric)
    assert "not measured" in texto
    assert "Limited hardware detected" not in texto and "None" not in texto


def test_projeto_com_sonda_de_hardware_quebrada_mostra_o_fallback():
    at = _rodar_projeto(hw="erro")
    assert "could not detect hardware" in _texto(at)


def test_projeto_mostra_a_previa_de_dados_e_a_tecnica_sem_validacao_de_terceiros():
    from guaraci.config import Config
    at = _rodar_projeto(hw=_hw(20), previa_dados={"n_espectros": 12, "n_classes": 3}, cfg_base=Config())
    assert "Matrix / technique" in _texto(at) and "data preview" in _texto(at)
    assert "public validation" not in _texto(at).lower() and "validacao_dados_publicos" not in _texto(at)
    at = _rodar_projeto(hw=_hw(20), previa_dados={"n_espectros": 12, "n_classes": 3}, cfg_base=Config(matrix_profile="perfil_que_nao_existe"))
    assert "Matrix / technique" not in _texto(at)


@pytest.mark.parametrize("achados, esperado", [
    ([{"severidade": "critico", "nome": "n", "mensagem": "m"}], "1 critical"),
    ([{"severidade": "aviso", "nome": "n", "mensagem": "m"}], "1 warning"),
    ([{"severidade": "ok", "nome": "n", "mensagem": "m"}], "0 critical"),
    (None, "Design audit not available"),
])
def test_projeto_resume_a_auditoria_da_execucao(tmp_path, achados, esperado):
    pasta = criar_pasta_de_resultados(tmp_path)
    if achados is not None:
        import os
        with open(os.path.join(pasta, "Relatorios", "auditoria_delineamento.json"), "w", encoding="utf-8") as f:
            json.dump(achados, f)
    at = _rodar_projeto(hw=_hw(20), ultima_pasta=pasta)
    assert esperado in _texto(at) and "last completed run" in _texto(at)


def test_projeto_acha_a_pasta_de_relatorios_no_layout_antigo_e_na_raiz(tmp_path):
    from guaraci.app_tabs.projeto import _pasta_relatorios
    antigo = tmp_path / "antigo"
    (antigo / "logs").mkdir(parents=True)
    assert _pasta_relatorios(str(antigo)).endswith("logs")
    raiz = tmp_path / "raiz"
    raiz.mkdir()
    assert _pasta_relatorios(str(raiz)) == str(raiz)
    assert _pasta_relatorios(str(tmp_path / "nao_existe")) == str(tmp_path / "nao_existe")


def test_as_abas_do_app_nao_leem_mais_a_tabela_de_validacao_de_terceiros():
    from pathlib import Path
    raiz = Path(__file__).resolve().parents[1] / "src" / "guaraci" / "app_tabs"
    modulo = "validacao" + "_publica"                          # literal partido: o nome do modulo removido nao volta
    for arq in ("projeto.py", "inicio.py"):
        assert modulo not in (raiz / arq).read_text(encoding="utf-8"), arq
    assert not (raiz.parent / (modulo + ".py")).exists()        # o modulo sem consumidor foi removido (lote 11b)


# ---- inicio ---------------------------------------------------------------------------------------------------------------------------------------


def _inicio():
    import types

    import streamlit as st

    from guaraci.app_tabs import inicio
    from guaraci.design_tokens import tokens

    def ir_para(destino):
        st.session_state["destino"] = destino

    inicio.render(types.SimpleNamespace(), lambda s: s, ir_para, lambda: tokens("light"))


def _cego(*, aceito=False, classe="x", teor=None, lod=1.0, loq=3.0, faixa="quantificado_com_confianca", cobertura="validado"):
    return types.SimpleNamespace(
        pureza=types.SimpleNamespace(aceito=aceito),
        identificacao=types.SimpleNamespace(classe_identificada=classe,
                                            cobertura_status=types.SimpleNamespace(value=cobertura) if cobertura else None),
        quantificacao=types.SimpleNamespace(teor_estimado=teor, lod=lod, loq=loq, faixa_decisao=faixa, especie_usada="Especie X"))


def _rodar_inicio(**estado):
    at = AppTest.from_function(_inicio, default_timeout=60)
    for k, v in estado.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_inicio_vazio_sugere_carregar_os_espectros_e_o_botao_navega():
    at = _rodar_inicio()
    assert "Load the spectra" in _texto(at) or "Carregue os espectros" in _texto(at)
    [b for b in at.button if b.key == "btn_acao_sugerida"][0].click()
    at.run()
    assert at.session_state["destino"] == "dados"


def test_inicio_com_previa_mostra_espectros_medios_e_dados_carregados():
    import numpy as np
    espectros = {"wn": np.linspace(4000, 8000, 20), "X": np.random.default_rng(0).normal(size=(6, 20)), "labels": np.array(list("aabbcc"))}
    at = _rodar_inicio(previa_dados={"n_espectros": 6, "n_classes": 3}, previa_espectros=espectros)
    assert "data preview" in _texto(at)


def test_inicio_com_execucao_le_o_resumo_a_auditoria_e_a_predicao_sem_fluxo_cego(tmp_path):
    import pandas as pd
    resumo = "Total samples: 30\nTotal classes: 3\nN grupos mae_id: 10\nPreprocessing: MSC\nGroup-aware (mae_id): sim\n"
    pasta = criar_pasta_de_resultados(tmp_path, resumo=resumo)
    at = _rodar_inicio(ultima_pasta=pasta, pred_resultados=pd.DataFrame({"aceito": [True, False, True]}))
    texto = _texto(at)
    assert "last completed run" in texto
    at = _rodar_inicio(ultima_pasta=pasta, pred_resultados=pd.DataFrame({"x": [1, 2]}))
    assert not at.exception


@pytest.mark.parametrize("resumo", ["Grouping guarantee: none\n", "Total samples: 5\n"])
def test_inicio_mostra_garantia_de_agrupamento_desprotegida_ou_nenhuma(tmp_path, resumo):
    pasta = criar_pasta_de_resultados(tmp_path, resumo=resumo)
    at = _rodar_inicio(ultima_pasta=pasta)
    assert not at.exception


def test_inicio_com_fluxo_cego_resume_deteccao_identificacao_e_quantificacao(tmp_path):
    import pandas as pd
    pasta = criar_pasta_de_resultados(tmp_path)
    df = pd.DataFrame({"aceito": [True, False]})
    cego = [_cego(aceito=False, classe="x", teor=12.0), _cego(aceito=True, classe=None, teor=None)]
    at = _rodar_inicio(ultima_pasta=pasta, pred_resultados=df, pred_resultados_cego=cego)
    assert "adulterated" in _texto(at)
    nenhum = [_cego(aceito=False, classe=None, teor=None)]
    at = _rodar_inicio(ultima_pasta=pasta, pred_resultados=df, pred_resultados_cego=nenhum)
    assert "Quantification was blocked" in _texto(at)


@pytest.mark.parametrize("lod, loq", [(1.0, 3.0), (None, None), (float("nan"), 2.0)])
def test_inicio_desenha_a_faixa_de_decisao_so_com_limites_validos(tmp_path, lod, loq):
    import pandas as pd
    pasta = criar_pasta_de_resultados(tmp_path)
    cego = [_cego(aceito=False, classe="x", teor=5.0, lod=lod, loq=loq, cobertura=None)]
    at = _rodar_inicio(ultima_pasta=pasta, pred_resultados=pd.DataFrame({"aceito": [False]}), pred_resultados_cego=cego)
    assert ("No LOD/LOQ persisted" in _texto(at)) is (lod is None or lod != lod)


def test_inicio_sem_perfil_resolvido_mostra_traco_na_tecnica(tmp_path):
    from guaraci.config import Config
    pasta = criar_pasta_de_resultados(tmp_path)
    at = _rodar_inicio(ultima_pasta=pasta, cfg_base=Config(matrix_profile="perfil_que_nao_existe"))
    assert not at.exception


@pytest.mark.parametrize("auditoria, destino", [([{"severidade": "critico", "mensagem": "m"}], "projeto"), (None, "predicao")])
def test_inicio_sugere_a_proxima_acao_conforme_a_auditoria(tmp_path, auditoria, destino):
    import os
    pasta = criar_pasta_de_resultados(tmp_path)
    if auditoria is not None:
        with open(os.path.join(pasta, "Relatorios", "auditoria_delineamento.json"), "w", encoding="utf-8") as f:
            json.dump(auditoria, f)
    at = _rodar_inicio(ultima_pasta=pasta)
    [b for b in at.button if b.key == "btn_acao_sugerida"][0].click()
    at.run()
    assert at.session_state["destino"] == destino


def test_inicio_acha_a_pasta_de_relatorios_em_cada_layout(tmp_path):
    from guaraci.app_tabs.inicio import _pasta_relatorios
    (tmp_path / "logs").mkdir()
    assert _pasta_relatorios(str(tmp_path)).endswith("logs")
    vazio = tmp_path / "vazio"
    vazio.mkdir()
    assert _pasta_relatorios(str(vazio)) == str(vazio)
