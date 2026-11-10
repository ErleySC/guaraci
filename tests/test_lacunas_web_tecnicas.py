"""Aba Tecnicas Avancadas do app web (Streamlit AppTest): bordas de ASCA, EPO/GLSW, MCR-ALS e fusao multibloco.

O motor de cada tecnica e' trocado por um duble (os motores tem testes proprios); aqui se confere o roteiro da tela.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import guaraci.mcr_als as mcr
import guaraci.tecnicas_avancadas as ta
import guaraci.web_seguranca as ws


def _texto(at):
    itens = [e.value for e in at.markdown] + [e.value for e in at.caption] + [e.value for e in at.info] + [e.value for e in at.warning]
    itens += [e.value for e in at.error] + [e.value for e in at.success]
    return "\n".join(str(i) for i in itens)


def _app():
    import types

    import numpy as np
    import pandas as pd
    import streamlit as st

    from guaraci.app_tabs import tecnicas
    from guaraci.config import Config

    classes = np.repeat(["A", "B", "C"], 6)
    rng = np.random.default_rng(0)
    X = np.abs(rng.normal(size=(18, 8))) + 1.0
    conc = rng.uniform(1, 5, 18) if st.session_state.get("com_conc", True) else None
    meta = pd.DataFrame({"sessao": np.tile(["s1", "s2"], 9)})

    def carregar(cfg):
        if st.session_state.get("falha_carga"):
            raise RuntimeError("pasta ilegivel")
        if st.session_state.get("bloco_sem_y") and getattr(cfg, "csv_file", "") == "b2":
            return np.arange(8.0), X, classes, None, None, meta
        return np.arange(8.0), X, classes, conc, np.array([f"g{i // 2}" for i in range(18)]), meta

    pq = types.SimpleNamespace(Config=Config, load_data=carregar, validate_input=lambda X_, wn, r, c, m: (X_, wn, r, c, m, {}))
    tecnicas.render(pq, Config(), lambda s: s)


def _rodar(**estado):
    at = AppTest.from_function(_app, default_timeout=60)
    for k, v in estado.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _clicar(at, chave):
    [b for b in at.button if b.key == chave][0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]


def test_carga_com_erro_mostra_a_mensagem_local_e_a_do_servidor_com_codigo(monkeypatch):
    monkeypatch.setenv("GUARACI_LOCAL_MODE", "1")
    at = _rodar(falha_carga=True)
    _clicar(at, "btn_tec_asca_load")
    assert "Error loading data" in _texto(at) and "pasta ilegivel" in _texto(at)
    monkeypatch.delenv("GUARACI_LOCAL_MODE", raising=False)
    at = _rodar(falha_carga=True)
    _clicar(at, "btn_tec_asca_load")
    assert "codigo do erro" in _texto(at) and "Error loading data" not in _texto(at)


def test_asca_exige_fator_e_mostra_erro_do_motor_e_p_valor(monkeypatch):
    at = _rodar()
    _clicar(at, "btn_tec_asca_load")
    [m for m in at.multiselect if m.key == "tec_asca_fatores"][0].set_value([])
    at.run()
    assert "Choose at least 1 factor" in _texto(at)
    [m for m in at.multiselect if m.key == "tec_asca_fatores"][0].set_value(["sessao"])
    at.run()
    monkeypatch.setattr(ta, "rodar_asca", lambda *a, **k: (_ for _ in ()).throw(ValueError("fator desbalanceado")))
    _clicar(at, "btn_tec_asca_run")
    assert "fator desbalanceado" in _texto(at)
    rel = types.SimpleNamespace(group_aware=True, permutacao={"sessao": {"p_value": 0.01}},
                                decomposicao={"efeitos": {"sessao": {"fracao_ss_total": 0.3}}, "ss_desbalanco": 0.1})
    monkeypatch.setattr(ta, "rodar_asca", lambda *a, **k: rel)
    _clicar(at, "btn_tec_asca_run")
    assert "SIGNIFICANT" in _texto(at)


def test_epo_glsw_recusa_fatores_iguais_e_mostra_erro_e_resultado(monkeypatch):
    at = _rodar()
    _clicar(at, "btn_tec_epo_load")
    nomes = list(ta.fatores_categoricos_disponiveis(np.repeat(["A", "B", "C"], 6), pd.DataFrame({"sessao": np.tile(["s1", "s2"], 9)})))
    for chave in ("tec_epo_f1", "tec_epo_f2"):
        [s for s in at.selectbox if s.key == chave][0].set_value(nomes[0])
    at.run()
    assert "must be different" in _texto(at)
    [s for s in at.selectbox if s.key == "tec_epo_f2"][0].set_value(nomes[-1])
    at.run()
    monkeypatch.setattr(ta, "rodar_epo_glsw", lambda *a, **k: (_ for _ in ()).throw(ValueError("poucos pares")))
    _clicar(at, "btn_tec_epo_run")
    assert "poucos pares" in _texto(at)
    monkeypatch.setattr(ta, "rodar_epo_glsw", lambda *a, **k: types.SimpleNamespace(metodo="EPO", n_pares_diferenca=3, variancia_removida_fracao=0.2))
    _clicar(at, "btn_tec_epo_run")
    assert "difference pairs" in _texto(at)


@pytest.mark.parametrize("com_conc", [True, False])
def test_mcr_als_com_e_sem_restricao_de_correlacao_e_com_erro(monkeypatch, com_conc):
    at = _rodar(com_conc=com_conc)
    _clicar(at, "btn_tec_mcr_load")
    simples = types.SimpleNamespace(n_iter=7, convergiu=True, lof_percent=1.2)

    class _Supervisionado:
        n_iter, convergiu, lof_percent, correlacao_calibracao = 5, False, 0.7, 0.93

    monkeypatch.setattr(mcr, "MCRALSResultadoSupervisionado", _Supervisionado)
    monkeypatch.setattr(ta, "rodar_mcr_als", lambda *a, **k: _Supervisionado() if k.get("conc") is not None else simples)
    if com_conc:
        [c for c in at.checkbox if c.key == "tec_mcr_restricao"][0].set_value(True)
        at.run()
        [n for n in at.number_input if n.key == "tec_mcr_alvo"][0].set_value(1)
        at.run()
    _clicar(at, "btn_tec_mcr_run")
    assert ("0.930" in _texto(at)) is com_conc
    monkeypatch.setattr(ta, "rodar_mcr_als", lambda *a, **k: (_ for _ in ()).throw(ValueError("nao negatividade impossivel")))
    _clicar(at, "btn_tec_mcr_run")
    assert "nao negatividade impossivel" in _texto(at)


def test_fusao_no_modo_local_pede_os_dois_blocos_e_cobre_cada_recusa(monkeypatch):
    monkeypatch.setenv("GUARACI_LOCAL_MODE", "1")
    at = _rodar()
    _clicar(at, "btn_tec_fusao_run")
    assert "Fill in both folders/files" in _texto(at)
    for i, caminho in ((1, "b1"), (2, "b2")):
        [t for t in at.text_input if t.key == f"tec_fusao_caminho_{i}"][0].set_value(caminho)
    at.run()
    _clicar(at, "btn_tec_fusao_run")
    assert "Neither block" not in _texto(at)
    monkeypatch.setattr(ta, "rodar_fusao_multibloco", lambda blocos, y: (_ for _ in ()).throw(ValueError("linhas diferentes")))
    _clicar(at, "btn_tec_fusao_run")
    assert "linhas diferentes" in _texto(at)
    monkeypatch.setattr(ta, "rodar_fusao_multibloco",
                        lambda blocos, y: (types.SimpleNamespace(rmsep=0.5, r2cal=0.9, r2val=0.8, n_lv=3, n_cal=12, n_val=6), ["a", "b"]))
    _clicar(at, "btn_tec_fusao_run")
    assert "RMSEP" in _texto(at)


def test_fusao_sem_teor_em_nenhum_bloco_e_com_erro_de_leitura(monkeypatch):
    monkeypatch.setenv("GUARACI_LOCAL_MODE", "1")
    at = _rodar(com_conc=False)
    for i, caminho in ((1, "b1"), (2, "b2")):
        [t for t in at.text_input if t.key == f"tec_fusao_caminho_{i}"][0].set_value(caminho)
    at.run()
    _clicar(at, "btn_tec_fusao_run")
    assert "Neither block has content" in _texto(at)
    at = _rodar(falha_carga=True)
    for i, caminho in ((1, "b1"), (2, "b2")):
        [t for t in at.text_input if t.key == f"tec_fusao_caminho_{i}"][0].set_value(caminho)
    at.run()
    _clicar(at, "btn_tec_fusao_run")
    assert "Error loading" in _texto(at)


def test_fusao_no_modo_servidor_recebe_csv_enviado_e_mostra_so_o_codigo_do_erro_de_leitura(monkeypatch):
    monkeypatch.delenv("GUARACI_LOCAL_MODE", raising=False)
    monkeypatch.setattr(ws, "salvar_upload", lambda sessao, nome, dados: f"/sessao/{nome}")
    at = AppTest.from_function(_app, default_timeout=60)
    at.session_state["_sessao_id"] = "sid"
    at.session_state["falha_carga"] = True
    at.run()
    for i in (1, 2):
        [u for u in at.file_uploader if u.key == f"tec_fusao_upload_{i}"][0].upload(f"b{i}.csv", b"a,b\n1,2\n", "text/csv")
    at.run()
    _clicar(at, "btn_tec_fusao_run")
    assert "codigo do erro" in _texto(at) and not at.exception
