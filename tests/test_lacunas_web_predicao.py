"""Aba Predicao do app web (Streamlit AppTest): carga do modelo (portatil ou .joblib), do CSV, erros e todos os formatos de resultado."""
from __future__ import annotations

import json
import types

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import guaraci.app_tabs.predicao as aba


def _texto(at):
    itens = [e.value for e in at.markdown] + [e.value for e in at.caption] + [e.value for e in at.info] + [e.value for e in at.warning]
    itens += [e.value for e in at.error] + [e.value for e in at.success]
    itens += [f"{m.label} {m.value}" for m in at.metric]
    return "\n".join(str(i) for i in itens)


def _tela():
    import streamlit as st

    from guaraci.app_tabs import predicao
    from guaraci.design_tokens import tokens
    predicao.render(bool(st.session_state.get("bloqueado", True)), lambda: tokens("light"), lambda s: s)


def _rodar(**estado):
    at = AppTest.from_function(_tela, default_timeout=60)
    for k, v in estado.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _predizer(at):
    [b for b in at.button if b.key == "btn_predizer"][0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _csv_dummy(monkeypatch, n=3, falha=None):
    meta = pd.DataFrame({"amostra": [f"s{i}" for i in range(n)]})

    def carregar(fonte):
        if falha:
            raise ValueError(falha)
        return np.ones((n, 4)), np.arange(4.0), meta

    monkeypatch.setattr(aba, "_carregar_csv_predicao", carregar)


def _enviar_csv(at):
    [u for u in at.file_uploader if u.key == "pred_csv_upload"][0].upload("novos.csv", b"1;2\n3;4\n", "text/csv")
    at.run()


def _enviar_modelo(at, conteudo):
    [u for u in at.file_uploader if u.key == "pred_model_upload_json"][0].upload("m.json", conteudo, "application/json")
    at.run()


_PORTATIL = {"formato": "guaraci-modelo-portatil", "classes": ["a", "b"], "wn_min": 4000, "wn_max": 8000, "dominio_aplicabilidade": True}


# ---- modo servidor (modelo portatil em JSON) -------------------------------------------------------------------------------------------------------------


def test_servidor_sem_modelo_e_sem_csv_lista_os_dois_erros(monkeypatch):
    at = _predizer(_rodar())
    assert "No valid model provided (upload)" in _texto(at) and "No spectra CSV provided" in _texto(at)


@pytest.mark.parametrize("conteudo, esperado", [(b"\xff\xfe", "not a valid portable model"), (json.dumps({"formato": "outro"}).encode(), "not a GUARACI portable model"),
                                                (json.dumps([1, 2]).encode(), "not a GUARACI portable model")])
def test_servidor_recusa_modelo_que_nao_e_json_portatil(monkeypatch, conteudo, esperado):
    _csv_dummy(monkeypatch)
    at = _rodar()
    _enviar_modelo(at, conteudo)
    _enviar_csv(at)
    _predizer(at)
    assert esperado in _texto(at) or "codigo do erro" in _texto(at)


@pytest.mark.parametrize("ad, fora", [(True, 0), (True, 2), (False, 0)])
def test_servidor_aplica_o_modelo_portatil_e_avisa_sobre_o_dominio(monkeypatch, ad, fora):
    _csv_dummy(monkeypatch)
    resultado = {"classe": ["a", "b", "a"]}
    if ad:
        resultado["AD_dentro_dominio"] = [True] * (3 - fora) + [False] * fora
    monkeypatch.setattr(aba._me, "predict_portable", lambda modelo, X, wn: resultado)
    portatil = dict(_PORTATIL, dominio_aplicabilidade=ad)
    at = _rodar()
    _enviar_modelo(at, json.dumps(portatil).encode())
    _enviar_csv(at)
    _predizer(at)
    texto = _texto(at)
    assert "Prediction complete: 3 samples" in texto and "Model loaded" in texto
    assert ("outside the model's applicability domain" in texto) is (fora > 0) and ("no applicability-domain check" in texto) is (not ad)


def test_servidor_com_erro_ao_aplicar_o_modelo_mostra_so_o_codigo_do_erro(monkeypatch):
    _csv_dummy(monkeypatch)
    monkeypatch.setattr(aba._me, "predict_portable", lambda m, X, wn: (_ for _ in ()).throw(RuntimeError("canais incompativeis")))
    at = _rodar()
    _enviar_modelo(at, json.dumps(_PORTATIL).encode())
    _enviar_csv(at)
    _predizer(at)
    assert "codigo do erro" in _texto(at)


def test_servidor_com_csv_ilegivel_avisa_que_nao_leu(monkeypatch):
    _csv_dummy(monkeypatch, falha="colunas misturadas")
    at = _rodar()
    _enviar_modelo(at, json.dumps(_PORTATIL).encode())
    _enviar_csv(at)
    _predizer(at)
    assert "Error reading CSV" in _texto(at)


# ---- modo local (.joblib por caminho) ---------------------------------------------------------------------------------------------------------------


def _local(tmp_path, monkeypatch, *, confia=True, modelo=True, csv=True, pacote=None):
    _csv_dummy(monkeypatch)
    at = _rodar(bloqueado=False)
    if modelo:
        arq = tmp_path / "m.joblib"
        arq.write_bytes(b"x")
        [t for t in at.text_input if t.key == "pred_model_path"][0].set_value(str(arq))
    if csv:
        arq_csv = tmp_path / "n.csv"
        arq_csv.write_text("1;2\n", encoding="utf-8")
        [t for t in at.text_input if t.key == "pred_csv_path"][0].set_value(str(arq_csv))
    [c for c in at.checkbox if c.key == "pred_model_confia"][0].set_value(confia)
    at.run()
    import guaraci.predicao as pr
    monkeypatch.setattr(pr, "load_model", lambda caminho, confiar: pacote if pacote is not None else {})
    monkeypatch.setattr(aba, "_validar_pacote_modelo", lambda p: None)
    return at


def test_local_exige_confiar_na_origem_do_modelo(tmp_path, monkeypatch):
    at = _predizer(_local(tmp_path, monkeypatch, confia=False))
    assert "I trust the source" in _texto(at) or "trust" in _texto(at).lower()


def test_local_sem_arquivo_de_modelo_ou_com_modelo_nao_assinado_avisa(tmp_path, monkeypatch):
    at = _predizer(_local(tmp_path, monkeypatch, modelo=False))
    assert "No valid model provided (upload or path)" in _texto(at)
    import guaraci.predicao as pr
    at = _local(tmp_path, monkeypatch)
    monkeypatch.setattr(pr, "load_model", lambda caminho, confiar: (_ for _ in ()).throw(pr.ModeloNaoAssinadoError("sem assinatura")))
    _predizer(at)
    assert "sem assinatura" in _texto(at)
    monkeypatch.setattr(pr, "load_model", lambda caminho, confiar: (_ for _ in ()).throw(OSError("arquivo corrompido")))
    _predizer(at)
    assert "Error loading model" in _texto(at)


def test_local_sem_csv_ou_com_erro_de_predicao(tmp_path, monkeypatch):
    at = _predizer(_local(tmp_path, monkeypatch, csv=False))
    assert "No spectra CSV provided" in _texto(at)
    at = _local(tmp_path, monkeypatch)
    monkeypatch.setattr(aba, "_predizer", lambda pkg, X, wn: (_ for _ in ()).throw(RuntimeError("canais incompativeis")))
    _predizer(at)
    assert "Prediction error" in _texto(at)


def test_local_aplica_o_modelo_simples_e_o_de_fluxo_cego(tmp_path, monkeypatch):
    at = _local(tmp_path, monkeypatch)
    monkeypatch.setattr(aba, "_predizer", lambda pkg, X, wn: pd.DataFrame({"classe_pred": ["a", "b", "a"], "aceito": [True, True, False]}))
    _predizer(at)
    assert "Prediction complete" in _texto(at) and "Within PLS-DA model fit" in _texto(at)
    cego = [_resultado_cego(aceito=False, classe="x", teor=4.0)] * 3
    pacote = {"identification_ensemble": object()}
    at = _local(tmp_path, monkeypatch, pacote=pacote)
    monkeypatch.setattr(aba, "_predizer_cego", lambda pkg, X, wn: (pd.DataFrame({"classe_pred": list("aaa"), "classe_identificada": list("xxx")}), cego))
    monkeypatch.setattr(aba, "_anexar_colunas_cego", lambda df, res: None)
    _predizer(at)
    assert "Blind flow" in _texto(at)


# ---- exibicao dos resultados ----------------------------------------------------------------------------------------------------------------


def _resultado_cego(*, aceito=False, classe="x", teor=None, faixa=None):
    return types.SimpleNamespace(pureza=types.SimpleNamespace(aceito=aceito),
                                 identificacao=types.SimpleNamespace(classe_identificada=classe),
                                 quantificacao=types.SimpleNamespace(teor_estimado=teor, faixa_decisao=faixa))


def test_resultados_com_dominio_conformal_e_sentinela(monkeypatch):
    import guaraci.sentinela_deriva as sent
    df = pd.DataFrame({"aceito": [True, False], "AD_dentro_dominio": [True, False], "classes_plausiveis": ["a|b", ""],
                       "conjunto_cobertura_nominal": [0.95, 0.95]})
    for retorno in (types.SimpleNamespace(alerta=True, n=4, mensagem="deriva"), types.SimpleNamespace(alerta=False, n=4, mensagem="estavel"), None):
        monkeypatch.setattr(sent, "hook_apos_predicao", lambda cam, d, r=retorno: r)
        at = _rodar(pred_resultados=df, pred_modelo_caminho="modelo.joblib")
        texto = _texto(at)
        assert "Within applicability domain" in texto and "Plausible class set (conformal, 95% coverage)" in texto
    monkeypatch.setattr(sent, "hook_apos_predicao", lambda cam, d: (_ for _ in ()).throw(RuntimeError("sentinela quebrou")))
    assert "Drift sentinel unavailable" in _texto(_rodar(pred_resultados=df, pred_modelo_caminho="modelo.joblib"))


def test_resultados_com_conjunto_conforme_nao_validado():
    df = pd.DataFrame({"classes_plausiveis": ["a", "b"]})
    assert "NOT VALIDATED" in _texto(_rodar(pred_resultados=df))


def test_resultados_sem_colunas_especiais_so_mostram_a_tabela():
    df = pd.DataFrame({"x": [1, 2]})
    at = _rodar(pred_resultados=df)
    assert not at.metric


@pytest.mark.parametrize("faixas, esperado", [
    ([None, None], "No LOD/LOQ available"),
    (["nao_detectavel", "zona_cinzenta", "quantificado_com_confianca", None], "no decision range"),
])
def test_resultados_do_fluxo_cego_mostram_a_faixa_de_decisao_so_quando_ha_limites(faixas, esperado):
    cego = [_resultado_cego(aceito=bool(i % 2), classe="x" if i % 2 else None, teor=1.0 + i, faixa=f) for i, f in enumerate(faixas)]
    df = pd.DataFrame({"classe_identificada": ["x"] * len(cego), "faixa_decisao": faixas})
    at = _rodar(pred_resultados=df, pred_resultados_cego=cego)
    texto = _texto(at)
    assert esperado in texto
    assert len(at.get("progress")) == (3 if len(faixas) == 4 else 0)
