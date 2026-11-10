"""Tela de predicao em lote e painel de hardware de `guaraci.guaraci`.

A predicao usa dubles para o motor (o motor tem os seus testes); aqui se confere o roteiro da tela: o que pergunta, o que recusa e o que
resume na tela para cada formato de resultado (simples, com dominio de aplicabilidade, conjunto conforme, fluxo cego).
"""
from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

import guaraci.guaraci as g
import guaraci.predicao as pred


def _arquivos(tmp_path):
    modelo = tmp_path / "m.joblib"
    modelo.write_bytes(b"x")
    csv = tmp_path / "novos.csv"
    csv.write_text("1;2\n3;4\n", encoding="utf-8")
    return str(modelo), str(csv)


def _motor(monkeypatch, df_res, *, pacote=None, cego=None, meta=None):
    monkeypatch.setattr(g, "_carregar_modelo_cli", lambda cam, pt: pacote or {})
    monkeypatch.setattr(pred, "validate_model_package", lambda pkg: None)
    monkeypatch.setattr(pred, "load_prediction_csv", lambda cam: (None, None, meta if meta is not None else pd.DataFrame()))
    monkeypatch.setattr(pred, "predict_samples", lambda pkg, X, wn: df_res.copy())
    monkeypatch.setattr(pred, "predict_blind", lambda pkg, X, wn: (df_res.copy(), cego))
    monkeypatch.setattr(pred, "anexar_colunas_fluxo_cego", lambda df, res: None)
    monkeypatch.setattr(pred, "anexar_metadados", lambda meta_df, df: df.assign(meta_extra=1))


def _df(**colunas):
    base = {"classe_pred": ["a", "a", "b"], "aceito": [True, False, True]}
    base.update(colunas)
    return pd.DataFrame(base)


def test_predicao_recusa_modelo_ausente_voltar_e_confirmacao_negada(cli, tmp_path):
    modelo, csv = _arquivos(tmp_path)
    cli.dizer("0")
    g._menu_prediction(g.Config())
    cli.dizer(str(tmp_path / "nao_existe.joblib"), "")
    g._menu_prediction(g.Config())
    assert "Arquivo nao encontrado" in cli.saida()
    cli.dizer(modelo, "n", "")
    g._menu_prediction(g.Config())
    assert g._t("cancelado") in cli.saida()


def test_predicao_alterna_idioma_antes_do_modelo_e_recusa_csv_vazio_ou_ausente(cli, tmp_path):
    modelo, csv = _arquivos(tmp_path)
    cli.dizer("i", "0")                       # prompt de valor: "i" e' texto (nao troca o idioma); "0" cancela
    g._menu_prediction(g.Config())
    g._STATE["lang"] = "EN"
    cli.dizer(modelo, "y", str(tmp_path / "nao.csv"), "")
    g._menu_prediction(g.Config())
    assert "File not found" in cli.saida()


def test_predicao_so_sobrescreve_a_saida_se_confirmado(cli, tmp_path, monkeypatch):
    modelo, csv = _arquivos(tmp_path)
    saida = tmp_path / "saida.csv"
    saida.write_text("antigo", encoding="utf-8")
    _motor(monkeypatch, _df())
    cli.dizer(modelo, "s", csv, str(saida), "n", "")
    g._menu_prediction(g.Config())
    assert saida.read_text(encoding="utf-8") == "antigo"
    cli.dizer(modelo, "s", csv, str(saida), "s", "")
    g._menu_prediction(g.Config())
    assert "classe_pred" in saida.read_text(encoding="utf-8")


def test_predicao_com_erro_do_motor_mostra_a_mensagem(cli, tmp_path, monkeypatch):
    modelo, csv = _arquivos(tmp_path)
    _motor(monkeypatch, _df())
    monkeypatch.setattr(pred, "validate_model_package", lambda pkg: (_ for _ in ()).throw(ValueError("pacote invalido")))
    cli.dizer(modelo, "s", csv, "", "")
    g._menu_prediction(g.Config())
    assert "pacote invalido" in cli.saida()


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_predicao_resume_dominio_de_aplicabilidade_e_sentinela(cli, tmp_path, monkeypatch, lang):
    import guaraci.sentinela_deriva as sent
    monkeypatch.setitem(g._STATE, "lang", lang)
    modelo, csv = _arquivos(tmp_path)
    _motor(monkeypatch, _df(AD_dentro_dominio=[True, True, False]), meta=pd.DataFrame({"id": [1, 2, 3]}))
    alerta = types.SimpleNamespace(alerta=True, n=3, mensagem="deriva detectada")
    for retorno in (alerta, types.SimpleNamespace(alerta=False, n=3, mensagem="estavel"), None):
        monkeypatch.setattr(sent, "hook_apos_predicao", lambda cam, df, r=retorno: r)
        cli.dizer(modelo, "s", csv, str(tmp_path / f"s_{id(retorno)}.csv"), "")
        g._menu_prediction(g.Config())
    monkeypatch.setattr(sent, "hook_apos_predicao", lambda cam, df: (_ for _ in ()).throw(RuntimeError("sentinela quebrou")))
    cli.dizer(modelo, "s", csv, str(tmp_path / "s_erro.csv"), "")
    g._menu_prediction(g.Config())
    saida = cli.saida()
    assert "deriva detectada" in saida and "estavel" in saida and "sentinela quebrou" in saida


@pytest.mark.parametrize("com_cobertura", [True, False])
def test_predicao_resume_o_conjunto_conforme_validado_ou_nao(cli, tmp_path, monkeypatch, com_cobertura):
    modelo, csv = _arquivos(tmp_path)
    colunas = {"classes_plausiveis": ["a|b", "a", ""]}
    if com_cobertura:
        colunas["conjunto_cobertura_nominal"] = [0.95, 0.95, 0.95]
    _motor(monkeypatch, _df(**colunas))
    cli.dizer(modelo, "s", csv, "", "")
    g._menu_prediction(g.Config())
    assert ("plausiveis (conforme" in cli.saida()) is com_cobertura and ("NAO_VALIDADO" in cli.saida()) is (not com_cobertura)


def _resultado(aceito, classe, teor, faixa):
    return types.SimpleNamespace(pureza=types.SimpleNamespace(aceito=aceito),
                                 identificacao=types.SimpleNamespace(classe_identificada=classe),
                                 quantificacao=types.SimpleNamespace(teor_estimado=teor, faixa_decisao=faixa))


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_predicao_resume_o_fluxo_cego_completo(cli, tmp_path, monkeypatch, lang):
    monkeypatch.setitem(g._STATE, "lang", lang)
    modelo, csv = _arquivos(tmp_path)
    cego = [_resultado(True, "x", 5.0, "nao_detectavel"), _resultado(False, "y", 9.0, "zona_cinzenta"),
            _resultado(None, None, None, None), _resultado(False, "z", 7.0, "quantificado_com_confianca"),
            _resultado(True, "w", 3.0, "sem_faixa")]
    df = pd.DataFrame({"classe_pred": list("aabbb"), "classe_identificada": list("xyqzw")})
    _motor(monkeypatch, df, pacote={"identification_ensemble": object()}, cego=cego)
    cli.dizer(modelo, "s", csv, "", "")
    g._menu_prediction(g.Config())
    saida = cli.saida()
    assert ("Fluxo cego" if lang == "PT" else "Blind flow") in saida


def test_predicao_com_ensemble_mas_sem_coluna_de_identificacao_nao_resume_o_fluxo_cego(cli, tmp_path, monkeypatch):
    modelo, csv = _arquivos(tmp_path)
    _motor(monkeypatch, _df(), pacote={"identification_ensemble": object()}, cego=[])
    cli.dizer(modelo, "s", csv, "", "")
    g._menu_prediction(g.Config())
    assert "Fluxo cego" not in cli.saida()


class _Mem:
    def __init__(self, total, livre, pct):
        self.total, self.available, self.percent = total * 1024**3, livre * 1024**3, pct


@pytest.mark.parametrize("total_gb, rotulo", [(32, "Alto Desempenho"), (12, "Desempenho Medio"), (6, "Desempenho Basico"), (2, "Limitado")])
@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_hardware_classifica_a_maquina_por_faixa_de_memoria(cli, monkeypatch, total_gb, rotulo, lang):
    import psutil
    monkeypatch.setitem(g._STATE, "lang", lang)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _Mem(total_gb, total_gb / 2, 50))
    cli.dizer("0")
    g._menu_hardware(g.Config())
    if lang == "PT":
        assert rotulo in cli.saida()
    cli.dizer("0")
    g._menu_hardware(None)


def test_hardware_sem_psutil_avisa_que_nao_esta_instalado(cli, monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    cli.dizer("0")
    g._menu_hardware(g.Config())
    assert "Nao instalado" in cli.saida()
