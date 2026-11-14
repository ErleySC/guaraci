# -*- coding: utf-8 -*-
"""A opcao `saturacao` percorrida pelas portas reais do usuario: menu interativo, web e config.yaml."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import guaraci.guaraci as g
import guaraci.preferencias_visuais as prefs
from guaraci.config_io import load_config, save_config
from guaraci.contexto import campos_visiveis

_RAIZ = Path(__file__).resolve().parents[1]


def test_menu_dados_oferece_saturacao_so_nos_modos_de_absorbancia():
    campos = ["saturacao", "faixas_excluir"]
    for modo in ("dx", "csv", "opus", "spc", "sp"):
        assert "saturacao" in campos_visiveis(g.Config(mode=modo), campos)
    for modo in ("hsi", "rmn", "hplc", "gcms", "eem", "multielementar", "imagem"):
        assert "saturacao" not in campos_visiveis(g.Config(mode=modo), campos)


def test_menu_edita_saturacao_pelo_numero_da_escolha(cli):
    cfg = g.Config(mode="csv")
    assert cfg.saturacao == "auto"
    cli.dizer("2", "s")                       # 2 = avisar; campo ANALITICO pede confirmacao
    assert g._editar_campo(cfg, "saturacao") is True
    assert cfg.saturacao == "avisar"
    cli.dizer("3", "s")
    assert g._editar_campo(cfg, "saturacao") is True and cfg.saturacao == "nenhuma"


def test_menu_recusa_valor_fora_da_lista(cli):
    cfg = g.Config(mode="csv")
    cli.dizer("9")
    assert g._editar_campo(cfg, "saturacao") is False and cfg.saturacao == "auto"


def test_config_yaml_grava_e_le_saturacao(tmp_path):
    cfg = g.Config(mode="csv", saturacao="avisar")
    caminho = str(tmp_path / "c.yaml")
    save_config(cfg, caminho)
    assert "saturacao: avisar" in Path(caminho).read_text(encoding="utf-8")
    assert load_config(caminho).saturacao == "avisar"


def test_config_yaml_recusa_valor_invalido(tmp_path):
    caminho = tmp_path / "c.yaml"
    save_config(g.Config(mode="csv"), str(caminho))
    caminho.write_text(caminho.read_text(encoding="utf-8").replace("saturacao: auto", "saturacao: talvez"),
                       encoding="utf-8")
    with pytest.raises(Exception):  # noqa: B017 -- o tipo do erro e' detalhe da validacao do YAML
        load_config(str(caminho))


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("GUARACI_SESSION_DIR", tempfile.mkdtemp(prefix="gs"))
    monkeypatch.setattr(prefs, "VISUAL_PATH", tmp_path / "visual_config.json")
    pasta = tmp_path / "appdir"
    pasta.mkdir()
    shutil.copy(_RAIZ / "app_quimiometria.py", pasta / "app_quimiometria.py")
    shutil.copytree(_RAIZ / "assets", pasta / "assets")
    return str(pasta / "app_quimiometria.py")


def test_web_aba_dados_mostra_e_muda_saturacao(app):
    at = AppTest.from_file(app, default_timeout=120)
    at.session_state["pagina"] = "dados"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    # no modo sintetico (padrao) a opcao nao se aplica; troca para csv e ela aparece
    modo = [w for w in at.selectbox if w.key == "w_modo_entrada"][0]
    modo.set_value("csv")
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    sat = [w for w in at.selectbox if w.key == "w_saturacao"]
    assert sat, "a aba Dados nao oferece a opcao saturacao no modo csv"
    sat[0].set_value("avisar")
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert [w for w in at.selectbox if w.key == "w_saturacao"][0].value == "avisar"
