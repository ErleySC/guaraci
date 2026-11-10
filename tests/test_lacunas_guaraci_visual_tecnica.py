"""Telas de visualizacao (paleta, fonte, grade, transparencia) e de tecnica analitica de `guaraci.guaraci`."""
from __future__ import annotations

import json

import pytest

import guaraci.guaraci as g


def _visual():
    return json.loads(g._VISUAL_PATH.read_text(encoding="utf-8")) if g._VISUAL_PATH.exists() else {}


def test_paleta_fonte_e_transparencia_gravam_a_escolha_e_ignoram_o_resto(cli):
    cli.dizer("8", "2", "", "8", "99", "", "8", "", "",
              "9", "4", "", "9", "9", "",
              "11", "1", "", "11", "x", "", "0")
    g._menu_visualization(g.Config())
    v = _visual()
    assert v["paleta"] == list(g._PALETAS_COR)[1] and v["tamanho_fonte"] == "l" and v["alpha_pontos"] == "baixo"


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_grade_alterna_estilo_e_confere_a_transparencia_antes_de_gravar(cli, monkeypatch, lang):
    monkeypatch.setitem(g._STATE, "lang", lang)
    cli.dizer("10", "1", "", "10", "2", "", "10", "3", "", "10", "3", "",
              "10", "4", "0,6", "", "10", "4", "9", "", "10", "4", "abc", "", "10", "4", "", "", "10", "x", "", "0")
    g._menu_visualization(g.Config())
    v = _visual()
    assert v["grid_major"] is False and v["grid_minor"] is True and v["grid_alpha"] == 0.6
    assert v["grid_style"] in ("solid", "dotted", "dashed")


def test_grade_com_estilo_desconhecido_volta_ao_pontilhado(cli):
    g._salvar_visual_cfg({"grid_style": "estranho"})
    cli.dizer("10", "3", "", "0")
    g._menu_visualization(g.Config())
    assert _visual()["grid_style"] == "dotted"


def test_tecnica_ordem_inclui_as_nao_categorizadas_no_fim(monkeypatch):
    monkeypatch.setitem(g._TECNICAS, "tecnica_extra", {"PT": {"nome": "Extra"}, "preproc": "msc"})
    assert g._tecnica_ordem()[-1] == "tecnica_extra"


def test_tecnica_mostra_detalhe_ajuda_da_tela_e_ignora_numero_fora_da_lista(cli):
    cli.dizer("?", "2", "", "?", "", "", "?", "999", "0")
    g._menu_technique(g.Config())
    assert "Faixa:" in cli.saida() and "Arquivos:" in cli.saida()


def test_tecnica_aplica_a_escolhida_alterna_idioma_e_recusa_lixo(cli, monkeypatch):
    monkeypatch.setitem(g._TECNICAS, "tecnica_extra", {"PT": {"nome": "Extra"}, "preproc": "msc", "faixa_min": 500, "faixa_max": 1500})
    monkeypatch.setattr(g, "_TECNICA_CATEGORIAS", list(g._TECNICA_CATEGORIAS) + [("Vazia", "Empty", ["nao_existe"])])
    cfg = g.Config()
    for chave in ("key", "nome"):
        monkeypatch.setitem(g._TECNICA_SELECIONADA, chave, g._TECNICA_SELECIONADA.get(chave, "ft-nir"))     # restaurado ao fim do teste
    cli.dizer("zzz", "", "i", "1", "")
    g._menu_technique(cfg)
    assert g._lang() == "EN" and g._TECNICA_SELECIONADA["key"] == g._tecnica_ordem()[0]
    assert "selected" in cli.saida()


def test_tecnica_cuja_aplicacao_e_recusada_mostra_o_erro(cli, monkeypatch):
    for chave in ("key", "nome"):
        monkeypatch.setitem(g._TECNICA_SELECIONADA, chave, g._TECNICA_SELECIONADA.get(chave, "ft-nir"))
    monkeypatch.setattr(g, "_set_val", lambda *a: (_ for _ in ()).throw(ValueError("faixa invalida para a tecnica")))
    cli.dizer("1", "")
    g._menu_technique(g.Config())
    assert "faixa invalida para a tecnica" in cli.saida()
