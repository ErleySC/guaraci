"""Edicao de campo, ajuda por campo, laco unico das telas de configuracao e perfil combinado de `guaraci.guaraci`.

A pessoa digita numero, texto, `?`, Enter ou lixo; o teste confere o que mudou (ou nao) na configuracao e o aviso mostrado.
"""
from __future__ import annotations

import types

import pytest

import guaraci.guaraci as g


def test_campo_inexistente_e_avisado(cli):
    assert g._editar_campo(g.Config(), "campo_que_nao_existe") is False
    assert "nao encontrado" in cli.saida()
    g._STATE["lang"] = "EN"
    assert g._editar_campo(g.Config(), "campo_que_nao_existe") is False
    assert "not found" in cli.saida()


def test_campo_booleano_so_aceita_1_ou_2_e_enter_mantem(cli):
    cfg = g.Config(detailed_figures=False)
    cli.dizer("", "?", "", "7", "1")
    assert g._editar_campo(cfg, "figuras_detalhadas") is False
    assert g._editar_campo(cfg, "figuras_detalhadas") is False        # `?` abre a ajuda (que pausa) e nao muda nada
    assert g._editar_campo(cfg, "figuras_detalhadas") is False        # 7: pede 1 ou 2
    assert "Digite 1 ou 2" in cli.saida() and cfg.detailed_figures is False
    assert g._editar_campo(cfg, "figuras_detalhadas") is True and cfg.detailed_figures is True


def test_campo_de_escolha_aceita_numero_do_menu_e_trata_enter_e_ajuda(cli):
    cfg = g.Config(output_format="png")
    cli.dizer("", "?", "", "2")
    assert g._editar_campo(cfg, "formato_figura") is False
    assert g._editar_campo(cfg, "formato_figura") is False
    assert g._editar_campo(cfg, "formato_figura") is True
    assert cfg.output_format != "png"


def test_campo_de_texto_livre_valida_o_valor_e_mostra_o_erro(cli):
    cfg = g.Config()
    cli.dizer("", "?", "", "abc", "s", "rodada_1")
    assert g._editar_campo(cfg, "tag") is False
    assert g._editar_campo(cfg, "tag") is False
    assert g._editar_campo(cfg, "max_lvs") is False and "Erro:" in cli.saida()
    assert g._editar_campo(cfg, "tag") is True and cfg.tag == "rodada_1"


def test_campo_analitico_pede_confirmacao_e_cancela_sem_ela(cli):
    cfg = g.Config()
    antes = cfg.max_lvs
    cli.dizer("7", "n")
    assert g._editar_campo(cfg, "max_lvs") is False and cfg.max_lvs == antes
    cli.dizer("7", "s")
    assert g._editar_campo(cfg, "max_lvs") is True and cfg.max_lvs == 7


def test_mudar_o_nivel_desliga_o_que_fica_inerte_e_avisa(cli):
    cfg = g.Config(level="N2", run_ddsimca=True)
    nomes = [str(o) for o in g._SPEC_BY_KEY["nivel"]["opcoes"]]
    cli.dizer(str(nomes.index("N1") + 1), "s")
    assert g._editar_campo(cfg, "nivel") is True
    assert cfg.level == "N1" and cfg.run_ddsimca is False


def test_ajuste_por_nivel_ignora_toggle_que_nao_esta_no_catalogo(cli, monkeypatch):
    monkeypatch.delitem(g._SPEC_BY_KEY, "ddsimca")
    assert "ddsimca" not in g._ajustar_toggles_por_nivel(g.Config(level="N1", run_ddsimca=True))


def test_ajuda_de_campo_mostra_uso_exemplos_e_o_texto_de_reserva(cli, monkeypatch):
    monkeypatch.setitem(g._HELP_DB, "campo_x", {"PT": {"desc": "faz x", "use": "use quando A", "nao_use": "nao use quando B",
                                                       "impacto": "alto", "exemplos": {"a": "1", "b": "2"}},
                                                "EN": {}, "default": 3, "range": "1-9"})
    monkeypatch.setitem(g._SPEC_BY_KEY, "campo_x", {"attr": "x", "tipo": "int", "key": "campo_x"})
    cli.dizer("")
    g._mostrar_ajuda("campo_x")
    saida = cli.saida()
    for trecho in ("use quando A", "nao use quando B", "Exemplos:", "1-9"):
        assert trecho in saida, trecho
    monkeypatch.setitem(g._STATE, "lang", "EN")
    cli.dizer("")
    g._mostrar_ajuda("campo_x")
    cli.dizer("")
    g._mostrar_ajuda("max_lvs")


def test_ajuda_de_campo_por_numero_nome_chave_e_tela(cli):
    campos = ["max_lvs", "n_permutacoes"]
    cli.dizer("1", "", "n_permutacoes", "", "permut", "", "zzz", "", "", "")
    g._ajuda_de_campo(campos, "modelo")
    g._ajuda_de_campo(campos, "modelo")
    g._ajuda_de_campo(campos, "modelo")
    g._ajuda_de_campo(campos, "modelo")
    assert g._t("invalido") in cli.saida()
    g._ajuda_de_campo(campos, "modelo")
    g._ajuda_de_campo(campos, None)


def test_tecla_comum_i_alterna_o_idioma(cli):
    assert g._tecla_comum("i", "modelo", g.Config()) == "tratada" and g._lang() == "EN"


def test_laco_da_tela_edita_campo_alterna_avancados_e_recusa_lixo(cli):
    cfg = g.Config()
    cli.dizer("zzz", "", "1", "", "+", "i", "0")
    g._loop_menu("modelo", cfg)
    assert g._t("invalido") in cli.saida() or "invalid" in cli.saida().lower()


def test_laco_da_tela_chama_extra_e_tecla_de_outra_tela(cli):
    chamadas = []
    # acoes da tela sao numeradas depois dos campos visiveis (Basico, modo dx: 7 campos de Dados,
    # com `faixas_excluir` entre os basicos desde a fase 2)
    cli.dizer("8", "0")
    g._loop_menu("dados", g.Config(), extras=[("F", "fundir")], on_extra={"F": lambda: chamadas.append(1)})
    assert chamadas == [1]


def _perfis(monkeypatch, *, tecnica="", matriz="m1", erro=None, falha_ao_salvar=False):
    import guaraci.perfil_matriz as pm

    def carregar(nome):
        if erro:
            raise pm.UnknownProfileError(erro)
        return types.SimpleNamespace(nome=nome)

    monkeypatch.setattr(pm, "load_profile", carregar)
    monkeypatch.setattr(pm, "combine_profiles", lambda nome, a, b: types.SimpleNamespace(nome=nome))

    def salvar(perfil, caminho):
        if falha_ao_salvar:
            raise OSError("disco cheio")
        open(caminho, "w", encoding="utf-8").write("ok")

    monkeypatch.setattr(pm, "save_profile", salvar)
    return g.Config(matrix_profile=matriz, acquisition_profile=tecnica)


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_perfil_combinado_pede_os_dois_perfis_e_o_nome(cli, tmp_path, monkeypatch, lang):
    monkeypatch.setitem(g._STATE, "lang", lang)
    monkeypatch.setattr(g, "_DIR_PERFIS_COMBINADOS", tmp_path / "combinados")
    (tmp_path / "combinados").mkdir()
    cli.dizer("")
    g._salvar_perfil_combinado(_perfis(monkeypatch, matriz=""))
    cli.dizer("")
    g._salvar_perfil_combinado(_perfis(monkeypatch, tecnica=""))
    cli.dizer("", "")
    g._salvar_perfil_combinado(_perfis(monkeypatch, tecnica="t1"))
    cli.dizer("")
    g._salvar_perfil_combinado(_perfis(monkeypatch, tecnica="t1", erro="perfil sumiu"))
    assert "perfil sumiu" in cli.saida()
    cli.dizer("meu_perfil", "")
    g._salvar_perfil_combinado(_perfis(monkeypatch, tecnica="t1"))
    assert (tmp_path / "combinados" / "meu_perfil.yaml").exists()
    cli.dizer("outro", "")
    g._salvar_perfil_combinado(_perfis(monkeypatch, tecnica="t1", falha_ao_salvar=True))
    assert "disco cheio" in cli.saida()


def test_grade_com_transparencia_ruim_volta_ao_padrao():
    assert g._grid_alpha_valido("0.7") == 0.7
    assert g._grid_alpha_valido("abc") == 0.4 and g._grid_alpha_valido(5) == 0.4 and g._grid_alpha_valido(0) == 0.4
