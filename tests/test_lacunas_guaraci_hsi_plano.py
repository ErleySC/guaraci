"""Telas de imageamento hiperespectral (HSI) e de planejamento de coleta de `guaraci.guaraci`."""
from __future__ import annotations

import types

import pytest

import guaraci.guaraci as g
import guaraci.hsi_extras as hx
import guaraci.hsi_pipeline as hp
import guaraci.plano_coleta as pc


def _resumo(externa=True, baixa=True, vazia=False):
    val = types.SimpleNamespace(n_objetos_teste_externo=4 if externa else 0, n_objetos_teste_interno=6, classes=["a", "b"],
                                sensibilidade_interna={"a": 0.9, "b": 0.8}, sensibilidade_externa={"a": 0.7, "b": 0.6})
    conf = {} if vazia else {f"o{i}": types.SimpleNamespace(heterogeneidade=0.5 if baixa else 0.1) for i in range(7)}
    return {"n_gravacoes_aceitas": 5, "n_gravacoes_total": 6, "n_gravacoes_rejeitadas": 1, "n_components": 3,
            "validacao_externa": val, "confianca_por_objeto": conf}


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_hsi_pasta_invalida_e_voltar(cli, tmp_path, monkeypatch, lang):
    monkeypatch.setitem(g._STATE, "lang", lang)
    cli.dizer("0")
    g._menu_hsi(g.Config())
    cli.dizer(str(tmp_path / "nao_existe"), "")
    g._menu_hsi(g.Config())
    assert "Pasta invalida" in cli.saida() or "Invalid folder" in cli.saida()
    cli.dizer("", "")
    g._menu_hsi(g.Config())


@pytest.mark.parametrize("externa, baixa, vazia", [(True, True, False), (False, False, False), (True, False, True)])
def test_hsi_resume_a_validacao_interna_ou_externa_e_a_confianca_por_objeto(cli, tmp_path, monkeypatch, externa, baixa, vazia):
    monkeypatch.setattr(hp, "run_hsi_pipeline", lambda cfg: _resumo(externa, baixa, vazia))
    cfg = g.Config()
    cfg.hsi_dataset_folder = ""
    cli.dizer(str(tmp_path), "")
    g._menu_hsi(cfg)
    saida = cli.saida()
    assert ("Validacao externa" in saida) is externa and cfg.mode != "hsi"
    assert ("baixa concordancia" in saida) is (baixa and not vazia)


def test_hsi_com_erro_no_pipeline_mostra_a_mensagem_e_reaproveita_a_pasta_anterior(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(hp, "run_hsi_pipeline", lambda cfg: (_ for _ in ()).throw(RuntimeError("cubo ilegivel")))
    cfg = g.Config()
    cfg.hsi_dataset_folder = str(tmp_path)
    cli.dizer("", "")
    g._menu_hsi(cfg)
    assert "cubo ilegivel" in cli.saida()


def test_hsi_caminho_e_erro_com_colchetes_nao_viram_etiqueta_do_rich(cli, tmp_path, monkeypatch):
    # "[/tmp/x]" e' lido pelo rich como etiqueta de fechamento; em Windows o tmp_path nao comeca por "/", entao forca-se o formato Linux
    cfg = g.Config()
    cfg.hsi_dataset_folder = "/nao/existe/x"
    cli.dizer("", "")
    g._menu_hsi(cfg)
    assert "/nao/existe/x" in cli.saida()
    monkeypatch.setattr(hp, "run_hsi_pipeline", lambda cfg: (_ for _ in ()).throw(RuntimeError("cubo em [/tmp/x] ilegivel")))
    cli.dizer(str(tmp_path), "")
    g._menu_hsi(g.Config())
    assert "[/tmp/x]" in cli.saida()


def _dados_hsi(monkeypatch, *, falha_na_carga=False):
    chamadas = []
    if falha_na_carga:
        monkeypatch.setattr(hx, "preparar_dados_hsi", lambda pasta: (_ for _ in ()).throw(OSError("pasta sem cubos")))
    else:
        monkeypatch.setattr(hx, "preparar_dados_hsi", lambda pasta: chamadas.append("dados") or object())
    monkeypatch.setattr(hx, "rodar_hsi_dominio_aplicabilidade",
                        lambda d: {"fracao_dentro": 0.8, "n_objetos_treino": 5, "n_objetos_teste": 3, "fracao_dentro_por_objeto": {"o1": 0.5}})
    monkeypatch.setattr(hx, "rodar_hsi_reamostragem",
                        lambda d: {"avaliabilidade": {"a": {"n_grupos": 2, "nota": "poucos"}, "b": {"n_grupos": 3, "nota": ""}},
                                   "antes": {"a": {"pixels": 10}, "b": {"pixels": 20}}, "depois": {"a": {"pixels": 20}, "b": {"pixels": 20}}})
    monkeypatch.setattr(hx, "rodar_hsi_multiway", lambda d: {"balanced_accuracy_npls": 0.9, "balanced_accuracy_pixelwise": 0.8})
    return chamadas


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_hsi_extras_rodam_dominio_reamostragem_e_multiway_carregando_os_cubos_uma_vez(cli, tmp_path, monkeypatch, lang):
    monkeypatch.setitem(g._STATE, "lang", lang)
    chamadas = _dados_hsi(monkeypatch)
    cli.dizer("1", "2", "3", "")
    g._menu_hsi_extras(str(tmp_path), lang == "PT")
    saida = cli.saida()
    assert chamadas == ["dados"] and "80.0%" in saida and "0.900" in saida and "poucos" in saida


def test_hsi_extras_com_falha_mostra_a_mensagem_e_continua_perguntando(cli, tmp_path, monkeypatch):
    _dados_hsi(monkeypatch, falha_na_carga=True)
    cli.dizer("1", "")
    g._menu_hsi_extras(str(tmp_path), True)
    assert "pasta sem cubos" in cli.saida()


# ---- planejamento de coleta ---------------------------------------------------------------------------------------------------------------


def test_plano_voltar_e_entradas_recusadas(cli):
    cli.dizer("0")
    g._menu_plan(g.Config())
    cli.dizer("i", "")
    g._menu_plan(g.Config())
    g._STATE["lang"] = "PT"
    cli.dizer(" , ")
    g._menu_plan(g.Config())
    for respostas in (("a,b", "x", ""), ("a,b", "0", ""), ("a,b", "2", "z", ""), ("a,b", "2", "c", "abc", "")):
        cli.dizer(*respostas)
        g._menu_plan(g.Config())
    saida = cli.saida()
    for trecho in ("Numero invalido", "pelo menos 1 sessao", "Opcao invalida"):
        assert trecho in saida, trecho


def test_plano_com_alvo_que_o_motor_recusa_mostra_o_erro(cli, monkeypatch):
    monkeypatch.setattr(pc, "plan_from_statistical_target", lambda *a, **k: (_ for _ in ()).throw(ValueError("alpha impossivel")))
    cli.dizer("a,b", "2", "c", "0.001", "")
    g._menu_plan(g.Config())
    assert "alpha impossivel" in cli.saida()


@pytest.mark.parametrize("alvo, valor", [("c", "0.05"), ("d", "0.90")])
def test_plano_grava_markdown_e_planilha_e_so_sobrescreve_se_confirmado(cli, tmp_path, monkeypatch, alvo, valor):
    monkeypatch.setattr(pc, "export_pdf", lambda plano, cam: open(cam, "w", encoding="utf-8").write("pdf"))
    prefixo = str(tmp_path / "meu_plano")
    cli.dizer("a,b", "2", alvo, valor, prefixo, "s", "n", "")
    g._menu_plan(g.Config())
    assert (tmp_path / "meu_plano.md").exists() and (tmp_path / "meu_plano.xlsx").exists() and (tmp_path / "meu_plano.pdf").exists()
    (tmp_path / "meu_plano.md").write_text("antigo", encoding="utf-8")
    cli.dizer("a,b", "2", alvo, valor, prefixo, "s", "n", "")
    g._menu_plan(g.Config())
    assert (tmp_path / "meu_plano.md").read_text(encoding="utf-8") == "antigo"
    cli.dizer("a,b", "2", alvo, valor, prefixo, "", "s", "n", "")
    g._menu_plan(g.Config())
    assert (tmp_path / "meu_plano.md").read_text(encoding="utf-8") != "antigo"


def test_plano_usa_a_pasta_atual_por_padrao_e_avisa_erro_de_gravacao(cli, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pc, "export_excel", lambda plano, cam: (_ for _ in ()).throw(OSError("planilha aberta")))
    cli.dizer("a,b", "2", "c", "0.05", "", "n", "")
    g._menu_plan(g.Config())
    assert "planilha aberta" in cli.saida()


def test_plano_refina_com_amostragem_ativa_em_cada_cenario(cli, tmp_path, monkeypatch):
    from guaraci import amostragem_ativa as aa
    modelo = tmp_path / "m.joblib"
    modelo.write_bytes(b"x")
    cli.dizer("")
    g._refinar_plano_com_amostragem_ativa(True)                            # sem caminho
    cli.dizer(str(modelo), "n")
    g._refinar_plano_com_amostragem_ativa(False)                           # nao confirmou
    monkeypatch.setattr(g, "_carregar_modelo_cli", lambda cam, pt: {})
    cli.dizer(str(modelo), "s")
    g._refinar_plano_com_amostragem_ativa(True)                            # sem ensemble
    monkeypatch.setattr(g, "_carregar_modelo_cli", lambda cam, pt: {"identification_ensemble": object()})
    monkeypatch.setattr(aa, "priorizar_amostragem", lambda ens: (_ for _ in ()).throw(RuntimeError("ensemble ilegivel")))
    cli.dizer(str(modelo), "s")
    g._refinar_plano_com_amostragem_ativa(True)
    linhas = [types.SimpleNamespace(especie="e", adulterante="x", n_sessoes_atual=1, sessoes_faltantes=1, prioridade=1.0,
                                    cobertura_status=types.SimpleNamespace(value="parcial")),
              types.SimpleNamespace(especie="e", adulterante="y", n_sessoes_atual=2, sessoes_faltantes=0, prioridade=0.0, cobertura_status=None)]
    monkeypatch.setattr(aa, "priorizar_amostragem", lambda ens: linhas)
    cli.dizer(str(modelo), "s")
    g._refinar_plano_com_amostragem_ativa(False)
    saida = " ".join(cli.saida().split())
    for trecho in ("ensemble de identificacao", "ensemble ilegivel", "parcial"):
        assert trecho in saida, trecho
