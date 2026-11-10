"""Telas-assistente de `guaraci.guaraci`: auditoria, monitorar, comparar, relatorios, fluxo de planejamento, exemplo guiado, perfis e YAML."""
from __future__ import annotations

import types

import pytest

import guaraci.guaraci as g


def _quebra(msg="falhou", exc=ValueError):
    def f(*a, **k):
        raise exc(msg)
    return f


# ---- auditoria --------------------------------------------------------------------------------------------------------------------------------


def _achado(nome, sev):
    return types.SimpleNamespace(nome=nome, severidade=sev, mensagem="m")


def _auditoria(monkeypatch, achados, *, pasta_ok=(True, "dados ok")):
    import guaraci.auditoria_delineamento as ad
    import guaraci.config_io as cio
    monkeypatch.setattr(cio, "_validar_pasta_dados", lambda cfg: pasta_ok)
    monkeypatch.setattr(g.pq, "load_data", lambda cfg: (1, 2, 3, 4, 5, 6))
    monkeypatch.setattr(g.pq, "validate_input", lambda *a: (2, 1, ["A", "B"], 4, ["A-01-01-2026", "B-02-01-2026"], None))
    monkeypatch.setattr(ad, "run_audit", lambda *a, **k: achados)


@pytest.mark.parametrize("achados", [[_achado("a", "ok")], [_achado("a", "aviso")], [_achado("a", "critico"), _achado("b", "silenciado")]])
def test_auditoria_resume_os_achados_por_severidade(cli, monkeypatch, achados):
    _auditoria(monkeypatch, achados)
    cli.dizer("", "")
    g._menu_audit(g.Config())
    assert "checagem" in cli.saida()
    assert "Plano de coleta a partir dos seus dados" in cli.saida() and "UMA classe so" in cli.saida()
    cli.mod._STATE["lang"] = "EN"
    cli.dizer("", "")
    g._menu_audit(g.Config())
    assert "Collection plan from your data" in cli.saida() and "SINGLE class" in cli.saida()


def test_auditoria_volta_antes_de_rodar_recusa_dados_ausentes_e_mostra_erro_de_carga(cli, monkeypatch):
    _auditoria(monkeypatch, [], pasta_ok=(False, "pasta vazia"))
    cli.dizer("")
    g._menu_audit(g.Config())
    assert "pasta vazia" in cli.saida()
    _auditoria(monkeypatch, [])
    cli.dizer("0")
    g._menu_audit(None)
    cli.dizer("i", "")
    monkeypatch.setattr(g.pq, "load_data", _quebra("dados corrompidos"))
    cli.dizer("", "")
    g._menu_audit(g.Config())
    assert "dados corrompidos" in cli.saida()


# ---- monitorar --------------------------------------------------------------------------------------------------------------------------------


def _csvs(tmp_path):
    ref = tmp_path / "ref.csv"
    ref.write_text("lote,a,b\n1,1,2\n2,2,3\n", encoding="utf-8")
    novos = tmp_path / "novos.csv"
    novos.write_text("lote,a,b\n3,2,2\n", encoding="utf-8")
    return str(ref), str(novos)


def test_monitorar_monta_os_argumentos_e_detecta_a_coluna_de_lote(cli, tmp_path, monkeypatch):
    from guaraci import monitorar_cli as mon
    ref, novos = _csvs(tmp_path)
    chamadas = []
    monkeypatch.setattr(mon, "executar", lambda args: chamadas.append(args))
    cli.dizer(ref, novos, "", "")
    g._menu_monitorar(g.Config())
    assert chamadas[0][:4] == ["--referencia", ref, "--novos", novos] and chamadas[0][-2:] == ["--lote", "lote"]


def test_monitorar_recusa_arquivos_ausentes_csv_ilegivel_e_passos_cancelados(cli, tmp_path, monkeypatch):
    from guaraci import monitorar_cli as mon
    ref, novos = _csvs(tmp_path)
    monkeypatch.setattr(mon, "executar", lambda args: None)
    cli.dizer("0")
    g._menu_monitorar(g.Config())                                       # volta no passo 1
    cli.dizer(str(tmp_path / "nao.csv"), "")
    g._menu_monitorar(g.Config())
    cli.dizer(ref, str(tmp_path / "nao2.csv"), "")
    g._menu_monitorar(g.Config())
    cli.dizer(ref, "0")
    g._menu_monitorar(g.Config())                                       # volta no passo 2
    cli.dizer(ref, novos, "0")
    g._menu_monitorar(g.Config())                                       # volta no passo 3
    monkeypatch.setattr(g, "_ler_csv_detectando_separador", _quebra("csv ilegivel", OSError))
    cli.dizer(ref, novos, "")
    g._menu_monitorar(g.Config())
    assert cli.saida().count("Arquivo nao encontrado") + cli.saida().count("File not found") >= 2 and "csv ilegivel" in cli.saida()


def test_monitorar_com_exemplo_sintetico(cli, tmp_path, monkeypatch):
    from guaraci import cli_comandos as cc
    from guaraci import exemplos as ex
    monkeypatch.chdir(tmp_path)
    chamadas = []
    monkeypatch.setattr(ex, "gerar", lambda cen, pasta: chamadas.append(("gerar", cen)))
    monkeypatch.setattr(cc, "_rodar_monitorar_exemplo", lambda pasta: chamadas.append(("rodar",)))
    cli.dizer("x", "")
    g._menu_monitorar(g.Config())
    assert chamadas == [("gerar", "monitorar"), ("rodar",)]


# ---- comparar / relatorios ------------------------------------------------------------------------------------------------------------------------------


def test_comparar_monta_a_chamada_da_anova(cli, tmp_path, monkeypatch):
    from guaraci import anova_cli
    tab = tmp_path / "t.csv"
    tab.write_text("teor,origem,amostra\n1,a,x\n", encoding="utf-8")
    chamadas = []
    monkeypatch.setattr(anova_cli, "executar", lambda args: chamadas.append(args))
    for lang in ("PT", "EN"):
        g._STATE["lang"] = lang
        cli.dizer(str(tab), "teor, outra", "origem", "amostra", "Tukey", "")
        g._menu_comparar(g.Config())
    assert chamadas[0][0] == str(tab) and "--metodo" in chamadas[0] and chamadas[1][-4:-2] == ["--idioma", "en"] or "--idioma" in chamadas[1]


def test_comparar_recusa_arquivo_ausente_passos_vazios_e_aceita_o_exemplo(cli, tmp_path, monkeypatch):
    from guaraci import anova_cli
    from guaraci import cli_comandos as cc
    from guaraci import exemplos as ex
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(anova_cli, "executar", lambda args: None)
    cli.dizer("0")
    g._menu_comparar(g.Config())
    cli.dizer(str(tmp_path / "nao.csv"), "")
    g._menu_comparar(g.Config())
    tab = tmp_path / "t.csv"
    tab.write_text("a\n1\n", encoding="utf-8")
    cli.dizer(str(tab), "")
    g._menu_comparar(g.Config())                                       # resposta vazia
    cli.dizer(str(tab), "teor", "")
    g._menu_comparar(g.Config())                                       # grupo vazio
    cli.dizer(str(tab), "teor", "origem", "", "")
    g._menu_comparar(g.Config())                                       # sem unidade, metodo padrao
    chamadas = []
    monkeypatch.setattr(ex, "gerar", lambda cen, pasta: chamadas.append(cen))
    monkeypatch.setattr(cc, "_rodar_anova_exemplo", lambda pasta: chamadas.append("rodou"))
    cli.dizer("x", "")
    g._menu_comparar(g.Config())
    assert chamadas == ["comparar", "rodou"]


def test_relatorios_pergunta_pasta_e_formato(cli, tmp_path, monkeypatch):
    from guaraci import relatorio_cli
    chamadas = []
    monkeypatch.setattr(relatorio_cli, "executar", lambda args: chamadas.append(args))
    cli.dizer("")
    g._menu_relatorios(g.Config())                                      # sem pasta
    cli.dizer(str(tmp_path), "0")
    g._menu_relatorios(g.Config())                                      # volta no formato
    cfg = g.Config()
    cfg.output_folder = str(tmp_path)
    cli.dizer("", "", "")
    g._menu_relatorios(cfg)                                             # pasta e formato padrao
    assert chamadas == [[str(tmp_path), "--formato", "pdf"]]


def test_assistente_de_campo_alterna_idioma_no_meio_do_passo(cli, tmp_path):
    cli.dizer("0")
    assert g._perguntar("comparar", g.Config(), 1, 3, "Rotulo", "ex") is None
    cli.dizer("i")                                  # prompt de valor: letra de tecla global e' so' texto
    assert g._perguntar("comparar", g.Config(), 1, 3, "Rotulo", "ex") == "i"


# ---- fluxo de planejamento --------------------------------------------------------------------------------------------------------------------------


def test_fluxo_de_planejamento_executa_pula_e_volta(cli, monkeypatch):
    chamadas = []
    for nome in ("_menu_plan", "_menu_selecao_amostras", "_menu_audit"):
        monkeypatch.setattr(g, nome, (lambda n: lambda cfg: chamadas.append(n))(nome))
    cli.dizer("", "2", "")
    g._menu_fluxo_planejamento(g.Config())
    assert chamadas == ["_menu_plan", "_menu_audit"]
    cli.dizer("v")
    g._menu_fluxo_planejamento(None)
    cli.dizer("i", "2", "2", "2")
    g._menu_fluxo_planejamento(g.Config())


# ---- exemplo guiado ----------------------------------------------------------------------------------------------------------------------------


def test_exemplo_guiado_lista_recusa_lixo_e_gera_o_cenario_pelo_nome_ou_numero(cli, tmp_path, monkeypatch):
    from guaraci import exemplos as ex
    monkeypatch.chdir(tmp_path)
    cli.dizer("zzz", "", "0")
    g._menu_exemplo(g.Config())
    cli.dizer("i", "0")
    g._menu_exemplo(g.Config())
    g._STATE["lang"] = "PT"
    monkeypatch.setattr(ex, "gerar", lambda cen, pasta: {"pasta": pasta, "arquivos": ["a.csv"], "comando": f"guaraci {cen}"})
    cli.dizer("1", "", "n", "", "0")                                      # numero, pasta padrao, nao carregar
    g._menu_exemplo(g.Config())
    assert "a.csv" in cli.saida()


def test_exemplo_guiado_roda_executor_sem_configuracao_se_pedido(cli, tmp_path, monkeypatch):
    from guaraci import cli_comandos as cc
    from guaraci import exemplos as ex
    monkeypatch.chdir(tmp_path)
    rodados = []
    monkeypatch.setattr(ex, "gerar", lambda cen, pasta: {"pasta": pasta, "arquivos": [], "comando": "c"})
    monkeypatch.setitem(cc.EXECUTORES_SEM_CONFIG, "monitorar", lambda pasta: rodados.append(pasta))
    cli.dizer("monitorar", "", "r", "", "monitorar", "", "n", "", "0")
    g._menu_exemplo(g.Config())
    assert len(rodados) == 1


@pytest.mark.parametrize("resposta", ["s", "r", "n"])
def test_exemplo_guiado_carrega_a_configuracao_na_sessao(cli, tmp_path, monkeypatch, resposta):
    from guaraci import exemplos as ex
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ex, "gerar", lambda cen, pasta: {"pasta": pasta, "arquivos": [], "comando": "c"})
    novo = g.Config(max_lvs=9)
    monkeypatch.setattr(g, "_load_config", lambda cam: novo)
    rodadas = []
    monkeypatch.setattr(g, "_rodar_pipeline", lambda cfg: rodadas.append(cfg))
    cfg = g.Config()
    cli.dizer("classificar", "", resposta, "", "0")
    g._menu_exemplo(cfg)
    assert (cfg.max_lvs == 9) is (resposta != "n") and bool(rodadas) is (resposta == "r")


def test_exemplo_guiado_com_configuracao_ilegivel_mostra_o_erro(cli, tmp_path, monkeypatch):
    from guaraci import exemplos as ex
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ex, "gerar", lambda cen, pasta: {"pasta": pasta, "arquivos": [], "comando": "c"})
    monkeypatch.setattr(g, "_load_config", _quebra("yaml quebrado", RuntimeError))
    cli.dizer("classificar", "", "s", "", "0")
    g._menu_exemplo(g.Config())
    assert "yaml quebrado" in cli.saida()


# ---- perfis -------------------------------------------------------------------------------------------------------------------------------


def test_aplicar_perfil_devolve_so_o_que_mudou():
    cfg = g.Config()
    primeira = g.aplicar_perfil(cfg, "explorar", "rapido")
    assert primeira and g.aplicar_perfil(cfg, "explorar", "rapido") == []


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_perfis_aplica_mostra_o_resumo_e_so_roda_com_confirmacao(cli, monkeypatch, lang):
    monkeypatch.setitem(g._STATE, "lang", lang)
    rodadas = []
    monkeypatch.setattr(g, "_rodar_pipeline", lambda cfg: rodadas.append(1))
    cfg = g.Config()
    cli.dizer("1", "1", "", "", "")                      # explorar + rapido: aplica, nao roda
    g._menu_profiles(cfg)
    cli.dizer("1", "1", "n")                              # recusa aplicar
    g._menu_profiles(cfg)
    cli.dizer("1", "1", "", "s")                          # aplica e roda (nada muda: segunda vez)
    g._menu_profiles(cfg)
    assert rodadas == [1]


def test_perfis_transferir_mostra_o_comando_em_vez_de_configurar(cli):
    from guaraci import catalogo as cat
    n = list(cat.PERFIS_OBJETIVO).index("transferir") + 1
    cli.dizer(str(n), "")
    g._menu_profiles(g.Config())
    assert "guaraci transferir" in cli.saida()


def test_perfis_navega_pelo_menu_grava_carrega_idioma_lixo_e_voltar(cli):
    n = len(g._cat.PERFIS_OBJETIVO)          # acoes da tela: n+1 = gravar, n+2 = carregar (depois dos itens)
    cli.dizer(str(n + 1), "meu perfil!", "", str(n + 2), "1", "", "i", "99", "", "0")
    g._menu_profiles(g.Config())
    assert any(p.suffix == ".yaml" for p in g._PERFIS_DIR.iterdir())
    cli.dizer("1", "0")
    g._menu_profiles(g.Config())                         # volta no passo 2


def test_perfil_yaml_cancelado_sem_nome_e_carregar_sem_perfis_ou_com_numero_ruim(cli, monkeypatch):
    cli.dizer("", "")
    g._salvar_yaml(g.Config())
    cli.dizer("")
    g._carregar_yaml(g.Config())
    assert "Nenhum perfil salvo" in cli.saida()
    cli.dizer("meu", "")
    g._salvar_yaml(g.Config(max_lvs=11))
    cli.dizer("x", "")
    g._carregar_yaml(g.Config())
    cfg = g.Config()
    cli.dizer("1", "")
    g._carregar_yaml(cfg)
    assert cfg.max_lvs == 11
    monkeypatch.setattr(g, "_load_config", _quebra("perfil antigo", ValueError))
    cli.dizer("1", "")
    g._carregar_yaml(g.Config())
    monkeypatch.setattr(g, "_save_config", _quebra("sem espaco", OSError))
    cli.dizer("outro", "")
    g._salvar_yaml(g.Config())
    assert "perfil antigo" in cli.saida() and "sem espaco" in cli.saida()
