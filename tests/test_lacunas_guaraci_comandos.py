"""Subcomandos (`web`, `doctor`, `run`, `demo`, `perfis`), validacao de opcoes e o laco principal de `guaraci.guaraci`."""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import pathlib
import subprocess
import sys

import pytest

import guaraci.guaraci as g


@pytest.fixture
def flags_da_linha_de_comando():
    """`_SEM_AJUSTE` e `_MAIS_GRAFICOS` sao listas de um elemento (estado do processo): cada teste as devolve como achou."""
    antes = (g._SEM_AJUSTE[0], g._MAIS_GRAFICOS[0])
    g._SEM_AJUSTE[0] = g._MAIS_GRAFICOS[0] = False
    yield
    g._SEM_AJUSTE[0], g._MAIS_GRAFICOS[0] = antes


def _quebra(msg="falhou", exc=OSError):
    def f(*a, **k):
        raise exc(msg)
    return f


# ---- web ------------------------------------------------------------------------------------------------------------------------------------


def test_web_recusa_quando_o_app_nao_esta_na_instalacao_e_delega_ao_streamlit_quando_esta(monkeypatch, capsys):
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: False)
    with pytest.raises(SystemExit) as e:
        g._comando_web()
    assert e.value.code == 1 and "app_web.py" in capsys.readouterr().err
    monkeypatch.undo()
    chamadas = []
    monkeypatch.setattr(subprocess, "call", lambda cmd, env=None: chamadas.append((cmd, env)) or 0)
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: True)
    with pytest.raises(SystemExit) as e:
        g._comando_web()
    assert e.value.code == 0 and chamadas[0][1]["GUARACI_LOCAL_MODE"] == "1" and "--server.address=127.0.0.1" in chamadas[0][0]
    # modo local mantem o detalhe completo do erro e o limite de upload de antes
    # (o config.toml do repositorio e' o do modo servidor: "none" e 50 MB)
    assert "--client.showErrorDetails=full" in chamadas[0][0] and "--server.maxUploadSize=500" in chamadas[0][0]


# ---- doctor ----------------------------------------------------------------------------------------------------------------------------


def test_versao_pelos_metadados_cai_em_interrogacao_quando_nao_ha_distribuicao(monkeypatch):
    assert g._versao_por_metadados("numpy") != "?"
    monkeypatch.setattr(importlib.metadata, "version", _quebra("sem dist", importlib.metadata.PackageNotFoundError))
    assert g._versao_por_metadados("pacote_que_nao_existe") == "?"


def test_doctor_lista_dependencias_e_grava_o_relatorio_na_pasta_atual(cli, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    g._comando_doctor()
    assert (tmp_path / "guaraci_doctor.txt").exists() and "Todas as dependencias obrigatorias" in cli.saida()


def test_doctor_aponta_dependencia_faltando_import_quebrado_psutil_ausente_e_relatorio_nao_gravado(cli, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real_find = importlib.util.find_spec
    real_import = importlib.import_module
    monkeypatch.setattr(importlib.util, "find_spec", lambda nome, *a: None if nome == "yaml" else real_find(nome, *a))
    monkeypatch.setattr(importlib, "import_module", lambda nome, *a: (_ for _ in ()).throw(ImportError("quebrado")) if nome == "scipy"
                        else real_import(nome, *a))
    hw = dict(g.pq.hardware_probe())
    hw["psutil_ok"] = False
    monkeypatch.setattr(g.pq, "hardware_probe", lambda: hw)
    monkeypatch.setattr(pathlib.Path, "write_text", _quebra("disco cheio"))
    g._comando_doctor()
    saida = cli.saida()
    for trecho in ("[FALTA] yaml", "erro ao importar", "psutil ausente", "Nao foi possivel salvar", "faltando"):
        assert trecho in saida, trecho


# ---- run -----------------------------------------------------------------------------------------------------------------------------


def _config(tmp_path, texto=None, **extra):
    caminho = tmp_path / "config.yaml"
    if texto is not None:
        caminho.write_text(texto, encoding="utf-8")
    else:
        g._save_config(g.Config(mode="sintetico", **extra), str(caminho))
    return str(caminho)


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_run_recusa_config_ausente_yaml_quebrado_e_config_invalida(tmp_path, monkeypatch, capsys, lang):
    monkeypatch.setitem(g._STATE, "lang", lang)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        g._comando_run(None)
    assert e.value.code == 2 and ("nao encontrado" if lang == "PT" else "not found") in capsys.readouterr().err
    for texto in ("a: [1, 2\nb: {", "chave_que_nao_existe: 1\n"):
        with pytest.raises(SystemExit) as e:
            g._comando_run(_config(tmp_path, texto))
        assert e.value.code == 2


def test_run_recusa_pasta_de_dados_ausente(tmp_path, monkeypatch, capsys):
    caminho = _config(tmp_path, f"modo_entrada: dx\npasta_dados: {(tmp_path / 'nao_existe').as_posix()}\n")
    with pytest.raises(SystemExit) as e:
        g._comando_run(caminho)
    assert e.value.code == 2 and capsys.readouterr().err


@pytest.mark.parametrize("sem_ajuste, mais_graficos", [(False, False), (True, True)])
def test_run_executa_uma_vez_e_aplica_as_flags(tmp_path, monkeypatch, capsys, flags_da_linha_de_comando, sem_ajuste, mais_graficos):
    import guaraci.dados_io as dio
    recebidas = []
    monkeypatch.setattr(g.pq, "executar", lambda cfg: recebidas.append(cfg))
    monkeypatch.setattr(dio, "garantia_agrupamento_prevista", lambda cfg: "none")
    g._SEM_AJUSTE[0], g._MAIS_GRAFICOS[0] = sem_ajuste, mais_graficos
    g._comando_run(_config(tmp_path))
    assert len(recebidas) == 1 and (recebidas[0].auto_adjust is False) is sem_ajuste
    assert (recebidas[0].figure_selection == "todos") is mais_graficos
    assert "Concluido" in capsys.readouterr().out


def test_run_com_execucao_que_falha_sai_com_codigo_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(g.pq, "executar", _quebra("sem memoria", RuntimeError))
    with pytest.raises(SystemExit) as e:
        g._comando_run(_config(tmp_path))
    assert e.value.code == 1 and "sem memoria" in capsys.readouterr().err


# ---- demo / perfis --------------------------------------------------------------------------------------------------------------------------------


def _executar_que_cria_a_pasta(raiz_base):
    def f(cfg):
        pathlib.Path(cfg.output_root_folder, "Amostra", "Modo", "PLSDA_OE_001").mkdir(parents=True, exist_ok=True)
    return f


def test_demo_que_falha_ou_nao_produz_pasta_sai_com_codigo_1(cli, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(g.pq, "executar", _quebra("sintetico quebrou", RuntimeError))
    with pytest.raises(SystemExit) as e:
        g._comando_demo()
    assert e.value.code == 1
    monkeypatch.setattr(g.pq, "executar", lambda cfg: None)
    with pytest.raises(SystemExit) as e:
        g._comando_demo()
    assert e.value.code == 1 and "nenhuma pasta de saida" in cli.saida()


@pytest.mark.parametrize("plataforma", ["win32", "darwin", "linux"])
def test_demo_concluida_abre_a_pasta_conforme_o_sistema(cli, tmp_path, monkeypatch, plataforma):
    monkeypatch.chdir(tmp_path)
    abertos = []
    monkeypatch.setattr(g.pq, "executar", _executar_que_cria_a_pasta(tmp_path))
    monkeypatch.setattr(sys, "platform", plataforma)
    monkeypatch.setattr(g.os, "startfile", lambda p: abertos.append(("startfile", p)), raising=False)
    monkeypatch.setattr(subprocess, "run", lambda cmd, check=False: abertos.append(tuple(cmd)))
    g._comando_demo()
    assert "Demo concluida" in cli.saida() and len(abertos) == 1


def test_demo_tolera_falha_ao_abrir_o_explorador(cli, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(g.pq, "executar", _executar_que_cria_a_pasta(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", _quebra("sem xdg-open"))
    g._comando_demo()
    assert "Demo concluida" in cli.saida()


def test_listagem_de_perfis_mostra_o_que_ha(capsys):
    g._listar_perfis()
    assert "Perfis de tecnica disponiveis" in capsys.readouterr().out


# ---- opcoes ------------------------------------------------------------------------------------------------------------------------


def test_extrair_opcoes_separa_chave_valor_de_posicionais():
    assert g._extrair_opcoes(["--modo=controle", "demo", "--x", "--perfil=a=b"]) == ({"modo": "controle", "perfil": "a=b"}, ["demo", "--x"])


@pytest.mark.parametrize("opcoes, trecho", [({"foo": "1"}, "desconhecida"), ({"perfil": ""}, "exige um valor"), ({"mode": "x"}, "invalido"),
                                             ({"perfil": "perfil_inexistente_xyz"}, "perfil")])
def test_opcoes_invalidas_saem_com_codigo_2(opcoes, trecho, capsys):
    with pytest.raises(SystemExit) as e:
        g._validar_opcoes(dict(opcoes))
    assert e.value.code == 2 and trecho in capsys.readouterr().err.lower()


def test_opcao_modo_vira_mode_e_perfil_valido_passa():
    opcoes = {"modo": "cego", "perfil": "generico"}
    g._validar_opcoes(opcoes)
    assert opcoes["mode"] == "cego" and "modo" not in opcoes


# ---- main: despacho --------------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def principal(monkeypatch, flags_da_linha_de_comando):
    monkeypatch.setattr(sys, "argv", ["guaraci"])
    monkeypatch.setattr(g, "_OPCOES_CLI", {})
    monkeypatch.setattr(g, "_AJUDA_TELA", None)


@pytest.mark.parametrize("comando, alvo", [("version", "_comando_versao"), ("demo", "_comando_demo"), ("doctor", "_comando_doctor"),
                                           ("perfis", "_listar_perfis"), ("web", "_comando_web")])
def test_main_despacha_cada_comando(principal, monkeypatch, comando, alvo):
    chamadas = []
    monkeypatch.setattr(g, alvo, lambda *a: chamadas.append(alvo))
    g.main([comando])
    assert chamadas == [alvo]


def test_main_despacha_run_ajuda_e_comando_desconhecido(principal, monkeypatch, capsys):
    chamadas = []
    monkeypatch.setattr(g, "_comando_run", lambda caminho: chamadas.append(caminho))
    g.main(["run", "x.yaml"])
    g.main(["run"])
    assert chamadas == ["x.yaml", None]
    g.main(["--help"])
    assert capsys.readouterr().out.strip()
    with pytest.raises(SystemExit) as e:
        g.main(["comando_que_nao_existe"])
    assert e.value.code == 2


def test_main_repassa_subcomandos_e_o_codigo_de_saida(principal, monkeypatch):
    sub = next(iter(g._cmd.SUBCOMANDOS))
    monkeypatch.setattr(g._cmd, "subcomando", lambda nome, args: 0)
    g.main([sub, "--mais-graficos"])
    monkeypatch.setattr(g._cmd, "subcomando", lambda nome, args: 3)
    with pytest.raises(SystemExit) as e:
        g.main([sub])
    assert e.value.code == 3


def test_main_ativa_o_modo_linear_quando_pedido(principal, monkeypatch):
    chamadas = []
    monkeypatch.setenv("GUARACI_MODO_LINEAR", "1")
    monkeypatch.setattr(g, "_ativar_modo_linear", lambda: chamadas.append(1))
    monkeypatch.setattr(g, "_comando_versao", lambda: None)
    g.main(["version"])
    assert chamadas == [1]


# ---- main: laco interativo ---------------------------------------------------------------------------------------------------------


def test_main_sai_pelo_q_e_pelo_fim_da_entrada(cli, principal):
    cli.dizer("q")
    g.main([])
    assert cli.saida().strip()
    g.main([])                                                              # EOF: despedida


def test_main_carrega_a_config_do_usuario_aplica_opcoes_e_idioma_salvo(cli, principal, monkeypatch):
    monkeypatch.delenv("GUARACI_IDIOMA", raising=False)   # sem variavel, vale o idioma gravado com [I]
    g._USER_DIR.mkdir(parents=True, exist_ok=True)
    g._save_config(g.Config(max_lvs=7), str(g._CFG_PATH))
    g._LANG_FLAG.write_text("EN", encoding="utf-8")
    cli.dizer("q")
    g.main(["--perfil=generico", "--mode=cego", "--sem-ajuste-automatico"])
    assert g._lang() == "EN" and g._OPCOES_CLI["perfil"] == "generico"
    g._CFG_PATH.write_text("chave_invalida: 1\n", encoding="utf-8")        # config ilegivel: segue com os padroes
    cli.dizer("q")
    g.main([])


def test_main_despacha_as_teclas_do_menu(cli, principal, monkeypatch):
    chamadas = []
    for tid, tela in g._cat.TELAS.items():
        monkeypatch.setattr(g, tela["handler"], (lambda n: lambda cfg=None: chamadas.append(n))(tela["handler"]))
    monkeypatch.setattr(g, "_menu_hardware", lambda cfg: chamadas.append("hw"))
    monkeypatch.setattr(g, "_abrir_assistente", lambda *a: chamadas.append("assistente"))
    monkeypatch.setattr(g, "_abrir_confianca", lambda *a: chamadas.append("confianca"))
    monkeypatch.setattr(g, "_menu_about", lambda cfg: chamadas.append("sobre"))
    monkeypatch.setattr(g, "_menu_help", lambda cfg: chamadas.append("ajuda"))
    monkeypatch.setattr(g, "_rodar_pipeline", lambda cfg: chamadas.append("rodar"))
    teclas = [t for t in g._cat.tela_por_tecla() if t != "R"]          # caminhos grupo.item (2.3 = Dados)
    cli.dizer(*teclas, "G", "A", "C", "?", "", "", "R", "q")
    g.main([])
    assert "hw" in chamadas and "assistente" in chamadas and "rodar" in chamadas and len(chamadas) >= len(teclas)


def test_main_trata_modo_idioma_nome_salvar_carregar_e_opcao_invalida(cli, principal):
    cli.dizer("M", "", "I", "1.4", "perfil_a", "", "1.5", "1", "", "zz", "", "Q")
    g.main([])
    saida = cli.saida()
    assert "Modo" in saida or "Mode" in saida
    assert any(p.suffix == ".yaml" for p in g._PERFIS_DIR.iterdir())


def test_main_sair_pela_tecla_s_de_uma_tela_pede_confirmacao_do_que_nao_foi_salvo(cli, principal, monkeypatch):
    monkeypatch.setattr(g, "_menu_project", lambda cfg: (_ for _ in ()).throw(g._SairDoPrograma()))
    tecla = next(t for t, tid in g._cat.tela_por_tecla().items() if g._cat.TELAS[tid]["handler"] == "_menu_project")
    cli.dizer(tecla)
    g.main([])
    assert cli.saida().strip()


def test_main_com_edicao_nao_gravada_volta_ao_menu_se_o_usuario_nao_confirma_a_saida(cli, principal, monkeypatch):
    def edita(cfg):
        cfg.max_lvs = cfg.max_lvs + 1
    monkeypatch.setattr(g, "_menu_project", edita)
    tecla = next(t for t, tid in g._cat.tela_por_tecla().items() if g._cat.TELAS[tid]["handler"] == "_menu_project")
    cli.dizer(tecla, "q", "", "q", "s")
    g.main([])
    assert "ajuste(s) de menu ainda nao gravado(s)" in cli.saida()
