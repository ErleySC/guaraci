"""`guaraci.guaraci`: estado do usuario (idioma, modo, codigos, visual), migracao do estado legado, entrada de teclado e rotulos.

O estado gravado vai sempre para uma pasta temporaria (fixture `cli`); falhas de disco sao injetadas para provar que a tela avisa em vez
de calar.
"""
from __future__ import annotations

import pathlib
import runpy
import sys
import types

import pytest

import guaraci.guaraci as g


def _quebra(msg="disco cheio", exc=OSError):
    def f(*a, **k):
        raise exc(msg)
    return f


def _arquivo_no_lugar_da_pasta(tmp_path, monkeypatch):
    bloqueio = tmp_path / "bloqueio"
    bloqueio.write_text("x", encoding="utf-8")
    monkeypatch.setattr(g, "_USER_DIR", bloqueio)
    for nome in ("_LANG_FLAG", "_MODO_FLAG", "_CFG_PATH", "_CODIGOS_PATH"):
        monkeypatch.setattr(g, nome, bloqueio / nome.strip("_"))
    return bloqueio


def test_config_visual_que_nao_grava_avisa_e_codigos_corrompidos_viram_vazio(cli, monkeypatch):
    monkeypatch.setattr(g._prefs_visuais, "save_visual_config", _quebra())
    g._salvar_visual_cfg({"dpi": 100})
    assert "Falha ao salvar config visual" in cli.saida()
    g._USER_DIR.mkdir(parents=True, exist_ok=True)
    g._CODIGOS_PATH.write_text("{ nao e json", encoding="utf-8")
    assert g._carregar_codigos_usuario() == {}


def test_idioma_e_modo_alternam_e_o_disco_cheio_nao_derruba(cli, tmp_path, monkeypatch):
    assert g._toggle_idioma() == "EN" and g._lang() == "EN" and g._LANG_FLAG.read_text(encoding="utf-8") == "EN"
    assert g._toggle_idioma() == "PT"
    assert g._toggle_modo_usuario() == "avancado" and g._modo_usuario() == "avancado"
    assert g._toggle_modo_usuario() == "iniciante"
    _arquivo_no_lugar_da_pasta(tmp_path, monkeypatch)
    assert g._toggle_idioma() == "EN"                  # a escolha vale na sessao mesmo sem gravar
    assert g._toggle_modo_usuario() == "avancado"


def test_texto_traduzido_com_marcador_que_falta_volta_sem_formatar(cli, monkeypatch):
    monkeypatch.setitem(g._I18N["PT"], "chave_de_teste", "ola {nome}")
    assert g._t("chave_de_teste", nome="Ana") == "ola Ana"
    assert g._t("chave_de_teste", outro=1) == "ola {nome}"


def test_limpar_a_tela_chama_o_sistema_salvo_no_modo_linear(cli, monkeypatch):
    monkeypatch.setattr(g.os, "name", "posix")      # ramo POSIX; o do Windows tem teste proprio (test_cli_inicio_e_idioma)
    g._cls()
    assert len(cli.comandos) == 1
    monkeypatch.setenv("GUARACI_MODO_LINEAR", "1")
    g._cls()
    assert len(cli.comandos) == 1


def test_migracao_do_estado_legado_copia_so_o_que_falta_e_tolera_falhas(cli, tmp_path, monkeypatch):
    antigo = tmp_path / "antigo"
    (antigo / "perfis").mkdir(parents=True)
    (antigo / "perfis" / "p.yaml").write_text("a: 1\n", encoding="utf-8")
    for nome in ("config.yaml", ".cli_wizard_done", "codigos_usuario.json", ".cli_modo_usuario", "visual_config.json"):
        (antigo / nome).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(g, "_BASE_DIR", antigo)
    g._migrar_estado_legado()
    assert g._CFG_PATH.exists() and g._VISUAL_PATH.exists() and (g._PERFIS_DIR / "p.yaml").exists()


def test_migracao_recupera_o_visual_da_pasta_de_trabalho_e_falhas_de_copia_sao_toleradas(cli, tmp_path, monkeypatch):
    vazio = tmp_path / "vazio"
    vazio.mkdir()
    monkeypatch.setattr(g, "_BASE_DIR", vazio)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "visual_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(cwd)
    g._migrar_estado_legado()
    assert g._VISUAL_PATH.exists()
    g._VISUAL_PATH.unlink()
    monkeypatch.setattr(g.shutil, "copy2", _quebra())
    g._migrar_estado_legado()
    assert not g._VISUAL_PATH.exists()


def test_migracao_com_copia_de_arquivo_e_de_perfis_que_falham_segue_adiante(cli, tmp_path, monkeypatch):
    antigo = tmp_path / "antigo2"
    (antigo / "perfis").mkdir(parents=True)
    (antigo / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(g, "_BASE_DIR", antigo)
    monkeypatch.setattr(g.shutil, "copy2", _quebra())
    monkeypatch.setattr(g.shutil, "copytree", _quebra())
    g._migrar_estado_legado()
    assert not g._CFG_PATH.exists() and not g._PERFIS_DIR.exists()


def test_migracao_sem_pasta_do_usuario_gravavel_simplesmente_volta(cli, tmp_path, monkeypatch):
    _arquivo_no_lugar_da_pasta(tmp_path, monkeypatch)
    g._migrar_estado_legado()


def test_entrada_devolve_o_padrao_no_fim_do_arquivo_e_ajuda_de_tela_so_com_texto(cli, monkeypatch):
    assert g._input("p: ", default="d") == "d"
    assert g._ask("[a]p[/a] ", default="e") == "e"
    g._pause()
    monkeypatch.setattr(g, "_AJUDA_TELA", {"PT": "texto de ajuda", "EN": "help text"})
    cli.dizer("?", " resposta ")
    assert g._ask("[a]p[/a] ") == "resposta"
    assert "texto de ajuda" in cli.saida()
    monkeypatch.setattr(g, "_AJUDA_TELA", None)
    cli.dizer("?")
    assert g._ask() == "?"


def test_modelo_nao_assinado_so_e_assinado_com_o_sim_explicito(cli, monkeypatch):
    import guaraci.predicao as p

    class _NaoAssinado(Exception):
        pass

    monkeypatch.setattr(p, "ModeloNaoAssinadoError", _NaoAssinado)
    chamadas = []

    def carregar(caminho, confiar):
        chamadas.append("carregar")
        if len(chamadas) == 1:
            raise _NaoAssinado("sem assinatura")
        return "modelo"

    monkeypatch.setattr(p, "load_model", carregar)
    monkeypatch.setattr(p, "assinar_modelo", lambda c: chamadas.append("assinar"))
    cli.dizer("n")
    with pytest.raises(_NaoAssinado):
        g._carregar_modelo_cli("m.joblib", True)
    assert "assinar" not in chamadas
    chamadas.clear()
    cli.dizer("s")
    assert g._carregar_modelo_cli("m.joblib", True) == "modelo" and "assinar" in chamadas


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_rotulos_de_perfil_cobrem_vazio_quebrado_validado_e_nao_validado(cli, monkeypatch, lang):
    import guaraci.perfil_matriz as pm
    monkeypatch.setitem(g._STATE, "lang", lang)
    assert "declar" in g._rotulo_opcao("perfil_matriz", "")
    perfis = {"bom": types.SimpleNamespace(descricao="Bom (extra)", referencia="paper", nivel_agrupamento_tipico="alta"),
              "cru": types.SimpleNamespace(descricao="", referencia="", nivel_agrupamento_tipico=""),
              "generico": types.SimpleNamespace(descricao="Neutro", referencia="", nivel_agrupamento_tipico="")}

    def carregar(nome):
        if nome == "quebrado":
            raise ValueError("sumiu")
        return perfis[nome]

    monkeypatch.setattr(pm, "load_profile", carregar)
    assert g._rotulo_opcao("perfil_matriz", "quebrado") == "quebrado"
    assert g._rotulo_opcao("perfil_matriz", "bom").startswith("bom — Bom")
    assert "alta" in g._rotulo_opcao("perfil_tecnica", "bom")
    assert "cru" in g._rotulo_opcao("perfil_matriz", "cru") and "valid" in g._rotulo_opcao("perfil_matriz", "cru")
    if lang == "PT":
        assert g._rotulo_opcao("perfil_matriz", "generico") == "generico — Neutro"
    else:   # em ingles o perfil neutro tem nome e descricao proprios (antes saia "genérico — matriz não especificada")
        assert g._rotulo_opcao("perfil_matriz", "generico").startswith("generic — ")
    assert g._rotulo_opcao("qualquer", 5) == "5"


def test_saida_com_edicoes_nao_gravadas_pergunta_e_so_grava_se_pedido(cli, tmp_path, monkeypatch):
    cfg = g.Config()
    foto = g._foto_cfg(cfg)
    assert g._confirmar_saida(cfg, foto) is True
    cfg.max_lvs = cfg.max_lvs + 1
    cli.dizer("", "s")
    assert g._confirmar_saida(cfg, foto) is False and g._confirmar_saida(cfg, foto) is True
    cli.dizer("g")
    assert g._confirmar_saida(cfg, foto) is True and g._CFG_PATH.exists()
    monkeypatch.setattr(g, "_save_config", _quebra("sem permissao"))
    cli.dizer("g")
    assert g._confirmar_saida(cfg, foto) is False and "sem permissao" in cli.saida()


def test_pausa_de_cafe_e_validacao_de_faixa_usam_a_configuracao(cli):
    g._sugerir_pausa()
    assert cli.saida().strip()
    cfg = g.Config(wn_min=900.0, wn_max=500.0)
    assert g._validate_ranges(cfg)


def test_dpi_do_visual_vai_para_a_config_e_valor_ruim_e_ignorado(cli, monkeypatch):
    cfg = g.Config()
    monkeypatch.setattr(g, "_carregar_visual_cfg", lambda: {"dpi": "300"})
    g._sincronizar_dpi(cfg)
    assert cfg.dpi == 300
    monkeypatch.setattr(g, "_carregar_visual_cfg", lambda: {"dpi": "abc"})
    g._sincronizar_dpi(cfg)
    assert cfg.dpi == 300


# ---- topo do modulo: UTF-8 no Windows e flag de modo lidos na importacao ------------------------------------------------------------------


def _reexecutar(monkeypatch, tmp_path, *, plataforma="win32", flag=None, flag_ilegivel=False, subprocesso=None, stdout=None):
    casa = tmp_path / "casa_reexec"
    pasta = casa / ".guaraci"
    pasta.mkdir(parents=True, exist_ok=True)
    if flag is not None:
        (pasta / ".cli_modo_usuario").write_text(flag, encoding="utf-8")
    if flag_ilegivel:
        (pasta / ".cli_modo_usuario").mkdir()
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: casa))
    monkeypatch.setattr(sys, "platform", plataforma)
    if subprocesso is not None:
        import subprocess
        monkeypatch.setattr(subprocess, "run", subprocesso)
    if stdout is not None:
        monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stdout if stdout is not None else sys.stderr)
    return runpy.run_module("guaraci.guaraci", run_name="guaraci_reexecutado")


class _SaidaQueNaoReconfigura:
    def reconfigure(self, **k):
        raise ValueError("sem reconfigure")

    def write(self, s):
        return len(s)

    def flush(self):
        pass


def test_topo_do_modulo_no_windows_tolera_saida_sem_reconfigure_e_chcp_indisponivel(tmp_path, monkeypatch):
    mod = _reexecutar(monkeypatch, tmp_path, subprocesso=_quebra("sem cmd"), stdout=_SaidaQueNaoReconfigura())
    assert mod["_STATE"]["modo_usuario"] == "iniciante"


def test_topo_do_modulo_le_o_modo_gravado_e_ignora_flag_ilegivel(tmp_path, monkeypatch):
    mod = _reexecutar(monkeypatch, tmp_path, plataforma="linux", flag="avancado")
    assert mod["_STATE"]["modo_usuario"] == "avancado"
    mod = _reexecutar(monkeypatch, tmp_path / "outra", plataforma="linux", flag_ilegivel=True)
    assert mod["_STATE"]["modo_usuario"] == "iniciante"
