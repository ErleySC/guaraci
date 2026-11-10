"""Ramos restantes de `guaraci.guaraci`: indicadores de status e RAM, rotulos de ajuda, avisos de validacao e bordas de varias telas."""
from __future__ import annotations

import json
import pathlib
import runpy
import sys
import types

import pandas as pd
import pytest

import guaraci.guaraci as g


def _quebra(msg="falhou", exc=OSError):
    def f(*a, **k):
        raise exc(msg)
    return f


class _Mem:
    def __init__(self, total, livre):
        self.total, self.available = total * 1024**3, livre * 1024**3


@pytest.mark.parametrize("livre", [12, 6, 2])
def test_caixa_de_execucao_indica_a_ram_livre_em_tres_niveis_e_tolera_falha_da_sonda(cli, monkeypatch, livre):
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _Mem(16, livre))
    g._print_run_box(g.Config())
    assert f"{livre}.0/16 GB" in cli.saida()
    monkeypatch.setattr(psutil, "virtual_memory", _quebra("sonda quebrou", RuntimeError))
    g._print_run_box(g.Config())
    assert "RAM N/A" in cli.saida()


def test_barra_de_status_e_uma_linha_com_proximo_passo_e_sem_hardware(cli, monkeypatch):
    """Guia de design v2: o hardware saiu da tela inicial (fica em [H]); o status
    cabe em uma linha e traz o proximo passo, sem sondar a memoria da maquina."""
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", _quebra("nao deveria sondar", RuntimeError))
    g._print_status(g.Config())
    saida = cli.saida()
    assert "Hardware" not in saida and "Proximo passo" in saida and "Dados:" in saida
    assert saida.count("Status") == 1


def test_descricao_curta_cai_na_do_catalogo_quando_nao_ha_ajuda_do_campo(monkeypatch):
    monkeypatch.setattr(g, "_HELP_DB", {})
    assert g._desc_curta("max_lvs", 60)


def test_revisao_com_opls_e_ddsimca_ligados_nao_os_chama_de_desativados(cli, monkeypatch):
    monkeypatch.setattr(g, "_cfgv", lambda c, k, d=None: {"opls_da": True, "ddsimca": True}.get(k, d))
    g._guaraci_revisar_config(g.Config())
    assert "desativado" not in cli.saida()


def test_navegar_secoes_ignora_chave_desconhecida(cli):
    cli.dizer("zzz")
    g._guaraci_navegar_secoes(g.Config())


def test_fluxo_de_decisao_que_so_muda_o_nivel_nao_mexe_no_preprocessamento(cli, monkeypatch):
    opcao = {"rotulo": {"PT": "p", "EN": "e"}, "tecnicas": [], "validacao": {"PT": "v", "EN": "v"}, "criterio": {"PT": "c", "EN": "c"},
             "nivel": "N3"}
    monkeypatch.setattr(g, "_FLUXO_DECISAO", [opcao])
    cfg = g.Config()
    antes = cfg.default_preprocessing
    cli.dizer("1", "s", "")
    g._guaraci_fluxo_decisao(cfg)
    assert cfg.level == "N3" and cfg.default_preprocessing == antes


def test_trocar_de_nivel_sem_toggle_para_desligar_nao_avisa(cli, monkeypatch):
    monkeypatch.setattr(g, "_ajustar_toggles_por_nivel", lambda cfg: [])
    cfg = g.Config(level="N2")
    nomes = [str(o) for o in g._SPEC_BY_KEY["nivel"]["opcoes"]]
    cli.dizer(str(nomes.index("N3") + 1), "s")
    assert g._editar_campo(cfg, "nivel") is True and cfg.level == "N3"


def test_validacao_sem_agrupamento_mostra_o_alerta_de_vazamento(cli, monkeypatch):
    monkeypatch.setattr(g, "_cfgv", lambda c, k, d=None: False if k == "validacao_group_aware" else d)
    cli.dizer("0")
    g._menu_validation(g.Config())
    assert "GroupKFold DESATIVADO" in cli.saida()


def test_codificacao_lista_os_codigos_do_usuario_e_nao_grava_especie_vazia(cli):
    g._USER_DIR.mkdir(parents=True, exist_ok=True)
    g._CODIGOS_PATH.write_text(json.dumps({"ZZ": "Especie Z"}), encoding="utf-8")
    cli.dizer("1", "", "2", "ab", "", "", "0")
    g._menu_encoding(g.Config())
    assert "Especie Z" in cli.saida() and json.loads(g._CODIGOS_PATH.read_text(encoding="utf-8")) == {"ZZ": "Especie Z"}


def test_predicao_sem_coluna_de_classe_nao_monta_a_tabela_por_classe(cli, tmp_path, monkeypatch):
    import guaraci.predicao as pred
    modelo = tmp_path / "m.joblib"
    modelo.write_bytes(b"x")
    csv = tmp_path / "n.csv"
    csv.write_text("1;2\n", encoding="utf-8")
    monkeypatch.setattr(g, "_carregar_modelo_cli", lambda cam, pt: {})
    monkeypatch.setattr(pred, "validate_model_package", lambda p: None)
    monkeypatch.setattr(pred, "load_prediction_csv", lambda c: (None, None, pd.DataFrame()))
    monkeypatch.setattr(pred, "predict_samples", lambda p, X, wn: pd.DataFrame({"valor": [1.0, 2.0]}))
    cli.dizer(str(modelo), "s", str(csv), "", "")
    g._menu_prediction(g.Config())
    assert "amostras processadas" in cli.saida()


def test_hsi_sem_configuracao_usa_a_padrao(cli):
    cli.dizer("0")
    g._menu_hsi()


def test_selecao_de_amostras_sobrescreve_quando_confirmado(cli, tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)
    csv = tmp_path / "e.csv"
    pd.DataFrame({"v0": rng.normal(size=12), "v1": rng.normal(size=12), "mae_id": [f"A{i // 3}" for i in range(12)]}).to_csv(csv, index=False)
    saida = tmp_path / "sel.csv"
    saida.write_text("antigo", encoding="utf-8")
    cli.dizer(str(csv), "", "", "1", "", str(saida), "s", "")
    g._menu_selecao_amostras(g.Config())
    assert saida.read_text(encoding="utf-8") != "antigo"


def test_epo_glsw_sem_dataset_carregado_apenas_volta(cli, monkeypatch):
    monkeypatch.setattr(g, "_tecavan_carregar_dataset", lambda cfg, is_pt: None)
    g._menu_tec_epo_glsw(g.Config(), True)


class _ConfigQueRecusaCampo(g.Config):
    """Depois de criada (`travar()`), recusa gravar `max_lvs`, como faria um campo que a versao atual da Config nao aceita."""

    def travar(self):
        object.__setattr__(self, "_travada", True)
        return self

    def __setattr__(self, nome, valor):
        if nome == "max_lvs" and self.__dict__.get("_travada"):
            raise AttributeError("campo somente leitura")
        super().__setattr__(nome, valor)


def test_exemplo_e_perfil_salvo_toleram_campo_que_a_config_nao_aceita(cli, tmp_path, monkeypatch):
    from guaraci import exemplos as ex
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ex, "gerar", lambda cen, pasta: {"pasta": pasta, "arquivos": [], "comando": "c"})
    monkeypatch.setattr(g, "_load_config", lambda cam: g.Config(max_lvs=9))
    cli.dizer("classificar", "", "s", "", "0")
    g._menu_exemplo(_ConfigQueRecusaCampo().travar())
    g._PERFIS_DIR.mkdir(parents=True, exist_ok=True)
    (g._PERFIS_DIR / "p.yaml").write_text("x: 1\n", encoding="utf-8")
    cli.dizer("1", "")
    g._carregar_yaml(_ConfigQueRecusaCampo().travar())
    assert "Carregado" in cli.saida()


def test_perfis_mostra_o_tempo_medido_quando_ha(cli, monkeypatch):
    obj, rig = next(iter(g._cat.TEMPO_PERFIL))
    o = list(g._cat.PERFIS_OBJETIVO).index(obj) + 1
    r = list(g._cat.PERFIS_RIGOR).index(rig) + 1
    cli.dizer(str(o), str(r), "n")
    g._menu_profiles(g.Config())
    assert "Tempo medido" in cli.saida()


def test_execucao_ignora_codigos_do_usuario_ilegiveis_e_aceita_grade_desligada(cli, monkeypatch):
    monkeypatch.setattr(g, "_print_checklist", lambda cfg: True)
    monkeypatch.setattr(g, "_executar", lambda cfg: None)
    monkeypatch.setattr(g, "_carregar_codigos_usuario", _quebra("codigos ilegiveis"))
    g._USER_DIR.mkdir(parents=True, exist_ok=True)
    g._VISUAL_PATH.write_text(json.dumps({"grid_major": False}), encoding="utf-8")
    import matplotlib
    with matplotlib.rc_context():
        cli.dizer("", "s", "")
        g._rodar_pipeline(g.Config())
    assert g._t("exec_concluido").upper() in cli.saida().upper()


def test_main_ignora_idioma_salvo_desconhecido_e_pede_confirmacao_ao_sair_de_uma_tela(cli, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["guaraci"])
    g._USER_DIR.mkdir(parents=True, exist_ok=True)
    g._LANG_FLAG.write_text("XX", encoding="utf-8")

    def edita_e_sai(cfg):
        cfg.max_lvs = cfg.max_lvs + 1
        raise g._SairDoPrograma()

    monkeypatch.setattr(g, "_menu_project", edita_e_sai)
    tecla = next(t for t, tid in g._cat.tela_por_tecla().items() if g._cat.TELAS[tid]["handler"] == "_menu_project")
    cli.dizer(tecla, "", "q", "s")
    g.main([])
    assert g._lang() == "PT" and "ajuste(s) de menu ainda nao gravado(s)" in cli.saida()


def test_executar_o_modulo_como_programa_chama_main(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["guaraci", "version"])
    runpy.run_module("guaraci.guaraci", run_name="__main__")
    assert "GUARACI v" in capsys.readouterr().out


class _SaidaSemReconfigure:
    def write(self, s):
        return len(s)

    def flush(self):
        pass


def test_topo_do_modulo_no_windows_sem_reconfigure_na_saida_nao_tenta_reconfigurar(tmp_path, monkeypatch):
    casa = tmp_path / "casa_reexec2"
    (casa / ".guaraci").mkdir(parents=True)
    (casa / ".guaraci" / ".cli_modo_usuario").write_text("estranho", encoding="utf-8")
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: casa))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "stdout", _SaidaSemReconfigure())
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0))
    mod = runpy.run_module("guaraci.guaraci", run_name="guaraci_reexecutado2")
    assert mod["_STATE"]["modo_usuario"] == "iniciante"


def test_perfis_sem_tempo_medido_para_a_combinacao_nao_mostra_estimativa(cli, monkeypatch):
    monkeypatch.setattr(g._cat, "TEMPO_PERFIL", {})
    cli.dizer("1", "1", "n")
    g._menu_profiles(g.Config())
    assert "Tempo medido" not in cli.saida()
