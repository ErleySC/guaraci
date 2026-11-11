"""Ganchos do assistente no fluxo da CLI (assistente/IMPLEMENTACAO.md, "O que fica para as outras frentes").

1. painel de falha com codigo GUA-E (menu [R] e `guaraci run`);
2. pre-voo antes de rodar (menu pergunta com achado critico; `run` para com 1, como `guaraci checar`, salvo --sem-preflight);
3. ficha anti-vazamento em <execucao>/Relatorios depois de rodar e achados de resultado no resumo;
4. "Revisar configuracao" le o registro de regras (assistente_regras);
5. subcomando `guaraci checar` e tela [6.5].
O motor (`_executar`/`pq.executar`) e' trocado por dubles rapidos; regras, ficha e diagnostico sao os de verdade.
"""
from __future__ import annotations

import os

import pytest

import guaraci.dados_io as dio
import guaraci.guaraci as g
from guaraci import assistente_regras, assistente_vazamento, catalogo, cli_comandos
from guaraci.catalogo_dados import MODULOS_FORA_DO_CLI

RESUMO_RUIM = ("Resumo\n"
               "  Accuracy (CV) : 1.0\n"
               "  N unidades independentes : 12\n"
               "  Permutation p-value : 0.30\n")


@pytest.fixture(autouse=True)
def _estado_da_cli_restaurado(monkeypatch):
    """Idioma, nivel e a ajuda da tela aberta (_abrir_tela) sao globais da CLI; nenhum teste
    daqui pode vaza-los para os outros arquivos ("?" passaria a explicar a tela [6.5])."""
    monkeypatch.setattr(g, "_AJUDA_TELA", None)
    antes = dict(g._STATE)
    yield
    g._STATE.clear()
    g._STATE.update(antes)


def _pasta_execucao(tmp_path, texto=RESUMO_RUIM):
    pasta = tmp_path / "execucao"
    (pasta / "Relatorios").mkdir(parents=True)
    (pasta / "Relatorios" / "resumo_modelo.txt").write_text(texto, encoding="utf-8")
    return pasta


def _preparar_menu(monkeypatch, executor):
    monkeypatch.setattr(g, "_print_checklist", lambda cfg: True)
    monkeypatch.setattr(dio, "garantia_agrupamento_prevista", lambda cfg: "high")
    monkeypatch.setattr(g, "_executar", executor)
    monkeypatch.setattr(g.pq, "CODIGO_ESPECIE", {})
    monkeypatch.setattr(g, "_bytes_da_entrada", lambda cfg: None)   # menu: so' regras de configuracao, sem ler disco


# ------------------------------------------------------------------ 1. painel de falha com GUA-E
def test_painel_de_falha_do_menu_mostra_codigo_causa_e_mensagem_original(cli, monkeypatch):
    def falha(cfg):
        raise ValueError("Input X contains NaN.")
    _preparar_menu(monkeypatch, falha)
    cli.dizer("", "s", "")
    g._rodar_pipeline(g.Config())
    saida = cli.saida()
    assert g._t("exec_erro") in saida
    assert "Input X contains NaN." in saida            # a mensagem original continua visivel
    assert "GUA-E001" in saida and "Causa provavel" in saida and "O que fazer" in saida


def test_painel_de_falha_em_ingles_e_interrupcao_sem_diagnostico(cli, monkeypatch):
    def falha(cfg):
        raise ValueError("Input X contains NaN.")
    _preparar_menu(monkeypatch, falha)
    g._STATE["lang"] = "EN"
    cli.dizer("", "s", "")
    g._rodar_pipeline(g.Config())
    assert "Likely cause" in cli.saida() and "GUA-E001" in cli.saida()

    def interrompe(cfg):
        raise KeyboardInterrupt
    monkeypatch.setattr(g, "_executar", interrompe)
    cli.dizer("", "y", "")
    g._rodar_pipeline(g.Config())
    assert "no known code" not in cli.saida()


def test_run_que_falha_mostra_gua_e_no_stderr_e_sai_com_1(tmp_path, monkeypatch, capsys):
    caminho = tmp_path / "config.yaml"
    (tmp_path / "dados").mkdir()
    g.pq.save_config(g.Config(input_folder=str(tmp_path / "dados")), str(caminho))

    def falha(cfg):
        raise FileNotFoundError("CSV nao encontrado: 'x.csv'. Confira o caminho em arquivo_csv (modo csv).")
    monkeypatch.setattr(g.pq, "executar", falha)
    with pytest.raises(SystemExit) as exc:
        g._comando_run(str(caminho))
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "a execucao falhou" in err and "CSV nao encontrado" in err and "GUA-E008" in err


def test_run_com_etapa_principal_que_falhou_sai_com_1(tmp_path, monkeypatch, capsys):
    """A execucao terminou e salvou o resto, mas a etapa do resultado principal falhou:
    `guaraci run` nao pode sair com 0 nem dizer "Concluido" (analise de 2026-11-11, 2.3)."""
    caminho = _config_run(tmp_path, n_permutations=30)
    pasta = _pasta_execucao(tmp_path)

    def executar(cfg):
        cfg.output_folder = str(pasta)
        g.pq._registrar_falha_etapa("regressao por especie (Quantificacao): quebrou")
    monkeypatch.setattr(g.pq, "executar", executar)
    monkeypatch.setattr(g, "_SEM_PREFLIGHT", [False])
    with pytest.raises(SystemExit) as exc:
        g._comando_run(str(caminho))
    assert exc.value.code == 1
    cap = capsys.readouterr()
    assert "INCOMPLETA" in cap.err and "regressao por especie" in cap.err
    assert "Concluido" not in cap.out
    g.pq.resetar_falhas_etapa()


def test_diagnostico_que_falha_ainda_mostra_a_mensagem(monkeypatch):
    import guaraci.assistente_erros as ae
    monkeypatch.setattr(ae, "diagnosticar_excecao", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert g._diagnostico_erro(ValueError("original"), "PT") == "original"


# ------------------------------------------------------------------ 2. pre-voo
@pytest.mark.parametrize("resposta, roda", [("", False), ("n", False), ("s", True)])
def test_menu_pergunta_antes_de_rodar_com_achado_critico_e_o_padrao_e_nao(cli, monkeypatch, resposta, roda):
    rodou = []
    _preparar_menu(monkeypatch, lambda cfg: rodou.append(1))
    cli.dizer("", "s", resposta, "")
    g._rodar_pipeline(g.Config(n_permutations=5))
    saida = cli.saida()
    assert "Pre-voo" in saida and "1/6" in saida and "Rodar mesmo assim" in saida
    assert bool(rodou) is roda


def test_menu_sem_achado_critico_nao_pergunta(cli, monkeypatch):
    rodou = []
    _preparar_menu(monkeypatch, lambda cfg: rodou.append(1))
    cli.dizer("", "s", "")
    g._rodar_pipeline(g.Config())
    assert rodou and "Rodar mesmo assim" not in cli.saida()
    assert "Nenhum problema encontrado" in cli.saida()


def test_preflight_carrega_dados_pequenos_e_aplica_regras_de_dados(monkeypatch):
    cfg = g.Config(mode="sintetico", n_per_class=4, n_synthetic_points=30, wn_min=400.0, wn_max=2001.0)
    achados, dados = g._preflight(cfg, "PT")
    assert isinstance(dados, assistente_regras.ResumoDados) and dados.n_leituras > 0
    # dados acima do limite: so' configuracao (a execucao carrega de novo)
    csv = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    monkeypatch.setattr(g, "_PREFLIGHT_MAX_BYTES", 10)
    assert g._bytes_da_entrada(g.Config(mode="csv", csv_file=csv)) is None
    assert g._bytes_da_entrada(g.Config(mode="hsi")) is None
    assert g._bytes_da_entrada(g.Config(mode="csv", csv_file="/nao/existe.csv")) is None


def test_preflight_que_falha_nao_impede_a_execucao(monkeypatch):
    monkeypatch.setattr(assistente_regras, "avaliar", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("regra quebrada")))
    assert g._preflight(g.Config(), "PT") == ([], None)


def _config_run(tmp_path, **kw):
    caminho = tmp_path / "config.yaml"
    (tmp_path / "dados").mkdir(exist_ok=True)
    g.pq.save_config(g.Config(input_folder=str(tmp_path / "dados"), **kw), str(caminho))
    return caminho


def test_run_para_com_achado_critico_como_guaraci_checar(tmp_path, monkeypatch, capsys):
    caminho = _config_run(tmp_path, n_permutations=5)
    chamado = []
    monkeypatch.setattr(g.pq, "executar", lambda cfg: chamado.append(cfg))
    monkeypatch.setattr(g, "_SEM_PREFLIGHT", [False])
    with pytest.raises(SystemExit) as exc:
        g._comando_run(str(caminho))
    assert exc.value.code == 1 and not chamado
    sai = capsys.readouterr()
    assert "[CRITICO]" in sai.out and "--sem-preflight" in sai.err
    # o mesmo criterio de `guaraci checar`
    assert cli_comandos.subcomando("checar", [str(caminho)]) == 1


def test_run_com_sem_preflight_avisa_e_executa(tmp_path, monkeypatch, capsys):
    caminho = _config_run(tmp_path, n_permutations=5)
    chamado = []
    monkeypatch.setattr(g.pq, "executar", lambda cfg: chamado.append(cfg))
    monkeypatch.setattr(g, "_SEM_PREFLIGHT", [False])
    g.main(["run", str(caminho), "--sem-preflight"])
    assert chamado and g._SEM_PREFLIGHT[0] is True
    assert "--sem-preflight" in capsys.readouterr().err


def test_run_sem_achado_critico_so_avisa_e_executa(tmp_path, monkeypatch, capsys):
    caminho = _config_run(tmp_path, n_permutations=30)
    chamado = []
    monkeypatch.setattr(g.pq, "executar", lambda cfg: chamado.append(cfg))
    monkeypatch.setattr(g, "_SEM_PREFLIGHT", [False])
    g._comando_run(str(caminho))
    out = capsys.readouterr().out
    assert chamado and "Pre-voo" in out and "[AVISO]" in out and "Concluido" in out


# ------------------------------------------------------------------ 3. ficha anti-vazamento depois de rodar
def test_pos_execucao_grava_ficha_em_relatorios_e_devolve_achados_de_resultado(tmp_path):
    pasta = _pasta_execucao(tmp_path)
    cfg = g.Config()
    cfg.output_folder = str(pasta)
    destino, achados = g._assistente_pos_execucao(cfg, None, "PT")
    assert destino == str(pasta / "Relatorios" / "ficha_vazamento.md")
    texto = (pasta / "Relatorios" / "ficha_vazamento.md").read_text(encoding="utf-8")
    assert texto.startswith("# Ficha anti-vazamento") and "[?]" in texto
    ids = {a.id for a in achados}
    assert {"permutacao_nao_significativa", "acuracia_perfeita"} <= ids
    assert all(a.momento == "resultado" for a in achados)


def test_pos_execucao_sem_pasta_nao_faz_nada_e_falha_so_avisa(tmp_path, monkeypatch, capsys):
    assert g._assistente_pos_execucao(g.Config(), None, "PT") is None
    pasta = _pasta_execucao(tmp_path)
    cfg = g.Config()
    cfg.output_folder = str(pasta)
    monkeypatch.setattr(assistente_vazamento, "ficha", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disco cheio")))
    assert g._assistente_pos_execucao(cfg, None, "PT") is None
    assert "ficha anti-vazamento nao gravada: disco cheio" in capsys.readouterr().out


def test_run_grava_a_ficha_e_mostra_achados_de_resultado(tmp_path, monkeypatch, capsys):
    caminho = _config_run(tmp_path, n_permutations=30)
    pasta = _pasta_execucao(tmp_path)

    def executar(cfg):
        cfg.output_folder = str(pasta)
    monkeypatch.setattr(g.pq, "executar", executar)
    monkeypatch.setattr(g, "_SEM_PREFLIGHT", [False])
    g._comando_run(str(caminho))
    out = capsys.readouterr().out
    assert (pasta / "Relatorios" / "ficha_vazamento.md").is_file()
    assert "Leitura do resultado" in out and "p-valor de permutacao = 0.300" in out and "Ficha anti-vazamento" in out
    assert out.index("Ficha anti-vazamento") < out.index("Concluido")


def test_run_com_ficha_quebrada_termina_bem(tmp_path, monkeypatch, capsys):
    caminho = _config_run(tmp_path, n_permutations=30)
    pasta = _pasta_execucao(tmp_path)
    monkeypatch.setattr(g.pq, "executar", lambda cfg: setattr(cfg, "output_folder", str(pasta)))
    monkeypatch.setattr(g, "_SEM_PREFLIGHT", [False])
    monkeypatch.setattr(assistente_vazamento, "ficha", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("quebrou")))
    g._comando_run(str(caminho))       # nao levanta
    out = capsys.readouterr().out
    assert "nao gravada" in out and "Concluido" in out


def test_menu_mostra_ficha_e_achados_de_resultado_depois_de_rodar(cli, monkeypatch, tmp_path):
    pasta = _pasta_execucao(tmp_path)
    _preparar_menu(monkeypatch, lambda cfg: setattr(cfg, "output_folder", str(pasta)))
    cli.dizer("", "s", "")
    g._rodar_pipeline(g.Config())
    saida = cli.saida()
    assert "Leitura do resultado" in saida and "Ficha anti-vazamento" in saida
    assert (pasta / "Relatorios" / "ficha_vazamento.md").is_file()


# ------------------------------------------------------------------ 4. revisar configuracao pelo registro
def test_revisao_le_o_registro_de_regras(cli, monkeypatch):
    falsa = assistente_regras.Regra("regra_de_teste", "config", "aviso", lambda c: True,
                                    ("Mensagem vinda do registro.", "Message from the registry."),
                                    ("causa", "cause"), ("Faca isto.", "Do this."), campo="max_lvs", valor=lambda c: 7)
    monkeypatch.setattr(assistente_regras, "REGRAS", (falsa,))
    g._guaraci_revisar_config(g.Config())
    saida = cli.saida()
    assert "Mensagem vinda do registro." in saida and "Faca isto." in saida and "max_lvs: 7" in saida


def test_revisao_cobre_as_regras_de_configuracao_do_registro(cli):
    cfg = g.Config(wn_min=5000.0, wn_max=4000.0, n_permutations=5, frac_holdout=0.0)
    g._guaraci_revisar_config(cfg)
    saida = cli.saida()
    for a in assistente_regras.avaliar(cfg, momentos=["config"]):
        assert a.mensagem[:40] in saida.replace("\n", " ") or a.mensagem[:20] in saida, a.id


# ------------------------------------------------------------------ 5. guaraci checar e tela [6.5]
def test_checar_e_subcomando_listado_na_ajuda_e_no_catalogo():
    assert "checar" in cli_comandos.SUBCOMANDOS
    assert "checar" in catalogo.COMANDOS and catalogo.COMANDOS["checar"]["grupo"] == "confianca"
    assert "checar CONFIG" in cli_comandos.texto_ajuda("PT") and "--sem-preflight" in cli_comandos.texto_ajuda("PT")
    assert "assistente_checar" not in MODULOS_FORA_DO_CLI
    for lang in (0, 1):
        assert catalogo.texto_proibido(catalogo.COMANDOS["checar"]["faz"][lang]) is None


def test_guaraci_checar_aponta_para_assistente_checar(tmp_path, monkeypatch, capsys):
    import guaraci.assistente_checar as ac
    vistos = []
    monkeypatch.setattr(ac, "main", lambda argv: vistos.append(argv) or 0)
    g.main(["checar", "cfg.yaml", "--dados", "--resultado", "PASTA", "--ficha"])
    assert vistos == [["cfg.yaml", "--dados", "--resultado", "PASTA", "--ficha"]]
    monkeypatch.setattr(ac, "main", lambda argv: 1)
    with pytest.raises(SystemExit) as exc:
        g.main(["checar", "cfg.yaml"])
    assert exc.value.code == 1


def test_guaraci_checar_le_o_resultado_na_pasta_relatorios(tmp_path, capsys):
    caminho = _config_run(tmp_path, n_permutations=30)
    pasta = _pasta_execucao(tmp_path)
    codigo = cli_comandos.subcomando("checar", [str(caminho), "--resultado", str(pasta), "--ficha"])
    out = capsys.readouterr().out
    assert codigo == 1 and "p-valor de permutacao" in out and "# Ficha anti-vazamento" in out


def test_tela_6_5_no_catalogo_e_no_menu(cli, monkeypatch, tmp_path):
    t = catalogo.tela("checar")
    assert t["tecla"] == "6.5" and t["handler"] == "_menu_checar" and callable(getattr(g, "_menu_checar"))
    assert "checar" in [x["id"] for x in catalogo.telas_do_grupo_cli("6")]
    assert not [p for p in catalogo.validar_catalogo() if "checar" in p]
    pasta = _pasta_execucao(tmp_path)
    monkeypatch.setattr(g, "_resumo_dados_assistente", lambda cfg, forcar=False: None)
    cli.dizer("", str(pasta), "")   # Enter = checar agora; depois a pasta da execucao
    g._abrir_tela("checar", g.Config(n_permutations=5))
    saida = cli.saida()
    assert "Pre-voo: configuracao e dados" in saida and "1/6" in saida
    assert "Leitura do resultado" in saida and (pasta / "Relatorios" / "ficha_vazamento.md").is_file()
    assert "guaraci checar CONFIG" in saida
