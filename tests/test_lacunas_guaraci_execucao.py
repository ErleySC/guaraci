"""Execucao do pipeline pela tela [R] de `guaraci.guaraci`: resumo, painel ao vivo, confirmacoes e erros.

O motor (`_executar`) e' trocado por uma funcao rapida que escreve no log como o real; o resto da tela e' o de verdade.
"""
from __future__ import annotations

import matplotlib
import pytest

import guaraci.dados_io as dio
import guaraci.guaraci as g


@pytest.fixture(autouse=True)
def _rcparams_restaurado():
    with matplotlib.rc_context():
        yield


def _executor(texto="[1/7] Carregando\n  [AVISO] algo estranho\n", erro=None):
    def f(cfg):
        print(texto)
        if erro is not None:
            raise erro
    return f


def _preparar(monkeypatch, *, pode=True, garantia="high", executor=None):
    monkeypatch.setattr(g, "_print_checklist", lambda cfg: pode)
    monkeypatch.setattr(dio, "garantia_agrupamento_prevista", lambda cfg: garantia)
    monkeypatch.setattr(g, "_executar", executor or _executor())
    monkeypatch.setattr(g.pq, "CODIGO_ESPECIE", {})


def test_resumo_mostra_booleanos_marcacoes_e_garantia_de_agrupamento(cli, monkeypatch):
    for garantia in ("high", "medium", "verificar", "none"):
        monkeypatch.setattr(dio, "garantia_agrupamento_prevista", lambda cfg, g_=garantia: g_)
        g._print_resumo(g.Config(tag="rodada"))
    g._STATE["lang"] = "EN"
    g._print_resumo(g.Config(run_benchmark=True))
    saida = cli.saida()
    for trecho in ("GroupKFold (mae_id)", "StratifiedKFold", "rodada", "Yes", "No"):
        assert trecho in saida, trecho


def test_painel_de_execucao_mostra_progresso_figuras_e_so_os_ultimos_avisos(cli):
    log = "\n".join(["[1/7] Carregando dados", "  [AVISO] a", "  [AVISO] b", "  [AVISO] c", "  [AVISO] d", "  [AVISO] e",
                     "  Salvo: figura_1.png", "[3/7] Modelo"])
    painel = g._montar_painel_execucao(log, 120.0, "Classificar", ["f1", "f2"])
    g.console.print(painel)
    assert "Classificar" in cli.saida()
    vazio = g._montar_painel_execucao("", 1.0, "Classificar", [])
    g.console.print(vazio)


def test_execucao_recusada_quando_o_checklist_bloqueia(cli, monkeypatch):
    _preparar(monkeypatch, pode=False)
    cli.dizer("")
    g._rodar_pipeline(g.Config())
    assert g._t("pip_sem_dados") in cli.saida()


def test_execucao_cancelada_na_pergunta_final(cli, monkeypatch):
    _preparar(monkeypatch)
    cli.dizer("", "n", "")
    g._rodar_pipeline(g.Config())
    assert g._t("cancelado") in cli.saida()


@pytest.mark.parametrize("resposta, roda", [("n", False), ("s", True)])
def test_execucao_sem_chave_de_grupo_pede_confirmacao_extra(cli, monkeypatch, resposta, roda):
    rodou = []
    _preparar(monkeypatch, garantia="none", executor=lambda cfg: rodou.append(1))
    cli.dizer("", "s", resposta, "")
    g._rodar_pipeline(g.Config())
    assert bool(rodou) is roda


def test_execucao_completa_da_o_nome_grava_a_config_e_mostra_a_conclusao(cli, monkeypatch):
    _preparar(monkeypatch)
    g._USER_DIR.mkdir(parents=True, exist_ok=True)
    g._CODIGOS_PATH.write_text('{"ZZ": "Especie Z"}', encoding="utf-8")
    cfg = g.Config(run_benchmark=True)
    cli.dizer("minha rodada!", "s", "")
    g._rodar_pipeline(cfg)
    assert cfg.tag == "minha_rodada_" and g._CFG_PATH.exists() and g.pq.CODIGO_ESPECIE == {"ZZ": "Especie Z"}
    assert g._t("exec_concluido").upper() in cli.saida().upper()


def test_execucao_limpa_o_nome_com_interrogacao_e_tolera_visual_e_gravacao_que_falham(cli, monkeypatch):
    _preparar(monkeypatch)
    monkeypatch.setattr(g, "_aplicar_paleta", lambda *a: (_ for _ in ()).throw(RuntimeError("paleta quebrada")))
    monkeypatch.setattr(g, "_save_config", lambda *a: (_ for _ in ()).throw(OSError("somente leitura")))
    cfg = g.Config(tag="antigo")
    cli.dizer("-", "s", "")
    g._rodar_pipeline(cfg)
    assert cfg.tag == "" and "somente leitura" in cli.saida()
    g._STATE["lang"] = "EN"
    cli.dizer("", "y", "")
    g._rodar_pipeline(cfg)
    assert "could not be saved" in cli.saida()


@pytest.mark.parametrize("erro, trecho", [(RuntimeError("explodiu"), "explodiu"), (KeyboardInterrupt(), None)])
def test_execucao_que_falha_ou_e_interrompida_mostra_o_painel_de_erro(cli, monkeypatch, erro, trecho):
    _preparar(monkeypatch, executor=_executor(erro=erro))
    cli.dizer("", "s", "")
    g._rodar_pipeline(g.Config())
    assert g._t("exec_erro") in cli.saida() and (trecho is None or trecho in cli.saida())
