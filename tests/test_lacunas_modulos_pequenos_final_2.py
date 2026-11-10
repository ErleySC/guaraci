"""Mais bordas pequenas: sessoes do modo servidor, politica da Config, relatorio por blocos, registro de modelos, manual e exemplos."""
from __future__ import annotations

import os
import re
import types

import pytest

import guaraci.contexto as cx
import guaraci.docs_pacote as dp
import guaraci.exemplos as ex
import guaraci.model_registry as mr
import guaraci.relatorio_blocos as rb
import guaraci.web_seguranca as ws
from guaraci.config import Config


@pytest.fixture
def sessoes(tmp_path, monkeypatch):
    monkeypatch.setenv(ws.VAR_RAIZ_SESSOES, str(tmp_path / "sessoes"))
    return tmp_path / "sessoes"


def test_apagar_sessao_com_id_malformado_nao_levanta_e_a_guarda_apaga_ao_encerrar(sessoes):
    ws.apagar_sessao("id com caracteres invalidos!")
    sid = ws.nova_sessao()
    pasta = ws.pasta_sessao(sid, "uploads")
    assert pasta.is_dir()
    guarda = ws.GuardaSessao(sid)
    guarda.encerrar()
    assert not ws.pasta_sessao(sid, criar=False).exists()


def test_limpeza_de_sessoes_antigas_tolera_arquivo_que_some_no_meio(sessoes, monkeypatch):
    sid = ws.nova_sessao()
    ws.salvar_upload(sid, "a.csv", b"x")
    real_stat = os.stat

    def stat_falho(caminho, *a, **k):
        if str(caminho).endswith("a.csv"):
            raise OSError("sumiu")
        return real_stat(caminho, *a, **k)

    monkeypatch.setattr(os, "stat", stat_falho)
    assert ws._ultima_modificacao(ws.pasta_sessao(sid, criar=False)) > 0
    monkeypatch.undo()
    monkeypatch.setenv(ws.VAR_RAIZ_SESSOES, str(sessoes))
    monkeypatch.setattr(ws, "_ultima_modificacao", lambda p: (_ for _ in ()).throw(OSError("sem permissao")))
    assert ws.limpar_sessoes_antigas(max_horas=0.0, agora=1e12) == 0


def test_mensagem_ao_visitante_em_ingles_e_politica_sem_arquivo_de_grupo_fora_da_sessao(sessoes, monkeypatch):
    msg, eid = ws.mensagem_visitante(RuntimeError("falhou aqui"), "leitura", pt=False)
    assert "Could not complete: leitura" in msg and eid in msg
    sid = ws.nova_sessao()
    cfg = types.SimpleNamespace(mode="sintetico", csv_file="", group_file="/fora/da/sessao/grupos.csv", input_folder="x")
    ws.impor_politica_servidor(cfg, sid)
    assert cfg.group_file is None and cfg.input_folder == "" and not hasattr(cfg, "hsi_dataset_folder")
    completa = Config(mode="hsi")
    completa.hsi_dataset_folder = "/qualquer"
    ws.impor_politica_servidor(completa, sid)
    assert completa.hsi_dataset_folder == "" and completa.mode == "sintetico"


def test_rotulo_da_pasta_no_modo_local_e_o_caminho_absoluto(monkeypatch, tmp_path):
    monkeypatch.setenv(ws.VAR_MODO_LOCAL, "1")
    assert ws.rotulo_pasta(tmp_path) == os.path.abspath(str(tmp_path))
    monkeypatch.delenv(ws.VAR_MODO_LOCAL)
    assert "isolada" in ws.rotulo_pasta(tmp_path)


def test_relatorio_por_blocos_escapa_barra_vertical_e_html_nas_tabelas():
    blocos = [("h1", "Titulo"), ("p", "texto <b>"), ("img", "f.png", "alt"), ("table", ["a", "b"], [["x|y", "<z>"]])]
    md, html = rb.renderizar_md(blocos), rb.renderizar_html(blocos)
    tex = rb.renderizar_tex(blocos)
    assert "x\\|y" in md and "&lt;z&gt;" in html and "&lt;b&gt;" in html and "\\begin" in tex


def test_registro_de_modelos_pula_construtor_que_nao_devolve_nada_nem_motivo(monkeypatch):
    monkeypatch.setattr(mr, "_REGISTRO", [("sem_nada", lambda n, cfg: None, False)])
    assert mr.build_benchmark_list(3, Config()) == []


def test_manual_devolve_a_ultima_secao_ate_o_fim_do_texto():
    linhas = dp.ler("manual").splitlines()
    numeros = [m.group(2) for ln in linhas if (m := dp._TITULO.match(ln))]
    ultimo = numeros[-1]
    texto = dp.secao_do_manual(ultimo)
    assert texto is not None and texto.startswith(next(ln for ln in linhas if dp._TITULO.match(ln) and dp._TITULO.match(ln).group(2) == ultimo))
    assert dp.secao_do_manual("999.999") is None
    assert re.search(r"guaraci manual", dp.lista())


def test_exemplo_desconhecido_e_campo_de_configuracao_inexistente(tmp_path):
    with pytest.raises(ValueError, match="cenario desconhecido"):
        ex.gerar("cenario_que_nao_existe", str(tmp_path))
    with pytest.raises(KeyError):
        cx._attr("campo_que_nao_existe")


_MANUAL_FALSO = "\n".join([
    "# Manual",
    "## 1 Primeira secao",
    "texto 1",
    "### 1.1 Subsecao",
    "texto 1.1",
    "#### 1.1.1 Mais fundo",
    "texto 1.1.1",
    "## 2 Segunda secao",
    "texto 2",
    "# Apendice sem numero",
    "texto do apendice",
    "## 3 Secao vazia",
    "## 4 Ultima",
    "texto 4",
])


def test_secao_do_manual_vai_ate_o_proximo_titulo_de_nivel_igual_ou_maior(monkeypatch):
    monkeypatch.setattr(dp, "ler", lambda topico: _MANUAL_FALSO)
    assert dp.secao_do_manual("1") == "## 1 Primeira secao\ntexto 1\n### 1.1 Subsecao\ntexto 1.1\n#### 1.1.1 Mais fundo\ntexto 1.1.1\n"
    # a subsecao de nivel 3 leva o titulo de nivel 4 e para no proximo nivel 2
    assert dp.secao_do_manual("1.1") == "### 1.1 Subsecao\ntexto 1.1\n#### 1.1.1 Mais fundo\ntexto 1.1.1\n"
    assert dp.secao_do_manual("1.1.1") == "#### 1.1.1 Mais fundo\ntexto 1.1.1\n"
    # titulo de nivel 1 sem numero (so' o regex generico o reconhece) tambem encerra a secao
    assert dp.secao_do_manual("2") == "## 2 Segunda secao\ntexto 2\n"
    # secao sem corpo: o titulo seguinte, logo na linha de baixo, ja a encerra
    assert dp.secao_do_manual("3") == "## 3 Secao vazia\n"
    assert dp.secao_do_manual("4") == "## 4 Ultima\ntexto 4\n"                  # a ultima vai ate o fim do texto
