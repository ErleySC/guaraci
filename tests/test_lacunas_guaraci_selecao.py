"""Tela de selecao de amostras (Kennard-Stone, Duplex, SPXY) de `guaraci.guaraci`, com CSV pequeno de verdade."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import guaraci.guaraci as g


def _csv(tmp_path, nome="espectros.csv", n=12, sep=",", grupos=True, alvo=True, numericas=3):
    rng = np.random.default_rng(0)
    d = {f"v{i}": rng.normal(size=n) for i in range(numericas)}
    if grupos:
        d["mae_id"] = [f"A{i // 3}" for i in range(n)]
    if alvo:
        d["teor"] = rng.uniform(0, 10, size=n)
    caminho = tmp_path / nome
    pd.DataFrame(d).to_csv(caminho, index=False, sep=sep, decimal="," if sep == ";" else ".")
    return str(caminho)


def test_selecao_voltar_e_arquivos_ruins(cli, tmp_path):
    cli.dizer("0")
    g._menu_selecao_amostras(g.Config())
    cli.dizer(str(tmp_path / "nao.csv"), "")
    g._menu_selecao_amostras(g.Config())
    assert "Arquivo nao encontrado" in cli.saida()     # o idioma da tela vale desde que ela abriu
    g._STATE["lang"] = "PT"
    ruim = tmp_path / "vazio.csv"
    ruim.write_bytes(b"")
    cli.dizer(str(ruim), "")
    g._menu_selecao_amostras(g.Config())
    assert "Erro ao ler CSV" in cli.saida()


def test_selecao_recusa_colunas_inexistentes_e_csv_sem_numeros(cli, tmp_path):
    csv = _csv(tmp_path)
    cli.dizer(csv, "coluna_fantasma", "")
    g._menu_selecao_amostras(g.Config())
    cli.dizer(csv, "", "grupo_fantasma", "")
    g._menu_selecao_amostras(g.Config())
    so_texto = tmp_path / "texto.csv"
    pd.DataFrame({"a": list("abcde"), "b": list("vwxyz")}).to_csv(so_texto, index=False)
    cli.dizer(str(so_texto), "", "", "")
    g._menu_selecao_amostras(g.Config())
    saida = cli.saida()
    assert saida.count("Coluna nao encontrada") == 2 and "sem colunas numericas" in saida


def test_selecao_recusa_metodo_e_fracao_invalidos(cli, tmp_path):
    csv = _csv(tmp_path)
    for respostas in ((csv, "", "", "9", ""), (csv, "", "", "1", "abc", ""), (csv, "", "", "1", "1.5", "")):
        cli.dizer(*respostas)
        g._menu_selecao_amostras(g.Config())
    saida = cli.saida()
    for trecho in ("Opcao invalida", "Numero invalido", "entre 0 e 1"):
        assert trecho in saida, trecho


@pytest.mark.parametrize("metodo, alvo, sep", [("1", "", ","), ("2", "", ";"), ("3", "teor", ",")])
def test_selecao_particiona_e_grava_o_csv_com_a_coluna_do_conjunto(cli, tmp_path, metodo, alvo, sep):
    csv = _csv(tmp_path, sep=sep)
    saida = tmp_path / f"sel_{metodo}.csv"
    cli.dizer(csv, alvo, "", metodo, "0.6", str(saida), "")
    g._menu_selecao_amostras(g.Config())
    df = pd.read_csv(saida)
    assert set(df["conjunto"]) == {"calibracao", "validacao"}
    assert "Selecao gerada" in cli.saida()


def test_selecao_sem_grupos_suficientes_avisa_e_usa_a_pasta_atual(cli, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = _csv(tmp_path, grupos=False, alvo=False)
    cli.dizer(csv, "", "", "1", "", "")
    g._menu_selecao_amostras(g.Config())
    assert (tmp_path / "selecao_amostras.csv").exists() and "conjunto" in pd.read_csv(tmp_path / "selecao_amostras.csv").columns


def test_selecao_pede_confirmacao_para_sobrescrever_e_avisa_erro_de_gravacao(cli, tmp_path, monkeypatch):
    csv = _csv(tmp_path)
    saida = tmp_path / "sel.csv"
    saida.write_text("antigo", encoding="utf-8")
    cli.dizer(csv, "", "", "1", "", str(saida), "n", "")
    g._menu_selecao_amostras(g.Config())
    assert saida.read_text(encoding="utf-8") == "antigo"
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, *a, **k: (_ for _ in ()).throw(OSError("destino bloqueado")))
    cli.dizer(csv, "", "", "1", "", str(tmp_path / "novo.csv"), "")
    g._menu_selecao_amostras(g.Config())
    assert "destino bloqueado" in cli.saida()


def test_leitor_de_csv_detecta_ponto_e_virgula_tabulacao_e_virgula(tmp_path):
    for sep, nome in ((";", "a.csv"), ("\t", "b.csv"), (",", "c.csv")):
        p = tmp_path / nome
        p.write_text(sep.join(["x", "y"]) + "\n" + sep.join(["1", "2"]) + "\n", encoding="utf-8")
        assert list(g._ler_csv_detectando_separador(str(p)).columns) == ["x", "y"]
