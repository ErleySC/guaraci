"""Tres entradas ruins (arquivo vazio, coluna ausente, formato errado) por comando principal: mensagem em PT e em EN, sem rastro
tecnico (traceback, nome de excecao do pandas/yaml) e com a acao sugerida."""
from __future__ import annotations

import pytest

from guaraci import anova_cli, monitorar_cli, relatorio_cli, transferir_cli, vip_cli
from guaraci import guaraci as g

OK_CSV = "amostra,grupo,teor,x1,x2\na1,A,1.0,1,2\na2,A,1.2,2,3\na3,B,2.0,3,4\na4,B,2.2,4,5\na5,C,3.0,5,6\na6,C,3.1,6,7\n"
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe\x00\x01"
RASTRO = ("Traceback", "EmptyDataError", "UnicodeDecodeError", "ParserError", "Could not determine delimiter", "pandas", "ScannerError", "ParserError", "yaml.scanner")
ACAO_PT = ("Abra ", "Confira", "Escolha", "Aponte", "Rode ", "corrija", "recomece", "copie")
ACAO_EN = ("Open ", "Check", "Pick", "Set ", "Run ", "fix ", "start again", "copy ")


def _arquivos(d):
    (d / "vazio.csv").write_bytes(b"")
    (d / "so_cabecalho.csv").write_text("amostra,grupo,teor\n", encoding="utf-8")
    (d / "binario.csv").write_bytes(PNG)
    (d / "ok.csv").write_text(OK_CSV, encoding="utf-8")
    (d / "quebrado.yaml").write_text("pasta_dados: [aberto\n  x: : :\n", encoding="utf-8")
    (d / "sem_pasta.yaml").write_text("pasta_dados: nao_existe_aqui\nmodo_entrada: dx\n", encoding="utf-8")
    (d / "pasta_vazia").mkdir()


def _comandos(d, lang):
    p = lambda n: str(d / n)  # noqa: E731
    i = ["--idioma", lang]
    return {
        ("anova", "vazio"): (anova_cli.executar, [p("vazio.csv"), "--resposta", "teor", "--grupo", "grupo", "--unidade", "amostra", *i]),
        ("anova", "coluna"): (anova_cli.executar, [p("ok.csv"), "--resposta", "nao_existe", "--grupo", "grupo", "--unidade", "amostra", *i]),
        ("anova", "formato"): (anova_cli.executar, [p("binario.csv"), "--resposta", "teor", "--grupo", "grupo", "--unidade", "amostra", *i]),
        ("vip", "vazio"): (vip_cli.executar, [p("vazio.csv"), "--classe", "grupo", "--grupo", "amostra", *i]),
        ("vip", "coluna"): (vip_cli.executar, [p("ok.csv"), "--classe", "nao_existe", "--grupo", "amostra", *i]),
        ("vip", "formato"): (vip_cli.executar, [p("binario.csv"), "--classe", "grupo", "--grupo", "amostra", *i]),
        ("monitorar", "vazio"): (monitorar_cli.executar, ["--referencia", p("vazio.csv"), "--novos", p("ok.csv"), "--lote", "grupo", *i]),
        ("monitorar", "coluna"): (monitorar_cli.executar, ["--referencia", p("ok.csv"), "--novos", p("ok.csv"), "--lote", "nao_existe", *i]),
        ("monitorar", "formato"): (monitorar_cli.executar, ["--referencia", p("binario.csv"), "--novos", p("ok.csv"), "--lote", "grupo", *i]),
        ("transferir", "vazio"): (transferir_cli.executar, ["--mestre", p("vazio.csv"), "--escravo", p("ok.csv"), "--novos", p("ok.csv"), *i]),
        ("transferir", "coluna"): (transferir_cli.executar, ["--mestre", p("ok.csv"), "--escravo", p("ok.csv"), "--novos", p("inexistente.csv"), *i]),
        ("transferir", "formato"): (transferir_cli.executar, ["--mestre", p("binario.csv"), "--escravo", p("ok.csv"), "--novos", p("ok.csv"), *i]),
        ("relatorio", "vazio"): (relatorio_cli.executar, [p("pasta_vazia"), *i]),
        ("relatorio", "coluna"): (relatorio_cli.executar, [p("inexistente"), *i]),
        ("relatorio", "formato"): (relatorio_cli.executar, [p("ok.csv"), *i]),
    }


CASOS = [(c, k) for c in ("anova", "vip", "monitorar", "transferir", "relatorio") for k in ("vazio", "coluna", "formato")]


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("cmd, caso", CASOS)
def test_entrada_ruim_recusa_com_mensagem_legivel_e_acao(tmp_path, capsys, cmd, caso, lang):
    _arquivos(tmp_path)
    funcao, args = _comandos(tmp_path, lang)[(cmd, caso)]
    assert funcao(args) == 2
    err = capsys.readouterr().err
    assert err.startswith("Erro:" if lang == "pt" else "Error:"), err
    assert not any(r in err for r in RASTRO), err
    assert any(a in err for a in (ACAO_PT if lang == "pt" else ACAO_EN)), err
    # a mensagem esta no idioma pedido (nao na outra lingua)
    assert (" nao " in err or "nao " in err) if lang == "pt" else ("not " in err or "does not" in err or "could not" in err), err


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_so_com_cabecalho_tambem_e_arquivo_sem_dados(tmp_path, capsys, lang):
    _arquivos(tmp_path)
    assert monitorar_cli.executar(["--referencia", str(tmp_path / "so_cabecalho.csv"), "--novos", str(tmp_path / "ok.csv"), "--lote", "grupo",
                                   "--idioma", lang]) == 2
    err = capsys.readouterr().err
    assert ("vazio" if lang == "pt" else "empty") in err


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("caso, trecho_pt, trecho_en", [
    ("vazio", "pasta de dados", "data folder"),
    ("coluna", "pasta de dados", "data folder"),
    ("formato", "YAML", "YAML"),
])
def test_run_com_config_ruim_recusa_com_acao(tmp_path, monkeypatch, capsys, caso, trecho_pt, trecho_en, lang):
    _arquivos(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GUARACI_IDIOMA", lang)
    arq = {"vazio": "vazio.yaml", "coluna": "sem_pasta.yaml", "formato": "quebrado.yaml"}[caso]
    (tmp_path / "vazio.yaml").write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        g._comando_run(str(tmp_path / arq))
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("Erro:" if lang == "pt" else "Error:"), err
    assert (trecho_pt if lang == "pt" else trecho_en) in err
    assert not any(r in err for r in RASTRO), err
    assert any(a in err for a in (ACAO_PT if lang == "pt" else ACAO_EN)), err


def test_run_sem_config_diz_como_gerar_um(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        g._comando_run(None)
    assert e.value.code == 2
    assert "guaraci exemplo classificar" in capsys.readouterr().err
