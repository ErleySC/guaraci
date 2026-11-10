"""Telas de terminal de `guaraci.guaraci` percorridas com digitacao scriptada (fixture `cli`): codificacao de especies.

Cada teste dita o que a pessoa digitaria e confere o que a tela entrega (arquivo gravado, aviso na tela), nao o desenho.
"""
from __future__ import annotations

import json

import guaraci.guaraci as g


def test_codificacao_lista_adiciona_recusa_codigo_ruim_e_grava_o_codigo_do_usuario(cli):
    cli.dizer("1", "", "2", "zz", "Especie Z", "", "2", "1", "", "x", "", "0")
    g._menu_encoding(g.Config())
    saida = cli.saida()
    assert "Codigos cadastrados" in saida and "Especie Z" in saida
    assert json.loads(g._CODIGOS_PATH.read_text(encoding="utf-8")) == {"ZZ": "Especie Z"}


def test_codificacao_importa_csv_com_ponto_e_virgula_ignora_cabecalho_e_linhas_ruins(cli, tmp_path):
    csv = tmp_path / "cod.csv"
    csv.write_text("codigo;especie\nab;Especie A\nxyz;\n1;Numerico\nsolo\ncde;Especie CDE\n", encoding="utf-8")
    cli.dizer("3", str(csv), "", "3", "", "3", str(tmp_path / "nao_existe.csv"), "", "4", "", "0")
    g._menu_encoding(g.Config())
    saida = cli.saida()
    assert "2 codigo(s) importado(s)" in saida and "Arquivo nao encontrado" in saida and "Exportado para" in saida
    assert json.loads(g._CODIGOS_PATH.read_text(encoding="utf-8")) == {"AB": "Especie A", "CDE": "Especie CDE"}
    assert (g._USER_DIR / "codigos_exportados.csv").read_text(encoding="utf-8-sig").splitlines()[0] == "codigo,especie,origem"


def test_codificacao_importa_csv_ilegivel_e_mostra_o_erro(cli, tmp_path):
    csv = tmp_path / "ruim.csv"
    csv.write_bytes(b"\xff\xfe\x00\x80 isto nao e utf-8 \x81\x82\n")
    cli.dizer("3", str(csv), "", "0")
    g._menu_encoding(g.Config())
    assert "codec" in cli.saida().lower() or "decode" in cli.saida().lower()


def test_codificacao_em_ingles_troca_o_idioma_e_tolera_arquivo_de_codigos_corrompido(cli):
    g._USER_DIR.mkdir(parents=True, exist_ok=True)
    g._CODIGOS_PATH.write_text("{ isto nao e json", encoding="utf-8")
    cli.dizer("I", "1", "", "3", "", "0")
    g._menu_encoding(g.Config())
    assert "Registered codes" in cli.saida() and "Import from CSV" in cli.saida()


def test_codificacao_avisa_quando_nao_consegue_gravar_ou_exportar(cli, tmp_path, monkeypatch):
    bloqueio = tmp_path / "arquivo_no_lugar_da_pasta"
    bloqueio.write_text("x", encoding="utf-8")
    monkeypatch.setattr(g, "_USER_DIR", bloqueio)
    monkeypatch.setattr(g, "_CODIGOS_PATH", bloqueio / "codigos.json")
    cli.dizer("2", "ab", "Especie AB", "", "4", "", "0")
    g._menu_encoding(g.Config())
    assert "Falha ao salvar codigos" in cli.saida()
