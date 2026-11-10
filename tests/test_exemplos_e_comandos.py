"""`guaraci exemplo`, `guaraci --help` derivado do catalogo e subcomandos."""
import ast
import inspect
import os
import tempfile
from pathlib import Path

import pytest

import guaraci.guaraci as g
from guaraci import catalogo as cat
from guaraci import cli_comandos as cmd
from guaraci import exemplos as ex


@pytest.mark.parametrize("cenario", ex.listar())
def test_exemplo_gera_dado_e_config_validos(tmp_path, cenario):
    r = ex.gerar(cenario, str(tmp_path / cenario))
    for a in r["arquivos"]:
        assert (tmp_path / cenario / a).exists()
    if cenario == "hsi":                                                  # pasta de cubos ENVI, uma subpasta por classe
        cfg = g._load_config(str(tmp_path / cenario / "config.yaml"))
        assert cfg.mode == "hsi" and len(list(Path(cfg.hsi_dataset_folder).rglob("*.hdr"))) == 24   # 2 classes x 6 objetos x 2 vistas
    elif cenario == "imagem":                                             # pasta de fotos, uma subpasta por classe/amostra
        cfg = g._load_config(str(tmp_path / cenario / "config.yaml"))
        assert cfg.mode == "imagem" and len(list(Path(cfg.input_folder).rglob("*.png"))) == 36
    elif cenario in ("fluorescencia_eem", "cromatograma_gcms"):           # pasta de arquivos de instrumento, uma subpasta por classe
        cfg = g._load_config(str(tmp_path / cenario / "config.yaml"))
        extensao = {"fluorescencia_eem": "*.dat", "cromatograma_gcms": "*.cdf"}[cenario]
        assert cfg.mode == {"fluorescencia_eem": "eem", "cromatograma_gcms": "gcms"}[cenario]
        assert len(list(Path(cfg.input_folder).rglob(extensao))) == 18  # 3 classes x 6 amostras
    elif cenario not in ("comparar", "monitorar", "transferir", "vip"):  # estes rodam por comando proprio
        cfg = g._load_config(str(tmp_path / cenario / "config.yaml"))     # o config e aceito pelo validador
        assert cfg.csv_file and os.path.isabs(cfg.csv_file)
        assert os.path.isfile(cfg.csv_file)


def test_exemplos_declaram_so_grupos_e_comandos_do_catalogo():
    ids = {x["id"] for x in cat.grupos()}
    for k, v in ex.EXEMPLOS.items():
        assert v["grupo"] in ids, k
        assert all(cat.texto_proibido(f) is None for f in v["faz"]), k     # (PT, EN)


def test_exemplo_nao_grava_dado_real_e_e_reprodutivel(tmp_path):
    ex.gerar("classificar", str(tmp_path / "a"))
    ex.gerar("classificar", str(tmp_path / "b"))
    assert (tmp_path / "a" / "dados.csv").read_bytes() == (tmp_path / "b" / "dados.csv").read_bytes()


def test_exemplo_cenario_desconhecido_e_erro_de_uso(capsys):
    assert cmd.subcomando("exemplo", ["nao_existe"]) == 2
    assert "cenario desconhecido" in capsys.readouterr().err


def test_exemplo_sem_argumento_lista_os_cenarios(capsys):
    assert cmd.subcomando("exemplo", []) == 0
    saida = capsys.readouterr().out
    for k in ex.listar():
        assert k in saida


def test_menu_exemplo_gera_na_pasta_escolhida(monkeypatch, tmp_path):
    monkeypatch.setattr(g, "_cls", lambda: None)
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    destino = str(tmp_path / "ex1")
    respostas = iter(["2", destino, "n", "v"])        # classificar, pasta, nao carregar, voltar
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas, "v"))
    g._menu_exemplo(g.Config())
    assert (Path(destino) / "config.yaml").is_file()


def test_menu_exemplo_carrega_a_configuracao_na_sessao(monkeypatch, tmp_path):
    monkeypatch.setattr(g, "_cls", lambda: None)
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    destino = str(tmp_path / "ex2")
    respostas = iter(["3", destino, "s", "v"])        # quantificar, pasta, carregar, voltar
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas, "v"))
    cfg = g.Config()
    g._menu_exemplo(cfg)
    assert cfg.level == "N3" and cfg.csv_file.endswith("dados.csv")


# ---------------------------------------------------------------- --help
def test_help_lista_todo_comando_do_catalogo_agrupado_pelos_grupos_do_menu():
    texto = cmd.texto_ajuda("PT")
    for c in cat.comandos():
        assert c["uso"].split()[0] in texto, c["id"]
    grupos_com_cmd = [x for x in cat.grupos() if cat.comandos(x["id"])]
    posicoes = [texto.index(x["nome"][0] + ":") for x in grupos_com_cmd]
    assert posicoes == sorted(posicoes)                         # mesma ordem dos grupos do menu
    assert max(len(l) for l in texto.splitlines() if not l.startswith(" " * 20)) <= 100


def test_texto_ajuda_do_guaraci_e_o_derivado_do_catalogo():
    assert g._TEXTO_AJUDA == cmd.texto_ajuda("PT")


def test_todo_comando_do_catalogo_tem_despacho_no_main():
    arvore = ast.parse(inspect.getsource(g.main))
    literais = {n.value for n in ast.walk(arvore) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for c in cat.comandos():
        assert c["id"] in literais or c["id"] in cmd.SUBCOMANDOS, f"comando {c['id']} sem despacho em main()"


def test_help_em_ingles_existe():
    assert "Usage:" in cmd.texto_ajuda("EN")


def test_confianca_cli_lista_e_mostra_pagina(capsys):
    assert cmd.subcomando("confianca", []) == 0
    assert "PLS-DA" in capsys.readouterr().out
    assert cmd.subcomando("confianca", ["pls_da"]) == 0
    saida = capsys.readouterr().out
    assert "Evidencia" in saida and "pytest tests/test_classificadores.py" in saida
    assert "ponto de partida" in saida and "validacao/" not in saida
    assert cmd.subcomando("confianca", ["xyz_inexistente"]) == 2


@pytest.mark.slow
@pytest.mark.parametrize("cenario", ["classificar", "quantificar", "multielementar"])
def test_exemplo_roda_de_ponta_a_ponta_pelo_cli(cenario):
    pasta = tempfile.mkdtemp(prefix="gx")
    try:
        ex.gerar(cenario, pasta)
        g.main(["run", os.path.join(pasta, "config.yaml")])
        achados = list(Path(pasta).rglob("resumo_modelo.txt"))
        assert achados, "a execucao nao gerou resumo_modelo.txt"
        if cenario == "multielementar":
            assert list(Path(pasta).rglob("multielementar.json"))
    finally:
        import shutil
        shutil.rmtree(pasta, ignore_errors=True)


def test_guaraci_manual_esta_no_catalogo_em_pt_e_en_na_ajuda_e_no_despacho(capsys):
    """`guaraci manual` e' comando de primeira classe: catalogo (PT/EN), grupo de ajuda, lista de ajuda nos dois idiomas, despacho e busca."""
    c = next(x for x in cat.comandos() if x["id"] == "manual")
    pt, en = c["faz"]
    assert pt and en and pt != en and c["uso"].startswith("manual") and c["grupo"] == "ajuda"
    assert "manual" in cmd.SUBCOMANDOS
    assert en in " ".join(cmd.texto_ajuda("EN").split()) and pt in " ".join(cmd.texto_ajuda("PT").split())
    assert cmd.subcomando("manual", []) == 0
    assert "Documentos que acompanham o programa" in capsys.readouterr().out
    assert cmd.subcomando("buscar", ["manual"]) == 0
    assert "guaraci manual" in capsys.readouterr().out
