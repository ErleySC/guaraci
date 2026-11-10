"""Partida da CLI sem processos filhos nem varreduras repetidas; tela 5.1 com rotulos inteiros;
nenhuma palavra de interface em portugues quando o idioma e' ingles (2026-10-16)."""
import io
import os

import pytest

import guaraci.guaraci as g
from guaraci import catalogo as cat
from guaraci import cli_logic, confianca


@pytest.fixture(autouse=True)
def _cli(monkeypatch, tmp_path):
    monkeypatch.setattr(g, "_cls", lambda: None)
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    monkeypatch.setattr(g, "_LANG_FLAG", tmp_path / ".cli_lang")
    monkeypatch.setattr(g, "_MODO_FLAG", tmp_path / ".cli_modo_usuario")
    monkeypatch.setattr(g, "_AJUDA_TELA", None)
    antes = dict(g._STATE)
    g._STATE["modo_usuario"] = "iniciante"
    yield
    g._STATE.clear()
    g._STATE.update(antes)


def _tela(tid, lang, largura=100):
    from rich.console import Console
    g._STATE["lang"] = lang
    buf = io.StringIO()
    velho = g.console
    g.console = Console(file=buf, width=largura, force_terminal=False)
    it = iter(["", "", ""])
    import builtins
    original = builtins.input
    builtins.input = lambda *a, **k: next(it, "")
    try:
        g._abrir_tela(tid, g.Config())
    finally:
        builtins.input = original
        g.console = velho
    return buf.getvalue()


def test_contagem_de_dx_e_guardada_e_refeita_quando_a_pasta_muda(tmp_path):
    cli_logic._CACHE_DX.clear()
    (tmp_path / "a.dx").write_text("x")
    assert cli_logic.count_dx(str(tmp_path)) == 1
    (tmp_path / "b.dx").write_text("x")
    os.utime(tmp_path, ns=(1, 1))               # a pasta mudou: mtime novo invalida a contagem guardada
    assert cli_logic.count_dx(str(tmp_path)) == 2
    visto = cli_logic._CACHE_DX[os.path.abspath(str(tmp_path))]
    assert cli_logic.count_dx(str(tmp_path)) == 2 and cli_logic._CACHE_DX[os.path.abspath(str(tmp_path))] == visto


def test_contagem_de_dx_em_subpastas_e_pasta_inexistente(tmp_path):
    cli_logic._CACHE_DX.clear()
    (tmp_path / "classe1").mkdir()
    (tmp_path / "classe1" / "a.DX").write_text("x")
    assert cli_logic.count_dx(str(tmp_path)) == 1
    assert cli_logic.count_dx(str(tmp_path / "nao_existe")) == 0


def test_limpar_a_tela_no_windows_nao_abre_cmd(monkeypatch):
    chamadas = []
    monkeypatch.undo()                          # a fixture trocou _cls; aqui interessa o original
    monkeypatch.setattr(g.os, "name", "nt")
    monkeypatch.setattr(g.os, "system", lambda c: chamadas.append(c))
    limpou = []
    monkeypatch.setattr(g.console, "clear", lambda *a, **k: limpou.append(1))
    monkeypatch.delenv("GUARACI_MODO_LINEAR", raising=False)
    g._cls()
    assert chamadas == [] and limpou == [1]


@pytest.mark.parametrize("largura", [80, 100])
def test_rotulo_transparencia_dos_pontos_fica_inteiro_numa_linha(largura):
    saida = _tela("visualizacao", "PT", largura)
    linha = next(ln for ln in saida.splitlines() if "[11]" in ln)
    assert "Transparencia dos Pontos" in linha
    saida_en = _tela("visualizacao", "EN", largura)
    assert any("Point Transparency" in ln and "[11]" in ln for ln in saida_en.splitlines())


def test_todo_rotulo_de_acao_das_telas_fica_na_coluna_do_nome():
    """Rotulo sem '(detalhe)' nao e' repartido entre as colunas Nome e Descricao (cores diferentes)."""
    for lang in ("PT", "EN"):
        saida = _tela("visualizacao", lang, 80)
        for ln in saida.splitlines():
            if "[8]" in ln or "[9]" in ln or "[10]" in ln or "[11]" in ln:
                assert "  Pontos" not in ln.replace("Transparencia dos Pontos", "")


PROIBIDAS_EM_INGLES = ("Analitico", "Avancado", "Classificacao", " padrao ", "Nivel", "pasta nao", "Chamado de",
                       "Ola, pesquisador", "Modulos pesados")


@pytest.mark.parametrize("tid", ["visualizacao", "preproc", "avancado", "auditoria", "assistente", "confianca", "dados"])
def test_tela_em_ingles_nao_mostra_palavras_de_interface_em_portugues(tid):
    saida = _tela(tid, "EN", 110)
    achadas = [p for p in PROIBIDAS_EM_INGLES if p in saida]
    assert not achadas, (tid, achadas)


def test_status_e_menu_principal_em_ingles():
    from rich.console import Console
    g._STATE["lang"] = "EN"
    buf = io.StringIO()
    velho = g.console
    g.console = Console(file=buf, width=110, force_terminal=False)
    try:
        g._print_status(g.Config())
    finally:
        g.console = velho
    saida = buf.getvalue()
    assert "Level: Species" in saida and "Classificacao" not in saida


def test_confianca_em_ingles_traduz_nomes_de_tecnicas_e_opcoes():
    pt = confianca.lista(None, "PT", False)
    en = confianca.lista(None, "EN", False)
    assert "Perfil de tecnica" in pt and "Technique profile" in en and "Perfil de tecnica" not in en
    assert "Modelo unico x modelo por especie" in pt and "Pooled model vs. per-species model" in en
    assert {t["id"] for t in cat.tecnicas("EN")} == {t["id"] for t in cat.tecnicas("PT")}


def test_mensagem_de_validacao_da_pasta_vai_para_ingles_so_na_exibicao():
    g._STATE["lang"] = "EN"
    assert g._msg_validacao("pasta nao encontrada: 'x' (confira o caminho)") == "folder not found: 'x' (check the path)"
    assert g._msg_validacao("mensagem desconhecida") == "mensagem desconhecida"
    g._STATE["lang"] = "PT"
    assert g._msg_validacao("pasta nao encontrada: 'x' (confira o caminho)").startswith("pasta nao")


def test_valores_de_opcao_em_ingles_so_na_exibicao():
    g._STATE["lang"] = "EN"
    assert g._valor_para_exibir("padrao") == "default" and g._valor_para_exibir("png") == "png"
    g._STATE["lang"] = "PT"
    assert g._valor_para_exibir("padrao") == "padrao"
