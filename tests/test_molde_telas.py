"""Contrato do molde unico de tela: trilha, o que faz, rodape padrao e teclas comuns."""
import ast
import inspect

import pytest

import guaraci.guaraci as g
from guaraci import catalogo as cat

TELAS_MENU = [t for t in cat.telas() if t["handler"].startswith("_menu_")]


@pytest.fixture(autouse=True)
def _sem_tela_limpa(monkeypatch, tmp_path):
    monkeypatch.setattr(g, "_cls", lambda: None)
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    # Estado do usuario (modo e idioma) NUNCA no ~/.guaraci real: redireciona e restaura.
    monkeypatch.setattr(g, "_MODO_FLAG", tmp_path / ".cli_modo_usuario")
    monkeypatch.setattr(g, "_LANG_FLAG", tmp_path / ".cli_lang")
    antes = dict(g._STATE)
    yield
    g._STATE.clear()
    g._STATE.update(antes)


def _entradas(monkeypatch, *respostas):
    it = iter(respostas)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(it, "0"))


def _chama(fn, cfg):
    with g.console.capture() as cap:
        fn(cfg)
    return cap.get()


def _alcanca(nome, vistos=None):
    """Nomes de funcoes do modulo chamadas (transitivamente) por `nome`."""
    vistos = vistos if vistos is not None else set()
    if nome in vistos or not hasattr(g, nome):
        return vistos
    vistos.add(nome)
    try:
        arvore = ast.parse(inspect.getsource(getattr(g, nome)))
    except (OSError, TypeError, SyntaxError):
        return vistos
    for n in ast.walk(arvore):
        if isinstance(n, ast.Name) and n.id.startswith("_") and hasattr(g, n.id):
            _alcanca(n.id, vistos)
    return vistos


@pytest.mark.parametrize("tela", TELAS_MENU, ids=[t["id"] for t in TELAS_MENU])
def test_toda_tela_do_catalogo_tem_handler_que_usa_o_molde(tela):
    assert hasattr(g, tela["handler"]), tela["handler"]
    alcance = _alcanca(tela["handler"])
    assert alcance & {"_loop_menu", "_cabecalho_tela"}, (
        f"{tela['handler']} nao usa o molde (_loop_menu/_cabecalho_tela)")


@pytest.mark.parametrize("tela", TELAS_MENU, ids=[t["id"] for t in TELAS_MENU])
def test_tela_mostra_trilha_faz_e_rodape_padrao(monkeypatch, tela):
    _entradas(monkeypatch, "v", "v", "v")
    fn = getattr(g, tela["handler"])
    saida = _chama(lambda c: fn(c), g.Config())
    assert cat.trilha(tela["id"]) in saida
    assert tela["faz"][0][:40] in saida
    assert cat.RODAPE_TELA["PT"].split("|")[0].strip() in saida     # "? ajuda"
    assert "V voltar" in saida and "G Guaraci" in saida and "I idioma" in saida


@pytest.mark.parametrize("tela", [t for t in TELAS_MENU if t["tipo"] == "config"], ids=lambda t: t["id"])
def test_teclas_comuns_funcionam_nas_telas_de_configuracao(monkeypatch, tela):
    fn = getattr(g, tela["handler"])
    _entradas(monkeypatch, "c", "", "v")                      # confianca, volta, sai da tela
    assert "confianca" in _chama(fn, g.Config()).lower()
    _entradas(monkeypatch, "?", "", "v")                      # ajuda da tela
    saida = _chama(fn, g.Config())
    assert tela["nome"][0] in saida
    _entradas(monkeypatch, "s", "v")                          # [s] nao encerra o programa em tela interna (so' [q] no menu principal)
    _chama(fn, g.Config())


def test_mostrar_avancadas_usa_mais_e_nao_v(monkeypatch):
    cfg = g.Config()
    _entradas(monkeypatch, "+", "v")
    g._set_modo_usuario("iniciante")
    saida = _chama(g._menu_data, cfg)
    assert "[+] Mostrar opcoes avancadas" in saida
    assert "Elementos (ICP)" in saida.split("[+]", 1)[1]      # depois do "+", as pastas de avancados aparecem


def test_menu_principal_mostra_so_os_grupos_e_as_telas_ficam_nas_pastas(monkeypatch):
    g._set_modo_usuario("iniciante")
    with g.console.capture() as cap:
        g._print_main_menu()
    inicial = cap.get()
    for grupo in cat.grupos_cli():
        assert f"[{grupo['num']}]" in inicial and grupo["nome"][0] in inicial
    assert "Monitorar lotes" not in inicial and "Auditoria de delineamento" not in inicial
    assert cat.RODAPE_MENU["PT"].split("|")[0].strip() in inicial
    for modo in ("iniciante", "avancado"):
        g._set_modo_usuario(modo)
        for grupo in cat.grupos_cli():
            _entradas(monkeypatch, "v")
            with g.console.capture() as cap:
                g._menu_grupo(grupo["num"], g.Config())
            pasta = cap.get()
            for t in cat.telas_do_grupo_cli(grupo["num"]):
                visivel = modo == "avancado" or t["nivel"] == "basico"
                assert (t["nome"][0] in pasta) == visivel, (modo, t["id"])


def test_sobre_do_menu_nao_colide_com_assistente():
    assert cat.tela("sobre")["tecla"] == "6.3"
    assert "A" not in cat.tela_por_tecla()                    # "a" e' o assistente em toda tela
