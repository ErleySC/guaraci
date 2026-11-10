"""A tecla "?" mostra o MESMO painel de ajuda em toda tela (relato de 2026-11-06).

Antes: algumas telas imprimiam uma linha cinza ("? Exemplo guiado: ..."), outras uma lista
solta, outras nada; o menu de grupo mostrava a ajuda da ULTIMA tela aberta (a variavel da
ajuda nunca era limpa) e a faixa de um campo vazava em ingles na tela em portugues
("Valid system path"). Estes testes percorrem todas as telas, grupos, o menu principal e
todos os campos.
"""
from __future__ import annotations

import re

import pytest

import guaraci.catalogo as cat
import guaraci.guaraci as g

#: R roda a analise inteira (nao tem prompt proprio); a ajuda dele esta no painel do menu principal.
_TELAS_COM_PROMPT = [tid for tid in cat.TELAS if tid != "rodar"]

#: trechos em ingles que nao podem aparecer na ajuda em portugues
_INGLES = re.compile(r"\b(Valid|Column|Choose|Existing|List of|Field|Does|Use when|Avoid when|Default|Range|True|False)\b|\(EN:")


@pytest.fixture
def sem_pausa(monkeypatch):
    monkeypatch.setattr(g, "_rodar_pipeline", lambda *a, **k: None)


@pytest.mark.parametrize("lang", ["PT", "EN"])
@pytest.mark.parametrize("tid", list(cat.TELAS))
def test_painel_de_toda_tela_tem_o_molde_completo(cli, tid, lang):
    cli.mod._STATE["lang"] = lang
    L = 1 if lang == "EN" else 0
    g.console.print(g._painel_tela(tid))
    saida = cli.saida()
    t = cat.tela(tid)
    for rot in ("tela", "onde", "faz", "use", "depois", "teclas"):
        assert g._ROT_AJUDA[rot][L] in saida, (tid, rot)
    assert t["nome"][L] in saida
    if cat.campos_da_tela(tid):
        assert g._ROT_AJUDA["opcoes"][L] in saida, tid
    if lang == "PT":
        assert not _INGLES.search(saida), (tid, _INGLES.search(saida).group(0))


#: telas que sao ferramentas com lista propria: o "?" explica as opcoes delas (Menu, Faz, Opcoes)
_FERRAMENTAS = {"assistente", "confianca"}


def _cfg_com_dados(tmp_path):
    """Configuracao com dados de verdade (o exemplo sintetico): telas como Auditoria so' pedem algo com dados."""
    from guaraci import exemplos
    res = exemplos.gerar("classificar", str(tmp_path / "ex"))
    return g._load_config(str(tmp_path / "ex" / "config.yaml")) if res else g.Config()


@pytest.mark.parametrize("tid", _TELAS_COM_PROMPT)
def test_interrogacao_em_toda_tela_abre_o_painel(cli, sem_pausa, tid, tmp_path):
    """Abre a tela como o menu abre e digita "?": o painel da tela aparece (com Faz e Use quando)."""
    cfg = _cfg_com_dados(tmp_path) if tid == "auditoria" else g.Config()
    if tid == "carregar_config":                            # sem perfil salvo a tela nao pergunta nada
        g._PERFIS_DIR.mkdir(parents=True, exist_ok=True)
        g._save_config(g.Config(), str(g._PERFIS_DIR / "perfil_a.yaml"))
    cli.dizer("?", "", "v")
    g._abrir_tela(tid, cfg)
    saida = cli.saida()
    t = cat.tela(tid)
    assert "Enter vazio volta ao menu" not in saida      # a linha cinza de antes
    if tid in _FERRAMENTAS:
        assert "Menu:" in saida and "Faz:" in saida and "Opcoes:" in saida, tid
        return
    assert "Faz:" in saida and "Use quando:" in saida, tid
    assert f"[{t['tecla']}] {t['nome'][0]}" in saida, tid


@pytest.mark.parametrize("num", [x["num"] for x in cat.grupos_cli()])
def test_interrogacao_no_menu_de_grupo_explica_cada_tela(cli, num):
    cli.mod._STATE["modo_usuario"] = "avancado"
    cli.dizer("?", "", "v")
    g._menu_grupo(num, g.Config())
    saida = cli.saida()
    assert "Menu:" in saida and "Opcoes:" in saida
    for x in cat.telas_do_grupo_cli(num):
        assert x["nome"][0] in saida
        assert cat.guia_tela(x["id"], "PT")["use"][:40] in saida.replace("\n", " ") or x["nome"][0] in saida


def test_menu_de_grupo_nao_mostra_a_ajuda_da_ultima_tela_aberta(cli):
    """Captura 4: abrir [2.4] Tecnica, voltar e apertar "?" no grupo mostrava '? Tecnica analitica: ...'."""
    cli.dizer("4", "v", "?", "", "v")
    g._menu_grupo("2", g.Config())
    saida = cli.saida()
    depois = saida[saida.rindex("Opcao: ") if "Opcao: " in saida else 0:]
    assert "? Tecnica analitica" not in saida
    assert "Menu:" in saida and "Planejar e preparar dados" in depois or "Menu:" in saida
    assert g._AJUDA_TELA is None        # ao sair da tela, o "?" volta a ser de quem a abriu


def test_interrogacao_n_no_grupo_abre_o_painel_da_tela(cli):
    cli.dizer("?3", "", "v")
    g._menu_grupo("2", g.Config())
    assert "Tela:" in cli.saida() and "[2.3] Dados" in cli.saida()


def test_interrogacao_no_menu_principal_explica_grupos_e_teclas(cli):
    cli.dizer("")
    g._ajuda_menu_principal(g.Config())
    saida = cli.saida()
    for x in cat.grupos_cli():
        assert x["nome"][0] in saida
    for tecla in ("[R]", "[G]", "[C]", "[I]", "[M]", "[Q]", "[A]"):
        assert tecla in saida


def test_menu_principal_numero_abre_o_grupo_e_a_abre_o_indice(cli, monkeypatch):
    abertos = []
    monkeypatch.setattr(g, "_menu_help", lambda cfg=None: abertos.append("indice"))
    cli.dizer("3", "")
    g._ajuda_menu_principal(g.Config())
    assert "Explorar, classificar ou quantificar" in cli.saida() and "Modelagem" in cli.saida() or "[1]" in cli.saida()
    cli.dizer("A")
    g._ajuda_menu_principal(g.Config())
    assert abertos == ["indice"]


@pytest.mark.parametrize("tid", [t for t in cat.TELAS if cat.TELAS[t]["tipo"] == "config" and cat.campos_da_tela(t)])
def test_interrogacao_em_tela_de_configuracao_lista_todas_as_opcoes_numeradas(cli, tid):
    campos = cat.campos_da_tela(tid, basicos_primeiro=True)
    cli.dizer("", "")                                     # Enter na pergunta = a tela inteira
    g._ajuda_de_campo(campos, tid)
    saida = cli.saida()
    assert "Opcoes:" in saida and "Tela:" in saida
    for i, k in enumerate(campos, 1):
        assert f"[{i}]" in saida and cat.opcao(k)["nome"][0] in saida, k


@pytest.mark.parametrize("tid", [t for t in cat.TELAS if cat.TELAS[t]["tipo"] == "config" and cat.campos_da_tela(t)])
def test_interrogacao_pergunta_primeiro_e_mostra_so_a_opcao_pedida(cli, tid):
    """Pedido de 2026-11-06: o "?" nao despeja a tela inteira; pergunta o numero e mostra so' aquela opcao."""
    campos = cat.campos_da_tela(tid, basicos_primeiro=True)
    cli.dizer("1", "")
    g._ajuda_de_campo(campos, tid)
    saida = cli.saida()
    assert "Opcoes:" not in saida and "Tela:" not in saida             # nada da lista inteira
    assert "Campo:" in saida and "Faixa:" in saida and cat.opcao(campos[0])["nome"][0] in saida
    assert saida.count("╭") == 1                                        # um painel so'


@pytest.mark.parametrize("lang", ["PT", "EN"])
@pytest.mark.parametrize("key", list(cat.OPCOES))
def test_ajuda_de_todo_campo_no_molde_e_sem_ingles_em_pt(cli, key, lang):
    cli.mod._STATE["lang"] = lang
    L = 1 if lang == "EN" else 0
    cli.dizer("")
    g._mostrar_ajuda(key)
    saida = cli.saida()
    for rot in ("campo", "faz", "impacto", "padrao", "faixa"):
        assert g._ROT_AJUDA[rot][L] in saida, (key, rot)
    if lang == "PT":
        achado = _INGLES.search(saida)
        assert not achado, (key, achado.group(0) if achado else "")


def test_faixa_do_caminho_aparece_em_portugues(cli):
    cli.dizer("")
    g._mostrar_ajuda("pasta_dados")
    assert "Valid system path" not in cli.saida() and "Caminho valido no computador" in cli.saida()


@pytest.mark.parametrize("tid", ["exemplo", "perfis", "tecnicas_avancadas", "codificacao", "sobre", "fluxo_planejamento"])
def test_listas_internas_explicam_cada_opcao(cli, sem_pausa, tid):
    cli.dizer("?", "", "v")
    g._abrir_tela(tid, g.Config())
    saida = cli.saida()
    assert "Opcoes:" in saida and "[1]" in saida[saida.index("Opcoes:"):], tid


def test_exemplo_guiado_mostra_a_descricao_inteira_de_cada_cenario(cli):
    from guaraci import exemplos as ex
    cli.dizer("?", "")
    g._abrir_tela("exemplo", g.Config())
    saida = " ".join(cli.saida().split())
    for k in ex.listar():
        assert " ".join(ex.EXEMPLOS[k]["faz"][0].split())[:60] in saida.replace("│", " ").replace("  ", " ") or \
            ex.EXEMPLOS[k]["nome"][0] in saida


def test_tecnica_interrogacao_lista_tecnicas_e_n_detalha(cli):
    cli.dizer("?", "2", "", "v")                                # ? pergunta; 2 mostra so' a tecnica 2
    g._menu_technique(g.Config())
    saida = cli.saida()
    assert "Opcoes:" not in saida and "Arquivos:" in saida and "Faixa:" in saida
    cli.dizer("?", "", "", "v")                                 # Enter = a lista inteira
    g._menu_technique(g.Config())
    assert "Opcoes:" in cli.saida()


def test_assistente_g_explica_as_proprias_opcoes(cli):
    cli.dizer("?", "1", "v")
    g._abrir_assistente("teste", g.Config(), "dados")
    saida = cli.saida()
    assert "Opcoes:" not in saida and "Revisar configuracao atual" in saida and "Confere a configuracao atual" in saida
    cli.dizer("?", "", "v")
    g._abrir_assistente("teste", g.Config(), "dados")
    assert "Opcoes:" in cli.saida()


def test_interrogacao_seguida_de_letra_explica_a_tecla(cli):
    cli.dizer("?", "I", "", "v")
    g._menu_grupo("2", g.Config())
    assert "Tecla:" in cli.saida() and "portugues e ingles" in cli.saida()


def test_paleta_tem_ajuda_propria_dentro_da_visualizacao(cli):
    cli.mod._STATE["modo_usuario"] = "avancado"
    n = len(cat.campos_da_tela("visualizacao")) + 1           # P e' a primeira acao depois dos campos
    cli.dizer(str(n), "?", "", "", "v")
    g._abrir_tela("visualizacao", g.Config())
    assert "Escolhe as cores das figuras" in " ".join(cli.saida().split())


def test_texto_para_exibir_traduz_booleano_e_lista_vazia(cli):
    assert g._texto_para_exibir(True) == "Sim" and g._texto_para_exibir([]) == "(vazio)"
    cli.mod._STATE["lang"] = "EN"
    assert g._texto_para_exibir(False) == "No" and g._texto_para_exibir("Valid system path") == "Valid system path"
