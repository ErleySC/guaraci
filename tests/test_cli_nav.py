"""Guia de design v2 da CLI (docs/DESIGN.md secao 7): teclas globais,
rodape gerado, arvore de menus, modo Basico/Avancado e ajuda por tela.

`cli_nav` e' dado puro; os testes de integracao no fim dirigem `main()` e
as telas de `guaraci.py` com uma sequencia de entradas, como um usuario."""
from __future__ import annotations

import inspect
import re

import pytest

import guaraci.cli_nav as nav
import guaraci.guaraci as g


# ── interpretar: o que cada entrada significa num prompt de menu ─────────

@pytest.mark.parametrize("raw,esperado", [
    ("3", ("item", 3)),
    (" 12 ", ("item", 12)),
    ("?", ("ajuda", None)),
    ("?3", ("ajuda", 3)),
    ("? 3", ("ajuda", 3)),
    ("g", ("guaraci", None)),
    ("I", ("idioma", None)),
    ("r", ("rodar", None)),
    ("M", ("avancados", None)),
    ("V", ("voltar", None)),
    ("", ("voltar", None)),
    ("0", ("voltar", None)),
    ("Q", ("voltar", None)),    # compatibilidade: Q voltava de submenu
    ("B", ("outro", None)),
    ("?x", ("outro", None)),
])
def test_interpretar_tela_interna(raw, esperado):
    assert nav.interpretar(raw) == esperado


@pytest.mark.parametrize("raw,esperado", [
    ("Q", ("sair", None)),
    ("", ("nada", None)),      # Enter no menu principal nao sai nem "volta"
    ("V", ("nada", None)),
    ("0", ("nada", None)),
    ("2", ("item", 2)),
    ("B", ("outro", None)),    # atalho legado: a tela decide
])
def test_interpretar_menu_principal(raw, esperado):
    assert nav.interpretar(raw, raiz=True) == esperado


def test_rodape_tem_as_quatro_teclas_e_sair_so_na_raiz():
    interno = [k for k, _ in nav.rodape("PT")]
    raiz = [k for k, _ in nav.rodape("PT", raiz=True)]
    assert interno == ["?", "G", "I", "V"]
    assert raiz == ["?", "G", "I", "Q"]


def test_rotulo_do_idioma_nao_diz_so_assistente():
    """`I` troca o idioma da interface inteira (`_toggle_idioma`), nao so'
    do assistente -- o rotulo do rascunho original estava errado."""
    rot = dict(nav.rodape("PT"))["I"]
    assert rot == "Idioma"


def test_linha_de_acoes_so_mostra_m_onde_ha_algo_oculto():
    assert nav.linha_acoes("PT", rodar=True, n_ocultos=None, modo_avancado=False) == [
        ("R", "Rodar análise")]
    acoes = dict(nav.linha_acoes("PT", rodar=False, n_ocultos=3, modo_avancado=False))
    assert acoes == {"M": "Mostrar avançados (3 ocultos)"}
    assert dict(nav.linha_acoes("PT", rodar=False, n_ocultos=1, modo_avancado=False))["M"] \
        == "Mostrar avançados (1 oculto)"
    assert dict(nav.linha_acoes("EN", rodar=False, n_ocultos=0, modo_avancado=True))["M"] \
        == "Hide advanced"


def test_atalhos_legados_nao_colidem_com_teclas_globais():
    assert not set(nav.ATALHOS_LEGADOS) & set(nav.TECLAS_GLOBAIS)
    ids = {it.id for gr in nav.MENU for it in gr.itens}
    assert set(nav.ATALHOS_LEGADOS.values()) <= ids


# ── arvore de menus ───────────────────────────────────────────────────────

def _todas_entradas():
    yield nav.MENU
    for gr in nav.MENU:
        yield gr.itens


@pytest.mark.parametrize("entradas", list(_todas_entradas()))
def test_numero_de_item_basico_nao_muda_entre_os_modos(entradas):
    """Avancados vem sempre DEPOIS dos basicos: o numero que o usuario
    decorou no modo Basico continua valendo no Avancado."""
    basico, _ = nav.ordenar_visiveis(entradas, False)
    avancado, n_oc = nav.ordenar_visiveis(entradas, True)
    assert avancado[:len(basico)] == basico
    assert n_oc == 0
    assert all(not e.avancado for e in basico)


def test_menu_principal_basico_tem_seis_macroetapas():
    basico, n_oc = nav.ordenar_visiveis(nav.MENU, False)
    assert [gr.id for gr in basico] == [
        "comecar", "preparar", "modelar", "prever", "graficos", "sistema"]
    assert n_oc == 2


def test_auditoria_fica_visivel_no_modo_basico():
    """Decisao do guia v2 (desvio do rascunho): a auditoria anti-vazamento
    e' o que mais protege um iniciante -- nao pode estar escondida."""
    visiveis, _ = nav.ordenar_visiveis(nav.GRUPOS["preparar"].itens, False)
    assert "auditoria" in [it.id for it in visiveis]


def test_validacao_tem_lugar_no_menu_basico():
    visiveis, _ = nav.ordenar_visiveis(nav.GRUPOS["modelar"].itens, False)
    assert "validacao" in [it.id for it in visiveis]


def test_hsi_aparece_no_basico_quando_a_tecnica_e_hsi():
    sem, _ = nav.ordenar_visiveis(nav.MENU, False)
    com, n_oc = nav.ordenar_visiveis(nav.MENU, False, frozenset({"hsi"}))
    assert "hsi" not in [gr.id for gr in sem]
    assert "hsi" in [gr.id for gr in com]
    assert n_oc == 1


def test_caminho_de_tela_e_legivel_e_sem_tecla():
    assert nav.caminho("dados", "PT") == "Preparar dados › Dados"
    assert nav.caminho("predicao", "PT") == "Prever novas amostras"   # grupo direto
    assert nav.caminho("hsi", "EN") == "Hyperspectral imaging"
    assert "[" not in nav.caminho("validacao", "PT")


def test_proximo_passo():
    assert "Começar" in nav.proximo_passo("PT", dados_ok=False, hsi=False)
    assert "[R]" in nav.proximo_passo("PT", dados_ok=True, hsi=False)
    assert "hiperespectral" in nav.proximo_passo("PT", dados_ok=True, hsi=True)


# ── ajuda por tela ────────────────────────────────────────────────────────

@pytest.mark.parametrize("tela", nav.ids_telas())
@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_toda_tela_tem_os_tres_textos_de_ajuda(tela, lang):
    textos = nav.ajuda_tela(tela, lang)
    assert textos is not None, f"tela '{tela}' sem ajuda em {lang}"
    assert len(textos) == 3 and all(t.strip() for t in textos)


# ── integracao com guaraci.py ─────────────────────────────────────────────

@pytest.fixture()
def cli(monkeypatch, tmp_path):
    for nome, valor in [("_USER_DIR", tmp_path), ("_CFG_PATH", tmp_path / "config.yaml"),
                        ("_LANG_FLAG", tmp_path / "lang"), ("_PERFIS_DIR", tmp_path / "perfis"),
                        ("_MODO_FLAG", tmp_path / "modo"), ("_VISUAL_PATH", tmp_path / "v.json")]:
        monkeypatch.setattr(g, nome, valor)
    monkeypatch.setattr(g, "_cls", lambda: None)
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    monkeypatch.setattr(g, "_exibir_boas_vindas", lambda: None)
    monkeypatch.setattr(g, "_exibir_despedida", lambda: None)
    monkeypatch.setitem(g._STATE, "modo_usuario", "iniciante")
    monkeypatch.setitem(g._STATE, "lang", "PT")

    def entradas(*seq):
        it = iter(seq)

        def _fake(*a, **k):
            try:
                return next(it)
            except StopIteration:
                raise EOFError
        monkeypatch.setattr("builtins.input", _fake)
    return entradas


def test_toda_tela_do_menu_tem_destino(cli, monkeypatch):
    """`_abrir_tela` despacha toda tela da arvore -- uma tela nova em
    `cli_nav.MENU` sem destino quebraria aqui, nao na mao do usuario."""
    alvos = ["_rodar_exemplo", "_menu_profiles", "_guaraci_fluxo_decisao",
             "_menu_project", "_menu_data", "_menu_technique", "_menu_preprocessing",
             "_menu_audit", "_menu_plan", "_menu_selecao_amostras", "_menu_encoding",
             "_menu_objetivo_lvs", "_menu_validation", "_menu_metodos_classe",
             "_menu_selecao_variaveis", "_menu_advanced", "_menu_prediction",
             "_menu_visualization", "_salvar_yaml", "_carregar_yaml",
             "_editar_nome_execucao", "_menu_about", "_menu_hardware",
             "_menu_tecnicas_avancadas", "_menu_hsi", "_toggle_idioma",
             "_alternar_modo"]
    chamados = []
    for nome in alvos:
        monkeypatch.setattr(g, nome, lambda *a, _n=nome, **k: chamados.append(_n))
    for gr in nav.MENU:
        for it in gr.itens:
            antes = len(chamados)
            g._abrir_tela(it.id, g.Config())
            assert len(chamados) == antes + 1, it.id


def test_menu_principal_numero_abre_grupo_e_atalho_legado_continua(cli, monkeypatch):
    abertos = []
    monkeypatch.setattr(g, "_menu_objetivo_lvs", lambda cfg: abertos.append("objetivo"))
    monkeypatch.setattr(g, "_menu_prediction", lambda cfg: abertos.append("predicao"))
    # 3 = Modelar e validar; 1 = Objetivo; V volta; B (legado) = predicao; Q sai
    cli("3", "1", "V", "B", "Q")
    g.main([])
    assert abertos == ["objetivo", "predicao"]


def test_m_no_grupo_revela_avancados_sem_mudar_os_numeros(cli, monkeypatch):
    abertos = []
    monkeypatch.setattr(g, "_menu_audit", lambda cfg: abertos.append("auditoria"))
    monkeypatch.setattr(g, "_menu_encoding", lambda cfg: abertos.append("codificacao"))
    # 2 = Preparar dados; 5 = auditoria (basico); M; 5 de novo; 8 = codificacao
    cli("2", "5", "M", "5", "8", "V", "Q")
    g.main([])
    assert abertos == ["auditoria", "auditoria", "codificacao"]


def test_q_com_configuracao_alterada_pergunta_antes_de_sair(cli, monkeypatch, tmp_path):
    def _muda(cfg):
        cfg.max_lvs = 7
    monkeypatch.setattr(g, "_menu_project", _muda)
    # 2 > 1 muda a config; V; Q -> pergunta; Enter = fica; Q -> 1 salva e sai
    cli("2", "1", "V", "Q", "", "Q", "1")
    g.main([])
    assert (tmp_path / "config.yaml").exists()
    assert re.search(r"max_lvs:\s*7\b",
                     (tmp_path / "config.yaml").read_text(encoding="utf-8"))


def test_q_sem_alteracao_sai_direto(cli, tmp_path):
    cli("Q")
    g.main([])
    assert not (tmp_path / "config.yaml").exists()


def test_nome_da_execucao_limpa_com_hifen_e_nao_com_interrogacao(cli):
    """`?` e' ajuda em todo lugar; antes, no nome da execucao, limpava."""
    cfg = g.Config(); cfg.tag = "abc"
    cli("-")
    g._editar_nome_execucao(cfg)
    assert cfg.tag == ""


def test_g_abre_o_assistente_no_assunto_da_tela(cli, monkeypatch):
    vistos = []
    monkeypatch.setattr(g, "_painel_ajuda_tela", lambda tela: vistos.append(tela))
    cli("V")
    g._abrir_assistente("Validação", g.Config(), tela="validacao")
    assert vistos == ["validacao"]


def test_extras_locais_viram_numeros_depois_dos_campos(cli, monkeypatch):
    """Visualizacao: paleta/fonte/grade/transparencia eram P/F/D/A; agora
    sao 7-10 (6 campos + 4 acoes)."""
    chamados = []
    monkeypatch.setattr(g, "_carregar_visual_cfg", lambda: chamados.append("pal") or {})
    cli("7", "", "V")
    g._menu_visualization(g.Config())
    assert chamados and chamados[0] == "pal"


def test_nenhuma_tela_escreve_rodape_a_mao():
    """O rodape vem de `_print_rodape`; nenhum `[0] Voltar` ou `[G]
    Guaraci` digitado a mao pode voltar a aparecer."""
    fonte = inspect.getsource(g)
    assert not re.search(r"\[\{PM\}\]\[0\]\[/\{PM\}\]", fonte)
    assert not re.search(r"\[\{PA\}\]\[G\]\[/\{PA\}\] Guaraci", fonte)
