"""Menu por objetivo (Fase 4): contratos de navegacao derivados do catalogo."""
from guaraci import catalogo as cat
from guaraci import cli_comandos

ORDEM_OBJETIVO = ["comecar", "dados", "explorar", "modelar", "comparar", "monitorar", "imagens", "graficos",
                  "confianca", "ajuda"]


def test_grupos_com_tela_seguem_a_ordem_por_objetivo():
    com_tela = [g["id"] for g in cat.grupos() if cat.telas(g["id"])]
    assert com_tela == ORDEM_OBJETIVO


def test_toda_tela_esta_em_exatamente_um_grupo_valido_e_tem_tecla_unica():
    ids = {g["id"] for g in cat.grupos()}
    teclas = [t["tecla"].upper() for t in cat.TELAS.values()]
    assert all(t["grupo"] in ids for t in cat.TELAS.values())
    assert len(teclas) == len(set(teclas))
    assert not (set(teclas) & set(cat.ACOES_MENU))


def test_toda_opcao_fica_a_no_maximo_tres_niveis_do_menu():
    """menu principal (1) -> tela (2) -> campo (3): toda opcao pertence a uma tela listada no menu."""
    for oid, o in cat.OPCOES.items():
        assert o["tela"] in cat.TELAS, oid


def test_toda_tela_tem_handler_no_cli():
    from guaraci import guaraci as g
    for tid, t in cat.TELAS.items():
        assert callable(getattr(g, t["handler"], None)), (tid, t["handler"])


def test_help_lista_os_comandos_na_ordem_dos_grupos_do_menu():
    texto = cli_comandos.texto_ajuda("PT")
    posicoes = [texto.find(f"\n{g['nome'][0]}:") for g in cat.grupos() if cat.comandos(g["id"])]
    assert all(p > 0 for p in posicoes) and posicoes == sorted(posicoes)
    for c in cat.comandos():
        assert c["uso"].split()[0] in texto


def test_fluxo_planejar_selecionar_auditar_esta_no_grupo_de_preparo_na_ordem():
    tels = [t["id"] for t in cat.telas("dados")]
    assert tels[0] == "fluxo_planejamento"
    assert tels.index("planejamento") < tels.index("selecao_amostras") < tels.index("auditoria")
