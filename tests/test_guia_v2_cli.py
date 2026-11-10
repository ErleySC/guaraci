"""Contrato do guia de design v2 da CLI (2026-10-14): tela inicial que cabe na
janela, teclas globais so' em prompt de menu, numeracao estavel entre os modos,
ajuda da tela (faz / quando usar / depois), `G` contextual e textos curtos."""
import pytest

import guaraci.guaraci as g
from guaraci import catalogo as cat
from guaraci import catalogo_curta


@pytest.fixture(autouse=True)
def _sem_tela_limpa(monkeypatch, tmp_path):
    monkeypatch.setattr(g, "_cls", lambda: None)
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    monkeypatch.setattr(g, "_MODO_FLAG", tmp_path / ".cli_modo_usuario")
    monkeypatch.setattr(g, "_LANG_FLAG", tmp_path / ".cli_lang")
    antes = dict(g._STATE)
    g._STATE["lang"] = "PT"
    g._STATE["modo_usuario"] = "iniciante"
    yield
    g._STATE.clear()
    g._STATE.update(antes)


def _capturar(fn, largura=100):
    import io
    from rich.console import Console
    buf = io.StringIO()
    velho = g.console
    g.console = Console(file=buf, width=largura, force_terminal=False)
    try:
        fn()
    finally:
        g.console = velho
    return buf.getvalue()


def _entradas(monkeypatch, *respostas):
    it = iter(respostas)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(it, "0"))


# ---- tela inicial ---------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("largura", [80, 100, 120])
def test_tela_inicial_do_modo_basico_cabe_numa_janela_de_30_linhas(largura):
    """O foco do terminal abre no topo do menu, nao abaixo dele: a tela toda tem de caber."""
    cfg = g.Config()

    def tela():
        g._print_header(cfg)
        g._print_status(cfg)
        g._print_main_menu(cfg)
        g._print_acao_rodar(cfg)

    linhas = _capturar(tela, largura).rstrip("\n").split("\n")
    assert len(linhas) <= 30, "\n".join(linhas)


def test_faixa_de_identidade_tem_moldura_e_uma_linha_so():
    """A faixa do GUARACI volta com a moldura de antes, mas com uma linha de texto."""
    for largura in (80, 100, 120):
        linhas = _capturar(lambda: g._print_header(g.Config()), largura).rstrip("\n").split("\n")
        assert len(linhas) == 3 and "╔" in linhas[0] and "GUARACI" in linhas[1], linhas
        assert "Tecnica: FT-NIR" in linhas[1]
    assert "Plataforma quimiometrica" in _capturar(lambda: g._print_header(g.Config()), 120)


def test_tela_inicial_nao_tem_caixa_de_execucao_nem_hardware():
    cfg = g.Config()
    saida = _capturar(lambda: (g._print_header(cfg), g._print_status(cfg), g._print_main_menu(cfg), g._print_acao_rodar(cfg)))
    assert "EXECUCAO" not in saida and "RAM" not in saida and "Hardware" not in saida


def test_status_tem_proximo_passo_e_aponta_o_exemplo_sem_dados():
    saida = _capturar(lambda: g._print_status(g.Config()))
    assert "Proximo passo" in saida and "[2.3]" in saida and "[1.1]" in saida


def test_status_com_dados_prontos_sugere_rodar(tmp_path):
    saida = _capturar(lambda: g._print_status(g.Config(mode="sintetico")))
    assert "Proximo passo" in saida and "[R]" in saida


def test_menu_principal_mostra_o_numero_de_telas_ocultas_no_basico():
    saida = _capturar(lambda: g._print_main_menu(g.Config()))
    assert "telas avancadas ocultas" in saida
    g._STATE["modo_usuario"] = "avancado"
    assert "ocultas" not in _capturar(lambda: g._print_main_menu(g.Config()))


def test_auditoria_fica_visivel_no_basico_e_hsi_so_com_tecnica_hsi(monkeypatch):
    _entradas(monkeypatch, "v")
    dados = _capturar(lambda: g._menu_grupo("2", g.Config()))
    assert "[6]" in dados and "Auditoria de delineamento" in dados
    _entradas(monkeypatch, "v")
    assert "Imageamento hiperespectral" not in _capturar(lambda: g._menu_grupo("3", g.Config()))
    _entradas(monkeypatch, "v")
    com_hsi = _capturar(lambda: g._menu_grupo("3", g.Config(mode="hsi")))
    assert "[6]" in com_hsi and "Imageamento hiperespectral" in com_hsi


# ---- rodape ---------------------------------------------------------------------------------------------------------------------------------

def test_rodape_da_tela_tem_as_teclas_globais_e_voltar():
    for lang in ("PT", "EN"):
        r = cat.RODAPE_TELA[lang]
        assert all(t in r for t in ("?", "G ", "I ", "V ", "R "))
        assert "S " not in r.replace("Guaraci", "")


def test_rodape_do_menu_principal_tem_q_sair_e_nao_voltar():
    assert "Q sair" in cat.RODAPE_MENU["PT"] and "V " not in cat.RODAPE_MENU["PT"]


# ---- teclas globais so' em prompt de menu --------------------------------------------------------------------------------------

@pytest.mark.parametrize("valor", ["S", "s", "Q", "G", "I", "C", "R", "V", "classe_S"])
def test_prompt_de_valor_trata_letras_como_texto(valor):
    assert g._tecla_valor(valor, "dados") is None


def test_prompt_de_valor_so_cancela_com_enter_ou_zero_e_abre_ajuda_com_interrogacao(monkeypatch):
    chamadas = []
    monkeypatch.setattr(g, "_ajuda_da_tela", lambda tid: chamadas.append(tid))
    assert g._tecla_valor("", "dados") == "voltar" and g._tecla_valor("0", "dados") == "voltar"
    assert g._tecla_valor("?", "dados") == "tratada" and chamadas == ["dados"]


def test_s_em_tela_interna_nao_encerra_o_programa():
    assert g._tecla_comum("s", "dados", g.Config()) is None
    assert g._tecla_comum("S", "dados", g.Config()) is None


def test_classe_chamada_s_nao_derruba_o_planejamento(monkeypatch):
    """Antes, digitar S na pergunta de classes encerrava o programa."""
    _entradas(monkeypatch, "S,Q", "", "", "", "", "", "0")
    g._menu_plan(g.Config())        # nao levanta _SairDoPrograma


def test_confirmacao_de_saida_usa_numeros_e_nao_g_nem_s(monkeypatch):
    cfg = g.Config()
    foto = g._foto_cfg(cfg)
    cfg.max_lvs = 7
    perguntas = []
    monkeypatch.setattr(g, "_input", lambda msg="", default="": perguntas.append(msg) or "2")
    assert g._confirmar_saida(cfg, foto) is True
    assert "[1]" in perguntas[0] and "[2]" in perguntas[0] and "[G]" not in perguntas[0] and "[S]" not in perguntas[0]


# ---- numeracao estavel ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tela", [t["id"] for t in cat.telas() if t["tipo"] == "config"])
def test_campos_basicos_vem_antes_dos_avancados_e_nao_mudam_de_numero(tela):
    todos = cat.campos_da_tela(tela, basicos_primeiro=True)
    basicos = cat.campos_da_tela(tela, avancado=False, basicos_primeiro=True)
    assert todos[:len(basicos)] == basicos           # o numero de uma opcao basica e' o mesmo nos dois modos


def test_alerta_de_vazamento_cita_o_numero_certo_nos_dois_modos(monkeypatch):
    campos = cat.campos_da_tela("validacao", basicos_primeiro=True)
    n = campos.index("validacao_group_aware") + 1
    cfg = g.Config()
    setattr(cfg, g._SPEC_BY_KEY["validacao_group_aware"]["attr"], False)
    for modo in ("iniciante", "avancado"):
        g._STATE["modo_usuario"] = modo
        _entradas(monkeypatch, "0")
        saida = _capturar(lambda: g._menu_validation(cfg))
        assert "DESATIVADO" in saida and f"campo [{n}]" in saida, modo


def test_acoes_da_tela_sao_numeradas_depois_dos_campos(monkeypatch):
    _entradas(monkeypatch, "0")
    saida = _capturar(lambda: g._menu_visualization(g.Config()))
    n = len(cat.campos_da_tela("visualizacao", avancado=False, basicos_primeiro=True))
    for j in range(1, 5):
        assert f"[{n + j}]" in saida
    assert "[P]" not in saida and "[T]" not in saida


# ---- ajuda da tela e G contextual ---------------------------------------------------------------------------------------------

def test_toda_tela_tem_quando_usar_e_proximo_passo_nos_dois_idiomas():
    for tid in cat.TELAS:
        for lang in ("PT", "EN"):
            guia = cat.guia_tela(tid, lang)
            assert guia["faz"] and guia["use"] and guia["proximo"], (tid, lang)


def test_ajuda_da_tela_mostra_faz_quando_e_depois():
    saida = _capturar(lambda: g._ajuda_da_tela("validacao"))
    assert "Faz:" in saida and "Use quando:" in saida and "Depois:" in saida


def test_ajuda_da_tela_marca_os_campos_avancados_ocultos_no_basico():
    saida = _capturar(lambda: g._ajuda_da_tela("modelo"))
    assert "(avancado: ative com [+])" in saida


def test_g_abre_o_assistente_no_assunto_da_tela(monkeypatch):
    _entradas(monkeypatch, "Q")
    saida = _capturar(lambda: g._abrir_assistente("Validacao", g.Config(), "validacao"))
    assert "Use quando:" in saida and "Chamado de: Validacao" in saida


def test_interrogacao_n_abre_a_ajuda_do_campo_n(monkeypatch):
    abertos = []
    monkeypatch.setattr(g, "_mostrar_ajuda", lambda k: abertos.append(k))
    assert g._tecla_comum("?2", "validacao", g.Config(), ["a", "b", "c"]) == "tratada"
    assert abertos == ["b"]


def test_r_de_qualquer_tela_roda_a_analise(monkeypatch):
    rodou = []
    monkeypatch.setattr(g, "_rodar_pipeline", lambda cfg: rodou.append(1))
    assert g._tecla_comum("r", "dados", g.Config()) == "tratada" and rodou == [1]


# ---- nome da execucao e descricoes curtas -------------------------------------------------------------------------------------

def test_toda_opcao_tem_descricao_curta_completa_nos_dois_idiomas():
    assert set(catalogo_curta.CURTA) == set(cat.OPCOES)
    for oid, (pt, en) in catalogo_curta.CURTA.items():
        for t in (pt, en):
            assert t and len(t) <= 44 and "…" not in t and "..." not in t, (oid, t)
            assert not cat.texto_proibido(t), (oid, t)


def test_lista_de_campos_usa_a_descricao_curta():
    assert g._desc_curta("max_lvs", 20) == catalogo_curta.CURTA["max_lvs"][0]
    g._STATE["lang"] = "EN"
    assert g._desc_curta("max_lvs", 20) == catalogo_curta.CURTA["max_lvs"][1]


# ---- grupos e subpastas (pedido de 2026-10-14: menos informacao ao abrir) ----------------------------------------

def test_toda_tela_tem_caminho_grupo_item_e_so_r_fica_como_letra():
    for t in cat.telas():
        if t["id"] == "rodar":
            assert t["tecla"] == "R"
            continue
        num, item = t["tecla"].split(".")
        assert num in {g_["num"] for g_ in cat.grupos_cli()} and item.isdigit(), t["id"]
    assert cat.validar_catalogo() == []


def test_no_grupo_as_telas_basicas_vem_antes_e_o_numero_nao_muda_com_o_modo(monkeypatch):
    for g_ in cat.grupos_cli():
        itens = cat.telas_do_grupo_cli(g_["num"])
        assert [int(t["tecla"].split(".")[1]) for t in itens] == list(range(1, len(itens) + 1))
        niveis = [t["nivel"] for t in itens]
        assert niveis == sorted(niveis, key=lambda n: n != "basico")


def test_menu_principal_abre_o_grupo_pelo_numero_e_a_tela_pelo_caminho(monkeypatch):
    abertos = []
    monkeypatch.setattr(g, "_menu_data", lambda cfg: abertos.append("dados"))
    monkeypatch.setattr(g, "_menu_grupo", lambda num, cfg: abertos.append(f"grupo {num}"))
    monkeypatch.setattr(g, "_exibir_boas_vindas", lambda: None)
    monkeypatch.setattr(g, "_CFG_PATH", g._USER_DIR / "nao_existe" / "config.yaml")
    monkeypatch.setattr("sys.argv", ["guaraci"])
    _entradas(monkeypatch, "2", "2.3", "q")
    _capturar(lambda: g.main([]))
    assert abertos == ["grupo 2", "dados"]


def test_dentro_do_grupo_o_numero_do_item_abre_a_tela(monkeypatch):
    abertos = []
    monkeypatch.setattr(g, "_menu_validation", lambda cfg: abertos.append("validacao"))
    _entradas(monkeypatch, "2", "v")
    _capturar(lambda: g._menu_grupo("3", g.Config()))
    assert abertos == ["validacao"]


def test_tela_avancada_oculta_no_grupo_abre_com_um_toque(monkeypatch):
    """O numero de uma tela avancada oculta abre a tela direto (antes pedia [+] e um segundo toque)."""
    abertos = []
    monkeypatch.setattr(g, "_menu_plan", lambda cfg: abertos.append("planejamento"))
    _entradas(monkeypatch, "7", "v")
    _capturar(lambda: g._menu_grupo("2", g.Config()))
    assert abertos == ["planejamento"]


def test_tecla_antiga_do_menu_diz_para_onde_a_tela_foi():
    saida = _capturar(lambda: g._aviso_tecla_antiga("D"))
    assert "Esta tecla mudou" in saida and "[4.2]" in saida and "Monitorar lotes" in saida
    assert g._aviso_tecla_antiga("zz") is False


@pytest.mark.parametrize("tela", ["dados", "modelo", "validacao"])
def test_telas_densas_mostram_os_avancados_em_pastas(monkeypatch, tela):
    """Modo avancado: os basicos ficam na lista e cada pasta conta as opcoes que tem."""
    g._STATE["modo_usuario"] = "avancado"
    _entradas(monkeypatch, "v")
    handler = getattr(g, cat.TELAS[tela]["handler"])
    saida = _capturar(lambda: handler(g.Config(mode="sintetico" if tela == "dados" else "dx")), 120)
    basicos = cat.campos_da_tela(tela, avancado=False)
    pastas = [s for s in cat.secoes_da_tela(tela) if s["campos"]]
    assert any(s["nome"][0] in saida for s in pastas)
    rotulos = " ".join(s["nome"][0] + " " + s["resumo"][0] for s in pastas)   # o resumo da pasta cita os campos dela
    linhas = [ln for ln in saida.splitlines() if not any(s["resumo"][0][:20] in ln for s in pastas)]
    for k in cat.campos_avancados(tela):
        nome = cat.opcao(k)["nome"][0]
        assert nome in rotulos or not any(nome in ln for ln in linhas), k   # avancado so' dentro da pasta
    assert any(cat.opcao(k)["nome"][0] in saida for k in basicos)


def test_pasta_de_avancados_lista_e_edita_so_os_campos_dela(monkeypatch):
    g._STATE["modo_usuario"] = "avancado"
    editados = []
    monkeypatch.setattr(g, "_editar_campo", lambda cfg, k: editados.append(k))
    n_basicos = len(cat.campos_da_tela("modelo", avancado=False))
    sel = next(i for i, s in enumerate(cat.secoes_da_tela("modelo")) if s["id"] == "selecao")
    _entradas(monkeypatch, str(n_basicos + sel + 1), "1", "v", "v")
    saida = _capturar(lambda: g._menu_modeling(g.Config()), 120)
    assert "Modelagem › Selecao de variaveis" in saida
    assert editados == [cat.secoes_da_tela("modelo")[sel]["campos"][0]]


def test_no_basico_o_mais_revela_as_pastas_e_o_numero_dos_basicos_fica(monkeypatch):
    _entradas(monkeypatch, "+", "v")
    saida = _capturar(lambda: g._menu_modeling(g.Config()), 120)
    antes, depois = saida.split("Ocultar as pastas avancadas")[0].rsplit("Mostrar opcoes avancadas", 1)
    assert "[1]" in depois and "Nivel de analise" in depois and "Selecao de variaveis" in depois


def test_assistente_usa_numeros_e_v_para_voltar(monkeypatch):
    _entradas(monkeypatch, "v")
    saida = _capturar(lambda: g._abrir_assistente("teste", g.Config()))
    assert "[10]" in saida and "[11]" in saida and "[V]" in saida and "[D]" not in saida and "[B]" not in saida


def test_perfil_embutido_mostra_perfil_de_partida_e_o_do_usuario_sem_fonte_mostra_aviso(tmp_path, monkeypatch):
    from guaraci import guaraci as g
    monkeypatch.setattr(g, "_lang", lambda: "PT")
    for nome in ("nir_ft_difuso", "nir_dispersivo_1100_2498nm", "nir_transmitancia_8mm", "visnir_400_1100nm"):
        rot = g._rotulo_opcao("perfil_matriz", nome)
        assert "perfil de partida" in rot and "nao validado" not in rot, rot
    meu = tmp_path / "meu.yaml"
    meu.write_text("descricao: 'meu'\nunidade_eixo: 'nm'\nvocabulario:\n  matriz: 'x'\n", encoding="utf-8")
    assert "nao validado" in g._rotulo_opcao("perfil_matriz", str(meu))
    meu.write_text("descricao: 'meu'\nunidade_eixo: 'nm'\nreferencia: 'Autor (2020)'\nvocabulario:\n  matriz: 'x'\n", encoding="utf-8")
    assert "✅" in g._rotulo_opcao("perfil_matriz", str(meu))


def test_sobre_arte_so_em_terminal_largo_e_fora_do_modo_linear(monkeypatch):
    from guaraci import arte
    monkeypatch.delenv("GUARACI_ARTE", raising=False)
    assert not arte.pode_mostrar(200, linear=True)
    assert not arte.pode_mostrar(arte.LARGURA, linear=False)
    monkeypatch.setenv("GUARACI_ARTE", "0")
    assert not arte.pode_mostrar(200, linear=False)
    assert len(arte.SOL_BRAILLE.splitlines()) == 20


@pytest.mark.parametrize("tela,modo", [("dados", "dx"), ("modelo", "dx")])
def test_ajuda_de_cada_pasta_e_acao_numerada_funciona(monkeypatch, tela, modo):
    """?N vale para todo numero da tela: campos, pastas e acoes (antes, ?7..?11 em Dados dava 'comando invalido')."""
    g._STATE["modo_usuario"] = "avancado"
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    import re as _re
    pastas = [s for s in cat.secoes_da_tela(tela) if s["campos"]]
    handler = getattr(g, cat.TELAS[tela]["handler"])
    _entradas(monkeypatch, "v")
    tela_txt = _capturar(lambda: handler(g.Config(mode=modo)), 120)
    pastas = [s for s in pastas if s["nome"][0] in tela_txt]   # o modo (dx) pode esconder pasta vazia
    assert pastas
    nums = []
    for s in pastas:   # numero que a tela mostra para a pasta (campos ocultos pelo modo mudam a conta)
        ln = next(ln for ln in tela_txt.splitlines() if s["nome"][0] in ln)
        nums.append(_re.search(r"\[(\d+)\]", ln).group(1))
    acoes = _re.findall(r"\[(\d+)\][^\n]*›?", tela_txt)
    nums += [n for n in acoes if int(n) > max(map(int, nums))]   # acoes numeradas depois das pastas
    _entradas(monkeypatch, *[f"?{n}" for n in nums], "v")
    saida = _capturar(lambda: handler(g.Config(mode=modo)), 120)
    assert "invalid" not in saida.lower()
    for s in pastas:
        assert cat.opcao(s["campos"][0])["nome"][0] in saida, s["id"]


def test_pastas_alinham_com_os_campos(monkeypatch):
    """A linha da pasta usa as mesmas colunas dos campos (sem tabela separada com recuo proprio)."""
    g._STATE["modo_usuario"] = "avancado"
    _entradas(monkeypatch, "v")
    saida = _capturar(lambda: g._menu_modeling(g.Config()), 120)
    pastas = [s["nome"][0] for s in cat.secoes_da_tela("modelo") if s["campos"]]
    col_campo = {ln.index("[1]") for ln in saida.splitlines() if "[1]" in ln}
    col_pasta = {ln.index("[" + str(len(cat.campos_da_tela("modelo", avancado=False)) + 1) + "]")
                 for ln in saida.splitlines() if pastas[0] in ln}
    assert col_campo and col_campo == col_pasta


def test_tela_de_tecnica_alcanca_todo_modo_de_entrada_do_motor():
    """Todo leitor de dado (modo_entrada) tem uma tecnica/formato na tela [2.4] -- antes ICP, EEM, imagem, HSI,
    OPUS, SPC, .sp, RMN Bruker, HPLC de fabricante e GC-MS ANDI so' eram alcancaveis editando o campo a mao."""
    from guaraci.config_io import _CONFIG_SPEC
    modos = set(next(s["opcoes"] for s in _CONFIG_SPEC if s["key"] == "modo_entrada")) - {"sintetico"}
    usados = {m for k in cat.INSTRUMENTOS for m, _r in g._formatos_tecnica(k, "PT")}
    assert modos <= usados, sorted(modos - usados)
    assert {"icp", "eem", "imagem", "hsi"} <= set(g._tecnica_ordem())


def test_escolher_tecnica_com_formato_aplica_o_leitor(monkeypatch):
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    ordem = g._tecnica_ordem()
    n_mir = ordem.index("mir") + 1
    sp = [m for m, _r in g._formatos_tecnica("mir", "PT")].index("sp") + 1
    _entradas(monkeypatch, str(n_mir), str(sp))
    cfg = g.Config()
    saida = _capturar(lambda: g._menu_technique(cfg), 100)
    assert cfg.mode == "sp" and "Proximo" in saida
    _entradas(monkeypatch, str(ordem.index("icp") + 1))
    cfg = g.Config()
    _capturar(lambda: g._menu_technique(cfg), 100)
    assert cfg.mode == "multielementar" and cfg.default_preprocessing == "autoscaling"


def test_tecnicas_avancadas_rodam_vip_e_transferencia_com_dados_do_usuario(monkeypatch, tmp_path):
    from guaraci import exemplos as ex
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    ex.gerar("vip", str(tmp_path / "vip"))
    tab = str(tmp_path / "vip" / "tabela.csv")
    _entradas(monkeypatch, "9", tab, "classe", "amostra", "--repeticoes 2 --sem-figuras")
    _capturar(lambda: g._menu_tecnicas_avancadas(g.Config()), 100)
    assert (tmp_path / "vip" / "resultados_vip" / "tabelas" / "vip_estabilidade.csv").is_file()
    ex.gerar("transferir", str(tmp_path / "tr"))
    d = tmp_path / "tr"
    _entradas(monkeypatch, "10", str(d / "mestre.csv"), str(d / "escravo.csv"), str(d / "novos_escravo.csv"), "ds", "")
    _capturar(lambda: g._menu_tecnicas_avancadas(g.Config()), 100)
    assert (d / "resultados_transferencia" / "novos_no_espaco_do_mestre.csv").is_file()


def test_opcoes_extras_recusam_flag_desconhecida_e_repassam_as_validas(monkeypatch):
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    _entradas(monkeypatch, "--inventada 1", "--alpha 0.01 --sem-bootstrap")
    saida_args = []
    _capturar(lambda: saida_args.append(g._opcoes_extras("comparar", g.Config(), [
        ("--alpha=", "a", "a"), ("--sem-bootstrap", "b", "b")])), 100)
    assert saida_args == [["--alpha", "0.01", "--sem-bootstrap"]]


def test_manual_na_ajuda_e_diagnostico_no_hardware(monkeypatch):
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    from guaraci import docs_pacote as dp
    topico = next(iter(dp.TOPICOS))
    _entradas(monkeypatch, "0", topico, "", "v")
    saida = _capturar(lambda: g._menu_help(g.Config()), 100)
    assert dp.ler(topico).strip().splitlines()[0].lstrip("# ")[:15] in saida
    chamou = []
    monkeypatch.setattr(g, "_comando_doctor", lambda: chamou.append(1))
    _entradas(monkeypatch, "1")
    _capturar(lambda: g._menu_hardware(g.Config()), 100)
    assert chamou == [1]


def test_voltar_no_passo_2_do_perfil_volta_ao_passo_1(monkeypatch):
    """Bug relatado: no Passo 2 (rigor), V saia da tela inteira em vez de voltar ao Passo 1 (objetivo)."""
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    _entradas(monkeypatch, "1", "v", "v")
    saida = _capturar(lambda: g._menu_profiles(g.Config()), 100)
    assert saida.count("Passo 1 de 2") == 2 and saida.count("Passo 2 de 2") == 1


def test_voltar_na_pasta_de_destino_do_exemplo_nao_cria_pasta_v(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    _entradas(monkeypatch, "1", "v", "v")
    _capturar(lambda: g._menu_exemplo(g.Config()), 100)
    assert not (tmp_path / "v").exists() and not (tmp_path / "GUARACI_Exemplo").exists()


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_por_que_guaraci_explica_nome_metodo_e_compara_sem_relato_proibido(monkeypatch, lang):
    """[1] do Sobre: origem do nome, os seis passos do metodo e o comparativo com pagos e livres."""
    g._STATE["lang"] = lang
    monkeypatch.setattr(g, "_W", lambda: 80)
    _entradas(monkeypatch, "1", "V")
    out = _capturar(lambda: g._menu_about(g.Config()), 80)
    assert "kûarasy" in out
    for passo in (("Auditar", "Validar", "Registrar") if lang == "PT" else ("Audit", "Validate", "Record")):
        assert passo in out
    assert "GUARACI" in out and ("Livres**" in out or "Free**" in out)
    assert "21 CFR Part 11" in out, "o comparativo diz onde os pagos ainda vencem"
    assert "…" not in out, "nenhuma celula do comparativo pode ser cortada"
