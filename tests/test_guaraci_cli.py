"""Testes do CLI guaraci.py: rotulos amigaveis de campos "choice" cujo valor
interno gravado no config (ex.: 'puros'/'todos' do DD-SIMCA) nao e'
autoexplicativo por si so. Import direto (mesmo padrao de
test_predicao.py::test_menu_predicao_cli_end_to_end) -- guaraci.py e' seguro
de importar (guard `if __name__ == "__main__"`, sem I/O bloqueante em nivel
de modulo).
"""

import pytest


@pytest.fixture(scope="module")
def guaraci_mod():
    import guaraci.guaraci as mod
    return mod


def test_modo_ddsimca_get_val_e_autoexplicativo(guaraci_mod):
    """O valor exibido (nao o gravado) deixa claro o MECANISMO de treino --
    'somente puras' (nao so' 'autenticacao') e 'todas as amostras' (nao so'
    'exploratorio'), evitando a confusao de linguagem reportada pelo usuario."""
    cfg_puros = guaraci_mod.Config(ddsimca_treinar_em="puros")
    cfg_todos = guaraci_mod.Config(ddsimca_treinar_em="todos")
    val_puros = guaraci_mod._get_val(cfg_puros, "modo_ddsimca")
    val_todos = guaraci_mod._get_val(cfg_todos, "modo_ddsimca")
    assert "puras" in val_puros.lower()
    assert "todas" in val_todos.lower() or "todos" in val_todos.lower()
    # valor interno gravado no config NAO muda (so' a exibicao)
    assert cfg_puros.ddsimca_treinar_em == "puros"
    assert cfg_todos.ddsimca_treinar_em == "todos"


def test_modo_ddsimca_rotulo_opcao_consistente_com_get_val(guaraci_mod):
    """Regressao: _rotulo_opcao (usado no menu numerado de _editar_campo)
    tinha esquecido o alias de modo_ddsimca e mostrava o valor CRU
    ('puros'/'todos'), inconsistente com _get_val (que ja mostrava o rotulo
    amigavel) -- o usuario via textos diferentes pro mesmo campo dependendo
    de onde olhava no CLI. Ambos devem concordar agora."""
    cfg = guaraci_mod.Config(ddsimca_treinar_em="puros")
    rotulo_valor_atual = guaraci_mod._get_val(cfg, "modo_ddsimca")
    rotulo_no_menu = guaraci_mod._rotulo_opcao("modo_ddsimca", "puros")
    assert rotulo_valor_atual == rotulo_no_menu


def test_modo_ddsimca_set_val_aceita_rotulo_novo_e_valor_cru(guaraci_mod):
    """_set_val aceita tanto o rotulo novo autoexplicativo quanto o valor
    interno cru (compatibilidade) -- grava sempre o codigo interno correto."""
    cfg = guaraci_mod.Config()
    guaraci_mod._set_val(cfg, "modo_ddsimca", "todas as amostras (exploratorio)")
    assert cfg.ddsimca_treinar_em == "todos"

    guaraci_mod._set_val(cfg, "modo_ddsimca", "puros")
    assert cfg.ddsimca_treinar_em == "puros"

    guaraci_mod._set_val(cfg, "modo_ddsimca", "somente puras (autenticacao)")
    assert cfg.ddsimca_treinar_em == "puros"


def test_modo_ddsimca_set_val_rejeita_valor_invalido(guaraci_mod):
    cfg = guaraci_mod.Config()
    with pytest.raises(ValueError):
        guaraci_mod._set_val(cfg, "modo_ddsimca", "valor-que-nao-existe")


# ── Painel de acompanhamento ao vivo (auditoria jul/2026, item 5) ──────────
# _montar_painel_execucao foi extraida de _rodar_pipeline (fecho local) para
# ser testavel sem rodar o pipeline de verdade nem simular entrada
# interativa (_rodar_pipeline pede tag/confirmacao via input()).

def test_montar_painel_execucao_retorna_renderable_sem_erro(guaraci_mod):
    """Renderiza sem lançar exceção e produz texto reconhecível (objetivo,
    percentual, contagem de figuras) quando capturado por um Console."""
    from rich.console import Console
    painel = guaraci_mod._montar_painel_execucao(
        texto_log="[1/7] Validating input\n"
                   "  -> /x/Graficos/fig1_pca_scores.png\n",
        elapsed=10.0, objetivo_rotulo="Classificacao",
        plano_figuras=["a", "b", "c"])
    console = Console(width=100, file=__import__("io").StringIO())
    console.print(painel)
    saida = console.file.getvalue()
    assert "Classificacao" in saida
    assert "fig1_pca_scores" in saida
    assert "1/3" in saida


def test_montar_painel_execucao_mostra_avisos_quando_presentes(guaraci_mod):
    from rich.console import Console
    painel = guaraci_mod._montar_painel_execucao(
        texto_log="[AVISO] Bootstrap VIP: 0 iteracoes validas\n",
        elapsed=5.0, objetivo_rotulo="Quantificacao", plano_figuras=["x"])
    console = Console(width=100, file=__import__("io").StringIO())
    console.print(painel)
    saida = console.file.getvalue()
    assert "Bootstrap VIP" in saida


def test_montar_painel_execucao_sem_avisos_nao_mostra_secao(guaraci_mod):
    from rich.console import Console
    painel = guaraci_mod._montar_painel_execucao(
        texto_log="[1/7] ok\n", elapsed=1.0,
        objetivo_rotulo="Exploratorio", plano_figuras=[])
    console = Console(width=100, file=__import__("io").StringIO())
    console.print(painel)
    saida = console.file.getvalue()
    assert "Avisos" not in saida


def test_painel_nao_estoura_altura_do_terminal(guaraci_mod):
    """Regressao do bug da "tela preta" (2026-08-07).

    `figuras_concluidas` e `avisos_do_log` cresciam sem teto; numa corrida
    completa o painel passava de 35 linhas num terminal de 24. O Live do
    Rich perde o controle do cursor quando o bloco nao cabe na janela e a
    tela fica preta com so' o cursor piscando -- o calculo continua, mas o
    usuario nao ve mais nada. O painel tem que ter altura LIMITADA
    independentemente de quantos avisos/figuras aparecam.
    """
    from rich.console import Console
    figs = "".join(f"  -> C:/x/Graficos/fig_numero_{i}_nome_longo.png\n"
                   for i in range(40))
    avisos = "".join(f"  [AVISO] aviso distinto {i} com texto bem longo "
                     f"que ocupa espaco na horizontal tambem\n"
                     for i in range(100))
    painel = guaraci_mod._montar_painel_execucao(
        texto_log=figs + "[7/7]\n" + avisos, elapsed=600.0,
        objetivo_rotulo="Classificacao",
        plano_figuras=[f"f{i}" for i in range(40)])

    for largura in (80, 100, 120):
        altura = len(Console(width=largura).render_lines(painel, pad=False))
        assert altura <= 24, (
            f"painel com {altura} linhas em largura {largura} — nao cabe num "
            "terminal padrao de 24 linhas; o bug da tela preta voltou")


def test_painel_indica_quantos_avisos_foram_ocultados(guaraci_mod):
    """Truncar nao pode ESCONDER informacao silenciosamente: o total real
    e quantos ficaram de fora tem que aparecer."""
    from rich.console import Console
    avisos = "".join(f"  [AVISO] problema numero {i}\n" for i in range(30))
    painel = guaraci_mod._montar_painel_execucao(
        texto_log=avisos, elapsed=10.0, objetivo_rotulo="Classificacao",
        plano_figuras=["a"])
    console = Console(width=110, file=__import__("io").StringIO())
    console.print(painel)
    saida = console.file.getvalue()
    assert "(30)" in saida, "total real de avisos sumiu do painel"
    assert "+26" in saida, "contador de avisos ocultos ausente"
    assert "problema numero 29" in saida, "aviso mais recente deveria aparecer"


def test_console_sem_pin_engole_saida_durante_redirect_global(guaraci_mod):
    """Reproduz a CAUSA RAIZ do bug da "tela preta" (2026-08-07/08), isolada
    do resto do CLI.

    `guaraci_theme.console` e' construido sem `file=`, entao `Console.file`
    resolve `sys.stdout` DINAMICAMENTE a cada escrita (rich/console.py:
    `self._file or sys.stdout`). `contextlib.redirect_stdout` troca
    `sys.stdout` GLOBALMENTE no processo, nao por thread -- entao enquanto
    uma thread de trabalho segura esse redirect (como `_run()` faz durante
    toda a execucao do pipeline), qualquer `console.print()` do thread
    principal (como o `Live` do painel) escreve no MESMO buffer
    redirecionado, nao no terminal. O painel nunca aparecia atualizado --
    nao porque travasse, mas porque escrevia no lugar errado o tempo todo.

    Sem pin (`console._file = None`, o estado por default), com `sys.stdout`
    redirecionado por outra thread, a escrita do `console.print()` do thread
    principal tem que ir parar no buffer redirecionado, nao no "terminal".
    """
    import contextlib
    import io
    import sys as _sys
    import threading
    import time

    console = guaraci_mod.console
    original_file = console._file
    terminal = io.StringIO()
    logger_buf = io.StringIO()
    real_stdout = _sys.stdout
    try:
        console._file = None        # comportamento default: resolve stdout
        _sys.stdout = terminal      # "terminal" antes do redirect comecar

        def trabalho():
            with contextlib.redirect_stdout(logger_buf):
                time.sleep(0.05)

        thr = threading.Thread(target=trabalho)
        thr.start()
        time.sleep(0.01)            # garante que o redirect ja esta ativo
        console.print("progresso")
        thr.join()

        assert "progresso" in logger_buf.getvalue(), (
            "premissa do teste nao se confirmou — sem pin, a escrita deveria "
            "ter sido engolida pelo buffer redirecionado")
        assert "progresso" not in terminal.getvalue()
    finally:
        console._file = original_file
        _sys.stdout = real_stdout


def test_pin_console_file_impede_saida_de_ser_engolida(guaraci_mod):
    """Com `console._file` FIXADO na referencia real (o fix aplicado em
    `_rodar_pipeline`), a escrita chega ao destino certo mesmo com um
    redirect global de `sys.stdout` ativo em outra thread."""
    import contextlib
    import io
    import sys as _sys
    import threading
    import time

    console = guaraci_mod.console
    original_file = console._file
    real_out = io.StringIO()
    logger_buf = io.StringIO()
    try:
        console._file = real_out   # <-- o fix: pin na referencia real

        def trabalho():
            with contextlib.redirect_stdout(logger_buf):
                time.sleep(0.05)

        _sys.stdout = logger_buf   # como ficaria durante a execucao real
        thr = threading.Thread(target=trabalho)
        thr.start()
        console.print("progresso")
        thr.join()

        assert "progresso" in real_out.getvalue(), (
            "saida nao chegou ao terminal 'real' mesmo com console._file fixado")
        assert "progresso" not in logger_buf.getvalue(), (
            "saida vazou para o buffer de log — pin nao esta protegendo")
    finally:
        console._file = original_file
        _sys.stdout = _sys.__stdout__


def test_montar_painel_execucao_progresso_zero_sem_log(guaraci_mod):
    """Sem nenhuma linha de progresso ainda (inicio da execucao), nao
    lanca excecao e mostra ETA como 'calculando' em vez de dividir por zero."""
    painel = guaraci_mod._montar_painel_execucao(
        texto_log="", elapsed=0.5, objetivo_rotulo="Classificacao",
        plano_figuras=["a"])
    assert painel is not None


# ── Presets por objetivo cientifico (CLAUDE.md secao 6 / auditoria 2026-07-12,
#    item "Modo Iniciante/Avancado + presets"): "Explorar Dados"/"Autenticar
#    Pureza"/"Quantificar Teor" escolhem O QUE analisar, sem o usuario ter
#    que entender nivel/objetivo primeiro. Sem sufixo N1/N2/N3 no nome (P8).
@pytest.mark.parametrize("pname,nivel_esperado,objetivo_esperado", [
    ("Explorar Dados",    "N1", "exploratorio"),
    ("Autenticar Pureza", "N2", "auto"),
    ("Quantificar Teor",  "N3", "auto"),
])
def test_presets_objetivo_existem_e_tem_descricao_bilingue(
        guaraci_mod, pname, nivel_esperado, objetivo_esperado):
    assert pname in guaraci_mod.PROFILES
    assert guaraci_mod.PROFILES[pname]["nivel"] == nivel_esperado
    assert guaraci_mod.PROFILES[pname]["objetivo"] == objetivo_esperado
    for lang in ("PT", "EN"):
        assert guaraci_mod.PROFILE_DESC[pname][lang].strip()
        assert guaraci_mod.PROFILE_KEY_SUMMARY[pname][lang].strip()


@pytest.mark.parametrize("pname,attr,valor_esperado", [
    ("Explorar Dados",    "nivel", "N1"),
    ("Explorar Dados",    "objetivo", "exploratorio"),
    ("Explorar Dados",    "executar_ddsimca", False),
    ("Autenticar Pureza", "nivel", "N2"),
    ("Autenticar Pureza", "executar_ddsimca", True),
    ("Autenticar Pureza", "ddsimca_treinar_em", "puros"),
    ("Quantificar Teor",  "nivel", "N3"),
    ("Quantificar Teor",  "executar_ddsimca", False),
])
def test_preset_objetivo_aplica_no_config_via_spec(
        guaraci_mod, pname, attr, valor_esperado):
    """Mesma resolucao de chave->attr usada por menu_perfis._aplicar
    (_SPEC_BY_KEY): confirma que aplicar o preset grava o ATRIBUTO REAL
    do Config, nao so' a chave amigavel do dict PROFILES."""
    cfg = guaraci_mod.Config()
    pdata = guaraci_mod.PROFILES[pname]
    for k, v in pdata.items():
        if k.startswith("_"):
            continue
        sp = guaraci_mod._SPEC_BY_KEY.get(k)
        if sp:
            setattr(cfg, sp["attr"], v)
    assert getattr(cfg, attr) == valor_esperado


# ── Modo Iniciante/Avancado (CLAUDE.md secao 6 / auditoria 2026-07-12) ───────
@pytest.fixture(autouse=False)
def _modo_iniciante_limpo(guaraci_mod, monkeypatch, tmp_path):
    """Reseta o estado global de modo antes/depois de cada teste desta secao
    -- _STATE e' um dict de modulo, persiste entre testes sem isolamento.

    _MODO_FLAG tambem e' redirecionado para tmp_path: _toggle_modo_usuario()
    grava em disco de verdade (_set_modo_usuario), e sem isso o teste
    escrevia em _USER_DIR (~/.guaraci por padrao) -- o HOME real de quem
    roda os testes, achado ao mexer em _CFG_PATH/_LANG_FLAG (2026-08-07)."""
    monkeypatch.setattr(guaraci_mod, "_USER_DIR", tmp_path)
    monkeypatch.setattr(guaraci_mod, "_MODO_FLAG", tmp_path / ".cli_modo_usuario")
    anterior = guaraci_mod._modo_usuario()
    guaraci_mod._STATE["modo_usuario"] = "iniciante"
    yield
    guaraci_mod._STATE["modo_usuario"] = anterior


def test_toggle_modo_usuario_alterna_iniciante_avancado(guaraci_mod, _modo_iniciante_limpo):
    assert guaraci_mod._modo_usuario() == "iniciante"
    assert guaraci_mod._toggle_modo_usuario() == "avancado"
    assert guaraci_mod._modo_usuario() == "avancado"
    assert guaraci_mod._toggle_modo_usuario() == "iniciante"


def test_print_submenu_compact_esconde_avancados_no_modo_iniciante(
        guaraci_mod, _modo_iniciante_limpo):
    """No modo Iniciante, sem 'mostrar_avancado', os campos em
    campos_avancados NAO aparecem na lista retornada (que e' a fonte da
    verdade p/ indexacao numerica do menu)."""
    cfg = guaraci_mod.Config()
    fields = ["nivel", "max_lvs", "opls_da", "ddsimca"]
    visiveis = guaraci_mod._print_submenu_compact(
        "t", "d", fields, cfg,
        campos_avancados={"opls_da", "ddsimca"}, mostrar_avancado=False)
    assert visiveis == ["nivel", "max_lvs"]


def test_print_submenu_compact_revela_avancados_quando_pedido(
        guaraci_mod, _modo_iniciante_limpo):
    """Com mostrar_avancado=True (usuario apertou [V] naquela visita ao
    menu), a lista completa volta a aparecer, mesmo em modo Iniciante."""
    cfg = guaraci_mod.Config()
    fields = ["nivel", "max_lvs", "opls_da", "ddsimca"]
    visiveis = guaraci_mod._print_submenu_compact(
        "t", "d", fields, cfg,
        campos_avancados={"opls_da", "ddsimca"}, mostrar_avancado=True)
    assert visiveis == fields


def test_print_submenu_compact_modo_avancado_ignora_ocultacao(
        guaraci_mod, _modo_iniciante_limpo):
    """No modo Avancado (sessao inteira), campos_avancados nao esconde nada
    -- so' faz sentido filtrar quando o usuario esta no modo Iniciante."""
    guaraci_mod._STATE["modo_usuario"] = "avancado"
    cfg = guaraci_mod.Config()
    fields = ["nivel", "max_lvs", "opls_da", "ddsimca"]
    visiveis = guaraci_mod._print_submenu_compact(
        "t", "d", fields, cfg,
        campos_avancados={"opls_da", "ddsimca"}, mostrar_avancado=False)
    assert visiveis == fields


def test_print_submenu_compact_sem_campos_avancados_nunca_filtra(
        guaraci_mod, _modo_iniciante_limpo):
    """Compatibilidade: chamadores que nao passam campos_avancados (a
    maioria dos menus existentes) continuam vendo TODOS os campos --
    comportamento identico ao de antes desta feature."""
    cfg = guaraci_mod.Config()
    fields = ["pre_processamento", "comparar_pre_processamentos"]
    visiveis = guaraci_mod._print_submenu_compact("t", "d", fields, cfg)
    assert visiveis == fields


# ── _estimar_tempo ───────────────────────────────────────────────────────────
# Existe para o usuario nao apertar [R] sem saber se espera 5 min ou 3 horas.
# E' ESTIMATIVA de ordem de grandeza — os testes travam MONOTONICIDADE e
# formato, nunca um valor exato (que dependeria de hardware).

def _min_inferior(txt: str) -> float:
    """Limite inferior da estimativa, sempre em MINUTOS.

    Normalizar a unidade e' obrigatorio: "~0.4-1.0 h" e' MAIOR que
    "~16-40 min", mas a comparacao ingenua dos numeros crus diria o oposto.
    """
    import re
    m = re.search(r"([\d.]+)", txt)
    if not m:
        return 0.0
    valor = float(m.group(1))
    return valor * 60.0 if "h" in txt else valor


def test_estimar_tempo_sem_amostras_retorna_none(guaraci_mod):
    """Sem base para estimar, nao inventa um numero."""
    cfg = guaraci_mod.Config()
    assert guaraci_mod._estimar_tempo(cfg, 0) is None
    assert guaraci_mod._estimar_tempo(cfg, None) is None


def test_estimar_tempo_cresce_com_o_numero_de_amostras(guaraci_mod):
    """Propriedade basica: mais amostras nao pode estimar MENOS tempo."""
    cfg = guaraci_mod.Config()
    cfg.n_permutacoes = 200
    valores = [guaraci_mod._estimar_tempo(cfg, n) for n in (200, 1000, 5000)]
    assert all(v is not None for v in valores)
    # Compara pelo limite inferior numerico extraido do texto ("~16-40 min").
    assert _min_inferior(valores[0]) <= _min_inferior(valores[1])


def test_paralelismo_reduz_a_estimativa(guaraci_mod):
    """n_jobs_permutacao e' o unico campo que muda a ordem de grandeza sem
    mudar nenhum resultado — a estimativa precisa refletir isso, senao a
    dica que o checklist exibe seria falsa."""
    cfg = guaraci_mod.Config()
    cfg.n_permutacoes = 200
    cfg.n_jobs_permutacao = 1
    seq = guaraci_mod._estimar_tempo(cfg, 1673)
    cfg.n_jobs_permutacao = 4
    par = guaraci_mod._estimar_tempo(cfg, 1673)
    assert _min_inferior(par) < _min_inferior(seq), \
        f"paralelo ({par}) deveria ser menor que sequencial ({seq})"


def test_modulos_pesados_aumentam_a_estimativa(guaraci_mod):
    """Ligar benchmark/SHAP/AG tem de aparecer no numero — sao justamente os
    que transformam minutos em horas."""
    base = guaraci_mod.Config()
    base.n_permutacoes = 200
    t_base = _min_inferior(guaraci_mod._estimar_tempo(base, 1673))
    for attr in ("executar_benchmark", "executar_shap", "executar_ag"):
        cfg = guaraci_mod.Config()
        cfg.n_permutacoes = 200
        setattr(cfg, attr, True)
        assert _min_inferior(guaraci_mod._estimar_tempo(cfg, 1673)) > t_base, \
            f"{attr}=True deveria aumentar a estimativa"


def test_estimativa_usa_faixa_nunca_valor_exato(guaraci_mod):
    """Formato: a incerteza precisa estar visivel. Um numero seco seria lido
    como promessa."""
    cfg = guaraci_mod.Config()
    cfg.n_permutacoes = 200
    txt = guaraci_mod._estimar_tempo(cfg, 1673)
    assert ("-" in txt and ("min" in txt or "h" in txt)) or txt.startswith("<"), \
        f"esperava faixa (ex.: '~16-40 min'), veio {txt!r}"


# ── i18n: vazamento de idioma e markup cru ───────────────────────────────────
# Achados em 2026-08-06 numa tela real do usuario em modo EN: "[g]Yes[/g]"
# impresso LITERALMENTE (Text() em vez de Text.from_markup) e varios textos
# ainda em portugues. Os testes abaixo travam os dois problemas de uma vez,
# renderizando os paineis de verdade e inspecionando a saida.

def _render(guaraci_mod, fn, cfg, lang):
    """Renderiza um painel no idioma pedido e devolve o texto SEM cores.

    `console.file` e' uma PROPERTY: le' `self._file or sys.stdout`
    (resolvida dinamicamente a cada chamada). Salvar `console.file` guarda o
    stdout JA' RESOLVIDO daquele instante; reatribuir esse valor na volta
    prende `_file` nesse objeto concreto para sempre, em vez de devolve-lo a
    `None` (dinamico) -- quebra o `capsys` de QUALQUER teste que rodar depois
    deste, no processo inteiro, nao so' o painel testado aqui. Precisa
    restaurar o atributo `_file` (que pode ser `None`), nao a property.
    """
    import io
    import re as _re
    import contextlib
    lang_antes = guaraci_mod._STATE["lang"]
    guaraci_mod._STATE["lang"] = lang
    buf = io.StringIO()
    file_antes = guaraci_mod.console._file
    try:
        guaraci_mod.console.file = buf
        with contextlib.redirect_stdout(io.StringIO()):
            fn(cfg)
    finally:
        guaraci_mod.console._file = file_antes
        guaraci_mod._STATE["lang"] = lang_antes
    return _re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())


@pytest.mark.parametrize("lang", ["PT", "EN"])
@pytest.mark.parametrize("ligado", [True, False])
def test_paineis_nao_imprimem_markup_cru(guaraci_mod, lang, ligado):
    """`Text("[g]Yes[/g]")` trata a string como LITERAL e vaza a tag para a
    tela. Tem de ser Text.from_markup. Vale nos DOIS idiomas — o bug original
    afetava "[g]Sim[/g]" igualmente, so' foi notado em ingles.

    Parametrizado por `ligado` porque os ramos True/False sao strings
    DIFERENTES no codigo: um teste so' com o Config() padrao (tudo False)
    exercitaria apenas o ramo "Nao/No" e passaria com o ramo "Sim/Yes"
    quebrado. Foi exatamente o que aconteceu ao validar este teste.
    """
    import re as _re
    cfg = guaraci_mod.Config()
    for attr in ("executar_opls", "executar_ddsimca", "executar_benchmark",
                 "executar_monte_carlo", "executar_shap"):
        setattr(cfg, attr, ligado)
    for fn in (guaraci_mod._print_resumo, guaraci_mod._print_checklist):
        txt = _render(guaraci_mod, fn, cfg, lang)
        cru = _re.findall(r"\[/?(?:g|m|err|b|bold)\]", txt)
        assert not cru, (f"{fn.__name__} [{lang}, ligado={ligado}] imprimiu "
                          f"markup cru: {sorted(set(cru))}")


def test_painel_em_ingles_nao_vaza_portugues(guaraci_mod):
    """Em modo EN nenhuma palavra inequivocamente portuguesa pode aparecer.

    Lista curta e especifica de proposito: termos que so' existem em PT e que
    ja' vazaram de verdade (rotulo de nivel, "Modo", "(automatico)"), para o
    teste falhar por motivo real e nao por acidente de vocabulario comum aos
    dois idiomas.
    """
    import re as _re
    cfg = guaraci_mod.Config()
    proibidas = _re.compile(
        r"\b(Modo|automatico|Classificacao|Discriminacao|Quantificacao|"
        r"Configuracao|Cientifica|Nivel|Tecnica|Ativo|Nenhum)\b")
    for fn in (guaraci_mod._print_resumo, guaraci_mod._print_checklist):
        txt = _render(guaraci_mod, fn, cfg, "EN")
        achados = sorted(set(proibidas.findall(txt)))
        assert not achados, f"{fn.__name__} vazou portugues em modo EN: {achados}"


def test_rotulo_de_nivel_traduz_mas_nao_muda_o_arquivo_de_saida(guaraci_mod):
    """O rotulo do nivel traduz na TELA, mas `pq._NIVEL_NOME` (que alimenta o
    resumo_modelo.txt) continua em portugues — traduzir la' faria um ARQUIVO
    DE SAIDA mudar conforme o idioma da interface, quebrando a comparabilidade
    entre execucoes."""
    lang_antes = guaraci_mod._STATE["lang"]
    try:
        guaraci_mod._STATE["lang"] = "EN"
        assert "Species" in guaraci_mod._rotulo_opcao("nivel", "N1")
        guaraci_mod._STATE["lang"] = "PT"
        assert "Classificacao" in guaraci_mod._rotulo_opcao("nivel", "N1")
    finally:
        guaraci_mod._STATE["lang"] = lang_antes
    # Fonte unica do arquivo de saida: sempre PT, independente do idioma.
    assert guaraci_mod.pq._NIVEL_NOME["N1"] == "Classificacao por especie"


def test_todo_rotulo_de_nivel_tem_traducao(guaraci_mod):
    """Se alguem adicionar um nivel N4, a traducao nao pode ficar para tras."""
    assert set(guaraci_mod._NIVEL_NOME_EN) == set(guaraci_mod.pq._NIVEL_NOME)


# ── Padronizacao de campos booleanos (achado 2026-08-06) ─────────────────────
# Antes: os 20 campos bool do _CONFIG_SPEC caiam em texto livre (`opcoes=None`
# no spec), e _coagir_valor so' reconhecia um punhado de palavras magicas
# ("true"/"sim"/"1"/"yes"/"s"/"v") como True -- QUALQUER outra coisa virava
# False SEM AVISO. Digitar "y" (comum em outros softwares) virava False
# silenciosamente; um erro de digitacao nunca dava mensagem de erro. Agora
# todo campo bool e' escolha numerada [1]=Sim/Yes [2]=Nao/No, igual a
# qualquer outro campo de opcao (nivel, modo_ddsimca) -- "modelo padrao"
# pedido pelo usuario.

def test_editar_campo_bool_escolha_1_liga(guaraci_mod, monkeypatch):
    cfg = guaraci_mod.Config(mostrar_elipses_grupo=False)
    respostas = iter(["1"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
    ok = guaraci_mod._editar_campo(cfg, "figuras_mostrar_elipses")
    assert ok is True
    assert cfg.mostrar_elipses_grupo is True


def test_editar_campo_bool_escolha_2_desliga(guaraci_mod, monkeypatch):
    cfg = guaraci_mod.Config(mostrar_elipses_grupo=True)
    respostas = iter(["2"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
    ok = guaraci_mod._editar_campo(cfg, "figuras_mostrar_elipses")
    assert ok is True
    assert cfg.mostrar_elipses_grupo is False


def test_editar_campo_bool_enter_mantem_valor(guaraci_mod, monkeypatch):
    cfg = guaraci_mod.Config(mostrar_elipses_grupo=True)
    respostas = iter([""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
    ok = guaraci_mod._editar_campo(cfg, "figuras_mostrar_elipses")
    assert ok is False
    assert cfg.mostrar_elipses_grupo is True


@pytest.mark.parametrize("entrada", ["9", "abc", "y", "true", "sim"])
def test_editar_campo_bool_entrada_fora_de_1_2_e_rejeitada(guaraci_mod, monkeypatch, entrada):
    """O bug original: digitar "y" (ou qualquer coisa fora do vocabulario
    magico) virava False SEM AVISO. Agora qualquer coisa que nao seja "1"
    ou "2" e' rejeitada explicitamente e o valor NAO muda -- nem "y", nem
    palavras que faziam parte do vocabulario antigo ("true"/"sim"), porque
    a escolha agora e' estritamente numerada."""
    cfg = guaraci_mod.Config(mostrar_elipses_grupo=False)
    respostas = iter([entrada])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
    ok = guaraci_mod._editar_campo(cfg, "figuras_mostrar_elipses")
    assert ok is False
    assert cfg.mostrar_elipses_grupo is False, (
        f"entrada {entrada!r} nao devia ter alterado o campo")


def test_editar_campo_bool_com_confirmacao_analitica_aceita(guaraci_mod, monkeypatch):
    """Campo ANALITICO (ddsimca) pede confirmacao extra -- fluxo completo:
    escolhe [1] e confirma com 's'."""
    cfg = guaraci_mod.Config(executar_ddsimca=False)
    respostas = iter(["1", "s"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
    ok = guaraci_mod._editar_campo(cfg, "ddsimca")
    assert ok is True
    assert cfg.executar_ddsimca is True


def test_editar_campo_bool_com_confirmacao_analitica_recusada(guaraci_mod, monkeypatch):
    """Mesmo fluxo, mas recusando a confirmacao -- o valor NAO muda."""
    cfg = guaraci_mod.Config(executar_ddsimca=False)
    respostas = iter(["1", "n"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
    ok = guaraci_mod._editar_campo(cfg, "ddsimca")
    assert ok is False
    assert cfg.executar_ddsimca is False


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_editar_campo_bool_grava_valor_canonico_independente_do_idioma(
        guaraci_mod, monkeypatch, lang):
    """O valor GRAVADO no config (true/false internos) nao pode depender do
    idioma da interface -- so' o ROTULO exibido muda."""
    lang_antes = guaraci_mod._STATE["lang"]
    try:
        guaraci_mod._STATE["lang"] = lang
        cfg = guaraci_mod.Config(mostrar_elipses_grupo=False)
        respostas = iter(["1"])
        monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
        guaraci_mod._editar_campo(cfg, "figuras_mostrar_elipses")
        assert cfg.mostrar_elipses_grupo is True
    finally:
        guaraci_mod._STATE["lang"] = lang_antes


# ── Reset automatico de toggles inertes ao trocar "nivel" (2026-08-06) ───────
# Pedido do usuario: "quando mudo de modo... continua ativado a dd simca e
# semelhantes... gostaria que ao mudar o N, mudasse as opcoes de modos como
# esse que nao agrega a analise". A regra espelha EXATAMENTE o que
# pipeline.executar()/modos_analise.py decidem (ver docstring de
# _ajustar_toggles_por_nivel), nao uma aproximacao da UI.

def test_ajustar_toggles_ddsimca_desliga_em_n1(guaraci_mod):
    """DD-SIMCA em N1 e' sempre ignorado por pipeline.executar() (aviso, nao
    bloqueio no Config) -- a UI passa a refletir isso desligando o toggle."""
    cfg = guaraci_mod.Config(nivel="N1", executar_ddsimca=True)
    mudou = guaraci_mod._ajustar_toggles_por_nivel(cfg)
    assert cfg.executar_ddsimca is False
    assert "ddsimca" in mudou


def test_ajustar_toggles_ddsimca_liga_em_n2(guaraci_mod):
    """N2 forca DD-SIMCA internamente em pipeline.executar() (linha ~1105)
    independente do toggle -- a UI passa a mostrar True de antemao, refletindo
    o que vai acontecer de qualquer forma, em vez de mostrar 'desligado' e
    surpreender o usuario com o log dizendo o contrario."""
    cfg = guaraci_mod.Config(nivel="N2", executar_ddsimca=False)
    mudou = guaraci_mod._ajustar_toggles_por_nivel(cfg)
    assert cfg.executar_ddsimca is True
    assert "ddsimca" in mudou


def test_ajustar_toggles_ddsimca_desliga_em_n3(guaraci_mod):
    """N3 (Quantificacao): deve_gerar(cfg,'ddsimca') e' False (figura so'
    pertence a Classificacao) -- toggle inerte, igual N1, mas SEM aviso no
    pipeline (achado: N1 tem log explicito, N3 nao tinha nenhum)."""
    cfg = guaraci_mod.Config(nivel="N3", executar_ddsimca=True)
    mudou = guaraci_mod._ajustar_toggles_por_nivel(cfg)
    assert cfg.executar_ddsimca is False
    assert "ddsimca" in mudou


def test_ajustar_toggles_classificacao_only_desligam_fora_de_classificacao(guaraci_mod):
    """opls_da, etapa4, comparar_pre_processamentos, teste_wold,
    teste_cv_anova, teste_martens, benchmark, monte_carlo, shap_benchmark:
    todos pertencem exclusivamente a objetivo=Classificacao (ver
    modos_analise._FIG_OBJETIVOS) -- inertes em N3."""
    cfg = guaraci_mod.Config(
        nivel="N3", executar_opls=True, executar_etapa4=True,
        comparar_pipelines=True, executar_wold=True, executar_cv_anova=True,
        executar_martens=True, executar_benchmark=True,
        executar_monte_carlo=True, executar_shap=True,
    )
    mudou = guaraci_mod._ajustar_toggles_por_nivel(cfg)
    assert cfg.executar_opls is False
    assert cfg.executar_etapa4 is False
    assert cfg.comparar_pipelines is False
    assert cfg.executar_wold is False
    assert cfg.executar_cv_anova is False
    assert cfg.executar_martens is False
    assert cfg.executar_benchmark is False
    assert cfg.executar_monte_carlo is False
    assert cfg.executar_shap is False
    assert set(mudou) >= {
        "opls_da", "selecao_variaveis_etapa4", "comparar_pre_processamentos",
        "teste_wold", "teste_cv_anova", "teste_martens", "benchmark",
        "monte_carlo", "shap_benchmark",
    }


def test_ajustar_toggles_classificacao_only_preservados_em_n1_e_n2(guaraci_mod):
    """Contraparte positiva: os mesmos toggles NAO podem ser mexidos em N1/N2
    -- la' eles SAO pertinentes (objetivo=Classificacao)."""
    for nivel in ("N1", "N2"):
        cfg = guaraci_mod.Config(nivel=nivel, executar_opls=True,
                                 executar_benchmark=True)
        guaraci_mod._ajustar_toggles_por_nivel(cfg)
        assert cfg.executar_opls is True, f"opls_da nao devia mudar em {nivel}"
        assert cfg.executar_benchmark is True, f"benchmark nao devia mudar em {nivel}"


def test_ajustar_toggles_benchmark_regressao_desliga_fora_de_quantificacao(guaraci_mod):
    cfg = guaraci_mod.Config(nivel="N1", executar_benchmark_regressao=True)
    mudou = guaraci_mod._ajustar_toggles_por_nivel(cfg)
    assert cfg.executar_benchmark_regressao is False
    assert "benchmark_regressao" in mudou


def test_ajustar_toggles_benchmark_regressao_preservado_em_n3(guaraci_mod):
    cfg = guaraci_mod.Config(nivel="N3", executar_benchmark_regressao=True)
    mudou = guaraci_mod._ajustar_toggles_por_nivel(cfg)
    assert cfg.executar_benchmark_regressao is True
    assert "benchmark_regressao" not in mudou


def test_ajustar_toggles_respeita_objetivo_explicito_sobre_nivel(guaraci_mod):
    """Se o usuario sobrepoe cfg.objetivo explicitamente (avancado), a regra
    segue o OBJETIVO resolvido, nao uma leitura ingenua do nivel -- mesma
    precedencia de modos_analise.resolver_objetivo()."""
    cfg = guaraci_mod.Config(nivel="N3", objetivo="classificacao",
                             executar_opls=True)
    guaraci_mod._ajustar_toggles_por_nivel(cfg)
    assert cfg.executar_opls is True, (
        "objetivo=classificacao explicito deveria preservar toggle "
        "de classificacao mesmo em nivel=N3")


def test_ajustar_toggles_nao_mexe_quando_ja_esta_correto(guaraci_mod):
    """Idempotencia: se os toggles ja estao no estado certo para o nivel,
    a lista de 'mudou' fica vazia -- nao reporta ajuste que nao aconteceu."""
    cfg = guaraci_mod.Config(nivel="N1", executar_ddsimca=False,
                             executar_opls=False)
    mudou = guaraci_mod._ajustar_toggles_por_nivel(cfg)
    assert mudou == []


def test_editar_campo_nivel_dispara_ajuste_e_avisa(guaraci_mod, monkeypatch):
    """Fluxo completo via _editar_campo: trocar nivel N2->N1 (que tem
    confirmacao ANALITICA) desliga ddsimca automaticamente e a mensagem de
    aviso e' impressa."""
    cfg = guaraci_mod.Config(nivel="N2", executar_ddsimca=True)
    respostas = iter(["1", "s"])   # [1]=N1 no menu de opcoes; 's'=confirma
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
    from rich.console import Console
    import io
    buf_console = Console(file=io.StringIO(), force_terminal=False, width=100)
    monkeypatch.setattr(guaraci_mod, "console", buf_console)
    ok = guaraci_mod._editar_campo(cfg, "nivel")
    assert ok is True
    assert cfg.nivel == "N1"
    assert cfg.executar_ddsimca is False
    saida = buf_console.file.getvalue()
    assert "DD-SIMCA" in saida or "ddsimca" in saida.lower()


def test_editar_campo_nivel_sem_mudanca_nao_mexe_em_nada(guaraci_mod, monkeypatch):
    """Reescolher o MESMO nivel (Enter=manter, ou escolher o numero atual)
    nao deve reportar nenhum ajuste."""
    cfg = guaraci_mod.Config(nivel="N1", executar_ddsimca=False)
    respostas = iter([""])   # Enter = mantem
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
    ok = guaraci_mod._editar_campo(cfg, "nivel")
    assert ok is False
    assert cfg.nivel == "N1"
    assert cfg.executar_ddsimca is False


# ── Alcancabilidade de campos do _CONFIG_SPEC (achado 2026-08-06) ────────────
# n_jobs_permutacao (e mais 6 campos: teste_martens, benchmark_regressao,
# figuras_detalhadas, imagem_incluir_textura, selecao_ag, selecao_spa,
# objetivo) existiam no Config/_CONFIG_SPEC/HELP_DB, mas nunca tinham sido
# colocados em NENHUM menu -- so' editaveis a mao no YAML. O usuario so'
# descobriu porque o assistente recomendou mudar um campo que era, na
# pratica, inacessivel pela interface. Teste sistemico para a classe inteira
# do bug, nao so' o campo que apareceu desta vez.

def test_todo_campo_do_spec_e_alcancavel_por_algum_menu(guaraci_mod):
    """Varre toda funcao menu_* via AST (nao regex sobre texto de origem --
    a primeira tentativa de diagnostico teve um bug proprio nisso) e
    confirma que toda chave do _CONFIG_SPEC aparece como string literal em
    pelo menos uma delas, exceto os aliases com caminho de edicao proprio
    documentados abaixo."""
    import ast
    import inspect

    # Aliases com caminho de edicao PROPRIO e melhor que o generico
    # _editar_campo -- nao sao bugs, sao design deliberado:
    #   nome_execucao: mesmo atributo de "tag" (attr="tag" no spec), editado
    #   pelo prompt dedicado "Identificador atual / Novo identificador" em
    #   _rodar_pipeline()/main(), nao pelo menu numerado generico.
    ALIASES_COM_CAMINHO_PROPRIO = {"nome_execucao"}

    todas_chaves = set(guaraci_mod._SPEC_BY_KEY.keys())
    alcancaveis = set()
    for nome in dir(guaraci_mod):
        if not nome.startswith("menu_"):
            continue
        fn = getattr(guaraci_mod, nome)
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in todas_chaves:
                    alcancaveis.add(node.value)

    faltando = todas_chaves - alcancaveis - ALIASES_COM_CAMINHO_PROPRIO
    assert not faltando, (
        f"Campos no _CONFIG_SPEC sem NENHUM menu que os alcance: "
        f"{sorted(faltando)} -- editaveis so' a mao no YAML. Adicione a um "
        f"menu_* existente ou a ALIASES_COM_CAMINHO_PROPRIO (com "
        f"justificativa) se houver caminho de edicao alternativo legitimo.")


def test_nome_execucao_alias_tem_mesmo_attr_que_tag(guaraci_mod):
    """Trava a premissa do whitelist acima: nome_execucao SO' pode ficar de
    fora do teste de alcancabilidade porque aponta pro MESMO atributo que
    "tag" (editado por um prompt dedicado). Se algum dia apontar para outro
    atributo, deixa de ser um alias legitimo e o teste acima precisa
    cobri-lo de novo."""
    spec_tag = guaraci_mod._SPEC_BY_KEY.get("tag")
    spec_nome_exec = guaraci_mod._SPEC_BY_KEY.get("nome_execucao")
    assert spec_tag is not None and spec_nome_exec is not None
    assert spec_tag["attr"] == spec_nome_exec["attr"] == "tag"


# ── main(): saida graciosa em EOF de stdin (achado 2026-08-07) ─────────────
# `_input()` engole EOFError/KeyboardInterrupt internamente e devolve "" --
# no loop principal de main(), "" nao bate com NENHUMA opcao de menu, entao
# cai no ramo "invalida" + _pause() (tambem EOF-safe) e o loop volta a
# chamar cls() e ler de novo, sempre "" de novo em EOF permanente. O
# try/except (EOFError, KeyboardInterrupt) que EXISTIA ao redor da leitura
# nunca disparava, porque a excecao ja tinha sido engolida por _input()
# antes de chegar la -- girava para sempre (reproduzido: >350 redesenhos em
# 8s sem terminar, chamando os.system("cls") a cada iteracao). Corrigido
# trocando a chamada por input() direto nesse UNICO ponto, deixando o
# EOFError propagar ate o handler que ja existia.

def test_main_sai_rapido_com_eof_no_stdin(guaraci_mod, monkeypatch, tmp_path):
    """Propriedade que falhava antes da correcao: main() tem que RETORNAR
    (nao girar para sempre) quando input() sempre levanta EOFError -- o
    mesmo efeito de um pipe/redirecionamento de stdin vazio, ou uma sessao
    interativa que perde a conexao."""
    import time

    def _input_eof(*_a, **_kw):
        raise EOFError()

    monkeypatch.setattr("builtins.input", _input_eof)
    # cls() spawna um subprocesso via os.system a cada iteracao -- sem
    # mockar, o teste ficaria lento e poluiria a saida do pytest sem
    # testar nada a mais sobre a correcao.
    monkeypatch.setattr(guaraci_mod, "cls", lambda: None)
    # Evita escrever/migrar estado no HOME real de quem roda os testes
    # (_USER_DIR aponta por padrao para ~/.guaraci -- ver _migrar_estado_legado).
    monkeypatch.setattr(guaraci_mod, "_USER_DIR", tmp_path)
    monkeypatch.setattr(guaraci_mod, "_CFG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(guaraci_mod, "_LANG_FLAG", tmp_path / ".cli_wizard_done")
    monkeypatch.setattr(guaraci_mod, "_PERFIS_DIR", tmp_path / "perfis")
    monkeypatch.setattr(guaraci_mod, "_CODIGOS_PATH", tmp_path / "codigos_usuario.json")
    monkeypatch.setattr(guaraci_mod, "_MODO_FLAG", tmp_path / ".cli_modo_usuario")

    inicio = time.monotonic()
    guaraci_mod.main()   # NAO pode travar -- se travar, o teste tambem trava
    duracao = time.monotonic() - inicio
    assert duracao < 5.0, (
        f"main() nao retornou rapido com EOF permanente -- ainda gira? "
        f"({duracao:.2f}s)")


# ── _CFG_PATH/_LANG_FLAG fora do diretorio de instalacao (achado 2026-08-07) ─
# config.yaml/perfis/flags de idioma-modo/codigos de usuario eram gravados
# dentro de _BASE_DIR (o diretorio de INSTALACAO do pacote) -- quebra em
# qualquer instalacao read-only (pip de sistema, Docker, `pip install
# --user` em alguns casos). Movido para _USER_DIR (~/.guaraci), com
# migracao best-effort do estado gravado pela versao anterior.

def test_estado_do_usuario_fica_fora_do_diretorio_de_instalacao(guaraci_mod):
    """Propriedade que falhava antes da correcao: nenhum dos caminhos de
    estado do usuario pode estar DENTRO de _BASE_DIR (o pacote instalado)."""
    base = str(guaraci_mod._BASE_DIR)
    for caminho in (guaraci_mod._CFG_PATH, guaraci_mod._PERFIS_DIR,
                    guaraci_mod._LANG_FLAG, guaraci_mod._CODIGOS_PATH,
                    guaraci_mod._MODO_FLAG):
        assert not str(caminho).startswith(base), (
            f"{caminho} ainda esta dentro do diretorio de instalacao do "
            "pacote -- quebra em instalacao read-only")


def test_migrar_estado_legado_copia_arquivos_que_faltam(guaraci_mod, monkeypatch, tmp_path):
    """Arquivos existentes no local ANTIGO (_BASE_DIR) e ausentes no NOVO
    (_USER_DIR) devem ser copiados -- efeito pratico: quem ja usava o CLI
    antes desta correcao nao perde config/perfis/codigos salvos."""
    base_antigo = tmp_path / "pacote_antigo"
    base_antigo.mkdir()
    (base_antigo / "config.yaml").write_text("nivel: N2\n", encoding="utf-8")
    (base_antigo / ".cli_wizard_done").write_text("EN", encoding="utf-8")
    (base_antigo / "codigos_usuario.json").write_text('{"XYZ": "Teste"}',
                                                       encoding="utf-8")
    perfis_antigo = base_antigo / "perfis"
    perfis_antigo.mkdir()
    (perfis_antigo / "meu_perfil.yaml").write_text("tag: x\n", encoding="utf-8")

    dir_novo = tmp_path / "home_novo" / ".guaraci"
    monkeypatch.setattr(guaraci_mod, "_BASE_DIR", base_antigo)
    monkeypatch.setattr(guaraci_mod, "_USER_DIR", dir_novo)
    monkeypatch.setattr(guaraci_mod, "_CFG_PATH", dir_novo / "config.yaml")
    monkeypatch.setattr(guaraci_mod, "_LANG_FLAG", dir_novo / ".cli_wizard_done")
    monkeypatch.setattr(guaraci_mod, "_CODIGOS_PATH", dir_novo / "codigos_usuario.json")
    monkeypatch.setattr(guaraci_mod, "_MODO_FLAG", dir_novo / ".cli_modo_usuario")
    monkeypatch.setattr(guaraci_mod, "_PERFIS_DIR", dir_novo / "perfis")

    guaraci_mod._migrar_estado_legado()

    assert (dir_novo / "config.yaml").read_text(encoding="utf-8") == "nivel: N2\n"
    assert (dir_novo / ".cli_wizard_done").read_text(encoding="utf-8") == "EN"
    assert '"XYZ"' in (dir_novo / "codigos_usuario.json").read_text(encoding="utf-8")
    assert (dir_novo / "perfis" / "meu_perfil.yaml").exists()
    # arquivo antigo continua intacto -- migracao NUNCA apaga a origem
    assert (base_antigo / "config.yaml").exists()


def test_migrar_estado_legado_nao_sobrescreve_arquivo_ja_existente(guaraci_mod, monkeypatch, tmp_path):
    """Se o NOVO local ja tem um config.yaml (usuario ja rodou apos a
    correcao e mudou algo), a migracao nao pode pisar em cima."""
    base_antigo = tmp_path / "pacote_antigo"
    base_antigo.mkdir()
    (base_antigo / "config.yaml").write_text("nivel: N2\n", encoding="utf-8")

    dir_novo = tmp_path / "home_novo" / ".guaraci"
    dir_novo.mkdir(parents=True)
    (dir_novo / "config.yaml").write_text("nivel: N3\n", encoding="utf-8")

    monkeypatch.setattr(guaraci_mod, "_BASE_DIR", base_antigo)
    monkeypatch.setattr(guaraci_mod, "_USER_DIR", dir_novo)
    monkeypatch.setattr(guaraci_mod, "_CFG_PATH", dir_novo / "config.yaml")
    monkeypatch.setattr(guaraci_mod, "_LANG_FLAG", dir_novo / ".cli_wizard_done")
    monkeypatch.setattr(guaraci_mod, "_CODIGOS_PATH", dir_novo / "codigos_usuario.json")
    monkeypatch.setattr(guaraci_mod, "_MODO_FLAG", dir_novo / ".cli_modo_usuario")
    monkeypatch.setattr(guaraci_mod, "_PERFIS_DIR", dir_novo / "perfis")

    guaraci_mod._migrar_estado_legado()

    assert (dir_novo / "config.yaml").read_text(encoding="utf-8") == "nivel: N3\n"


def test_migrar_estado_legado_sem_arquivos_antigos_nao_lanca(guaraci_mod, monkeypatch, tmp_path):
    """Instalacao nova (nunca rodou a versao antiga) -- nada para migrar,
    nao pode lancar excecao nem criar arquivos vazios."""
    base_antigo = tmp_path / "pacote_sem_nada"
    base_antigo.mkdir()
    dir_novo = tmp_path / "home_novo" / ".guaraci"

    monkeypatch.setattr(guaraci_mod, "_BASE_DIR", base_antigo)
    monkeypatch.setattr(guaraci_mod, "_USER_DIR", dir_novo)
    monkeypatch.setattr(guaraci_mod, "_CFG_PATH", dir_novo / "config.yaml")
    monkeypatch.setattr(guaraci_mod, "_LANG_FLAG", dir_novo / ".cli_wizard_done")
    monkeypatch.setattr(guaraci_mod, "_CODIGOS_PATH", dir_novo / "codigos_usuario.json")
    monkeypatch.setattr(guaraci_mod, "_MODO_FLAG", dir_novo / ".cli_modo_usuario")
    monkeypatch.setattr(guaraci_mod, "_PERFIS_DIR", dir_novo / "perfis")

    guaraci_mod._migrar_estado_legado()   # nao deve lancar

    assert dir_novo.exists()   # _USER_DIR e' criado mesmo sem nada a copiar
    assert not (dir_novo / "config.yaml").exists()
