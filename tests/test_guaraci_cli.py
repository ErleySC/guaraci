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
def _modo_iniciante_limpo(guaraci_mod):
    """Reseta o estado global de modo antes/depois de cada teste desta secao
    -- _STATE e' um dict de modulo, persiste entre testes sem isolamento."""
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
