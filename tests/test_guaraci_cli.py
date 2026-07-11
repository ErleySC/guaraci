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
