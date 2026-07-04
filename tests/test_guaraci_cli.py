"""Testes do CLI guaraci.py: rotulos amigaveis de campos "choice" cujo valor
interno gravado no config (ex.: 'puros'/'todos' do DD-SIMCA) nao e'
autoexplicativo por si so. Import direto (mesmo padrao de
test_predicao.py::test_menu_predicao_cli_end_to_end) -- guaraci.py e' seguro
de importar (guard `if __name__ == "__main__"`, sem I/O bloqueante em nivel
de modulo).
"""
import importlib.util
import os

import pytest


@pytest.fixture(scope="module")
def guaraci_mod():
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "guaraci_cli_test_naming", os.path.join(proj_root, "guaraci.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
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
