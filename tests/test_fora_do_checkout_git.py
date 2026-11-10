"""A3-03 (vistoria 2026-10): fora de um checkout git (zip/sdist) os testes que leem o historico do PROPRIO repositorio pulam com o
motivo cadastrado; os demais testes do mesmo arquivo continuam rodando."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

RAIZ = Path(__file__).resolve().parents[1]


def _conftest_raiz():
    spec = importlib.util.spec_from_file_location("_conftest_raiz_para_teste", RAIZ / "conftest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Item(SimpleNamespace):
    def add_marker(self, m):
        self.marcas.append(m)


def _item(arquivo, nome):
    return _Item(fspath=str(RAIZ / "tests" / arquivo), originalname=nome, name=nome, marcas=[])


def test_quem_le_o_historico_do_repositorio_pula_e_o_resto_do_arquivo_roda():
    ct = _conftest_raiz()
    lidos = [_item("test_verificacao_publicada.py", "test_resultado_publicado_e_do_hash_a_publicar"),
             _item("test_verificacao_publicada.py", "test_so_arquivo_gerado_de_qualidade_pode_mudar_depois_da_verificacao"),
             _item("test_medir_vault.py", "test_nota_gerada_em_commit_anterior_ao_relevante_e_defasada"),
             _item("test_medir_vault.py", "test_defasagem_por_carimbo_de_commit")]
    sem_git = [_item("test_medir_vault.py", "test_ids_do_livro_ignora_linha_com_estado_invalido")]
    ct.marcar_fora_do_git(lidos + sem_git, "PULAR")
    assert all(i.marcas == ["PULAR"] for i in lidos)
    assert sem_git[0].marcas == []


def test_entradas_da_lista_existem():
    """Uma entrada que nao casa nenhum arquivo/teste real nao protege nada (renomeacao silenciosa)."""
    ct = _conftest_raiz()
    for entrada in ct._TESTES_QUE_EXIGEM_CHECKOUT_GIT:
        arquivo, _, teste = entrada.partition("::")
        caminho = RAIZ / "tests" / arquivo
        assert caminho.exists(), entrada
        if teste:
            assert f"def {teste}(" in caminho.read_text(encoding="utf-8"), entrada
