"""m-06 (parecer 2026-09-25): a Etapa 4 avisa que a escolha entre metodos usa a mesma validacao."""
from guaraci import selecao_variaveis as sv


def test_aviso_de_escolha_entre_metodos_existe_e_diz_o_que_fazer():
    assert "mesma validacao cruzada" in sv.AVISO_ESCOLHA_ENTRE_METODOS
    assert "holdout" in sv.AVISO_ESCOLHA_ENTRE_METODOS
