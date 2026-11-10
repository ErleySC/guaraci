"""Textos do VIP: o comando `vip` aponta para a selecao aninhada do `run` (V19) e o exemplo explica v08/v11 (V26), lote 4."""
from pathlib import Path

from guaraci import cli_comandos as cmd

RAIZ = Path(__file__).resolve().parent.parent


def test_vip_nao_diz_que_a_selecao_aninhada_nao_existe_e_aponta_o_run():
    from guaraci.vip_estabilidade import TEXTO_VIP
    assert "nao oferece" not in TEXTO_VIP
    assert "guaraci run" in TEXTO_VIP and "aninhada" in TEXTO_VIP and "nao reduz variaveis" in TEXTO_VIP


def test_readme_nao_promete_selecao_aninhada_para_o_comando_vip():
    for nome, frase in (("README.md", "variable selection in `guaraci run` is nested"),
                        ("README.pt-BR.md", "seleção de variáveis do `guaraci run` fica aninhada")):
        txt = (RAIZ / nome).read_text(encoding="utf-8")
        assert frase in txt, nome
        assert "selection is nested inside the validation" not in txt


def test_exemplo_vip_explica_por_que_v08_e_v11_aparecem(tmp_path, capsys):
    assert cmd.subcomando("exemplo", ["vip", str(tmp_path / "ex_vip")]) == 0
    saida = capsys.readouterr().out
    assert "v08" in saida and "v11" in saida and "por acaso" in saida
    assert saida.strip().endswith("regra pratica, nao teste.")
