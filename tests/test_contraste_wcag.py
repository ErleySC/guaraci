"""M-18b do parecer de 2026-09-25: texto branco da barra lateral e do botao
primario com contraste >= 4,5:1 (WCAG AA) nos dois temas."""
import pytest

from guaraci.design_tokens import TOKENS, contraste, escurecer_para_contraste


@pytest.mark.parametrize("tema", ["light", "dark"])
def test_botao_primario_com_texto_branco_passa_aa(tema):
    assert contraste(TOKENS[tema]["primary"], "#FFFFFF") >= 4.5


@pytest.mark.parametrize("tema", ["light", "dark"])
@pytest.mark.parametrize("ponto", ["primary", "accent", "success"])
def test_degrade_da_barra_lateral_passa_aa(tema, ponto):
    cor = escurecer_para_contraste(TOKENS[tema][ponto])
    assert contraste(cor, "#FFFFFF") >= 4.5


def test_dourado_puro_reprova_e_e_corrigido():
    assert contraste("#FFC100", "#FFFFFF") < 2
    assert contraste(escurecer_para_contraste("#FFC100"), "#FFFFFF") >= 4.5


def test_marcadores_por_forma_ligados_por_padrao():
    """M-18a: a paleta padrao confunde vermelho/verde na deuteranopia; o canal
    de forma e' o que garante a identidade da classe, entao vem ligado."""
    from guaraci.config import Config
    assert Config().show_class_markers is True


def test_estilos_de_linha_distintos_para_ate_8_classes():
    """R5-13: figuras de linha distinguem classes tambem pelo estilo."""
    from guaraci.paleta_cores import estilo_linha
    estilos = [repr(estilo_linha(i)) for i in range(8)]
    assert len(set(estilos)) == 8
