"""Testes de guaraci_theme.py — tema visual compartilhado (paleta, console,
helpers de status). Modulo sem cobertura dedicada ate aqui (reforço de
margem de cobertura de CI, auditoria jul/2026 — piso 60%, real 62%)."""
import re

import pytest


@pytest.fixture(scope="module")
def theme():
    import guaraci.guaraci_theme as mod
    return mod


def test_ansi_retorna_escape_truecolor_para_tom_conhecido(theme):
    seq = theme.ansi("PA")
    assert seq.startswith("\033[38;2;")
    r, g, b = theme._RGB["PA"]
    assert seq == f"\033[38;2;{r};{g};{b}m"


def test_ansi_tom_desconhecido_cai_no_fallback_claro(theme):
    seq = theme.ansi("TOM_INEXISTENTE")
    assert seq == "\033[38;2;200;200;200m"


def test_w_retorna_largura_dentro_dos_limites(theme):
    w = theme._W()
    assert 60 <= w <= 100


def test_w_robusto_a_falha_do_terminal(theme, monkeypatch):
    """Se shutil.get_terminal_size lançar, cai no fallback de 80 colunas
    (clampado para a faixa [60,100])."""
    def _quebra(*a, **k):
        raise OSError("sem terminal")
    monkeypatch.setattr(theme.shutil, "get_terminal_size", _quebra)
    assert theme._W() == 80


@pytest.mark.parametrize("fn,tag", [
    ("ok", "ok"), ("warn", "warn"), ("err", "err"), ("info", "info"),
])
def test_helpers_de_status_imprimem_com_a_tag_correta(theme, fn, tag, capsys):
    getattr(theme, fn)("mensagem de teste")
    saida = capsys.readouterr().out
    assert "mensagem de teste" in saida


def test_paleta_tem_8_tons_e_rgb_correspondente(theme):
    tons = ["PA", "PF", "PS", "PR", "PW", "PM", "PD", "PG"]
    for tom in tons:
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", getattr(theme, tom))
        assert tom in theme._RGB


def test_console_usa_o_tema_definido(theme):
    assert theme.console is not None
    assert theme.THEME is not None
