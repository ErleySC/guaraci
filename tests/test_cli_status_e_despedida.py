"""M-17(b,c) do parecer de 2026-09-25: o painel nao diz 'Pasta invalida' com modo
csv valido, e a despedida nao promete salvar edicoes de menu que nao foram rodadas."""
import io

import guaraci.guaraci as g


def _status(cfg):
    from rich.console import Console
    buf = io.StringIO()
    velho = g.console
    g.console = Console(file=buf, width=120, force_terminal=False)
    try:
        g._print_status(cfg)
    finally:
        g.console = velho
    return buf.getvalue()


def test_modo_csv_valido_nao_diz_pasta_invalida(tmp_path):
    csv = tmp_path / "meus_dados.csv"
    csv.write_text("classe,1000\nA,0.1\n", encoding="utf-8")
    cfg = g.Config(mode="csv", csv_file=str(csv))
    saida = _status(cfg)
    assert "meus_dados.csv" in saida
    assert "invalida" not in saida.lower() and "invalid" not in saida.lower()


def test_modo_csv_ausente_avisa_arquivo_nao_pasta(tmp_path):
    saida = _status(g.Config(mode="csv", csv_file=str(tmp_path / "nao_existe.csv")))
    assert "CSV" in saida


def test_despedida_nao_promete_salvar_edicoes_de_menu():
    for lang, frases in g._DESPEDIDAS.items():
        assert not any("estao salvos" in f or "are saved and ready" in f for f in frases), lang


def test_painel_de_status_a_80_colunas_nao_corta_valores():
    """M-17d/R5-12: a 80 colunas os valores saiam cortados com reticencias."""
    from rich.console import Console
    buf = io.StringIO()
    velho = g.console
    g.console = Console(file=buf, width=80, force_terminal=False)
    try:
        g._print_status(g.Config(mode="sintetico"))
    finally:
        g.console = velho
    assert "…" not in buf.getvalue(), buf.getvalue()
