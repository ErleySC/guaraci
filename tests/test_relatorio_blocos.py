"""`relatorio_blocos`: o mesmo conteudo nos tres formatos, com saida exata."""
from guaraci.relatorio_blocos import renderizar_html, renderizar_md, renderizar_tex

BLOCOS = [
    ("h1", "Titulo & <um>"),
    ("h2", "Secao"),
    ("h3", "Sub"),
    ("p", "Texto com 50% e a_b $x$ #1"),
    ("table", ["g", "n"], [["A", "3"], ["B|C", "5"]]),
    ("img", "../figuras/f.png", "Figura: x"),
]


def test_markdown_exato():
    assert renderizar_md(BLOCOS) == "\n".join([
        "# Titulo & <um>", "", "## Secao", "", "### Sub", "", "Texto com 50% e a_b $x$ #1", "",
        "| g | n |", "|---|---|", "| A | 3 |", "| B\\|C | 5 |", "", "![Figura: x](../figuras/f.png)", ""])


def test_html_escapa_e_tem_estrutura():
    h = renderizar_html(BLOCOS)
    assert h.startswith("<!doctype html>") and h.rstrip().endswith("</html>")
    assert "<h1>Titulo &amp; &lt;um&gt;</h1>" in h and "<h2>Secao</h2>" in h and "<h3>Sub</h3>" in h
    assert "<p>Texto com 50% e a_b $x$ #1</p>" in h
    assert '<table border="1" cellpadding="4"><tr><th>g</th><th>n</th></tr>' in h
    assert "<tr><td>A</td><td>3</td></tr><tr><td>B|C</td><td>5</td></tr></table>" in h
    assert '<img src="../figuras/f.png" alt="Figura: x" style="max-width:100%">' in h
    assert 'charset="utf-8"' in h and "viewport" in h


def test_latex_escapa_caracteres_especiais_e_monta_tabela_e_figura():
    t = renderizar_tex(BLOCOS)
    assert "\\section*{Titulo \\& <um>}" in t and "\\subsection*{Secao}" in t and "\\subsubsection*{Sub}" in t
    assert "Texto com 50\\% e a\\_b \\$x\\$ \\#1" in t
    assert "\\begin{tabular}{ll}\\hline" in t
    assert "g & n \\\\ \\hline" in t and "A & 3 \\\\" in t and "B|C & 5 \\\\" in t
    assert "\\hline\\end{tabular}\\end{center}" in t
    assert "\\includegraphics[width=\\linewidth]{../figuras/f.png}" in t
    assert t.startswith("\\documentclass{article}") and t.rstrip().endswith("\\end{document}")


def test_latex_escapa_barra_chaves_til_e_circunflexo():
    t = renderizar_tex([("p", "a\\b {c} d~e f^g")])
    assert "a\\textbackslash{}b \\{c\\} d\\textasciitilde{}e f\\textasciicircum{}g" in t


def test_tabela_com_tres_colunas_tem_tres_separadores_e_tres_l():
    b = [("table", ["a", "b", "c"], [["1", "2", "3"]])]
    assert "|---|---|---|" in renderizar_md(b)
    assert "\\begin{tabular}{lll}" in renderizar_tex(b)
    assert renderizar_html(b).count("<th>") == 3


def test_blocos_vazios_e_desconhecido_nao_quebram():
    assert renderizar_md([]) == "" and "<body>" in renderizar_html([])
    assert renderizar_tex([("desconhecido", "x")]).count("\\begin{document}") == 1
