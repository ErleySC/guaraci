"""relatorio_blocos.py -- relatorio .md/.html/.tex a partir de blocos neutros.

Blocos: ("h1"|"h2"|"h3", texto), ("p", texto), ("aviso", texto) -- paragrafo em destaque --,
("img", caminho_relativo, alt), ("table", cabecalho, linhas). Usado por `guaraci anova` e `guaraci vip`: o
conteudo e' montado uma vez e renderizado nos tres formatos, com os caminhos
das figuras sempre RELATIVOS ao arquivo do relatorio.
"""
from __future__ import annotations

import html
from typing import List

__all__ = ["renderizar_md", "renderizar_html", "renderizar_tex", "renderizar_tex_corpo"]


def renderizar_md(blocos: List[tuple]) -> str:
    L: List[str] = []
    for b in blocos:
        if b[0] in ("h1", "h2", "h3"):
            L += ["#" * int(b[0][1]) + " " + b[1], ""]
        elif b[0] == "p":
            L += [b[1], ""]
        elif b[0] == "aviso":
            L += [f"**{b[1]}**", ""]
        elif b[0] == "img":
            L += [f"![{b[2]}]({b[1]})", ""]
        elif b[0] == "table":
            L.append("| " + " | ".join(b[1]) + " |")
            L.append("|" + "---|" * len(b[1]))
            L += ["| " + " | ".join(x.replace("|", "\\|") for x in lin) + " |" for lin in b[2]]
            L.append("")
    return "\n".join(L)


def renderizar_html(blocos: List[tuple]) -> str:
    e = html.escape
    P: List[str] = []
    for b in blocos:
        if b[0] in ("h1", "h2", "h3"):
            P.append(f"<{b[0]}>{e(b[1])}</{b[0]}>")
        elif b[0] == "p":
            P.append(f"<p>{e(b[1])}</p>")
        elif b[0] == "aviso":
            P.append(f"<p><strong>{e(b[1])}</strong></p>")
        elif b[0] == "img":
            P.append(f'<p><img src="{e(b[1], quote=True)}" alt="{e(b[2], quote=True)}" style="max-width:100%"></p>')
        elif b[0] == "table":
            cab = "".join(f"<th>{e(c)}</th>" for c in b[1])
            corpo = "".join("<tr>" + "".join(f"<td>{e(x)}</td>" for x in lin) + "</tr>" for lin in b[2])
            P.append(f"<table border=\"1\" cellpadding=\"4\"><tr>{cab}</tr>{corpo}</table>")
    return ("<!doctype html>\n<html><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            "content=\"width=device-width, initial-scale=1\"><title>guaraci anova</title></head>\n<body>\n"
            + "\n".join(P) + "\n</body></html>\n")


def _tex_esc(t: str) -> str:
    sub = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(sub.get(c, c) for c in t)


def _colunas_tex(cab: List[str], linhas: List[List[str]]) -> str:
    """Especificacao de colunas do tabular: `l`, e `p{..}` (quebra de linha) nas colunas de texto longo."""
    longas = [any(len(str(lin[j])) > 40 for lin in linhas if j < len(lin)) for j in range(len(cab))]
    n_longas = sum(longas)
    if not n_longas:
        return "l" * len(cab)
    largura = max(0.2, 0.75 / n_longas)
    return "".join(f"p{{{largura:.2f}\\linewidth}}" if lg else "l" for lg in longas)


def renderizar_tex_corpo(blocos: List[tuple]) -> str:
    """So' o corpo LaTeX (sem preambulo nem \\begin{document}): para embutir num modelo de artigo."""
    esc = _tex_esc
    P: List[str] = []
    cmd = {"h1": "section*", "h2": "subsection*", "h3": "subsubsection*"}
    for b in blocos:
        if b[0] in cmd:
            P.append(f"\\{cmd[b[0]]}{{{esc(b[1])}}}")
        elif b[0] == "p":
            P.append(esc(b[1]) + "\n")
        elif b[0] == "aviso":
            P.append("\\textbf{" + esc(b[1]) + "}\n")
        elif b[0] == "img":
            P.append(f"\\begin{{center}}\\includegraphics[width=\\linewidth]{{{b[1]}}}\\end{{center}}")
        elif b[0] == "table":
            P.append("\\begin{center}\\begin{tabular}{" + _colunas_tex(b[1], b[2]) + "}\\hline")
            P.append(" & ".join(esc(c) for c in b[1]) + r" \\ \hline")
            P += [" & ".join(esc(x) for x in lin) + r" \\" for lin in b[2]]
            P.append("\\hline\\end{tabular}\\end{center}")
    return "\n".join(P) + "\n"


def renderizar_tex(blocos: List[tuple]) -> str:
    P = [r"\documentclass{article}", r"\usepackage[utf8]{inputenc}", r"\usepackage[T1]{fontenc}",
         r"\usepackage{graphicx}", r"\usepackage[margin=2cm]{geometry}", r"\begin{document}"]
    return "\n".join(P) + "\n" + renderizar_tex_corpo(blocos) + r"\end{document}" + "\n"
