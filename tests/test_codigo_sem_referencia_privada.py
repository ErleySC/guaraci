"""Comentarios e docstrings do pacote explicam o MOTIVO no proprio texto, sem apontar para arquivo privado ou de processo.

O repositorio e' publico: quem le o codigo nao tem `CLAUDE.md` nem a pasta local do autor. Aponta-se o motivo, nao o arquivo.
Fica de fora, de proposito, o texto de execucao (o nome da pasta local usado como caminho pelo codigo em `dados_io.py`).
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
# Os literais proibidos sao partidos para este proprio arquivo nao os conter.
_PROIBIDO = re.compile("CLAUDE" + r"\.md|" + r"\.guaraci" + "_local|" + "guaraci" + "_local")


def _comentarios_e_docstrings(codigo: str):
    arvore = ast.parse(codigo)
    docs = set()
    for n in ast.walk(arvore):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = n.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
                docs.add((b[0].value.lineno, b[0].value.col_offset))
    for tk in tokenize.generate_tokens(io.StringIO(codigo).readline):
        if tk.type == tokenize.COMMENT or (tk.type == tokenize.STRING and tk.start in docs):
            yield tk.start[0], tk.string


def _achados(pasta: Path):
    ruins = []
    for f in sorted(pasta.rglob("*.py")):
        for linha, texto in _comentarios_e_docstrings(f.read_text(encoding="utf-8")):
            m = _PROIBIDO.search(texto)
            if m:
                ruins.append(f"{f.name}:{linha}: {m.group(0)}")
    return ruins


def test_src_nao_cita_arquivo_privado_em_comentario_nem_docstring():
    assert _achados(RAIZ / "src" / "guaraci") == []


def test_pyproject_nao_cita_arquivo_de_processo():
    texto = (RAIZ / "pyproject.toml").read_text(encoding="utf-8")
    assert not _PROIBIDO.search(texto)


def test_o_detector_pega_comentario_e_docstring_mas_nao_texto_de_execucao(tmp_path):
    alvo = "CLAUDE" + ".md"
    pasta = "~/.guaraci" + "_local"
    (tmp_path / "m.py").write_text(
        f'"""ver {alvo}"""\n# veja {alvo}\nX = "{alvo}"\n\n\ndef f():\n    """{pasta}/x"""\n', encoding="utf-8")
    achados = _achados(tmp_path)
    assert len(achados) == 3 and not any(":3:" in a for a in achados)
