"""Python 3.11 e o minimo suportado (pyproject: requires-python >= 3.11).

Rodando em 3.12+, o parser aceita construcoes que o 3.11 recusa (PEP 701:
barra invertida, aspas iguais ou comentario dentro da expressao de uma
f-string; PEP 695: `type X = ...` e generics `def f[T]`). `ast.parse` com
`feature_version=(3, 11)` pega as ultimas, mas NAO as da PEP 701; por isso a
checagem de f-string abaixo e feita pelo tokenizador.
"""
import ast
import importlib
import io
import shutil
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest

_R = Path(__file__).resolve().parents[1]
_PASTAS = ("src", "tests", "scripts")


def _arquivos():
    fs = [p for d in _PASTAS if (_R / d).is_dir() for p in (_R / d).rglob("*.py")]
    fs += [_R / "app_quimiometria.py"]
    return sorted(p for p in fs if p.is_file() and "__pycache__" not in p.parts)


def _delim(texto):
    """Delimitador (aspas) de um token de string/FSTRING_START, sem prefixo."""
    t = texto.lstrip("rRbBuUfF")
    return t[:3] if t[:3] in ('"""', "'''") else t[:1]


def violacoes_fstring_py311(fonte):
    """Lista (linha, motivo) das f-strings que so existem no Python >= 3.12.

    Sem efeito no 3.11, onde o proprio parser as recusa.
    """
    if not hasattr(tokenize, "FSTRING_START"):
        return []
    out = []
    pilha = []  # um item por f-string aberta: [delimitador, profundidade de chaves]
    for tk in tokenize.generate_tokens(io.StringIO(fonte).readline):
        linha = tk.start[0]
        if tk.type == tokenize.FSTRING_START:
            d = _delim(tk.string)
            if any(f[1] > 0 and d.startswith(f[0]) for f in pilha):
                out.append((linha, "aspas iguais as da f-string dentro da expressao"))
            pilha.append([d, 0])
            continue
        if tk.type == tokenize.FSTRING_END:
            pilha.pop()
            continue
        if not pilha:
            continue
        topo = pilha[-1]
        if tk.type == tokenize.FSTRING_MIDDLE:
            # texto literal de f-string aninhada dentro da expressao de outra
            if len(pilha) > 1 and any(f[1] > 0 for f in pilha[:-1]) and "\\" in tk.string:
                out.append((linha, "barra invertida dentro da expressao"))
            continue
        if tk.type == tokenize.OP and tk.string == "{":
            topo[1] += 1
        elif tk.type == tokenize.OP and tk.string == "}":
            topo[1] -= 1
        elif topo[1] > 0:
            if tk.type == tokenize.COMMENT:
                out.append((linha, "comentario dentro da expressao"))
            elif tk.type == tokenize.STRING:
                if "\\" in tk.string:
                    out.append((linha, "barra invertida dentro da expressao"))
                if any(f[1] > 0 and _delim(tk.string).startswith(f[0]) for f in pilha):
                    out.append((linha, "aspas iguais as da f-string dentro da expressao"))
            elif tk.type == tokenize.NL and len(topo[0]) == 1:
                out.append((linha, "quebra de linha na expressao de f-string simples"))
    return out


@pytest.mark.parametrize(
    "fonte",
    [
        "x = 1\ns = f\"\"\"{'' if x else ' a \\\\b'}\"\"\"\n",   # barra na expressao
        "d = {'a': 1}\ns = f\"{d[\"a\"]}\"\n",                   # aspas iguais
        "x = 1\ns = f\"\"\"{x # c\n}\"\"\"\n",                    # comentario
        "x = 1\ns = f\"{f'{x!r:\\\\n}'}\"\n",                      # barra em f-string aninhada
    ],
)
def test_detector_de_fstring_pega_construcoes_so_3_12(fonte):
    if not hasattr(tokenize, "FSTRING_START"):
        pytest.skip("o parser do 3.11 ja recusa estas construcoes")
    assert violacoes_fstring_py311(fonte)


def test_detector_nao_acusa_fstring_valida_no_3_11():
    ok = (
        "x = 1\n"
        "a = f\"\"\"{x} \\\\citet{{k}} {'a' if x else 'b'}\"\"\"\n"  # barra so no texto literal
        "b = f'{x:>{x}}' + f\"{'a'}\" + f'{{lit}}'\n"
    )
    assert violacoes_fstring_py311(ok) == []


def test_ha_arquivos_para_checar():
    nomes = {p.name for p in _arquivos()}
    assert {"reports.py", "app_quimiometria.py", "test_sintaxe_py311.py"} <= nomes


def test_todo_o_codigo_parseia_com_feature_version_3_11():
    erros = []
    for p in _arquivos():
        try:
            ast.parse(p.read_text(encoding="utf-8"), str(p), feature_version=(3, 11))
        except SyntaxError as e:
            erros.append(f"{p.relative_to(_R)}:{e.lineno}: {e.msg}")
    assert not erros, "sintaxe so valida no Python >= 3.12:\n" + "\n".join(erros)


def test_nenhuma_fstring_usa_sintaxe_da_pep701():
    erros = []
    for p in _arquivos():
        for linha, motivo in violacoes_fstring_py311(p.read_text(encoding="utf-8")):
            erros.append(f"{p.relative_to(_R)}:{linha}: {motivo}")
    assert not erros, "f-string so valida no Python >= 3.12:\n" + "\n".join(erros)


def test_reports_importa():
    assert importlib.import_module("guaraci.reports") is not None


@pytest.mark.skipif(
    sys.version_info[:2] == (3, 11) or shutil.which("python3.11") is None,
    reason="sem interpretador 3.11 separado",
)
def test_compila_de_verdade_com_python311():
    exe = shutil.which("python3.11")
    arquivos = [str(p) for p in _arquivos()]
    codigo = (
        "import sys\n"
        "bad = []\n"
        "for f in sys.argv[1:]:\n"
        "    try: compile(open(f, encoding='utf-8').read(), f, 'exec')\n"
        "    except SyntaxError as e: bad.append(f'{f}:{e.lineno}: {e.msg}')\n"
        "print('\\n'.join(bad)); sys.exit(1 if bad else 0)\n"
    )
    r = subprocess.run([exe, "-I", "-c", codigo, *arquivos], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
