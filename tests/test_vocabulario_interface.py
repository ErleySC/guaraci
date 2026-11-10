"""Vocabulario interno fora da interface (V22, lote 4): nenhum texto mostrado ao usuario cita Passo, Bloco, sprint,
achado numerado nem nome de arquivo .py. Cada teste falha no codigo anterior ao lote (conferido por `git archive`).
"""
import ast
import os
import re
from pathlib import Path

from guaraci import cli_comandos as cmd
from guaraci import docs_pacote as dp

RAIZ = Path(__file__).resolve().parent.parent

#: Jargao de desenvolvimento que nao pode aparecer em texto mostrado ao usuario.
_JARGAO = re.compile(r"Passo \d|Bloco ?\d|\b[Ss]print ?\d|achado #|objetivo_cfg")
#: Idem, mais nome de arquivo .py (nas saidas de `demo` e `--help`).
_JARGAO_COM_PY = re.compile(_JARGAO.pattern + r"|\b[a-z][a-z_0-9]+\.py\b")


def test_help_nao_traz_jargao_nem_nome_de_arquivo_py():
    for lang in ("PT", "EN"):
        txt = cmd.texto_ajuda(lang)
        assert not _JARGAO_COM_PY.search(txt), _JARGAO_COM_PY.search(txt).group(0)


def test_help_diz_modo_e_nao_mode_no_texto_em_portugues():
    txt = cmd.texto_ajuda("PT")
    assert "o unico modo que corresponde" in txt
    assert "unico mode" not in txt


def test_manual_nao_traz_passo_bloco_nem_sprint():
    textos = [dp.lista()] + [dp.ler(t) for t in dp.TOPICOS]
    for t in textos:
        m = _JARGAO.search(t)
        assert m is None, m and t[max(0, m.start() - 60): m.end() + 60]


def test_manual_secao_5_8_nao_tem_nome_de_arquivo_no_titulo():
    for sec in ("5.8", "5.9", "5.10", "5.1", "5.3", "5.4", "5.5"):
        titulo = (dp.secao_do_manual(sec) or "").splitlines()[0]
        assert titulo and ".py" not in titulo, titulo


def _strings_de_usuario(caminho: Path):
    """Constantes de texto de um modulo, sem docstrings (o que o programa pode mostrar ao usuario)."""
    arvore = ast.parse(caminho.read_text(encoding="utf-8"))
    docs = set()
    for n in ast.walk(arvore):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = n.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
                docs.add(id(b[0].value))
    for n in ast.walk(arvore):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
            yield n.lineno, n.value


def test_nenhum_texto_do_programa_cita_passo_bloco_ou_sprint():
    ignorar = {"catalogo.py"}   # contem o proprio filtro (lista de padroes proibidos)
    achados = []
    for f in sorted((RAIZ / "src" / "guaraci").rglob("*.py")):
        if f.name in ignorar:
            continue
        for linha, s in _strings_de_usuario(f):
            m = _JARGAO.search(s)
            if m:
                achados.append(f"{f.name}:{linha}: {s[:70]!r}")
    assert not achados, "\n".join(achados)


def test_figuras_gravadas_nao_levam_o_nome_do_sprint():
    fonte = (RAIZ / "src" / "guaraci" / "figuras.py").read_text(encoding="utf-8")
    nomes = re.findall(r'save\(\s*fig2?,\s*"([^"]+)"', fonte)
    assert nomes and not [n for n in nomes if "sprint" in n.lower()]
    assert {"fig_selectivity_ratio_vip", "fig_opls_scores"} <= set(nomes)
    assert '"fig_ddsimca_aceitacao"' in fonte   # paginada: o nome vai para o helper de pagina


def test_demo_nao_mostra_jargao_na_saida_inteira(tmp_path, monkeypatch, capsys, caplog):
    """Roda o `demo` de verdade (dados sinteticos pequenos): nada de Passo/Bloco/sprint/.py na saida inteira."""
    import subprocess

    from guaraci import guaraci as g
    real = g.Config

    def pequeno(**kw):
        kw.update(n_per_class=6, n_synthetic_points=60, n_permutations=5, n_permutations_wold=5,
                  n_bootstrap_vip=3, n_bootstrap_bca=20, n_monte_carlo=3, max_lvs=5)
        return real(**kw)

    monkeypatch.setattr(g, "Config", pequeno)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(os, "startfile", lambda *a, **k: None, raising=False)
    monkeypatch.chdir(tmp_path)
    with caplog.at_level("DEBUG"):
        g._comando_demo()
    saida = capsys.readouterr().out + "\n" + caplog.text
    m = _JARGAO_COM_PY.search(re.sub(r"[A-Za-z]:[^\s]*", "", saida))
    assert m is None, m and saida[max(0, m.start() - 80): m.end() + 80]
