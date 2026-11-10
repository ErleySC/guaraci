"""Todo modulo de src/guaraci e' alcancavel a partir do CLI ou esta declarado.

Alcance = grafo de imports (inclusive os feitos dentro de funcoes; fora os de
`if TYPE_CHECKING:`, que nao executam) a partir das
entradas do CLI (`python -m guaraci`, telas e subcomandos). O que ficar de fora
precisa estar em `catalogo_dados.MODULOS_FORA_DO_CLI` com categoria e motivo, e
a lista nao pode ter modulo que ja e' alcancavel (declaracao velha) nem crescer
sem justificativa.
"""
import ast
from pathlib import Path

from guaraci.catalogo_dados import MODULOS_FORA_DO_CLI

SRC = Path(__file__).resolve().parents[1] / "src" / "guaraci"
ENTRADAS = ("__main__", "guaraci", "cli_comandos")
CATEGORIAS = {"pacote", "interface web", "biblioteca nao exposta"}


def _modulos():
    out = {}
    for p in SRC.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        out[".".join(p.relative_to(SRC).with_suffix("").parts)] = p
    return out


def _sem_type_checking(arvore):
    """Tira os blocos `if TYPE_CHECKING:` -- import so' de anotacao nao executa nada."""
    for n in ast.walk(arvore):
        for campo in ("body", "orelse"):
            corpo = getattr(n, campo, None)
            if isinstance(corpo, list):
                setattr(n, campo, [f for f in corpo if not (isinstance(f, ast.If) and "TYPE_CHECKING" in ast.unparse(f.test))])
    return arvore


def _deps(m, mods):
    arvore = _sem_type_checking(ast.parse(mods[m].read_text(encoding="utf-8")))
    out = set()
    for n in ast.walk(arvore):
        nomes = []
        if isinstance(n, ast.Import):
            nomes = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level:
                pk = m.split(".")[: len(m.split(".")) - n.level]
                base = ".".join(pk + ([base] if base else []))
            nomes = [base] + [f"{base}.{a.name}" for a in n.names]
        for nm in nomes:
            nm = nm[len("guaraci."):] if nm.startswith("guaraci.") else nm
            if nm in mods and nm != m:
                out.add(nm)
    return out


def _alcancaveis():
    mods = _modulos()
    vistos, pilha = set(), list(ENTRADAS)
    while pilha:
        m = pilha.pop()
        if m in vistos or m not in mods:
            continue
        vistos.add(m)
        pilha.extend(_deps(m, mods))
    return set(mods), vistos


def test_todo_modulo_e_alcancavel_pelo_cli_ou_declarado():
    todos, alc = _alcancaveis()
    fora = todos - alc
    nao_declarados = fora - set(MODULOS_FORA_DO_CLI)
    assert not nao_declarados, f"modulo(s) que o CLI nao alcanca e nao estao declarados: {sorted(nao_declarados)}"


def test_declaracao_nao_tem_modulo_alcancavel_nem_inexistente():
    todos, alc = _alcancaveis()
    assert not (set(MODULOS_FORA_DO_CLI) & alc), sorted(set(MODULOS_FORA_DO_CLI) & alc)
    assert set(MODULOS_FORA_DO_CLI) <= todos, sorted(set(MODULOS_FORA_DO_CLI) - todos)


def test_declaracao_tem_categoria_motivo_e_teto():
    for m, (cat, motivo) in MODULOS_FORA_DO_CLI.items():
        assert cat in CATEGORIAS and len(motivo) > 25, m
    assert sum(1 for c, _ in MODULOS_FORA_DO_CLI.values() if c == "biblioteca nao exposta") <= 5   # 3 -> 5: robpca, pcv_regressao
    assert len(MODULOS_FORA_DO_CLI) <= 22
