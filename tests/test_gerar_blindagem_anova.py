"""`scripts/gerar_blindagem_anova.py` e `scripts/mutacao_modulo.py`: o relatorio sai das medicoes, sem caminho privado."""
import importlib.util
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def _carregar(nome):
    spec = importlib.util.spec_from_file_location(nome, RAIZ / "scripts" / f"{nome}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cob():
    return {"files": {f"src\\guaraci\\{m}.py": {"summary": {"covered_lines": 95, "num_statements": 100, "covered_branches": 40,
                                                          "num_branches": 44, "percent_covered": 94.5},
                                                 "missing_lines": [3, 4]}
                      for m in ("anova_grupos", "anova_cli", "vip_estabilidade", "vip_cli", "relatorio_blocos")}}


def _mut(modulo, total, sobreviventes):
    viv = [{"linha": ln, "operador": op, "trecho": "x = 1", "estado": "sobrevivente", "no": 1, "variante": [0, 0]}
           for ln, op in sobreviventes]
    return {"modulo": f"src/guaraci/{modulo}.py", "total": total, "sobreviventes": viv, "todos": []}


def test_relatorio_conta_mortos_e_classificados_e_nao_tem_caminho_de_usuario():
    g = _carregar("gerar_blindagem_anova")
    mut = [_mut("anova_grupos", 100, [(10, "Lt->LtE"), (11, "const 0")]), _mut("vip_cli", 50, [])]
    cls = {"anova_grupos:10:Lt->LtE": "igualdade exata de ponto flutuante (probabilidade zero): teste"}
    t = g.gerar("2026-10-06", _cob(), mut, cls)
    assert "## 1. Inventario" in t and "## 6. Mutacao" in t and "## 8. Referencias citadas" in t
    assert "| `anova_grupos.py` | 100 | 98 | 2 | 1 | 1 |" in t and "SEM CLASSIFICACAO" in t
    assert "149/150 mortos ou equivalentes classificados; 1 sem classificacao" in t
    assert "| Dunn 1964 | 10.1080/00401706.1964.10490181 | sim |" in t
    usuario = Path.home().name.lower()                      # o relatorio nao pode carregar o usuario de quem o gerou
    assert "Users" not in t and (len(usuario) < 4 or usuario not in t.lower())
    assert "test_anova_blindagem.py" in t and "casos** com parametrizacao" in t


def test_limpar_remove_caminho_de_usuario_e_de_tmp():
    g = _carregar("gerar_blindagem_anova")
    assert g._limpar("C:\\Users\\alguem\\x\\y.py e C:\\Temp\\mut_ab12\\z") == "<usuario>\\x\\y.py e <tmp>"


def test_mutacao_enumera_e_aplica_um_mutante_por_operador():
    import ast
    mm = _carregar("mutacao_modulo")
    fonte = "def f(a, b):\n    if a < b and not a:\n        return a + 1\n    raise ValueError('x')\n"
    arv = ast.parse(fonte)
    sites = mm.enumerar(arv)
    ops = {s[2] for s in sites}
    assert {"Lt->LtE", "Lt->Gt", "And<->Or", "remove Not", "const 1", "negar teste", "return None", "raise->pass",
            "Add->Sub"} <= ops
    for idx, var, desc, ln in sites:
        nova = ast.unparse(mm.aplicar(arv, idx, var))
        assert nova != ast.unparse(arv), desc
        ast.parse(nova)
    # strings, docstrings e anotacoes nao viram mutantes
    fonte2 = 'def g(x: int = 3) -> int:\n    """doc"""\n    return x\n'
    assert all(s[2].startswith("const 3") or s[2] == "return None" for s in mm.enumerar(ast.parse(fonte2)))
    assert not [s for s in mm.enumerar(ast.parse("log.info('a')\nx = 'texto'\n"))]


def test_json_de_mutacao_versionado_e_coerente_e_sem_caminho_privado():
    pasta = RAIZ / "docs" / "qualidade" / "anova"
    cls = json.loads((pasta / "mutantes_classificados.json").read_text(encoding="utf-8"))
    for mod in ("anova_grupos", "anova_cli", "vip_estabilidade", "vip_cli", "relatorio_blocos"):
        j = json.loads((pasta / f"mutacao_{mod}.json").read_text(encoding="utf-8"))
        assert j["total"] == len(j["todos"]) and j["mortos_no_total"] == j["total"] - len(j["sobreviventes"])
        assert all(r["estado"] != "sobrevivente" or f"{mod}:{r['linha']}:{r['operador']}" in cls for r in j["todos"])
        assert all(r["prova"] for r in j["todos"] if r["estado"] == "morto")      # todo morto cita o teste que falhou
        texto = json.dumps(j)
        assert "Users" not in texto and "AppData" not in texto
