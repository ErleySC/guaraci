"""Verificacao por amostra dos mutantes classificados como EQUIVALENTES.

Classificar um sobrevivente como "equivalente" e' uma afirmacao do autor da classificacao. Este script
tenta REFUTA-LA: sorteia (semente fixa) ate N mutantes classificados como equivalentes, aplica cada um ao
modulo e compara, em milhares de entradas aleatorias de borda (empates, variancia zero, desbalanceado,
blocos, colunas constantes), a saida do original com a do mutante. Qualquer diferenca prova que o mutante
NAO e' equivalente (o sobrevivente passa a exigir teste novo). Nenhuma diferenca nao prova equivalencia,
mas junto com o argumento escrito na classificacao e' a evidencia que um revisor pode reproduzir.

Uso: python scripts/verificar_equivalentes.py [--amostra 10] [--tentativas 400] [--saida ARQ.md]
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib.util
import json
import math
import random
import sys
import types
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
AUDIT = RAIZ / "docs" / "qualidade" / "anova"


def _mm():
    spec = importlib.util.spec_from_file_location("mutacao_modulo", RAIZ / "scripts" / "mutacao_modulo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def carregar_mutante(modulo_rel: str, idx: int, variante) -> types.ModuleType:
    mm = _mm()
    fonte = (RAIZ / modulo_rel).read_text(encoding="utf-8")
    arv = ast.parse(fonte)
    mutada = ast.unparse(ast.fix_missing_locations(mm.aplicar(arv, idx, tuple(variante))))
    nome = "guaraci._mutante_" + Path(modulo_rel).stem
    mod = types.ModuleType(nome)
    mod.__package__ = "guaraci"
    mod.__file__ = str(RAIZ / modulo_rel)
    sys.modules[nome] = mod
    exec(compile(mutada, nome, "exec"), mod.__dict__)
    return mod


def carregar_original(modulo_rel: str) -> types.ModuleType:
    nome = "guaraci." + Path(modulo_rel).stem
    return importlib.import_module(nome)


def igual(a, b) -> bool:
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        a, b = dataclasses.asdict(a), dataclasses.asdict(b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(igual(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(igual(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            return bool(np.allclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True))
        return bool(np.array_equal(a, b))
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        a, b = float(a), float(b)
        return (math.isnan(a) and math.isnan(b)) or a == b or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _chamar(f):
    try:
        return f()
    except Exception as e:                      # noqa: BLE001 -- o erro tambem e' comportamento comparavel
        return ("erro", type(e).__name__, str(e))


# ------------------------------------------------------------------ geradores de entradas de borda

def _dados_grupos(rng: np.random.Generator):
    k = int(rng.integers(2, 5))
    n = [int(rng.integers(2, 9)) for _ in range(k)]
    modo = rng.choice(["normal", "inteiros", "grupo_constante", "var_muito_desigual", "tudo_igual", "aditivo"])
    y, g = [], []
    for i in range(k):
        if modo == "normal":
            v = rng.normal(i * rng.uniform(0, 2), 1.0, n[i])
        elif modo == "inteiros":
            v = rng.integers(0, 4, n[i]).astype(float) + i
        elif modo == "aditivo":
            v = np.full(n[i], float(rng.integers(0, 6)))        # com blocos: dados aditivos, QM residual 0 (alcancavel)
        elif modo == "grupo_constante":
            v = np.full(n[i], float(i)) if i == 0 else rng.normal(i, 1.0, n[i])
        elif modo == "var_muito_desigual":
            v = rng.normal(0, 10.0 ** (-1 + 2 * i / max(1, k - 1)), n[i])
        else:
            v = np.full(n[i], 3.0)
        y += list(v)
        g += [f"g{i}"] * n[i]
    u = [f"u{j}" for j in range(len(y))]
    bloco = None
    if rng.random() < 0.4 or modo == "aditivo":
        nb = int(rng.integers(2, 4))
        bloco = [f"b{j % nb}" for j in range(len(y))]
        if modo == "aditivo":
            ef = rng.integers(0, 5, nb)
            y = [yy + float(ef[j % nb]) for j, yy in enumerate(y)]
    return y, g, u, bloco


def driver_anova_grupos(mod, rng):
    y, g, u, bloco = _dados_grupos(rng)
    saidas = []
    for var in ("auto", "iguais", "desiguais"):
        saidas.append(_chamar(lambda var=var: mod.anova_uma_via(y, g, u, bloco=bloco, variancias=var, n_boot=20)))
    saidas.append(_chamar(lambda: mod.kruskal_dunn(y, g, u, ajuste="holm")))
    saidas.append(_chamar(lambda: mod.kruskal_dunn(y, g, u, ajuste="bh")))
    saidas.append(_chamar(lambda: mod.diagnosticar_premissas(y, g, u)))
    d = _chamar(lambda: mod.diagnosticar_premissas(y, g, u))
    saidas.append(_chamar(lambda: mod.recomendar_metodo(d)) if isinstance(d, dict) else d)
    return saidas


def driver_anova_cli(mod, rng):
    k = int(rng.integers(2, 8))
    nomes = [f"g{i}" for i in range(k)]
    medias = {n: float(rng.normal()) for n in nomes}
    comps = [{"a": nomes[i], "b": nomes[j], "p_ajustado": float(rng.choice([0.001, 0.049, 0.05, 0.051, 0.5, 1.0]))}
             for i in range(k) for j in range(i + 1, k)]
    return _chamar(lambda: mod.letras_de_grupos(nomes, medias, comps, 0.05))


def driver_vip_estabilidade(mod, rng):
    n, p = int(rng.integers(24, 40)), int(rng.integers(4, 9))
    X = rng.normal(size=(n, p))
    if rng.random() < 0.4:
        X[:, int(rng.integers(0, p))] = 1.0             # coluna constante
    y = np.tile(np.arange(3), n // 3 + 1)[:n]
    X[:, 0] += y * 2.0
    grupos = [f"a{i // 2}" for i in range(n)]
    return _chamar(lambda: mod.vip_estabilidade(X, y, 2, grupos, tarefa="classificacao", n_divisoes=3,
                                                n_repeticoes=2, seed=int(rng.integers(0, 50))))


DRIVERS = {"anova_grupos": driver_anova_grupos, "anova_cli": driver_anova_cli, "vip_estabilidade": driver_vip_estabilidade}


def verificar(amostra: int, tentativas: int, semente: int = 0) -> list[dict]:
    cls = json.loads((AUDIT / "mutantes_classificados.json").read_text(encoding="utf-8"))
    candidatos = []
    for modulo in DRIVERS:
        j = json.loads((AUDIT / f"mutacao_{modulo}.json").read_text(encoding="utf-8"))
        for s in j["sobreviventes"]:
            chave = f"{modulo}:{s['linha']}:{s['operador']}"
            if cls.get(chave, "").startswith("equivalente"):
                candidatos.append((modulo, s, cls[chave]))
    random.Random(semente).shuffle(candidatos)
    resultados = []
    for modulo, s, motivo in candidatos[:amostra]:
        rel = f"src/guaraci/{modulo}.py"
        original = carregar_original(rel)
        mutante = carregar_mutante(rel, s["no"], s["variante"])
        if modulo == "anova_grupos":
            # o valor critico do IC simultaneo custa ~0,3 s por chamada e so' entra nos limites do IC95;
            # fixado igual nos dois lados para os sorteios caberem no tempo (p, F, eta2 continuam comparados).
            for m in (original, mutante):
                m._q_critico = lambda *a, **k: 0.0
        rng_a, rng_b = np.random.default_rng(1234), np.random.default_rng(1234)
        difere = 0
        primeira = None
        for t in range(tentativas):
            sa, sb = DRIVERS[modulo](original, rng_a), DRIVERS[modulo](mutante, rng_b)
            if not igual(sa, sb):
                difere += 1
                primeira = primeira if primeira is not None else t
        resultados.append({"modulo": modulo, "linha": s["linha"], "operador": s["operador"], "trecho": s["trecho"],
                           "tentativas": tentativas, "diferencas": difere, "primeira_diferenca": primeira, "motivo": motivo})
    return resultados


DRIVERS_COM_CONTROLE = ("anova_grupos", "vip_estabilidade")


def controle(tentativas: int = 60, n: int = 4, semente: int = 1) -> list[dict]:
    """Controle de poder: mutantes MORTOS (algum teste os pegou) de `anova_grupos` e `vip_estabilidade` precisam
    mostrar diferenca nos mesmos geradores; se nao mostrassem, 'nenhuma diferenca' nos equivalentes nao valeria nada."""
    out = []
    for modulo in ("anova_grupos", "vip_estabilidade"):
        j = json.loads((AUDIT / f"mutacao_{modulo}.json").read_text(encoding="utf-8"))
        ate = 173 if modulo == "vip_estabilidade" else 10**9           # a figura (linha 174 em diante) nao e' exercida pelo gerador
        mortos = [r for r in j["todos"] if r["estado"] == "morto" and r["linha"] < ate
                  and r["operador"] in ("Lt->Gt", "Add->Sub", "Sub->Add", "Gt->Lt", "Mult->Div")]
        random.Random(semente).shuffle(mortos)
        rel = f"src/guaraci/{modulo}.py"
        original = carregar_original(rel)
        for s in mortos[:n // 2]:
            mutante = carregar_mutante(rel, s["no"], s["variante"])
            if modulo == "anova_grupos":
                for m in (original, mutante):
                    m._q_critico = lambda *a, **k: 0.0
            ra, rb = np.random.default_rng(99), np.random.default_rng(99)
            dif = sum(not igual(DRIVERS[modulo](original, ra), DRIVERS[modulo](mutante, rb)) for _ in range(tentativas))
            out.append({"modulo": modulo, "linha": s["linha"], "operador": s["operador"], "tentativas": tentativas, "diferencas": dif})
    return out


def _md(res: list[dict], total_equiv: int, ctrl: list[dict] | None = None) -> str:
    L = ["# Verificacao por amostra dos mutantes equivalentes", "", "**Público:** quem avalia os números · **Estado:** gerado por `scripts/verificar_equivalentes.py` (não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`", "",
         f"Sorteados {len(res)} de {total_equiv} classificados como equivalentes (semente 0). Cada um foi aplicado ao modulo e a saida "
         "foi comparada com a do original em entradas aleatorias de borda (`scripts/verificar_equivalentes.py`).", "",
         "| mutante | trecho | tentativas | saidas diferentes | veredito |", "|---|---|---|---|---|"]
    for r in res:
        trecho = r["trecho"].replace("|", "\\|")[:70]
        ver = "equivalente (nenhuma diferenca)" if r["diferencas"] == 0 else f"**NAO equivalente** (1a diferenca na tentativa {r['primeira_diferenca']})"
        L.append(f"| `{r['modulo']}:{r['linha']}` {r['operador']} | `{trecho}` | {r['tentativas']} | {r['diferencas']} | {ver} |")
    if ctrl:
        L += ["", "Controle de poder (mutantes MORTOS nos mesmos geradores; ao menos um por modulo precisa mostrar diferenca; zero em um mutante morto significa que o gerador nao chega naquela linha, por exemplo uma recusa de entrada):", "",
              "| mutante morto | tentativas | saidas diferentes |", "|---|---|---|"]
        L += [f"| `{c['modulo']}:{c['linha']}` {c['operador']} | {c['tentativas']} | {c['diferencas']} |" for c in ctrl]
    L += ["", "Nenhuma diferenca e' evidencia, nao prova; a prova e' o argumento escrito de cada classificacao, abaixo.", "",
          "Argumentos escritos de cada classificacao:", ""]
    for r in res:
        L.append(f"- `{r['modulo']}:{r['linha']}` {r['operador']}: {r['motivo']}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--amostra", type=int, default=10)
    ap.add_argument("--tentativas", type=int, default=400)
    ap.add_argument("--saida", default=None)
    a = ap.parse_args(argv)
    res = verificar(a.amostra, a.tentativas)
    cls = json.loads((AUDIT / "mutantes_classificados.json").read_text(encoding="utf-8"))
    ctrl = controle()
    texto = _md(res, sum(1 for v in cls.values() if v.startswith("equivalente")), ctrl)
    print(texto)
    if a.saida:
        Path(a.saida).write_text(texto, encoding="utf-8", newline="\n")
    return 1 if any(r["diferencas"] for r in res) or not all(any(c["diferencas"] for c in ctrl if c["modulo"] == m) for m in DRIVERS_COM_CONTROLE) else 0


if __name__ == "__main__":
    sys.exit(main())
