#!/usr/bin/env python3
"""mutacao_modulo.py -- teste de mutacao de UM modulo, em paralelo, com copia isolada.

    python scripts/mutacao_modulo.py src/guaraci/anova_grupos.py \
        --testes tests/test_anova_grupos.py tests/test_anova_blindagem.py \
        [--extras ARQUIVO ...] [--deselecionar NODEID ...] [--workers 12] [--timeout 240] [--linhas 100-200] \
        [--saida docs/qualidade/anova/mutacao_anova_grupos.json]

Por que existe (alem do cosmic-ray): roda com o interpretador ABSOLUTO desta execucao (armadilha do
PATH), em copias isoladas por worker (o arquivo do repositorio nunca e' tocado: sem risco de deixar
mutante em disco), grava o PRIMEIRO teste que falhou de cada mutante morto (evidencia: "killed" sem
saida de pytest e' suspeito) e recusa rodar se a linha de base (sem mutacao) nao passa.

Operadores (sobre a AST; strings, docstrings, anotacoes e chamadas de log nao sao mutadas):
  comparacao (Eq/NotEq/Lt/LtE/Gt/GtE/Is/In e suas negacoes), aritmetica (+ - * / ** // %), And<->Or,
  remocao de `not` e de `-` unario, constantes numericas (+1) e booleanas (inverte), negacao do teste
  de `if`/`while`/ternario, `return X` -> `return None`, `raise` -> `pass`, `continue`/`break` -> `pass`.

Resultado de cada mutante: morto (um teste falhou), morto-por-importacao (o mutante quebra o
import/coleta: contado como morto, listado a parte), tempo-esgotado (contado como morto, como no
cosmic-ray) ou SOBREVIVENTE (todos os testes passaram) -- estes precisam de teste novo ou de
classificacao como equivalente, um a um.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

_CMP = {ast.Eq: [ast.NotEq], ast.NotEq: [ast.Eq], ast.Lt: [ast.LtE, ast.Gt], ast.LtE: [ast.Lt, ast.Gt],
        ast.Gt: [ast.GtE, ast.Lt], ast.GtE: [ast.Gt, ast.Lt], ast.Is: [ast.IsNot], ast.IsNot: [ast.Is],
        ast.In: [ast.NotIn], ast.NotIn: [ast.In]}
_BIN = {ast.Add: [ast.Sub], ast.Sub: [ast.Add], ast.Mult: [ast.Div], ast.Div: [ast.Mult],
        ast.Pow: [ast.Mult], ast.FloorDiv: [ast.Div], ast.Mod: [ast.Mult]}


def _ignorados(arvore: ast.AST) -> set:
    """ids de nos que nao se mutam: anotacoes, docstrings, strings, chamadas de log."""
    ign = set()

    def marca(no):
        for n in ast.walk(no):
            ign.add(id(n))

    for n in ast.walk(arvore):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for a in n.args.args + n.args.kwonlyargs + n.args.posonlyargs:
                if a.annotation is not None:
                    marca(a.annotation)
            if n.args.vararg is not None and n.args.vararg.annotation is not None:
                marca(n.args.vararg.annotation)
            if n.args.kwarg is not None and n.args.kwarg.annotation is not None:
                marca(n.args.kwarg.annotation)
            if n.returns is not None:
                marca(n.returns)
        if isinstance(n, ast.AnnAssign):
            marca(n.annotation)
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            marca(n)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                and n.func.value.id == "log":
            marca(n)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets):
            marca(n)
        if isinstance(n, ast.JoinedStr):
            marca(n)
    return ign


def enumerar(arvore: ast.AST, linhas=None):
    """Lista de (indice_do_no_na_ordem_do_walk, variante_k, descricao, lineno)."""
    ign = _ignorados(arvore)
    saida = []
    for idx, n in enumerate(ast.walk(arvore)):
        if id(n) in ign or not hasattr(n, "lineno"):
            continue
        ln = n.lineno
        if linhas and not (linhas[0] <= ln <= linhas[1]):
            continue
        if isinstance(n, ast.Compare):
            for i, op in enumerate(n.ops):
                for k, novo in enumerate(_CMP.get(type(op), [])):
                    saida.append((idx, (i, k), f"{type(op).__name__}->{novo.__name__}", ln))
        elif isinstance(n, ast.BinOp):
            for k, novo in enumerate(_BIN.get(type(n.op), [])):
                saida.append((idx, (0, k), f"{type(n.op).__name__}->{novo.__name__}", ln))
        elif isinstance(n, ast.AugAssign):
            for k, novo in enumerate(_BIN.get(type(n.op), [])):
                saida.append((idx, (0, k), f"Aug{type(n.op).__name__}->{novo.__name__}", ln))
        elif isinstance(n, ast.BoolOp):
            saida.append((idx, (0, 0), "And<->Or", ln))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.Not, ast.USub)):
            saida.append((idx, (0, 0), f"remove {type(n.op).__name__}", ln))
        elif isinstance(n, ast.Constant) and not isinstance(n.value, (str, bytes, type(None), type(Ellipsis))):
            saida.append((idx, (0, 0), f"const {n.value!r}", ln))
        elif isinstance(n, (ast.If, ast.While, ast.IfExp)):
            saida.append((idx, (0, 0), "negar teste", ln))
        elif isinstance(n, ast.Return) and n.value is not None and not (
                isinstance(n.value, ast.Constant) and n.value.value is None):
            saida.append((idx, (0, 0), "return None", ln))
        elif isinstance(n, ast.Raise):
            saida.append((idx, (0, 0), "raise->pass", ln))
        elif isinstance(n, (ast.Continue, ast.Break)):
            saida.append((idx, (0, 0), f"{type(n).__name__}->pass", ln))
    return saida


def aplicar(arvore: ast.AST, idx: int, variante) -> ast.AST:
    nova = copy.deepcopy(arvore)
    alvo = list(ast.walk(nova))[idx]
    i, k = variante
    if isinstance(alvo, ast.Compare):
        alvo.ops[i] = _CMP[type(alvo.ops[i])][k]()
    elif isinstance(alvo, (ast.BinOp, ast.AugAssign)):
        alvo.op = _BIN[type(alvo.op)][k]()
    elif isinstance(alvo, ast.BoolOp):
        alvo.op = ast.Or() if isinstance(alvo.op, ast.And) else ast.And()
    elif isinstance(alvo, ast.UnaryOp):
        _substituir_no_pai(nova, alvo, alvo.operand)
    elif isinstance(alvo, ast.Constant):
        _substituir_no_pai(nova, alvo, ast.Constant(value=(not alvo.value) if isinstance(alvo.value, bool)
                                                    else alvo.value + 1))
    elif isinstance(alvo, (ast.If, ast.While, ast.IfExp)):
        alvo.test = ast.UnaryOp(op=ast.Not(), operand=alvo.test)
    elif isinstance(alvo, ast.Return):
        alvo.value = ast.Constant(value=None)
    elif isinstance(alvo, (ast.Raise, ast.Continue, ast.Break)):
        _substituir_no_pai(nova, alvo, ast.Pass())
    ast.fix_missing_locations(nova)
    return nova


def _substituir(pai, velho, novo):
    for campo, valor in ast.iter_fields(pai):
        if valor is velho:
            setattr(pai, campo, novo)
            return True
        if isinstance(valor, list):
            for j, v in enumerate(valor):
                if v is velho:
                    valor[j] = novo
                    return True
    return False


def _substituir_no_pai(arvore, velho, novo):
    for p in ast.walk(arvore):
        if _substituir(p, velho, novo):
            return
    raise RuntimeError("no sem pai")


def _preparar_worker(base: Path, modulo_rel: str, testes, extras) -> Path:
    base.mkdir(parents=True)
    shutil.copy2(RAIZ / "conftest.py", base / "conftest.py")
    shutil.copytree(RAIZ / "src" / "guaraci", base / "src" / "guaraci",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (base / "tests").mkdir()
    if not modulo_rel.startswith("src/guaraci/"):          # modulo fora do pacote (ex.: scripts/): a copia leva o original
        (base / modulo_rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RAIZ / modulo_rel, base / modulo_rel)
    for t in set(testes) | set(extras):
        origem = RAIZ / t
        destino = base / t
        destino.parent.mkdir(parents=True, exist_ok=True)
        if origem.is_dir():
            shutil.copytree(origem, destino, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(origem, destino)
    return base


def _pytest(base: Path, testes, deselecionar, timeout: int):
    cmd = [sys.executable, "-m", "pytest", *testes, "-x", "-q", "-p", "no:cacheprovider", "-m", "not slow",
           "--no-header", "-o", "addopts="]
    for d in deselecionar:
        cmd += ["--deselect", d]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    env.pop("PYTHONPATH", None)
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, cwd=str(base), env=env, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        return r.returncode, r.stdout + r.stderr, time.perf_counter() - t0
    except subprocess.TimeoutExpired as e:
        return None, (e.stdout or "") if isinstance(e.stdout, str) else "", time.perf_counter() - t0


def _classificar(rc, saida: str):
    if rc is None:
        return "tempo_esgotado", ""
    if rc == 0:
        return "sobrevivente", ""
    m = re.findall(r"^FAILED (\S+)", saida, flags=re.M)
    if rc == 1 and m:
        return "morto", m[0]
    if rc == 1:
        m2 = re.findall(r"^(\S+\.py::\S+)", saida, flags=re.M)
        return "morto", m2[0] if m2 else "(teste nao identificado)"
    erro = re.findall(r"^E\s+(.*)$", saida, flags=re.M)
    return "morto_por_importacao", (erro[0] if erro else saida[-200:])


def executar(args) -> int:
    modulo = (RAIZ / args.modulo).resolve()
    modulo_rel = modulo.relative_to(RAIZ).as_posix()
    fonte_bytes = modulo.read_bytes()
    fonte = fonte_bytes.decode("utf-8").replace("\r\n", "\n")
    arvore = ast.parse(fonte)
    linhas = tuple(int(x) for x in args.linhas.split("-")) if args.linhas else None
    sites = enumerar(arvore, linhas)
    anterior = None
    if args.reexecutar:
        anterior = json.loads(Path(args.reexecutar).read_text(encoding="utf-8"))
        pendentes = {(r["no"], tuple(r["variante"])) for r in anterior["todos"] if r["estado"] == "sobrevivente"}
        sites = [s for s in sites if (s[0], tuple(s[1])) in pendentes]
    print(f"{modulo_rel}: {len(sites)} mutantes; {args.workers} workers", flush=True)
    base = Path(tempfile.mkdtemp(prefix="mut_"))
    extras = ["tests/__init__.py", "tests/_pulados.py", "tests/pulados_esperados.json", "validacao/pulados_esperados.json", "tests/golden",
              *args.extras]
    workers = [_preparar_worker(base / f"w{i}", modulo_rel, args.testes, extras) for i in range(args.workers)]
    # linha de base: sem mutacao, tem de passar (senao todo "morto" e' falso)
    rc, saida, dt = _pytest(workers[0], args.testes, args.deselecionar, args.timeout * 2)
    if rc != 0:
        print("LINHA DE BASE FALHOU -- abortando (kills seriam falsos):\n" + saida[-1500:])
        shutil.rmtree(base, ignore_errors=True)
        return 3
    print(f"linha de base ok em {dt:.0f}s", flush=True)
    fila: "queue.Queue" = queue.Queue()
    for s in sites:
        fila.put(s)
    resultados = []
    trava = threading.Lock()
    linhas_fonte = fonte.split("\n")

    def trabalhar(i):
        w = workers[i]
        alvo = w / modulo_rel
        while True:
            try:
                idx, var, desc, ln = fila.get_nowait()
            except queue.Empty:
                return
            try:
                nova = aplicar(arvore, idx, var)
                texto = ast.unparse(nova)
                alvo.write_text(texto, encoding="utf-8")
                rc, saida, dt = _pytest(w, args.testes, args.deselecionar, args.timeout)
                estado, prova = _classificar(rc, saida)
            except Exception as e:   # noqa: BLE001 -- mutante que nao gera codigo e' registrado, nao some
                estado, prova, dt = "gerador_falhou", f"{type(e).__name__}: {e}", 0.0
            with trava:
                resultados.append({"linha": ln, "operador": desc, "estado": estado, "prova": prova,
                                   "trecho": linhas_fonte[ln - 1].strip()[:140], "segundos": round(dt, 1),
                                   "no": idx, "variante": list(var)})
                if len(resultados) % 25 == 0:
                    n_viv = sum(1 for r in resultados if r["estado"] == "sobrevivente")
                    media = sum(r["segundos"] for r in resultados) / len(resultados)
                    print(f"  {len(resultados)}/{len(sites)}  sobreviventes ate agora: {n_viv}; "
                          f"{media:.0f}s por mutante", flush=True)

    ths = [threading.Thread(target=trabalhar, args=(i,)) for i in range(args.workers)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    shutil.rmtree(base, ignore_errors=True)
    if anterior is not None:       # funde: mortos da rodada anterior ficam; os reexecutados substituem os sobreviventes
        novos = {(r["no"], tuple(r["variante"])): r for r in resultados}
        resultados = [novos.get((r["no"], tuple(r["variante"])), r) for r in anterior["todos"]]
    resultados.sort(key=lambda r: (r["linha"], r["operador"]))
    cont = {}
    for r in resultados:
        cont[r["estado"]] = cont.get(r["estado"], 0) + 1
    total = len(resultados)
    mortos = total - cont.get("sobrevivente", 0) - cont.get("gerador_falhou", 0)
    resumo = {"modulo": modulo_rel, "total": total, "mortos_no_total": mortos, "contagem": cont,
              "python": sys.version.split()[0], "testes": args.testes, "deselecionados": args.deselecionar,
              "sobreviventes": [r for r in resultados if r["estado"] == "sobrevivente"],
              "todos": resultados}
    saida = Path(args.saida) if args.saida else RAIZ / "docs" / "qualidade" / "anova" / f"mutacao_{modulo.stem}.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{modulo_rel}: {mortos}/{total} mortos; contagem {cont}")
    for r in resumo["sobreviventes"]:
        print(f"  SOBREVIVE linha {r['linha']}: {r['operador']}  | {r['trecho']}")
    print(f"detalhes: {saida}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("modulo")
    p.add_argument("--testes", nargs="+", required=True)
    p.add_argument("--deselecionar", nargs="*", default=[])
    p.add_argument("--extras", nargs="*", default=[], help="arquivos/pastas a copiar junto (relativos a raiz), alem dos testes")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    p.add_argument("--timeout", type=int, default=240)
    p.add_argument("--linhas")
    p.add_argument("--reexecutar", help="JSON de uma rodada anterior: roda de novo so' os sobreviventes dela")
    p.add_argument("--saida")
    return executar(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
