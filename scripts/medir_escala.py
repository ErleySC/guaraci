"""Mede tempo e memoria de pico do pipeline com cada vez mais espectros (limitacoes.md, secao 8: "escala").

Cada tamanho roda num processo novo (memoria de pico = ru_maxrss do filho), com a mesma configuracao do
`guaraci demo` (classificacao N2 com DD-SIMCA, dados sinteticos de 300 pontos) e so' o numero de amostras
mudando. Opcionalmente mede tambem as funcoes que crescem com n^2 isoladas (Kennard-Stone, Q leave-one-out).

Uso:
    python scripts/medir_escala.py --por-classe 15 70 140 350 --pontos 300 --saida escala.json
    python scripts/medir_escala.py --so-funcoes --n 2000 5000 10000 20000 --pontos 1000

O resultado depende da maquina: o JSON grava CPU, nucleos e memoria total junto com os numeros."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

FILHO_PIPELINE = r"""
import json, os, resource, sys, time, contextlib, io
os.environ.setdefault("GUARACI_IDIOMA", "pt")
from guaraci.config import Config
from guaraci import pipeline as pq
saida, por_classe, pontos = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = Config(input_folder=os.path.join(saida, "dummy"), output_root_folder=saida, mode="sintetico", tag="escala",
             level="N2", objective="auto", n_per_class=por_classe, n_synthetic_points=pontos, wn_min=400.0,
             wn_max=4001.0, synthetic_adulterants=("S", "M", "A"), max_lvs=15, n_splits_cv=3, n_repeats_cv=1,
             n_permutations=50, n_permutations_wold=50, n_bootstrap_vip=10, n_bootstrap_bca=100, n_monte_carlo=20,
             run_benchmark=False, run_monte_carlo=False, run_shap=False, run_wold=False, run_cv_anova=False,
             run_opls=False, executar_etapa4=False, comparar_pipelines=False, compare_hca_pipelines=False)
os.makedirs(cfg.input_folder, exist_ok=True)
t0 = time.perf_counter()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    pq.executar(cfg)
dt = time.perf_counter() - t0
import re
m = re.search(r"Synthetic MODE .*?: (\d+) espectros", buf.getvalue())
print(json.dumps({"espectros": int(m.group(1)) if m else None, "segundos": round(dt, 1),
                  "pico_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0)}))
"""

FILHO_FUNCOES = r"""
import json, resource, sys, time
import numpy as np
from guaraci.dados_io import kennard_stone_split
from guaraci.chemometric_stats import q_residuals_loo
n, p, qual = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
X = np.random.default_rng(0).normal(size=(n, p))
t0 = time.perf_counter()
if qual == "ks":
    kennard_stone_split(X, 0.7)
else:
    q_residuals_loo(X, 5)
print(json.dumps({"funcao": qual, "n": n, "p": p, "segundos": round(time.perf_counter() - t0, 2),
                  "pico_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0)}))
"""


def _maquina() -> dict:
    mem = None
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            mem = round(int(f.readline().split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    return {"python": platform.python_version(), "sistema": platform.platform(), "nucleos": os.cpu_count(),
            "memoria_gb": mem, "processador": platform.processor() or platform.machine()}


def _rodar(codigo: str, args, limite: int) -> dict:
    t0 = time.perf_counter()
    try:
        r = subprocess.run([sys.executable, "-c", codigo, *map(str, args)], capture_output=True, text=True,
                           timeout=limite, check=False)
    except subprocess.TimeoutExpired:
        return {"args": list(map(str, args)), "erro": f"passou de {limite} s"}
    if r.returncode != 0:
        return {"args": list(map(str, args)), "erro": (r.stderr or r.stdout)[-400:],
                "segundos": round(time.perf_counter() - t0, 1)}
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--por-classe", type=int, nargs="*", default=[15, 70, 140])
    ap.add_argument("--pontos", type=int, default=300)
    ap.add_argument("--so-funcoes", action="store_true")
    ap.add_argument("--n", type=int, nargs="*", default=[2000, 5000, 10000])
    ap.add_argument("--limite", type=int, default=3600, help="segundos por tamanho")
    ap.add_argument("--saida", default="")
    a = ap.parse_args()
    resultado = {"maquina": _maquina(), "pipeline": [], "funcoes": []}
    if not a.so_funcoes:
        for k in a.por_classe:
            with tempfile.TemporaryDirectory() as d:
                r = _rodar(FILHO_PIPELINE, [d, k, a.pontos], a.limite)
            r["por_classe"] = k
            r["pontos"] = a.pontos
            print(json.dumps(r), flush=True)
            resultado["pipeline"].append(r)
            if "erro" in r:
                break
    else:
        for qual in ("ks", "q_loo"):
            for n in a.n:
                r = _rodar(FILHO_FUNCOES, [n, a.pontos, qual], a.limite)
                print(json.dumps(r), flush=True)
                resultado["funcoes"].append(r)
                if "erro" in r:
                    break
    if a.saida:
        with open(a.saida, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=1, ensure_ascii=True)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
