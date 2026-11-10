#!/usr/bin/env python3
"""medir_perfis.py -- mede, pelo CLI, o tempo de cada perfil (objetivo x rigor) no exemplo sintetico
do objetivo e imprime o dicionario TEMPO_PERFIL para colar em catalogo_dados.py.

    python scripts/medir_perfis.py [--so OBJETIVO ...] [--saida arquivo.txt]

Cada combinacao: `guaraci exemplo <exemplo do objetivo>`, as chaves do perfil acrescentadas ao
config.yaml, `guaraci run` (subprocesso, HOME isolado). Falha vira None, com o motivo impresso.
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from guaraci import catalogo as cat  # noqa: E402


def _yaml(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def medir(so=None) -> dict:
    base = Path(tempfile.mkdtemp(prefix="perf"))
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "HOME": str(base), "USERPROFILE": str(base)}
    out = {}
    for obj, po in cat.PERFIS_OBJETIVO.items():
        if not po["cfg"] or (so and obj not in so):
            continue
        for rig, pr in cat.PERFIS_RIGOR.items():
            pasta = base / f"{obj}_{rig}"
            subprocess.run([sys.executable, "-m", "guaraci", "exemplo", po["exemplo"], str(pasta)], env=env,
                           capture_output=True, timeout=600, check=True)
            chaves = dict(po["cfg"], **pr["cfg"])
            cfg = pasta / "config.yaml"
            linhas = [ln for ln in cfg.read_text(encoding="utf-8").splitlines() if ln.split(":")[0] not in chaves]
            linhas += [f"{k}: {_yaml(v)}" for k, v in chaves.items()]
            cfg.write_text("\n".join(linhas) + "\n", encoding="utf-8")
            t0 = time.perf_counter()
            r = subprocess.run([sys.executable, "-m", "guaraci", "run", str(cfg)], env=env, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=7200)
            dt = time.perf_counter() - t0
            out[(obj, rig)] = round(dt) if r.returncode == 0 else None
            print(f"{obj:<12} {rig:<11} {'%6.0f s' % dt if r.returncode == 0 else 'FALHOU: ' + r.stdout[-300:]}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", nargs="*")
    ap.add_argument("--saida", default="")
    a = ap.parse_args()
    res = medir(a.so)
    quando = f"{date.today().isoformat()}, {platform.processor() or platform.machine()}, {os.cpu_count()} nucleos logicos"
    txt = "TEMPO_PERFIL = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in res.items()) + "}\n"
    txt += f"TEMPO_PERFIL_MEDIDO_EM = {quando!r}\n"
    print(txt)
    if a.saida:
        Path(a.saida).write_text(txt, encoding="utf-8")
