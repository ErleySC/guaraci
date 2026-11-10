#!/usr/bin/env python3
"""camadas_de_teste.py -- UM comando por camada de teste.

    python scripts/camadas_de_teste.py rapido     # cada commit: suite sem `slow`, sem cobertura (alvo <= 10 min)
    python scripts/camadas_de_teste.py cobertura  # uma vez por ciclo: suite COMPLETA (com `slow`) com cobertura de ramos e resumo por camada
    python scripts/camadas_de_teste.py completo   # so' no hash final: tudo, com `slow` (a verificacao final adiciona o estresse)

Paralelismo: `-n` = metade dos nucleos, no maximo 4 (pytest-xdist). A pasta temporaria e' curta (`--basetemp`): no Windows o caminho de
temporario do xdist estoura o limite de 260 caracteres. Argumentos extras depois de `--` vao direto ao pytest.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

RAIZ = Path(__file__).resolve().parents[1]
CAMADAS = ("rapido", "cobertura", "completo")


def workers(nucleos: Optional[int] = None) -> int:
    """Metade dos nucleos, no maximo 4 (e no minimo 1)."""
    n = nucleos if nucleos is not None else (os.cpu_count() or 2)
    return max(1, min(4, n // 2))


def comando(camada: str, n_workers: int, basetemp: str, saida_xml: str = "", extra: Sequence[str] = ()) -> List[str]:
    """Linha de comando do pytest para a camada."""
    if camada not in CAMADAS:
        raise ValueError(f"camada desconhecida: {camada!r}; use {', '.join(CAMADAS)}")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", f"--basetemp={basetemp}"]
    if n_workers > 1:
        cmd += ["-n", str(n_workers)]
    if camada == "rapido":                       # a cobertura e' medida na suite completa: os testes `slow` cobrem figuras e extras
        cmd += ["-m", "not slow"]
    if camada == "cobertura":
        cmd += ["--cov=guaraci", "--cov-branch"]
        if saida_xml:
            cmd += [f"--cov-report=xml:{saida_xml}"]
    return cmd + list(extra)


def resumo_da_cobertura(xml: Path) -> str:
    """Linhas, ramos e combinada: global e do nucleo cientifico (o mesmo calculo da verificacao final)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import congelar_nucleo as cn
    import estado_do_projeto as ep
    c = ep.cobertura_por_camada(xml, cn.arquivos_do_nucleo())
    return "\n".join(f"{k}: linhas {v['linhas']}% | ramos {v['ramos']}% | combinada {v['combinada']}%" for k, v in c.items())


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("camada", choices=CAMADAS)
    ap.add_argument("--workers", type=int, help="padrao: metade dos nucleos, no maximo 4")
    ap.add_argument("extra", nargs="*", help="argumentos para o pytest (depois de --)")
    a = ap.parse_args(argv)
    base_curta = Path(os.environ.get("SystemDrive", "C:") + "/") / "gt" if os.name == "nt" else Path("/tmp/gt")
    base_curta.mkdir(parents=True, exist_ok=True)
    basetemp = str(base_curta / "camada")
    saida_xml = str(base_curta / "cobertura.xml") if a.camada == "cobertura" else ""
    cmd = comando(a.camada, a.workers or workers(), basetemp, saida_xml, a.extra)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"}
    rc = subprocess.run(cmd, cwd=RAIZ, env=env).returncode
    if a.camada == "cobertura" and Path(saida_xml).exists():
        print(resumo_da_cobertura(Path(saida_xml)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
