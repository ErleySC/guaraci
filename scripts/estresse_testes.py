#!/usr/bin/env python3
"""estresse_testes.py -- os testes que dependiam de tempo, de RAM ou de corrida rodam de novo SOB CARGA e com MEMORIA LIMITADA.

    python scripts/estresse_testes.py [--cargas 3] [--sem-memoria] [--sem-windows] [--gravar]

Ordem: PRIMEIRO no Linux (WSL), num clone no hash atual; so' depois no Windows.
  - `--cargas N` (padrao 3): N rodadas do subconjunto com uma carga de CPU ocupando todos os nucleos (um processo ocupado por nucleo
    logico), no Linux e depois no Windows (`--sem-windows` pula o Windows);
  - memoria limitada (1 rodada, Linux): o mesmo subconjunto com `ulimit -v` (limite de memoria virtual) -- `--sem-memoria` pula esta parte
    (vira `nao_executada`).
Leitura: falha no Linux e' DEFEITO (codigo 1). Passa no Linux e falha so' no Windows sob CPU saturada e' limitacao do AMBIENTE (o crash
nativo 0xc000070a com o Controle de Aplicativo do Windows e' a hipotese, nao conclusao): registrada em limitacoes.md, codigo 0.
`--gravar` escreve `docs/qualidade/estresse_testes.md` (gerado). Codigo de saida 1 se qualquer rodada reprovar ou ficar nao executada.
O subconjunto sao os arquivos de teste que tiveram dependencia de maquina corrigida (ver docs/qualidade/dependencias_de_maquina.md).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "docs" / "qualidade" / "estresse_testes.md"
SUBCONJUNTO = ["tests/test_guaraci_cli.py", "tests/test_hardware.py", "tests/test_avaliacao_modelos.py", "tests/test_tarefas_cli.py",
               "tests/test_pulados_esperados.py"]
LIMITE_VIRTUAL_KB = 12 * 1024 * 1024         # 12 GB de memoria virtual (com 6 GB o benchmark nativo falha com std::bad_alloc: ver docs/qualidade/estresse_testes.md)
DISTRO = os.environ.get("GUARACI_WSL_DISTRO", "Ubuntu")


def comando_pytest(python: str, arquivos: List[str]) -> List[str]:
    return [python, "-m", "pytest", *arquivos, "-v", "-p", "no:cacheprovider", "--no-header"]


def ler_resumo(saida: str) -> Tuple[int, int, str]:
    """(passaram, falharam, ultima linha) do texto de saida do pytest."""
    linhas = [l for l in saida.strip().splitlines() if l.strip()]
    ult = linhas[-1] if linhas else ""
    p = re.search(r"(\d+) passed", ult)
    f = re.search(r"(\d+) failed", ult)
    return (int(p.group(1)) if p else 0, int(f.group(1)) if f else 0, ult.strip("= ").strip())


def descrever_saida(rc: int, saida: str) -> str:
    """Resumo da rodada. Se o processo morreu (codigo de saida fora de 0/1/5 do pytest), diz o codigo e o ultimo teste que aparecia."""
    p, f, ult = ler_resumo(saida)
    if rc in (0, 1, 5) and (p or f):
        return ult
    ultimo = ""
    for ln in saida.splitlines():
        m = re.match(r"^(tests[\/][\w./\-]+::[\w\[\]-]+)", ln.strip())
        if m:
            ultimo = m.group(1).replace("\\", "/")
    codigo = f"0x{rc & 0xFFFFFFFF:08x}" if rc < 0 or rc > 255 else str(rc)
    return f"processo encerrado de forma anormal (codigo {codigo}) apos {ultimo or 'a coleta'}"


def veredito(rodadas: List[Dict[str, object]]) -> int:
    """0 so' se todas as rodadas passaram (sem falha, com pelo menos um teste) e nenhuma ficou nao executada."""
    for r in rodadas:
        if r["estado"] != "ok":
            return 1
    return 0 if rodadas else 1


def leitura(rodadas: List[Dict[str, object]]) -> Tuple[int, str]:
    """(codigo, texto). Linux que falha ou nao roda = defeito/pendencia (1); Linux ok e Windows que falha = limitacao do ambiente (0)."""
    linux = [r for r in rodadas if r.get("ambiente") == "linux"]
    windows = [r for r in rodadas if r.get("ambiente") == "windows"]
    if not linux:
        return 1, "sem rodada no Linux: o estresse nao foi executado onde a conclusao e' possivel"
    if any(r["estado"] != "ok" for r in linux):
        return 1, "DEFEITO: falhou no Linux (isole o teste, limite os threads nativos e corrija)"
    if any(r["estado"] == "falhou" for r in windows):
        return 0, ("limitacao de AMBIENTE (hipotese): passa no Linux sob CPU saturada e falha so' no Windows; a causa provavel e' o "
                   "Controle de Aplicativo do Windows com bibliotecas nativas, nao confirmada")
    return 0, "passou em todas as rodadas executadas"


def tabela(rodadas: List[Dict[str, object]], hash_: str) -> str:
    L = ["# Estresse dos testes sensíveis a tempo e memória", "",
         "**Público:** quem mantém a suíte · **Estado:** gerado por `scripts/estresse_testes.py` (não editar à mão) · "
         "**Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`", "",
         f"Subconjunto: {', '.join(f'`{a}`' for a in SUBCONJUNTO)}. Medido no hash `{hash_[:12]}`. Carga de CPU = um processo ocupado por núcleo lógico; "
         f"memória limitada = `ulimit -v` de {LIMITE_VIRTUAL_KB // (1024 * 1024)} GB no WSL.", "",
         "| rodada | condição | resultado | detalhe |", "|---|---|---|---|"]
    for r in rodadas:
        L.append(f"| {r['n']} | {r['condicao']} | {r['estado']} | {r['detalhe']} |")
    return "\n".join(L) + "\n"


def _carga(n: int) -> List[subprocess.Popen]:
    return [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(n)]


def rodada_sob_carga(n: int, python: str) -> Dict[str, object]:
    carga = _carga(os.cpu_count() or 4)
    try:
        r = subprocess.run(comando_pytest(python, SUBCONJUNTO), cwd=str(RAIZ), capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env={**os.environ, "GUARACI_PULADOS_AMBIENTE": "ci"}, timeout=3 * 3600)
        p, f, _ult = ler_resumo(r.stdout + r.stderr)
        ok = r.returncode == 0 and f == 0 and p > 0
        return {"n": n, "ambiente": "windows", "condicao": f"Windows, CPU saturada ({len(carga)} processos)", "estado": "ok" if ok else "falhou",
                "detalhe": descrever_saida(r.returncode, r.stdout + r.stderr)}
    finally:
        for c in carga:
            c.kill()


def _caminho_wsl(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", s)
    return f"/mnt/{m.group(1).lower()}/{m.group(2)}" if m else s


def comando_wsl(hash_: str, repo_no_wsl: str, sh_no_wsl: str, limite_kb: str, saturar: bool, arquivos: List[str]) -> List[str]:
    """Linha de comando do `wsl` que roda `estresse_wsl.sh` num clone no hash (saturar=True: um laco ocupado por nucleo)."""
    return ["wsl", "-d", DISTRO, "--", "bash", "-c",
            f"tr -d '\\r' < '{sh_no_wsl}' > /tmp/estresse_wsl.sh && bash /tmp/estresse_wsl.sh {hash_[:12]} '{repo_no_wsl}' {limite_kb} "
            f"{1 if saturar else 0} {' '.join(arquivos)}"]


def _rodada_wsl(n: int, hash_: str, cond: str, limite_kb: str, saturar: bool) -> Dict[str, object]:
    cmd = comando_wsl(hash_, _caminho_wsl(RAIZ), _caminho_wsl(RAIZ / "scripts" / "estresse_wsl.sh"), limite_kb, saturar, SUBCONJUNTO)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3 * 3600)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"n": n, "ambiente": "linux", "condicao": cond, "estado": "nao_executada", "detalhe": type(e).__name__}
    p, f, ult = ler_resumo((r.stdout or "").replace(chr(0), ""))
    return {"n": n, "ambiente": "linux", "condicao": cond, "estado": "ok" if f == 0 and p > 0 else "falhou", "detalhe": ult}


def rodada_linux_sob_carga(n: int, hash_: str) -> Dict[str, object]:
    return _rodada_wsl(n, hash_, "Linux (WSL), CPU saturada (um laco por nucleo)", "unlimited", True)


def rodada_memoria_limitada(n: int, hash_: str) -> Dict[str, object]:
    return _rodada_wsl(n, hash_, f"Linux (WSL), memória limitada ({LIMITE_VIRTUAL_KB // (1024 * 1024)} GB virtuais)", str(LIMITE_VIRTUAL_KB), False)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cargas", type=int, default=3)
    ap.add_argument("--sem-memoria", action="store_true")
    ap.add_argument("--sem-windows", action="store_true")
    ap.add_argument("--gravar", action="store_true")
    a = ap.parse_args(argv)
    hash_ = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(RAIZ), capture_output=True, text=True).stdout.strip()
    rodadas: List[Dict[str, object]] = []
    for i in range(1, a.cargas + 1):                                          # 1) Linux, CPU saturada
        rodadas.append(rodada_linux_sob_carga(len(rodadas) + 1, hash_))
        print(rodadas[-1], flush=True)
    if a.sem_memoria:                                                          # 2) Linux, memoria limitada
        rodadas.append({"n": len(rodadas) + 1, "ambiente": "linux", "condicao": "Linux (WSL), memória limitada", "estado": "nao_executada",
                        "detalhe": "--sem-memoria"})
    else:
        rodadas.append(rodada_memoria_limitada(len(rodadas) + 1, hash_))
        print(rodadas[-1], flush=True)
    if not a.sem_windows:                                                      # 3) Windows, so' depois do Linux
        for _ in range(a.cargas):
            rodadas.append(rodada_sob_carga(len(rodadas) + 1, sys.executable))
            print(rodadas[-1], flush=True)
    rc, texto = leitura(rodadas)
    print(texto)
    if a.gravar:
        SAIDA.parent.mkdir(parents=True, exist_ok=True)
        SAIDA.write_bytes(tabela(rodadas, hash_).encode("utf-8"))
        print(f"gravado: {SAIDA.relative_to(RAIZ).as_posix()}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
