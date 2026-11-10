#!/usr/bin/env python3
"""auditar_pacote_construido.py -- constroi wheel + sdist, confere o CONTEUDO e (opcional) percorre a ajuda do wheel instalado.

    python scripts/auditar_pacote_construido.py                    # so' conteudo do wheel e do sdist
    python scripts/auditar_pacote_construido.py --instalar PYTHON  # + venv limpo com esse Python, pip install do wheel e
                                                                   #   `percorrer_ajuda_instalada.py` (rede so' para o pip)

Reprova se: o wheel traz arquivo fora do pacote (dados de terceiros, validacao, datasets, testes, docs do repositorio, privados),
se falta `dados_docs/*.md` ou `perfis_matriz/*.yaml`, se o sdist leva `validacao/`, `datasets_publicos/` ou `docs/qualidade/`, ou se
a ajuda do wheel instalado aponta para fora do pacote. Imprime so' contagens e nomes de arquivos do pacote (sem caminho de maquina).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

RAIZ = Path(__file__).resolve().parents[1]

#: extensoes de DADO que nunca podem estar no wheel (dado de terceiros / medicao)
EXTENSOES_DE_DADO = (".csv", ".xlsx", ".xls", ".spc", ".jdx", ".dx", ".sp", ".opus", ".npy", ".npz", ".h5", ".mat", ".tif", ".tiff",
                     ".zip", ".joblib", ".pkl", ".parquet")
PREFIXOS_PROIBIDOS_NO_SDIST = ("validacao/", "datasets_publicos/", "docs/qualidade/", ".git/", "docs/auditoria/")
PREFIXOS_PROIBIDOS_NO_WHEEL = ("tests/", "docs/", "scripts/", "validacao/", "datasets_publicos/")


#: arquivos de licenca que precisam estar em `*.dist-info/licenses/` (e na raiz do sdist)
LICENCAS_EXIGIDAS_NO_WHEEL = ("LICENSE", "LICENSES/BSD-3-Clause-specio.txt", "THIRD_PARTY_NOTICES.md")


def problemas_do_wheel(nomes: List[str]) -> List[str]:
    p: List[str] = []
    if not any(n.startswith("guaraci/dados_docs/") and n.endswith(".md") for n in nomes):
        p.append("wheel sem guaraci/dados_docs/*.md")
    if not any(n.startswith("guaraci/perfis_matriz/") and n.endswith(".yaml") for n in nomes):
        p.append("wheel sem guaraci/perfis_matriz/*.yaml")
    # `guaraci web` roda este arquivo: sem ele o comando falha a partir do wheel (V07)
    if "guaraci/app_web.py" not in nomes:
        p.append("wheel sem guaraci/app_web.py (`guaraci web` nao subiria)")
    if "guaraci/assets/guaraci_icon.png" not in nomes:
        p.append("wheel sem guaraci/assets/guaraci_icon.png (icone do app web)")
    # V104/V111: o texto da GPL e os avisos de terceiros (BSD-3 do porte do leitor .sp) viajam no wheel
    for exigido in LICENCAS_EXIGIDAS_NO_WHEEL:
        if not any(n.endswith("/licenses/" + exigido) for n in nomes):
            p.append(f"wheel sem {exigido} em dist-info/licenses/")
    for n in nomes:
        if n.lower().endswith(EXTENSOES_DE_DADO):
            p.append(f"wheel traz arquivo de dado: {n}")
        elif n.startswith(PREFIXOS_PROIBIDOS_NO_WHEEL):
            p.append(f"wheel traz arquivo fora do pacote: {n}")
    return p


def problemas_do_sdist(nomes: List[str]) -> List[str]:
    p: List[str] = []
    relativos = {n.split("/", 1)[1] for n in nomes if "/" in n}
    for exigido in LICENCAS_EXIGIDAS_NO_WHEEL:
        if exigido not in relativos:
            p.append(f"sdist sem {exigido}")
    for n in nomes:
        raiz_relativa = n.split("/", 1)[1] if "/" in n else n          # tira `guaraci-X.Y.Z/`
        if raiz_relativa.startswith(PREFIXOS_PROIBIDOS_NO_SDIST):
            p.append(f"sdist traz {raiz_relativa}")
    return p


def construir(destino: Path) -> int:
    r = subprocess.run([sys.executable, "-m", "build", "--outdir", str(destino), str(RAIZ)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("build falhou:", (r.stdout + r.stderr).strip().splitlines()[-1][:200])
    return r.returncode


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instalar", metavar="PYTHON", help="interprete (3.10..3.13) para o venv limpo onde o wheel e' instalado")
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d, "dist")
        if construir(out) != 0:
            return 2
        whl = next(out.glob("*.whl"))
        sdist = next(out.glob("*.tar.gz"))
        with zipfile.ZipFile(whl) as z:
            nw = z.namelist()
        with tarfile.open(sdist) as t:
            ns = t.getnames()
        problemas = problemas_do_wheel(nw) + problemas_do_sdist(ns)
        print(f"wheel: {len(nw)} arquivos; sdist: {len(ns)} arquivos; problemas de conteudo: {len(problemas)}")
        for pr in problemas[:30]:
            print("PROBLEMA", pr)
        rc = 1 if problemas else 0
        if a.instalar:
            venv = Path(d, "venv")
            subprocess.run([a.instalar, "-m", "venv", str(venv)], check=True)
            py = venv / ("Scripts/python.exe" if (venv / "Scripts").is_dir() else "bin/python")
            r = subprocess.run([str(py), "-m", "pip", "install", "--quiet", f"{whl}[reports]"], capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            if r.returncode != 0:
                print("pip install falhou:", (r.stdout + r.stderr).strip().splitlines()[-1][:200])
                return 2
            trabalho = Path(d, "trabalho")
            trabalho.mkdir()
            r = subprocess.run([str(py), "-I", str(RAIZ / "scripts" / "percorrer_ajuda_instalada.py")], cwd=str(trabalho),
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               env={"PATH": "", "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""), "MPLBACKEND": "Agg",
                                    "HOME": str(trabalho), "USERPROFILE": str(trabalho), "PYTHONIOENCODING": "utf-8"})
            print((r.stdout + r.stderr).strip()[-1500:])
            rc = rc or (0 if r.returncode == 0 else 1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
