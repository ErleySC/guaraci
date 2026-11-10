#!/usr/bin/env python3
"""executar_docs.py -- todo comando `guaraci ...` escrito num bloco de codigo dos documentos RODA, com dado sintetico.

    python scripts/executar_docs.py [--listar] [--so ARQUIVO ...] [--json]

Le os blocos ```bash dos documentos de entrada (README, MANUAL, tutoriais de `docs/usuario/`), classifica cada comando e executa
os que sao do produto num diretorio temporario com HOME isolado e rede bloqueada, na ordem em que aparecem (o `exemplo` de uma
linha prepara o `run` da seguinte). Cada comando termina em:
  ok            rodou com codigo de saida 0
  falhou        rodou e devolveu codigo diferente de 0 (ou estourou o tempo)
  excluido      nao e' comando do produto, ou precisa de terminal interativo/servidor/rede -- COM o motivo na tabela
  nao_classificado  comando novo que nenhuma regra conhece: o script reprova, e' preciso classificar (rodar ou excluir com motivo)
Codigo de saida 1 se houver `falhou` ou `nao_classificado`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RAIZ = Path(__file__).resolve().parents[1]
FONTES_FIXAS = ["README.md", "README.pt-BR.md", "MANUAL.md"]
LINGUAGENS = ("bash", "sh", "console", "shell", "")
RX_CERCA = re.compile(r"^```(\w*)\s*$")
RX_PARECE_COMANDO = re.compile(r"^\s*(\$ )?(guaraci|python|pip|cp|streamlit|PYTHONPATH|pytest|export|git|bash|ruff|mypy|cd|set)\b")
TIMEOUT = 900

#: Prefixos de comando que nao sao do produto (instalacao, ambiente do desenvolvedor, git) e o motivo.
EXCLUIDOS_FORA_DO_PRODUTO = {
    "pip": "instalacao de pacote (rede e ambiente do usuario)",
    "cp": "copia de arquivo no ambiente do usuario",
    "git": "comando do git, nao do produto",
    "pytest": "comando de desenvolvimento (a suite roda na CI)",
    "ruff": "comando de desenvolvimento (a suite roda na CI)",
    "mypy": "comando de desenvolvimento (a suite roda na CI)",
    "export": "variavel de ambiente do shell",
    "cd": "navegacao no shell",
    "set": "variavel de ambiente do shell",
    "streamlit": "sobe um servidor",
    "bash": "script de desenvolvimento",
}


def extrair_comandos(texto: str, arquivo: str) -> List[Tuple[str, int, str]]:
    """(arquivo, linha, comando) de cada linha de comando dentro de bloco de codigo de shell; comentarios `# ...` saem."""
    out: List[Tuple[str, int, str]] = []
    dentro, lang = False, ""
    for n, linha in enumerate(texto.splitlines(), 1):
        m = RX_CERCA.match(linha.strip())
        if m:
            dentro = not dentro
            lang = m.group(1) if dentro else ""
            continue
        if dentro and lang in LINGUAGENS and RX_PARECE_COMANDO.match(linha):
            bruto = linha.strip()
            if bruto.startswith("$ "):
                bruto = bruto[2:]
            try:
                partes = shlex.split(bruto, comments=True)
            except ValueError:
                partes = bruto.split()
            if partes:
                out.append((arquivo, n, " ".join(shlex.quote(p) if re.search(r"\s", p) else p for p in partes)))
    return out


def classificar(cmd: str, ha_config: bool) -> Dict[str, object]:
    """{'acao': 'executar'|'excluir'|'nao_classificado', 'argv': [...], 'preparo': [...], 'motivo': str}."""
    p = shlex.split(cmd)
    if not p:
        return {"acao": "nao_classificado", "motivo": "linha vazia"}
    if p[0] in EXCLUIDOS_FORA_DO_PRODUTO:
        return {"acao": "excluir", "motivo": EXCLUIDOS_FORA_DO_PRODUTO[p[0]]}
    if p[0] == "PYTHONPATH=src" or p[0].startswith("PYTHONPATH="):
        return {"acao": "excluir", "motivo": "prefixo de caminho para uso sem instalar"}
    if p[0] == "python":
        if p[1:3] == ["-m", "venv"]:
            return {"acao": "excluir", "motivo": "criacao de ambiente virtual (ambiente do usuario)"}
        if p[1:3] == ["-m", "guaraci.pipeline"] and "--rodar" in p:
            return {"acao": "executar", "argv": [sys.executable, "-m", "guaraci.pipeline", "--rodar"],
                    "preparo": [] if ha_config else [["exemplo", "classificar", "."]]}
        return {"acao": "nao_classificado", "motivo": "python desconhecido"}
    if p[0] != "guaraci":
        return {"acao": "nao_classificado", "motivo": f"comando desconhecido: {p[0]}"}
    resto = p[1:]
    if not resto or resto[0].startswith(("--modo", "--mode", "--perfil")):
        return {"acao": "excluir", "motivo": "abre o menu interativo (precisa de terminal)"}
    if resto[0] == "web":
        return {"acao": "excluir", "motivo": "sobe um servidor web"}
    preparo: List[List[str]] = []
    if resto[0] == "run" and len(resto) == 1 and not ha_config:
        preparo = [["exemplo", "classificar", "."]]
    return {"acao": "executar", "argv": [sys.executable, "-m", "guaraci", *resto], "preparo": preparo}


def _env(home: Path) -> Dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8", "USERPROFILE": str(home), "HOME": str(home),
            "HTTP_PROXY": "http://0.0.0.0:9", "HTTPS_PROXY": "http://0.0.0.0:9", "MPLBACKEND": "Agg", "GUARACI_IDIOMA": "pt"}


def _rodar(argv: List[str], cwd: Path, home: Path) -> Tuple[int, float, str]:
    t0 = time.perf_counter()
    try:
        r = subprocess.run(argv, cwd=str(cwd), env=_env(home), capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=TIMEOUT)
        return r.returncode, time.perf_counter() - t0, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return 124, time.perf_counter() - t0, f"estourou {TIMEOUT}s"


def executar_documento(arquivo: str, comandos: List[Tuple[str, int, str]]) -> List[dict]:
    """Roda os comandos de UM documento, em ordem, num diretorio temporario proprio."""
    res: List[dict] = []
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        home, cwd = base / "home", base / "trabalho"
        home.mkdir()
        cwd.mkdir()
        for _arq, linha, cmd in comandos:
            c = classificar(cmd, (cwd / "config.yaml").exists())
            item = {"arquivo": arquivo, "linha": linha, "comando": cmd}
            if c["acao"] != "executar":
                item.update(estado="excluido" if c["acao"] == "excluir" else "nao_classificado", motivo=c.get("motivo", ""))
                res.append(item)
                continue
            for prep in c.get("preparo", []):                                  # type: ignore[union-attr]
                _rodar([sys.executable, "-m", "guaraci", *prep], cwd, home)
            rc, seg, saida = _rodar(c["argv"], cwd, home)                      # type: ignore[arg-type]
            item.update(estado="ok" if rc == 0 else "falhou", rc=rc, segundos=round(seg, 1))
            if rc != 0:
                item["fim_da_saida"] = saida.strip()[-300:]
            res.append(item)
    return res


def fontes() -> List[str]:
    extra = sorted(os.path.relpath(p, RAIZ).replace("\\", "/") for p in glob.glob(str(RAIZ / "docs" / "usuario" / "*.md")))
    return [f for f in FONTES_FIXAS + extra if (RAIZ / f).exists()]


def comandos_dos_documentos(so: Optional[List[str]] = None) -> Dict[str, List[Tuple[str, int, str]]]:
    out: Dict[str, List[Tuple[str, int, str]]] = {}
    for f in fontes():
        if so and f not in so:
            continue
        out[f] = extrair_comandos((RAIZ / f).read_text(encoding="utf-8"), f)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--listar", action="store_true", help="so' classifica, nao executa")
    ap.add_argument("--so", nargs="*", default=[])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    cmds = comandos_dos_documentos(a.so or None)
    todos: List[dict] = []
    for arq, lista in cmds.items():
        if a.listar:
            ha = False
            for _a, n, c in lista:
                k = classificar(c, ha)
                todos.append({"arquivo": arq, "linha": n, "comando": c, "estado": {"executar": "a_executar", "excluir": "excluido"}.get(
                    str(k["acao"]), "nao_classificado"), "motivo": k.get("motivo", "")})
                ha = ha or c.startswith("guaraci exemplo")
        else:
            todos += executar_documento(arq, lista)
    ruins = [t for t in todos if t["estado"] in ("falhou", "nao_classificado")]
    if a.json:
        print(json.dumps(todos, ensure_ascii=False, indent=2))
    else:
        for t in todos:
            extra = f" ({t.get('motivo') or t.get('segundos', '')})" if t["estado"] != "ok" or t.get("segundos") else ""
            print(f"[{t['estado']:>16}] {t['arquivo']}:{t['linha']}: {t['comando']}{extra}")
        n = {e: sum(1 for t in todos if t["estado"] == e) for e in sorted({t['estado'] for t in todos})}
        print(f"\n{len(todos)} comando(s): {n}")
    return 1 if ruins else 0


if __name__ == "__main__":
    sys.exit(main())
