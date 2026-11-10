#!/usr/bin/env python3
"""medir_vault.py -- CONTAGENS do vault (rede de notas do mantenedor); nunca imprime caminho nem conteudo.

    GUARACI_VAULT=<pasta do vault> [GUARACI_LIVRO_RAZAO=<arquivo do livro-razao>] python scripts/medir_vault.py [--gerar]

`--gerar` roda antes o gerador privado do mantenedor (`GUARACI_VAULT_GERADOR`, script .py que le `GUARACI_VAULT_DIR`).
Sem `GUARACI_VAULT`: sai com 3 e diz "nao executado" (isso e' pendencia do autor, nao aprovacao).
Medidas: notas (fora de `70-Notas-Pessoais/` e `99-Arquivo/`, que sao do autor e nunca sao tocadas), notas do livro-razao e ids
sem nota, notas sem id, notas defasadas (modificadas antes do ultimo commit que mexeu em codigo, manual ou documentos cientificos),
wikilinks quebrados e notas com dado sensivel (caminho de maquina, usuario, e-mail, termo local, codigo de amostra).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

PASTAS_DO_AUTOR = ("70-Notas-Pessoais", "99-Arquivo")
PASTA_LIVRO = "65-Livro-Razao"
_ESTADOS = ("FEITO", "PARCIAL", "ABERTO", "NÃO SE APLICA", "RETRATADO")
_LINK = re.compile(r"\[\[([^\]]+)\]\]")
_ID = re.compile(r"^id_livro: '(.*)'$", re.M)
_CARIMBO = re.compile(r"^commit: ([0-9a-f]{7,40})\s*$", re.M)
CAMINHOS_RELEVANTES = ("src", "MANUAL.md", "docs/cientifico", "tests")


def ids_do_livro(arquivo: Path) -> Dict[str, str]:
    """{id: estado} das linhas de tabela do livro-razao (colunas: id | origem | descricao | dono | estado | evidencia | bloqueia)."""
    itens: Dict[str, str] = {}
    for ln in arquivo.read_text(encoding="utf-8").splitlines():
        if ln.startswith("|"):
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            if len(c) >= 7 and c[4] in _ESTADOS:
                itens[c[0]] = c[4]
    return itens


def _links(texto: str) -> List[str]:
    return [m.group(1).split("|")[0].split("#")[0].strip() for m in _LINK.finditer(texto)]


def notas_do_vault(vault: Path) -> Dict[str, Path]:
    """{caminho relativo (/): arquivo} das notas .md do vault fora das pastas do autor e de pastas ocultas."""
    out: Dict[str, Path] = {}
    for f in vault.rglob("*.md"):
        rel = f.relative_to(vault).as_posix()
        if rel.split("/")[0] in PASTAS_DO_AUTOR or any(p.startswith(".") for p in rel.split("/")):
            continue
        out[rel] = f
    return out


def ultimo_commit_relevante(raiz: Path = RAIZ) -> Optional[str]:
    r = subprocess.run(["git", "log", "-1", "--format=%H %ct", "--", *CAMINHOS_RELEVANTES], cwd=str(raiz), capture_output=True, text=True)
    partes = r.stdout.split()
    return f"{partes[0]} {partes[1]}" if len(partes) == 2 else None


def cobre_o_commit(carimbo: str, relevante: str, raiz: Path = RAIZ) -> bool:
    """A nota foi gerada em `carimbo`; ela esta' em dia se o commit relevante e' ancestral de (ou igual a) esse carimbo."""
    return subprocess.run(["git", "merge-base", "--is-ancestor", relevante, carimbo], cwd=str(raiz), capture_output=True).returncode == 0


def ocorrencias_sensiveis(texto: str) -> int:
    import auditar_exposicao as ae
    rx, locais = ae._regexes(), ae.termos_locais()
    return sum(1 for ln in texto.splitlines() if ae.achar_na_linha(ln, rx, locais))


def medir(vault: Path, livro: Optional[Path], relevante: Optional[str]) -> Dict[str, int]:
    notas = notas_do_vault(vault)
    textos = {rel: f.read_text(encoding="utf-8", errors="replace") for rel, f in notas.items()}
    stems: Set[str] = {Path(rel).stem for rel in notas}
    ids = ids_do_livro(livro) if livro and livro.is_file() else {}
    com_id: Dict[str, str] = {}
    sem_id = 0
    for rel, t in textos.items():
        if rel.startswith(PASTA_LIVRO + "/lr-"):
            m = _ID.search(t)
            if not m:
                sem_id += 1
            else:
                com_id[m.group(1).replace("''", "'")] = rel
    quebrados = sum(1 for t in textos.values() for a in _links(t) if a and a not in stems)
    defasadas = 0
    if relevante:
        hash_rel, instante = relevante.split()
        cache: Dict[str, bool] = {}
        for rel, f in notas.items():
            m = _CARIMBO.search(textos[rel])
            if m:
                c = m.group(1)
                if c not in cache:
                    cache[c] = cobre_o_commit(c, hash_rel)
                defasadas += 0 if cache[c] else 1
            elif f.stat().st_mtime < float(instante):
                defasadas += 1
    return {"notas": len(notas), "ids_no_livro": len(ids), "notas_do_livro": len(com_id),
            "ids_sem_nota": sum(1 for i in ids if i not in com_id),
            "notas_sem_id": sem_id + sum(1 for i in com_id if ids and i not in ids),
            "notas_defasadas": defasadas, "wikilinks_quebrados": quebrados,
            "notas_com_dado_sensivel": sum(1 for t in textos.values() if ocorrencias_sensiveis(t))}


def gerar(vault: Path, gerador: Path) -> int:
    r = subprocess.run([sys.executable, str(gerador)], cwd=str(gerador.parents[1]), capture_output=True, text=True,
                       env={**os.environ, "GUARACI_VAULT_DIR": str(vault)})
    return r.returncode


def zerado(m: Dict[str, int]) -> bool:
    return all(m[k] == 0 for k in ("ids_sem_nota", "notas_sem_id", "notas_defasadas", "wikilinks_quebrados", "notas_com_dado_sensivel"))


def texto(m: Dict[str, int]) -> str:
    return ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in m.items())


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    v = os.environ.get("GUARACI_VAULT")
    if not v or not Path(v).is_dir():
        print("vault: nao executado (GUARACI_VAULT ausente ou invalida) -- pendencia do autor")
        return 3
    vault = Path(v)
    if "--gerar" in args:
        g = os.environ.get("GUARACI_VAULT_GERADOR")
        if not g or not Path(g).is_file():
            print("vault: nao executado (GUARACI_VAULT_GERADOR ausente) -- pendencia do autor")
            return 3
        if gerar(vault, Path(g)) != 0:
            print("vault: o gerador falhou")
            return 1
    livro = os.environ.get("GUARACI_LIVRO_RAZAO")
    m = medir(vault, Path(livro) if livro else None, ultimo_commit_relevante())
    print("vault: " + texto(m))
    return 0 if zerado(m) else 1


if __name__ == "__main__":
    sys.exit(main())
