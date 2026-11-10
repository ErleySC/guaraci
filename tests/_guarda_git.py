"""Guarda de sessao do repositorio real (vistoria 2026-10, fato 0.1-6).

Um teste que rode `git stash`, `git checkout`, `git reset` ou `git commit` no
repositorio REAL (em vez de num repositorio temporario) pode sumir com o
trabalho nao commitado do desenvolvedor sem nenhum aviso. Esta guarda fotografa
o estado no inicio da sessao de testes e reprova a sessao no fim se ele mudou.

SOMENTE LEITURA: so' roda `git rev-parse`, `git symbolic-ref`, `git stash list`
e `git --no-optional-locks status` (este ultimo nao reescreve o indice).

O que e' comparado, e por que:
* HEAD (commit) e o ramo atual -- commit, reset ou checkout feitos por teste;
* `git stash list` -- stash feito (ou consumido) por teste;
* `git status --porcelain --untracked-files=no` -- SO' arquivos rastreados:
  pega `git checkout -- arq`/`git restore`/`reset --hard` que desfazem edicao
  nao commitada. Arquivos NAO rastreados ficam de fora de proposito: testes e
  ferramentas criam caches e saidas ignoraveis, e o desenvolvedor costuma
  criar arquivos novos enquanto a suite roda (falso alarme). Editar um arquivo
  RASTREADO durante a sessao ainda dispara o aviso; a mensagem diz isso.

Sem `.git` na raiz (zip, sdist, copia exportada) ou sem o executavel `git`, a
guarda nao fotografa nada e a checagem e' pulada sem erro.
"""
from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional

_TEMPO_MAXIMO_S = 30


def _git(raiz: str, *args: str) -> Optional[str]:
    try:
        r = subprocess.run(  # nosec B603 B607 -- lista fixa, sem shell, so' leitura
            ["git", "--no-optional-locks", "-C", raiz, *args],
            capture_output=True, text=True, timeout=_TEMPO_MAXIMO_S,
            encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def fotografar(raiz: str) -> Optional[Dict[str, str]]:
    """Estado do repositorio em `raiz`, ou None se nao der para fotografar
    (sem `.git` na propria raiz, sem git instalado, repositorio quebrado)."""
    if not os.path.exists(os.path.join(raiz, ".git")):
        return None
    head = _git(raiz, "rev-parse", "HEAD")
    if head is None:
        return None
    ramo = _git(raiz, "symbolic-ref", "--quiet", "HEAD")
    stash = _git(raiz, "stash", "list")
    status = _git(raiz, "status", "--porcelain", "--untracked-files=no")
    if stash is None or status is None:
        return None
    return {"HEAD": head.strip(), "ramo": (ramo or "(HEAD destacado)").strip(),
            "stash": stash.strip(), "status_rastreados": status.strip()}


def comparar(antes: Optional[Dict[str, str]],
             depois: Optional[Dict[str, str]]) -> List[str]:
    """Lista de diferencas legiveis. Vazia se igual ou se alguma foto faltar."""
    if antes is None or depois is None:
        return []
    problemas = []
    for chave, nome in (("HEAD", "HEAD (commit)"), ("ramo", "ramo atual"),
                        ("stash", "git stash list"),
                        ("status_rastreados", "status dos arquivos rastreados")):
        if antes.get(chave) != depois.get(chave):
            problemas.append(
                f"{nome} mudou durante os testes: antes={antes.get(chave)!r} "
                f"depois={depois.get(chave)!r}")
    if problemas:
        problemas.append(
            "Algum teste mexeu no repositorio real (use um repositorio "
            "temporario). Se foi voce quem editou/commitou durante a sessao, "
            "ignore e rode de novo.")
    return problemas
