"""verificacao_incremental.py -- quais etapas do `verificar_hash_final.py` podem ser reaproveitadas de uma execucao aprovada.

Cada etapa declara os CAMINHOS DE ENTRADA de que depende (`ENTRADAS`, em pathspec do git). O resultado guarda, por etapa, o hash
desses caminhos no commit verificado. Se o hash de agora e' igual ao de uma execucao anterior em que a etapa foi `ok`, a etapa e'
reaproveitada ("reaproveitado de <hash>"); mudou uma entrada -> executa de novo. Etapas baratas e as que olham o repositorio inteiro
(arvore limpa, exposicao, links, estado, auditores, vault) nunca sao reaproveitadas.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

#: o que a suite e os testes por versao leem: codigo, testes, dados do pacote, scripts, documentos executados (exceto o gerado)
_SUITE = ["src", "tests", "scripts", "app_quimiometria.py", "pyproject.toml", "requirements-lock.txt", "MANIFEST.in", "conftest.py",
          "MANUAL.md", "README.md", "README.pt-BR.md", "SECURITY.md", "CHANGELOG.md", "CITATION.cff", "CONTRIBUTING.md",
          "ACKNOWLEDGMENTS.md", "SUPPORT.md", "GOVERNANCE.md", "docs", "validacao", ".semgrep", ":(exclude)docs/qualidade"]
_CODIGO = ["src", "app_quimiometria.py", "pyproject.toml", "requirements-lock.txt", "MANIFEST.in"]

ENTRADAS: Dict[str, List[str]] = {
    "suite_completa": _SUITE,
    "ruff": ["*.py", "pyproject.toml"],
    "mypy": _CODIGO,
    "matriz_cenarios": _CODIGO + ["scripts/matriz_cenarios.py"],
    "bandit": ["src", "pyproject.toml"],
    "semgrep": ["src", "app_quimiometria.py", "scripts", ".semgrep"],
    "pip_audit_pisos": ["pyproject.toml", "scripts/pisos_de_versao.py"],
    "pip_audit_lock": ["requirements-lock.txt"],
    "instalacao_limpa_3.10": _CODIGO + ["scripts/instalacao_limpa.sh", "scripts/percorrer_ajuda_instalada.py", "MANUAL.md"],
    "instalacao_limpa_3.13": _CODIGO + ["scripts/instalacao_limpa.sh", "scripts/percorrer_ajuda_instalada.py", "MANUAL.md"],
    "pytest_3.10": _SUITE, "pytest_3.11": _SUITE, "pytest_3.13": _SUITE,
    "pacote_construido": _CODIGO + ["scripts/auditar_pacote_construido.py", "scripts/percorrer_ajuda_instalada.py", "MANUAL.md", "docs"],
}

REUTILIZAVEIS = frozenset(ENTRADAS)

Git = Callable[..., str]


def _git_padrao(raiz: Path) -> Git:
    def git(*args: str) -> str:
        r = subprocess.run(["git", *args], cwd=str(raiz), capture_output=True, text=True, encoding="utf-8", errors="replace")
        return r.stdout
    return git


def hash_de_entradas(raiz: Path, nome: str, git: Optional[Git] = None) -> Optional[str]:
    """SHA-256 do indice do git (modo, blob, caminho) dos caminhos de entrada de `nome` (arvore limpa: igual a HEAD); None se a etapa nao e' reutilizavel ou nao ha arquivo de entrada."""
    if nome not in ENTRADAS:
        return None
    g = git or _git_padrao(raiz)
    saida = g("ls-files", "-s", "--", *ENTRADAS[nome])
    if not saida.strip():                      # git falhou ou o caminho nao existe: sem entradas nao ha o que comparar
        return None
    return hashlib.sha256(saida.encode("utf-8")).hexdigest()


def carregar_anteriores(pasta: Path) -> List[dict]:
    """Resultados `resultado_*.json` ja gravados, do mais antigo ao mais novo (por `concluido_em`)."""
    out = []
    for f in sorted(pasta.glob("resultado_*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(r, dict) and isinstance(r.get("etapas"), list):
            out.append(r)
    return sorted(out, key=lambda r: str(r.get("concluido_em", "")))


def reaproveitar(nome: str, hash_atual: Optional[str], anteriores: List[dict]) -> Optional[Tuple[str, dict]]:
    """(hash do commit de origem, etapa) da execucao mais recente em que `nome` foi `ok` com as mesmas entradas; None se nao ha."""
    if nome not in REUTILIZAVEIS or not hash_atual:
        return None
    for r in reversed(anteriores):
        for e in r["etapas"]:
            if e.get("nome") == nome and e.get("estado") == "ok" and e.get("entradas") == hash_atual:
                return str(e.get("reaproveitada_de") or r.get("hash", "?"))[:12], e
    return None
