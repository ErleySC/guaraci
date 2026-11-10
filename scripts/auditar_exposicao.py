#!/usr/bin/env python3
"""auditar_exposicao.py -- o que NAO pode estar em arquivo versionado nem no historico que sera publicado.

    python scripts/auditar_exposicao.py [--historico RANGO] [--local ARQUIVO_TERMOS] [--json]

Varre os arquivos versionados (e, com `--historico`, todo `git log -p` do intervalo) atras de:
  genericos (valem para qualquer um, ficam na CI): e-mail, telefone, caminho de usuario/OneDrive, token/chave, URL interna, marcas de
    rascunho de contato (`rascunho`/`minuta` + contato), nome de arquivo de amostra de terceiros;
  locais (a lista do autor, FORA do git): `~/.guaraci_local/termos_sensiveis.txt`, um termo por linha (pessoa, instituicao, codigo de
    amostra...). O agente cria o arquivo vazio e le o que o autor colocar; nunca escreve termo real em lugar versionado.

Imprime arquivo:linha e a contagem. NUNCA imprime o trecho (o trecho pode ser exatamente o dado sensivel).
Codigo de saida 1 se houver ocorrencia.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RAIZ = Path(__file__).resolve().parents[1]
LOCAL_PADRAO = Path(os.path.expanduser("~")) / ".guaraci_local" / "termos_sensiveis.txt"

#: Padroes genericos (nome -> regex). Cada um e' testado com mutacao em tests/test_auditar_exposicao.py.
GENERICOS: Dict[str, str] = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}",
    "telefone": r"(?<![\w.])(?:\+?55[\s-]?)?\(?\d{2}\)?[\s-]?9?\d{4}[\s-]\d{4}(?![\w.])",
    "caminho_usuario_windows": r"[A-Za-z]:\\+Users\\+(?!<usuario>|alguem|fulano|seu_usuario|usuario)[A-Za-z0-9_.-]+",
    "caminho_usuario_posix": r"/(?:home|Users)/(?!<usuario>|alguem|fulano|seu_usuario|usuario|runner)[A-Za-z0-9_.-]+/",
    "onedrive": r"(?:[\\/-]OneDrive\b|\bOneDrive[\\/-])",
    "chave_ou_token": r"(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)",
    "url_interna": r"https?://(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|[A-Za-z0-9-]+\.(?:local|internal|intranet))\S*",
    "rascunho_de_contato": r"(?i)\b(?:rascunho|minuta)\b.{0,40}\b(?:contato|e-?mail|carta|oficio|oficio)\b",
}
#: Enderecos que podem aparecer de proposito em arquivo versionado (convencao do GitHub) e dominios reservados para exemplo.
EMAILS_PERMITIDOS = {"noreply@github.com"}
DOMINIOS_DE_EXEMPLO = ("example.com", "example.org", "example.net", "exemplo.com", "exemplo.org", "exemplo.net")
#: Arquivos onde um e-mail pode aparecer. VAZIO de proposito (decisao do autor, lote 10): o projeto nao publica e-mail; o contato e'
#: o das GitHub Discussions e o canal privado de vulnerabilidade do GitHub. Nome mantido porque a reconstrucao do historico o consulta.
EMAIL_PERMITIDO_EM: Tuple[str, ...] = ()
#: Onde os padroes genericos nao se aplicam (o auditor e o seu teste carregam os padroes).
IGNORAR = ("scripts/auditar_exposicao.py", "tests/test_auditar_exposicao.py")
EXT = (".py", ".md", ".yaml", ".yml", ".toml", ".json", ".txt", ".cff", ".bat", ".cfg", ".ini", ".sh", ".html", ".ipynb", ".bib", ".R")


def _norm(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")


def termos_locais(caminho: Optional[Path] = None) -> List[str]:
    """Lista local do autor (um termo por linha, `#` comenta). Cria o arquivo VAZIO se nao existir; nunca escreve nele."""
    p = Path(caminho) if caminho else LOCAL_PADRAO
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# Um termo por linha (nome de pessoa, instituicao, codigo de amostra...). Este arquivo NAO e' versionado.\n", encoding="utf-8")
    out = []
    for l in p.read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if l and not l.startswith("#"):
            out.append(_norm(l))
    return out


def termos_locais_se_existirem(caminho: Optional[Path] = None) -> Optional[List[str]]:
    """Como `termos_locais`, mas SEM criar o arquivo: devolve `None` se ele nao existe (maquina sem a lista do autor, CI)."""
    p = Path(caminho) if caminho else LOCAL_PADRAO
    return termos_locais(p) if p.exists() else None


def _regexes() -> Dict[str, re.Pattern]:
    return {k: re.compile(v) for k, v in GENERICOS.items()}


def achar_na_linha(linha: str, rx: Dict[str, re.Pattern], locais: List[str], arquivo: str = "") -> List[str]:
    """Nomes das regras que a linha viola (nunca o trecho). `arquivo` so entra na regra se `EMAIL_PERMITIDO_EM` voltar a ter itens."""
    achados = []
    for nome, r in rx.items():
        for m in r.finditer(linha):
            if nome == "email" and (m.group(0).lower() in EMAILS_PERMITIDOS or arquivo in EMAIL_PERMITIDO_EM
                                    or m.group(0).lower().endswith(DOMINIOS_DE_EXEMPLO)):
                continue
            achados.append(nome)
            break
    n = _norm(linha)
    for t in locais:
        if t and t in n:
            achados.append("termo_local")
            break
    return achados


def _ha_arquivos_versionados(raiz: Path) -> bool:
    """False sem git ou fora de um clone: sem isto a varredura sairia '0 ocorrencias' sem ter olhado nada."""
    try:
        r = subprocess.run(["git", "ls-files"], cwd=raiz, capture_output=True, text=True, encoding="utf-8")
    except OSError:
        return False
    return bool(r.stdout.split())


def varrer_arvore(raiz: Optional[Path] = None, locais: Optional[List[str]] = None) -> Dict[str, List[Tuple[int, str]]]:
    raiz = raiz or RAIZ
    rx = _regexes()
    locais = termos_locais() if locais is None else locais
    r = subprocess.run(["git", "ls-files"], cwd=raiz, capture_output=True, text=True, encoding="utf-8")
    out: Dict[str, List[Tuple[int, str]]] = {}
    for f in r.stdout.split():
        if f.endswith(EXT) and f not in IGNORAR:
            try:
                linhas = (raiz / f).read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for n, l in enumerate(linhas, 1):
                for regra in achar_na_linha(l, rx, locais, f):
                    out.setdefault(f, []).append((n, regra))
    return out


def varrer_historico(intervalo: str, raiz: Optional[Path] = None, locais: Optional[List[str]] = None,
                     incluir_removidas: bool = False) -> Dict[str, int]:
    """Conta violacoes por regra em `git log -p INTERVALO`; nunca guarda o trecho.

    Conta as linhas ADICIONADAS (o que a publicacao do intervalo expoe de novo). As linhas removidas ja estavam na base do intervalo,
    isto e', ja eram publicas: so' entram com `incluir_removidas=True`. O arquivo de uma linha removida vem do cabecalho `--- a/`."""
    raiz = raiz or RAIZ
    rx = _regexes()
    locais = termos_locais() if locais is None else locais
    p = subprocess.Popen(["git", "log", "-p", "--no-color", "--format=commit %H", intervalo], cwd=raiz, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    cont: Dict[str, int] = {}
    arquivo = ""
    assert p.stdout is not None
    for bruto in p.stdout:
        linha = bruto.decode("utf-8", errors="replace").rstrip("\n")
        if linha.startswith("--- a/"):
            arquivo = linha[6:]
            continue
        if linha.startswith("+++ b/"):
            arquivo = linha[6:]
            continue
        sinais = ("+", "-") if incluir_removidas else ("+",)
        if linha.startswith(sinais) and not linha.startswith(("+++", "---")) and arquivo.endswith(EXT) and arquivo not in IGNORAR:
            for regra in achar_na_linha(linha[1:], rx, locais, arquivo):
                cont[regra] = cont.get(regra, 0) + 1
    p.wait()
    return cont


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--historico", metavar="INTERVALO", help="ex.: origin/master..HEAD")
    ap.add_argument("--local", metavar="ARQUIVO", help=f"lista local de termos (padrao {LOCAL_PADRAO.name} em ~/.guaraci_local)")
    ap.add_argument("--incluir-removidas", action="store_true", help="no historico, conta tambem as linhas removidas (padrao: so' as adicionadas)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if not _ha_arquivos_versionados(RAIZ):
        print("ERRO: nenhum arquivo versionado encontrado (git ausente ou pasta que nao e' um clone): nada foi verificado", file=sys.stderr)
        return 2
    locais = termos_locais(Path(a.local) if a.local else None)
    arvore = varrer_arvore(locais=locais)
    total = sum(len(v) for v in arvore.values())
    hist = varrer_historico(a.historico, locais=locais, incluir_removidas=a.incluir_removidas) if a.historico else None
    if a.json:
        print(json.dumps({"arvore": {k: len(v) for k, v in arvore.items()}, "total_arvore": total, "historico": hist,
                          "termos_locais": len(locais), "r5": "executado" if locais else "nao_executado"}, indent=2))
    else:
        for f, v in sorted(arvore.items()):
            regras = sorted({r for _, r in v})
            print(f"{f}: {len(v)} ocorrencia(s) [{', '.join(regras)}] linhas {', '.join(str(n) for n, _ in v[:10])}")
        print(f"arvore: {total} ocorrencia(s) em {len(arvore)} arquivo(s); termos locais lidos: {len(locais)}")
        if not locais:
            print("R5 (lista local de termos) NAO EXECUTADO: a lista esta vazia; as demais regras rodaram")
        if hist is not None:
            print(f"historico {a.historico}: {sum(hist.values())} ocorrencia(s) " + (str(hist) if hist else ""))
    return 1 if total or (hist and sum(hist.values())) else 0


if __name__ == "__main__":
    sys.exit(main())
