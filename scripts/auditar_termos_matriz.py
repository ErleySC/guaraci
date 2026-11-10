#!/usr/bin/env python3
"""auditar_termos_matriz.py -- o produto e' por TECNICA e TAREFA, nao por matriz (milho, oleo, azeite, carne...).

    python scripts/auditar_termos_matriz.py [--por-arquivo] [--json]

Varre os arquivos versionados e conta, por arquivo:linha, os termos de matriz que aparecem FORA dos lugares onde sao
permitidos (referencias bibliograficas, metadados de citacao e atribuicao). Conta tambem nomes de repositorio de dados
(citar o repositorio e' citar o conjunto) e, quando o dossie privado de calibracao existe (fora do git), os nomes e DOIs
dos conjuntos de terceiros listados nele -- os nomes nunca ficam no repositorio. Imprime arquivo:linha e a contagem,
nunca o trecho da linha. Codigo de saida 1 se houver ocorrencia fora da lista permitida.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RAIZ = Path(__file__).resolve().parents[1]

#: Termos de matriz (sem acento e em minusculas; a comparacao ignora acento e caixa).
TERMOS = [
    "milho", "maize", "corn", "oleo", "oleos", "azeite", "oliva", "olive", "lavanda", "lavender", "suino", "porco", "pork",
    "carne", "meat", "beef", "leite", "milk", "mel", "honey", "vinho", "wine", "solo", "soil", "mandioca", "cassava", "urina", "urine",
    "esgoto", "wastewater", "kiwi", "kaki", "persimmon", "mango", "manga", "fruta", "fruit", "graos", "grao", "grain", "cafe", "coffee",
    "trigo", "wheat", "soja", "soy", "arroz", "rice", "castanha", "dende", "palma", "oil",
    "oils", "sesamo", "girassol", "sunflower", "canola",
    # outras matrizes de aplicacao comum (a lista precisa cobrir mais do que o que ja' apareceu no projeto)
    "peixe", "fish", "cacau", "cocoa", "cerveja", "tomate", "tomato", "batata", "potato", "queijo", "cheese", "manteiga", "butter",
    "frango", "chicken", "gasolina", "gasoline", "diesel", "biodiesel", "petroleo", "petroleum", "farinha", "flour", "gordura",
    "hortela", "gengibre", "ginger", "propolis", "tabaco", "tobacco", "algodao", "cotton", "couro", "leather", "suco", "juice",
    "farmaco", "aspirina", "cafeina", "caffeine",
    # repositorios de dados: citar o repositorio e' citar o conjunto (politica de dados de terceiros)
    "zenodo", "figshare", "mendeley", "dryad", "kaggle",
]
#: Palavras ambiguas que so' contam como matriz em contexto (evita falso positivo de "solo" = "somente" etc.).
CONTEXTO = {"olive": r"\bolive\s+oils?\b|\bolives?\s+(?:and|or)\s+\w+\s+oils?\b", "lavender": r"\blavender\s+(?:oil|gc|essential|samples?)",
            "solo": r"\bsolos?\b", "mel": r"\bmel\b", "oil": r"\b(?:edible|olive|vegetable|fixed|essential)\s+oils?\b|\boils?\s+(?:profile|sample)",
            "oils": r"\b(?:edible|vegetable|fixed)\s+oils\b", "corn": r"\bcorn\b", "palma": r"\bpalma\b", "fruit": r"\bfruits?\b",
            "manga": r"\bmanga\b", "milk": r"\bmilk\b", "rice": r"\brice\b", "wine": r"\bwine\b", "meat": r"\bmeat\b", "honey": r"\bhoney\b"}

#: Onde a matriz e' permitida (prefixos de caminho).
PERMITIDO = (
    "docs/cientifico/referencias_verificadas.yaml", "CITATION.cff", "paper/paper.bib",
    # o proprio auditor carrega o lexico de termos (e seu teste)
    "scripts/auditar_termos_matriz.py", "tests/test_auditar_termos_matriz.py",
)
#: Arquivos que nao sao produto nem documentacao do produto (historico e dados de teste).
IGNORAR_PREFIXOS = ("CHANGELOG.md", "docs/revisao_pares_", "tests/fixtures/")
#: Linha com termo de matriz que e' parte de uma referencia bibliografica (titulo de artigo citado) esta a ate' 2 linhas de um DOI.
RX_DOI = re.compile(r"doi[:\s]|10\.\d{4,}/", re.I)
#: DOI de repositorio de dados NAO isenta: citar o deposito e' citar o conjunto (politica de dados de terceiros).
RX_DOI_DADOS = re.compile(r"(?:doi[:\s]*)?10\.(?:17632|6084|25678|5061|7910|5281/zenodo)[./]\S*", re.I)


def _isenta_por_doi(janela: str) -> bool:
    """A janela tem DOI de publicacao (nao de repositorio de dados)?"""
    return bool(RX_DOI.search(RX_DOI_DADOS.sub(" ", janela)))
EXT = (".py", ".md", ".yaml", ".yml", ".toml", ".json", ".txt", ".cff", ".bat", ".cfg", ".html")
#: Dossie privado de calibracao do mantenedor (fora do git); lido so' quando existe.
DOSSIE = Path(os.path.expanduser("~")) / ".guaraci_local" / "validacao" / "DOSSIE_CALIBRACAO.md"


def termos_do_dossie(caminho: Path) -> List[str]:
    """Nomes de conjunto (palavras capitalizadas do titulo de cada secao de conjunto) e DOIs do dossie privado.

    Secoes de conjunto: as `## ` entre a de conjuntos sem licenca e a de arquivos de terceiros do historico."""
    if not caminho.is_file():
        return []
    termos = set()
    dentro = False
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if not linha.startswith("## "):
            continue
        titulo = linha[3:]
        if titulo.lower().startswith("conjuntos sem"):
            dentro = True
            continue
        if titulo.lower().startswith("arquivos de terceiros"):
            break
        if not dentro:
            continue
        for tok in re.split(r"[\s/]+", titulo.split("(")[0]):
            if re.fullmatch(r"[A-Z][A-Za-z]{3,}", tok):
                termos.add(tok)
        termos.update(d.rstrip(".,;") for d in re.findall(r"10\.\d{4,}/[^\s()]+", titulo))
    return sorted(termos)


def _rx_dossie(caminho: Path) -> Optional[re.Pattern]:
    """Nomes e DOIs do dossie, com caixa (um nome proprio nao casa com a palavra comum em minusculas)."""
    termos = termos_do_dossie(caminho)
    if not termos:
        return None
    return re.compile("|".join(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])" for t in termos))


def _norm(t: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")


#: Em prosa e metadado (nao em codigo) o termo colado por sublinhado (`oleos_essenciais`) tambem e' matriz: `\b` nao separa em `_`.
EXT_PROSA = (".md", ".yaml", ".yml", ".toml", ".txt", ".cff")


def _rx(composto: bool = False) -> re.Pattern:
    """`composto`: o termo conta mesmo colado a outro por `_` (so' em prosa; em codigo `_` faz parte do identificador)."""
    simples = [t for t in TERMOS if t not in CONTEXTO]
    if composto:
        partes = [rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])" for t in simples]
    else:
        partes = [rf"\b{re.escape(t)}\b" for t in simples]
    return re.compile("|".join(partes + list(CONTEXTO.values())))


def arquivos(raiz: Optional[Path] = None) -> List[str]:
    r = subprocess.run(["git", "ls-files"], cwd=raiz or RAIZ, capture_output=True, text=True, encoding="utf-8")
    return [f for f in r.stdout.split() if f.endswith(EXT)]


def varrer(raiz: Optional[Path] = None, dossie: Optional[Path] = None) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """(fora do permitido, dentro do permitido): arquivo -> linhas."""
    raiz = raiz or RAIZ
    rx = _rx()
    rx_prosa = _rx(composto=True)
    rxd = _rx_dossie(dossie or DOSSIE)
    fora: Dict[str, List[int]] = {}
    dentro: Dict[str, List[int]] = {}
    for f in arquivos(raiz):
        if f.startswith(IGNORAR_PREFIXOS) or "/__pycache__/" in f:
            continue
        try:
            linhas = (raiz / f).read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        alvo = dentro if f.startswith(PERMITIDO) else fora
        for n, linha in enumerate(linhas, 1):
            if (rx_prosa if f.endswith(EXT_PROSA) else rx).search(_norm(linha)) or (rxd is not None and rxd.search(linha)):
                janela = " ".join(linhas[max(0, n - 3):n + 2])
                (dentro if _isenta_por_doi(janela) else alvo).setdefault(f, []).append(n)
    return fora, dentro


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--por-arquivo", action="store_true", help="imprime arquivo:linha (sem o trecho)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if not arquivos():
        print("ERRO: nenhum arquivo versionado encontrado (git ausente ou pasta que nao e' um clone): nada foi verificado", file=sys.stderr)
        return 2
    fora, dentro = varrer()
    total_fora = sum(len(v) for v in fora.values())
    total_dentro = sum(len(v) for v in dentro.values())
    if a.json:
        print(json.dumps({"fora": {k: len(v) for k, v in fora.items()}, "total_fora": total_fora, "total_permitido": total_dentro}, indent=2))
    else:
        if a.por_arquivo:
            for f, ls in sorted(fora.items()):
                print(f"{f}: {len(ls)} linha(s): {', '.join(map(str, ls[:15]))}{' ...' if len(ls) > 15 else ''}")
        por_pasta = Counter()
        for f, ls in fora.items():
            por_pasta[f.split("/")[0] if "/" in f else "(raiz)"] += len(ls)
        print(f"termos de matriz fora do permitido: {total_fora} linha(s) em {len(fora)} arquivo(s) (permitido: {total_dentro})")
        for k, v in por_pasta.most_common():
            print(f"  {k}: {v}")
    return 1 if total_fora else 0


if __name__ == "__main__":
    sys.exit(main())
