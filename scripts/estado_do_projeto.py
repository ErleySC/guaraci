#!/usr/bin/env python3
"""estado_do_projeto.py -- o estado do projeto em UMA tela, medido, nunca digitado.

    python scripts/estado_do_projeto.py [--junit R.xml] [--cobertura C.xml] [--matriz docs/qualidade/matriz_cenarios.md]
        [--resultado docs/qualidade/resultado_<hash>.json] [--sem-lint] [--json] [--gravar]

Cada linha traz o numero e a origem dele. O que nao foi medido aparece como "nao medido" (nunca como verde): a suite completa leva
dezenas de minutos e entra por `--junit` (arquivo de `pytest --junitxml`), a cobertura por `--cobertura`, e o veredito final por
`--resultado` (o `resultado_<hash>.json` de `scripts/verificar_hash_final.py`). Sem esses arquivos o script mede o que e' barato:
ramo e hash, consistencia de versao, ruff, mypy, contrato do catalogo, os dois auditores (exposicao e termos de matriz), links
quebrados, referencias nao conferidas e os documentos por destino. `--gravar` escreve `docs/qualidade/estado_do_projeto.md`.
Codigo de saida 1 se alguma medicao falhou.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SAIDA = "docs/qualidade/estado_do_projeto.md"
PRIVADO_LOCAL = Path.home() / ".guaraci_local" / "docs_privados"


def _git(*args: str, raiz: Path = RAIZ) -> str:
    r = subprocess.run(["git", *args], cwd=str(raiz), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout.strip()


def git_estado(raiz: Path = RAIZ) -> dict:
    return {"ramo": _git("rev-parse", "--abbrev-ref", "HEAD", raiz=raiz),
            "hash": _git("rev-parse", "--short", "HEAD", raiz=raiz),
            "sujo": bool(_git("status", "--porcelain", raiz=raiz))}


def _achar(rx: str, texto: str) -> Optional[str]:
    m = re.search(rx, texto, flags=re.M)
    return m.group(1) if m else None


def versoes(raiz: Path = RAIZ) -> Dict[str, Optional[str]]:
    """Versao declarada em cada fonte; `None` = a fonte nao diz."""
    def ler(rel: str) -> str:
        p = raiz / rel
        return p.read_text(encoding="utf-8") if p.exists() else ""
    return {
        "pyproject": _achar(r'^version\s*=\s*"([^"]+)"', ler("pyproject.toml")),
        "CITATION.cff": _achar(r'^version:\s*"?([^"\s]+)"?', ler("CITATION.cff")),
        "config.__version__": _achar(r'^__version__\s*=\s*"([^"]+)"', ler("src/guaraci/config.py")),
        "README (selo)": _achar(r"badge/version-([0-9A-Za-z.]+)-", ler("README.md")),
        "CHANGELOG": _achar(r"hoje `([^`]+)`", ler("CHANGELOG.md")),
    }


def versoes_consistentes(v: Dict[str, Optional[str]]) -> bool:
    valores = {x for x in v.values() if x is not None}
    return len(valores) == 1 and all(x is not None for x in v.values())


def _arquivo_de_teste(tc: ET.Element) -> str:
    return next((x for x in tc.get("classname", "").split(".") if x.startswith("test_")), "")


def ler_junit(caminho: Path, reexecutados: Optional[List[Path]] = None) -> dict:
    """Passados/pulados (com motivo)/falhas/erros de um junitxml do pytest.

    `reexecutados`: junit de reexecucao focada; para cada arquivo de teste que aparece neles, o resultado
    MAIS NOVO substitui o da execucao unica (uma falha corrigida depois da rodada nao fica como falha).
    """
    def casos(c: Path) -> List[ET.Element]:
        raiz = ET.parse(c).getroot()
        suites = [raiz] if raiz.tag == "testsuite" else list(raiz)
        return [tc for s in suites for tc in s.iter("testcase")]

    todos = casos(caminho)
    for extra in reexecutados or []:
        novos = casos(extra)
        arquivos = {_arquivo_de_teste(tc) for tc in novos} - {""}
        todos = [tc for tc in todos if _arquivo_de_teste(tc) not in arquivos] + novos
    passou = falhou = erro = 0
    motivos: List[str] = []
    for tc in todos:
        if tc.find("failure") is not None:
            falhou += 1
        elif tc.find("error") is not None:
            erro += 1
        elif tc.find("skipped") is not None:
            motivos.append(tc.find("skipped").get("message", ""))
        else:
            passou += 1
    return {"passaram": passou, "falharam": falhou, "erros": erro, "pulados": len(motivos), "motivos": motivos}


def pulados_fora_do_previsto(motivos: List[str], ambiente: str = "ci") -> List[str]:
    from tests._pulados import avaliar, carregar_esperados
    _contagem, problemas = avaliar(motivos, carregar_esperados(ambiente))
    return problemas


def pulados_por_motivo(motivos: List[str], ambiente: str = "ci") -> Dict[str, int]:
    from tests._pulados import avaliar, carregar_esperados
    contagem, _ = avaliar(motivos, carregar_esperados(ambiente))
    return dict(contagem)


def ler_cobertura(caminho: Path) -> Optional[float]:
    taxa = ET.parse(caminho).getroot().get("line-rate")
    return None if taxa is None else round(100 * float(taxa), 2)


def _pct(feito: int, total: int) -> float:
    return round(100 * feito / total, 2) if total else 100.0


def cobertura_por_camada(caminho: Path, nucleo: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """Cobertura de linhas, de ramos e combinada, global e do nucleo cientifico, lida do coverage.xml (formato cobertura).

    `nucleo`: caminhos relativos a `src/guaraci/` (ex.: `pipeline.py`) -- o mesmo conjunto de `scripts/congelar_nucleo.py`."""
    alvo = {n.replace("\\", "/").split("src/guaraci/")[-1] for n in nucleo}
    acum = {"global": [0, 0, 0, 0], "nucleo": [0, 0, 0, 0]}          # linhas validas/cobertas, ramos validos/cobertos
    for cls in ET.parse(caminho).getroot().iter("class"):
        fn = (cls.get("filename") or "").replace("\\", "/")
        rel = fn.split("src/guaraci/")[-1] if "src/guaraci/" in fn else fn.split("guaraci/", 1)[-1]
        grupos = [acum["global"]] + ([acum["nucleo"]] if rel in alvo else [])
        for ln in cls.iter("line"):
            for g in grupos:
                g[0] += 1
                g[1] += 1 if int(ln.get("hits", "0")) > 0 else 0
            m = re.search(r"\((\d+)/(\d+)\)", ln.get("condition-coverage") or "")
            if m:
                for g in grupos:
                    g[2] += int(m.group(2))
                    g[3] += int(m.group(1))
    return {k: {"linhas": _pct(v[1], v[0]), "ramos": _pct(v[3], v[2]), "combinada": _pct(v[1] + v[3], v[0] + v[2])}
            for k, v in acum.items()}


def _linhas_tabela(texto: str) -> List[List[str]]:
    out = []
    for ln in texto.splitlines():
        if ln.startswith("|") and not re.match(r"^\|[\s:|-]+\|?$", ln):
            out.append([c.strip() for c in ln.strip().strip("|").split("|")])
    return out


def contar_cenarios(texto: str) -> Dict[str, int]:
    cont: Counter = Counter()
    for c in _linhas_tabela(texto):
        if len(c) >= 5 and c[0] != "cenario":
            r = c[4].lower()
            cont["passou" if r == "passou" else ("nao executado" if r.startswith("nao exec") else "falhou")] += 1
    return dict(cont)


def total_cenarios() -> int:
    import matriz_cenarios
    return len(matriz_cenarios.CENARIOS)


def violacoes_do_catalogo() -> List[str]:
    from guaraci.catalogo import validar_catalogo
    return list(validar_catalogo())


def ocorrencias_de_privacidade(raiz: Path = RAIZ) -> int:
    """Identificador de amostra ou caminho de maquina REAL em arquivo versionado (0 = limpo).
    A sentinela declarada (usuario `alguem`, ano 2099) nao conta: e' exemplo, nao dado."""
    from privacidade_amostras import CATEGORIAS_REAIS, classificar_ocorrencia, varrer_conteudos
    itens: Dict[str, str] = {}
    for rel in _git("ls-files", raiz=raiz).splitlines():
        p = raiz / rel
        if p.suffix.lower() in {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".cff", ".json", ".cfg", ".sh", ".bat"} and p.is_file():
            try:
                itens[rel] = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
    return sum(1 for _r, _n, achado in varrer_conteudos(itens) if classificar_ocorrencia(achado) in CATEGORIAS_REAIS)


# ── Documentos, auditores, links e referencias ───────────────────────────────
def documentos_por_destino(arquivos: List[str], privados: Optional[int] = None) -> Dict[str, int]:
    """Quantos documentos versionados em cada destino publico; `privado-local` e' so' a contagem de arquivos fora do git."""
    cont: Counter = Counter()
    for f in arquivos:
        if f.startswith("docs/") and f.endswith((".md", ".yaml", ".json")):
            partes = f.split("/")
            cont[partes[1] if len(partes) > 2 else "docs (raiz)"] += 1
        elif f in ("README.md", "README.pt-BR.md", "MANUAL.md", "CHANGELOG.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md",
                   "ACKNOWLEDGMENTS.md", "SUPPORT.md", "GOVERNANCE.md"):
            cont["entrada (raiz)"] += 1
    out = dict(sorted(cont.items()))
    if privados is None:
        privados = sum(1 for p in PRIVADO_LOCAL.rglob("*") if p.is_file()) if PRIVADO_LOCAL.exists() else 0
    out["privado-local (fora do git)"] = privados
    return out


def ocorrencias_exposicao() -> int:
    import auditar_exposicao as ae
    return sum(len(v) for v in ae.varrer_arvore(RAIZ).values())


def ocorrencias_termos_de_matriz() -> int:
    import auditar_termos_matriz as at
    fora, _dentro = at.varrer(RAIZ)
    return sum(len(v) for v in fora.values())


def links_quebrados() -> int:
    import verificar_links_docs as vl
    return sum(len(v) for v in vl.quebrados_na_arvore(RAIZ).values())


def referencias_nao_conferidas() -> int:
    import gerar_tabela_referencias as gt
    return len(gt.carregar()[1])


def ler_resultado(caminho: Path) -> Dict[str, Any]:
    """Resumo do `resultado_<hash>.json` do verificar_hash_final: hash verificado, falhas e etapas.

    `reprovadas` = etapas `falhou` ou `nao_executada`; `pendentes_do_autor` e `nao_se_aplica` sao listas separadas (nao sao falha)."""
    r = json.loads(caminho.read_text(encoding="utf-8"))
    etapas = r.get("etapas", [])
    nomes = lambda *estados: [e["nome"] for e in etapas if e.get("estado") in estados]    # noqa: E731
    return {"hash": r.get("hash"), "falhas": r.get("falhas"), "etapas": len(etapas),
            "reprovadas": nomes("falhou", "nao_executada"), "pendentes_do_autor": nomes("pendente_do_autor"),
            "nao_se_aplica": nomes("nao_se_aplica"), "reaproveitadas": nomes_reaproveitadas(etapas),
            "versoes_python": r.get("versoes_python", [])}


def nomes_reaproveitadas(etapas: List[dict]) -> List[str]:
    return [e["nome"] for e in etapas if e.get("origem") == "reaproveitada"]


def _ferramenta(modulo: str, *args: str) -> str:
    r = subprocess.run([sys.executable, "-m", modulo, *args], cwd=str(RAIZ), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    saida = (r.stdout + r.stderr).strip().splitlines()
    return "ok" if r.returncode == 0 else f"FALHOU: {saida[-1] if saida else 'rc=' + str(r.returncode)}"


def medir(args: argparse.Namespace) -> dict:
    m: dict = {"git": git_estado(), "versoes": versoes()}
    m["versoes_ok"] = versoes_consistentes(m["versoes"])
    m["privacidade"] = ocorrencias_de_privacidade()
    m["catalogo"] = violacoes_do_catalogo()
    m["cenarios_total"] = total_cenarios()
    m["documentos"] = documentos_por_destino(_git("ls-files").splitlines())
    m["exposicao"] = ocorrencias_exposicao()
    m["termos_de_matriz"] = ocorrencias_termos_de_matriz()
    m["links_quebrados"] = links_quebrados()
    m["referencias_nao_conferidas"] = referencias_nao_conferidas()
    if not args.sem_lint:
        m["ruff"] = _ferramenta("ruff", "check", "src", "tests", "scripts", "validacao")
        m["mypy"] = _ferramenta("mypy", "src/guaraci")
    if args.junit:
        j = ler_junit(Path(args.junit), [Path(x) for x in (getattr(args, "reexecutado", None) or [])])
        m["testes"] = {k: j[k] for k in ("passaram", "falharam", "erros", "pulados")}
        m["pulados_por_motivo"] = pulados_por_motivo(j["motivos"])
        m["pulados_problemas"] = pulados_fora_do_previsto(j["motivos"])
    if args.cobertura:
        m["cobertura"] = ler_cobertura(Path(args.cobertura))
    matriz = Path(args.matriz) if args.matriz else RAIZ / "docs" / "qualidade" / "matriz_cenarios.md"
    if matriz.exists():
        m["cenarios"] = contar_cenarios(matriz.read_text(encoding="utf-8"))
        m["cenarios_fonte"] = matriz.name
    if args.resultado and Path(args.resultado).exists():
        m["verificacao"] = ler_resultado(Path(args.resultado))
    return m


def falhas(m: dict) -> List[str]:
    f: List[str] = []
    if not m["versoes_ok"]:
        f.append("versao inconsistente entre as fontes")
    if m["privacidade"]:
        f.append(f"{m['privacidade']} ocorrencia(s) de privacidade")
    if m["catalogo"]:
        f.append(f"{len(m['catalogo'])} violacao(oes) do contrato do catalogo")
    for k, rotulo in (("exposicao", "ocorrencia(s) de exposicao"), ("termos_de_matriz", "ocorrencia(s) de termo de matriz fora da validacao"),
                      ("links_quebrados", "link(s) quebrado(s)"), ("referencias_nao_conferidas", "referencia(s) nao conferida(s)")):
        if m.get(k):
            f.append(f"{m[k]} {rotulo}")
    for k in ("ruff", "mypy"):
        if k in m and m[k] != "ok":
            f.append(f"{k}: {m[k]}")
    t = m.get("testes")
    if t and (t["falharam"] or t["erros"]):
        f.append(f"{t['falharam']} falha(s) e {t['erros']} erro(s) nos testes")
    if m.get("pulados_problemas"):
        f.append(f"{len(m['pulados_problemas'])} problema(s) de pulados fora do previsto")
    v = m.get("verificacao")
    if v and (v.get("falhas") or 0):
        f.append(f"verificacao final com {v['falhas']} falha(s)")
    return f


def formatar(m: dict, privado: bool = True) -> str:
    """A tela. `privado=False` (o documento versionado) omite a contagem de arquivos privados: nao interessa a quem le o repositorio."""
    g = m["git"]
    nm = "nao medido"
    L = [f"RAMO {g['ramo']} @ {g['hash']}  arvore {'SUJA' if g['sujo'] else 'limpa'}"]
    v = m["versoes"]
    L.append(f"VERSAO {'consistente' if m['versoes_ok'] else 'INCONSISTENTE'}: " + ", ".join(f"{k}={x}" for k, x in v.items()))
    t = m.get("testes")
    if t:
        pm = "; ".join(f"{k} x{n}" for k, n in m["pulados_por_motivo"].items()) or "-"
        L.append(f"TESTES {t['passaram']} passaram, {t['pulados']} pulados, {t['falharam']} falharam, {t['erros']} erros")
        L.append(f"  pulados por motivo: {pm}")
        L.append("  guarda de pulados: " + ("fora do previsto -> " + "; ".join(m["pulados_problemas"]) if m["pulados_problemas"] else "dentro do previsto"))
    else:
        L.append(f"TESTES {nm} (passe --junit)")
    L.append(f"LINT ruff: {m.get('ruff', nm)}  |  mypy: {m.get('mypy', nm)}")
    c = m.get("cobertura")
    if c is not None:
        import cobertura_excecoes as ce
        L.append(f"COBERTURA global {c}% ({ce.frase_de_escopo(ce.linhas_excluidas())})")
    else:
        L.append(f"COBERTURA {nm} (passe --cobertura)")
    ce = m.get("cenarios")
    tot = m["cenarios_total"]
    if ce:
        L.append(f"CENARIOS pelo CLI ({m.get('cenarios_fonte', '?')}): {ce.get('passou', 0)}/{tot} passaram"
                 + (f", {ce['nao executado']} nao executado(s)" if ce.get('nao executado') else "") + (f", {ce['falhou']} falharam" if ce.get('falhou') else ""))
    else:
        L.append(f"CENARIOS {tot} no catalogo; resultado {nm} (passe --matriz)")
    L.append(f"CONTRATO DE TELA/CATALOGO violacoes: {len(m['catalogo'])}")
    L.append(f"PRIVACIDADE ocorrencias em arquivo versionado: {m['privacidade']}")
    if "exposicao" in m:
        L.append(f"AUDITORES exposicao: {m['exposicao']}  |  termos de matriz fora da validacao: {m['termos_de_matriz']}")
        L.append(f"LINKS quebrados: {m['links_quebrados']}  |  REFERENCIAS nao conferidas: {m['referencias_nao_conferidas']}")
    dc = m.get("documentos")
    if dc and not privado:
        dc = {k: n for k, n in dc.items() if not k.startswith("privado-local")}
    L.append("DOCUMENTOS por destino: " + ", ".join(f"{k} {n}" for k, n in dc.items()) if dc else f"DOCUMENTOS {nm}")
    ve = m.get("verificacao")
    if ve:
        L.append(f"VERIFICACAO FINAL @ {ve['hash']}: {ve['falhas']} falha(s) em {ve['etapas']} etapa(s); Python {', '.join(ve['versoes_python']) or '?'}"
                 + (f"; reprovadas: {', '.join(ve['reprovadas'])}" if ve['reprovadas'] else "")
                 + (f"; pendentes do autor: {', '.join(ve.get('pendentes_do_autor', []))}" if ve.get('pendentes_do_autor') else "")
                 + (f"; nao se aplica: {', '.join(ve.get('nao_se_aplica', []))}" if ve.get('nao_se_aplica') else ""))
    else:
        L.append(f"VERIFICACAO FINAL {nm} (passe --resultado)")
    f = falhas(m)
    L.append("RESULTADO " + ("OK" if not f else "FALHOU -> " + "; ".join(f)))
    return "\n".join(L)


def documento_gerado(m: dict) -> str:
    """`docs/qualidade/estado_do_projeto.md`: a mesma tela, em Markdown, sem caminho de maquina."""
    g = m["git"]
    return ("# Estado do projeto\n\n"
            f"**Público:** quem avalia o estado atual · **Estado:** gerado por `scripts/estado_do_projeto.py` (não editar à mão) · "
            f"**Atualizado em:** `{g['hash']}`\n\n"
            "Cada número vem de um comando; o que não foi medido diz `nao medido`. Testes, cobertura e veredito final vêm dos arquivos "
            "`--junit`, `--cobertura` e `--resultado` informados (a execução pode ser de um commit anterior ao do cabeçalho).\n\n"
            "```text\n" + formatar(m, privado=False) + "\n```\n")


def recusa_de_gravar(m: dict) -> Optional[str]:
    """Motivo para NAO gravar o documento versionado, ou None. Um estado medido em arvore suja nao descreve nenhum commit."""
    g = m["git"]
    if g["sujo"]:
        return (f"--gravar recusado: a arvore esta suja (ramo {g['ramo']} @ {g['hash']}); o estado gravado teria numeros que nenhum commit "
                "reproduz. Faca commit ou descarte as mudancas e rode de novo.")
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--junit")
    ap.add_argument("--reexecutado", action="append", default=[], metavar="JUNIT", help="junit de reexecucao focada; o resultado mais novo de cada arquivo de teste vale")
    ap.add_argument("--cobertura")
    ap.add_argument("--matriz")
    ap.add_argument("--resultado", help="resultado_<hash>.json do verificar_hash_final")
    ap.add_argument("--sem-lint", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--gravar", action="store_true", help=f"grava {SAIDA}")
    a = ap.parse_args(argv)
    m = medir(a)
    print(json.dumps(m, ensure_ascii=False, indent=2) if a.json else formatar(m))
    if a.gravar:
        motivo = recusa_de_gravar(m)
        if motivo:
            print(motivo, file=sys.stderr)
            return 2
        dest = RAIZ / SAIDA
        dest.parent.mkdir(parents=True, exist_ok=True)
        crlf = dest.exists() and b"\r\n" in dest.read_bytes()
        texto = documento_gerado(m)
        dest.write_bytes((texto.replace("\n", "\r\n") if crlf else texto).encode("utf-8"))
    return 1 if falhas(m) else 0


if __name__ == "__main__":
    raise SystemExit(main())
