#!/usr/bin/env python
"""Verifica TODA citacao com DOI do repositorio contra o registro oficial do DOI.

Origem: parecer de revisao por pares de 2026-09-25 (B6, m-01). Uma referencia
com DOI de outro artigo chegou a ser apresentada ao usuario como base
cientifica porque, nas rodadas anteriores, Crossref e doi.org estavam
inacessiveis e a conferencia foi pulada. Este script torna a conferencia uma
etapa AUTOMATICA e repetivel.

O que faz
---------
1. Extrai todo DOI de todo arquivo versionado (codigo, docs, .bib, YAML,
   textos de interface). Um DOI que nao esta no registro
   `docs/cientifico/referencias_verificadas.yaml` FALHA.
2. Para cada entrada do registro consulta o registro do DOI (Crossref, ou
   DataCite para DOIs de repositorios de dados) e
   compara com o que esta declarado no registro:
   - sobrenome do primeiro autor (quando declarado),
   - ano (quando declarado),
   - um trecho distintivo do titulo (quando declarado).
   Qualquer divergencia FALHA.
3. `--sugerir` lista os DOIs ainda fora do registro, com os metadados do
   registro oficial e o contexto onde aparecem, para conferencia HUMANA
   (a decisao "esta citacao e' o trabalho que o texto diz" e' de quem le o
   contexto; o script so' garante que o registro nao envelheca nem seja
   digitado errado).
4. `--sem-rede` so' faz a parte 1 (uso offline / CI sem rede).
5. `--rede` acrescenta a conferencia de que cada DOI resolve em doi.org (uma entrada
   `retirado: true` deve NAO resolver). Desligada por padrao.

Uso
---
    python scripts/verificar_citacoes.py            # verifica tudo (rede)
    python scripts/verificar_citacoes.py --sugerir  # o que falta registrar
    python scripts/verificar_citacoes.py --sem-rede # so' cobertura do registro

Codigo de saida: 0 tudo conferido | 1 divergencia/DOI nao registrado |
2 erro de uso ou de rede (nunca "passa" por falta de rede sem --sem-rede).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

RAIZ = Path(__file__).resolve().parents[1]
REGISTRO = RAIZ / "docs" / "cientifico" / "referencias_verificadas.yaml"
CACHE = RAIZ / ".cache_citacoes.json"          # nao versionado (.gitignore)
_UA = "guaraci-verificar-citacoes/1.0 (+https://github.com/ErleySC/guaraci)"

# Arquivos que nao sao texto pesquisavel.
_EXT_IGNORADAS = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".joblib",
                  ".mat", ".npz", ".zip", ".xlsx", ".pptx", ".docx", ".ipynb_checkpoints"}
# Pastas de material de terceiros / anexo do parecer (contem DOIs errados
# de proposito, citados como achados).
_PREFIXOS_IGNORADOS = ("docs/revisao_pares_", "docs/cientifico/referencias.md", ".claude/", "build/", "dist/")
# Arquivos historicos/de auditoria que CITAM DOIs errados como achados; os DOIs
# errados conhecidos ficam em `conhecidos_incorretos` no registro.
_RX_DOI = re.compile(
    r"(?i:10\.1002/\(SICI\)[^\s\"']+)|10\.\d{4,9}/[^\s\"'<>\[\]{},;|`\\]+")


def _limpar_doi(d: str) -> str:
    d = d.rstrip(".:;*_")
    while d.endswith(")") and d.count("(") < d.count(")"):
        d = d[:-1]
    d = d.rstrip(".:;*_")
    return d


def arquivos_versionados() -> List[Path]:
    saida = subprocess.run(["git", "ls-files"], cwd=RAIZ, capture_output=True,
                           text=True, check=True).stdout.splitlines()
    res = []
    for rel in saida:
        rel = rel.strip()
        if not rel or any(rel.startswith(p) for p in _PREFIXOS_IGNORADOS):
            continue
        if Path(rel).suffix.lower() in _EXT_IGNORADAS:
            continue
        res.append(RAIZ / rel)
    return res


def extrair_dois() -> Dict[str, List[Tuple[str, int, str]]]:
    """DOI (minusculo) -> [(arquivo, linha, texto da linha)]."""
    achados: Dict[str, List[Tuple[str, int, str]]] = {}
    for arq in arquivos_versionados():
        try:
            linhas = arq.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for i, linha in enumerate(linhas, 1):
            for m in _RX_DOI.finditer(linha):
                d = _limpar_doi(m.group(0))
                if len(d) < 8 or re.fullmatch(r"10\.6084/m9\.[a-z]+", d.lower()):
                    continue         # prefixo cortado / placeholder
                rel = str(arq.relative_to(RAIZ)).replace("\\", "/")
                achados.setdefault(d.lower(), []).append((rel, i, linha.strip()[:220]))
    return achados


# ── Consulta ao registro do DOI ─────────────────────────────────────────────
def _get_json(url: str, tentativas: int = 4) -> Optional[dict]:
    ult = None
    for k in range(tentativas):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA,
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            ult = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            ult = e
        time.sleep(1.5 * (k + 1))
    raise ConnectionError(f"sem resposta de {url}: {ult}")


def _carregar_cache() -> dict:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _salvar_cache(c: dict) -> None:
    try:
        CACHE.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


def consultar(doi: str, cache: dict) -> Optional[dict]:
    """Metadados normalizados {titulo, autores[familia], ano, tipo, agencia}
    ou None se o DOI NAO EXISTE. Levanta ConnectionError se a rede falhar."""
    if doi in cache:
        return cache[doi]
    q = urllib.parse.quote(doi, safe="/()")
    ra = _get_json(f"https://doi.org/doiRA/{q}")
    agencia = ""
    if ra and isinstance(ra, list) and ra and "RA" in ra[0]:
        agencia = ra[0]["RA"]
    meta: Optional[dict] = None
    if agencia == "Crossref":
        d = _get_json(f"https://api.crossref.org/works/{q}")
        if d and "message" in d:
            m = d["message"]
            ano = None
            for k in ("issued", "published-print", "published-online", "created"):
                partes = (m.get(k) or {}).get("date-parts") or [[None]]
                if partes[0] and partes[0][0]:
                    ano = int(partes[0][0])
                    break
            meta = {
                "titulo": " ".join((m.get("title") or [""])[:1]),
                "autores": [a.get("family") or a.get("name") or ""
                            for a in (m.get("author") or m.get("editor") or [])],
                "ano": ano, "tipo": m.get("type", ""),
                "periodico": " ".join((m.get("container-title") or [""])[:1]),
                "agencia": "Crossref",
            }
    elif agencia == "DataCite":
        d = _get_json(f"https://api.datacite.org/dois/{q}")
        if d and "data" in d:
            a = d["data"]["attributes"]
            meta = {
                "titulo": " ".join(t.get("title", "") for t in (a.get("titles") or [])[:1]),
                "autores": [c.get("familyName") or c.get("name") or ""
                            for c in (a.get("creators") or [])],
                "ano": a.get("publicationYear"),
                "tipo": (a.get("types") or {}).get("resourceTypeGeneral", ""),
                "periodico": (a.get("publisher") or ""),
                "agencia": "DataCite",
            }
    cache[doi] = meta
    return meta


def status_http(doi: str, tempo: float = 20.0) -> Optional[int]:
    """Codigo HTTP final de `https://doi.org/<doi>` seguindo redirecionamentos; None se nao houve resposta (rede)."""
    q = urllib.parse.quote(doi, safe="/()")
    req = urllib.request.Request(f"https://doi.org/{q}", method="HEAD", headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=tempo) as r:
            return int(r.status)
    except urllib.error.HTTPError as e:
        return int(e.code)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return None


def avaliar_resolucao(doi: str, esperado: dict, status: Optional[int]) -> Optional[str]:
    """Problema de resolucao do DOI, ou None. Entrada `retirado: true` NAO pode resolver (>= 400); as demais precisam resolver (< 400).
    `None` de `status` (sem resposta) e' erro de rede, tratado a parte pelo chamador."""
    if status is None:
        return f"{doi}: sem resposta de doi.org"
    if esperado.get("retirado"):
        return f"{doi}: marcado como retirado, mas resolve (HTTP {status})" if status < 400 else None
    return f"{doi}: nao resolve em doi.org (HTTP {status})" if status >= 400 else None


def _norm(t: str) -> str:
    t = re.sub(r"\\[~'`^\"=.]", "", t or "")      # acentos LaTeX: Galv\~ao -> Galvao
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


# ── Registro ────────────────────────────────────────────────────────────────
def carregar_registro() -> Tuple[Dict[str, dict], Set[str]]:
    dados = yaml.safe_load(REGISTRO.read_text(encoding="utf-8")) or {}
    refs = {}
    for e in dados.get("referencias", []):
        refs[str(e["doi"]).lower()] = e
    incorretos = {str(d).lower() for d in dados.get("conhecidos_incorretos", [])}
    return refs, incorretos


def comparar(doi: str, esperado: dict, meta: Optional[dict]) -> List[str]:
    if meta is None:
        return [f"{doi}: o DOI NAO EXISTE no registro oficial"]
    erros = []
    if esperado.get("autor1"):
        primeiro = _norm(meta["autores"][0]) if meta["autores"] else ""
        if _norm(esperado["autor1"]) not in primeiro and primeiro not in _norm(esperado["autor1"]):
            erros.append(f"{doi}: 1o autor esperado '{esperado['autor1']}', "
                         f"registro oficial: '{(meta['autores'] or ['?'])[0]}'")
    ano_ref = esperado.get("ano_oficial", esperado.get("ano"))
    if ano_ref and meta.get("ano") and int(ano_ref) != int(meta["ano"]):
        # Crossref pode datar pela publicacao online (ano anterior/seguinte)
        if abs(int(ano_ref) - int(meta["ano"])) > 1:
            erros.append(f"{doi}: ano esperado {ano_ref}, registro oficial: {meta['ano']}")
    if esperado.get("titulo") and _norm(esperado["titulo"]) not in _norm(meta.get("titulo", "")):
        erros.append(f"{doi}: titulo esperado contem '{esperado['titulo']}', "
                     f"registro oficial: '{meta.get('titulo')}'")
    return erros


# ── Entradas .bib: o que o .bib DIZ tem que bater com o registro oficial ─────
_RX_ENTRADA = re.compile(r"@(\w+)\{([^,]+),(.*?)\n\}", re.S)
_RX_CAMPO = re.compile(r"(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?\s*(?=\n\s*\w+\s*=|\Z)", re.S)


def entradas_bib() -> List[dict]:
    res = []
    for arq in arquivos_versionados():
        if arq.suffix.lower() != ".bib":
            continue
        txt = arq.read_text(encoding="utf-8")
        for m in _RX_ENTRADA.finditer(txt + "\n"):
            campos = {k.lower(): re.sub(r"\s+", " ", v.replace("{", "").replace("}", "")).strip()
                      for k, v in _RX_CAMPO.findall(m.group(3) + "\n")}
            campos["_chave"] = m.group(2).strip()
            campos["_arquivo"] = str(arq.relative_to(RAIZ)).replace("\\", "/")
            res.append(campos)
    return res


def verificar_bib(cache: dict) -> List[str]:
    """Para cada entrada do .bib com DOI: 1o autor, ano e inicio do titulo
    tem que bater com o registro oficial do DOI."""
    erros: List[str] = []
    for e in entradas_bib():
        doi = (e.get("doi") or "").lower().strip()
        rot = f"{e['_arquivo']}:{e['_chave']}"
        if not doi:
            continue
        meta = consultar(doi, cache)
        if meta is None:
            erros.append(f"{rot}: DOI {doi} nao existe")
            continue
        primeiro = e.get("author", "").split(" and ")[0]
        familia = primeiro.split(",")[0] if "," in primeiro else primeiro.split()[-1]
        oficial1 = _norm(" ".join(meta["autores"][:1]))
        if _norm(familia) not in oficial1 and oficial1 not in _norm(familia):
            erros.append(f"{rot}: 1o autor '{familia}' != registro oficial "
                         f"'{(meta['autores'] or ['?'])[0]}'")
        if e.get("year") and meta.get("ano") and abs(int(e["year"]) - int(meta["ano"])) > 1:
            erros.append(f"{rot}: ano {e['year']} != registro oficial {meta['ano']}")
        t_bib = _norm(e.get("title", ""))
        t_ofi = _norm(meta.get("titulo", ""))
        inicio = " ".join(t_ofi.split()[:5])
        if inicio and inicio not in t_bib and " ".join(t_bib.split()[:5]) not in t_ofi:
            erros.append(f"{rot}: titulo '{e.get('title')}' != registro oficial "
                         f"'{meta.get('titulo')}'")
    return erros


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sugerir", action="store_true",
                    help="lista DOIs fora do registro, com metadados e contexto")
    ap.add_argument("--rede", action="store_true",
                    help="alem dos metadados, confere que cada DOI RESOLVE em doi.org (HTTP < 400; entradas `retirado` devem falhar); "
                         "etapa com rede, desligada por padrao")
    ap.add_argument("--sem-rede", action="store_true",
                    help="so' checa que todo DOI do repositorio esta no registro")
    args = ap.parse_args(argv)

    try:
        registro, incorretos = carregar_registro()
    except (OSError, yaml.YAMLError, KeyError) as e:
        print(f"ERRO: registro ilegivel ({REGISTRO}): {e}", file=sys.stderr)
        return 2
    achados = extrair_dois()
    erros: List[str] = []

    # 1) cobertura
    fora = sorted(d for d in achados if d not in registro)
    incorretos_no_repo = sorted(d for d in achados if d in incorretos)
    proibidos = []
    for d in incorretos_no_repo:
        for arq, lin, _t in achados[d]:
            if not arq.startswith("docs/") or arq.startswith("docs/MANUAL"):
                proibidos.append(f"{d} em {arq}:{lin}")
    if proibidos:
        erros += [f"DOI CONHECIDO COMO INCORRETO presente: {p}" for p in proibidos]
    for d in fora:
        if d in incorretos:
            continue
        erros.append(f"DOI nao registrado em {REGISTRO.name}: {d}  "
                     f"(ex.: {achados[d][0][0]}:{achados[d][0][1]})")

    if args.sem_rede and not args.sugerir:
        return _fim(erros, len(achados), 0)

    cache = _carregar_cache()
    try:
        if args.sugerir:
            for d in fora:
                meta = consultar(d, cache)
                print(f"\n{d}")
                print(f"  oficial : {json.dumps(meta, ensure_ascii=False) if meta else 'NAO EXISTE'}")
                for arq, lin, txt in achados[d][:2]:
                    print(f"  contexto: {arq}:{lin}: {txt}")
            _salvar_cache(cache)
            return 0 if not fora else 1
        n = 0
        for d, esp in sorted(registro.items()):
            if d not in achados and not esp.get("manter_sem_uso"):
                erros.append(f"{d}: esta no registro mas nao aparece no repositorio "
                             f"(remova, ou marque manter_sem_uso)")
                continue
            meta = consultar(d, cache)
            n += 1
            erros += comparar(d, esp, meta)
        erros += verificar_bib(cache)
        if args.rede:
            for d, esp in sorted(registro.items()):
                st = status_http(d)
                if st is None:
                    raise ConnectionError(f"doi.org nao respondeu para {d}")
                p = avaliar_resolucao(d, esp, st)
                if p:
                    erros.append(p)
        _salvar_cache(cache)
    except ConnectionError as e:
        print(f"ERRO DE REDE (a verificacao NAO foi feita): {e}", file=sys.stderr)
        return 2
    return _fim(erros, len(achados), n)


def _fim(erros: List[str], n_dois: int, n_consultados: int) -> int:
    if erros:
        print(f"FALHOU: {len(erros)} problema(s) em {n_dois} DOI(s) do repositorio:")
        for e in erros:
            print("  -", e)
        return 1
    print(f"OK: {n_dois} DOI(s) no repositorio, todos no registro; "
          f"{n_consultados} conferidos contra Crossref/DataCite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
