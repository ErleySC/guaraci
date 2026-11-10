#!/usr/bin/env python3
"""gerar_cobertura_lacunas.py -- docs/qualidade/cobertura_lacunas.md a partir do coverage.xml da suite e do registro de excecoes.

    python scripts/gerar_cobertura_lacunas.py --cobertura coverage.xml            # grava docs/qualidade/cobertura_lacunas.md
    python scripts/gerar_cobertura_lacunas.py --cobertura coverage.xml --verificar  # sai 1 se o documento gravado difere do gerado

O documento declara: a cobertura medida (global e por camada: nucleo cientifico, web, CLI e menus, resto) contra a medicao inicial; as
categorias das lacunas (A coberta por teste novo, B codigo inalcancavel removido, C plataforma ou dependencia opcional, D interface web);
cada linha ou ramo que ainda falta, com a categoria e o motivo registrados em `tests/cobertura_excecoes.json` (chave `lacunas_aceitas`) ou a
marca SEM CLASSIFICACAO; e as exclusoes de cobertura. Nada de caminho de maquina, nome de pessoa ou dado de terceiros.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import congelar_nucleo as cn  # noqa: E402
import cobertura_excecoes as ce  # noqa: E402

SAIDA = RAIZ / "docs" / "qualidade" / "cobertura_lacunas.md"
REGISTRO = RAIZ / "tests" / "cobertura_excecoes.json"
CAMADAS = ("nucleo", "web", "cli", "resto")
ROTULOS = {"nucleo": "Núcleo científico", "web": "Web (app_tabs)", "cli": "CLI e menus", "resto": "Demais módulos", "global": "Global"}
_CLI = ("guaraci.py", "cli_assistente.py", "cli_comandos.py", "cli_logic.py", "catalogo.py", "catalogo_dados.py", "catalogo_validacao.py",
        "catalogo_ajuda.py", "assistente.py", "confianca.py", "guaraci_theme.py")


def camada(arquivo: str, nucleo: set) -> str:
    if arquivo in nucleo:
        return "nucleo"
    if arquivo.startswith("app_tabs/"):
        return "web"
    if arquivo in _CLI:
        return "cli"
    return "resto"


def ler_xml(caminho: Path) -> Dict[str, Dict[str, object]]:
    """{arquivo relativo a src/guaraci: {linhas, perdidas, ramos, ramos_perdidos, linhas_perdidas, linhas_de_ramo_perdidas}}."""
    out: Dict[str, Dict[str, object]] = {}
    for cls in ET.parse(caminho).getroot().iter("class"):
        fn = (cls.get("filename") or "").replace("\\", "/")
        rel = fn.split("src/guaraci/")[-1] if "src/guaraci/" in fn else fn.split("guaraci/", 1)[-1]
        reg = out.setdefault(rel, {"linhas": 0, "cobertas": 0, "ramos": 0, "ramos_cobertos": 0, "linhas_perdidas": [], "linhas_de_ramo_perdidas": []})
        for ln in cls.iter("line"):
            n = int(ln.get("number", "0"))
            reg["linhas"] += 1                                                   # type: ignore[operator]
            if int(ln.get("hits", "0")) > 0:
                reg["cobertas"] += 1                                             # type: ignore[operator]
            else:
                reg["linhas_perdidas"].append(n)                                 # type: ignore[union-attr]
            m = re.search(r"\((\d+)/(\d+)\)", ln.get("condition-coverage") or "")
            if m:
                reg["ramos"] += int(m.group(2))                                  # type: ignore[operator]
                reg["ramos_cobertos"] += int(m.group(1))                         # type: ignore[operator]
                if int(m.group(1)) < int(m.group(2)):
                    reg["linhas_de_ramo_perdidas"].append(n)                     # type: ignore[union-attr]
    return out


def _pct(a: int, b: int) -> float:
    return round(100.0 * a / b, 2) if b else 100.0


def agregar(dados: Dict[str, Dict[str, object]], nucleo: set) -> Dict[str, Dict[str, float]]:
    acum: Dict[str, List[int]] = {k: [0, 0, 0, 0] for k in (*CAMADAS, "global")}
    for arq, v in dados.items():
        for alvo in (camada(arq, nucleo), "global"):
            a = acum[alvo]
            a[0] += int(v["linhas"])                                             # type: ignore[call-overload]
            a[1] += int(v["cobertas"])                                           # type: ignore[call-overload]
            a[2] += int(v["ramos"])                                              # type: ignore[call-overload]
            a[3] += int(v["ramos_cobertos"])                                     # type: ignore[call-overload]
    return {k: {"linhas": _pct(a[1], a[0]), "ramos": _pct(a[3], a[2]), "combinada": _pct(a[1] + a[3], a[0] + a[2])} for k, a in acum.items()}


def _faixas(numeros: List[int]) -> str:
    if not numeros:
        return "—"
    numeros = sorted(set(numeros))
    partes, ini, ant = [], numeros[0], numeros[0]
    for n in numeros[1:]:
        if n == ant + 1:
            ant = n
            continue
        partes.append(f"{ini}" if ini == ant else f"{ini}–{ant}")
        ini = ant = n
    partes.append(f"{ini}" if ini == ant else f"{ini}–{ant}")
    return ", ".join(partes)


def lacunas(dados: Dict[str, Dict[str, object]], nucleo: set) -> List[Tuple[str, str, List[int], List[int]]]:
    return [(arq, camada(arq, nucleo), list(v["linhas_perdidas"]), list(v["linhas_de_ramo_perdidas"]))             # type: ignore[call-overload]
            for arq, v in sorted(dados.items()) if v["linhas_perdidas"] or v["linhas_de_ramo_perdidas"]]


def _assuntos_de_remocao(raiz: Path) -> List[str]:
    """Assuntos dos commits que removeram codigo inalcancavel (categoria B), sem hash: os hashes mudam na reconstrucao."""
    r = subprocess.run(["git", "log", "--format=%s", "-n", "400"], cwd=raiz, capture_output=True, text=True, encoding="utf-8", errors="replace")
    vistos: List[str] = []
    for s in r.stdout.splitlines():
        # so' os commits do fechamento da cobertura (um assunto de outro tema pode citar caminho de documento que nao existe mais)
        if s.startswith("test(cobertura)") and re.search(r"remove[^\n]*(inalcan|guarda|ramo|pragma|laco)", s, re.I) and s not in vistos:
            vistos.append(s)
    return vistos


def gerar(xml: Path, registro: Optional[Dict] = None, raiz: Path = RAIZ) -> str:
    registro = registro if registro is not None else json.loads(REGISTRO.read_text(encoding="utf-8"))
    nucleo = {a.replace("src/guaraci/", "") for a in cn.arquivos_do_nucleo()}
    dados = ler_xml(xml)
    depois = agregar(dados, nucleo)
    antes = registro.get("medicao_inicial", {})
    L = ["# Cobertura e lacunas", "",
         "**Público:** quem mantém a suíte · **Estado:** gerado por `scripts/gerar_cobertura_lacunas.py` (não editar à mão) · "
         "**Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`", "",
         "Medida pela suíte completa (`tests/`, ramos), a mesma da verificação final, em "
         f"{ce.frase_de_escopo(ce.linhas_excluidas())}. 100% de cobertura **não prova correção**: a prova de "
         "qualidade dos testes do núcleo é a mutação (`scripts/mutacao_modulo.py`).", "",
         "## 1. Medição", "", "| Camada | Linhas (antes → depois) | Ramos (antes → depois) | Combinada (antes → depois) |", "|---|---|---|---|"]
    for k in ("global", *CAMADAS):
        a, d = antes.get(k, {}), depois[k]
        L.append(f"| {ROTULOS[k]} | {a.get('linhas', '—')}% → {d['linhas']}% | {a.get('ramos', '—')}% → {d['ramos']}% | "
                 f"{a.get('combinada', '—')}% → {d['combinada']}% |")
    L += ["", "Alvos: núcleo 100% de linhas e de ramos; global ≥ 95% (combinada).", ""]
    L += ["## 2. Categorias", "",
          "- **A — testável barato → teste escrito.** Arquivos `tests/test_lacunas_*.py` (núcleo, CLI, web, figuras, relatórios): "
          f"{len(list((raiz / 'tests').glob('test_lacunas_*.py')))} arquivo(s).",
          "- **B — defensivo inalcançável → código removido** (prova: o teste da verificação anterior recusa a entrada; nunca `pragma`):", ""]
    L += [f"  - {s}" for s in _assuntos_de_remocao(raiz)] or ["  - (nenhuma remoção no histórico recente)"]
    L += ["", "- **C — plataforma ou dependência opcional → injeção do ambiente em teste;** sem como injetar, `# pragma: no cover` com motivo "
          "(seção 4).",
          f"- **D — interface web → `streamlit.testing.v1.AppTest`** sem navegador: {len(list((raiz / 'tests').glob('test_lacunas_web_*.py')))} "
          "arquivo(s) novo(s) mais os existentes. O que o AppTest não alcança fica declarado em `docs/cientifico/limitacoes.md`.", ""]
    L += ["## 3. O que ainda falta", ""]
    aceitas = {(e["arquivo"], e.get("funcao", "")): e for e in registro.get("lacunas_aceitas", [])}
    faltas = lacunas(dados, nucleo)
    if not faltas:
        L.append("Nada: todas as linhas e todos os ramos medidos estão cobertos.")
    else:
        L += ["| Arquivo | Camada | Linhas sem execução | Linhas com ramo parcial | Categoria · motivo |", "|---|---|---|---|---|"]
        for arq, cam, perdidas, ramos in faltas:
            reg = next((e for (a, _f), e in aceitas.items() if a == arq), None)
            motivo = f"{reg['categoria']} · {reg['motivo']}" if reg else "**SEM CLASSIFICAÇÃO**"
            L.append(f"| `{arq}` | {ROTULOS[cam]} | {_faixas(perdidas)} | {_faixas(ramos)} | {motivo} |")
    L += ["", "## 4. Exclusões registradas", "", "Cada linha que o coverage exclui (`# pragma: no cover`, `if TYPE_CHECKING:`, stub `...`) e cada "
          "`omit`/`exclude_lines` está em `tests/cobertura_excecoes.json`; "
          "`tests/test_cobertura_excecoes.py` falha se o número crescer sem entrada.", ""]
    pragmas = registro.get("pragmas", [])
    if pragmas:
        L += ["| Arquivo:função | Quantidade | Categoria | Motivo |", "|---|---:|---|---|"]
        L += [f"| `{e['arquivo']}:{e['funcao']}` | {e.get('quantidade', 1)} | {e['categoria']} | {e['motivo']} |" for e in pragmas]
    else:
        L.append("Nenhuma exclusão.")
    L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cobertura", required=True, help="coverage.xml da suite completa")
    ap.add_argument("--verificar", action="store_true")
    a = ap.parse_args(argv)
    texto = gerar(Path(a.cobertura))
    atual = SAIDA.read_bytes().decode("utf-8").replace("\r\n", "\n") if SAIDA.exists() else ""
    if a.verificar:
        if atual != texto:
            print("desatualizado: docs/qualidade/cobertura_lacunas.md")
            return 1
        return 0
    crlf = SAIDA.exists() and b"\r\n" in SAIDA.read_bytes()
    SAIDA.write_bytes((texto.replace("\n", "\r\n") if crlf else texto).encode("utf-8"))
    print("gravado: docs/qualidade/cobertura_lacunas.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
