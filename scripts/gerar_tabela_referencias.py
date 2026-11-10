#!/usr/bin/env python3
"""gerar_tabela_referencias.py -- a tabela unica de referencias: fonte | DOI | conferido | onde e' citada.

    python scripts/gerar_tabela_referencias.py              # grava docs/cientifico/referencias.md
    python scripts/gerar_tabela_referencias.py --verificar  # sai 1 se estiver desatualizada ou houver referencia nao conferida

A fonte e' `docs/cientifico/referencias_verificadas.yaml` (cada entrada traz `conferido`: data e registro oficial contra o qual foi lida);
"onde e' citada" vem de uma varredura dos arquivos versionados por DOI. Referencia sem `conferido` nao entra na tabela e reprova a
verificacao (regra: nao conferida, removida).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
SAIDA = RAIZ / "docs" / "cientifico" / "referencias.md"
REGISTRO = RAIZ / "docs" / "cientifico" / "referencias_verificadas.yaml"
MAX_ONDE = 4


def _onde(dois: Dict[str, List[Tuple[str, int, str]]], doi: str) -> str:
    arqs = sorted({a for a, _n, _l in dois.get(doi.lower(), []) if not a.endswith("referencias_verificadas.yaml")
                   and a != SAIDA.relative_to(RAIZ).as_posix()})
    if not arqs:
        return "-"
    txt = ", ".join(f"`{a}`" for a in arqs[:MAX_ONDE])
    return txt + (f" e mais {len(arqs) - MAX_ONDE}" if len(arqs) > MAX_ONDE else "")


def carregar() -> Tuple[List[dict], List[dict]]:
    refs = (yaml.safe_load(REGISTRO.read_text(encoding="utf-8")) or {}).get("referencias", [])
    return [r for r in refs if r.get("conferido")], [r for r in refs if not r.get("conferido")]


def gerar(onde: Dict[str, List[Tuple[str, int, str]]]) -> Tuple[str, List[dict]]:
    todas, nao = carregar()
    ok = [r for r in todas if not r.get("retirado")]
    retiradas = [r for r in todas if r.get("retirado")]
    L = ["# Referências", "",
         "**Público:** quem confere as fontes · **Estado:** gerado por `scripts/gerar_tabela_referencias.py` a partir de "
         "`docs/cientifico/referencias_verificadas.yaml` (não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`", "",
         f"{len(ok)} referências, todas conferidas contra o registro oficial do DOI (Crossref ou DataCite). Uma referência não conferida é removida, "
         "não listada. Para conferir de novo: `python scripts/verificar_citacoes.py`.", "",
         "| fonte | DOI | conferido | onde é citada |", "|---|---|---|---|"]
    for r in sorted(ok, key=lambda x: (str(x.get("autor1", "")).lower(), x.get("ano", 0))):
        fonte = f"{r.get('autor1', '?')} ({r.get('ano', '?')}), {r.get('titulo', '')}".replace("|", "/")
        conf = str(r["conferido"]).split(" (")[0]
        L.append(f"| {fonte} | `{r['doi']}` | {conf} | {_onde(onde, r['doi'])} |")
    if retiradas:
        L += ["", "## Registros retirados", "",
              "O registro de DOI do depósito do próprio software foi retirado pelo repositório de depósito e não resolve mais "
              "(resposta HTTP 410). Ele fica listado só como lápide: não é referência do software e não deve ser citado.", "",
              "| registro | DOI | situação |", "|---|---|---|"]
        for r in retiradas:
            L.append(f"| depósito do próprio software | `{r['doi']}` | retirado; metadados ainda constam no registro de DOI, a resolução falha |")
    return "\n".join(L) + "\n", nao


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verificar", action="store_true")
    a = ap.parse_args(argv)
    import verificar_citacoes as vc
    texto, nao = gerar(vc.extrair_dois())
    for r in nao:
        print(f"nao conferida: {r.get('doi')}")
    atual = SAIDA.read_bytes().decode("utf-8").replace("\r\n", "\n") if SAIDA.exists() else ""
    if a.verificar:
        if atual != texto:
            print(f"desatualizada: {SAIDA.relative_to(RAIZ).as_posix()}")
        return 1 if (nao or atual != texto) else 0
    crlf = SAIDA.exists() and b"\r\n" in SAIDA.read_bytes()
    SAIDA.write_bytes((texto.replace("\n", "\r\n") if crlf else texto).encode("utf-8"))
    print(f"gravado: {SAIDA.relative_to(RAIZ).as_posix()}")
    return 1 if nao else 0


if __name__ == "__main__":
    sys.exit(main())
