#!/usr/bin/env python3
"""gerar_mutacao_linhas_alteradas.py -- docs/qualidade/mutacao_linhas_alteradas.md a partir dos JSON de docs/qualidade/mutacao/.

    python scripts/gerar_mutacao_linhas_alteradas.py            # grava
    python scripts/gerar_mutacao_linhas_alteradas.py --verificar  # sai 1 se o documento gravado difere do gerado

Cada JSON em `docs/qualidade/mutacao/` e' a saida de `scripts/mutacao_modulo.py` sobre UMA faixa de linhas alterada desde a ultima
verificacao completa (nome do arquivo: `modulo__linhas.json`). `pipeline__..._reexecucao.json` e' a nova rodada dos sobreviventes da
primeira rodada da mesma faixa (depois de testes novos): vale a reexecucao. Os modulos que ficaram sem mutacao nesta rodada estao em
`NAO_EXECUTADOS`, com o motivo. Sobrevivente so' fica sem teste se estiver classificado em
`docs/qualidade/anova/mutantes_classificados.json` (chave `modulo:linha:operador`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "docs" / "qualidade" / "mutacao"
CLASSIFICACAO = RAIZ / "docs" / "qualidade" / "anova" / "mutantes_classificados.json"
SAIDA = RAIZ / "docs" / "qualidade" / "mutacao_linhas_alteradas.md"

#: modulo, linhas alteradas, motivo (decisao de escopo: tempo e aquecimento da maquina; entram na v1.1)
NAO_EXECUTADOS: List[Tuple[str, str, str]] = [
    ("hsi_extras", "106-107", "mutação não executada (escopo desta rodada: núcleo estatístico e leitores)"),
    ("auditoria_delineamento", "376-378", "mutação não executada (escopo desta rodada)"),
]
#: faixas alteradas sem nenhum operador mutavel (comentarios, importacoes, texto)
SEM_MUTANTES = [("predicao", "379"), ("linearity", "178"), ("vip_cli", "214"), ("config_io", "263-265"), ("selecao_variaveis", "599"),
                ("dados_io", "1613"), ("pipeline", "1113"), ("pipeline", "2174")]


def _carregar() -> List[Dict]:
    itens = []
    for p in sorted(PASTA.glob("*.json")):
        modulo, _, resto = p.stem.partition("__")
        faixa = resto.replace("_1a_rodada", "").replace("_reexecucao", "")
        d = json.loads(p.read_text(encoding="utf-8"))
        itens.append({"modulo": modulo, "faixa": faixa, "arquivo": p.name, "dados": d,
                      "tipo": "reexecucao" if p.stem.endswith("_reexecucao") else ("primeira" if p.stem.endswith("_1a_rodada") else "unica")})
    return itens


def _linhas() -> List[Tuple[str, str, int, int, List[Dict]]]:
    """(modulo, faixa, mutantes, mortos, sobreviventes finais)."""
    saida = []
    itens = _carregar()
    primeiras = {(i["modulo"], i["faixa"]): i for i in itens if i["tipo"] == "primeira"}
    for i in itens:
        d = i["dados"]
        if i["tipo"] == "primeira":
            continue
        if i["tipo"] == "reexecucao":
            ini = primeiras[(i["modulo"], i["faixa"])]["dados"]
            total = ini["total"]
            surv = d["sobreviventes"]
            saida.append((i["modulo"], i["faixa"], total, total - len(surv), surv))
        else:
            saida.append((i["modulo"], i["faixa"], d["total"], d["total"] - len(d["sobreviventes"]), d["sobreviventes"]))
    return sorted(saida, key=lambda x: (x[0], int(x[1].split("-")[0])))


def gerar() -> str:
    classif = json.loads(CLASSIFICACAO.read_text(encoding="utf-8")) if CLASSIFICACAO.exists() else {}
    linhas = _linhas()
    tot = sum(x[2] for x in linhas)
    mortos = sum(x[3] for x in linhas)
    L = ["# Mutação nas linhas alteradas desde a última verificação completa", "",
         "**Público:** quem avalia os números · **Estado:** gerado por `scripts/gerar_mutacao_linhas_alteradas.py` (não editar à mão) · "
         "**Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`", "",
         "Teste de mutação (`scripts/mutacao_modulo.py`): o programa troca um operador do código por outro e roda os testes; um teste que "
         "falha *mata* o mutante, e o que sobrevive mostra um comportamento que nenhum teste confere. Aqui a mutação foi aplicada só às "
         "**linhas alteradas** desde a última verificação completa, módulo por módulo, com os testes de cada módulo. O ANOVA, a CLI do ANOVA e o "
         "VIP têm a rodada completa em `docs/qualidade/anova_blindagem.md`.", "",
         "## 1. Resultado por faixa de linhas", "",
         "| módulo | linhas | mutantes | mortos | sobreviventes | sobreviventes classificados |", "|---|---|---|---|---|---|"]
    abertos = 0
    for modulo, faixa, total, morto, surv in linhas:
        cls = 0
        for s in surv:
            if f"{modulo}:{s['linha']}:{s['operador']}" in classif:
                cls += 1
        abertos += len(surv) - cls
        L.append(f"| `{modulo}.py` | {faixa} | {total} | {morto} | {len(surv)} | {cls} |")
    L += ["", f"**{mortos} de {tot} mutantes mortos; {tot - mortos} sobrevivente(s); {abertos} sem classificação.**", ""]
    surv_todos = [(m, f, s) for m, f, _, _, ss in linhas for s in ss]
    if surv_todos:
        L += ["## 2. Sobreviventes e a classificação de cada um", "", "| módulo | linha | operador | trecho | classificação |", "|---|---|---|---|---|"]
        for m, _f, s in surv_todos:
            chave = f"{m}:{s['linha']}:{s['operador']}"
            L.append(f"| `{m}.py` | {s['linha']} | {s['operador']} | `{s['trecho'].strip()[:90]}` | {classif.get(chave, 'SEM CLASSIFICAÇÃO')} |")
        L.append("")
    L += ["## 3. Faixas alteradas sem operador mutável", "",
          "Comentários, importações ou texto: não há o que mutar.", "", "| módulo | linhas |", "|---|---|"]
    L += [f"| `{m}.py` | {f} |" for m, f in SEM_MUTANTES]
    L += ["", "## 4. Não executado nesta rodada", "", "Fica registrado como **não executado** (não como verde); entra na v1.1.", "",
          "| módulo | linhas | situação |", "|---|---|---|"]
    L += [f"| `{m}.py` | {f} | {motivo} |" for m, f, motivo in NAO_EXECUTADOS]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verificar", action="store_true")
    a = ap.parse_args(argv)
    novo = gerar()
    if a.verificar:
        atual = SAIDA.read_bytes().decode("utf-8").replace("\r\n", "\n") if SAIDA.exists() else ""
        if atual != novo:
            print("mutacao_linhas_alteradas.md desatualizado: rode scripts/gerar_mutacao_linhas_alteradas.py")
            return 1
        return 0
    SAIDA.write_text(novo, encoding="utf-8", newline="\n")
    print("gravado:", SAIDA.relative_to(RAIZ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
