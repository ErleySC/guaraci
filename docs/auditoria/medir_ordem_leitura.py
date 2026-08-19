# -*- coding: utf-8 -*-
"""Mede a ORDEM REAL DE LEITURA dos espectros e testa se ela esta
correlacionada com o teor de adulteracao (Passe 1, sec. 3.2 do
PROMPT_AUDITORIA_MESTRE).

POR QUE IMPORTA: se as concentracoes foram lidas em ordem crescente dentro
de uma sessao, deriva instrumental (aquecimento da fonte, deriva do
detector) e evaporacao/oxidacao da amostra ficam CONFUNDIDAS com a
variavel de interesse. O PLS-R estaria modelando a deriva, nao o teor --
e o R2cv continuaria alto, porque a deriva e' reprodutivel dentro da
sessao. Auditorias anteriores usaram a data do ##TITLE (data da AMOSTRA);
esta usa o ##AUDIT TRAIL (data/hora da LEITURA), que e' outro campo.

Uso:
    python docs/auditoria/medir_ordem_leitura.py <raiz_dos_dx>

Nao escreve nada em disco e nao imprime identificador de amostra --
so' agregados por especie/adulterante.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from guaraci.dados_io import extrair_title_do_dx, parse_title  # noqa: E402

_RE_AUDIT = re.compile(
    r"\(\s*1\s*,\s*<(?P<ts>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})[^>]*>\s*,"
    r"\s*<(?P<quem>[^>]*)>\s*,\s*<(?P<onde>[^>]*)>")


def ler_audit(caminho):
    """Devolve (datetime, operador, local) da 1a entrada do AUDIT TRAIL."""
    with open(caminho, "r", encoding="latin-1", errors="replace") as f:
        for linha in f:
            m = _RE_AUDIT.search(linha)
            if m:
                return (datetime.strptime(m.group("ts"), "%Y/%m/%d %H:%M:%S"),
                        m.group("quem").strip(), m.group("onde").strip())
            if linha.startswith("##XYDATA") or linha.startswith("##XYPOINTS"):
                break
    return None, None, None


def main(raiz):
    arquivos = sorted(Path(raiz).rglob("*.dx"))
    print("arquivos .dx encontrados: %d" % len(arquivos))

    regs = []
    sem_title = sem_audit = sem_parse = 0
    operadores = defaultdict(int)
    locais = defaultdict(int)
    for p in arquivos:
        titulo = extrair_title_do_dx(str(p))
        if not titulo:
            sem_title += 1
            continue
        info = parse_title(titulo)
        if not info:
            sem_parse += 1
            continue
        ts, quem, onde = ler_audit(p)
        if ts is None:
            sem_audit += 1
            continue
        operadores[quem] += 1
        locais[onde] += 1
        regs.append({
            "ts": ts, "especie": info["especie"], "cod": info["cod"],
            "adulterante": info["adulterante_nome"],
            "teor": 0.0 if info["puro"] else float(info["teor"]),
            "puro": info["puro"], "mae_id": info["mae_id"],
            "data_titulo": info["data"], "trip": info["triplicata"],
        })

    print("  sem TITLE: %d | TITLE nao parseavel: %d | sem AUDIT: %d | "
          "usados: %d" % (sem_title, sem_parse, sem_audit, len(regs)))
    print("\noperadores distintos no AUDIT TRAIL: %d (contagens: %s)"
          % (len(operadores), sorted(operadores.values(), reverse=True)))
    print("locais distintos no AUDIT TRAIL: %d -> %s"
          % (len(locais), sorted(locais)))
    if not regs:
        return 1

    # ---------------------------------------------------------------- 1
    dias = defaultdict(int)
    for r in regs:
        dias[r["ts"].date()] += 1
    print("\n== 1. SESSOES DE LEITURA (dias distintos): %d ==" % len(dias))
    print("   janela: %s .. %s" % (min(dias), max(dias)))
    for d in sorted(dias):
        print("   %s: %4d espectros" % (d, dias[d]))

    # ---------------------------------------------------------------- 2
    print("\n== 2. PURO E ADULTERADO COMPARTILHAM A SESSAO? "
          "(por especie x dia de LEITURA) ==")
    print("%-20s %5s %13s %14s %14s"
          % ("especie", "dias", "dias c/ puro", "dias c/ adult",
             "dias c/ AMBOS"))
    for esp in sorted({r["especie"] for r in regs}):
        sub = [r for r in regs if r["especie"] == esp]
        dd = defaultdict(lambda: [0, 0])
        for r in sub:
            dd[r["ts"].date()][0 if r["puro"] else 1] += 1
        ambos = sum(1 for v in dd.values() if v[0] and v[1])
        cp = sum(1 for v in dd.values() if v[0])
        ca = sum(1 for v in dd.values() if v[1])
        print("%-20s %5d %13d %14d %14d" % (esp, len(dd), cp, ca, ambos))

    # ---------------------------------------------------------------- 3
    print("\n== 3. ORDEM DE LEITURA x TEOR (Spearman, dentro de cada "
          "especie x adulterante x sessao) ==")
    print("   H0: a ordem em que as amostras foram lidas nao tem relacao "
          "com o teor.")
    print("%-20s %-10s %-12s %4s %7s %9s"
          % ("especie", "adult", "dia", "n", "rho", "p"))
    blocos = defaultdict(list)
    for r in regs:
        if r["adulterante"]:
            blocos[(r["especie"], r["adulterante"], r["ts"].date())].append(r)

    rhos = []
    n_sig = 0
    for chave in sorted(blocos):
        b = blocos[chave]
        # colapsa replicas: uma leitura por mae_id (primeira do grupo)
        por_mae = {}
        for r in sorted(b, key=lambda x: x["ts"]):
            por_mae.setdefault(r["mae_id"], r)
        pts = sorted(por_mae.values(), key=lambda x: x["ts"])
        if len(pts) < 5:
            continue
        ordem = np.arange(len(pts), dtype=float)
        teor = np.array([p["teor"] for p in pts], dtype=float)
        if np.std(teor) < 1e-12:
            continue
        rho, pv = stats.spearmanr(ordem, teor)
        rhos.append(rho)
        if pv < 0.05:
            n_sig += 1
        marca = " <<<" if pv < 0.05 else ""
        print("%-20s %-10s %-12s %4d %7.3f %9.4f%s"
              % (chave[0], chave[1], str(chave[2]), len(pts), rho, pv, marca))

    if rhos:
        rhos_a = np.array(rhos)
        print("\n   blocos avaliados: %d | significativos a 5%%: %d "
              "(%.1f%%; esperado sob H0: 5%%)"
              % (len(rhos), n_sig, 100.0 * n_sig / len(rhos)))
        print("   rho medio: %+.3f | mediana: %+.3f | |rho| medio: %.3f"
              % (np.mean(rhos_a), np.median(rhos_a), np.mean(np.abs(rhos_a))))
        pos = int(np.sum(rhos_a > 0))
        bino = stats.binomtest(pos, len(rhos_a), 0.5)
        print("   rho > 0 em %d/%d blocos (teste binomial p=%.4g)"
              % (pos, len(rhos_a), bino.pvalue))

    # ---------------------------------------------------------------- 4
    print("\n== 4. DERIVA: releituras do MESMO mae_id em momentos "
          "diferentes ==")
    span = defaultdict(list)
    for r in regs:
        span[r["mae_id"]].append(r["ts"])
    multi_dia = sum(1 for v in span.values()
                    if len({t.date() for t in v}) > 1)
    dts = [max(v) - min(v) for v in span.values() if len(v) > 1]
    print("   mae_ids: %d | lidos em >1 dia: %d" % (len(span), multi_dia))
    if dts:
        segs = np.array([d.total_seconds() for d in dts])
        print("   intervalo dentro do grupo de replicas (s): mediana %.0f | "
              "p95 %.0f | max %.0f"
              % (np.median(segs), np.percentile(segs, 95), segs.max()))

    # ---------------------------------------------------------------- 5
    print("\n== 5. TEOR x HORA DO DIA (deriva intra-sessao) ==")
    ad = [r for r in regs if r["adulterante"]]
    if ad:
        h = np.array([r["ts"].hour + r["ts"].minute / 60.0 for r in ad])
        t = np.array([r["teor"] for r in ad])
        rho, pv = stats.spearmanr(h, t)
        print("   n=%d rho(hora, teor) = %+.3f (p=%.3g)" % (len(ad), rho, pv))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
