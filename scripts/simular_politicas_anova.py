#!/usr/bin/env python3
"""simular_politicas_anova.py -- qual politica de comparacao de grupos o `guaraci anova` deve usar por padrao? (medida, nao opiniao)

    python scripts/simular_politicas_anova.py [--repeticoes N] [--processos P] [--saida-json ARQ] [--saida-md ARQ]

Politicas comparadas (todas sobre a unidade independente, k grupos, nivel 0,05):
  tukey          ANOVA classica + Tukey/Tukey-Kramer fixo
  games-howell   Welch + Games-Howell fixo
  auto           o modo atual: o Levene decide Tukey x Games-Howell (procedimento em duas etapas)
  recomendar     o diagnostico de premissas escolhe o metodo (Tukey, Games-Howell ou Kruskal-Wallis/Dunn)
  kruskal-dunn   Kruskal-Wallis + Dunn (Holm) fixo

Metricas por cenario: erro por familia (FWER: pelo menos um par significativo com medias iguais), poder (par com diferenca real) e
discordancia (fracao dos conjuntos de dados em que o conjunto de pares significativos difere do Games-Howell).

Desenho para nao julgar a regra pela simulacao que a calibrou:
  CALIBRACAO  grade A (cenarios e sementes de um lado) -- escolhe-se a politica;
  CONFIRMACAO grade B (outros n, razoes de variancia, k e distribuicoes; outras sementes) -- mede-se a politica escolhida.
Criterio: erro por familia <= 0,06 em TODOS os cenarios da calibracao (com a margem de Monte Carlo declarada), e entre as que cumprem, a
de maior poder medio. Nenhum limiar e' ajustado aqui; a simulacao so' ESCOLHE entre as politicas existentes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import zlib
from pathlib import Path

import numpy as np

import guaraci.anova_grupos as _ag
from guaraci.anova_grupos import anova_uma_via, diagnosticar_premissas, kruskal_dunn, recomendar_metodo

ALFA = 0.05
LIMITE_FWER = 0.06
POLITICAS = ("tukey", "games-howell", "auto", "recomendar", "kruskal-dunn")
EFEITO = 2.0       # diferenca do ultimo grupo, em desvios-padrao medios


def _gerar_dist(rng, dist: str, n: int, sd: float) -> np.ndarray:
    if dist == "normal":
        z = rng.normal(0.0, 1.0, n)
    elif dist == "lognormal":                       # assimetrica, media 0, variancia 1
        z = (rng.lognormal(0.0, 1.0, n) - np.exp(0.5)) / np.sqrt((np.exp(1) - 1) * np.exp(1))
    elif dist == "t3":                              # caudas pesadas, variancia 1
        z = rng.standard_t(3, n) / np.sqrt(3.0)
    elif dist == "contaminada":                     # 90% N(0,1) + 10% N(0,5^2), reescalada para variancia 1
        base = rng.normal(0.0, 1.0, n)
        mancha = rng.random(n) < 0.10
        base[mancha] = rng.normal(0.0, 5.0, int(mancha.sum()))
        z = base / np.sqrt(0.9 + 0.1 * 25.0)
    else:
        raise ValueError(dist)
    return sd * z


def _tamanhos(k: int, n: int, aloc: str) -> list:
    if aloc == "igual":
        return [n] * k
    ns = [max(2, int(round(n * f))) for f in np.linspace(0.5, 1.5, k)]      # 0,5n ... 1,5n
    return ns


def _cenario(dist, razao_var, n, k, aloc, gl_extra=None):
    nome = f"{dist}|var1:{razao_var:g}|n={n}|k={k}|{aloc}"
    return dict(nome=nome, dist=dist, razao_var=razao_var, n=n, k=k, aloc=aloc)


def grade_calibracao() -> list:
    out = []
    for dist, n in itertools.product(("normal", "lognormal", "t3"), (3, 6, 10, 30)):
        out.append(_cenario(dist, 1.0, n, 3, "igual"))
        out.append(_cenario(dist, 1.0, n, 3, "n_desigual"))
        for rv in (4.0, 10.0):
            out.append(_cenario(dist, rv, n, 3, "igual"))
            out.append(_cenario(dist, rv, n, 3, "var_alta_n_pequeno"))
            out.append(_cenario(dist, rv, n, 3, "var_alta_n_grande"))
    for k in (5, 10):
        for rv in (1.0, 10.0):
            out.append(_cenario("normal", rv, 6, k, "igual"))
            out.append(_cenario("normal", rv, 6, k, "var_alta_n_pequeno" if rv > 1 else "n_desigual"))
    return out


def grade_confirmacao() -> list:
    out = []
    for dist, n in itertools.product(("normal", "lognormal", "t3", "contaminada"), (4, 8, 20)):
        out.append(_cenario(dist, 1.0, n, 4, "igual"))
        out.append(_cenario(dist, 1.0, n, 4, "n_desigual"))
        for rv in (2.5, 6.0, 16.0):
            out.append(_cenario(dist, rv, n, 4, "igual"))
            out.append(_cenario(dist, rv, n, 4, "var_alta_n_pequeno"))
            out.append(_cenario(dist, rv, n, 4, "var_alta_n_grande"))
    for rv in (1.0, 6.0):
        out.append(_cenario("normal", rv, 5, 6, "igual"))
        out.append(_cenario("normal", rv, 5, 6, "var_alta_n_pequeno" if rv > 1 else "n_desigual"))
    return out


def _amostra(sc, rng, com_efeito):
    k, n, aloc, rv = sc["k"], sc["n"], sc["aloc"], sc["razao_var"]
    ns = [n] * k if aloc == "igual" else _tamanhos(k, n, aloc)
    base = list(np.linspace(1.0, np.sqrt(rv), k)) if rv != 1.0 else [1.0] * k
    sds = base[::-1] if aloc == "var_alta_n_pequeno" else base      # n cresce com o indice: a maior variancia cai no menor (ou maior) grupo
    sd_medio = float(np.sqrt(np.mean(np.square(sds))))
    ys, gs = [], []
    for i in range(k):
        v = _gerar_dist(rng, sc["dist"], ns[i], sds[i])
        if com_efeito and i == k - 1:
            v = v + EFEITO * sd_medio
        ys.append(v)
        gs += [f"g{i + 1}"] * ns[i]
    y = np.concatenate(ys)
    return y, np.array(gs), np.arange(y.size)


def _sem_ic_caro():
    """So' o p ajustado importa: o valor critico do IC simultaneo (~0,3 s por par) e o bootstrap do tamanho de efeito nao mudam p."""
    _ag._q_critico = lambda *a, **kw: 0.0


def _pares(comps):
    return frozenset((c["a"], c["b"]) for c in comps if c["p_ajustado"] < ALFA)


def _rodar(metodo, y, g, u):
    if metodo == "tukey":
        return _pares(anova_uma_via(y, g, u, variancias="iguais", n_boot=5).comparacoes)
    if metodo == "games-howell":
        return _pares(anova_uma_via(y, g, u, variancias="desiguais", n_boot=5).comparacoes)
    if metodo == "auto":
        return _pares(anova_uma_via(y, g, u, variancias="auto", n_boot=5).comparacoes)
    return _pares(kruskal_dunn(y, g, u).comparacoes)


def _politicas_sobre(y, g, u):
    """Conjunto de pares significativos de cada politica (calcula cada metodo uma vez)."""
    res = {m: _rodar(m, y, g, u) for m in ("tukey", "games-howell", "auto", "kruskal-dunn")}
    rec = recomendar_metodo(diagnosticar_premissas(y, g, u, alpha=ALFA))["metodo"]
    res["recomendar"] = res[rec]
    return res


def simular_cenario(args) -> dict:
    sc, repeticoes, semente0 = args
    _sem_ic_caro()
    fwer = {p: 0 for p in POLITICAS}
    poder = {p: 0 for p in POLITICAS}
    discord = {p: 0 for p in POLITICAS}
    ultimo = f"g{sc['k']}"
    alvo = ("g1", ultimo)
    for r in range(repeticoes):
        rng0 = np.random.default_rng([semente0, 0, r, zlib.crc32(sc["nome"].encode()) & 0xFFFF])
        y, g, u = _amostra(sc, rng0, False)
        c0 = _politicas_sobre(y, g, u)
        for p in POLITICAS:
            fwer[p] += bool(c0[p])
            discord[p] += c0[p] != c0["games-howell"]
        rng1 = np.random.default_rng([semente0, 1, r, zlib.crc32(sc["nome"].encode()) & 0xFFFF])
        y1, g1, u1 = _amostra(sc, rng1, True)
        c1 = _politicas_sobre(y1, g1, u1)
        for p in POLITICAS:
            poder[p] += (alvo in c1[p]) or (alvo[::-1] in c1[p])
    return dict(cenario=sc["nome"], k=sc["k"], dist=sc["dist"], razao_var=sc["razao_var"], n=sc["n"], aloc=sc["aloc"],
                fwer={p: fwer[p] / repeticoes for p in POLITICAS}, poder={p: poder[p] / repeticoes for p in POLITICAS},
                discordancia={p: discord[p] / repeticoes for p in POLITICAS})


def rodar_grade(grade, repeticoes, semente0, processos):
    tarefas = [(sc, repeticoes, semente0) for sc in grade]
    if processos > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=processos) as ex:
            return list(ex.map(simular_cenario, tarefas, chunksize=1))
    return [simular_cenario(t) for t in tarefas]


def resumir(linhas: list, repeticoes: int) -> dict:
    se = float(np.sqrt(ALFA * (1 - ALFA) / repeticoes))
    r = {}
    for p in POLITICAS:
        f = np.array([l["fwer"][p] for l in linhas])
        w = np.array([l["poder"][p] for l in linhas])
        r[p] = dict(fwer_max=float(f.max()), fwer_medio=float(f.mean()), n_acima_limite=int((f > LIMITE_FWER).sum()),
                    n_acima_limite_com_margem=int((f > LIMITE_FWER + 2 * se).sum()), poder_medio=float(w.mean()),
                    pior_cenario=linhas[int(f.argmax())]["cenario"])
    return dict(n_cenarios=len(linhas), erro_padrao_mc=se, politicas=r)


def escolher(resumo_cal: dict) -> dict:
    """Entre as politicas com ZERO cenarios acima do limite (com a margem de Monte Carlo), a de maior poder medio; se nenhuma cumpre,
    a de menor numero de violacoes (empate: menor erro maximo) -- e o resultado diz que nenhuma cumpre."""
    pol = resumo_cal["politicas"]
    cumprem = [p for p in POLITICAS if pol[p]["n_acima_limite_com_margem"] == 0]
    if cumprem:
        melhor = max(cumprem, key=lambda p: pol[p]["poder_medio"])
        return dict(politica=melhor, cumpre_o_criterio=True, candidatas=cumprem)
    melhor = min(POLITICAS, key=lambda p: (pol[p]["n_acima_limite_com_margem"], pol[p]["fwer_max"], -pol[p]["poder_medio"]))
    return dict(politica=melhor, cumpre_o_criterio=False, candidatas=[])


def _tabela_resumo(resumo: dict) -> list:
    L = ["| politica | FWER maximo | FWER medio | cenarios acima de 0,06 (com margem de MC) | poder medio | pior cenario |", "|---|---|---|---|---|---|"]
    for p in POLITICAS:
        d = resumo["politicas"][p]
        L.append(f"| {p} | {d['fwer_max']:.3f} | {d['fwer_medio']:.3f} | {d['n_acima_limite']} ({d['n_acima_limite_com_margem']}) | "
                 f"{d['poder_medio']:.3f} | {d['pior_cenario']} |")
    return L


def markdown(res: dict) -> str:
    cal, conf, esc = res["calibracao"], res["confirmacao"], res["escolha"]
    L = ["# Politicas de comparacao de grupos (ANOVA): simulacao com verdade conhecida", "", "**Público:** quem avalia os números · **Estado:** gerado por `scripts/simular_politicas_anova.py` (não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`", "",
         f"{res['repeticoes']} repeticoes por cenario e por condicao; nivel {ALFA}; limite de erro por familia {LIMITE_FWER}; erro padrao de "
         f"Monte Carlo ~{cal['resumo']['erro_padrao_mc']:.3f} perto de 0,05. Efeito de {EFEITO} desvios no ultimo grupo para o poder.", "",
         "## 1. Calibracao (grade A: " + str(cal['resumo']['n_cenarios']) + " cenarios)", ""] + _tabela_resumo(cal["resumo"]) + [
         "", f"**Politica escolhida na calibracao: `{esc['politica']}`** "
         + ("(cumpre o criterio: nenhum cenario acima do limite com a margem de Monte Carlo)." if esc["cumpre_o_criterio"] else
            "(**nenhuma politica cumpre o criterio**; esta e' a de menos violacoes)."), "",
         "## 2. Confirmacao independente (grade B: " + str(conf['resumo']['n_cenarios']) + " cenarios, outros n, razoes, k, distribuicoes e sementes)", ""] + \
        _tabela_resumo(conf["resumo"]) + [""]
    pol = conf["resumo"]["politicas"][esc["politica"]]
    L.append(f"Na confirmacao, `{esc['politica']}` tem FWER maximo {pol['fwer_max']:.3f}, {pol['n_acima_limite']} cenario(s) acima de {LIMITE_FWER} "
             f"({pol['n_acima_limite_com_margem']} com a margem) e poder medio {pol['poder_medio']:.3f}.")
    L += ["", "## 3. Onde cada politica falha (calibracao + confirmacao, FWER > 0,06)", ""]
    todas = cal["linhas"] + conf["linhas"]
    for p in POLITICAS:
        ruins = [l for l in todas if l["fwer"][p] > LIMITE_FWER + 2 * cal["resumo"]["erro_padrao_mc"]]
        L.append(f"- **{p}**: {len(ruins)} de {len(todas)} cenarios acima do limite com margem" +
                 ("; piores: " + "; ".join(f"{l['cenario']} ({l['fwer'][p]:.3f})" for l in sorted(ruins, key=lambda l: -l['fwer'][p])[:4]) if ruins else "."))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeticoes", type=int, default=600)
    ap.add_argument("--processos", type=int, default=1)
    ap.add_argument("--saida-json")
    ap.add_argument("--saida-md")
    ap.add_argument("--rapido", action="store_true", help="so' 6 cenarios de cada grade (teste de fumaca)")
    a = ap.parse_args(argv)
    gA, gB = grade_calibracao(), grade_confirmacao()
    if a.rapido:
        gA, gB = gA[::max(1, len(gA) // 6)][:6], gB[::max(1, len(gB) // 6)][:6]
    linhasA = rodar_grade(gA, a.repeticoes, 1000, a.processos)
    linhasB = rodar_grade(gB, a.repeticoes, 777000, a.processos)
    resA, resB = resumir(linhasA, a.repeticoes), resumir(linhasB, a.repeticoes)
    res = dict(repeticoes=a.repeticoes, alfa=ALFA, limite_fwer=LIMITE_FWER, efeito=EFEITO, escolha=escolher(resA),
               calibracao=dict(resumo=resA, linhas=linhasA), confirmacao=dict(resumo=resB, linhas=linhasB))
    texto = markdown(res)
    print(texto)
    if a.saida_json:
        Path(a.saida_json).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    if a.saida_md:
        Path(a.saida_md).write_text(texto, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
