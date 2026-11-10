"""Simulacao com verdade conhecida da recomendacao de metodo do `guaraci anova` (D-M a D-V).

Para cada cenario (3 grupos, unidades independentes) mede, em repeticoes de Monte Carlo:
  1. acerto da recomendacao: comparada com o metodo que o MECANISMO GERADOR pede
     (normal e homocedastico -> Tukey; variancias muito diferentes -> Games-Howell;
     caudas pesadas ou assimetria forte -> Kruskal-Wallis/Dunn);
  2. erro tipo I por familia (FWER: pelo menos um par significativo com medias iguais) do
     metodo recomendado e dos tres metodos fixos;
  3. poder para o par com diferenca real (grupo 1 contra o grupo 3, diferenca de 1,5 desvio).
Sem rede e sem dado real: o gerador e' o numpy com semente fixa.

Uso: python scripts/simular_recomendacao.py [--repeticoes N] [--saida ARQ.json] [--md ARQ.md]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import guaraci.anova_grupos as _ag
from guaraci.anova_grupos import anova_uma_via, diagnosticar_premissas, kruskal_dunn, recomendar_metodo

ALFA = 0.05
DELTA = 1.5        # diferenca do grupo 3, em desvios (na escala do grupo)
K = 3


def _normal(sd, n):
    return lambda rng, i, d: rng.normal(d if i == K - 1 else 0.0, sd[i], n[i])


def _t3(n):
    return lambda rng, i, d: (d if i == K - 1 else 0.0) + rng.standard_t(3, n[i]) / np.sqrt(3.0)   # variancia 1


def _lognormal(n):
    return lambda rng, i, d: (d if i == K - 1 else 0.0) + (rng.lognormal(0.0, 1.0, n[i]) - np.exp(0.5)) / np.sqrt((np.exp(1) - 1) * np.exp(1))


# nome -> (gerador, metodo correto, descricao)
CENARIOS = {
    "normal, variancias iguais, n=10": (_normal([1, 1, 1], [10, 10, 10]), "tukey", "bem comportado"),
    "normal, variancias iguais, n=4": (_normal([1, 1, 1], [4, 4, 4]), "tukey", "n minimo"),
    "normal, variancias iguais, n desigual 5/10/25": (_normal([1, 1, 1], [5, 10, 25]), "tukey", "desbalanceado"),
    "normal, variancia 2:1, n=10": (_normal([1, 1.0, np.sqrt(2)], [10, 10, 10]), "tukey", "razao 2 (leve)"),
    "normal, variancias 1:10, n=10": (_normal([1, 1.78, 3.16], [10, 10, 10]), "games-howell", "razao 10"),
    "normal, variancias 1:10, n desigual (grupo de maior variancia tem n=5)": (_normal([3.16, 1.78, 1.0], [5, 10, 25]), "games-howell", "caso classicamente liberal p/ Tukey"),
    "t com 3 gl (caudas pesadas), n=15": (_t3([15, 15, 15]), "kruskal-dunn", "caudas pesadas"),
    "lognormal (assimetrica), n=15": (_lognormal([15, 15, 15]), "kruskal-dunn", "assimetria forte"),
}


def _amostra(gerador, seed, delta):
    rng = np.random.default_rng(seed)
    ys, gs = [], []
    for i in range(K):
        v = np.asarray(gerador(rng, i, delta), dtype=float)
        ys.append(v)
        gs += [f"g{i + 1}"] * len(v)
    y = np.concatenate(ys)
    return y, np.array(gs), np.arange(len(y))


def _sig_par(comps, a, b, chave="p_ajustado"):
    return any({c["a"], c["b"]} == {a, b} and c[chave] < ALFA for c in comps)


def _sem_ic_caro():
    """So' o p ajustado importa aqui; o valor critico do IC simultaneo (`_q_critico`, dezenas de
    ms por par) e o bootstrap do tamanho de efeito nao mudam p e deixariam a simulacao 10x mais
    lenta. `tests/test_simular_recomendacao.py` confere que o p e' identico com e sem este atalho."""
    _ag._q_critico = lambda *a, **k: 0.0


def _rodar(metodo, y, g, u):
    if metodo == "tukey":
        return anova_uma_via(y, g, u, variancias="iguais", n_boot=20).comparacoes
    if metodo == "games-howell":
        return anova_uma_via(y, g, u, variancias="desiguais", n_boot=20).comparacoes
    return kruskal_dunn(y, g, u).comparacoes


def _simular_cenario(args) -> tuple:
    nome, repeticoes = args
    _sem_ic_caro()
    metodos = ("tukey", "games-howell", "kruskal-dunn")
    gerador, correto, desc = CENARIOS[nome]
    recs = {m: 0 for m in metodos}
    fwer = {m: 0 for m in (*metodos, "recomendado")}
    poder = {m: 0 for m in (*metodos, "recomendado")}
    for r in range(repeticoes):
        y, g, u = _amostra(gerador, 10_000 + r, 0.0)
        rec = recomendar_metodo(diagnosticar_premissas(y, g, u, alpha=ALFA))["metodo"]
        recs[rec] += 1
        rejeita = {m: any(c["p_ajustado"] < ALFA for c in _rodar(m, y, g, u)) for m in metodos}
        for m in metodos:
            fwer[m] += rejeita[m]
        fwer["recomendado"] += rejeita[rec]
        y1, g1, u1 = _amostra(gerador, 500_000 + r, DELTA)
        rec1 = recomendar_metodo(diagnosticar_premissas(y1, g1, u1, alpha=ALFA))["metodo"]
        sig = {m: _sig_par(_rodar(m, y1, g1, u1), "g1", "g3") for m in metodos}
        for m in metodos:
            poder[m] += sig[m]
        poder["recomendado"] += sig[rec1]
    return nome, {
        "descricao": desc, "metodo_correto": correto,
        "acerto": recs[correto] / repeticoes,
        "recomendacao": {m: recs[m] / repeticoes for m in metodos},
        "fwer": {m: v / repeticoes for m, v in fwer.items()},
        "poder": {m: v / repeticoes for m, v in poder.items()},
    }


def simular(repeticoes: int, processos: int = 1) -> dict:
    tarefas = [(nome, repeticoes) for nome in CENARIOS]
    if processos > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=processos) as ex:
            pares = list(ex.map(_simular_cenario, tarefas))
    else:
        pares = [_simular_cenario(t) for t in tarefas]
    return {"repeticoes": repeticoes, "alfa": ALFA, "delta": DELTA, "cenarios": dict(pares)}


def _md(r: dict) -> str:
    n = r["repeticoes"]
    L = [f"# Simulacao da recomendacao de metodo ({n} repeticoes por cenario e por condicao)", "", "**Público:** quem avalia os números · **Estado:** gerado por `scripts/simular_recomendacao.py` (não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`", "",
         "Verdade conhecida: o metodo correto e' o que o mecanismo gerador pede. Medias iguais para o erro tipo I; "
         f"diferenca de {r['delta']} desvio no grupo 3 para o poder. Nivel nominal {r['alfa']}.", "",
         "## 1. A recomendacao acerta?", "",
         "| cenario | correto | recomendou Tukey | Games-Howell | Kruskal/Dunn | acerto |", "|---|---|---|---|---|---|"]
    for nome, c in r["cenarios"].items():
        rc = c["recomendacao"]
        L.append(f"| {nome} | {c['metodo_correto']} | {rc['tukey']:.1%} | {rc['games-howell']:.1%} | {rc['kruskal-dunn']:.1%} | **{c['acerto']:.1%}** |")
    L += ["", "## 2. Erro tipo I por familia (FWER) e poder do par com diferenca real", "",
          "| cenario | FWER Tukey | FWER G-H | FWER Dunn | FWER recomendado | poder Tukey | poder G-H | poder Dunn | poder recomendado |",
          "|---|---|---|---|---|---|---|---|---|"]
    for nome, c in r["cenarios"].items():
        f, p = c["fwer"], c["poder"]
        L.append(f"| {nome} | {f['tukey']:.3f} | {f['games-howell']:.3f} | {f['kruskal-dunn']:.3f} | **{f['recomendado']:.3f}** | "
                 f"{p['tukey']:.2f} | {p['games-howell']:.2f} | {p['kruskal-dunn']:.2f} | **{p['recomendado']:.2f}** |")
    L += ["", f"Com {n} repeticoes o erro padrao de uma taxa perto de 0,05 e' cerca de {np.sqrt(0.05 * 0.95 / n):.3f}."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeticoes", type=int, default=500)
    ap.add_argument("--saida", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--processos", type=int, default=1)
    a = ap.parse_args(argv)
    r = simular(a.repeticoes, a.processos)
    texto = _md(r)
    print(texto)
    if a.saida:
        with open(a.saida, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=2, ensure_ascii=False)
    if a.md:
        with open(a.md, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(texto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
