"""Prova empirica de que a selecao de variaveis fica DENTRO da validacao (rotulos sem sinal).

Com rotulos sorteados e variaveis de ruido puro nenhum modelo honesto passa do acaso
(1/K). Cada metodo e' avaliado de dois jeitos: ANINHADO (a mascara e' refeita a cada
dobra so' com o treino, como o pipeline faz) e INGENUO (a mascara sai de todos os dados
e so' depois vem a CV, o erro classico de Ambroise & McLachlan 2002). O ingenuo precisa
passar do acaso, senao o experimento nao teria poder de detectar vazamento.

Uso: python scripts/prova_selecao_aninhada.py [--sementes N] [--saida ARQ.json]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from guaraci.selecao_variaveis import (
    _avaliar_busca_nested_cv,
    _avaliar_subset_cv,
    _avaliar_subset_nested_cv,
    _mask_melhor_intervalo,
    _mask_sr_top_frac,
    _mask_vip_threshold,
    cars_selecao,
    selecao_spa,
    sparse_plsda_mask,
)
from guaraci.validacao_estatistica import StableStratifiedGroupKFold

N_AMOSTRAS, N_VARIAVEIS, N_CLASSES, N_LV = 60, 200, 3, 2


def _dados(seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N_AMOSTRAS, N_VARIAVEIS))
    y = np.repeat(np.arange(N_CLASSES), N_AMOSTRAS // N_CLASSES)
    rng.shuffle(y)
    Y = np.eye(N_CLASSES)[y]
    grupos = np.arange(N_AMOSTRAS)
    cv = list(StableStratifiedGroupKFold(n_splits=5, seed=seed).split(X, y, grupos))
    return X, Y, y, grupos, cv


def _metodos(seed: int):
    """nome -> (mascara no treino | None, busca (com CV interna) | None)."""
    return {
        "VIP>=1": (lambda Xt, Yt, nlv: _mask_vip_threshold(Xt, Yt, nlv, 1.0), None),
        "SR top 10%": (lambda Xt, Yt, nlv: _mask_sr_top_frac(Xt, Yt, nlv, 0.10), None),
        "sPLS-DA": (lambda Xt, Yt, nlv: sparse_plsda_mask(Xt, Yt, nlv, 10), None),
        "iPLS": (lambda Xt, Yt, nlv: _mask_melhor_intervalo(Xt, Yt, nlv, 10, seed), None),
        "SPA": (None, lambda Xt, Yt, yt, cvin: selecao_spa(Xt, Yt, yt, cvin, N_LV, 10, 3, seed)[1]),
        "CARS": (None, lambda Xt, Yt, yt, cvin: cars_selecao(Xt, Yt, yt, cvin, N_LV, 6, 0.8, seed)[1]),
    }


def medir(n_sementes: int) -> dict:
    out: dict = {"n_amostras": N_AMOSTRAS, "n_variaveis": N_VARIAVEIS, "n_classes": N_CLASSES,
                 "acaso": 1 / N_CLASSES, "n_sementes": n_sementes, "metodos": {}}
    acumulado: dict = {}
    for s in range(n_sementes):
        X, Y, y, grupos, cv = _dados(s)
        todas = _avaliar_subset_cv(X, Y, y, cv, N_LV)["balanced_accuracy"]
        acumulado.setdefault("todas as variaveis", {"aninhado": [], "ingenuo": []})["aninhado"].append(todas)
        for nome, (mascara, busca) in _metodos(s).items():
            reg = acumulado.setdefault(nome, {"aninhado": [], "ingenuo": []})
            if mascara is not None:
                aninhado = _avaliar_subset_nested_cv(X, Y, y, cv, N_LV, mascara)["balanced_accuracy"]
                m = mascara(X, Y, N_LV)
            else:
                aninhado = _avaliar_busca_nested_cv(X, Y, y, cv, N_LV, busca, s, mae_id=grupos)["balanced_accuracy"]
                m = busca(X, Y, y, cv)
            m = np.asarray(m, dtype=bool)
            if m.sum() < 2:
                m = np.ones(N_VARIAVEIS, dtype=bool)
            ingenuo = _avaliar_subset_cv(X[:, m], Y, y, cv, N_LV)["balanced_accuracy"]
            reg["aninhado"].append(aninhado)
            reg["ingenuo"].append(ingenuo)
    for nome, reg in acumulado.items():
        out["metodos"][nome] = {k: {"media": float(np.mean(v)), "maximo": float(np.max(v)), "valores": [round(float(x), 4) for x in v]}
                                for k, v in reg.items() if v}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sementes", type=int, default=20)
    ap.add_argument("--saida", default=None)
    a = ap.parse_args(argv)
    r = medir(a.sementes)
    print(f"rotulos sem sinal: {r['n_amostras']} amostras, {r['n_variaveis']} variaveis, acaso = {r['acaso']:.3f}")
    print(f"{'metodo':<22}{'aninhado (media)':>18}{'ingenuo (media)':>18}")
    for nome, v in r["metodos"].items():
        ing = f"{v['ingenuo']['media']:.3f}" if "ingenuo" in v else "-"
        print(f"{nome:<22}{v['aninhado']['media']:>18.3f}{ing:>18}")
    if a.saida:
        with open(a.saida, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
