#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""portao_emsc_osc.py -- Passo 134 (Bloco 21): roda EMSC e OSC atraves do
portao de aceite (Bloco 20) contra o acervo de oleo privado (quantificacao
pooled de teor de adulterante, todas as especies/adulterantes juntos,
split group-aware por mae_id -- mesma unidade ja usada em
`pls_regressao_pooled`) e o dataset publico Corn (RMSEP de proteina,
baseline conhecido ~0,148 no m5 sozinho).

Uso:
    python scripts/medicoes/portao_emsc_osc.py

Requer GUARACI_DADOS_REAIS (acervo privado) e GUARACI_DATASETS_DIR/corn.mat
(dataset publico) -- ver docstring de validar_mcr_als_oleos_reais.py pro
motivo de nao hardcodar nenhum dos dois caminhos aqui."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ / "src"))

import numpy as np

from guaraci.dados_io import load_dx
from guaraci.preprocessamento import EMSC, OSC
from guaraci.portao_correcao_sinal import avaliar_correcao_sinal_pls

FAIXA_MIN_CM, FAIXA_MAX_CM = 4550.0, 9000.0


def _pasta_acervo() -> str:
    caminho = os.environ.get("GUARACI_DADOS_REAIS")
    if not caminho:
        raise SystemExit("Defina GUARACI_DADOS_REAIS (acervo privado de oleo).")
    return caminho


def _caminho_corn() -> str:
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        raise SystemExit("Defina GUARACI_DATASETS_DIR (contendo corn.mat).")
    return str(Path(raiz) / "corn.mat")


def _dataset_oleo_pooled():
    """Quantificacao POOLED (todas as especies/adulterantes juntos) de teor
    declarado -- mesma unidade de agrupamento (`mae_id`) e mesma janela
    espectral (`config.yaml`) ja usadas no Passo 131."""
    wn, X, _rot, conc, mae, meta = load_dx(_pasta_acervo())
    janela = (wn >= FAIXA_MIN_CM) & (wn <= FAIXA_MAX_CM)
    mask = (~meta["puro"].values) & np.isfinite(conc)
    X_sel = np.clip(X[mask][:, janela], 0.0, None)
    y_sel = conc[mask]
    grupos_sel = mae[mask]
    return X_sel, y_sel, grupos_sel, wn[janela]


def _dataset_corn():
    import scipy.io as sio
    m = sio.loadmat(_caminho_corn())
    X = np.asarray(m["m5spec"]["data"][0, 0], dtype=float)
    eixo = np.asarray(m["m5spec"]["axisscale"][0, 0][1, 0], dtype=float).ravel()
    Y = np.asarray(m["propvals"]["data"][0, 0], dtype=float)
    proteina = Y[:, 2]
    grupos = np.array([f"corn_{i}" for i in range(X.shape[0])])
    return X, proteina, grupos, eixo


def main() -> None:
    cenarios = []
    print("Carregando acervo de oleo (pooled, todas especies/adulterantes)...")
    X_oleo, y_oleo, g_oleo, eixo_oleo = _dataset_oleo_pooled()
    print(f"  {X_oleo.shape[0]} amostras, {X_oleo.shape[1]} canais, "
          f"{len(set(g_oleo))} grupos (mae_id), teor {y_oleo.min():.2f}-{y_oleo.max():.2f}%")
    cenarios.append(("oleo_pooled", X_oleo, y_oleo, g_oleo, eixo_oleo))

    print("Carregando Corn (m5, proteina)...")
    X_corn, y_corn, g_corn, eixo_corn = _dataset_corn()
    print(f"  {X_corn.shape[0]} amostras, {X_corn.shape[1]} canais")
    cenarios.append(("corn_m5_proteina", X_corn, y_corn, g_corn, eixo_corn))

    for nome_cenario, X, y, grupos, eixo in cenarios:
        for nome_tecnica, transformador in (
                ("EMSC", EMSC(eixo=eixo, ordem_polinomial=2)),
                ("OSC", OSC(n_componentes=1)),
        ):
            print(f"\n--- {nome_tecnica} em {nome_cenario} ---")
            v = avaliar_correcao_sinal_pls(
                f"{nome_tecnica}_{nome_cenario}", X, y, grupos, transformador,
                metrica="RMSEP", n_componentes=7, n_seeds=10)
            print(v.resumo())


if __name__ == "__main__":
    main()
