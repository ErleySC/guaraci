#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""portao_mcr_als_como_feature.py -- Passo 136 (Bloco 23, item opcional):
avalia, atraves do portao de aceite (Bloco 20), se usar as proporcoes
resolvidas pelo MCR-ALS como VARIAVEIS DE ENTRADA EXTRA (concatenadas ao
espectro bruto) ajuda o PLS-R supervisionado a quantificar teor -- em vez
de MCR-ALS predizer teor diretamente (ja' testado e rejeitado no Passo
131/mcr_als.py).

Escopo do experimento: uma UNICA combinacao especie+adulterante (nao o
dataset pooled inteiro) -- MCR-ALS com 2 componentes so' faz sentido
conceitual quando a mistura tem 2 "puros" de verdade (oleo + adulterante);
rodar com 2 componentes sobre o pooled inteiro (13 especies x 3
adulterantes) nao teria interpretacao fisica coerente."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ / "src"))

import numpy as np
from scipy.optimize import nnls
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from guaraci.dados_io import load_dx
from guaraci.mcr_als import mcr_als
from guaraci.portao_correcao_sinal import avaliar_correcao_sinal
from guaraci.validacao_estatistica import StableStratifiedGroupKFold

FAIXA_MIN_CM, FAIXA_MAX_CM = 4550.0, 9000.0
ESPECIE, ADULTERANTE = "Babaçu", "milho"


def _pasta_acervo() -> str:
    caminho = os.environ.get("GUARACI_DADOS_REAIS")
    if not caminho:
        raise SystemExit("Defina GUARACI_DADOS_REAIS.")
    return caminho


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    wn, X, _rot, conc, mae, meta = load_dx(_pasta_acervo())
    janela = (wn >= FAIXA_MIN_CM) & (wn <= FAIXA_MAX_CM)
    mask = (meta["especie"] == ESPECIE) & (
        (meta["adulterante_nome"] == ADULTERANTE) | (meta["puro"] == True))  # noqa: E712
    X_sel = np.clip(X[mask.values][:, janela], 0.0, None)
    y_sel = np.nan_to_num(conc[mask.values], nan=0.0)
    grupos_sel = mae[mask.values]
    print(f"{X_sel.shape[0]} amostras, {X_sel.shape[1]} canais, "
          f"{len(set(grupos_sel))} grupos, teor {y_sel.min():.2f}-{y_sel.max():.2f}%")

    def _rodar(seed: int, usar_mcr_extra: bool) -> float:
        splitter = StableStratifiedGroupKFold(n_splits=3, seed=seed)
        folds = list(splitter.split(np.zeros(len(X_sel)), np.zeros(len(X_sel)),
                                    groups=grupos_sel))
        y_hat = np.zeros(len(y_sel))
        contador = np.zeros(len(y_sel), dtype=int)
        for idx_tr, idx_va in folds:
            X_tr, X_va = X_sel[idx_tr], X_sel[idx_va]
            if usar_mcr_extra:
                res = mcr_als(X_tr, n_componentes=2, max_iter=300)
                prop_tr = res.C / res.C.sum(axis=1, keepdims=True)
                prop_va = np.array([nnls(res.S.T, xv)[0] for xv in X_va])
                soma_va = prop_va.sum(axis=1, keepdims=True)
                soma_va[soma_va < 1e-12] = 1.0
                prop_va = prop_va / soma_va
                X_tr_uso = np.hstack([X_tr, prop_tr])
                X_va_uso = np.hstack([X_va, prop_va])
            else:
                X_tr_uso, X_va_uso = X_tr, X_va
            mc = StandardScaler(with_std=False)
            X_tr_uso = mc.fit_transform(X_tr_uso)
            X_va_uso = mc.transform(X_va_uso)
            n_comp = int(max(1, min(5, X_tr_uso.shape[1], len(idx_tr) - 1)))
            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(X_tr_uso, y_sel[idx_tr])
            y_hat[idx_va] += pls.predict(X_va_uso).ravel()
            contador[idx_va] += 1
        contador[contador == 0] = 1
        return _rmse(y_sel, y_hat / contador)

    v = avaliar_correcao_sinal(
        "mcr_als_como_feature_extra",
        avaliar_sem_fn=lambda seed: _rodar(seed, False),
        avaliar_com_fn=lambda seed: _rodar(seed, True),
        metrica="RMSEP", n_seeds=10)
    print(v.resumo())
    print("scores_sem:", [f"{x:.3f}" for x in v.scores_sem])
    print("scores_com:", [f"{x:.3f}" for x in v.scores_com])


if __name__ == "__main__":
    main()
