#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""validar_mcr_als_oleos_reais.py -- Passo 131: valida MCR-ALS (Bloco 14)
contra mistura REAL do acervo privado (Andiroba puro + Andiroba adulterado
com algodao, 15 niveis de teor declarado de 1,05% a 15,18%, 3 replicas
cada + 3 amostras puras = 48 amostras), comparando a concentracao
recuperada com o teor DECLARADO.

Caminho do acervo: variavel de ambiente GUARACI_DADOS_REAIS (ou 1o
argumento posicional) -- NUNCA hardcoded aqui, mesmo padrao ja
estabelecido em `medir_sessoes_especie_adulterante.py`/`medir_especie_vs_
adulterante_permanova.py` (o caminho real e' privado -- por isso
`config.yaml`/`run_benchmark_tcc.py` sao gitignored; um script rastreado
pelo git nao pode hardcodar esse caminho, so' referencia-lo por variavel
de ambiente). NAO e' a pasta `dados/` do repo (vazia de proposito --
dado de terceiro nunca versionado). Achado do Passo 130->131: a
"limitacao real" reportada antes (`dados/` vazia) era erro de caminho
meu, nao ausencia de dado -- o acervo sempre esteve acessivel (ver
`config.yaml` local, gitignored, que ja apontava pra ele), so' nao
chequei antes de concluir que faltava.

JANELA ESPECTRAL: 4550-9000 cm^-1 (mesma de `config.yaml`, faixa_min_cm/
faixa_max_cm) -- fora dela ha' saturacao do detector (A>=2,99, fora da
faixa linear de Beer-Lambert, comentario do proprio config.yaml). SEM
pre-processamento SNV/MSC/SG: MCR-ALS pressupoe mistura ADITIVA/BILINEAR
(D = C @ S) sobre a absorbancia bruta -- SNV/MSC/derivada quebram essa
premissa (nao sao operacoes lineares na escala de concentracao). Valores
residuais < 0 (ruido/artefato de background, nunca absorbancia negativa
de verdade) sao clipados a 0 -- exigencia de nao-negatividade do proprio
MCR-ALS."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ / "src"))

import numpy as np
from scipy.stats import pearsonr

from guaraci.dados_io import load_dx
from guaraci.mcr_als import mcr_als, avaliar_incerteza_rotacional

FAIXA_MIN_CM, FAIXA_MAX_CM = 4550.0, 9000.0


def _pasta_acervo() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    caminho = os.environ.get("GUARACI_DADOS_REAIS")
    if not caminho:
        raise SystemExit(
            "Pasta do acervo nao informada. Uso: python "
            f"{os.path.basename(__file__)} <pasta_dados>  ou defina a "
            "variavel de ambiente GUARACI_DADOS_REAIS.")
    return caminho

#: Duas combinacoes deliberadamente contrastantes -- nao so' a "melhor
#: caso" (visto de tras pra' frente seria sicofancia). Escolhidas por uma
#: varredura previa de sanity-check SUPERVISIONADO (PLS-R simples, CV
#: 4-fold nao-group-aware, so' pra' TRIAGEM) em todas as 35 combinacoes
#: especie x adulterante do acervo: Andiroba+algodao foi a PIOR (Q2
#: supervisionado = -0.29, ou seja nem PLS-R acha sinal), Babacu+milho a
#: MELHOR (Q2 = 0.82). Rodar as duas mostra se o resultado do MCR-ALS
#: depende so' de o combo ter sinal supervisionado forte, ou se e' um
#: problema mais geral do metodo NAO-SUPERVISIONADO nesse tipo de dado.
COMBOS = [
    ("Andiroba", "algodão", "pior sinal supervisionado (Q2 PLS-R sanity-check=-0.29)"),
    ("Babaçu", "milho", "melhor sinal supervisionado (Q2 PLS-R sanity-check=0.82)"),
]


def _validar_combo(wn: np.ndarray, X: np.ndarray, conc: np.ndarray,
                   meta, especie: str, adulterante: str, nota: str) -> None:
    janela = (wn >= FAIXA_MIN_CM) & (wn <= FAIXA_MAX_CM)
    mask = (meta["especie"] == especie) & (
        (meta["adulterante_nome"] == adulterante) | (meta["puro"] == True))  # noqa: E712
    D = np.clip(X[mask.values][:, janela], 0.0, None)
    teor_declarado = np.nan_to_num(conc[mask.values], nan=0.0)

    print(f"\n{'='*70}\n{especie} + {adulterante} ({nota})\n{'='*70}")
    print(f"Amostras: {D.shape[0]}  Canais (janela {FAIXA_MIN_CM:.0f}-"
          f"{FAIXA_MAX_CM:.0f} cm-1): {D.shape[1]}")
    print(f"Teor declarado: {teor_declarado.min():.2f}% a "
          f"{teor_declarado.max():.2f}% ({len(set(teor_declarado.tolist()))} "
          f"niveis distintos)")

    res = mcr_als(D, n_componentes=2, max_iter=500)
    print(f"MCR-ALS: lof_percent={res.lof_percent:.2f}%  "
          f"n_iter={res.n_iter}  convergiu={res.convergiu}")

    proporcoes = res.C / res.C.sum(axis=1, keepdims=True)
    corr0 = pearsonr(proporcoes[:, 0], teor_declarado)
    corr1 = pearsonr(proporcoes[:, 1], teor_declarado)
    print(f"Correlacao componente 0 x teor declarado: r={corr0.statistic:.4f} "
          f"(p={corr0.pvalue:.2e})")
    print(f"Correlacao componente 1 x teor declarado: r={corr1.statistic:.4f} "
          f"(p={corr1.pvalue:.2e})")

    idx_adulterante = int(np.argmax([abs(corr0.statistic), abs(corr1.statistic)]))
    score_adulterante = proporcoes[:, idx_adulterante]
    r_escolhido = (corr0, corr1)[idx_adulterante].statistic

    coefs = np.polyfit(score_adulterante, teor_declarado, deg=1)
    pred = np.polyval(coefs, score_adulterante)
    ss_res = float(np.sum((teor_declarado - pred) ** 2))
    ss_tot = float(np.sum((teor_declarado - teor_declarado.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((teor_declarado - pred) ** 2)))
    print(f"Melhor componente (idx={idx_adulterante}, r={r_escolhido:.4f}) "
          f"calibrado linearmente -> teor declarado:")
    print(f"  R2={r2:.4f}  RMSE={rmse:.3f}pp  "
          f"erro_abs_medio={np.mean(np.abs(teor_declarado - pred)):.3f}pp  "
          f"erro_abs_max={np.max(np.abs(teor_declarado - pred)):.3f}pp")

    diag = avaliar_incerteza_rotacional(D, n_componentes=2, n_inicializacoes=5,
                                        seed=0, max_iter=500)
    print(f"Incerteza de rotacao (5 inicializacoes): "
          f"desvio_padrao_medio={diag['desvio_padrao_medio']:.4f}")


def main() -> None:
    pasta_acervo = _pasta_acervo()
    print(f"Carregando acervo real de: {pasta_acervo}")
    wn, X, _rot, conc, _mae, meta = load_dx(pasta_acervo)
    for especie, adulterante, nota in COMBOS:
        _validar_combo(wn, X, conc, meta, especie, adulterante, nota)


if __name__ == "__main__":
    main()
