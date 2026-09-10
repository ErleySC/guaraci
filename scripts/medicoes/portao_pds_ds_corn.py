#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""portao_pds_ds_corn.py -- Passo 135 (Bloco 22): reaplica o portao de
aceite (Bloco 20) a' transferencia de calibracao PDS/DS no Corn real,
reproduzindo formalmente (10 seeds independentes, nao so' o seed=0 fixo
do teste original) o resultado ja conhecido de
`tests/test_validacao_publica.py::
test_transferencia_de_calibracao_reduz_erro_entre_instrumentos_do_corn`
(RMSEP sem~0.51, com PDS~0.16). Contra-prova de que o mecanismo do Bloco
20 reproduz um resultado ja validado ANTES de confiar nele para tecnicas
novas (EMSC/OSC, Passo 134).

REGRA DE PAUSA (instrucao do usuario): se o portao NAO confirmar PDS
aprovado com o mesmo padrao ja conhecido, e' achado grave a investigar,
nao a esconder -- este script imprime o resultado cru, sem forcar
conclusao."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ / "src"))

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from guaraci.transferencia_calibracao import (
    apply_standardization, direct_standardization, piecewise_direct_standardization)
from guaraci.portao_correcao_sinal import avaliar_correcao_sinal


def _caminho_corn() -> str:
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        raise SystemExit("Defina GUARACI_DATASETS_DIR (contendo corn.mat).")
    return str(Path(raiz) / "corn.mat")


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    import scipy.io as sio

    m = sio.loadmat(_caminho_corn())
    X_m5 = np.asarray(m["m5spec"]["data"][0, 0], dtype=float)
    X_mp5 = np.asarray(m["mp5spec"]["data"][0, 0], dtype=float)
    Y = np.asarray(m["propvals"]["data"][0, 0], dtype=float)
    proteina = Y[:, 2]

    def _rodar(seed: int, metodo) -> float:
        """Mesma estrutura de `test_transferencia_de_calibracao_reduz_
        erro_entre_instrumentos_do_corn` (transferencia/calibracao/teste,
        15/40/25), so' que parametrizada por `seed` em vez de fixa em 0 --
        e' o que garante split BLOQUEADO mas repetido em 10 particoes
        independentes, exigencia do Bloco 20."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(80)
        idx_transf, idx_cal, idx_teste = idx[:15], idx[15:55], idx[55:]

        pls = PLSRegression(n_components=7, scale=False)
        pls.fit(X_m5[idx_cal], proteina[idx_cal])
        y_teste = proteina[idx_teste]

        if metodo is None:
            pred = pls.predict(X_mp5[idx_teste]).ravel()
            return _rmse(pred, y_teste)

        transform = metodo(X_m5[idx_transf], X_mp5[idx_transf])
        X_padronizado = apply_standardization(X_mp5[idx_teste], transform)
        pred = pls.predict(X_padronizado).ravel()
        return _rmse(pred, y_teste)

    for nome, metodo in (
            ("PDS", lambda a, b: piecewise_direct_standardization(a, b, janela=5, alpha=0.001)),
            ("DS", lambda a, b: direct_standardization(a, b)),
    ):
        v = avaliar_correcao_sinal(
            nome,
            avaliar_sem_fn=lambda seed: _rodar(seed, None),
            avaliar_com_fn=lambda seed, m=metodo: _rodar(seed, m),
            metrica="RMSEP", n_seeds=10)
        print(v.resumo())
        print(f"  scores_sem={[f'{x:.3f}' for x in v.scores_sem]}")
        print(f"  scores_com={[f'{x:.3f}' for x in v.scores_com]}")


if __name__ == "__main__":
    main()
