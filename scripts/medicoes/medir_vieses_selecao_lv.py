# -*- coding: utf-8 -*-
"""Mede o viés de seleção de nº de variáveis latentes (VLs) na métrica de CV
reportada por `pipeline.executar()` (achado #12, rodada multiagente
2026-09-10, Passo 202).

Hoje: `n_opt` é escolhido pelo RMSECV mínimo (parcimônia de Wold) nos MESMOS
5 folds cuja predição (`preds_por_lv[n_opt]`) vira a métrica de classificação
reportada (`balanced_accuracy` etc.) -- é a mesma unidade de dado usada para
ESCOLHER o modelo e para AVALIÁ-LO, o que tende a inflar a métrica (viés de
seleção). O corretivo textual (CV aninhada / nested CV) escolhe `n_opt`
usando só uma partição INTERNA do treino de cada fold externo, nunca vendo o
fold de validação externo.

Este script mede a diferença REAL entre as duas metodologias no dataset
privado, reusando build_preprocessor/StableStratifiedGroupKFold. NÃO altera
o pipeline de produção -- decisão de propagar fica para depois de reportar o
número (mesma disciplina do Passo 201).

Uso:
    python scripts/medicoes/medir_vieses_selecao_lv.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import threadpoolctl
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

_RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ / "src"))
sys.path.insert(0, str(_RAIZ / "scripts" / "medicoes"))

from guaraci.chemometric_stats import expandir_binario_um_quente, rmse_flat  # noqa: E402
from guaraci.pipeline import classification_metrics  # noqa: E402
from guaraci.preprocessamento import build_preprocessor  # noqa: E402
from guaraci.validacao_estatistica import StableStratifiedGroupKFold  # noqa: E402
from medir_mae_id_vs_sessao import _carregar_dados_reais  # noqa: E402


def _fabrica(cfg, n_lv):
    return Pipeline([("preproc", build_preprocessor(cfg)),
                      ("pls", PLSRegression(n_components=n_lv, scale=False))])


def _escolher_n_opt_por_rmsecv(cfg, X, Y_bin, cv_indices, max_lvs):
    """Reproduz a heurística de parcimônia de Wold de `pipeline.executar()`."""
    erros = []
    preds = {}
    for n in range(1, max_lvs + 1):
        y_hat = np.zeros_like(Y_bin)
        contador = np.zeros(len(Y_bin), dtype=int)
        for tr, va in cv_indices:
            pipe = _fabrica(cfg, n)
            pipe.fit(X[tr], Y_bin[tr])
            y_hat[va] += pipe.predict(X[va])
            contador[va] += 1
        contador[contador == 0] = 1
        y_hat = y_hat / contador[:, None]
        erros.append(rmse_flat(Y_bin, y_hat))
        preds[n] = y_hat
    erros_arr = np.array(erros)
    tol = erros_arr.min() * 1.02
    n_opt = int(np.where(erros_arr <= tol)[0][0]) + 1
    return n_opt, preds[n_opt]


def main() -> None:
    print("[1/3] Carregando dataset privado ...")
    cfg, X_raw, rotulos, conc, mae_id = _carregar_dados_reais()
    lb = LabelBinarizer()
    Y_bin = np.asarray(lb.fit_transform(rotulos), dtype=float)
    Y_bin = expandir_binario_um_quente(Y_bin)
    y_int = np.argmax(Y_bin, axis=1)
    classes = np.unique(rotulos)

    n_grupos = len(np.unique(mae_id))
    n_splits = min(cfg.n_splits_cv, n_grupos)
    max_lvs = cfg.max_lvs

    with threadpoolctl.threadpool_limits(1):
        t0 = time.time()
        print("[2/3] Metodologia ATUAL (n_opt escolhido e avaliado nos "
              "MESMOS 5 folds externos) ...")
        cv_ext = StableStratifiedGroupKFold(n_splits=max(n_splits, 2), seed=cfg.seed)
        cv_indices_ext = list(cv_ext.split(X_raw, y_int, groups=mae_id))
        n_opt_naive, y_hat_naive = _escolher_n_opt_por_rmsecv(
            cfg, X_raw, Y_bin, cv_indices_ext, max_lvs)
        pred_naive = np.argmax(y_hat_naive, axis=1)
        m_naive = classification_metrics(y_int, pred_naive, np.arange(len(classes)))
        t_naive = time.time() - t0
        print(f"    n_opt={n_opt_naive}  balanced_accuracy={m_naive['balanced_accuracy']:.4f}"
              f"  ({t_naive:.0f}s)")

        print("[3/3] Metodologia ANINHADA (n_opt escolhido só com o TREINO "
              "de cada fold externo, nunca vendo a validação externa) ...")
        t0 = time.time()
        y_hat_aninhado = np.zeros_like(Y_bin)
        n_opts_por_fold = []
        for i, (tr, va) in enumerate(cv_indices_ext):
            # CV INTERNA: só dentro do treino deste fold externo.
            X_tr, Y_tr, y_tr_int = X_raw[tr], Y_bin[tr], y_int[tr]
            grp_tr = mae_id[tr]
            n_grupos_tr = len(np.unique(grp_tr))
            n_splits_int = min(cfg.n_splits_cv, n_grupos_tr)
            cv_int = StableStratifiedGroupKFold(
                n_splits=max(n_splits_int, 2), seed=cfg.seed)
            cv_indices_int = list(cv_int.split(X_tr, y_tr_int, groups=grp_tr))
            n_opt_i, _ = _escolher_n_opt_por_rmsecv(
                cfg, X_tr, Y_tr, cv_indices_int, max_lvs)
            n_opts_por_fold.append(n_opt_i)
            # Ajusta com n_opt_i em TODO o treino externo, aplica na
            # validacao externa -- esta e' a predicao "honesta" (nunca viu
            # `va` nem para escolher n_opt nem para treinar).
            pipe = _fabrica(cfg, n_opt_i)
            pipe.fit(X_tr, Y_tr)
            y_hat_aninhado[va] = pipe.predict(X_raw[va])
            print(f"    fold externo {i}: n_opt_interno={n_opt_i} "
                  f"(|va|={len(va)})")
        pred_aninhado = np.argmax(y_hat_aninhado, axis=1)
        m_aninhado = classification_metrics(
            y_int, pred_aninhado, np.arange(len(classes)))
        t_aninhado = time.time() - t0

    print("\n===================== RESULTADO =====================")
    print(f"n_opts por fold externo (aninhado): {n_opts_por_fold}")
    print(f"balanced_accuracy  NAIVE (atual, n_opt={n_opt_naive})   = "
          f"{m_naive['balanced_accuracy']:.4f}  ({t_naive:.0f}s)")
    print(f"balanced_accuracy  ANINHADA (nested, n_opt por fold)   = "
          f"{m_aninhado['balanced_accuracy']:.4f}  ({t_aninhado:.0f}s)")
    print(f"delta (naive - aninhada) = "
          f"{m_naive['balanced_accuracy'] - m_aninhado['balanced_accuracy']:+.4f}")
    print(f"custo relativo (aninhada / naive): {t_aninhado / max(t_naive, 1e-9):.1f}x")


if __name__ == "__main__":
    main()
