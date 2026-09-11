# -*- coding: utf-8 -*-
"""medir_vieses_selecao_lv_replicado.py -- replica a medicao de UMA
execucao de `medir_vieses_selecao_lv.py` (achado #12, Passo 202) em
`N_SEEDS` seeds de split INDEPENDENTES + teste de Wilcoxon pareado, no
mesmo padrao ja usado em `comparar_npls_pixelwise_mango.py` (10 seeds +
`scipy.stats.wilcoxon`).

POR QUE. A medicao original (Passo 202) foi 1 UNICA execucao: CV aninhada
deu balanced_accuracy MAIOR (0,8499) que a atual (0,8299) -- direcao
OPOSTA a hipotese do relatorio de que a selecao atual infla a metrica.
1 medicao nao decide nada (pode ser artefato da particao especifica
daquele seed); esta replicacao roda a MESMA comparacao em 10 seeds
diferentes de split e aplica Wilcoxon pareado pra' confirmar se a
diferenca e' consistente entre replicas, nao ruido de 1 particao.

CUSTO (medido no Passo 202, 1 execucao): naive=48s + aninhada=177s =
225s/seed. Para 10 seeds: ~2250s (~37,5 min). NAO altera o pipeline de
producao -- so' mede e reporta.

Uso:
    python scripts/medicoes/medir_vieses_selecao_lv_replicado.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

_RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ / "src"))
sys.path.insert(0, str(_RAIZ / "scripts" / "medicoes"))

from guaraci.chemometric_stats import expandir_binario_um_quente  # noqa: E402
from guaraci.pipeline import classification_metrics  # noqa: E402
from guaraci.validacao_estatistica import StableStratifiedGroupKFold  # noqa: E402
from medir_mae_id_vs_sessao import _carregar_dados_reais  # noqa: E402
from medir_vieses_selecao_lv import _escolher_n_opt_por_rmsecv, _fabrica  # noqa: E402
from sklearn.preprocessing import LabelBinarizer  # noqa: E402

N_SEEDS = 10


def _rodar_1_seed(cfg, X_raw, Y_bin, y_int, mae_id, classes, max_lvs, seed):
    n_grupos = len(np.unique(mae_id))
    n_splits = min(cfg.n_splits_cv, n_grupos)

    t0 = time.time()
    cv_ext = StableStratifiedGroupKFold(n_splits=max(n_splits, 2), seed=seed)
    cv_indices_ext = list(cv_ext.split(X_raw, y_int, groups=mae_id))
    n_opt_naive, y_hat_naive = _escolher_n_opt_por_rmsecv(
        cfg, X_raw, Y_bin, cv_indices_ext, max_lvs)
    pred_naive = np.argmax(y_hat_naive, axis=1)
    m_naive = classification_metrics(y_int, pred_naive, np.arange(len(classes)))
    t_naive = time.time() - t0

    t0 = time.time()
    y_hat_aninhado = np.zeros_like(Y_bin)
    n_opts_por_fold = []
    for tr, va in cv_indices_ext:
        X_tr, Y_tr, y_tr_int = X_raw[tr], Y_bin[tr], y_int[tr]
        grp_tr = mae_id[tr]
        n_grupos_tr = len(np.unique(grp_tr))
        n_splits_int = min(cfg.n_splits_cv, n_grupos_tr)
        cv_int = StableStratifiedGroupKFold(
            n_splits=max(n_splits_int, 2), seed=seed)
        cv_indices_int = list(cv_int.split(X_tr, y_tr_int, groups=grp_tr))
        n_opt_i, _ = _escolher_n_opt_por_rmsecv(
            cfg, X_tr, Y_tr, cv_indices_int, max_lvs)
        n_opts_por_fold.append(n_opt_i)
        pipe = _fabrica(cfg, n_opt_i)
        pipe.fit(X_tr, Y_tr)
        y_hat_aninhado[va] = pipe.predict(X_raw[va])
    pred_aninhado = np.argmax(y_hat_aninhado, axis=1)
    m_aninhado = classification_metrics(y_int, pred_aninhado, np.arange(len(classes)))
    t_aninhado = time.time() - t0

    return {
        "n_opt_naive": n_opt_naive,
        "bal_acc_naive": m_naive["balanced_accuracy"],
        "t_naive": t_naive,
        "n_opts_aninhado": n_opts_por_fold,
        "bal_acc_aninhado": m_aninhado["balanced_accuracy"],
        "t_aninhado": t_aninhado,
    }


def main() -> None:
    print("[1/2] Carregando dataset privado ...")
    cfg, X_raw, rotulos, conc, mae_id = _carregar_dados_reais()
    lb = LabelBinarizer()
    Y_bin = np.asarray(lb.fit_transform(rotulos), dtype=float)
    Y_bin = expandir_binario_um_quente(Y_bin)
    y_int = np.argmax(Y_bin, axis=1)
    classes = np.unique(rotulos)
    max_lvs = cfg.max_lvs

    custo_estimado_s = N_SEEDS * (48 + 177)
    print(f"[2/2] Rodando {N_SEEDS} seeds independentes (naive vs. aninhada) "
          f"-- custo estimado ~{custo_estimado_s}s (~{custo_estimado_s/60:.0f} min), "
          f"com base no tempo medido no Passo 202 (48s + 177s por seed) ...")

    naive_scores, aninhado_scores = [], []
    t_total0 = time.time()
    for seed in range(N_SEEDS):
        r = _rodar_1_seed(cfg, X_raw, Y_bin, y_int, mae_id, classes, max_lvs, seed)
        naive_scores.append(r["bal_acc_naive"])
        aninhado_scores.append(r["bal_acc_aninhado"])
        print(f"  seed={seed}  naive(n_opt={r['n_opt_naive']})="
              f"{r['bal_acc_naive']:.4f} ({r['t_naive']:.0f}s)   "
              f"aninhada(n_opts={r['n_opts_aninhado']})="
              f"{r['bal_acc_aninhado']:.4f} ({r['t_aninhado']:.0f}s)   "
              f"delta={r['bal_acc_naive'] - r['bal_acc_aninhado']:+.4f}")

    naive_scores = np.array(naive_scores)
    aninhado_scores = np.array(aninhado_scores)
    t_total = time.time() - t_total0

    print("\n===================== RESULTADO REPLICADO =====================")
    print(f"NAIVE (atual):    media={naive_scores.mean():.4f}  "
          f"desvio={naive_scores.std():.4f}")
    print(f"ANINHADA (nested):media={aninhado_scores.mean():.4f}  "
          f"desvio={aninhado_scores.std():.4f}")
    print(f"delta medio (naive - aninhada) = "
          f"{(naive_scores - aninhado_scores).mean():+.4f}")
    print(f"ANINHADA venceu em {int((aninhado_scores > naive_scores).sum())}"
          f"/{N_SEEDS} seeds")
    if np.allclose(naive_scores, aninhado_scores):
        print("Wilcoxon pareado: NAO APLICAVEL -- diferencas todas zero "
              "(naive e aninhada identicos em todas as seeds).")
    else:
        w = wilcoxon(naive_scores, aninhado_scores)
        print(f"Wilcoxon pareado: statistic={w.statistic:.2f}  p={w.pvalue:.4f}")
    print(f"custo real total: {t_total:.0f}s (~{t_total/60:.1f} min)")


if __name__ == "__main__":
    main()
