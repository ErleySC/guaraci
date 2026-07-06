"""
validacao_estatistica.py — Testes de significancia e intervalos de confianca
do pipeline: cross_val_predict manual, IC BCa (Efron), CV-ANOVA (Eriksson),
teste de permutacao e teste de Wold (com paralelismo opcional via joblib).

Extraido de pipeline.py como parte da modularizacao (Fase H). Sem acoplamento a
Config — funcoes puras sobre X/Y/cv/pipeline_factory. pipeline.py reexporta os
nomes, entao pipeline.teste_permutacao(...), pipeline._cv_predict_manual(...)
etc. seguem inalterados (usados por comparar_pipelines, etapa4 e executar).
Coberto por tests/test_pipeline_smoke.py e tests/test_pipeline_core.py.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import f as f_dist, norm as _norm_dist
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline


def _cv_predict_manual(pipeline_factory, X, Y_bin, cv_indices):
    """Manual cross_val_predict, compatible with multilabel Y + stratified CV."""
    y_hat = np.zeros_like(Y_bin, dtype=float)
    contador = np.zeros(len(Y_bin), dtype=int)
    for tr, va in cv_indices:
        pipe = pipeline_factory()
        pipe.fit(X[tr], Y_bin[tr])
        y_hat[va] += pipe.predict(X[va])
        contador[va] += 1
    contador[contador == 0] = 1
    return y_hat / contador[:, None]


def bootstrap_bca_ci(y_true: np.ndarray, y_pred: np.ndarray,
                      metric_fn: Callable, n_boot: int = 500,
                      alpha: float = 0.05, seed: int = 42
                      ) -> Tuple[float, float, float]:
    """BCa confidence interval (bias-corrected & accelerated, Efron 1987)
    for a classification metric via stratified bootstrap.

    Returns (low, high, observed_value).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    observed = float(metric_fn(y_true, y_pred))

    classes = np.unique(y_true)
    idx_por_classe = {c: np.where(y_true == c)[0] for c in classes}

    boot_stats = []
    for _ in range(n_boot):
        partes = []
        for c in classes:
            ic = idx_por_classe[c]
            partes.append(rng.choice(ic, size=len(ic), replace=True))
        idx = np.concatenate(partes)
        try:
            boot_stats.append(float(metric_fn(y_true[idx], y_pred[idx])))
        except Exception:
            continue
    boot_stats = np.asarray(boot_stats)

    if len(boot_stats) < 20:
        return float("nan"), float("nan"), observed

    # Bias-correction z0
    prop_less = float(np.mean(boot_stats < observed))
    if prop_less <= 0 or prop_less >= 1:
        return (float(np.percentile(boot_stats, 100 * alpha / 2)),
                float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
                observed)
    z0 = _norm_dist.ppf(prop_less)

    # Acceleration via jackknife
    jack = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        try:
            jack[i] = float(metric_fn(y_true[mask], y_pred[mask]))
        except Exception:
            jack[i] = observed
    mean_jack = jack.mean()
    diffs = mean_jack - jack
    num = float(np.sum(diffs ** 3))
    den = 6.0 * (float(np.sum(diffs ** 2))) ** 1.5
    a = num / den if den > 0 else 0.0

    z_a_lo = _norm_dist.ppf(alpha / 2)
    z_a_hi = _norm_dist.ppf(1 - alpha / 2)
    denom_lo = 1 - a * (z0 + z_a_lo)
    denom_hi = 1 - a * (z0 + z_a_hi)
    if denom_lo == 0 or denom_hi == 0:
        return (float(np.percentile(boot_stats, 100 * alpha / 2)),
                float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
                observed)
    alpha_lo = _norm_dist.cdf(z0 + (z0 + z_a_lo) / denom_lo)
    alpha_hi = _norm_dist.cdf(z0 + (z0 + z_a_hi) / denom_hi)

    low  = float(np.percentile(boot_stats, 100 * alpha_lo))
    high = float(np.percentile(boot_stats, 100 * alpha_hi))
    return low, high, observed


def cv_anova_eriksson(Y: np.ndarray, Y_cv: np.ndarray, n_components: int
                       ) -> Dict[str, float]:
    """CV-ANOVA of Eriksson, Trygg & Wold (J. Chemometrics 22:594-600, 2008).

    Tests whether PRESS (CV residual) is significantly smaller than SS_total
    (variance around the mean of Y). H0: model does not improve prediction.

    F = ((SS_total - PRESS) / df_model) / (PRESS / df_residual)
    """
    Y = np.asarray(Y, dtype=float)
    Y_cv = np.asarray(Y_cv, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
        Y_cv = Y_cv.reshape(-1, 1)
    n, m = Y.shape

    ss_total = float(np.sum((Y - Y.mean(axis=0)) ** 2))
    press    = float(np.sum((Y - Y_cv) ** 2))

    if ss_total <= 0:
        return {"F": float("nan"), "p_value": 1.0, "Q2": 0.0,
                "df_model": 0, "df_resid": 0}
    if press >= ss_total:
        return {"F": 0.0, "p_value": 1.0,
                "Q2": 1.0 - press / ss_total,
                "df_model": n_components * m, "df_resid": max(n * m - n_components * m, 1)}

    # Eriksson et al. (2008) derivation assumes scalar Y (m=1).
    # With one-hot Y_bin (m=K classes), columns are NOT independent (row-sums = 1),
    # so treating m as effective observations inflates df by K-fold and produces
    # an artificially small p-value. Correction: use m_eff=1 for df calculation,
    # equivalent to reporting the test on the pooled univariate residual.
    m_eff    = 1
    df_model = n_components * m_eff
    df_resid = n * m_eff - df_model
    if df_resid <= 0:
        return {"F": float("nan"), "p_value": 1.0,
                "Q2": 1.0 - press / ss_total,
                "df_model": df_model, "df_resid": df_resid}

    if press <= 0:
        # Predicao perfeita (PRESS=0): o modelo explica tudo. F -> infinito e
        # p -> 0. Sem esta guarda, `press` no denominador causava
        # ZeroDivisionError (achado pela rede de seguranca / test_validacao_estatistica).
        return {"F": float("inf"), "p_value": 0.0, "Q2": 1.0,
                "df_model": int(df_model), "df_resid": int(df_resid)}

    F = ((ss_total - press) / df_model) / (press / df_resid)
    p = float(1.0 - f_dist.cdf(F, df_model, df_resid))
    return {"F":        float(F),
             "p_value":  p,
             "Q2":       1.0 - press / ss_total,
             "df_model": int(df_model),
             "df_resid": int(df_resid)}


def _iter_wold(pipeline_factory, X, Y_perm, y_perm_int, y_int, cv, groups):
    """Uma iteracao (pura, sem estado global) do teste de Wold. Usada tanto
    sequencialmente quanto via joblib.Parallel — o resultado depende so dos
    argumentos, entao rodar em paralelo nao muda o valor calculado, so a
    ordem de execucao (a ordem dos resultados e sempre preservada pelo
    chamador). Replica exatamente a logica de teste_wold pre-paralelizacao.
    Retorna ("ok", sim, r2, q2) | ("skip", ...) | ("fail", ...).
    """
    try:
        sim = float(np.mean(y_perm_int == y_int))
        cv_perm_idx = list(cv.split(X, y_perm_int, groups=groups))
        pipe = pipeline_factory(); pipe.fit(X, Y_perm)
        Y_tr_p = pipe.predict(X)
        Y_cv_p = _cv_predict_manual(pipeline_factory, X, Y_perm, cv_perm_idx)
        ss_tot_p = float(np.sum((Y_perm - Y_perm.mean(axis=0)) ** 2))
        if ss_tot_p < 1e-12:
            return ("skip", None, None, None)
        ss_res_r2 = float(np.sum((Y_perm - Y_tr_p) ** 2))
        ss_res_q2 = float(np.sum((Y_perm - Y_cv_p) ** 2))
        if not (np.isfinite(ss_res_r2) and np.isfinite(ss_res_q2)):
            return ("fail", None, None, None)
        r2 = max(-1.0, 1.0 - ss_res_r2 / ss_tot_p)
        q2 = max(-1.0, 1.0 - ss_res_q2 / ss_tot_p)
        return ("ok", sim, r2, q2)
    except Exception:
        return ("fail", None, None, None)


def teste_wold(pipeline_factory: Callable[[], Pipeline],
                X: np.ndarray, Y_bin: np.ndarray, y_int: np.ndarray,
                cv, n_perm: int, seed: int,
                groups: Optional[np.ndarray] = None,
                n_jobs: int = 1) -> Dict[str, object]:
    """Permutation test in the style of Wold/Westerhuis (J. Chemometrics 22:578-585):
    tracks R2Y and Q2Y for each permutation as a function of the similarity
    of the permuted Y with the original. Fits a line and reports intercepts.

    Classic model validity criteria (one-hot Y):
        R2Y intercept < 0.4
        Q2Y intercept < 0.05
    """
    rng = np.random.default_rng(seed)
    cv_indices = list(cv.split(X, y_int, groups=groups))

    # --- Observed ---------------------------------------------------------
    pipe = pipeline_factory(); pipe.fit(X, Y_bin)
    Y_train_obs = pipe.predict(X)
    Y_cv_obs    = _cv_predict_manual(pipeline_factory, X, Y_bin, cv_indices)
    ss_total    = float(np.sum((Y_bin - Y_bin.mean(axis=0)) ** 2))
    r2_obs = 1.0 - float(np.sum((Y_bin - Y_train_obs) ** 2)) / ss_total \
                if ss_total > 0 else 0.0
    q2_obs = 1.0 - float(np.sum((Y_bin - Y_cv_obs)    ** 2)) / ss_total \
                if ss_total > 0 else 0.0

    sims: List[float] = []
    r2s:  List[float] = []
    q2s:  List[float] = []
    n_validos = 0
    n_falhos  = 0

    import time as _time
    t0 = _time.time()
    progress_every = max(1, n_perm // 20)   # ~20 updates

    # Permutacoes geradas SEMPRE na mesma ordem/sequencia do rng, independente
    # de n_jobs — garante que o resultado (para um dado seed) e identico seja
    # a execucao sequencial ou paralela; so o tempo de parede muda.
    permutacoes = [rng.permutation(len(Y_bin)) for _ in range(n_perm)]

    if n_jobs <= 1:
        for i, idx in enumerate(permutacoes):
            Y_perm = Y_bin[idx]
            y_perm_int = np.argmax(Y_perm, axis=1)
            status, sim, r2, q2 = _iter_wold(
                pipeline_factory, X, Y_perm, y_perm_int, y_int, cv, groups)
            if status == "ok":
                sims.append(sim); r2s.append(r2); q2s.append(q2)
                n_validos += 1
            elif status == "fail":
                n_falhos += 1
            # "skip": ss_tot_p degenerada — ignorado silenciosamente (igual antes)

            if (i + 1) % progress_every == 0 or (i + 1) == n_perm:
                elapsed = _time.time() - t0
                taxa    = (i + 1) / max(elapsed, 1e-6)
                eta_s   = (n_perm - i - 1) / max(taxa, 1e-6)
                pct = (i + 1) / n_perm * 100
                print(f"    Wold {i+1:4d}/{n_perm}  ({pct:5.1f}%)  "
                      f"valid={n_validos} failed={n_falhos}  "
                      f"elapsed={elapsed:5.1f}s  ETA={eta_s:5.1f}s",
                      flush=True)
    else:
        from joblib import Parallel, delayed
        import threadpoolctl
        print(f"    Wold: {n_perm} permutacoes em paralelo (n_jobs={n_jobs})...",
              flush=True)
        # backend="loky" (processos, nao threads): medido que threading NAO
        # acelera aqui — grande parte do tempo e overhead Python do sklearn
        # (validacao/roteamento de metadados), que segura o GIL e nao roda em
        # paralelo entre threads. Processos separados (loky) contornam o GIL.
        # threadpool_limits(1) evita que o BLAS interno (ex.: OpenBLAS com 16
        # threads nesta maquina) sobrecarregue os nucleos por cima do
        # paralelismo externo (oversubscription), o que também prejudicava o
        # ganho medido.
        with threadpoolctl.threadpool_limits(1):
            resultados = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_iter_wold)(
                    pipeline_factory, X, Y_bin[idx], np.argmax(Y_bin[idx], axis=1),
                    y_int, cv, groups)
                for idx in permutacoes)
        for status, sim, r2, q2 in resultados:
            if status == "ok":
                sims.append(sim); r2s.append(r2); q2s.append(q2)
                n_validos += 1
            elif status == "fail":
                n_falhos += 1
        print(f"    Wold: concluido em {_time.time() - t0:.1f}s  "
              f"(valid={n_validos} failed={n_falhos})", flush=True)

    sims_arr = np.asarray(sims); r2s_arr = np.asarray(r2s); q2s_arr = np.asarray(q2s)
    # Add observed point (sim=1)
    sims_all = np.append(sims_arr, 1.0)
    r2_all   = np.append(r2s_arr, r2_obs)
    q2_all   = np.append(q2s_arr, q2_obs)

    # Filter non-finite values (numerical blow-up in degenerate permutations)
    fin_mask = np.isfinite(sims_all) & np.isfinite(r2_all) & np.isfinite(q2_all)
    sims_f = sims_all[fin_mask]; r2_f = r2_all[fin_mask]; q2_f = q2_all[fin_mask]

    if len(sims_f) >= 2 and (sims_f.max() - sims_f.min()) > 1e-8:
        slope_r2, int_r2 = np.polyfit(sims_f, r2_f, 1)
        slope_q2, int_q2 = np.polyfit(sims_f, q2_f, 1)
    else:
        slope_r2 = int_r2 = slope_q2 = int_q2 = float("nan")

    return {
        "sims":         sims_arr,
        "r2s":          r2s_arr,
        "q2s":          q2s_arr,
        "r2_obs":       float(r2_obs),
        "q2_obs":       float(q2_obs),
        "intercept_r2": float(int_r2),
        "intercept_q2": float(int_q2),
        "slope_r2":     float(slope_r2),
        "slope_q2":     float(slope_q2),
        "valid_r2":     bool(int_r2 < 0.40) if np.isfinite(int_r2) else False,
        "valid_q2":     bool(int_q2 < 0.05) if np.isfinite(int_q2) else False,
        "n_validos":    n_validos,
        "n_falhos":     n_falhos,
    }


def _iter_permutacao(pipeline_factory, X, Y_perm, y_perm_int, cv, groups):
    """Uma iteracao (pura, sem estado global) do teste de permutacao. Mesma
    logica de teste_permutacao pre-paralelizacao — usada sequencialmente ou
    via joblib.Parallel. Retorna ("ok", acc) | ("fail", None)."""
    try:
        cv_perm_idx = list(cv.split(X, y_perm_int, groups=groups))
        y_hat = _cv_predict_manual(pipeline_factory, X, Y_perm, cv_perm_idx)
        acc = float(balanced_accuracy_score(y_perm_int, np.argmax(y_hat, axis=1)))
        return ("ok", acc)
    except Exception:
        return ("fail", None)


def teste_permutacao(pipeline_factory: Callable[[], Pipeline],
                      X: np.ndarray, Y_bin: np.ndarray, y_int: np.ndarray,
                      cv, n_perm: int, seed: int,
                      groups: Optional[np.ndarray] = None,
                      n_jobs: int = 1
                      ) -> Dict[str, object]:
    """Robust Y-randomization. Iterations that fail (e.g., stratification
    impossible after shuffle) are recorded and excluded from the p-value.

    Returns dict with:
        acc_observada      - balanced_accuracy_score with true Y (consistent with
                             the main metric; unbiased for class imbalance)
        accs_permutadas    - array of H0 balanced accuracies (valid iterations only)
        p_value            - (sum(accs >= obs) + 1) / (n_validos + 1)
        n_validos          - iterations completed successfully
        n_falhos           - iterations aborted due to error
        failure_rate       - n_falhos / n_perm
    """
    rng = np.random.default_rng(seed)

    cv_indices = list(cv.split(X, y_int, groups=groups))
    y_hat = _cv_predict_manual(pipeline_factory, X, Y_bin, cv_indices)
    acc_obs = balanced_accuracy_score(y_int, np.argmax(y_hat, axis=1))

    accs: List[float] = []
    n_falhos = 0
    import time as _time
    t0 = _time.time()
    progress_every = max(1, n_perm // 20)

    # Mesma sequencia de permutacoes independente de n_jobs (reprodutibilidade
    # do seed identica, sequencial ou paralelo — so o tempo de parede muda).
    permutacoes = [rng.permutation(len(Y_bin)) for _ in range(n_perm)]

    if n_jobs <= 1:
        for i, idx in enumerate(permutacoes):
            Y_perm = Y_bin[idx]
            y_perm_int = np.argmax(Y_perm, axis=1)
            status, acc = _iter_permutacao(pipeline_factory, X, Y_perm,
                                            y_perm_int, cv, groups)
            if status == "ok":
                accs.append(acc)
            else:
                n_falhos += 1

            # Progress with ETA
            if (i + 1) % progress_every == 0 or (i + 1) == n_perm:
                elapsed = _time.time() - t0
                taxa    = (i + 1) / max(elapsed, 1e-6)
                eta_s   = (n_perm - i - 1) / max(taxa, 1e-6)
                pct = (i + 1) / n_perm * 100
                print(f"    Perm {i+1:4d}/{n_perm}  ({pct:5.1f}%)  "
                      f"valid={len(accs)} failed={n_falhos}  "
                      f"elapsed={elapsed:5.1f}s  ETA={eta_s:5.1f}s",
                      flush=True)
    else:
        from joblib import Parallel, delayed
        import threadpoolctl
        print(f"    Perm: {n_perm} permutacoes em paralelo (n_jobs={n_jobs})...",
              flush=True)
        # Ver comentario equivalente em teste_wold: threading nao acelera
        # (overhead Python do sklearn segura o GIL) — loky (processos) sim;
        # threadpool_limits(1) evita oversubscription do BLAS interno.
        with threadpoolctl.threadpool_limits(1):
            resultados = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_iter_permutacao)(
                    pipeline_factory, X, Y_bin[idx], np.argmax(Y_bin[idx], axis=1),
                    cv, groups)
                for idx in permutacoes)
        for status, acc in resultados:
            if status == "ok":
                accs.append(acc)
            else:
                n_falhos += 1
        print(f"    Perm: concluido em {_time.time() - t0:.1f}s  "
              f"(valid={len(accs)} failed={n_falhos})", flush=True)

    n_validos = len(accs)
    failure_rate = n_falhos / n_perm if n_perm > 0 else 0.0
    accs_arr = np.asarray(accs, dtype=float)

    if failure_rate > 0.30:
        print(f"[WARNING] Permutation test: failure rate = "
              f"{failure_rate:.1%} ({n_falhos}/{n_perm}). "
              f"Result may be unreliable (classes too "
              f"imbalanced for stratified CV after shuffle).")

    if n_validos == 0:
        print("[ERROR] Permutation test: 0 valid iterations. "
              "p_value returned as 1.0 (non-informative).")
        p_val = 1.0
    else:
        p_val = float((np.sum(accs_arr >= acc_obs) + 1) / (n_validos + 1))

    return {
        "acc_observada":   float(acc_obs),
        "accs_permutadas": accs_arr,
        "p_value":         p_val,
        "n_validos":       n_validos,
        "n_falhos":        n_falhos,
        "failure_rate":    failure_rate,
    }
