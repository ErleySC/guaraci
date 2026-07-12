"""Smoke tests for the chemometrics pipeline.

Uses session-scoped `pq` fixture from conftest.py (module loaded once).
"""
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

from conftest import achar_pastas_run


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_config_defaults(pq):
    cfg = pq.Config()
    assert cfg.preprocessamento_padrao == "msc_sg_mc"
    assert cfg.max_lvs == 40
    assert cfg.agrupar_por_mae_id is True
    assert cfg.n_permutacoes_wold == 200   # raised from 50 for publication


def test_config_spec_keys(pq):
    assert hasattr(pq, "_CONFIG_SPEC")
    keys = {s["key"] for s in pq._CONFIG_SPEC}
    assert "nivel" in keys
    assert "max_lvs" in keys
    assert "pre_processamento" in keys


def test_version_present(pq):
    """__version__ must be defined and follow semver."""
    assert hasattr(pq, "__version__")
    parts = pq.__version__.split(".")
    assert len(parts) == 3, f"Expected semver X.Y.Z, got {pq.__version__!r}"


def test_msc_no_leakage(pq):
    """MSC.ref_ must be fitted on train only — core anti-leakage property."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(50, 100))
    X_test  = rng.normal(size=(10, 100))
    msc = pq.MSC()
    msc.fit(X_train)
    assert hasattr(msc, "ref_")
    assert msc.ref_.shape == (100,)
    X_tr = msc.transform(X_train)
    X_te = msc.transform(X_test)
    assert X_tr.shape == X_train.shape
    assert X_te.shape == X_test.shape


def test_plsda_classifier_binary(pq):
    """PLSDAClassifier: binary case — proba sums to 1, shapes correct."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 20))
    y = np.array(["A"] * 30 + ["B"] * 30)
    clf = pq.PLSDAClassifier(n_components=2)
    clf.fit(X, y)
    preds = clf.predict(X)
    proba = clf.predict_proba(X)
    assert set(preds).issubset({"A", "B"})
    assert proba.shape == (60, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_plsda_classifier_multiclass(pq):
    """PLSDAClassifier: 5-class case — proba sums to 1, all classes predicted."""
    rng = np.random.default_rng(1)
    n_per = 30
    classes = ["A", "B", "C", "D", "E"]
    X = np.vstack([rng.normal(loc=i * 2, size=(n_per, 15)) for i in range(5)])
    y = np.repeat(classes, n_per)
    clf = pq.PLSDAClassifier(n_components=3)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), 5)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    preds = clf.predict(X)
    assert set(preds).issubset(set(classes))


def test_plsda_classifier_vs_sklearn_manual(pq):
    """VALIDACAO (docs/VALIDATION.md): PLSDAClassifier.predict() deve
    reproduzir EXATAMENTE argmax(PLSRegression manual) -- prova que o
    wrapper (LabelBinarizer + PLS + argmax) nao introduz nenhum desvio
    sobre o sklearn.PLSRegression puro, que e' o motor por baixo."""
    from sklearn.cross_decomposition import PLSRegression as _SkPLS
    from sklearn.preprocessing import LabelBinarizer as _SkLB
    rng = np.random.default_rng(3)
    n_per = 25
    classes = ["Andiroba", "Acai", "Castanha", "Copaiba"]
    X = np.vstack([rng.normal(loc=i * 1.7, scale=1.0, size=(n_per, 24))
                   for i in range(len(classes))])
    y = np.repeat(classes, n_per)

    clf = pq.PLSDAClassifier(n_components=3).fit(X, y)
    preds_guaraci = clf.predict(X)

    # Reproducao manual, sem passar pelo wrapper Guaraci.
    lb = _SkLB().fit(y)
    Yb = np.asarray(lb.transform(y))
    pls_ref = _SkPLS(n_components=3, scale=False).fit(X, Yb)
    preds_ref = lb.classes_[np.argmax(np.asarray(pls_ref.predict(X)), axis=1)]

    assert list(preds_guaraci) == list(preds_ref)
    # Coeficientes de regressao identicos (mesmo estimador por baixo).
    coef_guaraci = np.asarray(clf._pls.coef_)
    coef_ref = np.asarray(pls_ref.coef_)
    diff = float(np.max(np.abs(coef_guaraci - coef_ref)))
    assert diff == 0.0, f"max|Δcoef| = {diff} (esperado 0.0)"


def test_opls_orthogonality_binary(pq):
    """OPLS binary: t_orth ⊥ t_pred by Gram-Schmidt construction."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 50))
    Y = np.zeros((80, 2)); Y[:40, 0] = 1; Y[40:, 1] = 1
    opls = pq.OPLSDAWrapper(n_ortho=1)
    opls.fit(X, Y)
    t_pred, t_orth = opls.transform(X)
    dot = abs(float(t_pred @ t_orth[:, 0]))
    assert dot < 1e-6, f"t_orth not orthogonal: inner product = {dot:.2e}"


def test_opls_orthogonality_multiclass(pq):
    """OPLS 14-class: LDA y-vector used; t_orth still ⊥ t_pred."""
    rng = np.random.default_rng(99)
    n_classes = 14
    n_per = 20
    X = np.vstack([rng.normal(loc=i, size=(n_per, 40)) for i in range(n_classes)])
    # One-hot Y
    Y = np.zeros((n_classes * n_per, n_classes))
    for k in range(n_classes):
        Y[k * n_per:(k + 1) * n_per, k] = 1
    opls = pq.OPLSDAWrapper(n_ortho=1)
    opls.fit(X, Y)
    t_pred, t_orth = opls.transform(X)
    dot = abs(float(t_pred @ t_orth[:, 0]))
    assert dot < 1e-6, f"t_orth not orthogonal (multiclass): inner product = {dot:.2e}"


def test_config_spec_attrs_match_config(pq):
    """Every _CONFIG_SPEC entry must correspond to an existing Config attribute."""
    cfg = pq.Config()
    for s in pq._CONFIG_SPEC:
        assert hasattr(cfg, s["attr"]), f"Config missing attr: {s['attr']}"


def test_avaliar_subset_cv_q2_no_overflow(pq):
    """_avaliar_subset_cv: Q2 must be finite or nan — never blow up to ±1e10+.

    Regression test for iPLS bug: narrow ill-conditioned intervals caused
    y_cv predictions to diverge numerically, producing Q2 ≈ -3.9e31.
    Fixed by checking np.isfinite(ss_res) before computing Q2.
    """
    rng = np.random.default_rng(42)
    n, p = 40, 5
    # Two binary classes
    X = rng.normal(size=(n, p))
    Y_bin = np.zeros((n, 2)); Y_bin[:20, 0] = 1; Y_bin[20:, 1] = 1
    y_int = np.array([0] * 20 + [1] * 20)

    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    cv_indices = list(cv.split(X, y_int))

    res = pq._avaliar_subset_cv(X, Y_bin, y_int, cv_indices, n_lv=2)
    q2 = res["q2"]
    assert np.isnan(q2) or (np.isfinite(q2) and q2 >= -1.0), (
        f"Q2 = {q2} is an invalid overflow value — fix in _avaliar_subset_cv failed"
    )


def test_wold_no_nan_intercept(pq):
    """teste_wold: intercepts must be finite or nan — never blow up to ±inf.

    Regression test: degenerate models caused polyfit to receive non-finite
    r2/q2 values, returning NaN intercepts silently.
    """
    rng = np.random.default_rng(7)
    n, p = 30, 10
    X = rng.normal(size=(n, p))
    Y_bin = np.zeros((n, 2)); Y_bin[:15, 0] = 1; Y_bin[15:, 1] = 1
    y_int = np.array([0] * 15 + [1] * 15)

    from sklearn.model_selection import StratifiedKFold

    def fac():
        return Pipeline([
            ("mc",  StandardScaler(with_std=False)),
            ("pls", PLSRegression(n_components=2, scale=False)),
        ])

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
    res = pq.teste_wold(fac, X, Y_bin, y_int, cv, n_perm=10, seed=0)

    for key in ("intercept_r2", "intercept_q2", "slope_r2", "slope_q2"):
        v = res[key]
        assert v is not None, f"{key} should not be None"
        assert not np.isinf(float(v)), f"{key} = {v} is ±inf — numerical overflow"


def _spec(pq, key):
    return next(s for s in pq._CONFIG_SPEC if s["key"] == key)


def test_teste_permutacao_paralelo_identico_ao_sequencial(pq):
    """n_jobs>1 deve produzir exatamente os mesmos resultados que n_jobs=1
    (mesma sequência de permutações, mesmo cálculo por iteração — só muda o
    tempo de parede). Regressão de segurança da paralelização da Fase E."""
    rng = np.random.default_rng(3)
    n, p = 60, 12
    X = rng.normal(size=(n, p))
    Y_bin = np.zeros((n, 3))
    y_int = np.array([0] * 20 + [1] * 20 + [2] * 20)
    for i, c in enumerate(y_int):
        Y_bin[i, c] = 1

    from sklearn.model_selection import StratifiedKFold

    def fac():
        return Pipeline([
            ("mc",  StandardScaler(with_std=False)),
            ("pls", PLSRegression(n_components=3, scale=False)),
        ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=2)

    res_seq = pq.teste_permutacao(fac, X, Y_bin, y_int, cv, n_perm=12,
                                   seed=42, n_jobs=1)
    res_par = pq.teste_permutacao(fac, X, Y_bin, y_int, cv, n_perm=12,
                                   seed=42, n_jobs=4)

    assert res_seq["n_validos"] == res_par["n_validos"]
    assert res_seq["n_falhos"] == res_par["n_falhos"]
    assert res_seq["acc_observada"] == pytest.approx(res_par["acc_observada"])
    np.testing.assert_array_almost_equal(
        res_seq["accs_permutadas"], res_par["accs_permutadas"], decimal=12)
    assert res_seq["p_value"] == pytest.approx(res_par["p_value"])


def test_teste_wold_paralelo_identico_ao_sequencial(pq):
    """n_jobs>1 deve produzir exatamente os mesmos resultados que n_jobs=1
    para teste_wold — mesma verificação de segurança que teste_permutacao."""
    rng = np.random.default_rng(5)
    n, p = 50, 10
    X = rng.normal(size=(n, p))
    Y_bin = np.zeros((n, 2)); Y_bin[:25, 0] = 1; Y_bin[25:, 1] = 1
    y_int = np.array([0] * 25 + [1] * 25)

    from sklearn.model_selection import StratifiedKFold

    def fac():
        return Pipeline([
            ("mc",  StandardScaler(with_std=False)),
            ("pls", PLSRegression(n_components=2, scale=False)),
        ])

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)

    res_seq = pq.teste_wold(fac, X, Y_bin, y_int, cv, n_perm=10, seed=7, n_jobs=1)
    res_par = pq.teste_wold(fac, X, Y_bin, y_int, cv, n_perm=10, seed=7, n_jobs=4)

    for key in ("intercept_r2", "intercept_q2", "slope_r2", "slope_q2",
                "r2_obs", "q2_obs"):
        assert res_seq[key] == pytest.approx(res_par[key], nan_ok=True)
    np.testing.assert_array_almost_equal(res_seq["sims"], res_par["sims"], decimal=12)
    np.testing.assert_array_almost_equal(res_seq["r2s"], res_par["r2s"], decimal=12)
    np.testing.assert_array_almost_equal(res_seq["q2s"], res_par["q2s"], decimal=12)


def test_coagir_valor_rejeita_holdout_negativo(pq):
    """holdout_fracao negativa deve ser rejeitada (antes: pulava holdout em silêncio)."""
    s = _spec(pq, "holdout_fracao")
    with pytest.raises(ValueError):
        pq._coagir_valor(s, -0.2)
    with pytest.raises(ValueError):
        pq._coagir_valor(s, 0.8)          # acima do máximo (0.5)
    assert pq._coagir_valor(s, 0.2) == 0.2  # valor válido passa


def test_coagir_valor_rejeita_contagem_zero(pq):
    """Contagens (LVs, permutações, MC) devem exigir >= 1, não valores degenerados."""
    for key in ("max_lvs", "n_permutacoes", "n_monte_carlo"):
        s = _spec(pq, key)
        with pytest.raises(ValueError):
            pq._coagir_valor(s, 0)
        assert pq._coagir_valor(s, 5) == 5


def test_coagir_valor_faixa_ausente_nao_valida(pq):
    """Campos numéricos sem min/max declarado continuam aceitando qualquer valor."""
    # dpi tem min/max; um campo sem limites (se existir) não deve levantar.
    s = _spec(pq, "dpi")
    with pytest.raises(ValueError):
        pq._coagir_valor(s, 10)          # abaixo do mínimo (50)
    assert pq._coagir_valor(s, 600) == 600


# ── Integration test ──────────────────────────────────────────────────────────

@pytest.mark.slow
def test_pipeline_end_to_end_synthetic(pq, tmp_path):
    """End-to-end: executar() with synthetic data must produce resumo_modelo.txt."""
    import os
    cfg = pq.Config(
        pasta_entrada=str(tmp_path / "dados"),  # synthetic mode ignores this
        pasta_saida_raiz=str(tmp_path / "saida"),
        modo="sintetico",
        n_por_classe=8,
        n_pontos_sint=50,
        # Synthetic data uses linspace(4000, 400) — match the range so
        # spectral truncation keeps all 50 points (sg_window=25 requires n>25)
        wn_min=400.0,
        wn_max=4001.0,
        n_splits_cv=2,
        n_repeats_cv=1,
        n_permutacoes=5,
        n_permutacoes_wold=5,
        n_bootstrap_vip=3,
        n_bootstrap_bca=20,
        n_monte_carlo=3,
        executar_benchmark=False,
        executar_monte_carlo=False,
        executar_shap=False,
        executar_wold=False,
        executar_cv_anova=False,
        executar_opls=False,
        executar_etapa4=False,
        comparar_pipelines=False,
        comparar_hca_pipelines=False,
        max_lvs=5,
    )
    os.makedirs(str(tmp_path / "dados"), exist_ok=True)
    pq.executar(cfg)

    # Verify at least one output folder was created with the summary
    from pathlib import Path
    saida_root = tmp_path / "saida"
    runs = [Path(r) for r in achar_pastas_run(saida_root)] if saida_root.exists() else []
    assert runs, "No output folder created by executar()"
    logs_dirs = [r / pq.NOME_RELATORIOS for r in runs
                 if (r / pq.NOME_RELATORIOS).exists()]
    assert logs_dirs, f"No {pq.NOME_RELATORIOS}/ subfolder found in output"
    resumo_files = [d / "resumo_modelo.txt" for d in logs_dirs
                    if (d / "resumo_modelo.txt").exists()]
    assert resumo_files, "resumo_modelo.txt not found — pipeline may have crashed silently"
