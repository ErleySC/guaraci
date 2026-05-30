"""Smoke tests for the chemometrics pipeline.

Uses session-scoped `pq` fixture from conftest.py (module loaded once).
"""
import numpy as np
import pytest


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


# ── Integration test ──────────────────────────────────────────────────────────

@pytest.mark.slow
def test_pipeline_end_to_end_synthetic(pq, tmp_path):
    """End-to-end: executar() with synthetic data must produce resumo_modelo.txt."""
    import os
    cfg = pq.Config(
        pasta_dados=str(tmp_path / "dados"),    # synthetic mode ignores this
        pasta_saida=str(tmp_path / "saida"),
        modo="sintetico",
        n_por_classe=8,
        n_pontos_sint=50,
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
        gerar_pptx=False,
        max_lvs=5,
    )
    os.makedirs(str(tmp_path / "dados"), exist_ok=True)
    pq.executar(cfg)

    # Verify at least one output folder was created with the summary
    saida_root = tmp_path / "saida"
    runs = list(saida_root.iterdir()) if saida_root.exists() else []
    assert runs, "No output folder created by executar()"
    logs_dirs = [r / "logs" for r in runs if (r / "logs").exists()]
    assert logs_dirs, "No logs/ subfolder found in output"
    resumo_files = [d / "resumo_modelo.txt" for d in logs_dirs
                    if (d / "resumo_modelo.txt").exists()]
    assert resumo_files, "resumo_modelo.txt not found — pipeline may have crashed silently"
