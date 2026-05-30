"""Smoke test for the chemometrics pipeline using synthetic data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

def _load_pipeline():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pq", os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "pineline_quimiometria_14.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_config_defaults():
    pq = _load_pipeline()
    cfg = pq.Config()
    assert cfg.preprocessamento_padrao == "msc_sg_mc"
    assert cfg.max_lvs == 40
    assert cfg.agrupar_por_mae_id is True

def test_config_spec_keys():
    pq = _load_pipeline()
    assert hasattr(pq, "_CONFIG_SPEC")
    keys = {s["key"] for s in pq._CONFIG_SPEC}
    assert "nivel" in keys
    assert "max_lvs" in keys
    assert "pre_processamento" in keys

def test_msc_no_leakage():
    """MSC.ref_ must be computed from train only (anti-leakage)."""
    pq = _load_pipeline()
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(50, 100))
    X_test  = rng.normal(size=(10, 100))
    msc = pq.MSC()
    msc.fit(X_train)
    # ref_ (mean training spectrum) set from train only
    assert hasattr(msc, "ref_")
    assert msc.ref_.shape == (100,)
    X_tr = msc.transform(X_train)
    X_te = msc.transform(X_test)
    assert X_tr.shape == X_train.shape
    assert X_te.shape == X_test.shape

def test_plsda_classifier_binary():
    pq = _load_pipeline()
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

def test_opls_orthogonality():
    """t_orth must be strictly orthogonal to t_pred (Gram-Schmidt guarantee)."""
    pq = _load_pipeline()
    rng = np.random.default_rng(7)
    n, p = 80, 50
    X = rng.normal(size=(n, p))
    # Binary response (2 classes)
    Y = np.zeros((n, 2))
    Y[:40, 0] = 1; Y[40:, 1] = 1
    opls = pq.OPLSDAWrapper(n_ortho=1)
    opls.fit(X, Y)
    t_pred, t_orth = opls.transform(X)
    # Inner product must be < 1e-9 (machine precision)
    dot = abs(float(t_pred @ t_orth[:, 0]))
    assert dot < 1e-6, f"t_orth not orthogonal to t_pred: inner product = {dot:.2e}"

def test_config_spec_attrs_match_config():
    pq = _load_pipeline()
    cfg = pq.Config()
    for s in pq._CONFIG_SPEC:
        assert hasattr(cfg, s["attr"]), f"Config missing attr: {s['attr']}"
