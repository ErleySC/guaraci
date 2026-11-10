"""Lote 3.4 da vistoria (achado A5-03 / V65): o Auto-Benchmark usa o splitter
estavel do projeto, n_splits pelo numero de GRUPOS da menor classe, roda sem
grupos, aplica Holm aos p de Nadeau-Bengio e so' traz Wilcoxon com 6+ dobras."""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer

matplotlib.use("Agg")

import guaraci.validacao_estatistica as ve  # noqa: E402
from guaraci import avaliacao_modelos as am  # noqa: E402
from guaraci.config import NOME_TABELAS, Config  # noqa: E402


def _cenario_s7():
    """Cenario s7 de A5: classe 0 com 2 grupos x 3 replicas, classe 1 com 5
    grupos x 3 replicas. Pelo numero de amostras (6) o codigo anterior pedia 5
    dobras e 3 delas ficavam sem a classe 0."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 6 + [1] * 15)
    g = np.repeat([f"g{i}" for i in range(7)], 3)
    X = rng.normal(size=(len(y), 10)) + y[:, None] * 3.0
    return X, y, g, LabelBinarizer().fit(y)


def _preparar(monkeypatch, tmp_path, **cfg_kw):
    cfg = Config(**{**dict(n_splits_cv=5, seed=0, run_shap=False, default_preprocessing="mc"), **cfg_kw})
    os.makedirs(os.path.join(str(tmp_path), NOME_TABELAS), exist_ok=True)
    monkeypatch.setattr(am, "build_benchmark_list", lambda n_opt, cfg, incluir_opcionais=True: [
        ("PLS-DA", am.PLSDAClassifier(n_components=2)),
        ("Logistica", LogisticRegression(max_iter=200)),
        ("Logistica C", LogisticRegression(C=0.01, max_iter=200)),
    ])
    for nome in ("fig_benchmark_classifiers", "fig_det_curvas"):
        monkeypatch.setattr(am, nome, lambda *a, **k: None)
    return cfg, str(tmp_path)


def _espiar(monkeypatch):
    capturados: list = []
    real = ve.StableStratifiedGroupKFold

    class _Espiao(real):
        def split(self, X, y, groups=None):
            for tr, te in super().split(X, y, groups):
                capturados.append((np.asarray(tr), np.asarray(te)))
                yield tr, te
    monkeypatch.setattr(ve, "StableStratifiedGroupKFold", _Espiao)
    return capturados


def test_s7_todas_as_dobras_de_validacao_tem_todas_as_classes(monkeypatch, tmp_path):
    X, y, g, lb = _cenario_s7()
    cfg, pasta = _preparar(monkeypatch, tmp_path)
    caps = _espiar(monkeypatch)
    am.benchmark_classifiers(X, y, g, lb, 2, cfg, pasta)
    assert caps, "o splitter estavel nao foi usado"
    # n_splits pelos grupos da menor classe (2), nao por amostras (5)
    assert len(caps) % 2 == 0 and len(caps) // 2 >= 1
    for tr, te in caps:
        assert set(y[te]) == {0, 1}, "dobra de validacao sem uma das classes"
        assert set(y[tr]) == {0, 1}
        assert not (set(g[tr]) & set(g[te])), "grupo vazou entre treino e validacao"


def test_benchmark_roda_sem_grupos(monkeypatch, tmp_path):
    X, y, g, lb = _cenario_s7()
    cfg, pasta = _preparar(monkeypatch, tmp_path)
    caps = _espiar(monkeypatch)
    df = am.benchmark_classifiers(X, y, None, lb, 2, cfg, pasta)
    assert len(df) == 3 and not caps          # StratifiedKFold comum, sem o splitter de grupos
    assert df["Bal.Acc mean"].between(0, 1).all()


def test_particao_do_benchmark_e_a_do_splitter_estavel(monkeypatch, tmp_path):
    """Mesma (y, groups, n_splits, seed) => mesma particao: e' a garantia do
    splitter estavel, independente da versao do scikit-learn."""
    X, y, g, lb = _cenario_s7()
    cfg, pasta = _preparar(monkeypatch, tmp_path)
    caps = _espiar(monkeypatch)
    am.benchmark_classifiers(X, y, g, lb, 2, cfg, pasta)
    esperado = list(ve.StableStratifiedGroupKFold.__mro__[1](n_splits=2, seed=0).split(X, y, g))
    n = len(esperado)
    assert len(caps) >= n == 2
    for (tr, te), (etr, ete) in zip(caps[:n], esperado):
        assert np.array_equal(te, ete) and np.array_equal(tr, etr)


def test_holm_valores_conhecidos():
    aj = am._ajustar_holm(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(aj, [0.03, 0.06, 0.06])
    assert am._ajustar_holm(np.array([])).size == 0
    assert np.allclose(am._ajustar_holm(np.array([0.5, 0.9])), [1.0, 1.0])


def test_coluna_holm_e_o_ajuste_dos_p_de_nadeau_bengio(monkeypatch, tmp_path):
    X, y, g, lb = _cenario_s7()
    # mais grupos por classe para ter 3 dobras e variancia nas diferencas
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 24)
    g = np.repeat([f"h{i}" for i in range(16)], 3)
    X = rng.normal(size=(48, 10)) + y[:, None] * 0.8
    lb = LabelBinarizer().fit(y)
    cfg, pasta = _preparar(monkeypatch, tmp_path, n_splits_cv=4)
    df = am.benchmark_classifiers(X, y, g, lb, 2, cfg, pasta).set_index("Classifier")
    col_nb = "p t corrigido (Nadeau-Bengio, vs PLS-DA)"
    col_h = "p Holm (Nadeau-Bengio, vs PLS-DA)"
    assert df.loc["PLS-DA", col_h] == "-"
    nomes = [n for n in df.index if isinstance(df.loc[n, col_nb], float)]
    brutos = np.array([df.loc[n, col_nb] for n in nomes])
    esperados = am._ajustar_holm(brutos)
    for n, e in zip(nomes, esperados):
        assert df.loc[n, col_h] == pytest.approx(round(float(e), 4), abs=2e-4)
        assert df.loc[n, col_h] >= df.loc[n, col_nb]


def test_wilcoxon_so_aparece_com_seis_ou_mais_dobras(monkeypatch, tmp_path):
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1], 36)
    g = np.repeat([f"k{i}" for i in range(24)], 3)
    X = rng.normal(size=(72, 10)) + y[:, None] * 0.8
    lb = LabelBinarizer().fit(y)
    cfg, pasta = _preparar(monkeypatch, tmp_path, n_splits_cv=5)
    df5 = am.benchmark_classifiers(X, y, g, lb, 2, cfg, pasta)
    assert "p Wilcoxon (vs PLS-DA)" not in df5.columns
    cfg6, pasta = _preparar(monkeypatch, tmp_path, n_splits_cv=6)
    df6 = am.benchmark_classifiers(X, y, g, lb, 2, cfg6, pasta)
    assert "p Wilcoxon (vs PLS-DA)" in df6.columns
