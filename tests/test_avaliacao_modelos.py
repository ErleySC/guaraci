"""Testes de avaliacao_modelos.py: Auto-Benchmark, Monte Carlo CV, curvas DET
e SHAP (PLSDAClassifier em si já é coberto por test_pipeline_smoke.py).

Marcados 'slow' — treinam SVM/RF/GBM/XGBoost de verdade e rodam SHAP
TreeExplainer. Tanto `xgboost` quanto `shap` estão instalados neste projeto
(dependência opcional 'benchmark' do pyproject.toml), então os testes
exercitam os caminhos REAIS, não mocks — era exatamente essa parte do
código (11% de cobertura) que nunca rodava na suíte antes.
"""
import glob
import os

import numpy as np
import pytest
from sklearn.preprocessing import LabelBinarizer


def _dados_benchmark(seed=0, n_por_classe=15, p=25, n_classes=3):
    """Dados sintéticos com grupos (estilo mae_id) para exercitar CV
    group-aware — classes bem separadas para os classificadores convergirem
    rápido (o objetivo é cobertura de código, não desempenho)."""
    rng = np.random.default_rng(seed)
    X_list, y_list, grp_list = [], [], []
    for c in range(n_classes):
        centro = rng.normal(loc=c * 4.0, size=p)
        for i in range(n_por_classe):
            X_list.append(centro + rng.normal(scale=1.0, size=p))
            y_list.append(c)
            grp_list.append(f"grupo_{c}_{i}")
    X = np.array(X_list)
    y_int = np.array(y_list)
    grupos = np.array(grp_list)
    lb = LabelBinarizer().fit(y_int)
    return X, y_int, grupos, lb


@pytest.mark.slow
def test_benchmark_classificadores_roda_e_gera_saidas(pq, tmp_path):
    """Auto-Benchmark: PLS-DA/SVM/RF/GBM/XGBoost sob a mesma CV group-aware.
    Verifica DataFrame + CSV + figura de boxplot + curvas DET (>=2
    classificadores válidos)."""
    X, y_int, grupos, lb = _dados_benchmark()
    cfg = pq.Config(n_splits_cv=3, seed=0, executar_shap=False)
    pasta = str(tmp_path)
    os.makedirs(os.path.join(pasta, "dados"), exist_ok=True)

    df = pq.benchmark_classificadores(X, y_int, grupos, lb, n_opt=2,
                                       cfg=cfg, pasta=pasta)

    nomes = set(df["Classificador"].values)
    assert "PLS-DA" in nomes
    assert "XGBoost" in nomes  # confirma que o import opcional funcionou de verdade
    assert os.path.exists(
        os.path.join(pasta, "dados", "benchmark_classificadores.csv"))
    assert os.path.exists(
        os.path.join(pasta, "figuras", "fig_benchmark_classificadores.png"))
    assert glob.glob(os.path.join(pasta, "figuras", "fig_det_curvas*.png"))
    for v in df["Bal.Acc media"]:
        assert 0.0 <= v <= 1.0


@pytest.mark.slow
def test_monte_carlo_cv_apenas_plsda(pq, tmp_path):
    """Monte Carlo CV no modo padrão (só PLS-DA, monte_carlo_incluir_todos=False):
    roda rápido, gera CSV + DataFrame com IC95%."""
    X, y_int, grupos, lb = _dados_benchmark(seed=1)
    cfg = pq.Config(n_monte_carlo=8, monte_carlo_test_size=0.3,
                     monte_carlo_incluir_todos=False, seed=1)
    pasta = str(tmp_path)
    os.makedirs(os.path.join(pasta, "dados"), exist_ok=True)

    df = pq.monte_carlo_cv(X, y_int, grupos, lb, n_opt=2, cfg=cfg, pasta=pasta)

    assert list(df["Classificador"]) == ["PLS-DA"]
    assert df["Iteracoes validas"].iloc[0] > 0
    assert 0.0 <= df["IC95% inf"].iloc[0] <= df["IC95% sup"].iloc[0] <= 1.0
    assert os.path.exists(os.path.join(pasta, "dados", "monte_carlo_cv.csv"))


@pytest.mark.slow
def test_monte_carlo_cv_todos_os_modelos(pq, tmp_path):
    """Monte Carlo CV com monte_carlo_incluir_todos=True: roda PLS-DA + SVM +
    RF + GBM + XGBoost, com >= 5 iterações válidas cada (gate para a figura
    violino, que exige exatamente esse mínimo)."""
    X, y_int, grupos, lb = _dados_benchmark(seed=2, n_por_classe=20)
    cfg = pq.Config(n_monte_carlo=6, monte_carlo_test_size=0.3,
                     monte_carlo_incluir_todos=True, seed=2)
    pasta = str(tmp_path)
    os.makedirs(os.path.join(pasta, "dados"), exist_ok=True)

    df = pq.monte_carlo_cv(X, y_int, grupos, lb, n_opt=2, cfg=cfg, pasta=pasta)

    nomes = set(df["Classificador"])
    assert {"PLS-DA", "SVM RBF", "Random Forest", "Grad. Boost.", "XGBoost"} <= nomes


@pytest.mark.slow
def test_fig_shap_benchmark_gera_figura(pq, tmp_path):
    """SHAP TreeExplainer (RF, já que 3 classes > 2 desabilita GBM/SHAP
    multiclasse por limitação do próprio SHAP): não crasha, gera ao menos
    uma figura fig_shap_*."""
    # p=30: preprocessamento padrão (msc_sg_mc) usa Savitzky-Golay com janela
    # default 25 — precisa n_variaveis > sg_window.
    X, y_int, _grupos, _lb = _dados_benchmark(seed=3, p=30)
    cfg = pq.Config(shap_max_amostras=100, seed=3)
    pasta = str(tmp_path)
    wavenumbers = np.linspace(4000, 400, X.shape[1])

    pq.fig_shap_benchmark(X, y_int, n_opt=2, cfg=cfg, pasta=pasta,
                           wavenumbers=wavenumbers)

    assert glob.glob(os.path.join(pasta, "figuras", "fig_shap_*.png"))
