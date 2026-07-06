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

from conftest import achar_pastas_run


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
    os.makedirs(os.path.join(pasta, pq.NOME_TABELAS), exist_ok=True)

    df = pq.benchmark_classificadores(X, y_int, grupos, lb, n_opt=2,
                                       cfg=cfg, pasta=pasta)

    nomes = set(df["Classificador"].values)
    assert "PLS-DA" in nomes
    assert "XGBoost" in nomes  # confirma que o import opcional funcionou de verdade
    assert os.path.exists(
        os.path.join(pasta, pq.NOME_TABELAS, "benchmark_classificadores.csv"))
    assert os.path.exists(
        os.path.join(pasta, pq.NOME_GRAFICOS, "fig_benchmark_classificadores.png"))
    assert glob.glob(os.path.join(pasta, pq.NOME_GRAFICOS, "fig_det_curvas*.png"))
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
    os.makedirs(os.path.join(pasta, pq.NOME_TABELAS), exist_ok=True)

    df = pq.monte_carlo_cv(X, y_int, grupos, lb, n_opt=2, cfg=cfg, pasta=pasta)

    assert list(df["Classificador"]) == ["PLS-DA"]
    assert df["Iteracoes validas"].iloc[0] > 0
    assert 0.0 <= df["IC95% inf"].iloc[0] <= df["IC95% sup"].iloc[0] <= 1.0
    assert os.path.exists(os.path.join(pasta, pq.NOME_TABELAS, "monte_carlo_cv.csv"))


@pytest.mark.slow
def test_monte_carlo_cv_todos_os_modelos(pq, tmp_path):
    """Monte Carlo CV com monte_carlo_incluir_todos=True: roda PLS-DA + SVM +
    RF + GBM + XGBoost, com >= 5 iterações válidas cada (gate para a figura
    violino, que exige exatamente esse mínimo)."""
    X, y_int, grupos, lb = _dados_benchmark(seed=2, n_por_classe=20)
    cfg = pq.Config(n_monte_carlo=6, monte_carlo_test_size=0.3,
                     monte_carlo_incluir_todos=True, seed=2)
    pasta = str(tmp_path)
    os.makedirs(os.path.join(pasta, pq.NOME_TABELAS), exist_ok=True)

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

    assert glob.glob(os.path.join(pasta, pq.NOME_GRAFICOS, "fig_shap_*.png"))


# ── Auto-Benchmark de regressao (Ridge/Lasso/EN/SVR/RF vs PLS-R) ───────────

def _dados_regressao_multi_especie(seed=0, n_por_especie=24, p=30,
                                   n_especies=3, n_replicas=3):
    """Dados sinteticos multi-especie com replicas fisicas (mae_id) e teor
    de adulterante correlacionado ao espectro (mesmo estilo de
    gerar_dados_sinteticos, mas construido diretamente p/ o teste)."""
    rng = np.random.default_rng(seed)
    X_list, conc_list, rot_list, mae_list = [], [], [], []
    especies = [f"Esp_{chr(65+i)}" for i in range(n_especies)]
    for e_idx, especie in enumerate(especies):
        w_true = rng.normal(size=p)
        centro = rng.normal(loc=e_idx * 6.0, size=p)
        n_pontos = n_por_especie // n_replicas
        concs = np.linspace(0, 40, n_pontos)
        for p_idx, c in enumerate(concs):
            espectro_base = centro + w_true * (c / 40.0) * 3.0
            grupo_id = f"{especie[:3].upper()}{p_idx:02d}"
            for r in range(n_replicas):
                X_list.append(espectro_base + rng.normal(scale=0.15, size=p))
                conc_list.append(c)
                rot_list.append(especie)
                mae_list.append(grupo_id)
    X = np.array(X_list)
    conc = np.array(conc_list)
    rotulos = np.array(rot_list)
    mae_id = np.array(mae_list)
    classes_unicas = np.array(especies)
    return X, conc, rotulos, mae_id, classes_unicas


@pytest.mark.slow
def test_benchmark_regressao_roda_e_gera_saidas(pq, tmp_path):
    """Auto-Benchmark de regressao: PLS-R (reaproveitado de
    pls_regressao_por_especie, sem refit) + Ridge/Lasso/EN/SVR/RF, mesmo
    split por especie. Verifica DataFrame + CSV + figura."""
    X, conc, rotulos, mae_id, classes_unicas = _dados_regressao_multi_especie()
    cfg = pq.Config(seed=0, max_lvs=5, frac_cal=0.7)
    pasta = str(tmp_path)
    os.makedirs(os.path.join(pasta, pq.NOME_TABELAS), exist_ok=True)
    os.makedirs(os.path.join(pasta, pq.NOME_GRAFICOS), exist_ok=True)

    reg_esp = pq.pls_regressao_por_especie(
        X, conc, rotulos, mae_id, classes_unicas, cfg, pasta, n_splits=3)
    assert reg_esp is not None, "fixture nao gerou dados suficientes p/ PLS-R"

    df = pq.benchmark_regressao_por_especie(
        X, conc, rotulos, mae_id, classes_unicas, cfg, pasta, reg_esp)

    assert df is not None
    modelos_esperados = {"PLS-R", "Ridge", "Lasso", "Elastic Net",
                         "SVR (RBF)", "Random Forest"}
    assert modelos_esperados.issubset(set(df["Modelo"]))
    assert (df["RMSEP (pooled)"] >= 0).all()
    assert os.path.exists(
        os.path.join(pasta, pq.NOME_TABELAS, "benchmark_regressao.csv"))
    assert os.path.exists(
        os.path.join(pasta, pq.NOME_GRAFICOS, "fig_benchmark_regressores.png"))

    # PLS-R do benchmark bate com o ja calculado por pls_regressao_por_especie
    # (reaproveitado, nao deve ser refeito com numeros diferentes)
    linha_pls = df[df["Modelo"] == "PLS-R"].iloc[0]
    assert linha_pls["RMSEP (pooled)"] == pytest.approx(
        round(float(reg_esp["rmsep"]), 3))


def test_benchmark_regressao_sem_especies_suficientes_retorna_none(pq, tmp_path):
    """Sem nenhuma especie com amostras adulteradas suficientes, retorna
    None (mesmo criterio de pls_regressao_por_especie) em vez de crashar."""
    rng = np.random.default_rng(9)
    X = rng.normal(size=(10, 15))
    conc = np.zeros(10)          # nenhuma amostra adulterada
    rotulos = np.array(["Esp_A"] * 10)
    classes_unicas = np.array(["Esp_A"])
    cfg = pq.Config(seed=9)
    reg_esp_fake = {"rmsep": 0.0, "r2v": 0.0, "n_especies": 0,
                    "tabela_especie": []}

    df = pq.benchmark_regressao_por_especie(
        X, conc, rotulos, None, classes_unicas, cfg, str(tmp_path),
        reg_esp_fake)
    assert df is None


@pytest.mark.slow
def test_regressao_pooled_com_benchmark_ligado_roda_sem_erro(pq, tmp_path):
    """Integracao real: executar() em N3 sintetico com
    executar_benchmark_regressao=True gera o CSV/figura do benchmark de
    regressao junto com o restante do pipeline, sem quebrar nada."""
    cfg = pq.Config(
        pasta_entrada=str(tmp_path / "dados"),
        pasta_saida_raiz=str(tmp_path / "saida"),
        modo="sintetico", nivel="N3",
        n_por_classe=10, n_pontos_sint=60, n_replicas_sint=3,
        wn_min=400.0, wn_max=4001.0,
        n_splits_cv=2, n_repeats_cv=1, n_permutacoes=5,
        n_permutacoes_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20,
        n_monte_carlo=3, max_lvs=5,
        executar_benchmark_regressao=True,
    )
    os.makedirs(cfg.pasta_entrada, exist_ok=True)
    pq.executar(cfg)

    runs = achar_pastas_run(cfg.pasta_saida_raiz)
    assert runs, "executar() nao criou pasta de saida"
    pasta_run = runs[0]
    assert os.path.exists(
        os.path.join(pasta_run, pq.NOME_TABELAS, "benchmark_regressao.csv"))
    assert os.path.exists(
        os.path.join(pasta_run, pq.NOME_GRAFICOS, "fig_benchmark_regressores.png"))
