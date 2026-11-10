"""Ramos de borda de guaraci.avaliacao_modelos (benchmark, Monte Carlo, DET, SHAP, regressao por especie): um teste por ramo.

Os classificadores/regressores caros sao trocados por equivalentes minusculos ou por dublês que falham de proposito; o que se testa e'
a LOGICA de tolerancia a falha e de agregacao deste modulo, nao o desempenho dos modelos.
"""
from __future__ import annotations

import builtins
import os
import sys
from types import ModuleType, SimpleNamespace

import matplotlib
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC

matplotlib.use("Agg")

from guaraci import avaliacao_modelos as am  # noqa: E402
PLSDAClassifier = am.PLSDAClassifier
from guaraci.config import NOME_TABELAS, Config  # noqa: E402


def _dados(n_per_class=12, p=10, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c * 4.0, 1.0, size=(n_per_class, p)) for c in range(n_classes)])
    y = np.repeat(np.arange(n_classes), n_per_class)
    grupos = np.array([f"g{c}_{i}" for c in range(n_classes) for i in range(n_per_class)])
    return X, y, grupos, LabelBinarizer().fit(y)


def _cfg(**kw):
    base = dict(n_splits_cv=3, seed=0, run_shap=False, n_monte_carlo=4, monte_carlo_test_size=0.3, default_preprocessing="mc")
    base.update(kw)
    return Config(**base)


def _pasta(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), NOME_TABELAS), exist_ok=True)
    return str(tmp_path)


class _SemProba(LinearSVC):                      # sem predict_proba: o OOF dessa linha fica zerado
    pass


class _ProbaQuebra(LogisticRegression):
    def predict_proba(self, X):
        raise ValueError("predict_proba degenerado neste fold")


class _FitQuebra(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        raise RuntimeError("nao ajusta")


class _FitValueError(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        raise ValueError("iteracao degenerada")

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def _lista(*pares):
    return lambda n_opt, cfg, incluir_opcionais=True: [(n, c) for n, c in pares]


# ── benchmark_classifiers ───────────────────────────────────────────────────
def test_benchmark_tolera_modelo_sem_proba_proba_degenerada_fit_que_falha_e_compara_com_o_plsda(monkeypatch, tmp_path, capsys):
    X, y, g, lb = _dados()
    monkeypatch.setattr(am, "build_benchmark_list", _lista(
        ("PLS-DA", PLSDAClassifier(n_components=2)), ("PLS-DA copia", PLSDAClassifier(n_components=2)),
        ("Sem proba", _SemProba()), ("Proba quebra", _ProbaQuebra()), ("Quebra", _FitQuebra())))
    monkeypatch.setattr(am, "fig_det_curvas", lambda *a, **k: (_ for _ in ()).throw(ValueError("det quebrou")))
    import scipy.stats
    real_wilcoxon = scipy.stats.wilcoxon

    def wilcoxon_so_para_a_copia(ref, alt, **k):
        if np.array_equal(ref, alt):
            raise ValueError("diferencas todas zero")
        return real_wilcoxon(ref, alt, **k)
    monkeypatch.setattr(scipy.stats, "wilcoxon", wilcoxon_so_para_a_copia)
    df = am.benchmark_classifiers(X, y, g, lb, 2, _cfg(n_splits_cv=6), _pasta(tmp_path))
    saida = capsys.readouterr().out
    assert set(df["Classifier"]) == {"PLS-DA", "PLS-DA copia", "Sem proba", "Proba quebra"}            # o que falhou nao entra
    assert "FALHA (nao ajusta)" in saida and "DET curves falhou: det quebrou" in saida
    linha = df.set_index("Classifier").loc["PLS-DA copia"]
    assert linha["p Wilcoxon (vs PLS-DA)"] == "n/a"                      # diferencas todas zero: nao inventa p-valor


def test_p_do_wilcoxon_contra_o_plsda_e_arredondado_a_4_casas(monkeypatch, tmp_path):
    """O p devolvido pelo teste de Wilcoxon entra na tabela arredondado a 4 casas (0,123456789 -> 0,1235)."""
    X, y, g, lb = _dados()
    monkeypatch.setattr(am, "build_benchmark_list", _lista(
        ("PLS-DA", PLSDAClassifier(n_components=2)), ("Logistica", LogisticRegression(max_iter=200))))
    monkeypatch.setattr(am, "fig_det_curvas", lambda *a, **k: None)
    monkeypatch.setattr(am, "fig_benchmark_classifiers", lambda *a, **k: None)
    import scipy.stats
    monkeypatch.setattr(scipy.stats, "wilcoxon", lambda ref, alt, **k: (0.0, 0.123456789))
    df = am.benchmark_classifiers(X, y, g, lb, 2, _cfg(n_splits_cv=6), _pasta(tmp_path))
    assert df.set_index("Classifier").loc["Logistica", "p Wilcoxon (vs PLS-DA)"] == 0.1235


def test_benchmark_em_que_todos_os_modelos_falham_devolve_tabela_vazia_sem_figuras(monkeypatch, tmp_path):
    X, y, g, lb = _dados()
    monkeypatch.setattr(am, "build_benchmark_list", _lista(("Quebra", _FitQuebra())))
    chamadas = []
    monkeypatch.setattr(am, "fig_benchmark_classifiers", lambda *a, **k: chamadas.append("box"))
    df = am.benchmark_classifiers(X, y, g, lb, 2, _cfg(), _pasta(tmp_path))
    assert len(df) == 0 and chamadas == []


@pytest.mark.parametrize("ram_ok", [True, False])
def test_benchmark_com_shap_ligado_so_gera_a_figura_se_houver_ram(monkeypatch, tmp_path, ram_ok):
    from guaraci import modos_analise
    X, y, g, lb = _dados()
    monkeypatch.setattr(am, "build_benchmark_list", _lista(("PLS-DA", PLSDAClassifier(n_components=2))))
    monkeypatch.setattr(modos_analise, "should_generate", lambda cfg, nome: True)
    monkeypatch.setattr(am, "_verificar_ram", lambda gb, rotulo: ram_ok)
    chamadas = []
    monkeypatch.setattr(am, "_fig_shap_benchmark", lambda *a, **k: chamadas.append("shap"))
    am.benchmark_classifiers(X, y, g, lb, 2, _cfg(run_shap=True), _pasta(tmp_path))
    assert chamadas == (["shap"] if ram_ok else [])


# ── monte_carlo_cv ──────────────────────────────────────────────────────────
def test_monte_carlo_sem_grupos_usa_split_estratificado_simples(monkeypatch, tmp_path):
    X, y, g, lb = _dados()
    monkeypatch.setattr(am, "build_benchmark_list", _lista(("PLS-DA", PLSDAClassifier(n_components=2))))
    df = am.monte_carlo_cv(X, y, None, lb, 2, _cfg(), _pasta(tmp_path))
    assert len(df) == 1


def test_monte_carlo_com_poucos_grupos_de_teste_para_as_classes_e_pulado(monkeypatch, tmp_path, capsys):
    X, y, g, lb = _dados(n_per_class=4)
    grupos_poucos = np.array([f"c{c}" for c in y])                           # 1 grupo por classe: 3 grupos no total
    monkeypatch.setattr(am, "build_benchmark_list", _lista(("PLS-DA", PLSDAClassifier(n_components=2))))
    df = am.monte_carlo_cv(X, y, grupos_poucos, lb, 2, _cfg(monte_carlo_test_size=0.2), _pasta(tmp_path))
    assert len(df) == 0 and "Monte Carlo CV PULADO" in capsys.readouterr().out


def test_monte_carlo_descarta_iteracoes_degeneradas_sem_derrubar_o_resto(monkeypatch, tmp_path):
    X, y, g, lb = _dados()
    n = len(y)
    sem_classe_no_treino = (np.flatnonzero(y != 2), np.flatnonzero(y == 2))               # treino sem a classe 2
    teste_de_uma_classe = (np.arange(n)[: n - 1], np.array([n - 1]))                       # teste com 1 amostra: 1 classe so'
    boa = (np.arange(0, n, 2), np.arange(1, n, 2))
    monkeypatch.setattr(am, "_stratified_group_shuffle_splits", lambda *a, **k: [sem_classe_no_treino, teste_de_uma_classe, boa])
    monkeypatch.setattr(am, "build_benchmark_list", _lista(("PLS-DA", PLSDAClassifier(n_components=2)), ("Degenerado", _FitValueError())))
    df = am.monte_carlo_cv(X, y, g, lb, 2, _cfg(), _pasta(tmp_path))
    assert "PLS-DA" in set(df.iloc[:, 0])


# ── curvas DET ──────────────────────────────────────────────────────────────
def test_interpolacao_da_curva_det_de_curva_vazia_e_nan():
    assert np.isnan(am.interpolar_det(np.array([]), np.array([]), np.linspace(0.01, 1, 5))).all()


def test_det_sem_det_curve_do_sklearn_pula_com_aviso(monkeypatch, tmp_path, capsys):
    real = builtins.__import__

    def sem_det(nome, *a, **k):
        if nome == "sklearn.metrics" and a and a[2] and "det_curve" in a[2]:
            raise ImportError("sem det_curve")
        return real(nome, *a, **k)
    monkeypatch.setattr(builtins, "__import__", sem_det)
    am.fig_det_curvas({"A": np.zeros((4, 2))}, np.array([0, 1, 0, 1]), 2, _cfg(), str(tmp_path))
    assert "det_curve indisponivel" in capsys.readouterr().out


def test_det_binaria_com_classe_rara_probabilidade_curta_e_classe_degenerada(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    y = np.array([0] * 9 + [1])                                               # classe 1 com 1 amostra: pulada (soma < 2)
    ok = np.column_stack([1 - rng.uniform(size=10), rng.uniform(size=10)])
    curta = rng.uniform(size=(10, 1))                                         # proba com so' 1 coluna: k >= proba.shape[1]
    am.fig_det_curvas({"ok": ok, "curta": curta}, y, 2, _cfg(), _pasta(tmp_path))
    y3 = np.array([0, 1, 2] * 4)
    proba = rng.uniform(size=(12, 3))
    monkeypatch.setattr("sklearn.metrics.det_curve", lambda *a, **k: (_ for _ in ()).throw(ValueError("classe degenerada")))
    am.fig_det_curvas({"a": proba, "b": proba}, y3, 3, _cfg(), _pasta(tmp_path))                  # n_valid == 0: classificador omitido


# ── SHAP ────────────────────────────────────────────────────────────────────
class _ExplicadorFalso:
    formato = "3d"

    def __init__(self, modelo):
        pass

    def shap_values(self, X):
        n, p = X.shape
        rng = np.random.default_rng(0)
        if _ExplicadorFalso.formato == "lista":
            return [rng.normal(size=(n, p)) for _ in range(3)]
        if _ExplicadorFalso.formato == "3d":
            return rng.normal(size=(n, p, 3))
        if _ExplicadorFalso.formato == "2d":
            return rng.normal(size=(n, p))
        return rng.normal(size=(n, 1, p))                    # 3D mas tratado como "fallback" nao: ver 4d abaixo


def _shap_falso(monkeypatch, formato):
    _ExplicadorFalso.formato = formato
    fake = ModuleType("shap")
    fake.TreeExplainer = _ExplicadorFalso
    monkeypatch.setitem(sys.modules, "shap", fake)
    monkeypatch.setattr(am, "shap", fake, raising=False)
    xgboost = pytest.importorskip("xgboost")   # o SHAP e' falso, mas o benchmark instancia o XGBClassifier de verdade
    import sklearn.ensemble
    from sklearn.ensemble import GradientBoostingClassifier as GBC
    from sklearn.ensemble import RandomForestClassifier as RFC
    monkeypatch.setattr(sklearn.ensemble, "RandomForestClassifier", lambda **k: RFC(n_estimators=5, random_state=0))
    monkeypatch.setattr(sklearn.ensemble, "GradientBoostingClassifier", lambda **k: GBC(n_estimators=5, random_state=0))
    monkeypatch.setattr(xgboost, "XGBClassifier", lambda **k: RFC(n_estimators=5, random_state=0))


@pytest.mark.parametrize("formato", ["lista", "3d"])
def test_shap_multiclasse_com_valores_em_lista_ou_em_3d_e_amostragem_limitada(monkeypatch, tmp_path, formato):
    _shap_falso(monkeypatch, formato)
    X, y, g, lb = _dados(n_per_class=6)
    cfg = _cfg(run_shap=True, shap_max_samples=8)
    am.fig_shap_benchmark(X, y, 2, cfg, _pasta(tmp_path), wavenumbers=np.linspace(4000, 400, X.shape[1]))
    assert any(f.startswith("fig_shap_") for f in os.listdir(tmp_path) + [f for _, _, fs in os.walk(tmp_path) for f in fs])


def test_shap_binario_inclui_gradient_boosting_valores_2d_e_muitas_classes_dispensam_legenda(monkeypatch, tmp_path, capsys):
    _shap_falso(monkeypatch, "2d")
    X, y, g, lb = _dados(n_per_class=15, n_classes=2)
    am.fig_shap_benchmark(X, y, 2, _cfg(run_shap=True), _pasta(tmp_path))
    saida = capsys.readouterr().out
    # a importancia 2D e' por VARIAVEL (media sobre as amostras): com 30 amostras e 10 variaveis, media no eixo errado indexaria fora
    assert "FALHA" not in saida and saida.count("salvo.") >= 2
    _shap_falso(monkeypatch, "2d")
    X12, y12, *_ = _dados(n_per_class=3, n_classes=12)
    am.fig_shap_benchmark(X12, y12, 2, _cfg(run_shap=True), _pasta(tmp_path))                 # 12 classes: sem legenda no dependence plot


def test_shap_tolera_xgboost_bloqueado_e_falha_de_um_explicador(monkeypatch, tmp_path, capsys):
    _shap_falso(monkeypatch, "2d")
    import xgboost

    class _XGBoostError(ValueError):
        pass

    def bloqueado(**k):
        raise _XGBoostError("Application Control bloqueou a DLL")
    monkeypatch.setattr(xgboost, "XGBClassifier", bloqueado)
    monkeypatch.setattr(am, "_e_dependencia_indisponivel", lambda e: True)

    def explicador_que_falha(modelo):
        raise RuntimeError("sem memoria")
    sys.modules["shap"].TreeExplainer = explicador_que_falha
    X, y, g, lb = _dados(n_per_class=6)
    am.fig_shap_benchmark(X, y, 2, _cfg(run_shap=True), _pasta(tmp_path))
    assert "FALHA (sem memoria)" in capsys.readouterr().out


def test_shap_re_levanta_erro_de_xgboost_que_nao_e_dependencia_indisponivel(monkeypatch, tmp_path):
    _shap_falso(monkeypatch, "2d")
    import xgboost

    def erro_real(**k):
        raise ValueError("hiperparametro invalido")
    monkeypatch.setattr(xgboost, "XGBClassifier", erro_real)
    monkeypatch.setattr(am, "_e_dependencia_indisponivel", lambda e: False)
    X, y, g, lb = _dados(n_per_class=6)
    with pytest.raises(ValueError, match="hiperparametro invalido"):
        am.fig_shap_benchmark(X, y, 2, _cfg(run_shap=True), _pasta(tmp_path))


# ── regressao por especie ───────────────────────────────────────────────────
def _regressao_dados(n=48, p=12, seed=0):
    rng = np.random.default_rng(seed)
    rotulos = np.array(["A"] * 24 + ["B"] * 24)
    conc = np.tile(np.concatenate([np.zeros(6), np.linspace(1, 10, 18)]), 2)
    X = rng.normal(size=(n, p)) + conc[:, None] * 0.3
    mae = np.array([f"{r}{i // 3}" for r, i in zip(rotulos, np.tile(np.arange(24), 2))])
    return X, conc, rotulos, mae


_PLS_REF = {"rmsep": 1.0, "r2v": 0.5, "n_especies": 2, "tabela_especie": [{"rmsep": 1.0}, {"rmsep": 1.2}]}


def _leves(monkeypatch):
    import sklearn.ensemble
    from sklearn.ensemble import RandomForestRegressor as RFR
    monkeypatch.setattr(sklearn.ensemble, "RandomForestRegressor", lambda **k: RFR(n_estimators=5, random_state=0))


def test_regressao_por_especie_ignora_especie_ausente_e_sem_split_e_segue_com_as_demais(monkeypatch, tmp_path):
    _leves(monkeypatch)
    X, conc, rotulos, mae = _regressao_dados()
    X = np.vstack([X, np.random.default_rng(1).normal(size=(4, X.shape[1]))])
    conc = np.concatenate([conc, [0, 5, 6, 7]])
    rotulos = np.concatenate([rotulos, ["C"] * 4])                                         # C tem so' 4 amostras: sem split viavel
    mae = np.concatenate([mae, ["c0", "c1", "c2", "c3"]])
    df = am.benchmark_regression_by_species(X, conc, rotulos, mae, np.array(["A", "B", "C", "Z"]), _cfg(), _pasta(tmp_path),
                                            _PLS_REF, min_amostras_adult=3)
    assert df is not None and "PLS-R" in set(df["Model"]) and "Ridge" in set(df["Model"])


def test_regressao_por_especie_com_todos_os_modelos_falhando_so_traz_o_baseline(monkeypatch, tmp_path, capsys):
    _leves(monkeypatch)
    X, conc, rotulos, mae = _regressao_dados()

    class _PreprocQuebra(BaseEstimator):
        def fit(self, X, y=None):
            raise RuntimeError("preproc quebrou")
    monkeypatch.setattr(am, "build_preprocessor", lambda cfg: _PreprocQuebra())
    df = am.benchmark_regression_by_species(X, conc, rotulos, mae, np.array(["A", "B"]), _cfg(), _pasta(tmp_path),
                                            {**_PLS_REF, "tabela_especie": []})
    assert list(df["Model"]) == ["PLS-R"] and "falhou em especie" in capsys.readouterr().out


def test_figura_de_regressores_pula_modelo_sem_valores(tmp_path):
    am.fig_benchmark_regressors({"vazio": np.array([]), "Ridge": np.array([1.0, 2.0, 3.0])}, 3, _cfg(), _pasta(tmp_path))
    assert SimpleNamespace  # a funcao nao levanta com uma serie vazia
