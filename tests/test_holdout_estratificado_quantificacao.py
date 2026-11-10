"""M-05 do parecer de 2026-09-25: protocolo de avaliacao externa.

(a) holdout agrupado ESTRATIFICADO (nenhuma classe ausente em silencio);
(b) a quantificacao e' avaliada no holdout externo (RMSEP publicavel);
(c) o IC BCa nunca tem largura zero.
Reproduz `revisor2/holdout_classes.py`.
"""
import numpy as np
import pytest

import guaraci.pipeline as pq
from guaraci.validacao_estatistica import bootstrap_bca_ci
from sklearn.metrics import balanced_accuracy_score
from tests._resumo import canonico


def _y_g(n_cls=4, grupos_por_classe=3, rep=3):
    y, g = [], []
    for k in range(n_cls):
        for s in range(grupos_por_classe):
            for _ in range(rep):
                y.append(f"C{k}")
                g.append(f"C{k}_S{s}")
    return np.array(y), np.array(g)


def test_holdout_estratificado_cobre_toda_classe_com_2_ou_mais_grupos_em_muitas_sementes():
    y, g = _y_g()
    for seed in range(60):
        tr, ho = pq.holdout_agrupado_estratificado(y, g, 0.2, seed)
        assert set(y[ho]) == set(y), f"seed {seed}: classes fora do holdout"
        # grupos indivisiveis: nenhum grupo em treino e holdout ao mesmo tempo
        assert not (set(g[tr]) & set(g[ho]))
        assert len(set(tr) & set(ho)) == 0 and len(tr) + len(ho) == len(y)
        # a classe continua com >=1 grupo no treino
        assert all(c in set(y[tr]) for c in set(y))


def test_classe_com_um_unico_grupo_nao_e_dividida():
    y, g = _y_g(grupos_por_classe=3)
    y = np.concatenate([y, np.array(["RARA"] * 3)])
    g = np.concatenate([g, np.array(["RARA_S0"] * 3)])
    tr, ho = pq.holdout_agrupado_estratificado(y, g, 0.2, 0)
    lados = ({"tr" for i in tr if y[i] == "RARA"} | {"ho" for i in ho if y[i] == "RARA"})
    assert len(lados) == 1          # inteira de um lado so'


def test_holdout_legado_ainda_deixava_classe_de_fora_e_o_novo_nao():
    """Contra-prova: o GroupShuffleSplit legado, nas mesmas sementes, deixava
    classes fora do holdout; o estratificado nao."""
    from sklearn.model_selection import GroupShuffleSplit
    y, g = _y_g()
    fora_legado = 0
    for seed in range(60):
        _tr, ho = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(y, y, g))
        fora_legado += len(set(y) - set(y[ho])) > 0
    assert fora_legado > 30                       # o defeito era a regra, nao a excecao


def test_execucao_reporta_classes_ausentes_do_holdout_no_resumo(tmp_path):
    """Fluxo real: 4 classes com 3 grupos cada -> nenhuma ausente."""
    from guaraci.io_registry import register_reader
    rng = np.random.default_rng(0)
    wn = np.linspace(4000, 10000, 100)
    X, y, g = [], [], []
    for k in range(4):
        base = np.exp(-((wn - 5000 - k * 600) / 300) ** 2)
        for s in range(3):
            amp = 1 + 0.2 * rng.standard_normal()
            for _ in range(3):
                X.append(1 + base * amp + 0.01 * rng.standard_normal(100))
                y.append(f"C{k}"); g.append(f"C{k}_S{s}")
    X, y, g = np.array(X), np.array(y), np.array(g)
    register_reader("holdout_estrat_teste", lambda cfg: (wn, X, y, None, g, None))
    cfg = pq.Config(mode="holdout_estrat_teste", level="N1", objective="classificacao",
                    output_root_folder=str(tmp_path), show_plots=False,
                    run_benchmark=False, run_monte_carlo=False, run_shap=False,
                    run_wold=False, run_cv_anova=False, run_opls=False,
                    run_ddsimca=False, executar_etapa4=False, n_permutations=5,
                    frac_holdout=0.2, seed=0, max_lvs=4, wn_min=4000, wn_max=10000,
                    selecao_lv_cv_aninhada=False)
    pq.executar(cfg)
    r = canonico(list(tmp_path.rglob("resumo_modelo.txt"))[0])
    assert "Holdout classes ausentes" in r and "nenhuma" in r.split("Holdout classes ausentes")[1].splitlines()[0]


def test_bca_com_um_unico_grupo_por_classe_nao_devolve_intervalo_de_largura_zero():
    y = np.array(["A"] * 6 + ["B"] * 6)
    pred = y.copy()
    pred[0] = "B"
    g = np.array(["a"] * 6 + ["b"] * 6)               # 1 grupo por classe
    lo, hi, obs = bootstrap_bca_ci(y, pred, balanced_accuracy_score, n_boot=200,
                                   seed=0, groups=g)
    assert np.isnan(lo) and np.isnan(hi) and 0 < obs < 1


def test_bca_com_grupos_suficientes_continua_dando_intervalo():
    y, g = _y_g(2, 8, 2)
    rng = np.random.default_rng(0)
    pred = y.copy()
    trocar = rng.choice(len(y), 8, replace=False)
    pred[trocar] = np.where(y[trocar] == "C0", "C1", "C0")
    lo, hi, obs = bootstrap_bca_ci(y, pred, balanced_accuracy_score, n_boot=300,
                                   seed=1, groups=g)
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi


def test_quantificacao_e_avaliada_no_holdout_externo():
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 15))
    beta = rng.normal(size=15)
    y = X @ beta + 0.05 * rng.normal(size=40)
    pipe = Pipeline([("mc", StandardScaler(with_std=False)),
                     ("pls", PLSRegression(n_components=5, scale=False))]).fit(X[:30], y[:30])
    r = pq.avaliar_quantificacao_holdout(
        {"ESA": {"pipeline": pipe}}, X[30:], np.array(["ESA"] * 10), y[30:])
    assert r["n"] == 10 and r["n_sem_modelo"] == 0
    esperado = float(np.sqrt(np.mean((np.asarray(pipe.predict(X[30:])).ravel() - y[30:]) ** 2)))
    assert r["rmsep"] == pytest.approx(esperado)
    assert r["tabela_especie"][0]["especie"] == "ESA"
    # classe sem modelo de regressao: fica de fora e e' contada
    r2 = pq.avaliar_quantificacao_holdout(
        {"ESA": {"pipeline": pipe}}, X[30:],
        np.array(["ESA"] * 6 + ["OUTRA"] * 4), y[30:])
    assert r2["n"] == 6 and r2["n_sem_modelo"] == 4


def test_n3_publica_rmsep_do_holdout_no_resumo(tmp_path):
    cfg = pq.Config(mode="sintetico", level="N3", objective="quantificacao",
                    output_root_folder=str(tmp_path), show_plots=False,
                    synthetic_adulterants=("A", "M", "S"), n_permutations=5,
                    frac_holdout=0.25, seed=0, max_lvs=6,
                    run_benchmark=False, run_monte_carlo=False, run_shap=False,
                    run_wold=False, run_cv_anova=False, run_opls=False,
                    run_ddsimca=False, executar_etapa4=False,
                    selecao_lv_cv_aninhada=False)
    pq.executar(cfg)
    r = canonico(list(tmp_path.rglob("resumo_modelo.txt"))[0])
    assert "RMSEP (holdout externo)" in r and "numero publicavel" in r
    assert "validacao INTERNA" in r


def test_n3_uma_matriz_so_tambem_publica_rmsep_do_holdout(tmp_path):
    """Caso de uma matriz so' (1 classe, modelo pooled): o holdout tambem e' avaliado."""
    from guaraci.io_registry import register_reader
    rng = np.random.default_rng(0)
    wn = np.linspace(1100, 2498, 120)
    y = rng.uniform(8, 10, 60)
    X = (np.outer(y, np.exp(-((wn - 1700) / 80) ** 2))
         + 0.02 * rng.standard_normal((60, 120)) + 1.0)
    register_reader("pooled_holdout_teste",
                    lambda cfg: (wn, X, np.array(["adulterante M"] * 60), y, None, None))
    cfg = pq.Config(mode="pooled_holdout_teste", level="N3", objective="quantificacao",
                    output_root_folder=str(tmp_path), show_plots=False,
                    group_by_mae_id=False, n_permutations=3, frac_holdout=0.25,
                    seed=0, max_lvs=5, wn_min=1100, wn_max=2498,
                    run_benchmark=False, run_monte_carlo=False, run_shap=False,
                    run_wold=False, run_cv_anova=False, run_opls=False,
                    run_ddsimca=False, executar_etapa4=False,
                    selecao_lv_cv_aninhada=False)
    pq.executar(cfg)
    r = canonico(list(tmp_path.rglob("resumo_modelo.txt"))[0])
    assert "RMSEP (holdout externo)" in r
