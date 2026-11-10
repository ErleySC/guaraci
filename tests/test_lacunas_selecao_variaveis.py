"""Ramos de borda de guaraci.selecao_variaveis (iPLS/VIP/sPLS-DA/SPA/AG/CARS/UVE/Moran e a Etapa 4): um teste por ramo."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from guaraci import selecao_variaveis as sv
from guaraci.config import Config


@pytest.fixture(autouse=True)
def _backend_sem_tela():
    """As figuras da Etapa 4 usam o backend ativo; sem isto o matplotlib pode escolher o Tk (sem display / sem init.tcl) e o teste vira intermitente."""
    import matplotlib
    anterior = matplotlib.get_backend()
    matplotlib.use("Agg", force=True)
    yield
    matplotlib.use(anterior, force=True)


def _dados(seed=0, n=36, p=14, informativas=(2, 3)):
    rng = np.random.default_rng(seed)
    cls = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, p))
    X[np.ix_(np.flatnonzero(cls == 1), [i for i in informativas if i < p])] += 3.0
    Y = np.eye(2)[cls]
    cv = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X, cls))
    return X, Y, cls, cv


def _y_constante(n=36):
    """Todas as amostras da mesma classe: a soma de quadrados total do alvo e' zero."""
    Y = np.zeros((n, 2))
    Y[:, 0] = 1.0
    return Y, np.zeros(n, dtype=int)


# ── Q2 indefinido quando o alvo nao varia ───────────────────────────────────
def test_q2_e_nan_quando_o_alvo_nao_varia_nos_tres_avaliadores():
    X, _, _, cv = _dados()
    Y, y = _y_constante(len(X))
    assert np.isnan(sv._avaliar_subset_cv(X, Y, y, cv, 2)["q2"])
    assert np.isnan(sv._avaliar_subset_nested_cv(X, Y, y, cv, 2, lambda a, b, c: np.ones(X.shape[1], dtype=bool))["q2"])
    assert np.isnan(sv._avaliar_busca_nested_cv(X, Y, y, cv, 2, lambda a, b, c, d: np.ones(X.shape[1], dtype=bool), 0)["q2"])


def test_busca_aninhada_cai_para_todas_as_variaveis_quando_a_busca_nao_devolve_mascara():
    X, Y, y, cv = _dados()
    r = sv._avaliar_busca_nested_cv(X, Y, y, cv, 2, lambda a, b, c, d: None, 0)
    assert r["n_vars_min"] == r["n_vars_max"] == X.shape[1]


def test_cv_interna_com_um_so_grupo_cai_para_estratificada():
    y = np.array([0, 1] * 10)
    folds = sv._cv_local(y, seed=0, grupos_local=np.array(["g"] * 20))
    assert sorted(np.concatenate([va for _, va in folds]).tolist()) == list(range(20))


# ── iPLS / interval-VIP ─────────────────────────────────────────────────────
def test_interval_vip_com_mais_intervalos_que_variaveis_pula_o_intervalo_vazio():
    X, Y, y, cv = _dados(p=3)
    res, mask = sv.selecao_interval_vip(X, Y, y, np.array([4000.0, 3000.0, 2000.0]), cv, 2, 5)
    assert len(res) < 5 and mask.shape == (3,)


# ── sPLS-DA ─────────────────────────────────────────────────────────────────
def test_splsda_para_quando_a_svd_falha_quando_o_loading_e_plano_e_quando_o_escore_some(monkeypatch):
    X, Y, _, _ = _dados()

    def falha(*a, **k):
        raise np.linalg.LinAlgError("nao convergiu")
    monkeypatch.setattr(np.linalg, "svd", falha)
    assert not sv.sparse_plsda_mask(X, Y, 2, 3).any()                      # parou no primeiro componente: nada selecionado
    p = X.shape[1]

    def plano(M, full_matrices=False):
        U = np.zeros((p, 2))
        U[:, 0] = 1.0 / np.sqrt(p)                                          # todos |w| iguais: o soft-threshold zera tudo
        return U, np.ones(2), np.zeros((2, 2))
    monkeypatch.setattr(np.linalg, "svd", plano)
    assert not sv.sparse_plsda_mask(X, Y, 2, 3).any()


def test_splsda_com_escore_minusculo_para_o_laco():
    X, Y, _, _ = _dados()
    mask = sv.sparse_plsda_mask(X * 1e-7, Y, 3, 2)
    assert mask.shape == (X.shape[1],)


# ── SPA ─────────────────────────────────────────────────────────────────────
def test_spa_cadeia_a_partir_de_coluna_nula_para_na_primeira_projecao():
    X, *_ = _dados()
    X = X.copy()
    X[:, 0] = 0.0
    assert sv._spa_cadeia(X, 0, 4).tolist() == [0]


def test_spa_cadeia_devolve_exatamente_n_vars_max_variaveis_distintas():
    X, *_ = _dados()
    for n_max in (2, 3, 4):
        cadeia = sv._spa_cadeia(X, 0, n_max).tolist()
        assert len(cadeia) == n_max and len(set(cadeia)) == n_max and cadeia[0] == 0


def test_spa_com_cadeia_de_uma_variavel_so_e_ignorada():
    X, Y, y, cv = _dados()
    res, mask = sv.selecao_spa(X, Y, y, cv, 2, n_vars_max=1, n_starts=3, seed=0)
    assert res == [] and not mask.any()


# ── AG ──────────────────────────────────────────────────────────────────────
def test_ag_com_populacao_vazia_de_variaveis_ainda_devolve_mascara_valida():
    X, Y, y, cv = _dados(p=6)
    hist, mask = sv.ga_selection(X, Y, y, cv, 2, tam_populacao=4, n_geracoes=2, prob_mutacao=0.5, frac_inicial=0.0, seed=0)
    assert mask.shape == (6,) and mask.sum() >= 2 and len(hist) == 2


# ── coeficientes, Moran ─────────────────────────────────────────────────────
def test_coeficiente_por_variavel_aceita_coef_unidimensional():
    class _Pls:
        coef_ = np.array([-1.0, 2.0, -3.0])
    assert sv._coef_por_variavel(_Pls(), 3).tolist() == [1.0, 2.0, 3.0]


def test_moran_sem_vizinhos_nao_seleciona_nada():
    X, *_ = _dados()
    coords = np.random.default_rng(1).normal(size=(len(X), 2))
    info, mask = sv.moran_i_mask(X, coords, k_vizinhos=0, n_permutacoes=9)
    assert not mask.any() and not info["moran_i"].any() and (info["p_valor"] == 1).all()


def test_moran_aninhado_cai_para_todas_as_variaveis_quando_nada_e_significativo():
    X, Y, y, cv = _dados()
    coords = np.random.default_rng(2).normal(size=(len(X), 2))
    r = sv._avaliar_subset_nested_cv_moran(X, Y, y, coords, cv, 2, k_vizinhos=0, n_permutacoes=9)
    assert r["n_vars_min"] == r["n_vars_max"] == X.shape[1]


# ── Etapa 4 completa, com todos os metodos opcionais ligados ────────────────
def _cfg(**kw):
    base = dict(seed=3, ipls_n_intervalos=4, run_spa=True, executar_ag=True, run_cars=True, run_uve=True, spa_n_starts=2, spa_n_vars_max=4,
                ag_tam_populacao=4, ag_n_geracoes=2, cars_n_iteracoes=3, uve_n_repeticoes=3)
    base.update(kw)
    return Config(**base)


def test_etapa4_com_spa_ag_cars_e_uve_ligados_grava_tabela_e_csvs(tmp_path):
    X, Y, y, cv = _dados(p=16)
    resumo = sv.etapa4_selecao_variaveis(X, Y, y, np.linspace(4000, 400, X.shape[1]), cv, 2, _cfg(), str(tmp_path), str(tmp_path))
    metodos = {t["metodo"] for t in resumo["tabela"]}
    assert {"SPA (APS)", "AG (Genetico)", "CARS", "UVE"} <= metodos
    assert (tmp_path / "etapa4_cars_iteracoes.csv").is_file() and (tmp_path / "etapa4_ag_historico.csv").is_file()
    for nome in ("etapa4_cars_iteracoes.csv", "etapa4_ag_historico.csv"):                # sem coluna de indice do pandas
        cabecalho = (tmp_path / nome).read_text(encoding="utf-8").splitlines()[0]
        assert cabecalho and not cabecalho.startswith(";"), nome


def test_etapa4_omite_da_tabela_o_metodo_que_nao_escolheu_variaveis(tmp_path, monkeypatch):
    X, Y, y, cv = _dados(p=16)
    real = sv._avaliar_subset_nested_cv
    vazio = lambda *a, **k: {**real(*a, **k), "n_vars_max": 1}                  # noqa: E731 -- nenhum fold escolheu >= 2 variaveis
    monkeypatch.setattr(sv, "_avaliar_subset_nested_cv", lambda *a, **k: vazio(*a, **k))
    real_busca = sv._avaliar_busca_nested_cv
    monkeypatch.setattr(sv, "_avaliar_busca_nested_cv", lambda *a, **k: {**real_busca(*a, **k), "n_vars_max": 1})
    resumo = sv.etapa4_selecao_variaveis(X, Y, y, np.linspace(4000, 400, X.shape[1]), cv, 2, _cfg(), str(tmp_path), str(tmp_path))
    metodos = {t["metodo"] for t in resumo["tabela"]}
    assert not metodos & {"SPA (APS)", "AG (Genetico)", "CARS", "UVE", "sPLS-DA"} and not any(m.startswith("VIP") for m in metodos)
    assert "Full (todas)" in metodos


def test_etapa4_sem_spa_ag_cars_uve_nao_os_executa(tmp_path):
    X, Y, y, cv = _dados(p=16)
    cfg = _cfg(run_spa=False, executar_ag=False, run_cars=False, run_uve=False)
    resumo = sv.etapa4_selecao_variaveis(X, Y, y, np.linspace(4000, 400, X.shape[1]), cv, 2, cfg, str(tmp_path), str(tmp_path))
    assert not {"SPA (APS)", "AG (Genetico)", "CARS", "UVE"} & {t["metodo"] for t in resumo["tabela"]}



def test_moran_aninhado_com_gradiente_espacial_seleciona_as_variaveis_que_o_seguem():
    rng = np.random.default_rng(5)
    n, p = 30, 6
    coords = rng.uniform(size=(n, 2))
    X = rng.normal(scale=0.05, size=(n, p))
    X[:, :3] += coords[:, [0]] * 5.0                              # tres canais seguem o gradiente leste-oeste
    cls = np.array([0, 1] * (n // 2))
    cv = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, cls))
    r = sv._avaliar_subset_nested_cv_moran(X, np.eye(2)[cls], cls, coords, cv, 2, k_vizinhos=4, n_permutacoes=999)
    assert r["n_vars_max"] < p and r["n_vars_min"] >= 2


def test_etapa4_sem_cadeias_spa_sem_geracoes_do_ag_nao_grava_esses_csvs(tmp_path):
    X, Y, y, cv = _dados(p=16)
    cfg = _cfg(spa_n_vars_max=1, ag_n_geracoes=0)
    sv.etapa4_selecao_variaveis(X, Y, y, np.linspace(4000, 400, X.shape[1]), cv, 2, cfg, str(tmp_path), str(tmp_path))
    assert not (tmp_path / "etapa4_spa_cadeias.csv").exists() and not (tmp_path / "etapa4_ag_historico.csv").exists()
