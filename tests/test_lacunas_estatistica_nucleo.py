"""Ramos de borda do nucleo estatistico (pre-processamento, estatistica quimiometrica, DD-SIMCA/OPLS-DA): um teste por ramo.

Cada teste forca UM ramo que a suite principal nao exercia (entrada degenerada, convencao antiga do sklearn, escala minuscula,
quantil zero). Os dados sao sinteticos e minusculos; nada aqui mede desempenho.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from guaraci import chemometric_stats as cs
from guaraci import classificadores as cl
from guaraci import preprocessamento as pp


# ── preprocessamento ────────────────────────────────────────────────────────
def test_msc_linha_constante_tem_inclinacao_zero_e_nao_divide_por_zero():
    X = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0], [2.0, 4.0, 6.0, 8.0]])
    out = pp._msc_aplicar(X, X.mean(axis=0))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[1], 0.0)                 # b = 0: so' remove o nivel (a = media da linha)


def test_osc_aceita_y_unidimensional():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(20, 8)), rng.normal(size=20)
    assert pp.OSC(n_componentes=1).fit(X, y).transform(X).shape == X.shape


def test_airpls_sem_iteracoes_devolve_o_proprio_espectro_como_linha_de_base():
    X = np.abs(np.random.default_rng(1).normal(size=(3, 12)))
    np.testing.assert_allclose(pp.AirPLS(itermax=0).fit(X).transform(X), 0.0, atol=1e-12)


def test_airpls_espectro_todo_zero_nao_tem_residuo_negativo():
    X = np.zeros((2, 10))
    out = pp.AirPLS().fit(X).transform(X)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, 0.0, atol=1e-9)


# ── chemometric_stats ───────────────────────────────────────────────────────
def _pls_com_coef(coef):
    """PLS real com o `coef_` trocado: so' a convencao de forma muda (a antiga do sklearn era transposta / 1-D)."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 6))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=30)
    real = PLSRegression(n_components=2, scale=False).fit(X, y.reshape(-1, 1))

    class _Delegado:
        coef_ = coef(real)

        def __getattr__(self, nome):
            return getattr(real, nome)
    return _Delegado(), X, np.asarray(real.coef_, dtype=float)


def test_selectivity_ratio_aceita_coef_unidimensional_e_transposto():
    modelo, X, coef = _pls_com_coef(lambda r: np.asarray(r.coef_).reshape(-1))
    esperado = cs.compute_selectivity_ratio(SimpleNamespace(coef_=coef), X)
    np.testing.assert_allclose(cs.compute_selectivity_ratio(modelo, X), esperado)
    transposto, X2, _ = _pls_com_coef(lambda r: np.asarray(r.coef_).reshape(-1, 1))
    np.testing.assert_allclose(cs.compute_selectivity_ratio(transposto, X2), esperado)


class _PLSFalso:
    """Substitui PLSRegression dentro de chemometric_stats: cada `fit` consome uma saida pronta (coef_ ou excecao)."""
    saidas: list = []

    def __init__(self, **_):
        pass

    def fit(self, X, Y):
        s = _PLSFalso.saidas.pop(0)
        if isinstance(s, Exception):
            raise s
        self.coef_ = s
        return self


def test_martens_descarta_fold_degenerado_ou_de_forma_diferente_e_aceita_coef_unidimensional(monkeypatch):
    p = 4
    ok = np.full((1, p), 0.5)
    _PLSFalso.saidas = [np.full(p, 0.4),                 # 1-D: reformatado para (1, p)
                        ok + 0.1,
                        np.full((p, 1), 0.5),            # forma transposta: nao confere com coef_completo -> descartado
                        ValueError("fold degenerado"),   # classe ausente no treino -> descartado
                        ok + 0.2, ok]
    monkeypatch.setattr(cs, "PLSRegression", _PLSFalso)
    X, Y = np.zeros((12, p)), np.zeros((12, 1))
    cv = [(np.arange(12), np.arange(2)) for _ in range(6)]
    r = cs.martens_uncertainty_test(X, Y, 1, cv, np.full(p, 0.5))            # coef_completo 1-D tambem e' reformatado
    assert int(r["n_folds_validos"]) == 4
    assert np.isfinite(r["t_valores"]).all()


def test_q_residuals_loo_sem_amostras_suficientes_para_ajustar_devolve_zero():
    X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 5.0]])
    q = cs.q_residuals_loo(X, 2, groups=np.array([0, 1]))        # tirar um grupo deixa 1 linha: nao ha' PCA
    np.testing.assert_array_equal(q, [0.0, 0.0])


def _modelo_e_replicas(identicas):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 10))
    modelo = PLSRegression(n_components=2, scale=False).fit(X, (X[:, 0] * 3).reshape(-1, 1))
    grupos = [np.tile(X[i], (3, 1)) + (0 if identicas else rng.normal(scale=0.05, size=(3, 10))) for i in range(5)]
    return modelo, X, grupos


def test_figuras_de_merito_com_replicas_identicas_nao_estimam_limites():
    modelo, X, grupos = _modelo_e_replicas(identicas=True)
    f = cs.regression_figures_of_merit(modelo, X, grupos)
    assert f["delta_x_ruido"] < 1e-12
    assert np.isnan(f["lod"]) and np.isnan(f["sensibilidade_analitica"])


@pytest.mark.parametrize("alpha_ic, lado_pulado, lado_calculado", [(0.0, "lod_ic_alto", "lod_ic_baixo"), (2.0, "lod_ic_baixo", "lod_ic_alto")])
def test_figuras_de_merito_alpha_degenerado_pula_o_lado_do_ic_com_quantil_zero(alpha_ic, lado_pulado, lado_calculado):
    modelo, X, grupos = _modelo_e_replicas(identicas=False)
    f = cs.regression_figures_of_merit(modelo, X, grupos, alpha_ic=alpha_ic)
    assert np.isnan(f[lado_pulado]) and not np.isnan(f[lado_calculado])


def test_diagnostico_de_faixa_com_janela_par_usa_a_proxima_impar():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(10, 60)).cumsum(axis=1)
    r = cs.diagnose_spectral_range(X, np.linspace(4000, 10000, 60), janela_suave=10)
    assert r["snr"].shape == (60,)


def test_diagnostico_de_faixa_as_bordas_da_media_movel_repetem_o_valor_interno_mais_proximo():
    """A convolucao 'same' atenua meia janela em cada ponta: o SNR dessas pontas e' substituido pelo do vizinho interno
    (esquerda: o primeiro ponto valido; direita: o ultimo ponto valido), nao fica com o valor atenuado."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(10, 60)).cumsum(axis=1)
    r = cs.diagnose_spectral_range(X, np.linspace(4000, 10000, 60), janela_suave=5)       # janela 5 => borda de 2 pontos
    snr = r["snr"]
    assert np.all(snr[:2] == snr[2]) and np.all(snr[-2:] == snr[-3])
    assert snr[2] != snr[3] and snr[-3] != snr[-4]                         # o interior segue variando: so' as pontas foram copiadas


# ── classificadores ─────────────────────────────────────────────────────────
def _duas_classes(rng, n=12, k=5):
    X = np.vstack([rng.normal(0, 0.3, size=(n, k)), rng.normal(5, 0.3, size=(n, k))])
    return X, np.array(["A"] * n + ["B"] * n)


def test_ddsimca_amostra_aceita_por_duas_classes_e_ambigua(monkeypatch):
    X, y = _duas_classes(np.random.default_rng(5))
    dd = cl.DDSimca(n_components=2).fit(X, y)
    monkeypatch.setattr(cl.DDSimca, "_f_distance", lambda self, T2, Q, m: np.zeros_like(np.atleast_1d(T2), dtype=float))
    assert set(dd.predict(X[:3])) == {"Ambiguo"}


def test_ddsimca_score_matrix_avisa_quando_o_x_nao_confere_com_o_do_treino(caplog):
    X, y = _duas_classes(np.random.default_rng(6))
    dd = cl.DDSimca(n_components=2).fit(X, y)
    mask = np.zeros(len(X), dtype=bool)
    mask[:5] = True                                # 5 linhas de treino da classe A contra 12 valores de Q_train
    with caplog.at_level(logging.WARNING, logger=cl.log.name):
        res = dd.score_matrix(X, mask_treino=mask, y=y)
    assert "X nao confere" in caplog.text and set(res) == {"A", "B"}


def test_opls_com_escala_minuscula_para_no_nipals_e_nao_ha_componente_ortogonal():
    rng = np.random.default_rng(7)
    m = cl.OPLSDAWrapper(n_ortho=1).fit(rng.normal(size=(30, 6)) * 1e-8, np.sign(rng.normal(size=30)))
    assert len(m.W_orth_) == 0                    # nt < 1e-12: o NIPALS para e o loop ortogonal tambem


def test_opls_com_segunda_direcao_quase_nula_para_no_escore_ortogonal_minusculo():
    rng = np.random.default_rng(0)
    a1, a2 = rng.normal(size=30), rng.normal(size=30)
    a1, a2 = a1 - a1.mean(), a2 - a2.mean()
    y = np.sign(a1 + 0.3 * rng.normal(size=30))
    X = np.outer(a1, rng.normal(size=6)) + 1e-8 * np.outer(a2, rng.normal(size=6))     # posto 1 + ruido de 1e-8
    assert len(cl.OPLSDAWrapper(n_ortho=1).fit(X, y).W_orth_) == 0      # w_orth existe (>= 1e-10) mas t_orth e' ~ 1e-8


def test_opls_transform_com_escore_preditivo_nulo_nao_ortogonaliza(monkeypatch):
    rng = np.random.default_rng(8)
    X, y = rng.normal(size=(30, 6)), np.sign(rng.normal(size=30))
    m = cl.OPLSDAWrapper(n_ortho=1).fit(X, y)
    assert len(m.W_orth_) == 1
    m._pls_pred = SimpleNamespace(transform=lambda Xr: np.zeros((len(Xr), 1)))
    t_pred, t_orth = m.transform(X)
    assert not t_pred.any() and t_orth.shape[0] == len(X)


def _puros(n_grupos, reps=3, k=5):
    rng = np.random.default_rng(9)
    X = np.vstack([rng.normal(0, 0.2, size=(reps, k)) for _ in range(n_grupos)])
    return X, np.array([f"g{i}" for i in range(n_grupos) for _ in range(reps)])


def test_logo_com_dez_grupos_ou_mais_nao_traz_aviso_de_poucos_grupos():
    X, g = _puros(10)
    r = cl.ddsimca_logo_sensitivity(X, g, n_components=2)
    assert r["n_grupos"] == 10 and r["aviso"] is None


def test_logo_pula_a_dobra_em_que_o_modelo_nao_e_ajustavel(monkeypatch):
    X, g = _puros(4)
    original = cl.DDSimca.score_matrix
    chamadas = {"n": 0}

    def a_primeira_dobra_nao_tem_modelo(self, *a, **k):
        chamadas["n"] += 1
        return {} if chamadas["n"] == 1 else original(self, *a, **k)
    monkeypatch.setattr(cl.DDSimca, "score_matrix", a_primeira_dobra_nao_tem_modelo)
    r = cl.ddsimca_logo_sensitivity(X, g, n_components=2)
    assert r["n_grupos_validos"] == 3


def test_pcv_falha_do_prcv_vira_aviso(monkeypatch):
    pytest.importorskip("prcv")
    import prcv.methods

    def quebra(*a, **k):
        raise ValueError("matriz mal condicionada")
    monkeypatch.setattr(prcv.methods, "pcvpca", quebra)
    X, g = _puros(4)
    r = cl.ddsimca_pcv_sensitivity(X, g, n_components=2)
    assert "PCV falhou" in r["aviso"] and np.isnan(r["sensibilidade"])


def test_pcv_modelo_nao_ajustavel_vira_aviso(monkeypatch):
    pytest.importorskip("prcv")
    monkeypatch.setattr(cl.DDSimca, "score_matrix", lambda self, *a, **k: {})
    X, g = _puros(4)
    assert "nao ajustavel" in cl.ddsimca_pcv_sensitivity(X, g, n_components=2)["aviso"]


def test_pcv_com_dez_grupos_ou_mais_nao_traz_aviso():
    pytest.importorskip("prcv")
    X, g = _puros(10)
    r = cl.ddsimca_pcv_sensitivity(X, g, n_components=2)
    assert r["aviso"] is None and 0.0 <= r["sensibilidade"] <= 1.0


def test_nipals_que_nao_converge_em_max_iter_devolve_a_ultima_iteracao():
    rng = np.random.default_rng(10)
    X, y = rng.normal(size=(20, 5)), rng.normal(size=20)
    w, t, p = cl.OPLSDAWrapper._nipals_pls1(X, y, max_iter=1, tol=0.0)      # tol = 0: nunca "converge"
    assert w.shape == (5,) and t.shape == (20,) and p.shape == (5,) and abs(np.linalg.norm(w) - 1.0) < 1e-9
