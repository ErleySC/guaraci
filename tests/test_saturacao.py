"""Saturacao do detector: variavel/espectro no teto saem do modelo, pico isolado nao."""
import numpy as np

from guaraci.saturacao import detectar_saturacao, regioes_de_mascara


def _espectros(n=60, p=400, seed=0):
    rng = np.random.default_rng(seed)
    wn = np.linspace(4000, 10000, p)
    base = np.exp(-((wn - 7000) / 900.0) ** 2) * 2.0
    X = base + rng.normal(0, 0.01, (n, p)) + rng.uniform(0, 0.3, (n, 1))
    return wn, X


def test_sem_saturacao_nao_mexe():
    wn, X = _espectros()
    r = detectar_saturacao(X, wn)
    assert not r["ativo"]
    assert r["manter_variaveis"].all() and r["manter_espectros"].all()


def test_pico_isolado_nao_e_saturacao():
    wn, X = _espectros()
    X[3, 200] = X.max() + 5.0          # um ponto muito alto, sem plato
    assert not detectar_saturacao(X, wn)["ativo"]


def test_banda_saturada_em_varios_espectros_sai():
    wn, X = _espectros()
    X[:, 100:130] = 3.0                # plato no teto em todas as amostras
    X = np.minimum(X, 3.0)
    r = detectar_saturacao(X, wn)
    assert r["ativo"] and abs(r["teto"] - 3.0) < 0.01
    assert (~r["manter_variaveis"]).sum() >= 30
    assert r["regioes"] and r["regioes"][0][0] < r["regioes"][0][1]
    assert r["manter_espectros"].all()


def test_espectro_inteiro_saturado_sai_e_os_outros_ficam():
    wn, X = _espectros()
    X[:, 100:130] = 3.0
    X[5, :] = 3.0                      # espectro todo no teto
    X[9, 150:300] = 3.0                # 150/400 do espectro no teto
    X = np.minimum(X, 3.0)
    r = detectar_saturacao(X, wn)
    assert r["ativo"]
    assert not r["manter_espectros"][5] and not r["manter_espectros"][9]
    assert r["manter_espectros"].sum() == X.shape[0] - 2


def test_tabela_pequena_nao_e_espectro():
    X = np.full((20, 8), 3.0)
    assert not detectar_saturacao(X, np.arange(8.0))["ativo"]


def test_tudo_saturado_nao_mexe():
    wn, X = _espectros()
    X[:] = 3.0
    r = detectar_saturacao(X, wn)
    assert not r["ativo"]


def test_regioes_de_mascara():
    wn = np.arange(10.0)
    m = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1], dtype=bool)
    assert regioes_de_mascara(wn, m) == [(1.0, 2.0), (5.0, 5.0), (8.0, 9.0)]


def test_limite_q_cauda_pesada_nao_marca_todas_as_amostras():
    """Q com poucos valores enormes (espectros saturados) fazia o ajuste chi2 por
    momentos desabar: limite ~0 e 100 % das amostras 'outlier Q'."""
    from guaraci.chemometric_stats import q_residuals_limit
    rng = np.random.default_rng(1)
    q = rng.lognormal(-10, 4.0, 1500)
    lim = q_residuals_limit(q)
    assert np.mean(q > lim) <= 0.06
    assert lim >= np.percentile(q, 94)


def test_limite_q_comportado_continua_chi2():
    from guaraci.chemometric_stats import q_residuals_limit
    rng = np.random.default_rng(2)
    q = rng.chisquare(8, 800) * 1e-4
    lim = q_residuals_limit(q)
    assert 0.02 <= np.mean(q > lim) <= 0.09
