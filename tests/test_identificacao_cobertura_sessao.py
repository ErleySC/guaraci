"""M-04b do parecer de 2026-09-25: identificacao conforme calibrada e decidida na
MESMA unidade, com centroide leave-one-session-out e cobertura MEDIDA.

Versao automatizada (reduzida) de `revisor3/simulacoes.py::S3`: com 25 sessoes de
calibracao a cobertura por espectro de sessoes NOVAS era 0,854 contra 0,95
nominal (centroide in-sample + mediana por sessao, decisao por espectro).
"""
import numpy as np
from sklearn.decomposition import PCA

from guaraci.conformal import (ConformalOneClass, conformal_threshold,
                               conformal_threshold_agrupado)
from guaraci.identificacao import (CoverageStatus, cobertura_empirica_loso,
                                   escores_loso, train_identification_ensemble)


def _sessao(rng, L, esc, i_s, k=4, p=40):
    z_s = rng.normal(size=k) * esc
    linhas, maes, concs = [], [], []
    for c in (2.0, 5.0, 10.0, 20.0):
        z_m = z_s + rng.normal(size=k) * esc * 0.3
        for _ in range(3):
            z = z_m + rng.normal(size=k) * esc * 0.3
            linhas.append(z @ L.T + 0.05 * rng.normal(size=p))
            maes.append(f"ESA-{i_s:03d}-01-2099-A{c:.2f}")
            concs.append(c)
    return linhas, maes, concs


def _cenario(seed, n_sess_cal=25, n_sess_novas=30):
    rng = np.random.default_rng(seed)
    L = np.linalg.qr(rng.normal(size=(40, 4)))[0]
    esc = np.array([3.0, 2.0, 1.5, 1.0])
    X, M, C = [], [], []
    for s in range(n_sess_cal):
        a, b, c = _sessao(rng, L, esc, s)
        X += a; M += b; C += c
    Xn, Mn = [], []
    for s in range(n_sess_novas):
        a, b, _ = _sessao(rng, L, esc, 500 + s)
        Xn += a; Mn += b
    return (np.asarray(X), np.asarray(M), np.asarray(C),
            np.asarray(Xn), np.asarray(Mn))


def test_cobertura_por_espectro_de_sessoes_novas_fica_no_nominal():
    cobert, status = [], set()
    for seed in range(12):
        X, M, C, Xn, _Mn = _cenario(seed)
        pca = PCA(n_components=4).fit(X)
        T = pca.transform(X)
        var_t = T.var(axis=0, ddof=1)
        ens = train_identification_ensemble(
            pca, var_t, X, np.array(["ESA"] * len(X)), C, M,
            mapa_adulterante={"A": "adulterante A"})
        info = ens[("ESA", "adulterante A")]
        status.add(info["cobertura_status"])
        sc = np.sum((pca.transform(Xn) - info["centroide"]) ** 2 / var_t, axis=1)
        cobert.append(float(np.mean(info["conformal"].predict(sc))))
    assert status == {CoverageStatus.VALIDATED}
    assert np.mean(cobert) >= 0.93, (
        f"cobertura de sessoes novas = {np.mean(cobert):.3f} (nominal 0,95; "
        "antes da correcao: 0,854)")


def test_o_ensemble_registra_a_cobertura_empirica_medida():
    X, M, C, _Xn, _Mn = _cenario(0)
    pca = PCA(n_components=4).fit(X)
    var_t = pca.transform(X).var(axis=0, ddof=1)
    info = train_identification_ensemble(
        pca, var_t, X, np.array(["ESA"] * len(X)), C, M,
        mapa_adulterante={"A": "adulterante A"})[("ESA", "adulterante A")]
    assert info["cobertura_empirica"] is not None
    assert 0.85 <= info["cobertura_empirica"] <= 1.0
    assert info["cobertura_empirica_se"] >= 0.0


def test_status_validado_exige_cobertura_medida(monkeypatch):
    """Se a cobertura empirica em sessoes nao vistas fica abaixo do nominal, o
    status NAO e' VALIDADO, mesmo com n suficiente para o alpha."""
    import guaraci.identificacao as ident
    monkeypatch.setattr(ident, "cobertura_empirica_loso",
                        lambda *a, **k: (0.60, 0.01, 25))
    X, M, C, _Xn, _Mn = _cenario(1)
    pca = PCA(n_components=4).fit(X)
    var_t = pca.transform(X).var(axis=0, ddof=1)
    info = ident.train_identification_ensemble(
        pca, var_t, X, np.array(["ESA"] * len(X)), C, M,
        mapa_adulterante={"A": "adulterante A"})[("ESA", "adulterante A")]
    assert info["cobertura_status"] == CoverageStatus.NOT_VALIDATED_COVERAGE


def test_escores_loso_sao_maiores_que_os_do_centroide_in_sample():
    X, M, C, _Xn, _Mn = _cenario(2, n_sess_cal=8)
    sess = np.array(["-".join(m.split("-")[:-1]) for m in M])
    pca = PCA(n_components=4).fit(X)
    T = pca.transform(X)
    var_t = T.var(axis=0, ddof=1)
    in_sample = np.sum((T - T.mean(axis=0)) ** 2 / var_t, axis=1)
    assert escores_loso(T, sess, var_t).mean() > in_sample.mean()


def test_limiar_agrupado_com_um_escore_por_grupo_e_o_conformal_padrao():
    rng = np.random.default_rng(0)
    s = rng.normal(size=30)
    g = np.arange(30)
    a = conformal_threshold_agrupado(s, g, 0.1)
    b = conformal_threshold(s, 0.1)
    assert a["limiar"] == b["limiar"] and a["alcancavel"] and b["alcancavel"]


def test_limiar_agrupado_da_peso_igual_a_cada_grupo():
    """Um grupo com 100 replicas altas nao pode decidir sozinho o limiar."""
    s = np.concatenate([np.full(100, 9.0), np.arange(1.0, 20.0)])
    g = np.concatenate([np.zeros(100, int), np.arange(1, 20)])
    r = conformal_threshold_agrupado(s, g, 0.1)
    # 20 grupos, massa ceil(21*0.9)=19: o grupo gordo vale 1 (nao 100)
    assert r["alcancavel"] and r["limiar"] == 18.0
    # Sem pesos (espectro a espectro) o grupo gordo domina e o limiar cai em 9.0
    assert conformal_threshold(s, 0.1)["limiar"] == 9.0


def test_alpha_inalcancavel_com_poucos_grupos_nao_aceita_tudo():
    s = np.arange(10.0)
    g = np.repeat(np.arange(5), 2)
    r = conformal_threshold_agrupado(s, g, 0.05)
    assert not r["alcancavel"] and np.isnan(r["limiar"])
    c = ConformalOneClass(0.05, unidade="espectro").fit(s, mae_id=g)
    assert not c.predict(np.array([0.0, 100.0])).any()


def test_cobertura_empirica_loso_iid_perto_do_nominal():
    rng = np.random.default_rng(5)
    T = rng.normal(size=(25 * 12, 3))
    grupos = np.repeat(np.arange(25), 12)
    cob, se, n = cobertura_empirica_loso(T, grupos, np.ones(3), 0.05)
    assert n == 25 and 0.85 < cob <= 1.0 and se >= 0
