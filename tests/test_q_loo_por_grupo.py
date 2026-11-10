"""M-04a do parecer de 2026-09-25: o Q de treino tem que ser leave-one-GROUP-out.

Antes, `q_residuals_loo` deixava fora UM espectro: as replicas irmas ficavam no
ajuste da PCA e o DD-SIMCA aceitava so' 0,62 das amostras puras NOVAS (3 replicas,
10 grupos) contra 0,80 alcancavel. Versao automatizada de `revisor3/s2b.py`.
"""
import numpy as np
import pytest

from guaraci.chemometric_stats import q_residuals_loo
from guaraci.classificadores import DDSimca


def _dados(rng, n_grupos, n_rep, p=40, k=5):
    L = np.linalg.qr(rng.normal(size=(p, k)))[0]
    esc = np.linspace(3, 1, k)
    X, g = [], []
    for i in range(n_grupos):
        z = rng.normal(size=k) * esc
        for _ in range(n_rep):
            X.append((z + rng.normal(size=k) * esc * 0.05) @ L.T + 0.05 * rng.normal(size=p))
            g.append(i)
    return np.asarray(X), np.asarray(g)


def test_q_por_grupo_usa_pca_sem_nenhuma_irma_do_grupo():
    rng = np.random.default_rng(0)
    X, g = _dados(rng, 8, 3)
    q_grupo = q_residuals_loo(X, 3, groups=g)
    q_loo = q_residuals_loo(X, 3)
    # Com irmas no ajuste (LOO por espectro) o residuo e' MENOR (otimista)
    assert q_grupo.mean() > q_loo.mean()
    # Manualmente: PCA sem o grupo 0 reconstroi as amostras do grupo 0
    from sklearn.decomposition import PCA
    fora = g == 0
    pca = PCA(n_components=3).fit(X[~fora])
    esperado = np.sum((X[fora] - pca.inverse_transform(pca.transform(X[fora]))) ** 2, axis=1)
    np.testing.assert_allclose(q_grupo[fora], esperado)


def test_sem_groups_e_leave_one_out_como_antes():
    rng = np.random.default_rng(1)
    X, _g = _dados(rng, 6, 2)
    np.testing.assert_allclose(q_residuals_loo(X, 2),
                               q_residuals_loo(X, 2, groups=np.arange(len(X))))


def test_groups_de_tamanho_errado_e_recusado():
    with pytest.raises(ValueError, match="groups"):
        q_residuals_loo(np.zeros((5, 4)), 2, groups=np.arange(3))


def test_ddsimca_aceita_perto_do_nominal_alcancavel_em_amostras_novas():
    """3 replicas por amostra, 10 amostras de calibracao: a aceitacao de
    amostras puras NOVAS fica perto de 0,80 (antes: 0,62)."""
    acc = []
    for it in range(40):
        rng = np.random.default_rng(31 + it)
        L = np.linalg.qr(rng.normal(size=(40, 5)))[0]
        esc = np.linspace(3, 1, 5)

        def amostra(ng):
            X, g = [], []
            for i in range(ng):
                z = rng.normal(size=5) * esc
                for _ in range(3):
                    X.append((z + rng.normal(size=5) * esc * 0.0) @ L.T
                             + 0.05 * rng.normal(size=40))
                    g.append(i)
            return np.asarray(X), np.asarray(g)
        Xc, gc = amostra(10)
        Xn, _ = amostra(50)
        m = DDSimca(n_components=3).fit(Xc, np.array(["c"] * len(Xc)), mae_id=gc.astype(str))
        sm = m.score_matrix(Xn)["c"]
        acc.append(np.mean(sm["f"] <= sm["f_crit"]))
    assert np.mean(acc) > 0.75, f"aceitacao de puras novas = {np.mean(acc):.3f}"


def test_um_unico_grupo_nao_degenera_o_limite_para_zero():
    """Regressao achada pelo golden numerico: com UM grupo (ex.: 3 replicas de
    uma so' amostra pura) tirar o grupo inteiro deixa o ajuste vazio -> Q=0 para
    todos -> limite 0 -> o DD-SIMCA rejeitava ate' o proprio treino."""
    import numpy as np
    from guaraci.chemometric_stats import q_residuals_loo
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 20))
    q = q_residuals_loo(X, 2, groups=np.array(["G", "G", "G"]))
    assert np.all(q > 0), q
    assert np.allclose(q, q_residuals_loo(X, 2, groups=None))
