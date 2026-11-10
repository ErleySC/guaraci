"""VIP com estabilidade: oraculo independente (PLS1 por NIPALS na mao), propriedades e dados sinteticos."""
import numpy as np
import pytest
from hypothesis import example, given, settings, strategies as st
from sklearn.cross_decomposition import PLSRegression

from guaraci.chemometric_stats import vip_scores
from guaraci.vip_estabilidade import (
    FREQ_ESTAVEL, FREQ_INSTAVEL, TEXTO_VIP, figura_vip_estabilidade, vip_estabilidade)


def _dados(n_grupos=24, rep=3, p=12, informativas=(0, 1, 2), seed=0, ruido=0.6):
    """Regressao: y depende so' das variaveis `informativas`; replicas por grupo (mae)."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_grupos), rep)
    base = rng.normal(0, 1, (n_grupos, p))
    X = base[g] + rng.normal(0, 0.05, (len(g), p))
    y = X[:, list(informativas)].sum(axis=1) + rng.normal(0, ruido, len(g))
    return X, y, g.astype(str)


def _vip_nipals(Xc, y, A):
    """Oraculo: PLS1 por NIPALS e VIP pela formula de Wold, em lacos explicitos."""
    X = Xc.copy()
    yy = y - y.mean()
    p = X.shape[1]
    W, ss = [], []
    for _ in range(A):
        w = X.T @ yy
        w /= np.linalg.norm(w)
        t = X @ w
        tt = t @ t
        pl = X.T @ t / tt
        q = (yy @ t) / tt
        W.append(w)
        ss.append(q * q * tt)
        X = X - np.outer(t, pl)
        yy = yy - t * q
    return np.array([np.sqrt(p * sum(ss[k] * W[k][j] ** 2 for k in range(A)) / sum(ss)) for j in range(p)])


@pytest.mark.parametrize("A", [1, 2, 4])
def test_vip_confere_com_pls1_nipals_feito_a_mao(A):
    X, y, _ = _dados(seed=3)
    Xc = (X - X.mean(0)) / X.std(0, ddof=1)
    pls = PLSRegression(n_components=A, scale=False).fit(Xc, y.reshape(-1, 1))
    np.testing.assert_allclose(vip_scores(pls), _vip_nipals(Xc, y, A), rtol=1e-8)


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 10_000), st.integers(1, 3), st.integers(4, 15))
@example(0, 1, 4)
def test_soma_de_vip_ao_quadrado_e_o_numero_de_variaveis(seed, A, p):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(30, p))
    y = X[:, 0] + rng.normal(0, 0.5, 30)
    A = min(A, p)
    pls = PLSRegression(n_components=A, scale=False).fit(X - X.mean(0), y.reshape(-1, 1))
    assert float((vip_scores(pls) ** 2).sum()) == pytest.approx(p, rel=1e-8)


def test_informativas_tem_vip_alto_e_estavel_e_ruido_nao():
    X, y, g = _dados(seed=1)
    r = vip_estabilidade(X, y, 2, g, tarefa="regressao", n_repeticoes=6, seed=1)
    inf, ruido = [0, 1, 2], list(range(3, 12))
    assert np.all(r["frequencia"][inf] >= FREQ_ESTAVEL)
    assert np.all(r["classe"][inf] == "estavel_acima")
    assert r["media"][inf].min() > r["media"][ruido].max()
    assert np.all(r["frequencia"][ruido] <= 1 - FREQ_ESTAVEL + 0.3)
    assert r["soma_vip2"] == pytest.approx(12, rel=1e-8)
    assert np.all(r["ic95_inf"] <= r["media"] + 1e-12) and np.all(r["media"] <= r["ic95_sup"] + 1e-12)
    assert r["n_ajustes"] == 5 * 6 and r["n_falhos"] == 0 and r["n_grupos"] == 24
    assert r["texto"] == TEXTO_VIP and "regra pratica" in TEXTO_VIP and "aninhada" in TEXTO_VIP
    assert 0 <= FREQ_INSTAVEL < FREQ_ESTAVEL <= 1


def test_classificacao_pls_da_acha_a_variavel_que_separa_as_classes():
    rng = np.random.default_rng(5)
    n_g, rep, p = 30, 2, 10
    classe_g = np.array(["a", "b", "c"] * 10)
    g = np.repeat(np.arange(n_g), rep)
    X = rng.normal(0, 1, (n_g, p))[g] + rng.normal(0, 0.05, (n_g * rep, p))
    desloc = {"a": -3.0, "b": 0.0, "c": 3.0}
    X[:, 4] += np.array([desloc[classe_g[i]] for i in g])
    y = classe_g[g]
    r = vip_estabilidade(X, y, 2, g.astype(str), tarefa="classificacao", n_repeticoes=4)
    assert int(np.argmax(r["media"])) == 4 and r["frequencia"][4] == 1.0


def test_invariancia_a_ordem_das_amostras_e_das_variaveis():
    X, y, g = _dados(seed=2)
    r0 = vip_estabilidade(X, y, 2, g, tarefa="regressao", n_repeticoes=3)
    idx = np.random.default_rng(0).permutation(len(y))
    r1 = vip_estabilidade(X[idx], y[idx], 2, g[idx], tarefa="regressao", n_repeticoes=3)
    np.testing.assert_allclose(r1["vip"], r0["vip"], rtol=1e-8)
    np.testing.assert_allclose(r1["frequencia"], r0["frequencia"])
    np.testing.assert_allclose(r1["media"], r0["media"], rtol=1e-8)
    col = np.random.default_rng(1).permutation(X.shape[1])
    r2 = vip_estabilidade(X[:, col], y, 2, g, tarefa="regressao", n_repeticoes=3)
    np.testing.assert_allclose(r2["vip"], r0["vip"][col], rtol=1e-8)
    np.testing.assert_allclose(r2["frequencia"], r0["frequencia"][col])


def test_determinismo_e_a_semente_muda_so_as_dobras():
    X, y, g = _dados(seed=4)
    a = vip_estabilidade(X, y, 2, g, tarefa="regressao", n_repeticoes=3, seed=7)
    b = vip_estabilidade(X, y, 2, g, tarefa="regressao", n_repeticoes=3, seed=7)
    c = vip_estabilidade(X, y, 2, g, tarefa="regressao", n_repeticoes=3, seed=8)
    np.testing.assert_array_equal(a["media"], b["media"])
    np.testing.assert_array_equal(a["vip"], c["vip"])             # VIP completo nao depende da semente


def test_sem_grupos_avisa_e_garantia_baixa_avisa():
    X, y, _ = _dados(seed=1)
    r = vip_estabilidade(X, y, 2, None, tarefa="regressao", n_repeticoes=2, grouping_guarantee="medium")
    texto = " ".join(r["avisos"])
    assert "sem coluna de grupo" in texto and "garantia de agrupamento 'medium'" in texto


def test_poucos_grupos_reduz_as_divisoes_e_um_grupo_e_erro():
    X, y, g = _dados(n_grupos=4, rep=6, seed=1)
    r = vip_estabilidade(X, y, 1, g, tarefa="regressao", n_divisoes=10, n_repeticoes=2)
    assert r["n_divisoes"] == 4 and any("reduzido de 10 para 4" in a for a in r["avisos"])
    with pytest.raises(ValueError, match="menos de 2 grupos"):
        vip_estabilidade(X, y, 1, ["g"] * len(y), tarefa="regressao")


@pytest.mark.parametrize("kw, trecho", [
    (dict(tarefa="outra"), "tarefa"),
    (dict(n_componentes=0), "n_componentes"),
    (dict(n_componentes=99), "n_componentes"),
    (dict(n_divisoes=1), "n_divisoes"),
    (dict(n_repeticoes=0), "n_repeticoes"),
])
def test_entradas_invalidas_recusam(kw, trecho):
    X, y, g = _dados(seed=1)
    args = dict(n_componentes=2, tarefa="regressao")
    args.update(kw)
    nc = args.pop("n_componentes")
    with pytest.raises(ValueError, match=trecho):
        vip_estabilidade(X, y, nc, g, **args)


def test_dados_ruins_recusam():
    X, y, g = _dados(seed=1)
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        vip_estabilidade(Xn, y, 2, g, tarefa="regressao")
    with pytest.raises(ValueError, match="constante"):
        vip_estabilidade(X, np.ones(len(y)), 2, g, tarefa="regressao")
    yn = y.copy()
    yn[0] = np.inf
    with pytest.raises(ValueError, match="NaN"):
        vip_estabilidade(X, yn, 2, g, tarefa="regressao")
    with pytest.raises(ValueError, match="2 classes"):
        vip_estabilidade(X, ["a"] * len(y), 2, g, tarefa="classificacao")
    with pytest.raises(ValueError, match="linhas"):
        vip_estabilidade(X, y[:-1], 2, g, tarefa="regressao")
    with pytest.raises(ValueError, match="grupos tem"):
        vip_estabilidade(X, y, 2, g[:-1], tarefa="regressao")
    with pytest.raises(ValueError, match=">= 4 linhas"):
        vip_estabilidade(X[:3], y[:3], 1, g[:3], tarefa="regressao")


def test_replicas_do_mesmo_grupo_nunca_ficam_separadas_nas_dobras():
    """Garantia herdada do StableStratifiedGroupKFold: o ajuste de cada dobra de
    treino nunca ve' replicas do grupo que ficou de fora."""
    from guaraci.validacao_estatistica import StableStratifiedGroupKFold
    X, y, g = _dados(seed=1)
    _, estrato = np.unique(np.digitize(y, np.quantile(y, [1 / 3, 2 / 3])), return_inverse=True)
    for tr, va in StableStratifiedGroupKFold(5, seed=0).split(X, estrato, groups=g):
        assert not set(g[tr]) & set(g[va])


def test_figura_de_barras_e_gerada_para_poucas_e_muitas_variaveis(tmp_path):
    X, y, g = _dados(seed=1)
    r = vip_estabilidade(X, y, 2, g, tarefa="regressao", n_repeticoes=2)
    f = figura_vip_estabilidade(r, str(tmp_path / "a.png"), nomes=[f"v{i}" for i in range(12)])
    assert open(f, "rb").read(4) == b"\x89PNG"
    rng = np.random.default_rng(0)
    Xs = rng.normal(size=(40, 200))
    ys = Xs[:, 5] + rng.normal(0, 0.3, 40)
    r2 = vip_estabilidade(Xs, ys, 2, np.arange(40).astype(str), tarefa="regressao", n_repeticoes=1)
    assert open(figura_vip_estabilidade(r2, str(tmp_path / "b.png")), "rb").read(4) == b"\x89PNG"


def test_estratos_da_regressao_sao_tercos_e_o_alvo_da_classificacao_e_one_hot():
    from guaraci.vip_estabilidade import _alvo
    y = np.arange(30, dtype=float)
    Y, estrato = _alvo(y, "regressao")
    assert Y.shape == (30, 1)
    assert np.bincount(estrato).tolist() == [10, 10, 10]               # terco inferior, medio, superior
    assert list(estrato[:10]) == [0] * 10 and list(estrato[-10:]) == [2] * 10
    Yc, e = _alvo(["b", "a", "c", "a"], "classificacao")
    assert Yc.tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]] and list(e) == [1, 0, 2, 0]
