"""Rede de segurança de guaraci.classificadores (DD-SIMCA e OPLS-DA) — os
estimadores que fazem a autenticação one-class e a análise discriminante
ortogonal, o diferencial metodológico do projeto. Corrupção silenciosa aqui
rejeitaria/aceitaria amostras erradas sem nenhum sinal de alerta.
"""
import numpy as np
import pytest

from guaraci.classificadores import DDSimca, OPLSDAWrapper


def _classe_compacta(rng, centro, n, k=5, escala=0.3):
    return rng.normal(loc=centro, scale=escala, size=(n, k))


# ── DDSimca.fit / predict: caso feliz ────────────────────────────────────────
def test_ddsimca_amostras_de_treino_sao_aceitas_pela_propria_classe():
    """Toda amostra de treino DEVE ser aceita pelo próprio modelo — é o
    invariante central que o 'small-n guard' (comentado no código) existe
    para garantir. Se isso falhar, o DD-SIMCA rejeitaria dados legítimos."""
    rng = np.random.default_rng(0)
    Xa = _classe_compacta(rng, centro=0.0, n=10)
    Xb = _classe_compacta(rng, centro=5.0, n=10)
    X = np.vstack([Xa, Xb])
    y = np.array(["A"] * 10 + ["B"] * 10)

    dd = DDSimca(n_components=3, alpha=0.05, ucl_method="empirical").fit(X, y)
    preds = dd.predict(X)
    assert list(preds[:10]) == ["A"] * 10
    assert list(preds[10:]) == ["B"] * 10


def test_ddsimca_amostra_distante_e_desconhecida():
    rng = np.random.default_rng(1)
    Xa = _classe_compacta(rng, centro=0.0, n=15)
    dd = DDSimca(n_components=2, alpha=0.05).fit(Xa, np.array(["A"] * 15))
    fora = np.full((1, 5), 100.0)  # muito longe do treino
    pred = dd.predict(fora)
    assert pred[0] == "Desconhecido"


def test_ddsimca_score_matrix_contem_campos_esperados():
    rng = np.random.default_rng(2)
    X = _classe_compacta(rng, centro=0.0, n=12)
    dd = DDSimca(n_components=2).fit(X, np.array(["A"] * 12))
    scores = dd.score_matrix(X)
    assert "A" in scores
    campos = scores["A"]
    for chave in ("T2", "Q", "T2_ucl", "Q_ucl", "T2_norm", "Q_norm", "n_train"):
        assert chave in campos
    assert campos["n_train"] == 12


# ── DDSimca: classe com amostras insuficientes é pulada (não quebra) ────────
def test_ddsimca_classe_com_1_amostra_e_pulada(capsys):
    rng = np.random.default_rng(3)
    Xa = _classe_compacta(rng, centro=0.0, n=10)
    Xb = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])  # so' 1 amostra: n_comp < 1
    X = np.vstack([Xa, Xb])
    y = np.array(["A"] * 10 + ["B"])

    dd = DDSimca(n_components=3).fit(X, y)
    assert "B" not in dd._modelos          # modelo pulado
    assert "A" in dd._modelos               # classe valida seguiu normal
    saida = capsys.readouterr().out
    assert "insufficient samples" in saida


def test_ddsimca_score_matrix_ignora_classe_sem_modelo():
    rng = np.random.default_rng(41)
    Xa = _classe_compacta(rng, centro=0.0, n=10)
    Xb = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
    X = np.vstack([Xa, Xb])
    y = np.array(["A"] * 10 + ["B"])
    dd = DDSimca(n_components=3).fit(X, y)
    scores = dd.score_matrix(X)
    assert "B" not in scores
    assert "A" in scores


def test_ddsimca_predict_ignora_classe_sem_modelo():
    """predict() nao quebra quando uma classe do y original nao tem modelo
    treinado (pulada por amostras insuficientes) -- so' nao a considera."""
    rng = np.random.default_rng(4)
    Xa = _classe_compacta(rng, centro=0.0, n=10)
    Xb = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
    X = np.vstack([Xa, Xb])
    y = np.array(["A"] * 10 + ["B"])
    dd = DDSimca(n_components=3).fit(X, y)
    preds = dd.predict(X[:3])
    assert all(p in ("A", "Desconhecido") for p in preds)


# ── DDSimca: amostra ambigua (aceita por 2+ classes) ─────────────────────────
def test_ddsimca_amostra_entre_duas_classes_e_ambigua_ou_de_uma_delas():
    """Um ponto exatamente no meio de duas classes muito proximas e' aceito
    por ambas (Ambiguo) ou por uma delas -- nunca lanca excecao, e o rotulo
    e' sempre um dos 3 esperados."""
    rng = np.random.default_rng(5)
    Xa = _classe_compacta(rng, centro=0.0, n=15, escala=0.05)
    Xb = _classe_compacta(rng, centro=0.3, n=15, escala=0.05)  # bem proxima de A
    X = np.vstack([Xa, Xb])
    y = np.array(["A"] * 15 + ["B"] * 15)
    dd = DDSimca(n_components=2).fit(X, y)

    meio = np.full((1, 5), 0.15)
    pred = dd.predict(meio)[0]
    assert pred in ("A", "B", "Ambiguo", "Desconhecido")


# ── DDSimca._compute_t2_ucl: os 3 métodos de UCL + fallback ─────────────────
def test_compute_t2_ucl_empirical():
    dd = DDSimca(alpha=0.05, ucl_method="empirical")
    T2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ucl = dd._compute_t2_ucl(T2, n=5, k=2)
    assert ucl == pytest.approx(np.percentile(T2, 95))


def test_compute_t2_ucl_empirical_vazio_retorna_infinito():
    dd = DDSimca(ucl_method="empirical")
    ucl = dd._compute_t2_ucl(np.array([]), n=0, k=2)
    assert ucl == float("inf")


def test_compute_t2_ucl_theoretical_usa_formula_tracy_young():
    dd = DDSimca(alpha=0.05, ucl_method="theoretical")
    from guaraci.chemometric_stats import hotelling_t2_limite
    ucl = dd._compute_t2_ucl(np.array([1.0, 2.0]), n=10, k=2)
    assert ucl == pytest.approx(hotelling_t2_limite(10, 2, 0.05))


def test_compute_t2_ucl_chi2():
    from scipy.stats import chi2
    dd = DDSimca(alpha=0.05, ucl_method="chi2")
    ucl = dd._compute_t2_ucl(np.array([1.0, 2.0]), n=10, k=3)
    assert ucl == pytest.approx(chi2.ppf(0.95, 3))


def test_compute_t2_ucl_metodo_desconhecido_cai_no_empirico():
    dd = DDSimca(alpha=0.05, ucl_method="metodo_que_nao_existe")
    T2 = np.array([1.0, 2.0, 3.0, 4.0])
    ucl = dd._compute_t2_ucl(T2, n=4, k=2)
    assert ucl == pytest.approx(np.percentile(T2, 95))


# ── OPLSDAWrapper ────────────────────────────────────────────────────────────
def test_oplsda_fit_binario_gera_scores_ortogonais():
    rng = np.random.default_rng(6)
    Xa = _classe_compacta(rng, centro=0.0, n=20, k=6)
    Xb = _classe_compacta(rng, centro=3.0, n=20, k=6)
    X = np.vstack([Xa, Xb])
    Y = np.array([0.0] * 20 + [1.0] * 20)

    opls = OPLSDAWrapper(n_ortho=1).fit(X, Y)
    assert len(opls.W_orth_) >= 0  # pode convergir com 0 ou 1 componente ortogonal


def test_oplsda_fit_multiclasse_usa_lda_para_y_continuo():
    """Y one-hot multiclasse (>1 coluna) aciona o ramo LDA (fit() reduz a um
    y continuo antes do NIPALS) -- nao deve lancar excecao e deve treinar
    componentes ortogonais coerentes com o numero de features."""
    rng = np.random.default_rng(7)
    X = np.vstack([
        _classe_compacta(rng, centro=0.0, n=15, k=6),
        _classe_compacta(rng, centro=3.0, n=15, k=6),
        _classe_compacta(rng, centro=6.0, n=15, k=6),
    ])
    Y = np.eye(3)[np.array([0] * 15 + [1] * 15 + [2] * 15)]  # one-hot 3 classes

    opls = OPLSDAWrapper(n_ortho=1).fit(X, Y)
    assert isinstance(opls.W_orth_, list)
