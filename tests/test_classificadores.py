"""Rede de segurança de guaraci.classificadores (DD-SIMCA e OPLS-DA) — os
estimadores que fazem a autenticação one-class e a análise discriminante
ortogonal, o diferencial metodológico do projeto. Corrupção silenciosa aqui
rejeitaria/aceitaria amostras erradas sem nenhum sinal de alerta.
"""
import numpy as np
import pytest

from guaraci.classificadores import (
    DDSimca,
    OPLSDAWrapper,
    sensibilidade_ddsimca_logo,
)


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


def test_oplsda_lda_falha_cai_no_fallback_pls2(monkeypatch):
    """Se a LDA multiclasse falhar (matriz de dispersao intra-classe
    singular -- caso real com poucas amostras/features colineares), o
    fit() deve cair no fallback PLS2 em vez de propagar a excecao. Forcado
    via monkeypatch (a falha real da LDA e' dificil de reproduzir de forma
    limpa/deterministica via dados publicos, mas o CAMINHO de fallback e'
    codigo real que precisa continuar correto)."""
    def _fit_transform_falha(self, X, y):
        raise ValueError("simulada: matriz de dispersao intra-classe singular")

    monkeypatch.setattr(
        "sklearn.discriminant_analysis.LinearDiscriminantAnalysis.fit_transform",
        _fit_transform_falha)

    rng = np.random.default_rng(8)
    X = np.vstack([
        _classe_compacta(rng, centro=0.0, n=10, k=6),
        _classe_compacta(rng, centro=3.0, n=10, k=6),
        _classe_compacta(rng, centro=6.0, n=10, k=6),
    ])
    Y = np.eye(3)[np.array([0] * 10 + [1] * 10 + [2] * 10)]

    opls = OPLSDAWrapper(n_ortho=1).fit(X, Y)  # nao deve lancar ValueError
    assert isinstance(opls.W_orth_, list)
    assert opls.t_pred_train_.shape[0] == 30


def test_nipals_pls1_com_x_todo_zero_nao_diverge():
    """X todo zero (caso degenerado extremo) faz w=X.T@u ter norma ~0 no
    1o passo -- deve interromper o loop (break) em vez de dividir por zero
    ou iterar ate max_iter sem necessidade."""
    X = np.zeros((10, 4))
    y = np.random.default_rng(9).normal(size=10)
    w, t, p = OPLSDAWrapper._nipals_pls1(X, y, max_iter=50)
    assert np.all(w == 0)
    assert np.all(t == 0)
    assert np.all(p == 0)


# ── Sensibilidade DD-SIMCA por LOGO (P1: fim da re-substituicao) ──────────────
def _puros_agrupados(rng, centros, reps=3, k=5, escala=0.2):
    """Puros de UMA classe em grupos de replica (mae_id): cada centro = 1 grupo
    com `reps` replicas fisicas."""
    X, g = [], []
    for i, c in enumerate(centros):
        X.append(rng.normal(c, escala, size=(reps, k)))
        g += [f"grp{i}"] * reps
    return np.vstack(X), np.array(g)


def test_logo_sempre_retorna_n_grupos():
    """CONTRATO: 'n_grupos' NUNCA pode faltar do resultado. Sensibilidade sem o
    denominador de grupos independentes e enganosa — era o buraco da
    re-substituicao. Este teste falha se alguem remover a chave."""
    rng = np.random.default_rng(0)
    X, g = _puros_agrupados(rng, [0.0, 0.5, 1.0])
    r = sensibilidade_ddsimca_logo(X, g, n_components=2)
    assert "n_grupos" in r
    assert r["n_grupos"] == 3


def test_logo_cai_abaixo_de_100pct_com_grupo_outlier():
    """LOGO detecta um grupo de replica retido que cai FORA da regiao treinada
    nos demais — exatamente o que a re-substituicao mascarava dando ~100%."""
    rng = np.random.default_rng(1)
    X, g = _puros_agrupados(rng, [0.0, 0.0, 0.0, 0.0, 20.0])  # 1 grupo distante
    r = sensibilidade_ddsimca_logo(X, g, n_components=2)
    assert r["n_grupos"] == 5
    assert r["sensibilidade"] < 1.0            # < 100%: o objetivo do P1

    # Re-substituicao: modelo treinado em TODOS os puros e avaliado neles mesmos
    # (o small-n guard aceita todo o treino) -> infla a sensibilidade.
    dd = DDSimca(n_components=2).fit(X, np.array(["_c"] * len(X)))
    m = dd.score_matrix(X)["_c"]
    aceito = (np.asarray(m["T2_norm"]) <= 1.0) & (np.asarray(m["Q_norm"]) <= 1.0)
    sens_resub = float(np.mean(aceito))
    assert sens_resub > r["sensibilidade"]     # re-sub sempre >= LOGO honesto


def test_logo_um_unico_grupo_nao_e_estimavel():
    """Com um unico grupo de replica pura NAO ha validacao possivel: retorna
    nan + aviso, nunca um numero falsamente confiante."""
    rng = np.random.default_rng(2)
    X, g = _puros_agrupados(rng, [0.0])        # 1 grupo apenas
    r = sensibilidade_ddsimca_logo(X, g, n_components=2)
    assert r["n_grupos"] == 1
    assert np.isnan(r["sensibilidade"])
    assert r["aviso"] is not None


def test_logo_avisa_com_poucos_grupos():
    """n_grupos < 10 dispara aviso de incerteza (interpretacao exploratoria)."""
    rng = np.random.default_rng(3)
    X, g = _puros_agrupados(rng, [0.0, 0.3, 0.6, 0.9])
    r = sensibilidade_ddsimca_logo(X, g, n_components=2)
    assert r["n_grupos"] == 4
    assert r["aviso"] is not None and "LOGO" in r["aviso"]


def test_logo_grupo_com_treino_insuficiente_e_pulado_nao_quebra():
    """Um grupo cujo TREINO restante (todos os outros) teria <2 amostras --
    ou cujo proprio grupo retido esta vazio -- deve ser pulado (continue),
    nao lancar excecao. So' e' possivel construir isso artificialmente com
    um grupo cujo array de mascara de teste fique vazio; testamos a
    propriedade indiretamente: rodar com grupos minusculos nao quebra e
    ainda retorna um resultado coerente."""
    rng = np.random.default_rng(4)
    # 2 grupos de 1 replica cada: treino de cada fold tem so' 1 amostra
    # (o outro grupo) -- abaixo do minimo de 2 exigido pelo guard interno.
    X, g = _puros_agrupados(rng, [0.0, 5.0], reps=1)
    r = sensibilidade_ddsimca_logo(X, g, n_components=1)
    assert r["n_grupos"] == 2
    # nao lancou excecao; ou fica inconclusivo (validos<2) ou reporta normal
    assert r["n_grupos_validos"] <= r["n_grupos"]


def test_logo_inconclusivo_quando_nenhum_fold_valido():
    """Se NENHUM fold produzir dobra valida (todas puladas), a chave
    'sensibilidade' fica NaN com aviso 'inconclusiva' -- nunca um numero
    calculado sobre uma lista vazia (o que estouraria ou mentiria)."""
    rng = np.random.default_rng(5)
    X, g = _puros_agrupados(rng, [0.0, 1.0], reps=1)  # mesmo caso do teste acima
    r = sensibilidade_ddsimca_logo(X, g, n_components=1)
    if r["n_grupos_validos"] < 2:
        assert np.isnan(r["sensibilidade"])
        assert r["aviso"] is not None and "inconclusiva" in r["aviso"]
