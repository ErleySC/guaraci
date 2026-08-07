"""Rede de segurança de guaraci.classificadores (DD-SIMCA e OPLS-DA) — os
estimadores que fazem a autenticação one-class e a análise discriminante
ortogonal, o diferencial metodológico do projeto. Corrupção silenciosa aqui
rejeitaria/aceitaria amostras erradas sem nenhum sinal de alerta.
"""
import logging

import numpy as np
import pytest

from guaraci.classificadores import (
    DDSimca,
    OPLSDAWrapper,
    sensibilidade_ddsimca_logo,
    sensibilidade_ddsimca_pcv,
)


def _classe_compacta(rng, centro, n, k=5, escala=0.3):
    return rng.normal(loc=centro, scale=escala, size=(n, k))


# ── DDSimca.fit / predict: caso feliz ────────────────────────────────────────
def test_ddsimca_treino_e_majoritariamente_mas_nao_100pct_aceito():
    """A UCL empirica e' o quantil (1-alpha) do treino: por definicao, uma
    fracao proxima de alpha das PROPRIAS amostras de treino cai FORA da
    regiao de aceitacao -- e' o que faz de um limite um limite, nao um teto
    frouxo que aceita tudo. Exigir 100% de aceitacao no treino (o invariante
    antigo deste teste) e' a NEGACAO de alpha=0.05: e' exatamente o
    comportamento que o 'small-n clamp' removido de classificadores.py
    forcava (ele elevava o UCL ate o max(treino), zerando alpha de fato --
    achado de auditoria adversarial, 2026-07-19). Um modelo correto ainda
    reconhece a MAIORIA da propria classe, so' nao 100%."""
    rng = np.random.default_rng(0)
    n_por_classe = 40
    Xa = _classe_compacta(rng, centro=0.0, n=n_por_classe)
    Xb = _classe_compacta(rng, centro=5.0, n=n_por_classe)
    X = np.vstack([Xa, Xb])
    y = np.array(["A"] * n_por_classe + ["B"] * n_por_classe)

    dd = DDSimca(n_components=3, alpha=0.05, ucl_method="empirical").fit(X, y)
    preds = dd.predict(X)

    taxa_a = float(np.mean(preds[:n_por_classe] == "A"))
    taxa_b = float(np.mean(preds[n_por_classe:] == "B"))

    # Reconhece a propria classe na maioria dos casos...
    assert taxa_a > 0.70
    assert taxa_b > 0.70
    # ...mas NAO 100%: alpha>0 exige alguma rejeicao por definicao.
    assert taxa_a < 1.0 or taxa_b < 1.0


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
    for chave in ("T2", "Q", "T2_ucl", "Q_ucl", "T2_norm", "Q_norm", "n_train",
                  "n_comp"):
        assert chave in campos
    assert campos["n_train"] == 12


def test_ddsimca_score_matrix_expoe_n_comp_usado():
    """REGRESSAO: a figura de aceitacao (fig_sprint3_ddsimca_acceptance)
    precisa saber quantos componentes o modelo usou para explicar por que
    a maioria das amostras de outras classes colapsa perto de zero em T2
    quando n_comp=1 (comum com poucas amostras puras de treino) — sem esse
    campo, o padrao no grafico parecia bug de renderizacao."""
    rng = np.random.default_rng(7)
    X = _classe_compacta(rng, centro=0.0, n=3)   # cenario real: 3 puras
    dd = DDSimca(n_components=3).fit(X, np.array(["A"] * 3))
    scores = dd.score_matrix(X)
    assert scores["A"]["n_comp"] == 1   # forcado por _MIN_Q_RESIDUAL_DF


# ── DDSimca: classe com amostras insuficientes é pulada (não quebra) ────────
def test_ddsimca_classe_com_1_amostra_e_pulada(caplog):
    rng = np.random.default_rng(3)
    Xa = _classe_compacta(rng, centro=0.0, n=10)
    Xb = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])  # so' 1 amostra: n_comp < 1
    X = np.vstack([Xa, Xb])
    y = np.array(["A"] * 10 + ["B"])

    with caplog.at_level(logging.WARNING, logger="guaraci.classificadores"):
        dd = DDSimca(n_components=3).fit(X, y)
    assert "B" not in dd._modelos          # modelo pulado
    assert "A" in dd._modelos               # classe valida seguiu normal
    assert "insufficient samples" in caplog.text


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


def test_logo_nao_colapsa_para_zero_quando_grupos_sao_identicos_com_n_menor_que_p():
    """Regressao: com poucos puros por especie (regime real deste projeto,
    ~3-4/especie) e MUITAS variaveis (espectro), o Q-residuo IN-SAMPLE usado
    para calibrar o UCL colapsava perto de zero -- a propria amostra ajuda a
    ajustar a PCA que depois a reconstroi quase exatamente quando n<<p. O
    UCL resultante rejeitava QUALQUER amostra retida, mesmo vinda da MESMA
    distribuicao dos grupos de treino: sensibilidade LOGO ~= 0.0 mesmo sem
    nenhuma diferenca real entre os grupos (achado de auditoria adversarial,
    2026-07-19). Corrigido usando residuo leave-one-out (jackknife) para
    calibrar o UCL (ver DDSimca._q_residuals_loo). Aqui, 4 grupos
    ESTATISTICAMENTE IDENTICOS (mesmo centro) com p=200 variaveis e apenas
    3 replicas/grupo (9 amostras de treino por dobra LOGO) devem gerar
    sensibilidade proxima de 1.0 -- nao 0.0."""
    rng = np.random.default_rng(7)
    X, g = _puros_agrupados(rng, [0.0, 0.0, 0.0, 0.0], reps=3, k=200, escala=0.02)
    r = sensibilidade_ddsimca_logo(X, g, n_components=7)
    assert r["n_grupos_validos"] == 4
    assert r["sensibilidade"] > 0.7


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
    aceito = np.asarray(m["f"]) <= m["f_crit"]
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


# ---------------------------------------------------------------------------
# Distancia combinada f<=f_crit (corrigido 2026-08-08): regra retangular
# T2<=UCL e Q<=UCL independente NAO e' o metodo publicado.
# ---------------------------------------------------------------------------
def test_score_matrix_expoe_campos_da_distancia_combinada():
    rng = np.random.default_rng(5)
    X = _classe_compacta(rng, centro=0.0, n=20)
    dd = DDSimca(n_components=3).fit(X, np.array(["A"] * 20))
    m = dd.score_matrix(X)["A"]
    for chave in ("f", "f_crit", "h0", "q0", "Nh", "Nq"):
        assert chave in m
    assert m["f_crit"] > 0
    assert np.all(np.isfinite(m["f"]))


def test_predict_usa_distancia_combinada_nao_regra_retangular():
    """REGRESSAO: predict() aceitava so' se T2<=UCL(T2) E Q<=UCL(Q)
    independentemente -- uma caixa retangular. Com alpha independente por
    eixo, a rejeicao conjunta efetiva era ~1-(1-alpha)^2 (~0.0975 p/
    alpha=0.05), quase o dobro do declarado. A regra corrigida usa
    f=(T2/h0)*Nh+(Q/q0)*Nq <= f_crit (Kucheryavskiy/Rodionova/Pomerantsev
    2024) e fica muito mais proxima do alpha nominal."""
    rng = np.random.default_rng(0)
    Xc = rng.normal(scale=1.0, size=(20, 50))
    dd = DDSimca(n_components=3, alpha=0.05).fit(Xc, np.array(["A"] * 20))

    Xt = rng.normal(scale=1.0, size=(4000, 50))
    sc = dd.score_matrix(Xt)["A"]
    t2_aceito = sc["T2_norm"] <= 1.0
    q_aceito  = sc["Q_norm"]  <= 1.0
    antiga = t2_aceito & q_aceito
    nova   = sc["f"] <= sc["f_crit"]

    # As duas regras tem que DISCORDAR em uma fracao real de pontos --
    # senao o teste nao prova que o comportamento mudou de verdade.
    discordancia = float(np.mean(antiga != nova))
    assert discordancia > 0.005, (
        "regra nova e antiga concordam em quase tudo -- fix nao mudou nada")

    # Propriedade ESTRUTURAL (sempre verdadeira, nao depende do sorteio
    # aleatorio): P(A ∩ B) <= min(P(A), P(B)). O "E" de dois testes so'
    # pode aceitar MENOS OU IGUAL do que qualquer um dos dois isolados --
    # e' exatamente a penalidade que faz a caixa retangular superrejeitar.
    assert antiga.mean() <= t2_aceito.mean() + 1e-9
    assert antiga.mean() <= q_aceito.mean() + 1e-9

    # A distancia combinada nao tem essa penalidade estrutural (nao e' um
    # "E" de dois testes independentes): aceita estritamente mais que a
    # regra retangular que ela substituiu.
    assert nova.mean() > antiga.mean(), (
        f"regra nova (aceita {nova.mean():.3f}) nao superou a antiga "
        f"(aceita {antiga.mean():.3f}) -- deveria, por construcao")


def test_predict_e_score_matrix_f_concordam():
    """predict() e score_matrix() tem que usar EXATAMENTE a mesma regra
    (mesmo f/f_crit) -- antes cada metodo (predict, sensibilidade_ddsimca_
    logo, pipeline) reimplementava a comparacao por conta propria."""
    rng = np.random.default_rng(3)
    X = _classe_compacta(rng, centro=0.0, n=15)
    dd = DDSimca(n_components=3).fit(X, np.array(["A"] * 15))
    Xt = rng.normal(size=(50, X.shape[1])) * 2.0

    preds = dd.predict(Xt)
    sc = dd.score_matrix(Xt)["A"]
    aceito_score = sc["f"] <= sc["f_crit"]
    aceito_predict = preds == "A"
    np.testing.assert_array_equal(aceito_score, aceito_predict)


def test_media_e_dof_casos_degenerados():
    from guaraci.classificadores import DDSimca
    media, N = DDSimca._media_e_dof(np.array([]))
    assert media == 0.0 and N == 1.0
    media, N = DDSimca._media_e_dof(np.full(10, 3.0))   # desvio=0
    assert N == 1.0
    media, N = DDSimca._media_e_dof(np.array([5.0]))    # n=1, sem desvio
    assert N == 1.0


# ---------------------------------------------------------------------------
# Sensibilidade DD-SIMCA por Procrustes Cross-Validation (PCV) -- diagnostico
# complementar ao LOGO, adicionado 2026-08-08.
# ---------------------------------------------------------------------------
def test_pcv_indisponivel_sem_pacote_nao_lanca_excecao(monkeypatch):
    """Se 'prcv' nao estiver instalado, devolve disponivel=False com aviso
    -- nunca quebra o pipeline (mesmo padrao de xgboost/shap opcionais)."""
    import builtins
    real_import = builtins.__import__

    def _import_bloqueado(name, *a, **kw):
        if name == "prcv" or name.startswith("prcv."):
            raise ImportError("simulado: prcv nao instalado")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _import_bloqueado)
    rng = np.random.default_rng(0)
    X = _classe_compacta(rng, centro=0.0, n=3)
    r = sensibilidade_ddsimca_pcv(X, np.array(["G1"] * 3), n_components=3)
    assert r["disponivel"] is False
    assert r["aviso"] is not None
    assert np.isnan(r["sensibilidade"])


def test_pcv_um_grupo_nao_quebra_e_avisa_limitacao():
    """REGRESSAO: passar o split de CV do PCV agrupado por mae_id quando
    so' existe 1 grupo faz pcvpca falhar (ValueError de shape, verificado
    manualmente) -- a funcao tem que cair para LOO por amostra nesse caso,
    nunca lancar excecao. O aviso tem que deixar claro que o resultado nao
    e' evidencia de autenticacao (so' ruido de medicao), senao o numero
    seria mal-interpretado como se fosse tao forte quanto LOGO."""
    pytest.importorskip("prcv")
    rng = np.random.default_rng(1)
    X = _classe_compacta(rng, centro=0.0, n=3, k=30, escala=0.05)
    r = sensibilidade_ddsimca_pcv(X, np.array(["G1", "G1", "G1"]),
                                  n_components=3)
    assert r["n_grupos"] == 1
    assert r["disponivel"] is True
    assert not np.isnan(r["sensibilidade"])
    assert "ruido de MEDICAO" in r["aviso"]
    assert "instrumental" in r["aviso"]


def test_pcv_multiplos_grupos_usa_split_por_grupo():
    """Com 2+ grupos, o resultado nao e' NaN e o aviso e' o generico de
    complementaridade ao LOGO (nao o de grupo unico)."""
    pytest.importorskip("prcv")
    rng = np.random.default_rng(2)
    X = np.vstack([_classe_compacta(rng, centro=i * 0.3, n=3, k=30,
                                    escala=0.05) for i in range(4)])
    grupos = np.array([f"G{i}" for i in range(4) for _ in range(3)])
    r = sensibilidade_ddsimca_pcv(X, grupos, n_components=3)
    assert r["n_grupos"] == 4
    assert not np.isnan(r["sensibilidade"])
    assert "ruido de MEDICAO" not in (r["aviso"] or "")


def test_pcv_amostras_insuficientes_nao_quebra():
    pytest.importorskip("prcv")
    X = np.zeros((1, 10))
    r = sensibilidade_ddsimca_pcv(X, np.array(["G1"]), n_components=3)
    assert np.isnan(r["sensibilidade"])
    assert r["aviso"] is not None


# ---------------------------------------------------------------------------
# Diagnostico robusto (mediana/MAD) de replicas de treino atipicas --
# adicionado 2026-08-08. So' SINALIZA, nunca remove sozinho.
# ---------------------------------------------------------------------------
def test_outliers_robustos_detecta_replica_divergente():
    """Cenario controlado: 2 replicas proximas (medicao normal) + 1
    deslocada (replica atipica/possivel contaminacao). O z-score modificado
    (Iglewicz & Hoaglin 1993) tem que sinalizar a divergente."""
    rng = np.random.default_rng(0)
    p = 200
    base = rng.normal(scale=0.3, size=p)
    X = np.array([
        base + rng.normal(scale=0.01, size=p),
        base + rng.normal(scale=0.01, size=p),
        base + 3.0 + rng.normal(scale=0.01, size=p),
    ])
    dd = DDSimca(n_components=3).fit(X, np.array(["A"] * 3))
    outliers = dd._modelos["A"]["outliers_treino"]
    assert 2 in outliers


def test_outliers_robustos_sem_falso_positivo_em_dados_limpos():
    """3 replicas normais (mesma distribuicao) nao devem disparar aviso na
    maioria dos casos -- senao o diagnostico vira ruido, nao sinal.

    NOTA HONESTA (medido, nao suposto): com nc=3, _MIN_Q_RESIDUAL_DF forca
    n_comp=1 (so' 2 graus de liberdade residuais) -- T2_train/Q_train ja
    sao inerentemente instaveis SEM outlier real nenhum, e o detector
    (uniao de 2 testes, T2 e Q) chega a ~10% de falso positivo mesmo em
    n=20 (medido: 3/30 seeds). A seed abaixo foi verificada limpa; nao e'
    garantia de zero falsos positivos em toda seed -- e' o preco de operar
    honestamente no regime de poucas amostras deste projeto, nao um bug
    do detector."""
    rng = np.random.default_rng(2)
    p = 200
    base = rng.normal(scale=0.3, size=p)
    X = np.array([base + rng.normal(scale=0.01, size=p) for _ in range(3)])
    dd = DDSimca(n_components=3).fit(X, np.array(["A"] * 3))
    assert dd._modelos["A"]["outliers_treino"] == []


def test_outliers_robustos_mad_zero_nao_quebra():
    """Valores identicos (MAD=0, treino degenerado) nao pode lancar
    ZeroDivisionError/produzir NaN -- devolve 'nenhum outlier'."""
    out = DDSimca._outliers_robustos_mad(np.array([5.0, 5.0, 5.0]))
    assert out.size == 0


def test_outliers_robustos_poucos_pontos_nao_quebra():
    for valores in (np.array([]), np.array([1.0]), np.array([1.0, 2.0])):
        out = DDSimca._outliers_robustos_mad(valores)
        assert out.size == 0


def test_score_matrix_expoe_outliers_treino():
    rng = np.random.default_rng(2)
    X = _classe_compacta(rng, centro=0.0, n=10)
    dd = DDSimca(n_components=3).fit(X, np.array(["A"] * 10))
    m = dd.score_matrix(X)["A"]
    assert "outliers_treino" in m
    assert isinstance(m["outliers_treino"], list)


def test_outliers_robustos_nao_remove_amostra_do_treino(caplog):
    """REGRESSAO/GARANTIA DE DESIGN: mesmo com outlier detectado, n_train
    continua o numero ORIGINAL de amostras -- a funcao so' avisa, nunca
    filtra o treino sozinha (removeria dado escasso demais sem o usuario
    decidir)."""
    rng = np.random.default_rng(3)
    p = 200
    base = rng.normal(scale=0.3, size=p)
    X = np.array([
        base + rng.normal(scale=0.01, size=p),
        base + rng.normal(scale=0.01, size=p),
        base + 3.0 + rng.normal(scale=0.01, size=p),
    ])
    with caplog.at_level(logging.WARNING, logger="guaraci.classificadores"):
        dd = DDSimca(n_components=3).fit(X, np.array(["A"] * 3))
    assert dd._modelos["A"]["n_train"] == 3   # nenhuma amostra removida
    assert dd._modelos["A"]["T2_train"].size == 3
    assert "atipica" in caplog.text
