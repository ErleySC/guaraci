"""Rede de segurança da validação estatística (guaraci.validacao_estatistica).

BCa (IC de confiança) e CV-ANOVA (significância do modelo) são o que torna os
resultados "publication-grade" — uma regressão silenciosa aqui corromperia
intervalos de confiança e p-valores reportados em monografia/artigo. Testes
das duas funções numéricas PURAS (as demais — teste_wold/permutação — exigem
pipeline+CV e são cobertas pelos testes end-to-end 'slow').
"""
import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from guaraci.validacao_estatistica import (bootstrap_bca_ci, cv_anova_eriksson,
                                          StratifiedGroupKFoldEstavel)
# Alias com prefixo _ para o pytest NAO coletar a funcao importada como teste
# (o nome 'teste_permutacao' casa com o padrao de coleta 'test*').
from guaraci.validacao_estatistica import teste_permutacao as _teste_permutacao
from guaraci.validacao_estatistica import teste_wold as _teste_wold
from guaraci.validacao_estatistica import _gerar_permutacoes_rotulo


class _CVFalhaApartirDaSegundaChamada:
    """cv fake que delega para um StratifiedKFold real na 1a chamada (a
    observada, computada ANTES do loop de permutacao em teste_permutacao/
    teste_wold) e levanta ValueError em TODAS as chamadas seguintes (as do
    loop de permutacao) -- simula "fold impossivel apos embaralhar
    rotulos" de forma deterministica, sem depender de uma coincidencia
    estatistica fragil para acionar o caminho de erro/falha de
    _iter_permutacao/_iter_wold."""

    def __init__(self, n_splits=4):
        self._real = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        self._chamadas = 0

    def split(self, X, y=None, groups=None):
        self._chamadas += 1
        if self._chamadas == 1:
            return self._real.split(X, y, groups)
        raise ValueError("simulada: estratificacao impossivel neste fold")


# ── bootstrap_bca_ci ─────────────────────────────────────────────────────────
def test_bca_predicao_perfeita_ic_em_um():
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    low, high, obs = bootstrap_bca_ci(y, y.copy(), accuracy_score, n_boot=200, seed=1)
    assert obs == 1.0
    assert low == 1.0 and high == 1.0


def test_bca_observed_bate_com_metrica():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=40)
    yp = y.copy()
    yp[:10] = (yp[:10] + 1) % 3  # 25% de erro
    _, _, obs = bootstrap_bca_ci(y, yp, accuracy_score, n_boot=200, seed=1)
    assert obs == accuracy_score(y, yp)


def test_bca_intervalo_contem_observado_e_dentro_de_0_1():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=60)
    yp = y.copy()
    yp[::5] = 1 - yp[::5]  # ~20% erro
    low, high, obs = bootstrap_bca_ci(y, yp, accuracy_score, n_boot=300, seed=7)
    assert 0.0 <= low <= obs <= high <= 1.0


def test_bca_reprodutivel_com_mesma_seed():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, size=50)
    yp = y.copy(); yp[:8] = (yp[:8] + 1) % 3
    r1 = bootstrap_bca_ci(y, yp, accuracy_score, n_boot=200, seed=99)
    r2 = bootstrap_bca_ci(y, yp, accuracy_score, n_boot=200, seed=99)
    assert r1 == r2


def test_bca_n_boot_baixo_retorna_nan():
    y = np.array([0, 0, 1, 1, 0, 1])
    low, high, obs = bootstrap_bca_ci(y, y.copy(), accuracy_score, n_boot=5, seed=1)
    assert np.isnan(low) and np.isnan(high)
    assert obs == 1.0


# ── cv_anova_eriksson ────────────────────────────────────────────────────────
def test_cv_anova_predicao_perfeita_q2_alto_p_baixo():
    Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    r = cv_anova_eriksson(Y, Y.copy(), n_components=2)
    assert r["Q2"] > 0.999
    assert r["p_value"] < 0.05


def test_cv_anova_predicao_ruim_q2_baixo_p_alto():
    Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    Y_cv = np.full_like(Y, Y.mean())  # prever a média = PRESS ~ SS_total
    r = cv_anova_eriksson(Y, Y_cv, n_components=2)
    assert r["Q2"] <= 0.0
    assert r["p_value"] == 1.0
    assert r["F"] == 0.0


def test_cv_anova_y_constante_retorna_nan():
    Y = np.full(10, 3.0)          # variância zero
    r = cv_anova_eriksson(Y, Y.copy(), n_components=1)
    assert np.isnan(r["F"])
    assert r["p_value"] == 1.0
    assert r["Q2"] == 0.0


def test_cv_anova_aceita_y_2d_onehot():
    # Y one-hot (m=K classes); a função reduz para univariado (m_eff=1)
    Y = np.eye(3)[np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])]
    Y_cv = Y * 0.9 + 0.033  # quase perfeito
    r = cv_anova_eriksson(Y, Y_cv, n_components=2)
    assert 0.0 < r["Q2"] <= 1.0
    assert 0.0 <= r["p_value"] <= 1.0


def test_cv_anova_q2_formula():
    # Q2 = 1 - PRESS/SS_total; monta um caso com valor conhecido
    Y = np.array([0.0, 2.0, 4.0, 6.0])           # média 3, SS_total = 9+1+1+9 = 20
    Y_cv = np.array([1.0, 2.0, 4.0, 5.0])        # PRESS = 1+0+0+1 = 2
    r = cv_anova_eriksson(Y, Y_cv, n_components=1)
    assert abs(r["Q2"] - (1.0 - 2.0 / 20.0)) < 1e-9


# ── teste_permutacao (Y-randomization) ──────────────────────────────────────
def _dados_perm(separavel: bool, seed: int):
    """Dataset binario de 2x20 amostras, 10 variaveis. `separavel=True` cria
    duas nuvens bem afastadas (sinal real); False = ruido puro (rotulos sem
    relacao com X)."""
    rng = np.random.default_rng(seed)
    n = 40
    y_int = np.array([0] * (n // 2) + [1] * (n // 2))
    if separavel:
        X = np.vstack([rng.normal(-3.0, 0.5, size=(n // 2, 10)),
                       rng.normal(+3.0, 0.5, size=(n // 2, 10))])
    else:
        X = rng.normal(0.0, 1.0, size=(n, 10))
    Y_bin = np.zeros((n, 2)); Y_bin[np.arange(n), y_int] = 1.0
    return X, Y_bin, y_int


def _factory_pls():
    return Pipeline([("pls", PLSRegression(n_components=2, scale=False))])


def test_permutacao_da_p_baixo_com_sinal_real():
    """VALIDACAO: com classes bem separadas, a acuracia observada deve superar
    quase todas as permutacoes -> p pequeno. Se o teste desse p alto aqui,
    estaria mascarando sinal real como se fosse acaso."""
    X, Y_bin, y_int = _dados_perm(separavel=True, seed=1)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    res = _teste_permutacao(_factory_pls, X, Y_bin, y_int, cv,
                           n_perm=40, seed=1)
    assert res["p_value"] < 0.05
    assert res["acc_observada"] > 0.9


def test_permutacao_da_p_alto_com_rotulos_aleatorios():
    """VALIDACAO: sob H0 (rotulos sem relacao com X) a acuracia observada e'
    apenas mais uma amostra da distribuicao permutada -> p NAO deve ser
    pequeno. Se desse p baixo aqui, seria um falso positivo (acha estrutura
    onde so ha ruido)."""
    X, Y_bin, y_int = _dados_perm(separavel=False, seed=2)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    res = _teste_permutacao(_factory_pls, X, Y_bin, y_int, cv,
                           n_perm=40, seed=2)
    assert res["p_value"] > 0.10


def test_permutacao_todas_as_iteracoes_falham_da_p_1_nao_informativo():
    """Se TODA iteracao do loop de permutacao falhar (fold impossivel apos
    embaralhar rotulos -- caso real com classes muito desbalanceadas), o
    p-valor deve ser 1.0 (nao-informativo), nunca um numero calculado sobre
    uma lista vazia de acertos. n_falhos deve contar TODAS as permutacoes."""
    X, Y_bin, y_int = _dados_perm(separavel=True, seed=3)
    cv = _CVFalhaApartirDaSegundaChamada(n_splits=4)
    res = _teste_permutacao(_factory_pls, X, Y_bin, y_int, cv,
                           n_perm=10, seed=3)
    assert res["n_validos"] == 0
    assert res["n_falhos"] == 10
    assert res["failure_rate"] == 1.0
    assert res["p_value"] == 1.0


# ── _gerar_permutacoes_rotulo (achado A1, auditoria 2026-08-07) ────────────
# O teste de permutacao/Wold permutava ROTULOS POR AMOSTRA, ignorando
# `groups` (mae_id) -- quebra a coerencia de replica fisica e estreita o
# nulo artificialmente (medido: falso positivo sobe de 5% nominal p/ 15%,
# ver docs/auditoria/medir_permutacao_grupos.py). Estes testes travam a
# propriedade que corrige isso: FALHAM com permutacao por amostra.

def test_permutacoes_por_grupo_preservam_coerencia_dentro_do_grupo():
    """Cada bloco de replicas fisicas (mesmo grupo) precisa manter o MESMO
    rotulo permutado em toda permutacao -- e' a propriedade que define uma
    permutacao group-aware. Uma permutacao por amostra quebraria isso com
    probabilidade praticamente 1 (grupos de tamanho >= 2)."""
    rng_dados = np.random.default_rng(0)
    groups = np.repeat(np.arange(15), 4)
    y_int = np.repeat(rng_dados.integers(0, 3, size=15), 4)
    rng = np.random.default_rng(1)
    permutacoes = _gerar_permutacoes_rotulo(y_int, groups, n_perm=30, rng=rng)
    assert len(permutacoes) == 30
    for y_perm in permutacoes:
        for g in np.unique(groups):
            rotulos_no_grupo = np.unique(y_perm[groups == g])
            assert len(rotulos_no_grupo) == 1, (
                f"grupo {g} recebeu rotulos permutados diferentes entre "
                "suas replicas -- coerencia de grupo quebrada")


def test_permutacoes_por_grupo_preservam_o_multiset_de_rotulos_por_grupo():
    """A permutacao reatribui rotulos ENTRE grupos (nao inventa rotulo
    novo): o conjunto de rotulos-por-grupo antes e depois deve ser o
    mesmo, so a atribuicao muda."""
    groups = np.repeat(np.arange(10), 3)
    y_int = np.repeat(np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 2]), 3)
    rot_original = sorted(y_int[groups == g][0] for g in np.unique(groups))
    rng = np.random.default_rng(2)
    permutacoes = _gerar_permutacoes_rotulo(y_int, groups, n_perm=20, rng=rng)
    for y_perm in permutacoes:
        rot_perm = sorted(y_perm[groups == g][0] for g in np.unique(groups))
        assert rot_perm == rot_original


def test_gerar_permutacoes_sem_groups_cai_para_permutacao_por_amostra():
    """Sem `groups`, nao ha estrutura a preservar -- deve reproduzir
    exatamente `y_int[rng.permutation(n)]` (mesmo rng, mesma sequencia)."""
    y_int = np.array([0, 0, 1, 1, 1, 2, 2, 0])
    rng_a = np.random.default_rng(5)
    rng_b = np.random.default_rng(5)
    esperado = [y_int[rng_a.permutation(len(y_int))] for _ in range(10)]
    obtido = _gerar_permutacoes_rotulo(y_int, None, n_perm=10, rng=rng_b)
    for e, o in zip(esperado, obtido):
        np.testing.assert_array_equal(e, o)


def test_permutacao_end_to_end_respeita_grupos():
    """teste_permutacao com `groups` fornecido deve, de ponta a ponta, gerar
    apenas permutacoes coerentes por grupo (nao so' o helper isolado)."""
    rng_dados = np.random.default_rng(9)
    n_grupos, n_rep = 20, 2
    groups = np.repeat(np.arange(n_grupos), n_rep)
    y_int_grupo = np.array([i % 2 for i in range(n_grupos)])
    y_int = np.repeat(y_int_grupo, n_rep)
    X = rng_dados.normal(0, 1, size=(n_grupos * n_rep, 10))
    Y_bin = np.zeros((len(y_int), 2)); Y_bin[np.arange(len(y_int)), y_int] = 1.0
    cv = StratifiedGroupKFoldEstavel(n_splits=4, seed=0)
    res = _teste_permutacao(_factory_pls, X, Y_bin, y_int, cv,
                           n_perm=15, seed=9, groups=groups)
    assert 0.0 <= res["p_value"] <= 1.0


def test_wold_todas_as_iteracoes_falham_nao_quebra():
    """Mesma propriedade de teste_permutacao, para teste_wold: se toda
    iteracao falhar, n_falhos conta todas, n_validos fica 0, e o ajuste
    linear (slope/intercept) vira NaN em vez de tentar np.polyfit com
    menos de 2 pontos validos."""
    X, Y_bin, y_int = _dados_perm(separavel=True, seed=4)
    cv = _CVFalhaApartirDaSegundaChamada(n_splits=4)
    res = _teste_wold(_factory_pls, X, Y_bin, y_int, cv, n_perm=10, seed=4)
    assert res["n_validos"] == 0
    assert res["n_falhos"] == 10
    assert np.isnan(res["intercept_r2"])
    assert np.isnan(res["intercept_q2"])


# ── StratifiedGroupKFoldEstavel ──────────────────────────────────────────────
# Existe porque o StratifiedGroupKFold do sklearn muda a particao entre versoes
# mesmo com random_state fixo (42% das amostras trocaram de fold entre 1.7.2 e
# 1.9.0). Os testes abaixo travam as propriedades que justificam a classe.

def _dados_agrupados(n_grupos=24, n_replicas=3, n_classes=3, seed=0):
    """Estrutura do dataset real: cada grupo = replicas fisicas (mae_id)."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_grupos), n_replicas)
    classe_do_grupo = np.array([g % n_classes for g in range(n_grupos)])
    y = np.repeat(classe_do_grupo, n_replicas)
    X = rng.normal(size=(n_grupos * n_replicas, 8))
    return X, y, groups


def test_grupo_nunca_se_divide_entre_treino_e_teste():
    """A propriedade que da nome ao metodo: replicas do mesmo mae_id ficam
    SEMPRE do mesmo lado. E' o diferencial cientifico do projeto — se este
    teste falhar, ha vazamento de replica e toda metrica esta inflada."""
    X, y, groups = _dados_agrupados()
    cv = StratifiedGroupKFoldEstavel(n_splits=4, seed=42)
    for tr, va in cv.split(X, y, groups):
        assert not (set(groups[tr]) & set(groups[va])), \
            "um grupo apareceu em treino E teste — vazamento de replica"


def test_particao_e_deterministica_entre_chamadas():
    """Mesma entrada -> mesma particao. Base de toda a reprodutibilidade."""
    X, y, groups = _dados_agrupados()
    a = [(tr.tolist(), va.tolist())
         for tr, va in StratifiedGroupKFoldEstavel(n_splits=3, seed=7).split(X, y, groups)]
    b = [(tr.tolist(), va.tolist())
         for tr, va in StratifiedGroupKFoldEstavel(n_splits=3, seed=7).split(X, y, groups)]
    assert a == b


def test_particao_nao_depende_da_ordem_das_amostras():
    """Permutar as LINHAS nao pode mudar a que fold cada AMOSTRA pertence.

    Garante que a particao e' funcao de (grupo, classe) e nao da ordem em que
    o arquivo foi lido — dois usuarios com o mesmo dataset em ordem diferente
    precisam obter o mesmo resultado.
    """
    X, y, groups = _dados_agrupados()
    cv = StratifiedGroupKFoldEstavel(n_splits=3, seed=1)

    def fold_por_grupo(Xa, ya, ga):
        destino = {}
        for i, (_tr, va) in enumerate(cv.split(Xa, ya, ga)):
            for g in np.unique(ga[va]):
                destino[g] = i
        return destino

    original = fold_por_grupo(X, y, groups)
    perm = np.random.default_rng(99).permutation(len(y))
    embaralhado = fold_por_grupo(X[perm], y[perm], groups[perm])
    assert original == embaralhado


def test_todas_as_amostras_aparecem_em_exatamente_um_fold_de_validacao():
    """Particao de verdade: nenhuma amostra fica de fora nem se repete."""
    X, y, groups = _dados_agrupados()
    cv = StratifiedGroupKFoldEstavel(n_splits=4, seed=5)
    vistos = np.concatenate([va for _tr, va in cv.split(X, y, groups)])
    assert sorted(vistos.tolist()) == list(range(len(y)))


def test_seeds_diferentes_podem_dar_particoes_diferentes():
    """O seed precisa REALMENTE variar a particao — senao repeticoes com
    seeds distintos mediriam sempre a mesma coisa."""
    X, y, groups = _dados_agrupados(n_grupos=30)
    def destino(seed):
        cv = StratifiedGroupKFoldEstavel(n_splits=3, seed=seed)
        d = np.zeros(len(y), dtype=int)
        for i, (_tr, va) in enumerate(cv.split(X, y, groups)):
            d[va] = i
        return d.tolist()
    assert any(destino(s) != destino(0) for s in (1, 2, 3, 4, 5))


def test_exige_groups_explicitamente():
    """Sem `groups` nao ha como manter replicas juntas: falhar e' obrigatorio,
    porque cair num split sem grupos vazaria replicas em silencio."""
    X, y, _groups = _dados_agrupados()
    with pytest.raises(ValueError, match="groups"):
        list(StratifiedGroupKFoldEstavel(n_splits=3).split(X, y, None))


def test_recusa_n_splits_maior_que_numero_de_grupos():
    X, y, groups = _dados_agrupados(n_grupos=3)
    with pytest.raises(ValueError, match="n_splits"):
        list(StratifiedGroupKFoldEstavel(n_splits=5).split(X, y, groups))


def test_estratificacao_distribui_classes_entre_os_folds():
    """Nao basta agrupar: cada fold precisa ver mais de uma classe, senao a
    metrica por fold fica degenerada."""
    X, y, groups = _dados_agrupados(n_grupos=30, n_classes=3)
    cv = StratifiedGroupKFoldEstavel(n_splits=3, seed=42)
    for _tr, va in cv.split(X, y, groups):
        assert len(np.unique(y[va])) >= 2, "fold de validacao com uma unica classe"


def test_compativel_com_cross_val_predict_do_sklearn():
    """Precisa funcionar como splitter do sklearn, nao so' isolado."""
    from sklearn.model_selection import cross_val_predict
    from sklearn.linear_model import LogisticRegression
    X, y, groups = _dados_agrupados(n_grupos=18)
    cv = StratifiedGroupKFoldEstavel(n_splits=3, seed=42)
    pred = cross_val_predict(LogisticRegression(max_iter=500), X, y,
                             cv=cv.split(X, y, groups))
    assert pred.shape == y.shape
