"""Kruskal-Wallis + Dunn, diagnostico de premissas e recomendacao de metodo."""
import math

import numpy as np
import pytest
from hypothesis import example, given, settings, strategies as st
from scipy import stats

from guaraci.anova_grupos import (
    ASSIMETRIA_LIMITE, CURTOSE_LIMITE, RAZAO_VARIANCIA_LIMITE, TEXTO_KRUSKAL,
    anova_uma_via, diagnosticar_premissas, kruskal_dunn, recomendar_metodo)

G0 = [24.5, 23.5, 26.4, 27.1, 29.9, 25.0]
G1 = [28.4, 34.2, 29.5, 32.2, 30.1, 31.0]
G2 = [26.1, 28.3, 24.3, 26.2, 27.8, 26.9]


def _dados(*grupos):
    y, g, u = [], [], []
    for i, a in enumerate(grupos):
        for j, v in enumerate(a):
            y.append(v)
            g.append(str(i))
            u.append(f"{i}-{j}")
    return y, g, u


def _kw(*grupos, **kw):
    y, g, u = _dados(*grupos)
    return kruskal_dunn(y, g, u, **kw)


def _par(r, a, b):
    for c in r.comparacoes:
        if {c["a"], c["b"]} == {a, b}:
            return c
    raise KeyError((a, b))


# ------------------------------------------------------- oraculos ingenuos

def _postos_ingenuos(valores):
    """Postos medios por definicao (laco duplo), sem rankdata."""
    v = list(valores)
    return [(sum(1 for x in v if x < a) + (sum(1 for x in v if x == a) + 1) / 2) for a in v]


def _dunn_ingenuo(grupos):
    todos = [x for g in grupos for x in g]
    N = len(todos)
    postos = _postos_ingenuos(todos)
    medias, i0 = [], 0
    for g in grupos:
        medias.append(sum(postos[i0:i0 + len(g)]) / len(g))
        i0 += len(g)
    empates = sum(t ** 3 - t for t in (todos.count(x) for x in set(todos)))
    var = N * (N + 1) / 12 - empates / (12 * (N - 1))
    out = {}
    for i in range(len(grupos)):
        for j in range(i + 1, len(grupos)):
            z = (medias[i] - medias[j]) / math.sqrt(var * (1 / len(grupos[i]) + 1 / len(grupos[j])))
            out[(i, j)] = (z, 2 * (1 - stats.norm.cdf(abs(z))))
    return out


def _holm_ingenuo(ps):
    m = len(ps)
    ordem = sorted(range(m), key=lambda i: ps[i])
    aj = [0.0] * m
    for rank, i in enumerate(ordem):
        aj[i] = min(1.0, max((m - r) * ps[ordem[r]] for r in range(rank + 1)))
    return aj


def _bh_ingenuo(ps):
    m = len(ps)
    ordem = sorted(range(m), key=lambda i: ps[i])
    aj = [0.0] * m
    for rank, i in enumerate(ordem):
        aj[i] = min(1.0, min(ps[ordem[r]] * m / (r + 1) for r in range(rank, m)))
    return aj


def test_kw_bate_com_scipy_e_epsilon2_pela_formula():
    r = _kw(G0, G1, G2)
    ref = stats.kruskal(G0, G1, G2)
    assert r.H == pytest.approx(ref.statistic, rel=1e-12)
    assert r.p == pytest.approx(ref.pvalue, rel=1e-12)
    assert r.gl == 2
    n = 18
    assert r.epsilon2 == pytest.approx(r.H / ((n ** 2 - 1) / (n + 1)), rel=1e-12)
    assert 0 <= r.epsilon2 <= 1
    assert TEXTO_KRUSKAL in r.texto and "dominancia" in r.texto and "medianas" in r.texto


@pytest.mark.parametrize("ajuste, f", [("holm", _holm_ingenuo), ("bh", _bh_ingenuo)])
def test_dunn_bate_com_oraculo_ingenuo_sem_e_com_empates(ajuste, f):
    for grupos in ([G0, G1, G2],
                   [[1, 2, 2, 3, 3, 3], [2, 3, 4, 4, 5, 5], [1, 1, 2, 2, 2, 6], [5, 5, 6, 7, 7, 7]]):
        r = _kw(*grupos, ajuste=ajuste)
        ref = _dunn_ingenuo([list(map(float, g)) for g in grupos])
        pares = sorted(ref)
        ajust = f([ref[k][1] for k in pares])
        for k, aj in zip(pares, ajust):
            c = _par(r, str(k[0]), str(k[1]))
            assert c["z"] == pytest.approx(ref[k][0], rel=1e-9, abs=1e-12)
            assert c["p_bruto"] == pytest.approx(ref[k][1], rel=1e-9, abs=1e-15)
            assert c["p_ajustado"] == pytest.approx(aj, rel=1e-9, abs=1e-15)
    assert ajuste in r.regra_ajuste.lower() or "Holm" in r.regra_ajuste or "Benjamini" in r.regra_ajuste


def test_k2_dunn_z2_e_h_e_concorda_com_mann_whitney_assintotico():
    """k=2: z^2 de Dunn == H do Kruskal e o p bruto e' o do Mann-Whitney
    assintotico (sem correcao de continuidade, com correcao de empates)."""
    a, b = [1, 2, 2, 3, 5, 8, 8], [2, 4, 6, 8, 9, 9, 10]
    r = _kw(a, b)
    c = r.comparacoes[0]
    assert c["z"] ** 2 == pytest.approx(r.H, rel=1e-10)
    assert c["p_bruto"] == pytest.approx(r.p, rel=1e-10)
    mw = stats.mannwhitneyu(a, b, use_continuity=False, method="asymptotic", alternative="two-sided")
    assert c["p_bruto"] == pytest.approx(mw.pvalue, rel=1e-8)
    # com 1 comparacao, Holm nao altera o p
    assert c["p_ajustado"] == pytest.approx(c["p_bruto"])


def test_empates_em_massa_continuam_definidos():
    r = _kw([1, 1, 1, 1, 2, 2], [1, 1, 2, 2, 2, 2], [2, 2, 2, 3, 3, 3])
    assert math.isfinite(r.H) and all(math.isfinite(c["z"]) for c in r.comparacoes)


def test_todos_iguais_e_erro():
    with pytest.raises(ValueError, match="iguais"):
        _kw([3.0] * 4, [3.0] * 4)


def test_n1_e_excluido_com_aviso_e_todos_n1_e_erro():
    r = _kw(G0, G1, [9.0])
    assert r.grupos == ["0", "1"] and any("1 unidade" in a for a in r.avisos)
    with pytest.raises(ValueError, match="menos de 2 grupos"):
        _kw([1.0], [2.0], [3.0])


def test_n_pequeno_avisa_poder_baixo():
    r = _kw([1, 2, 3], [4, 5, 6])
    assert any("poder baixo" in a for a in r.avisos)


def test_nan_descartado_com_aviso():
    y, g, u = _dados(G0, G1)
    y[2] = float("nan")
    r = kruskal_dunn(y, g, u)
    assert any("descartada" in a for a in r.avisos)
    assert r.p == pytest.approx(stats.kruskal(G0[:2] + G0[3:], G1).pvalue, rel=1e-12)


@pytest.mark.parametrize("transf", [np.log, np.sqrt, lambda x: x ** 3, lambda x: 1 / x])
def test_invariancia_a_transformacao_monotona_crescente(transf):
    base = _kw(G0, G1, G2)
    r = _kw(*[list(transf(np.array(g))) for g in (G0, G1, G2)])
    if transf(2.0) < transf(1.0):                     # decrescente inverte so' o sinal
        for c0 in base.comparacoes:
            assert _par(r, c0["a"], c0["b"])["p_ajustado"] == pytest.approx(c0["p_ajustado"], rel=1e-9)
    else:
        for c0 in base.comparacoes:
            c = _par(r, c0["a"], c0["b"])
            assert c["z"] == pytest.approx(c0["z"], rel=1e-9)
            assert c["p_ajustado"] == pytest.approx(c0["p_ajustado"], rel=1e-9)
    assert r.H == pytest.approx(base.H, rel=1e-10)


def test_ordem_das_linhas_nao_altera():
    y, g, u = _dados(G0, G1, G2)
    r0 = kruskal_dunn(y, g, u)
    idx = np.random.default_rng(1).permutation(len(y))
    r = kruskal_dunn([y[i] for i in idx], [g[i] for i in idx], [u[i] for i in idx])
    assert r.H == pytest.approx(r0.H, rel=1e-12)
    for c0 in r0.comparacoes:
        assert _par(r, c0["a"], c0["b"])["p_ajustado"] == pytest.approx(c0["p_ajustado"], rel=1e-12)


def test_recusas_de_unidade_e_ajuste_invalido():
    y, g, _ = _dados(G0, G1)
    with pytest.raises(ValueError, match="unidade"):
        kruskal_dunn(y, g, None)
    with pytest.raises(ValueError, match="sem id"):
        kruskal_dunn(y, g, [""] * len(y))
    with pytest.raises(ValueError, match="dobras"):
        kruskal_dunn(y, g, [str(i) for i in range(len(y))], origem="dobras_cv")
    with pytest.raises(ValueError, match="ajuste"):
        _kw(G0, G1, ajuste="bonferroni")


_gr = st.lists(st.integers(min_value=0, max_value=20).map(float), min_size=3, max_size=10)


@settings(max_examples=80, deadline=None)
@given(_gr, _gr, _gr)
@example([1.0, 1.0, 2.0], [1.0, 2.0, 2.0], [3.0, 3.0, 3.0])
@example([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0])
def test_propriedades_kw_dunn(a, b, c):
    try:
        r = _kw(a, b, c)
    except ValueError:
        return
    assert 0 <= r.p <= 1 and 0 <= r.epsilon2 <= 1 + 1e-12
    for cmp in r.comparacoes:
        assert 0 <= cmp["p_bruto"] <= cmp["p_ajustado"] + 1e-12 <= 1 + 1e-12
    rr = _kw(c, b, a)
    assert rr.H == pytest.approx(r.H, rel=1e-9, abs=1e-12)
    for cmp in r.comparacoes:
        inv = _par(rr, {"0": "2", "1": "1", "2": "0"}[cmp["a"]], {"0": "2", "1": "1", "2": "0"}[cmp["b"]])
        assert inv["p_ajustado"] == pytest.approx(cmp["p_ajustado"], rel=1e-9, abs=1e-12)


# ------------------------------------------------- variancias forcadas / aviso

def test_variancias_forcadas_mudam_metodo_e_ficam_declaradas():
    y, g, u = _dados(G0, G1, G2)
    assert anova_uma_via(y, g, u, n_boot=0, variancias="desiguais").metodo == "Welch ANOVA"
    r = anova_uma_via(y, g, u, n_boot=0, variancias="desiguais")
    # declarado sem afirmar que foi "pedido": o padrao de um comando tambem passa variancias='desiguais' (vistoria V20)
    assert r.pos_teste == "Games-Howell" and "Welch ANOVA + Games-Howell (variancias='desiguais')" in r.avisos
    assert not any("como solicitado" in a for a in r.avisos)
    assert anova_uma_via(y, g, u, n_boot=0, variancias="iguais").pos_teste in ("Tukey", "Tukey-Kramer")
    with pytest.raises(ValueError, match="variancias"):
        anova_uma_via(y, g, u, variancias="talvez")


def test_variancias_iguais_forcado_ignora_levene():
    rng = np.random.default_rng(5)
    a, b = rng.normal(0, 0.05, 12), rng.normal(0.3, 5.0, 12)
    y, g, u = _dados(a, b)
    assert anova_uma_via(y, g, u, n_boot=0).metodo == "Welch ANOVA"
    assert anova_uma_via(y, g, u, n_boot=0, variancias="iguais").metodo == "ANOVA classica"


# ----------------------------------------------- diagnostico + recomendacao

def _diag(*grupos, **kw):
    y, g, u = _dados(*grupos)
    return diagnosticar_premissas(y, g, u, **kw)


def test_diagnostico_calcula_residuos_qq_e_testes():
    d = _diag(G0, G1, G2)
    res = np.concatenate([np.array(x) - np.mean(x) for x in (G0, G1, G2)])
    np.testing.assert_allclose(d["residuos"], res)
    assert len(d["qq_teorico"]) == len(d["qq_amostral"]) == 18
    assert np.all(np.diff(d["qq_amostral"]) >= 0) and np.all(np.diff(d["qq_teorico"]) > 0)
    assert d["shapiro_p"] == pytest.approx(stats.shapiro(res).pvalue)
    assert d["levene_p"] == pytest.approx(stats.levene(G0, G1, G2, center="mean").pvalue)
    assert d["brown_forsythe_p"] == pytest.approx(stats.levene(G0, G1, G2, center="median").pvalue)
    v = [np.var(x, ddof=1) for x in (G0, G1, G2)]
    assert d["razao_variancias"] == pytest.approx(max(v) / min(v))
    assert d["razao_n"] == 1.0
    assert "porteiro" in d["texto"] and "quantil" in d["texto"] and "Games-Howell" in d["texto"]


def test_diagnostico_recusas_e_variancia_zero():
    with pytest.raises(ValueError, match="menos de 2 grupos"):
        _diag([1.0], [2.0])
    d = _diag([5.0] * 4, [5.0] * 4)
    assert d["shapiro_p"] is None and d["razao_variancias"] == 1.0 and d["assimetria"] == 0.0
    d2 = _diag([5.0] * 4, [1.0, 2.0, 3.0, 4.0])
    assert d2["razao_variancias"] == float("inf")


def test_recomendacao_para_cada_violacao_sintetica():
    rng = np.random.default_rng(7)
    ok = _diag(*[rng.normal(m, 1.0, 12) for m in (0, 0.2, 0.1)])
    assert recomendar_metodo(ok)["metodo"] == "tukey"
    het = _diag(rng.normal(0, 0.3, 12), rng.normal(0, 3.0, 12), rng.normal(0, 1.0, 12))   # 1:10 em dp
    assert recomendar_metodo(het)["metodo"] == "games-howell"
    assim = _diag(*[rng.lognormal(0, 1.2, 15) for _ in range(3)])
    assert recomendar_metodo(assim)["metodo"] == "kruskal-dunn"
    # curtose amostral de caudas pesadas e' instavel (t com 2 gl: 0,8 a 51 conforme a
    # semente) -- por isso o texto manda olhar o grafico; aqui, semente fixa conhecida.
    rng = np.random.default_rng(7)
    cauda = _diag(*[rng.standard_t(2.0, 40) for _ in range(3)])
    assert recomendar_metodo(cauda)["metodo"] == "kruskal-dunn"
    desig_n = _diag(rng.normal(0, 1, 30), rng.normal(0, 1, 4), rng.normal(0, 1, 5))   # n muito desigual, premissas ok
    assert desig_n["razao_n"] == pytest.approx(7.5)


def _d(**kw):
    base = {"assimetria": 0.0, "curtose_excesso": 0.0, "razao_variancias": 1.0,
            "brown_forsythe_p": 0.5, "alpha": 0.05}
    base.update(kw)
    return base


@pytest.mark.parametrize("campo, limite, esperado_acima", [
    ("razao_variancias", RAZAO_VARIANCIA_LIMITE, "games-howell"),
    ("assimetria", ASSIMETRIA_LIMITE, "kruskal-dunn"),
    ("curtose_excesso", CURTOSE_LIMITE, "kruskal-dunn"),
])
def test_limiares_nas_bordas_logo_abaixo_no_limite_e_logo_acima(campo, limite, esperado_acima):
    assert recomendar_metodo(_d(**{campo: limite - 1e-6}))["metodo"] == "tukey"
    assert recomendar_metodo(_d(**{campo: limite}))["metodo"] == "tukey"          # no limite: nao troca
    assert recomendar_metodo(_d(**{campo: limite + 1e-6}))["metodo"] == esperado_acima


def test_assimetria_negativa_tambem_conta_e_borda_do_brown_forsythe():
    assert recomendar_metodo(_d(assimetria=-(ASSIMETRIA_LIMITE + 1e-6)))["metodo"] == "kruskal-dunn"
    assert recomendar_metodo(_d(brown_forsythe_p=0.05))["metodo"] == "tukey"       # p == alpha: nao
    assert recomendar_metodo(_d(brown_forsythe_p=0.05 - 1e-9))["metodo"] == "games-howell"
    # forma tem prioridade sobre variancia
    assert recomendar_metodo(_d(assimetria=2.0, razao_variancias=50.0))["metodo"] == "kruskal-dunn"


def test_recomendacao_traz_razao_em_uma_linha():
    for d in (_d(), _d(razao_variancias=10.0), _d(assimetria=2.0)):
        r = recomendar_metodo(d)
        assert "\n" not in r["razao"] and "->" in r["razao"]


def test_regressao_variancias_desiguais_nao_viram_caudas_pesadas():
    """Achado da integracao: a curtose dos residuos JUNTADOS explodia so' por
    causa de variancias 1:100 (mistura de escalas) e a recomendacao saia
    Kruskal em vez de Games-Howell. A forma agora usa residuos padronizados
    por grupo."""
    rng = np.random.default_rng(1)
    d = _diag(rng.normal(0, 0.3, 40), rng.normal(0, 3.0, 40), rng.normal(0, 1.0, 40))
    assert abs(d["assimetria"]) < ASSIMETRIA_LIMITE and d["curtose_excesso"] < CURTOSE_LIMITE
    assert recomendar_metodo(d)["metodo"] == "games-howell"
