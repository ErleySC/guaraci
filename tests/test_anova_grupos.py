"""guaraci.anova_grupos: ANOVA + Tukey/Games-Howell sobre a UNIDADE independente."""
import math

import numpy as np
import pytest
from hypothesis import example, given, settings, strategies as st
from scipy import stats

from guaraci.anova_grupos import (
    N_MINIMO_PODER, agregar_por_unidade, anova_uma_via, anova_varias_respostas)

G0 = [24.5, 23.5, 26.4, 27.1, 29.9]
G1 = [28.4, 34.2, 29.5, 32.2, 30.1]
G2 = [26.1, 28.3, 24.3, 26.2, 27.8]


_SCIPY_SEM_EQUAL_VAR = "equal_var" not in __import__("inspect").signature(stats.tukey_hsd).parameters


class _Ref:
    """Oraculo do pos-teste que NAO depende da versao do SciPy: formula ingenua
    (Tukey-Kramer / Games-Howell com Welch-Satterthwaite + amplitude
    studentizada). Quando o SciPy tem `equal_var` (>= 1.16), confere tambem
    contra `tukey_hsd` -- assim nenhum ambiente pula este teste."""

    def __init__(self, amostras, igual, confianca=0.95):
        k = len(amostras)
        n = [len(a) for a in amostras]
        gl_in = sum(n) - k
        msw = sum((len(a) - 1) * np.var(a, ddof=1) for a in amostras) / gl_in
        self.statistic = np.zeros((k, k))
        self.pvalue = np.ones((k, k))
        self.low = np.zeros((k, k))
        self.high = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                d = np.mean(amostras[i]) - np.mean(amostras[j])
                if igual:
                    se, gl = math.sqrt(msw / 2 * (1 / n[i] + 1 / n[j])), gl_in
                else:
                    va, vb = np.var(amostras[i], ddof=1) / n[i], np.var(amostras[j], ddof=1) / n[j]
                    se = math.sqrt((va + vb) / 2)
                    gl = (va + vb) ** 2 / (va ** 2 / (n[i] - 1) + vb ** 2 / (n[j] - 1))
                crit = stats.studentized_range.isf(1 - confianca, k, gl)
                self.statistic[i, j] = d
                self.pvalue[i, j] = stats.studentized_range.sf(abs(d) / se, k, gl)
                self.low[i, j], self.high[i, j] = d - crit * se, d + crit * se
        if not _SCIPY_SEM_EQUAL_VAR:
            sp = stats.tukey_hsd(*amostras, equal_var=igual)
            ic = sp.confidence_interval(confidence_level=confianca)
            np.testing.assert_allclose(sp.pvalue[np.triu_indices(k, 1)], self.pvalue[np.triu_indices(k, 1)], atol=1e-6)
            np.testing.assert_allclose(ic.low[np.triu_indices(k, 1)], self.low[np.triu_indices(k, 1)], atol=1e-6)


def _scipy_doc():
    y = G0 + G1 + G2
    g = ["0"] * 5 + ["1"] * 5 + ["2"] * 5
    u = [f"u{i}" for i in range(15)]
    return y, g, u


# ---------------------------------------------------------------- oraculos

def test_oraculo_da_documentacao_do_scipy_tukey_hsd():
    """Valores publicados no exemplo de scipy.stats.tukey_hsd."""
    r = anova_uma_via(*_scipy_doc(), n_boot=0)
    assert r.pos_teste == "Tukey" and r.metodo == "ANOVA classica"
    c = {(x["a"], x["b"]): x for x in r.comparacoes}
    assert c[("0", "1")]["diferenca"] == pytest.approx(-4.600, abs=5e-4)
    assert c[("0", "1")]["p_ajustado"] == pytest.approx(0.014, abs=5e-4)
    assert c[("0", "1")]["ic95"][0] == pytest.approx(-8.249, abs=5e-4)
    assert c[("0", "1")]["ic95"][1] == pytest.approx(-0.951, abs=5e-4)
    assert c[("0", "2")]["diferenca"] == pytest.approx(-0.260, abs=5e-4)
    assert c[("0", "2")]["p_ajustado"] == pytest.approx(0.980, abs=5e-4)
    assert c[("1", "2")]["diferenca"] == pytest.approx(4.340, abs=5e-4)
    assert c[("1", "2")]["p_ajustado"] == pytest.approx(0.020, abs=5e-4)


def test_oraculo_independente_anova_feita_a_mao():
    """SQ/QM/F calculados a mao (formulas de livro), sem scipy."""
    gs = [G0, G1, G2]
    todos = [x for g in gs for x in g]
    m = sum(todos) / len(todos)
    sqe = sum(len(g) * (sum(g) / len(g) - m) ** 2 for g in gs)
    sqd = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in gs)
    F = (sqe / 2) / (sqd / 12)
    r = anova_uma_via(*_scipy_doc(), n_boot=0)
    assert r.tabela["entre"]["SQ"] == pytest.approx(sqe)
    assert r.tabela["dentro"]["SQ"] == pytest.approx(sqd)
    assert r.tabela["entre"]["gl"] == 2 and r.tabela["dentro"]["gl"] == 12
    assert r.F == pytest.approx(F)
    assert r.p == pytest.approx(stats.f_oneway(G0, G1, G2).pvalue)
    assert r.eta2 == pytest.approx(sqe / (sqe + sqd))
    assert r.omega2 == pytest.approx((sqe - 2 * sqd / 12) / (sqe + sqd + sqd / 12))


# ------------------------------------------------- unidade (fail-closed)

def test_sem_unidade_recusa_com_mensagem_acionavel():
    y, g, _ = _scipy_doc()
    with pytest.raises(ValueError, match="pseudo-replicacao") as e:
        anova_uma_via(y, g, None)
    assert "15 linha(s), 3 grupo(s)" in str(e.value)


def test_unidade_vazia_em_alguma_linha_recusa():
    y, g, u = _scipy_doc()
    u[3] = ""
    with pytest.raises(ValueError, match="sem id de unidade"):
        anova_uma_via(y, g, u)


@pytest.mark.parametrize("origem", ["dobras_cv", "replicas", "espectros"])
def test_dobras_de_cv_replicas_e_espectros_sao_recusados(origem):
    """Dobras de CV sao dependentes (treino compartilhado; Nadeau & Bengio
    2003): ANOVA sobre elas e' invalida e o modulo recusa."""
    with pytest.raises(ValueError, match="ANOVA recusada"):
        anova_uma_via(*_scipy_doc(), origem=origem)


def test_replicas_sao_agregadas_antes_do_teste():
    """3 replicas por amostra: o teste usa 5 unidades por grupo, nao 15."""
    y, g, u = [], [], []
    for gi, grupo in enumerate([G0, G1, G2]):
        for k, v in enumerate(grupo):
            for rep in (-0.1, 0.0, 0.1):
                y.append(v + rep)
                g.append(str(gi))
                u.append(f"g{gi}a{k}")
    r = anova_uma_via(y, g, u, n_boot=0)
    assert r.n_linhas == 45 and r.n_unidades == 15
    assert r.tabela["dentro"]["gl"] == 12          # nao 42
    assert r.p == pytest.approx(stats.f_oneway(G0, G1, G2).pvalue)


def test_unidade_em_dois_grupos_e_erro():
    with pytest.raises(ValueError, match="mais de um grupo"):
        agregar_por_unidade([1, 2, 3, 4], ["a", "b", "a", "b"], ["x", "x", "y", "z"])


# ------------------------------------------------------------ casos limite

def test_grupo_com_uma_unidade_e_excluido_com_aviso():
    y, g, u = _scipy_doc()
    y, g, u = y + [99.0], g + ["3"], u + ["unica"]
    r = anova_uma_via(y, g, u, n_boot=0)
    assert r.grupos == ["0", "1", "2"]
    assert any("1 unidade so'" in a for a in r.avisos)


def test_um_grupo_so_e_erro():
    with pytest.raises(ValueError, match="menos de 2 grupos"):
        anova_uma_via(G0, ["a"] * 5, list("abcde"))


def test_grupos_constantes_variancia_zero_e_erro():
    with pytest.raises(ValueError, match="variancia dentro dos grupos e' zero"):
        anova_uma_via([1, 1, 1, 2, 2, 2], list("aaabbb"), list("uvwxyz"))


def test_nan_e_descartado_com_aviso():
    y, g, u = _scipy_doc()
    y[0] = float("nan")
    r = anova_uma_via(y, g, u, n_boot=0)
    assert r.n_unidades == 14
    assert any("NaN" in a for a in r.avisos)


def test_poucas_unidades_avisa_baixo_poder_sem_bloquear():
    r = anova_uma_via(*_scipy_doc(), n_boot=0)
    assert 5 < N_MINIMO_PODER
    assert any("poder baixo" in a for a in r.avisos)


def test_variancias_desiguais_levam_a_welch_e_games_howell():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.1, 12)
    b = rng.normal(0, 5.0, 12)
    c = rng.normal(0, 0.1, 12)
    y = np.concatenate([a, b, c])
    g = ["a"] * 12 + ["b"] * 12 + ["c"] * 12
    r = anova_uma_via(y, g, [f"u{i}" for i in range(36)], n_boot=0)
    assert r.levene_p < 0.05
    assert r.metodo == "Welch ANOVA" and r.pos_teste == "Games-Howell"
    assert r.tabela["entre"]["p_welch"] == pytest.approx(r.p)


def test_desbalanceado_usa_tukey_kramer():
    y, g, u = _scipy_doc()
    r = anova_uma_via(y[:-1], g[:-1], u[:-1], n_boot=0)
    assert r.pos_teste == "Tukey-Kramer"


def test_kruskal_opcional_rotulado_como_postos():
    r = anova_uma_via(*_scipy_doc(), kruskal=True, n_boot=0)
    assert r.kruskal["p"] == pytest.approx(stats.kruskal(G0, G1, G2).pvalue)
    assert "POSTOS" in r.kruskal["nota"]


def test_ic_bootstrap_de_eta2_contem_a_estimativa():
    r = anova_uma_via(*_scipy_doc(), n_boot=500, seed=3)
    lo, hi = r.eta2_ic95
    assert lo <= r.eta2 <= hi


# -------------------------------------------------- varias respostas (FDR)

def test_varias_respostas_aplica_bh_entre_respostas():
    y, g, u = _scipy_doc()
    rng = np.random.default_rng(0)
    respostas = {"efeito": y, "ruido": list(rng.normal(0, 1, 15))}
    out = anova_varias_respostas(respostas, g, u, n_boot=0)
    ps = [out["resultados"][n].p for n in ("efeito", "ruido")]
    esperado = stats.false_discovery_control(ps, method="bh")
    assert [out["resultados"][n].p_fdr for n in ("efeito", "ruido")] == pytest.approx(list(esperado))
    assert "Benjamini-Hochberg" in out["correcao"]


def test_varias_respostas_lista_a_recusada_sem_derrubar_as_outras():
    y, g, u = _scipy_doc()
    out = anova_varias_respostas({"ok": y, "constante": [5.0] * 15}, g, u, n_boot=0)
    assert "ok" in out["resultados"] and "constante" in out["recusadas"]


def test_varias_respostas_sem_unidade_propaga_a_recusa():
    y, g, _ = _scipy_doc()
    with pytest.raises(ValueError, match="nao informada"):
        anova_varias_respostas({"a": y}, g, None)


# ------------------------------------------------------- bloco (2 vias)

def test_bloco_balanceado_confere_com_conta_a_mao():
    """Delineamento em blocos completos balanceado: SQ do fator e do bloco
    pelas formulas de livro (medias marginais)."""
    tratamentos = ["A", "B", "C"]
    blocos = ["s1", "s2", "s3", "s4"]
    dados = {("A", "s1"): 10.0, ("A", "s2"): 12.0, ("A", "s3"): 11.0, ("A", "s4"): 13.0,
             ("B", "s1"): 14.0, ("B", "s2"): 15.5, ("B", "s3"): 13.0, ("B", "s4"): 16.0,
             ("C", "s1"): 9.0, ("C", "s2"): 11.5, ("C", "s3"): 10.0, ("C", "s4"): 12.5}
    y = [dados[(t, b)] for t in tratamentos for b in blocos]
    g = [t for t in tratamentos for _ in blocos]
    bl = [b for _ in tratamentos for b in blocos]
    u = [f"{t}{b}" for t in tratamentos for b in blocos]
    m = sum(y) / len(y)
    sq_t = 4 * sum((sum(dados[(t, b)] for b in blocos) / 4 - m) ** 2 for t in tratamentos)
    sq_b = 3 * sum((sum(dados[(t, b)] for t in tratamentos) / 3 - m) ** 2 for b in blocos)
    sq_tot = sum((v - m) ** 2 for v in y)
    sq_res = sq_tot - sq_t - sq_b
    F = (sq_t / 2) / (sq_res / 6)
    r = anova_uma_via(y, g, u, bloco=bl, n_boot=0)
    assert r.metodo.startswith("ANOVA com bloco")
    assert r.tabela["entre"]["SQ"] == pytest.approx(sq_t)
    assert r.tabela["bloco"]["SQ"] == pytest.approx(sq_b)
    assert r.tabela["dentro"]["gl"] == 6
    assert r.F == pytest.approx(F)
    c = {(x["a"], x["b"]): x for x in r.comparacoes}
    medias = {t: sum(dados[(t, b)] for b in blocos) / 4 for t in tratamentos}
    assert c[("A", "B")]["diferenca"] == pytest.approx(medias["A"] - medias["B"])
    # balanceado: Tukey classico com QM residual do modelo com bloco
    q = abs(medias["A"] - medias["B"]) / math.sqrt((sq_res / 6) / 4)
    assert c[("A", "B")]["p_ajustado"] == pytest.approx(stats.studentized_range.sf(q, 3, 6))


# ------------------------------------- pseudo-replicacao (simulacao, H0)

def _simular_h0(rng, n_amostras=6, n_replicas=5, sd_amostra=1.0, sd_replica=0.3):
    y, g, u = [], [], []
    for gi in range(3):
        for a in range(n_amostras):
            efeito = rng.normal(0, sd_amostra)
            for _r in range(n_replicas):
                y.append(efeito + rng.normal(0, sd_replica))
                g.append(str(gi))
                u.append(f"g{gi}a{a}")
    return np.asarray(y), np.asarray(g), np.asarray(u)


def _ic_clopper_pearson(k, n, conf=0.95):
    a = 1 - conf
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


@pytest.mark.slow
def test_pseudo_replicacao_infla_falso_positivo_e_a_agregacao_corrige():
    """H0 (sem efeito de grupo), 3 grupos x 6 amostras x 5 replicas
    correlacionadas, 2000 execucoes. (a) ANOVA tratando cada replica como
    independente: falso positivo inflado (medido). (b) ANOVA do modulo
    (agregada por unidade): taxa compativel com alpha=0,05 -- o teste falha
    se 0,05 ficar fora do IC 95% de Clopper-Pearson da taxa observada."""
    rng = np.random.default_rng(20261002)
    n_exec, alpha = 2000, 0.05
    fp_replicas = fp_unidade = 0
    for _ in range(n_exec):
        y, g, u = _simular_h0(rng)
        fp_replicas += stats.f_oneway(*[y[g == k] for k in "012"]).pvalue < alpha
        fp_unidade += anova_uma_via(y, g, u, n_boot=0, calcular_pos_teste=False).p < alpha
    taxa_rep, taxa_uni = fp_replicas / n_exec, fp_unidade / n_exec
    lo, hi = _ic_clopper_pearson(fp_unidade, n_exec)
    print(f"\nfalso positivo: replicas={taxa_rep:.3f}; agregado={taxa_uni:.3f} "
          f"[IC95 {lo:.3f}; {hi:.3f}]")
    assert taxa_rep > 0.30                     # inflacao grosseira (medida ~0,5)
    assert lo <= alpha <= hi


@pytest.mark.slow
def test_poder_cresce_com_o_numero_de_unidades():
    """Efeito conhecido (delta = 1 dp entre amostras) no grupo '2'; poder
    estimado em 400 execucoes por n. Reportado e monotono."""
    rng = np.random.default_rng(7)
    poder = {}
    for n in (3, 6, 12):
        acertos = 0
        for _ in range(400):
            y, g, u = _simular_h0(rng, n_amostras=n)
            y = y + (g == "2") * 1.0
            acertos += anova_uma_via(y, g, u, n_boot=0, calcular_pos_teste=False).p < 0.05
        poder[n] = acertos / 400
    print(f"\npoder por n de unidades/grupo: {poder}")
    assert poder[3] < poder[6] < poder[12]
    assert poder[12] > 0.6


# ------------------------------------------------------------- hypothesis

@settings(deadline=None, max_examples=60)
@example(tamanhos=(2, 2), escala=1.0, empate=False)       # n=2 por grupo
@example(tamanhos=(2, 7, 3), escala=1.0, empate=False)    # desbalanceado
@example(tamanhos=(4, 4), escala=1.0, empate=True)        # empates
@example(tamanhos=(5, 5, 5), escala=1e6, empate=False)    # escala grande
@example(tamanhos=(3, 2, 2), escala=1e-3, empate=True)    # grupo de variancia zero (bug real: Welch -> NaN)
@given(tamanhos=st.lists(st.integers(2, 8), min_size=2, max_size=4).map(tuple),
       escala=st.sampled_from([1e-3, 1.0, 1e6]), empate=st.booleans())
def test_propriedades_basicas(tamanhos, escala, empate):
    rng = np.random.default_rng(sum(tamanhos))
    y, g, u = [], [], []
    for gi, n in enumerate(tamanhos):
        vals = np.round(rng.normal(gi * 0.1, 1, n)) if empate else rng.normal(gi * 0.1, 1, n)
        for k, v in enumerate(vals):
            y.append(float(v) * escala)
            g.append(f"g{gi}")
            u.append(f"g{gi}u{k}")
    try:
        r = anova_uma_via(y, g, u, n_boot=0)
    except ValueError as e:
        assert "variancia dentro dos grupos e' zero" in str(e)
        return
    assert 0.0 <= r.p <= 1.0
    assert 0.0 <= r.eta2 <= 1.0 + 1e-12
    assert len(r.comparacoes) == len(r.grupos) * (len(r.grupos) - 1) // 2
    for c in r.comparacoes:
        assert c["ic95"][0] <= c["diferenca"] <= c["ic95"][1]
        assert 0.0 <= c["p_ajustado"] <= 1.0


def test_sem_pos_teste_mantem_o_p_e_declara_que_nao_calculou():
    com = anova_uma_via(*_scipy_doc(), n_boot=0)
    sem = anova_uma_via(*_scipy_doc(), n_boot=0, calcular_pos_teste=False)
    assert sem.p == com.p and sem.comparacoes == []
    assert "nao calculado" in sem.pos_teste


# ------------------------------------- oraculos que a mutacao pediu (2026-10-02)

def _welch_a_mao(grupos):
    """Welch (1951), recodificado aqui de forma independente do modulo."""
    k = len(grupos)
    n = [len(g) for g in grupos]
    m = [sum(g) / len(g) for g in grupos]
    v = [sum((x - mi) ** 2 for x in g) / (len(g) - 1) for g, mi in zip(grupos, m)]
    w = [ni / vi for ni, vi in zip(n, v)]
    W = sum(w)
    mw = sum(wi * mi for wi, mi in zip(w, m)) / W
    num = sum(wi * (mi - mw) ** 2 for wi, mi in zip(w, m)) / (k - 1)
    lam = sum((1 - wi / W) ** 2 / (ni - 1) for wi, ni in zip(w, n))
    F = num / (1 + 2 * (k - 2) * lam / (k * k - 1))
    gl2 = (k * k - 1) / (3 * lam)
    return F, k - 1, gl2


def _grupos_heterocedasticos(k):
    rng = np.random.default_rng(11)
    escalas = [0.2, 4.0, 0.5, 2.0][:k]
    tamanhos = [9, 14, 11, 7][:k]
    return [list(rng.normal(i * 0.5, s, n)) for i, (s, n) in enumerate(zip(escalas, tamanhos))]


def _rodar(grupos, **kw):
    y = [x for g in grupos for x in g]
    gr = [f"g{i}" for i, g in enumerate(grupos) for _ in g]
    u = [f"u{i}" for i in range(len(y))]
    return anova_uma_via(y, gr, u, n_boot=0, **kw)


def test_welch_com_2_grupos_e_o_t_de_welch_ao_quadrado():
    """Oraculo independente: com k=2, F de Welch = t^2 do teste t de Welch
    (scipy.stats.ttest_ind equal_var=False) e gl2 = gl de Welch-Satterthwaite."""
    g = _grupos_heterocedasticos(2)
    r = _rodar(g)
    assert r.metodo == "Welch ANOVA"
    t = stats.ttest_ind(g[0], g[1], equal_var=False)
    assert r.F == pytest.approx(t.statistic ** 2)
    assert r.tabela["entre"]["gl2_welch"] == pytest.approx(t.df)
    assert r.tabela["entre"]["gl1_welch"] == 1
    assert r.p == pytest.approx(t.pvalue)


@pytest.mark.parametrize("k", [3, 4])
def test_welch_com_k_grupos_confere_com_a_formula_de_welch(k):
    g = _grupos_heterocedasticos(k)
    r = _rodar(g)
    assert r.metodo == "Welch ANOVA"
    F, gl1, gl2 = _welch_a_mao(g)
    assert r.F == pytest.approx(F)
    assert r.tabela["entre"]["gl1_welch"] == gl1
    assert r.tabela["entre"]["gl2_welch"] == pytest.approx(gl2)
    assert r.p == pytest.approx(stats.f.sf(F, gl1, gl2))


def test_bloco_tabela_completa_e_ic_das_comparacoes():
    tratamentos, blocos = ["A", "B", "C"], ["s1", "s2", "s3", "s4"]
    vals = [10.0, 12.0, 11.0, 13.0, 14.0, 15.5, 13.0, 16.0, 9.0, 11.5, 10.0, 12.5]
    g = [t for t in tratamentos for _ in blocos]
    bl = [b for _ in tratamentos for b in blocos]
    u = [f"{t}{b}" for t, b in zip(g, bl)]
    r = anova_uma_via(vals, g, u, bloco=bl, n_boot=0)
    m = sum(vals) / 12
    sq_tot = sum((v - m) ** 2 for v in vals)
    tb = r.tabela
    assert tb["total"]["SQ"] == pytest.approx(sq_tot) and tb["total"]["gl"] == 11
    assert tb["entre"]["QM"] == pytest.approx(tb["entre"]["SQ"] / 2)
    assert tb["bloco"]["gl"] == 3
    qm_res = tb["dentro"]["SQ"] / 6
    assert tb["bloco"]["F"] == pytest.approx((tb["bloco"]["SQ"] / 3) / qm_res)
    assert tb["bloco"]["p"] == pytest.approx(stats.f.sf(tb["bloco"]["F"], 3, 6))
    se = math.sqrt(2 * qm_res / 4)
    crit = stats.studentized_range.ppf(0.95, 3, 6) / math.sqrt(2)
    for c in r.comparacoes:
        assert c["ic95"][0] == pytest.approx(c["diferenca"] - crit * se)
        assert c["ic95"][1] == pytest.approx(c["diferenca"] + crit * se)


def test_bloco_sem_graus_de_liberdade_residuais_e_erro():
    with pytest.raises(ValueError, match="graus de liberdade residuais"):
        # cada bloco contem um grupo so': grupo e bloco ficam confundidos
        anova_uma_via([1, 2, 3, 4.5], ["a", "a", "b", "b"], ["u1", "u2", "u3", "u4"],
                      bloco=["s1", "s2", "s3", "s4"], n_boot=0)


def test_ic_bootstrap_reproduz_a_reamostragem_por_unidade():
    """Mesma reamostragem (por grupo, com reposicao, semente fixa) feita aqui."""
    r = anova_uma_via(*_scipy_doc(), n_boot=300, seed=5)
    rng = np.random.default_rng(5)
    gs = [np.array(G0), np.array(G1), np.array(G2)]
    e = []
    for _ in range(300):
        re = [a[rng.integers(0, len(a), len(a))] for a in gs]
        todos = np.concatenate(re)
        sst = ((todos - todos.mean()) ** 2).sum()
        if sst == 0:
            continue
        sse = sum(len(x) * (x.mean() - todos.mean()) ** 2 for x in re)
        e.append(sse / sst)
    lo, hi = np.percentile(e, [2.5, 97.5])
    assert r.eta2_ic95 == pytest.approx((lo, hi))


def test_shapiro_usa_os_residuos_e_avisa_so_quando_deve():
    rng = np.random.default_rng(2)
    normais = [list(rng.normal(i, 1, 15)) for i in range(2)]      # 30 unidades
    r = _rodar(normais)
    res = np.concatenate([np.array(g) - np.mean(g) for g in normais])
    assert r.shapiro_p == pytest.approx(stats.shapiro(res).pvalue)
    assert not any("Shapiro com poucas" in a for a in r.avisos)    # >= 20 unidades
    assert not any("desvio de normalidade" in a for a in r.avisos)
    tortos = [list(np.exp(rng.normal(0, 1.5, 15)) + i) for i in range(2)]
    r2 = _rodar(tortos)
    assert r2.shapiro_p < 0.05 and any("desvio de normalidade" in a for a in r2.avisos)


def test_variancia_nao_nula_com_desvios_inteiros_nao_e_confundida_com_zero():
    r = anova_uma_via([0.0, 4.0, 10.0, 14.0], ["a", "a", "b", "b"],
                      ["u1", "u2", "u3", "u4"], n_boot=0)
    assert r.tabela["dentro"]["SQ"] == pytest.approx(16.0)


def test_omega2_em_escala_pequena():
    y, g, u = _scipy_doc()
    r = anova_uma_via([v * 1e-3 for v in y], g, u, n_boot=0)
    base = anova_uma_via(y, g, u, n_boot=0)
    assert r.omega2 == pytest.approx(base.omega2) and r.eta2 == pytest.approx(base.eta2)


def test_tamanhos_diferentes_sao_erro():
    with pytest.raises(ValueError, match="mesmo tamanho"):
        agregar_por_unidade([1, 2, 3], ["a", "b"], ["x", "y", "z"])


def test_aviso_de_linha_igual_a_unidade_so_aparece_sem_replicas():
    r = anova_uma_via(*_scipy_doc(), n_boot=0)
    assert any("cada linha e' uma unidade" in a for a in r.avisos)
    y, g, u = _scipy_doc()
    r2 = anova_uma_via(y + y, g + g, u + u, n_boot=0)
    assert not any("cada linha e' uma unidade" in a for a in r2.avisos)


# ------------------------- sobreviventes da reaplicacao de mutantes (2026-10-02)

def test_tamanhos_em_ordem_crescente_tambem_sao_erro():
    """`==` encadeado virando `<=` passava com len(y) < len(g) == len(u)."""
    with pytest.raises(ValueError, match="mesmo tamanho"):
        agregar_por_unidade([1, 2], ["a", "b", "c"], ["x", "y", "z"])


def test_bloco_com_zero_gl_residual_mas_fator_estimavel_e_erro():
    # a: s1,s2 ; b: s1,s3 -> 4 parametros para 4 unidades: gl residual 0, gl do fator 1
    with pytest.raises(ValueError, match="graus de liberdade residuais"):
        anova_uma_via([1.0, 2.0, 3.5, 4.0], ["a", "a", "b", "b"], ["u1", "u2", "u3", "u4"],
                      bloco=["s1", "s2", "s1", "s3"], n_boot=0)


def test_bloco_confundido_com_o_grupo_e_erro_mesmo_com_gl_residual():
    # a so' em s1, b so' em s2: o fator nao e' estimavel (gl do fator 0), gl residual 2
    with pytest.raises(ValueError, match="graus de liberdade residuais"):
        anova_uma_via([1.0, 2.0, 3.5, 4.0], ["a", "a", "b", "b"], ["u1", "u2", "u3", "u4"],
                      bloco=["s1", "s1", "s2", "s2"], n_boot=0)


def test_f_do_bloco_com_qm_pequeno_e_dois_blocos():
    tratamentos, blocos = ["A", "B", "C"], ["s1", "s2"]
    vals = [0.010, 0.013, 0.015, 0.019, 0.009, 0.0125]
    g = [t for t in tratamentos for _ in blocos]
    bl = [b for _ in tratamentos for b in blocos]
    u = [f"{t}{b}" for t, b in zip(g, bl)]
    r = anova_uma_via(vals, g, u, bloco=bl, n_boot=0)
    tb = r.tabela
    qm_res = tb["dentro"]["SQ"] / tb["dentro"]["gl"]
    assert 0 < qm_res < 1 and tb["bloco"]["gl"] == 1
    assert tb["bloco"]["F"] == pytest.approx(tb["bloco"]["SQ"] / qm_res)
    assert math.isfinite(tb["bloco"]["p"])


def _confere_pos_teste_contra_scipy(tamanhos, escalas):
    from guaraci.anova_grupos import _pos_teste_par_a_par
    rng = np.random.default_rng(sum(tamanhos))
    amostras = [rng.normal(i * 0.3, escalas[i], n) for i, n in enumerate(tamanhos)]
    nomes = [f"g{i}" for i in range(len(amostras))]
    for igual in (True, False):
        ref = _Ref(amostras, igual)
        meus = {(c["a"], c["b"]): c for c in _pos_teste_par_a_par(amostras, nomes, igual, 0.05)}
        for i in range(len(amostras)):
            for j in range(i + 1, len(amostras)):
                c = meus[(nomes[i], nomes[j])]
                assert c["diferenca"] == pytest.approx(ref.statistic[i, j])
                assert c["p_ajustado"] == pytest.approx(ref.pvalue[i, j], abs=1e-6)
                assert c["ic95"][0] == pytest.approx(ref.low[i, j], abs=1e-6)
                assert c["ic95"][1] == pytest.approx(ref.high[i, j], abs=1e-6)


# Os tres casos fixos do Hypothesis, SEM Hypothesis: ficam na camada rapida (o teste com busca aleatoria e' `slow`).
@pytest.mark.parametrize("tamanhos, escalas", [
    ((5, 5, 5), (1.0, 1.0, 1.0)),
    ((4, 9, 6), (0.2, 5.0, 1.0)),
    ((3, 3), (1.0, 8.0)),
])
def test_pos_teste_proprio_igual_ao_tukey_hsd_do_scipy_nos_exemplos_fixos(tamanhos, escalas):
    _confere_pos_teste_contra_scipy(tamanhos, escalas)


@pytest.mark.slow          # ~190 s (busca aleatoria); o gemeo deterministico e os exemplos fixos acima ficam na camada rapida
@settings(deadline=None, max_examples=40)
@example(tamanhos=(5, 5, 5), escalas=(1.0, 1.0, 1.0))
@example(tamanhos=(4, 9, 6), escalas=(0.2, 5.0, 1.0))
@example(tamanhos=(3, 3), escalas=(1.0, 8.0))
@given(tamanhos=st.lists(st.integers(3, 9), min_size=2, max_size=5).map(tuple),
       escalas=st.lists(st.sampled_from([0.2, 1.0, 5.0]), min_size=5, max_size=5).map(tuple))
def test_pos_teste_proprio_igual_ao_tukey_hsd_do_scipy(tamanhos, escalas):
    """O pos-teste foi reescrito (cache do quantil critico, ~5x mais rapido com
    muitas respostas); aqui ele e' conferido contra scipy.stats.tukey_hsd nos
    dois casos (variancias iguais = Tukey-Kramer; desiguais = Games-Howell)."""
    _confere_pos_teste_contra_scipy(tamanhos, escalas)


@pytest.mark.parametrize("igual", [True, False])
def test_pos_teste_proprio_igual_ao_scipy_em_casos_fixos(igual):
    """Versao deterministica do teste com Hypothesis (rapida o bastante para a
    mutacao): desbalanceado e com variancias diferentes."""
    from guaraci.anova_grupos import _pos_teste_par_a_par
    rng = np.random.default_rng(42)
    amostras = [rng.normal(0.0, 0.5, 6), rng.normal(0.8, 2.0, 9), rng.normal(0.3, 1.0, 4),
                rng.normal(1.5, 0.7, 7)]
    nomes = ["a", "b", "c", "d"]
    ref = _Ref(amostras, igual)
    meus = {(c["a"], c["b"]): c for c in _pos_teste_par_a_par(amostras, nomes, igual, 0.05)}
    assert len(meus) == 6
    for i in range(4):
        for j in range(i + 1, 4):
            c = meus[(nomes[i], nomes[j])]
            assert c["diferenca"] == pytest.approx(ref.statistic[i, j])
            assert c["p_ajustado"] == pytest.approx(ref.pvalue[i, j], abs=1e-6)
            assert c["ic95"] == pytest.approx((ref.low[i, j], ref.high[i, j]), abs=1e-6)


def test_pos_teste_com_alfa_diferente_muda_o_ic():
    from guaraci.anova_grupos import _pos_teste_par_a_par
    am = [np.array(G0), np.array(G1), np.array(G2)]
    c95 = _pos_teste_par_a_par(am, ["0", "1", "2"], True, 0.05)[0]
    c99 = _pos_teste_par_a_par(am, ["0", "1", "2"], True, 0.01)[0]
    ref = stats.tukey_hsd(*am).confidence_interval(confidence_level=0.99)
    assert c99["ic95"] == pytest.approx((ref.low[0, 1], ref.high[0, 1]), abs=1e-6)
    assert (c99["ic95"][1] - c99["ic95"][0]) > (c95["ic95"][1] - c95["ic95"][0])


def _tres_grupos(n_pequeno, n_grande=12, dp_pequeno=5.0):
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(0, 0.1, n_grande), rng.normal(0, dp_pequeno, n_pequeno), rng.normal(0, 0.1, n_grande)])
    g = ["a"] * n_grande + ["b"] * n_pequeno + ["c"] * n_grande
    return y, g, [f"u{i}" for i in range(len(y))]


def test_games_howell_com_grupo_de_menos_de_6_unidades_avisa_o_erro_por_familia_medido():
    r = anova_uma_via(*_tres_grupos(5), n_boot=0)
    assert r.pos_teste == "Games-Howell"
    av = [a for a in r.avisos if a.startswith("Games-Howell com grupo(s)")]
    assert len(av) == 1 and "0.27" in av[0] and "['b']" in av[0] and "exploratorias" in av[0]


def test_games_howell_sem_grupo_pequeno_nao_traz_o_aviso():
    r = anova_uma_via(*_tres_grupos(6), n_boot=0)             # 6 = limite: o aviso e' so' para < 6
    assert r.pos_teste == "Games-Howell"
    assert not any(a.startswith("Games-Howell com grupo(s)") for a in r.avisos)


def test_tukey_com_grupo_pequeno_nao_traz_o_aviso_do_games_howell():
    r = anova_uma_via(*_scipy_doc(), n_boot=0)                # 5 por grupo, variancias iguais -> Tukey
    assert r.pos_teste == "Tukey"
    assert not any("Games-Howell" in a for a in r.avisos)
    assert any("poder baixo" in a for a in r.avisos)
