"""Testes escritos para matar sobreviventes da mutacao de `anova_grupos.py` (cada um cita o que fecha).

Limiares sao conferidos com NUMEROS LITERAIS, nunca com a constante do modulo: um teste que importa a
constante para testar a borda acompanha o mutante e nao pega nada.
"""
import numpy as np
import pytest
from scipy import stats

from guaraci.anova_grupos import (
    anova_uma_via, anova_varias_respostas, diagnosticar_premissas, kruskal_dunn, recomendar_metodo)


def _dados(*grupos):
    y, g, u = [], [], []
    for i, a in enumerate(grupos):
        for j, v in enumerate(a):
            y.append(v)
            g.append(str(i))
            u.append(f"{i}-{j}")
    return y, g, u


def _rng_grupos(n, k=2, seed=0):
    r = np.random.default_rng(seed)
    return [r.normal(i * 0.2, 1.0, n) for i in range(k)]


# --------------------------------------------------- limiares com numeros literais

def test_poucas_unidades_avisa_com_5_e_nao_com_6_por_grupo_no_anova_e_no_kruskal():
    for n, avisa in ((5, True), (6, False)):
        g = _rng_grupos(n)
        y, gr, u = _dados(*g)
        assert any("poder baixo" in a for a in anova_uma_via(y, gr, u, n_boot=0).avisos) is avisa
        assert any("poder baixo" in a for a in kruskal_dunn(y, gr, u).avisos) is avisa


def _d(**kw):
    base = {"assimetria": 0.0, "curtose_excesso": 0.0, "razao_variancias": 1.0, "brown_forsythe_p": 0.5, "alpha": 0.05}
    base.update(kw)
    return base


def test_limiares_da_recomendacao_sao_4_1_e_3_literais():
    assert recomendar_metodo(_d(razao_variancias=4.0))["metodo"] == "tukey"
    assert recomendar_metodo(_d(razao_variancias=4.01))["metodo"] == "games-howell"
    assert recomendar_metodo(_d(assimetria=1.0))["metodo"] == "tukey"
    assert recomendar_metodo(_d(assimetria=1.01))["metodo"] == "kruskal-dunn"
    assert recomendar_metodo(_d(assimetria=-1.01))["metodo"] == "kruskal-dunn"
    assert recomendar_metodo(_d(curtose_excesso=3.0))["metodo"] == "tukey"
    assert recomendar_metodo(_d(curtose_excesso=3.01))["metodo"] == "kruskal-dunn"


# --------------------------------------------------- tabela, bootstrap e defaults

def test_tabela_total_tem_gl_n_menos_1_e_soma_dos_quadrados():
    g = _rng_grupos(7, k=3)
    r = anova_uma_via(*_dados(*g), n_boot=0)
    t = r.tabela
    todos = np.concatenate(g)
    assert t["total"]["gl"] == 20 and t["entre"]["gl"] == 2 and t["dentro"]["gl"] == 18
    assert t["total"]["SQ"] == pytest.approx(((todos - todos.mean()) ** 2).sum())
    assert t["total"]["SQ"] == pytest.approx(t["entre"]["SQ"] + t["dentro"]["SQ"])


def _eta_omega_ref(amostras):
    todos = np.concatenate(amostras)
    k, n = len(amostras), len(todos)
    ssb = sum(len(a) * (a.mean() - todos.mean()) ** 2 for a in amostras)
    sse = sum(((a - a.mean()) ** 2).sum() for a in amostras)
    msw = sse / (n - k)
    return ssb / (ssb + sse), (ssb - (k - 1) * msw) / (ssb + sse + msw)


@pytest.mark.parametrize("grupos", [[[1.0, 2.0], [5.0, 6.0]], [[1.0, 2.0, 4.0], [5.0, 6.0, 9.0], [2.0, 3.0, 7.0]]])
def test_ic_bootstrap_reproduz_a_reamostragem_com_descarte_da_degenerada(grupos):
    """Resultado exato do bootstrap por unidade: mesma sequencia de sorteios, descartando a reamostragem
    cuja variacao DENTRO dos grupos e' zero (fecha as mutacoes do descarte e dos percentis 2,5/97,5)."""
    amostras = [np.array(g) for g in grupos]
    y, gr, u = _dados(*amostras)
    r = anova_uma_via(y, gr, u, n_boot=400, seed=3, calcular_pos_teste=False)
    rng = np.random.default_rng(3)
    e, o = [], []
    for _ in range(400):
        re = [a[rng.integers(0, len(a), len(a))] for a in amostras]
        if sum(((x - x.mean()) ** 2).sum() for x in re) == 0:
            continue
        ee, oo = _eta_omega_ref(re)
        e.append(ee)
        o.append(oo)
    assert r.eta2_ic95 == pytest.approx(tuple(np.percentile(e, [2.5, 97.5])), rel=1e-9)
    assert r.omega2_ic95 == pytest.approx(tuple(np.percentile(o, [2.5, 97.5])), rel=1e-9)
    assert len(e) < 400 or len(grupos[0]) > 2          # o caso de 2 por grupo descarta de fato reamostragens


def test_valores_padrao_n_boot_2000_seed_0_e_sem_kruskal():
    y, g, u = _dados(*_rng_grupos(6, k=3))
    padrao = anova_uma_via(y, g, u, calcular_pos_teste=False)
    explicito = anova_uma_via(y, g, u, n_boot=2000, seed=0, kruskal=False, calcular_pos_teste=False)
    assert padrao == explicito
    assert padrao.kruskal is None


# --------------------------------------------------- avisos e condicoes de Shapiro

def test_aviso_de_linha_unica_aparece_mesmo_com_nan_e_some_com_replicas_e_nan():
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, float("nan"), 8.0]
    g = ["a"] * 4 + ["b"] * 4
    r = anova_uma_via(y, g, [f"u{i}" for i in range(8)], n_boot=0)
    assert any("NAO sao replicas" in a for a in r.avisos)
    y2 = [1.0, 1.1, 2.0, 2.1, 5.0, 5.1, float("nan"), 6.1]
    r2 = anova_uma_via(y2, g, ["a", "a", "b", "b", "c", "c", "d", "d"], n_boot=0)
    assert not any("NAO sao replicas" in a for a in r2.avisos)


def test_shapiro_roda_em_escala_minuscula_e_nos_limites_de_5000_e_5001_unidades():
    r = anova_uma_via([0.0, 0.1, 0.3, 0.5, 0.2, 0.9], ["a"] * 3 + ["b"] * 3, list("uvwxyz"), n_boot=0)
    assert r.shapiro_p is not None                      # amplitude dos residuos < 1: ainda calcula
    rng = np.random.default_rng(1)
    for n_tot, esperado in ((5000, True), (5001, False)):
        y = rng.normal(0, 1, n_tot)
        g = ["a"] * (n_tot // 2) + ["b"] * (n_tot - n_tot // 2)
        r = anova_uma_via(y, g, [f"u{i}" for i in range(n_tot)], n_boot=0, calcular_pos_teste=False)
        assert (r.shapiro_p is not None) is esperado


def test_aviso_de_shapiro_com_poucas_unidades_some_com_20():
    rng = np.random.default_rng(2)
    for n_tot, avisa in ((19, True), (20, False)):
        y = rng.normal(0, 1, n_tot)
        g = ["a"] * 10 + ["b"] * (n_tot - 10)
        r = anova_uma_via(y, g, [f"u{i}" for i in range(n_tot)], n_boot=0, calcular_pos_teste=False)
        assert r.shapiro_p > 0.05                       # sem rejeicao: so' o aviso de pouco poder esta em jogo
        assert any("pouco poder" in a for a in r.avisos) is avisa


# --------------------------------------------------- bloco

def test_bloco_com_um_grau_de_liberdade_residual_e_aceito():
    r = anova_uma_via([1.0, 2.0, 3.0, 5.0], ["a", "a", "b", "b"], list("wxyz"), bloco=["p", "q", "p", "q"], n_boot=0)
    assert r.tabela["dentro"]["gl"] == 1 and r.tabela["entre"]["gl"] == 1 and np.isfinite(r.p)


def test_bloco_unico_nao_tem_f_de_bloco_e_o_qm_do_bloco_e_sq_sobre_gl():
    y = [1.0, 2.0, 3.0, 5.0, 6.0, 8.0]
    r = anova_uma_via(y, ["a"] * 3 + ["b"] * 3, list("uvwxyz"), bloco=["z"] * 6, n_boot=0)
    b = r.tabela["bloco"]
    assert b["gl"] == 0 and np.isnan(b["QM"]) and np.isnan(b["F"]) and np.isnan(b["p"])
    y2 = [1.0, 2.0, 4.0, 3.0, 5.0, 8.0, 2.0, 3.0, 7.0, 5.0, 6.0, 9.0]
    r2 = anova_uma_via(y2, ["a"] * 6 + ["b"] * 6, [f"u{i}" for i in range(12)],
                       bloco=["x", "y", "z"] * 4, n_boot=0)
    t = r2.tabela
    assert t["bloco"]["QM"] == pytest.approx(t["bloco"]["SQ"] / t["bloco"]["gl"])
    assert t["entre"]["QM"] == pytest.approx(t["entre"]["SQ"] / t["entre"]["gl"])
    assert t["bloco"]["F"] == pytest.approx(t["bloco"]["QM"] / t["dentro"]["QM"])


def test_bloco_comparacoes_batem_com_o_modelo_aditivo_feito_a_mao():
    """3 grupos x 4 blocos, uma observacao por celula (balanceado): diferenca = diferenca de medias,
    q = |d| / sqrt(QMres / 4); p e IC pela amplitude studentizada (Tukey)."""
    rng = np.random.default_rng(4)
    efeito_g = {"a": 0.0, "b": 1.0, "c": 2.5}
    efeito_b = [0.0, 0.7, -0.4, 1.1]
    y, g, b, u = [], [], [], []
    for gi, ge in efeito_g.items():
        for bi, be in enumerate(efeito_b):
            y.append(10 + ge + be + rng.normal(0, 0.3))
            g.append(gi)
            b.append(f"b{bi}")
            u.append(f"{gi}{bi}")
    r = anova_uma_via(y, g, u, bloco=b, n_boot=0)
    y_a, g_a, b_a = np.array(y), np.array(g), np.array(b)
    X = np.column_stack([np.ones(12)] + [(g_a == n).astype(float) for n in ("b", "c")]
                        + [(b_a == f"b{i}").astype(float) for i in (1, 2, 3)])
    beta, *_ = np.linalg.lstsq(X, y_a, rcond=None)
    qm = float(((y_a - X @ beta) ** 2).sum()) / (12 - 6)
    se = np.sqrt(qm / 4)                                   # erro-padrao da media do grupo (4 blocos)
    medias = {n: y_a[g_a == n].mean() for n in "abc"}
    for c in r.comparacoes:
        d = medias[c["a"]] - medias[c["b"]]
        assert c["diferenca"] == pytest.approx(d, rel=1e-9)
        assert c["p_ajustado"] == pytest.approx(stats.studentized_range.sf(abs(d) / se, 3, 6), rel=1e-6)
        crit = stats.studentized_range.isf(0.05, 3, 6)
        assert c["ic95"] == pytest.approx((d - crit * se, d + crit * se), rel=1e-6)


# --------------------------------------------------- varias respostas

def test_unidade_em_dois_grupos_vira_resposta_recusada_e_nao_derruba_as_demais():
    g = ["a"] * 4 + ["b"] * 4
    u = ["x", "x", "y", "z", "x", "p", "q", "r"]       # a unidade 'x' esta nos dois grupos
    out = anova_varias_respostas({"r1": list(range(8)), "r2": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]}, g, u, n_boot=0)
    assert set(out["recusadas"]) == {"r1", "r2"} and "mais de um grupo" in out["recusadas"]["r1"]


# --------------------------------------------------- Kruskal/Dunn: postos e n minimo

def test_posto_medio_e_diferenca_de_postos_do_dunn_batem_com_rankdata():
    g = [[3.0, 1.0, 4.0, 1.5], [5.0, 9.0, 2.0, 6.0, 5.0], [8.0, 7.0, 7.5]]
    y, gr, u = _dados(*g)
    r = kruskal_dunn(y, gr, u)
    postos = stats.rankdata(np.concatenate(g))
    cortes = np.cumsum([0] + [len(x) for x in g])
    pm = [postos[cortes[i]:cortes[i + 1]].mean() for i in range(3)]
    assert [r.posto_medio_por_grupo[str(i)] for i in range(3)] == pytest.approx(pm)
    for c in r.comparacoes:
        i, j = int(c["a"]), int(c["b"])
        assert c["diferenca_postos"] == pytest.approx(pm[i] - pm[j])


def test_grupo_de_2_unidades_e_mantido_no_kruskal_e_de_1_e_excluido():
    y, g, u = _dados([1.0, 2.0], [5.0, 6.0], [9.0])
    r = kruskal_dunn(y, g, u)
    assert r.grupos == ["0", "1"] and r.n_por_grupo == {"0": 2, "1": 2}


# --------------------------------------------------- diagnostico

def test_diagnostico_com_n_desigual_usa_variancia_amostral_e_forma_padronizada_por_grupo():
    a = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 3.0, 5.0])        # assimetrico, media alta
    b = np.array([100.0, 100.5, 101.0])                       # media bem diferente, n pequeno, amplitude 1
    c = np.array([50.0, 50.2])                                # n = 2, amplitude < 1
    y, g, u = _dados(a, b, c)
    d = diagnosticar_premissas(y, g, u)
    assert d["grupos"] == ["0", "1", "2"]
    assert d["variancia_por_grupo"] == pytest.approx({"0": a.var(ddof=1), "1": b.var(ddof=1), "2": c.var(ddof=1)})
    assert d["razao_variancias"] == pytest.approx(a.var(ddof=1) / c.var(ddof=1))
    pad = np.concatenate([(x - x.mean()) / x.std(ddof=1) for x in (a, b, c)])
    assert d["assimetria"] == pytest.approx(stats.skew(pad))
    assert d["curtose_excesso"] == pytest.approx(stats.kurtosis(pad))
    assert d["razao_n"] == pytest.approx(3.5)


def test_diagnostico_mantem_grupos_de_2_descarta_de_1_e_trata_tudo_constante():
    y, g, u = _dados([1.0, 2.0], [5.0, 7.0], [9.0])
    d = diagnosticar_premissas(y, g, u)
    assert d["grupos"] == ["0", "1"]
    d2 = diagnosticar_premissas(*_dados([5.0] * 3, [5.0] * 3))
    assert d2["assimetria"] == 0.0 and d2["curtose_excesso"] == 0.0


def test_diagnostico_calcula_shapiro_com_5000_residuos_e_nao_com_5001():
    rng = np.random.default_rng(3)
    for n_tot, esperado in ((5000, True), (5001, False)):
        y = rng.normal(0, 1, n_tot)
        g = ["a"] * (n_tot // 2) + ["b"] * (n_tot - n_tot // 2)
        d = diagnosticar_premissas(y, g, [f"u{i}" for i in range(n_tot)])
        assert (d["shapiro_p"] is not None) is esperado


def test_bloco_desbalanceado_confere_f_p_e_comparacoes_com_minimos_quadrados_independentes():
    """Desenho desbalanceado (4, 3 e 5 unidades; blocos desiguais): F e p do fator e do bloco, e
    diferenca/erro-padrao/p/IC de cada par, por contrastes sobre (X'X)^-1 do modelo completo."""
    rng = np.random.default_rng(9)
    grupos = ["a"] * 4 + ["b"] * 3 + ["c"] * 5
    blocos = ["x", "y", "x", "z", "y", "y", "x", "z", "x", "y", "z", "z"]
    y = np.array([10.0 + {"a": 0, "b": 1.2, "c": 2.0}[g] + {"x": 0, "y": 0.8, "z": -0.5}[b] + rng.normal(0, 0.4)
                  for g, b in zip(grupos, blocos)])
    r = anova_uma_via(y, grupos, [f"u{i}" for i in range(12)], bloco=blocos, n_boot=0)
    g, b = np.array(grupos), np.array(blocos)
    um = np.ones(12)
    dg = [(g == n).astype(float) for n in ("b", "c")]
    db = [(b == n).astype(float) for n in ("y", "z")]

    def ajuste(*cols):
        X = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return X, beta, float(((y - X @ beta) ** 2).sum()), np.linalg.matrix_rank(X)

    X, beta, sse, rk = ajuste(um, *dg, *db)
    _, _, sse_sem_g, rk_sem_g = ajuste(um, *db)
    _, _, sse_sem_b, rk_sem_b = ajuste(um, *dg)
    gl_res = 12 - rk
    qm = sse / gl_res
    F_g = ((sse_sem_g - sse) / (rk - rk_sem_g)) / qm
    F_b = ((sse_sem_b - sse) / (rk - rk_sem_b)) / qm
    assert r.F == pytest.approx(F_g, rel=1e-9)
    assert r.p == pytest.approx(stats.f.sf(F_g, rk - rk_sem_g, gl_res), rel=1e-9)
    assert r.tabela["bloco"]["F"] == pytest.approx(F_b, rel=1e-9)
    assert r.tabela["bloco"]["p"] == pytest.approx(stats.f.sf(F_b, rk - rk_sem_b, gl_res), rel=1e-9)
    assert r.tabela["dentro"]["SQ"] == pytest.approx(sse) and r.tabela["dentro"]["gl"] == gl_res
    cov = qm * np.linalg.pinv(X.T @ X)
    contrastes = {("a", "b"): np.array([0, -1, 0, 0, 0.0]), ("a", "c"): np.array([0, 0, -1, 0, 0.0]),
                  ("b", "c"): np.array([0, 1, -1, 0, 0.0])}
    for (n1, n2), c in contrastes.items():
        d, se = float(c @ beta), float(np.sqrt(c @ cov @ c))
        par = next(x for x in r.comparacoes if {x["a"], x["b"]} == {n1, n2})
        sinal = 1.0 if (par["a"], par["b"]) == (n1, n2) else -1.0
        assert par["diferenca"] == pytest.approx(sinal * d, rel=1e-9)
        assert par["p_ajustado"] == pytest.approx(
            min(1.0, stats.studentized_range.sf(abs(d) / se * np.sqrt(2), 3, gl_res)), rel=1e-6)
        crit = stats.studentized_range.isf(0.05, 3, gl_res) / np.sqrt(2)
        assert (par["ic95"][1] - par["ic95"][0]) == pytest.approx(2 * crit * se, rel=1e-6)
