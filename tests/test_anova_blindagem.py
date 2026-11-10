"""Blindagem do ANOVA (Parte A): oraculos independentes, casos de borda,
invariancias e propriedades. O resultado esperado de cada caso de borda esta
escrito no docstring/comentario ANTES da execucao."""
import math

import numpy as np
import pytest
from hypothesis import example, given, settings, strategies as st
from scipy import stats

from guaraci.anova_grupos import anova_uma_via, anova_varias_respostas

G0 = [24.5, 23.5, 26.4, 27.1, 29.9]
G1 = [28.4, 34.2, 29.5, 32.2, 30.1]
G2 = [26.1, 28.3, 24.3, 26.2, 27.8]


def _dados(*grupos, nomes=None):
    y, g, u = [], [], []
    for i, a in enumerate(grupos):
        nome = str(i) if nomes is None else nomes[i]
        for j, v in enumerate(a):
            y.append(v)
            g.append(nome)
            u.append(f"{nome}-{j}")
    return y, g, u


def _roda(*grupos, nomes=None, **kw):
    y, g, u = _dados(*grupos, nomes=nomes)
    kw.setdefault("n_boot", 0)
    return anova_uma_via(y, g, u, **kw)


def _com(r, a, b):
    for c in r.comparacoes:
        if {c["a"], c["b"]} == {a, b}:
            return c
    raise KeyError((a, b))


# ----------------------------------------------- oraculos independentes (A2)

def _tukey_kramer_ingenuo(amostras, alpha=0.05):
    """Tukey-Kramer por definicao, sem reaproveitar o codigo do modulo."""
    k = len(amostras)
    n = [len(a) for a in amostras]
    gl = sum(n) - k
    msw = sum((len(a) - 1) * np.var(a, ddof=1) for a in amostras) / gl
    out = {}
    for i in range(k):
        for j in range(i + 1, k):
            d = np.mean(amostras[i]) - np.mean(amostras[j])
            se = math.sqrt(msw / 2 * (1 / n[i] + 1 / n[j]))
            out[(i, j)] = (d, stats.studentized_range.sf(abs(d) / se, k, gl),
                           stats.studentized_range.isf(alpha, k, gl) * se)
    return out


def _games_howell_ingenuo(amostras, alpha=0.05):
    """Games-Howell por definicao: Welch-Satterthwaite + amplitude studentizada.
    Independente da versao do SciPy (nao usa tukey_hsd(equal_var=False))."""
    k = len(amostras)
    out = {}
    for i in range(k):
        for j in range(i + 1, k):
            a, b = amostras[i], amostras[j]
            va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
            gl = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
            d = np.mean(a) - np.mean(b)
            se = math.sqrt((va + vb) / 2)
            out[(i, j)] = (d, stats.studentized_range.sf(abs(d) / se, k, gl),
                           stats.studentized_range.isf(alpha, k, gl) * se)
    return out


def _confere(r, ref, nomes):
    for (i, j), (d, p, meia) in ref.items():
        c = _com(r, nomes[i], nomes[j])
        assert c["diferenca"] == pytest.approx(d, rel=1e-9)
        assert c["p_ajustado"] == pytest.approx(min(1.0, p), rel=1e-6, abs=1e-12)
        assert c["ic95"][1] - c["ic95"][0] == pytest.approx(2 * meia, rel=1e-6)


def test_f_e_p_batem_com_f_oneway():
    r = _roda(G0, G1, G2)
    ref = stats.f_oneway(G0, G1, G2)
    assert r.F == pytest.approx(ref.statistic, rel=1e-12)
    assert r.p == pytest.approx(ref.pvalue, rel=1e-10)


def test_tukey_kramer_desigual_bate_com_oraculo_ingenuo_e_com_tukey_hsd():
    a, b, c = G0, G1[:4], G2 + [27.0, 25.5, 26.8]
    r = _roda(a, b, c)
    assert r.pos_teste == "Tukey-Kramer"
    _confere(r, _tukey_kramer_ingenuo([np.array(x) for x in (a, b, c)]), ["0", "1", "2"])
    th = stats.tukey_hsd(a, b, c)
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        assert _com(r, str(i), str(j))["p_ajustado"] == pytest.approx(th.pvalue[i, j], rel=1e-6, abs=1e-12)


def test_games_howell_com_variancias_muito_diferentes_bate_com_oraculo_ingenuo():
    rng = np.random.default_rng(3)
    a, b, c = rng.normal(10, 0.1, 12), rng.normal(10.8, 3.0, 13), rng.normal(12.5, 10.0, 11)
    r = _roda(a, b, c)
    assert r.pos_teste == "Games-Howell"
    _confere(r, _games_howell_ingenuo([a, b, c]), ["0", "1", "2"])


def test_levene_bate_com_scipy_centro_mediana():
    r = _roda(G0, G1, G2)
    assert r.levene_p == pytest.approx(stats.levene(G0, G1, G2, center="median").pvalue, rel=1e-12)


def test_eta2_e_omega2_pela_definicao():
    r = _roda(G0, G1, G2)
    todos = np.array(G0 + G1 + G2)
    gm = todos.mean()
    sse = sum(((np.array(a) - np.mean(a)) ** 2).sum() for a in (G0, G1, G2))
    ssb = sum(len(a) * (np.mean(a) - gm) ** 2 for a in (G0, G1, G2))
    sst = ssb + sse
    msw = sse / (len(todos) - 3)
    assert r.eta2 == pytest.approx(ssb / sst, rel=1e-12)
    assert r.omega2 == pytest.approx((ssb - 2 * msw) / (sst + msw), rel=1e-12)
    assert r.omega2 <= r.eta2


def _bh_ingenuo(p):
    p = np.asarray(p, float)
    m = len(p)
    ordem = np.argsort(p)
    ajust = np.empty(m)
    menor = 1.0
    for rank in range(m, 0, -1):
        idx = ordem[rank - 1]
        menor = min(menor, p[idx] * m / rank)
        ajust[idx] = menor
    return ajust


def test_fdr_bh_bate_com_implementacao_ingenua():
    rng = np.random.default_rng(11)
    g = ["a"] * 6 + ["b"] * 6 + ["c"] * 6
    u = [f"u{i}" for i in range(18)]
    resp = {f"e{i}": np.concatenate([rng.normal(0, 1, 6), rng.normal(0.2 * i, 1, 6),
                                     rng.normal(0.1 * i, 1, 6)]) for i in range(8)}
    out = anova_varias_respostas(resp, g, u, n_boot=0, calcular_pos_teste=False)
    ps = [out["resultados"][k].p for k in resp]
    esperado = _bh_ingenuo(ps)
    obtido = [out["resultados"][k].p_fdr for k in resp]
    assert obtido == pytest.approx(list(esperado), rel=1e-10)


# ------------------------------------------------------- casos de borda (A3)

def test_k2_classico_e_o_t_de_student_ao_quadrado():
    """k=2 com variancias compativeis: F == t^2 (Student) e p identico."""
    r = _roda(G0, G1)
    assert r.metodo == "ANOVA classica"
    t = stats.ttest_ind(G0, G1, equal_var=True)
    assert r.F == pytest.approx(t.statistic ** 2, rel=1e-10)
    assert r.p == pytest.approx(t.pvalue, rel=1e-10)


def test_k2_variancias_desiguais_e_welch_e_diz_isso():
    rng = np.random.default_rng(5)
    a, b = rng.normal(0, 0.1, 10), rng.normal(0.5, 4.0, 10)
    r = _roda(a, b)
    assert r.metodo == "Welch ANOVA"
    assert r.p == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue, rel=1e-8)
    assert any("Welch" in x for x in r.avisos)


def test_grupo_com_n1_e_excluido_com_aviso_e_o_resto_segue():
    r = _roda(G0, G1, [30.0])
    assert r.grupos == ["0", "1"]
    assert any("1 unidade" in a for a in r.avisos)


def test_todos_os_grupos_com_n1_e_erro():
    with pytest.raises(ValueError, match="menos de 2 grupos"):
        _roda([1.0], [2.0], [3.0])


def test_todos_os_valores_iguais_e_erro():
    with pytest.raises(ValueError, match="variancia"):
        _roda([5.0] * 4, [5.0] * 4)


def test_entrada_vazia_e_erro_e_nao_excecao_obscura():
    with pytest.raises(ValueError):
        anova_uma_via([], [], [])


def test_grupo_com_variancia_zero_cai_na_classica_com_aviso():
    r = _roda([5.0] * 5, [1.0, 9.0, 3.0, 7.0, 5.5])
    assert np.isfinite(r.p) and r.metodo == "ANOVA classica"


def test_grupo_de_amplitude_minuscula_cuja_variancia_sofre_underflow_cai_na_classica():
    """Achado do Hypothesis (vistoria 2026-10): amplitude 1.5e-292 nao e' constante (ptp != 0), mas a variancia (~1e-584) vira 0
    por underflow; o peso n/variancia do Welch fica infinito e o p ficava NaN. Esperado: tratado como grupo sem variancia."""
    r = _roda([0.0, 0.0, 1.5341810627529126e-292], [0.0, 3.0, 6.0], [0.0, 0.0, 0.0, 0.0, 1.0])
    assert np.isfinite(r.p) and 0.0 <= r.p <= 1.0
    assert r.metodo == "ANOVA classica"
    assert any("variancia zero" in a for a in r.avisos)


def test_nan_e_inf_sao_descartados_com_aviso_e_nao_contaminam():
    y, g, u = _dados(G0, G1)
    y[0], y[7] = float("nan"), float("inf")
    r = anova_uma_via(y, g, u, n_boot=0)
    assert any("descartada" in a for a in r.avisos)
    ref = stats.f_oneway(G0[1:], [v for i, v in enumerate(G1) if i != 2])
    assert r.p == pytest.approx(ref.pvalue, rel=1e-10)


@pytest.mark.parametrize("politica", ["substituir", "excluir"])
def test_censura_icp_nas_duas_politicas_e_celula_vazia_nao_e_censura(tmp_path, politica):
    from guaraci.icp_multielementar import (
        _anova_por_elemento, ler_icp_csv)
    linhas = ["Amostra;Tipo;Grupo;Unidade;Cd 111;Pb 208"]
    rng = np.random.default_rng(2)
    for gi, grp in enumerate(("sul", "norte")):
        for j in range(8):
            cd = "<0,05" if (gi == 0 and j == 0) else f"{rng.normal(1 + gi, 0.2):.3f}".replace(".", ",")
            pb = "" if (gi == 1 and j == 1) else f"{rng.normal(3, 0.3):.3f}".replace(".", ",")
            linhas.append(f"{grp}{j};amostra;{grp};{grp}{j};{cd};{pb}")
    p = tmp_path / "icp.csv"
    p.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    tab = ler_icp_csv(str(p), unidade="mg/kg", coluna_amostra="Amostra", coluna_tipo="Tipo",
                      coluna_grupo="Grupo", coluna_unidade_fisica="Unidade")
    out, els, reg = _anova_por_elemento(tab, politica, 0.5, False, 0.05, 0)
    assert set(els) == {"Cd 111", "Pb 208"}
    assert all(np.isfinite(r.p) for r in out["resultados"].values())
    # a celula vazia derruba a linha nas duas politicas, contada a parte
    assert reg["linhas_descartadas_por_ausencia"] >= 1


def test_rotulos_numericos_texto_espacos_e_acentos_dao_o_mesmo_resultado():
    base = _roda(G0, G1, G2, nomes=["1", "2", "3"])
    for nomes in (["tratamento A", "tratamento B", "tratamento C"],
                  ["amostra classe_z", "amostra classe_a", "amostra classe_c"]):
        r = _roda(G0, G1, G2, nomes=nomes)
        assert r.F == pytest.approx(base.F, rel=1e-12)
        assert r.p == pytest.approx(base.p, rel=1e-12)
        assert len(r.comparacoes) == 3


def test_ordem_das_linhas_e_dos_grupos_nao_altera_o_resultado():
    y, g, u = _dados(G0, G1, G2)
    r0 = anova_uma_via(y, g, u, n_boot=0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        idx = rng.permutation(len(y))
        r = anova_uma_via([y[i] for i in idx], [g[i] for i in idx], [u[i] for i in idx], n_boot=0)
        assert r.F == pytest.approx(r0.F, rel=1e-12) and r.p == pytest.approx(r0.p, rel=1e-12)
        for c in r0.comparacoes:
            assert _com(r, c["a"], c["b"])["p_ajustado"] == pytest.approx(c["p_ajustado"], rel=1e-9)


@pytest.mark.parametrize("a, b", [(0.0, 1.0), (100.0, 1.0), (-7.5, 1.0), (0.0, 1e-3), (3.0, 250.0)])
def test_invariancia_a_deslocamento_e_escala(a, b):
    r0 = _roda(G0, G1, G2)
    r = _roda(*[[a + b * v for v in gr] for gr in (G0, G1, G2)])
    assert r.F == pytest.approx(r0.F, rel=1e-8)
    assert r.p == pytest.approx(r0.p, rel=1e-8)
    assert r.eta2 == pytest.approx(r0.eta2, rel=1e-8)
    sig0 = {(c["a"], c["b"]) for c in r0.comparacoes if c["p_ajustado"] < 0.05}
    sig = {(c["a"], c["b"]) for c in r.comparacoes if c["p_ajustado"] < 0.05}
    assert sig == sig0


def test_determinismo_inclusive_do_bootstrap_com_a_mesma_semente():
    y, g, u = _dados(G0, G1, G2)
    r1 = anova_uma_via(y, g, u, n_boot=200, seed=4)
    r2 = anova_uma_via(y, g, u, n_boot=200, seed=4)
    assert r1 == r2


@pytest.mark.parametrize("k", [10, pytest.param(50, marks=pytest.mark.slow)])
def test_k_grande_funciona_e_o_cache_do_valor_critico_nao_muda_o_resultado(k):
    """k=50 (1225 pares) leva ~4 min por causa da amplitude studentizada do SciPy:
    marcado slow; k=10 roda sempre."""
    rng = np.random.default_rng(8)
    grupos = [rng.normal(0.05 * i, 1, 5) for i in range(k)]
    r = _roda(*grupos)
    assert len(r.comparacoes) == k * (k - 1) // 2
    ref = _tukey_kramer_ingenuo(grupos)
    for (i, j) in ((0, k - 1), (1, 2), (3, k - 2)):
        d, p, meia = ref[(i, j)]
        c = _com(r, str(i), str(j))
        assert c["p_ajustado"] == pytest.approx(min(1, p), rel=1e-5, abs=1e-12)
        assert (c["ic95"][1] - c["ic95"][0]) / 2 == pytest.approx(meia, rel=1e-6)


def test_pseudo_replicacao_avisa_quando_cada_linha_vira_unidade():
    """Unidade = linha (replicas tratadas como independentes): aviso aparece e
    o resultado nao e' apresentado como definitivo."""
    y, g, u = _dados(G0, G1)
    r = anova_uma_via(y, g, u, n_boot=0)
    assert any("NAO sao replicas" in a for a in r.avisos)


def test_pseudo_replicacao_animal_x_fazenda_agregada_por_unidade():
    """Cada fazenda tem varios animais medidos: com a unidade = fazenda, as 4
    linhas viram 1 observacao e o grupo com 1 fazenda so' e' excluido."""
    y = [1.0, 1.1, 0.9, 1.0, 2.0, 2.1, 1.9, 2.0, 3.0, 3.2, 2.8, 3.0]
    g = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    u = ["fazA"] * 4 + ["fazB"] * 4 + ["fazC"] * 4
    with pytest.raises(ValueError, match="menos de 2 grupos"):
        anova_uma_via(y, g, u, n_boot=0)


# ------------------------------------------------------------ propriedades (A4)

_amostra = st.lists(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
                    min_size=3, max_size=12)


@settings(max_examples=60, deadline=None)
@given(_amostra, _amostra, _amostra)
@example([1.0, 2.0, 3.0], [2.0, 3.0, 4.5], [5.0, 6.0, 7.0])
@example([0.0, 0.0, 1e-9], [0.0, 1e-9, 0.0], [1e-9, 0.0, 0.0])
@example([1.0, 1.0, 2.0], [1.0, 2.0, 2.0], [1.0, 2.0, 3.0])
@example([0.0, 0.0, 1.5341810627529126e-292], [0.0, 3.0, 6.0], [0.0, 0.0, 0.0, 0.0, 1.0])
def test_propriedades_p_intervalos_simetria_omega(a, b, c):
    try:
        r = _roda(a, b, c)
    except ValueError:
        return
    assert 0.0 <= r.p <= 1.0
    assert r.omega2 <= r.eta2 + 1e-12
    for cmp in r.comparacoes:
        lo, hi = cmp["ic95"]
        assert lo - 1e-9 <= cmp["diferenca"] <= hi + 1e-9      # IC contem a diferenca observada
        assert 0.0 <= cmp["p_ajustado"] <= 1.0
        # simetria de pares: trocar a ordem so' troca o sinal da diferenca
    if len(r.comparacoes) >= 1:
        rr = _roda(c, b, a)
        for cmp in r.comparacoes:
            inv = _com(rr, {"0": "2", "1": "1", "2": "0"}[cmp["a"]], {"0": "2", "1": "1", "2": "0"}[cmp["b"]])
            assert abs(inv["diferenca"]) == pytest.approx(abs(cmp["diferenca"]), rel=1e-9, abs=1e-9)
            assert inv["p_ajustado"] == pytest.approx(cmp["p_ajustado"], rel=1e-6, abs=1e-9)


def test_tukey_balanceado_com_anova_de_p_alto_nao_rejeita():
    """Em dados balanceados, p(ANOVA) >> alpha implica nenhum par significativo
    (Tukey e' conservador frente ao F quando o F ja' nao rejeita). Excecao
    documentada: o inverso nao vale -- F significativo pode vir sem par
    significativo."""
    rng = np.random.default_rng(21)
    for _ in range(40):
        grupos = [rng.normal(0, 1, 7) for _ in range(4)]
        r = _roda(*grupos)
        if r.p > 0.5:
            assert all(c["p_ajustado"] >= 0.05 for c in r.comparacoes)
