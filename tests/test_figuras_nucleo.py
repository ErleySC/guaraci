"""Defeitos de FIGURA em modulos do nucleo (auditoria de figuras, secao B,
itens #6, #11, #12, #15, #17, #20). So' a parte grafica muda: estes testes
conferem o que a figura mostra, nunca numeros novos."""
from __future__ import annotations

import types

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.container import ErrorbarContainer

from guaraci.config import Config, NOME_GRAFICOS


@pytest.fixture
def figs_salvas(monkeypatch):
    """Intercepta figuras.save: guarda (nome, fig) sem gravar nada."""
    from guaraci import avaliacao_modelos, figuras, selecao_variaveis
    salvas = []
    for mod in (figuras, avaliacao_modelos, selecao_variaveis):
        monkeypatch.setattr(mod, "save",
                            lambda fig, nome, pasta, cfg, subpasta="": salvas.append((nome, fig)))
    yield salvas
    plt.close("all")


# ── #11 Q-Q com residuos padronizados ─────────────────────────────────────
def test_qq_da_anova_usa_residuos_padronizados_sem_mudar_o_diagnostico():
    from guaraci.anova_cli import _qq_padronizado
    from guaraci.anova_grupos import diagnosticar_premissas
    rng = np.random.default_rng(1)
    grupos = np.repeat(["a", "b", "c"], 8)
    y = rng.normal(10, 0.7, grupos.size) + (grupos == "c") * 3
    d = diagnosticar_premissas(y, grupos, [f"u{i}" for i in range(y.size)])
    antes = d["qq_amostral"].copy()
    teo, amo = _qq_padronizado(d)
    assert np.std(amo, ddof=1) == pytest.approx(1.0)
    assert abs(np.mean(amo)) < 1e-12
    np.testing.assert_array_equal(d["qq_amostral"], antes)      # numero reportado intacto
    np.testing.assert_array_equal(teo, d["qq_teorico"])
    # sem padronizar (DP ~0,7) a inclinacao contra os quantis N(0,1) seria ~0,7
    assert np.polyfit(teo, amo, 1)[0] == pytest.approx(1.0, abs=0.15)


# ── #12 mapa HSI fora da amostra ─────────────────────────────────────────
def test_mapa_hsi_e_predito_por_modelo_que_nao_viu_o_objeto_mostrado(monkeypatch):
    from guaraci import hsi_pipeline as hp
    masc = [np.ones((2, 2), bool), np.ones((2, 3), bool), np.ones((1, 2), bool), np.ones((2, 2), bool)]
    grupos = ["obj0", "obj1", "obj0b", "obj3"]
    rotulos = ["madura", "verde", "madura", "verde"]
    n = [int(m.sum()) for m in masc]
    X = np.vstack([np.full((k, 3), i, float) for i, k in enumerate(n)])
    y = np.concatenate([[r] * k for r, k in zip(rotulos, n)])
    g = np.concatenate([[gr] * k for gr, k in zip(grupos, n)])
    chamadas, mapas = [], []

    def _fake(Xtr, ytr, gtr, Xte, gte, **kw):
        chamadas.append((Xtr, gtr, Xte, gte))
        return {"n_components": 2, "predicoes_pixel": np.array(["verde"] * len(Xte)),
                "predicoes_objeto": {}}
    monkeypatch.setattr(hp, "fit_predict_pixel_plsda", _fake)
    monkeypatch.setattr(hp, "_fig_hsi_classification_map",
                        lambda *a, **k: mapas.append((a, k)))
    hp._mapa_fora_da_amostra(X, y, g, masc, grupos, rotulos, ["madura", "verde"], 2,
                             types.SimpleNamespace(output_folder="x"), 0)
    Xtr, gtr, Xte, gte = chamadas[0]
    assert "obj0" not in set(gtr)                       # objeto mostrado fora do treino
    assert set(gte) == {"obj0"} and len(Xte) == n[0]
    np.testing.assert_array_equal(Xte, X[:n[0]])
    titulo = mapas[0][1]["titulo"]
    assert "obj0" in titulo and "madura" in titulo and "FORA DA AMOSTRA" in titulo


def test_mapa_hsi_grava_no_dpi_configurado(tmp_path):
    from PIL import Image
    from guaraci.hsi_figures import fig_hsi_classification_map
    masc = np.zeros((4, 4), bool)
    masc[1:3, 1:3] = True
    fig_hsi_classification_map(masc, np.array(["a", "b", "a", "a"]), ["a", "b"],
                               Config(output_format="png", save_dpi=200), str(tmp_path), nome="m")
    dpi = Image.open(tmp_path / NOME_GRAFICOS / "hsi" / "m.png").info["dpi"][0]
    assert dpi == pytest.approx(200, abs=1)


# ── #17 carta T2/Q: marca por estatistica, alpha na legenda, lotes no eixo ─
def test_carta_t2_q_marca_cada_painel_so_pela_propria_estatistica(tmp_path, figs_salvas):
    from guaraci.monitorar_cli import _figuras
    res = {"t2": np.array([5.0, 1.0, 1.0, 1.0]), "q": np.array([0.1, 0.9, 0.1, 0.1]),
           "dentro_dominio": np.array([False, False, False, True]),   # status combinado
           "t2_limite": 3.0, "q_limite": 0.5}
    _figuras(str(tmp_path), res, ["L1", "L1", "L2", "L2"], False, None, None, alpha=0.01)
    fig = dict(figs_salvas)["carta_t2_q"]
    ax_t2, ax_q = fig.axes[:2]

    def _x_marcados(ax):
        col = [c for c in ax.collections if c.get_label().endswith("> limite")][0]
        return sorted(col.get_offsets()[:, 0].tolist())
    assert _x_marcados(ax_t2) == [0.0]          # so' T2 acima do limite de T2
    assert _x_marcados(ax_q) == [1.0]           # so' Q acima do limite de Q
    for ax in (ax_t2, ax_q):
        assert any("alpha = 0.01" in t.get_text() for t in ax.get_legend().get_texts())
    assert [t.get_text() for t in ax_q.get_xticklabels()] == ["L1", "L2"]


# ── #15 ICP: barras de erro, pontos, paleta, dpi/formato via save ─────────
def test_figura_icp_tem_barra_de_erro_e_pontos_e_grava_em_graficos(tmp_path):
    from PIL import Image
    from guaraci.icp_multielementar import _figura_medias
    r = types.SimpleNamespace(p_fdr=0.001, grupos=["A", "B"],
                              media_por_grupo={"A": 1.0, "B": 2.0})
    valores = {"Fe": (np.array([0.9, 1.0, 1.1, 1.9, 2.0, 2.1]),
                      np.array(["A", "A", "A", "B", "B", "B"]))}
    pasta_g = tmp_path / NOME_GRAFICOS
    capt = []
    orig = plt.close
    try:
        plt.close = lambda fig=None: capt.append(fig)
        final = _figura_medias({"Fe": r}, "mg/kg", str(pasta_g / "fig_multielementar_medias.png"),
                               0.05, valores=valores, ctx={"fig_dpi": 150, "fig_formato": "png"})
    finally:
        plt.close = orig
    fig = capt[0]
    ax = fig.axes[0]
    assert any(isinstance(c, ErrorbarContainer) for c in ax.containers)
    eb = [c for c in ax.containers if isinstance(c, ErrorbarContainer)][0]
    segs = eb.lines[2][0].get_segments()
    assert [abs(s[1][1] - s[0][1]) / 2 for s in segs] == pytest.approx([0.1, 0.1])  # +-DP
    assert sum(len(c.get_offsets()) for c in ax.collections if hasattr(c, "get_offsets")) >= 6
    cores = {tuple(np.round(p.get_facecolor()[:3], 3)) for p in ax.patches}
    assert len(cores) == 2                        # uma cor por classe
    assert final and final.endswith("fig_multielementar_medias.png")
    assert Image.open(final).info["dpi"][0] == pytest.approx(150, abs=1)
    plt.close("all")


# ── #6 / #20 rotulos ─────────────────────────────────────────────────────
def test_shap_legenda_usa_nome_da_classe():
    from guaraci.avaliacao_modelos import _nome_classe
    assert _nome_classe(1, ["especie_A", "especie_B"]) == "especie_B"
    assert _nome_classe(1, None) == "Cls 1"


def test_ipls_sem_perfil_nao_presume_cm1(figs_salvas, tmp_path):
    from guaraci.selecao_variaveis import fig_etapa4_ipls
    res = [{"intervalo": i, "balanced_accuracy": b, "wn_ini": 100.0 * i, "wn_fim": 100.0 * i + 90,
            "n_vars": 10} for i, b in enumerate([0.6, 0.9, 0.7], start=1)]
    fig_etapa4_ipls(res, None, 0.8, Config(), str(tmp_path))
    fig = dict(figs_salvas)["fig_etapa4_ipls_intervalos"]
    textos = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "melhor: 200-290" in textos and "cm" not in textos


def test_etapa4_diz_que_n_vars_aninhado_e_media_entre_dobras(figs_salvas, tmp_path):
    from guaraci.selecao_variaveis import fig_etapa4_comparacao
    tab = [{"metodo": "Full (todas)", "balanced_accuracy": 0.9, "n_vars": 200},
           {"metodo": "VIP>=1", "balanced_accuracy": 0.8, "n_vars": 62.2,
            "n_vars_min": 60, "n_vars_max": 63}]
    fig_etapa4_comparacao(tab, Config(), str(tmp_path))
    ax = dict(figs_salvas)["fig_etapa4_comparacao_metodos"].axes[0]
    textos = [t.get_text() for t in ax.texts]
    assert textos[0].strip() == "0.900 | 200 vars"
    assert "media 62.2 vars/dobra [60-63]" in textos[1]
    assert "MEDIA" in ax.get_xlabel()


def test_benchmark_regressao_com_2_especies_mostra_pontos_sem_boxplot(figs_salvas, tmp_path):
    from guaraci.avaliacao_modelos import fig_benchmark_regressors
    fig_benchmark_regressors({"PLS-R": np.array([0.4, 0.5]), "RF": np.array([1.0, 4.0])},
                             2, Config(), str(tmp_path))
    ax = dict(figs_salvas)["fig_benchmark_regressors"].axes[0]
    assert not ax.patches                                   # nenhuma caixa
    assert sum(len(c.get_offsets()) for c in ax.collections if hasattr(c, "get_offsets")) >= 4


def test_dependencia_shap_usa_a_classe_de_cada_amostra_e_nao_a_media_que_se_anula():
    import numpy as np
    from guaraci.avaliacao_modelos import _shap_da_propria_classe
    rng = np.random.default_rng(0)
    base = rng.normal(size=(6, 4))
    sv = [base, -base / 2, -base / 2]             # soma zero entre classes, como no RF multiclasse
    y = np.array([0, 1, 2, 0, 1, 2])
    v, rot = _shap_da_propria_classe(sv, y, 1)
    esperado = np.array([sv[c][i, 1] for i, c in enumerate(y)])
    assert np.allclose(v, esperado) and "own class" in rot
    assert np.abs(v).max() > 1e-3
    arr = np.stack(sv, axis=2)                    # formato (n, f, k)
    v3, _ = _shap_da_propria_classe(arr, y, 1)
    assert np.allclose(v3, esperado)
    vf, rotf = _shap_da_propria_classe(sv, np.array(list("abcabc")), 1)   # rotulo nao inteiro: |SHAP| medio
    assert "mean |SHAP|" in rotf and (vf >= 0).all()
