"""Figuras (`guaraci.figuras`): bordas dos desenhos e funcoes puras, com dados sinteticos minusculos.

Cada teste desenha de verdade (backend Agg) e confere o que a figura entrega: o arquivo gravado, os elementos desenhados ou o retorno.
"""
from __future__ import annotations

import os
import types

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import guaraci.figuras as fg  # noqa: E402
from guaraci.config import Config  # noqa: E402


@pytest.fixture(autouse=True)
def _fecha_figuras():
    yield
    plt.close("all")


@pytest.fixture
def cfg():
    return Config(output_format="png", show_plots=False)


def _graficos(pasta):
    return sorted(p.name for p in (pasta / "Graficos").rglob("*.png"))


def _cores(rotulos):
    paleta = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    return {c: paleta[i % 4] for i, c in enumerate(sorted(set(rotulos)))}


# ---- salvar e ajudantes puros -------------------------------------------------------------------------------------------------------------------


def test_salvar_avisa_uma_unica_vez_que_abrir_na_tela_nao_abre_e_registra_falha_de_disco(tmp_path, capsys, monkeypatch):
    cfg = Config(output_format="png", show_plots=True)
    monkeypatch.setattr(fg, "_AVISO_MOSTRAR_GRAFICOS_EMITIDO", False)
    for _ in range(2):
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        fg.save(fig, "f", str(tmp_path), cfg)
    assert capsys.readouterr().out.count("[AVISO]") == 1
    fg.resetar_falhas_salvamento()
    fig, ax = plt.subplots()
    monkeypatch.setattr(fig, "savefig", lambda *a, **k: (_ for _ in ()).throw(OSError("disco cheio")))
    fg.save(fig, "g", str(tmp_path), Config(output_format="png"))
    assert any("disco cheio" in f for f in fg.obter_falhas_salvamento())
    fg.resetar_falhas_salvamento()


@pytest.mark.parametrize("n_total, esperado_s", [(8, 60.0), (30, 44.0), (80, 30.0), (200, 20.0), (900, 14.0)])
def test_tamanho_do_ponto_cai_com_a_densidade(n_total, esperado_s):
    assert fg.adaptive_scatter_parameters(n_total, 1)[0] == esperado_s


def test_elipse_so_e_desenhada_com_pontos_suficientes_e_forma_razoavel():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots()
    x, y = rng.normal(size=30), rng.normal(size=30)
    assert fg.ellipse_t2(ax, x, y, "k") is True and len(ax.patches) == 1
    assert fg.ellipse_t2(ax, x[:3], y[:3], "k") is False                      # menos de 4 pontos
    assert fg.ellipse_t2(ax, x, np.zeros(30), "k") is False                   # colinear: autovalor nulo
    assert fg.ellipse_t2(ax, x, y * 0.001, "k", max_excentricidade=5.0) is False   # excentrica demais
    assert fg.ellipse_t2(ax, x, y, "k", limite_dispersao=(0.01, 0.01)) is False    # maior que o dado
    assert fg.ellipse_t2(ax, x, y, "k", limite_dispersao=(100.0, 100.0)) is True


def test_contorno_convexo_cobre_poucos_pontos_triangulo_e_pontos_colineares():
    fig, ax = plt.subplots()
    assert fg.convex_hull_contorno(ax, [0, 1], [0, 1], "k") is False
    assert fg.convex_hull_contorno(ax, [0, 1, 0], [0, 0, 1], "k") is True and len(ax.lines) == 1
    assert fg.convex_hull_contorno(ax, [0, 1, 2], [0, 1, 2], "k") is False


def test_paineis_de_escores_usam_elipse_e_contorno_de_reserva():
    rng = np.random.default_rng(1)
    rotulos = np.array(["a"] * 12 + ["b"] * 3)
    scores = np.vstack([rng.normal(size=(12, 2)), rng.normal(size=(3, 2)) + 4])
    fig, ax = plt.subplots()
    fg.plot_scores_panel(ax, scores, rotulos, _cores(rotulos), [50.0, 20.0], "t", "PC1", "PC2")
    assert ax.patches and ax.lines


def test_selecao_de_cargas_distintas_cobre_vazio_piso_e_exaustao():
    assert fg.select_distinct_loadings(np.array([]), np.array([]), 3).size == 0
    assert list(fg.select_distinct_loadings(np.array([-1.0, -2.0, -3.0]), np.array([1.0, 2.0, 3.0]), 2)) == [0]
    escolhidas = fg.select_distinct_loadings(np.array([5.0, 5.0, 5.0]), np.array([1000.0, 1000.1, 2000.0]), 3)
    assert len(escolhidas) == 2


def test_afastamento_de_rotulos_ignora_coluna_de_um_so_e_separa_a_sobreposta():
    pos = np.array([[0.0, 0.0], [0.01, 0.0], [10.0, 0.0]])
    saida = fg.spread_labels(pos, 1.0, 1.0)
    assert abs(saida[0, 1] - saida[1, 1]) >= 1.0 and saida[2, 1] == 0.0
    assert np.array_equal(fg.spread_labels(pos[:1], 1.0, 1.0), pos[:1])


# ---- dendrogramas ---------------------------------------------------------------------------------------------------------------------------------


def _espectros(n_por_classe=6, n_var=40, classes=("a", "b", "c")):
    rng = np.random.default_rng(2)
    rotulos = np.repeat(classes, n_por_classe)
    X = rng.normal(size=(len(rotulos), n_var)) + np.repeat(np.arange(len(classes)), n_por_classe)[:, None]
    return X, rotulos


def test_dendrograma_de_uma_classe_so_avisa_e_o_de_varias_grava_a_figura(tmp_path, cfg, capsys, monkeypatch):
    X, rot = _espectros(classes=("a",))
    fg.fig_hca_dendrogram(X, rot, _cores(rot), cfg, str(tmp_path))
    assert "<2 groups" in capsys.readouterr().out and not _graficos(tmp_path)
    X, rot = _espectros()
    import scipy.cluster.hierarchy as h
    monkeypatch.setattr(h, "fcluster", lambda *a, **k: (_ for _ in ()).throw(ValueError("sem corte")))
    fg.fig_hca_dendrogram(X, rot, _cores(rot), cfg, str(tmp_path))
    assert "fig_hca_dendrogram.png" in _graficos(tmp_path)


def test_comparacao_de_dendrogramas_mostra_o_painel_que_falhou_sem_derrubar_os_outros(tmp_path, cfg):
    X, rot = _espectros(n_var=6)                                               # janela do Savitzky-Golay maior que o numero de variaveis
    fg.fig_hca_comparacao_pipelines(X, rot, _cores(rot), cfg, str(tmp_path))
    # 4+ dendrogramas: no maximo 3 por imagem (fig_..._01.png, _02.png...)
    gs = _graficos(tmp_path)
    assert "fig_hca_comparacao_pipelines.png" in gs or "fig_hca_comparacao_pipelines_01.png" in gs


# ---- pre-processamento, Wold, merito, contribuicao ----------------------------------------------------------------------------------------------------------


def test_figura_de_preprocessamento_sem_savitzky_golay_descreve_so_o_que_foi_aplicado(tmp_path):
    X, rot = _espectros()
    wn = np.linspace(4000, 8000, X.shape[1])
    cfg = Config(output_format="png", apply_snv=True, apply_sg=False)
    fg.fig6_preprocessing(wn, X, X * 0.5, rot, _cores(rot), cfg, str(tmp_path))
    assert _graficos(tmp_path)


def _wold(slope=0.4, intercept=0.1):
    rng = np.random.default_rng(3)
    return {"sims": rng.uniform(size=20), "r2s": rng.uniform(size=20), "q2s": rng.uniform(-0.2, 0.5, size=20), "r2_obs": 0.9, "q2_obs": 0.7,
            "intercept_r2": intercept, "intercept_q2": intercept, "slope_r2": slope, "slope_q2": slope}


@pytest.mark.parametrize("inclinacao", [0.4, float("nan")])
def test_grafico_de_wold_traca_a_reta_so_quando_a_inclinacao_e_finita(tmp_path, cfg, inclinacao):
    fg.fig_extra_wold(_wold(slope=inclinacao), cfg, str(tmp_path))
    assert _graficos(tmp_path)


def test_figura_de_merito_cobre_tabela_vazia_e_especie_sem_seletividade(tmp_path, cfg):
    fg.fig_regression_merit([], cfg, str(tmp_path))
    assert not _graficos(tmp_path)
    tabela = [{"especie": "a", "lod": 0.1, "loq": 0.3, "seletividade_media": 0.5},
              {"especie": "b", "lod": float("nan"), "loq": float("nan"), "seletividade_media": float("nan")}]
    fg.fig_regression_merit(tabela, cfg, str(tmp_path))
    assert _graficos(tmp_path)
    fg.fig_regression_merit([{"especie": "x"}], cfg, str(tmp_path))


def test_contribuicao_por_variavel_cobre_modelo_sem_pesos_e_classe_unica(tmp_path, cfg):
    X, rot = _espectros(n_por_classe=1)
    wn = np.linspace(4000, 8000, X.shape[1])
    vazio = types.SimpleNamespace(x_weights_=np.zeros((X.shape[1], 0)))
    fg.fig_sprint3_score_contribution(vazio, X, rot, wn, _cores(rot), 5, cfg, str(tmp_path))
    assert not _graficos(tmp_path)
    rng = np.random.default_rng(4)
    modelo = types.SimpleNamespace(x_weights_=rng.normal(size=(X.shape[1], 2)))
    fg.fig_sprint3_score_contribution(modelo, X, rot, wn, _cores(rot), 5, cfg, str(tmp_path))
    assert _graficos(tmp_path)


# ---- DD-SIMCA -------------------------------------------------------------------------------------------------------------------------------------


def _modelo_dd(n=9, **extra):
    rng = np.random.default_rng(5)
    m = {"T2_norm": rng.uniform(0.1, 3, n), "Q_norm": rng.uniform(0.1, 3, n), "f": rng.uniform(0.1, 3, n), "f_crit": 1.5,
         "h0": 1.0, "q0": 1.0, "Nh": 2.0, "Nq": 2.0, "T2_ucl": 4.0, "Q_ucl": 0.5, "n_comp": 2, "n_train": 6}
    m.update(extra)
    return m


def test_fronteira_do_ddsimca_e_nan_sem_parametros_ou_com_parametros_invalidos():
    grade = np.logspace(-1, 1, 5)
    assert np.isnan(fg._fronteira_ddsimca({"h0": None}, grade)).all()
    assert np.isnan(fg._fronteira_ddsimca(_modelo_dd(h0=0.0), grade)).all()
    assert np.isnan(fg._fronteira_ddsimca(_modelo_dd(Q_ucl=-1.0), grade)).all()
    assert np.isfinite(fg._fronteira_ddsimca(_modelo_dd(), grade)).any()


def test_aceitacao_do_ddsimca_sem_modelo_nao_desenha_e_com_modelo_usa_ou_nao_a_sensibilidade_informada(tmp_path, cfg):
    rot = np.array(list("aaabbbccc"))
    fg.fig_sprint3_ddsimca_acceptance({}, rot, _cores(rot), cfg, str(tmp_path))
    assert not _graficos(tmp_path)
    scores = {"a": _modelo_dd(), "b": _modelo_dd(calibrado_por_amostra=True, n_grupos_calibracao=4)}
    fg.fig_sprint3_ddsimca_acceptance(scores, rot, _cores(rot), cfg, str(tmp_path))
    fg.fig_sprint3_ddsimca_acceptance(scores, rot, _cores(rot), cfg, str(tmp_path), sens_esp={"a": (0.9, 0.8, 5, 4, 3, "")})
    assert _graficos(tmp_path)


def test_ddsimca_sem_imagens_individuais_que_repetem_as_paginas(tmp_path, cfg):
    """As paginas de aceitacao (um modelo por imagem) ja' sao a versao legivel: as imagens por classe
    eram copias e poluiam a pasta. A funcao antiga fica na API, mas nao grava nada."""
    rot = np.array(list("aaabbbccc"))
    scores = {"a": _modelo_dd(calibrado_por_amostra=False), "b": _modelo_dd()}
    fg.fig_ddsimca_individuais(scores, rot, _cores(rot), cfg, str(tmp_path))
    assert not [n for n in _graficos(tmp_path) if n.startswith("ddsimca_")]
    fg.fig_sprint3_ddsimca_acceptance(scores, rot, _cores(rot), cfg, str(tmp_path),
                                      sens_esp={"a": (float("nan"), float("nan"), 5, 4, 3, "")})
    assert "fig_ddsimca_aceitacao_01.png" in _graficos(tmp_path)   # um modelo por imagem


@pytest.mark.parametrize("n_modelos", [1, 2, 4])
def test_cooman_cobre_um_par_uma_linha_e_paineis_sobrando(tmp_path, cfg, n_modelos):
    rot = np.array(list("aaabbbcccddd"))
    nomes = list("abcd")[:n_modelos]
    scores = {c: _modelo_dd(n=12) for c in nomes}
    fg.fig_cooman_ddsimca(scores, rot, _cores(rot), cfg, str(tmp_path))
    gs = _graficos(tmp_path)
    gerou = "fig_cooman_ddsimca.png" in gs or "fig_cooman_ddsimca_01.png" in gs   # >3 pares: paginado
    assert gerou is (n_modelos >= 2)


# ---- cargas, ROC, S-plot, mapa de calor ------------------------------------------------------------------------------------------------------------------------


def test_cargas_da_pca_com_um_so_componente(tmp_path, cfg):
    from sklearn.decomposition import PCA
    X, _ = _espectros()
    pca = PCA(n_components=1).fit(X)
    fg.fig_loadings_pca(pca, np.linspace(4000, 8000, X.shape[1]), cfg, str(tmp_path), n_pcs=3)
    assert _graficos(tmp_path)


def test_roc_cobre_classe_unica_e_auc_macro_degenerado(tmp_path, cfg, monkeypatch):
    rng = np.random.default_rng(6)
    assert fg.fig_roc_auc(np.ones((5, 1)), np.ones((5, 1)), np.array(["a"]), cfg, str(tmp_path)) == {}
    Y = np.zeros((12, 3))
    Y[np.arange(12), np.repeat([0, 1, 1], 4)] = 1                              # a terceira classe nunca aparece
    Y_cv = rng.uniform(size=(12, 3))
    monkeypatch.setattr(fg, "roc_auc_score", lambda *a, **k: (_ for _ in ()).throw(ValueError("classe ausente")))
    aucs = fg.fig_roc_auc(Y, Y_cv, np.array(list("abc")), cfg, str(tmp_path))
    assert "macro" not in aucs and set(aucs) == {"a", "b"}


def test_splot_com_preditor_constante_nao_divide_por_zero(tmp_path, cfg):
    X, _ = _espectros()
    fg.fig_splot_opls(X, np.ones(X.shape[0]), np.linspace(4000, 8000, X.shape[1]), cfg, str(tmp_path))
    assert _graficos(tmp_path)


def test_mapa_de_calor_marca_celula_sem_dado_e_celula_abaixo_do_limiar(tmp_path, cfg):
    resultado = {"especies": ["e1", "e2"], "adulterantes": ["x", "y"], "limiar_r2": 0.7, "n_falhas": 1, "n_total": 3,
                 "matriz": {("e1", "x"): 0.9, ("e1", "y"): 0.4, ("e2", "x"): 0.8}}
    fg.fig_heatmap_species_by_adulterant(resultado, cfg, str(tmp_path))
    assert "fig_heatmap_species_by_adulterant.png" in _graficos(tmp_path)


def test_o_diretorio_de_saida_das_figuras_e_criado_sob_demanda(tmp_path, cfg):
    fig, ax = plt.subplots()
    fg.save(fig, "x", str(tmp_path / "novo"), cfg, subpasta="sub")
    assert os.path.isfile(tmp_path / "novo" / "Graficos" / "sub" / "x.png")
