"""Organizacao de Graficos/ por grupo de analise e correcoes da auditoria de
figuras (sinal do residuo, orientacao do eixo X, fig2 com 1 VL)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import guaraci.figuras as figuras
from guaraci.config import NOME_GRAFICOS, Config
from guaraci.grupos_figuras import (
    DESCRICAO_GRUPOS, GRUPOS_FIGURAS, chave_ordem_figura, subpasta_do_grupo,
)


def _cfg(**kw):
    cfg = Config()
    cfg.output_format = "png"
    cfg.save_dpi = 40
    cfg.show_plots = False
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _salvar(nome, pasta, subpasta=""):
    fig, _ax = plt.subplots(figsize=(1, 1))
    figuras.save(fig, nome, str(pasta), _cfg(), subpasta=subpasta)


@pytest.fixture
def chave_ligada(monkeypatch):
    monkeypatch.setattr(figuras, "ORGANIZAR_POR_GRUPO", True)


# ---------------------------------------------------------------- chave
def test_chave_desligada_mantem_caminho_plano(tmp_path):
    assert figuras.ORGANIZAR_POR_GRUPO is False      # padrao
    _salvar("fig1_pca_scores", tmp_path)
    _salvar("ddsimca_especie_A", tmp_path, subpasta="ddsimca")
    base = tmp_path / NOME_GRAFICOS
    assert (base / "fig1_pca_scores.png").exists()
    assert (base / "ddsimca" / "ddsimca_especie_A.png").exists()
    assert not (base / "README.txt").exists()


def test_chave_ligada_mapeia_prefixos_conhecidos(tmp_path, chave_ligada):
    base = tmp_path / NOME_GRAFICOS
    casos = {
        "fig0_espectros_medios_classe": "01_dados_preprocessamento",
        "fig1_pca_scores": "02_exploracao_PCA_HCA",
        "fig4_confusao_e_metricas_por_classe": "03_PLS-DA",
        "fig3_outliers_T2_Q": "04_diagnostico_outliers",
        "fig_etapa4_ipls_intervalos": "05_VIP_selecao_variaveis",
        "fig_det_curvas_log": "06_validacao",
        "fig_opls_scores": "08_OPLS-DA",
        "fig_shap_dep_random_forest_feat1": "09_benchmark_SHAP",
        "figS2_pls_regressao": "10_quantificacao",
    }
    for nome, pasta in casos.items():
        _salvar(nome, tmp_path)
        assert (base / pasta / f"{nome}.png").exists(), nome
    leia = (base / "README.txt").read_text(encoding="utf-8")
    for pasta in DESCRICAO_GRUPOS:
        assert pasta in leia


def test_chave_ligada_deixa_nome_desconhecido_solto(tmp_path, chave_ligada):
    _salvar("carta_t2_q", tmp_path)                  # monitoramento
    _salvar("hsi_mapa", tmp_path, subpasta="hsi")    # subpasta propria
    base = tmp_path / NOME_GRAFICOS
    assert (base / "carta_t2_q.png").exists()
    assert (base / "hsi" / "hsi_mapa.png").exists()
    assert subpasta_do_grupo("nome_qualquer") == ""


def test_chave_ligada_ddsimca_vai_para_grupo(tmp_path, chave_ligada):
    _salvar("ddsimca_especie_A", tmp_path, subpasta="ddsimca")
    _salvar("fig_cooman_ddsimca", tmp_path, subpasta="ddsimca")
    pasta = tmp_path / NOME_GRAFICOS / "07_DD-SIMCA"
    assert (pasta / "ddsimca_especie_A.png").exists()
    assert (pasta / "fig_cooman_ddsimca.png").exists()


def test_tabela_so_aponta_para_pastas_descritas():
    assert {p for _pre, p in GRUPOS_FIGURAS} <= set(DESCRICAO_GRUPOS)


def test_list_figures_mesma_ordem_com_e_sem_grupos(tmp_path, monkeypatch):
    """Os relatorios pegam as N primeiras: a selecao nao pode depender da
    ordem das pastas de grupo."""
    from guaraci.app_logic import list_figures
    nomes = ["fig_splot_opls", "fig0_espectros_medios_classe", "figS1_selecao_lvs",
             "fig4_confusao_e_metricas_por_classe", "fig_extra_holdout",
             "fig1_pca_scores", "fig_ddsimca_aceitacao", "carta_t2_q"]
    planos, agrupados = tmp_path / "plano", tmp_path / "grupo"
    for n in nomes:
        _salvar(n, planos)
    _salvar("ddsimca_x", planos, subpasta="ddsimca")
    monkeypatch.setattr(figuras, "ORGANIZAR_POR_GRUPO", True)
    for n in nomes:
        _salvar(n, agrupados)
    _salvar("ddsimca_x", agrupados, subpasta="ddsimca")
    a = [os.path.relpath(p, planos) for p in list_figures(str(planos))]
    b = [chave_ordem_figura(os.path.relpath(p, agrupados))
         for p in list_figures(str(agrupados))]
    assert a == b


# ---------------------------------------------------------- #1 residuo
def test_residuo_tem_o_sinal_do_bias(monkeypatch):
    capt = {}
    monkeypatch.setattr(figuras, "save", lambda fig, nome, *a, **k: capt.setdefault(nome, fig))
    yv = np.array([1.0, 2.0, 3.0, 4.0])
    yv_hat = yv + 0.5                                 # superestima
    bias = float(np.mean(yv_hat - yv))
    figuras.fig7_pls_regression(yv, yv + 0.1, yv, yv_hat, [1.0, 0.8], 2,
                                0.9, 0.9, 0.1, 0.2, 0.5, bias, _cfg(), "x")
    ax = capt["figS2_pls_regressao"].axes[2]
    res = ax.collections[0].get_offsets()[:, 1]
    assert np.isclose(np.mean(res), bias)
    assert "pred" in ax.get_ylabel().lower()
    plt.close("all")


# ---------------------------------------------------------- #4 eixo X
@pytest.mark.parametrize("modo,decrescente", [("gcms", False), ("eem", False),
                                              ("hplc", False), ("csv", True),
                                              ("dx", True)])
def test_eixo_x_so_inverte_numero_de_onda(monkeypatch, modo, decrescente):
    capt = {}
    monkeypatch.setattr(figuras, "save", lambda fig, nome, *a, **k: capt.setdefault(nome, fig))
    eixo = np.linspace(2.0, 30.0, 50)                 # crescente (ex. tempo de retencao)
    X = np.random.default_rng(0).normal(size=(6, 50))
    rot = np.array(["a"] * 3 + ["b"] * 3)
    figuras.fig_class_mean_spectra(eixo, X, rot, {"a": "r", "b": "b"},
                                   _cfg(mode=modo), "x")
    lo, hi = capt["fig0_espectros_medios_classe"].axes[0].get_xlim()
    assert bool(lo > hi) is decrescente
    plt.close("all")


# ---------------------------------------------------------- #13 fig2 1 VL
def test_fig2_com_uma_vl_e_gerada(tmp_path):
    rng = np.random.default_rng(1)
    T = rng.normal(size=(12, 1))
    rot = np.array(["a"] * 6 + ["b"] * 6)
    figuras.fig2_plsda_scores(T, np.array([42.0]), rot, {"a": "r", "b": "b"},
                              _cfg(), str(tmp_path))
    assert (tmp_path / NOME_GRAFICOS / "fig2_plsda_scores.png").exists()
