"""Graficos quimiometricos que faltavam (auditoria de figuras, secao D):
cada funcao de `figuras_extras` grava o arquivo, e as grandezas desenhadas
batem com as formulas/funcoes do motor (h*, alavancagem, residuo, RMS da
transferencia, scree <= 100%)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import guaraci.figuras as figuras
import guaraci.figuras_extras as fx
from guaraci.config import NOME_GRAFICOS, Config
from guaraci.eixo_espectral import ROTULO_EIXO_NEUTRO
from guaraci.grupos_figuras import DESCRICAO_GRUPOS, subpasta_do_grupo


def _cfg(**kw):
    cfg = Config()
    cfg.output_format = "png"
    cfg.save_dpi = 40
    cfg.show_plots = False
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _existe(caminho):
    assert os.path.isfile(caminho) and os.path.getsize(caminho) > 1000, caminho


@pytest.fixture
def capturar(monkeypatch):
    capt = {}
    monkeypatch.setattr(figuras, "save", lambda fig, nome, *a, **k: capt.setdefault(nome, fig))
    yield capt
    plt.close("all")


def _dados_reg(n=30, p=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X[:, :3] @ np.array([1.0, 2.0, -1.5]) + 5 + rng.normal(scale=0.1, size=n)
    return X, y


# ------------------------------------------------------------- numericos
def test_limiar_williams_formula():
    assert fx.limiar_williams(3, 40) == pytest.approx(3 * 4 / 40)
    assert fx.limiar_williams(0, 10) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        fx.limiar_williams(2, 0)


def test_alavancagem_e_a_diagonal_da_matriz_chapeu():
    X, y = _dados_reg()
    pls = PLSRegression(3, scale=False).fit(X, y)
    T = pls.x_scores_
    Z = np.column_stack([np.ones(len(T)), T])
    H = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    h = fx.alavancagem_scores(T)
    assert np.allclose(h, np.diag(H))
    assert h.sum() == pytest.approx(3 + 1)          # traco = A + 1


def test_dados_calibracao_recupera_y_e_predicao_do_modelo_salvo():
    X, y = _dados_reg()
    pipe = Pipeline([("preproc", StandardScaler(with_std=False)),
                     ("pls", PLSRegression(3, scale=False))]).fit(X, y)
    d = fx.dados_calibracao_pls(pipe.named_steps["pls"])
    assert np.allclose(d["y"], y)
    assert np.allclose(d["y_hat"], pipe.predict(X).ravel())


def test_residuo_padronizado_usa_s_com_n_menos_a_menos_1():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    yh = y + np.array([0.1, -0.2, 0.1, 0.0, 0.2, -0.1])
    r, s = fx.residuos_padronizados(y, yh, n_lv=2)
    e = yh - y
    assert s == pytest.approx(np.sqrt(np.sum(e ** 2) / (6 - 2 - 1)))
    assert np.allclose(r, e / s)


def test_scree_soma_no_maximo_100(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(25, 40))
    var = PCA(n_components=10).fit(X).explained_variance_ratio_ * 100
    acum = fx.variancia_acumulada(var)
    assert np.all(np.diff(acum) >= 0) and acum[-1] <= 100 + 1e-9
    cam = fx.fig_pca_scree(var, _cfg(), str(tmp_path))
    _existe(cam)


# ------------------------------------------------------------ regressao
def test_predito_medido_faixa_tem_banda_rmsep_e_holdout(tmp_path, capturar):
    rng = np.random.default_rng(2)
    yc = rng.uniform(0, 20, 20); yv = rng.uniform(0, 20, 8); yh = rng.uniform(0, 20, 6)
    fx.fig_predito_medido_faixa(yc, yc + 0.1, yv, yv + 0.3, 0.5, _cfg(), str(tmp_path),
                                y_holdout=yh, yhat_holdout=yh - 0.2, rmsep_holdout=0.6)
    ax = capturar["figS2b_predito_medido_faixa"].axes[0]
    banda = ax.collections[0].get_paths()[0].vertices
    xs, ys = banda[:, 0], banda[:, 1]
    lo = xs.min()
    nas_bordas = np.sort(np.unique(np.round(ys[np.isclose(xs, lo)], 9)))
    assert np.allclose(nas_bordas, [lo - 0.5, lo + 0.5])
    rotulos = [t.get_text() for t in capturar["figS2b_predito_medido_faixa"].axes[1].get_legend().get_texts()]
    assert any("holdout" in t.lower() for t in rotulos)


def test_predito_medido_faixa_grava_arquivo(tmp_path):
    y = np.linspace(0, 10, 12)
    _existe(fx.fig_predito_medido_faixa(y, y, y, y + 0.2, 0.2, _cfg(), str(tmp_path)))


def test_fig7_tambem_gera_faixa(tmp_path):
    yv = np.array([1.0, 2.0, 3.0, 4.0])
    figuras.fig7_pls_regression(yv, yv + 0.1, yv, yv + 0.2, [1.0, 0.8], 2,
                                0.9, 0.9, 0.1, 0.2, 0.2, 0.2, _cfg(), str(tmp_path))
    _existe(os.path.join(str(tmp_path), NOME_GRAFICOS, "figS2b_predito_medido_faixa.png"))


def test_coeficientes_eixo_neutro_e_orientacao(tmp_path, capturar):
    eixo = np.linspace(4000, 10000, 60)
    fx.fig_coeficientes_regressao(eixo, {"A": np.sin(eixo / 500), "B": np.cos(eixo / 500)},
                                  _cfg(mode="csv"), str(tmp_path), n_lv={"A": 2, "B": 3})
    fig = capturar["figS2c_coeficientes_regressao_02"]   # um grafico por especie: A em _01, B em _02
    assert fig.axes[-1].get_xlabel() == ROTULO_EIXO_NEUTRO
    lo, hi = fig.axes[0].get_xlim()
    assert lo > hi                                    # numero de onda: decrescente
    plt.close("all")
    capturar.clear()
    fx.fig_coeficientes_regressao(np.linspace(2, 30, 60), {"A": np.ones(60)},
                                  _cfg(mode="gcms"), str(tmp_path))
    lo, hi = capturar["figS2c_coeficientes_regressao"].axes[0].get_xlim()
    assert lo < hi


def test_williams_linha_em_h_estrela(tmp_path, capturar):
    X, y = _dados_reg(n=24)
    pls = PLSRegression(2, scale=False).fit(X, y)
    d = fx.dados_calibracao_pls(pls)
    r, _s = fx.residuos_padronizados(d["y"], d["y_hat"], 2)
    fx.fig_williams({"m": {"h_cal": fx.alavancagem_scores(d["T"]), "r_cal": r, "n_lv": 2}},
                    _cfg(), str(tmp_path))
    ax = capturar["figS2d_williams"].axes[0]
    verticais = [ln.get_xdata()[0] for ln in ax.lines
                 if len(set(np.atleast_1d(ln.get_xdata()))) == 1 and ln.get_linestyle() == "--"]
    assert np.isclose(verticais, 3 * 3 / 24).any()
    horizontais = {round(float(ln.get_ydata()[0]), 6) for ln in ax.lines
                   if ln.get_linestyle() == ":"}
    assert {3.0, -3.0} <= horizontais


def test_figuras_regressao_do_modelo(tmp_path, monkeypatch):
    X, y = _dados_reg(n=20, p=30)
    pipes = {}
    for esp, sl in (("A", slice(0, 10)), ("B", slice(10, 20))):
        pipes[esp] = {"pipeline": Pipeline([("preproc", StandardScaler(with_std=False)),
                                            ("pls", PLSRegression(2, scale=False))]).fit(X[sl], y[sl])}
    pkg = {"wavenumbers": np.linspace(4000, 5000, 30), "regressao_por_especie": pipes}
    run = tmp_path / "run"
    (run / "Modelos").mkdir(parents=True)
    (run / "Modelos" / "modelo_plsda.joblib").write_bytes(b"x")
    import guaraci.predicao as pred
    monkeypatch.setattr(pred, "load_model", lambda caminho, confiar=False: pkg)
    cams = fx.figuras_regressao_do_modelo(str(run), _cfg(objective="quantificacao", level="N3"))
    assert len(cams) == 2
    for c in cams:
        _existe(c)
    # fora da Quantificacao nao abre o modelo
    assert fx.figuras_regressao_do_modelo(str(run), _cfg(objective="classificacao")) == []


# ------------------------------------------------- tecnicas especiais [3.5]
def test_mcr_als(tmp_path):
    from guaraci.tecnicas_avancadas import rodar_mcr_als
    rng = np.random.default_rng(3)
    eixo = np.linspace(400, 700, 80)
    S = np.vstack([np.exp(-((eixo - c) / 30) ** 2) for c in (480, 600)])
    C = rng.uniform(0.1, 1.0, size=(16, 2))
    X = C @ S + rng.normal(scale=0.002, size=(16, 80)).clip(0)
    res = rodar_mcr_als(X.clip(0), 2, nao_negativo_c=True, nao_negativo_s=True)
    rot = np.array(["a"] * 8 + ["b"] * 8)
    _existe(fx.fig_mcr_als(res.C, res.S, eixo, rot, _cfg(), str(tmp_path),
                           lof_percent=res.lof_percent))


def test_parafac_eem(tmp_path):
    pytest.importorskip("tensorly")
    from guaraci.tecnicas_avancadas import rodar_parafac_eem
    ex = np.arange(250, 360, 10.0); em = np.arange(300, 420, 8.0)
    rng = np.random.default_rng(4)
    a = np.exp(-((ex - 290) / 20) ** 2); b = np.exp(-((em - 340) / 20) ** 2)
    X = np.array([(c * np.outer(b, a)).ravel() for c in rng.uniform(0.5, 2, 9)])
    r = rodar_parafac_eem(X, list(ex), list(em), 1)
    _existe(fx.fig_parafac_eem(r, ex, em, np.array(["x"] * 4 + ["y"] * 5), _cfg(),
                               str(tmp_path)))


def test_asca_dois_fatores_e_sinal_dos_loadings(tmp_path):
    from guaraci.tecnicas_avancadas import rodar_asca
    rng = np.random.default_rng(5)
    eixo = np.linspace(1000, 1800, 50)
    f1 = np.repeat(["p", "q", "r"], 8); f2 = np.tile(["u", "v"], 12)
    X = rng.normal(scale=0.05, size=(24, 50))
    X += (f1 == "q")[:, None] * np.sin(eixo / 80) + (f2 == "v")[:, None] * 0.5 * np.cos(eixo / 60)
    rel = rodar_asca(X, {"f1": f1, "f2": f2}, testar_significancia=False)
    for nome in ("f1", "f2"):
        T, P = fx._svd_efeito(rel.decomposicao["efeitos"][nome])
        assert np.allclose(T, rel.decomposicao["efeitos"][nome]["scores"])
        assert np.allclose(rel.decomposicao["efeitos"][nome]["matriz"] @ P.T, T)
    _existe(fx.fig_asca(rel.decomposicao, {"f1": f1, "f2": f2}, eixo, _cfg(mode="csv"),
                        str(tmp_path)))


# ------------------------------------------------- EEM e cromatograma
def test_eem_media_classe_reshape_emissao_por_excitacao(tmp_path, capturar):
    ex = np.array([250.0, 260, 270, 280]); em = np.array([300.0, 310, 320])
    M = np.zeros((3, 4)); M[2, 1] = 10.0              # pico em em=320, ex=260
    X = np.array([M.ravel(), M.ravel() * 0.5])
    fx.fig_eem_media_classe(ex, em, X, np.array(["a", "a"]), _cfg(), str(tmp_path))
    fig = capturar["fig0_espectros_medios_classe"]
    ax = fig.axes[0]
    assert "Excitation" in ax.get_xlabel() and "Emission" in ax.get_ylabel()


def test_fig0_eem_vira_mapa_de_contorno(tmp_path):
    from guaraci.exemplos import gerar
    r = gerar("fluorescencia_eem", str(tmp_path / "ex"))
    from guaraci.config_io import load_config
    cfg = load_config(os.path.join(r["pasta"], "config.yaml"))
    cfg.output_format, cfg.save_dpi = "png", 40
    grade = fx.grade_eem_da_pasta(cfg.input_folder)
    assert grade is not None
    ex, em = grade
    from guaraci.dados_io import load_data
    wn, X, rot, *_ = load_data(cfg)
    capt = {}
    orig = figuras.save
    try:
        figuras.save = lambda fig, nome, *a, **k: capt.setdefault(nome, fig)
        figuras.fig_class_mean_spectra(wn, X, rot, {c: "r" for c in set(rot)}, cfg, "x")
    finally:
        figuras.save = orig
    ax = capt["fig0_espectros_medios_classe_01"].axes[0]   # um mapa EEM por classe
    assert "Excitation" in ax.get_xlabel()
    assert X.shape[1] == len(ex) * len(em)
    plt.close("all")


@pytest.mark.parametrize("modo", ["gcms", "hplc"])
def test_cromatograma_sobreposto_com_tempo_de_retencao(capturar, modo):
    eixo = np.linspace(2, 30, 50)
    X = np.random.default_rng(0).normal(size=(6, 50))
    figuras.fig_class_mean_spectra(eixo, X, np.array(["a"] * 3 + ["b"] * 3),
                                   {"a": "r", "b": "b"}, _cfg(mode=modo), "x")
    ax = capturar["fig0_espectros_medios_classe"].axes[0]
    assert ax.get_xlabel() == "Retention time (min)"
    assert "chromatogram" in ax.get_title(loc="left").lower()


# --------------------------------------------------------- transferencia
def test_transferencia_rms_igual_ao_do_cli(tmp_path):
    from guaraci import transferir_cli
    rng = np.random.default_rng(6)
    eixo = np.linspace(0, 1, 40)
    Xm = np.array([np.exp(-((eixo - 0.4) / 0.08) ** 2) * a for a in rng.uniform(0.5, 1.5, 8)])
    Xs = np.roll(Xm, 1, axis=1) * 1.05 + 0.02
    cam, antes, depois = fx.fig_transferencia_calibracao(eixo, Xm, Xs, "pds", None,
                                                          str(tmp_path), janela=3, alpha=0.5)
    _existe(cam)
    ref = transferir_cli.erro_loo("pds", Xm, Xs, 3, 0.5)
    assert (antes, depois) == pytest.approx(ref)


def test_subcomando_transferir_gera_figura(tmp_path):
    from guaraci import cli_comandos
    from guaraci.exemplos import gerar
    r = gerar("transferir", str(tmp_path / "t"))
    p = r["pasta"]
    saida = os.path.join(p, "res")
    cod = cli_comandos.subcomando("transferir", [
        "--mestre", os.path.join(p, "mestre.csv"), "--escravo", os.path.join(p, "escravo.csv"),
        "--novos", os.path.join(p, "novos_escravo.csv"), "--saida", saida])
    assert cod == 0
    _existe(os.path.join(saida, NOME_GRAFICOS, "fig_transferencia_mestre_escravo.png"))


# ---------------------------------------------------------------- grupos
def test_novos_prefixos_tem_grupo_descrito():
    casos = {"fig1b_pca_scree": "02_exploracao_PCA_HCA",
             "figS2b_predito_medido_faixa": "10_quantificacao",
             "figS2c_coeficientes_regressao": "10_quantificacao",
             "figS2d_williams": "10_quantificacao",
             "fig_mcr_als_perfis_C_S": "12_resolucao_MCR",
             "fig_parafac_loadings": "13_multivias_PARAFAC",
             "fig_asca_efeitos": "14_ASCA",
             "fig_transferencia_mestre_escravo": "15_transferencia"}
    for nome, pasta in casos.items():
        assert subpasta_do_grupo(nome) == pasta
        assert pasta in DESCRICAO_GRUPOS


def test_save_com_grupo_vai_para_subpasta(tmp_path, monkeypatch):
    monkeypatch.setattr(figuras, "ORGANIZAR_POR_GRUPO", True)
    cam = fx.fig_pca_scree([60.0, 30.0, 5.0], _cfg(), str(tmp_path))
    assert os.sep + "02_exploracao_PCA_HCA" + os.sep in cam
    _existe(cam)
