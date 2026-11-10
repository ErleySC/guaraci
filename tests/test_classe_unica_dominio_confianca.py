"""Lote 3.3 da vistoria (achados A4-01, A4-02, A4-04, A4-06, A4-07):
classe unica / poucos grupos, dominio de aplicabilidade e ressalvas de
confianca. Cada teste fixa um comportamento que o codigo anterior nao tinha.
"""
from __future__ import annotations

import inspect
import logging

import numpy as np

import guaraci.app_tabs.predicao as app_predicao
import guaraci.classificadores as clf
import guaraci.guaraci as cli
import guaraci.predicao as pr
from guaraci.chemometric_stats import AD_MIN_GRUPOS_POR_CLASSE, ad_small_n_warning
from guaraci.classificadores import DDSimca, ddsimca_alpha_sustentado
from guaraci.conformal import ConformalOneClass


# ── A4-01: DD-SIMCA com poucos grupos ───────────────────────────────────────

def _classe_hierarquica(n_grupos: int, rng, sd_entre: float = 1.0, p: int = 30):
    """Espectros sinteticos: efeito de grupo (entre amostras fisicas) forte e
    3 replicas por grupo com ruido pequeno -- o regime em que a calibracao
    com poucos grupos rejeita puros novos."""
    X, g = [], []
    for i in range(n_grupos):
        base = rng.normal(0, sd_entre, p)
        for _ in range(3):
            X.append(base + rng.normal(0, 0.1, p))
            g.append(f"g{i}")
    return np.asarray(X), np.asarray(g)


def _entrada_pacote(X, g):
    dd = DDSimca(n_components=2, alpha=0.05).fit(X, np.array(["A"] * len(X)),
                                                 mae_id=g)
    m = dd._modelos["A"]
    return {"pca": m["pca"], "var_t": m["var_t"], "h0": m["h0"], "q0": m["q0"],
            "Nh": m["Nh"], "Nq": m["Nq"], "f_crit": m["f_crit"],
            "n_grupos_calibracao": m["n_grupos_calibracao"],
            "calibrado_por_amostra": m["calibrado_por_amostra"]}


def test_tres_grupos_com_variancia_entre_grupos_alta_nao_e_confiavel():
    """Antes: `confiavel = n_grupos >= 3` e alpha_nominal=0.05 declarado."""
    rng = np.random.default_rng(7)
    X, g = _classe_hierarquica(3, rng)
    entrada = _entrada_pacote(X, g)
    assert entrada["n_grupos_calibracao"] == 3
    r = pr.detect_purity({"ddsimca_por_especie": {"A": entrada}}, "A", X[0])
    assert r.confiavel is False
    assert r.alpha_nominal is None
    assert r.aceito is not None          # a decisao continua a existir


def test_alpha_sustentado_por_n_de_grupos_ou_por_logo_medido():
    # n >= (1-alpha)/alpha = 19 sustenta sozinho
    assert ddsimca_alpha_sustentado(19) is True
    assert ddsimca_alpha_sustentado(18) is False
    # LOGO medido e compativel com 1-alpha, com grupos suficientes
    assert ddsimca_alpha_sustentado(6, 0.95) is True
    assert ddsimca_alpha_sustentado(6, 0.60) is False
    # LOGO com poucos grupos nao e' evidencia, nem se der 1.0
    assert ddsimca_alpha_sustentado(3, 1.0) is False
    assert ddsimca_alpha_sustentado(6, float("nan")) is False
    assert ddsimca_alpha_sustentado(6, None) is False


def test_detect_purity_usa_sens_logo_persistida():
    rng = np.random.default_rng(3)
    X, g = _classe_hierarquica(6, rng, sd_entre=0.2)
    entrada = _entrada_pacote(X, g)
    pkg = {"ddsimca_por_especie": {"A": dict(entrada, sens_logo=0.95)}}
    r = pr.detect_purity(pkg, "A", X[0])
    assert r.confiavel is True and r.alpha_nominal == 0.05
    pkg = {"ddsimca_por_especie": {"A": dict(entrada, sens_logo=0.5)}}
    r = pr.detect_purity(pkg, "A", X[0])
    assert r.confiavel is False and r.alpha_nominal is None


def test_aviso_de_poucos_grupos_nao_chama_a_regiao_de_conservadora(caplog):
    rng = np.random.default_rng(1)
    X, g = _classe_hierarquica(1, rng)
    with caplog.at_level(logging.WARNING, logger="guaraci.classificadores"):
        DDSimca(n_components=2).fit(X, np.array(["A"] * len(X)), mae_id=g)
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "poucos" in msgs or "tao poucos" in msgs
    assert "desconhecida" in msgs
    assert "conservador" not in msgs.lower()
    assert "nao um defeito" not in msgs
    fonte = inspect.getsource(clf)
    assert "larga/conservadora" not in fonte
    assert "conservadora por construcao" not in fonte


# ── A4-02: dominio de aplicabilidade com poucos grupos ─────────────────────

def test_aviso_de_ad_com_5_grupos_por_classe():
    aviso = ad_small_n_warning({"A": 5, "B": 5, "C": 40})
    assert aviso is not None
    assert "desconhecida" in aviso and "5" in aviso
    assert "taxa" in aviso.lower()


def test_sem_aviso_de_ad_com_grupos_suficientes():
    assert ad_small_n_warning({"A": AD_MIN_GRUPOS_POR_CLASSE, "B": 80}) is None
    assert ad_small_n_warning({}) is None


def test_pacote_real_traz_aviso_de_ad_e_taxa_esperada(modelo_e_dados):
    """O pipeline real (dataset sintetico pequeno) grava o aviso de n pequeno
    e a taxa de rejeicao em CV usada como 'taxa esperada de alarmes'."""
    pkg, _X, _wn = modelo_e_dados
    assert pkg.get("ad_aviso_n_pequeno")
    assert pkg["ad_cv_n"] > 0


# ── A4-04: confianca_% nao e' probabilidade ─────────────────────────────────

def test_predict_samples_traz_a_ressalva_no_dataframe_e_no_csv(
        modelo_e_dados, tmp_path):
    pkg, X_novos, wn = modelo_e_dados
    df = pr.predict_samples(pkg, X_novos, wn)
    assert "confianca_%" in df.columns        # nome mantido (v1.1)
    assert (df["confianca_nota"] == pr.NOTA_CONFIANCA_RELATIVA).all()
    assert "probabilidade" in pr.NOTA_CONFIANCA_RELATIVA
    caminho = tmp_path / "saida.csv"
    df.to_csv(caminho, index=False, sep=";", decimal=",")
    assert "nao e' probabilidade calibrada" in caminho.read_text(encoding="utf-8")


def test_telas_do_modo_joblib_dizem_escore_relativo():
    assert "relative score" in inspect.getsource(app_predicao)
    assert "escore relativo" in inspect.getsource(cli)


# ── A4-06: ConformalOneClass padrao ─────────────────────────────────────────

def test_conformal_padrao_cobre_espectros_novos():
    """Antes o padrao era unidade='grupo' (mediana): aceitacao por espectro de
    0,875 (sd entre grupos 0) e 0,92 (sd 1) com 40 grupos, alvo 0,95."""
    rng = np.random.default_rng(20261011)
    assert ConformalOneClass().unidade == "espectro"
    for sd_g in (0.0, 1.0):
        cov = []
        for _ in range(150):
            G = 40
            g = np.repeat(np.arange(G), 3)
            s = np.repeat(rng.normal(0, sd_g, G), 3) + rng.normal(0, 1, 3 * G)
            m = ConformalOneClass().fit(s, g)
            novos = (np.repeat(rng.normal(0, sd_g, 50), 3)
                     + rng.normal(0, 1, 150))
            cov.append(np.mean(m.predict(novos)))
        assert np.mean(cov) >= 0.93, (sd_g, np.mean(cov))


def test_conformal_docstring_diz_que_a_garantia_vale_na_unidade_da_calibracao():
    assert "UNIDADE DA CALIBRACAO" in ConformalOneClass.__init__.__doc__
    assert "MESMA unidade" in ConformalOneClass.predict.__doc__


# ── A4-07: regra de aceitacao descrita corretamente ────────────────────────

def test_textos_nao_descrevem_regra_retangular():
    for modulo in (cli, app_predicao):
        fonte = inspect.getsource(modulo)
        assert "Hotelling T2 <= limit and Q-residual <= limit" not in fonte
        assert "T2 <= limite e resíduo Q <= limite" not in fonte
        assert "both within their statistical limit" not in fonte
    assert "combined" in inspect.getsource(app_predicao).lower()
    assert "T2_ok" in inspect.getsource(cli)


# ── A4-01: o pipeline persiste a sensibilidade LOGO da pureza ──────────────

def test_pipeline_persiste_sens_logo_da_pureza(pq, tmp_path, monkeypatch):
    """Dataset sintetico tem 3 grupos puros por especie; baixamos o minimo de
    grupos do LOGO para 3 so' para exercitar o caminho do export."""
    import os

    import joblib

    from conftest import achar_pastas_run
    monkeypatch.setattr(pq, "PUREZA_MIN_GRUPOS_LOGO", 3)
    cfg = pq.Config(
        input_folder=str(tmp_path / "dados"),
        output_root_folder=str(tmp_path / "saida"),
        mode="sintetico", level="N3",
        n_per_class=12, n_synthetic_points=60, n_synthetic_replicates=3,
        synthetic_adulterants=("S", "M"),
        wn_min=400.0, wn_max=4001.0,
        n_splits_cv=2, n_repeats_cv=1, n_permutations=5,
        n_permutations_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20,
        n_monte_carlo=3, max_lvs=5,
    )
    os.makedirs(cfg.input_folder, exist_ok=True)
    pq.executar(cfg)
    runs = achar_pastas_run(cfg.output_root_folder)
    pkg = joblib.load(os.path.join(runs[0], pq.NOME_MODELOS,
                                   "modelo_plsda.joblib"))
    por_especie = pkg["ddsimca_por_especie"]
    assert por_especie
    medidas = [v["sens_logo"] for v in por_especie.values()
               if "sens_logo" in v]
    assert medidas, "nenhuma especie recebeu sensibilidade LOGO"
    assert all(0.0 <= s <= 1.0 for s in medidas)
    resumo = pr._summarize_purity_coverage(pkg)
    assert resumo is not None and resumo["n_confiavel"] <= len(por_especie)
