"""Regressao PLS de modelo unico (pooled): pontos do HOLDOUT externo no
predito x medido e figuras de coeficientes b / Williams do modelo final.

So' figuras: os numeros devolvidos pela regressao nao mudam com o holdout
nem com as figuras novas."""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from guaraci import figuras, figuras_extras  # noqa: E402
from guaraci import pipeline as pl  # noqa: E402
from guaraci.config import NOME_GRAFICOS, Config  # noqa: E402


def _dados(n=40, p=30, seed=0):
    rng = np.random.default_rng(seed)
    conc = np.linspace(0, 10, n)
    eixo = np.linspace(4000, 1000, p)
    base = np.exp(-((eixo - 2500) / 400) ** 2)
    X = rng.normal(scale=0.05, size=(n, p)) + conc[:, None] * base[None, :] * 0.1
    rot = np.array(["A"] * n)
    mae = np.array([f"A-{i // 2:03d}" for i in range(n)])
    return X, conc, rot, mae, eixo


def _cfg():
    return Config(seed=0, max_lvs=3, n_splits_cv=2, frac_cal=0.7,
                  default_preprocessing="mc", output_format="png",
                  objective="quantificacao")


def _arquivos(pasta):
    saida = set()
    for _r, _d, fs in os.walk(os.path.join(pasta, NOME_GRAFICOS)):
        saida.update(fs)
    return saida


def _rodar(tmp_path, com_holdout):
    X, conc, rot, mae, eixo = _dados()
    kw = {}
    if com_holdout:
        Xh, ch, *_ = _dados(n=10, seed=7)
        kw = {"X_holdout": Xh, "conc_holdout": ch}
    logs = tmp_path / "Logs"
    os.makedirs(logs, exist_ok=True)
    return pl.pls_regressao_pooled(X, conc, rot, mae, _cfg(), str(tmp_path),
                                   str(logs), n_splits=2, wavenumbers=eixo, **kw)


@pytest.fixture
def captura(monkeypatch):
    """Guarda os eixos (b) da figS2 e as chamadas das figuras extras."""
    capt = {"legendas": [], "extras": {}}
    salvar = figuras.save

    def _save(fig, nome, *a, **k):
        if nome == "figS2_pls_regressao":
            ax = fig.axes[1]
            capt["legendas"] = [t.get_text() for t in ax.get_legend().get_texts()]
            capt["marcadores"] = len(ax.collections)
        return salvar(fig, nome, *a, **k)
    monkeypatch.setattr(figuras, "save", _save)
    for nome in ("fig_predito_medido_faixa", "fig_williams"):
        orig = getattr(figuras_extras, nome)

        def _env(*a, _o=orig, _n=nome, **k):
            capt["extras"][_n] = (a, k)
            return _o(*a, **k)
        monkeypatch.setattr(figuras_extras, nome, _env)
    return capt


def test_holdout_aparece_no_predito_medido_e_nas_figuras_extras(tmp_path, captura):
    r = _rodar(tmp_path, com_holdout=True)
    assert r["holdout"] is not None and r["holdout"]["n"] == 10
    assert any("holdout" in t.lower() for t in captura["legendas"])
    assert captura["marcadores"] == 3            # calibracao, validacao, holdout
    _a, k = captura["extras"]["fig_predito_medido_faixa"]
    assert len(np.ravel(k["y_holdout"])) == 10
    assert np.isclose(k["rmsep_holdout"], r["holdout"]["rmsep"])
    (mods, *_), _k = captura["extras"]["fig_williams"]
    m = mods["PLS"]
    assert len(m["h_holdout"]) == 10 and len(m["r_holdout"]) == 10
    arqs = _arquivos(str(tmp_path))
    assert {"figS2_pls_regressao.png", "figS2b_predito_medido_faixa.png",
            "figS2c_coeficientes_regressao.png", "figS2d_williams.png"} <= arqs
    plt.close("all")


def test_sem_holdout_figura_igual_a_antes_e_figuras_do_modelo_saem(tmp_path, captura):
    _rodar(tmp_path, com_holdout=False)
    assert not any("holdout" in t.lower() for t in captura["legendas"])
    assert captura["marcadores"] == 2
    _a, k = captura["extras"]["fig_predito_medido_faixa"]
    assert "y_holdout" not in k
    assert "h_holdout" not in captura["extras"]["fig_williams"][0][0]["PLS"]
    arqs = _arquivos(str(tmp_path))
    assert {"figS2c_coeficientes_regressao.png", "figS2d_williams.png"} <= arqs


def test_numeros_nao_mudam_com_o_holdout(tmp_path):
    a = _rodar(tmp_path / "a", com_holdout=False)
    b = _rodar(tmp_path / "b", com_holdout=True)
    for chave in ("r2c", "r2v", "rmsec", "rmsecv", "rmsep", "bias", "rpd"):
        assert a[chave] == b[chave]


def test_williams_limiar_e_linhas(tmp_path):
    h = np.linspace(0.01, 0.3, 20)
    r = np.linspace(-2, 2, 20)
    cam = figuras_extras.fig_williams(
        {"PLS": {"h_cal": h, "r_cal": r, "n_lv": 2,
                 "h_holdout": h[:4], "r_holdout": r[:4]}},
        _cfg(), str(tmp_path))
    assert os.path.isfile(cam)
    assert figuras_extras.limiar_williams(2, 20) == pytest.approx(3 * 3 / 20)


def test_falha_nas_figuras_do_modelo_nao_derruba(tmp_path, monkeypatch):
    def falha(*a, **k):
        raise RuntimeError("sem figura")
    monkeypatch.setattr(figuras_extras, "fig_williams", falha)
    r = _rodar(tmp_path, com_holdout=True)
    assert np.isfinite(r["rmsep"])
