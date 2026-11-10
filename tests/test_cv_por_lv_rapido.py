# -*- coding: utf-8 -*-
"""O atalho de `_predicoes_cv_por_lv` (um PLS com K VLs por fold, predicoes
de 1..K derivadas dele) tem de dar as MESMAS predicoes de CV que ajustar um
pipeline por (n, fold), que era o caminho antigo."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from guaraci.pipeline import _predicoes_cv_por_lv
from guaraci.preprocessamento import SNV, SavGol


def _dados():
    rng = np.random.default_rng(7)
    n, p = 60, 120
    y = np.repeat([0, 1, 2], 20)
    eixo = np.linspace(0, 1, p)
    picos = np.stack([np.exp(-((eixo - c) / 0.05) ** 2) for c in (.2, .5, .8)])
    X = (rng.uniform(.5, 1.5, (n, 1)) * (np.eye(3)[y] * .3 + rng.uniform(0, 1, (n, 3))) @ picos
         + rng.normal(0, .01, (n, p)) + rng.uniform(0, .2, (n, 1)))
    return X, np.eye(3)[y].astype(float), y


@pytest.mark.parametrize("pre", ["snv_sg", "autoscale"])
def test_atalho_bate_com_um_ajuste_por_lv(pre):
    X, Y, y = _dados()
    passos = ([("snv", SNV()), ("sg", SavGol(11, 2, 1))] if pre == "snv_sg"
              else [("sc", StandardScaler())])

    def fabrica(n):
        return Pipeline([("preproc", Pipeline(passos)),
                         ("pls", PLSRegression(n_components=n, scale=False))])

    cv = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    rapido = _predicoes_cv_por_lv(fabrica, X, Y, cv, 12)

    def fabrica_lenta(n):          # forca o caminho antigo
        return Pipeline([("outro", Pipeline(passos)),
                         ("pls", PLSRegression(n_components=n, scale=False))])
    lento = _predicoes_cv_por_lv(fabrica_lenta, X, Y, cv, 12)
    for n in range(1, 13):
        np.testing.assert_allclose(rapido[n], lento[n], rtol=1e-9, atol=1e-11)


def test_pipeline_fora_do_padrao_usa_caminho_antigo():
    X, Y, y = _dados()

    def fabrica(n):
        return Pipeline([("preproc", FunctionTransformer()),
                         ("pls", PLSRegression(n_components=n, scale=True))])
    cv = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))
    out = _predicoes_cv_por_lv(fabrica, X, Y, cv, 3)
    pipe = fabrica(2).fit(X[cv[0][0]], Y[cv[0][0]])
    np.testing.assert_allclose(out[2][cv[0][1]], pipe.predict(X[cv[0][1]]))
