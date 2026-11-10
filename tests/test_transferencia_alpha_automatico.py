"""DS/PDS: o alpha padrao e' escolhido por validacao e funciona em escala de absorbancia.

Achado da auditoria de 2026-11-10: com `alpha=1.0` fixo, em absorbancia (variacao entre
amostras ~1e-2) a regularizacao zerava a transformacao e a saida virava a media do mestre,
enquanto a tela mostrava uma melhora so' pela correcao do deslocamento medio.
"""
import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from guaraci.transferencia_calibracao import (
    apply_standardization, direct_standardization, escolher_alpha, piecewise_direct_standardization)


def _cenario(seed=5):
    rng = np.random.default_rng(seed)
    wl = np.linspace(0, 1, 100)
    S = np.stack([np.exp(-((wl - .3) / .05) ** 2), np.exp(-((wl - .7) / .05) ** 2)])

    def espectros(c, inst):
        A = 0.3 + 0.02 * (c @ S) + 0.0003 * rng.standard_normal((len(c), 100))
        return A if inst == 0 else A * 1.05 + 0.01 * (1 + wl)

    c_cal = rng.uniform(0, 1, (60, 2))
    pls = PLSRegression(2, scale=False).fit(espectros(c_cal, 0), c_cal[:, 0])
    c_tr = rng.uniform(0, 1, (12, 2))
    c_novo = rng.uniform(0, 1, (40, 2))
    return pls, espectros(c_tr, 0), espectros(c_tr, 1), espectros(c_novo, 1), c_novo[:, 0]


@pytest.mark.parametrize("metodo", ["ds", "pds"])
def test_padrao_em_absorbancia_prediz_bem_apos_transferir(metodo):
    pls, Xm, Xs, Xn, y = _cenario()
    t = (direct_standardization(Xm, Xs) if metodo == "ds"
         else piecewise_direct_standardization(Xm, Xs, janela=5))
    rmsep = float(np.sqrt(np.mean((pls.predict(apply_standardization(Xn, t)).ravel() - y) ** 2)))
    assert rmsep < 0.5 * float(y.std())      # antes: ~1,0 x o desvio (o modelo nao predizia nada)


def test_alpha_fixo_continua_sendo_respeitado():
    _, Xm, Xs, _, _ = _cenario()
    assert direct_standardization(Xm, Xs, alpha=2.5).alpha == 2.5
    assert piecewise_direct_standardization(Xm, Xs, janela=3, alpha=0.1).alpha == 0.1


def test_o_alpha_escolhido_e_registrado_e_acompanha_a_escala():
    _, Xm, Xs, _, _ = _cenario()
    a1 = direct_standardization(Xm, Xs).alpha
    a2 = direct_standardization(Xm * 100, Xs * 100).alpha
    assert a1 > 0 and a1 < 1.0               # em absorbancia o alpha 1.0 antigo seria enorme
    assert a2 == pytest.approx(a1 * 1e4, rel=1e-6)   # grade relativa: escala quadratica dos dados


def test_escolher_alpha_com_poucas_amostras_nao_quebra():
    _, Xm, Xs, _, _ = _cenario()
    assert escolher_alpha(Xm[:2], Xs[:2]) > 0
    assert escolher_alpha(np.zeros((5, 10)), np.zeros((5, 10))) == 1.0
