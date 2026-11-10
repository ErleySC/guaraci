# -*- coding: utf-8 -*-
"""Testes basicos (nao-Hypothesis) de `aumento_dados.py` -- forma de
saida, preservacao de rotulo, heranca de grupo (caso pontual, ver
`test_aumento_dados_hypothesis.py` para a propriedade geral) e validacao
de parametros. Espectros SINTETICOS (sem dependencia de dado real)."""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.aumento_dados import AumentoVRM, aumentar_vrm


def _espectros(seed=0, n=6, p=40):
    rng = np.random.default_rng(seed)
    eixo = np.arange(p, dtype=float)
    base = np.exp(-0.5 * ((eixo - 20) / 4) ** 2)
    X = base[None, :] + rng.normal(0, 0.01, size=(n, p))
    y = rng.uniform(0, 10, size=n)
    grupos = np.array([f"mae_{i}" for i in range(n)])
    return X, y, grupos


def test_gerar_produz_forma_esperada():
    X, y, grupos = _espectros()
    av = AumentoVRM(n_aumentos=3, seed=1)
    X_novo, y_novo, grupos_novo = av.gerar(X, y, grupos)
    assert X_novo.shape == (len(X) * 3, X.shape[1])
    assert y_novo.shape == (len(X) * 3,)
    assert grupos_novo.shape == (len(X) * 3,)
    assert np.all(np.isfinite(X_novo))


def test_aumentar_concatena_original_e_novo():
    X, y, grupos = _espectros()
    av = AumentoVRM(n_aumentos=2, seed=2)
    X_full, y_full, grupos_full = av.aumentar(X, y, grupos)
    assert X_full.shape == (len(X) * 3, X.shape[1])  # original + 2 variantes
    np.testing.assert_array_equal(X_full[:len(X)], X)
    np.testing.assert_array_equal(y_full[:len(X)], y)
    np.testing.assert_array_equal(grupos_full[:len(X)], grupos)


def test_rotulo_e_grupo_preservados_por_variante():
    """Cada bloco de `n_aumentos` variantes preserva o `y` e o `grupo` da
    amostra original correspondente -- checagem pontual (a propriedade
    geral, sob split group-aware, esta' em
    test_aumento_dados_hypothesis.py)."""
    X, y, grupos = _espectros(n=5)
    k = 4
    av = AumentoVRM(n_aumentos=k, seed=3)
    _X_novo, y_novo, grupos_novo = av.gerar(X, y, grupos)
    y_novo = y_novo.reshape(len(X), k)
    grupos_novo = grupos_novo.reshape(len(X), k)
    for i in range(len(X)):
        assert np.all(y_novo[i] == y[i])
        assert np.all(grupos_novo[i] == grupos[i])


def test_amplitude_zero_produz_variante_muito_proxima_do_original():
    """Com todas as amplitudes em 0, a variante deve ser (quase) identica
    ao original -- a perturbacao 'vicinal' desliga por completo, servindo
    de sanity-check de que a implementacao nao introduz deslocamento
    faltante nem NaN quando os parametros de amplitude sao zerados."""
    X, y, grupos = _espectros(n=4)
    av = AumentoVRM(n_aumentos=1, amplitude_multiplicativa=0.0,
                     amplitude_baseline=0.0, amplitude_ruido_estruturado=0.0,
                     seed=4)
    X_novo, _y_novo, _grupos_novo = av.gerar(X, y, grupos)
    np.testing.assert_allclose(X_novo, X, atol=1e-10)


def test_perturbacao_produz_espectro_diferente_mas_proximo():
    """Com amplitude > 0, a variante deve DIFERIR do original (a
    perturbacao teve efeito) mas permanecer 'vicinal' -- proxima em norma
    relativa, nao uma distorcao arbitraria."""
    X, y, grupos = _espectros(n=10, seed=5)
    av = AumentoVRM(n_aumentos=1, amplitude_multiplicativa=0.02,
                     amplitude_baseline=0.02, amplitude_ruido_estruturado=0.01,
                     seed=5)
    X_novo, _y_novo, _grupos_novo = av.gerar(X, y, grupos)
    assert not np.allclose(X_novo, X, atol=1e-6)
    diff_relativa = np.linalg.norm(X_novo - X, axis=1) / np.linalg.norm(X, axis=1)
    assert np.all(diff_relativa < 0.5), (
        "perturbacao produziu desvio grande demais para ser considerada "
        "'vicinal' (proxima do original)")


def test_seed_e_deterministico():
    X, y, grupos = _espectros(n=5, seed=6)
    av1 = AumentoVRM(n_aumentos=2, amplitude_multiplicativa=0.03,
                      amplitude_baseline=0.03, seed=42)
    av2 = AumentoVRM(n_aumentos=2, amplitude_multiplicativa=0.03,
                      amplitude_baseline=0.03, seed=42)
    X1, y1, g1 = av1.gerar(X, y, grupos)
    X2, y2, g2 = av2.gerar(X, y, grupos)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(g1, g2)


def test_seeds_diferentes_dao_variantes_diferentes():
    X, y, grupos = _espectros(n=5, seed=7)
    av1 = AumentoVRM(n_aumentos=2, amplitude_multiplicativa=0.03,
                      amplitude_baseline=0.03, seed=1)
    av2 = AumentoVRM(n_aumentos=2, amplitude_multiplicativa=0.03,
                      amplitude_baseline=0.03, seed=2)
    X1, _y1, _g1 = av1.gerar(X, y, grupos)
    X2, _y2, _g2 = av2.gerar(X, y, grupos)
    assert not np.allclose(X1, X2)


def test_ruido_estruturado_ligado_nao_quebra():
    X, y, grupos = _espectros(n=6, p=60, seed=8)
    av = AumentoVRM(n_aumentos=2, amplitude_ruido_estruturado=0.02,
                     janela_suavizacao_ruido=15, seed=8)
    X_novo, _y_novo, _grupos_novo = av.gerar(X, y, grupos)
    assert np.all(np.isfinite(X_novo))


def test_ruido_estruturado_com_poucos_canais_nao_quebra():
    """`p` menor que a janela de suavizacao (fallback documentado no
    docstring: cai pro ruido branco em vez de quebrar)."""
    X, y, grupos = _espectros(n=4, p=3, seed=9)
    av = AumentoVRM(n_aumentos=1, amplitude_ruido_estruturado=0.05,
                     janela_suavizacao_ruido=11, seed=9)
    X_novo, _y_novo, _grupos_novo = av.gerar(X, y, grupos)
    assert np.all(np.isfinite(X_novo))


@pytest.mark.parametrize("kwargs", [
    dict(n_aumentos=0),
    dict(amplitude_multiplicativa=-0.1),
    dict(amplitude_baseline=-0.1),
    dict(amplitude_ruido_estruturado=-0.1),
    dict(ordem_polinomial_baseline=-1),
])
def test_parametros_invalidos_levantam_valueerror(kwargs):
    with pytest.raises(ValueError):
        AumentoVRM(**kwargs)


def test_gerar_rejeita_tamanhos_incompativeis():
    X, y, grupos = _espectros(n=5)
    av = AumentoVRM()
    with pytest.raises(ValueError):
        av.gerar(X, y[:-1], grupos)
    with pytest.raises(ValueError):
        av.gerar(X, y, grupos[:-1])


def test_funcao_atalho_aumentar_vrm_equivale_a_classe():
    X, y, grupos = _espectros(n=5, seed=10)
    X_a, y_a, g_a = aumentar_vrm(X, y, grupos, n_aumentos=2,
                                  amplitude_multiplicativa=0.02, seed=10)
    X_b, y_b, g_b = AumentoVRM(n_aumentos=2, amplitude_multiplicativa=0.02,
                                seed=10).aumentar(X, y, grupos)
    np.testing.assert_array_equal(X_a, X_b)
    np.testing.assert_array_equal(y_a, y_b)
    np.testing.assert_array_equal(g_a, g_b)
