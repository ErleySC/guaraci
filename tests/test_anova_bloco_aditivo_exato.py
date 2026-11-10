"""ANOVA com bloco em dados PERFEITAMENTE aditivos: residuo exatamente zero e' alcancavel (nao e' codigo morto)."""
import warnings

import numpy as np
import pytest

import guaraci.anova_grupos as ag
from guaraci.anova_grupos import anova_uma_via


def _dados():
    efeito_g, efeito_b = [5, 1], [0, 3]
    y, g, b, u = [], [], [], []
    for i in range(2):
        for j in range(2):
            for _ in range(3):
                y.append(float(efeito_g[i] + efeito_b[j] + 10))
                g.append(f"g{i}")
                b.append(f"b{j}")
                u.append(f"u{len(y)}")
    return y, g, b, u


@pytest.fixture
def lstsq_exato(monkeypatch):
    """O LAPACK de cada plataforma devolve 0 ou ~1e-30 neste desenho; arredondar beta torna o zero exato em todas."""
    original = np.linalg.lstsq

    def arredondado(a, b, rcond=None):
        beta, *resto = original(a, b, rcond=rcond)
        return (np.round(beta, 9), *resto)

    monkeypatch.setattr(ag.np.linalg, "lstsq", arredondado)


def test_residuo_exatamente_zero_da_F_infinito_p_zero_e_comparacao_sem_erro_padrao(lstsq_exato):
    y, g, b, u = _dados()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # scipy.levene avisa quando os desvios da mediana sao constantes
        r = anova_uma_via(y, g, u, bloco=b, n_boot=0)
    assert r.tabela["dentro"]["SQ"] == 0.0 and r.tabela["dentro"]["QM"] == 0.0
    assert r.F == float("inf") and r.p == 0.0 and r.tabela["entre"]["p"] == 0.0
    assert np.isnan(r.tabela["bloco"]["F"]) and np.isnan(r.tabela["bloco"]["p"])      # F do bloco indefinido com QM = 0
    (c,) = r.comparacoes
    assert c["diferenca"] == pytest.approx(4.0) and c["p_ajustado"] == 0.0
    assert c["ic95"][0] == pytest.approx(4.0) and c["ic95"][1] == pytest.approx(4.0)       # erro-padrao zero: IC degenerado


def test_residuo_positivo_tem_F_finito_e_p_entre_0_e_1():
    y, g, b, u = _dados()
    y = list(np.array(y) + np.array([0.3, -0.2, 0.1, -0.4, 0.5, 0.0, 0.2, -0.1, 0.0, 0.3, -0.3, 0.1]))
    r = anova_uma_via(y, g, u, bloco=b, n_boot=0)
    assert np.isfinite(r.F) and 0 < r.p < 1 and r.comparacoes[0]["p_ajustado"] > 0
    assert np.isfinite(r.tabela["bloco"]["F"])
