"""S-1 do parecer de 2026-09-25: o benchmark ganha o t corrigido de Nadeau &
Bengio (2003) -- Wilcoxon com 5 folds nunca passa de p=0,0625."""
import numpy as np
from scipy.stats import t as _t

from guaraci.avaliacao_modelos import t_corrigido_nadeau_bengio


def test_formula_bate_com_o_calculo_manual():
    d = np.array([0.05, 0.08, 0.06, 0.07, 0.09])
    r = 0.25
    esperado = 2 * _t.sf(abs(d.mean() / np.sqrt((1 / 5 + r) * d.var(ddof=1))), df=4)
    assert np.isclose(t_corrigido_nadeau_bengio(d, r), esperado)


def test_diferenca_consistente_pode_ser_significativa_com_5_folds():
    d = np.array([0.10, 0.11, 0.09, 0.10, 0.12])
    assert t_corrigido_nadeau_bengio(d, 0.25) < 0.05


def test_sem_variacao_ou_poucos_folds_nao_inventa_p():
    assert np.isnan(t_corrigido_nadeau_bengio(np.array([0.1, 0.1, 0.1]), 0.25))
    assert np.isnan(t_corrigido_nadeau_bengio(np.array([0.1]), 0.25))
