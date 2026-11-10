"""m-04 do parecer de 2026-09-25: preset `custom` sem etapa com estado
(so' SNV, so' SavGol, so' AirPLS) dava NotFittedError com o sklearn 1.9 fixado."""
import numpy as np
import pytest

import guaraci.pipeline as pq
from guaraci.preprocessamento import build_preprocessor


@pytest.mark.parametrize("flags", [
    dict(apply_snv=True, apply_sg=False, apply_mc=False),
    dict(apply_snv=False, apply_sg=True, apply_mc=False),
    dict(apply_airpls=True, apply_snv=False, apply_sg=False, apply_mc=False),
])
def test_custom_so_com_etapas_sem_estado_ajusta_e_transforma(flags):
    cfg = pq.Config(default_preprocessing="custom", **flags)
    X = np.random.default_rng(0).random((30, 60)) + 1.0
    Xt = build_preprocessor(cfg).fit(X).transform(X)
    assert Xt.shape == X.shape and np.isfinite(Xt).all()
