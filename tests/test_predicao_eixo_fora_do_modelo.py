"""M-06 do parecer de 2026-09-25: a predicao nao extrapola em silencio.

`np.interp` repetia o valor da borda: um espectro que cobre so' metade do eixo
do modelo saia com 99,3% de confianca, "aceito" e "dentro do dominio"
(`revisor2/pred_test.py`). Agora a predicao e' RECUSADA, com mensagem que diz o
que faltou -- nos DOIS caminhos (.joblib e modelo portatil JSON).
"""
import numpy as np
import pytest

from guaraci.eixo_espectral import (COBERTURA_EIXO_MINIMA, EixoForaDoModeloError,
                                    cobertura_do_eixo, exigir_cobertura_do_eixo)
from guaraci.model_export import export_portable_json, predict_portable
from guaraci.predicao import _interpolate_to_reference
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer

from guaraci.config import Config
from guaraci.preprocessamento import build_preprocessor


def _pacote_minimo(preset, seed=0, n=60, p=40, n_classes=3):
    rng = np.random.default_rng(seed)
    y_int = rng.integers(0, n_classes, size=n)
    X_raw = rng.normal(loc=0.5, scale=0.2, size=(n, p)) + y_int[:, None] * 0.3
    wavenumbers = np.linspace(4000.0, 400.0, p)
    preproc = build_preprocessor(Config(default_preprocessing=preset)).fit(X_raw)
    lb = LabelBinarizer().fit(y_int)
    pls = PLSRegression(n_components=2, scale=False).fit(
        preproc.transform(X_raw), np.asarray(lb.transform(y_int), dtype=float))
    return {"preprocessador": preproc, "pls_final": pls, "label_binarizer": lb,
            "classes": list(lb.classes_), "wavenumbers": wavenumbers,
            "preset": preset, "wn_min": 400.0, "wn_max": 4000.0}


def test_cobertura_completa_parcial_e_nula():
    ref = np.linspace(4000, 10000, 101)
    assert cobertura_do_eixo(ref, np.linspace(3900, 10100, 50)) == 1.0
    assert cobertura_do_eixo(ref, np.linspace(4000, 7000, 50)) == pytest.approx(0.505, abs=0.02)
    assert cobertura_do_eixo(ref, np.linspace(500, 900, 20)) == 0.0
    assert cobertura_do_eixo(ref, np.array([])) == 0.0


def test_exigir_cobertura_recusa_e_diz_o_que_faltou():
    ref = np.linspace(4000, 10000, 101)
    with pytest.raises(EixoForaDoModeloError, match="cobre so'"):
        exigir_cobertura_do_eixo(ref, np.linspace(4000, 7000, 50))
    assert exigir_cobertura_do_eixo(ref, ref) == 1.0
    # o limiar e' configuravel
    assert exigir_cobertura_do_eixo(ref, np.linspace(4000, 7000, 50), minima=0.4) > 0.4


def test_interpolacao_para_o_eixo_do_modelo_recusa_eixo_truncado():
    pkg = {"wavenumbers": np.linspace(4000, 400, 60)}
    X = np.random.default_rng(0).random((2, 30))
    with pytest.raises(EixoForaDoModeloError):
        _interpolate_to_reference(pkg, X, np.linspace(4000, 2200, 30))
    # eixo completo (ainda que decrescente, como em muitos .dx) segue valendo
    Xc = np.random.default_rng(1).random((2, 60))
    saida = _interpolate_to_reference(pkg, Xc, np.linspace(4000, 400, 60))
    assert saida.shape == (2, 60)


def test_pacote_pode_declarar_cobertura_minima_propria():
    pkg = {"wavenumbers": np.linspace(4000, 400, 60), "cobertura_eixo_minima": 0.4}
    X = np.random.default_rng(0).random((2, 30))
    assert _interpolate_to_reference(pkg, X, np.linspace(4000, 2200, 30)).shape[0] == 2


def test_modelo_portatil_tambem_recusa_extrapolacao():
    pkg = _pacote_minimo("snv_sg_mc")
    portatil = export_portable_json(pkg)
    wn = np.asarray(pkg["wavenumbers"], dtype=float)
    X = np.random.default_rng(0).random((3, len(wn))) + 0.5
    ok = predict_portable(portatil, X, wn)                       # eixo completo
    assert len(ok["classe_pred"]) == 3
    metade = len(wn) // 2
    with pytest.raises(EixoForaDoModeloError):
        predict_portable(portatil, X[:, :metade], wn[:metade])   # metade do eixo


def test_limiar_padrao_e_95_por_cento():
    assert COBERTURA_EIXO_MINIMA == 0.95
