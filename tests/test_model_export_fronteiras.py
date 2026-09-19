# -*- coding: utf-8 -*-
"""Fronteiras de `guaraci.model_export` com pacotes construidos A MAO.

Achado da rodada de mutacao (Passo 219): 38 mutantes sobreviviam em
`model_export.py` porque os testes existentes so' comparavam o export
com o pacote real nos presets normais -- nunca exercitavam: eixo
recortado (`wn_min/wn_max` dentro do eixo), `x_mean` != 0, linha SNV
constante, soma de probabilidades ~0, tipos de passo desconhecidos,
`versao_formato`, JSON nao-ASCII. Os valores esperados abaixo sao
calculados a mao (independentes do codigo sob teste).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from guaraci.model_export import (
    UnsupportedModelError,
    _aplicar_passo,
    _serializar_passo,
    export_portable_json,
    predict_portable,
    save_portable_json,
)


def _portatil(wn, passos, x_mean, coef, intercept, classes, wn_min=None, wn_max=None):
    wn = list(map(float, wn))
    return {
        "formato": "guaraci-modelo-portatil", "versao_formato": 1,
        "wavenumbers": wn,
        "wn_min": min(wn) if wn_min is None else wn_min,
        "wn_max": max(wn) if wn_max is None else wn_max,
        "preset": "mc", "passos_preprocessamento": passos,
        "pls": {"x_mean": x_mean, "coef": coef, "intercept": intercept},
        "classes": classes,
    }


# ── predict_portable ────────────────────────────────────────────────

def test_x_mean_e_subtraido_antes_dos_coeficientes():
    """Y = (X - x_mean) @ coef.T + intercept, calculado a mao."""
    p = _portatil([1, 2], [], x_mean=[0.5, 0.25], coef=[[1.0, 0.0], [0.0, 1.0]],
                  intercept=[0.0, 0.0], classes=["a", "b"])
    r = predict_portable(p, np.array([[0.75, 0.75]]), np.array([1.0, 2.0]))
    # (0.75-0.5, 0.75-0.25) = (0.25, 0.5) -> normaliza -> (1/3, 2/3)
    assert r["classe_pred"] == ["b"]
    assert r["confianca_%"][0] == pytest.approx(200.0 / 3.0)


def test_eixo_recortado_por_wn_min_e_wn_max():
    """Eixo 1..6; so' 3..4 entram no modelo (2 variaveis). Mascara errada
    (OR em vez de AND, ou so' um dos limites) dá formato incompatível."""
    wn = [1, 2, 3, 4, 5, 6]
    p = _portatil(wn, [], x_mean=[0.0, 0.0], coef=[[1.0, 0.0], [0.0, 1.0]],
                  intercept=[0.0, 0.0], classes=["a", "b"], wn_min=3.0, wn_max=4.0)
    X = np.array([[0.0, 0.0, 0.1, 0.9, 0.0, 0.0]])
    r = predict_portable(p, X, np.array(wn, float))
    assert r["classe_pred"] == ["b"]
    assert r["confianca_%"][0] == pytest.approx(90.0)


def test_soma_zero_nao_gera_nan_e_confianca_e_zero():
    """Todas as saidas <= 0 -> apos clip soma 0 -> guarda evita 0/0."""
    p = _portatil([1, 2], [], x_mean=[0.0, 0.0], coef=[[1.0, 0.0], [0.0, 1.0]],
                  intercept=[-1.0, -1.0], classes=["a", "b"])
    r = predict_portable(p, np.array([[0.0, 0.0]]), np.array([1.0, 2.0]))
    assert r["confianca_%"] == [0.0]
    assert r["classe_pred"] == ["a"]


def test_soma_menor_que_um_e_renormalizada_para_100_por_cento():
    p = _portatil([1, 2], [], x_mean=[0.0, 0.0], coef=[[1.0, 0.0], [0.0, 1.0]],
                  intercept=[0.0, 0.0], classes=["a", "b"])
    r = predict_portable(p, np.array([[0.5, 0.0]]), np.array([1.0, 2.0]))
    assert r["confianca_%"][0] == pytest.approx(100.0)


# ── _aplicar_passo ──────────────────────────────────────────────────

def test_snv_linha_constante_vira_zero_e_linha_normal_e_padronizada():
    X = np.array([[3.0, 3.0, 3.0], [1.0, 2.0, 3.0]])
    out = _aplicar_passo({"tipo": "SNV"}, X)
    assert np.all(out[0] == 0.0)          # sd == 0 -> divide por 1, nao NaN
    esp = (X[1] - X[1].mean()) / X[1].std()
    assert out[1] == pytest.approx(esp)


def test_standard_scaler_aplica_media_e_escala_e_aceita_none():
    X = np.array([[4.0, 8.0]])
    full = {"tipo": "StandardScaler", "mean": [2.0, 4.0], "scale": [2.0, 4.0]}
    assert _aplicar_passo(full, X) == pytest.approx(np.array([[1.0, 1.0]]))
    so_media = {"tipo": "StandardScaler", "mean": [2.0, 4.0], "scale": None}
    assert _aplicar_passo(so_media, X) == pytest.approx(np.array([[2.0, 4.0]]))
    so_escala = {"tipo": "StandardScaler", "mean": None, "scale": [2.0, 4.0]}
    assert _aplicar_passo(so_escala, X) == pytest.approx(np.array([[2.0, 2.0]]))


@pytest.mark.parametrize("tipo", ["Alfa", "Zulu", "Detrend"])
def test_passo_desconhecido_no_json_levanta_unsupported(tipo):
    """Tipos alfabeticamente antes/depois dos suportados: a igualdade tem
    que ser `==` (nao `<=`/`>=`)."""
    with pytest.raises(UnsupportedModelError, match="desconhecido"):
        _aplicar_passo({"tipo": tipo}, np.zeros((1, 3)))


@pytest.mark.parametrize("nome_classe", ["Alfa", "Zulu", "Detrend"])
def test_serializar_passo_de_tipo_desconhecido_recusa(nome_classe):
    passo = type(nome_classe, (), {})()
    with pytest.raises(UnsupportedModelError, match="serializavel"):
        _serializar_passo("x", passo)


# ── export / save ───────────────────────────────────────────────────

def _pkg_falso(pls, classes=("a", "b")):
    return {
        "preset": "mc",
        "pls_final": pls,
        "preprocessador": SimpleNamespace(named_steps={}),
        "label_binarizer": SimpleNamespace(classes_=list(classes)),
        "wavenumbers": np.array([1.0, 2.0]),
    }


def _pls_falso(**extra):
    return SimpleNamespace(_x_mean=np.zeros(2), coef_=np.eye(2),
                           intercept_=np.zeros(2), **extra)


def test_pls_sem_atributo_scale_nao_e_recusado():
    """`getattr(pls, 'scale', False)`: sem o atributo assume scale=False."""
    dados = export_portable_json(_pkg_falso(_pls_falso()))
    assert dados["classes"] == ["a", "b"]


def test_pls_com_scale_true_e_recusado_e_com_scale_false_aceito():
    with pytest.raises(UnsupportedModelError, match="scale=False"):
        export_portable_json(_pkg_falso(_pls_falso(scale=True)))
    assert export_portable_json(_pkg_falso(_pls_falso(scale=False)))["preset"] == "mc"


def test_export_versao_do_formato_e_1_e_wn_limites_default_do_eixo():
    dados = export_portable_json(_pkg_falso(_pls_falso()))
    assert dados["versao_formato"] == 1
    assert (dados["wn_min"], dados["wn_max"]) == (1.0, 2.0)


def test_save_grava_utf8_legivel_e_indentado(tmp_path):
    """`ensure_ascii=False` (classe com acento fica literal) e `indent=2`."""
    pkg = _pkg_falso(_pls_falso(), classes=("Açaí", "Andiroba"))
    caminho = str(tmp_path / "m.json")
    save_portable_json(pkg, caminho)
    texto = open(caminho, encoding="utf-8").read()
    assert "Açaí" in texto and "\\u00" not in texto
    assert '\n  "formato"' in texto
    assert json.loads(texto)["classes"] == ["Açaí", "Andiroba"]
