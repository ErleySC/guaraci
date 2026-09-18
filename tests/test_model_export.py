# -*- coding: utf-8 -*-
"""Testes de guaraci.model_export -- exportacao portatil de modelo (Grupo
2, fechamento 2026-09-18): JSON puro (sem pickle/joblib) para o caminho
de predicao PRINCIPAL (pre-processamento + PLS-DA + classes).

Duas camadas de contra-prova: (1) UNITARIA, por preset -- constroi um
pacote minimo direto (sem rodar o pipeline inteiro) e compara
`predict_portable` contra `pls_final.predict()`/`preprocessador.
transform()` reais, byte a byte (tolerancia numerica apertada); (2)
PONTA A PONTA -- roda `pipeline.executar()` de verdade (sintetico,
preset default `msc_sg_mc`) e compara `predict_portable` contra
`predicao.predict_samples` no MESMO pacote `.joblib` real.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer

from guaraci.model_export import (
    UnsupportedModelError,
    export_portable_json,
    load_portable_json,
    predict_portable,
    save_portable_json,
)
from guaraci.preprocessamento import build_preprocessor


def _pacote_minimo(preset: str, seed: int = 0, n: int = 60, p: int = 40,
                    n_classes: int = 3):
    """Pacote minimo (so' as chaves que `export_portable_json` precisa),
    construido DIRETO (sem rodar `pipeline.executar()`) -- isola a
    correcao da replicacao numerica do resto da mecanica do pipeline."""
    from guaraci.config import Config

    rng = np.random.default_rng(seed)
    y_int = rng.integers(0, n_classes, size=n)
    X_raw = rng.normal(loc=0.5, scale=0.2, size=(n, p)) + y_int[:, None] * 0.3
    wavenumbers = np.linspace(4000.0, 400.0, p)

    cfg = Config(default_preprocessing=preset)
    preproc = build_preprocessor(cfg).fit(X_raw)
    X_proc = preproc.transform(X_raw)

    lb = LabelBinarizer().fit(y_int)
    Y_bin = np.asarray(lb.transform(y_int), dtype=float)
    pls = PLSRegression(n_components=2, scale=False).fit(X_proc, Y_bin)

    pkg = {
        "preprocessador": preproc,
        "pls_final": pls,
        "label_binarizer": lb,
        "classes": list(lb.classes_),
        "wavenumbers": wavenumbers,
        "preset": preset,
        "wn_min": float(wavenumbers.min()),
        "wn_max": float(wavenumbers.max()),
    }
    return pkg, X_raw, wavenumbers


def _prever_via_pacote_real(pkg, X_new_raw, wn_new):
    """Mesma formula de `predicao.predict_samples` (softmax-like
    clip+normalize+argmax), aplicada DIRETO ao pacote real -- usado como
    oraculo de comparacao (sem depender de `predicao.py` para nao
    acoplar este teste a mudancas la')."""
    preproc = pkg["preprocessador"]
    pls = pkg["pls_final"]
    X_proc = preproc.transform(X_new_raw)
    Y_soft = np.asarray(pls.predict(X_proc), dtype=float)
    Y_clip = np.clip(Y_soft, 0.0, 1.0)
    totais = Y_clip.sum(axis=1, keepdims=True)
    totais[totais < 1e-12] = 1.0
    Y_norm = Y_clip / totais
    classes = list(pkg["label_binarizer"].classes_)
    idx_pred = Y_norm.argmax(axis=1)
    return {
        "classe_pred": [classes[i] for i in idx_pred],
        "confianca_%": (Y_norm.max(axis=1) * 100.0).tolist(),
    }


@pytest.mark.parametrize("preset", ["snv_sg_mc", "msc_sg_mc", "mc", "autoscaling"])
def test_predict_portable_bate_com_pacote_real_por_preset(preset):
    pkg, X_raw, wn = _pacote_minimo(preset)
    portable = export_portable_json(pkg)

    rng = np.random.default_rng(99)
    X_novos = rng.normal(loc=0.5, scale=0.2, size=(10, X_raw.shape[1]))

    esperado = _prever_via_pacote_real(pkg, X_novos, wn)
    obtido = predict_portable(portable, X_novos, wn)

    # `export_portable_json` normaliza classes p/ string (JSON nao tem um
    # tipo "classe" nativo) -- o oraculo usa o tipo nativo do
    # LabelBinarizer (aqui, int sintetico); comparar por str() dos dois
    # lados isola essa normalizacao esperada do que o teste quer provar
    # (a PREDICAO em si bate).
    assert obtido["classe_pred"] == [str(c) for c in esperado["classe_pred"]]
    assert np.allclose(obtido["confianca_%"], esperado["confianca_%"], atol=1e-6)


def test_export_recusa_preset_nao_suportado():
    pkg, _X, _wn = _pacote_minimo("airpls_sg_mc")
    with pytest.raises(UnsupportedModelError, match="airpls_sg_mc"):
        export_portable_json(pkg)


def test_export_recusa_pls_com_scale_true():
    pkg, X_raw, _wn = _pacote_minimo("mc")
    # Substitui por um PLS com scale=True -- nunca usado pelo pipeline
    # real, mas o exportador tem que recusar em vez de produzir um
    # export que prediria errado (formula manual assume scale=False).
    X_proc = pkg["preprocessador"].transform(X_raw)
    y_int = np.arange(len(X_raw)) % len(pkg["classes"])
    Y_bin = np.asarray(pkg["label_binarizer"].transform(y_int), dtype=float)
    pkg["pls_final"] = PLSRegression(n_components=2, scale=True).fit(X_proc, Y_bin)
    with pytest.raises(UnsupportedModelError, match="scale=False"):
        export_portable_json(pkg)


def test_save_e_load_portable_json_roundtrip(tmp_path):
    pkg, X_raw, wn = _pacote_minimo("msc_sg_mc")
    caminho = str(tmp_path / "modelo_portatil.json")
    save_portable_json(pkg, caminho)

    # Confirma que e' JSON puro (nunca pickle) -- carrega sem nenhuma lib
    # do projeto, so' a stdlib.
    with open(caminho, encoding="utf-8") as f:
        bruto = json.load(f)
    assert bruto["formato"] == "guaraci-modelo-portatil"

    portable = load_portable_json(caminho)
    rng = np.random.default_rng(5)
    X_novos = rng.normal(loc=0.5, scale=0.2, size=(4, X_raw.shape[1]))
    resultado = predict_portable(portable, X_novos, wn)
    assert len(resultado["classe_pred"]) == 4


def test_load_portable_json_recusa_arquivo_sem_marca_de_formato(tmp_path):
    caminho = str(tmp_path / "nao_e_modelo.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump({"qualquer": "coisa"}, f)
    with pytest.raises(ValueError, match="formato"):
        load_portable_json(caminho)


# ── Ponta a ponta: pacote real produzido por pipeline.executar() ───────

@pytest.mark.slow
def test_predict_portable_bate_com_predict_samples_em_modelo_real(pq, tmp_path):
    import joblib
    import guaraci.predicao as pr
    from conftest import achar_pastas_run

    cfg = pq.Config(
        input_folder=str(tmp_path / "dados"),
        output_root_folder=str(tmp_path / "saida"),
        mode="sintetico", n_per_class=8, n_synthetic_points=40,
        wn_min=400.0, wn_max=4001.0,
        n_splits_cv=2, n_repeats_cv=1, n_permutations=3,
        n_permutations_wold=3, n_bootstrap_vip=2, n_bootstrap_bca=5,
        n_monte_carlo=2, max_lvs=4, frac_holdout=0.2,
        run_benchmark=False, run_monte_carlo=False, run_shap=False,
        run_wold=False, run_cv_anova=False, run_opls=False,
        executar_etapa4=False, show_plots=False,
    )
    os.makedirs(cfg.input_folder, exist_ok=True)
    pq.executar(cfg)

    runs = achar_pastas_run(cfg.output_root_folder)
    cam_modelo = os.path.join(runs[0], pq.NOME_MODELOS, "modelo_plsda.joblib")
    pkg = joblib.load(cam_modelo)

    portable = export_portable_json(pkg)
    wn = np.asarray(pkg["wavenumbers"], dtype=float)
    rng = np.random.default_rng(7)
    X_novos = rng.normal(loc=0.5, scale=0.05, size=(6, len(wn)))

    df_real = pr.predict_samples(pkg, X_novos, wn)
    resultado_portatil = predict_portable(portable, X_novos, wn)

    assert resultado_portatil["classe_pred"] == list(df_real["classe_pred"])
    assert np.allclose(resultado_portatil["confianca_%"],
                        df_real["confianca_%"].values, atol=1e-6)


@pytest.mark.slow
def test_pipeline_executar_grava_sidecar_portatil_ao_lado_do_joblib(pq, tmp_path):
    """Wiring em `pipeline.executar()` (Grupo 2): com o preset default
    (`msc_sg_mc`, `PLSRegression(scale=False)` -- unica convencao do
    projeto), o `.portatil.json` tem que aparecer ao lado do `.joblib`
    SEM nenhuma acao extra do usuario -- nao so' uma funcao de biblioteca
    alcancavel so' via API Python direta."""
    from conftest import achar_pastas_run

    cfg = pq.Config(
        input_folder=str(tmp_path / "dados"),
        output_root_folder=str(tmp_path / "saida"),
        mode="sintetico", n_per_class=6, n_synthetic_points=30,
        wn_min=400.0, wn_max=4001.0,
        n_splits_cv=2, n_repeats_cv=1, n_permutations=2,
        n_permutations_wold=2, n_bootstrap_vip=2, n_bootstrap_bca=3,
        n_monte_carlo=2, max_lvs=3, frac_holdout=0.0,
        run_benchmark=False, run_monte_carlo=False, run_shap=False,
        run_wold=False, run_cv_anova=False, run_opls=False,
        executar_etapa4=False, show_plots=False,
    )
    os.makedirs(cfg.input_folder, exist_ok=True)
    pq.executar(cfg)

    runs = achar_pastas_run(cfg.output_root_folder)
    cam_modelo = os.path.join(runs[0], pq.NOME_MODELOS, "modelo_plsda.joblib")
    cam_portatil = cam_modelo.replace(".joblib", ".portatil.json")
    assert os.path.isfile(cam_portatil), (
        "pipeline.executar() com preset default nao gerou o sidecar "
        "portatil -- deveria, dado que msc_sg_mc/scale=False sao "
        "suportados")

    portable = load_portable_json(cam_portatil)
    assert portable["preset"] == "msc_sg_mc"
