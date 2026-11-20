# -*- coding: utf-8 -*-
"""Conversao do sinal bruto em absorbancia dentro do Guaraci (`sinal_entrada`): transmitancia,
reflectancia (log 1/R), Kubelka-Munk e feixe simples (I/I0, com branco em arquivo ou classe), e a
mesma conversao reaplicada na predicao. Dados sinteticos com resposta conhecida."""
from __future__ import annotations

import numpy as np
import pytest

from guaraci import conversao_sinal as cs
from guaraci.config import Config

RNG = np.random.default_rng(2099)
WN = np.linspace(4000.0, 10000.0, 50)
A = RNG.uniform(0.1, 1.5, (8, 50))           # absorbancia "verdadeira"


def test_absorbancia_nao_muda_nada():
    X, info = cs.converter(A, "absorbancia")
    assert X is not None and np.array_equal(X, A) and info["sinal"] == "absorbancia"


@pytest.mark.parametrize("escala", [1.0, 100.0])
def test_transmitancia_fracao_e_porcentagem(escala):
    T = 10.0 ** (-A) * escala
    X, info = cs.converter(T, "transmitancia")
    np.testing.assert_allclose(X, A, rtol=1e-10)
    assert info["percentual"] is (escala == 100.0)


def test_reflectancia_log_1_r_e_kubelka_munk():
    R = 10.0 ** (-A)
    np.testing.assert_allclose(cs.converter(R, "reflectancia")[0], A, rtol=1e-10)
    km, info = cs.converter(R, "kubelka_munk")
    np.testing.assert_allclose(km, (1 - R) ** 2 / (2 * R))
    assert "(1 - R)^2" in info["formula"]


def test_feixe_simples_com_referencia():
    I0 = 1000.0 + 200.0 * np.sin(WN / 900.0)
    I = I0[None, :] * 10.0 ** (-A)
    X, info = cs.converter(I, "feixe_simples", WN, (WN, I0), "arquivo fundo.dx")
    np.testing.assert_allclose(X, A, rtol=1e-10)
    assert info["origem_referencia"] == "arquivo fundo.dx"


def test_feixe_simples_sem_referencia_explica():
    with pytest.raises(ValueError, match="sinal_referencia"):
        cs.converter(A, "feixe_simples", WN)


def test_referencia_que_nao_cobre_a_faixa_recusa():
    with pytest.raises(ValueError, match="sem extrapolar"):
        cs.converter(A, "feixe_simples", WN, (WN[10:], np.ones(40)))


def test_muitos_valores_nao_positivos_recusa_e_poucos_avisam():
    T = 10.0 ** (-A)
    T_ruim = T.copy()
    T_ruim[:, :5] = 0.0                       # 10 % dos pontos
    with pytest.raises(ValueError, match="ja' podem estar em"):
        cs.converter(T_ruim, "transmitancia")
    T_pouco = T.copy()
    T_pouco[0, 0] = -0.001
    X, info = cs.converter(T_pouco, "transmitancia")
    assert np.all(np.isfinite(X)) and any("<= 0" in a for a in info["avisos"])


def test_absorbancia_passada_como_transmitancia_e_recusada():
    """Erro mais provavel do usuario: arquivo ja' em absorbancia marcado como T. Absorbancia entre 1 e
    1,5 seria T > 1 (impossivel) e linha de base negativa seria T <= 0: os dois casos param."""
    with pytest.raises(ValueError, match="ja' estao em absorbancia"):
        cs.converter(A, "transmitancia")                     # A vai ate' 1,5
    with pytest.raises(ValueError, match="ja' podem estar em"):
        cs.converter(np.clip(A, None, 1.0) - 0.3, "transmitancia")
    with pytest.raises(ValueError, match="parecem estar em"):
        cs.converter(A * 2.5, "transmitancia")               # absorbancia ate' ~3,8


def test_pacote_reaplica_a_conversao_na_predicao():
    T = 10.0 ** (-A) * 100.0
    _X, info = cs.converter(T, "transmitancia")
    pkg = {"sinal_entrada": cs.info_para_pacote(info)}
    np.testing.assert_allclose(cs.converter_pelo_pacote(pkg, T, WN), A, rtol=1e-10)
    with pytest.raises(ValueError, match="porcentagem"):
        cs.converter_pelo_pacote(pkg, T / 100.0, WN)        # escala diferente do treino
    assert np.array_equal(cs.converter_pelo_pacote({}, T, WN), T)   # pacote antigo: nada muda


def test_pacote_do_feixe_simples_guarda_o_branco():
    I0 = np.full(50, 800.0)
    I = I0 * 10.0 ** (-A)
    _X, info = cs.converter(I, "feixe_simples", WN, (WN, I0), "classe 'branco'")
    pkg = {"sinal_entrada": cs.info_para_pacote(info)}
    np.testing.assert_allclose(cs.converter_pelo_pacote(pkg, I, WN), A, rtol=1e-10)


def test_pipeline_usa_classe_de_branco_como_i0_e_a_retira(tmp_path):
    from guaraci.pipeline import _converter_sinal_entrada
    I0 = 1000.0 + 100.0 * np.cos(WN / 700.0)
    brancos = I0[None, :] * (1 + RNG.normal(0, 1e-4, (3, 50)))
    amostras = I0[None, :] * 10.0 ** (-A)
    X = np.vstack([brancos, amostras])
    rot = np.array(["branco"] * 3 + ["A"] * 4 + ["B"] * 4)
    mae = np.array([f"g{i}" for i in range(11)])
    cfg = Config(mode="csv")
    cfg.signal_type, cfg.signal_reference = "feixe_simples", "branco"
    Xc, rot2, conc2, mae2, _meta = _converter_sinal_entrada(cfg, X, WN, rot, None, mae, None)
    assert list(rot2) == ["A"] * 4 + ["B"] * 4 and len(mae2) == 8 and conc2 is None
    np.testing.assert_allclose(Xc, A, atol=1e-3)
    assert cfg.sinal_info["sinal"] == "feixe_simples"


def test_pipeline_branco_em_arquivo_e_classe_inexistente(tmp_path):
    from guaraci.pipeline import _converter_sinal_entrada
    I0 = np.full(50, 500.0)
    arq = tmp_path / "fundo.csv"
    arq.write_text("\n".join(f"{x},{y}" for x, y in zip(WN, I0)), encoding="utf-8")
    cfg = Config(mode="csv")
    cfg.signal_type, cfg.signal_reference = "feixe_simples", str(arq)
    rot = np.array(["A"] * 8)
    Xc, *_ = _converter_sinal_entrada(cfg, I0 * 10.0 ** (-A), WN, rot, None, None, None)
    np.testing.assert_allclose(Xc, A, rtol=1e-8)
    cfg.signal_reference = "nao_existe"
    with pytest.raises(ValueError, match="nem uma classe"):
        _converter_sinal_entrada(cfg, A, WN, rot, None, None, None)


def test_modos_que_nao_sao_espectro_nao_convertem():
    from guaraci.pipeline import _converter_sinal_entrada
    cfg = Config(mode="multielementar")
    cfg.signal_type = "transmitancia"
    Xc, *_ = _converter_sinal_entrada(cfg, A, WN, np.array(["A"] * 8), None, None, None)
    assert np.array_equal(Xc, A)


def test_ponta_a_ponta_csv_em_transmitancia_treina_e_prediz(pq, tmp_path):
    """Modo csv com a tabela em transmitancia (%): executar() converte, o pacote guarda a conversao,
    a predicao (joblib e JSON portatil) aceita T % e recusa T em fracao (escala diferente do treino)."""
    import os

    import joblib
    import pandas as pd

    from conftest import achar_pastas_run
    from guaraci import model_export as me
    from guaraci import predicao as pr
    rng = np.random.default_rng(7)
    wn = np.linspace(4000.0, 9000.0, 60)
    base_a = 0.6 + 0.3 * np.exp(-((wn - 5200) / 300) ** 2)
    base_b = 0.6 + 0.3 * np.exp(-((wn - 7000) / 300) ** 2)
    linhas = []
    for classe, base in (("A", base_a), ("B", base_b)):
        for i in range(12):
            a = base + rng.normal(0, 0.01, wn.size)
            linhas.append([classe, f"{classe}{i}"] + list(100.0 * 10.0 ** (-a)))
    df = pd.DataFrame(linhas, columns=["classe", "id"] + [f"{w:g}" for w in wn])
    arq = tmp_path / "t.csv"
    df.to_csv(arq, index=False)
    cfg = pq.Config(mode="csv", csv_file=str(arq), class_column="classe",
                    output_root_folder=str(tmp_path / "saida"), wn_min=4000.0, wn_max=9000.0,
                    n_splits_cv=2, n_repeats_cv=1, n_permutations=5, n_permutations_wold=5,
                    n_bootstrap_vip=3, n_bootstrap_bca=20, n_monte_carlo=3, max_lvs=4)
    cfg.signal_type = "transmitancia"
    pq.executar(cfg)
    runs = achar_pastas_run(cfg.output_root_folder)
    pkg = joblib.load(os.path.join(runs[0], pq.NOME_MODELOS, "modelo_plsda.joblib"))
    assert pkg["sinal_entrada"]["sinal"] == "transmitancia" and pkg["sinal_entrada"]["percentual"]
    T_novos = df[[f"{w:g}" for w in wn]].to_numpy(dtype=float)[[0, 20]]
    res = pr.predict_samples(pkg, T_novos, wn)
    assert list(res["classe_pred"]) == ["A", "B"]
    with pytest.raises(ValueError, match="porcentagem"):
        pr.predict_samples(pkg, T_novos / 100.0, wn)
    port = me.export_portable_json(pkg)
    assert port["sinal_entrada"]["sinal"] == "transmitancia"
    assert list(me.predict_portable(port, T_novos, wn)["classe_pred"]) == ["A", "B"]
