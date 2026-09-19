# -*- coding: utf-8 -*-
"""Testes do teste R2 do sentinela de deriva (Passo 220): duas amostras
(Fisher exato unilateral) contra a taxa de rejeicao em VALIDACAO CRUZADA
da propria calibracao, em vez do `alpha` nominal.

Cada valor esperado e' calculado por caminho INDEPENDENTE do codigo sob
teste (distribuicao hipergeometrica, nao `fisher_exact`).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, hypergeom

import guaraci.chemometric_stats as cs
from guaraci.chemometric_stats import ad_rejection_rate_cv
from guaraci.sentinela_deriva import (
    SentinelState,
    check_drift,
    hook_apos_predicao,
    load_state,
    save_state,
)


def _p_hipergeometrico(k, n, r, m):
    """P(X >= k) com X ~ Hipergeometrica: entre n+m amostras ha' k+r
    rejeitadas; sorteia-se n (producao). = p unilateral de Fisher."""
    return float(hypergeom.sf(k - 1, n + m, k + r, n))


def _estado(k, n, r=None, m=None, **kw):
    e = SentinelState(ref_rejeitadas=r, ref_n=m, **kw)
    e.historico = [False] * k + [True] * (n - k)
    return e


# ── check_drift com referencia (R2) ─────────────────────────────────

def test_p_valor_bate_com_hipergeometrica_independente():
    for k, n, r, m in [(6, 24, 3, 40), (10, 40, 3, 40), (2, 30, 0, 50),
                       (12, 32, 5, 60)]:
        d = check_drift(_estado(k, n, r, m))
        assert d.p_valor == pytest.approx(_p_hipergeometrico(k, n, r, m))
        assert d.teste == "duas_amostras_cv"
        assert d.taxa_referencia == pytest.approx(r / m)


def test_alerta_quando_producao_rejeita_muito_mais_que_a_calibracao():
    d = check_drift(_estado(k=12, n=30, r=2, m=40))
    assert d.alerta is True
    assert "DERIVA PROVAVEL" in d.mensagem
    assert "validacao cruzada" in d.mensagem and "LEGADO" not in d.mensagem


def test_r2_nao_alerta_quando_a_taxa_e_a_da_calibracao_mas_o_teste_legado_alertaria():
    """A CAUSA RAIZ do achado do Passo 219: uma calibracao que rejeita ~12%
    em controle (bem acima do nominal 5%). Producao com a MESMA taxa
    (5/40 = 12,5%) e' processo em controle: R2 nao alerta; o teste
    binomial contra 5% (legado) alerta -- falso alarme."""
    r2 = check_drift(_estado(k=5, n=40, r=12, m=100))
    legado = check_drift(_estado(k=5, n=40))
    assert legado.teste == "binomial_nominal"
    assert legado.alerta is True                 # falso alarme do teste antigo
    assert r2.alerta is False                    # R2 nao cai nele
    assert r2.p_valor > 0.05


def test_sem_referencia_cai_no_binomial_legado_e_avisa():
    e = _estado(k=10, n=40)
    d = check_drift(e)
    esperado = binomtest(10, 40, 0.05, alternative="greater").pvalue
    assert d.p_valor == pytest.approx(esperado)
    assert d.teste == "binomial_nominal" and d.taxa_referencia is None
    assert "LEGADO" in d.mensagem


def test_referencia_vazia_ou_parcial_e_tratada_como_ausente():
    assert check_drift(_estado(10, 40, r=3, m=0)).teste == "binomial_nominal"
    assert check_drift(_estado(10, 40, r=None, m=40)).teste == "binomial_nominal"
    assert check_drift(_estado(10, 40, r=3, m=None)).teste == "binomial_nominal"


def test_abaixo_do_minimo_nao_testa_e_devolve_a_referencia():
    d = check_drift(_estado(3, 10, r=2, m=40))
    assert d.alerta is False and np.isnan(d.p_valor)
    assert d.teste == "duas_amostras_cv"
    assert d.taxa_referencia == pytest.approx(0.05)


def test_referencia_com_zero_rejeicoes_na_cv_e_valida():
    """r=0: taxa de referencia 0 nao pode dividir por zero nem quebrar."""
    d = check_drift(_estado(k=8, n=30, r=0, m=40))
    assert d.p_valor == pytest.approx(_p_hipergeometrico(8, 30, 0, 40))
    assert d.alerta is True


def test_significancia_e_estrita_tambem_no_r2():
    e = _estado(k=8, n=30, r=1, m=40)
    p = check_drift(e).p_valor
    assert check_drift(e, significancia=p).alerta is False
    assert check_drift(e, significancia=p * 1.001).alerta is True


# ── persistencia ────────────────────────────────────────────────────

def test_save_load_preserva_referencia(tmp_path):
    e = _estado(k=3, n=25, r=4, m=40, janela=50)
    c = str(tmp_path / "s.json")
    save_state(e, c)
    e2 = load_state(c)
    assert (e2.ref_rejeitadas, e2.ref_n, e2.janela) == (4, 40, 50)
    assert e2.historico == e.historico


def test_estado_antigo_sem_chaves_de_referencia_carrega_como_legado(tmp_path):
    c = tmp_path / "antigo.json"
    c.write_text(json.dumps({"alpha_nominal": 0.05, "janela": None,
                             "historico": [True, False]}), encoding="utf-8")
    e = load_state(str(c))
    assert e.ref_rejeitadas is None and e.ref_n is None
    assert e.n == 2


# ── hook: referencia vem das colunas AD_ref_cv_* do lote ────────────

def _lote(n_fora, n_dentro, r=None, m=None):
    df = pd.DataFrame({"AD_dentro_dominio": [False] * n_fora + [True] * n_dentro})
    if r is not None:
        df["AD_ref_cv_rejeitadas"] = r
        df["AD_ref_cv_n"] = m
    return df


def test_hook_grava_referencia_do_lote_e_usa_r2(tmp_path):
    modelo = str(tmp_path / "m.joblib")
    d = hook_apos_predicao(modelo, _lote(3, 27, r=4, m=40))
    assert d.teste == "duas_amostras_cv"
    e = load_state(modelo + ".sentinela.json")
    assert (e.ref_rejeitadas, e.ref_n) == (4, 40)
    assert d.p_valor == pytest.approx(_p_hipergeometrico(3, 30, 4, 40))


def test_hook_sem_colunas_de_referencia_usa_legado(tmp_path):
    d = hook_apos_predicao(str(tmp_path / "m.joblib"), _lote(3, 27))
    assert d.teste == "binomial_nominal"


def test_hook_modelo_recalibrado_zera_historico_e_troca_referencia(tmp_path):
    modelo = str(tmp_path / "m.joblib")
    hook_apos_predicao(modelo, _lote(10, 10, r=2, m=40))
    d = hook_apos_predicao(modelo, _lote(1, 19, r=9, m=40))    # outra calibracao
    e = load_state(modelo + ".sentinela.json")
    assert (e.ref_rejeitadas, e.ref_n) == (9, 40)
    assert e.n == 20                                  # so' o lote novo
    assert d.n == 20


def test_hook_mesma_referencia_acumula_historico(tmp_path):
    modelo = str(tmp_path / "m.joblib")
    hook_apos_predicao(modelo, _lote(1, 9, r=2, m=40))
    d = hook_apos_predicao(modelo, _lote(1, 9, r=2, m=40))
    assert d.n == 20


def test_hook_estado_legado_ganha_referencia_sem_perder_historico(tmp_path):
    modelo = str(tmp_path / "m.joblib")
    hook_apos_predicao(modelo, _lote(1, 9))                    # legado
    d = hook_apos_predicao(modelo, _lote(1, 9, r=2, m=40))     # modelo retreinado c/ ref
    assert d.n == 20 and d.teste == "duas_amostras_cv"


def test_hook_referencia_nan_ou_n_zero_e_ignorada(tmp_path):
    df = _lote(2, 8, r=np.nan, m=np.nan)
    assert hook_apos_predicao(str(tmp_path / "a.joblib"), df).teste == "binomial_nominal"
    df0 = _lote(2, 8, r=0, m=0)
    assert hook_apos_predicao(str(tmp_path / "b.joblib"), df0).teste == "binomial_nominal"


# ── ad_rejection_rate_cv ────────────────────────────────────────────

def _X(n=30, p=20, seed=0):
    return np.random.default_rng(seed).normal(size=(n, p))


def test_cv_devolve_contagens_coerentes():
    r, m = ad_rejection_rate_cv(_X(), 2)
    assert m == 30 and 0 <= r <= m


def test_cv_deterministica():
    assert ad_rejection_rate_cv(_X(), 2) == ad_rejection_rate_cv(_X(), 2)


def test_cv_group_aware_replicas_nunca_ficam_no_treino_do_proprio_fold(monkeypatch):
    """Replicas (mesmo `groups`) tem que sair juntas: o treino de cada fold
    nao pode conter NENHUMA linha do grupo julgado."""
    X = _X(n=24, p=15, seed=3)
    grupos = np.repeat(np.arange(8), 3)
    vistos = []
    orig = cs.training_applicability_domain

    def espia(pca, X_tr, alpha=0.05):
        vistos.append(np.asarray(X_tr))
        return orig(pca, X_tr, alpha=alpha)
    monkeypatch.setattr(cs, "training_applicability_domain", espia)
    r, m = ad_rejection_rate_cv(X, 2, groups=grupos)
    assert m == 24 and len(vistos) == 8            # 1 fold por grupo
    for X_tr in vistos:
        assert X_tr.shape[0] == 21                 # 24 - 3 replicas do grupo
    # cada linha ausente do treino pertence a UM unico grupo
    for X_tr in vistos:
        ausentes = [i for i in range(24)
                    if not np.any(np.all(np.isclose(X_tr, X[i]), axis=1))]
        assert len(set(grupos[ausentes])) == 1 and len(ausentes) == 3


def test_cv_muitos_grupos_usa_k_folds_estaveis(monkeypatch):
    X = _X(n=40, p=12, seed=4)
    n_fits = []
    orig = cs.training_applicability_domain
    monkeypatch.setattr(cs, "training_applicability_domain",
                        lambda pca, Xt, alpha=0.05: (n_fits.append(1),
                                                     orig(pca, Xt, alpha=alpha))[1])
    r, m = ad_rejection_rate_cv(X, 2, max_loo_groups=10, n_splits_grupos=4)
    assert len(n_fits) == 4 and m == 40


def test_cv_acima_de_max_grupos_usa_subconjunto_deterministico():
    X = _X(n=50, p=10, seed=5)
    r1, m1 = ad_rejection_rate_cv(X, 2, max_grupos=20)
    r2, m2 = ad_rejection_rate_cv(X, 2, max_grupos=20)
    assert m1 == m2 == 20 and (r1, m1) == (r2, m2)


def test_cv_groups_de_tamanho_errado_levanta():
    with pytest.raises(ValueError, match="diferem em tamanho"):
        ad_rejection_rate_cv(_X(10), 2, groups=np.arange(9))


def test_cv_treino_pequeno_demais_e_ignorado():
    """n=3 amostras, k=2: treino de 2 < k+2 -> nenhum fold avaliavel."""
    assert ad_rejection_rate_cv(_X(3, 5), 2) == (0, 0)


# ── fronteiras (achados da mutacao do Passo 220) ────────────────────

def test_referencia_com_1_amostra_e_valida_e_negativa_nao():
    assert check_drift(_estado(10, 40, r=0, m=1)).teste == "duas_amostras_cv"
    assert check_drift(_estado(10, 40, r=1, m=-5)).teste == "binomial_nominal"


def _ref(df):
    from guaraci.sentinela_deriva import _referencia_do_lote
    return _referencia_do_lote(df)


def test_referencia_do_lote_exige_as_DUAS_colunas():
    base = pd.DataFrame({"AD_dentro_dominio": [True, False]})
    so_r = base.assign(AD_ref_cv_rejeitadas=2)
    so_n = base.assign(AD_ref_cv_n=40)
    assert _ref(so_r) is None and _ref(so_n) is None
    assert _ref(base.assign(AD_ref_cv_rejeitadas=2, AD_ref_cv_n=40)) == (2, 40)


def test_referencia_do_lote_usa_a_primeira_linha():
    df = pd.DataFrame({"AD_ref_cv_rejeitadas": [2, 3, 4],
                       "AD_ref_cv_n": [40, 50, 60]})
    assert _ref(df) == (2, 40)


def test_referencia_do_lote_vazio_ou_de_uma_linha():
    vazio = pd.DataFrame({"AD_ref_cv_rejeitadas": [], "AD_ref_cv_n": []})
    assert _ref(vazio) is None
    um = pd.DataFrame({"AD_ref_cv_rejeitadas": [1], "AD_ref_cv_n": [10]})
    assert _ref(um) == (1, 10)


@pytest.mark.parametrize("r,m,esperado", [
    (np.nan, 40, None),     # so' r NaN
    (2, np.nan, None),      # so' n NaN
    (2, 0, None),           # n = 0
    (2, -3, None),          # n negativo
    (0, 1, (0, 1)),         # menor referencia valida
    (0, 40, (0, 40)),
])
def test_referencia_do_lote_nan_zero_e_negativo(r, m, esperado):
    df = pd.DataFrame({"AD_ref_cv_rejeitadas": [r], "AD_ref_cv_n": [m]})
    assert _ref(df) == esperado


def test_hook_estado_com_referencia_parcial_nao_e_tratado_como_recalibracao(tmp_path):
    modelo = str(tmp_path / "m.joblib")
    (tmp_path / "m.joblib.sentinela.json").write_text(json.dumps({
        "alpha_nominal": 0.05, "janela": None,
        "historico": [True] * 10, "ref_rejeitadas": 2, "ref_n": None}),
        encoding="utf-8")
    d = hook_apos_predicao(modelo, _lote(1, 9, r=5, m=40))
    assert d.n == 20                          # historico preservado


def test_hook_referencia_menor_ou_n_diferente_tambem_zera_historico(tmp_path):
    modelo = str(tmp_path / "m.joblib")
    hook_apos_predicao(modelo, _lote(2, 8, r=9, m=40))
    d = hook_apos_predicao(modelo, _lote(1, 9, r=2, m=40))     # r MENOR
    assert d.n == 10
    d = hook_apos_predicao(modelo, _lote(1, 9, r=2, m=50))     # so' n muda
    assert d.n == 10


# ── ponta a ponta: pipeline real -> pacote -> predicao -> sentinela ──

@pytest.mark.slow
def test_pipeline_real_grava_referencia_e_sentinela_usa_r2(pq, tmp_path):
    """`pipeline.executar()` (dados sinteticos) grava `ad_cv_rejeitadas`/
    `ad_cv_n` no pacote; `predict_samples` as expoe como colunas; o hook do
    sentinela, sobre essa saida REAL, usa o teste de 2 amostras."""
    import joblib
    import os

    import guaraci.predicao as pr
    from conftest import achar_pastas_run

    cfg = pq.Config(
        input_folder=str(tmp_path / "dados"),
        output_root_folder=str(tmp_path / "saida"),
        mode="sintetico", n_per_class=8, n_synthetic_points=40,
        wn_min=400.0, wn_max=4001.0,
        n_splits_cv=2, n_repeats_cv=1, n_permutations=2,
        n_permutations_wold=2, n_bootstrap_vip=2, n_bootstrap_bca=3,
        n_monte_carlo=2, max_lvs=3, frac_holdout=0.2,
        run_benchmark=False, run_monte_carlo=False, run_shap=False,
        run_wold=False, run_cv_anova=False, run_opls=False,
        executar_etapa4=False, show_plots=False,
    )
    os.makedirs(cfg.input_folder, exist_ok=True)
    pq.executar(cfg)
    runs = achar_pastas_run(cfg.output_root_folder)
    cam = os.path.join(runs[0], pq.NOME_MODELOS, "modelo_plsda.joblib")
    pkg = joblib.load(cam)

    assert pkg["ad_cv_n"] > 0 and 0 <= pkg["ad_cv_rejeitadas"] <= pkg["ad_cv_n"]

    wn = np.asarray(pkg["wavenumbers"], dtype=float)
    X = np.random.default_rng(3).normal(loc=0.5, scale=0.05, size=(24, len(wn)))
    df = pr.predict_samples(pkg, X, wn)
    assert set(df["AD_ref_cv_n"]) == {pkg["ad_cv_n"]}
    assert set(df["AD_ref_cv_rejeitadas"]) == {pkg["ad_cv_rejeitadas"]}

    alerta = hook_apos_predicao(cam, df)
    assert alerta is not None and alerta.teste == "duas_amostras_cv"
    e = load_state(cam + ".sentinela.json")
    assert (e.ref_rejeitadas, e.ref_n) == (pkg["ad_cv_rejeitadas"], pkg["ad_cv_n"])
