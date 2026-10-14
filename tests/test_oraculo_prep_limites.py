# -*- coding: utf-8 -*-
"""Contra-prova do PRE-PROCESSAMENTO espectral (SNV, MSC, EMSC, PQN,
Savitzky-Golay), do limite T2 de Hotelling, do limite Q e do IC BCa contra
implementacoes de REFERENCIA EXTERNAS em R -- mesma disciplina de
`test_oraculo_referencias.py`:

  * mdatools 0.16.0 (Kucheryavskiy) -- prep.norm (snv, pqn), prep.emsc
    (MSC = EMSC de grau 0), prep.savgol, hotelling.crit;
  * boot 1.3 (Canty & Ripley, recomendado do R) -- intervalo BCa
    (Efron 1987) via `boot:::bca.ci`;
  * stats::qchisq do R para o limite Q (aproximacao chi2 por momentos).

Saidas em `tests/fixtures/oraculo_prep_limites/`, geradas por
`gen_replicas.py` (so' as replicas bootstrap, porque o RNG do numpy nao
existe no R) e depois `gen_oracle.R`. Os espectros sao SINTETICOS (o
pre-processamento e' uma formula por amostra; nao ha' propriedade de dado
real que um espectro sintetico nao exercite) e ficam versionados.

Diferencas de CONVENCAO (nao bugs), travadas aqui de proposito para que
qualquer mudanca de convencao apareca como teste vermelho:
  * SNV: GUARACI divide pelo desvio com ddof=0 (np.std), mdatools com
    ddof=1 (sd do R) -> fator constante sqrt((p-1)/p) por espectro.
  * PQN: com a MESMA referencia, o mdatools normaliza por area antes
    (passo 1 de Dieterle et al. 2006) -> fator constante sum|ref|.
    A REFERENCIA padrao difere: GUARACI usa o espectro mediano do treino
    (bruto); mdatools a media dos espectros normalizados por area.
  * T2 de Hotelling: GUARACI usa o limite de FASE II de Tracy, Young &
    Mason (1992), mdatools a forma k(n-1)/(n-k)*F -> fator (n+1)/n.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score

from guaraci.chemometric_stats import hotelling_t2_limit, q_residuals_limit
from guaraci.preprocessamento import EMSC, MSC, PQN, SNV, SavGol
from guaraci.validacao_estatistica import bootstrap_bca_ci

_FIXDIR = Path(__file__).parent / "fixtures" / "oraculo_prep_limites"
_EXATO = dict(rtol=1e-8, atol=1e-10)
_N_TREINO = 10       # linhas 0..9 de X.csv ajustam MSC/EMSC/PQN; todas sao transformadas


def _ler(nome: str) -> np.ndarray:
    return np.loadtxt(_FIXDIR / nome, delimiter=",", skiprows=1, ndmin=2)


# ------------------------------------------------- casos do IC BCa (bootstrap)

def _casos_bca():
    rng = np.random.default_rng(20261015)
    y = np.repeat([0, 1], [28, 32])
    escore = y * 1.2 + rng.normal(size=y.size)
    rotulo = (escore > 0.6).astype(int)
    return {
        # metrica quase continua (AUC) e metrica discreta com muitos empates
        # (acuracia) -- o z0 de ambos usa '<' estrito, como o boot.
        "auc": (y, escore, roc_auc_score, 999),
        "acc": (y, rotulo, accuracy_score, 999),
    }


CASOS_BCA = _casos_bca()


def capturar_replicas(y_true, y_pred, metrica, n_boot):
    """Roda `bootstrap_bca_ci` registrando cada chamada da metrica, na
    ordem em que a funcao as faz: valor observado, `n_boot` replicas
    bootstrap, `n` replicas jackknife."""
    chamadas = []

    def espia(a, b):
        v = float(metrica(a, b))
        chamadas.append(v)
        return v

    bootstrap_bca_ci(y_true, y_pred, espia, n_boot=n_boot)
    v = np.asarray(chamadas)
    assert v.size == 1 + n_boot + len(y_true), "replica descartada"
    return v[0], v[1:1 + n_boot], v[1 + n_boot:]


# ------------------------------------------------------- pre-processamento

def _X():
    X = _ler("X.csv")
    assert X.shape == (15, 101)
    return X


def test_snv_bate_com_mdatools_a_menos_do_ddof():
    X = _X()
    p = X.shape[1]
    np.testing.assert_allclose(SNV().fit_transform(X) * np.sqrt((p - 1) / p),
                               _ler("snv.csv"), **_EXATO)


def test_msc_bate_com_mdatools_emsc_grau_0():
    X = _X()
    msc = MSC().fit(X[:_N_TREINO])
    np.testing.assert_allclose(msc.transform(X), _ler("msc.csv"), **_EXATO)


def test_emsc_grau_2_bate_com_mdatools():
    """O GUARACI usa o indice do canal padronizado (z-score), o mdatools
    o indice levado a [-1, 1]: as duas bases polinomiais geram o MESMO
    subespaco, entao a correcao tem de sair identica."""
    X = _X()
    emsc = EMSC(ordem_polinomial=2).fit(X[:_N_TREINO])
    np.testing.assert_allclose(emsc.transform(X), _ler("emsc2.csv"), **_EXATO)


def test_emsc_grau_0_coincide_com_msc():
    X = _X()
    np.testing.assert_allclose(
        EMSC(ordem_polinomial=0).fit(X[:_N_TREINO]).transform(X),
        MSC().fit(X[:_N_TREINO]).transform(X), **_EXATO)


def test_pqn_bate_com_mdatools_com_a_mesma_referencia():
    X = _X()
    pqn = PQN().fit(X[:_N_TREINO])
    np.testing.assert_allclose(pqn.ref_, np.median(X[:_N_TREINO], axis=0))
    escala = np.sum(np.abs(pqn.ref_))
    np.testing.assert_allclose(pqn.transform(X), _ler("pqn.csv") * escala,
                               **_EXATO)


@pytest.mark.parametrize("janela,ordem,derivada", [
    (11, 2, 0), (11, 2, 1), (11, 3, 2), (7, 3, 1), (15, 4, 2)])
def test_savgol_bate_com_mdatools_inclusive_nas_bordas(janela, ordem, derivada):
    X = _X()
    sg = SavGol(window_length=janela, polyorder=ordem, deriv=derivada)
    np.testing.assert_allclose(
        sg.fit_transform(X),
        _ler(f"savgol_w{janela}_p{ordem}_d{derivada}.csv"), **_EXATO)


# ------------------------------------------------------------------ limites

def test_limite_t2_e_o_de_fase_ii_do_mdatools_vezes_n_mais_1_sobre_n():
    grade = _ler("hotelling.csv")
    for n, k, alpha, t2_mda in grade:
        n, k = int(n), int(k)
        np.testing.assert_allclose(hotelling_t2_limit(n, k, alpha),
                                   t2_mda * (n + 1) / n, **_EXATO)


def test_limite_q_bate_com_qchisq_do_r():
    q = _ler("q.csv").ravel()
    for alpha, ucl in _ler("q_limite.csv"):
        np.testing.assert_allclose(q_residuals_limit(q, alpha), ucl, **_EXATO)


# -------------------------------------------------------------------- BCa

@pytest.mark.parametrize("nome", sorted(CASOS_BCA))
def test_bca_bate_com_pacote_boot(nome):
    y_true, y_pred, metrica, n_boot = CASOS_BCA[nome]
    t0, t, jack = capturar_replicas(y_true, y_pred, metrica, n_boot)
    # guarda do RNG: o oraculo foi gerado com ESTAS replicas
    np.testing.assert_allclose(t, _ler(f"bca_{nome}_t.csv").ravel(), **_EXATO)
    np.testing.assert_allclose(jack, _ler(f"bca_{nome}_jack.csv").ravel(),
                               **_EXATO)

    lo, hi, obs = bootstrap_bca_ci(y_true, y_pred, metrica, n_boot=n_boot)
    assert obs == pytest.approx(t0)
    boot_lo, boot_hi, adj_lo, adj_hi, q7_lo, q7_hi = \
        _ler(f"bca_{nome}_ci.csv").ravel()
    # Mesmos z0, aceleracao e alfas ajustados do boot; o quantil empirico
    # e' o tipo 7 (np.percentile) -> tem de bater exatamente.
    np.testing.assert_allclose([lo, hi], [q7_lo, q7_hi], **_EXATO)
    # Contra o intervalo final do boot (interpolacao norm.inter): a unica
    # diferenca e' a interpolacao entre estatisticas de ordem vizinhas.
    ts = np.sort(t)
    for valor, ref, adj in ((lo, boot_lo, adj_lo), (hi, boot_hi, adj_hi)):
        r = (n_boot + 1) * adj
        faixa = ts[max(int(np.floor(r)) - 2, 0):int(np.ceil(r)) + 1]
        folga = 1e-12      # write.csv do R grava 15 algarismos
        assert faixa.min() - folga <= valor <= faixa.max() + folga
        assert faixa.min() - folga <= ref <= faixa.max() + folga
