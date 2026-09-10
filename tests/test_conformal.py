"""Testes de guaraci.conformal — a camada de decisao que substitui o
limiar chi2 do DD-SIMCA na autenticacao one-class.

O invariante que importa aqui nao e' "roda sem erro": e' que o metodo
RECUSE produzir limiar quando o n nao sustenta o alpha pedido. Foi
exatamente o comportamento oposto (piso silencioso em Nh=Nq=1) que tornou
o DD-SIMCA nao-estimavel sem avisar neste dataset.
"""
import numpy as np
import pytest

from guaraci.conformal import (
    ConformalOneClass,
    achievable_alpha,
    conformal_threshold,
    n_minimum_for_alpha,
)


# ── O limite duro 1/(n+1) ───────────────────────────────────────────────
def test_alpha_alcancavel_e_um_sobre_n_mais_um():
    assert achievable_alpha(1) == pytest.approx(0.5)
    assert achievable_alpha(9) == pytest.approx(0.1)
    assert achievable_alpha(19) == pytest.approx(0.05)
    assert achievable_alpha(0) == pytest.approx(1.0)   # degenerado


def test_n_minimum_for_alpha_e_o_inverso():
    assert n_minimum_for_alpha(0.05) == 19
    assert n_minimum_for_alpha(0.10) == 9
    assert n_minimum_for_alpha(0.25) == 3
    for a in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            n_minimum_for_alpha(a)


# ── A propriedade central: recusar em vez de inventar ───────────────────
def test_limiar_recusa_quando_n_nao_sustenta_o_alpha():
    """REGRESSAO do mode de falha do DD-SIMCA: com n insuficiente NAO se
    devolve numero. `limiar` fica NaN, `alcancavel` False, e o aviso diz
    qual alpha seria alcancavel."""
    r = conformal_threshold(np.array([1.0]), alpha=0.05)     # n=1
    assert r["alcancavel"] is False
    assert np.isnan(r["limiar"])
    assert r["achievable_alpha"] == pytest.approx(0.5)
    assert "NAO e' alcancavel" in r["aviso"]
    assert "n>=19" in r["aviso"]


def test_limiar_definido_assim_que_o_n_permite():
    rng = np.random.default_rng(0)
    s = rng.normal(size=19)
    r = conformal_threshold(s, alpha=0.05)
    assert r["alcancavel"] is True
    assert np.isfinite(r["limiar"])
    # k = ceil((n+1)(1-alpha)) = ceil(20*0.95) = 19 -> o maior escore
    assert r["k"] == 19
    assert r["limiar"] == pytest.approx(np.max(s))


def test_limiar_e_o_k_esimo_menor_escore():
    """Propriedade exata da correcao de amostra finita, nao aproximada."""
    s = np.arange(1.0, 101.0)          # n=100, escores 1..100
    r = conformal_threshold(s, alpha=0.10)
    assert r["k"] == int(np.ceil(101 * 0.90))   # 91
    assert r["limiar"] == pytest.approx(91.0)


# ── Cobertura empirica: a garantia tem que valer de fato ────────────────
def test_cobertura_empirica_atinge_o_nominal():
    """Com calibracao e teste da MESMA distribuicao (permutaveis), a
    fracao aceita tem que ficar >= 1-alpha. E' a garantia conformal; se
    este teste falhar, a implementacao do quantil esta errada."""
    rng = np.random.default_rng(7)
    coberturas = []
    for _ in range(200):
        calib = rng.normal(size=50)
        teste = rng.normal(size=200)
        r = conformal_threshold(calib, alpha=0.10)
        coberturas.append(float(np.mean(teste <= r["limiar"])))
    media = float(np.mean(coberturas))
    assert media >= 0.88, f"cobertura {media:.3f} abaixo do nominal 0.90"
    assert media <= 0.95, f"cobertura {media:.3f} conservadora demais"


# ── Group-aware: replicas nao sao amostras independentes ────────────────
def test_replicas_nao_inflam_o_n_de_calibracao():
    """3 replicas de 2 amostras fisicas = n=2 para o limite 1/(n+1), NAO
    n=6. Tratar replicas como independentes violaria permutabilidade e
    daria uma garantia que nao existe -- e' o mesmo erro que o projeto
    combate no GroupKFold, aqui no passo de calibracao."""
    scores = np.array([1.0, 1.1, 0.9, 5.0, 5.1, 4.9])
    grupos = np.array(["g1", "g1", "g1", "g2", "g2", "g2"])

    cc = ConformalOneClass(alpha=0.10).fit(scores, mae_id=grupos)
    assert cc.info_["n_calibracao"] == 2            # nao 6
    assert cc.info_["achievable_alpha"] == pytest.approx(1 / 3)
    assert cc.info_["alcancavel"] is False          # 0.10 exige n>=9

    # Sem mae_id o n seria 6 -- a diferenca e' justamente o achado
    cc_sem = ConformalOneClass(alpha=0.10).fit(scores)
    assert cc_sem.info_["n_calibracao"] == 6


def test_colapso_por_grupo_usa_mediana_resistente_a_replica_atipica():
    """Uma replica descolada nao pode arrastar o escore da amostra."""
    scores = np.array([1.0, 1.0, 99.0])            # 3a replica atipica
    grupos = np.array(["g1", "g1", "g1"])
    colapsado = ConformalOneClass._colapsar_por_grupo(scores, grupos)
    assert colapsado.size == 1
    assert colapsado[0] == pytest.approx(1.0)      # mediana, nao 33.7


# ── predict() sem limiar valido ─────────────────────────────────────────
def test_predict_sem_limiar_rejeita_tudo_em_vez_de_aceitar_tudo():
    """Sem garantia nao ha' autenticacao a declarar. Devolver tudo aceito
    seria indistinguivel de um modelo permissivo bem calibrado -- o tipo
    de ambiguidade que esta auditoria existe para eliminar."""
    cc = ConformalOneClass(alpha=0.05).fit(np.array([1.0]))   # n=1
    assert cc.info_["alcancavel"] is False
    pred = cc.predict(np.array([0.001, 1.0, 1e9]))
    assert pred.dtype == bool
    assert not pred.any()


def test_predict_aceita_abaixo_do_limiar_quando_valido():
    rng = np.random.default_rng(3)
    cc = ConformalOneClass(alpha=0.10).fit(rng.normal(size=100))
    pred = cc.predict(np.array([-10.0, cc.limiar_, cc.limiar_ + 1e-9]))
    assert pred[0] is np.True_ or bool(pred[0])
    assert bool(pred[1])            # <= e' inclusivo
    assert not bool(pred[2])
