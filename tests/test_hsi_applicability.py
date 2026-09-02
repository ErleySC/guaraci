"""Testes de hsi_applicability.py (Passo 108) -- dominio de
aplicabilidade no espaco de pixels do HSI. A contra-prova
`test_evaluate_hsi_ad_rejeita_pixels_sinteticos_fora_do_dominio` e'
OBRIGATORIA pela instrucao: uma cena sintetica deliberadamente fora do
dominio deve ser marcada corretamente."""
from __future__ import annotations

import numpy as np
import pytest

from guaraci.hsi_applicability import (evaluate_hsi_applicability_domain,
                                       train_hsi_applicability_domain)


def _pixels_treino(n=200, n_bandas=15, seed=0):
    rng = np.random.default_rng(seed)
    centro = rng.normal(loc=0.4, scale=0.02, size=n_bandas)
    return centro + rng.normal(scale=0.02, size=(n, n_bandas))


# ── train_hsi_applicability_domain ───────────────────────────────────────

def test_train_hsi_ad_devolve_dominio_com_n_bandas_correto():
    X = _pixels_treino(n_bandas=15)
    dominio = train_hsi_applicability_domain(X)
    assert dominio.n_bandas == 15


def test_train_hsi_ad_dados_insuficientes_levanta_erro():
    X = np.zeros((0, 10))
    with pytest.raises(ValueError, match="insuficientes"):
        train_hsi_applicability_domain(X)


# ── evaluate_hsi_applicability_domain: amostras do proprio treino ───────

def test_evaluate_hsi_ad_aceita_maioria_dos_proprios_pixels_de_treino():
    X = _pixels_treino(seed=1)
    dominio = train_hsi_applicability_domain(X, alpha=0.05)
    resultado = evaluate_hsi_applicability_domain(dominio, X)
    assert resultado["sensor_compativel"] is True
    # ~95% esperado nominal (alpha=0.05) -- folga generosa, e' um teste
    # de sanidade, nao uma checagem exata de calibracao.
    assert resultado["fracao_dentro"] > 0.7


# ── incompatibilidade de sensor (cameras com numero de bandas diferente) ──

def test_evaluate_hsi_ad_sensor_incompativel_nao_lanca_excecao():
    """Achado real do Passo 104: Kaki/VIS=224 bandas, Kaki/VIS_COR=249 --
    avaliar dados de uma camera contra o dominio de outra NAO pode
    tentar pca.transform com shape incompativel (isso seria um erro
    cru do sklearn, nao uma decisao interpretavel)."""
    X_vis = _pixels_treino(n_bandas=224, seed=2)
    dominio_vis = train_hsi_applicability_domain(X_vis)
    X_vis_cor = _pixels_treino(n_bandas=249, seed=3)

    resultado = evaluate_hsi_applicability_domain(dominio_vis, X_vis_cor)
    assert resultado["sensor_compativel"] is False
    assert "incompativel" in resultado["motivo"].lower()
    assert np.all(resultado["dentro_dominio"] == False)  # noqa: E712


def test_evaluate_hsi_ad_forma_errada_levanta_erro_claro():
    X = _pixels_treino(n_bandas=10)
    dominio = train_hsi_applicability_domain(X)
    with pytest.raises(ValueError, match="2D"):
        evaluate_hsi_applicability_domain(dominio, np.zeros(10))


# ── generalizacao entre FRUTAS no MESMO sensor (mesmo eixo espectral) ────

def test_evaluate_hsi_ad_rejeita_fruta_diferente_mesmo_sensor():
    """2 combinacoes com o MESMO numero de bandas (mesmo modelo de
    camera, ex. Specim FX10 usado no VIS de varias frutas -- ver
    docstring do modulo) mas quimica/reflectancia DIFERENTE: o dominio
    calibrado numa deve rejeitar a maioria dos pixels da outra."""
    rng = np.random.default_rng(4)
    n_bandas = 20
    X_fruta_a = rng.normal(loc=0.3, scale=0.02, size=(150, n_bandas))
    X_fruta_b = rng.normal(loc=0.75, scale=0.02, size=(150, n_bandas))  # bem diferente

    dominio_a = train_hsi_applicability_domain(X_fruta_a, alpha=0.05)
    resultado = evaluate_hsi_applicability_domain(dominio_a, X_fruta_b)
    assert resultado["sensor_compativel"] is True
    assert resultado["fracao_dentro"] < 0.3  # maioria REJEITADA


# ── CONTRA-PROVA OBRIGATORIA (Passo 108) ─────────────────────────────────

def test_evaluate_hsi_ad_rejeita_pixels_sinteticos_fora_do_dominio():
    """Cena sintetica DELIBERADAMENTE fora do dominio (deslocamento
    grande + ruido bem maior que o treino) deve ser marcada
    corretamente como fora do dominio na quase totalidade dos pixels."""
    X_treino = _pixels_treino(n=200, n_bandas=12, seed=5)
    dominio = train_hsi_applicability_domain(X_treino, alpha=0.05)

    rng = np.random.default_rng(6)
    X_fora = rng.normal(loc=3.0, scale=0.5, size=(50, 12))  # bem longe do treino

    resultado = evaluate_hsi_applicability_domain(dominio, X_fora)
    assert resultado["sensor_compativel"] is True
    assert resultado["fracao_dentro"] < 0.1, (
        f"esperado quase todos os pixels sinteticos fora do dominio, "
        f"fracao_dentro={resultado['fracao_dentro']}")
