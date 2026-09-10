"""Heatmap R2cv especie x adulterante (P2): granularidade honesta da
quantificacao. A regressao pooled por especie junta os adulterantes e esconde
que alguns nao sao quantificaveis; este modulo cobre o parser de adulterante,
o calculo R2cv por combinacao (com o contador de falhas) e a figura.
"""
import os

import numpy as np
import pytest

import guaraci.pipeline as pq
from guaraci.dados_io import adulterant_from_mae_id, session_from_mae_id


# ── Parser de adulterante a partir do mae_id ─────────────────────────────────
def test_adulterant_from_mae_id_real_e_sintetico():
    assert adulterant_from_mae_id("CAP-04-11-2099-A1.03") == "algodão"
    assert adulterant_from_mae_id("AND-01-01-2099-S5.00") == "soja"
    assert adulterant_from_mae_id("ESA-M08.00") == "milho"


def test_adulterant_from_mae_id_puro_e_orfao_sao_none():
    assert adulterant_from_mae_id("CAP-04-11-2099") is None   # puro real
    assert adulterant_from_mae_id("ESA-P00") is None          # puro sintetico
    assert adulterant_from_mae_id("orfao_arquivo.dx") is None
    assert adulterant_from_mae_id(None) is None


# ── Sessao de coleta a partir do mae_id (Bloco 9b) ───────────────────────────
def test_session_from_mae_id_remove_o_token_de_teor():
    """Achado real (2026-08-25): 15 niveis de teor da MESMA sessao tem 15
    mae_id diferentes (o teor entra no token final) -- sem remove-lo, a
    calibracao conformal de identificacao (Bloco 9b) contaria 15 'grupos'
    onde ha' 1 sessao de coleta so'."""
    assert (session_from_mae_id("AND-10-06-2099-A1.05")
            == session_from_mae_id("AND-10-06-2099-A2.11")
            == "AND-10-06-2099")
    assert session_from_mae_id("ESA-S05.00") == "ESA"


def test_session_from_mae_id_amostra_pura_e_sua_propria_sessao():
    assert session_from_mae_id("CAP-04-11-2099") == "CAP-04-11-2099"
    assert session_from_mae_id("orfao_arquivo.dx") == "orfao_arquivo.dx"
    assert session_from_mae_id(None) is None


# ── R2cv por especie x adulterante ───────────────────────────────────────────
@pytest.fixture(scope="module")
def dados_adulterados():
    cfg = pq.Config(mode="sintetico", n_per_class=12, n_synthetic_points=60,
                    n_synthetic_replicates=3, synthetic_adulterants=("S", "M"),
                    max_lvs=5, n_splits_cv=3, seed=1)
    _wn, X, rot, conc, mae = pq.generate_synthetic_data(cfg)
    return cfg, X, rot, conc, mae


def test_r2cv_conta_combinacoes(dados_adulterados):
    cfg, X, rot, conc, mae = dados_adulterados
    res = pq.r2cv_species_by_adulterant(X, conc, rot, mae, cfg)
    assert res is not None
    assert set(res["adulterantes"]) == {"soja", "milho"}
    # 3 especies x 2 adulterantes = 6 combinacoes potenciais (avaliadas + n/a)
    assert res["n_total"] + res["n_na"] == 6


def test_r2cv_contador_bate_com_a_matriz(dados_adulterados):
    """O contador de falhas DEVE ser exatamente o numero de celulas finitas
    abaixo do limiar. Corolario do criterio de aceite: se todas passassem, o
    contador seria 0/N."""
    cfg, X, rot, conc, mae = dados_adulterados
    res = pq.r2cv_species_by_adulterant(X, conc, rot, mae, cfg)
    finitos = [float(v) for v in res["matriz"].values() if np.isfinite(v)]
    abaixo = sum(1 for v in finitos if v < res["limiar_r2"])
    acima = sum(1 for v in finitos if v >= res["limiar_r2"])
    assert res["n_falhas"] == abaixo
    assert res["n_ok"] == acima
    assert res["n_total"] == abaixo + acima


def test_r2cv_sem_adulterante_retorna_none():
    """Modo sintetico legado (sem adulterantes) nao tem combinacao especie x
    adulterante -> None (nao inventa heatmap vazio)."""
    cfg = pq.Config(mode="sintetico", n_per_class=10, n_synthetic_replicates=3)
    _wn, X, rot, conc, mae = pq.generate_synthetic_data(cfg)
    assert pq.r2cv_species_by_adulterant(X, conc, rot, mae, cfg) is None


# ── Figura ───────────────────────────────────────────────────────────────────
def test_heatmap_gera_png_valido(tmp_path, dados_adulterados):
    cfg, X, rot, conc, mae = dados_adulterados
    res = pq.r2cv_species_by_adulterant(X, conc, rot, mae, cfg)
    pasta = str(tmp_path)
    os.makedirs(os.path.join(pasta, pq.NOME_GRAFICOS), exist_ok=True)
    pq.fig_heatmap_species_by_adulterant(res, cfg, pasta)
    caminho = os.path.join(pasta, pq.NOME_GRAFICOS,
                           f"fig_heatmap_species_by_adulterant.{cfg.output_format}")
    assert os.path.isfile(caminho)
    assert os.path.getsize(caminho) > 3000   # PNG real, nao arquivo vazio
