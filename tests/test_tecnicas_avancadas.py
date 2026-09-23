# -*- coding: utf-8 -*-
"""Testes de tecnicas_avancadas.py -- orquestracao de ASCA/EPO-GLSW/MCR-ALS/
fusao multibloco sobre dado ja' carregado, fechando a lacuna encontrada na
auditoria de acessibilidade CLI/web (funcoes cientificas implementadas e
testadas em isolamento, mas sem NENHUM caminho de execucao real)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guaraci.fusao_multibloco import BlockMismatchError
from guaraci.mcr_als import MCRALSResultado, MCRALSResultadoSupervisionado
from guaraci.tecnicas_avancadas import (
    fatores_categoricos_disponiveis,
    rodar_asca,
    rodar_epo_glsw,
    rodar_fusao_multibloco,
    rodar_mcr_als,
)


def _dataset_sintetico(n_por_classe=8, p=60, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.repeat(["Copaiba", "Andiroba", "Buriti"], n_por_classe)
    n = len(classes)
    base = {"Copaiba": 1.0, "Andiroba": 2.0, "Buriti": 3.0}
    X = np.array([base[c] + rng.normal(0, 0.1, p) for c in classes])
    mae_id = np.array([f"G{i // 2}" for i in range(n)])  # 2 replicas/grupo
    sessao = np.tile(["S1", "S2"], n // 2 + 1)[:n]
    return X, classes, mae_id, sessao


# --- fatores_categoricos_disponiveis ---------------------------------------

def test_fatores_disponiveis_sempre_inclui_especie():
    rotulos = np.array(["A", "A", "B", "B"])
    fatores = fatores_categoricos_disponiveis(rotulos, None)
    assert set(fatores) == {"especie"}
    np.testing.assert_array_equal(fatores["especie"], rotulos)


def test_fatores_disponiveis_inclui_coluna_categorica_baixa_cardinalidade():
    rotulos = np.array(["A", "A", "B", "B"])
    meta = pd.DataFrame({
        "especie": rotulos,
        "adulterante": ["X", "Y", "X", "Y"],
        "teor": [1.0, 2.0, 3.0, 4.0],          # continua -> excluida
        "mae_id": ["g1", "g1", "g2", "g2"],    # identificadora -> excluida
    })
    fatores = fatores_categoricos_disponiveis(rotulos, meta)
    assert "adulterante" in fatores
    assert "teor" not in fatores
    assert "mae_id" not in fatores


def test_fatores_disponiveis_exclui_coluna_alta_cardinalidade():
    rotulos = np.array(["A"] * 20)
    meta = pd.DataFrame({"especie": rotulos, "id_unico": [str(i) for i in range(20)]})
    fatores = fatores_categoricos_disponiveis(rotulos, meta)
    assert "id_unico" not in fatores


# --- ASCA --------------------------------------------------------------

def test_asca_decompoe_e_detecta_fator_real():
    X, classes, mae_id, _sessao = _dataset_sintetico()
    rel = rodar_asca(X, {"especie": classes}, n_componentes=2, mae_id=mae_id,
                      n_perm=50)
    assert "especie" in rel.decomposicao["efeitos"]
    assert rel.group_aware is True
    assert rel.permutacao is not None
    # 3 classes bem separadas -> p-valor baixo (fator real, nao ruido)
    assert rel.permutacao["especie"]["p_value"] < 0.05


def test_asca_sem_fatores_da_erro_claro():
    X, classes, mae_id, _s = _dataset_sintetico()
    with pytest.raises(ValueError, match="fator"):
        rodar_asca(X, {}, mae_id=mae_id)


def test_asca_sem_permutacao_pula_teste_significancia():
    X, classes, mae_id, _s = _dataset_sintetico()
    rel = rodar_asca(X, {"especie": classes}, mae_id=mae_id,
                      testar_significancia=False)
    assert rel.permutacao is None


# --- EPO/GLSW ------------------------------------------------------------

def test_epo_remove_variancia_do_fator_incomodo():
    X, classes, mae_id, sessao = _dataset_sintetico()
    rel = rodar_epo_glsw(X, classes, sessao, "especie", "sessao", metodo="EPO",
                          n_componentes=1)
    assert rel.metodo == "EPO"
    assert rel.X_corrigido.shape == X.shape
    assert 0.0 <= rel.variancia_removida_fracao <= 1.0


def test_glsw_roda_e_atenua():
    X, classes, mae_id, sessao = _dataset_sintetico()
    rel = rodar_epo_glsw(X, classes, sessao, "especie", "sessao", metodo="GLSW",
                          alpha=1e-2)
    assert rel.metodo == "GLSW"
    assert rel.X_corrigido.shape == X.shape


def test_epo_metodo_invalido_da_erro_claro():
    X, classes, mae_id, sessao = _dataset_sintetico()
    with pytest.raises(ValueError, match="metodo"):
        rodar_epo_glsw(X, classes, sessao, "especie", "sessao", metodo="XYZ")


# --- MCR-ALS ---------------------------------------------------------------

def test_mcr_als_puro_sem_conc():
    X, classes, mae_id, _s = _dataset_sintetico(p=40)
    resultado = rodar_mcr_als(np.abs(X), n_componentes=2)
    assert isinstance(resultado, MCRALSResultado)
    assert resultado.C.shape[0] == X.shape[0]


def test_mcr_als_com_restricao_correlacao_quando_conc_disponivel():
    X, classes, mae_id, _s = _dataset_sintetico(p=40)
    rng = np.random.default_rng(1)
    conc = np.where(np.isin(classes, ["Andiroba"]), rng.uniform(1, 10, len(classes)),
                     np.nan)
    resultado = rodar_mcr_als(np.abs(X), n_componentes=2, conc=conc,
                               indice_componente_alvo=0)
    assert isinstance(resultado, MCRALSResultadoSupervisionado)


def test_mcr_als_restricao_poucas_amostras_referencia_da_erro_claro():
    X, classes, mae_id, _s = _dataset_sintetico(p=40)
    conc = np.full(len(classes), np.nan)
    conc[:2] = [1.0, 2.0]
    with pytest.raises(ValueError, match="pelo menos 3"):
        rodar_mcr_als(np.abs(X), n_componentes=2, conc=conc,
                       indice_componente_alvo=0)


# --- Fusao multibloco -------------------------------------------------------

def test_fusao_multibloco_roda_dois_blocos():
    rng = np.random.default_rng(2)
    n, p1, p2 = 30, 50, 40
    y = rng.uniform(0, 10, n)
    bloco_a = y[:, None] * 0.1 + rng.normal(0, 0.05, (n, p1))
    bloco_b = y[:, None] * 0.2 + rng.normal(0, 0.05, (n, p2))
    resultado, fatias = rodar_fusao_multibloco(
        {"NIR": bloco_a, "MIR": bloco_b}, y, seed=0)
    assert resultado.n_cal + resultado.n_val == n
    assert set(fatias) == {"NIR", "MIR"}
    assert resultado.rmsep >= 0.0


def test_fusao_multibloco_um_bloco_so_da_erro_claro():
    y = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match=">=2 blocos"):
        rodar_fusao_multibloco({"NIR": np.zeros((10, 5))}, y)


def test_fusao_multibloco_tamanhos_diferentes_da_block_mismatch():
    y = np.arange(10, dtype=float)
    blocos = {"NIR": np.zeros((10, 5)), "MIR": np.zeros((9, 5))}
    with pytest.raises(BlockMismatchError):
        rodar_fusao_multibloco(blocos, y)
