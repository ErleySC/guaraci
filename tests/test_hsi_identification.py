"""Testes de hsi_identification.py (Passo 106) -- conjunto aberto no
nivel de objeto do HSI. A contra-prova
`test_identify_hsi_object_tipo_desconhecido_retorna_none` e' OBRIGATORIA
pela instrucao: um objeto de tipo NAO presente no treino deve retornar
"desconhecido" (objeto_identificado=None), nunca forcar um palpite."""
from __future__ import annotations

import numpy as np

from guaraci.hsi_identification import (aggregate_object_spectrum,
                                        identify_hsi_object,
                                        train_hsi_identification_ensemble)
from guaraci.identificacao import CoverageStatus


def _cubo_com_espectro(espectro_medio: np.ndarray, seed: int, n_pixels: int = 20,
                        ruido: float = 0.02) -> tuple:
    rng = np.random.default_rng(seed)
    cubo = np.tile(espectro_medio, (n_pixels, 1, 1)).reshape(n_pixels, 1, -1)
    cubo = cubo + rng.normal(scale=ruido, size=cubo.shape)
    mascara = np.ones((n_pixels, 1), dtype=bool)
    return cubo, mascara


def _dataset_treino(n_objetos_por_tipo=25, n_bandas=10, seed=0):
    rng = np.random.default_rng(seed)
    tipo_a = rng.normal(loc=0.3, scale=0.05, size=n_bandas)
    tipo_b = rng.normal(loc=0.8, scale=0.05, size=n_bandas)
    cubos, mascaras, group_ids, frutas, cameras = [], [], [], [], []
    for i in range(n_objetos_por_tipo):
        cubo, mascara = _cubo_com_espectro(
            tipo_a + rng.normal(scale=0.01, size=n_bandas), seed=100 + i)
        cubos.append(cubo); mascaras.append(mascara)
        group_ids.append(f"tipoA_{i}"); frutas.append("TipoA"); cameras.append("VIS")
    for i in range(n_objetos_por_tipo):
        cubo, mascara = _cubo_com_espectro(
            tipo_b + rng.normal(scale=0.01, size=n_bandas), seed=200 + i)
        cubos.append(cubo); mascaras.append(mascara)
        group_ids.append(f"tipoB_{i}"); frutas.append("TipoB"); cameras.append("VIS")
    return cubos, mascaras, group_ids, frutas, cameras, tipo_a, tipo_b


# ── aggregate_object_spectrum ────────────────────────────────────────────

def test_aggregate_object_spectrum_e_media_dos_pixels_da_roi():
    cubo = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    mascara = np.array([[True, False], [True, False]])
    espectro = aggregate_object_spectrum(cubo, mascara)
    esperado = cubo[mascara].mean(axis=0)
    np.testing.assert_allclose(espectro, esperado)


# ── train_hsi_identification_ensemble ────────────────────────────────────

def test_train_ensemble_calibra_1_entrada_por_combinacao_fruta_camera():
    cubos, mascaras, group_ids, frutas, cameras, *_ = _dataset_treino()
    ensemble = train_hsi_identification_ensemble(
        cubos, mascaras, group_ids, frutas, cameras, alpha_nominal=0.2)
    assert set(ensemble.keys()) == {("TipoA", "VIS"), ("TipoB", "VIS")}
    for info in ensemble.values():
        assert info["n_grupos"] == 25
        assert info["cobertura_status"] == CoverageStatus.VALIDATED


def test_train_ensemble_frente_e_costas_do_mesmo_objeto_nao_infla_n_grupos():
    """2 gravacoes com o MESMO group_id (frente/costas) devem contar como
    1 OBJETO so' -- nao 2 -- ao calibrar (mesmo espirito do Passo 97: o
    objeto fisico e' a unidade independente, nao a gravacao)."""
    cubo, mascara = _cubo_com_espectro(np.full(5, 0.5), seed=1)
    cubos = [cubo, cubo.copy()]
    mascaras = [mascara, mascara.copy()]
    group_ids = ["obj1", "obj1"]  # frente + costas do MESMO objeto
    ensemble = train_hsi_identification_ensemble(
        cubos, mascaras, group_ids, ["Fruta", "Fruta"], ["VIS", "VIS"],
        alpha_nominal=0.2)
    entrada = ensemble[("Fruta", "VIS")]
    assert entrada["n_grupos"] == 1
    assert entrada["cobertura_status"] == CoverageStatus.NOT_VALIDATED_N1
    assert entrada["pca"] is None  # nao da' pra' fitar PCA com 1 amostra so'


# ── identify_hsi_object ───────────────────────────────────────────────────

def test_identify_hsi_object_reconhece_objeto_do_proprio_tipo():
    cubos, mascaras, group_ids, frutas, cameras, tipo_a, _tipo_b = _dataset_treino()
    ensemble = train_hsi_identification_ensemble(
        cubos, mascaras, group_ids, frutas, cameras, alpha_nominal=0.2)

    cubo_novo, mascara_novo = _cubo_com_espectro(tipo_a, seed=999)
    resultado = identify_hsi_object(ensemble, cubo_novo, mascara_novo, camera="VIS")
    assert resultado.objeto_identificado == "TipoA|VIS"
    assert resultado.cobertura_status == CoverageStatus.VALIDATED


# ── CONTRA-PROVA OBRIGATORIA (Passo 106) ─────────────────────────────────

def test_identify_hsi_object_tipo_desconhecido_retorna_none():
    """Objeto de um tipo espectral COMPLETAMENTE diferente dos 2
    calibrados no treino -- nao deve ser aceito por nenhuma entrada do
    ensemble (score muito alto p/ os 2 centroides calibrados)."""
    cubos, mascaras, group_ids, frutas, cameras, tipo_a, tipo_b = _dataset_treino()
    ensemble = train_hsi_identification_ensemble(
        cubos, mascaras, group_ids, frutas, cameras, alpha_nominal=0.2)

    n_bandas = len(tipo_a)
    espectro_desconhecido = np.full(n_bandas, 5.0)  # bem longe de 0.3 e 0.8
    cubo_novo, mascara_novo = _cubo_com_espectro(espectro_desconhecido, seed=777,
                                                 ruido=0.01)
    resultado = identify_hsi_object(ensemble, cubo_novo, mascara_novo, camera="VIS")
    assert resultado.objeto_identificado is None
    assert resultado.candidatos_ambiguos or resultado.escores  # nao esconde os escores


def test_identify_hsi_object_camera_incompativel_retorna_none():
    cubos, mascaras, group_ids, frutas, cameras, tipo_a, _ = _dataset_treino()
    ensemble = train_hsi_identification_ensemble(
        cubos, mascaras, group_ids, frutas, cameras, alpha_nominal=0.2)

    cubo_novo, mascara_novo = _cubo_com_espectro(tipo_a, seed=42)
    resultado = identify_hsi_object(ensemble, cubo_novo, mascara_novo, camera="NIR")
    assert resultado.objeto_identificado is None
    assert resultado.escores == {}


def test_identify_hsi_object_ensemble_vazio_nao_lanca_excecao():
    resultado = identify_hsi_object({}, np.zeros((2, 2, 3)),
                                    np.ones((2, 2), dtype=bool), camera="VIS")
    assert resultado.objeto_identificado is None
