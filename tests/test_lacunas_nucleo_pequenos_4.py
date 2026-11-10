"""Ramos de borda de modulos pequenos do nucleo, parte 4 (transferencia, plano amostral/coleta, hsi_*, fusao, EEM, sentinela)."""
from __future__ import annotations


import numpy as np
import pytest

from guaraci import eem_multiway as em
from guaraci import fusao_multibloco as fm
from guaraci import hsi_pixels as hpx
from guaraci import hsi_resampling as hrs
from guaraci import hsi_validation as hv
from guaraci import plano_amostral as pa
from guaraci import transferencia_calibracao as tc


def test_pds_com_janela_negativa_e_recusada():
    X = np.random.default_rng(0).normal(size=(6, 5))
    with pytest.raises(ValueError, match="janela=-1 invalida"):
        tc.piecewise_direct_standardization(X, X, janela=-1)


def test_a_tabela_medida_alcanca_o_plato_e_acima_dele_so_o_conformal_serve(monkeypatch):
    maior = max(cov for _, cov in pa.MEASURED_DDSIMCA_COVERAGE_TABLE)
    assert maior >= pa.PLATO_COBERTURA_DDSIMCA                                  # e' o que garante que o ramo "nao alcancavel na tabela" nao ocorre hoje
    abaixo = pa.ddsimca_sample_size_guidance(pa.PLATO_COBERTURA_DDSIMCA)
    assert abaixo.alcancavel and abaixo.n_sugerido is not None and not abaixo.recomendar_conformal
    acima = pa.ddsimca_sample_size_guidance(min(0.999, pa.PLATO_COBERTURA_DDSIMCA + 0.02))
    assert not acima.alcancavel and acima.recomendar_conformal


def test_orientacao_de_n_quando_a_tabela_nao_alcanca_o_alvo_continua_honesta(monkeypatch):
    monkeypatch.setattr(pa, "MEASURED_DDSIMCA_COVERAGE_TABLE", [(10, 0.80), (20, 0.85)])
    g = pa.ddsimca_sample_size_guidance(0.90)
    assert g.alcancavel is False and g.n_sugerido is None and g.recomendar_conformal and "Nenhum ponto da tabela" in g.ressalva


def test_validacao_externa_recusa_quando_todas_as_gravacoes_sao_do_dia_externo():
    cubos = [np.ones((8, 8, 3))] * 2
    masc = [np.ones((8, 8), dtype=bool)] * 2
    with pytest.raises(ValueError, match="Todas as gravacoes pertencem a dias_externos"):
        hv.run_external_validation_by_day(cubos, masc, ["g1", "g2"], ["a", "b"], ["d1", "d1"], ["d1"])


def test_sobreamostragem_de_conjunto_vazio_devolve_o_proprio_vazio():
    X, y, g = hrs.oversample_minority_groups(np.zeros((0, 3)), np.array([]), np.array([]))
    assert X.shape == (0, 3) and len(y) == len(g) == 0


def test_espectros_da_roi_recusam_cubo_nao_3d():
    with pytest.raises(ValueError, match="cubo 3D"):
        hpx.extract_roi_spectra(np.ones((4, 4)), np.ones((4, 4), dtype=bool))


def test_fusao_multibloco_sem_bloco_e_recusada():
    with pytest.raises(ValueError, match=">=1 bloco"):
        fm.build_multiblock_dataset({})


def test_tensor_eem_recusa_matriz_nao_2d():
    with pytest.raises(ValueError, match="2D"):
        em.construir_tensor_eem({"a": np.ones(5)})


def test_sentinela_usa_o_n_minimo_calculado_do_alpha_quando_nao_e_informado():
    from guaraci import sentinela_deriva as sd
    estado = sd.SentinelState(alpha_nominal=0.05)
    for _ in range(3):                                                        # menos amostras que o minimo para esse alpha: sem poder para testar
        estado.registrar(True)
    alerta = sd.check_drift(estado)
    assert alerta.alerta is False and sd.n_minimum_for_alpha(0.05) > 3


def test_sentinela_aceita_o_n_minimo_informado():
    from guaraci import sentinela_deriva as sd
    estado = sd.SentinelState(alpha_nominal=0.05)
    for _ in range(4):
        estado.registrar(True)
    assert sd.check_drift(estado, n_minimo=1000).alerta is False               # abaixo do minimo declarado pelo chamador


def test_pdf_do_plano_de_coleta_sem_alertas_nao_tem_a_secao_de_alertas(tmp_path):
    import dataclasses

    from guaraci.plano_coleta import export_pdf, plan_collection
    plano = dataclasses.replace(plan_collection(["A", "B"], 4, 2, seed=0), alertas=[])
    caminho = tmp_path / "sem_alertas.pdf"
    export_pdf(plano, str(caminho))
    assert caminho.read_bytes()[:5] == b"%PDF-"
