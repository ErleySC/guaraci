# -*- coding: utf-8 -*-
"""Testes de amostragem_ativa.py (Bloco 25) -- ensemble SINTETICO (mesma
forma de `identificacao.train_identification_ensemble`, sem PCA/conformal
de verdade -- so' testa a logica de priorizacao). Sanidade contra o
acervo REAL fica em `scripts/medicoes/amostragem_ativa_oleos_reais.py`
(dado privado, mesmo padrao dos demais scripts de medicao desta sessao).
"""
from __future__ import annotations

from guaraci.amostragem_ativa import priorizar_amostragem
from guaraci.identificacao import CoverageStatus


def _info(n_grupos, status):
    return {"n_grupos": n_grupos, "n_amostras": n_grupos * 3,
            "cobertura_status": status, "centroide": None, "conformal": None,
            "alpha_alcancavel": None}


def test_combinacao_validada_fica_com_prioridade_zero():
    ensemble = {
        ("Andiroba", "soja"): _info(19, CoverageStatus.VALIDATED),
    }
    lista = priorizar_amostragem(ensemble, alpha=0.05)
    assert lista[0].prioridade == 0.0
    assert lista[0].cobertura_status == CoverageStatus.VALIDATED


def test_combinacao_mais_perto_da_validacao_tem_prioridade_maior():
    """1 sessao faltando (18/19) precisa ranquear ACIMA de 17 faltando
    (2/19) -- maior 'impacto esperado' por sessao investida."""
    ensemble = {
        ("Andiroba", "soja"): _info(18, CoverageStatus.NOT_VALIDATED_N2_WEAK),
        ("Maracuja", "algodao"): _info(2, CoverageStatus.NOT_VALIDATED_N2_WEAK),
    }
    lista = priorizar_amostragem(ensemble, alpha=0.05)
    assert lista[0].especie == "Andiroba"
    assert lista[0].sessoes_faltantes == 1
    assert lista[1].sessoes_faltantes == 17
    assert lista[0].prioridade > lista[1].prioridade


def test_combinacao_com_1_sessao_fica_mais_urgente_que_validada():
    ensemble = {
        ("Goiaba", "milho"): _info(1, CoverageStatus.NOT_VALIDATED_N1),
        ("Andiroba", "soja"): _info(19, CoverageStatus.VALIDATED),
    }
    lista = priorizar_amostragem(ensemble, alpha=0.05)
    assert lista[0].especie == "Goiaba"
    assert lista[0].prioridade > 0.0
    assert lista[1].prioridade == 0.0


def test_sessoes_necessario_bate_com_n_minimum_for_alpha():
    from guaraci.conformal import n_minimum_for_alpha
    ensemble = {("X", "Y"): _info(5, CoverageStatus.NOT_VALIDATED_N2_WEAK)}
    lista = priorizar_amostragem(ensemble, alpha=0.10)
    assert lista[0].n_sessoes_necessario == n_minimum_for_alpha(0.10)


def test_erro_por_especie_aumenta_prioridade_de_especie_pior():
    """Duas combinacoes com o MESMO numero de sessoes faltando -- a
    especie com erro medido pior tem que ranquear primeiro quando
    `erro_por_especie` e' fornecido."""
    ensemble = {
        ("EspecieBoa", "soja"): _info(10, CoverageStatus.NOT_VALIDATED_N2_WEAK),
        ("EspeciePior", "soja"): _info(10, CoverageStatus.NOT_VALIDATED_N2_WEAK),
    }
    erro = {"EspecieBoa": 0.05, "EspeciePior": 0.40}
    lista = priorizar_amostragem(ensemble, alpha=0.05, erro_por_especie=erro)
    assert lista[0].especie == "EspeciePior"


def test_sem_erro_por_especie_as_duas_ficam_empatadas():
    ensemble = {
        ("A", "soja"): _info(10, CoverageStatus.NOT_VALIDATED_N2_WEAK),
        ("B", "soja"): _info(10, CoverageStatus.NOT_VALIDATED_N2_WEAK),
    }
    lista = priorizar_amostragem(ensemble, alpha=0.05)
    assert lista[0].prioridade == lista[1].prioridade


def test_ensemble_vazio_devolve_lista_vazia():
    assert priorizar_amostragem({}, alpha=0.05) == []


def test_todos_os_campos_de_prioridadeamostragem_presentes():
    ensemble = {("Andiroba", "soja"): _info(3, CoverageStatus.NOT_VALIDATED_N2_WEAK)}
    lista = priorizar_amostragem(ensemble)
    r = lista[0]
    assert r.especie == "Andiroba"
    assert r.adulterante == "soja"
    assert r.n_sessoes_atual == 3
    assert r.n_sessoes_necessario > 0
    assert r.sessoes_faltantes == r.n_sessoes_necessario - 3
