# -*- coding: utf-8 -*-
"""Testes de guaraci.identificacao — ensemble conformal por combinacao
especie x adulterante (Bloco 9b).

O invariante que importa aqui (mesmo espirito de test_conformal.py): a
cobertura reportada tem que refletir o `n` de GRUPOS (mae_id) real, nunca um
numero inflado ou inventado, e uma classe so' pode ser "identificada" quando
a garantia estatistica realmente sustenta -- nunca por default/silencio.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
from sklearn.decomposition import PCA

from guaraci.chemometric_stats import training_applicability_domain
from guaraci.identificacao import (
    CoverageStatus,
    combine_alpha_bonferroni,
    identify_sample,
    train_identification_ensemble,
)


def _mae_id(prefixo: str, grupo_idx: int, letra_adult: str) -> str:
    """mae_id valido para `adulterant_from_mae_id` (ultimo token = letra+digitos)."""
    return f"{prefixo}-{grupo_idx:03d}-{letra_adult}05.00"


def _monta_dataset(rng, especies, combos_n_grupos, p=12, sep=8.0, noise=0.05):
    """Constroi X (espectros sinteticos), rotulos, conc, mae_id.

    `combos_n_grupos`: dict {(especie, letra_adult): n_grupos}. Cada
    combinacao recebe um centro proprio bem separado dos demais (sep),
    e cada grupo daquela combinacao e' UMA amostra fisica (mae_id unico),
    com pequeno ruido em torno do centro da combinacao.
    """
    X_list, rot_list, conc_list, mae_list = [], [], [], []
    centros = {}
    idx_centro = 0
    for chave in combos_n_grupos:
        centros[chave] = rng.normal(loc=idx_centro * sep, scale=1.0, size=p)
        idx_centro += 1

    for (especie, letra), n_grupos in combos_n_grupos.items():
        centro = centros[(especie, letra)]
        for g in range(n_grupos):
            X_list.append(centro + rng.normal(scale=noise, size=p))
            rot_list.append(especie)
            conc_list.append(10.0)
            mae_list.append(_mae_id(especie[:3].upper(), g, letra))

    # ancoras puras (conc=0), uma por especie -- nao entram em nenhum combo
    # (conc<=0 e' filtrado por train_identification_ensemble).
    for especie in especies:
        X_list.append(rng.normal(scale=noise, size=p))
        rot_list.append(especie)
        conc_list.append(0.0)
        mae_list.append(f"{especie[:3].upper()}-PURO")

    X = np.array(X_list)
    rotulos = np.array(rot_list, dtype=str)
    conc = np.array(conc_list, dtype=float)
    mae_id = np.array(mae_list, dtype=str)
    return X, rotulos, conc, mae_id, centros


def _pca_e_var_t(X, n_components=3):
    pca = PCA(n_components=n_components, random_state=0).fit(X)
    treino = training_applicability_domain(pca, X)
    return pca, np.asarray(treino["var_t"], dtype=float)


# =========================================================================
#  Classificacao de cobertura por n_grupos
# =========================================================================

def test_combo_n1_marca_nao_validado_n1_com_alpha_none():
    rng = np.random.default_rng(0)
    combos = {("Andiroba", "S"): 1, ("Andiroba", "M"): 2}
    X, rotulos, conc, mae_id, _ = _monta_dataset(rng, ["Andiroba"], combos)
    pca, var_t = _pca_e_var_t(X)

    ensemble = train_identification_ensemble(pca, var_t, X, rotulos, conc, mae_id)

    info = ensemble[("Andiroba", "soja")]
    assert info["n_grupos"] == 1
    assert info["cobertura_status"] == CoverageStatus.NOT_VALIDATED_N1
    assert info["alpha_alcancavel"] is None
    assert info["conformal"] is None


def test_combo_n2_marca_nao_validado_n2_fraco_com_alpha_um_terco():
    rng = np.random.default_rng(1)
    combos = {("Andiroba", "S"): 1, ("Andiroba", "M"): 2}
    X, rotulos, conc, mae_id, _ = _monta_dataset(rng, ["Andiroba"], combos)
    pca, var_t = _pca_e_var_t(X)

    ensemble = train_identification_ensemble(pca, var_t, X, rotulos, conc, mae_id)

    info = ensemble[("Andiroba", "milho")]
    assert info["n_grupos"] == 2
    assert info["cobertura_status"] == CoverageStatus.NOT_VALIDATED_N2_WEAK
    assert info["alpha_alcancavel"] == pytest.approx(1.0 / 3.0)
    assert info["conformal"] is not None


def test_combo_com_muitos_grupos_e_validado():
    rng = np.random.default_rng(2)
    combos = {("Andiroba", "S"): 25}
    X, rotulos, conc, mae_id, _ = _monta_dataset(rng, ["Andiroba"], combos)
    pca, var_t = _pca_e_var_t(X)

    ensemble = train_identification_ensemble(pca, var_t, X, rotulos, conc, mae_id)

    info = ensemble[("Andiroba", "soja")]
    assert info["n_grupos"] == 25
    assert info["cobertura_status"] == CoverageStatus.VALIDATED
    assert info["alpha_alcancavel"] == pytest.approx(0.05)


def test_ensemble_sem_mae_id_retorna_vazio(caplog):
    rng = np.random.default_rng(3)
    combos = {("Andiroba", "S"): 5}
    X, rotulos, conc, _mae_id, _ = _monta_dataset(rng, ["Andiroba"], combos)
    pca, var_t = _pca_e_var_t(X)

    with caplog.at_level(logging.WARNING):
        ensemble = train_identification_ensemble(
            pca, var_t, X, rotulos, conc, mae_id=None)
    assert ensemble == {}
    assert "mae_id ausente" in caplog.text


def test_diluicoes_da_mesma_sessao_nao_inflam_n_grupos():
    """Regressao do achado real (2026-08-25, medido contra o dataset):
    varias amostras com o MESMO prefixo de sessao (especie+data) mas teores
    diferentes tem mae_id DIFERENTES (o token final inclui o teor) -- se
    contadas por mae_id bruto, pareceriam N grupos independentes; sao
    UMA SO sessao de coleta diluida em N niveis. `train_identification_
    ensemble` tem que colapsar para n_grupos=1 usando `session_from_mae_id`,
    nao o `mae_id` bruto (que produziria n_grupos=15 aqui)."""
    rng = np.random.default_rng(42)
    p = 8
    centro = rng.normal(size=p)
    X_list, rot_list, conc_list, mae_list = [], [], [], []
    for nivel, teor in enumerate(np.linspace(1.0, 15.0, 15)):
        X_list.append(centro + rng.normal(scale=0.05, size=p))
        rot_list.append("Andiroba")
        conc_list.append(teor)
        mae_list.append(f"AND-10-06-2099-A{teor:.2f}")   # MESMA sessao (data)
    X_list.append(rng.normal(scale=0.05, size=p))
    rot_list.append("Andiroba"); conc_list.append(0.0)
    mae_list.append("AND-PURO")

    X = np.array(X_list)
    rotulos = np.array(rot_list, dtype=str)
    conc = np.array(conc_list, dtype=float)
    mae_id = np.array(mae_list, dtype=str)
    pca, var_t = _pca_e_var_t(X, n_components=2)

    ensemble = train_identification_ensemble(pca, var_t, X, rotulos, conc, mae_id)
    info = ensemble[("Andiroba", "algodão")]
    assert info["n_amostras"] == 15
    assert info["n_grupos"] == 1, (
        "15 diluicoes da MESMA sessao foram contadas como grupos "
        "independentes -- pseudo-replicacao na calibracao conformal")
    assert info["cobertura_status"] == CoverageStatus.NOT_VALIDATED_N1


def test_combinacao_ausente_dos_dados_nao_entra_no_ensemble():
    """So' combinacoes com pelo menos 1 amostra adulterada entram -- nunca
    inventa uma entrada vazia para uma combinacao nao observada."""
    rng = np.random.default_rng(4)
    combos = {("Andiroba", "S"): 3}
    X, rotulos, conc, mae_id, _ = _monta_dataset(rng, ["Andiroba"], combos)
    pca, var_t = _pca_e_var_t(X)

    ensemble = train_identification_ensemble(pca, var_t, X, rotulos, conc, mae_id)
    assert ("Andiroba", "milho") not in ensemble
    assert ("Andiroba", "algodão") not in ensemble


# =========================================================================
#  identify_sample -- nunca forca classe
# =========================================================================

def test_identify_aceita_quando_validado_e_proximo():
    # p/n_components pequenos e sep >> noise: geometria bem-condicionada
    # (com dimensao alta e poucas amostras, T2 normalizado por var_t fica
    # instavel -- ver nota de verificacao no relatorio do Bloco 9b).
    rng = np.random.default_rng(5)
    combos = {("Andiroba", "S"): 25, ("Castanha", "M"): 25}
    X, rotulos, conc, mae_id, centros = _monta_dataset(
        rng, ["Andiroba", "Castanha"], combos, p=4, sep=10.0, noise=0.1)
    pca, var_t = _pca_e_var_t(X, n_components=2)
    ensemble = train_identification_ensemble(pca, var_t, X, rotulos, conc, mae_id)

    # amostra nova: bem proxima do centro real da combinacao Andiroba x soja
    X_nova = centros[("Andiroba", "S")] + rng.normal(scale=0.03, size=len(pca.mean_))

    res = identify_sample(ensemble, pca, var_t, X_nova)
    assert res.cobertura_status == CoverageStatus.VALIDATED
    assert res.classe_identificada == "Andiroba|soja"
    assert res.candidatos_ambiguos == []


def test_identify_nunca_forca_classe_quando_nao_validado():
    """Contra-prova central (D7-b): com cobertura fraca (n=2), mesmo uma
    amostra EXATAMENTE no centro de uma combinacao nao pode virar
    `classe_identificada` -- so' aparece em candidatos_ambiguos."""
    rng = np.random.default_rng(6)
    combos = {("Andiroba", "S"): 2, ("Andiroba", "M"): 1}
    X, rotulos, conc, mae_id, centros = _monta_dataset(
        rng, ["Andiroba"], combos)
    pca, var_t = _pca_e_var_t(X)
    ensemble = train_identification_ensemble(pca, var_t, X, rotulos, conc, mae_id)

    X_nova = centros[("Andiroba", "S")].copy()   # exatamente no centro
    res = identify_sample(ensemble, pca, var_t, X_nova)

    assert res.classe_identificada is None
    assert "Andiroba|soja" in res.candidatos_ambiguos
    assert res.cobertura_status == CoverageStatus.NOT_VALIDATED_N2_WEAK
    assert res.alpha_alcancavel == pytest.approx(1.0 / 3.0)


def test_identify_amostra_longe_de_tudo_ainda_reporta_candidatos():
    rng = np.random.default_rng(7)
    combos = {("Andiroba", "S"): 25}
    X, rotulos, conc, mae_id, centros = _monta_dataset(
        rng, ["Andiroba"], combos)
    pca, var_t = _pca_e_var_t(X)
    ensemble = train_identification_ensemble(pca, var_t, X, rotulos, conc, mae_id)

    X_longe = centros[("Andiroba", "S")] + 500.0   # muito longe do centroide
    res = identify_sample(ensemble, pca, var_t, X_longe)

    assert res.classe_identificada is None
    assert res.candidatos_ambiguos == ["Andiroba|soja"]


def test_identify_com_ensemble_vazio_nao_quebra():
    res = identify_sample({}, PCA(n_components=1).fit(np.eye(3)),
                          np.array([1.0]), np.zeros(3))
    assert res.classe_identificada is None
    assert res.candidatos_ambiguos == []
    assert res.cobertura_status is None


def test_identify_duas_combinacoes_validadas_aceitas_e_ambiguo():
    """Se MAIS de uma combinacao validada aceita a mesma amostra, a garantia
    estatistica nao decide entre elas -- ambiguo por construcao, nunca
    escolhe uma arbitrariamente."""
    rng = np.random.default_rng(8)
    # centros bem proximos (sep pequeno) para que uma amostra no meio
    # caia dentro do limiar conformal de AMBAS as combinacoes.
    combos = {("Andiroba", "S"): 25, ("Andiroba", "M"): 25}
    X, rotulos, conc, mae_id, centros = _monta_dataset(
        rng, ["Andiroba"], combos, sep=0.3)
    pca, var_t = _pca_e_var_t(X)
    ensemble = train_identification_ensemble(pca, var_t, X, rotulos, conc, mae_id)

    meio = (centros[("Andiroba", "S")] + centros[("Andiroba", "M")]) / 2.0
    res = identify_sample(ensemble, pca, var_t, meio)

    if len(res.candidatos_ambiguos) >= 2 or res.classe_identificada is None:
        # ambiguo (aceito por >=2, ou nao aceito por nenhuma) -- ambos os
        # desfechos sao aceitaveis aqui; o que NAO pode acontecer e' uma
        # classe escolhida sozinha sem justificativa.
        pass
    else:
        pytest.fail("uma unica classe foi escolhida sem que apenas uma "
                    "combinacao a aceitasse")


# =========================================================================
#  Propagacao de incerteza (Bonferroni/uniao)
# =========================================================================

def test_combine_alpha_bonferroni_soma_direta():
    assert combine_alpha_bonferroni(0.05, 0.10, 0.03) == pytest.approx(0.18)


def test_combine_alpha_bonferroni_none_se_qualquer_etapa_sem_alpha():
    assert combine_alpha_bonferroni(0.05, None, 0.03) is None
    assert combine_alpha_bonferroni(None) is None


def test_combine_alpha_bonferroni_satura_em_um_com_aviso(caplog):
    with caplog.at_level(logging.WARNING):
        total = combine_alpha_bonferroni(0.6, 0.7)
    assert total == pytest.approx(1.0)
    assert "1.0" in caplog.text or "deixou de ser informativo" in caplog.text


def test_combine_alpha_bonferroni_sem_argumentos_e_zero():
    assert combine_alpha_bonferroni() == pytest.approx(0.0)
