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


# =========================================================================
#  Lacunas achadas por mutacao (cosmic-ray, auditoria 2026-09-19)
#  identificacao.py: 40/196 mutantes sobreviveram na 1a rodada -- os
#  testes acima checavam estrutura (status/cobertura), nunca os NUMEROS
#  do escore nem as guardas de filtragem por especie/teor. Cada teste
#  abaixo fecha um grupo de sobreviventes reais.
# =========================================================================

def _mais_amostras(X, rotulos, conc, mae_id, extras):
    """Anexa amostras (espectro, especie, teor, mae_id) a um dataset."""
    for x, esp, teor, mid in extras:
        X = np.vstack([X, x])
        rotulos = np.append(rotulos, esp)
        conc = np.append(conc, teor)
        mae_id = np.append(mae_id, mid)
    return X, rotulos, conc, mae_id


def test_teor_ausente_nan_nao_conta_como_amostra_adulterada():
    """NaN de teor = amostra pura (mesma convencao do resto do projeto).
    Um mae_id COM token de adulterante mas teor NaN nao pode entrar no
    combo -- mutante `0.0 -> 1.0` no np.where(isnan) o fazia entrar."""
    rng = np.random.default_rng(10)
    X, rot, conc, mae, _ = _monta_dataset(
        rng, ["Andiroba"], {("Andiroba", "S"): 3})
    X, rot, conc, mae = _mais_amostras(
        X, rot, conc, mae,
        [(X[0], "Andiroba", np.nan, "AND-099-S05.00")])
    pca, var_t = _pca_e_var_t(X)
    ens = train_identification_ensemble(pca, var_t, X, rot, conc, mae)
    assert ens[("Andiroba", "soja")]["n_amostras"] == 3


def test_teor_zero_fica_fora_e_teor_fracionario_entra():
    """Fronteira exata do filtro `conc > 0.0`: teor 0.0 fica FORA (pura),
    teor 0.5 (entre 0 e 1) entra -- mutantes `>=`, `> 1.0` e `> -1.0`."""
    rng = np.random.default_rng(11)
    X, rot, conc, mae, _ = _monta_dataset(
        rng, ["Andiroba"], {("Andiroba", "S"): 3})
    X, rot, conc, mae = _mais_amostras(
        X, rot, conc, mae,
        [(X[0], "Andiroba", 0.0, "AND-098-S05.00"),
         (X[1], "Andiroba", 0.5, "AND-097-S05.00")])
    pca, var_t = _pca_e_var_t(X)
    ens = train_identification_ensemble(pca, var_t, X, rot, conc, mae)
    assert ens[("Andiroba", "soja")]["n_amostras"] == 4   # 3 + o de 0.5, sem o de 0.0


def test_especies_diferentes_nunca_se_misturam_no_mesmo_combo():
    """Combo (Andiroba, milho) NAO pode existir quando so' Castanha foi
    adulterada com milho -- o filtro de especie e' igualdade estrita
    (mutante `==` -> `>=` deixava especies alfabeticamente posteriores
    vazarem para o combo)."""
    rng = np.random.default_rng(12)
    combos = {("Andiroba", "S"): 3, ("Castanha", "M"): 3}
    X, rot, conc, mae, _ = _monta_dataset(rng, ["Andiroba", "Castanha"], combos)
    pca, var_t = _pca_e_var_t(X)
    ens = train_identification_ensemble(pca, var_t, X, rot, conc, mae)
    assert set(ens) == {("Andiroba", "soja"), ("Castanha", "milho")}
    assert ens[("Andiroba", "soja")]["n_amostras"] == 3
    assert ens[("Castanha", "milho")]["n_amostras"] == 3


class _PcaIdentidade:
    """Duplo de PCA que devolve o proprio espectro -- score conhecido de
    cabeca, sem depender de ajuste numerico."""

    def transform(self, X):
        return np.asarray(X, dtype=float)


def _combo(centroide, status=CoverageStatus.VALIDATED, alpha=0.05,
           limiar_scores=(0.1, 0.2, 0.3, 0.4)):
    from guaraci.conformal import ConformalOneClass
    cc = ConformalOneClass(alpha=0.25).fit(
        np.array(limiar_scores),
        mae_id=np.array([f"g{i}" for i in range(len(limiar_scores))]))
    return {"centroide": np.asarray(centroide, dtype=float), "conformal": cc,
            "n_grupos": len(limiar_scores), "n_amostras": len(limiar_scores),
            "cobertura_status": status, "alpha_alcancavel": alpha}


def test_escore_e_a_distancia_t2_normalizada_exata():
    """Oraculo independente: score = sum((t - c)^2 / var_t). Antes desta
    asserção, 8 mutantes aritmeticos (/ -> + * - ** // %) da linha do
    escore sobreviviam -- nenhum teste olhava o NUMERO."""
    ens = {("A", "x"): _combo([1.0, 2.0], status=CoverageStatus.NOT_VALIDATED_N1)}
    var_t = np.array([4.0, 0.3])
    # valores com resto fracionario: divisao inteira (//) daria outro numero
    res = identify_sample(ens, _PcaIdentidade(), var_t, np.array([3.5, 3.0]))
    esperado = (3.5 - 1.0) ** 2 / 4.0 + (3.0 - 2.0) ** 2 / 0.3   # 1,5625 + 3,3333
    assert res.escores["A|x"] == pytest.approx(esperado)


def test_amostra_com_numero_impar_de_variaveis_e_aceita_como_vetor_1d():
    ens = {("A", "x"): _combo([0.0, 0.0, 0.0])}
    res = identify_sample(ens, _PcaIdentidade(), np.ones(3), np.zeros(3))
    assert res.escores["A|x"] == pytest.approx(0.0)


def test_exatamente_duas_combinacoes_aceitas_sao_ambiguas_e_listadas():
    """Caminho `len(aceitos) > 1` com EXATAMENTE 2 (o teste antigo aceitava
    qualquer desfecho). Ambas com limiar folgado e centroide na amostra:
    as duas aceitam; a resposta e' ambigua, ordenada, com o MENOR alpha."""
    ens = {("A", "x"): _combo([0.0, 0.0], alpha=0.05),
           ("B", "y"): _combo([0.0, 0.0], alpha=0.02)}
    res = identify_sample(ens, _PcaIdentidade(), np.ones(2), np.zeros(2))
    assert res.classe_identificada is None
    assert res.candidatos_ambiguos == ["A|x", "B|y"]
    assert res.cobertura_status == CoverageStatus.VALIDATED
    assert res.alpha_alcancavel == pytest.approx(0.02)


def test_uma_unica_combinacao_aceita_e_identificada_com_seu_alpha():
    ens = {("A", "x"): _combo([0.0, 0.0], alpha=0.05),
           ("B", "y"): _combo([50.0, 50.0], alpha=0.02)}
    res = identify_sample(ens, _PcaIdentidade(), np.ones(2), np.zeros(2))
    assert res.classe_identificada == "A|x"
    assert res.candidatos_ambiguos == []
    assert res.alpha_alcancavel == pytest.approx(0.05)


def test_numero_de_candidatos_default_e_tres_e_e_respeitado():
    ens = {(f"E{i}", "x"): _combo([float(10 * i)] * 2,
                                  status=CoverageStatus.NOT_VALIDATED_N1)
           for i in range(6)}
    var_t = np.ones(2)
    res = identify_sample(ens, _PcaIdentidade(), var_t, np.zeros(2))
    assert res.candidatos_ambiguos == ["E0|x", "E1|x", "E2|x"]        # default 3
    res2 = identify_sample(ens, _PcaIdentidade(), var_t, np.zeros(2),
                           n_candidatos=2)
    assert res2.candidatos_ambiguos == ["E0|x", "E1|x"]


def test_combinacao_nao_validada_nunca_e_aceita_mesmo_com_conformal_calibrado():
    """`cobertura_status` != VALIDATED bloqueia a aceitacao mesmo que o
    ConformalOneClass exista e o escore caia dentro do limiar."""
    ens = {("A", "x"): _combo([0.0, 0.0],
                              status=CoverageStatus.NOT_VALIDATED_N2_WEAK)}
    res = identify_sample(ens, _PcaIdentidade(), np.ones(2), np.zeros(2))
    assert res.classe_identificada is None


def test_bonferroni_no_limite_exato_de_um_loga_aviso_e_devolve_um(caplog):
    with caplog.at_level(logging.WARNING):
        total = combine_alpha_bonferroni(0.5, 0.5)
    assert total == pytest.approx(1.0)
    assert "deixou de ser informativo" in caplog.text


def test_teor_negativo_nunca_conta_como_adulterado():
    """Fronteira `conc > 0.0` (nao `!= 0`): teor negativo (dado corrompido)
    fica FORA do combo."""
    rng = np.random.default_rng(13)
    X, rot, conc, mae, _ = _monta_dataset(
        rng, ["Andiroba"], {("Andiroba", "S"): 3})
    X, rot, conc, mae = _mais_amostras(
        X, rot, conc, mae, [(X[0], "Andiroba", -5.0, "AND-096-S05.00")])
    pca, var_t = _pca_e_var_t(X)
    ens = train_identification_ensemble(pca, var_t, X, rot, conc, mae)
    assert ens[("Andiroba", "soja")]["n_amostras"] == 3
