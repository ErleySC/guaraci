# -*- coding: utf-8 -*-
"""Teste de integracao do fluxo completo do mode cego (Bloco 9b):
Detectar -> Identificar -> Quantificar, contra um pacote de modelo REAL
exportado por `executar()` (nao um pkg montado a mao) -- mesmo padrao de
`test_predicao.py` (roda `executar()` sintetico UMA vez, sessao inteira).

Verificado tambem contra o dataset real (2026-08-25, fora da suite): 36
combinacoes especie x adulterante com n_grupos=1 (sessao unica) e 2 com
n_grupos=2 (Andiroba x soja, Maracuja x algodao) -- exatamente o numero
documentado em docs/MANUAL.md. O mode sintetico usado aqui NAO tem token de
data no mae_id (`ESA-S05.00`, sem sessao real codificada), entao toda
combinacao cai em NOT_VALIDATED_N1 -- o que e' o comportamento ESPERADO e
util para testar o gate (D4/D7-b), nao um bug deste teste.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import achar_pastas_run
from guaraci.identificacao import CoverageStatus
from guaraci.predicao import predict_blind


@pytest.fixture(scope="module")
def pkg_bloco9b(pq, tmp_path_factory):
    base = tmp_path_factory.mktemp("bloco9b")
    cfg = pq.Config(
        input_folder=str(base / "dados"),
        output_root_folder=str(base / "saida"),
        mode="sintetico", level="N3",
        n_per_class=12, n_synthetic_points=60, n_synthetic_replicates=3,
        synthetic_adulterants=("S", "M"),
        wn_min=400.0, wn_max=4001.0,
        n_splits_cv=2, n_repeats_cv=1, n_permutations=5,
        n_permutations_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20,
        n_monte_carlo=3, max_lvs=5,
    )
    os.makedirs(cfg.input_folder, exist_ok=True)
    pq.executar(cfg)

    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs, "executar() nao criou pasta de saida"
    cam_modelo = os.path.join(runs[0], pq.NOME_MODELOS, "modelo_plsda.joblib")
    assert os.path.isfile(cam_modelo)
    pkg = joblib.load(cam_modelo)
    return pkg, runs[0], cam_modelo


# =========================================================================
#  Persistencia (D1/D6): o ensemble e a regressao por especie tem que
#  estar no pacote exportado por executar(), nao so' calculaveis a parte.
# =========================================================================

def test_ensemble_de_identificacao_foi_persistido(pkg_bloco9b):
    pkg, _pasta, _cam = pkg_bloco9b
    assert "identification_ensemble" in pkg
    ensemble = pkg["identification_ensemble"]
    assert ensemble, "nenhuma combinacao especie x adulterante calibrada"
    assert {"Esp_A", "Esp_B", "Esp_C"} == {esp for esp, _ in ensemble}
    assert {"soja", "milho"} == {ad for _, ad in ensemble}


def test_regressao_por_especie_foi_persistida(pkg_bloco9b):
    """Quantificar (Bloco 9b) so' tem o que aplicar em amostra nova se os
    pipelines de PLS-R por especie (ajustados em `pls_regression_by_species`,
    D4) forem persistidos no pacote -- ate' aqui, `executar()` so' calculava
    metricas de CV, nunca o modelo pronto para produzir."""
    pkg, _pasta, _cam = pkg_bloco9b
    assert "regressao_por_especie" in pkg
    modelos = pkg["regressao_por_especie"]
    assert modelos, "nenhum modelo de regressao por especie foi persistido"
    for especie, info in modelos.items():
        assert hasattr(info["pipeline"], "predict")
        assert isinstance(info["n_lv"], int) and info["n_lv"] >= 1


def test_mae_id_sintetico_sem_data_cai_em_nao_validado_n1(pkg_bloco9b):
    """mae_id sintetico ('ESA-S05.00') nao tem token de sessao/data --
    session_from_mae_id colapsa toda combinacao numa unica sessao. Isso e'
    honesto (nao ha' replica de sessao real nos dados sinteticos), nao um
    defeito: serve para provar que o gate bloqueia mesmo com dezenas de
    ESPECTROS por combinacao, porque o que importa e' `n_grupos`."""
    pkg, _pasta, _cam = pkg_bloco9b
    for (_esp, _adult), info in pkg["identification_ensemble"].items():
        assert info["n_grupos"] == 1
        assert info["cobertura_status"] == CoverageStatus.NOT_VALIDATED_N1
        assert info["alpha_alcancavel"] is None


def test_ddsimca_por_especie_foi_persistido(pkg_bloco9b):
    """Fecha o gap do Detectar: DD-SIMCA por especie (pureza) precisa
    existir no pacote MESMO numa rodada N3 (cfg.run_ddsimca nao foi pedido
    explicitamente) -- ate' esta correcao, DD-SIMCA so' era calculado (e
    nunca persistido) em rodadas N2."""
    pkg, _pasta, _cam = pkg_bloco9b
    assert "ddsimca_por_especie" in pkg
    modelos = pkg["ddsimca_por_especie"]
    assert modelos, "nenhum modelo de pureza por especie foi persistido"
    assert {"Esp_A", "Esp_B", "Esp_C"} == set(modelos)
    for info in modelos.values():
        assert hasattr(info["pca"], "transform")
        for chave in ("h0", "q0", "Nh", "Nq", "f_crit", "n_grupos_calibracao"):
            assert chave in info


# =========================================================================
#  Detectar tem 2 sinais: AD (similaridade global) e DD-SIMCA (pureza da
#  especie predita) -- contra-prova de que NAO sao redundantes: uma
#  amostra adulterada pode passar no AD e ainda assim ser rejeitada pelo
#  DD-SIMCA de pureza.
# =========================================================================

def _pkg_deteccao_pureza(rng):
    """pkg minimo com AD (pooled puro+adulterado) e DD-SIMCA (so' puro),
    ambos ajustados nos MESMOS dados -- o suficiente para exercitar
    `detect_purity`/`applicability_domain_new_samples` sem rodar
    `executar()` inteiro."""
    from sklearn.decomposition import PCA

    from guaraci.chemometric_stats import training_applicability_domain
    from guaraci.classificadores import DDSimca

    p = 10
    centro_puro = rng.normal(size=p)
    # "adulterado": deslocado o suficiente para nao ser puro, mas ainda
    # dentro da nuvem geral (nao um outlier espectral extremo) -- e'
    # exatamente o regime em que AD aceita e pureza deveria rejeitar.
    X_puro = np.array([centro_puro + rng.normal(scale=0.05, size=p)
                        for _ in range(20)])
    X_adult = np.array([centro_puro + rng.normal(loc=1.2, scale=0.15, size=p)
                         for _ in range(20)])
    X_pool = np.vstack([X_puro, X_adult])
    rot_puro = np.array(["Esp_X"] * 20)
    mae_puro = np.array([f"g{i}" for i in range(20)])

    pca = PCA(n_components=3, random_state=0).fit(X_pool)
    treino_ad = training_applicability_domain(pca, X_pool)

    dds = DDSimca(n_components=2, alpha=0.05)
    dds.fit(X_puro, rot_puro, mae_id=mae_puro)
    m = dds._modelos["Esp_X"]

    pkg = {
        "pca": pca, "ad_var_t": treino_ad["var_t"], "ad_h0": treino_ad["h0"],
        "ad_q0": treino_ad["q0"], "ad_Nh": treino_ad["Nh"],
        "ad_Nq": treino_ad["Nq"], "ad_f_crit": treino_ad["f_crit"],
        "ddsimca_por_especie": {
            "Esp_X": {
                "pca": m["pca"], "var_t": m["var_t"], "h0": m["h0"],
                "q0": m["q0"], "Nh": m["Nh"], "Nq": m["Nq"],
                "f_crit": m["f_crit"],
                "n_grupos_calibracao": m["n_grupos_calibracao"],
                "calibrado_por_amostra": m["calibrado_por_amostra"],
            }
        },
    }
    amostra_adulterada = centro_puro + 1.2
    return pkg, amostra_adulterada


def test_ad_e_ddsimca_nao_sao_redundantes():
    """A contra-prova central do fechamento do gap: uma amostra adulterada
    pode passar no dominio de aplicabilidade (ela faz parte do treino do
    AD, pooled) e AINDA ASSIM ser rejeitada pelo DD-SIMCA de pureza
    (ajustado so' nos puros). Se os dois sempre concordassem, DD-SIMCA
    seria peso morto -- este teste prova que nao e'."""
    from guaraci.predicao import detect_purity
    from guaraci.chemometric_stats import applicability_domain_new_samples

    rng = np.random.default_rng(11)
    pkg, amostra = _pkg_deteccao_pureza(rng)

    ad = applicability_domain_new_samples(
        pkg["pca"], amostra.reshape(1, -1), pkg["ad_var_t"], pkg["ad_h0"],
        pkg["ad_q0"], pkg["ad_Nh"], pkg["ad_Nq"], pkg["ad_f_crit"])
    pureza = detect_purity(pkg, "Esp_X", amostra)

    assert bool(ad["dentro_dominio"][0]) is True, (
        "fixture nao produziu o regime esperado: a amostra adulterada "
        "deveria estar DENTRO do dominio de aplicabilidade pooled")
    assert pureza.aceito is False, (
        "DD-SIMCA de pureza deveria REJEITAR a amostra adulterada mesmo "
        "com o AD aceitando -- se nao rejeitou, os dois sinais colapsaram "
        "no mesmo resultado e o segundo Detectar nao adiciona nada")


def test_detect_purity_sem_modelo_para_especie_nao_inventa_resultado():
    from guaraci.predicao import PurityResult, detect_purity
    r = detect_purity({"ddsimca_por_especie": {}}, "Especie_Desconhecida",
                       np.zeros(5))
    assert r == PurityResult(aceito=None, f=None, f_crit=None,
                              n_grupos_calibracao=None, confiavel=False,
                              alpha_nominal=None)


def test_detect_purity_com_poucos_grupos_nao_declara_alpha_confiavel():
    """n_grupos_calibracao pequeno (o caso comum: 1 amostra pura por
    especie) -- o metodo ainda decide aceitar/rejeitar (nao se recusa como
    o conformal), mas `alpha_nominal` fica None (sem lastro para a soma de
    Bonferroni)."""
    from guaraci.predicao import detect_purity

    rng = np.random.default_rng(12)
    pkg, amostra = _pkg_deteccao_pureza(rng)
    pkg["ddsimca_por_especie"]["Esp_X"]["n_grupos_calibracao"] = 1
    r = detect_purity(pkg, "Esp_X", amostra)
    assert r.aceito in (True, False)   # decide mesmo assim
    assert r.confiavel is False
    assert r.alpha_nominal is None


# =========================================================================
#  D4/D7-b: Quantificar nunca forca classe/numero quando Identificar nao
#  valida -- contra-prova ao nivel de INTEGRACAO (test_identificacao.py ja
#  cobre isso isolado; aqui e' contra um pacote real, fim a fim).
# =========================================================================

def test_predict_blind_nunca_forca_classe_nem_numero(pkg_bloco9b):
    pkg, _pasta, _cam = pkg_bloco9b
    wn = np.asarray(pkg["wavenumbers"], dtype=float)
    rng = np.random.default_rng(7)
    X_novos = rng.normal(loc=0.5, scale=0.05, size=(4, len(wn)))

    df, resultados = predict_blind(pkg, X_novos, wn)
    assert len(df) == len(resultados) == 4
    for r in resultados:
        assert r.identificacao.classe_identificada is None
        assert r.quantificacao.teor_estimado is None
        assert r.quantificacao.motivo_bloqueio in (
            "identificacao_desconhecida", "identificacao_ambigua")
        # alpha_total = Bonferroni(alpha do Detectar, alpha do Identificar);
        # Identificar nao tem alpha_alcancavel aqui (NOT_VALIDATED_N1) ->
        # a soma fica indefinida (None), nao um numero inventado.
        assert r.alpha_total is None


def test_predict_blind_detectar_preenchido_quando_ad_disponivel(pkg_bloco9b):
    """Detectar (dominio de aplicabilidade) e' independente do Identificar
    -- continua funcionando (D do Bloco 9a, inalterado) mesmo quando toda
    combinacao de Identificar esta' bloqueada."""
    pkg, _pasta, _cam = pkg_bloco9b
    wn = np.asarray(pkg["wavenumbers"], dtype=float)
    rng = np.random.default_rng(8)
    X_novos = rng.normal(loc=0.5, scale=0.05, size=(3, len(wn)))
    _df, resultados = predict_blind(pkg, X_novos, wn)
    for r in resultados:
        assert r.detectado_no_dominio in (True, False)


# =========================================================================
#  D5: a ressalva de nao-validacao propaga para os 3 lugares.
# =========================================================================

def test_ressalva_aparece_no_model_card(pkg_bloco9b):
    _pkg, pasta, _cam = pkg_bloco9b
    caminho = os.path.join(pasta, pq_nome_relatorios(pasta), "model_card.md")
    assert os.path.isfile(caminho)
    texto = open(caminho, encoding="utf-8").read()
    assert "Identificacao especie x adulterante" in texto
    assert "nao_validado_n1" in texto


def test_addendum_identificacao_nao_forca_numero_de_secao(pkg_bloco9b):
    """Regressao de achado real (revisao 2026-08-25): o addendum de
    Identificacao e' anexado ANTES da regressao (secao 9, condicional) no
    fluxo de executar() -- numera-lo como '## 10.' fixo produzia '10.'
    aparecendo ANTES de '9.' no arquivo (append-only, ordem de escrita
    determina ordem no arquivo). O titulo tem que ficar sem numero."""
    _pkg, pasta, _cam = pkg_bloco9b
    caminho = os.path.join(pasta, pq_nome_relatorios(pasta), "model_card.md")
    texto = open(caminho, encoding="utf-8").read()
    assert "## 10." not in texto
    if "## 9. Addendum" in texto:
        assert (texto.index("Identificacao especie x adulterante")
                < texto.index("## 9. Addendum"))


def test_ressalva_aparece_no_manifesto(pkg_bloco9b):
    _pkg, _pasta, cam_modelo = pkg_bloco9b
    cam_manifesto = cam_modelo + ".manifest.json"
    assert os.path.isfile(cam_manifesto)
    manifesto = json.loads(open(cam_manifesto, encoding="utf-8").read())
    assert "identification_coverage" in manifesto
    cobertura = manifesto["identification_coverage"]
    assert cobertura is not None
    assert cobertura["n_combinacoes"] == len(_pkg["identification_ensemble"])
    assert cobertura["por_status"]["validado"] == 0


# =========================================================================
#  D6: CLI (menu_prediction) estende a predicao existente com as colunas
#  do fluxo cego, sem quebrar o caminho antigo (ver test_predicao.py para
#  o caminho sem ensemble).
# =========================================================================

def test_menu_predicao_cli_inclui_colunas_do_fluxo_cego(
        monkeypatch, tmp_path, pkg_bloco9b):
    import guaraci.guaraci as guaraci_mod

    pkg, _pasta, _cam = pkg_bloco9b
    wn = np.asarray(pkg["wavenumbers"], dtype=float)
    rng = np.random.default_rng(9)
    X_novos = rng.normal(loc=0.5, scale=0.05, size=(3, len(wn)))

    cam_modelo = tmp_path / "modelo_bloco9b.joblib"
    joblib.dump(pkg, cam_modelo)
    df_in = pd.DataFrame(X_novos, columns=[f"{w:.1f}" for w in wn])
    cam_csv = tmp_path / "novos.csv"
    df_in.to_csv(cam_csv, index=False, sep=";")

    respostas = iter([str(cam_modelo), "s", str(cam_csv), "", ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
    guaraci_mod.menu_prediction(guaraci_mod.Config())

    cam_saida = cam_csv.with_name(cam_csv.stem + "_predicao.csv")
    assert cam_saida.is_file()
    df_res = pd.read_csv(cam_saida, sep=";", decimal=",")
    for col in ("classe_identificada", "identificacao_cobertura",
                "identificacao_alpha_alcancavel", "teor_estimado",
                "quantificacao_motivo_bloqueio", "alpha_total"):
        assert col in df_res.columns
    # nunca forca classe/numero (D7-b), tambem visivel na saida do CLI.
    assert df_res["classe_identificada"].isna().all()
    assert df_res["teor_estimado"].isna().all()


def pq_nome_relatorios(_pasta):
    # NOME_RELATORIOS e' uma constante do pacote (Relatorios); import local
    # para nao acoplar o topo do arquivo a um simbolo so' usado aqui.
    from guaraci.pipeline import NOME_RELATORIOS
    return NOME_RELATORIOS
