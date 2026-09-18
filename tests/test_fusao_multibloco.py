# -*- coding: utf-8 -*-
"""Testes de guaraci.fusao_multibloco -- fechamento do Grupo 2 (prova de
conceito de fusao multibloco, ver docs/ESCOPO_FUSAO_MULTIBLOCO_E_MSPC.md).

Duas camadas: (1) testes SINTETICOS e rapidos da mecanica pura
(`build_multiblock_dataset`/`fit_evaluate_pls_regression`) -- guarda
contra desalinhamento, ausencia de vazamento cal->val, formato do
resultado; (2) UMA prova de conceito real contra o par Mendeley
NIR8mm+MIR (mesmas 100 garrafas de oleo medidas nas duas tecnicas, ja
usadas em `test_validacao_publica_mendeley.py`/`..._mir_raman.py`),
pulada sem o dataset baixado, mesma convencao das outras validacoes
publicas.

ACHADO REAL (medido nesta rodada, honesto -- nao escondido): fusao de
BAIXO NIVEL (concatenacao apos pre-processamento separado, sem
block-scaling) NAO melhora a regressao do indice de peroxido sobre usar
MIR sozinho neste dataset -- consistente em 3 seeds independentes
(0, 1, 2): RMSEP da fusao fica sempre pior (maior) que o do MIR sozinho,
puxado para baixo pelo bloco NIR (11512 variaveis, sinal mais fraco,
R2val ja NEGATIVO sozinho -- ver `test_validacao_publica_mendeley.py`)
dominando a decomposicao PLS por simples CONTAGEM de variaveis (3.4x
mais colunas que o MIR). Testado tambem com block-scaling (normalizacao
de cada bloco pela norma de Frobenius do lado de calibracao antes de
concatenar, fora do modulo publicado -- ver historico desta sessao): nao
mudou o resultado de forma consistente, entao NAO foi adotado como
feature do modulo (manteria o escopo "nivel 1, nenhum algoritmo novo" do
documento de escopo, e adicionar so' por nao ter ajudado seria
complexidade sem beneficio demonstrado). Reportado honestamente como
NAO-MELHORA, nao escondido nem forcado a parecer sucesso -- mesma
disciplina do resto do projeto (ver docs/VALIDACAO_PUBLICA.md).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from guaraci.config import Config
from guaraci.fusao_multibloco import (
    BlockMismatchError,
    build_multiblock_dataset,
    fit_evaluate_pls_regression,
)

# ── Mecanica sintetica (rapida, sem dataset externo) ────────────────────


def test_build_multiblock_dataset_recusa_blocos_desalinhados():
    rng = np.random.default_rng(0)
    bloco_a = rng.normal(size=(20, 5))
    bloco_b = rng.normal(size=(19, 4))   # 1 amostra a menos -- deliberado
    with pytest.raises(BlockMismatchError, match="n_amostras diferente"):
        build_multiblock_dataset({"a": bloco_a, "b": bloco_b})


def test_build_multiblock_dataset_concatena_na_ordem_e_reporta_fatias():
    rng = np.random.default_rng(1)
    bloco_a = rng.normal(size=(30, 6))
    bloco_b = rng.normal(size=(30, 4))
    # default_preprocessing="mc" (so' mean-centering) -- o preset default
    # (msc_sg_mc) exige sg_window (25) <= n_variaveis, que estes blocos
    # sinteticos pequenos (6/4 colunas) nao satisfazem; irrelevante para
    # o que este teste verifica (concatenacao/fatias, nao o preset).
    cfg = {"a": Config(default_preprocessing="mc"),
           "b": Config(default_preprocessing="mc")}
    X_fundida, preps, fatias = build_multiblock_dataset(
        {"a": bloco_a, "b": bloco_b}, cfg)
    assert X_fundida.shape == (30, 10)
    assert fatias["a"] == (0, 6)
    assert fatias["b"] == (6, 10)
    assert set(preps.keys()) == {"a", "b"}


def test_build_multiblock_dataset_ajusta_preprocessador_so_no_indices_ajuste():
    """MSC/mean-centering sao stateful (media de referencia vem do fit) --
    ajustar so' com `indices_ajuste` (calibracao) tem que dar um resultado
    DIFERENTE de ajustar com todas as amostras, quando cal e val tem
    medias bem diferentes -- prova de que nao ha' vazamento cal->val."""
    rng = np.random.default_rng(2)
    bloco = np.vstack([
        rng.normal(loc=0.0, scale=0.1, size=(20, 8)),    # "calibracao"
        rng.normal(loc=50.0, scale=0.1, size=(10, 8)),   # "validacao" (media MUITO diferente)
    ])
    idx_cal = np.arange(20)

    cfg = {"x": Config(default_preprocessing="mc")}
    _, preps_cal, _ = build_multiblock_dataset(
        {"x": bloco}, cfg, indices_ajuste=idx_cal)
    _, preps_tudo, _ = build_multiblock_dataset({"x": bloco}, cfg)

    media_cal = preps_cal["x"].named_steps["mc"].mean_
    media_tudo = preps_tudo["x"].named_steps["mc"].mean_
    assert not np.allclose(media_cal, media_tudo), (
        "ajustar com indices_ajuste=calibracao deveria dar uma media "
        "diferente de ajustar com o dataset inteiro -- se bater, o "
        "pre-processador esta' vazando informacao de validacao")


def test_fit_evaluate_pls_regression_recupera_sinal_linear_simples():
    """Contra-prova minima: com y linear numa unica variavel latente
    (sinal facil, sem ruido de dominio real), a regressao tem que
    recuperar R2 alto e RMSEP baixo -- se isto falhar, a mecanica de
    ajuste/avaliacao esta' quebrada, independente do dataset real."""
    rng = np.random.default_rng(3)
    n = 200
    latente = rng.normal(size=n)
    X = latente[:, None] * rng.normal(loc=1.0, scale=0.05, size=(1, 20)) \
        + rng.normal(scale=0.01, size=(n, 20))
    y = 3.0 * latente + rng.normal(scale=0.05, size=n)

    idx_cal, idx_val = train_test_split(
        np.arange(n), test_size=0.25, random_state=0)
    r = fit_evaluate_pls_regression(
        X[idx_cal], y[idx_cal], X[idx_val], y[idx_val], max_lvs=5, seed=0)
    assert r.n_cal == 150 and r.n_val == 50
    assert r.r2val > 0.9, f"R2val={r.r2val:.3f} baixo demais p/ sinal facil"
    assert r.rmsep < 0.5


# ── Prova de conceito real: Mendeley NIR8mm + MIR ───────────────────────


def _pasta_mendeley():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "mendeley_ctgg7k4m5g"
    if (pasta / "NIR8mm1A.csv").is_file() and (pasta / "MIR1A.csv").is_file():
        return pasta
    return None


requer_mendeley_nir_mir = pytest.mark.skipif(
    _pasta_mendeley() is None,
    reason=("NIR8mm1A.csv/MIR1A.csv do dataset Mendeley ctgg7k4m5g "
            "ausentes. Baixe com "
            "'python scripts/download_datasets/baixar_mendeley_oleos.py' "
            "e aponte GUARACI_DATASETS_DIR para a pasta que contem "
            "mendeley_ctgg7k4m5g/."))


def _carregar_blocos():
    pasta = _pasta_mendeley()
    nir = pd.read_csv(pasta / "NIR8mm1A.csv")
    mir = pd.read_csv(pasta / "MIR1A.csv")
    return nir, mir


@requer_mendeley_nir_mir
def test_nir8mm_e_mir_estao_alinhados_linha_a_linha():
    """Bloqueante do PoC (regra de reporte da instrucao): confirma por
    inspecao direta do dado real, nao presume, que as 100 linhas de
    NIR8mm1A.csv e MIR1A.csv sao a MESMA amostra fisica na mesma posicao
    -- unica base que sustenta concatenar linha-a-linha sem misturar
    amostras."""
    nir, mir = _carregar_blocos()
    assert len(nir) == len(mir) == 100
    assert (nir["Class"].values == mir["Class"].values).all()
    assert np.allclose(nir["PeroxideValue"].values, mir["PeroxideValue"].values)


@requer_mendeley_nir_mir
@pytest.mark.slow
def test_fusao_de_baixo_nivel_nao_melhora_sobre_mir_sozinho():
    """PROVA DE CONCEITO (Grupo 2): compara NIR-so', MIR-so' e a fusao
    (concatenacao apos pre-processamento separado) no MESMO split
    cal/val, reportando os 3 numeros lado a lado -- nunca so' o da fusao
    (mesma disciplina do resto do projeto). Ver ACHADO REAL no docstring
    do modulo: reproduzido em 3 seeds independentes, a fusao NAO supera
    o MIR sozinho -- o floor abaixo e' o resultado MEDIDO, no' um alvo de
    literatura."""
    nir, mir = _carregar_blocos()
    y = np.log10(nir["PeroxideValue"].values.astype(float))
    X_nir = nir.drop(columns=["Class", "PeroxideValue"]).values.astype(float)
    X_mir = mir.drop(columns=["Class", "PeroxideValue"]).values.astype(float)

    resultados = {"nir": [], "mir": [], "fusao": []}
    for seed in (0, 1, 2):
        idx = np.arange(len(y))
        idx_cal, idx_val = train_test_split(
            idx, test_size=0.25, random_state=seed)

        cfg_por_bloco = {"nir": Config(), "mir": Config()}
        X_fundida, preps, _fatias = build_multiblock_dataset(
            {"nir": X_nir, "mir": X_mir}, cfg_por_bloco,
            indices_ajuste=idx_cal)
        X_nir_proc = np.asarray(preps["nir"].transform(X_nir))
        X_mir_proc = np.asarray(preps["mir"].transform(X_mir))

        r_nir = fit_evaluate_pls_regression(
            X_nir_proc[idx_cal], y[idx_cal], X_nir_proc[idx_val], y[idx_val],
            max_lvs=15, seed=seed)
        r_mir = fit_evaluate_pls_regression(
            X_mir_proc[idx_cal], y[idx_cal], X_mir_proc[idx_val], y[idx_val],
            max_lvs=15, seed=seed)
        r_fusao = fit_evaluate_pls_regression(
            X_fundida[idx_cal], y[idx_cal], X_fundida[idx_val], y[idx_val],
            max_lvs=15, seed=seed)

        resultados["nir"].append(r_nir)
        resultados["mir"].append(r_mir)
        resultados["fusao"].append(r_fusao)

    rmsep_mir = [r.rmsep for r in resultados["mir"]]
    rmsep_fusao = [r.rmsep for r in resultados["fusao"]]
    print(f"\n[Fusao multibloco] RMSEP por seed -- "
          f"NIR={[round(r.rmsep, 3) for r in resultados['nir']]}  "
          f"MIR={[round(r, 3) for r in rmsep_mir]}  "
          f"FUSAO={[round(r, 3) for r in rmsep_fusao]}")

    # Contra-prova honesta: o MIR sozinho ja e' finito/positivo (checagem
    # de sanidade), e a fusao, nas 3 seeds medidas, NUNCA bate o MIR
    # sozinho -- e' o achado real, nao um piso arbitrario de literatura.
    assert all(np.isfinite(r) and r > 0 for r in rmsep_mir + rmsep_fusao)
    assert all(f >= m for f, m in zip(rmsep_fusao, rmsep_mir)), (
        "Fusao superou o MIR sozinho nesta rodada -- ACHADO MUDOU em "
        "relacao a' medicao registrada no docstring do modulo; investigar "
        "antes de aceitar (retratar a nota anterior se confirmado).")
