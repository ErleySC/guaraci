# -*- coding: utf-8 -*-
"""Validação do MSPC (monitoramento em linha, Bloco 13b) -- Grupo 2,
Camada 2: deriva REAL já documentada (dataset público Corn, mesmo usado
para validar PDS/DS em `test_validacao_publica.py`).

Por que o Corn serve para isto: as MESMAS 80 amostras físicas de milho
foram medidas em 3 espectrômetros diferentes (m5/mp5/mp6 -- FOSS
NIRSystems 5000/6500, ver `docs/VALIDACAO_PUBLICA.md`). A troca de
instrumento (m5 -> mp5) já tem diferença espectral REAL e MEDIDA:
aplicar um modelo calibrado em m5 a espectros mp5 sem correção degrada
o RMSEP de proteína de ~0,15 para ~0,5-0,9 (`test_transferencia_de_
calibracao_reduz_erro_entre_instrumentos_do_corn`,
`test_portao_correcao_sinal_reproduz_pds_e_RETRATA_ds_no_corn`) -- é
exatamente o tipo de "deriva" que o MSPC deveria pegar, sem precisar
inventar nada sintético.

Escolha de `n_components=2` para o PCA do Domínio de Aplicabilidade:
NÃO arbitrária -- medido diretamente (ver histórico desta sessão): a
1ª componente principal do m5 sozinha já explica 99,3% da variância,
a 2ª leva a 99,86%. Usar mais componentes (testado: 3/5/10) infla a
taxa de FALSO alarme na fase em controle (mesmo instrumento, deveria
ficar estável) para 20-30% -- overfitting do PCA num regime n≪p (40-60
amostras de calibração × 700 canais), não uma falha do MSPC em si.
`n_components=2` é a escolha correta pela variância explicada, não uma
escolha feita para o teste passar.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA

from guaraci.chemometric_stats import (
    ad_rejection_rate_cv,
    applicability_domain_new_samples,
    training_applicability_domain,
)
from guaraci.sentinela_deriva import SentinelState, check_drift


def _caminho_corn():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    caminho = Path(raiz) / "corn.mat"
    return caminho if caminho.is_file() else None


requer_corn = pytest.mark.skipif(
    _caminho_corn() is None,
    reason=("dataset publico Corn ausente. Baixe corn.mat de "
            "https://eigenvector.com/data/Corn/ e aponte "
            "GUARACI_DATASETS_DIR para a pasta que o contem."))


def _carregar_instrumentos():
    import scipy.io as sio
    m = sio.loadmat(str(_caminho_corn()))
    X_m5 = np.asarray(m["m5spec"]["data"][0, 0], dtype=float)
    X_mp5 = np.asarray(m["mp5spec"]["data"][0, 0], dtype=float)
    assert X_m5.shape == (80, 700) and X_mp5.shape == (80, 700)
    return X_m5, X_mp5


def _rodar_cenario(X_m5, X_mp5, seed: int, tam_lote: int = 8,
                    n_cal: int = 40, n_components: int = 2,
                    com_referencia: bool = True):
    """Calibra o Dominio de Aplicabilidade em `n_cal` amostras m5, depois
    processa o RESTANTE em lotes -- primeiro do proprio m5 (fase 'em
    controle'), depois as MESMAS amostras fisicas medidas em mp5 (fase
    'deriva real'). Retorna (alarme_disparou_na_fase_1,
    lote_de_deteccao_na_fase_2_ou_None).

    `com_referencia=True` (padrao): o sentinela testa contra a taxa de
    rejeicao em validacao cruzada da propria calibracao (R2, Passo 220 --
    o que o modelo salvo carrega). `False`: teste legado contra 5%."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(80)
    idx_cal, idx_resto = idx[:n_cal], idx[n_cal:]

    pca = PCA(n_components=n_components).fit(X_m5[idx_cal])
    art = training_applicability_domain(pca, X_m5[idx_cal], alpha=0.05)

    def _dentro_dominio(X, idxs):
        return applicability_domain_new_samples(
            pca, X[idxs], art["var_t"], art["h0"], art["q0"],
            art["Nh"], art["Nq"], art["f_crit"])["dentro_dominio"]

    estado = SentinelState(alpha_nominal=0.05)
    if com_referencia:
        estado.ref_rejeitadas, estado.ref_n = ad_rejection_rate_cv(
            X_m5[idx_cal], n_components)
    alarme_fase1 = False
    for i in range(0, len(idx_resto), tam_lote):
        lote = idx_resto[i:i + tam_lote]
        for b in _dentro_dominio(X_m5, lote):
            estado.registrar(bool(b))
        if estado.n >= 19 and check_drift(estado).alerta:
            alarme_fase1 = True

    lote_deteccao = None
    for j, i in enumerate(range(0, len(idx_resto), tam_lote)):
        lote = idx_resto[i:i + tam_lote]
        for b in _dentro_dominio(X_mp5, lote):
            estado.registrar(bool(b))
        if check_drift(estado).alerta and lote_deteccao is None:
            lote_deteccao = j

    return alarme_fase1, lote_deteccao


@requer_corn
def test_mspc_detecta_troca_de_instrumento_no_corn():
    """Prova principal da Camada 2: calibrado em m5, aplicado
    sequencialmente a m5 (deveria continuar em controle) e depois a mp5
    -- MESMAS amostras fisicas, instrumento diferente, degradacao de
    RMSEP ja documentada -- tem que disparar o alerta."""
    X_m5, X_mp5 = _carregar_instrumentos()
    _alarme_fase1, lote_deteccao = _rodar_cenario(X_m5, X_mp5, seed=0)
    assert lote_deteccao is not None, (
        "MSPC nao detectou a troca de instrumento m5->mp5 -- deriva "
        "real e documentada (degradacao de RMSEP ja medida em "
        "test_validacao_publica.py) nao foi pega pelo sentinela")
    assert lote_deteccao <= 1, (
        f"deteccao no lote {lote_deteccao} -- esperava deteccao em ate' "
        "1 lote (8 amostras) dado o tamanho real da diferenca "
        "espectral entre m5 e mp5")


@requer_corn
@pytest.mark.slow
def test_mspc_corn_r2_deteccao_em_ate_1_lote_e_falso_alarme_calibrado_30_seeds():
    """Contra-prova replicada do MSPC com R2 aplicado (Passo 220): 30
    splits aleatorios independentes de calibracao/holdout.

    Historico (medido, nao apagado): o teste LEGADO (binomial contra 5%)
    dava ~21% de falso alarme em controle (1500 splits; a leitura anterior
    de "~7-10%" era 3/30 seeds e foi RETRATADA no Passo 219). Causa raiz:
    a calibracao com n_cal=40 rejeita ~6,5% em controle e esse valor varia
    entre calibracoes (efeito de estimacao de parametros da Fase I,
    Jensen et al. 2006, DOI 10.1080/00224065.2006.11918623) -- a hipotese
    nula "taxa = 5%" era falsa. R2 compara contra a taxa da PROPRIA
    calibracao em validacao cruzada. Medido (`scripts/medicoes/
    medir_mspc_r2_corn.py`, codigo de producao): ver `docs/
    VALIDACAO_PUBLICA.md` §11 para os numeros com 300 splits.

    Aqui, com 30 seeds: deteccao 100% em ate' 1-2 lotes, e falso
    alarme <= 15% (<= 4/30; sob os ~3% medidos, P(>=5/30) < 0,3%) -- teto
    de regressao, nao afirmacao de que vale exatamente 3%.
    """
    X_m5, X_mp5 = _carregar_instrumentos()
    n_seeds = 30
    n_falso_alarme = 0
    n_detectado = 0
    atrasos = []
    for seed in range(n_seeds):
        alarme_fase1, lote_deteccao = _rodar_cenario(X_m5, X_mp5, seed=seed)
        if alarme_fase1:
            n_falso_alarme += 1
        if lote_deteccao is not None:
            n_detectado += 1
            atrasos.append(lote_deteccao)

    taxa_deteccao = n_detectado / n_seeds
    taxa_falso_alarme = n_falso_alarme / n_seeds
    print(f"\n[MSPC/Corn R2] deteccao={taxa_deteccao:.2f} "
          f"falso_alarme_fase1={taxa_falso_alarme:.2f} "
          f"atraso_medio(lotes)={np.mean(atrasos) if atrasos else float('nan'):.2f}")

    assert taxa_deteccao == 1.0, (
        f"deteccao da troca de instrumento caiu para {taxa_deteccao:.2f} "
        "-- deveria ser 100% dado o tamanho real da diferenca espectral")
    # R2 detecta no 1o OU no 2o lote pos-troca (medido em 300 splits: 177 no
    # lote 0, 123 no lote 1, nenhum depois; o teste legado era sempre no
    # lote 0). E' o custo medido de comparar contra a referencia de CV em
    # vez do 5% nominal -- ver docs/VALIDACAO_PUBLICA.md.
    assert max(atrasos) <= 2 and np.mean(atrasos) < 1.0, (
        f"atraso de deteccao com R2 subiu (max={max(atrasos)}, "
        f"medio={np.mean(atrasos):.2f} lotes; esperado <= 1 lote)")
    assert taxa_falso_alarme <= 0.15, (
        f"falso alarme {taxa_falso_alarme:.2f} com R2 -- esperado ~3% "
        "(medido em 300 splits); investigar antes de aceitar")


@requer_corn
def test_mspc_corn_teste_legado_e_r2_no_mesmo_split_o_legado_alarma_mais():
    """Mesmo split, mesma sequencia de amostras: R2 nunca alarma MENOS em
    deriva real (deteccao imediata nos dois) e o legado tem MAIS falso
    alarme em controle somado sobre 12 splits (soma, nao por split, para
    nao depender de um sorteio)."""
    X_m5, X_mp5 = _carregar_instrumentos()
    fa_leg = fa_r2 = 0
    for seed in range(12):
        f_l, d_l = _rodar_cenario(X_m5, X_mp5, seed, com_referencia=False)
        f_r, d_r = _rodar_cenario(X_m5, X_mp5, seed, com_referencia=True)
        assert d_l == 0 and d_r is not None and d_r <= 1
        fa_leg += f_l
        fa_r2 += f_r
    assert fa_r2 <= fa_leg
