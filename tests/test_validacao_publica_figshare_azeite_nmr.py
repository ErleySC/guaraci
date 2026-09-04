# -*- coding: utf-8 -*-
"""Validacao RMN contra dataset publico Figshare `10.6084/m9.figshare.
4307804` (Lamanna et al. 2017) -- Passo 142/143 da auditoria das 11
tecnicas do menu `cli_assistente.TECNICAS` (2026-09-04).

Dataset: 97 azeites de oliva da regiao Abruzzo/Italia, perfil 1H-NMR
JA BINADO pelos autores originais em 125 variaveis de deslocamento
quimico (ppm) -- zero NaN, tabela pronta pro motor CSV generico do
GUARACI sem nenhum pre-processamento de binning novo (o dataset ja'
resolve essa parte, ver Passo 144 sobre binning de RMN). Alvo:
provincia de origem, codificada no PREFIXO do `Sample_ID` ("pe" =
Pescara, 50 amostras; "te" = Teramo, 47 -- confirmado por leitura
direta das coordenadas lat/long de cada grupo). Licenca **CC0**
(dominio publico).

LIMITACAO DE ACESSO (nao do dado em si): o Figshare bloqueia download
automatizado deste arquivo com um desafio de bot da AWS WAF -- nao e'
CAPTCHA (a regra permanente contra burlar CAPTCHA nao se aplica aqui
tecnicamente), mas `scripts/download_datasets/baixar_figshare_azeite_
nmr.py` tenta o download automatico e, se falhar, orienta o download
MANUAL com instrucoes claras (ver docstring do script). O arquivo
usado nesta auditoria (2026-09-04) foi obtido assim -- SHA256
verificado bate com o pinado no script.

ACHADO REAL (medido em 2026-09-04, NAO escondido): ao contrario de
NIR/MIR/Raman/Fluorescencia (que aprenderam algo acima do acaso), a
classificacao por provincia com o motor GENERICO do GUARACI (PLS-DA,
125 variaveis, `matrix_profile="generico"`) fica em
**balanced_accuracy = 0.500 -- EXATAMENTE o nivel do acaso para um
problema binario** -- testado com 4 presets de pre-processamento
diferentes (msc_sg_mc/snv_mc/autoscaling/sg_mc), todos deram o MESMO
0.500. Hipotese razoavel (mesma disciplina do achado `unripe` do HSI):
o proprio GUARACI reporta que so' ~32% das 125 variaveis carregam
sinal acima do ruido (SNR>=3, aviso `[AVISO] Faixa espectral`) -- o
artigo original NAO usou PLS-DA ingenuo sobre todas as variaveis; usou
um teste geoestatistico (I de Moran) para SELECIONAR quais variaveis
tem autocorrelacao espacial antes de rodar LDA só' nelas. A separacao
por provincia provavelmente existe em poucas variaveis especificas,
nao no espectro inteiro -- reproduzir os 99% do artigo exigiria
implementar selecao de variavel por geoestatistica, fora do escopo
deste passo. Registrado como NEGATIVO, nao escondido: RMN classifica
no acaso com o motor generico atual do GUARACI para este dataset."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _pasta_figshare():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "figshare_4307804"
    nome = "oliando2012Frantoio_data_Rev_New.csv"
    return pasta if (pasta / nome).is_file() else None


requer_figshare_nmr = pytest.mark.skipif(
    _pasta_figshare() is None,
    reason=("oliando2012Frantoio_data_Rev_New.csv (Figshare 4307804) "
            "ausente. Baixe com "
            "'python scripts/download_datasets/baixar_figshare_azeite_"
            "nmr.py' (download automatico pode falhar por bloqueio do "
            "Figshare -- ver docstring do script para a instrucao "
            "manual) e aponte GUARACI_DATASETS_DIR para a pasta que "
            "contem figshare_4307804/."))


def _carregar_bruto():
    pasta = _pasta_figshare()
    df = pd.read_csv(pasta / "oliando2012Frantoio_data_Rev_New.csv")
    assert df.isna().sum().sum() == 0, (
        "dataset tinha NaN -- premissa do docstring deste arquivo mudou, "
        "revisar antes de prosseguir.")
    prefixo = df["Sample_ID"].str.extract(r"^([a-zA-Z]+)")[0]
    classe = prefixo.map({"pe": "Pescara", "te": "Teramo"})
    assert classe.isna().sum() == 0, (
        "algum Sample_ID nao comecava com 'pe'/'te' -- premissa de "
        "prefixo->provincia mudou.")
    df = df.drop(columns=["Sample_ID", "long", "lat"]).copy()
    df["classe"] = classe
    return df


@requer_figshare_nmr
@pytest.mark.slow
def test_nmr_roda_sem_excecao_mas_nao_separa_provincia_com_motor_generico(pq, tmp_path):
    """NAO e' um teste de "classificacao funciona" -- e' o oposto,
    registrado com a mesma disciplina do achado `unripe` do HSI. Ver
    ACHADO REAL no docstring do modulo para a hipotese de por que."""
    from conftest import achar_pastas_run

    sub = _carregar_bruto()
    cols_espectrais = [c for c in sub.columns if c != "classe"]
    wn_min = min(float(c) for c in cols_espectrais)
    wn_max = max(float(c) for c in cols_espectrais)

    csv = tmp_path / "nmr_classificacao.csv"
    sub.to_csv(csv, index=False)

    cfg = pq.Config(
        mode="csv", csv_file=str(csv),
        class_column="classe", conc_column="",
        matrix_profile="generico", wn_min=wn_min, wn_max=wn_max,
        objective="classificacao", level="N1",
        output_root_folder=str(tmp_path / "saida"),
        group_by_mae_id=False, show_plots=False,
        run_benchmark=False, run_monte_carlo=False, run_shap=False,
        run_wold=False, run_cv_anova=False, run_opls=False,
        run_ddsimca=False, executar_etapa4=False,
        n_permutations=20, frac_holdout=0.2, seed=0, max_lvs=10,
    )
    pq.executar(cfg)   # nao deve lancar excecao -- e' o unico requisito real

    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs, "executar() nao criou saida"
    resumo = (Path(runs[0]) / pq.NOME_RELATORIOS / "resumo_modelo.txt").read_text(
        encoding="utf-8", errors="replace")
    achado = re.search(r"Balanced accuracy\s*\.*:\s*([\d.]+)", resumo)
    assert achado, f"Balanced accuracy nao encontrada no resumo:\n{resumo[:600]}"
    bal_acc = float(achado.group(1))
    assert np.isfinite(bal_acc) and 0.0 <= bal_acc <= 1.0, (
        f"balanced_accuracy={bal_acc} fora do intervalo valido -- "
        f"sinal de bug (nao de dificuldade esperada).")
    # Sem gate de "aprendeu algo": o achado real medido e' 0.500 (acaso),
    # documentado no docstring do modulo. Um gate `> 0.6` aqui seria
    # inventar sucesso que a medicao nao mostra.
