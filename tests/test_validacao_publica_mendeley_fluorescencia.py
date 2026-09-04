# -*- coding: utf-8 -*-
"""Validacao de Fluorescencia Molecular contra dataset publico Mendeley
`10.17632/thkcz3h6n6.6` -- Passo 142/143 da auditoria das 11 tecnicas
do menu `cli_assistente.TECNICAS` (2026-09-04).

Dataset: 24 azeites de oliva (10 EXTRA, 8 VIRGEN, 6 LAMPANTE -- grau de
qualidade oficial), espectros de emissao de fluorescencia (1024 pontos,
2 LEDs de excitacao, 20 repeticoes tecnicas por amostra/LED = 960
medicoes). Licenca CC BY 4.0.

DECISOES METODOLOGICAS (documentadas, nao escondidas):
  - So' o LED 1 e' usado (dos 2 disponiveis) -- escolha arbitraria de
    UM canal de excitacao, mesmo espirito de `NIR8mm1A.csv` ter sido
    escolhido entre 4 tecnicas no dataset Mendeley ctgg7k4m5g (nao
    testado: combinar os 2 LEDs).
  - As 20 repeticoes tecnicas de cada (amostra, LED) sao MEDIAS antes
    de treinar -- regra 5 da instrucao ("group-aware em qualquer
    validacao nova"): o motor do GUARACI so' suporta agrupamento via
    convencao `mae_id` (formato proprio deste projeto), que nao se
    aplica a um dataset externo. Em vez de forcar um `mae_id`
    artificial (contaminaria o vocabulario, o proprio problema que o
    `perfil_matriz.py` foi criado para evitar -- ver Passo 141),
    colapsar as repeticoes tecnicas em 1 linha por amostra fisica
    elimina o risco de vazamento na raiz (nao sobra unidade repetida
    pra' vazar entre treino/teste). Resultado: n=24 (nao 480).
  - O dataset nao publica um eixo de emissao calibrado em nm -- o
    arquivo `..._background.csv` (medicao de fundo/branco do
    instrumento) NAO e' monotonico e nao serve como eixo (verificado
    por leitura direta em 2026-09-04). `wn_min`/`wn_max` usam indice de
    canal (0-1023) em vez de nm real -- limitacao do dataset, nao do
    GUARACI.

ACHADO REAL (medido em 2026-09-04): balanced_accuracy = 0,383 (CV),
so' um pouco acima do acaso para 3 classes (~0,333) -- sinal fraco,
n=24 e' pequeno (LAMPANTE so' 6 amostras). Testei a hipotese de que
subtrair o espectro de fundo do instrumento (fornecido no proprio
dataset) melhoraria o sinal -- **resultado identico** (0,383 com e sem
subtracao) porque o preset padrao MSC+SG+MC ja remove qualquer offset
constante por amostra antes do subtracao ter chance de importar
(mean-centering torna a subtracao de um offset comum redundante).
Registrado como achado fraco mas real, nao escondido -- nao e' o
mesmo nivel de sinal que MIR/Raman/NIR mostraram no dataset de oleos
por indice de peroxido."""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

#: Piso de sanidade -- "aprendeu um pouco", nao alvo de literatura. Chance
#: com 3 classes ~0.333; o valor medido (0.383) fica logo acima. Piso
#: deliberadamente baixo (n=24 e' pequeno, mesma disciplina de floors
#: pequenos usados em outros datasets publicos pequenos deste projeto).
BAL_ACC_MINIMA = 0.35
N_MINIMO_POR_CLASSE = 5


def _pasta_mendeley_fluor():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "mendeley_thkcz3h6n6"
    return pasta if (pasta / "Fluorescence_olive_oil_dataset.csv").is_file() else None


requer_mendeley_fluorescencia = pytest.mark.skipif(
    _pasta_mendeley_fluor() is None,
    reason=("Fluorescence_olive_oil_dataset.csv (Mendeley thkcz3h6n6) "
            "ausente. Baixe com 'python scripts/download_datasets/"
            "baixar_mendeley_fluorescencia_oleo.py' e aponte "
            "GUARACI_DATASETS_DIR para a pasta que contem "
            "mendeley_thkcz3h6n6/."))


def _carregar_bruto():
    pasta = _pasta_mendeley_fluor()
    df = pd.read_csv(pasta / "Fluorescence_olive_oil_dataset.csv")
    df = df[df["Led"] == 1].copy()
    df["Data"] = df["Data"].apply(ast.literal_eval)
    matriz = np.vstack(df["Data"].to_numpy())
    cols = [str(i) for i in range(matriz.shape[1])]
    espectros = pd.DataFrame(matriz, columns=cols, index=df.index)
    junto = pd.concat([df[["Sample", "Quality"]], espectros], axis=1)
    media_por_amostra = junto.groupby(["Sample", "Quality"], as_index=False)[cols].mean()
    return media_por_amostra, cols


@requer_mendeley_fluorescencia
@pytest.mark.slow
def test_fluorescencia_classifica_grau_qualidade_acima_do_acaso(pq, tmp_path):
    """3 classes de qualidade oficial (EXTRA/VIRGEN/LAMPANTE), n=24
    apos colapsar as 20 repeticoes tecnicas por amostra/LED (ver
    docstring do modulo)."""
    from conftest import achar_pastas_run

    media, cols = _carregar_bruto()
    contagem = media["Quality"].value_counts()
    assert (contagem >= N_MINIMO_POR_CLASSE).all(), (
        f"alguma classe abaixo do minimo de {N_MINIMO_POR_CLASSE} "
        f"amostras -- premissa do dataset mudou:\n{contagem}")

    sub = media.rename(columns={"Quality": "classe"}).drop(columns=["Sample"])
    csv = tmp_path / "fluorescencia_classificacao.csv"
    sub.to_csv(csv, index=False)

    cfg = pq.Config(
        mode="csv", csv_file=str(csv),
        class_column="classe", conc_column="",
        matrix_profile="generico", wn_min=0.0, wn_max=float(len(cols) - 1),
        objective="classificacao", level="N1",
        output_root_folder=str(tmp_path / "saida"),
        group_by_mae_id=False, show_plots=False,
        run_benchmark=False, run_monte_carlo=False, run_shap=False,
        run_wold=False, run_cv_anova=False, run_opls=False,
        run_ddsimca=False, executar_etapa4=False,
        n_permutations=20, frac_holdout=0.2, seed=0, max_lvs=8,
    )
    pq.executar(cfg)

    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs, "executar() nao criou saida"
    resumo = (Path(runs[0]) / pq.NOME_RELATORIOS / "resumo_modelo.txt").read_text(
        encoding="utf-8", errors="replace")
    achado = re.search(r"Balanced accuracy\s*\.*:\s*([\d.]+)", resumo)
    assert achado, f"Balanced accuracy nao encontrada no resumo:\n{resumo[:600]}"
    bal_acc = float(achado.group(1))
    assert bal_acc >= BAL_ACC_MINIMA, (
        f"balanced_accuracy={bal_acc:.3f} abaixo do piso {BAL_ACC_MINIMA} "
        f"-- modelo nao aprendeu nada com 3 graus de qualidade deste "
        f"dataset publico, sinal de bug (nao de dificuldade esperada).")
