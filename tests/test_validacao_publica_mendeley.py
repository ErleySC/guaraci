# -*- coding: utf-8 -*-
"""Validacao contra dataset PUBLICO Mendeley `10.17632/ctgg7k4m5g.2`
(Ottaway et al. 2021/2025) -- mesmo espirito de `test_validacao_publica.py`
(Corn), arquivo separado porque a integracao e' de outro dataset/matriz.

Dataset: 100 garrafas de 19 tipos de oleo comestivel (azeite extra-virgem/
leve/puro, abacate, amendoim, milho, semente de uva, canola, girassol,
varios blends, entre outros -- ver OilClassKey.csv), envelhecidas
naturalmente (5-7 anos), medidas por NIR (varios caminhos opticos)/MIR/
Raman. Licenca CC BY 4.0. Fonte: https://data.mendeley.com/datasets/ctgg7k4m5g/2

Os arquivos NAO sao versionados (licenca de terceiro, mesma politica do
Corn -- ver docs/VALIDACAO_PUBLICA.md e `datasets/README.md`). Baixados
por `scripts/download_datasets/baixar_mendeley_oleos.py` (SHA256+tamanho
pinados) para `$GUARACI_DATASETS_DIR/mendeley_ctgg7k4m5g/`; ausentes, o
teste PULA com instrucao de como obter -- nunca falha por falta de dado
nem baixa nada por conta propria dentro da suite.

Usamos so' o arquivo NIR8mm1A.csv (caminho optico de 8mm, a tecnica com o
RMSEP publicado mais confiavel entre as 4 testadas no artigo original --
o proprio artigo sinaliza o RMSEP do Raman como correlacao por acaso).
Os outros 8 arquivos do dataset (MIR, Raman, NIR 24mm/2mm) nao sao
baixados nem usados aqui.

ACHADO REAL (2026-08-27, nao escondido): o RMSEP publicado (4.9,
indice de peroxido) NAO reproduz com holdout independente usando os
presets padrao do GUARACI -- R2val fica NEGATIVO tanto no valor bruto
(RMSEP=25.9) quanto em log10 (RMSEP=0.49 em unidades log, R2cal=0.83 mas
R2val=-0.53). Com n=100 amostras e ~11500 canais NIR colineares, um
holdout de 25 amostras e' um teste de alta variancia; o artigo original
nao detalha o suficiente (protocolo exato de CV/holdout, remocao de
outliers) para reproduzir s2s numero com confianca -- tentar mais
combinacoes ate' bater com 4.9 seria ajustar o teste ao numero
publicado, nao reproduzi-lo de verdade. Por isso o teste de regressao
abaixo e' um SANITY CHECK (roda sem excecao, ajuste de calibracao
razoavel), NAO um gate contra o RMSEP publicado -- decisao do usuario,
registrada aqui e em docs/VALIDACAO_PUBLICA.md.

NaN: uma linha de Raman1A.csv (nao usado aqui) tem NaN em todas as
colunas espectrais -- amostra sem aquela medicao especifica tomada.
NIR8mm1A.csv (usado neste arquivo) tem ZERO NaN, verificado por leitura
direta -- nao ha' NaN para tratar no arquivo que este teste consome.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

#: Balanced accuracy minima para a classificacao de especies nao ser
#: degenerada (pior que sempre prever a classe majoritaria). NAO e' um
#: alvo de literatura -- e' so' o piso "o modelo aprendeu algo", dado que
#: n=62 amostras de treino / 8 classes / ~11500 variaveis e' um regime
#: genuinamente dificil (balanced_accuracy medida em 2026-08-27: CV
#: 0.35, holdout 0.475 -- ambos acima do chance ~0.125, mas modestos).
BAL_ACC_MINIMA = 0.25

#: Classes com pelo menos este numero de amostras entram na classificacao
#: -- classes com 1-2 amostras nao sustentam nenhuma divisao treino/teste
#: honesta (GroupKFold/StratifiedKFold exigem >=2 por classe; abaixo
#: disso e' fabricar uma metrica sem base estatistica real).
N_MINIMO_POR_CLASSE = 5


def _pasta_mendeley():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "mendeley_ctgg7k4m5g"
    return pasta if (pasta / "NIR8mm1A.csv").is_file() else None


requer_mendeley = pytest.mark.skipif(
    _pasta_mendeley() is None,
    reason=("dataset publico Mendeley ctgg7k4m5g ausente. Baixe com "
            "'python scripts/download_datasets/baixar_mendeley_oleos.py' "
            "e aponte GUARACI_DATASETS_DIR para a pasta que contem "
            "mendeley_ctgg7k4m5g/."))


def _carregar_bruto():
    pasta = _pasta_mendeley()
    df = pd.read_csv(pasta / "NIR8mm1A.csv")
    chave = pd.read_csv(pasta / "OilClassKey.csv")
    mapa_nomes = dict(zip(chave["Class Number"], chave["Class Name"]))
    assert df.isna().sum().sum() == 0, (
        "NIR8mm1A.csv tinha NaN -- premissa do docstring deste arquivo "
        "mudou, revisar antes de prosseguir (ver secao NaN acima).")
    return df, mapa_nomes


@requer_mendeley
@pytest.mark.slow
def test_multimatriz_declara_perfil_correto_e_classifica_acima_do_acaso(
        pq, tmp_path):
    """Prova do requisito multimatriz com este dataset: perfil
    `oleos_comestiveis_nir` (NIR, 8mm, oleos comestiveis) aplicado sem
    NENHUMA alteracao de codigo-fonte -- so' `cfg.matrix_profile`. Model
    card tem que declarar a matriz do PERFIL, nunca vocabulario de outra
    matriz (mesmo teste que `test_validacao_publica.py` faz para o Corn).

    Classes com <5 amostras sao excluidas (ver N_MINIMO_POR_CLASSE) --
    classificar sobre uma classe com 1 amostra nao e' uma checagem
    honesta, e' teatro estatistico.
    """
    from conftest import achar_pastas_run

    df, mapa_nomes = _carregar_bruto()
    contagem = df["Class"].value_counts()
    classes_ok = contagem[contagem >= N_MINIMO_POR_CLASSE].index
    sub = df[df["Class"].isin(classes_ok)].copy()

    cols = list(sub.columns)
    cols[0] = "classe"
    cols[1] = "conc"
    sub.columns = cols
    sub["classe"] = sub["classe"].map(mapa_nomes)
    sub = sub.drop(columns=["conc"])

    csv = tmp_path / "mendeley_classificacao.csv"
    sub.to_csv(csv, index=False)

    cfg = pq.Config(
        mode="csv", csv_file=str(csv),
        class_column="classe", conc_column="",
        matrix_profile="oleos_comestiveis_nir",
        objective="classificacao", level="N1",
        output_root_folder=str(tmp_path / "saida"),
        group_by_mae_id=False, show_plots=False,
        run_benchmark=False, run_monte_carlo=False, run_shap=False,
        run_wold=False, run_cv_anova=False, run_opls=False,
        run_ddsimca=False, executar_etapa4=False,
        n_permutations=20, frac_holdout=0.2, seed=0, max_lvs=10,
    )
    pq.executar(cfg)

    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs, "executar() nao criou saida"

    card = (Path(runs[0]) / pq.NOME_RELATORIOS / "model_card.md")
    conteudo = card.read_text(encoding="utf-8", errors="replace")
    assert "oleo comestivel (NIR, 8mm)" in conteudo
    assert "milho em grao" not in conteudo
    assert "oleo vegetal" not in conteudo   # vocabulario do dataset privado

    resumo = (Path(runs[0]) / pq.NOME_RELATORIOS / "resumo_modelo.txt")
    texto = resumo.read_text(encoding="utf-8", errors="replace")
    import re
    achado = re.search(r"Balanced accuracy\s*\.*:\s*([\d.]+)", texto)
    assert achado, f"Balanced accuracy nao encontrada no resumo:\n{texto[:600]}"
    bal_acc = float(achado.group(1))
    assert bal_acc >= BAL_ACC_MINIMA, (
        f"balanced_accuracy={bal_acc:.3f} abaixo do piso {BAL_ACC_MINIMA} "
        f"-- modelo nao aprendeu nada com 8 especies deste dataset "
        f"publico, sinal de bug (nao de dificuldade esperada do problema).")


@requer_mendeley
@pytest.mark.slow
def test_regressao_peroxido_roda_sem_excecao_e_calibra_razoavel(
        pq, tmp_path):
    """Sanity check da quantificacao (NAO gate de literatura -- ver
    ACHADO REAL no docstring do modulo). Confirma que o caminho de
    regressao pooled funciona neste dataset publico (100 amostras, alvo
    log10(indice de peroxido) por causa da assimetria 1.5-165) e produz
    um ajuste de CALIBRACAO razoavel -- nao trava, nao produz NaN/Inf.

    Nao afirma generalizacao (R2val) porque a medicao real (2026-08-27)
    mostrou R2val negativo com holdout de 25 amostras -- ver docstring
    do modulo para a analise completa. Registrar isso e' o ponto: um
    gate que fingisse sucesso aqui estaria mentindo sobre o que foi de
    fato validado.
    """
    from conftest import achar_pastas_run

    df, _mapa_nomes = _carregar_bruto()
    sub = df.copy()
    cols = list(sub.columns)
    cols[0] = "classe_original"
    cols[1] = "conc"
    sub.columns = cols
    sub["conc"] = np.log10(sub["conc"])
    sub.insert(0, "classe", "oleo")   # modelo GLOBAL/pooled -- mesma
    # escolha metodologica do artigo original ("a global peroxide value
    # model"), nao estratificado por especie.
    sub = sub.drop(columns=["classe_original"])

    csv = tmp_path / "mendeley_regressao.csv"
    sub.to_csv(csv, index=False)

    cfg = pq.Config(
        mode="csv", csv_file=str(csv),
        class_column="classe", conc_column="conc",
        matrix_profile="oleos_comestiveis_nir",
        objective="quantificacao", level="N3",
        output_root_folder=str(tmp_path / "saida"),
        group_by_mae_id=False, show_plots=False,
        n_permutations=5, frac_holdout=0.25, seed=0,
    )
    pq.executar(cfg)   # nao deve lancar excecao

    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs, "executar() nao criou saida"
    resumo = (Path(runs[0]) / pq.NOME_RELATORIOS / "resumo_modelo.txt")
    texto = resumo.read_text(encoding="utf-8", errors="replace")

    import re
    r2cal = re.search(r"R2cal\s*\.*:\s*([\-\d.]+)", texto)
    assert r2cal, f"R2cal nao encontrado no resumo:\n{texto[:600]}"
    valor = float(r2cal.group(1))
    assert np.isfinite(valor), "R2cal nao-finito -- regressao degenerada"
    assert valor > 0.3, (
        f"R2cal={valor:.3f} baixo demais para 'a calibracao capturou "
        f"algum sinal real' -- ver docstring do modulo para o que ESTE "
        f"teste garante (nao e' o mesmo que reproduzir o RMSEP publicado).")
