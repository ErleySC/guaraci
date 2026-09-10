# -*- coding: utf-8 -*-
"""Validacao de HPLC contra dataset publico Zenodo `10.5281/
zenodo.21245912` -- Passo 157/159 da busca ampliada por dataset de HPLC
ja' pre-processado (2026-09-05), apos o Passo 152 ter fechado a tecnica
como "suportavel, aguardando dataset" por falta de candidato compativel.

Dataset: de la Mata-Espinosa, Bosque-Sendra, Bro & Cuadros-Rodriguez
(2011) -- 120 amostras de oleo comestivel medidas por HPLC com detector
de aerossol carregado (CAD), perfil de triacilgliceroois. Mirror recente
(Zenodo, publicado 2026-07-07) da pagina original `https://
ucphchemometrics.com/olive/` (grupo de quimiometria da Universidade de
Copenhague -- mesmo Rasmus Bro ja' referenciado em `hsi_multiway.py`
para PARAFAC/N-PLS). Licenca **CC BY 4.0** (confirmado via API oficial
do Zenodo). Baixado e verificado nesta auditoria (SHA256 conferido
contra o pinado em `scripts/download_datasets/
baixar_zenodo_hplc_azeite.py`).

Formato: `HPLCforweb.mat` e' um objeto MATLAB "Dataset Object" (MESMA
convencao do PLS_Toolbox ja' usada pelo `corn.mat` deste projeto --
`m["HPLCforweb"]["data"][0,0]` da' a matriz 120x4001). O eixo de 4001
pontos (retencao cromatografica) NAO tem unidade calibrada no arquivo
(`axistype="none"`) -- usado aqui como indice generico 1..4001, mesmo
espirito do RMN (ppm ja' binado, sem re-processamento de sinal).

ALVO: campo `class` do objeto (`classlookup`: 1=not, 2=olive, 3=mix).
Das 120 amostras, 118 estao no subconjunto "ativo" definido pelos
proprios autores (`include`, linhas 3-120; as 2 primeiras, `Cac_CA`/
`Cac_LE`, ficam de fora -- decisao original preservada, nao uma escolha
deste projeto). Distribuicao ativa: 71 azeite, 42 nao-azeite, 5
mistura.

RETRATACAO METODOLOGICA (medido em 2026-09-05, ANTES de publicar
qualquer numero): a primeira versao deste teste tentou classificacao de
3 classes (not/olive/mix) e mediu balanced_accuracy=0,646 (CV) / 0,593
(holdout) -- BEM abaixo do esperado. Investigado: a classe "mix" (n=5,
menos de 1 amostra por fold em holdout de 20%) teve precision/recall/f1
= 0,00 -- nao aprendeu nada, arrastando a media macro para baixo,
enquanto "not" (f1=0,94) e "olive" (f1=0,97) discriminaram MUITO bem
sozinhas. Isso nao e' um bug nem uma limitacao do motor: e' um erro de
escopo deste teste, nao do dado. O arquivo NAO tem nenhum campo de
percentual de mistura continuo (`userdata`/`description` vazios,
confirmado por leitura direta) -- "mix" e' so' uma bandeira binaria
"e' uma mistura", sem informacao suficiente para virar uma classe
propria ou um alvo de quantificacao. Isso E' consistente com os 2
artigos originais: o primeiro (Anal Bioanal Chem 2011) faz EXATAMENTE
a discriminacao BINARIA azeite-vs-nao-azeite (as 5 misturas nao entram
nessa parte); o percentual de azeite em mistura e' o alvo do SEGUNDO
artigo (Talanta 2011), que usa uma tabela de referencia separada nao
incluida neste arquivo Zenodo. Corrigido: o teste abaixo classifica
SO' azeite-vs-nao-azeite (113 amostras, excluindo as 5 misturas) --
o mesmo escopo do artigo original, nao uma escolha deste projeto para
inflar o numero.

ACHADO REAL, medido em 2026-09-05 apos a correcao de escopo: com o
motor GENERICO do GUARACI (PLS-DA, 4001 variaveis, sem nenhuma selecao
de variavel) a classificacao binaria azeite-vs-nao-azeite fica em
**balanced_accuracy = 0,970 (CV) / 0,944 (holdout de 23 amostras)** --
consistente com a discriminacao forte reportada no artigo original.

GROUP-AWARE (regra 4 da instrucao, "group-aware em qualquer validacao
nova"): verificado por leitura direta que os 118 rotulos de amostra
ativos sao TODOS UNICOS (zero duplicata exata) -- cada linha e' 1
amostra fisica distinta (nao ha' replica tecnica registrada no dataset,
ao contrario do Corn que tem replicas por espectrometro mas nao por
amostra). `group_by_mae_id=False` e' correto aqui pela MESMA razao do
Corn e do RMN: nao ha' grupo/replica para vazar entre treino e holdout.
Auditoria de correlacao aproximada (limiar > 0.999, mesmo metodo usado
para o RMN no Passo 154) encontrou 6 pares de alta similaridade -- os 6
DENTRO da classe "olive" (varias amostras de mesma cultivar, ex.
`FRA_1..FRA_5`), nenhum cruzando classes: nao pode inflar a separacao
azeite-vs-nao-azeite que a metrica mede.

Reproduzir:
```
python scripts/download_datasets/baixar_zenodo_hplc_azeite.py
GUARACI_DATASETS_DIR=datasets_publicos pytest tests/test_validacao_publica_hplc_azeite.py -v
```

Citar ao usar este dataset:
  de la Mata-Espinosa, P.; Bosque-Sendra, J. M.; Bro, R.;
  Cuadros-Rodriguez, L. (2011). Discriminating olive and non-olive oils
  using HPLC-CAD and chemometrics. Analytical and Bioanalytical
  Chemistry, 399(6), 2083-2092. doi:10.1007/s00216-010-4366-4
  de la Mata-Espinosa, P.; Bosque-Sendra, J. M.; Bro, R.;
  Cuadros-Rodriguez, L. (2011). Olive oil quantification of edible
  vegetable oil blends using triacylglycerols chromatographic
  fingerprints and chemometric tools. Talanta, 85(1), 177-182.
  doi:10.1016/j.talanta.2011.03.049
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest


def _caminho_hplc():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    caminho = Path(raiz) / "zenodo_21245912_hplc_azeite" / "HPLCforweb.mat"
    return caminho if caminho.is_file() else None


requer_hplc = pytest.mark.skipif(
    _caminho_hplc() is None,
    reason=("HPLCforweb.mat (Zenodo 21245912) ausente. Baixe com "
            "'python scripts/download_datasets/baixar_zenodo_hplc_azeite.py' "
            "e aponte GUARACI_DATASETS_DIR para a pasta que contem "
            "zenodo_21245912_hplc_azeite/."))


def _carregar_ativo():
    import scipy.io as sio

    m = sio.loadmat(str(_caminho_hplc()))
    obj = m["HPLCforweb"]
    data = np.asarray(obj["data"][0, 0], dtype=float)
    cls_full = obj["class"][0, 0][0, 0].ravel()
    labels_full = obj["label"][0, 0][0, 0].ravel()
    include_idx = obj["include"][0, 0][0, 0].ravel().astype(int) - 1

    X = data[include_idx]
    y_int = cls_full[include_idx]
    labels = np.array([str(v).strip() for v in labels_full[include_idx]])

    mapa = {1: "not", 2: "olive", 3: "mix"}
    classe = np.array([mapa[int(v)] for v in y_int])
    return X, classe, labels


@requer_hplc
@pytest.mark.slow
def test_hplc_classifica_azeite_vs_nao_azeite_com_motor_generico(pq, tmp_path):
    """Classificacao BINARIA azeite-vs-nao-azeite (113 das 118 amostras
    ativas -- as 5 "mix" excluidas, ver RETRATACAO METODOLOGICA no
    docstring do modulo) com o motor GENERICO do GUARACI (PLS-DA sobre a
    matriz 113x4001 ja' corrigida de baseline e alinhada pelos autores
    originais). Mesmo escopo do artigo original (Anal Bioanal Chem
    2011)."""
    import pandas as pd

    from conftest import achar_pastas_run

    X, classe, labels = _carregar_ativo()
    assert X.shape == (118, 4001), f"HPLCforweb.mat inesperado: {X.shape}"
    assert not np.isnan(X).any() and not np.isinf(X).any()
    assert len(set(labels.tolist())) == len(labels), (
        "rotulos de amostra duplicados -- premissa de 1 amostra fisica "
        "por linha mudou, revisar group-awareness antes de prosseguir.")

    import collections
    contagem = collections.Counter(classe.tolist())
    assert contagem == {"olive": 71, "not": 42, "mix": 5}, (
        f"distribuicao de classes inesperada: {contagem}")

    mascara_binaria = classe != "mix"
    X = X[mascara_binaria]
    classe = classe[mascara_binaria]
    assert X.shape[0] == 113

    csv = tmp_path / "hplc_azeite.csv"
    df = pd.DataFrame(X, columns=[str(float(i)) for i in range(1, 4002)])
    df.insert(0, "classe", classe)
    df.to_csv(csv, index=False)

    cfg = pq.Config(
        mode="csv", csv_file=str(csv),
        class_column="classe", conc_column="",
        matrix_profile="generico", wn_min=1.0, wn_max=4001.0,
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
    resumo = (Path(runs[0]) / pq.NOME_RELATORIOS / "resumo_modelo.txt").read_text(
        encoding="utf-8", errors="replace")
    achado = re.search(r"Balanced accuracy\s*\.*:\s*([\d.]+)", resumo)
    assert achado, f"Balanced accuracy nao encontrada no resumo:\n{resumo[:600]}"
    bal_acc = float(achado.group(1))
    assert np.isfinite(bal_acc) and 0.0 <= bal_acc <= 1.0
    assert bal_acc > 0.85, (
        f"balanced_accuracy={bal_acc:.3f} abaixo do piso esperado (0.85) -- "
        f"medido em 2026-09-05: CV=0,970 / holdout=0,944 (piso com folga "
        f"sob os dois). O artigo original reporta discriminacao forte "
        f"azeite-vs-nao-azeite com este mesmo perfil cromatografico -- "
        f"uma queda ate perto do acaso aqui seria sinal de bug, nao de "
        f"dificuldade esperada.")
