# -*- coding: utf-8 -*-
"""Validacao de GC-IMS (IMS) contra dataset publico Zenodo `10.5281/
zenodo.19209004` -- Passo 158/159 da busca ampliada por dataset de IMS
ja' pre-processado (2026-09-05), apos os Passos 151/158 terem
confirmado que os 2 datasets GC-IMS "classicos" (Mendeley `fr9t5fkkvz`/
`jxj2r45t2x`, azeite/mel) sao espectros BRUTOS grandes demais (4,8-6,9
GB) para o escopo desta auditoria (ver `docs/VALIDACAO_PUBLICA.md`
secao 2j).

Dataset: "Targeted GC-IMS Urine Dataset for Anisole and 2-Heptanone
Analysis in Colorectal Cancer and Control Samples" (Fernandez, Univ. de
Barcelona/IBEC) -- 44 medicoes GC-IMS de urina (FlavourSpec, headspace
estatico), das quais 30 sao amostras CLINICAS (15 cancer colorretal/CRC,
15 controle/CTRL, 1 medicao por paciente) e 14 sao amostras de controle
de qualidade (urina pool com concentracoes conhecidas dos analitos-alvo
-- NAO entram na classificacao, mesma logica de excluir calibradores/
brancos de uma validacao supervisionada). Licenca **CC BY 4.0**
(confirmada no `README.txt` do proprio dataset e na API do Zenodo).

DIFERENCA CRUCIAL para os outros 2 datasets GC-IMS ja' descartados por
tamanho: os proprios autores JA' processaram os 44 espectros brutos
`.mea` (denoising, alinhamento, deteccao de picos por clustering) e
publicaram o RESULTADO como tabela de picos CSV -- `peak_table_
untargeted.csv`, 184 variaveis "Cluster<N>" (intensidade de pico),
zero NaN. So' esta tabela e' baixada (`scripts/download_datasets/
baixar_zenodo_gcims_urina.py` le o zip remoto via HTTP Range e extrai
so' os 4 arquivos pequenos -- annotations.csv, as 2 tabelas de picos e
o README -- SEM baixar os 44 `.mea` brutos, ~3,4 GB descomprimidos que
este projeto nao usa).

ALVO: coluna `patient_condition` (CRC/CTRL), so' nas 30 amostras com
`class=="patient"` (as 14 QC tem `patient_condition=NaN`, filtradas
antes de montar o CSV).

GROUP-AWARE (regra 4): cada amostra clinica tem `patient_id` UNICO (1
medicao por paciente, confirmado por leitura direta -- 30 IDs distintos
para 30 linhas) -- `group_by_mae_id=False` e' correto pela mesma razao
do Corn/RMN/HPLC (nao ha' grupo/replica para vazar entre treino e
holdout).

Reproduzir:
```
python scripts/download_datasets/baixar_zenodo_gcims_urina.py
GUARACI_DATASETS_DIR=datasets_publicos pytest tests/test_validacao_publica_gcims_urina.py -v
```

Citar ao usar este dataset: Fernandez, L. et al. (2026). Targeted
GC-IMS Urine Dataset for Anisole and 2-Heptanone Analysis in
Colorectal Cancer and Control Samples [Data set]. Zenodo.
doi:10.5281/zenodo.19209004

ACHADO REAL, medido em 2026-09-05 (NEGATIVO, registrado com a mesma
disciplina do achado `unripe` do HSI e da Fluorescencia fraca): com o
motor GENERICO do GUARACI (PLS-DA, 184 variaveis "Cluster<N>", sem
selecao) a classificacao CRC-vs-controle fica em **balanced_accuracy =
0,500 (CV) -- EXATAMENTE o acaso** para um problema binario (holdout de
6 amostras: 0,333, mas n tao pequeno que uma unica amostra move a
metrica em ~17 pontos -- nao interpretado como sinal adicional). Nao e'
o sintoma do bug do Passo 148 (colapso numa classe so'): a matriz de
confusao mostra as DUAS classes preditas, so' que sem acerto acima do
acaso (12/12 amostras, recall 0,42/0,58). Permutacao (20 iteracoes):
p=0,619 -- NAO significativo. Checagem direta nos 2 compostos-alvo do
proprio dataset (anisol/2-heptanona, os 2 biomarcadores que a instrucao
original do estudo pretendia avaliar): Mann-Whitney U CRC-vs-CTRL,
p=0,868 (anisol) e p=0,590 (heptanona) -- nenhuma diferenca
estatisticamente significativa nem nos 2 compostos-alvo isolados.
CONSISTENTE com o proprio escopo do dataset: o artigo associado
(Surrogate-matrix calibration for quantitative GC-IMS headspace
analysis of urine, 2026, ScienceDirect) foca em METODOLOGIA de
calibracao/quantificacao, nao em prova de eficacia diagnostica -- n=15
por grupo e' pequeno demais para biomarcador urinario de cancer
(sinal tipicamente sutil na literatura). Isso NAO invalida a tecnica:
o motor GENERICO le a tabela de picos, roda sem excecao, preditcoes nao
colapsam numa classe so' -- e' o engine que esta sendo validado contra
dado IMS real pela 1a vez, nao um veredito sobre o biomarcador em si.
Registrado HONESTO, mesma disciplina do achado fraco de Fluorescencia
(secao 2d) e Mendeley NIR8mm (secao 2)."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest


def _pasta_gcims():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "zenodo_19209004_gcims_urina"
    nome = "peak_table_untargeted.csv"
    return pasta if (pasta / nome).is_file() else None


requer_gcims = pytest.mark.skipif(
    _pasta_gcims() is None,
    reason=("peak_table_untargeted.csv (Zenodo 19209004) ausente. Baixe "
            "com 'python scripts/download_datasets/"
            "baixar_zenodo_gcims_urina.py' e aponte GUARACI_DATASETS_DIR "
            "para a pasta que contem zenodo_19209004_gcims_urina/."))


def _carregar_amostras_clinicas():
    import pandas as pd

    pasta = _pasta_gcims()
    df = pd.read_csv(pasta / "peak_table_untargeted.csv")
    cols_pico = [c for c in df.columns if c.startswith("Cluster")]
    assert len(cols_pico) == 184, f"esperava 184 colunas Cluster, achou {len(cols_pico)}"
    assert df[cols_pico].isna().sum().sum() == 0, (
        "tabela de picos tinha NaN -- premissa do docstring mudou.")

    clinicas = df[df["class"] == "patient"].copy()
    assert clinicas["patient_id"].duplicated().sum() == 0, (
        "patient_id duplicado -- premissa de 1 medicao por paciente mudou, "
        "revisar group-awareness antes de prosseguir.")
    assert clinicas["patient_condition"].isin(["CRC", "CTRL"]).all()

    X = clinicas[cols_pico].to_numpy(dtype=float)
    classe = clinicas["patient_condition"].to_numpy()
    return X, classe


@requer_gcims
@pytest.mark.slow
def test_gcims_classifica_crc_vs_controle_com_motor_generico(pq, tmp_path):
    """Classificacao binaria CRC-vs-controle (30 amostras clinicas) com
    o motor GENERICO do GUARACI (PLS-DA sobre a tabela de picos de 184
    clusters ja' processada pelos autores originais).

    NAO e' um teste de "classificacao funciona" -- e' o oposto,
    registrado com a mesma disciplina do achado `unripe` do HSI e da
    Fluorescencia fraca (secao 2d). Ver ACHADO REAL no docstring do
    modulo para os numeros completos e a checagem estatistica direta
    nos 2 compostos-alvo. Unico requisito real: nao pode colapsar numa
    classe so' (sintoma do bug do Passo 148, ver `pipeline.py`)."""
    import collections

    import pandas as pd

    from conftest import achar_pastas_run

    X, classe = _carregar_amostras_clinicas()
    assert X.shape == (30, 184), f"tabela clinica inesperada: {X.shape}"
    contagem = collections.Counter(classe.tolist())
    assert contagem == {"CRC": 15, "CTRL": 15}, f"distribuicao inesperada: {contagem}"

    csv = tmp_path / "gcims_urina.csv"
    df = pd.DataFrame(X, columns=[str(float(i)) for i in range(1, X.shape[1] + 1)])
    df.insert(0, "classe", classe)
    df.to_csv(csv, index=False)

    cfg = pq.Config(
        mode="csv", csv_file=str(csv),
        class_column="classe", conc_column="",
        matrix_profile="generico", wn_min=1.0, wn_max=float(X.shape[1]),
        objective="classificacao", level="N1",
        output_root_folder=str(tmp_path / "saida"),
        group_by_mae_id=False, show_plots=False,
        run_benchmark=False, run_monte_carlo=False, run_shap=False,
        run_wold=False, run_cv_anova=False, run_opls=False,
        run_ddsimca=False, executar_etapa4=False,
        n_permutations=20, frac_holdout=0.2, seed=0, max_lvs=5,
    )
    pq.executar(cfg)

    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs, "executar() nao criou saida"
    resumo = (Path(runs[0]) / pq.NOME_RELATORIOS / "resumo_modelo.txt").read_text(
        encoding="utf-8", errors="replace")
    achado = re.search(r"Balanced accuracy\s*\.*:\s*([\d.]+)", resumo)
    assert achado, f"Balanced accuracy nao encontrada no resumo:\n{resumo[:600]}"
    bal_acc = float(achado.group(1))
    assert np.isfinite(bal_acc) and 0.0 <= bal_acc <= 1.0, (
        f"balanced_accuracy={bal_acc} fora do intervalo valido -- sinal "
        f"de bug (nao de dificuldade esperada).")
    # Sem gate de "aprendeu algo": o achado real medido e' 0,500 (acaso),
    # documentado no docstring do modulo. Um gate `> 0.6` aqui seria
    # inventar sucesso que a medicao nao mostra.

    # Contra-prova direta do bug do Passo 148: as predicoes NAO podem
    # colapsar numa classe so' -- e' o sintoma exato do bug antigo.
    achados_por_classe = re.findall(r"Acc (\w+)\s*\.*:\s*([\d.]+)", resumo)
    assert len(achados_por_classe) == 2, (
        f"esperava 2 classes na tabela 'Acc <classe>', achou "
        f"{len(achados_por_classe)}: {achados_por_classe}")
    accs_por_classe = [float(v) for _c, v in achados_por_classe]
    assert min(accs_por_classe) > 0.0, (
        f"pelo menos uma classe com accuracy = 0,0 individual "
        f"({achados_por_classe}) -- sintoma do bug de colapso do Passo "
        f"148, nao da fraqueza honesta do sinal biologico documentada "
        f"no docstring do modulo.")
    print(f"\n[GC-IMS urina] balanced_accuracy (CV) = {bal_acc:.3f} "
          f"(n=30, 15 CRC/15 CTRL) -- achado negativo/nulo, ver docstring")
