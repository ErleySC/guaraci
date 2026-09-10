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

RETRATACAO (Passo 148, 2026-09-04): a nota anterior desta secao (achado
de 2026-09-04, Passo 142/143) registrava "balanced_accuracy = 0.500 --
EXATAMENTE o acaso" para este dataset, com 4 presets de
pre-processamento diferentes todos dando o MESMO 0.500 -- coincidencia
que deveria ter levantado suspeita na hora (0.500 exato E identico entre
4 presets distintos e' o padrao de um COLAPSO, nao de "sinal fraco").
**Era um bug real do GUARACI, nao uma limitacao do dado nem do motor
generico.** Investigado ao tentar aplicar o I de Moran (ver ACHADO REAL
abaixo): `pipeline.executar()` construia o alvo one-hot
(`Y_bin = LabelBinarizer().fit_transform(rotulos)`) e so' expandia para
2 colunas quando `Y_bin.ndim == 1` -- mas para EXATAMENTE 2 classes o
sklearn ja devolve shape `(n, 1)` (ndim=**2**, nao 1), entao a expansao
NUNCA disparava. Com 1 SO' coluna, `np.argmax(Y_bin, axis=1)` e' SEMPRE
0, e toda predicao downstream colapsava na PRIMEIRA classe -- balanced_
accuracy fica travado em exatamente 0.5 para QUALQUER dataset BINARIO,
disfarcado de "acaso genuino", independente de pre-processamento (por
isso os 4 presets davam o mesmo numero: nenhum deles chegava perto de
mudar o resultado, o colapso acontecia DEPOIS). A mesma checagem
CORRETA (`ndim == 1 or shape[1] == 1`) ja existia ha' muito tempo em
`avaliacao_modelos.PLSDAClassifier.fit`/`hsi_multiway.NPLSClassifier.
fit`/`portao_correcao_sinal` -- so' o caminho principal de classificacao
(`pipeline.executar()`, usado por TODA execucao N1/N2) tinha ficado pra
tras dessa correcao. Corrigido em `src/guaraci/pipeline.py`; contra-prova
em `tests/test_pipeline_core.py::
test_executar_classificacao_binaria_nao_colapsa_em_uma_classe_so`
(dataset sintetico 2-classes bem separado, que colapsava para 0.5 antes
da correcao e agora classifica >0.9). Este e' o UNICO dataset publico
deste projeto com exatamente 2 classes -- por isso o bug nunca apareceu
nos outros achados (Mendeley 8 especies, Fluorescencia 3 graus, HSI 3
estagios).

ACHADO REAL, CORRIGIDO (medido em 2026-09-04, apos o fix): com o bug
corrigido, a classificacao por provincia com o motor GENERICO do
GUARACI (PLS-DA, 125 variaveis, `matrix_profile="generico"`) fica em
**balanced_accuracy = 1,000 (CV) / 1,000 (holdout de 20 amostras)** --
robusto a 5 presets de pre-processamento diferentes. Ranking de
separabilidade CONSISTENTE com o proprio artigo original (que reporta
99% de acuracia) -- so' que o motor GENERICO do GUARACI, sem nenhuma
selecao de variavel geoestatistica, ja alcanca o mesmo patamar. A
tentativa de melhorar o resultado com I de Moran (motivacao original do
Passo 148) acabou desnecessaria: a separacao real e' forte e nao precisa
de selecao de variavel para aparecer -- ver `test_nmr_com_selecao_moran_
reavaliado_passo_148` abaixo, mantido como comparacao honesta (Full vs.
Moran, ambos ja perto do teto).

INDEPENDENCIA DO HOLDOUT VERIFICADA (Passo 154, 2026-09-05): o teste
principal abaixo usa `seed=0` -- a MESMA seed usada quando o bug de
colapso ainda estava presente (a divisao em si sempre foi valida; o bug
era na decodificacao da predicao, nao no split). Para descartar que
1,000 fosse sorte dessa divisao especifica, `test_nmr_holdout_robusto_a_
multiplas_seeds_passo_154` reroda com 5 seeds NOVAS (1-5); fora da suite
(nao repetido em teste por custo) foram medidas tambem as seeds 0-14
(15 no total): balanced_accuracy do CV interno do pipeline (Repeated
StratifiedKFold, LVs otimizados por Wold) E do holdout ficou EXATAMENTE
1,000 em TODAS as 15 -- nao 0,98-1,00 como uma nota anterior desta
secao chegou a registrar (esse "0,98-1,00, 10 seeds" media outra coisa:
o harness simplificado de `_avaliar_subset_cv`, LV FIXO=5 sem otimizacao
por fold, usado so' para comparar Full-vs-Moran -- ali sim ha' variacao
real, 0,926-0,958 em 10 seeds, ver `test_nmr_com_selecao_moran_
reavaliado_passo_148`; os dois numeros nao medem a mesma coisa e nao
devem ser confundidos). Independencia de grupo confirmada por leitura
direta: os 97 `Sample_ID` sao TODOS unicos (nenhuma replica fisica no
dataset), entao `group_by_mae_id=False` nao arrisca vazamento de grupo
entre treino e holdout. A auditoria de delineamento automatica
(`auditoria_delineamento.check_duplicates`) sinaliza 1 par de amostras
com correlacao aproximada > 0,99995 (`pe09`/`pe20`) -- confirmado por
leitura direta que as DUAS sao Pescara (mesma classe), entao esse par
nao pode ser a causa de separacao artificial ENTRE classes. Resultado
declarado DEFINITIVAMENTE FECHADO."""
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


def _carregar_bruto_com_coords():
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
    coords = df[["long", "lat"]].to_numpy(dtype=float)
    df = df.drop(columns=["Sample_ID", "long", "lat"]).copy()
    df["classe"] = classe
    return df, coords


def _carregar_bruto():
    df, _coords = _carregar_bruto_com_coords()
    return df


@requer_figshare_nmr
@pytest.mark.slow
def test_nmr_classifica_provincia_com_motor_generico_apos_correcao_do_bug(pq, tmp_path):
    """Ate 2026-09-04 este teste era o OPOSTO (nao separa) -- retratado
    no docstring do modulo: era um bug de classificacao binaria no
    GUARACI (`pipeline.py`), nao uma limitacao do dado. Corrigido, este
    dataset classifica MUITO bem (gate deliberadamente com folga sob o
    1,000 medido em 15 seeds -- ver `test_nmr_holdout_robusto_a_
    multiplas_seeds_passo_154` abaixo -- mesma disciplina de piso-com-
    folga do resto do projeto)."""
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
        f"balanced_accuracy={bal_acc:.3f} abaixo do piso esperado (0.85) "
        f"para este dataset apos a correcao do Passo 148 -- se isto "
        f"falhar, o bug de colapso em uma classe pode ter voltado (ver "
        f"docstring do modulo).")

    # Contra-prova direta do bug retratado: as predicoes NAO podem
    # colapsar numa classe so' -- e' o sintoma exato do bug antigo.
    achados_por_classe = re.findall(r"Acc (\w+)\s*\.*:\s*([\d.]+)", resumo)
    assert len(achados_por_classe) == 2, (
        f"esperava 2 classes na tabela 'Acc <classe>', achou "
        f"{len(achados_por_classe)}: {achados_por_classe}")
    accs_por_classe = [float(v) for _c, v in achados_por_classe]
    assert min(accs_por_classe) > 0.5, (
        f"pelo menos uma classe com accuracy <= 0.5 individual "
        f"({achados_por_classe}) -- sintoma do bug de colapso retratado "
        f"no docstring do modulo.")


@requer_figshare_nmr
@pytest.mark.slow
def test_nmr_holdout_robusto_a_multiplas_seeds_passo_154(pq, tmp_path):
    """Passo 154: a instrucao de fechamento pediu para confirmar que o
    holdout de 20 amostras que deu balanced_accuracy=1,000 nao e' um
    artefato de uma unica divisao favoravel (seed=0, a MESMA seed usada
    quando o bug de colapso ainda estava presente -- ver docstring do
    modulo). Reexecutado aqui com 5 seeds NOVAS (1-5, nunca usadas em
    nenhuma medicao anterior deste arquivo): balanced_accuracy do
    holdout continua exatamente 1,000 em TODAS -- medido tambem com
    seeds 0-14 fora da suite (15/15 em 1,000, CV interno do pipeline
    incluido), consistente com o proprio artigo original (99% com LDA +
    selecao geoestatistica). Nao e' sorte de uma divisao especifica.

    Pre-condicao de independencia verificada (fora da suite, nao repetida
    aqui por custo): as 97 `Sample_ID` sao TODAS unicas (nenhuma amostra
    fisica duplicada) -- `group_by_mae_id=False` e' correto porque nao ha'
    nenhum grupo/replica para vazar entre treino e holdout. O par de
    maior correlacao aproximada sinalizado pela auditoria de delineamento
    (`pe09`/`pe20`, corr=0,99997) e' DENTRO da mesma classe (Pescara-
    Pescara) -- nao pode inflar a separacao ENTRE classes, que e' o que
    este teste mede."""
    from conftest import achar_pastas_run

    sub = _carregar_bruto()
    cols_espectrais = [c for c in sub.columns if c != "classe"]
    wn_min = min(float(c) for c in cols_espectrais)
    wn_max = max(float(c) for c in cols_espectrais)

    resultados = {}
    for seed in range(1, 6):
        saida = tmp_path / f"saida_seed{seed}"
        csv = tmp_path / f"nmr_seed{seed}.csv"
        sub.to_csv(csv, index=False)
        cfg = pq.Config(
            mode="csv", csv_file=str(csv),
            class_column="classe", conc_column="",
            matrix_profile="generico", wn_min=wn_min, wn_max=wn_max,
            objective="classificacao", level="N1",
            output_root_folder=str(saida),
            group_by_mae_id=False, show_plots=False,
            run_benchmark=False, run_monte_carlo=False, run_shap=False,
            run_wold=False, run_cv_anova=False, run_opls=False,
            run_ddsimca=False, executar_etapa4=False,
            n_permutations=20, frac_holdout=0.2, seed=seed, max_lvs=10,
        )
        pq.executar(cfg)
        runs = achar_pastas_run(cfg.output_root_folder)
        assert runs, f"executar() nao criou saida para seed={seed}"
        resumo = (Path(runs[0]) / pq.NOME_RELATORIOS / "resumo_modelo.txt").read_text(
            encoding="utf-8", errors="replace")
        achado = re.search(r"Holdout balanced acc\s*\.*:\s*([\d.]+)", resumo)
        assert achado, (
            f"'Holdout balanced acc' nao encontrada no resumo (seed={seed}):"
            f"\n{resumo[:600]}")
        resultados[seed] = float(achado.group(1))

    for seed, bal_acc in resultados.items():
        assert bal_acc >= 0.85, (
            f"seed={seed}: holdout balanced_accuracy={bal_acc:.3f} abaixo "
            f"do piso (0.85) -- o resultado 1,000 medido na seed=0 pode "
            f"ter sido sorte de divisao, nao separabilidade real. "
            f"Resultados completos: {resultados}")


@requer_figshare_nmr
@pytest.mark.slow
def test_nmr_com_selecao_moran_reavaliado_passo_148(pq):
    """Passo 148 (Fase B): motivacao original -- tentar melhorar o RMN
    com a MESMA tecnica de selecao de variavel que o artigo original
    usou (I de Moran, ver docstring de `selecao_variaveis.moran_i_mask`)
    -- ficou PREJUDICADA pela descoberta do bug retratado no docstring
    do modulo: o achado negativo que motivava a Fase B nunca foi real,
    era um bug de classificacao binaria em `pipeline.py`. Mantido como
    comparacao HONESTA agora que o motor generico ja classifica quase
    perfeitamente sozinho (teste acima): 'Full (125 var)' vs. 'I de
    Moran (subconjunto)', mesmos folds de CV (amostras independentes, 1
    medicao por azeite -- mesmo raciocinio de `group_by_mae_id=False` do
    teste acima), via `_avaliar_subset_cv` (harness limpo do modulo de
    selecao de variaveis, NUNCA passou pelo bug de `pipeline.py` -- por
    isso os numeros medidos aqui, ANTES da correcao do pipeline, ja eram
    altos e foram a PISTA que levou a achar o bug).

    A mascara de Moran e' refeita a cada fold usando so' o treino
    daquele fold (`_avaliar_subset_nested_cv_moran`, nested-CV -- sem
    isso o numero seria inflado por vazamento, o mesmo vies que o
    Bloco 27 mediu para o iPLS)."""
    from sklearn.model_selection import StratifiedKFold

    from guaraci.selecao_variaveis import (
        _avaliar_subset_cv, _avaliar_subset_nested_cv_moran)

    df, coords = _carregar_bruto_com_coords()
    cols_espectrais = [c for c in df.columns if c != "classe"]
    X = df[cols_espectrais].to_numpy(dtype=float)
    y_int = (df["classe"] == "Teramo").to_numpy(dtype=int)
    Y_bin = np.eye(2)[y_int]

    cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
              .split(X, y_int))
    n_lv = 5

    completo = _avaliar_subset_cv(X, Y_bin, y_int, cv, n_lv)
    moran = _avaliar_subset_nested_cv_moran(
        X, Y_bin, y_int, coords, cv, n_lv,
        k_vizinhos=8, alpha=0.05, n_permutacoes=999, seed=0)

    print(f"\n[Passo 148] Full (125 var): balanced_accuracy="
          f"{completo['balanced_accuracy']:.3f}")
    print(f"[Passo 148] I de Moran ({moran['n_vars']:.0f} var em media, "
          f"{moran['n_vars_min']}-{moran['n_vars_max']}): "
          f"balanced_accuracy={moran['balanced_accuracy']:.3f}")

    assert np.isfinite(moran["balanced_accuracy"])
    # Piso com folga sob os dois numeros medidos em 2026-09-04 (Full~0.94,
    # Moran~0.96 com n_lv=5, seed=0) -- os dois ja' perto do teto, achado
    # honesto e' que a selecao de variavel NAO era o que faltava (o bug
    # de pipeline.py era). Ver docs/VALIDACAO_PUBLICA.md secao 2e/
    # docs/PROGRESSO.md Passo 148 para os numeros completos e a decisao.
    assert completo["balanced_accuracy"] > 0.85
    assert moran["balanced_accuracy"] > 0.85
