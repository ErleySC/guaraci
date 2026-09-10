# -*- coding: utf-8 -*-
"""Validacao das tecnicas MIR e Raman contra o MESMO dataset publico
Mendeley `10.17632/ctgg7k4m5g.2` (Ottaway et al. 2021) ja integrado para
NIR em `test_validacao_publica_mendeley.py` -- Passo 142/143 da auditoria
das 11 tecnicas do menu `cli_assistente.TECNICAS` (2026-09-04).

Por que este arquivo existe separado do de NIR: `MIR1A.csv` e
`Raman1A.csv` sao arquivos IRMAOS de `NIR8mm1A.csv` no MESMO repositorio
Mendeley -- mesmas 100 garrafas de oleo, mesmo alvo (indice de peroxido),
so' medidas em tecnica diferente. Confirmado por leitura direta
(`_carregar_bruto` abaixo): `Class`/`PeroxideValue` sao IDENTICOS
linha-a-linha nos 3 arquivos. Isso decide o escopo do Passo 142 para
essas 2 tecnicas SEM busca de dataset nova nenhuma -- so' reusar a
integracao ja feita (`scripts/download_datasets/baixar_mendeley_oleos.py`
estendido para baixar os 2 arquivos novos, hash pinado a partir da API
publica do Mendeley que ja' devolve o SHA256 calculado por eles).

Estrutura dos arquivos (verificado por leitura direta em 2026-09-04):
  - `MIR1A.csv`: 100 linhas x 3423 canais espectrais (699-3999 cm-1,
    faixa MIR classica), ZERO NaN.
  - `Raman1A.csv`: 100 linhas x 1340 canais (Raman shift -18 a 1974
    cm-1), 1 linha com NaN em TODAS as colunas espectrais (amostra sem
    aquela medicao Raman especifica tomada -- ja documentado no
    docstring de `test_validacao_publica_mendeley.py`); removida antes
    de treinar (`dropna()`).

Nenhum perfil de matriz dedicado existe para MIR/Raman de oleos
comestiveis (so' NIR tem `oleos_comestiveis_nir`) -- usar esse perfil
aqui aplicaria faixa/vocabulario ERRADOS (faixa NIR sobre canal
MIR/Raman). Os testes abaixo usam `matrix_profile="generico"` e `wn_min`/
`wn_max` explicitos por tecnica -- criar perfis dedicados
(`oleos_comestiveis_mir`/`_raman`) fica fora do escopo aprovado deste
passo, registrado aqui como pendencia honesta (Passo 144/145 futuro).

ACHADO REAL (medido em 2026-09-04, mesma disciplina do que foi feito
para NIR em 2026-08-27): ao contrario do NIR8mm (R2val NEGATIVO,
ver docstring do outro arquivo), tanto MIR quanto Raman produziram
R2val POSITIVO nesta mesma medicao pontual (holdout de 25 amostras,
seed=0): MIR R2cal=0.79/R2val=0.57/RMSEP=0.26 (log10); Raman
R2cal=0.67/R2val=0.43/RMSEP=0.26. Classificacao (8 especies com >=5
amostras, n=78, mesmo filtro do teste de NIR): MIR bal_acc=0.696, Raman
bal_acc=0.389 (acima do acaso ~0.125 mas mais fraco, coerente com o
proprio artigo original sinalizando o RMSEP do Raman como possivel
correlacao por acaso). Os limiares abaixo sao FLOOR de sanidade com
margem folgada sob o valor medido -- nao um alvo de literatura, mesma
politica de `test_validacao_publica_mendeley.py`. Um unico holdout com
n=100 e' um teste de alta variancia; nao gerar prosa de "MIR bate NIR"
a partir de UMA medicao (regra 2 da instrucao: nada de "validado" sem
disciplina de reproducao)."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

#: Mesmo floor/racional de `test_validacao_publica_mendeley.py` --
#: "aprendeu alguma coisa" (chance ~0.125 com 8 classes), nao alvo de
#: literatura. Raman fica com floor menor: o proprio artigo original
#: sinaliza esta tecnica como menos confiavel para este dataset.
BAL_ACC_MINIMA = {"MIR1A.csv": 0.4, "Raman1A.csv": 0.2}
N_MINIMO_POR_CLASSE = 5
R2CAL_MINIMO = {"MIR1A.csv": 0.5, "Raman1A.csv": 0.4}

#: Faixa real do eixo espectral de cada arquivo (verificado por leitura
#: direta) -- generico nao tem perfil proprio, entao wn_min/wn_max sao
#: passados explicitamente na config em vez de vir de matrix_profile.
FAIXA_EIXO = {"MIR1A.csv": (699.0, 3999.0), "Raman1A.csv": (-19.0, 1975.0)}


def _pasta_mendeley():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "mendeley_ctgg7k4m5g"
    if (pasta / "MIR1A.csv").is_file() and (pasta / "Raman1A.csv").is_file():
        return pasta
    return None


requer_mendeley_mir_raman = pytest.mark.skipif(
    _pasta_mendeley() is None,
    reason=("MIR1A.csv/Raman1A.csv do dataset Mendeley ctgg7k4m5g "
            "ausentes. Baixe com "
            "'python scripts/download_datasets/baixar_mendeley_oleos.py' "
            "e aponte GUARACI_DATASETS_DIR para a pasta que contem "
            "mendeley_ctgg7k4m5g/."))


def _carregar_bruto(nome_arquivo: str):
    pasta = _pasta_mendeley()
    df = pd.read_csv(pasta / nome_arquivo)
    chave = pd.read_csv(pasta / "OilClassKey.csv")
    mapa_nomes = dict(zip(chave["Class Number"], chave["Class Name"]))
    df = df.dropna()   # so' remove linha(s) em Raman1A.csv; MIR1A.csv nao tem NaN
    return df, mapa_nomes


def _rodar_classificacao(pq, tmp_path, nome_arquivo: str) -> float:
    df, mapa_nomes = _carregar_bruto(nome_arquivo)
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

    wn_min, wn_max = FAIXA_EIXO[nome_arquivo]
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

    from conftest import achar_pastas_run
    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs, "executar() nao criou saida"

    resumo = (Path(runs[0]) / pq.NOME_RELATORIOS / "resumo_modelo.txt").read_text(
        encoding="utf-8", errors="replace")
    achado = re.search(r"Balanced accuracy\s*\.*:\s*([\d.]+)", resumo)
    assert achado, f"Balanced accuracy nao encontrada no resumo:\n{resumo[:600]}"
    return float(achado.group(1))


def _rodar_regressao(pq, tmp_path, nome_arquivo: str) -> float:
    df, _mapa_nomes = _carregar_bruto(nome_arquivo)
    sub = df.copy()
    cols = list(sub.columns)
    cols[0] = "classe_original"
    cols[1] = "conc"
    sub.columns = cols
    sub["conc"] = np.log10(sub["conc"])
    sub.insert(0, "classe", "oleo")
    sub = sub.drop(columns=["classe_original"])

    csv = tmp_path / "mendeley_regressao.csv"
    sub.to_csv(csv, index=False)

    wn_min, wn_max = FAIXA_EIXO[nome_arquivo]
    cfg = pq.Config(
        mode="csv", csv_file=str(csv),
        class_column="classe", conc_column="conc",
        matrix_profile="generico", wn_min=wn_min, wn_max=wn_max,
        objective="quantificacao", level="N3",
        output_root_folder=str(tmp_path / "saida"),
        group_by_mae_id=False, show_plots=False,
        n_permutations=5, frac_holdout=0.25, seed=0,
    )
    pq.executar(cfg)   # nao deve lancar excecao

    from conftest import achar_pastas_run
    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs, "executar() nao criou saida"
    resumo = (Path(runs[0]) / pq.NOME_RELATORIOS / "resumo_modelo.txt").read_text(
        encoding="utf-8", errors="replace")

    r2cal = re.search(r"R2cal\s*\.*:\s*([\-\d.]+)", resumo)
    assert r2cal, f"R2cal nao encontrado no resumo:\n{resumo[:600]}"
    return float(r2cal.group(1))


@requer_mendeley_mir_raman
@pytest.mark.slow
def test_mir_classifica_acima_do_acaso(pq, tmp_path):
    """MIR1A.csv: mesmo filtro/split de `test_validacao_publica_mendeley.
    py::test_multimatriz_declara_perfil_correto_e_classifica_acima_do_
    acaso`, so' com o arquivo-irmao MIR."""
    bal_acc = _rodar_classificacao(pq, tmp_path, "MIR1A.csv")
    piso = BAL_ACC_MINIMA["MIR1A.csv"]
    assert bal_acc >= piso, (
        f"balanced_accuracy={bal_acc:.3f} abaixo do piso {piso} -- "
        f"modelo MIR nao aprendeu nada com 8 especies deste dataset "
        f"publico, sinal de bug (nao de dificuldade esperada).")


@requer_mendeley_mir_raman
@pytest.mark.slow
def test_raman_classifica_acima_do_acaso(pq, tmp_path):
    """Raman1A.csv: piso mais baixo que MIR de proposito -- o artigo
    original sinaliza esta tecnica como menos confiavel para este
    dataset (RMSEP publicado possivelmente por correlacao de acaso)."""
    bal_acc = _rodar_classificacao(pq, tmp_path, "Raman1A.csv")
    piso = BAL_ACC_MINIMA["Raman1A.csv"]
    assert bal_acc >= piso, (
        f"balanced_accuracy={bal_acc:.3f} abaixo do piso {piso} -- "
        f"modelo Raman nao aprendeu nada com 8 especies deste dataset "
        f"publico, sinal de bug (nao de dificuldade esperada).")


@requer_mendeley_mir_raman
@pytest.mark.slow
def test_mir_regressao_peroxido_roda_e_calibra_razoavel(pq, tmp_path):
    """Sanity check de calibracao (NAO gate de literatura -- ver ACHADO
    REAL no docstring do modulo para o R2val medido, positivo aqui ao
    contrario do NIR8mm)."""
    r2cal = _rodar_regressao(pq, tmp_path, "MIR1A.csv")
    assert np.isfinite(r2cal), "R2cal nao-finito -- regressao degenerada"
    piso = R2CAL_MINIMO["MIR1A.csv"]
    assert r2cal > piso, (
        f"R2cal={r2cal:.3f} baixo demais para 'a calibracao MIR capturou "
        f"algum sinal real' (piso {piso}).")


@requer_mendeley_mir_raman
@pytest.mark.slow
def test_raman_regressao_peroxido_roda_e_calibra_razoavel(pq, tmp_path):
    """Idem MIR, arquivo Raman1A.csv (1 linha NaN removida antes)."""
    r2cal = _rodar_regressao(pq, tmp_path, "Raman1A.csv")
    assert np.isfinite(r2cal), "R2cal nao-finito -- regressao degenerada"
    piso = R2CAL_MINIMO["Raman1A.csv"]
    assert r2cal > piso, (
        f"R2cal={r2cal:.3f} baixo demais para 'a calibracao Raman "
        f"capturou algum sinal real' (piso {piso}).")
