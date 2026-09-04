# -*- coding: utf-8 -*-
"""Validacao de UV-Vis contra dataset publico ERIC/Eawag -- Passo 147 da
Fase A (fechamento das 11 tecnicas do menu `cli_assistente.TECNICAS`,
2026-09-04).

Dataset: Lechevallier et al. (2025), "Dataset on wastewater quality
monitoring with adsorption and reflectance spectroscopy in the UV/Vis
range", Scientific Data 12:1296, doi:10.1038/s41597-025-05459-x. Campanha
de 25 semanas (esgoto bruto, Suiça) com 2 espectrofotometros UV-Vis
(Spectrolyser/scan e ISA) medindo absorbancia a cada 2 minutos, e 533
amostras coletadas manualmente e analisadas em laboratorio para 9
indicadores de poluicao convencionais (turbidez, DOC, TSS, TOC, N
dissolvido, N total, NH4, PO4, SO4). Licenca CC BY (confirmado via API
publica do portal CKAN `opendata.eawag.ch`).

So' o arquivo tabular `2_data.zip` e' baixado (~357 MB) -- o pacote
completo tem mais ~180 GB de cubos hiperespectrais que este projeto nao
usa (ver `scripts/download_datasets/baixar_eawag_esgoto_uvvis.py`).

DECISOES METODOLOGICAS (documentadas, nao escondidas):
  - Sensor usado: **scan** (Spectrolyser, 200-735 nm, resolucao 2,5 nm) --
    faixa completa publicada sem descarte de canais, ao contrario do ISA
    (que precisa descartar UV baixo por absorcao da fibra optica, ver o
    paper). Escolha arbitraria de UM dos 2 sensores disponiveis, mesmo
    espirito de outras escolhas ja registradas neste projeto (ex.: NIR
    8mm entre 4 tecnicas no Mendeley ctgg7k4m5g).
  - Alvo: **DOC** (carbono organico dissolvido, mg/L) -- um dos 5
    indicadores medidos para TODAS as 533 amostras (ao contrario de
    TSS/TOC/N total, medidos so' para 45) e o correlato classico de
    absorbancia UV-Vis na literatura de monitoramento de agua (UV254 como
    substituto de DOC).
  - Casamento amostra-de-laboratorio <-> espectro do sensor: pelo
    timestamp mais proximo, tolerancia de 3 minutos (o sensor mede a
    cada 2 min) -- amostras sem par dentro da tolerancia sao descartadas.
  - **Agrupamento (regra 5 da instrucao, "group-aware em qualquer
    validacao nova")**: o motor CSV do GUARACI (`load_csv`) trata toda
    coluna que nao seja classe/conc como CANAL espectral -- nao ha' como
    passar uma coluna de agrupamento arbitraria (`mae_id`) sem quebrar o
    parser. Em vez de ignorar o risco de vazamento (amostras do MESMO DIA
    sao temporalmente autocorrelacionadas -- varias coletas manuais por
    dia), as amostras sao AGREGADAS POR DIA (media do alvo e do espectro
    casado) ANTES de montar o CSV -- mesma solucao ja usada para a
    Fluorescencia (Passo 142/143, colapsar repeticoes tecnicas por
    amostra). Cada linha final e' 1 dia = 1 unidade fisica independente;
    nao sobra estrutura repetida para vazar entre treino/holdout, entao
    `group_by_mae_id=False` e' correto aqui (mesmo raciocinio do Corn:
    nenhum grupo a proteger).
  - Pre-processamento: **EMSC** (ja aprovado no portao de aceite contra o
    Corn, Passo 134 -- ver `docs/VALIDACAO_PUBLICA.md` secao 9) + MC, por
    pedido explicito da instrucao ("validar com EMSC ja disponivel").
  - `matrix_profile="generico"`: nao existe perfil dedicado para esgoto/
    agua ainda (mesma situacao de MIR/Raman de oleos, Passo 142/143).

Reproduzir:
    python scripts/download_datasets/baixar_eawag_esgoto_uvvis.py
    GUARACI_DATASETS_DIR=<pasta> pytest tests/test_validacao_publica_eawag_esgoto_uvvis.py -v
"""
from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

#: Tolerancia de casamento temporal lab<->sensor (o sensor scan mede a
#: cada 2 minutos).
_TOLERANCIA_MATCH = pd.Timedelta(minutes=3)

#: Piso de sanidade -- "a calibracao capturou algum sinal real", NAO um
#: alvo de literatura (nao ha' RMSEP publicado para este recorte
#: especifico: sensor scan, alvo DOC, agregacao diaria). Mesma disciplina
#: de outros pisos deste projeto para dataset publico sem numero
#: comparavel (ex.: Mendeley peroxido, RMN).
R2CAL_MINIMO = 0.3
N_DIAS_MINIMO = 40

_ARQUIVO_ZIP = "2_data.zip"
_CAMINHO_LAB = "2_data/5_laboratory_reference_measurements/laboratory_measurements.csv"
_CAMINHO_SCAN = "2_data/3_sensor_data/flume_scan_absorbance.csv"


def _pasta_eawag():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "eawag_esgoto_uvvis"
    return pasta if (pasta / _ARQUIVO_ZIP).is_file() else None


requer_eawag = pytest.mark.skipif(
    _pasta_eawag() is None,
    reason=(f"{_ARQUIVO_ZIP} (ERIC/Eawag) ausente. Baixe com 'python "
            "scripts/download_datasets/baixar_eawag_esgoto_uvvis.py' e "
            "aponte GUARACI_DATASETS_DIR para a pasta que contem "
            "eawag_esgoto_uvvis/."))


def _carregar_bruto():
    """Le lab+scan de dentro do zip, casa por timestamp mais proximo,
    agrega por dia. Retorna (DataFrame diario, lista de nomes de coluna
    de comprimento de onda em nm-string)."""
    pasta = _pasta_eawag()
    with zipfile.ZipFile(pasta / _ARQUIVO_ZIP) as z:
        with z.open(_CAMINHO_LAB) as f:
            lab = pd.read_csv(f)
        with z.open(_CAMINHO_SCAN) as f:
            scan = pd.read_csv(f)

    lab["ts"] = pd.to_datetime(lab["timestamp_iso"])
    scan["ts"] = pd.to_datetime(scan["timestamp_iso"])

    wave_cols = [c for c in scan.columns if c.startswith("scan_absorbance")]
    assert wave_cols, "colunas de absorbancia do scan nao encontradas -- formato mudou?"

    scan_valido = scan[scan["valid_data"] == 1].sort_values("ts").reset_index(drop=True)
    scan_ts = scan_valido["ts"].to_numpy(dtype="datetime64[ns]")
    lab_ts = lab["ts"].to_numpy(dtype="datetime64[ns]")

    idx = np.searchsorted(scan_ts, lab_ts)
    idx = np.clip(idx, 1, len(scan_ts) - 1)
    esquerda, direita = idx - 1, idx
    delta_esq = np.abs(scan_ts[esquerda] - lab_ts)
    delta_dir = np.abs(scan_ts[direita] - lab_ts)
    melhor = np.where(delta_esq <= delta_dir, esquerda, direita)
    melhor_delta = np.minimum(delta_esq, delta_dir)

    casado = melhor_delta <= np.timedelta64(_TOLERANCIA_MATCH.value, "ns")
    lab = lab.copy()
    lab["match_idx"] = melhor
    lab["casado"] = casado

    usar = lab[lab["casado"] & lab["lab_doc_mg_l"].notna()].copy()
    espectro_casado = scan_valido.iloc[usar["match_idx"].to_numpy()][wave_cols].reset_index(drop=True)
    usar = usar.reset_index(drop=True)
    usar["dia"] = usar["ts"].dt.date

    junto = pd.concat(
        [usar[["dia", "lab_doc_mg_l"]].reset_index(drop=True), espectro_casado], axis=1)
    diario = junto.groupby("dia").mean(numeric_only=True).reset_index()

    wl_nm = {c: f"{float(c[len('scan_absorbance'):-len('__m')]) / 10.0:.1f}"
             for c in wave_cols}
    diario = diario.rename(columns=wl_nm)
    cols_onda = list(wl_nm.values())
    return diario, cols_onda


@requer_eawag
@pytest.mark.slow
def test_uvvis_doc_calibra_com_emsc_sobre_esgoto_real(pq, tmp_path):
    """Sanity check de quantificacao (NAO gate de literatura -- nao ha'
    RMSEP publicado para este recorte). Confirma que o motor genérico do
    GUARACI, com EMSC, roda sem excecao sobre um espectro UV-Vis real
    (200-735nm) e captura sinal real (R2cal) prevendo DOC em esgoto
    bruto, agregado por dia (ver docstring do modulo para a metodologia
    de agregacao group-aware)."""
    from conftest import achar_pastas_run

    diario, cols_onda = _carregar_bruto()
    assert len(diario) >= N_DIAS_MINIMO, (
        f"so' {len(diario)} dias com par lab+sensor -- premissa do "
        f"dataset mudou (esperado >= {N_DIAS_MINIMO}).")

    saida = diario[["lab_doc_mg_l"] + cols_onda].copy()
    saida.insert(0, "classe", "esgoto_bruto")
    saida = saida.rename(columns={"lab_doc_mg_l": "conc"})

    csv = tmp_path / "eawag_uvvis_doc.csv"
    saida.to_csv(csv, index=False)

    cfg = pq.Config(
        mode="csv", csv_file=str(csv),
        class_column="classe", conc_column="conc",
        matrix_profile="generico",
        wn_min=float(cols_onda[0]), wn_max=float(cols_onda[-1]),
        objective="quantificacao", level="N3",
        output_root_folder=str(tmp_path / "saida"),
        group_by_mae_id=False, show_plots=False,
        default_preprocessing="custom",
        apply_snv=False, apply_sg=False, apply_airpls=False, apply_osc=False,
        apply_emsc=True, apply_mc=True,
        n_permutations=5, frac_holdout=0.25, seed=0,
    )
    pq.executar(cfg)   # nao deve lancar excecao

    runs = achar_pastas_run(cfg.output_root_folder)
    assert runs, "executar() nao criou saida"
    resumo = (Path(runs[0]) / pq.NOME_RELATORIOS / "resumo_modelo.txt").read_text(
        encoding="utf-8", errors="replace")

    r2cal = re.search(r"R2cal\s*\.*:\s*([\-\d.]+)", resumo)
    assert r2cal, f"R2cal nao encontrado no resumo:\n{resumo[:600]}"
    valor = float(r2cal.group(1))
    assert np.isfinite(valor), "R2cal nao-finito -- regressao degenerada"
    assert valor > R2CAL_MINIMO, (
        f"R2cal={valor:.3f} baixo demais para 'a calibracao capturou "
        f"algum sinal real' -- ver docstring do modulo para o que ESTE "
        f"teste garante.")
