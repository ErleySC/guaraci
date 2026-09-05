# -*- coding: utf-8 -*-
"""Validacao de GC-MS contra dataset publico Mendeley
`10.17632/pgkrc7wyj4.1` -- Passo 150 (Fase D da auditoria das 11
tecnicas, 2026-09-04).

Dataset: 55 amostras COMERCIAIS de oleo essencial de Lavandula
angustifolia (Pokajewicz 2024), usadas no estudo original para
deteccao de adulteracao. `L01.CDF`...`L55.CDF` -- GC-MS de varredura
completa, formato ANDI-MS/netCDF classico (ASTM E1948). Licenca **CC BY
4.0**.

BIBLIOTECAS AVALIADAS ANTES DE ESCREVER PARSER DO ZERO (exigido pela
instrucao): `netCDF4` e `pyms` (PyMassSpec) sao as opcoes maduras
citadas na literatura para ler ANDI-MS -- mas os arquivos reais deste
dataset se confirmaram netCDF CLASSICO (nao HDF5), formato que
`scipy.io.netcdf_file` **ja' le direto** -- scipy JA' e' dependencia
deste projeto, entao **zero dependencia nova** foi necessaria (nem
`netCDF4`, nem `pyms`, com toda a superficie de licenca/manutencao que
viriam junto). Ver `src/guaraci/gcms_io.py` para a implementacao.

ESCOPO DESTA VALIDACAO (decisao explicita, nao escondida): o dataset
bruto (1020 arquivos, so' `.CDF`/`.dat`) NAO inclui uma tabela de
referencia com rotulo de autenticidade/adulteracao por amostra -- esse
rotulo, se existir, esta' no artigo companheiro (nao no repositorio de
dados), e reconstrui-lo exigiria identificacao de compostos por indice
de retencao (trabalho de escopo proprio, maior que "escrever um
parser"). Por isso esta validacao NAO tenta classificar/quantificar
adulteracao -- valida a parte que O PASSO PEDE de fato: o parser (TIC
real extraido de 55 arquivos binarios reais) e o alinhamento COW
(Nielsen, Carstensen & Smedsgaard 1998, `alinhamento_retencao.py`),
medindo a melhora de correlacao cruzada ANTES/DEPOIS do alinhamento --
mesmo espirito "antes/depois" ja usado para validar PDS/DS no Corn.
Group-aware nao se aplica aqui: cada arquivo GC-MS e' 1 injecao por
amostra, sem repeticao a proteger."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

#: Medido em 2026-09-04 contra os 55 arquivos reais: corr. media par-a-
#: par 0,753 (antes) -> 0,884 (depois, referencia=L01, 30 segmentos,
#: slack=10, grade de 1200 pontos). Piso com folga generosa.
MELHORA_CORR_MINIMA = 0.05
CORR_DEPOIS_MINIMA = 0.75


def _pasta_gcms():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "mendeley_pgkrc7wyj4"
    return pasta if (pasta / "L01.CDF").is_file() else None


requer_gcms_lavanda = pytest.mark.skipif(
    _pasta_gcms() is None,
    reason=("arquivos L01.CDF..L55.CDF (Mendeley pgkrc7wyj4) ausentes. "
            "Baixe com 'python scripts/download_datasets/"
            "baixar_mendeley_gcms_lavanda.py' e aponte "
            "GUARACI_DATASETS_DIR para a pasta que contem "
            "mendeley_pgkrc7wyj4/."))


def _corr_media_pares(matriz: np.ndarray, n_pares: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = matriz.shape[0]
    pares = [(i, j) for i in range(n) for j in range(i + 1, n)]
    idx = rng.choice(len(pares), size=min(n_pares, len(pares)), replace=False)
    return float(np.mean([np.corrcoef(matriz[i], matriz[j])[0, 1]
                           for i, j in (pares[k] for k in idx)]))


@requer_gcms_lavanda
def test_carrega_tic_dos_55_arquivos_gcms_reais():
    """Contra-prova do parser (regra 9): confirma que os 55 arquivos
    ANDI-MS/netCDF reais sao lidos, produzem TIC com formato/faixa de
    tempo plausiveis, e que `scipy.io.netcdf_file` (zero dependencia
    nova) da' conta do formato sem precisar de netCDF4/pyms."""
    from guaraci.gcms_io import carregar_dataset_gcms

    dados = carregar_dataset_gcms(_pasta_gcms())
    assert len(dados) == 55, f"esperava 55 amostras, achou {len(dados)}"

    for nome, cromatograma in dados.items():
        assert cromatograma.tempo_min.ndim == 1
        assert cromatograma.intensidade.shape == cromatograma.tempo_min.shape
        assert np.all(np.diff(cromatograma.tempo_min) > 0), (
            f"{nome}: tempo de retencao nao-crescente -- formato inesperado")
        assert cromatograma.tempo_min.min() < 10, (
            f"{nome}: inicio da corrida > 10 min, fora do esperado para "
            f"este metodo de GC-MS")
        assert cromatograma.intensidade.max() > 0


@requer_gcms_lavanda
@pytest.mark.slow
def test_cow_melhora_alinhamento_entre_as_55_amostras_reais():
    """Passo 150: conecta o parser real ao COW (ate aqui so' provado
    por contra-prova sintetica, `tests/test_alinhamento_retencao.py`) e
    mede, contra as 55 amostras REAIS, se o alinhamento de retencao
    melhora a correlacao cruzada entre cromatogramas -- a mesma logica
    "antes/depois" usada para validar PDS/DS no Corn (Passo 135), so
    que sem rotulo supervisionado (nao ha' um na base bruta -- ver
    docstring do modulo)."""
    from guaraci.gcms_io import carregar_dataset_gcms
    from guaraci.alinhamento_retencao import cow

    dados = carregar_dataset_gcms(_pasta_gcms())
    ids = sorted(dados.keys(), key=lambda s: int(s[1:]))

    # Janela de tempo comum SEGURA (intersecao) -- diferentes lotes do
    # dataset tem duracao de corrida ligeiramente diferente (~75 vs
    # ~82 min); usar so' a intersecao evita extrapolar.
    t_ini = max(dados[i].tempo_min.min() for i in ids)
    t_fim = min(dados[i].tempo_min.max() for i in ids)
    assert t_fim - t_ini > 30, "janela de tempo comum suspeita demais"

    n_grade = 1200
    grade = np.linspace(t_ini, t_fim, n_grade)
    matriz = np.zeros((len(ids), n_grade))
    for i, nome in enumerate(ids):
        c = dados[nome]
        matriz[i] = np.interp(grade, c.tempo_min, c.intensidade)

    corr_antes = _corr_media_pares(matriz)

    referencia = matriz[0]
    matriz_alinhada = np.empty_like(matriz)
    matriz_alinhada[0] = referencia
    for i in range(1, len(ids)):
        resultado = cow(referencia, matriz[i], n_segmentos=30, slack=10)
        matriz_alinhada[i] = resultado.amostra_alinhada

    corr_depois = _corr_media_pares(matriz_alinhada)

    print(f"\n[Passo 150] Correlacao media par-a-par entre as 55 amostras "
          f"reais: antes={corr_antes:.3f}, depois do COW={corr_depois:.3f}")

    assert corr_depois > corr_antes + MELHORA_CORR_MINIMA, (
        f"COW deveria melhorar a correlacao cruzada entre cromatogramas "
        f"reais: antes={corr_antes:.3f}, depois={corr_depois:.3f}")
    assert corr_depois > CORR_DEPOIS_MINIMA
