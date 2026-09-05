# -*- coding: utf-8 -*-
"""gcms_io.py -- Leitor de GC-MS bruto no formato ANDI-MS/NetCDF (ASTM
E1948), Passo 150 da auditoria das 11 tecnicas analiticas (Fase D,
2026-09-04).

Formato confirmado por leitura DIRETA de arquivos reais (dataset
Mendeley `10.17632/pgkrc7wyj4.1`, Lavandula angustifolia, 55 amostras
comerciais): `.CDF` e' netCDF **classico** (netCDF-3, nao HDF5) --
`scipy.io.netcdf_file` (ja' dependencia do projeto via scipy, ZERO
dependencia nova) le' direto, sem precisar de `netCDF4` nem `pyms`
(avaliados antes de decidir -- ver Passo 150 em `docs/PROGRESSO.md`
para o comparativo de licenca/maturidade).

Variaveis ANDI-MS usadas (nomes padrao do formato, confirmados nos
arquivos reais): `scan_acquisition_time` (segundos, 1 por scan),
`total_intensity` (TIC, 1 por scan). O espectro de massa completo por
scan (`mass_values`/`intensity_values`/`scan_index`/`point_count`,
formato esparso -- so' pares massa/intensidade != 0 sao armazenados)
NAO e' extraido aqui -- o Passo 150 usa so' o TIC (cromatograma de
corrente ionica total) para alinhamento de tempo de retencao; extrair
espectro de massa completo por scan fica como extensao futura, se um
passo futuro pedir deconvolucao espectral (AMDIS)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

__all__ = ["CromatogramaTIC", "ler_tic_andi_ms", "carregar_dataset_gcms"]


@dataclass
class CromatogramaTIC:
    """`tempo_min` (minutos, crescente) e `intensidade` (contagens TIC
    brutas do instrumento, sem normalizacao) -- mesmo comprimento."""
    tempo_min: np.ndarray
    intensidade: np.ndarray


def ler_tic_andi_ms(caminho: "str | Path") -> CromatogramaTIC:
    """Le `scan_acquisition_time`/`total_intensity` de um arquivo
    ANDI-MS/netCDF classico. Levanta `ValueError` se as variaveis
    esperadas nao existirem (formato nao e' o assumido) ou se os dois
    vetores nao tiverem o mesmo comprimento -- nunca completa/trunca em
    silencio."""
    import scipy.io

    arquivo = scipy.io.netcdf_file(str(caminho), mmap=False)
    faltando = [v for v in ("scan_acquisition_time", "total_intensity")
                if v not in arquivo.variables]
    if faltando:
        raise ValueError(
            f"{caminho}: variavel(is) ANDI-MS ausente(s): {faltando} -- "
            f"nao e' um ANDI-MS/netCDF valido ou o esquema mudou.")

    tempo_s = np.asarray(arquivo.variables["scan_acquisition_time"][:], dtype=float)
    intensidade = np.asarray(arquivo.variables["total_intensity"][:], dtype=float)
    if tempo_s.shape != intensidade.shape:
        raise ValueError(
            f"{caminho}: scan_acquisition_time ({tempo_s.shape}) e "
            f"total_intensity ({intensidade.shape}) com formas diferentes.")
    return CromatogramaTIC(tempo_min=tempo_s / 60.0, intensidade=intensidade.copy())


def carregar_dataset_gcms(pasta: "str | Path", padrao: str = "*.CDF"
                           ) -> Dict[str, CromatogramaTIC]:
    """Carrega todos os arquivos `padrao` (case-sensitive no glob, mas
    tenta maiusculo e minusculo -- instrumentos diferentes exportam com
    extensao `.CDF` ou `.cdf`) de `pasta`, retorna
    `{nome_arquivo_sem_extensao: CromatogramaTIC}`. Levanta `ValueError`
    se nenhum arquivo for encontrado -- nunca devolve dataset vazio em
    silencio."""
    raiz = Path(pasta)
    candidatos: List[Path] = sorted(set(raiz.glob(padrao)) | set(raiz.glob(padrao.lower())))
    if not candidatos:
        raise ValueError(f"{raiz}: nenhum arquivo '{padrao}' encontrado")

    return {c.stem: ler_tic_andi_ms(c) for c in candidatos}
