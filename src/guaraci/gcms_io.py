# -*- coding: utf-8 -*-
"""gcms_io.py -- Leitor de GC-MS bruto no formato ANDI-MS/NetCDF (ASTM
E1948), Passo 150 da auditoria das 11 tecnicas analiticas (Fase D,
2026-09-04).

Formato confirmado por leitura DIRETA de arquivos reais de referencia:
`.CDF` e' netCDF **classico** (netCDF-3, nao HDF5) --
`scipy.io.netcdf_file` (ja' dependencia do projeto via scipy, ZERO
dependencia nova) le' direto, sem precisar de `netCDF4` nem `pyms`
(avaliados antes de decidir pelo criterio de licenca e maturidade).

Variaveis ANDI-MS usadas (nomes padrao do formato, confirmados nos
arquivos reais): `scan_acquisition_time` (segundos, 1 por scan),
`total_intensity` (TIC, 1 por scan). O espectro de massa completo por
scan (`mass_values`/`intensity_values`/`scan_index`/`point_count`,
formato esparso -- so' pares massa/intensidade != 0 sao armazenados)
NAO e' extraido por `ler_tic_andi_ms` (so' o TIC, cromatograma de corrente
ionica total, usado no alinhamento de tempo de retencao). O espectro por
scan opcional vem de `ler_ms_andi`: matriz tempo x m/z, com a massa
agregada em UNIDADES de massa (arredondada ao inteiro mais proximo, soma das
intensidades). NAO ha' deconvolucao espectral (AMDIS) aqui.

VALIDACAO: o layout esparso foi implementado do padrao ASTM E1948 e testado com arquivo sintetico escrito nesse
layout (verdade conhecida) e, desde 2026-11-10, com 8 arquivos ANDI-MS reais publicos de GC-MS (projeto PyMassSpec): o TIC
do arquivo e' a soma das intensidades do scan (exato em 7, 1.6e-6 relativo no 8o) e a matriz tempo x m/z bate com um laco
independente. Todos tinham `scale_factor` 1 ou ausente: o caso escalado so' foi testado com arquivo sintetico."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

__all__ = ["CromatogramaTIC", "ler_tic_andi_ms", "ler_ms_andi", "carregar_dataset_gcms"]


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


def _com_escala(var) -> np.ndarray:
    """Valores da variavel netCDF com `scale_factor`/`add_offset` aplicados quando existem (E1948 permite
    guardar massas/intensidades como inteiros escalados; `scipy.io.netcdf_file` nao aplica sozinho)."""
    v = np.asarray(var[:], dtype=float)
    fator = getattr(var, "scale_factor", None)
    desloc = getattr(var, "add_offset", None)
    if fator is not None:
        v = v * float(np.asarray(fator).ravel()[0])
    if desloc is not None:
        v = v + float(np.asarray(desloc).ravel()[0])
    return v


def ler_ms_andi(caminho: "str | Path", canal: str = "todos"):
    """Espectro de massas por scan de um ANDI-MS/netCDF classico (`mass_values`, `intensity_values`,
    `scan_index`, `point_count`), agregado em UNIDADES de massa.

    Devolve `(tempo_min, matriz, rotulos)`; `matriz[i, j]` = soma das intensidades do scan `i` cujas massas
    arredondadas valem `rotulos[j]` (m/z inteiro, como texto). `canal="todos"`: toda a faixa de m/z do arquivo
    (colunas densas do menor ao maior m/z); ou lista de m/z (`"57,71,85"`, cromatogramas de ion extraido). Um
    m/z fora da faixa do arquivo e' `ValueError`, nunca uma coluna zerada em silencio.

    Levanta `ValueError` se as variaveis do espectro por scan nao existirem, se `scan_index`/`point_count`
    forem inconsistentes com `mass_values`, ou se o numero de scans nao bater com o tempo."""
    import re
    import scipy.io

    arquivo = scipy.io.netcdf_file(str(caminho), mmap=False)
    try:
        faltando = [v for v in ("scan_acquisition_time", "mass_values", "intensity_values",
                                "scan_index", "point_count") if v not in arquivo.variables]
        if faltando:
            raise ValueError(
                f"{caminho}: sem espectro de massas por scan (variavel(is) ausente(s): {faltando}). "
                f"Deixe cromatografia_canal vazio para ler so' o TIC.")
        tempo_s = np.asarray(arquivo.variables["scan_acquisition_time"][:], dtype=float)
        massa = _com_escala(arquivo.variables["mass_values"])
        inten = _com_escala(arquivo.variables["intensity_values"])
        inicio = np.asarray(arquivo.variables["scan_index"][:], dtype=np.int64)
        n_pt = np.asarray(arquivo.variables["point_count"][:], dtype=np.int64)
    finally:
        arquivo.close()
    n_scan = tempo_s.size
    if inicio.size != n_scan or n_pt.size != n_scan or massa.shape != inten.shape:
        raise ValueError(
            f"{caminho}: scan_index ({inicio.size}), point_count ({n_pt.size}) e tempo ({n_scan}) ou "
            f"mass_values {massa.shape} / intensity_values {inten.shape} inconsistentes.")
    if n_scan == 0 or massa.size == 0:
        raise ValueError(f"{caminho}: espectro de massas vazio.")
    if np.any(inicio < 0) or np.any(inicio + n_pt > massa.size):
        raise ValueError(f"{caminho}: scan_index/point_count apontam para fora de mass_values.")
    scan_do_ponto = np.repeat(np.arange(n_scan), n_pt)
    pos = np.concatenate([np.arange(a, a + n) for a, n in zip(inicio, n_pt)]) if n_pt.sum() else np.array([], int)
    mz = np.rint(massa[pos]).astype(np.int64)
    inten_pt = inten[pos]
    mz_min, mz_max = int(mz.min()), int(mz.max())
    matriz = np.zeros((n_scan, mz_max - mz_min + 1), dtype=float)
    np.add.at(matriz, (scan_do_ponto, mz - mz_min), inten_pt)
    rotulos = [str(m) for m in range(mz_min, mz_max + 1)]
    if str(canal).strip().lower() not in ("todos", "all", "ms"):
        alvos = []
        for t in re.split(r"[,;\s]+", str(canal).strip()):
            if t:
                try:
                    alvos.append(int(round(float(t.replace(",", ".")))))
                except ValueError as e:
                    raise ValueError(f"cromatografia_canal='{canal}': use 'todos' ou m/z separados por "
                                     f"virgula (ex.: 57,71,85).") from e
        ausentes = [a for a in alvos if not (mz_min <= a <= mz_max)]
        if ausentes or not alvos:
            raise ValueError(f"{caminho}: m/z {ausentes or canal} fora da faixa do arquivo ({mz_min}-{mz_max}).")
        matriz = matriz[:, [a - mz_min for a in alvos]]
        rotulos = [str(a) for a in alvos]
    return tempo_s / 60.0, matriz, rotulos
