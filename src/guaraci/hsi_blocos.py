"""hsi_blocos.py -- Cubo hiperespectral lido SOB DEMANDA (np.memmap) e
processamento em blocos de linhas.

POR QUE EXISTE (R11-d, fase 2): cubos de camera real (640x1000x224 float32 =
570 MB cada) nao cabem em dezenas na memoria, e o fluxo antigo ainda os
copiava para float64 no controle de qualidade, na segmentacao e na extracao
de pixels. `CuboHSI` guarda so' o caminho do binario e os parametros do
ENVI; cada etapa le BLOCOS DE LINHAS (float32) pelo memmap e descarta. O
memmap e' recriado a cada bloco e solto em seguida, para que o RSS nao
acumule as paginas ja' lidas do arquivo.

`CuboHSI` imita o que as etapas HSI precisam de um array 3D
`(altura, largura, bandas)`: `shape`, `ndim`, `blocos()`, `pixels()` e
`np.asarray(cubo)` (materializa tudo em float32, para o codigo que ainda
espera um ndarray). A calibracao branco/escuro (R = (I - D)/(W - D)) e a
reamostragem de bandas sao aplicadas no proprio bloco, de modo que o
chamador nunca ve o valor bruto.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np

__all__ = ["CalibracaoBrancoEscuro", "CuboHSI", "iterar_blocos", "extrair_pixels",
           "aplicar_calibracao", "reamostrar_bandas", "pesos_reamostragem"]

#: Alvo de memoria de UM bloco de linhas (float32), em bytes.
_BYTES_BLOCO = 48 * 1024 * 1024


@dataclass
class CalibracaoBrancoEscuro:
    """Referencias ja' reduzidas a `(colunas|1, bandas)`: media sobre as
    linhas dos cubos de referencia. `denominador` = W - D, com NaN onde
    W <= D (calibracao indefinida ali, nunca divisao por zero em silencio)."""
    escuro: np.ndarray
    denominador: np.ndarray
    origem: str = ""


def aplicar_calibracao(bloco: np.ndarray, calib: CalibracaoBrancoEscuro) -> np.ndarray:
    """R = (I - D)/(W - D) em um bloco `(linhas, colunas, bandas)` (float32).
    `calib` tem forma `(colunas, bandas)` (referencia por coluna) ou
    `(1, bandas)` (espectro medio unico)."""
    return ((bloco - calib.escuro[None, :, :]) / calib.denominador[None, :, :]).astype(
        np.float32, copy=False)


def pesos_reamostragem(eixo_origem: np.ndarray, eixo_destino: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices `(i0, i1)` e peso `f` da interpolacao linear do eixo de origem
    nos pontos do eixo de destino: y_dest = y[i0]*(1-f) + y[i1]*f."""
    o = np.asarray(eixo_origem, dtype=float)
    d = np.asarray(eixo_destino, dtype=float)
    i1 = np.clip(np.searchsorted(o, d, side="left"), 1, len(o) - 1)
    i0 = i1 - 1
    f = np.clip((d - o[i0]) / np.where(o[i1] - o[i0] == 0, 1.0, o[i1] - o[i0]), 0.0, 1.0)
    return i0, i1, f.astype(np.float32)


def reamostrar_bandas(bloco: np.ndarray, pesos: Tuple[np.ndarray, np.ndarray, np.ndarray]
                      ) -> np.ndarray:
    i0, i1, f = pesos
    return (bloco[..., i0] * (1.0 - f) + bloco[..., i1] * f).astype(np.float32, copy=False)


class CuboHSI:
    """Cubo ENVI sob demanda. Construido por `hsi_io`; ver docstring do modulo."""

    def __init__(self, caminho: str, dtype_np: np.dtype, interleave: str,
                 n_lines: int, n_samples: int, n_bands: int, header_offset: int = 0,
                 fator: float = 1.0, calibracao: Optional[CalibracaoBrancoEscuro] = None,
                 reamostragem: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                 n_bands_saida: Optional[int] = None) -> None:
        self.caminho = caminho
        self._dtype = np.dtype(dtype_np)
        self.interleave = interleave
        self.n_lines, self.n_samples, self.n_bands = int(n_lines), int(n_samples), int(n_bands)
        self.header_offset = int(header_offset)
        self.fator = float(fator)
        self.calibracao = calibracao
        self.reamostragem = reamostragem
        self._nb_saida = int(n_bands_saida) if n_bands_saida else self.n_bands

    # -- aparencia de ndarray -------------------------------------------------
    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.n_lines, self.n_samples, self._nb_saida)

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float32)

    @property
    def size(self) -> int:
        return self.n_lines * self.n_samples * self._nb_saida

    def __len__(self) -> int:
        return self.n_lines

    # -- leitura --------------------------------------------------------------
    def _abrir(self) -> np.ndarray:
        """memmap somente-leitura visto como `(linhas, colunas, bandas)`."""
        L, S, B = self.n_lines, self.n_samples, self.n_bands
        if self.interleave == "bip":
            return np.memmap(self.caminho, dtype=self._dtype, mode="r",
                             offset=self.header_offset, shape=(L, S, B))
        if self.interleave == "bil":
            return np.memmap(self.caminho, dtype=self._dtype, mode="r",
                             offset=self.header_offset, shape=(L, B, S)).transpose(0, 2, 1)
        return np.memmap(self.caminho, dtype=self._dtype, mode="r",
                         offset=self.header_offset, shape=(B, L, S)).transpose(1, 2, 0)

    def _tratar(self, bruto: np.ndarray, colunas=None) -> np.ndarray:
        """float32 + fator de reflectancia + calibracao + reamostragem."""
        b = np.array(bruto, dtype=np.float32)          # copia: solta o memmap
        if self.fator not in (0.0, 1.0):
            b /= np.float32(self.fator)
        if self.calibracao is not None:
            c = self.calibracao
            if colunas is not None and c.escuro.shape[0] > 1:
                b = ((b - c.escuro[colunas]) / c.denominador[colunas]).astype(np.float32, copy=False)
            else:
                b = aplicar_calibracao(b, c) if b.ndim == 3 else (
                    (b - c.escuro[0]) / c.denominador[0]).astype(np.float32, copy=False)
        if self.reamostragem is not None:
            b = reamostrar_bandas(b, self.reamostragem)
        return b

    def bloco(self, r0: int, r1: int) -> np.ndarray:
        mm = self._abrir()
        try:
            return self._tratar(mm[r0:r1])
        finally:
            del mm

    def blocos(self, n_linhas: Optional[int] = None) -> Iterator[Tuple[int, int, np.ndarray]]:
        """Gera `(r0, r1, bloco float32 (r1-r0, colunas, bandas))`."""
        if n_linhas is None:
            n_linhas = max(1, _BYTES_BLOCO // max(1, self.n_samples * max(self.n_bands, self._nb_saida) * 4))
        for r0 in range(0, self.n_lines, n_linhas):
            r1 = min(self.n_lines, r0 + n_linhas)
            yield r0, r1, self.bloco(r0, r1)

    def pixels(self, linhas: np.ndarray, colunas: np.ndarray) -> np.ndarray:
        """Espectros `(n, bandas)` (float32) dos pixels (linha, coluna)."""
        linhas = np.asarray(linhas)
        colunas = np.asarray(colunas)
        mm = self._abrir()
        try:
            ordem = np.argsort(linhas, kind="stable")   # leitura sequencial no arquivo
            bruto = np.empty((len(linhas), self.n_bands), dtype=np.float32)
            bruto[ordem] = mm[linhas[ordem], colunas[ordem]]
        finally:
            del mm
        return self._tratar(bruto, colunas=colunas)

    def __array__(self, dtype=None, copy=None):
        out = np.empty(self.shape, dtype=np.float32)
        for r0, r1, b in self.blocos():
            out[r0:r1] = b
        return out if dtype is None else out.astype(dtype, copy=False)

    def __getitem__(self, idx):
        return np.asarray(self)[idx]


def iterar_blocos(cubo, n_linhas: Optional[int] = None) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Blocos de linhas de um `CuboHSI` OU de um ndarray `(H, W, B)`."""
    if isinstance(cubo, CuboHSI):
        yield from cubo.blocos(n_linhas)
        return
    cubo = np.asarray(cubo)
    if cubo.ndim != 3:
        raise ValueError(f"esperado um cubo 3D (altura, largura, bandas), recebeu shape {cubo.shape}.")
    h, w, b = cubo.shape
    if n_linhas is None:
        n_linhas = max(1, _BYTES_BLOCO // max(1, w * b * 4))
    for r0 in range(0, h, n_linhas):
        r1 = min(h, r0 + n_linhas)
        yield r0, r1, cubo[r0:r1]


def extrair_pixels(cubo, linhas: np.ndarray, colunas: np.ndarray) -> np.ndarray:
    """Espectros dos pixels (linha, coluna) de um `CuboHSI` ou ndarray."""
    if isinstance(cubo, CuboHSI):
        return cubo.pixels(linhas, colunas)
    return np.asarray(cubo)[linhas, colunas]
