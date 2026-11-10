"""Escritores de formatos binarios de instrumento com dados SINTETICOS, so' para os testes dos leitores.

Nenhum arquivo de terceiros e' versionado: cada teste gera aqui o arquivo no layout que o leitor documenta, com valores de verdade
conhecida. Ler um arquivo real do fabricante (conformidade do formato) e' validacao do mantenedor, em `validacao/tests`
(`test_formatos_reais_locais.py`, com arquivos que o mantenedor guarda localmente e que nao fazem parte do repositorio).
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Sequence

import numpy as np


def escrever_spc(caminho, x_ini: float, x_fim: float, subarquivos: Sequence[Sequence[float]]) -> str:
    """SPC 'novo formato' (Galactic/Thermo), Y em float32, eixo X uniforme implicito de `x_ini` a `x_fim` (sem TXVALS).
    Mais de um subarquivo liga a flag TMULTI (varios Y compartilhando o X)."""
    ys = [np.asarray(s, dtype="<f4") for s in subarquivos]
    n = ys[0].size
    if any(y.size != n for y in ys):
        raise ValueError("todos os subarquivos precisam ter o mesmo numero de pontos")
    multi = len(ys) > 1
    flags = 0x10 if multi else 0x00                     # TMULTI
    cab = struct.pack("<BBBbIddIBBBB", flags, 0x4B, 0, -128, n, x_ini, x_fim, len(ys), 1, 1, 0, 0)   # fexp = 0x80: float32
    cab = cab.ljust(512, b"\x00")
    corpo = b""
    for k, y in enumerate(ys):
        sub = struct.pack("<BbHfffIIf", 0, 0, k, 0.0, 0.0, 0.0, n, 0, 0.0).ljust(32, b"\x00")
        corpo += sub + y.tobytes()
    Path(caminho).write_bytes(cab + corpo)
    return str(caminho)


def _bloco_sp(id_bloco: int, dados: bytes) -> bytes:
    return struct.pack("<Hi", id_bloco, len(dados)) + dados


def escrever_sp(caminho, x_ini: float, x_fim: float, y: Sequence[float]) -> str:
    """`.sp` PerkinElmer minimo: assinatura 'PEPE', descricao de 40 bytes, diretorio de blocos ate' o bloco terminal 122, bloco de
    metadados e os tres blocos que `parse_sp` usa (faixa do eixo, numero de pontos e espectro em float64)."""
    y = np.asarray(y, dtype="<f8")
    # diretorio: o bloco 5104 declara 12 bytes (os dois cabecalhos seguintes, 7000 e o terminal 122); os dados comecam logo depois
    diretorio = (struct.pack("<Hi", 5000, 0) + struct.pack("<Hi", 5104, 12) + struct.pack("<Hi", 7000, 0) + struct.pack("<Hi", 122, 0))
    blocos = (_bloco_sp(35698, struct.pack("<H", 29981) + struct.pack("<dd", x_ini, x_fim))
              + _bloco_sp(35701, struct.pack("<H", 29995) + struct.pack("<I", y.size))
              + _bloco_sp(35708, struct.pack("<H", 29974) + struct.pack("<I", y.nbytes) + y.tobytes()))
    Path(caminho).write_bytes(b"PEPE" + b"\x00" * 40 + diretorio + blocos)
    return str(caminho)


def escrever_rmn_bruker(pasta, ppm_ini: float, ppm_fim: float, y: Sequence[float], sf_mhz: float = 400.0) -> str:
    """`pdata/1/` Bruker com `procs` e o espectro real `1r` (int32), mais o `acqus` do experimento. Devolve a pasta `pdata/1`
    (o argumento de `parse_rmn_bruker`). O eixo vai de `ppm_ini` (esquerda) a `ppm_fim` (direita), decrescente como no TopSpin."""
    y = np.asarray(y)
    n = y.size
    raiz = Path(pasta)
    pd = raiz / "pdata" / "1"
    pd.mkdir(parents=True, exist_ok=True)
    sw_hz = (ppm_ini - ppm_fim) * sf_mhz * n / (n - 1) if n > 1 else 1.0
    o1_hz = (ppm_ini + ppm_fim) / 2.0 * sf_mhz           # portadora no centro da janela (o leitor deriva o eixo dela)
    procs = "\n".join([
        "##TITLE= Parameter file, sintetico", "##JCAMPDX= 5.0", "##DATATYPE= Parameter Values",
        f"##$SI= {n}", f"##$SW_p= {sw_hz:.6f}", f"##$SF= {sf_mhz:.6f}", f"##$OFFSET= {ppm_ini:.6f}",
        "##$BYTORDP= 0", "##$DTYPP= 0", "##$NC_proc= 0", "##$AXNUC= <1H>", "##END=", ""])
    acqus = "\n".join([
        "##TITLE=Parameter List", "##JCAMPDX=4.24", "##DATATYPE=Parameter Values", "##$BYTORDA=0", f"##$TD={2 * n}",
        f"##$SW_h={sw_hz:.6f}", f"##$O1={o1_hz:.6f}", f"##$SFO1={sf_mhz + o1_hz / 1e6:.9f}", f"##$BF1={sf_mhz:.6f}", "##$NUC1=<1H>",
        "##$AQ_mod=3", "##END=", ""])
    (pd / "procs").write_text(procs, encoding="ascii")
    (raiz / "acqus").write_text(acqus, encoding="ascii")
    (pd / "1r").write_bytes(np.asarray(y, dtype="<i4").tobytes())
    return str(pd)


def instalar_rainbow_falso(monkeypatch, canais_por_detector) -> None:
    """Instala um modulo `rainbow` falso com a MESMA superficie que `parse_cromatograma_hplc` usa (`read`, `by_detector`,
    `detectors`, `xlabels`, `data`). `canais_por_detector`: {detector: [(tempo, intensidade), ...]}. `read` recusa o que nao e'
    diretorio, como o pacote real recusa um diretorio de fabricante invalido."""
    import os
    import sys
    from types import SimpleNamespace

    def read(pasta):
        if not os.path.isdir(pasta):
            raise FileNotFoundError(pasta)
        por_det = {d: [SimpleNamespace(xlabels=np.asarray(x, dtype=float), data=np.asarray(y, dtype=float).reshape(-1, 1))
                       for x, y in canais] for d, canais in canais_por_detector.items()}
        return SimpleNamespace(by_detector=por_det, detectors=set(por_det))

    monkeypatch.setitem(sys.modules, "rainbow", SimpleNamespace(read=read))

