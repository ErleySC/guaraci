"""formatos_sinteticos.py -- escritores de formatos de instrumento com dados SINTETICOS de verdade conhecida.

O produto nao distribui dado de terceiros: os exemplos (`guaraci exemplo`) e os testes dos leitores usam arquivos gerados aqui,
por codigo proprio, no layout que cada leitor documenta. Cada escritor devolve o caminho escrito. Determinismo: tudo vem de
`numpy.random.default_rng(seed)`. Sem dependencia nova (`scipy.io.netcdf_file` ja' e' do projeto).
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["escrever_andi_ms", "escrever_eem_dat", "gerar_cromatogramas_tic", "gerar_eem"]


def escrever_andi_ms(caminho: str, tempo_min: Union[Sequence[float], np.ndarray], intensidade: Union[Sequence[float], np.ndarray], *,
                     espectros: Optional[Sequence[Tuple[Sequence[float], Sequence[float]]]] = None) -> str:
    """ANDI-MS/netCDF classico minimo (as duas variaveis que `gcms_io.ler_tic_andi_ms` le): tempo em segundos e TIC.
    `espectros` (opcional): um par `(massas, intensidades)` por scan; grava o layout esparso do ASTM E1948
    (`mass_values`, `intensity_values`, `scan_index`, `point_count`) que `gcms_io.ler_ms_andi` le."""
    import scipy.io
    t = np.asarray(tempo_min, dtype=float) * 60.0
    y = np.asarray(intensidade, dtype=float)
    if t.shape != y.shape or t.ndim != 1:
        raise ValueError("tempo e intensidade precisam ser vetores do mesmo comprimento")
    if espectros is not None and len(espectros) != t.size:
        raise ValueError("espectros precisa de um par (massas, intensidades) por scan")
    os.makedirs(os.path.dirname(os.path.abspath(caminho)), exist_ok=True)
    f = scipy.io.netcdf_file(caminho, "w")
    try:
        f.createDimension("point_number", t.size)
        v = f.createVariable("scan_acquisition_time", "d", ("point_number",))
        v[:] = t
        v = f.createVariable("total_intensity", "d", ("point_number",))
        v[:] = y
        if espectros is not None:
            massas = np.concatenate([np.asarray(m, dtype=float) for m, _ in espectros])
            inten = np.concatenate([np.asarray(i, dtype=float) for _, i in espectros])
            cont = np.array([len(m) for m, _ in espectros], dtype=np.int32)
            ini = np.concatenate(([0], np.cumsum(cont)[:-1])).astype(np.int32)
            f.createDimension("ponto_espectro", massas.size)
            v = f.createVariable("mass_values", "d", ("ponto_espectro",))
            v[:] = massas
            v = f.createVariable("intensity_values", "d", ("ponto_espectro",))
            v[:] = inten
            v = f.createVariable("scan_index", "i", ("point_number",))
            v[:] = ini
            v = f.createVariable("point_count", "i", ("point_number",))
            v[:] = cont
    finally:
        f.close()
    return caminho


def escrever_eem_dat(caminho: str, excitacao: Union[Sequence[float], np.ndarray], emissao: Union[Sequence[float], np.ndarray], matriz: np.ndarray) -> str:
    """`.dat` de fluorimetro (separado por tabulacao): linha 1 = rotulo + excitacoes; 2 linhas de unidades; depois uma linha por
    emissao (emissao + intensidades), o layout que `eem_io.parse_eem_dat` le. `matriz[i, j]` = (emissao[i], excitacao[j])."""
    exc = np.asarray(excitacao, dtype=float)
    em = np.asarray(emissao, dtype=float)
    m = np.asarray(matriz, dtype=float)
    if m.shape != (em.size, exc.size):
        raise ValueError(f"matriz {m.shape} nao bate com (emissao, excitacao) = ({em.size}, {exc.size})")
    linhas = ["Wavelength\t" + "\t".join(f"{x:g}" for x in exc), "nm\t" + "\t".join("nm" for _ in exc), "\t"]
    linhas += [f"{e:g}\t" + "\t".join(f"{x:.6g}" for x in m[i]) for i, e in enumerate(em)]
    os.makedirs(os.path.dirname(os.path.abspath(caminho)), exist_ok=True)
    with open(caminho, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(linhas) + "\n")
    return caminho


def gerar_cromatogramas_tic(pasta: str, *, n_classes: int = 3, n_amostras: int = 6, seed: int = 31) -> Tuple[str, ...]:
    """`pasta/classe_k/amostra_i.cdf`: cromatogramas de TIC com 4 picos gaussianos; so' a razao entre 2 picos e o deslocamento do
    tempo de retencao carregam a classe (verdade conhecida). Devolve os caminhos escritos."""
    rng = np.random.default_rng(seed)
    t = np.linspace(2.0, 30.0, 700)
    escritos = []
    for k in range(n_classes):
        for i in range(n_amostras):
            razao = 1.0 + 0.8 * k
            desloc = rng.normal(0, 0.06)
            picos = ((8.0, 1.0), (14.0, razao), (19.0, 0.7), (24.5, 0.5 + 0.2 * k))
            y = sum(a * np.exp(-((t - (p + desloc)) / 0.35) ** 2) for p, a in picos) * 1.0e5
            y = y * rng.normal(1.0, 0.03) + rng.normal(0, 1.5e3, t.size)
            escritos.append(escrever_andi_ms(os.path.join(pasta, f"classe_{k + 1}", f"amostra_{i + 1:02d}.cdf"), t, y))
    return tuple(escritos)


def gerar_eem(pasta: str, *, n_classes: int = 3, n_amostras: int = 6, seed: int = 37) -> Tuple[str, ...]:
    """`pasta/classe_k/amostra_i.dat`: matrizes excitacao-emissao de 2 fluoroforos; a classe muda a proporcao entre eles (verdade
    conhecida, componentes de posto 2 + ruido, o caso em que a decomposicao PARAFAC recupera os perfis)."""
    rng = np.random.default_rng(seed)
    exc = np.arange(250.0, 451.0, 10.0)
    em = np.arange(300.0, 601.0, 5.0)
    perfil = []
    for cx, ce, sx, se in ((290.0, 340.0, 25.0, 30.0), (400.0, 520.0, 35.0, 45.0)):
        perfil.append(np.outer(np.exp(-((em - ce) / se) ** 2), np.exp(-((exc - cx) / sx) ** 2)))
    escritos = []
    for k in range(n_classes):
        for i in range(n_amostras):
            a = 1.0 + rng.normal(0, 0.05)
            b = (0.3 + 0.7 * k) * (1.0 + rng.normal(0, 0.05))
            m = 1.0e4 * (a * perfil[0] + b * perfil[1]) + rng.normal(0, 40.0, (em.size, exc.size))
            escritos.append(escrever_eem_dat(os.path.join(pasta, f"classe_{k + 1}", f"amostra_{i + 1:02d}.dat"), exc, em, m))
    return tuple(escritos)
