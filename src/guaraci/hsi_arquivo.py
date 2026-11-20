# -*- coding: utf-8 -*-
"""hsi_arquivo.py -- cubo hiperespectral guardado num arquivo de variaveis (MATLAB .mat v5/v7.3,
NumPy .npy/.npz, HDF5 .h5), alem do ENVI .hdr+.bin de `hsi_io.py`.

O que se detecta sozinho e o que nao:
  - O CUBO: a unica variavel numerica com 3 eixos maiores que 1. Se houver mais de uma (ex.:
    cubo + branco + escuro no mesmo arquivo), descarta as de nome de referencia (white/dark/
    branco/escuro/ref); se ainda sobrar mais de uma, PARA e lista as candidatas: o usuario
    escolhe com `arquivo_variavel_dados`.
  - O EIXO ESPECTRAL (comprimentos de onda): um vetor monotono cujo tamanho bate com um dos
    eixos do cubo -- de preferencia com nome de eixo (wavelength, wl, lambda, bands, nm...),
    ou o atributo `wavelength(s)` do proprio dataset HDF5. `arquivo_variavel_eixo` escolhe.
  - QUAL EIXO DO CUBO E' O ESPECTRAL: o que tem o tamanho do vetor de comprimentos de onda.
    Sem vetor, assume o ULTIMO (convencao do MATLAB e do ENVI BIP: linhas x colunas x bandas)
    e avisa; se a forma sugere bandas x linhas x colunas (os dois ultimos iguais e o primeiro
    diferente) ou o vetor bate com mais de um eixo, PARA e pede `hsi_eixo_espectral`.
Nunca se adivinha em silencio: toda suposicao vai no log e no `info` devolvido.

O cubo sai em float32 na forma (linhas, colunas, bandas), o mesmo contrato de
`hsi_io.load_envi_cube`. ASCII por convencao do projeto.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from guaraci import arquivo_matrizes as am

__all__ = ["CuboArquivo", "EXTENSOES_CUBO_ARQUIVO", "EIXOS_ESPECTRAIS", "ler_cubo_arquivo",
           "referencias_no_arquivo", "ler_referencia_arquivo"]

EXTENSOES_CUBO_ARQUIVO: Tuple[str, ...] = am.EXTENSOES_MATRIZ

#: Valores de `hsi_eixo_espectral` (posicao das bandas na forma que o programa de origem mostra).
EIXOS_ESPECTRAIS: Tuple[str, ...] = ("auto", "ultimo", "primeiro", "meio")
_POS = {"primeiro": 0, "meio": 1, "ultimo": 2}
_NOME_POS = {0: "primeiro", 1: "meio", 2: "ultimo"}

_RE_REFERENCIA = re.compile(r"(^|_)(white|dark|branco|escuro)|^[wd]_?ref$", re.IGNORECASE)
#: Nome (ultimo componente) de variavel de referencia DENTRO do arquivo do cubo.
_RE_REF_TIPO = {"white": re.compile(r"^(white|branco)(_?ref(erence|erencia)?)?$|^(white|branco)_?(ref|padrao)|^w_?ref$",
                                    re.IGNORECASE),
                "dark": re.compile(r"^(dark|escuro)(_?ref(erence|erencia)?)?$|^(dark|escuro)_?(ref|padrao)|^d_?ref$",
                                   re.IGNORECASE)}


@dataclass
class CuboArquivo:
    cubo: np.ndarray                         # float32 (linhas, colunas, bandas)
    wavelengths: Optional[np.ndarray]        # None = so' indice de banda
    info: Dict[str, Any] = field(default_factory=dict)


def _forma_txt(forma) -> str:
    return " x ".join(str(int(d)) for d in forma)


def _escolher_cubo(caminho: str, variaveis: List[am.VariavelArquivo], nome: str) -> am.VariavelArquivo:
    base = os.path.basename(caminho)
    if nome:
        v = am.achar_variavel(caminho, variaveis, nome, "arquivo_variavel_dados")
        if len(v.forma_util) != 3:
            raise ValueError(f"{base}: a variavel {v.descricao()} nao e' um cubo (precisa de 3 eixos: linhas, "
                             f"colunas e bandas).")
        if not v.numerica:
            raise ValueError(f"{base}: a variavel {v.descricao()} e' uma mascara logica, nao um cubo de dados.")
        return v
    cands = [v for v in variaveis if v.numerica and len(v.forma_util) == 3]
    if not cands:
        planas = [v for v in variaveis if v.numerica and len(v.forma_util) == 2]
        dica = ""
        if planas:
            dica = (f" Ha' matriz(es) 2D ({am.descrever_variaveis(planas, 4)}): se for o cubo desdobrado em "
                    f"pixels x bandas, o arquivo nao guarda as dimensoes da imagem -- no MATLAB, refaca o cubo "
                    f"com reshape(X, linhas, colunas, bandas) e salve de novo. Se for uma tabela de espectros "
                    f"(amostras x variaveis), use o modo de tabela (modo_entrada: csv) em vez do HSI.")
        raise ValueError(f"{base}: nenhuma variavel com 3 eixos (linhas x colunas x bandas). Variaveis: "
                         f"{am.descrever_variaveis(variaveis)}.{dica}")
    if len(cands) > 1:
        sem_ref = [v for v in cands if not _RE_REFERENCIA.search(v.nome.split(".")[-1])]
        if len(sem_ref) == 1:
            return sem_ref[0]
        raise ValueError(
            f"{base}: ha' {len(cands)} variaveis com 3 eixos e nao da' para saber qual e' o cubo da amostra: "
            f"{am.descrever_variaveis(cands)}. Diga qual e' com arquivo_variavel_dados (ex.: "
            f"arquivo_variavel_dados: {cands[0].nome}).")
    return cands[0]


def _eixo_do_atributo(v: am.VariavelArquivo) -> Optional[np.ndarray]:
    for k in ("wavelength", "wavelengths", "Wavelength", "Wavelengths", "wl", "bands"):
        if k in v.atributos:
            a = np.asarray(v.atributos[k], dtype=float).ravel()
            if am.eixo_monotono(a):
                return a
    return None


def _escolher_eixo(caminho: str, variaveis: List[am.VariavelArquivo], cubo: am.VariavelArquivo,
                   nome: str) -> Tuple[Optional[np.ndarray], str]:
    """(valores, origem) do eixo espectral; (None, motivo) se nao ha'."""
    base = os.path.basename(caminho)
    forma = cubo.forma_util
    if nome:
        v = am.achar_variavel(caminho, variaveis, nome, "arquivo_variavel_eixo")
        valores = np.asarray(v.ler(), dtype=float).ravel()
        if len(v.forma_util) != 1:
            raise ValueError(f"{base}: arquivo_variavel_eixo aponta {v.descricao()}, que nao e' um vetor.")
        if valores.size not in forma:
            raise ValueError(f"{base}: o eixo {v.descricao()} tem {valores.size} valores, mas nenhum eixo do cubo "
                             f"({_forma_txt(forma)}) tem esse tamanho.")
        if not am.eixo_monotono(valores):
            raise ValueError(f"{base}: o eixo {v.descricao()} nao e' monotono (tem repeticao, NaN ou vai e volta): "
                             f"nao pode ser um eixo de comprimentos de onda.")
        return valores, f"variavel '{v.nome}' (arquivo_variavel_eixo)"
    attr = _eixo_do_atributo(cubo)
    if attr is not None and attr.size in forma:
        return attr, f"atributo do dataset '{cubo.nome}'"
    cands = am.vetores_candidatos_a_eixo(variaveis, sorted(set(forma)), excluir=(cubo.nome,))
    com_nome = [c for c in cands if c[2]]
    usar = com_nome or cands
    if not usar:
        return None, "nenhum vetor de comprimentos de onda no arquivo"
    if len(usar) > 1:
        distintos = {(c[1].size, tuple(np.round(c[1], 6))) for c in usar}
        if len(distintos) > 1:
            raise ValueError(
                f"{base}: mais de um vetor pode ser o eixo de comprimentos de onda: "
                f"{am.descrever_variaveis([c[0] for c in usar])}. Diga qual e' com arquivo_variavel_eixo.")
    v, valores, sugere = usar[0]
    return valores, f"variavel '{v.nome}'" + ("" if sugere else " (unico vetor monotono com o tamanho de um eixo)")


def _posicao_espectral(base: str, forma: Tuple[int, ...], wl: Optional[np.ndarray], eixo: str,
                       avisos: List[str]) -> int:
    if eixo not in EIXOS_ESPECTRAIS:
        raise ValueError(f"hsi_eixo_espectral '{eixo}' invalido; use {list(EIXOS_ESPECTRAIS)}.")
    if eixo != "auto":
        p = _POS[eixo]
        if wl is not None and forma[p] != wl.size:
            raise ValueError(f"{base}: hsi_eixo_espectral: {eixo} aponta um eixo de {forma[p]} posicoes, mas o "
                             f"vetor de comprimentos de onda tem {wl.size} valores (cubo {_forma_txt(forma)}).")
        return p
    msg_escolha = (" Diga onde estao as bandas com hsi_eixo_espectral: primeiro, meio ou ultimo "
                   "(na ordem que o MATLAB mostra em size(cubo)).")
    if wl is not None:
        posicoes = [i for i, d in enumerate(forma) if d == wl.size]
        if len(posicoes) == 1:
            return posicoes[0]
        raise ValueError(f"{base}: o vetor de comprimentos de onda ({wl.size} valores) tem o tamanho de mais de um "
                         f"eixo do cubo ({_forma_txt(forma)}).{msg_escolha}")
    if forma[1] == forma[2] and forma[0] != forma[1]:
        raise ValueError(f"{base}: cubo {_forma_txt(forma)} sem vetor de comprimentos de onda; a forma sugere "
                         f"bandas x linhas x colunas, mas pode ser uma imagem de {forma[0]} x {forma[1]} pixels com "
                         f"{forma[2]} bandas.{msg_escolha}")
    avisos.append(f"{base}: sem vetor de comprimentos de onda no arquivo; assumido que as bandas sao o ULTIMO eixo "
                  f"({forma[2]} bandas, imagem {forma[0]} x {forma[1]}). Se estiver errado, use hsi_eixo_espectral.")
    return 2


def ler_cubo_arquivo(caminho: str, *, variavel_dados: str = "", variavel_eixo: str = "",
                     eixo_espectral: str = "auto", calibrando: bool = False) -> CuboArquivo:
    """Le o cubo de um .mat/.npy/.npz/.h5 e devolve `CuboArquivo` com o cubo (linhas, colunas, bandas)
    em float32, o eixo espectral (ou None) e `info` (formato, variaveis usadas, forma original, avisos)."""
    base = os.path.basename(caminho)
    fmt, variaveis = am.listar_variaveis(caminho)
    v = _escolher_cubo(caminho, variaveis, (variavel_dados or "").strip())
    wl, origem_wl = _escolher_eixo(caminho, variaveis, v, (variavel_eixo or "").strip())
    forma = v.forma_util
    avisos: List[str] = []
    p = _posicao_espectral(base, forma, wl, eixo_espectral or "auto", avisos)
    bruto = np.asarray(v.ler())
    bruto = bruto.reshape(forma)            # tira eixos de tamanho 1 (ex.: 1 x L x C x B)
    espaciais = [i for i in range(3) if i != p]
    cubo = np.ascontiguousarray(np.transpose(bruto, (*espaciais, p)), dtype=np.float32)
    invertido = False
    if wl is not None and wl.size >= 2 and wl[0] > wl[-1]:
        # eixo decrescente (comum em numero de onda): mesma orientacao crescente do resto do HSI
        wl = wl[::-1].copy()
        cubo = np.ascontiguousarray(cubo[..., ::-1])
        invertido = True
    if np.dtype(v.dtype).kind in "iu" and not calibrando:
        avisos.append(f"{base}: cubo inteiro ({v.dtype}); parece contagem bruta do detector (DN), nao reflectancia. "
                      f"A checagem de qualidade espera reflectancia (0 a 1): use hsi_calibracao: branco_escuro "
                      f"(com WHITEREF/DARKREF ou variaveis white/dark no proprio arquivo).")
    n_nan = int(np.count_nonzero(~np.isfinite(cubo)))
    if n_nan:
        avisos.append(f"{base}: {n_nan} valor(es) NaN/Inf no cubo.")
    info = {"formato": fmt, "variavel_dados": v.nome, "forma_original": list(forma),
            "eixo_espectral": _NOME_POS[p], "origem_eixo": origem_wl, "avisos": avisos,
            "dtype_original": str(v.dtype), "bandas_invertidas": invertido,
            "maximo_bruto": float(np.nanmax(bruto)) if bruto.size else 0.0}
    return CuboArquivo(cubo=cubo, wavelengths=None if wl is None else np.asarray(wl, dtype=float), info=info)


# --------------------------------------------------------------------------- referencias branco/escuro
def _reduzir_referencia(base: str, nome: str, arr: np.ndarray, n_bandas: int, pos_cubo: Optional[int],
                        invertido: bool) -> np.ndarray:
    """Referencia -> `(colunas|1, bandas)` (media sobre as linhas), no mesmo sentido de bandas do cubo.
    Aceita espectro (bandas), matriz (colunas x bandas, ou o transposto) ou cubo 3D."""
    a = np.asarray(arr, dtype=np.float64)
    a = a.reshape(tuple(d for d in a.shape if d != 1) or (1,))
    if a.ndim == 1:
        if a.size != n_bandas:
            raise ValueError(f"{base}: a referencia '{nome}' tem {a.size} valores e o cubo {n_bandas} bandas.")
        red = a[None, :]
    elif a.ndim == 2:
        if a.shape[1] == n_bandas:
            red = a
        elif a.shape[0] == n_bandas:
            red = a.T
        else:
            raise ValueError(f"{base}: a referencia '{nome}' ({_forma_txt(a.shape)}) nao tem um eixo de "
                             f"{n_bandas} bandas.")
    elif a.ndim == 3:
        posicoes = [i for i, d in enumerate(a.shape) if d == n_bandas]
        if not posicoes:
            raise ValueError(f"{base}: a referencia '{nome}' ({_forma_txt(a.shape)}) nao tem um eixo de "
                             f"{n_bandas} bandas.")
        p = pos_cubo if pos_cubo in posicoes else (posicoes[-1] if 2 in posicoes or len(posicoes) > 1
                                                    else posicoes[0])
        espaciais = [i for i in range(3) if i != p]
        red = np.nanmean(np.transpose(a, (*espaciais, p)), axis=0)
    else:
        raise ValueError(f"{base}: a referencia '{nome}' tem {a.ndim} eixos.")
    return red[:, ::-1].copy() if invertido else red


def _dict_referencia(origem: str, red: np.ndarray, bruto: np.ndarray, dtype: str,
                     wl: Optional[np.ndarray]) -> Dict[str, Any]:
    """Mesmo formato de `hsi_io._carregar_referencia` (ENVI)."""
    maximo = float(np.nanmax(bruto)) if bruto.size else 0.0
    frac_sat = 0.0
    dt = np.dtype(dtype)
    if dt.kind in "iu":
        teto = np.iinfo(dt).max
        topos = tuple(2 ** k - 1 for k in (8, 10, 12, 14, 16) if 2 ** k - 1 <= teto)
        if maximo in topos:
            frac_sat = float(np.mean(bruto >= maximo))
    return {"media": red.astype(np.float64), "n_linhas": 1, "n_colunas": int(red.shape[0]),
            "n_bandas": int(red.shape[1]), "wl": wl, "frac_saturada": frac_sat, "maximo": maximo,
            "tint": None, "arquivo": origem}


def referencias_no_arquivo(caminho: str, info: Dict[str, Any], n_bandas: int) -> Dict[str, Dict[str, Any]]:
    """Variaveis white/dark (branco/escuro, whiteref/darkref...) guardadas NO PROPRIO arquivo do cubo,
    ja' reduzidas como `hsi_io._carregar_referencia`. Chaves "white"/"dark" so' para as encontradas."""
    base = os.path.basename(caminho)
    _fmt, variaveis = am.listar_variaveis(caminho)
    pos = _POS.get(str(info.get("eixo_espectral", "")))
    saida: Dict[str, Dict[str, Any]] = {}
    for tipo, rx in _RE_REF_TIPO.items():
        achados = [v for v in variaveis if v.numerica and v.nome != info.get("variavel_dados")
                   and rx.search(v.nome.split(".")[-1])]
        if len(achados) > 1:
            raise ValueError(f"{base}: mais de uma variavel parece a referencia {'branca' if tipo == 'white' else 'escura'}: "
                             f"{am.descrever_variaveis(achados)}. Deixe so' uma no arquivo ou aponte o arquivo com "
                             f"hsi_referencia_{'branco' if tipo == 'white' else 'escuro'}.")
        if achados:
            v = achados[0]
            bruto = np.asarray(v.ler())
            red = _reduzir_referencia(base, v.nome, bruto, n_bandas, pos, bool(info.get("bandas_invertidas")))
            saida[tipo] = _dict_referencia(f"{caminho}::{v.nome}", red, bruto, v.dtype, None)
    return saida


def ler_referencia_arquivo(caminho: str, n_bandas: int, *, pos_cubo: Optional[int] = None,
                           invertido: bool = False) -> Dict[str, Any]:
    """Arquivo WHITEREF/DARKREF em .mat/.npy/.npz/.h5: a variavel com um eixo de `n_bandas` (cubo, matriz
    colunas x bandas ou espectro), reduzida como a referencia ENVI. Mais de uma candidata -> erro com a lista.
    `invertido`: o cubo teve as bandas invertidas (eixo decrescente); vale se o arquivo nao trouxer eixo."""
    base = os.path.basename(caminho)
    _fmt, variaveis = am.listar_variaveis(caminho)
    cands = [v for v in variaveis if v.numerica and n_bandas in v.forma_util and len(v.forma_util) >= 1]
    eixos = am.vetores_candidatos_a_eixo(variaveis, (n_bandas,))
    nomes_eixo = {c[0].nome for c in eixos if c[2]}
    cands = [v for v in cands if v.nome not in nomes_eixo]
    if not cands:
        raise ValueError(f"{base}: nenhuma variavel com {n_bandas} bandas para servir de referencia. "
                         f"Variaveis: {am.descrever_variaveis(variaveis)}.")
    if len(cands) > 1:
        maior = max(len(v.forma_util) for v in cands)
        cands = [v for v in cands if len(v.forma_util) == maior]
    if len(cands) > 1:
        raise ValueError(f"{base}: mais de uma variavel pode ser a referencia: {am.descrever_variaveis(cands)}. "
                         f"Deixe so' uma no arquivo de referencia.")
    v = cands[0]
    wl = None
    inv = invertido
    com_nome = [c for c in eixos if c[2]]
    if len(com_nome) == 1:
        wl = com_nome[0][1]
        inv = bool(wl.size >= 2 and wl[0] > wl[-1])
        if inv:
            wl = wl[::-1].copy()
    bruto = np.asarray(v.ler())
    red = _reduzir_referencia(base, v.nome, bruto, n_bandas, pos_cubo, inv)
    return _dict_referencia(caminho, red, bruto, v.dtype, wl)
