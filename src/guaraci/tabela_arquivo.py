# -*- coding: utf-8 -*-
"""tabela_arquivo.py -- a tabela amostras x variaveis do modo `csv` a partir de Excel (.xlsx/.xlsm)
ou de arquivo de variaveis (MATLAB .mat v5/v7.3, NumPy .npz, HDF5 .h5), devolvida como o MESMO
DataFrame que o CSV daria: colunas espectrais com o valor do eixo no cabecalho e colunas de texto/
numero para classe, concentracao e grupo. Assim o resto do modo `csv` (coluna_classe,
coluna_concentracao, coluna_grupo, checagens) vale sem mudanca.

Excel: le a primeira planilha (ou a de nome `arquivo_variavel_dados`).

Arquivo de variaveis (o caso comum do MATLAB: X, wl, classe, y):
  - X: a unica matriz numerica 2D (ou `arquivo_variavel_dados`). Duas ou mais (ex.: Xcal e Xval)
    -> para e lista; o usuario escolhe.
  - eixo: vetor monotono com o comprimento de um dos lados de X, de preferencia com nome de eixo
    (wavelength, wl, nm, axis...) ou `arquivo_variavel_eixo`. Ele decide a orientacao: se bate com
    as LINHAS, X esta' guardada variaveis x amostras e e' transposta (com aviso).
  - cada vetor (numerico, logico ou texto) com uma entrada por amostra vira uma coluna com o nome
    da variavel: aponte coluna_classe / coluna_concentracao / coluna_grupo para esses nomes.
ASCII por convencao do projeto.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from guaraci import arquivo_matrizes as am

__all__ = ["EXTENSOES_EXCEL", "EXTENSOES_TABELA_ARQUIVO", "e_tabela_arquivo", "ler_tabela_arquivo"]

EXTENSOES_EXCEL: Tuple[str, ...] = (".xlsx", ".xlsm")
#: .npy fica de fora: guarda uma matriz so', sem eixo nem rotulos (nao forma uma tabela de estudo).
EXTENSOES_TABELA_ARQUIVO: Tuple[str, ...] = EXTENSOES_EXCEL + (".mat", ".npz", ".h5", ".hdf5")


def e_tabela_arquivo(caminho: str) -> bool:
    """True se `caminho` e' Excel ou arquivo de variaveis (nao texto CSV/TXT)."""
    return str(caminho).lower().endswith(EXTENSOES_TABELA_ARQUIVO)


def _ler_excel(caminho: str, planilha: str, nrows: Optional[int]) -> pd.DataFrame:
    base = os.path.basename(caminho)
    try:
        import openpyxl  # noqa: F401 -- so' para checar a presenca com mensagem clara
    except ImportError as e:
        raise ImportError(f"{base}: ler Excel precisa do pacote openpyxl, que nao esta instalado. Instale com: "
                          f'pip install ".[reports]" (ou pip install openpyxl), ou salve a planilha como CSV.') from e
    with pd.ExcelFile(caminho, engine="openpyxl") as xl:
        nomes = [str(n) for n in xl.sheet_names]
        if planilha:
            if planilha not in nomes:
                raise ValueError(f"{base}: arquivo_variavel_dados = '{planilha}', mas a planilha nao existe. "
                                 f"Planilhas: {', '.join(nomes)}.")
            alvo = planilha
        else:
            alvo = nomes[0]
            if len(nomes) > 1:
                print(f"[INFO] {base}: {len(nomes)} planilhas ({', '.join(nomes)}); lida a primeira, '{alvo}'. "
                      f"Para outra, use arquivo_variavel_dados.")
        df = pd.read_excel(xl, sheet_name=alvo, nrows=nrows)
    df.columns = [_cabecalho(c) for c in df.columns]
    return df


def _cabecalho(c) -> str:
    """Cabecalho numerico do Excel vem como float (1100.0): grava como no CSV (1100)."""
    if isinstance(c, (int, float, np.integer, np.floating)) and not isinstance(c, bool):
        return np.format_float_positional(float(c), trim="-")
    return str(c)


# --------------------------------------------------------------------------- textos (rotulos)
def _textos_mat5(caminho: str) -> Dict[str, List[str]]:
    import scipy.io as sio
    nomes = [n for n, _f, c in sio.whosmat(caminho) if c in ("char", "cell")]
    if not nomes:
        return {}
    dados = sio.loadmat(caminho, variable_names=nomes, simplify_cells=True)
    saida: Dict[str, List[str]] = {}
    for n in nomes:
        v = dados.get(n)
        if isinstance(v, str):
            continue                       # um texto so' (titulo, unidade): nao e' coluna
        if isinstance(v, np.ndarray) and v.dtype.kind == "U":
            saida[n] = [str(s).strip() for s in v.ravel()]
        elif isinstance(v, (list, np.ndarray)):
            itens = list(np.asarray(v, dtype=object).ravel())
            if itens and all(isinstance(s, str) for s in itens):
                saida[n] = [s.strip() for s in itens]
    return saida


def _texto_uint16(a: np.ndarray) -> str:
    return "".join(chr(int(c)) for c in np.asarray(a).ravel()).strip()


def _textos_h5(caminho: str, matlab: bool) -> Dict[str, List[str]]:
    try:
        import h5py
    except ImportError:
        return {}
    saida: Dict[str, List[str]] = {}
    with h5py.File(caminho, "r") as f:
        def _visitar(nome: str, obj) -> None:
            if not isinstance(obj, h5py.Dataset) or nome.startswith("#"):
                return
            chave = nome.replace("/", ".") if matlab else nome
            classe = am._texto_atributo(obj.attrs.get("MATLAB_class", "")) if matlab else ""
            if matlab and classe == "char":
                a = np.asarray(obj[()]).T             # (n, k) no MATLAB: uma linha por amostra
                if a.ndim == 2 and a.shape[0] > 1:
                    saida[chave] = [_texto_uint16(lin) for lin in a]
            elif matlab and classe == "cell":
                refs = np.asarray(obj[()]).T.ravel()
                textos = []
                for r in refs:
                    alvo = f[r]
                    if am._texto_atributo(alvo.attrs.get("MATLAB_class", "")) != "char":
                        return
                    textos.append(_texto_uint16(alvo[()]))
                saida[chave] = textos
            elif not matlab and (obj.dtype.kind == "S" or h5py.check_string_dtype(obj.dtype) is not None):
                saida[chave] = [str(s).strip() for s in obj.asstr()[()].ravel()]
        f.visititems(_visitar)
    return saida


def _textos_npz(caminho: str) -> Dict[str, List[str]]:
    saida: Dict[str, List[str]] = {}
    with np.load(caminho, allow_pickle=False) as z:
        for k in z.files:
            a = z[k]
            if a.dtype.kind in "US" and a.ndim >= 1 and a.size > 1:
                saida[k] = [s.decode() if isinstance(s, bytes) else str(s) for s in a.ravel()]
    return saida


def _textos(caminho: str, fmt: str) -> Dict[str, List[str]]:
    if fmt == "mat5":
        return _textos_mat5(caminho)
    if fmt in ("mat73", "h5"):
        return _textos_h5(caminho, matlab=fmt == "mat73")
    if fmt == "npz":
        return _textos_npz(caminho)
    return {}


# --------------------------------------------------------------------------- matriz
def _escolher_matriz(caminho: str, variaveis: List[am.VariavelArquivo], nome: str) -> am.VariavelArquivo:
    base = os.path.basename(caminho)
    if nome:
        v = am.achar_variavel(caminho, variaveis, nome, "arquivo_variavel_dados")
        if len(v.forma_util) != 2 or not v.numerica:
            raise ValueError(f"{base}: a variavel {v.descricao()} nao e' uma matriz numerica amostras x variaveis.")
        return v
    cands = [v for v in variaveis if v.numerica and len(v.forma_util) == 2]
    if not cands:
        cubos = [v for v in variaveis if v.numerica and len(v.forma_util) == 3]
        dica = (f" Ha' variavel com 3 eixos ({am.descrever_variaveis(cubos, 3)}): se for imagem hiperespectral, "
                f"use o modo HSI (modo_entrada: hsi).") if cubos else ""
        raise ValueError(f"{base}: nenhuma matriz 2D (amostras x variaveis). Variaveis: "
                         f"{am.descrever_variaveis(variaveis)}.{dica}")
    if len(cands) > 1:
        raise ValueError(f"{base}: ha' {len(cands)} matrizes 2D e nao da' para saber qual e' a de espectros: "
                         f"{am.descrever_variaveis(cands)}. Diga qual e' com arquivo_variavel_dados "
                         f"(ex.: arquivo_variavel_dados: {cands[0].nome}).")
    return cands[0]


def _tabela_de_matriz(caminho: str, variavel_dados: str, variavel_eixo: str) -> pd.DataFrame:
    base = os.path.basename(caminho)
    fmt, variaveis = am.listar_variaveis(caminho)
    vx = _escolher_matriz(caminho, variaveis, variavel_dados)
    n_lin, n_col = vx.forma_util
    # eixo espectral
    eixo: Optional[np.ndarray] = None
    nome_eixo = ""
    if variavel_eixo:
        ve = am.achar_variavel(caminho, variaveis, variavel_eixo, "arquivo_variavel_eixo")
        eixo = np.asarray(ve.ler(), dtype=float).ravel()
        if len(ve.forma_util) != 1 or eixo.size not in (n_lin, n_col):
            raise ValueError(f"{base}: arquivo_variavel_eixo aponta {ve.descricao()}, que nao e' um vetor com o "
                             f"tamanho de um dos lados de {vx.descricao()}.")
        nome_eixo = ve.nome
    else:
        cands = am.vetores_candidatos_a_eixo(variaveis, (n_lin, n_col), excluir=(vx.nome,))
        com_nome = [c for c in cands if c[2]]
        if len(com_nome) > 1:
            raise ValueError(f"{base}: mais de um vetor pode ser o eixo espectral: "
                             f"{am.descrever_variaveis([c[0] for c in com_nome])}. Diga qual e' com "
                             f"arquivo_variavel_eixo.")
        if com_nome:
            eixo, nome_eixo = com_nome[0][1], com_nome[0][0].nome
    # vetores por amostra (numericos e texto)
    vetores: Dict[str, np.ndarray] = {}
    for v in variaveis:
        if v.nome in (vx.nome, nome_eixo) or len(v.forma_util) != 1:
            continue
        if v.forma_util[0] in (n_lin, n_col):
            vetores[v.nome] = np.asarray(v.ler()).ravel()
    for nome, textos in _textos(caminho, fmt).items():
        if len(textos) in (n_lin, n_col) and nome not in vetores:
            vetores[nome] = np.asarray(textos, dtype=object)
    # orientacao
    X = np.asarray(vx.ler(), dtype=float).reshape(n_lin, n_col)
    transpor = False
    if eixo is not None and n_lin != n_col:
        transpor = eixo.size == n_lin
    elif n_lin != n_col and vetores:
        por_col = [k for k, a in vetores.items() if a.size == n_col]
        por_lin = [k for k, a in vetores.items() if a.size == n_lin]
        transpor = bool(por_col) and not por_lin
    if transpor:
        X = X.T
        print(f"[AVISO] {base}: '{vx.nome}' esta' guardada variaveis x amostras ({n_lin} x {n_col}); "
              f"transposta para {n_col} amostras x {n_lin} variaveis.")
    n_amostras, n_var = X.shape
    if eixo is None:
        print(f"[AVISO] {base}: sem vetor de eixo espectral; as {n_var} variaveis ficam numeradas 1..{n_var}. "
              f"Para usar comprimentos de onda, aponte o vetor com arquivo_variavel_eixo.")
        cab = [str(i) for i in range(1, n_var + 1)]
    else:
        cab = [np.format_float_positional(float(x), trim="-") for x in eixo]
    df = pd.DataFrame(X, columns=cab)
    extras = {}
    for nome, a in vetores.items():
        if a.size != n_amostras:
            continue
        col = nome.split(".")[-1].split("/")[-1]
        if col in df.columns or col in extras:
            col = nome
        extras[col] = a if a.dtype == object else (a.astype(bool) if a.dtype.kind == "b" else a)
    if extras:
        df = pd.concat([pd.DataFrame(extras), df], axis=1)
    print(f"[INFO] {base}: matriz '{vx.nome}' ({n_amostras} amostras x {n_var} variaveis)"
          + (f", eixo '{nome_eixo}'" if nome_eixo else "")
          + (f", colunas por amostra: {', '.join(extras)}" if extras else ", sem vetores de classe/concentracao")
          + ".")
    return df


def ler_tabela_arquivo(caminho: str, *, variavel_dados: str = "", variavel_eixo: str = "",
                       nrows: Optional[int] = None) -> pd.DataFrame:
    """DataFrame amostras x colunas de um Excel ou arquivo de variaveis (ver docstring do modulo)."""
    if not os.path.isfile(str(caminho)):
        raise FileNotFoundError(f"Arquivo nao encontrado: '{caminho}'. Confira arquivo_csv.")
    caminho = str(caminho)
    if caminho.lower().endswith(EXTENSOES_EXCEL):
        return _ler_excel(caminho, (variavel_dados or "").strip(), nrows)
    df = _tabela_de_matriz(caminho, (variavel_dados or "").strip(), (variavel_eixo or "").strip())
    return df if nrows is None else df.head(nrows)
