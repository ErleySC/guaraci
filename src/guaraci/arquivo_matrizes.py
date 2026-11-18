# -*- coding: utf-8 -*-
"""arquivo_matrizes.py -- leitura de matrizes numericas guardadas em arquivo de
"variaveis" (MATLAB .mat v4/v5/v7 e v7.3, NumPy .npy/.npz, HDF5 .h5/.hdf5).

Estes formatos nao dizem o que cada variavel significa: um .mat pode ter o cubo,
uma mascara, o vetor de comprimentos de onda e o branco, com qualquer nome.
Este modulo so' LISTA as variaveis numericas (nome, forma, tipo) sem carregar os
dados, e carrega uma variavel quando pedida. Quem decide qual e' o cubo (modo
`hsi`, `hsi_arquivo.py`) ou a matriz amostras x variaveis (modo `csv`,
`tabela_arquivo.py`) e' o chamador -- com mensagem clara quando ha' mais de uma
candidata, em vez de chutar.

Ordem dos eixos: a forma devolvida e' a que o USUARIO ve no programa de origem.
No .mat v7.3 (HDF5) o MATLAB grava em ordem de coluna, entao o h5py ve os eixos
invertidos; aqui eles sao desinvertidos, para que `size(cubo)` no MATLAB e a
forma listada pelo Guaraci coincidam em todas as versoes do .mat.

Dependencias: .mat v4-v7 usa scipy (base); .mat v7.3 e .h5 usam h5py (extra
opcional `hdf5`): sem ele a mensagem diz o comando de instalacao.
ASCII por convencao do projeto.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["VariavelArquivo", "EXTENSOES_MATRIZ", "formato_do_arquivo", "listar_variaveis",
           "achar_variavel", "descrever_variaveis", "eixo_monotono", "vetores_candidatos_a_eixo",
           "PADRAO_NOME_EIXO", "texto_falta_h5py"]

#: Extensoes reconhecidas (minusculas).
EXTENSOES_MATRIZ: Tuple[str, ...] = (".mat", ".npy", ".npz", ".h5", ".hdf5")

#: Nomes que costumam guardar o eixo espectral (comprimento de onda, numero de onda, banda).
PADRAO_NOME_EIXO = re.compile(
    r"(wave|wavelength|wavenumber|lambda|^wl$|^wl_|_wl$|^nm$|comprimento|onda|^bandas?$|^bands?$"
    r"|^eixo|axis|^x$|^xaxis|^cm_?1$|numero_?de_?onda)", re.IGNORECASE)

_INSTALAR_H5PY = 'pip install ".[hdf5]"'


def texto_falta_h5py(caminho: str, motivo: str) -> str:
    return (f"{os.path.basename(caminho)}: {motivo} precisa do pacote opcional h5py, que nao esta "
            f"instalado. Instale com: {_INSTALAR_H5PY} (ou pip install h5py). Alternativa sem instalar "
            f"nada: no MATLAB, salve de novo com save('arquivo.mat', 'cubo', 'wl', '-v7').")


@dataclass
class VariavelArquivo:
    """Uma variavel numerica do arquivo, ainda nao carregada."""
    nome: str
    forma: Tuple[int, ...]
    dtype: str                     # dtype numpy (ex.: 'float64', 'uint16')
    logica: bool                   # mascara booleana (MATLAB logical / numpy bool)
    ler: Callable[[], np.ndarray]  # carrega o array, ja' na ordem de eixos do usuario
    atributos: Dict[str, Any]

    @property
    def forma_util(self) -> Tuple[int, ...]:
        """Forma sem os eixos de tamanho 1 (um vetor 1 x N do MATLAB vira (N,))."""
        return tuple(int(d) for d in self.forma if int(d) != 1)

    @property
    def numerica(self) -> bool:
        return np.dtype(self.dtype).kind in "iuf" and not self.logica

    def descricao(self) -> str:
        forma = " x ".join(str(d) for d in self.forma) or "escalar"
        return f"'{self.nome}' ({forma}, {'logical' if self.logica else self.dtype})"


def formato_do_arquivo(caminho: str) -> str:
    """'mat5' (MATLAB v4-v7), 'mat73' (MATLAB v7.3/HDF5), 'npy', 'npz' ou 'h5'."""
    ext = os.path.splitext(caminho)[1].lower()
    if ext == ".npy":
        return "npy"
    if ext == ".npz":
        return "npz"
    if ext in (".h5", ".hdf5"):
        return "h5"
    if ext != ".mat":
        raise ValueError(f"{os.path.basename(caminho)}: extensao '{ext}' nao e' de matriz "
                         f"({', '.join(EXTENSOES_MATRIZ)}).")
    with open(caminho, "rb") as f:
        cab = f.read(128)
    # Assinatura do v7.3: cabecalho de texto "MATLAB 7.3 MAT-file" seguido de um arquivo HDF5
    # (userblock de 512 bytes). O v5 tem "MATLAB 5.0 MAT-file"; o v4 nao tem texto.
    if b"MATLAB 7.3" in cab:
        return "mat73"
    with open(caminho, "rb") as f:
        f.seek(512)
        if f.read(8) == b"\x89HDF\r\n\x1a\n":
            return "mat73"
    return "mat5"


# --------------------------------------------------------------------------- MATLAB v4-v7
def _listar_mat5(caminho: str) -> List[VariavelArquivo]:
    import scipy.io as sio
    try:
        quem = sio.whosmat(caminho)
    except Exception as e:  # noqa: BLE001 -- arquivo corrompido/nao-MAT: erro claro com o nome
        raise ValueError(f"{os.path.basename(caminho)}: nao foi possivel ler como arquivo MATLAB ({e}).") from e
    saida: List[VariavelArquivo] = []
    structs: List[str] = []
    for nome, forma, classe in quem:
        if classe == "struct":
            structs.append(nome)
            continue
        if classe in ("cell", "char", "object", "function_handle", "opaque", "sparse"):
            continue
        logica = classe == "logical"
        dtype = {"double": "float64", "single": "float32", "int8": "int8", "uint8": "uint8",
                 "int16": "int16", "uint16": "uint16", "int32": "int32", "uint32": "uint32",
                 "int64": "int64", "uint64": "uint64", "logical": "bool"}.get(classe)
        if dtype is None:
            continue

        def _ler(n: str = nome) -> np.ndarray:
            return np.asarray(sio.loadmat(caminho, variable_names=[n])[n])
        saida.append(VariavelArquivo(nome, tuple(int(d) for d in forma), dtype, logica, _ler, {}))
    # Campos de struct (um nivel ou mais): s.cubo, s.wl ... Carrega so' as structs (o resto ja'
    # foi listado sem carregar).
    if structs:
        dados = sio.loadmat(caminho, variable_names=structs, simplify_cells=True)
        for nome in structs:
            _campos_struct(nome, dados.get(nome), saida)
    return saida


def _campos_struct(prefixo: str, valor: Any, saida: List[VariavelArquivo]) -> None:
    if isinstance(valor, dict):
        for k, v in valor.items():
            _campos_struct(f"{prefixo}.{k}", v, saida)
        return
    if isinstance(valor, np.ndarray) and valor.dtype.kind in "iufb" and valor.size > 0:
        arr: np.ndarray = valor

        def _ler_campo(a: np.ndarray = arr) -> np.ndarray:
            return a
        saida.append(VariavelArquivo(prefixo, tuple(int(d) for d in arr.shape), str(arr.dtype),
                                     arr.dtype.kind == "b", _ler_campo, {}))


# --------------------------------------------------------------------------- HDF5 (.mat v7.3 e .h5)
def _importar_h5py(caminho: str, motivo: str):
    try:
        import h5py
    except ImportError as e:
        raise ImportError(texto_falta_h5py(caminho, motivo)) from e
    return h5py


def _texto_atributo(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("ascii", "replace")
    if isinstance(v, np.ndarray) and v.dtype.kind == "S":
        return b"".join(v.ravel().tolist()).decode("ascii", "replace")
    return str(v)


def _listar_h5(caminho: str, matlab: bool) -> List[VariavelArquivo]:
    h5py = _importar_h5py(caminho, "arquivo MATLAB v7.3 (HDF5)" if matlab else "arquivo HDF5")
    saida: List[VariavelArquivo] = []
    with h5py.File(caminho, "r") as f:
        def _visitar(nome: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset) or nome.startswith("#"):
                return
            atributos = {k: obj.attrs[k] for k in obj.attrs.keys()}
            classe = _texto_atributo(atributos.get("MATLAB_class", "")) if matlab else ""
            if matlab and classe in ("cell", "char", "struct", "function_handle"):
                return
            if obj.dtype.kind not in "iufb":
                return
            if matlab and atributos.get("MATLAB_empty") is not None:
                return
            logica = classe == "logical" or obj.dtype.kind == "b"
            forma = tuple(int(d) for d in obj.shape)
            nome_usuario = nome.replace("/", ".") if matlab else nome
            if matlab:
                forma = forma[::-1]   # MATLAB grava em ordem de coluna: desinverte

            def _ler(n: str = nome, inverter: bool = matlab) -> np.ndarray:
                with h5py.File(caminho, "r") as g:
                    a = np.asarray(g[n][()])
                return a.T if inverter else a
            attrs = {} if matlab else {k: v for k, v in atributos.items()}
            saida.append(VariavelArquivo(nome_usuario, forma, str(obj.dtype), logica, _ler, attrs))
        f.visititems(_visitar)
    return saida


# --------------------------------------------------------------------------- NumPy
def _listar_npy(caminho: str) -> List[VariavelArquivo]:
    arr = np.load(caminho, mmap_mode="r", allow_pickle=False)
    nome = os.path.splitext(os.path.basename(caminho))[0]
    return [VariavelArquivo(nome, tuple(int(d) for d in arr.shape), str(arr.dtype), arr.dtype.kind == "b",
                            lambda: np.load(caminho, allow_pickle=False), {})]


def _listar_npz(caminho: str) -> List[VariavelArquivo]:
    saida: List[VariavelArquivo] = []
    with np.load(caminho, allow_pickle=False) as z:
        for k in z.files:
            a = z[k]
            if a.dtype.kind not in "iufb":
                continue
            def _ler_npz(n: str = k) -> np.ndarray:
                with np.load(caminho, allow_pickle=False) as zz:
                    return np.asarray(zz[n])
            saida.append(VariavelArquivo(k, tuple(int(d) for d in a.shape), str(a.dtype), a.dtype.kind == "b",
                                         _ler_npz, {}))
    return saida


def listar_variaveis(caminho: str) -> Tuple[str, List[VariavelArquivo]]:
    """`(formato, variaveis numericas)` do arquivo, sem carregar os dados (exceto structs do .mat v5
    e o .npz, que o formato obriga a abrir)."""
    if not os.path.isfile(caminho):
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")
    fmt = formato_do_arquivo(caminho)
    if fmt == "mat5":
        vs = _listar_mat5(caminho)
    elif fmt == "mat73":
        vs = _listar_h5(caminho, matlab=True)
    elif fmt == "h5":
        vs = _listar_h5(caminho, matlab=False)
    elif fmt == "npy":
        vs = _listar_npy(caminho)
    else:
        vs = _listar_npz(caminho)
    return fmt, vs


def _normalizar_nome(n: str) -> str:
    return n.strip().strip("/").replace("/", ".").lower()


def achar_variavel(caminho: str, variaveis: Sequence[VariavelArquivo], nome: str, chave: str
                   ) -> VariavelArquivo:
    """A variavel de nome `nome` (sem diferenciar maiuscula; `s/cubo` = `s.cubo`). Erro claro, com a
    lista do que existe, se nao houver."""
    alvo = _normalizar_nome(nome)
    exatos = [v for v in variaveis if v.nome == nome.strip()]
    if exatos:
        return exatos[0]
    achados = [v for v in variaveis if _normalizar_nome(v.nome) == alvo]
    if len(achados) == 1:
        return achados[0]
    raise ValueError(
        f"{os.path.basename(caminho)}: {chave} = '{nome}', mas o arquivo nao tem essa variavel numerica. "
        f"Variaveis encontradas: {descrever_variaveis(variaveis)}.")


def descrever_variaveis(variaveis: Sequence[VariavelArquivo], n: int = 12) -> str:
    if not variaveis:
        return "nenhuma variavel numerica"
    txt = ", ".join(v.descricao() for v in variaveis[:n])
    return txt + (f" e mais {len(variaveis) - n}" if len(variaveis) > n else "")


def eixo_monotono(v: np.ndarray) -> bool:
    """Vetor finito e estritamente crescente ou decrescente (um eixo espectral de verdade)."""
    v = np.asarray(v, dtype=float).ravel()
    if v.size < 2 or not np.all(np.isfinite(v)):
        return False
    d = np.diff(v)
    return bool(np.all(d > 0) or np.all(d < 0))


def _e_indice(v: np.ndarray) -> bool:
    """0..n-1 ou 1..n: numeracao de linha/coluna, nao comprimento de onda."""
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    return bool(np.array_equal(v, np.arange(n)) or np.array_equal(v, np.arange(1, n + 1)))


def vetores_candidatos_a_eixo(variaveis: Sequence[VariavelArquivo], tamanhos: Sequence[int],
                              excluir: Sequence[str] = ()) -> List[Tuple[VariavelArquivo, np.ndarray, bool]]:
    """Vetores numericos monotonos cujo comprimento esta em `tamanhos`: `(variavel, valores,
    nome_sugere_eixo)`. Vetores de numeracao (1..n) so' entram se o nome sugerir eixo."""
    saida = []
    for v in variaveis:
        if v.nome in excluir or not v.numerica or len(v.forma_util) != 1:
            continue
        if v.forma_util[0] not in tamanhos:
            continue
        valores = np.asarray(v.ler(), dtype=float).ravel()
        if not eixo_monotono(valores):
            continue
        sugere = bool(PADRAO_NOME_EIXO.search(v.nome.split(".")[-1]))
        if _e_indice(valores) and not sugere:
            continue
        saida.append((v, valores, sugere))
    return saida
