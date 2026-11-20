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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["VariavelArquivo", "EXTENSOES_MATRIZ", "formato_do_arquivo", "listar_variaveis",
           "achar_variavel", "descrever_variaveis", "eixo_monotono", "vetores_candidatos_a_eixo",
           "PADRAO_NOME_EIXO", "texto_falta_h5py", "ListaVariaveis", "DICA_CONVERTER"]

#: Extensoes reconhecidas (minusculas).
EXTENSOES_MATRIZ: Tuple[str, ...] = (".mat", ".npy", ".npz", ".h5", ".hdf5")

#: Nomes que costumam guardar o eixo espectral (comprimento de onda, numero de onda, banda).
PADRAO_NOME_EIXO = re.compile(
    r"(wave|wavelength|wavenumber|lambda|^wl$|^wl_|_wl$|^nm$|comprimento|onda|^bandas?$|^bands?$"
    r"|^eixo|axis|^x$|^xaxis|^cm_?1$|numero_?de_?onda)", re.IGNORECASE)

_INSTALAR_H5PY = 'pip install ".[hdf5]"'


def texto_falta_h5py(caminho: str, motivo: str, erro: Optional[BaseException] = None) -> str:
    alternativa = "Alternativa sem h5py: no MATLAB, salve de novo com save('arquivo.mat', 'cubo', 'wl', '-v7')."
    nome = os.path.basename(caminho)
    if erro is not None and not isinstance(erro, ModuleNotFoundError):
        # instalado, mas nao carrega (ex.: DLL bloqueada pelo Controle de Aplicativos do Windows)
        return (f"{nome}: {motivo} precisa do pacote opcional h5py, que esta instalado mas nao carregou "
                f"({type(erro).__name__}: {str(erro).strip()[:200]}). Em geral e' bloqueio do sistema "
                f"(Controle de Aplicativos/antivirus) ou instalacao quebrada; reinstale com "
                f"{_INSTALAR_H5PY} ou libere o h5py. {alternativa}")
    return (f"{nome}: {motivo} precisa do pacote opcional h5py, que nao esta instalado. Instale com: "
            f"{_INSTALAR_H5PY} (ou pip install h5py). {alternativa}")


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


class ListaVariaveis(list):
    """Lista de `VariavelArquivo` que tambem guarda o que foi IGNORADO no arquivo (`(nome, classe)`):
    tabela, string, objeto, celula de texto, numero complexo... Serve para a mensagem de erro dizer o
    que existe no arquivo e como converter, em vez de "nenhuma variavel numerica"."""

    def __init__(self, itens=(), ignoradas=()):
        super().__init__(itens)
        self.ignoradas: List[Tuple[str, str]] = list(ignoradas)


#: Como salvar de novo no MATLAB quando a variavel nao e' matriz numerica simples.
DICA_CONVERTER = (
    "No MATLAB, converta para matriz numerica e salve de novo: tabela -> X = table2array(T) (ou T{:,:}); "
    "DataSet do PLS_Toolbox -> X = ds.data; wl = ds.axisscale{2}; celula de matrizes -> salve cada uma "
    "com nome proprio (ou X = cat(3, c{:})); numero complexo -> real(X) ou abs(X). Depois: "
    "save('arquivo.mat', 'X', 'wl', '-v7').")

#: Classes MATLAB que sao matriz numerica (ou logica) simples.
_CLASSES_NUMERICAS = {"double": "float64", "single": "float32", "int8": "int8", "uint8": "uint8",
                      "int16": "int16", "uint16": "uint16", "int32": "int32", "uint32": "uint32",
                      "int64": "int64", "uint64": "uint64", "logical": "bool"}


def _array_numerico(valor: Any) -> Any:
    """`valor` como ndarray numerico real (esparsa vira densa), ou None se nao for numerico. Complexo
    devolve a string 'complexo' (o chamador registra como ignorado)."""
    import scipy.sparse as sp          # scipy e' dependencia base
    if sp.issparse(valor):
        valor = valor.toarray()
    if isinstance(valor, (bool, int, float, np.number, np.bool_)) and not isinstance(valor, complex):
        valor = np.asarray(valor)
    if not isinstance(valor, np.ndarray):
        return None
    if valor.dtype.kind == "c":
        return "complexo"
    if valor.dtype.kind not in "iufb" or valor.size == 0:
        return None
    return valor


def _erro_leitura(caminho: str, nome: str, e: Exception) -> ValueError:
    return ValueError(f"{os.path.basename(caminho)}: nao foi possivel carregar a variavel '{nome}' "
                      f"(arquivo truncado ou corrompido? detalhe: {e}).")


def formato_do_arquivo(caminho: str) -> str:
    """'mat5' (MATLAB v4-v7), 'mat73' (MATLAB v7.3/HDF5), 'npy', 'npz' ou 'h5' (inclui .mat que e' HDF5
    puro, sem o cabecalho do MATLAB, como o `save -hdf5` do Octave)."""
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
    if cab[:8] == b"\x89HDF\r\n\x1a\n":
        return "h5"      # HDF5 puro com extensao .mat (Octave `save -hdf5`, hdf5storage): sem layout MATLAB
    with open(caminho, "rb") as f:
        f.seek(512)
        if f.read(8) == b"\x89HDF\r\n\x1a\n":
            return "mat73"
    return "mat5"


# --------------------------------------------------------------------------- MATLAB v4-v7
def _loadmat(caminho: str, **kw: Any) -> Dict[str, Any]:
    """scipy.io.loadmat sem o DeprecationWarning de matriz esparsa (scipy >= 1.15 avisa que o tipo
    devolvido vai mudar; aqui a esparsa vira densa de qualquer jeito)."""
    import warnings
    import scipy.io as sio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return sio.loadmat(caminho, **kw)


def _listar_mat5(caminho: str) -> ListaVariaveis:
    """Matrizes numericas do .mat v4-v7. Matriz simples e' listada sem carregar; struct, objeto antigo
    (`@classe`, ex.: DataSet do PLS_Toolbox), celula e esparsa sao abertas e seus campos/elementos
    numericos entram como `s.campo`, `obj.campo`, `c{2}` e `s(2).campo` (indices do MATLAB, base 1).
    Tabela, string, datetime (objetos MCOS) e texto ficam em `ignoradas`."""
    import scipy.io as sio
    saida = ListaVariaveis()
    try:
        quem = sio.whosmat(caminho)
    except Exception as e:  # noqa: BLE001 -- whosmat falha com objetos MCOS (string do MATLAB): tenta o loadmat
        quem = None
        erro_whos = e
    if quem is None:
        try:
            tudo = _loadmat(caminho, simplify_cells=True)
        except Exception as e:  # noqa: BLE001 -- arquivo corrompido/nao-MAT: erro claro com o nome
            raise ValueError(f"{os.path.basename(caminho)}: nao foi possivel ler como arquivo MATLAB "
                             f"({erro_whos}).") from e
        for nome, valor in tudo.items():
            if not nome.startswith("__"):
                _abrir_valor(nome, valor, saida)
        return saida
    abrir: List[str] = []
    for nome, forma, classe in quem:
        if nome.startswith("__"):           # __function_workspace__ e afins: interno do MATLAB
            continue
        if classe in ("struct", "object", "cell", "sparse"):
            abrir.append(nome)
            continue
        dtype = _CLASSES_NUMERICAS.get(classe)
        if dtype is None:
            saida.ignoradas.append((nome, classe))
            continue

        def _ler(n: str = nome) -> np.ndarray:
            try:
                bruto = _loadmat(caminho, variable_names=[n])[n]
            except Exception as e:  # noqa: BLE001 -- truncado/corrompido: diz qual variavel e qual arquivo
                raise _erro_leitura(caminho, n, e) from e
            a = _array_numerico(bruto)
            if isinstance(a, str):
                raise ValueError(f"{os.path.basename(caminho)}: a variavel '{n}' e' de numeros complexos; "
                                 f"o Guaraci nao descarta a parte imaginaria por conta propria. {DICA_CONVERTER}")
            if a is None:
                raise ValueError(f"{os.path.basename(caminho)}: a variavel '{n}' nao e' matriz numerica.")
            return a
        saida.append(VariavelArquivo(nome, tuple(int(d) for d in forma), dtype, classe == "logical", _ler, {}))
    if abrir:
        try:
            dados = _loadmat(caminho, variable_names=abrir, simplify_cells=True)
        except Exception as e:  # noqa: BLE001
            raise _erro_leitura(caminho, ", ".join(abrir), e) from e
        for nome in abrir:
            _abrir_valor(nome, dados.get(nome), saida)
    return saida


def _abrir_valor(nome: str, valor: Any, saida: ListaVariaveis, profundidade: int = 0) -> None:
    """Desce em struct (dict), struct array/celula (list ou ndarray de objetos) e objeto antigo do MATLAB
    e registra cada matriz numerica achada. Profundidade limitada (um .mat de resultado de toolbox pode
    ter centenas de niveis)."""
    if profundidade > 6:
        return
    if isinstance(valor, dict):
        for k, v in valor.items():
            _abrir_valor(f"{nome}.{k}", v, saida, profundidade + 1)
        return
    if isinstance(valor, (list, tuple)) or (isinstance(valor, np.ndarray) and valor.dtype.kind == "O"):
        itens = list(valor) if isinstance(valor, (list, tuple)) else list(valor.ravel(order="F"))
        for i, v in enumerate(itens, start=1):
            rot = f"{nome}({i})" if isinstance(v, dict) else f"{nome}{{{i}}}"
            _abrir_valor(rot, v, saida, profundidade + 1)
        return
    if type(valor).__name__ == "MatlabOpaque":   # string/table/datetime (MCOS): sem dado numerico legivel
        saida.ignoradas.append((nome, "objeto MATLAB (string/table/datetime)"))
        return
    if isinstance(valor, np.ndarray) and valor.dtype.names:     # objeto/struct nao simplificado
        if valor.size == 1:
            for campo in valor.dtype.names:
                _abrir_valor(f"{nome}.{campo}", valor[campo].ravel()[0], saida, profundidade + 1)
        else:
            saida.ignoradas.append((nome, "objeto"))
        return
    a = _array_numerico(valor)
    if isinstance(a, str):
        saida.ignoradas.append((nome, "complexo"))
        return
    if a is None:
        if isinstance(valor, str) or (isinstance(valor, np.ndarray) and valor.dtype.kind in "US"):
            saida.ignoradas.append((nome, "texto"))
        return
    if a.ndim == 0:                     # escalar de configuracao: nao e' dado
        return
    arr: np.ndarray = a

    def _ler_campo(x: np.ndarray = arr) -> np.ndarray:
        return x
    saida.append(VariavelArquivo(nome, tuple(int(d) for d in arr.shape), str(arr.dtype),
                                 arr.dtype.kind == "b", _ler_campo, {}))


# --------------------------------------------------------------------------- HDF5 (.mat v7.3 e .h5)
def _importar_h5py(caminho: str, motivo: str):
    try:
        import h5py
    except ImportError as e:
        raise ImportError(texto_falta_h5py(caminho, motivo, e)) from e
    return h5py


def _texto_atributo(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("ascii", "replace")
    if isinstance(v, np.ndarray) and v.dtype.kind == "S":
        return b"".join(v.ravel().tolist()).decode("ascii", "replace")
    return str(v)


def _listar_h5(caminho: str, matlab: bool) -> ListaVariaveis:
    """No .mat v7.3, so' entra dataset com MATLAB_class numerica (double, single, int*, uint*, logical):
    os pedacos internos de esparsa (data/ir/jc), string/table/datetime (uint32 com MATLAB_object_decode)
    e texto (char) nao sao dado. Celulas e struct arrays (datasets de referencias) sao seguidos um nivel:
    `c{2}`, `s(2).campo`."""
    h5py = _importar_h5py(caminho, "arquivo MATLAB v7.3 (HDF5)" if matlab else "arquivo HDF5")
    saida = ListaVariaveis()
    try:
        f = h5py.File(caminho, "r")
    except Exception as e:  # noqa: BLE001 -- HDF5 truncado/corrompido
        raise ValueError(f"{os.path.basename(caminho)}: nao foi possivel abrir como HDF5 "
                         f"(arquivo truncado ou corrompido? detalhe: {e}).") from e
    with f:
        def _registrar(rotulo: str, caminho_h5, obj: Any) -> None:
            atributos = {k: obj.attrs[k] for k in obj.attrs.keys()}
            if matlab:
                classe = _texto_atributo(atributos.get("MATLAB_class", ""))
                if classe not in _CLASSES_NUMERICAS:
                    if classe and classe not in ("struct",):
                        saida.ignoradas.append((rotulo, classe))
                    return
                if atributos.get("MATLAB_empty") is not None:
                    return
            if obj.dtype.kind == "V" or (obj.dtype.names and "real" in obj.dtype.names):
                saida.ignoradas.append((rotulo, "complexo"))
                return
            if obj.dtype.kind not in "iufb":
                return
            classe = _texto_atributo(atributos.get("MATLAB_class", "")) if matlab else ""
            logica = classe == "logical" or obj.dtype.kind == "b"
            forma = tuple(int(d) for d in obj.shape)
            if matlab:
                forma = forma[::-1]   # MATLAB grava em ordem de coluna: desinverte

            def _ler(alvo=caminho_h5, inverter: bool = matlab, n: str = rotulo) -> np.ndarray:
                try:
                    with h5py.File(caminho, "r") as g:
                        a = np.asarray(g[alvo][()])
                except Exception as e:  # noqa: BLE001
                    raise _erro_leitura(caminho, n, e) from e
                return a.T if inverter else a
            attrs = {} if matlab else {k: v for k, v in atributos.items()}
            saida.append(VariavelArquivo(rotulo, forma, str(obj.dtype), logica, _ler, attrs))

        def _visitar(nome: str, obj: Any) -> None:
            if nome.startswith("#"):
                return
            if isinstance(obj, h5py.Group):
                if matlab and obj.attrs.get("MATLAB_sparse") is not None:
                    saida.ignoradas.append((nome.replace("/", "."), "esparsa (v7.3)"))
                return
            if not isinstance(obj, h5py.Dataset):
                return
            # pedaco interno de esparsa do v7.3 (grupo com MATLAB_sparse): ja' registrado no grupo
            pai = obj.parent
            if matlab and pai is not None and pai.attrs.get("MATLAB_sparse") is not None:
                return
            rotulo = nome.replace("/", ".") if matlab else nome
            if matlab and obj.dtype == h5py.ref_dtype:
                classe = _texto_atributo(obj.attrs.get("MATLAB_class", ""))
                campo_de_struct_array = classe == ""
                refs = obj[()].T.ravel(order="F") if obj.ndim else [obj[()]]
                for i, r in enumerate(refs, start=1):
                    if not r:
                        continue
                    alvo = f[r]
                    if not isinstance(alvo, h5py.Dataset):
                        sub_ig = f"{rotulo}({i})" if campo_de_struct_array else f"{rotulo}{{{i}}}"
                        saida.ignoradas.append((sub_ig, "struct dentro de celula"))
                        continue
                    if campo_de_struct_array:
                        base, _, campo = rotulo.rpartition(".")
                        sub = f"{base}({i}).{campo}" if base else f"{rotulo}({i})"
                    else:
                        sub = f"{rotulo}{{{i}}}"
                    if alvo.dtype == h5py.ref_dtype:
                        saida.ignoradas.append((sub, "celula aninhada"))
                        continue
                    _registrar(sub, r, alvo)
                return
            _registrar(rotulo, nome, obj)
        f.visititems(_visitar)
    return saida


# --------------------------------------------------------------------------- NumPy
def _listar_npy(caminho: str) -> ListaVariaveis:
    try:
        arr = np.load(caminho, mmap_mode="r", allow_pickle=False)
    except ValueError as e:   # dtype objeto (exige pickle, que o Guaraci nao carrega) ou arquivo invalido
        raise ValueError(f"{os.path.basename(caminho)}: nao e' matriz numerica do NumPy (dtype objeto ou "
                         f"arquivo invalido; o Guaraci nao carrega pickle). Salve com "
                         f"np.save(arquivo, X.astype(float)). Detalhe: {e}") from e
    nome = os.path.splitext(os.path.basename(caminho))[0]
    if arr.dtype.kind not in "iufb":
        return ListaVariaveis([], [(nome, str(arr.dtype))])
    return ListaVariaveis([VariavelArquivo(nome, tuple(int(d) for d in arr.shape), str(arr.dtype),
                                           arr.dtype.kind == "b", lambda: np.load(caminho, allow_pickle=False),
                                           {})])


def _listar_npz(caminho: str) -> ListaVariaveis:
    saida = ListaVariaveis()
    try:
        z = np.load(caminho, allow_pickle=False)
    except Exception as e:  # noqa: BLE001 -- zip truncado/corrompido
        raise ValueError(f"{os.path.basename(caminho)}: nao foi possivel abrir o .npz (truncado ou "
                         f"corrompido? detalhe: {e}).") from e
    with z:
        for k in z.files:
            try:
                a = z[k]
            except ValueError:            # array de objetos: precisaria de pickle
                saida.ignoradas.append((k, "objeto (pickle)"))
                continue
            if a.dtype.kind not in "iufb":
                saida.ignoradas.append((k, str(a.dtype)))
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
    """Lista curta das variaveis. Quando `variaveis` e' a `ListaVariaveis` do arquivo e tem coisa
    ignorada (tabela, string, objeto, complexo...), diz o que foi ignorado e como converter."""
    ignoradas = list(getattr(variaveis, "ignoradas", []) or [])
    if not variaveis:
        txt = "nenhuma variavel numerica"
    else:
        txt = ", ".join(v.descricao() for v in variaveis[:n])
        txt += f" e mais {len(variaveis) - n}" if len(variaveis) > n else ""
    if ignoradas:
        ign = ", ".join(f"'{nome}' ({classe})" for nome, classe in ignoradas[:6])
        ign += f" e mais {len(ignoradas) - 6}" if len(ignoradas) > 6 else ""
        txt += f"; ignoradas por nao serem matriz numerica: {ign}. {DICA_CONVERTER}"
    return txt


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
