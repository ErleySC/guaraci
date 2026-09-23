# -*- coding: utf-8 -*-
"""leitores_avancados.py -- registra no `io_registry` os leitores de formato
de instrumento ja' implementados em `importadores_proprietarios.py`/
`gcms_io.py` (OPUS, SPC, PerkinElmer .sp, RMN Bruker, HPLC/GC de fabricante,
GC-MS ANDI-MS), fechando a lacuna encontrada na auditoria de acessibilidade
CLI/web de preparacao para o primeiro usuario externo: os parsers de baixo
nivel (contrato `(X, Y)` por arquivo/pasta) existiam e estavam testados
contra fixture real, mas nenhum `cfg.mode` os alcancava -- nenhuma entrada
de `io_registry._LEITORES`, nenhum caminho de menu/wizard.

Deliberadamente um modulo NOVO e separado de `dados_io.py`: o docstring de
`importadores_proprietarios.py` avisa que generalizar `load_dx`/
`_listar_arquivos_espectro` (codigo ja' congelado, Bloco B) para esses
formatos e' "trabalho de escopo proprio" que arriscaria codigo testado --
este modulo faz esse trabalho SEM tocar em `dados_io.py`, reaproveitando so'
o contrato publico (`DadosCarregados`, `register_reader`) e duas funcoes
privadas de baixo risco (`_extrair_conc_filename`, puramente uma regex sobre
string) por composicao, nunca por edicao.

Convencao de pasta (mesma ideia de `load_dx`, adaptada a formato/arquivo vs.
formato/pasta-por-amostra):
    - leitores POR ARQUIVO (opus/spc/sp/gcms): cada arquivo dentro da pasta
      (ou de cada subpasta-classe) e' 1 amostra.
    - leitores POR PASTA (rmn/hplc): cada SUBPASTA que "parece uma amostra"
      (RMN: contem `pdata/<N>/`; HPLC: termina em `.D`/`.raw`) e' 1 amostra;
      subpastas-classe (se existirem) ficam 1 nivel acima.
Em ambos os casos: sem subpastas-classe -> 1 classe so' (nome da pasta
raiz); com subpastas-classe -> 1 classe por subpasta (mesma convencao de
`dx`/`imagem`). `mae_id` fica None (nenhum destes formatos tem uma nocao de
agrupamento por replica fisica no nome/estrutura, mesma situacao honesta ja'
documentada para `csv` generico) -- `grouping_guarantee` marcado "none"."""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from guaraci import dados_io as _dados_io
from guaraci.eem_io import parse_eem_dat
from guaraci.gcms_io import ler_tic_andi_ms
from guaraci.importadores_proprietarios import (
    parse_cromatograma_hplc,
    parse_opus,
    parse_rmn_bruker,
    parse_sp,
    parse_spc,
)
from guaraci.io_registry import DadosCarregados, register_reader

__all__ = [
    "load_generic_file_dataset",
    "load_generic_folder_dataset",
]

Parser1D = Callable[[str], Tuple[np.ndarray, np.ndarray]]


def _grade_comum(espectros: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    xmin = max(float(np.min(x)) for x, _ in espectros)
    xmax = min(float(np.max(x)) for x, _ in espectros)
    if xmin >= xmax:
        raise ValueError(
            "Amostras sem sobreposicao de eixo utilizavel -- verifique se "
            "todos os arquivos/pastas sao do mesmo instrumento/faixa.")
    n_pts = min(2000, min(len(x) for x, _ in espectros))
    return np.linspace(xmin, xmax, n_pts)


def _interpolar(espectros: List[Tuple[np.ndarray, np.ndarray]],
                 grade: np.ndarray) -> np.ndarray:
    linhas = []
    for x, y in espectros:
        idx = np.argsort(x)
        linhas.append(np.interp(grade, x[idx], y[idx]))
    return np.array(linhas, dtype=float)


def _rotulos_e_metadados(itens: List[Tuple[str, str, Optional[float]]]
                          ) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    """`itens`: lista de (rotulo_classe, nome_amostra, teor_ou_None)."""
    rotulos = np.array([r for r, _, _ in itens], dtype=str)
    concs = [c for _, _, c in itens]
    conc_arr: Optional[np.ndarray] = None
    if any(c is not None for c in concs):
        conc_arr = np.array([c if c is not None else 0.0 for c in concs],
                             dtype=float)
    metadados_df = pd.DataFrame({
        "amostra": [n for _, n, _ in itens],
        "especie": rotulos,
        "teor": concs,
    })
    return rotulos, conc_arr, metadados_df


def _subpastas_classe(raiz: str, eh_amostra: Callable[[str], bool]) -> List[str]:
    """Subpastas de `raiz` que contem >=1 amostra (por `eh_amostra`) --
    generalizacao de `dados_io._detectar_subpastas_classe` para formatos que
    nao usam extensao de arquivo fixa (OPUS) ou cuja "amostra" e' uma pasta
    (RMN/HPLC), sem tocar na versao original (so' DX/CSV/imagem/txt)."""
    if not os.path.isdir(raiz):
        return []
    subpastas = []
    for nome in sorted(os.listdir(raiz)):
        caminho = os.path.join(raiz, nome)
        if os.path.isdir(caminho):
            try:
                filhos = sorted(os.listdir(caminho))
            except OSError:
                continue
            if any(eh_amostra(os.path.join(caminho, f)) for f in filhos):
                subpastas.append(caminho)
    return subpastas


def load_generic_file_dataset(
        pasta: str, parser: Parser1D, nome_formato: str,
        extensoes: Optional[Tuple[str, ...]] = None) -> DadosCarregados:
    """Carrega um dataset em que CADA ARQUIVO e' 1 amostra (OPUS/SPC/.sp/
    GC-MS ANDI-MS). `extensoes=None` (caso do OPUS, sem extensao fixa
    filtravel -- ver docstring de `importadores_proprietarios.py`) tenta
    TODO arquivo regular da pasta como candidato, descartando com aviso
    (nao com excecao) os que `parser` rejeitar -- mesma disciplina
    defensiva de `dados_io.load_dx` (1 arquivo malformado nao derruba o
    dataset inteiro)."""
    if not os.path.isdir(pasta):
        raise FileNotFoundError(
            f"Pasta nao existe: {pasta}\n"
            f"  -> verifique cfg.input_folder (formato {nome_formato}).")

    def eh_candidato(caminho: str) -> bool:
        if not os.path.isfile(caminho):
            return False
        if extensoes is None:
            return True
        return os.path.splitext(caminho)[1].lower() in extensoes

    subpastas = _subpastas_classe(pasta, eh_candidato)
    arquivos: List[Tuple[str, str]] = []   # (caminho, classe)
    if subpastas:
        for sp in subpastas:
            classe = os.path.basename(sp)
            for nome in sorted(os.listdir(sp)):
                caminho = os.path.join(sp, nome)
                if eh_candidato(caminho):
                    arquivos.append((caminho, classe))
    else:
        classe_unica = os.path.basename(os.path.normpath(pasta)) or nome_formato
        for nome in sorted(os.listdir(pasta)):
            caminho = os.path.join(pasta, nome)
            if eh_candidato(caminho):
                arquivos.append((caminho, classe_unica))

    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo candidato a {nome_formato} encontrado em "
            f"{pasta}.")

    espectros: List[Tuple[np.ndarray, np.ndarray]] = []
    itens: List[Tuple[str, str, Optional[float]]] = []
    n_falhos = 0
    for caminho, classe in arquivos:
        try:
            x, y = parser(caminho)
        except Exception as e:  # noqa: BLE001 -- parsing defensivo de
            # arquivo binario de terceiro (formato variavel/instrumento
            # nao-garantido); 1 arquivo invalido nao derruba o dataset,
            # mesma disciplina de dados_io.load_dx.
            n_falhos += 1
            print(f"  [ERROR] {os.path.basename(caminho)}: {e}")
            continue
        if len(x) == 0:
            print(f"  [WARNING] {os.path.basename(caminho)} sem dado -- pulado")
            continue
        nome_amostra = os.path.splitext(os.path.basename(caminho))[0]
        conc = _dados_io._extrair_conc_filename(nome_amostra)
        espectros.append((x, y))
        itens.append((classe, nome_amostra, conc))

    if not espectros:
        raise ValueError(
            f"Nenhuma amostra {nome_formato} valida carregada "
            f"({n_falhos} arquivos com erro).")
    if n_falhos > 0:
        print(f"[WARNING] {n_falhos} arquivos {nome_formato} com erro de "
              f"parsing -- pulados.")

    grade = _grade_comum(espectros)
    X = _interpolar(espectros, grade)
    rotulos, conc_arr, metadados_df = _rotulos_e_metadados(itens)
    print(f"[INFO] {nome_formato}: {len(itens)} amostras, "
          f"{len(np.unique(rotulos))} classe(s).")
    return grade, X, rotulos, conc_arr, None, metadados_df


def load_generic_folder_dataset(
        pasta: str, parser: Parser1D, nome_formato: str,
        eh_amostra: Callable[[str], bool],
        resolver: Callable[[str], str]) -> DadosCarregados:
    """Carrega um dataset em que CADA SUBPASTA e' 1 amostra (RMN Bruker/
    HPLC). `eh_amostra(caminho)` decide se um item e' uma pasta-amostra;
    `resolver(caminho)` converte a pasta detectada no caminho exato que o
    `parser` espera (ex.: RMN: acrescenta `pdata/<N>`; HPLC: identidade)."""
    if not os.path.isdir(pasta):
        raise FileNotFoundError(
            f"Pasta nao existe: {pasta}\n"
            f"  -> verifique cfg.input_folder (formato {nome_formato}).")

    def eh_amostra_dir(caminho: str) -> bool:
        return os.path.isdir(caminho) and eh_amostra(caminho)

    subpastas = _subpastas_classe(pasta, eh_amostra_dir)
    amostras: List[Tuple[str, str]] = []   # (caminho_pasta_amostra, classe)
    if subpastas:
        for sp in subpastas:
            classe = os.path.basename(sp)
            for nome in sorted(os.listdir(sp)):
                caminho = os.path.join(sp, nome)
                if eh_amostra_dir(caminho):
                    amostras.append((caminho, classe))
    else:
        classe_unica = os.path.basename(os.path.normpath(pasta)) or nome_formato
        for nome in sorted(os.listdir(pasta)):
            caminho = os.path.join(pasta, nome)
            if eh_amostra_dir(caminho):
                amostras.append((caminho, classe_unica))
        # Caso degenerado: a propria `pasta` JA E' a amostra unica (ex.:
        # usuario aponta direto p/ 1 diretorio .D, ou 1 raiz de experimento
        # RMN) -- sem subpastas-amostra nenhuma, mas a raiz mesma qualifica.
        if not amostras and eh_amostra_dir(pasta):
            amostras.append((pasta, classe_unica))

    if not amostras:
        raise FileNotFoundError(
            f"Nenhuma pasta-amostra {nome_formato} encontrada em {pasta}.")

    espectros: List[Tuple[np.ndarray, np.ndarray]] = []
    itens: List[Tuple[str, str, Optional[float]]] = []
    n_falhos = 0
    for caminho, classe in amostras:
        nome_amostra = os.path.basename(os.path.normpath(caminho))
        try:
            x, y = parser(resolver(caminho))
        except Exception as e:  # noqa: BLE001 -- mesma disciplina defensiva
            # do loader por arquivo acima.
            n_falhos += 1
            print(f"  [ERROR] {nome_amostra}: {e}")
            continue
        if len(x) == 0:
            print(f"  [WARNING] {nome_amostra} sem dado -- pulado")
            continue
        conc = _dados_io._extrair_conc_filename(nome_amostra)
        espectros.append((x, y))
        itens.append((classe, nome_amostra, conc))

    if not espectros:
        raise ValueError(
            f"Nenhuma amostra {nome_formato} valida carregada "
            f"({n_falhos} pastas com erro).")
    if n_falhos > 0:
        print(f"[WARNING] {n_falhos} pastas {nome_formato} com erro de "
              f"parsing -- puladas.")

    grade = _grade_comum(espectros)
    X = _interpolar(espectros, grade)
    rotulos, conc_arr, metadados_df = _rotulos_e_metadados(itens)
    print(f"[INFO] {nome_formato}: {len(itens)} amostras, "
          f"{len(np.unique(rotulos))} classe(s).")
    return grade, X, rotulos, conc_arr, None, metadados_df


# --- Detectores de "pasta-amostra" (RMN/HPLC) ------------------------------

def _eh_pasta_rmn(caminho: str) -> bool:
    return os.path.isdir(os.path.join(caminho, "pdata"))


def _resolver_rmn(caminho: str) -> str:
    pdata = os.path.join(caminho, "pdata")
    numeros = sorted(
        (n for n in os.listdir(pdata) if os.path.isdir(os.path.join(pdata, n))),
        key=lambda n: (len(n), n))
    if not numeros:
        raise ValueError(f"{caminho}: 'pdata' sem nenhum subdiretorio numerado")
    return os.path.join(pdata, numeros[0])


def _eh_pasta_hplc(caminho: str) -> bool:
    return os.path.isdir(caminho) and caminho.lower().endswith((".d", ".raw"))


# --- Adaptadores LeitorDados (contrato `Callable[[Config], DadosCarregados]`) ---

def _leitor_opus(cfg) -> DadosCarregados:  # noqa: ANN001 -- Config via TYPE_CHECKING alhures
    return load_generic_file_dataset(cfg.input_folder, parse_opus, "OPUS",
                                      extensoes=None)


def _leitor_spc(cfg) -> DadosCarregados:
    return load_generic_file_dataset(cfg.input_folder, parse_spc, "SPC",
                                      extensoes=(".spc",))


def _leitor_sp(cfg) -> DadosCarregados:
    return load_generic_file_dataset(cfg.input_folder, parse_sp,
                                      "PerkinElmer .sp", extensoes=(".sp",))


def _leitor_gcms(cfg) -> DadosCarregados:
    def parser(caminho: str) -> Tuple[np.ndarray, np.ndarray]:
        tic = ler_tic_andi_ms(caminho)
        return tic.tempo_min, tic.intensidade
    return load_generic_file_dataset(cfg.input_folder, parser,
                                      "GC-MS ANDI-MS", extensoes=(".cdf",))


def _leitor_rmn(cfg) -> DadosCarregados:
    return load_generic_folder_dataset(
        cfg.input_folder, parse_rmn_bruker, "RMN Bruker",
        eh_amostra=_eh_pasta_rmn, resolver=_resolver_rmn)


def _leitor_hplc(cfg) -> DadosCarregados:
    detector = str(getattr(cfg, "hplc_detector", "UV") or "UV")

    def parser(caminho: str) -> Tuple[np.ndarray, np.ndarray]:
        return parse_cromatograma_hplc(caminho, detector=detector)
    return load_generic_folder_dataset(
        cfg.input_folder, parser, "HPLC/GC de fabricante",
        eh_amostra=_eh_pasta_hplc, resolver=lambda c: c)


def _leitor_eem(cfg) -> DadosCarregados:
    """EEM (fluorescencia excitacao-emissao) e' 2D por amostra, nao um
    espectro 1D -- achatado (flatten) em vetor 1D p/ caber no mesmo
    contrato/pipeline PLS-DA/PLS-R de todo outro mode aqui (perde a
    estrutura 2D explicita, mas o dado inteiro continua presente; quem
    quiser a decomposicao PARAFAC de verdade usa
    `eem_multiway.parafac_eem`/`construir_tensor_eem` diretamente sobre a
    pasta, ver [T] Tecnicas avancadas no menu). Eixo retornado e' um indice
    posicional (0..n-1), sem unidade fisica -- a grade emissao x excitacao
    original fica em `metadados_df.attrs['grade_excitacao']`."""
    import glob as _glob
    arquivos = sorted(_glob.glob(
        os.path.join(cfg.input_folder, "**", "*.dat"), recursive=True))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo .dat (EEM) encontrado em {cfg.input_folder} "
            f"(nem nas subpastas).")

    matrizes: List[np.ndarray] = []
    nomes: List[str] = []
    classes: List[str] = []
    grade_excitacao = None
    grade_emissao = None
    n_falhos = 0
    for arq in arquivos:
        try:
            excitacao, emissao, matriz, _relatorio = parse_eem_dat(arq)
        except Exception as e:  # noqa: BLE001 -- parsing defensivo, mesma
            # disciplina dos demais leitores deste modulo.
            n_falhos += 1
            print(f"  [ERROR] {os.path.basename(arq)}: {e}")
            continue
        if grade_excitacao is None:
            grade_excitacao = list(excitacao)
            grade_emissao = list(emissao)
        matrizes.append(np.asarray(matriz, dtype=float).ravel())
        nomes.append(os.path.splitext(os.path.basename(arq))[0])
        classes.append(os.path.basename(os.path.dirname(arq)) or "eem")

    if not matrizes:
        raise ValueError(
            f"Nenhuma amostra EEM valida carregada ({n_falhos} arquivos "
            f"com erro).")
    n_col = min(len(m) for m in matrizes)
    if any(len(m) != n_col for m in matrizes):
        print("[CAUTION] EEM: grades de emissao/excitacao heterogeneas entre "
              "amostras -- truncando ao menor vetor comum.")
    X = np.array([m[:n_col] for m in matrizes], dtype=float)
    grade = np.arange(n_col, dtype=float)
    rotulos = np.array(classes, dtype=str)
    metadados_df = pd.DataFrame({"amostra": nomes, "especie": rotulos})
    metadados_df.attrs["grade_excitacao"] = grade_excitacao
    metadados_df.attrs["grade_emissao"] = grade_emissao
    print(f"[INFO] EEM: {len(nomes)} amostras carregadas (achatadas p/ "
          f"vetor 1D, {n_col} pontos).")
    return grade, X, rotulos, None, None, metadados_df


register_reader("opus", _leitor_opus)
register_reader("spc", _leitor_spc)
register_reader("sp", _leitor_sp)
register_reader("rmn", _leitor_rmn)
register_reader("hplc", _leitor_hplc)
register_reader("gcms", _leitor_gcms)
register_reader("eem", _leitor_eem)
