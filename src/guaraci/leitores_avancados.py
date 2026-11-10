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
`dx`/`imagem`). `mae_id` fica None por padrao (nenhum destes formatos tem uma
nocao de agrupamento por replica fisica no nome/estrutura) --
`grouping_guarantee` marcado "none" e DECLARADO em toda saida. Quem TEM a
informacao a fornece em `arquivo_grupos` (CSV id->grupo, id = nome do arquivo
ou da pasta): ai' `mae_id` e' preenchido, a validacao e' agrupada e a
garantia e' "medium" (declarada pelo usuario, nao verificavel pelo software)."""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from guaraci import dados_io as _dados_io
from guaraci.eem_io import (forca_do_espalhamento, larguras_do_config, parse_eem_dat,
                            tratar_espalhamento)
from guaraci.gcms_io import ler_ms_andi, ler_tic_andi_ms
from guaraci.importadores_proprietarios import (
    parse_cromatograma_hplc,
    parse_cromatograma_hplc_canais,
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
                 grade: np.ndarray, integrar: bool = False) -> np.ndarray:
    """Leva cada espectro a `grade`. `integrar=True` (RMN): quando a grade e'
    mais grossa que o eixo nativo, cada ponto da grade vira a MEDIA do sinal
    no seu intervalo (bucketing/binning de area), nao uma amostra pontual --
    linhas de RMN (~0,003 ppm) sao mais finas que o passo da grade (~0,01 ppm)
    e a interpolacao pontual pegaria o pico ou o vale ao acaso (aliasing)."""
    linhas = []
    for x, y in espectros:
        idx = np.argsort(x)
        xs, ys = x[idx], y[idx]
        if integrar and grade.size > 1 and xs.size > grade.size:
            meio = (grade[1:] + grade[:-1]) / 2.0
            bordas = np.concatenate(([grade[0] - (meio[0] - grade[0])], meio,
                                     [grade[-1] + (grade[-1] - meio[-1])]))
            soma = np.concatenate(([0.0], np.cumsum(ys)))
            k = np.searchsorted(xs, bordas)
            n = np.diff(k)
            media = np.diff(soma[k]) / np.maximum(n, 1)
            linhas.append(np.where(n > 0, media, np.interp(grade, xs, ys)))
        else:
            linhas.append(np.interp(grade, xs, ys))
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


def _grupos_por_amostra(itens, arquivo_grupos: Optional[str], nome_formato: str
                        ) -> Optional[np.ndarray]:
    """`mae_id` a partir do arquivo de grupos (B4a): a chave e' o nome da
    amostra (arquivo sem extensao / pasta). Sem arquivo: None (sem garantia
    de agrupamento -- declarado pelo pipeline, nunca silencioso)."""
    if not arquivo_grupos:
        return None
    _col_id, mapa = _dados_io.carregar_mapa_grupos(arquivo_grupos)
    nomes = [nome for _r, nome, _c in itens]
    return _dados_io.atribuir_grupos(nomes, mapa, origem=nome_formato)


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
        extensoes: Optional[Tuple[str, ...]] = None,
        arquivo_grupos: Optional[str] = None,
        max_pontos: int = _dados_io.MAX_PONTOS_ESPECTRO) -> DadosCarregados:
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
        except ImportError:
            # pacote opcional ausente (brukeropus/spcfile): nao e' erro de 1
            # arquivo -- falha ja' no primeiro, com a instrucao de instalacao,
            # em vez de N linhas iguais e um "nenhuma amostra valida" ambiguo.
            raise
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

    if espectros[0][1].ndim == 2:
        grade, X, info_reamostragem = _reamostrar_multicanal(espectros, max_pontos, nome_formato)
    else:
        grade, X, info_reamostragem = _dados_io.reamostrar_espectros(espectros, max_pontos, nome_formato)
    rotulos, conc_arr, metadados_df = _rotulos_e_metadados(itens)
    metadados_df.attrs["reamostragem"] = info_reamostragem
    mae_id = _grupos_por_amostra(itens, arquivo_grupos, nome_formato)
    print(f"[INFO] {nome_formato}: {len(itens)} amostras, "
          f"{len(np.unique(rotulos))} classe(s).")
    return grade, X, rotulos, conc_arr, mae_id, metadados_df


def load_generic_folder_dataset(
        pasta: str, parser: Parser1D, nome_formato: str,
        eh_amostra: Callable[[str], bool],
        resolver: Callable[[str], str],
        arquivo_grupos: Optional[str] = None,
        integrar: bool = False,
        max_pontos: int = _dados_io.MAX_PONTOS_ESPECTRO) -> DadosCarregados:
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

    if espectros[0][1].ndim == 2:
        grade, X, info_reamostragem = _reamostrar_multicanal(espectros, max_pontos, nome_formato)
    else:
        grade, X, info_reamostragem = _dados_io.reamostrar_espectros(espectros, max_pontos, nome_formato,
                                                                      integrar=integrar)
    rotulos, conc_arr, metadados_df = _rotulos_e_metadados(itens)
    metadados_df.attrs["reamostragem"] = info_reamostragem
    mae_id = _grupos_por_amostra(itens, arquivo_grupos, nome_formato)
    print(f"[INFO] {nome_formato}: {len(itens)} amostras, "
          f"{len(np.unique(rotulos))} classe(s).")
    return grade, X, rotulos, conc_arr, mae_id, metadados_df


#: acima disto (tempos x canais) o modo multicanal exige um teto de pontos ou menos comprimentos de onda
MAX_VARIAVEIS_MULTICANAL = 500_000


def _reamostrar_multicanal(espectros, max_pontos: int, nome_formato: str):
    """`espectros`: `[(tempo, matriz[n_tempos, n_canais]), ...]`. Cada canal vai a' mesma grade de tempo
    (mesma regra de `reamostrar_espectros`) e os canais sao concatenados em sequencia: o vetor da amostra e'
    [canal 1 em todos os tempos, canal 2 em todos os tempos, ...] (matriz tempo x canal achatada por canal).
    O eixo devolvido e' a POSICAO 1..n (nao tem unidade); a grade de tempo e o numero de canais ficam em
    `info`."""
    n_c = {m.shape[1] for _t, m in espectros}
    if len(n_c) != 1:
        raise ValueError(f"{nome_formato}: amostras com numero diferente de canais ({sorted(n_c)}).")
    n_c = n_c.pop()
    n_t = min(len(t) for t, _m in espectros)
    n_t_final = n_t if int(max_pontos or 0) <= 0 else min(n_t, int(max_pontos))
    if n_t_final * n_c > MAX_VARIAVEIS_MULTICANAL:
        raise ValueError(
            f"{nome_formato}: {n_c} canais x {n_t_final} tempos = {n_t_final * n_c} variaveis (limite "
            f"{MAX_VARIAVEIS_MULTICANAL}). Defina cromatograma_max_pontos (ex.: 500) ou liste menos "
            f"comprimentos de onda em cromatografia_canal.")
    blocos, info = [], None
    grade_t = None
    for j in range(n_c):
        g, Xj, inf = _dados_io.reamostrar_espectros([(t, m[:, j]) for t, m in espectros], max_pontos,
                                                    nome_formato, avisar=False)
        blocos.append(Xj)
        grade_t = g if grade_t is None else grade_t
        info = info or inf
    if info is not None:
        info = dict(info, n_canais=int(n_c))
        _dados_io.avisar_reamostragem(info, f" em cada um dos {n_c} canais (tempo x canal achatado)")
    X = np.hstack(blocos)
    eixo = np.arange(1.0, X.shape[1] + 1.0)
    print(f"[INFO] {nome_formato}: {n_c} canais x {len(grade_t)} tempos = {X.shape[1]} variaveis por amostra "
          f"(eixo = posicao; canal 1 em todos os tempos, depois canal 2...).")
    return eixo, X, info


# --- Detectores de "pasta-amostra" (RMN/HPLC) ------------------------------

def _expnos_rmn(caminho: str) -> List[str]:
    """Subpastas EXPNO (numericas, com `pdata`) de uma pasta de experimento
    TopSpin: o layout real e' `<nome>/<expno>/pdata/<procno>/1r`."""
    try:
        filhos = os.listdir(caminho)
    except OSError:
        return []
    return sorted((n for n in filhos if n.isdigit()
                   and os.path.isdir(os.path.join(caminho, n, "pdata"))), key=int)


def _eh_pasta_rmn(caminho: str) -> bool:
    return os.path.isdir(os.path.join(caminho, "pdata")) or bool(_expnos_rmn(caminho))


def _resolver_rmn(caminho: str) -> str:
    if not os.path.isdir(os.path.join(caminho, "pdata")):
        caminho = os.path.join(caminho, _expnos_rmn(caminho)[0])   # menor EXPNO
    pdata = os.path.join(caminho, "pdata")
    numeros = sorted(
        (n for n in os.listdir(pdata) if os.path.isdir(os.path.join(pdata, n))),
        key=lambda n: (len(n), n))
    if not numeros:
        raise ValueError(f"{caminho}: 'pdata' sem nenhum subdiretorio numerado")
    return os.path.join(pdata, numeros[0])


def _eh_pasta_hplc(caminho: str) -> bool:
    return os.path.isdir(caminho) and caminho.lower().endswith((".d", ".raw"))


def _arq_grupos(cfg) -> Optional[str]:  # noqa: ANN001
    return getattr(cfg, "group_file", None) or None


# --- Adaptadores LeitorDados (contrato `Callable[[Config], DadosCarregados]`) ---

def _max_espectro(cfg) -> int:  # noqa: ANN001
    return int(getattr(cfg, "espectro_max_pontos", _dados_io.MAX_PONTOS_ESPECTRO))


def _max_cromatograma(cfg) -> int:  # noqa: ANN001
    """Cromatograma: padrao SEM limite (0). Decimar a 2000 pontos fazia a altura de um pico depender da
    fase da amostragem (ate' 8,5 % entre injecoes identicas)."""
    return int(getattr(cfg, "cromatograma_max_pontos", 0) or 0)


def _registrar(cfg, resultado: DadosCarregados) -> DadosCarregados:  # noqa: ANN001
    """Guarda em `cfg.reamostragem_aplicada` o que o leitor reduziu (None se nada), para o resumo do modelo."""
    meta = resultado[5]
    cfg.reamostragem_aplicada = meta.attrs.get("reamostragem") if meta is not None else None
    return resultado


def _leitor_opus(cfg) -> DadosCarregados:  # noqa: ANN001 -- Config via TYPE_CHECKING alhures
    return _registrar(cfg, load_generic_file_dataset(
        cfg.input_folder, parse_opus, "OPUS", extensoes=None,
        arquivo_grupos=_arq_grupos(cfg), max_pontos=_max_espectro(cfg)))


def _leitor_spc(cfg) -> DadosCarregados:
    return _registrar(cfg, load_generic_file_dataset(
        cfg.input_folder, parse_spc, "SPC", extensoes=(".spc",),
        arquivo_grupos=_arq_grupos(cfg), max_pontos=_max_espectro(cfg)))


def _leitor_sp(cfg) -> DadosCarregados:
    return _registrar(cfg, load_generic_file_dataset(
        cfg.input_folder, parse_sp, "PerkinElmer .sp", extensoes=(".sp",),
        arquivo_grupos=_arq_grupos(cfg), max_pontos=_max_espectro(cfg)))


def _leitor_gcms(cfg) -> DadosCarregados:
    canal = str(getattr(cfg, "cromatografia_canal", "") or "").strip()
    rotulos_canais: List[str] = []

    def parser(caminho: str) -> Tuple[np.ndarray, np.ndarray]:
        if not canal:     # padrao historico: TIC
            tic = ler_tic_andi_ms(caminho)
            return tic.tempo_min, tic.intensidade
        t, m, rots = ler_ms_andi(caminho, canal)
        if not rotulos_canais:
            rotulos_canais.extend(rots)
        elif rots != rotulos_canais:
            raise ValueError(
                f"faixa de m/z diferente da 1a amostra ({rots[0]}-{rots[-1]} vs {rotulos_canais[0]}-"
                f"{rotulos_canais[-1]}); liste os m/z em cromatografia_canal (ex.: 57,71,85).")
        return (t, m[:, 0]) if m.shape[1] == 1 else (t, m)
    res = _registrar(cfg, load_generic_file_dataset(
        cfg.input_folder, parser, "GC-MS ANDI-MS", extensoes=(".cdf",),
        arquivo_grupos=_arq_grupos(cfg), max_pontos=_max_cromatograma(cfg)))
    cfg.canais_dad_lidos = [f"m/z {r}" for r in rotulos_canais] if canal else None
    return res


def _leitor_rmn(cfg) -> DadosCarregados:
    return _registrar(cfg, load_generic_folder_dataset(
        cfg.input_folder, parse_rmn_bruker, "RMN Bruker",
        eh_amostra=_eh_pasta_rmn, resolver=_resolver_rmn,
        arquivo_grupos=_arq_grupos(cfg), integrar=True, max_pontos=_max_espectro(cfg)))


def _leitor_hplc(cfg) -> DadosCarregados:
    detector = str(getattr(cfg, "hplc_detector", "UV") or "UV")
    canal = str(getattr(cfg, "cromatografia_canal", "") or "").strip()
    rotulos_canais: List[str] = []

    def parser(caminho: str) -> Tuple[np.ndarray, np.ndarray]:
        if not canal:     # padrao historico: 1o canal do detector
            return parse_cromatograma_hplc(caminho, detector=detector)
        t, m, rots = parse_cromatograma_hplc_canais(caminho, detector=detector, canal=canal)
        if not rotulos_canais:
            rotulos_canais.extend(rots)
        elif rots != rotulos_canais:
            raise ValueError(f"canais diferentes dos da 1a amostra ({rots[:3]}... vs {rotulos_canais[:3]}...)")
        return (t, m[:, 0]) if m.shape[1] == 1 else (t, m)
    res = _registrar(cfg, load_generic_folder_dataset(
        cfg.input_folder, parser, "HPLC/GC de fabricante",
        eh_amostra=_eh_pasta_hplc, resolver=lambda c: c,
        arquivo_grupos=_arq_grupos(cfg), max_pontos=_max_cromatograma(cfg)))
    cfg.canais_dad_lidos = list(rotulos_canais) if canal else None
    return res


#: Razao crista/vizinhanca (`eem_io.forca_do_espalhamento`) a partir da qual a EEM bruta e' dita de espalhamento forte.
LIMIAR_ESPALHAMENTO_FORTE = 3.0


def _leitor_eem(cfg) -> DadosCarregados:
    """EEM (fluorescencia excitacao-emissao) e' 2D por amostra, nao um
    espectro 1D -- achatado (flatten) em vetor 1D p/ caber no mesmo
    contrato/pipeline PLS-DA/PLS-R de todo outro mode aqui (perde a
    estrutura 2D explicita, mas o dado inteiro continua presente; quem
    quiser a decomposicao PARAFAC de verdade usa
    `eem_multiway.parafac_eem`/`construir_tensor_eem` diretamente sobre a
    pasta, ver [T] Tecnicas avancadas no menu). Eixo retornado e' um indice
    posicional (0..n-1), sem unidade fisica -- a grade emissao x excitacao
    original fica em `metadados_df.attrs['grade_excitacao']`.

    Espalhamento Rayleigh/Raman (`eem_espalhamento`): `nenhum` (padrao) nao toca no dado e AVISA se a EEM bruta tem
    espalhamento forte; `nan`/`zero`/`interp` tratam as bandas (ver `eem_io.tratar_espalhamento`). O vetor achatado
    devolvido tem a regiao removida igual a 0 em TODAS as amostras (colunas constantes: o classificador as descarta no
    ajuste, e a predicao seleciona so as colunas mantidas, entao as duas etapas enxergam a mesma regiao); a mascara
    fica em `metadados_df.attrs['mascara_eem']` (vetor achatado, na mesma ordem) para o PARAFAC tratar como dado
    faltante (`mascara_eem_faltante`: o que ele deve ignorar)."""
    import glob as _glob
    arquivos = sorted(_glob.glob(
        os.path.join(cfg.input_folder, "**", "*.dat"), recursive=True))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo .dat (EEM) encontrado em {cfg.input_folder} "
            f"(nem nas subpastas).")

    modo_esp = str(getattr(cfg, "eem_espalhamento", "nenhum") or "nenhum")
    larguras = larguras_do_config(getattr(cfg, "eem_espalhamento_largura_nm", ()))
    usar_raman = bool(getattr(cfg, "eem_raman", False))
    brutas: List[np.ndarray] = []
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
            arq_ref = arq
        elif (len(excitacao) != len(grade_excitacao) or len(emissao) != len(grade_emissao)
              or not np.allclose(excitacao, grade_excitacao, atol=0.05)
              or not np.allclose(emissao, grade_emissao, atol=0.05)):
            # Truncar o vetor achatado desalinha TODAS as linhas de emissao
            # (cada valor passa a ocupar o lugar de outro comprimento de onda),
            # e grades de mesmo tamanho mas valores diferentes passavam caladas.
            raise ValueError(
                f"EEM: {os.path.basename(arq)} tem grade {len(emissao)} emissoes "
                f"({emissao[0]:g}-{emissao[-1]:g} nm) x {len(excitacao)} excitacoes "
                f"({excitacao[0]:g}-{excitacao[-1]:g} nm), diferente de "
                f"{os.path.basename(arq_ref)} ({len(grade_emissao)} x {len(grade_excitacao)}, "
                f"{grade_emissao[0]:g}-{grade_emissao[-1]:g} / {grade_excitacao[0]:g}-"
                f"{grade_excitacao[-1]:g} nm). Exporte todas as amostras com a mesma grade "
                f"(ou interpole para uma grade comum antes).")
        brutas.append(np.asarray(matriz, dtype=float))
        nomes.append(os.path.splitext(os.path.basename(arq))[0])
        classes.append(os.path.basename(os.path.dirname(arq)) or "eem")

    if not brutas:
        raise ValueError(
            f"Nenhuma amostra EEM valida carregada ({n_falhos} arquivos "
            f"com erro).")
    # tratamento do espalhamento (a regiao removida depende so' da grade: a mesma em todas as amostras)
    matrizes: List[np.ndarray] = []
    removida = np.zeros(brutas[0].shape, dtype=bool)
    ausente = np.zeros(brutas[0].shape, dtype=bool)
    forca = float(np.median([forca_do_espalhamento(m, grade_excitacao, grade_emissao, larguras) for m in brutas[:20]]))
    for m in brutas:
        mt, rem, aus = tratar_espalhamento(m, grade_excitacao, grade_emissao, modo_esp, larguras, raman=usar_raman)
        matrizes.append(mt)
        removida |= rem
        ausente |= aus
    if modo_esp in ("nan", "zero"):
        # a regiao fica identica (0) em todas as amostras: coluna constante, descartada igual no ajuste e na predicao
        for mt in matrizes:
            mt[removida] = 0.0
    if ausente.any():
        for mt in matrizes:
            mt[ausente] = 0.0     # NaN vindo do arquivo: faltante em TODAS as amostras (a coluna sai do classificador)
    n_col = brutas[0].size       # grade identica garantida acima
    matrizes = [mt.ravel() for mt in matrizes]
    X = np.array(matrizes, dtype=float)
    grade = np.arange(n_col, dtype=float)
    rotulos = np.array(classes, dtype=str)
    metadados_df = pd.DataFrame({"amostra": nomes, "especie": rotulos})
    metadados_df.attrs["grade_excitacao"] = grade_excitacao
    metadados_df.attrs["grade_emissao"] = grade_emissao
    faltante = (removida if modo_esp in ("nan", "interp") else np.zeros_like(removida)) | ausente
    metadados_df.attrs["mascara_eem"] = removida.ravel()
    metadados_df.attrs["mascara_eem_faltante"] = faltante.ravel()
    metadados_df.attrs["eem_espalhamento"] = {
        "modo": modo_esp, "larguras_nm": list(larguras), "raman": usar_raman,
        "forca_do_espalhamento": round(forca, 2), "n_celulas_removidas": int(removida.sum()),
        "n_celulas_faltantes": int(faltante.sum())}
    if modo_esp == "nenhum" and forca >= LIMIAR_ESPALHAMENTO_FORTE:
        aviso = (f"EEM bruta com espalhamento Rayleigh/Raman forte (crista de {forca:.0f}x a vizinhanca) e nenhum tratamento "
                 f"aplicado: PCA, PLS e sobretudo o PARAFAC ficam dominados pelas diagonais de espalhamento. Use "
                 f"eem_espalhamento: nan (ou interp) no config.yaml, ou exporte a EEM ja sem espalhamento.")
        metadados_df.attrs["aviso_espalhamento"] = aviso
        print(f"[CAUTION] {aviso}")
    if modo_esp != "nenhum" or ausente.any():
        print(f"[INFO] EEM: espalhamento '{modo_esp}': {int(removida.sum())} de {removida.size} celulas tratadas"
              f"{f', {int(ausente.sum())} NaN do arquivo tratados como faltantes' if ausente.any() else ''}.")
    mae_id = _grupos_por_amostra(
        [(r, n, None) for r, n in zip(classes, nomes)], _arq_grupos(cfg), "EEM")
    print(f"[INFO] EEM: {len(nomes)} amostras carregadas (achatadas p/ "
          f"vetor 1D, {n_col} pontos).")
    return grade, X, rotulos, None, mae_id, metadados_df


register_reader("opus", _leitor_opus)
register_reader("spc", _leitor_spc)
register_reader("sp", _leitor_sp)
register_reader("rmn", _leitor_rmn)
register_reader("hplc", _leitor_hplc)
register_reader("gcms", _leitor_gcms)
register_reader("eem", _leitor_eem)
