"""
dados_io.py — Parsing de JCAMP-DX/ASDF e carregamento de dados (.dx, CSV,
sintético) para o pipeline quimiométrico.

Extraído de pipeline.py como parte da modularização (Fase H). Depende de
Config só para type hints (import guardado por TYPE_CHECKING, para não criar
import circular com pipeline.py, que importa este módulo). pipeline.py
reexporta estes nomes, então `pipeline.carregar_dados(...)`,
`pipeline.parse_title(...)` etc. continuam funcionando sem alteração.
"""
from __future__ import annotations

import glob
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from guaraci.io_registry import registrar_leitor, obter_leitor

if TYPE_CHECKING:
    from guaraci.pipeline import Config

# =========================================================================
#  parse_title v3 — metadata extraction from ##TITLE= JCAMP-DX
#  Expected format (Amazonian oils, ABB MB3600 — GEAAp/UFPA):
#      PURE:        {COD}-{DD-MM-YYYY}_T{N}
#      ADULTERATED: {COD}-{DD-MM-YYYY}-AD-{A|M|S}-{N,NN}%_T{N}
# =========================================================================

CODIGO_ESPECIE: Dict[str, str] = {
    "ACA": "Açaí",       "AND": "Andiroba",        "BAB": "Babaçu",
    "BCB": "Bacaba",     "BUR": "Buriti",          "CAP": "Castanha do Pará",
    "COC": "Coco",       "COP": "Copaíba",         "GOI": "Goiaba",
    "GRA": "Graviola",   "MAR": "Maracujá",
    "AR":  "Maracujá",   # codificacao encontrada no dataset GEAAp/UFPA
    "PAL": "Palmiste",   "PAT": "Patauá",          "PRA": "Pracaxi",
}
ADULTERANTE_NOME: Dict[str, str] = {"A": "algodão", "M": "milho", "S": "soja"}

# Regex robust to deviations found in the real GEAAp/UFPA dataset:
#   - surrounding whitespace                   "## TITLE= GOI-..."
#   - separator after COD/DATE: "-" or "_"     "AND_10-06-2020_AD-S-..."
#   - optional separator before T              "...%T_3"  (no '-' or '_')
#   - Triplicate: "T1" or "T_1"
#   - optional decimal content                 "11%"  /  "1,1%"  /  "10,52%"
#   - "%%" sign (typo)                         "...4,13%%_T1"
_RE_TITLE = re.compile(
    r"^\s*"
    r"(?P<cod>[A-Z]{2,4})"
    r"[-_](?P<data>\d{2}-\d{2}-\d{4})"
    r"(?P<adulteracao>(?:[-_]AD-[AMS]-\d+(?:[.,]\d+)?%%?)?)"
    r"[-_]?T_?(?P<trip>[123])"
    r"\s*$"
)
# Accepts decimal with comma OR period (11,11% and 11.11% coexist in the dataset)
_RE_ADULT = re.compile(r"[-_]AD-([AMS])-(\d+(?:[.,]\d+)?)%%?")


def parse_title(title: str) -> Optional[Dict[str, Any]]:
    """JCAMP-DX TITLE parser. Returns complete dict or None if invalid.

    mae_id field: uniquely identifies the physical sampling point.
    Triplicates T1/T2/T3 of the same point share mae_id, enabling
    GroupKFold/GroupShuffleSplit to prevent replica leakage.

    mae_id format:
        Pure:        'CAP-04-11-2020'
        Adulterated: 'CAP-04-11-2020-A1.03'  (content always 2 decimal places)
    """
    m = _RE_TITLE.match(title.strip())
    if not m:
        return None
    cod  = m.group("cod").upper()
    data = m.group("data")
    trip = int(m.group("trip"))
    am   = _RE_ADULT.match(m.group("adulteracao"))
    adulterante: Optional[str]      = None
    teor:        Optional[float]    = None
    adulterante_nome: Optional[str] = None
    if am:
        adulterante = str(am.group(1))
        teor = float(str(am.group(2)).replace(",", "."))
        adulterante_nome = ADULTERANTE_NOME.get(adulterante, adulterante)
        if teor <= 0:
            return None
    mae_id = (f"{cod}-{data}-{adulterante}{teor:.2f}"
              if adulterante is not None and teor is not None
              else f"{cod}-{data}")
    return {
        "title_original":   title.strip(),
        "cod":              cod,
        "especie":          CODIGO_ESPECIE.get(cod, cod),
        "cod_conhecido":    cod in CODIGO_ESPECIE,
        "data":             data,
        "puro":             adulterante is None,
        "adulterante":      adulterante,
        "adulterante_nome": adulterante_nome,
        "teor":             teor,
        "triplicata":       trip,
        "mae_id":           mae_id,
    }


def extrair_title_do_dx(caminho: str) -> Optional[str]:
    """Extracts the ##TITLE= line without loading all 8192 spectral points."""
    try:
        with open(caminho, "r", encoding="latin-1", errors="replace") as f:
            for linha in f:
                linha = linha.strip()
                if linha.startswith("##TITLE="):
                    return linha[len("##TITLE="):]
                if linha.startswith("##XYDATA") or linha.startswith("##XYPOINTS"):
                    break
    except Exception:
        logging.getLogger(__name__).debug("suppressed non-critical exception", exc_info=True)
    return None


# =========================================================================
#  Data loading
# =========================================================================

def kennard_stone(X: np.ndarray, n_selecionar: int) -> np.ndarray:
    """Selecao de amostras de Kennard & Stone (1969), Technometrics 11:137-148.

    Escolhe `n_selecionar` amostras que cobrem o espaco espectral de forma
    MAXIMAMENTE representativa (uniforme), em vez de aleatoria/estratificada:
    comeca pelas amostras mais extremas e vai sempre adicionando a amostra
    mais distante (em distancia euclidiana) do conjunto ja escolhido. E o
    padrao para dividir calibracao/validacao em calibracao multivariada — a
    validacao herda amostras dentro do span do treino (nao extrapola) e o
    treino cobre bem as bordas do espaco.

    Semente: para evitar a matriz de distancias n×n (custosa em memoria para
    milhares de espectros), o par inicial e aproximado pelo ponto mais distante
    do centroide e, em seguida, o mais distante desse — variante eficiente e
    comum de KS. As iteracoes seguintes sao exatas (distancia minima ao
    conjunto selecionado, atualizada incrementalmente).

    Retorna os indices selecionados, na ORDEM de selecao (os primeiros sao os
    mais extremos), o que permite usar um prefixo como subconjunto menor.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    n_sel = int(min(max(n_selecionar, 1), n))

    centroide = X.mean(axis=0)
    i1 = int(np.argmax(np.sum((X - centroide) ** 2, axis=1)))
    selecionados = [i1]
    min_dist = np.sqrt(np.sum((X - X[i1]) ** 2, axis=1))

    if n_sel >= 2:
        i2 = int(np.argmax(min_dist))
        selecionados.append(i2)
        min_dist = np.minimum(min_dist,
                              np.sqrt(np.sum((X - X[i2]) ** 2, axis=1)))

    while len(selecionados) < n_sel:
        candidato_dist = min_dist.copy()
        candidato_dist[selecionados] = -np.inf   # nunca reescolher
        prox = int(np.argmax(candidato_dist))
        selecionados.append(prox)
        min_dist = np.minimum(min_dist,
                              np.sqrt(np.sum((X - X[prox]) ** 2, axis=1)))

    return np.array(selecionados, dtype=int)


def kennard_stone_split(X: np.ndarray, frac_treino: float = 0.7
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Divisao calibracao/validacao por Kennard-Stone: o treino recebe as
    `frac_treino` amostras mais representativas (bordas + cobertura uniforme);
    o resto vai para validacao. Retorna (idx_treino, idx_val), ambos ordenados.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    n_treino = int(round(max(0.0, min(1.0, frac_treino)) * n))
    n_treino = int(min(max(n_treino, 1), n)) if n > 0 else 0
    ordem = kennard_stone(X, n)          # ranking completo por KS
    idx_treino = np.sort(ordem[:n_treino])
    idx_val = np.sort(ordem[n_treino:])
    return idx_treino, idx_val


def kennard_stone_split_group_aware(
        X: np.ndarray, mae_subset: Optional[np.ndarray], frac_cal: float
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Split calibracao/validacao por Kennard-Stone (1969), group-aware.

    Kennard-Stone escolhe as amostras que cobrem o espaco espectral de
    forma maximamente representativa (bordas + cobertura uniforme), em vez
    de aleatoria -- o padrao classico p/ dividir cal/val em calibracao
    multivariada. Com mae_id disponivel (>=4 grupos), colapsa cada grupo de
    replicas fisicas (T1/T2/T3) num espectro MEDIO antes de rodar KS (no
    nivel de GRUPO, nao de amostra individual) e depois expande de volta
    para os indices de amostra -- preserva o invariante do projeto de nunca
    separar replicas fisicas entre calibracao e validacao. Sem mae_id
    suficiente, roda KS diretamente por amostra.

    Usada por `pipeline.pls_regressao_por_especie`/bloco de regressao
    pooled E por `avaliacao_modelos.benchmark_regressao_por_especie` (mesmo
    split reproduzido deterministicamente nos dois lugares -- comparacao
    apples-to-apples entre PLS-R e os modelos de benchmark).
    """
    if mae_subset is not None and len(np.unique(mae_subset)) >= 4:
        grupos_unicos = np.unique(mae_subset)
        X_grupo = np.array([X[mae_subset == g].mean(axis=0)
                             for g in grupos_unicos])
        idx_treino_g, _idx_val_g = kennard_stone_split(
            X_grupo, frac_treino=frac_cal)
        grupos_treino = set(grupos_unicos[idx_treino_g].tolist())
        mask_treino = np.isin(mae_subset, list(grupos_treino))
        return np.where(mask_treino)[0], np.where(~mask_treino)[0]
    return kennard_stone_split(X, frac_treino=frac_cal)


def gerar_dados_sinteticos(cfg: "Config"):
    """Gera espectros sinteticos de teste, incluindo REPLICAS FISICAS
    (n_replicas_sint por ponto amostral, como T1/T2/T3 do mesmo ponto real) e
    mae_id — sem isso, DD-SIMCA (N2) e as figuras de merito de regressao (N3)
    nunca tinham dados suficientes para treinar/estimar ruido em modo
    sintetico (so 1 amostra "pura" por especie, sem nocao de replica).

    mae_id usa um codigo de 3 letras POR ESPECIE (ESA/ESB/ESC) como prefixo —
    o mesmo formato de 3 letras maiusculas do dataset real (AND/ACA/CAP/...),
    do qual `executar()` deriva o numero de especies via prefixo."""
    print("[INFO] Synthetic MODE — generating test spectra.")
    rng = np.random.default_rng(cfg.seed)
    wavenumbers = np.linspace(4000, 400, cfg.n_pontos_sint)
    conc_base = np.linspace(0, 40, cfg.n_por_classe)
    n_replicas = max(1, int(cfg.n_replicas_sint))

    def esp(c, p1, p2, ruido=0.015):
        frac = c / 100
        return ((1 - frac) * np.exp(-((wavenumbers - p1) ** 2) / (2 * 50 ** 2))
                + frac     * np.exp(-((wavenumbers - (p1 + 20)) ** 2) / (2 * 45 ** 2))
                + 0.6      * np.exp(-((wavenumbers - p2) ** 2) / (2 * 30 ** 2))
                + rng.normal(0, ruido, cfg.n_pontos_sint))

    params  = [(2900, 1740), (2850, 1650), (2960, 1710)]
    classes = ["Esp_A", "Esp_B", "Esp_C"]
    codigos = {"Esp_A": "ESA", "Esp_B": "ESB", "Esp_C": "ESC"}
    X_list, rot_list, conc_list, mae_list = [], [], [], []
    for (p1, p2), cls in zip(params, classes):
        cod = codigos[cls]
        for i, c in enumerate(conc_base):
            mae_ponto = f"{cod}-{i:02d}"
            for _ in range(n_replicas):
                X_list.append(esp(c, p1, p2))
                rot_list.append(cls)
                conc_list.append(c)
                mae_list.append(mae_ponto)

    return (wavenumbers,
            np.array(X_list, dtype=float),
            np.array(rot_list, dtype=str),
            np.array(conc_list, dtype=float),
            np.array(mae_list, dtype=str))


def carregar_csv(caminho, col_classe, col_conc):
    print(f"[INFO] Loading CSV: {caminho}")
    df          = pd.read_csv(caminho)
    rotulos     = np.asarray(df[col_classe].values, dtype=str)
    conc        = np.asarray(df[col_conc].values, dtype=float) if col_conc else None
    excluir     = [c for c in [col_classe, col_conc] if c]
    cols_spec   = [c for c in df.columns if c not in excluir]
    X_raw       = np.asarray(df[cols_spec].values, dtype=float)
    wavenumbers = np.array([float(c) for c in cols_spec])
    return wavenumbers, X_raw, rotulos, conc


def _flush_asdf(y_raw, sign, digits, is_dif):
    if sign == 0:
        return
    num = int("".join(str(d) for d in digits)) if digits else 0
    val = sign * num
    if is_dif:
        val = (y_raw[-1] if y_raw else 0) + val
    y_raw.append(val)


def _decodificar_linha_asdf(s: str, SQZ: dict, DIF: dict, DUP: dict
                             ) -> Tuple[Optional[float], List[float]]:
    """Decodes an ASDF line '(X++(Y..Y))': returns (x_check, [y_raw]).

    x_check is the abscissa value (WITHOUT xfactor yet) that prefixes the line;
    used to anchor the block position on the global grid."""
    i, n = 0, len(s)
    x_str = ""
    while i < n and (s[i].isdigit() or s[i] in "+-"):
        x_str += s[i]; i += 1
    if not x_str:
        return None, []
    x_check = float(x_str)

    y_raw: List[float] = []
    sign, digits, is_dif = 0, [], False
    while i < n:
        ch = s[i]; i += 1
        if ch in SQZ:
            _flush_asdf(y_raw, sign, digits, is_dif)
            v = SQZ[ch]; sign = 1 if v >= 0 else -1
            digits = [abs(v)]; is_dif = False
        elif ch in DIF:
            _flush_asdf(y_raw, sign, digits, is_dif)
            v = DIF[ch]; sign = 1 if v >= 0 else -1
            digits = [abs(v)]; is_dif = True
        elif ch in DUP:
            _flush_asdf(y_raw, sign, digits, is_dif)
            sign = 0; last = y_raw[-1] if y_raw else 0
            for _ in range(DUP[ch]): y_raw.append(last)
        elif ch.isdigit() and sign != 0:
            digits.append(int(ch))
    _flush_asdf(y_raw, sign, digits, is_dif)
    return x_check, y_raw


def parse_dx(filepath):
    """JCAMP-DX parser for compressed '(X++(Y..Y))' format (ASDF).

    Robust axis reconstruction strategy:
      - The X axis is reconstructed as np.linspace(FIRSTX, LASTX, NPOINTS)
        using the header (authoritative), NEVER the encoded X values.
      - Each data line is ANCHORED by its check abscissa (x_check) to the
        correct index on the global grid. This is self-correcting: lines
        that over/under-decode are re-anchored by the next line (fixes the
        blind concatenation bug that used to lose points).
      - Residual gaps (NaN) are linearly interpolated.

    Fallback: if FIRSTX/LASTX/NPOINTS are missing, uses simple concatenation
    with encoded X (legacy mode, less reliable)."""
    SQZ, DIF, DUP = {"@": 0}, {"%": 0}, {}
    for i, c in enumerate("ABCDEFGHI", 1): SQZ[c] =  i
    for i, c in enumerate("abcdefghi", 1): SQZ[c] = -i
    for i, c in enumerate("JKLMNOPQR", 1): DIF[c] =  i
    for i, c in enumerate("jklmnopqr", 1): DIF[c] = -i
    for i, c in enumerate("STUVWXYZs", 1): DUP[c] =  i

    xfactor, yfactor, lendo_dados = 1.0, 1.0, False
    firstx = lastx = None
    npoints: Optional[int] = None
    linhas_dados: List[str] = []

    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        s = line.strip()
        if   s.startswith("##XFACTOR="): xfactor = float(s.split("=", 1)[1])
        elif s.startswith("##YFACTOR="): yfactor = float(s.split("=", 1)[1])
        elif s.startswith("##FIRSTX="):  firstx  = float(s.split("=", 1)[1])
        elif s.startswith("##LASTX="):   lastx   = float(s.split("=", 1)[1])
        elif s.startswith("##NPOINTS="):
            try: npoints = int(float(s.split("=", 1)[1]))
            except ValueError: npoints = None
        elif s.startswith("##XYDATA=") or s.startswith("##XYPOINTS="):
            lendo_dados = True; continue
        elif s.startswith("##END"):      break
        elif s.startswith("##") and lendo_dados: lendo_dados = False
        elif lendo_dados and s:
            linhas_dados.append(s)

    # --- Robust reconstruction anchored by X-check ----------------------
    if (firstx is not None and lastx is not None and npoints
            and npoints > 1 and lastx != firstx):
        dx = (lastx - firstx) / (npoints - 1)
        Y = np.full(npoints, np.nan, dtype=float)
        for s in linhas_dados:
            x_check, y_raw = _decodificar_linha_asdf(s, SQZ, DIF, DUP)
            if x_check is None or not y_raw:
                continue
            x_real = x_check * xfactor
            idx0 = int(round((x_real - firstx) / dx))
            for j, yv in enumerate(y_raw):
                pos = idx0 + j
                if 0 <= pos < npoints:
                    Y[pos] = yv * yfactor
        X = np.linspace(firstx, lastx, npoints)
        nan_mask = np.isnan(Y)
        n_nan = int(nan_mask.sum())
        if n_nan > 0 and n_nan < npoints:
            Y[nan_mask] = np.interp(X[nan_mask], X[~nan_mask], Y[~nan_mask])
        return X, Y

    # --- Legacy fallback (concatenation with encoded X) -----------------
    x_all: List[float] = []
    y_all: List[float] = []
    for s in linhas_dados:
        x_check, y_raw = _decodificar_linha_asdf(s, SQZ, DIF, DUP)
        if x_check is None:
            continue
        x_first = x_check * xfactor
        for yr in y_raw:
            x_all.append(x_first); y_all.append(yr * yfactor)
    if y_all and firstx is not None and lastx is not None:
        x_arr = np.linspace(firstx, lastx, len(y_all))
    else:
        x_arr = np.array(x_all, dtype=float)
    return np.asarray(x_arr, dtype=float), np.array(y_all, dtype=float)


_REGEX_FLOAT = re.compile(r"[-+]?\d+[.,]?\d*(?:[eE][-+]?\d+)?")
_REGEX_CONC  = re.compile(r"(\d+[.,]?\d*)\s*%")


def parse_spectrum(filepath):
    """Generic ASCII parser for .spectrum/.txt/.csv files with x,y per line.
    Tolerates headers, variable separators (space/tab/comma/;) and
    decimal comma. Returns (x, y) as ndarrays.

    Detects known binary formats (Bomem MB, PerkinElmer, Bruker OPUS)
    and emits a message with instructions to re-export as JCAMP-DX."""
    with open(filepath, "rb") as f:
        head = f.read(256)

    # Detect Bomem format (ABB Horizon MB) - UTF-16-LE "Bomem File"
    try:
        head_text = head.decode("utf-16-le", errors="ignore")
        if "Bomem" in head_text or "Horizon" in head_text:
            raise ValueError(
                f"Detected format: ABB Bomem Horizon MB (.spectrum binary).\n"
                f"  File: {os.path.basename(filepath)}\n"
                f"  THIS PARSER DOES NOT READ PROPRIETARY BINARIES for\n"
                f"  scientific integrity. To use with this pipeline:\n"
                f"    1. Open the spectra in Bomem Horizon software\n"
                f"    2. File -> Export -> JCAMP-DX (.dx) or ASCII (.txt)\n"
                f"    3. Point cfg.pasta_entrada to the newly exported folder\n"
                f"  The parser already supports .dx and .txt automatically.")
    except UnicodeDecodeError:
        pass

    # Detect PerkinElmer .sp / Bruker OPUS
    if head[:4] == b"PEPE" or head[:4] == b"\x00\x00\x00\x00" and b"OPUS" in head:
        raise ValueError(
            f"Proprietary binary format detected: {os.path.basename(filepath)}.\n"
            f"  -> export as JCAMP-DX (.dx) or ASCII (.txt) from the original software.")

    n_bad = sum(1 for b in head[:64] if b < 9 or (13 < b < 32) or b > 126)
    if n_bad > 64 * 0.2:
        raise ValueError(
            f"Unrecognized binary file: {os.path.basename(filepath)}\n"
            f"  -> export as JCAMP-DX (.dx) or ASCII (.txt) from the\n"
            f"     instrument software (Bomem/PerkinElmer/Bruker/etc).")

    x_list: List[float] = []
    y_list: List[float] = []
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            tokens = _REGEX_FLOAT.findall(line)
            if len(tokens) < 2:
                continue
            try:
                x = float(tokens[0].replace(",", "."))
                y = float(tokens[1].replace(",", "."))
            except ValueError:
                continue
            x_list.append(x); y_list.append(y)
    return np.array(x_list, dtype=float), np.array(y_list, dtype=float)


def _extrair_conc_filename(nome: str) -> Optional[float]:
    """Extracts the first occurrence of N,NN% from the filename."""
    m = _REGEX_CONC.search(nome)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except ValueError:
        return None


def _listar_arquivos_espectro(pasta: str
                                ) -> Tuple[List[str], Optional[str]]:
    """Searches for spectral files by extension in the folder. Returns (list, ext)."""
    extensoes = [".dx", ".spectrum", ".txt", ".csv"]
    for ext in extensoes:
        candidatos = sorted(glob.glob(os.path.join(pasta, f"*{ext}")))
        if candidatos:
            return candidatos, ext
    return [], None


def _detectar_subpastas_classe(raiz: str) -> List[str]:
    """Returns subfolders that contain >=1 .dx/.spectrum/.txt/.csv file."""
    if not os.path.isdir(raiz):
        return []
    subpastas = []
    for nome in sorted(os.listdir(raiz)):
        caminho = os.path.join(raiz, nome)
        if os.path.isdir(caminho):
            arqs, _ = _listar_arquivos_espectro(caminho)
            if arqs:
                subpastas.append(caminho)
    return subpastas


def carregar_dx(pasta: str, parte_classe: int = 0,
                 extrair_conc: bool = False,
                 usar_parse_title: bool = True
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                              Optional[np.ndarray], Optional[np.ndarray],
                              Optional[pd.DataFrame]]:
    """Loads spectra with auto-detection of folder structure:

        (A) root folder with subfolders (each subfolder = 1 species/class)
            -> recursive via _detectar_subpastas_classe
        (B) single folder with .dx/.spectrum/.txt/.csv files
            -> legacy mode (parte_classe)

    When usar_parse_title=True and the file is .dx, extracts ##TITLE= and
    uses parse_title() for rich metadata (species, adulterant, content,
    replicate, mae_id). Otherwise, uses the filename (fallback).

    Returns: (wavenumbers, X, rotulos, conc, mae_id, metadados_df).
        - mae_id      : ndarray of strings or None if not available
        - metadados_df: pd.DataFrame with all parsed fields
    """
    subpastas = _detectar_subpastas_classe(pasta)
    if subpastas:
        print(f"[INFO] Multi-folder structure detected: {len(subpastas)} "
              f"subfolders in {pasta}")
        arquivos: List[Tuple[str, str]] = []   # (caminho, nome_subpasta)
        for sp in subpastas:
            arqs, _ = _listar_arquivos_espectro(sp)
            arquivos.extend((a, os.path.basename(sp)) for a in arqs)
        ext_usada = os.path.splitext(arquivos[0][0])[1] if arquivos else None
    else:
        if not os.path.isdir(pasta):
            raise FileNotFoundError(
                f"Path does NOT exist: {pasta}\n"
                f"  -> check cfg.pasta_entrada or use cfg.modo='sintetico'.")
        arqs, ext_usada = _listar_arquivos_espectro(pasta)
        if not arqs:
            raise FileNotFoundError(
                f"Folder exists but contains no known spectral files.\n"
                f"  Folder: {pasta}\n"
                f"  Contents (up to 10 items): {os.listdir(pasta)[:10]}")
        arquivos = [(a, "") for a in arqs]

    parser = parse_dx if ext_usada == ".dx" else parse_spectrum
    print(f"[INFO] {len(arquivos)} {ext_usada} files found "
          f"(parser={parser.__name__})")

    pode_parse_title = usar_parse_title and ext_usada == ".dx"

    espectros: List[Tuple[np.ndarray, np.ndarray]] = []
    rotulos: List[str] = []
    concs:   List[Optional[float]] = []
    mae_ids: List[Optional[str]]   = []
    meta_rows: List[Dict[str, Any]] = []
    n_falhos = 0
    n_title_falhos = 0
    cods_desconhecidos: set = set()

    for arq, subpasta_nome in arquivos:
        try:
            x, y = parser(arq)
        except Exception as e:
            n_falhos += 1
            print(f"  [ERROR] {os.path.basename(arq)}: {e}")
            continue
        if len(x) == 0:
            print(f"  [WARNING] {os.path.basename(arq)} has no data — skipped")
            continue

        nome_arq = os.path.splitext(os.path.basename(arq))[0]
        title_parsed: Optional[Dict[str, Any]] = None
        if pode_parse_title:
            title = extrair_title_do_dx(arq)
            if title:
                title_parsed = parse_title(title)
            if title_parsed is None:
                n_title_falhos += 1

        if title_parsed is not None:
            classe = title_parsed["especie"]
            conc_i = title_parsed["teor"]
            mae_i  = title_parsed["mae_id"]
            if not title_parsed["cod_conhecido"]:
                cods_desconhecidos.add(title_parsed["cod"])
            meta = dict(title_parsed)
            meta["arquivo"]   = os.path.basename(arq)
            meta["subpasta"]  = subpasta_nome
        else:
            # Fallback: try to extract COD from filename prefix and map
            # to canonical name (avoids duplicate class due to accent:
            # subfolder 'Copaiba' vs CODIGO_ESPECIE['COP']='Copaíba').
            m_cod = re.match(r"^([A-Z]{2,4})[-_]", nome_arq)
            cod_fb = m_cod.group(1).upper() if m_cod else None
            if cod_fb and cod_fb in CODIGO_ESPECIE:
                classe = CODIGO_ESPECIE[cod_fb]
            elif subpasta_nome:
                classe = subpasta_nome
            else:
                partes = nome_arq.replace("_", "-").split("-")
                classe = partes[parte_classe] if abs(parte_classe) < len(partes) \
                          else nome_arq
            conc_i = _extrair_conc_filename(nome_arq) if extrair_conc else None
            mae_i  = None
            meta = {
                "arquivo":     os.path.basename(arq),
                "subpasta":    subpasta_nome,
                "especie":     classe,
                "teor":        conc_i,
                "puro":        conc_i is None,
                "mae_id":      None,
                "cod":         cod_fb,
                "cod_conhecido": bool(cod_fb and cod_fb in CODIGO_ESPECIE),
            }

        espectros.append((x, y))
        rotulos.append(classe)
        concs.append(conc_i)
        mae_ids.append(mae_i)
        meta_rows.append(meta)

    if not espectros:
        raise ValueError(
            f"No valid spectra loaded. ({n_falhos} files with errors)")
    if n_falhos > 0:
        print(f"[WARNING] {n_falhos} files with parsing errors — skipped.")
    if pode_parse_title:
        if n_title_falhos > 0:
            print(f"[WARNING] {n_title_falhos} files with non-conforming ##TITLE= "
                  f"— using fallback (name/subfolder).")
        if cods_desconhecidos:
            print(f"[CAUTION] CODs not mapped in CODIGO_ESPECIE: "
                  f"{sorted(cods_desconhecidos)}")

    # --- Detection of dominant acquisition range ------------------------
    # Real datasets may mix spectral ranges (e.g. full NIR
    # [0,15797] 8192pts vs narrow range [300,4000]). Mixing is
    # scientifically invalid. Detects the dominant range (mode of xmax
    # rounded to 100 cm-1) and DISCARDS incompatible files with a report.
    maxes = np.array([float(e[0].max()) for e in espectros])
    chave_faixa = np.round(maxes / 100.0) * 100.0
    valores, contagens = np.unique(chave_faixa, return_counts=True)
    faixa_dominante = float(valores[int(np.argmax(contagens))])
    compat = np.abs(chave_faixa - faixa_dominante) < 50.0
    n_drop = int((~compat).sum())

    if n_drop > 0:
        print(f"[CAUTION] Heterogeneous spectral ranges detected. "
              f"Dominant range: xmax~{faixa_dominante:.0f} cm-1 "
              f"({int(compat.sum())} files).")
        print(f"          DISCARDING {n_drop} files with "
              f"incompatible range (not comparable in the same spectral window):")
        # Report by species for discarded files
        from collections import Counter
        drop_especies = Counter(
            meta_rows[i].get("especie", "?")
            for i in range(len(espectros)) if not compat[i])
        for esp, n in sorted(drop_especies.items(), key=lambda kv: -kv[1]):
            exemplos = [meta_rows[i]["arquivo"]
                         for i in range(len(espectros))
                         if not compat[i]
                         and meta_rows[i].get("especie") == esp][:1]
            print(f"            {esp}: {n} arquivos (ex: {exemplos[0] if exemplos else '?'})")
        # Filter all parallel lists
        keep = [i for i in range(len(espectros)) if compat[i]]
        espectros = [espectros[i] for i in keep]
        rotulos   = [rotulos[i]   for i in keep]
        concs     = [concs[i]     for i in keep]
        mae_ids   = [mae_ids[i]   for i in keep]
        meta_rows = [meta_rows[i] for i in keep]
        print(f"          {len(espectros)} files remain in the "
              f"dominant range.")

    if not espectros:
        raise ValueError("No spectra remaining after range filter.")

    # Common grid by interpolation (now within the dominant range)
    xmin  = max(e[0].min() for e in espectros)
    xmax  = min(e[0].max() for e in espectros)
    n_pts = min(2000, min(len(e[0]) for e in espectros))
    grade = np.linspace(xmin, xmax, n_pts)
    X_raw = []
    for x, y in espectros:
        idx = np.argsort(x)
        X_raw.append(np.interp(grade, x[idx], y[idx]))

    # Concentrations (pure samples = 0% by convention)
    if any(c is not None for c in concs):
        conc_arr: Optional[np.ndarray] = np.array(
            [c if c is not None else 0.0 for c in concs], dtype=float)
        n_com_conc = sum(1 for c in concs if c is not None)
        print(f"[INFO] Concentrations extracted: {n_com_conc}/{len(concs)} "
              f"(pure samples treated as 0%).")
    else:
        conc_arr = None

    # mae_id array. For files without valid parse, assigns unique ID
    # (filename) — becomes a "group of 1", isolating them without
    # disabling the entire dataset's GroupKFold.
    n_com_mae = sum(1 for m in mae_ids if m is not None)
    if n_com_mae == 0:
        mae_arr: Optional[np.ndarray] = None
    else:
        n_orfaos = 0
        mae_final: List[str] = []
        for m, row in zip(mae_ids, meta_rows):
            if m is not None:
                mae_final.append(m)
            else:
                mae_final.append(f"orfao_{row['arquivo']}")
                n_orfaos += 1
        mae_arr = np.array(mae_final, dtype=str)
        n_grupos = int(len(np.unique(mae_arr)))
        msg_orfaos = f", {n_orfaos} isolated orphans" if n_orfaos > 0 else ""
        print(f"[INFO] mae_id: {n_com_mae}/{len(mae_ids)} samples parsed, "
              f"{n_grupos} unique groups{msg_orfaos}.")

    metadados_df = pd.DataFrame(meta_rows)

    return (grade,
            np.array(X_raw, dtype=float),
            np.array(rotulos, dtype=str),
            conc_arr, mae_arr, metadados_df)


def _leitor_sintetico(cfg: "Config"):
    wn, X, rot, conc, mae = gerar_dados_sinteticos(cfg)
    return wn, X, rot, conc, mae, None


def _leitor_csv(cfg: "Config"):
    wn, X, rot, conc = carregar_csv(
        cfg.arquivo_csv, cfg.coluna_classe, cfg.coluna_conc)
    return wn, X, rot, conc, None, None


def _leitor_dx(cfg: "Config"):
    return carregar_dx(cfg.pasta_entrada, cfg.parte_classe,
                        cfg.extrair_conc_filename, cfg.usar_parse_title)


def _leitor_imagem(cfg: "Config"):
    from guaraci.dados_imagem import carregar_imagens
    return carregar_imagens(cfg.pasta_entrada, cfg.imagem_recorte,
                             cfg.imagem_incluir_textura)


# Leitores built-in (item 20 da auditoria: registry em vez de if/elif fixo —
# ver io_registry.py para o contrato e como registrar um novo modo).
registrar_leitor("sintetico", _leitor_sintetico)
registrar_leitor("csv", _leitor_csv)
registrar_leitor("dx", _leitor_dx)
registrar_leitor("imagem", _leitor_imagem)


def carregar_dados(cfg: "Config"
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                Optional[np.ndarray], Optional[np.ndarray],
                                Optional[pd.DataFrame]]:
    """Unified data loader. Despacha para o leitor registrado em `cfg.modo`
    (ver io_registry.py). Returns 6-tuple:
        (wavenumbers, X, rotulos, conc, mae_id, metadados_df)
    metadados_df is always None in 'sintetico'/'csv' mode; mae_id is None
    only in 'csv'/'imagem' mode (sem replicas fisicas conhecidas)."""
    return obter_leitor(cfg.modo)(cfg)
