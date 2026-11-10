"""hsi_io.py — Estrutura interna de cubo hiperespectral (HSI) e leitor do
formato ENVI (par `.hdr` texto + `.bin` binario bruto), Passo 94 da
`INSTRUCAO_HSI_MINIMO_VIAVEL.md`.

Formato confirmado por leitura DIRETA (nao suposta) de cubos reais de
referencia: cada gravacao
e' um par `<nome>.hdr` (ASCII, chave=valor) + `<nome>.bin` (float32,
interleave BIP, sem cabecalho embutido -- "header offset = 0"). Exemplo real
inspecionado (uma gravacao de camera VIS):

    ENVI
    samples = 64
    lines = 64
    bands = 224
    header offset = 0
    file type = ENVI Standard
    data type = 4          <- float32 (codigo ENVI padrao)
    interleave = bip
    byte order = 0          <- little-endian

Ha' `.hdr` que NAO trazem a lista de comprimentos de onda (campo
`wavelength = {...}` do padrao ENVI) -- o mapeamento banda->comprimento de
onda vem a' parte, em outro arquivo do fornecedor. Por isso
`load_envi_cube` aceita `wavelengths` como parametro OPCIONAL -- quando o
`.hdr` nao traz a lista, quem chama fornece a de fora.

ENVI e' o formato dominante na literatura de HSI (confirmado no Passo 93);
nao ha suporte a outros formatos (MAT/HDF5) neste ciclo -- fora de escopo
do "minimo viavel", adicionar quando um dataset real exigir.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from guaraci import agrupamento_pastas as _ap
from guaraci.hsi_blocos import (CalibracaoBrancoEscuro, CuboHSI,
                                pesos_reamostragem)

__all__ = [
    "HSICubeMetadata",
    "parse_envi_header",
    "load_envi_cube",
    "load_hsi_folder_dataset",
]

# Codigos de tipo de dado do padrao ENVI (subconjunto que este leitor aceita
# -- os demais levantam erro explicito em vez de interpretar bytes errado).
_ENVI_DTYPES: Dict[int, str] = {
    1: "u1",    # byte / uint8
    2: "<i2",   # int16
    3: "<i4",   # int32
    4: "<f4",   # float32
    5: "<f8",   # float64
    12: "<u2",  # uint16
    13: "<u4",  # uint32
}


@dataclass
class HSICubeMetadata:
    """Metadados de UM cubo hiperespectral -- analogo ao `wavenumbers` do
    modo `.dx`, mas com 2 eixos espaciais a mais. `wavelengths` fica `None`
    quando nem o `.hdr` nem quem chamou `load_envi_cube` forneceu a lista --
    nesse caso o eixo espectral so' tem indice (mesma limitacao ja
    documentada e aceita em `dados_imagem.py` para o eixo simbolico)."""
    n_lines: int
    n_samples: int
    n_bands: int
    interleave: str
    dtype_envi: int
    byte_order: int
    header_offset: int
    wavelengths: Optional[np.ndarray] = None
    camera: Optional[str] = None
    extras: Dict[str, str] = field(default_factory=dict)


def parse_envi_header(texto: str) -> Dict[str, str]:
    """Parseia um `.hdr` ENVI (texto ASCII, `chave = valor`, uma por linha,
    exceto listas entre `{...}` que podem quebrar linha) para um dict de
    chave (minuscula, sem espaco nas pontas) -> valor cru (string, ainda sem
    conversao de tipo -- quem chama decide o tipo esperado por campo)."""
    linhas = texto.replace("\r\n", "\n").split("\n")
    campos: Dict[str, str] = {}
    i = 0
    while i < len(linhas):
        linha = linhas[i].strip()
        if not linha or linha.upper() == "ENVI" or "=" not in linha:
            i += 1
            continue
        chave, _, valor = linha.partition("=")
        chave = chave.strip().lower()
        valor = valor.strip()
        # Listas `{ ... }` podem continuar em linhas seguintes ate' fechar.
        if valor.startswith("{") and "}" not in valor:
            partes = [valor]
            i += 1
            while i < len(linhas) and "}" not in linhas[i]:
                partes.append(linhas[i].strip())
                i += 1
            if i < len(linhas):
                partes.append(linhas[i].strip())
            valor = " ".join(partes)
        campos[chave] = valor.strip("{} ").strip()
        i += 1
    return campos


def _wavelengths_do_header(campos: Dict[str, str]) -> Optional[np.ndarray]:
    bruto = campos.get("wavelength")
    if not bruto:
        return None
    try:
        return np.array([float(v) for v in bruto.split(",") if v.strip()],
                        dtype=float)
    except ValueError:
        return None  # campo presente mas ilegivel -- trata como ausente


#: Extensoes do arquivo de dados ENVI vistas na pratica (o padrao ENVI nao fixa
#: extensao: Specim grava `.raw`, ENVI/L3Harris `.img`/`.dat` ou nenhuma).
_EXTENSOES_DADOS_ENVI: Tuple[str, ...] = (".bin", ".raw", ".img", ".dat", ".bil", ".bsq", ".bip", "")


def _resolver_arquivo_dados_envi(caminho_hdr: str) -> str:
    """Arquivo binario que acompanha `caminho_hdr` (mesma raiz de nome),
    tentando as extensoes de `_EXTENSOES_DADOS_ENVI` em minuscula e
    maiuscula. Sem nenhum candidato, devolve o `.bin` (o erro de leitura
    cita o caminho esperado)."""
    raiz = caminho_hdr[:-4] if caminho_hdr.lower().endswith(".hdr") else caminho_hdr
    for ext in _EXTENSOES_DADOS_ENVI:
        for cand in ((raiz + ext, raiz + ext.upper()) if ext else (raiz,)):
            if os.path.isfile(cand) and not cand.lower().endswith(".hdr"):
                return cand
    return raiz + ".bin"


@dataclass
class _ParametrosEnvi:
    campos: Dict[str, str]
    n_samples: int
    n_lines: int
    n_bands: int
    dtype_envi: int
    interleave: str
    header_offset: int
    byte_order: int
    dtype_np: np.dtype
    fator: float          # 'reflectance scale factor' (1.0 se ausente/invalido)
    wl: Optional[np.ndarray]


def _ler_parametros_envi(caminho_hdr: str,
                         wavelengths: Optional[np.ndarray] = None) -> _ParametrosEnvi:
    """Le e valida o `.hdr` ENVI (sem tocar no binario)."""
    with open(caminho_hdr, "r", encoding="utf-8", errors="replace") as f:
        campos = parse_envi_header(f.read())

    obrigatorios = ("samples", "lines", "bands", "data type", "interleave")
    faltando = [c for c in obrigatorios if c not in campos]
    if faltando:
        raise ValueError(
            f"{caminho_hdr}: campo(s) ENVI obrigatorio(s) ausente(s): "
            f"{faltando}. Nao e' um header ENVI valido/completo.")

    n_samples = int(campos["samples"])
    n_lines = int(campos["lines"])
    n_bands = int(campos["bands"])
    dtype_envi = int(campos["data type"])
    interleave = campos["interleave"].strip().lower()
    header_offset = int(campos.get("header offset", "0"))
    byte_order = int(campos.get("byte order", "0"))

    if dtype_envi not in _ENVI_DTYPES:
        raise ValueError(
            f"{caminho_hdr}: 'data type' ENVI {dtype_envi} nao suportado "
            f"por este leitor. Suportados: {sorted(_ENVI_DTYPES)}.")
    if interleave not in ("bip", "bil", "bsq"):
        raise ValueError(
            f"{caminho_hdr}: interleave '{interleave}' desconhecido "
            f"(esperado bip/bil/bsq).")

    dtype_np = np.dtype(_ENVI_DTYPES[dtype_envi])
    if byte_order == 1:
        dtype_np = dtype_np.newbyteorder(">")

    fator = 1.0
    escala = campos.get("reflectance scale factor")
    if escala:
        try:
            fator = float(escala)
        except ValueError:
            fator = 1.0
        if not fator > 0:
            fator = 1.0

    wl = _wavelengths_do_header(campos)
    if wl is not None and campos.get("wavelength units", "").strip().lower() in (
            "micrometers", "micrometer", "microns", "micron", "um", "\u00b5m"):
        wl = wl * 1000.0   # eixo interno sempre em nm (faixa da tela [2.4] e figuras)
    if wl is None:
        wl = np.asarray(wavelengths, dtype=float) if wavelengths is not None else None
    if wl is not None and wl.size != n_bands:
        raise ValueError(
            f"{caminho_hdr}: {wl.size} comprimentos de onda fornecidos, "
            f"cubo tem {n_bands} bandas -- nao correspondem.")
    return _ParametrosEnvi(campos, n_samples, n_lines, n_bands, dtype_envi, interleave,
                           header_offset, byte_order, dtype_np, fator, wl)


def _validar_tamanho_binario(caminho_bin: str, p: _ParametrosEnvi) -> None:
    """Confere o tamanho do binario contra o `.hdr` sem le-lo."""
    n_esperado = p.n_samples * p.n_lines * p.n_bands
    n_lido = max(0, (os.path.getsize(caminho_bin) - p.header_offset) // p.dtype_np.itemsize)
    if n_lido != n_esperado:
        raise ValueError(
            f"{caminho_bin}: {n_lido} valores lidos, esperado "
            f"{n_esperado} ({p.n_lines}x{p.n_samples}x{p.n_bands}, "
            f"interleave={p.interleave}) -- arquivo truncado/corrompido ou "
            f".hdr nao corresponde a este .bin.")


def load_envi_cube(caminho_bin: str, caminho_hdr: Optional[str] = None,
                    wavelengths: Optional[np.ndarray] = None,
                    ) -> Tuple[np.ndarray, HSICubeMetadata]:
    """Le um par ENVI `.bin`+`.hdr` e devolve `(cubo, metadados)`, com o
    cubo SEMPRE normalizado para a forma `(altura, largura, n_bandas)`
    independente do `interleave` original do arquivo (bip/bil/bsq) -- o
    resto do modulo HSI (Passos 95+) nunca precisa saber que interleave o
    dataset de origem usava.

    `caminho_hdr`: default `caminho_bin` com a extensao trocada para
    `.hdr` (convencao universal do formato). `wavelengths`: usado SOMENTE
    se o `.hdr` nao trouxer o campo `wavelength` (ver docstring do modulo
    -- ha' cameras cujo `.hdr` nao traz).
    """
    if caminho_hdr is None:
        caminho_hdr = (caminho_bin[:-4] + ".hdr" if caminho_bin.lower().endswith(".bin")
                       else caminho_bin + ".hdr")

    par = _ler_parametros_envi(caminho_hdr, wavelengths)
    campos = par.campos
    n_samples, n_lines, n_bands = par.n_samples, par.n_lines, par.n_bands
    dtype_envi, interleave = par.dtype_envi, par.interleave
    header_offset, byte_order, dtype_np = par.header_offset, par.byte_order, par.dtype_np

    n_esperado = n_samples * n_lines * n_bands
    bruto = np.fromfile(caminho_bin, dtype=dtype_np, offset=header_offset)
    if bruto.size != n_esperado:
        raise ValueError(
            f"{caminho_bin}: {bruto.size} valores lidos, esperado "
            f"{n_esperado} ({n_lines}x{n_samples}x{n_bands}, "
            f"interleave={interleave}) -- arquivo truncado/corrompido ou "
            f".hdr nao corresponde a este .bin.")

    if interleave == "bip":       # (linha, coluna, banda) -- ja' na ordem alvo
        cubo = bruto.reshape(n_lines, n_samples, n_bands)
    elif interleave == "bil":     # (linha, banda, coluna)
        cubo = bruto.reshape(n_lines, n_bands, n_samples).transpose(0, 2, 1)
    else:                          # bsq: (banda, linha, coluna)
        cubo = bruto.reshape(n_bands, n_lines, n_samples).transpose(1, 2, 0)
    # float32: cubos de camera real (centenas de MB) dobrariam de tamanho em
    # float64 sem ganho de precisao (o sensor tem 12-16 bit).
    cubo = np.ascontiguousarray(cubo, dtype=np.float32)
    if par.fator != 1.0:   # ex.: int16 com reflectancia x10000
        cubo /= np.float32(par.fator)
    wl = par.wl

    meta = HSICubeMetadata(
        n_lines=n_lines, n_samples=n_samples, n_bands=n_bands,
        interleave=interleave, dtype_envi=dtype_envi, byte_order=byte_order,
        header_offset=header_offset, wavelengths=wl,
        extras={k: v for k, v in campos.items()
                if k not in ("samples", "lines", "bands", "data type",
                             "interleave", "header offset", "byte order",
                             "wavelength")})
    return cubo, meta


#: Extensao que marca 1 gravacao ENVI -- o `.bin` irmao e' assumido presente
#: (ver `load_envi_cube`, que resolve `.hdr` -> `.bin` pela mesma raiz de
#: nome). So' contar `.hdr` evita listar a mesma gravacao 2x.
_EXTENSOES_HSI: Tuple[str, ...] = (".hdr",)

#: Tolerancia (nm) para considerar iguais os eixos de bandas de dois cubos.
TOLERANCIA_EIXO_NM = 1.0

CALIBRACOES = ("nenhuma", "branco_escuro")
EIXOS_BANDAS = ("recusar", "reamostrar")

# ---------------------------------------------------------------------------
# Calibracao radiometrica branco/escuro (R11-a)
# ---------------------------------------------------------------------------
_RE_REFERENCIA = re.compile(r"^(white|dark)_?ref|_(white|dark)_?ref$")


def _e_arquivo_de_referencia(caminho: str) -> bool:
    """True para os cubos de referencia no padrao Specim (WHITEREF_<nome>,
    DARKREF_<nome>, <nome>_WHITEREF...): nao sao amostras do estudo."""
    raiz = os.path.splitext(os.path.basename(caminho))[0].lower()
    return bool(_RE_REFERENCIA.search(raiz))


#: Chaves do .hdr em que o usuario pode apontar a referencia de cada cubo
#: (convencao do GUARACI; o padrao ENVI nao define estas chaves).
_CHAVES_HDR_REF = {"white": ("white reference file", "white reference", "whiteref file", "whiteref"),
                   "dark": ("dark reference file", "dark reference", "darkref file", "darkref")}


def _hdr_com_raiz(caminho: str) -> Optional[str]:
    """`.hdr` correspondente a um arquivo de referencia dado (o proprio `.hdr`,
    ou o binario irmao); None se nao existe."""
    if os.path.isfile(caminho) and caminho.lower().endswith(".hdr"):
        return caminho
    raiz = os.path.splitext(caminho)[0] if os.path.isfile(caminho) else caminho
    for ext in (".hdr", ".HDR"):
        if os.path.isfile(raiz + ext):
            return raiz + ext
    return None


def _procurar_na_pasta(pasta: str, tipo: str, stem: Optional[str]) -> Optional[str]:
    """`.hdr` de referencia em `pasta`: nomes especificos do cubo (`WHITEREF_<stem>`,
    `<stem>_WHITEREF`) quando `stem` e' dado, senao os genericos (`WHITEREF`)."""
    base = "whiteref" if tipo == "white" else "darkref"
    alt = "white_ref" if tipo == "white" else "dark_ref"
    if stem is not None:
        nomes = [f"{base}_{stem}", f"{stem}_{base}", f"{alt}_{stem}", f"{stem}_{alt}"]
    else:
        nomes = [base, alt]
    try:
        entradas = {os.path.splitext(n)[0].lower(): n for n in os.listdir(pasta)
                    if n.lower().endswith(".hdr")}
    except OSError:
        return None
    for nome in nomes:
        if nome.lower() in entradas:
            return os.path.join(pasta, entradas[nome.lower()])
    return None


def localizar_referencia(caminho_hdr: str, raiz: str, campos: Dict[str, str], tipo: str,
                         configurada: str = "") -> Tuple[Optional[str], str]:
    """Onde esta a referencia `tipo` ("white"/"dark") do cubo `caminho_hdr`?
    Ordem: (1) caminho da config (`hsi_referencia_branco/escuro`: arquivo ou
    pasta); (2) chave do .hdr do cubo; (3) arquivos do padrao Specim ao lado do
    cubo (`WHITEREF_<nome>`); (4) `WHITEREF`/`DARKREF` genericos na pasta do
    cubo e nas pastas acima dela ate' a raiz do estudo (uma referencia por
    sessao/lote). Devolve (`.hdr` ou None, descricao da origem)."""
    pasta_cubo = os.path.dirname(os.path.abspath(caminho_hdr))
    stem = os.path.splitext(os.path.basename(caminho_hdr))[0]
    if configurada:
        if os.path.isdir(configurada):
            achado = (_procurar_na_pasta(configurada, tipo, stem)
                      or _procurar_na_pasta(configurada, tipo, None))
            if achado:
                return achado, f"pasta da config ({configurada})"
        else:
            achado = _hdr_com_raiz(configurada)
            if achado:
                return achado, f"arquivo da config ({configurada})"
        return None, f"config aponta para '{configurada}', que nao existe ou nao tem {tipo}ref"
    for chave in _CHAVES_HDR_REF[tipo]:
        v = campos.get(chave)
        if v:
            alvo = v if os.path.isabs(v) else os.path.join(pasta_cubo, v)
            achado = _hdr_com_raiz(alvo)
            if achado:
                return achado, f"chave '{chave}' do .hdr"
            return None, f"chave '{chave}' do .hdr aponta para '{v}', que nao existe"
    achado = _procurar_na_pasta(pasta_cubo, tipo, stem)
    if achado:
        return achado, "ao lado do cubo (padrao Specim)"
    atual = pasta_cubo
    raiz_abs = os.path.abspath(raiz)
    while True:
        achado = _procurar_na_pasta(atual, tipo, None)
        if achado:
            return achado, f"pasta da sessao ({atual})"
        if os.path.abspath(atual) == raiz_abs or os.path.dirname(atual) == atual:
            break
        atual = os.path.dirname(atual)
    return None, "nao encontrada"


def _topos_saturacao(par: "_ParametrosEnvi") -> Tuple[int, ...]:
    """Valores de saturacao plausiveis para um dado inteiro (8/10/12/14/16 bit)."""
    if par.dtype_np.kind not in "iu":
        return ()
    teto = np.iinfo(par.dtype_np).max
    return tuple(2 ** k - 1 for k in (8, 10, 12, 14, 16) if 2 ** k - 1 <= teto)


def _carregar_referencia(hdr: str) -> Dict[str, object]:
    """Le um cubo de referencia e reduz a media sobre as linhas: `(colunas, bandas)`."""
    arq = _resolver_arquivo_dados_envi(hdr)
    cubo, meta = load_envi_cube(arq, hdr)
    par = _ler_parametros_envi(hdr)
    topos = _topos_saturacao(par)
    maximo = float(np.nanmax(cubo)) if cubo.size else 0.0
    frac_sat = 0.0
    if maximo in topos:
        frac_sat = float(np.mean(cubo >= maximo))
    return {"media": np.nanmean(cubo, axis=0).astype(np.float64), "n_linhas": meta.n_lines,
            "n_colunas": meta.n_samples, "n_bandas": meta.n_bands, "wl": meta.wavelengths,
            "frac_saturada": frac_sat, "maximo": maximo, "tint": meta.extras.get("tint"),
            "arquivo": hdr}


#: Fracao de pontos (coluna x banda) com W <= D acima da qual a calibracao e' recusada.
LIMITE_W_MENOR_QUE_D = 0.01
#: Fracao de valores do branco no teto do conversor acima da qual a calibracao e' recusada (abaixo: aviso).
LIMITE_BRANCO_SATURADO = 0.05
_AVISO_BRANCO_SATURADO = 0.001


def _construir_calibracao(nome: str, par: "_ParametrosEnvi", branco: Dict[str, object],
                          escuro: Dict[str, object], origem: str) -> Tuple[CalibracaoBrancoEscuro, List[str]]:
    """R = (I - D)/(W - D) com as checagens do R11-a. Levanta ValueError (mensagem
    acionavel) quando a calibracao seria invalida; devolve avisos quando e' so' suspeita."""
    avisos: List[str] = []
    for rot, ref in (("WHITEREF", branco), ("DARKREF", escuro)):
        if int(ref["n_bandas"]) != par.n_bands:  # type: ignore[call-overload]
            raise ValueError(
                f"{nome}: {rot} ({os.path.basename(str(ref['arquivo']))}) tem {ref['n_bandas']} bandas e o "
                f"cubo {par.n_bands} -- referencia de outra camera/configuracao.")
        if par.wl is not None and ref["wl"] is not None:
            desvio = float(np.max(np.abs(np.asarray(ref["wl"]) - par.wl)))
            if desvio > TOLERANCIA_EIXO_NM:
                raise ValueError(
                    f"{nome}: o eixo de comprimentos de onda de {rot} difere do cubo em ate' "
                    f"{desvio:.1f} nm (tolerancia {TOLERANCIA_EIXO_NM:g} nm) -- referencia de outra camera.")
        for chave in ("tint",):
            a, b = par.campos.get(chave), ref.get(chave)
            try:
                if a is not None and b is not None and abs(float(a) - float(str(b))) > 0.01 * abs(float(a)):
                    avisos.append(f"{nome}: tempo de integracao do cubo ({a}) difere do de {rot} ({b}); "
                                  f"a razao so' vale com o mesmo tempo de integracao.")
            except ValueError:
                pass
    W = np.asarray(branco["media"], dtype=np.float64)
    D = np.asarray(escuro["media"], dtype=np.float64)
    if W.shape != D.shape:
        if W.shape[0] != D.shape[0]:           # colunas diferentes: espectro medio unico
            W = np.nanmean(W, axis=0, keepdims=True)
            D = np.nanmean(D, axis=0, keepdims=True)
            avisos.append(f"{nome}: WHITEREF e DARKREF tem larguras diferentes; usado o espectro medio de cada um.")
    if W.shape[0] != par.n_samples:
        W = np.nanmean(W, axis=0, keepdims=True)
        D = np.nanmean(D, axis=0, keepdims=True)
        if int(branco["n_colunas"]) != 1:  # type: ignore[call-overload]
            avisos.append(f"{nome}: a referencia tem {branco['n_colunas']} colunas e o cubo {par.n_samples}; "
                          f"usado o espectro medio da referencia (sem correcao por coluna).")
    den = W - D
    invalido = ~np.isfinite(den) | (den <= 0)
    frac_inv = float(np.mean(invalido))
    if frac_inv > LIMITE_W_MENOR_QUE_D:
        raise ValueError(
            f"{nome}: o branco nao e' maior que o escuro em {frac_inv:.1%} dos pontos (coluna x banda) "
            f"(limite {LIMITE_W_MENOR_QUE_D:.0%}). Confira se WHITEREF e DARKREF nao estao trocados, se o "
            f"branco foi fotografado com o mesmo tempo de integracao e se ha' sinal no padrao branco.")
    if frac_inv > 0:
        avisos.append(f"{nome}: W <= D em {frac_inv:.2%} dos pontos; nesses pontos R fica indefinida (NaN).")
    fs = float(branco["frac_saturada"])  # type: ignore[arg-type]
    if fs > LIMITE_BRANCO_SATURADO:
        raise ValueError(
            f"{nome}: o branco esta saturado ({fs:.1%} dos valores no teto do conversor, {branco['maximo']:g} DN): "
            f"R seria superestimada. Refaca o WHITEREF com menos exposicao.")
    if fs > _AVISO_BRANCO_SATURADO:
        avisos.append(f"{nome}: {fs:.2%} dos valores do branco estao no teto do conversor "
                      f"({branco['maximo']:g} DN); R pode estar superestimada nessas bandas.")
    den_ok = np.where(invalido, np.nan, den)
    mediana_w = float(np.nanmedian(W))
    if mediana_w > 0 and float(np.nanmedian(den_ok)) / mediana_w < 0.05:
        avisos.append(f"{nome}: o escuro e' quase do tamanho do branco (W-D < 5% de W): faixa dinamica baixa, "
                      f"R muito ruidosa.")
    return CalibracaoBrancoEscuro(escuro=D.astype(np.float32), denominador=den_ok.astype(np.float32),
                                  origem=origem), avisos


def _checar_amostra_vs_referencia(nome: str, cubo: CuboHSI, branco: Dict[str, object],
                                  escuro: Dict[str, object], par: "_ParametrosEnvi") -> List[str]:
    """Avisos que exigem olhar o proprio cubo (poucas linhas centrais)."""
    avisos: List[str] = []
    r0 = max(0, cubo.n_lines // 2 - 4)
    mm = cubo._abrir()
    try:
        bruto = np.array(mm[r0:r0 + 8], dtype=np.float64)
    finally:
        del mm
    if bruto.size == 0:
        return avisos
    mw = float(np.nanmedian(np.asarray(branco["media"])))
    if mw > 0 and float(np.nanmedian(bruto)) > 1.25 * mw:
        avisos.append(f"{nome}: a mediana do cubo ({np.nanmedian(bruto):.0f}) e' maior que a do branco "
                      f"({mw:.0f}): R > 1 em boa parte da cena (branco com menos sinal que a amostra, ou "
                      f"tempo de integracao diferente).")
    topos = _topos_saturacao(par)
    if topos and float(np.max(bruto)) in topos:
        f = float(np.mean(bruto >= np.max(bruto)))
        if f > 0.01:
            avisos.append(f"{nome}: {f:.1%} dos valores do cubo estao no teto do conversor "
                          f"({np.max(bruto):g} DN): amostra saturada, R subestimada nesses pixels.")
    return avisos


# ---------------------------------------------------------------------------
# Eixo de bandas entre cubos (R11-b)
# ---------------------------------------------------------------------------
def _rel(caminho: str, base: str) -> str:
    """Caminho relativo a `base` para mensagens/relatorio (cai no proprio caminho se for
    de outra unidade, caso do Windows)."""
    try:
        return os.path.relpath(caminho, base)
    except ValueError:
        return caminho


def _resumo_eixo(wl: np.ndarray) -> str:
    return f"{wl.min():g}-{wl.max():g} nm, {wl.size} bandas"


def _comparar_eixos(ref_nome: str, wl_ref: Optional[np.ndarray], n_ref: int,
                    nome: str, wl: Optional[np.ndarray], n: int, modo: str):
    """Devolve (reamostragem|None, desvio_max_nm). Levanta ValueError se os eixos
    nao sao comparaveis (ou nao reamostraveis)."""
    if n != n_ref:
        raise ValueError(
            f"{nome}: {n} bandas, mas outra(s) "
            f"gravacao(oes) desta pasta tem {n_ref} -- nao da "
            f"pra' formar um dataset coerente (sensores/cameras "
            f"incompativeis misturados na mesma pasta). Separe-os em "
            f"pastas diferentes.")
    if (wl_ref is None) != (wl is None):
        raise ValueError(
            f"{nome}: este cubo {'nao ' if wl is None else ''}traz 'wavelength' no .hdr e {ref_nome} "
            f"{'' if wl is None else 'nao '}traz -- nao ha' como saber se os eixos de bandas coincidem. "
            f"Complete os .hdr (ou separe as cameras em pastas diferentes).")
    if wl_ref is None or wl is None:
        return None, 0.0
    desvio = float(np.max(np.abs(wl - wl_ref)))
    if desvio <= TOLERANCIA_EIXO_NM:
        return None, desvio
    esp = float(np.median(np.abs(np.diff(wl_ref)))) if n_ref > 1 else float("inf")
    pequena = desvio <= 0.5 * esp and np.all(np.diff(wl) > 0) and np.all(np.diff(wl_ref) > 0)
    base = (f"{nome}: o eixo de comprimentos de onda ({_resumo_eixo(wl)}) difere do de {ref_nome} "
            f"({_resumo_eixo(wl_ref)}) em ate' {desvio:.1f} nm (tolerancia {TOLERANCIA_EIXO_NM:g} nm).")
    if modo == "reamostrar" and pequena:
        return pesos_reamostragem(wl, wl_ref), desvio
    if modo == "reamostrar":
        raise ValueError(
            base + f" A diferenca passa de meia banda ({0.5 * esp:.1f} nm), grande demais para reamostrar "
            f"sem inventar informacao: cameras diferentes (ex.: VNIR e SWIR) nao devem ser misturadas. "
            f"Separe-as em pastas diferentes.")
    raise ValueError(
        base + " Provavelmente sao cameras/configuracoes diferentes (ex.: VNIR e SWIR com o mesmo numero "
        "de bandas): misturar daria um modelo sem sentido. Separe-as em pastas diferentes"
        + (f"; se a diferenca for pequena (ate' meia banda, {0.5 * esp:.1f} nm) e voce quiser reamostrar "
           f"para o eixo do primeiro cubo, use hsi_eixo_bandas: reamostrar." if pequena else "."))


def load_hsi_folder_dataset(pasta: str, *, lazy: bool = False, calibracao: str = "nenhuma",
                            referencia_branco: str = "", referencia_escuro: str = "",
                            eixo_bandas: str = "recusar") -> Tuple[
        List[np.ndarray], np.ndarray, Optional[np.ndarray],
        np.ndarray, "pd.DataFrame"]:
    """Le uma pasta com cubos hiperespectrais do PROPRIO usuario (par
    `.hdr`+`.bin` ENVI por gravacao) -- Passo 111 (INSTRUCAO_HSI_DADO_
    PROPRIO.md). NAO exige `manifest.json` nem layout de terceiros:
    qualquer pasta
    no formato ENVI serve, com a MESMA convencao de pastas ja usada em
    `dados_io.py` (.dx) e `dados_imagem.py` (fotos) -- uma subpasta por
    classe (rotulo), ou arquivos soltos na raiz (fallback, 1 classe por
    arquivo).

    Agrupamento por amostra fisica (`mae_id`, protecao contra vazamento
    entre gravacoes do MESMO objeto no split treino/teste) segue a
    MESMA hierarquia de 3 niveis do Bloco 8 -- reaproveitada de
    `agrupamento_pastas.py`, nao reimplementada aqui:
      - "high"   -- subpasta por amostra fisica dentro da subpasta de
        classe (cada `.hdr`+`.bin` dentro dela e' uma replica/vista).
      - "medium" -- CSV `amostras.csv` na raiz (colunas `arquivo,
        id_amostra`), mesmo mecanismo de `dados_imagem.py`.
      - "none"   -- nem um nem outro: processa mesmo assim, mas `mae_id`
        fica `None` e a limitacao vai em `metadados_df.attrs[
        "grouping_guarantee"]` (quem chama -- `hsi_pipeline.py` -- deve
        declarar isso no log/relatorio, nunca silenciosamente, mesmo
        padrao ja' seguido por `dados_imagem.py`).

    Cubos de referencia (`WHITEREF_*`, `DARKREF_*`, padrao Specim) NUNCA entram
    como amostras; com `calibracao="branco_escuro"` servem para
    R = (I - D)/(W - D) (ver `localizar_referencia` para onde sao procurados).

    Eixo espectral (`wavelengths`): usa o do `.hdr` quando presente; se
    NENHUM `.hdr` da pasta trouxer o campo `wavelength` mas todos os
    cubos tiverem o MESMO numero de bandas, cai para um indice simbolico
    `0..n_bandas-1` (mesma limitacao ja aceita em `dados_imagem.py` para
    fotos). Cubos com numero de bandas DIFERENTE entre si levantam erro
    explicito -- sensores/cameras diferentes misturados na mesma pasta
    nao formam uma matriz X coerente. Com o mesmo numero de bandas mas
    comprimentos de onda diferentes (> `TOLERANCIA_EIXO_NM`) tambem recusa
    (ou, com `eixo_bandas="reamostrar"` e diferenca de ate' meia banda,
    reamostra para o eixo do primeiro cubo).

    `lazy=True` devolve `hsi_blocos.CuboHSI` (leitura por memmap, em blocos)
    em vez de ndarray float32 carregados por inteiro.

    Devolve `(cubos, rotulos, mae_id, wavelengths, metadados_df)` --
    `mae_id` e' `None` quando o nivel detectado e' "none" (mesmo contrato
    de `dados_imagem.load_images`). `metadados_df.attrs` traz tambem
    `calibracao` e `eixo_bandas` (dicts de resumo para o relatorio).
    """
    if calibracao not in CALIBRACOES:
        raise ValueError(f"calibracao '{calibracao}' invalida; use {list(CALIBRACOES)}.")
    if eixo_bandas not in EIXOS_BANDAS:
        raise ValueError(f"eixo_bandas '{eixo_bandas}' invalido; use {list(EIXOS_BANDAS)}.")
    ign = _e_arquivo_de_referencia
    subpastas = _ap.detectar_subpastas_por_extensao(pasta, _EXTENSOES_HSI, ign)
    if subpastas:
        arquivos: List[Tuple[str, str]] = []
        for sp in subpastas:
            classe = os.path.basename(sp)
            diretos = _ap.listar_arquivos_por_extensao(sp, _EXTENSOES_HSI, ign)
            if diretos:
                arquivos.extend((a, classe) for a in diretos)
            else:
                # nivel "high" em potencial: sem gravacao direta, mas com
                # subpastas de amostra fisica dentro da subpasta de classe.
                for n in sorted(os.listdir(sp)):
                    caminho_amostra = os.path.join(sp, n)
                    if os.path.isdir(caminho_amostra):
                        arquivos.extend(
                            (a, classe) for a in _ap.listar_arquivos_por_extensao(
                                caminho_amostra, _EXTENSOES_HSI, ign))
    else:
        if not os.path.isdir(pasta):
            raise FileNotFoundError(
                f"Pasta nao existe: {pasta}\n"
                f"  -> confira cfg.hsi_dataset_folder.")
        arqs = _ap.listar_arquivos_por_extensao(pasta, _EXTENSOES_HSI, ign)
        if not arqs:
            raise FileNotFoundError(
                f"Pasta existe mas nao contem cubos hiperespectrais ENVI "
                f"(pares .hdr/.bin).\n  Pasta: {pasta}")
        arquivos = [(a, "") for a in arqs]

    todos_arquivos = [a for a, _ in arquivos]
    grupos_high = _ap.detectar_nivel_high(subpastas, _EXTENSOES_HSI, ign)
    if grupos_high is not None:
        nivel = _ap.GROUPING_HIGH
        mapa_grupo = grupos_high
    else:
        grupos_medium = _ap.detectar_nivel_medium(pasta, todos_arquivos)
        if grupos_medium is not None:
            nivel = _ap.GROUPING_MEDIUM
            mapa_grupo = grupos_medium
        else:
            nivel = _ap.GROUPING_NONE
            mapa_grupo = {}

    if nivel == _ap.GROUPING_NONE:
        print("[WARNING] Grouping guarantee: NONE -- nenhuma subpasta por "
              "amostra fisica nem CSV de associacao (amostras.csv) foi "
              "encontrado. Validacao cai sem protecao contra vazamento "
              "entre gravacoes do MESMO objeto. Resultados devem "
              "ser tratados como exploratorios. Ver docstring de "
              "agrupamento_pastas.py para os niveis 'high'/'medium'.")
    else:
        print(f"[INFO] Grouping guarantee: {nivel.upper()} "
              f"({len(set(mapa_grupo.values()))} grupos de amostra fisica).")

    # Fase 1: cabecalhos e tamanho dos binarios (sem ler os dados).
    lidos: List[Tuple[str, str, str, "_ParametrosEnvi"]] = []
    n_falhos = 0
    for caminho_hdr, subpasta_nome in arquivos:
        try:
            arq_dados = _resolver_arquivo_dados_envi(caminho_hdr)
            par = _ler_parametros_envi(caminho_hdr)
            _validar_tamanho_binario(arq_dados, par)
        except Exception as e:  # noqa: BLE001 -- leitura defensiva de cubo
            # externo (arquivo truncado/corrompido/header incompativel);
            # erro impresso COM NOME DO ARQUIVO e contabilizado em
            # n_falhos, nunca silencioso (mesmo padrao de dados_imagem.py).
            n_falhos += 1
            print(f"  [ERROR] {os.path.basename(caminho_hdr)}: {e}")
            continue
        lidos.append((caminho_hdr, subpasta_nome, arq_dados, par))

    if not lidos:
        raise ValueError(
            f"Nenhum cubo hiperespectral valido carregado ({n_falhos} com erro).")

    # Fase 2: eixos de bandas contra o do primeiro cubo.
    ref_hdr, _, _, ref_par = lidos[0]
    ref_nome = os.path.basename(ref_hdr)
    reamostragens: List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = [None]
    desvio_max = 0.0
    for caminho_hdr, _sp, _arq, par in lidos[1:]:
        reamo, desvio = _comparar_eixos(ref_nome, ref_par.wl, ref_par.n_bands,
                                        os.path.basename(caminho_hdr), par.wl, par.n_bands, eixo_bandas)
        reamostragens.append(reamo)
        desvio_max = max(desvio_max, desvio)
    n_reamostrados = sum(r is not None for r in reamostragens)
    if n_reamostrados:
        print(f"[AVISO] HSI: {n_reamostrados} cubo(s) reamostrado(s) para o eixo de {ref_nome} "
              f"(desvio maximo {desvio_max:.2f} nm; hsi_eixo_bandas: reamostrar).")
    wl_ref = ref_par.wl
    info_eixo = {"modo": eixo_bandas, "n_bandas": int(ref_par.n_bands),
                 "faixa_nm": ([float(wl_ref.min()), float(wl_ref.max())] if wl_ref is not None else None),
                 "tolerancia_nm": TOLERANCIA_EIXO_NM, "n_reamostrados": int(n_reamostrados),
                 "desvio_max_nm": float(desvio_max)}

    # Fase 3: calibracao branco/escuro (todos os cubos ou nenhum).
    calibracoes: List[Optional[CalibracaoBrancoEscuro]] = [None] * len(lidos)
    refs_por_cubo: Dict[int, Tuple[Dict[str, object], Dict[str, object]]] = {}
    avisos_calib: List[str] = []
    info_calib: Dict[str, object] = {"modo": calibracao, "n_cubos_calibrados": 0, "referencias": [], "avisos": []}
    if calibracao == "branco_escuro":
        cache: Dict[str, Dict[str, object]] = {}
        faltam: List[str] = []
        refs_usadas: Dict[str, Dict[str, str]] = {}
        achadas: List[Tuple[str, str, str, str, str]] = []
        for caminho_hdr, _sp, _arq, par in lidos:
            w, ow = localizar_referencia(caminho_hdr, pasta, par.campos, "white", referencia_branco)
            d, od = localizar_referencia(caminho_hdr, pasta, par.campos, "dark", referencia_escuro)
            if w is None or d is None:
                faltam.append(f"{_rel(caminho_hdr, pasta)} (branco: {ow if w is None else 'ok'}; "
                              f"escuro: {od if d is None else 'ok'})")
            else:
                achadas.append((caminho_hdr, w, d, ow, od))
        if faltam:
            raise ValueError(
                f"hsi_calibracao=branco_escuro, mas {len(faltam)} cubo(s) ficaram sem WHITEREF/DARKREF: "
                + "; ".join(faltam[:5]) + ("..." if len(faltam) > 5 else "")
                + ". Coloque WHITEREF_<nome>.hdr/DARKREF_<nome>.hdr ao lado de cada cubo (padrao Specim), "
                "um WHITEREF.hdr/DARKREF.hdr na pasta de cada sessao, ou aponte hsi_referencia_branco/"
                "hsi_referencia_escuro. A calibracao vale para todos os cubos ou para nenhum (misturar "
                "cubos calibrados e brutos invalida o modelo).")
        idx_por_hdr = {c[0]: i for i, c in enumerate(lidos)}
        for caminho_hdr, w, d, ow, od in achadas:
            i = idx_por_hdr[caminho_hdr]
            par = lidos[i][3]
            for caminho in (w, d):
                if caminho not in cache:
                    try:
                        cache[caminho] = _carregar_referencia(caminho)
                    except Exception as e:  # noqa: BLE001 -- referencia ilegivel: erro claro com o nome
                        raise ValueError(f"referencia {caminho}: {e}") from e
            nome = _rel(caminho_hdr, pasta)
            calib, av = _construir_calibracao(nome, par, cache[w], cache[d], f"branco: {ow}; escuro: {od}")
            calibracoes[i] = calib
            refs_por_cubo[i] = (cache[w], cache[d])
            avisos_calib.extend(av)
            refs_usadas[nome] = {"branco": _rel(w, pasta), "escuro": _rel(d, pasta)}
        info_calib.update({"n_cubos_calibrados": len(achadas),
                           "referencias": [{"cubo": k, **v} for k, v in sorted(refs_usadas.items())]})

    cubos: List[np.ndarray] = []
    rotulos: List[str] = []
    grupos_arr: List[Optional[str]] = []
    meta_rows: List[Dict[str, object]] = []
    for i, (caminho_hdr, subpasta_nome, arq_dados, par) in enumerate(lidos):
        reamo = reamostragens[i]
        cubo_lazy = CuboHSI(arq_dados, par.dtype_np, par.interleave, par.n_lines, par.n_samples,
                            par.n_bands, par.header_offset, fator=par.fator, calibracao=calibracoes[i],
                            reamostragem=reamo)
        if calibracoes[i] is not None:
            w_ref, d_ref = refs_por_cubo[i]
            avisos_calib.extend(_checar_amostra_vs_referencia(
                _rel(caminho_hdr, pasta), cubo_lazy, w_ref, d_ref, par))
        # CuboHSI imita a interface de ndarray (shape/ndim/dtype/indexacao em blocos): com
        # lazy=True ele segue no lugar do array pelo pipeline HSI, que so' usa essa interface.
        cubos.append(cast(np.ndarray, cubo_lazy) if lazy else np.asarray(cubo_lazy))
        classe = subpasta_nome or os.path.splitext(os.path.basename(caminho_hdr))[0]
        rotulos.append(classe)
        grupos_arr.append(mapa_grupo.get(caminho_hdr))
        meta_rows.append({
            "arquivo": os.path.basename(caminho_hdr), "subpasta": subpasta_nome,
            "classe": classe, "grupo_id": mapa_grupo.get(caminho_hdr),
            "n_linhas": par.n_lines, "n_colunas": par.n_samples,
            "n_bandas": par.n_bands,
        })
    for a in dict.fromkeys(avisos_calib):
        print(f"[AVISO] HSI calibracao: {a}")
    info_calib["avisos"] = list(dict.fromkeys(avisos_calib))

    if n_falhos > 0:
        print(f"[WARNING] {n_falhos} cubos com erro de leitura -- pulados.")

    mae_id = (np.array(grupos_arr, dtype=object)
              if nivel != _ap.GROUPING_NONE else None)
    n_bandas_ref = ref_par.n_bands
    wavelengths = (wl_ref if wl_ref is not None
                   else np.arange(n_bandas_ref, dtype=float))
    metadados_df = pd.DataFrame(meta_rows)
    metadados_df.attrs["grouping_guarantee"] = nivel
    metadados_df.attrs["calibracao"] = info_calib
    metadados_df.attrs["eixo_bandas"] = info_eixo
    n_ign = _contar_referencias_ignoradas(pasta)
    if n_ign and calibracao == "nenhuma":
        print(f"[INFO] {n_ign} arquivo(s) WHITEREF/DARKREF ignorado(s) como amostras "
              f"(use hsi_calibracao: branco_escuro para calibrar a reflectancia com eles).")
    print(f"[INFO] {len(cubos)} cubos hiperespectrais carregados, "
          f"{n_bandas_ref} bandas.")

    return cubos, np.array(rotulos, dtype=str), mae_id, wavelengths, metadados_df


def _contar_referencias_ignoradas(pasta: str) -> int:
    n = 0
    for raiz, _d, arqs in os.walk(pasta):
        n += sum(1 for a in arqs if a.lower().endswith(".hdr") and _e_arquivo_de_referencia(a))
    return n
