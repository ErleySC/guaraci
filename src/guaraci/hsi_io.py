"""hsi_io.py — Estrutura interna de cubo hiperespectral (HSI) e leitor do
formato ENVI (par `.hdr` texto + `.bin` binario bruto), Passo 94 da
`INSTRUCAO_HSI_MINIMO_VIAVEL.md`.

Formato confirmado por leitura DIRETA (nao suposta) do dataset publico
escolhido no Passo 93 -- DeepHS Fruit (Varga, Makowski & Zell, IJCNN 2021,
arXiv:2104.09808, github.com/cogsys-tuebingen/deephs_fruit): cada gravacao
e' um par `<nome>.hdr` (ASCII, chave=valor) + `<nome>.bin` (float32,
interleave BIP, sem cabecalho embutido -- "header offset = 0"). Exemplo real
inspecionado (Kaki/VIS/day_9_m3/kaki_day_9_m3_04_front):

    ENVI
    samples = 64
    lines = 64
    bands = 224
    header offset = 0
    file type = ENVI Standard
    data type = 4          <- float32 (codigo ENVI padrao)
    interleave = bip
    byte order = 0          <- little-endian

Os `.hdr` deste dataset especifico NAO trazem a lista de comprimentos de
onda (campo `wavelength = {...}` do padrao ENVI) -- o mapeamento
banda->comprimento de onda vem à parte, no JSON de anotacoes
(`annotations/*.json`, chave `cameras[i].wavelengths`), uma lista por
camera (Specim FX10 / INNO-SPEC Redeye / Corning microHSI). Por isso
`load_envi_cube` aceita `wavelengths` como parametro OPCIONAL -- quando o
`.hdr` nao traz a lista, quem chama fornece a de fora (ver
`load_deephs_kaki_dataset`, que já faz essa ponte para este dataset).

ENVI e' o formato dominante na literatura de HSI (confirmado no Passo 93);
nao ha suporte a outros formatos (MAT/HDF5) neste ciclo -- fora de escopo
do "minimo viavel", adicionar quando um dataset real exigir.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "HSICubeMetadata",
    "parse_envi_header",
    "load_envi_cube",
    "load_deephs_fruit_dataset",
    "load_deephs_kaki_dataset",
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
    -- o dataset DeepHS nao traz).
    """
    if caminho_hdr is None:
        caminho_hdr = (caminho_bin[:-4] + ".hdr" if caminho_bin.lower().endswith(".bin")
                       else caminho_bin + ".hdr")

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
    cubo = np.ascontiguousarray(cubo, dtype=np.float64)

    wl = _wavelengths_do_header(campos)
    if wl is None:
        wl = np.asarray(wavelengths, dtype=float) if wavelengths is not None else None
    if wl is not None and wl.size != n_bands:
        raise ValueError(
            f"{caminho_hdr}: {wl.size} comprimentos de onda fornecidos, "
            f"cubo tem {n_bands} bandas -- nao correspondem.")

    meta = HSICubeMetadata(
        n_lines=n_lines, n_samples=n_samples, n_bands=n_bands,
        interleave=interleave, dtype_envi=dtype_envi, byte_order=byte_order,
        header_offset=header_offset, wavelengths=wl,
        extras={k: v for k, v in campos.items()
                if k not in ("samples", "lines", "bands", "data type",
                             "interleave", "header offset", "byte order",
                             "wavelength")})
    return cubo, meta


#: Extrai o numero do objeto fisico do nome do arquivo -- o segmento
#: numerico IMEDIATAMENTE ANTES de "_front"/"_back". Generalizado no
#: Passo 104 (INSTRUCAO_HSI_ROBUSTEZ_E_VALIDACAO.md): o padrao anterior
#: (`_m\d+_(\d+)_`, especifico do Kaki, que tem sufixo de serie de
#: medicao "_m3_" no nome do dia) NAO bate com o padrao de nome de
#: arquivo de Avocado/Kiwi/Mango/Papaya (ex.
#: `avocado_day_01_20_front.hdr`, sem "_m\d+_"). Este padrao mais
#: generico funciona nos 5 -- confirmado por leitura direta dos nomes
#: reais de cada fruta antes de generalizar (nao presumido).
_PADRAO_NUMERO_OBJETO = re.compile(r"_(\d+)_(?:front|back)\.")


def load_deephs_fruit_dataset(pasta: str, fruta: Optional[str] = None,
                              camera: Optional[str] = None):
    """Le o dataset publico DeepHS Fruit (uma ou mais frutas/cameras)
    baixado por `scripts/download_datasets/baixar_deephs_kaki.py` (so'
    Kaki/VIS) ou `baixar_deephs_fruit_todas.py` (as 5 frutas x cameras
    disponiveis, Passo 104) -- espera `manifest.json` na raiz de `pasta`,
    no formato `{"cameras": [...], "records": [...]}` (cada record com
    campos `fruit`/`camera_type`).

    `fruta`/`camera`: filtros opcionais (ex. `fruta="Kaki", camera="VIS"`).
    Como cada camera tem um eixo de comprimento de onda PROPRIO
    (numero de bandas diferente entre VIS/NIR/VIS_COR -- confirmado por
    leitura direta dos headers reais), o resultado filtrado DEVE conter
    UMA SO' camera -- levanta erro explicito se mais de uma sobrar
    (nunca mistura wavelengths incompativeis numa mesma matriz X).

    Devolve `(cubos, rotulos, group_id, wavelengths, metadados_df)` --
    mesmo contrato de antes:
      - cubos:      lista de arrays `(altura, largura, n_bandas)`, 1 por
                    gravacao (front/back sao gravacoes SEPARADAS do MESMO
                    objeto fisico -- ver `group_id`).
      - rotulos:    `ripeness_state` de cada gravacao (unripe/perfect/
                    overripe), string -- alvo do Passo 98.
      - group_id:   `f"{fruta}_{day}_<numero do objeto>"` -- MESMO id
                    para front E back do MESMO objeto fisico. Assuncao
                    verificada por leitura direta do JSON de anotacoes
                    para as 5 frutas (nao so' Kaki): front/back com o
                    mesmo (fruta, dia, numero) SEMPRE compartilham
                    `storage_days`/`ripeness_state` -- zero excecoes em
                    636 gravacoes (Passo 104). Equivalente do
                    `mae_id`/`session_from_mae_id` deste projeto.
      - wavelengths: array 1D (nm) da camera efetivamente usada.
      - metadados_df: 1 linha por gravacao com id/fruit/camera_type/day/
                    side/storage_days/firmness, para inspecao/relatorio.
    """
    import json
    import os

    import pandas as pd

    caminho_manifest = os.path.join(pasta, "manifest.json")
    if not os.path.isfile(caminho_manifest):
        raise FileNotFoundError(
            f"{caminho_manifest} nao encontrado -- rode "
            f"scripts/download_datasets/baixar_deephs_kaki.py (so' Kaki) "
            f"ou baixar_deephs_fruit_todas.py (demais frutas) primeiro.")
    with open(caminho_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    registros = manifest["records"]
    if fruta is not None:
        registros = [r for r in registros if r.get("fruit") == fruta]
    if camera is not None:
        registros = [r for r in registros if r.get("camera_type") == camera]
    if not registros:
        raise ValueError(
            f"Nenhuma gravacao para fruta={fruta!r} camera={camera!r} em "
            f"{caminho_manifest}.")

    cameras_presentes = {r.get("camera_type") for r in registros}
    if len(cameras_presentes) > 1:
        raise ValueError(
            f"{len(cameras_presentes)} cameras diferentes na selecao "
            f"({sorted(cameras_presentes)}) -- filtre por `camera` "
            f"(cada camera tem um eixo de comprimento de onda proprio, "
            f"nao da pra' misturar numa mesma matriz X).")
    camera_id = next(iter(cameras_presentes))
    camera_spec = next(c for c in manifest["cameras"] if c["id"] == camera_id)
    wavelengths = np.array(camera_spec["wavelengths"], dtype=float)

    cubos = []
    rotulos = []
    grupos = []
    meta_linhas = []
    for rec in registros:
        caminho_bin = os.path.join(pasta, rec["data_file"].replace("/", os.sep))
        caminho_hdr = os.path.join(pasta, rec["header_file"].replace("/", os.sep))
        cubo, _meta = load_envi_cube(caminho_bin, caminho_hdr,
                                     wavelengths=wavelengths)
        cubos.append(cubo)
        rotulos.append(rec["ripeness_state"])
        m = _PADRAO_NUMERO_OBJETO.search(rec["data_file"])
        numero_objeto = m.group(1) if m else rec["data_file"]
        fruta_rec = rec.get("fruit", "")
        grupos.append(f"{fruta_rec}_{rec['day']}_{numero_objeto}")
        meta_linhas.append({
            "id": rec["id"], "fruit": fruta_rec,
            "camera_type": rec.get("camera_type", ""),
            "day": rec["day"], "side": rec["side"],
            "storage_days": rec["storage_days"], "firmness": rec["firmness"],
            "group_id": grupos[-1],
        })

    return (cubos, np.array(rotulos, dtype=str), np.array(grupos, dtype=str),
            wavelengths, pd.DataFrame(meta_linhas))


def load_deephs_kaki_dataset(pasta: str):
    """Atalho equivalente a `load_deephs_fruit_dataset(pasta, fruta="Kaki",
    camera="VIS")` -- mantido pelo nome historico (Passo 94) para nao
    quebrar chamadores existentes. Requer que `manifest.json` esteja no
    formato novo (campos `fruit`/`camera_type` por record, `cameras`
    como lista) -- ver `baixar_deephs_kaki.py`."""
    return load_deephs_fruit_dataset(pasta, fruta="Kaki", camera="VIS")
