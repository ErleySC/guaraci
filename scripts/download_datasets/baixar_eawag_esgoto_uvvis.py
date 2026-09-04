#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baixar_eawag_esgoto_uvvis.py -- Baixa o dataset publico ERIC/Eawag
"Dataset on wastewater quality monitoring with adsorption and reflectance
spectroscopy in the UV/Vis range" (Lechevallier et al. 2025, Scientific
Data 12:1296, doi:10.1038/s41597-025-05459-x) para um diretorio de CACHE
LOCAL fora do controle de versao.

Passo 147 da auditoria das 11 tecnicas analiticas (Fase A, 2026-09-04) --
mesmo padrao de seguranca de `baixar_mendeley_oleos.py`: HTTPS sempre,
SHA256+tamanho pinados (calculados localmente a partir do arquivo
publicado, ja' que o portal CKAN da Eawag nao expoe hash na API), o
conteudo baixado e' gravado primeiro num arquivo temporario e so'
promovido ao destino final DEPOIS de bater tamanho E sha256 (nunca
carregado inteiro em memoria, ao contrario dos scripts de arquivo
pequeno -- este arquivo tem ~357 MB), URL hardcoded (nunca de entrada do
usuario), cache fora do controle de versao (`$GUARACI_DATASETS_DIR`).

So' baixa `2_data.zip` (~357 MB): o pacote completo do dataset tem mais
12 arquivos de cubos hiperespectrais (~180 GB no total) que este projeto
NAO usa -- a validacao de UV-Vis usa so' os espectros de absorbancia
tabulares (sensores Spectrolyser/ISA) e as medicoes de laboratorio, todos
dentro de `2_data.zip` (formato CSV, ver
`docs/VALIDACAO_PUBLICA.md`).

Licenca: CC BY (confirmado via API do portal CKAN,
`opendata.eawag.ch/api/3/action/package_show`, campo `license_id`).

Uso:
    python scripts/download_datasets/baixar_eawag_esgoto_uvvis.py [DESTINO]

DESTINO (opcional): pasta onde salvar. Default: $GUARACI_DATASETS_DIR/
eawag_esgoto_uvvis, ou ./datasets_publicos/eawag_esgoto_uvvis se a
variavel nao estiver definida.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

_URL = ("https://opendata.eawag.ch/dataset/"
        "5ea07dd0-c003-4173-957c-0e6753530911/resource/"
        "a7b02596-d78e-4ae3-b12e-283166d846e6/download/2_data.zip")
_NOME = "2_data.zip"
_SHA256 = "1e810853eede8f60737717ee8abe74412efec8e9e6160df7f350654ac82709d9"
_BYTES = 373981270


def _sha256_de(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def baixar_dataset(pasta_destino: "str | Path | None" = None) -> Path:
    if pasta_destino is None:
        raiz = os.environ.get("GUARACI_DATASETS_DIR", "datasets_publicos")
        pasta_destino = Path(raiz) / "eawag_esgoto_uvvis"
    pasta_destino = Path(pasta_destino)
    pasta_destino.mkdir(parents=True, exist_ok=True)

    destino = pasta_destino / _NOME
    if destino.is_file() and destino.stat().st_size == _BYTES:
        if _sha256_de(destino) == _SHA256:
            print(f"[OK] {_NOME} ja' em cache e integro -- pulando download.")
            return pasta_destino
        print(f"[AVISO] {_NOME} em cache mas com hash divergente -- "
              f"baixando de novo.")

    print(f"[INFO] Baixando {_NOME} ({_BYTES / 1e6:.0f} MB) de {_URL} ...")
    req = urllib.request.Request(
        _URL, headers={"User-Agent": "guaraci-dataset-downloader/1.0"})

    fd, tmp_nome = tempfile.mkstemp(dir=pasta_destino, prefix=".tmp_" + _NOME)
    tmp_path = Path(tmp_nome)
    try:
        h = hashlib.sha256()
        tamanho = 0
        with os.fdopen(fd, "wb") as tmp_f, \
                urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            for bloco in iter(lambda: resp.read(1 << 20), b""):
                tmp_f.write(bloco)
                h.update(bloco)
                tamanho += len(bloco)

        if tamanho != _BYTES:
            raise RuntimeError(
                f"{_NOME}: {tamanho} bytes baixados, esperado {_BYTES} -- "
                "a fonte pode ter mudado o conteudo. NAO gravando arquivo "
                "nao verificado.")

        sha_real = h.hexdigest()
        if sha_real != _SHA256:
            raise RuntimeError(
                f"{_NOME}: sha256 {sha_real} nao bate com o esperado "
                f"{_SHA256} -- a fonte pode ter mudado o conteudo. NAO "
                "gravando arquivo nao verificado.")

        shutil.move(str(tmp_path), str(destino))
        print(f"[OK] {_NOME}: {tamanho} bytes, sha256 confirmado.")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return pasta_destino


if __name__ == "__main__":
    destino = sys.argv[1] if len(sys.argv) > 1 else None
    pasta = baixar_dataset(destino)
    print(f"[OK] dataset Eawag esgoto UV-Vis pronto em: {pasta}")
