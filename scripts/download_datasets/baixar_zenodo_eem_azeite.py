#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baixar_zenodo_eem_azeite.py -- Baixa o dataset publico Zenodo
`10.5281/zenodo.19755088` ("EEM fluorescence spectral dataset of
olive-oil adulteration samples across five adulterant systems") para um
diretorio de CACHE LOCAL fora do controle de versao.

Passo 149 da auditoria das 11 tecnicas analiticas (Fase C, 2026-09-04):
EEM (matriz excitacao-emissao) REAL de fluorescencia, 330 amostras --
azeite de oliva extra-virgem (3 marcas) misturado com 5 tipos de
adulterante (milho, canola, amendoim, soja, noz) em 10 niveis de fracao
de azeite (9,09%-90,91%), 3 medicoes independentes ("rodadas") por
combinacao marca/adulterante/fracao. Cada arquivo `.dat` e' uma matriz
35 (excitacao) x 270 (emissao), texto separado por tabulacao -- formato
BEM mais tratavel que o dataset EEM cogitado anteriormente (Mendeley
`g6y69g8gwm`, registrado como "parser fora de escopo" em
`eem_multiway.py`; ESTE dataset e' o substituto real).

Licenca: **CC BY 4.0** (confirmado via API oficial do Zenodo,
`zenodo.org/api/records/19755088`, campo `metadata.license.id`).

Mesma disciplina de seguranca de `baixar_eawag_esgoto_uvvis.py`: HTTPS,
SHA-256+tamanho pinados, arquivo grande (~96 MB) transmitido em
streaming para um temporario e so' promovido ao destino final depois de
bater tamanho E hash. URL hardcoded (endpoint de download DIRETO do
arquivo, nao o link de pagina).

Uso:
    python scripts/download_datasets/baixar_zenodo_eem_azeite.py [DESTINO]

DESTINO (opcional): pasta onde salvar. Default: $GUARACI_DATASETS_DIR/
zenodo_19755088_eem_azeite, ou ./datasets_publicos/
zenodo_19755088_eem_azeite se a variavel nao estiver definida.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_URL = "https://zenodo.org/api/records/19755088/files/data.zip/content"
_NOME = "data.zip"
_SHA256 = "d65cad5d716d7b912c1735f896392d537b0a9850519fdcd7516d0b0e6eca3b83"
_BYTES = 95919124


def _sha256_de(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def baixar_dataset(pasta_destino: "str | Path | None" = None,
                    extrair: bool = True) -> Path:
    if pasta_destino is None:
        raiz = os.environ.get("GUARACI_DATASETS_DIR", "datasets_publicos")
        pasta_destino = Path(raiz) / "zenodo_19755088_eem_azeite"
    pasta_destino = Path(pasta_destino)
    pasta_destino.mkdir(parents=True, exist_ok=True)

    destino = pasta_destino / _NOME
    ja_integro = (destino.is_file() and destino.stat().st_size == _BYTES
                  and _sha256_de(destino) == _SHA256)
    if ja_integro:
        print(f"[OK] {_NOME} ja' em cache e integro -- pulando download.")
    else:
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
                    "a fonte pode ter mudado o conteudo. NAO gravando "
                    "arquivo nao verificado.")
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

    pasta_dados = pasta_destino / "data"
    if extrair and not pasta_dados.is_dir():
        print(f"[INFO] Extraindo {_NOME} ...")
        with zipfile.ZipFile(destino) as z:
            z.extractall(pasta_destino)
        print(f"[OK] Extraido em {pasta_dados}")

    return pasta_destino


if __name__ == "__main__":
    destino = sys.argv[1] if len(sys.argv) > 1 else None
    pasta = baixar_dataset(destino)
    print(f"[OK] dataset Zenodo 19755088 (EEM azeite) pronto em: {pasta}")
