#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baixar_mendeley_fluorescencia_oleo.py -- Baixa o dataset publico Mendeley
`10.17632/thkcz3h6n6.6` (espectros de fluorescencia de oleo de oliva, com
grau de qualidade EXTRA/VIRGEN/LAMPANTE) para um diretorio de CACHE LOCAL
fora do controle de versao.

Passo 142/143 da auditoria das 11 tecnicas analiticas (2026-09-04) --
mesmo padrao de seguranca de `baixar_mendeley_oleos.py`: HTTPS sempre,
SHA256+tamanho pinados (a partir da API publica do Mendeley, que ja
devolve o hash calculado por eles), verificados ANTES de gravar em
disco, URL hardcoded (nunca de entrada do usuario), cache fora do
controle de versao (`$GUARACI_DATASETS_DIR`).

Este e' um dataset DIFERENTE do de `baixar_mendeley_oleos.py`
(`ctgg7k4m5g`, NIR/MIR/Raman de 100 oleos por indice de peroxido) --
outro DOI, outra tecnica (fluorescencia), outro alvo (grau de
qualidade), scripts separados por dataset (mesmo padrao ja usado).

Baixa os 2 arquivos do dataset:
  - `Fluorescence_olive_oil_dataset.csv` (6,9 MB): 960 medicoes = 24
    amostras fisicas x 2 LEDs de excitacao x 20 repeticoes tecnicas,
    coluna `Data` com o espectro de emissao (1024 pontos) codificado
    como string de lista Python -- precisa `ast.literal_eval`.
  - `Fluorescence_olive_oil_dataset_background.csv` (11 KB): espectro
    de fundo/branco do instrumento (NAO e' eixo de comprimento de onda
    -- valores nao-monotonicos na faixa 1403-1541, confirmado por
    leitura direta em 2026-09-04). O dataset nao publica um eixo de
    emissao calibrado em nm; os testes usam indice de canal (0-1023)
    como eixo, limitacao registrada em
    `tests/test_validacao_publica_mendeley_fluorescencia.py`.

Uso:
    python scripts/download_datasets/baixar_mendeley_fluorescencia_oleo.py [DESTINO]

DESTINO (opcional): pasta onde salvar. Default: $GUARACI_DATASETS_DIR/
mendeley_thkcz3h6n6, ou ./datasets_publicos/mendeley_thkcz3h6n6 se a
variavel nao estiver definida.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# id de arquivo Mendeley -> (nome local, sha256 esperado, tamanho esperado
# em bytes). Pinado em 2026-09-04 a partir da API publica do Mendeley
# (`public-api/datasets/thkcz3h6n6/files?folder_id=root&version=6`).
_ARQUIVOS = {
    "Fluorescence_olive_oil_dataset.csv": {
        "url": ("https://data.mendeley.com/public-files/datasets/"
                 "thkcz3h6n6/files/2038eb5e-67d9-4698-ae1a-43b75bd83d97/"
                 "file_downloaded"),
        "sha256": "2a152b0308f45ad82760a94870ca2a0441d9b4e01a03e3f24082471ae3041c76",
        "bytes": 6935626,
    },
    "Fluorescence_olive_oil_dataset_background.csv": {
        "url": ("https://data.mendeley.com/public-files/datasets/"
                 "thkcz3h6n6/files/1d4e6468-e6f7-48d2-bb7b-733691884383/"
                 "file_downloaded"),
        "sha256": "210608025b7c0835c011da1a769efb781ddd4dac64f15c503c282a0f4f31f00d",
        "bytes": 11181,
    },
}


def _sha256_de(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _baixar_um(nome: str, spec: dict, pasta_destino: Path) -> None:
    destino = pasta_destino / nome
    if destino.is_file() and destino.stat().st_size == spec["bytes"]:
        if _sha256_de(destino) == spec["sha256"]:
            print(f"[OK] {nome} ja' em cache e integro -- pulando download.")
            return
        print(f"[AVISO] {nome} em cache mas com hash divergente -- "
              f"baixando de novo.")

    print(f"[INFO] Baixando {nome} de {spec['url']} ...")
    req = urllib.request.Request(
        spec["url"], headers={"User-Agent": "guaraci-dataset-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        conteudo = resp.read()

    tamanho = len(conteudo)
    if tamanho != spec["bytes"]:
        raise RuntimeError(
            f"{nome}: {tamanho} bytes baixados, esperado {spec['bytes']} -- "
            "a fonte pode ter mudado o conteudo. NAO gravando arquivo "
            "nao verificado.")

    sha_real = hashlib.sha256(conteudo).hexdigest()
    if sha_real != spec["sha256"]:
        raise RuntimeError(
            f"{nome}: sha256 {sha_real} nao bate com o esperado "
            f"{spec['sha256']} -- a fonte pode ter mudado o conteudo. "
            "NAO gravando arquivo nao verificado.")

    destino.write_bytes(conteudo)
    print(f"[OK] {nome}: {tamanho} bytes, sha256 confirmado.")


def baixar_dataset(pasta_destino: "str | Path | None" = None) -> Path:
    if pasta_destino is None:
        raiz = os.environ.get("GUARACI_DATASETS_DIR", "datasets_publicos")
        pasta_destino = Path(raiz) / "mendeley_thkcz3h6n6"
    pasta_destino = Path(pasta_destino)
    pasta_destino.mkdir(parents=True, exist_ok=True)

    for nome, spec in _ARQUIVOS.items():
        _baixar_um(nome, spec, pasta_destino)

    return pasta_destino


if __name__ == "__main__":
    destino = sys.argv[1] if len(sys.argv) > 1 else None
    pasta = baixar_dataset(destino)
    print(f"[OK] dataset Mendeley thkcz3h6n6 pronto em: {pasta}")
