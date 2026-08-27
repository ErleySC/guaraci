#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baixar_mendeley_oleos.py — Baixa o dataset publico Mendeley
`10.17632/ctgg7k4m5g.2` (Ottaway et al. 2021, oleos comestiveis por NIR/
MIR/Raman) para um diretorio de CACHE LOCAL fora do controle de versao.

NUNCA versiona o dado em si (licenca de terceiro, CC BY 4.0 -- ver
docs/VALIDACAO_PUBLICA.md para a citacao completa). Mesmo padrao do Corn
(`.github/workflows/test.yml`, job `validacao-publica`): pinagem por
SHA256+tamanho, para que o teste que consome o arquivo nunca rode contra
um conteudo diferente do verificado nesta auditoria (2026-08-26).

So' baixa os 2 arquivos que a suite de validacao publica de fato usa
(`tests/test_validacao_publica_mendeley.py`): NIR8mm1A.csv (RMSEP
publicado mais confiavel entre as 4 tecnicas testadas no artigo
original) e OilClassKey.csv (legenda numero->nome da classe). Os outros
8 arquivos do dataset (MIR, Raman, NIR 24mm/2mm) nao sao baixados --
sem uso nesta integracao, baixa-los so' aumentaria o tempo de CI sem
beneficio.

Uso:
    python scripts/download_datasets/baixar_mendeley_oleos.py [DESTINO]

DESTINO (opcional): pasta onde salvar. Default: $GUARACI_DATASETS_DIR/
mendeley_ctgg7k4m5g, ou ./datasets_publicos/mendeley_ctgg7k4m5g se a
variavel nao estiver definida.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# id de arquivo Mendeley -> (nome local, sha256 esperado, tamanho esperado
# em bytes). Pinado em 2026-08-26 -- ver docs/VALIDACAO_PUBLICA.md para
# a verificacao original (curl + HEAD confirmando redirect 302 para S3
# assinado, sem necessidade de sessao/credencial).
_ARQUIVOS = {
    "NIR8mm1A.csv": {
        "url": ("https://data.mendeley.com/public-files/datasets/"
                 "ctgg7k4m5g/files/6a7dc111-968a-4fdd-9ce4-c2fff34666b5/"
                 "file_downloaded"),
        "sha256": "d98758f4cde5c4d98b4e6a1cd71b545e34d42126849850c1a84cdd74b2468b47",
        "bytes": 13824578,
    },
    "OilClassKey.csv": {
        "url": ("https://data.mendeley.com/public-files/datasets/"
                 "ctgg7k4m5g/files/dfcd5aa4-9043-4cab-859f-d0bacfc76c97/"
                 "file_downloaded"),
        "sha256": "317abeb42aaa563177af19d42a7ce316f6c94963bcbf15a4bfcb8fc845655769",
        "bytes": 401,
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
    # User-Agent explicito: o servidor devolve 403 para o UA default do
    # urllib ("Python-urllib/3.x") -- achado ao testar este script pela
    # primeira vez (2026-08-26). curl (UA "curl/x.y") passa sem problema.
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
        pasta_destino = Path(raiz) / "mendeley_ctgg7k4m5g"
    pasta_destino = Path(pasta_destino)
    pasta_destino.mkdir(parents=True, exist_ok=True)

    for nome, spec in _ARQUIVOS.items():
        _baixar_um(nome, spec, pasta_destino)

    return pasta_destino


if __name__ == "__main__":
    destino = sys.argv[1] if len(sys.argv) > 1 else None
    pasta = baixar_dataset(destino)
    print(f"[OK] dataset Mendeley ctgg7k4m5g pronto em: {pasta}")
