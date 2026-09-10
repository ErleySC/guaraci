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

Baixa 4 arquivos: NIR8mm1A.csv + OilClassKey.csv (uso original, ver
`tests/test_validacao_publica_mendeley.py`) e, desde o Passo 142/143
(2026-09-04, auditoria das 11 tecnicas analiticas), MIR1A.csv e
Raman1A.csv -- MESMAS 100 amostras/mesmo alvo (indice de peroxido) do
artigo, so' medidas em tecnica diferente, o que permite validar MIR e
Raman sem buscar dataset novo nenhum (ver
`tests/test_validacao_publica_mendeley_mir_raman.py`). Os 4 arquivos
restantes do dataset (NIR 24mm/2mm 1A/1B, Raman2, ATR*) continuam nao
baixados -- sem uso em nenhuma suite ainda. SHA256/tamanho desses 2
arquivos novos pinados a partir da API publica do Mendeley
(`public-api/datasets/ctgg7k4m5g/files?folder_id=root&version=2`, que
ja devolve o hash calculado pelo proprio Mendeley -- conferido em
2026-09-04 que o hash da API bate exatamente com o de NIR8mm1A.csv
ja pinado abaixo desde 2026-08-26, ou seja a API e' fonte confiavel
para os 2 arquivos novos tambem).

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
    "MIR1A.csv": {
        "url": ("https://data.mendeley.com/public-files/datasets/"
                 "ctgg7k4m5g/files/c9eb515e-4a4e-4882-9b33-88e1d1c7fbf7/"
                 "file_downloaded"),
        "sha256": "3fa249fbf3a34c2a06a4c8b389512fd8c560694ee95f2f4b641f2c523f7d02fa",
        "bytes": 4268334,
    },
    "Raman1A.csv": {
        "url": ("https://data.mendeley.com/public-files/datasets/"
                 "ctgg7k4m5g/files/abf49aee-c5d7-4a55-aec4-b59eea61affb/"
                 "file_downloaded"),
        "sha256": "8fbd92e030605a48fd84f194bd48dbdba9fcf006b2368a6d278159930459e5b9",
        "bytes": 1029350,
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
