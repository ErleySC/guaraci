#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baixar_ropls_sacurine.py -- Baixa o dataset publico `sacurine` (metaboloma
urinario de 183 adultos, LC-HRMS, 109 metabolitos, intensidades log10) para
um diretorio de CACHE LOCAL fora do controle de versao.

Referencia: Thevenot E.A., Roux A., Xu Y., Ezan E. & Junot C. (2015).
Analysis of the human adult urinary metabolome variations with age, body
mass index, and gender by implementing a comprehensive workflow for
univariate and OPLS statistical analyses. J. Proteome Res. 14(8):3322-3335.
E' o dataset de exemplo do pacote ropls (Bioconductor, CeCILL), onde vem
como 3 TSVs em `inst/extdata/`. Usado por `tests/test_oraculo_dados_reais.py`
(PLS-DA, VIP, SR e OPLS-DA do GUARACI x mdatools/ropls).

Fonte: mirror do Bioconductor no GitHub (`bioc/ropls`), FIXADO no commit
a70ae7c2e27a7e029ce8a2520a133b082b90186c (ropls 1.45.0) -- a URL "raw" de
um commit nao muda; SHA-256 + tamanho pinados mesmo assim (mesma
disciplina dos demais scripts desta pasta: arquivo vai para um temporario
e so' e' promovido ao destino depois de bater tamanho E hash).

Uso:
    python scripts/download_datasets/baixar_ropls_sacurine.py [DESTINO]

DESTINO (opcional): pasta onde salvar. Default: $GUARACI_DATASETS_DIR/
ropls_sacurine, ou ./datasets_publicos/ropls_sacurine se a variavel nao
estiver definida.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _http_retry import urlopen_com_retry  # noqa: E402

_BASE = ("https://raw.githubusercontent.com/bioc/ropls/"
         "a70ae7c2e27a7e029ce8a2520a133b082b90186c/inst/extdata/")
_ARQUIVOS = {
    # nome: (sha256, bytes)
    "sacurine_dataMatrix.tsv": (
        "301acaba5fe7154377feb3ad6fcaa7128c4a989604637fcd2bfe76f80737e490",
        241164),
    "sacurine_sampleMetadata.tsv": (
        "877ad2306b097e70a3192e7e97edf822337adf348ed08ec5ecace35c1339c325",
        3492),
    "sacurine_variableMetadata.tsv": (
        "3205554a4ca4ef55eb2ce51916422ee560bba9a3fe614aae71d54887bbdebe74",
        4540),
}


def _sha256_de(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _baixar_um(nome: str, sha: str, n_bytes: int, pasta: Path) -> None:
    destino = pasta / nome
    if (destino.is_file() and destino.stat().st_size == n_bytes
            and _sha256_de(destino) == sha):
        print(f"[OK] {nome} ja' em cache e integro -- pulando download.")
        return
    req = urllib.request.Request(
        _BASE + nome, headers={"User-Agent": "guaraci-dataset-downloader/1.0"})
    fd, tmp_nome = tempfile.mkstemp(dir=pasta, prefix=".tmp_" + nome)
    tmp_path = Path(tmp_nome)
    try:
        h = hashlib.sha256()
        tamanho = 0
        with os.fdopen(fd, "wb") as tmp_f, \
                urlopen_com_retry(req, timeout=60) as resp:
            for bloco in iter(lambda: resp.read(1 << 20), b""):
                tmp_f.write(bloco)
                h.update(bloco)
                tamanho += len(bloco)
        if tamanho != n_bytes:
            raise RuntimeError(
                f"{nome}: {tamanho} bytes baixados, esperado {n_bytes} -- "
                "NAO gravando arquivo nao verificado.")
        if h.hexdigest() != sha:
            raise RuntimeError(
                f"{nome}: sha256 {h.hexdigest()} nao bate com o esperado "
                f"{sha} -- NAO gravando arquivo nao verificado.")
        shutil.move(str(tmp_path), str(destino))
        print(f"[OK] {nome}: {tamanho} bytes, sha256 confirmado.")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def baixar_dataset(pasta_destino: "str | Path | None" = None) -> Path:
    if pasta_destino is None:
        raiz = os.environ.get("GUARACI_DATASETS_DIR", "datasets_publicos")
        pasta_destino = Path(raiz) / "ropls_sacurine"
    pasta_destino = Path(pasta_destino)
    pasta_destino.mkdir(parents=True, exist_ok=True)
    for nome, (sha, n_bytes) in _ARQUIVOS.items():
        _baixar_um(nome, sha, n_bytes, pasta_destino)
    return pasta_destino


if __name__ == "__main__":
    destino_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pasta = baixar_dataset(destino_arg)
    print(f"[OK] dataset sacurine (ropls) pronto em: {pasta}")
