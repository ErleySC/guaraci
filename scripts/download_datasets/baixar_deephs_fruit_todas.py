#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baixar_deephs_fruit_todas.py — Baixa o DeepHS Fruit COMPLETO (as 5
frutas x ate' 2 cameras cada REALMENTE disponiveis -- Kaki nao tem NIR,
Avocado/Kiwi nao tem VIS_COR, Mango/Papaya nao tem NIR; medido por
leitura direta do JSON de anotacoes, nenhuma fruta tem as 3 cameras,
ver `docs/PROGRESSO.md` Passo 104) para um diretorio de CACHE LOCAL
fora do controle de versao -- mesmo padrao de
`baixar_deephs_kaki.py`/`baixar_mendeley_oleos.py` (SHA256+tamanho
pinados, nunca versiona o dado de terceiro, verificado ANTES de gravar).

Diferenca de convencao em relacao aos outros scripts desta pasta: os
outros pinam SHA256+tamanho INLINE no proprio .py (2 a 112 arquivos).
Aqui sao 636 gravacoes x 2 arquivos = 1272 pins -- inline viraria um
arquivo de dezenas de milhares de linhas, ilegivel. Os pins ficam em
`_deephs_fruit_todas_pins.json` (sidecar, VERSIONADO junto com este
script -- e' so' um mapa nome->hash, nao o dado em si), carregado em
tempo de execucao. A garantia de seguranca e' a MESMA: cada arquivo e'
verificado (tamanho + SHA256) ANTES de gravar em disco, nunca confia
silenciosamente no que a fonte devolve.

Uso:
    python scripts/download_datasets/baixar_deephs_fruit_todas.py [DESTINO]

DESTINO (opcional): pasta onde salvar. Default: $GUARACI_DATASETS_DIR/
deephs_fruit_all, ou ./datasets_publicos/deephs_fruit_all se a variavel
nao estiver definida.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_BASE_URL = "https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/"
_ANNOTATIONS_URL = _BASE_URL + "annotations-upd-2024-01-09.zip"
_ANNOTATIONS_SHA256 = "14275450e362684bc379a5aaf6c845cf82b0f9d5912036b9401d59a4b964a3f3"
_ANNOTATIONS_BYTES = 165730

_FRUTAS = ("Avocado", "Kiwi", "Mango", "Papaya")   # Kaki tem script proprio
_N_WORKERS_POR_FRUTA = 8

_PASTA_SCRIPT = Path(__file__).parent
_CAMINHO_PINS = _PASTA_SCRIPT / "_deephs_fruit_todas_pins.json"


class _HTTPRangeFile(io.IOBase):
    """Mesma implementacao de `baixar_deephs_kaki.py` -- ver docstring
    la' para o porque (le' so' o directorio central + membros pedidos de
    um .zip remoto de dezenas de GB, sem baixar o arquivo inteiro)."""

    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        fim = self.size - 1 if (n is None or n < 0) else min(self.pos + n, self.size) - 1
        if fim < self.pos:
            return b""
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{fim}"})
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            dados = r.read()
        self.pos += len(dados)
        return dados

    def readable(self):
        return True

    def seekable(self):
        return True


def _verificar_e_gravar(destino: Path, dados: bytes, sha_esperado: str,
                         bytes_esperado: int, nome: str) -> None:
    if len(dados) != bytes_esperado:
        raise RuntimeError(
            f"{nome}: {len(dados)} bytes baixados, esperado "
            f"{bytes_esperado} -- a fonte pode ter mudado o conteudo. "
            "NAO gravando arquivo nao verificado.")
    sha_real = hashlib.sha256(dados).hexdigest()
    if sha_real != sha_esperado:
        raise RuntimeError(
            f"{nome}: sha256 {sha_real} nao bate com o esperado "
            f"{sha_esperado} -- a fonte pode ter mudado o conteudo. "
            "NAO gravando arquivo nao verificado.")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(dados)


def _baixar_annotations(pasta_tmp: Path) -> dict:
    destino = pasta_tmp / "annotations.zip"
    if not (destino.is_file() and destino.stat().st_size == _ANNOTATIONS_BYTES
            and hashlib.sha256(destino.read_bytes()).hexdigest() == _ANNOTATIONS_SHA256):
        print(f"[INFO] Baixando anotacoes de {_ANNOTATIONS_URL} ...")
        req = urllib.request.Request(
            _ANNOTATIONS_URL,
            headers={"User-Agent": "guaraci-dataset-downloader/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            conteudo = resp.read()
        _verificar_e_gravar(destino, conteudo, _ANNOTATIONS_SHA256,
                            _ANNOTATIONS_BYTES, "annotations-upd-2024-01-09.zip")
    with zipfile.ZipFile(destino) as azf:
        return json.loads(azf.read("annotations/train_only_labeled_v2.json"))


def _baixar_fruta(fruta: str, records: list, pasta_destino: Path,
                   pins: dict) -> None:
    zip_url = _BASE_URL + f"{fruta}.zip"
    n_workers = min(_N_WORKERS_POR_FRUTA, max(1, len(records) // 5))
    chunks = [records[i::n_workers] for i in range(n_workers)]

    def worker(chunk):
        zf: "zipfile.ZipFile | None" = None
        for r in chunk:
            for nome in (r["files"]["header_file"], r["files"]["data_file"]):
                spec = pins.get(nome)
                if spec is None:
                    raise RuntimeError(
                        f"{nome}: sem pin em {_CAMINHO_PINS.name} -- a fonte "
                        f"mudou desde a auditoria que gerou os pins? Nao "
                        f"baixando arquivo nao verificavel.")
                local = pasta_destino / nome.replace("/", os.sep)
                if (local.is_file() and local.stat().st_size == spec["bytes"]
                        and hashlib.sha256(local.read_bytes()).hexdigest() == spec["sha256"]):
                    continue
                if zf is None:
                    zf = zipfile.ZipFile(_HTTPRangeFile(zip_url))
                dados = zf.read(nome)
                _verificar_e_gravar(local, dados, spec["sha256"], spec["bytes"], nome)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(worker, c) for c in chunks]
        for fut in as_completed(futs):
            fut.result()


def baixar_dataset(pasta_destino: "str | Path | None" = None) -> Path:
    if pasta_destino is None:
        raiz = os.environ.get("GUARACI_DATASETS_DIR", "datasets_publicos")
        pasta_destino = Path(raiz) / "deephs_fruit_all"
    pasta_destino = Path(pasta_destino)
    pasta_destino.mkdir(parents=True, exist_ok=True)

    if not _CAMINHO_PINS.is_file():
        raise RuntimeError(
            f"{_CAMINHO_PINS} nao encontrado -- este script depende do "
            f"sidecar de pins versionado junto com ele (ver docstring do "
            f"modulo).")
    with open(_CAMINHO_PINS, "r", encoding="utf-8") as f:
        pins = json.load(f)

    ann = _baixar_annotations(pasta_destino)
    anns_by_rec = {a["record_id"]: a for a in ann["annotations"]}

    manifest = {"cameras": ann["cameras"], "records": []}
    for fruta in _FRUTAS:
        recs = [r for r in ann["records"] if r["fruit"] == fruta]
        print(f"=== {fruta}: {len(recs)} gravacoes ===")
        _baixar_fruta(fruta, recs, pasta_destino, pins)
        for r in recs:
            a = anns_by_rec[r["id"]]
            manifest["records"].append({
                "id": r["id"], "fruit": fruta, "camera_type": r["camera_type"],
                "day": r["day"], "side": r["side"],
                "header_file": r["files"]["header_file"],
                "data_file": r["files"]["data_file"],
                "ripeness_state": a["ripeness_state"],
                "storage_days": a["storage_days"], "firmness": a["firmness"],
            })

    with open(pasta_destino / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"[OK] {len(manifest['records'])} gravacoes prontas em {pasta_destino}.")
    return pasta_destino


if __name__ == "__main__":
    destino_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pasta = baixar_dataset(destino_arg)
    print(f"[OK] DeepHS Fruit (Avocado/Kiwi/Mango/Papaya) pronto em: {pasta}")
