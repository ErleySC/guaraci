#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baixar_zenodo_hplc_azeite.py -- Baixa o dataset publico Zenodo
`10.5281/zenodo.21245912` ("Olive Oil Triacylglyceride Profiles by
HPLC-CAD") para um diretorio de CACHE LOCAL fora do controle de versao.

Passo 157/159 da auditoria das 11 tecnicas analiticas (busca ampliada
por dataset de HPLC ja' pre-processado, 2026-09-05): mirror recente
(publicado no Zenodo em 2026-07-07) do dataset CLASSICO de
de la Mata-Espinosa, Bosque-Sendra, Bro & Cuadros-Rodriguez (2011) --
mesmo Rasmus Bro ja' referenciado em `hsi_multiway.py` (PARAFAC/N-PLS)
-- pagina original `https://ucphchemometrics.com/olive/` (grupo de
quimiometria da Universidade de Copenhague). 120 amostras de oleo
comestivel medidas por HPLC com detector de aerossol carregado (CAD),
perfil de triacilgliceroois JA' corrigido de baseline e alinhado pelos
autores originais (mesma logica do RMN ja' binado -- motor CSV
generico do GUARACI sem nenhum pre-processamento de sinal novo). Alvo:
azeite vs. nao-azeite vs. mistura, codificado no campo `class` do
objeto MATLAB (`classlookup`: 1=not, 2=olive, 3=mix). Das 120 amostras,
118 estao no subconjunto "ativo" (`include`, linhas 3-120 -- as 2
primeiras, `Cac_CA`/`Cac_LE`, ficam de fora por decisao dos autores
originais, preservada aqui).

Licenca: **CC BY 4.0** (confirmado via API oficial do Zenodo,
`zenodo.org/api/records/21245912`, campo `metadata.license.id`).
Citar os 2 artigos originais ao usar (ver docstring de
`tests/test_validacao_publica_hplc_azeite.py`).

Formato: `HPLCforweb.mat` e' um objeto MATLAB "Dataset Object" (mesma
convencao do PLS_Toolbox usada pelo `corn.mat` ja' integrado neste
projeto -- `m["HPLCforweb"]["data"][0,0]` da' a matriz 120x4001,
`["class"][0,0][0,0]` da' os codigos de classe, `["include"][0,0][0,0]`
da' os indices 1-based do subconjunto ativo).

Mesma disciplina de seguranca dos demais scripts desta pasta: HTTPS,
SHA-256+tamanho pinados, arquivo pequeno (~3.2 MB) transmitido em
streaming para um temporario e so' promovido ao destino final depois de
bater tamanho E hash. URL hardcoded (endpoint de download DIRETO do
arquivo, nao o link de pagina).

Uso:
    python scripts/download_datasets/baixar_zenodo_hplc_azeite.py [DESTINO]

DESTINO (opcional): pasta onde salvar. Default: $GUARACI_DATASETS_DIR/
zenodo_21245912_hplc_azeite, ou ./datasets_publicos/
zenodo_21245912_hplc_azeite se a variavel nao estiver definida.
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

_URL = ("https://zenodo.org/api/records/21245912/files/"
        "olive_oil_hplc_cad_triacylglyceride_profiles.zip/content")
_NOME = "olive_oil_hplc_cad_triacylglyceride_profiles.zip"
_SHA256 = "40ad2f5c716ed820c297a15a9adcab2a949e2446d8450d53a73258a0696507dc"
_BYTES = 3237868


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
        pasta_destino = Path(raiz) / "zenodo_21245912_hplc_azeite"
    pasta_destino = Path(pasta_destino)
    pasta_destino.mkdir(parents=True, exist_ok=True)

    destino = pasta_destino / _NOME
    ja_integro = (destino.is_file() and destino.stat().st_size == _BYTES
                  and _sha256_de(destino) == _SHA256)
    if ja_integro:
        print(f"[OK] {_NOME} ja' em cache e integro -- pulando download.")
    else:
        print(f"[INFO] Baixando {_NOME} ({_BYTES / 1e6:.1f} MB) de {_URL} ...")
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

    mat_path = pasta_destino / "HPLCforweb.mat"
    if extrair and not mat_path.is_file():
        print(f"[INFO] Extraindo {_NOME} ...")
        with zipfile.ZipFile(destino) as z:
            z.extractall(pasta_destino)
        print(f"[OK] Extraido em {mat_path}")

    return pasta_destino


if __name__ == "__main__":
    destino_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pasta = baixar_dataset(destino_arg)
    print(f"[OK] dataset Zenodo 21245912 (HPLC-CAD azeite) pronto em: {pasta}")
