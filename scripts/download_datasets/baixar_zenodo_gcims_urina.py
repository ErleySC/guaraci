#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baixar_zenodo_gcims_urina.py -- Baixa SO' a tabela de picos ja'
processada (nao os 44 arquivos `.mea` brutos, ~3.4 GB descomprimidos)
do dataset publico Zenodo `10.5281/zenodo.19209004` ("Targeted GC-IMS
Urine Dataset for Anisole and 2-Heptanone Analysis in Colorectal Cancer
and Control Samples") para um diretorio de CACHE LOCAL fora do controle
de versao.

Passo 158/159 da busca ampliada por dataset de IMS/GC-IMS ja'
pre-processado (2026-09-05): os 2 datasets GC-IMS "classicos" ja'
achados antes desta sessao (Mendeley `fr9t5fkkvz` -- azeite, ~6,9 GB; e
`jxj2r45t2x` -- mel, ~4,8 GB) sao espectros BRUTOS `.mea` -- teriam que
passar por um pipeline completo de denoising/alinhamento/deteccao de
picos mesmo com biblioteca madura disponivel (`gc-ims-tools`),
confirmado adiado no Passo 151/158 (ver `docs/VALIDACAO_PUBLICA.md`
secao 2j). ESTE dataset e' diferente: os proprios autores JA'
processaram os 44 espectros brutos (denoising, alinhamento, deteccao de
picos, clustering) e publicaram o resultado como 2 tabelas de picos CSV
(`peak_table_untargeted.csv`, 184 variaveis "Cluster<N>"; e
`peak_table_targeted.csv`, so' os 2 compostos-alvo) -- mesma logica do
RMN ja' binado, zero pre-processamento de sinal novo necessario.

ARQUIVO INTEIRO NAO E' BAIXADO DE PROPOSITO: o zip do Zenodo (851 MB)
contem os 44 `.mea` brutos (nao usados por este projeto) MAIS as 2
tabelas de picos (~145 KB no total) MAIS `annotations.csv` (rotulo
clinico por amostra) MAIS `README.txt`. Baixar o zip inteiro so' para
ler 4 arquivos pequenos seria desperdicio de banda/tempo de CI --
`_HTTPRangeFile` abaixo abre o zip remoto via HTTP Range (confirmado
funcional neste servidor: GET com header `Range` devolve `206 Partial
Content`; um `HEAD` com `Range` e' ignorado, por isso o teste de
suporte deve usar GET, nao HEAD) e le SO' os 4 membros pedidos, sem
baixar os `.mea`. Mesma tecnica ja' usada em `baixar_deephs_kaki.py`
para o Kaki.zip de 2.2 GB.

Alvo (classificacao): `annotations.csv`/coluna `patient_condition` --
CRC (cancer colorretal, 15 amostras) vs. CTRL (controle, 15 amostras).
As 14 amostras de controle de qualidade (`class="control"`, matriz
"pool") NAO sao amostras clinicas -- ficam de fora da classificacao
(mesmo raciocinio de excluir calibradores/brancos de uma validacao
supervisionada). Cada amostra clinica tem `patient_id` UNICO (1 medicao
por paciente, sem replica) -- `group_by_mae_id=False` e' correto pela
mesma razao do Corn/RMN/HPLC.

Licenca: **CC BY 4.0** (confirmada no `README.txt` do proprio dataset
e na API do Zenodo, campo `metadata.license.id`). Autor de contato:
Luis Fernandez, Universidade de Barcelona/IBEC.

Mesma disciplina de seguranca dos demais scripts desta pasta: HTTPS,
SHA-256+tamanho pinados POR ARQUIVO (nao do zip inteiro, que nunca e'
gravado em disco).

Uso:
    python scripts/download_datasets/baixar_zenodo_gcims_urina.py [DESTINO]

DESTINO (opcional): pasta onde salvar. Default: $GUARACI_DATASETS_DIR/
zenodo_19209004_gcims_urina, ou ./datasets_publicos/
zenodo_19209004_gcims_urina se a variavel nao estiver definida.
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

_ZIP_URL = ("https://zenodo.org/api/records/19209004/files/"
            "Targeted%20GC-IMS%20Urine%20Dataset%20for%20Anisole%20and%20"
            "2-Heptanone%20Analysis%20in%20Colorectal%20Cancer%20and%20"
            "Control%20Samples.zip/content")
_PASTA_NO_ZIP = ("Targeted GC-IMS Urine Dataset for Anisole and "
                  "2-Heptanone Analysis in Colorectal Cancer and "
                  "Control Samples/")

# nome local -> (nome dentro do zip, sha256, bytes). Pinado em 2026-09-05
# a partir do arquivo baixado e verificado nesta auditoria (MD5 do zip
# completo tambem conferido contra o publicado pela API do Zenodo antes
# de extrair estes 4 membros).
_ARQUIVOS = {
    "peak_table_untargeted.csv": {
        "sha256": "9a6c78839c195151772bc7763114930c28e489bd74d3c1b4ebbf61ab7fbaddf5",
        "bytes": 141467,
    },
    "peak_table_targeted.csv": {
        "sha256": "d2d3f73dcdc6b06cfe551316f3a3813c0aa9e21c238465a6dfc80a8122eacb0d",
        "bytes": 3453,
    },
    "annotations.csv": {
        "sha256": "d7ca49f586f8037ce46da894670c52dbb031d51149ae13a4c89ab4e672d5b3e8",
        "bytes": 2008,
    },
    "README.txt": {
        "sha256": "14a592cbb8dd606316e4a54f51f8e4028f171a220259d6b345c7d8b47d654117",
        "bytes": 3636,
    },
}


class _HTTPRangeFile(io.IOBase):
    """Arquivo remoto lido sob demanda via HTTP Range -- permite abrir um
    .zip de 851 MB e ler so' os membros pedidos (o diretorio central do
    zip + os 4 arquivos pequenos), sem baixar os 44 `.mea` brutos.
    `zipfile.ZipFile` so' precisa de seek/tell/read para funcionar com
    acesso aleatorio. Mesma classe de `baixar_deephs_kaki.py`."""

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
    destino.write_bytes(dados)


def baixar_dataset(pasta_destino: "str | Path | None" = None) -> Path:
    if pasta_destino is None:
        raiz = os.environ.get("GUARACI_DATASETS_DIR", "datasets_publicos")
        pasta_destino = Path(raiz) / "zenodo_19209004_gcims_urina"
    pasta_destino = Path(pasta_destino)
    pasta_destino.mkdir(parents=True, exist_ok=True)

    faltando = {}
    for nome, spec in _ARQUIVOS.items():
        local = pasta_destino / nome
        if not (local.is_file() and local.stat().st_size == spec["bytes"]
                and hashlib.sha256(local.read_bytes()).hexdigest() == spec["sha256"]):
            faltando[nome] = spec

    if not faltando:
        print("[OK] arquivos ja' em cache e integros -- pulando download.")
        return pasta_destino

    print(f"[INFO] Abrindo {_ZIP_URL} via HTTP Range (sem baixar os "
          f"44 arquivos .mea brutos, ~3.4 GB descomprimidos) ...")
    zf = zipfile.ZipFile(_HTTPRangeFile(_ZIP_URL))
    for nome, spec in faltando.items():
        print(f"[INFO] Extraindo {nome} ...")
        dados = zf.read(_PASTA_NO_ZIP + nome)
        _verificar_e_gravar(pasta_destino / nome, dados, spec["sha256"],
                             spec["bytes"], nome)
        print(f"[OK] {nome}: {len(dados)} bytes, sha256 confirmado.")

    return pasta_destino


if __name__ == "__main__":
    destino_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pasta = baixar_dataset(destino_arg)
    print(f"[OK] dataset Zenodo 19209004 (GC-IMS urina, so' tabelas de "
          f"picos) pronto em: {pasta}")
