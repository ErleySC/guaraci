#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baixar_figshare_azeite_nmr.py -- Baixa o dataset publico Figshare
`10.6084/m9.figshare.4307804` (Lamanna et al. 2017 -- perfil 1H-NMR de
97 azeites de oliva da regiao Abruzzo/Italia, rotulados por provincia
de origem: "pe" = Pescara (50), "te" = Teramo (47)) para um diretorio
de CACHE LOCAL fora do controle de versao.

Passo 142/143 da auditoria das 11 tecnicas analiticas (2026-09-04).
LICENCA CC0 (dominio publico) -- a mais permissiva ja usada neste
projeto. So' 1 arquivo baixado: `oliando2012Frantoio_data_Rev_New.csv`
(97 amostras x 125 variaveis de deslocamento quimico ja BINADAS/
processadas pelos autores originais -- tabela pronta pro motor CSV
generico, zero NaN, verificado por leitura direta em 2026-09-04). Os
outros 4 arquivos do dataset (`.spect` bruto, script `.r`, `.list`/
`.txt` de regioes de binning) NAO sao baixados -- sem uso na validacao,
mesmo criterio ja usado para os arquivos NIR 24mm/2mm do Mendeley
ctgg7k4m5g.

LIMITACAO HONESTA -- download automatizado BLOQUEADO pelo Figshare:
o servidor do Figshare roda um desafio de bot da AWS WAF
(`x-amzn-waf-action: challenge`) tanto contra `curl`/`urllib` quanto
contra o navegador automatizado deste projeto (verificado em
2026-09-04) -- NAO e' CAPTCHA (a regra permanente deste projeto contra
burlar CAPTCHA nao se aplica tecnicamente aqui), mas e' um bloqueio de
bot deliberado que este script nao tenta contornar. Por isso `baixar_
dataset()` abaixo TENTA o download automatico (funciona se o WAF nao
desafiar a requisicao, ex. de uma rede/IP diferente) e, se falhar,
levanta erro claro com a instrucao de download MANUAL (baixe o arquivo
pela pagina https://figshare.com/articles/dataset/Dataset_for_
Territorial_origin_of_olive_oil_Representing_georeferenced_maps_of_
olive_oils_by_NMR_profiling_/4307804 -- botao "Download file" ao lado
de `oliando2012Frantoio_data_Rev_New.csv` -- e copie para a pasta
indicada). SHA256+tamanho pinados a partir do arquivo baixado
manualmente e verificado nesta auditoria (2026-09-04) -- mesmo padrao
de seguranca dos outros scripts (nunca confia em conteudo nao
verificado).

Uso:
    python scripts/download_datasets/baixar_figshare_azeite_nmr.py [DESTINO]

DESTINO (opcional): pasta onde salvar. Default: $GUARACI_DATASETS_DIR/
figshare_4307804, ou ./datasets_publicos/figshare_4307804 se a
variavel nao estiver definida.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

_NOME_ARQUIVO = "oliando2012Frantoio_data_Rev_New.csv"
_URL = "https://figshare.com/ndownloader/files/7025039"
_SHA256_ESPERADO = "e49042c539ee5f3ecabdd36dc55874dd90eac979710ac858f24c3e7ea6366d3b"
_BYTES_ESPERADO = 163710

_PAGINA_DATASET = (
    "https://figshare.com/articles/dataset/Dataset_for_Territorial_origin_"
    "of_olive_oil_Representing_georeferenced_maps_of_olive_oils_by_NMR_"
    "profiling_/4307804")


def _sha256_de(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def baixar_dataset(pasta_destino: "str | Path | None" = None) -> Path:
    if pasta_destino is None:
        raiz = os.environ.get("GUARACI_DATASETS_DIR", "datasets_publicos")
        pasta_destino = Path(raiz) / "figshare_4307804"
    pasta_destino = Path(pasta_destino)
    pasta_destino.mkdir(parents=True, exist_ok=True)
    destino = pasta_destino / _NOME_ARQUIVO

    if destino.is_file() and destino.stat().st_size == _BYTES_ESPERADO:
        if _sha256_de(destino) == _SHA256_ESPERADO:
            print(f"[OK] {_NOME_ARQUIVO} ja' em cache e integro -- pulando download.")
            return pasta_destino
        print(f"[AVISO] {_NOME_ARQUIVO} em cache mas com hash divergente -- "
              f"baixando de novo.")

    print(f"[INFO] Tentando baixar {_NOME_ARQUIVO} de {_URL} ...")
    try:
        req = urllib.request.Request(
            _URL, headers={"User-Agent": "guaraci-dataset-downloader/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            conteudo = resp.read()
    except Exception as e:
        raise RuntimeError(
            f"Download automatico de {_NOME_ARQUIVO} falhou ({e!r}) -- o "
            f"Figshare bloqueia requisicoes automatizadas deste dataset "
            f"(desafio de bot AWS WAF, ver docstring deste script). Baixe "
            f"MANUALMENTE pela pagina {_PAGINA_DATASET} (botao 'Download "
            f"file' ao lado de '{_NOME_ARQUIVO}') e copie o arquivo para "
            f"'{destino}'.") from e

    tamanho = len(conteudo)
    if tamanho != _BYTES_ESPERADO or hashlib.sha256(conteudo).hexdigest() != _SHA256_ESPERADO:
        raise RuntimeError(
            f"{_NOME_ARQUIVO}: conteudo baixado nao bate com o esperado "
            f"({tamanho} bytes) -- a fonte pode ter mudado ou o WAF "
            f"devolveu uma pagina de desafio em vez do arquivo. NAO "
            f"gravando arquivo nao verificado. Baixe manualmente (ver "
            f"docstring deste script).")

    destino.write_bytes(conteudo)
    print(f"[OK] {_NOME_ARQUIVO}: {tamanho} bytes, sha256 confirmado.")
    return pasta_destino


if __name__ == "__main__":
    destino_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pasta = baixar_dataset(destino_arg)
    print(f"[OK] dataset Figshare 4307804 pronto em: {pasta}")
