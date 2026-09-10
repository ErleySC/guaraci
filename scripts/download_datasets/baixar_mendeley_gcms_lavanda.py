#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baixar_mendeley_gcms_lavanda.py -- Baixa os 55 arquivos GC-MS brutos
(`L01.CDF`...`L55.CDF`, formato ANDI-MS/netCDF classico) do dataset
publico Mendeley `10.17632/pgkrc7wyj4.1` ("Lavandula angustifolia
essential oil adulteration dataset", Pokajewicz 2024) para um diretorio
de CACHE LOCAL fora do controle de versao.

Passo 150 da auditoria das 11 tecnicas analiticas (Fase D, 2026-09-04):
GC-MS -- 55 amostras COMERCIAIS de oleo essencial de lavanda inglesa,
usadas no estudo original para deteccao de adulteracao. Licenca **CC BY
4.0** (confirmado via API oficial do Mendeley, campo `data_licence`).

O dataset tem 1020 arquivos no total (GC-MS + GC-FID em 3 colunas +
coluna quiral); este script baixa SO' os 55 `L##.CDF` (GC-MS de
varredura completa, ~2,6-14,5 MB cada, ~347 MB total) -- os outros ~965
arquivos (GC-FID, replicas A/B/C, menores) nao sao usados nesta
integracao (so' o TIC de GC-MS entra na validacao de alinhamento por
COW, ver `docs/VALIDACAO_PUBLICA.md` secao 2i).

Mesma disciplina de seguranca dos demais scripts: HTTPS, SHA-256+tamanho
pinados (usados os hashes ja calculados pela PROPRIA API do Mendeley,
`content_details.sha256_hash` -- mesma pratica ja usada em
`baixar_mendeley_fluorescencia_oleo.py`), verificados ANTES de gravar em
disco, URL hardcoded.

Uso:
    python scripts/download_datasets/baixar_mendeley_gcms_lavanda.py [DESTINO]

DESTINO (opcional): pasta onde salvar. Default: $GUARACI_DATASETS_DIR/
mendeley_pgkrc7wyj4, ou ./datasets_publicos/mendeley_pgkrc7wyj4 se a
variavel nao estiver definida.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# {nome: {url, sha256, bytes}} -- pinado em 2026-09-04 a partir da API
# publica do Mendeley (`public-api/datasets/pgkrc7wyj4`), campo
# `content_details` de cada um dos 55 arquivos `L##.CDF`.
_ARQUIVOS = {
    "L01.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/55f768cc-f52d-4ab2-94fc-4de3b85b2e28/file_downloaded", "sha256": "d5b638dfe2f19737e94cef770f9d0d5919ac52be1fd92dcd70d134474c81447e", "bytes": 14361668},
    "L02.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/d25b56c6-3589-43d7-99f5-288b687147fc/file_downloaded", "sha256": "3007568aab5658a40978066adbf0b8b2e522fc03f5b8c5cd1248663b4c1ee2b3", "bytes": 14524448},
    "L03.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/2af35fd2-d3e1-4e27-9712-6593b42c39b6/file_downloaded", "sha256": "a2960e7c8b77e25801b8c9fb54db29e9cbd508796600b40e0a42a84e863e94d1", "bytes": 13026080},
    "L04.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/1c20a859-b240-4362-a6dc-2c10e528bb31/file_downloaded", "sha256": "b57f90966cebe6a6a3f64d3d376413e7d4dd21bc39828660c9061bd362360c06", "bytes": 13809632},
    "L05.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/13b96b68-7943-477d-a0cd-bbaebd1f8d70/file_downloaded", "sha256": "299744d378bf755438a8064a5f09bde509bd9770d8bc4f508b94f9220477362b", "bytes": 13548320},
    "L06.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/35ead519-9f40-443b-860b-a65fe85f5c1e/file_downloaded", "sha256": "98e22c9b9ce9d4d2b925cf2b41c54d7258d63d29ff1ae286f21b3b8cef917a42", "bytes": 12987812},
    "L07.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/c1ac3459-0dac-4702-a856-cfae6909f9fd/file_downloaded", "sha256": "3ad251557888175b2de2eb9957356ec2b553a99ea37b2dfe2c8b52d24dabdda6", "bytes": 14015168},
    "L08.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/ea2deced-464d-4463-a1ea-aa2aab74a935/file_downloaded", "sha256": "23d95aa66255dc61a52e01b4d575ddc2f87728a8f573846d308a92c74a0e201a", "bytes": 13069628},
    "L09.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/0adefdf8-1737-4ca6-95db-34d171dbd93f/file_downloaded", "sha256": "cc1f105abbe5e01d2dcdc5d7521ae1e9c74b88b0e5bc5dea754b4cac5371aee7", "bytes": 8736204},
    "L10.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/083cb421-bafc-497b-bd68-1ebd1dedd9cc/file_downloaded", "sha256": "deb1232484088d5b95e94e38fb1e4f7450ba06126a75254c58e093f76ed2f8fb", "bytes": 6863784},
    "L11.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/1d54d8b1-0ebe-42bc-844e-252f11f35790/file_downloaded", "sha256": "47ec556595d546b1b86867d6b3986da3b88450250a3caf018c66fa2f187a8589", "bytes": 7070040},
    "L12.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/b677fcbf-5e92-4309-9213-67df50e0890f/file_downloaded", "sha256": "4dbb84331a9dc9d714743b84f9b5c7664b25eabc4a21da5ae5db9ee4e0a1a35e", "bytes": 5904300},
    "L13.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/91e4287b-3c2a-4d43-849c-d8623dbdbdd4/file_downloaded", "sha256": "df7391f1b3f63a98c739c1ca2fd92f8b2e1970d3d0482692336bce341cb83f3d", "bytes": 5534628},
    "L14.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/82b49300-f863-4318-ab65-f2d21fc03dc3/file_downloaded", "sha256": "1cc4bcea535a69e40bbc4826c796bb8b6efcf9769f09ef55e77a1d9ac91d177e", "bytes": 5259096},
    "L15.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/60f492f6-2c0a-4494-8f7c-ad62f3c1e39d/file_downloaded", "sha256": "cbb0be4ff74ac51a6c07b98a28f4bc4558f0a7de320f083472b81779b26269dd", "bytes": 5156868},
    "L16.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/62a6152b-33cd-41c4-a54f-7dbe45f6013d/file_downloaded", "sha256": "1aecb757086fbf267cd7e2b2b07d522368dab903122f3ca7ad01f382d0e0e7ce", "bytes": 4538904},
    "L17.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/11fa1942-f86e-4995-8616-8125e78e48f5/file_downloaded", "sha256": "c05023b51d70a7afb1495b70070b0511ccf75483cb410c0f883c5a9acf26c4f5", "bytes": 5076764},
    "L18.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/4c7fefeb-b98a-48fa-8d03-d0e2d0c9fbb6/file_downloaded", "sha256": "86fb509defd1e84bc77cb4f647004f8a5a9b0678655cfea729a815f3027d0501", "bytes": 4974956},
    "L19.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/ded45c62-1bc4-4329-ac75-77fa4306b4e2/file_downloaded", "sha256": "7465b896e2461376d9bb26dcd93ec6a4a8952f58b6ef1ad5ef586d48b1e18cd2", "bytes": 4910684},
    "L20.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/ab792bcf-1457-467f-9387-19225a153cc2/file_downloaded", "sha256": "f5b6fcf02952a4dcfec0d92e115dadf6048f889496de6d4fc97a9d897ee32da0", "bytes": 4266248},
    "L21.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/6b44880a-586d-4750-a30c-91be60076ff5/file_downloaded", "sha256": "a972ee9bd3c584c8a03eb42df3ba0af91d4f81f800b3dc24762bb67b6b18e618", "bytes": 4338740},
    "L22.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/f1c4d4b7-4e57-4109-8a8e-881ec2825ad8/file_downloaded", "sha256": "8a5b7cf9f0a02c6e5a23978b73c29bb7c7a1ee56bd60db15e428638c430195cb", "bytes": 4893668},
    "L23.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/4d3f1119-cebf-4329-b977-311d91b8858c/file_downloaded", "sha256": "8ed03835c3b17685882a15eb2557c81ce0b2ff0cf3e81e4011e6edbad47b4e32", "bytes": 3979520},
    "L24.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/e67706c1-5e58-43bf-bf9c-11f9cc28ab13/file_downloaded", "sha256": "8c4d2525d096eb7a9e8c6d606f3c7e68316f6d382e2d859472890c5d5a6551c5", "bytes": 4023728},
    "L25.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/df248c0f-0a79-4ac9-ab45-a42e6538f946/file_downloaded", "sha256": "9d0221f5dbf006c7c09ef7962bd79e15c09258bb79dcf1b21fdabc7c0de84016", "bytes": 2768204},
    "L26.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/e7d68b56-1dd4-4b01-b354-54194483956d/file_downloaded", "sha256": "8dabeff33421472e319253cc456773438b2ef1789c3ba78905af5b4c0934d139", "bytes": 2987996},
    "L27.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/25d8bf37-7e9e-449c-ab6b-9d8397c933f0/file_downloaded", "sha256": "e03c7884d56c3b1f578a86ad5746c36baabf8fe10777ea0fdfff3b37527129ae", "bytes": 2735000},
    "L28.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/6b317376-845f-4e95-b2cd-62778699caf4/file_downloaded", "sha256": "eeb48bce41feb970043c1de72dea5677d14a28914a72b71e1e0f960a3c97e7ab", "bytes": 3383540},
    "L29.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/5122e30d-31ec-4c7c-8d0c-438eef4c1b52/file_downloaded", "sha256": "3d82e1e7b95d910202792a2cd8f67dcce03607c605f47259ff6c49a58a95734c", "bytes": 3003992},
    "L30.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/de32a781-fc36-4752-b51f-b2edc3cb2540/file_downloaded", "sha256": "aa263effefc28dbcfc38feac8e34dc9c0d6a24bb8642133f0edf3c2b0bb9d12c", "bytes": 3157292},
    "L31.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/2691aced-a76f-4b04-a435-53a8924bfbff/file_downloaded", "sha256": "65b95cdb1e1f9f986ec23fdfaca7cd9a73f5bb131493ef851781fd4c62b8f519", "bytes": 2655620},
    "L32.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/794c55e9-6559-411c-9c0c-ee5bec0cae4b/file_downloaded", "sha256": "3d38f7878774ceb2a3108a2b3d911d889eeb7544a6760f4f3b9fc3cdbb357933", "bytes": 2653556},
    "L33.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/ca5ad859-d157-4364-b5bb-36e621e02d7f/file_downloaded", "sha256": "482f182740089c363303594f9aa6e910a5d80731fc567e737114d377893d2fce", "bytes": 7817420},
    "L34.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/64ac5f4d-adfe-433f-a712-71f0342df84d/file_downloaded", "sha256": "313f87a1a150bdcdf3928e2a601ca9be205d27b2b272847dfe9f885e8549f3d4", "bytes": 6199460},
    "L35.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/1dbe734d-9816-4316-9622-1315e9facadb/file_downloaded", "sha256": "09b9b8c3e02cd1902a6143fdaec57b59f7d6dbdeb031ce161c6601e7856d4e0a", "bytes": 6140276},
    "L36.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/9b1dcd2e-a4b3-4a9d-84c4-12da902d89e6/file_downloaded", "sha256": "ab793b2c2f2e3d4f46cdce4a8155ed16ebfa93ad7e8bfca14915d6f4b7ff260b", "bytes": 6686432},
    "L37.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/27523333-8885-4b15-b855-4e28653e7635/file_downloaded", "sha256": "64f008b5524f6222736be47ddcf678a9c10bc1056ea9ecccc13874fe5ac70323", "bytes": 6036068},
    "L38.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/165a6aa7-b26e-4940-93ff-dfab368514aa/file_downloaded", "sha256": "02110b1e2f435c2af28262fad41fe93323e5f08fbdaa19a3b64e7381d12d29c6", "bytes": 4924340},
    "L39.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/15528a57-ffaa-4dcf-932c-13d7f493bed4/file_downloaded", "sha256": "6b65283ba8bdfb884bfe8e0d722b28aeb364c35e4d1a2d04722035bb6ae8fab4", "bytes": 6365036},
    "L40.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/8d181ab2-8e77-4f11-8ad6-12085b504e26/file_downloaded", "sha256": "78d9593d36919360d10c7cbadb3c18970681c154151831de7186969d81178b26", "bytes": 5142944},
    "L41.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/5c91bbcc-d039-40cb-9526-13f97ad2ad09/file_downloaded", "sha256": "d6a0e3bf3c2ecc40b3f6cb1c064581bca234c897a03266c5aee3e8280e78dd73", "bytes": 5118308},
    "L42.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/95721b79-7f17-41ab-bca6-10043e906033/file_downloaded", "sha256": "cc462a6097d640508fc2ceb0bc38ec795e2145c33cebf11bec9db14ceba6bab6", "bytes": 5518784},
    "L43.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/62e7c23a-38d0-4fb9-ad6d-e6b310d58b7d/file_downloaded", "sha256": "e0bd1bb4f04dff3b89ac06a18048bccad0fedb0d1f5784e8eac34255bc43d96c", "bytes": 6891584},
    "L44.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/19517e7b-a146-431f-8b1d-10b5fcb1dfe2/file_downloaded", "sha256": "43f32b685b1982ded33d7ac727c6d7290949d516f4c95595195f6d5f107a79f7", "bytes": 4865264},
    "L45.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/cd6998fe-32c7-484c-9721-60ce33145f95/file_downloaded", "sha256": "9d39c09555fc81e7cedd56b120c71456dc5eff330fd66c35ad73835d40457dad", "bytes": 5107976},
    "L46.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/fe7c57f5-29ba-4efe-873d-7174e7934511/file_downloaded", "sha256": "639dba8da8ac161e179c1c2dc000fcd8f04005ca1a8a5251da2c0a7c6cbcaf40", "bytes": 5551436},
    "L47.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/4f0983ae-48c8-4f3d-bf09-1aa1b9e0f730/file_downloaded", "sha256": "a9298371d5ba72b04fa2143e1282c2bc78d4886c96edee8573441ed6d6d038af", "bytes": 5176472},
    "L48.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/fdcd09d2-7fce-4dbe-bfd5-94af03ca46fd/file_downloaded", "sha256": "cc735b3a0e2f5a853255bba670e83cf493ee7c4a5a777feb4ef316bfc3111e5e", "bytes": 5229752},
    "L49.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/02ba94cc-7393-4fa5-8f73-475d36672afa/file_downloaded", "sha256": "716b14e7c6d7f52155d9bf917dd27f42d0aa1587bd7258111d2145b3b05e9c19", "bytes": 4507424},
    "L50.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/1bd0c89f-4c64-49ab-ac26-056e5eefb49e/file_downloaded", "sha256": "1b1aea58fe038a73ca428793447a08269efbe9d4eb49c8ee42bc3e66097e0e14", "bytes": 4742108},
    "L51.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/b21f627a-5dd6-44e8-8c89-a807ae914d1c/file_downloaded", "sha256": "faf01f6b659ace60cf2c8a67e3096636333f060d2fb365c6ea987b0ba67ffad7", "bytes": 5234372},
    "L52.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/63d85895-6478-4c30-b089-4766b10a2636/file_downloaded", "sha256": "015fb85b7355967ac87ea11223c39cde0de0f8f2a8c0bf57cfb99b302d49993b", "bytes": 5471288},
    "L53.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/84c1e6b9-f91e-45c0-916f-8a60fcf38083/file_downloaded", "sha256": "d8ad67a4da16b4fae3e112ad2f03d930fa11cfcdb3ad10d2eded4425408c8d10", "bytes": 4999220},
    "L54.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/afa833f3-0616-48b3-ab27-b5b95c9d9256/file_downloaded", "sha256": "de0247a34de8eac66bdc6ae98bacfe5eb2bd114eef65962bfc9525fada72a722", "bytes": 5544704},
    "L55.CDF": {"url": "https://data.mendeley.com/public-files/datasets/pgkrc7wyj4/files/b2b05de9-f984-47bc-a639-37804bf54337/file_downloaded", "sha256": "4b0fcbae5ff1ee289a7fdd0eb25a918ef490715f77c3d85b67019d470bc7abc7", "bytes": 5450000},
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

    print(f"[INFO] Baixando {nome} ({spec['bytes'] / 1e6:.1f} MB) ...")
    req = urllib.request.Request(
        spec["url"], headers={"User-Agent": "guaraci-dataset-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
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
        pasta_destino = Path(raiz) / "mendeley_pgkrc7wyj4"
    pasta_destino = Path(pasta_destino)
    pasta_destino.mkdir(parents=True, exist_ok=True)

    for nome, spec in _ARQUIVOS.items():
        _baixar_um(nome, spec, pasta_destino)

    return pasta_destino


if __name__ == "__main__":
    destino = sys.argv[1] if len(sys.argv) > 1 else None
    pasta = baixar_dataset(destino)
    print(f"[OK] dataset Mendeley pgkrc7wyj4 (GC-MS lavanda, 55 amostras) "
          f"pronto em: {pasta}")
