#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baixar_deephs_kaki.py — Baixa um SUBCONJUNTO do dataset publico DeepHS
Fruit (Varga, Makowski & Zell, IJCNN 2021, arXiv:2104.09808,
github.com/cogsys-tuebingen/deephs_fruit) para um diretorio de CACHE LOCAL
fora do controle de versao -- mesmo padrao de `baixar_mendeley_oleos.py`
(SHA256+tamanho pinados, nunca versiona o dado de terceiro).

Subconjunto escolhido (Passo 93 da INSTRUCAO_HSI_MINIMO_VIAVEL.md): fruta
Kaki (caqui), camera VIS (Specim FX10, 224 bandas, 397.66-1003.81 nm)
apenas -- as outras 4 frutas (Avocado 72G, Kiwi 44G, Papaya 22G, Mango
2.7G) e as outras 2 cameras (NIR, VIS_COR) NAO sao baixadas: o "minimo
viavel" nao precisa do dataset inteiro (o Kaki.zip completo sozinho ja'
tem 2.2G) para validar o pipeline HSI (Passos 94-102). 56 gravacoes VIS
de Kaki, 38 frutas fisicas distintas, rotulo `ripeness_state`
(unripe/perfect/overripe) -- ver `hsi_io.load_deephs_kaki_dataset`.

O dataset e' distribuido como .zip por fruta (nao ha' download por
arquivo individual) -- este script usa HTTP Range requests para ler
SOMENTE o directorio central do zip + os 112 membros deste subconjunto
(56 gravacoes x 2 arquivos, .hdr+.bin), sem baixar os outros ~1500
membros do Kaki.zip nem o resto do dataset (~212G para as outras 4
frutas). Cada arquivo extraido e' verificado por SHA256+tamanho ANTES de
gravar em disco -- mesma regra de `docs/VALIDACAO_PUBLICA.md` secao 6.

LICENCA -- achado, nao suposicao: o repositorio (README, LICENSE via
`api.github.com/repos/cogsys-tuebingen/deephs_fruit` -> `license: None`)
NAO declara uma licenca formal (SPDX) explicita. Os autores afirmam
publicamente "we make public" o dataset (README, paper IJCNN 2021) e o
disponibilizam para download HTTP sem autenticacao/sessao -- mesmo
tratamento ja' dado ao Corn neste projeto ("distribuido publicamente
... ver a fonte para os termos", `docs/VALIDACAO_PUBLICA.md` secao 4).
Consistente com a politica deste projeto (`datasets/README.md`): o dado
NUNCA e' versionado/redistribuido, so' usado para validacao em cache
local efemero -- o mesmo uso que o Corn ja' tem sem licenca SPDX
tampouco.

Uso:
    python scripts/download_datasets/baixar_deephs_kaki.py [DESTINO]

DESTINO (opcional): pasta onde salvar. Default: $GUARACI_DATASETS_DIR/
deephs_kaki_vis, ou ./datasets_publicos/deephs_kaki_vis se a variavel
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
from pathlib import Path

# HTTPS hardcoded -- nunca construido a partir de entrada do usuario/env
# (evita SSRF/redirect trivial, mesma regra de baixar_mendeley_oleos.py).
_ZIP_URL = ("https://cogsys.cs.uni-tuebingen.de/webprojects/"
            "DeepHS-Fruit-2023-Datasets/Kaki.zip")
_ANNOTATIONS_URL = ("https://cogsys.cs.uni-tuebingen.de/webprojects/"
                    "DeepHS-Fruit-2023-Datasets/annotations-upd-2024-01-09.zip")
_ANNOTATIONS_SHA256 = "14275450e362684bc379a5aaf6c845cf82b0f9d5912036b9401d59a4b964a3f3"
_ANNOTATIONS_BYTES = 165730

# 56 gravacoes VIS de Kaki (112 arquivos: .hdr+.bin por gravacao) --
# pinado em 2026-09-01 por leitura direta (download + sha256sum), ver
# docstring do modulo.
_ARQUIVOS = {
    "Kaki/VIS/day_1_m3/kaki_day_1_m3_15_back.bin": {"sha256": "88c477e86711fd0ae6e41792da6b5eb3effada0a5b216e43c239eeda15587d4e", "bytes": 3670016},
    "Kaki/VIS/day_1_m3/kaki_day_1_m3_15_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_1_m3/kaki_day_1_m3_15_front.bin": {"sha256": "2c17cd61815b520301ef9fa694c70cde1f4d919b2df7f38b094a3f8c9a439955", "bytes": 3670016},
    "Kaki/VIS/day_1_m3/kaki_day_1_m3_15_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_1_m3/kaki_day_1_m3_19_back.bin": {"sha256": "fbe835e4323fd65635564fee942f0d9a908d82dd922093191b34ac4e144e7222", "bytes": 3670016},
    "Kaki/VIS/day_1_m3/kaki_day_1_m3_19_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_1_m3/kaki_day_1_m3_19_front.bin": {"sha256": "888b924942c8cc0b4eadec63e140f97278595746485021f6deeebdad47b8a93b", "bytes": 3670016},
    "Kaki/VIS/day_1_m3/kaki_day_1_m3_19_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_2_m3/kaki_day_2_m3_21_back.bin": {"sha256": "763d49eec6990ad9a39a3b1dc53b87d83c2a62d08ef7cb860bafb194db2bc439", "bytes": 3670016},
    "Kaki/VIS/day_2_m3/kaki_day_2_m3_21_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_2_m3/kaki_day_2_m3_21_front.bin": {"sha256": "d8318eefa50e53b46256c3f434e60978c66aff606809b53506b5eff92bf5b71f", "bytes": 3670016},
    "Kaki/VIS/day_2_m3/kaki_day_2_m3_21_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_2_m3/kaki_day_2_m3_28_back.bin": {"sha256": "6122b77dda3c4569b1598087df4a8c495026e509cd22ff5820dc58c349b7deed", "bytes": 3670016},
    "Kaki/VIS/day_2_m3/kaki_day_2_m3_28_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_2_m3/kaki_day_2_m3_28_front.bin": {"sha256": "681bd6eb3d75688e24a25d451c4c086548ceda472377a5b10cc699f3cd1d2096", "bytes": 3670016},
    "Kaki/VIS/day_2_m3/kaki_day_2_m3_28_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_2_m3/kaki_day_2_m3_31_back.bin": {"sha256": "a70146d91e66e24669c7d40f9d7d50f7c4a5fb4638dbc1bfa35f73de3d8b520d", "bytes": 3670016},
    "Kaki/VIS/day_2_m3/kaki_day_2_m3_31_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_2_m3/kaki_day_2_m3_31_front.bin": {"sha256": "bdba4a15acfcc2570defa3bd69872b42030b6c06648bba1d4805015fbffa7871", "bytes": 3670016},
    "Kaki/VIS/day_2_m3/kaki_day_2_m3_31_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_3_m3/kaki_day_3_m3_02_back.bin": {"sha256": "d6816957a915fc24c14450a60f1c1d5782a81800a434e405f8461bb5e6cd0f1c", "bytes": 3670016},
    "Kaki/VIS/day_3_m3/kaki_day_3_m3_02_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_3_m3/kaki_day_3_m3_11_back.bin": {"sha256": "80b1eb1b4a7a928d573b566135b3a00e81ae06173c5ae3afdde793795b3f6531", "bytes": 3670016},
    "Kaki/VIS/day_3_m3/kaki_day_3_m3_11_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_3_m3/kaki_day_3_m3_11_front.bin": {"sha256": "f588d0795a2b2625a8308289cf380194e7f6da03371cc860b2b6387743f0cb89", "bytes": 3670016},
    "Kaki/VIS/day_3_m3/kaki_day_3_m3_11_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_3_m3/kaki_day_3_m3_35_back.bin": {"sha256": "4e1fabd3f098339d0fb3bcb7d20c85a5dddbee515e82e30a91bfe7d534802731", "bytes": 3670016},
    "Kaki/VIS/day_3_m3/kaki_day_3_m3_35_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_3_m3/kaki_day_3_m3_35_front.bin": {"sha256": "6484da2c3328956c9c033d0314522372338090fa724f10f517ea375afe2c9061", "bytes": 3670016},
    "Kaki/VIS/day_3_m3/kaki_day_3_m3_35_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_3_m3/kaki_day_3_m3_40_back.bin": {"sha256": "860c4401aa7bb8f44e14633ffe224518c5f0c8a9c76e9423044fbf276f5a9bb9", "bytes": 3670016},
    "Kaki/VIS/day_3_m3/kaki_day_3_m3_40_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_4_m3/kaki_day_4_m3_07_front.bin": {"sha256": "5aebe03976c4cc6b6a8a720ef81092c1442c3600b01f19748246fde44b7d0b6c", "bytes": 3670016},
    "Kaki/VIS/day_4_m3/kaki_day_4_m3_07_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_4_m3/kaki_day_4_m3_14_back.bin": {"sha256": "27743cc0b5f6f03cd7a12ecc3a4e0f25ca6e9ef7c36a2eea787ac9a03a7b45b2", "bytes": 3670016},
    "Kaki/VIS/day_4_m3/kaki_day_4_m3_14_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_4_m3/kaki_day_4_m3_20_back.bin": {"sha256": "7ad20759f0ce6704259491cb65bc89160f6b2771d8da3f692adc8d438877d872", "bytes": 3670016},
    "Kaki/VIS/day_4_m3/kaki_day_4_m3_20_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_4_m3/kaki_day_4_m3_20_front.bin": {"sha256": "cc9a772111f36c25fa6d945ddeee25b873bf28f9605b2b5d5fb43b50aab7731f", "bytes": 3670016},
    "Kaki/VIS/day_4_m3/kaki_day_4_m3_20_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_4_m3/kaki_day_4_m3_36_back.bin": {"sha256": "c4415580451d74c89976c0ca59a5cdcd6e6e662953a81dfb6da5facf33db87b2", "bytes": 3670016},
    "Kaki/VIS/day_4_m3/kaki_day_4_m3_36_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_4_m3/kaki_day_4_m3_36_front.bin": {"sha256": "9a8912e8fc33d118221d9bd61a79cd4289f08a48f3838d10bd4ea85bbcf45795", "bytes": 3670016},
    "Kaki/VIS/day_4_m3/kaki_day_4_m3_36_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_5_m3/kaki_day_5_m3_05_back.bin": {"sha256": "2fbf20ce66d003950d81f88e82a2352be18a971046a462418b0b2b97608941b0", "bytes": 3670016},
    "Kaki/VIS/day_5_m3/kaki_day_5_m3_05_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_5_m3/kaki_day_5_m3_05_front.bin": {"sha256": "460ef4fac1d484e7e3810544c38e03375025b7e230df7d88e92ec174eab080bd", "bytes": 3670016},
    "Kaki/VIS/day_5_m3/kaki_day_5_m3_05_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_5_m3/kaki_day_5_m3_13_back.bin": {"sha256": "a6530624afa7e76b7a3f64f94943c9e1266ec5c25888970ed52a4d03e7aa967e", "bytes": 3670016},
    "Kaki/VIS/day_5_m3/kaki_day_5_m3_13_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_5_m3/kaki_day_5_m3_13_front.bin": {"sha256": "dd37ca5c9ae9f9fd0bc46107051437de1096f957b09cb9bc142048985ba98b75", "bytes": 3670016},
    "Kaki/VIS/day_5_m3/kaki_day_5_m3_13_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_5_m3/kaki_day_5_m3_26_front.bin": {"sha256": "1e3634f8ade1dccd4de5c0ab0a65f66aa965c49e0bfc7408aa9ee65127e027fd", "bytes": 3670016},
    "Kaki/VIS/day_5_m3/kaki_day_5_m3_26_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_5_m3/kaki_day_5_m3_33_back.bin": {"sha256": "a3f3fc7b79d46dc325f461a761f9ae332c19f32982eb4a13960b476e467ea052", "bytes": 3670016},
    "Kaki/VIS/day_5_m3/kaki_day_5_m3_33_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_5_m3/kaki_day_5_m3_33_front.bin": {"sha256": "6fa7350cafec4f64c3ead34d90f8e0d2e1a6cfc1bdffd0e16aae79028e198537", "bytes": 3670016},
    "Kaki/VIS/day_5_m3/kaki_day_5_m3_33_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_12_back.bin": {"sha256": "04832138712cff668cba8015b8db0faf6c503ad583d72f2446fe3875581198bb", "bytes": 3670016},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_12_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_12_front.bin": {"sha256": "27a04169d376b04268ec8f67f75f3147d41fe3d62e89ef852068b703c7d60179", "bytes": 3670016},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_12_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_23_back.bin": {"sha256": "753813d4a74b0f5c0d4c6a0ab67345f465ad7c1cf2a2d8fdc945f56ca326934b", "bytes": 3670016},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_23_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_24_back.bin": {"sha256": "8ea46ac039837aaa021ed9fa7106dbf1898431cf13b6f3073bc782b263184a77", "bytes": 3670016},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_24_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_24_front.bin": {"sha256": "35067756b3073ac701086143a25c69499e657fae23dd9efe94fabe134933c7bf", "bytes": 3670016},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_24_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_29_back.bin": {"sha256": "0d392801f49d139900ed60c1b0cc0b3937d6074ec6dbca6b03306b0015055f7d", "bytes": 3670016},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_29_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_29_front.bin": {"sha256": "5b449c104a2859703a9cc7b668a07ca8e36cb2bfe9f9710878bab498e3e3fa27", "bytes": 3670016},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_29_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_37_back.bin": {"sha256": "9e6af4906f3a9be585fbfefb31e54c9c32ab79a8b7b80ec9d90ee50d3301e32f", "bytes": 3670016},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_37_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_39_back.bin": {"sha256": "0c520593356f8e75531b55c481be852616396f514e30b005c28710806a661527", "bytes": 3670016},
    "Kaki/VIS/day_7_m3/kaki_day_7_m3_39_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_06_back.bin": {"sha256": "12414e75e28dd8f25dead0e2002a107981515d024817a251bab66e421a8da9e3", "bytes": 3670016},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_06_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_08_back.bin": {"sha256": "3b10abaf357fceaa24d1057e73c1bf1160383c790793e26b12ea5c90413e8fec", "bytes": 3670016},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_08_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_08_front.bin": {"sha256": "f4d3a4c918b615f7faaa50ef8e2a7a49fef74e0b9817449c8b520dfb15f044d1", "bytes": 3670016},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_08_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_09_back.bin": {"sha256": "5c830c99a7616424ccc9233a1f9b0b832daa1234090326a7666c402a07576c64", "bytes": 3670016},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_09_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_09_front.bin": {"sha256": "856f2eb17581e209b8c00b17650e5f4b57c346c7178ce99e4762df677b82dea1", "bytes": 3670016},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_09_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_16_back.bin": {"sha256": "a1f1b3ce2765cf59d7ce5372acd3356b72dbb50d29235fd3b9b6b272ed67c60e", "bytes": 3670016},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_16_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_22_back.bin": {"sha256": "9bcbbd4c7f620d589df39ce068a0f47a8a16df975cbd81697775c0b6129b58f7", "bytes": 3670016},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_22_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_22_front.bin": {"sha256": "7e4914139433f49015f059ffa22aa1ed7128672b23da727c18d7e4a1ee62ee62", "bytes": 3670016},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_22_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_38_back.bin": {"sha256": "44a0cd9d44df1eec14b7f8f508ea419801acc9f71d2611ba45ab10fe923ffa5a", "bytes": 3670016},
    "Kaki/VIS/day_8_m3/kaki_day_8_m3_38_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_04_back.bin": {"sha256": "41d392160f3eaaae52d35c7cc1f9179366615c641e28fe71b603de8e5f4a4657", "bytes": 3670016},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_04_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_17_back.bin": {"sha256": "069b4359e785e66bc44626050c8d01d41aea388f0427826c801eb851168aded9", "bytes": 3670016},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_17_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_17_front.bin": {"sha256": "87633e1e511e32f15a856dbbc95a187e58193df1876f209a994d34b38a2e6d14", "bytes": 3670016},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_17_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_18_back.bin": {"sha256": "8ea2c1f3478ff1e1d226ebf32b2b65981eddf0892828f4d09258025e335fb9b6", "bytes": 3670016},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_18_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_18_front.bin": {"sha256": "36f7a9ecac284de4a78c1ae6bd99520f6bcb97de21a5eedf257b3e33c3147106", "bytes": 3670016},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_18_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_25_back.bin": {"sha256": "cf09efae7d2b8b7a1d89608475d3d61ca706a0dc5ee849c692848a1b497fedf5", "bytes": 3670016},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_25_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_32_front.bin": {"sha256": "dc99d71878d8dcd2e8c43200dcd64445fa39891ced6623e7a8952dd6ad5ea50e", "bytes": 3670016},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_32_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_34_back.bin": {"sha256": "c79668e9346fd6485ad52b3d9f1a6cd9f72546e194eab698da78c44be2e65c3e", "bytes": 3670016},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_34_back.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_34_front.bin": {"sha256": "a5483b1db0cb3c1e0f44a3926b31e4ab6cb6735385a20ea8691e721fd625fca7", "bytes": 3670016},
    "Kaki/VIS/day_9_m3/kaki_day_9_m3_34_front.hdr": {"sha256": "703ad9acc94626b52f0a94cc94a448943d3cce3bbae06ed86412c1288e344071", "bytes": 131},
}


class _HTTPRangeFile(io.IOBase):
    """Arquivo remoto lido sob demanda via HTTP Range -- permite abrir um
    .zip de 2.2G e ler so' o directorio central + os membros pedidos, sem
    baixar o arquivo inteiro. `zipfile.ZipFile` so' precisa de
    seek/tell/read para funcionar com acesso aleatorio."""

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


def baixar_dataset(pasta_destino: "str | Path | None" = None) -> Path:
    if pasta_destino is None:
        raiz = os.environ.get("GUARACI_DATASETS_DIR", "datasets_publicos")
        pasta_destino = Path(raiz) / "deephs_kaki_vis"
    pasta_destino = Path(pasta_destino)
    pasta_destino.mkdir(parents=True, exist_ok=True)

    ann = _baixar_annotations(pasta_destino)
    cam_vis = next(c for c in ann["cameras"] if c["id"] == "VIS")
    anns_by_rec = {a["record_id"]: a for a in ann["annotations"]}
    recs = [r for r in ann["records"]
            if r["fruit"] == "Kaki" and r["camera_type"] == "VIS"]

    faltando = [nome for r in recs
                for nome in (r["files"]["header_file"], r["files"]["data_file"])
                if nome not in _ARQUIVOS]
    if faltando:
        raise RuntimeError(
            f"{len(faltando)} arquivo(s) que a fonte lista para Kaki/VIS "
            f"nao estao pinados neste script (a fonte mudou desde a "
            f"auditoria de 2026-09-01?): {faltando[:5]}...")

    zf: "zipfile.ZipFile | None" = None
    manifest = {"camera": cam_vis, "records": []}
    for r in recs:
        a = anns_by_rec[r["id"]]
        for nome in (r["files"]["header_file"], r["files"]["data_file"]):
            spec = _ARQUIVOS[nome]
            local = pasta_destino / nome.replace("/", os.sep)
            if (local.is_file() and local.stat().st_size == spec["bytes"]
                    and hashlib.sha256(local.read_bytes()).hexdigest() == spec["sha256"]):
                continue
            if zf is None:
                print(f"[INFO] Abrindo {_ZIP_URL} via HTTP Range ...")
                zf = zipfile.ZipFile(_HTTPRangeFile(_ZIP_URL))
            print(f"[INFO] Extraindo {nome} ...")
            dados = zf.read(nome)
            _verificar_e_gravar(local, dados, spec["sha256"], spec["bytes"], nome)
        manifest["records"].append({
            "id": r["id"], "day": r["day"], "side": r["side"],
            "header_file": r["files"]["header_file"],
            "data_file": r["files"]["data_file"],
            "ripeness_state": a["ripeness_state"],
            "storage_days": a["storage_days"], "firmness": a["firmness"],
        })

    with open(pasta_destino / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"[OK] {len(recs)} gravacoes VIS de Kaki prontas em {pasta_destino}.")
    return pasta_destino


if __name__ == "__main__":
    destino_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pasta = baixar_dataset(destino_arg)
    print(f"[OK] dataset DeepHS Fruit/Kaki/VIS pronto em: {pasta}")
