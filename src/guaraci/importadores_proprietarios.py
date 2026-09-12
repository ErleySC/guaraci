# -*- coding: utf-8 -*-
"""importadores_proprietarios.py -- Leitores de formato binario proprietario
de instrumento (Bloco 18).

ESCOPO DESTE MODULO: converte um UNICO arquivo binario proprietario para o
MESMO contrato de `dados_io.parse_dx`/`dados_io.parse_spectrum` -- uma
tupla `(X, Y)` de arrays 1D (`eixo espectral`, `intensidade`). Isso o torna
plugavel em qualquer lugar que hoje chame `parse_dx`/`parse_spectrum` por
arquivo. NAO reimplementa a deteccao de estrutura de pasta/extracao de
metadados de `dados_io.load_dx` (subpastas por classe, ##TITLE=, mae_id):
arquivos OPUS nao tem uma extensao fixa e filtravel (`.0`, `.1`, `.2`... um
numero por repeticao de medida, nao um marcador de formato) -- generalizar
`_listar_arquivos_espectro`/`load_dx` para esse padrao e' trabalho de
escopo proprio, fora do que o Bloco 18 pediu ("converter para a estrutura
ja usada", nao "generalizar a varredura de pasta"), e arriscaria codigo ja
congelado (Bloco B) por uma extensao que nao e' pura adicao.

OPUS (Bruker FT-NIR/FT-MIR -- prioridade do bloco, formato historico deste
projeto): via `brukeropus` (Josh Duran, MIT -- compativel com
GPL-3.0-or-later deste projeto; verificado no classifier do pacote
publicado no PyPI em 2026-09-04, ver `pyproject.toml` extra `[opus]`).
Avaliado ANTES de escrever parser do zero (Bloco 18 pede explicitamente
para nao reinventar): `brukeropusreader` (mais antigo, GPLv3, mantido pela
ultima vez em 2019) tambem seria compativel de licenca, mas `brukeropus`
e' o mais recentemente mantido (release 2025-11-14) e MIT permite reuso
mais amplo (ex.: por um consumidor comercial da API deste projeto, ver
`roadmap_mercado`) -- import LAZY (so' ao chamar `parse_opus`), pacote
opcional (`pip install guaraci-chemometrics[opus]`), nao trava o pacote
base para quem nunca abre arquivo OPUS.

LIMITACAO HONESTA: sem um arquivo `.0`/OPUS real de teste disponivel neste
ambiente (a biblioteca `brukeropus` nao empacota exemplo, e nenhum arquivo
publico foi obtido para este checkout -- ver `tests/test_importadores_
proprietarios.py`), a extracao X/Y e' testada contra o CONTRATO DOCUMENTADO
e verificado no codigo-fonte da biblioteca (`brukeropus.file.data.Data.x`/
`.y`, `OPUSFile.data_keys`/`.is_opus`), via double de teste que reproduz
essa forma exatamente -- nao contra um binario OPUS de verdade. Cobertura
fim-a-fim com instrumento real fica pendente ate' haver um arquivo de
exemplo genuino.

SPC (Galactic/Thermo -- Grupo 1 do mapa de completude): via `spcfile`
(Nikolaj Langemark, LGPL-3.0 -- compativel com GPL-3.0-or-later deste
projeto: LGPL e' uma licenca permissiva de biblioteca desenhada
especificamente para uso por programas sob licenca mais restritiva,
incluindo GPL; verificado no LICENSE do repositorio, nao so' no README).
Repositorio criado e ativo em 2026 (39 commits, ultimo push 2026-01-25),
mas SEM release no PyPI ainda -- so' instalavel via
`pip install git+https://github.com/kogens/spcfile.git` (ver extra
opcional `[spc]` em `pyproject.toml`). ATENCAO PyPI: se este pacote
(`guaraci-chemometrics`) for publicado no PyPI antes de `spcfile` ganhar
um release proprio la', o extra `[spc]` com URL git direta sera' REJEITADO
pelo upload do Warehouse (PyPI nao aceita `Requires-Dist` com referencia
direta de URL) -- trocar para o nome do pacote puro assim que houver
release, ou documentar `[spc]` como "instale manualmente" no README nesse
cenario.
Avaliado ANTES: `specio` (BSD-3, tambem le SPC) e `spc_spectra`/
`rohanisaac/spc` (GPL-3.0, tambem compativel) foram descartados como
DEPENDENCIA porque ambos pararam de ser lancados em 2018 (specio 0.1.0,
2018-02-09; spc_spectra 0.4.0, 2018-05-03) -- exatamente o padrao que o
Bloco 18/Grupo 1 pede para tratar como sinal de alternativa, nao para
depender as cegas.
TESTADO COM ARQUIVO REAL (nao sintetico): `spectra.spc` (Raman, do
dataset de exemplo do pacote `specio`, BSD-3, baixado publicamente) e
`nir.spc` (NIR multi-subfile com 20 espectros, do `test_data/` de
`rohanisaac/spc`, GPL-3.0, baixado publicamente) -- ver
`tests/fixtures/spc/`. Multi-subfile: so' o primeiro subarquivo e'
retornado (mesma logica de "um espectro por arquivo" do resto deste
modulo -- ficheiros com repeticoes fisicas ja tem seu proprio mecanismo
de pasta/mae_id em `dados_io.load_dx`)."""
from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "parse_opus",
    "parse_spc",
]

# Ordem de preferencia: absorbancia (o que a maioria dos fluxos deste
# projeto consome) > transmitancia > reflectancia > espectro de amostra
# bruto (single-channel) -- so' cai pro proximo se o anterior nao existir
# no arquivo (nem todo arquivo OPUS grava todos os blocos).
_CHAVES_PREFERIDAS = ("a", "t", "r", "sm")


def parse_opus(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Le um arquivo binario OPUS (Bruker) e retorna `(X, Y)` -- mesmo
    contrato de `dados_io.parse_dx`/`dados_io.parse_spectrum`: dois arrays
    1D, eixo espectral (numero de onda, cm^-1) e intensidade.

    Escolhe o primeiro bloco de dados disponivel em ordem de preferencia
    (`_CHAVES_PREFERIDAS`: absorbancia > transmitancia > reflectancia >
    single-channel bruto); levanta `ValueError` se o arquivo nao for OPUS
    valido ou nao tiver nenhum bloco de dados espectrais reconhecivel.

    Requer o pacote opcional `brukeropus` (`pip install
    guaraci-chemometrics[opus]`) -- import LAZY, so' ao chamar esta funcao."""
    try:
        from brukeropus import read_opus
    except ImportError as e:
        raise ImportError(
            "Pacote opcional 'brukeropus' nao instalado -- leitura de "
            "arquivos OPUS indisponivel (pip install "
            "guaraci-chemometrics[opus])."
        ) from e

    opus_file = read_opus(filepath)
    if not getattr(opus_file, "is_opus", False):
        raise ValueError(f"{filepath}: nao reconhecido como arquivo OPUS valido")

    chaves_disponiveis = list(getattr(opus_file, "data_keys", []) or [])
    chave = next((k for k in _CHAVES_PREFERIDAS if k in chaves_disponiveis), None)
    if chave is None:
        chave = chaves_disponiveis[0] if chaves_disponiveis else None
    if chave is None:
        raise ValueError(
            f"{filepath}: nenhum bloco de dados espectrais encontrado no "
            f"arquivo OPUS (data_keys vazio)")

    bloco = getattr(opus_file, chave)
    X = np.asarray(bloco.x, dtype=float)
    Y = np.asarray(bloco.y, dtype=float)
    if X.shape != Y.shape or X.size == 0:
        raise ValueError(
            f"{filepath}: bloco '{chave}' com eixo/intensidade "
            f"inconsistentes (x={X.shape}, y={Y.shape})")
    return X, Y


def parse_spc(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Le um arquivo binario SPC (Galactic/Thermo GRAMS) e retorna `(X, Y)`
    -- mesmo contrato de `dados_io.parse_dx`/`dados_io.parse_spectrum`.

    Arquivos SPC multi-subarquivo (medidas repetidas num unico arquivo,
    flag `TMULTI`) retornam so' o PRIMEIRO subarquivo -- repeticoes fisicas
    ja tem seu proprio mecanismo de pasta/mae_id no restante do pipeline
    (ver nota do modulo); ler as demais exigiria um contrato de retorno
    diferente (lista de `(X, Y)` por arquivo), fora do escopo aqui.

    Levanta `ValueError` se o arquivo nao for SPC valido (versao de
    cabecalho desconhecida, ou eixo/intensidade do primeiro subarquivo
    inconsistentes).

    Requer o pacote opcional `spcfile` (`pip install
    'spcfile @ git+https://github.com/kogens/spcfile.git'`, ou
    `guaraci-chemometrics[spc]` -- ver ATENCAO PyPI na docstring do
    modulo) -- import LAZY, so' ao chamar esta funcao."""
    try:
        from spcfile import SPCFile
    except ImportError as e:
        raise ImportError(
            "Pacote opcional 'spcfile' nao instalado -- leitura de "
            "arquivos SPC indisponivel (pip install "
            "'spcfile @ git+https://github.com/kogens/spcfile.git')."
        ) from e

    try:
        spc_file = SPCFile(filepath)
    except ValueError as e:
        raise ValueError(f"{filepath}: nao reconhecido como arquivo SPC valido ({e})") from e

    try:
        subarquivo = spc_file[0]
    except (IndexError, KeyError) as e:
        raise ValueError(f"{filepath}: arquivo SPC sem nenhum subarquivo de dados") from e

    X = np.asarray(subarquivo.x, dtype=float)
    Y = np.asarray(subarquivo.y, dtype=float)
    if X.shape != Y.shape or X.size == 0:
        raise ValueError(
            f"{filepath}: subarquivo 0 com eixo/intensidade "
            f"inconsistentes (x={X.shape}, y={Y.shape})")
    return X, Y
