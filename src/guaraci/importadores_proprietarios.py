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
de pasta/mae_id em `dados_io.load_dx`).

PerkinElmer `.sp` (Grupo 1): NENHUMA biblioteca Python madura e ATIVA foi
encontrada. `spectrochempy` (que tem um plugin `spectrochempy-perkinelmer`
e le' `.sp`) usa licenca CeCILL-B -- CONFIRMADO INCOMPATIVEL com GPL na
lista oficial da FSF (CeCILL-B e' permissiva mas NAO tem a clausula de
compatibilidade GPL que a CeCILL "pura" tem; diferente de CeCILL-B, a
CeCILL simples e' compativel -- mas o pacote usado e' especificamente
CeCILL-B). `specio` (BSD-3, compativel de licenca) le' `.sp` nativamente,
mas parou em 2018 (0.1.0), depende de `six` (biblioteca de compatibilidade
Python 2/3 morta ha' anos) e teve um issue publico de instalacao quebrada
-- mesmo criterio de descarte do SPC acima.
DECISAO: como nenhuma biblioteca madura E ativa existe, e o formato `.sp`
NAO tem especificacao publica oficial da PerkinElmer (a estrutura de
blocos TLV aninhados abaixo foi reverso-projetada pelo autor do `specio`,
nao documentada pelo fabricante), a rota escolhida foi ADAPTAR (nao
depender em tempo de execucao) a logica de leitura ja publicada do
plugin `specio.plugins.sp` (Copyright 2017 Guillaume Lemaitre, BSD-3
Clause -- compativel com GPL-3.0-or-later, permite adaptacao com
atribuicao) para este modulo, removendo o uso de `six` e adequando ao
contrato `(X, Y)` daqui. Isso evita puxar um pacote inteiro abandonado
(com sua dependencia morta) so' por um unico leitor de formato, mas
tambem significa que a logica de caminhada pela arvore de blocos
aninhados (`_sp_percorrer_diretorio_de_blocos`) e' um PORTE, nao uma
implementacao derivada de especificacao propria -- um ramo dela (recuo
para nivel de aninhamento anterior) nunca foi exercitado pelo unico
arquivo real disponivel para teste e fica sem cobertura de execucao real
(ver docstring da funcao).
TESTADO COM ARQUIVO REAL (nao sintetico): `spectra.sp` (FT-IR, do dataset
de exemplo do pacote `specio`, BSD-3, baixado publicamente) -- ver
`tests/fixtures/sp/`. Saida conferida byte-a-byte contra o docstring
publicado do `specio` (`spectra.wavelength`/`spectra.amplitudes`, mesmos
5 primeiros valores).

RMN bruto Bruker (Grupo 1, fechamento final): via `nmrglue` (Jonathan J.
Helmus, BSD-3-Clause -- confirmado no LICENSE.txt do repositorio, nao so'
no classifier do PyPI -- compativel com GPL-3.0-or-later). `Requires:
numpy, scipy` -- AMBOS ja' dependencia base deste projeto, entao o extra
opcional `[rmn]` acrescenta ZERO dependencia transitiva nova (ver medicao
no `docs/MAPA_COMPLETUDE_V1.md`, Grupo 3).
ESCOPO DELIBERADO -- so' o ESPECTRO JA PROCESSADO, nao o FID bruto: um
experimento Bruker TopSpin grava tanto o FID (dominio do tempo, sinal
complexo) quanto o espectro ja' processado pelo proprio instrumento em
`pdata/<N>/1r` (FFT + correcao de fase JA aplicadas pelo TopSpin -- isso
e' o fluxo real de qualquer laboratorio, o software do espectrometro
sempre processa e grava o `pdata` automaticamente). Ler o FID cru
(`ng.bruker.read`, tambem suportado pelo `nmrglue` e viavel) exigiria
este modulo implementar sua PROPRIA FFT + correcao de fase -- isso e'
processamento de sinal, nao leitura de formato, e fica fora do escopo
deste modulo (mesmo raciocinio de escopo do cabecalho do arquivo).
Registrado como extensao futura possivel, nao como lacuna de biblioteca.
TESTADO COM ARQUIVO REAL (nao sintetico, nao um dataset publico
pre-processado por terceiro): diretorio `pdata/1/` de um experimento
Bruker BioSpin real (PRESS, 300MHz, TopSpin) do repositorio publico
`CIC-methods/FID-A` (dados via Git LFS, baixados pelo endpoint de midia
do GitHub) -- ver `tests/fixtures/rmn_bruker/PROVENANCIA.md`. Eixo ppm
construido via `ng.bruker.guess_udic`/`ng.fileiobase.uc_from_udic` (API
propria do `nmrglue` para essa conversao, nao uma formula derivada por
este modulo) -- decrescente, convencao padrao de RMN (mesmo principio de
nao forcar ordem crescente ja' aplicado a OPUS/SPC/`.sp` acima).

HPLC cromatograma bruto / UV-Vis comercial (Grupo 1, fechamento final):
via `rainbow-api` (Evan Shi e Eugene Kwan, LGPL-3.0-or-later -- verificado
no `COPYING.LESSER` do wheel instalado, nao so' no classifier -- mesmo
padrao ja' aplicado a `spcfile`; compativel com GPL-3.0-or-later).
Dependencia transitiva NOVA: `lxml` (BSD-3-Clause, compativel) -- unico
extra deste modulo que acrescenta peso nao-trivial (~14MB, ver medicao em
`docs/MAPA_COMPLETUDE_V1.md`, Grupo 3; ainda assim bem abaixo do maior
extra ja' medido, xgboost 58MB).
ESCOPO: `rainbow-api` le' diretorios NATIVOS de fabricante -- Agilent
(`.D`/`.dx`: canais `.uv`/`.ch`, UV/DAD e tambem FID de GC quando o
detector e' `FID`) e Waters (`.raw`) -- NAO o formato aberto ANDI/AIA
netCDF (ASTM E1947, especifico de cromatografia, primo do ANDI-MS/E1948
ja' lido por `gcms_io.py` via `scipy.io.netcdf_file`). Avaliado
explicitamente ANTES de escrever qualquer coisa: nenhum arquivo `.CDF`
ANDI-E1947 (cromatografia pura, canal unico) real e publicamente
disponivel foi encontrado nesta busca -- os datasets ANDI publicos
achados (ex.: `faahKO`/Bioconductor) sao todos ANDI-**MS** (E1948, mesmo
formato ja' coberto), nao ANDI-Chrom. Como `rainbow-api` cobre o caso
real de uso (dado de fabricante, o que um laboratorio de fato tem) com
biblioteca madura, licenca compativel e arquivo real disponivel, a rota
ANDI-Chrom por `scipy.io.netcdf_file` fica registrada como extensao
futura possivel (nao descartada por incompatibilidade, so' sem arquivo
real disponivel agora) -- ver `docs/MAPA_COMPLETUDE_V1.md`, Grupo 1.
TESTADO COM ARQUIVO REAL (nao sintetico): diretorio `pink.D` (2 canais UV
de comprimento de onda unico, 210nm e 230nm, 9000 pontos cada) da propria
suite de testes do `rainbow-api` -- ver
`tests/fixtures/hplc_agilent/PROVENANCIA.md`."""
from __future__ import annotations

import struct
from typing import Tuple

import numpy as np

__all__ = [
    "parse_opus",
    "parse_spc",
    "parse_sp",
    "parse_rmn_bruker",
    "parse_cromatograma_hplc",
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


# --- PerkinElmer .sp -------------------------------------------------------
# Porte adaptado de `specio.plugins.sp` (Copyright 2017 Guillaume Lemaitre,
# BSD-3-Clause -- ver nota de licenca na docstring do modulo). Removido o
# uso de `six` (compatibilidade Python 2, morta) e adequado ao contrato
# `(X, Y)` deste modulo; a logica de travessia de blocos e' preservada tal
# qual validada contra `tests/fixtures/sp/spectra.sp` (arquivo real).

_SP_ASSINATURA = b"PEPE"
_SP_TAMANHO_DESCRICAO = 40
_SP_TAMANHO_CABECALHO_BLOCO = 6  # '<Hi': id do bloco (u16) + tamanho (i32), little-endian
_SP_ID_BLOCO_FAIXA_ONDA = 35698
_SP_ID_BLOCO_N_PONTOS = 35701
_SP_ID_BLOCO_ESPECTRO = 35708


def _sp_info_bloco(dado: bytes) -> Tuple[int, int]:
    if len(dado) != _SP_TAMANHO_CABECALHO_BLOCO:
        raise ValueError(
            f"cabecalho de bloco truncado (esperado "
            f"{_SP_TAMANHO_CABECALHO_BLOCO} bytes, recebido {len(dado)})")
    return struct.unpack("<Hi", dado)


def _sp_decodificar_faixa_onda(dado: bytes) -> dict:
    var_id = struct.unpack("<H", dado[0:2])[0]
    if var_id != 29981:
        return {}
    minimo, maximo = struct.unpack("<dd", dado[2:18])
    return {"min_wavelength": minimo, "max_wavelength": maximo}


def _sp_decodificar_n_pontos(dado: bytes) -> dict:
    var_id = struct.unpack("<H", dado[0:2])[0]
    if var_id != 29995:
        return {}
    (n_pontos,) = struct.unpack("<I", dado[2:6])
    return {"n_points": n_pontos}


def _sp_decodificar_espectro(dado: bytes):
    var_id = struct.unpack("<H", dado[0:2])[0]
    if var_id != 29974:
        return None
    (tamanho,) = struct.unpack("<I", dado[2:6])
    return np.frombuffer(dado[6:6 + tamanho], dtype=np.float64)


_SP_DECODIFICADORES = {
    _SP_ID_BLOCO_FAIXA_ONDA: _sp_decodificar_faixa_onda,
    _SP_ID_BLOCO_N_PONTOS: _sp_decodificar_n_pontos,
    _SP_ID_BLOCO_ESPECTRO: _sp_decodificar_espectro,
}


def _sp_percorrer_diretorio_de_blocos(conteudo: bytes, inicio: int) -> int:
    """Percorre o diretorio de blocos TLV aninhados do `.sp` ate' o bloco
    terminal (id 122), devolvendo o offset onde o bloco de METADADOS
    (5104, descartado por este modulo -- so' precisamos dos blocos de
    eixo/espectro que vem depois) termina.

    LIMITACAO HONESTA: o ramo que recua um nivel de aninhamento quando o
    proximo id de bloco tem o 2o byte == 117 (`chr(117) == 'u'`, marcador
    do formato original em `specio`) NUNCA foi exercitado pelo unico
    arquivo real disponivel (`spectra.sp`, aninhamento de 1 nivel so') --
    portado tal qual do `specio.plugins.sp` sem validacao propria contra
    um arquivo `.sp` de aninhamento mais profundo."""
    pilha_offsets_finais: list[int] = []
    id_bloco, tamanho_bloco = _sp_info_bloco(
        conteudo[inicio:inicio + _SP_TAMANHO_CABECALHO_BLOCO])
    pos = inicio + _SP_TAMANHO_CABECALHO_BLOCO
    pilha_offsets_finais.append(pos + tamanho_bloco)

    while id_bloco != 122 and pos < len(conteudo) - 2:
        proximo_id = conteudo[pos:pos + 2]
        if proximo_id[1] == 117:
            # Ramo nunca exercitado (ver LIMITACAO HONESTA acima) -- porte
            # tal qual do original (`specio.plugins.sp._read_sp`), que tem
            # a MESMA reatribuicao de lista->escalar aqui. Preservado sem
            # "correcao" para nao mudar comportamento sem prova real.
            pos = pilha_offsets_finais[-1]
            pilha_offsets_finais = pilha_offsets_finais[:-1]
            while pos >= pilha_offsets_finais[-1]:
                pilha_offsets_finais = pilha_offsets_finais[-1]  # type: ignore[assignment]
        else:
            id_bloco, tamanho_bloco = _sp_info_bloco(
                conteudo[pos:pos + _SP_TAMANHO_CABECALHO_BLOCO])
            pos += _SP_TAMANHO_CABECALHO_BLOCO
            pilha_offsets_finais.append(pos + tamanho_bloco)

    return pilha_offsets_finais[1]


def parse_sp(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Le um arquivo binario PerkinElmer `.sp` e retorna `(X, Y)` -- mesmo
    contrato de `dados_io.parse_dx`/`dados_io.parse_spectrum`: eixo
    espectral (comprimento de onda) e intensidade.

    Nao precisa de pacote opcional -- struct parsing puro (`numpy` ja e'
    dependencia base do projeto). Ver nota de adaptacao/atribuicao BSD-3
    na docstring do modulo (portado de `specio.plugins.sp`).

    Levanta `ValueError` se o arquivo nao tiver a assinatura `PEPE` (nao e'
    um `.sp` da PerkinElmer) ou nao tiver os blocos minimos de eixo/n de
    pontos/espectro."""
    with open(filepath, "rb") as fh:
        conteudo = fh.read()

    if conteudo[:4] != _SP_ASSINATURA:
        raise ValueError(
            f"{filepath}: nao reconhecido como arquivo PerkinElmer .sp "
            f"valido (assinatura 'PEPE' ausente)")

    try:
        inicio_pos_metadados = _sp_percorrer_diretorio_de_blocos(
            conteudo, _SP_TAMANHO_DESCRICAO + 4)

        meta: dict = {}
        espectro = None
        pos = inicio_pos_metadados
        while pos < len(conteudo):
            id_bloco, tamanho_bloco = _sp_info_bloco(
                conteudo[pos:pos + _SP_TAMANHO_CABECALHO_BLOCO])
            pos += _SP_TAMANHO_CABECALHO_BLOCO
            decodificador = _SP_DECODIFICADORES.get(id_bloco)
            if decodificador is not None:
                resultado = decodificador(conteudo[pos:pos + tamanho_bloco])
                if isinstance(resultado, dict):
                    meta.update(resultado)
                elif resultado is not None:
                    espectro = resultado
            pos += tamanho_bloco
    except (struct.error, IndexError, ValueError) as e:
        raise ValueError(f"{filepath}: arquivo .sp truncado ou corrompido ({e})") from e

    if espectro is None or "min_wavelength" not in meta or "n_points" not in meta:
        raise ValueError(
            f"{filepath}: blocos obrigatorios de eixo/espectro nao "
            f"encontrados no arquivo .sp")

    X = np.linspace(meta["min_wavelength"], meta["max_wavelength"], meta["n_points"])
    Y = np.asarray(espectro, dtype=float)
    if X.shape != Y.shape or X.size == 0:
        raise ValueError(
            f"{filepath}: eixo/intensidade inconsistentes "
            f"(x={X.shape}, y={Y.shape})")
    return X, Y


def parse_rmn_bruker(pasta: str) -> Tuple[np.ndarray, np.ndarray]:
    """Le o ESPECTRO RMN Bruker JA PROCESSADO (`pdata/<N>/1r`, gravado
    pelo proprio TopSpin apos FFT + correcao de fase) e retorna `(X, Y)`
    -- mesmo contrato de `dados_io.parse_dx`/`dados_io.parse_spectrum`:
    eixo em ppm (decrescente, convencao padrao de RMN -- nao reordenado)
    e intensidade (parte real).

    `pasta` deve apontar DIRETO para o diretorio `pdata/<N>/` (mesmo
    contrato de `nmrglue.bruker.read_pdata`) -- nao para a raiz do
    experimento nem para o diretorio que contem o `fid` bruto.

    ESCOPO DELIBERADO: le so' o espectro JA PROCESSADO pelo instrumento,
    nao o FID bruto (dominio do tempo) -- ver docstring do modulo para o
    raciocinio completo (processar o FID do zero exigiria este modulo
    implementar FFT + fase propria, fora do escopo de um leitor de
    formato).

    Levanta `ValueError` se o diretorio nao existir, nao tiver o binario
    `pdata` esperado, ou faltar algum dos parametros de eixo (`SW_p`,
    `SF`, `OFFSET` em `procs`) necessarios para construir o eixo ppm.

    Requer o pacote opcional `nmrglue` (`pip install
    guaraci-chemometrics[rmn]`) -- import LAZY, so' ao chamar esta
    funcao."""
    try:
        import nmrglue as ng
    except ImportError as e:
        raise ImportError(
            "Pacote opcional 'nmrglue' nao instalado -- leitura de "
            "espectro RMN Bruker indisponivel (pip install "
            "guaraci-chemometrics[rmn])."
        ) from e

    try:
        dic, dados = ng.bruker.read_pdata(pasta)
    except OSError as e:
        raise ValueError(f"{pasta}: nao reconhecido como pdata Bruker valido ({e})") from e

    try:
        udic = ng.bruker.guess_udic(dic, dados)
        uc = ng.fileiobase.uc_from_udic(udic)
        X = np.asarray(uc.ppm_scale(), dtype=float)
    except (KeyError, ZeroDivisionError) as e:
        raise ValueError(
            f"{pasta}: parametros de eixo ppm ausentes/invalidos em "
            f"'procs' ({e})") from e

    Y = np.asarray(dados, dtype=float)
    if X.shape != Y.shape or X.size == 0:
        raise ValueError(
            f"{pasta}: eixo/intensidade inconsistentes "
            f"(x={X.shape}, y={Y.shape})")
    return X, Y


def parse_cromatograma_hplc(pasta: str, detector: str = "UV") -> Tuple[np.ndarray, np.ndarray]:
    """Le um diretorio de cromatograma bruto de fabricante -- Agilent
    (`.D`/`.dx`) ou Waters (`.raw`), formato detectado automaticamente
    pelo sufixo do diretorio -- e retorna `(X, Y)`: tempo de retencao
    (minutos) e intensidade de UM canal do `detector` pedido.

    `detector` seleciona o TIPO de detector (`"UV"` para UV/DAD -- o
    default e o caso de uso mais comum de HPLC --, `"FID"` para GC-FID,
    outros conforme `rainbow.DataDirectory.detectors`); se o diretorio
    tiver MAIS DE UM canal desse tipo (ex.: varios comprimentos de onda
    de UV), retorna so' o PRIMEIRO -- mesma logica de "um espectro por
    arquivo" ja' aplicada a SPC multi-subarquivo acima (ler os demais
    exigiria um contrato de retorno diferente, fora do escopo aqui).

    Levanta `ValueError` se o diretorio nao for reconhecido pelo
    `rainbow-api` (formato invalido/nao suportado) ou se o `detector`
    pedido nao existir no diretorio.

    Requer o pacote opcional `rainbow-api` (`pip install
    guaraci-chemometrics[hplc]`) -- import LAZY, so' ao chamar esta
    funcao."""
    try:
        import rainbow as rb
    except ImportError as e:
        raise ImportError(
            "Pacote opcional 'rainbow-api' nao instalado -- leitura de "
            "cromatograma HPLC/GC de fabricante indisponivel (pip "
            "install guaraci-chemometrics[hplc])."
        ) from e

    try:
        diretorio = rb.read(pasta)
    except Exception as e:
        raise ValueError(
            f"{pasta}: nao reconhecido como diretorio Agilent/Waters "
            f"valido ({e})") from e

    arquivos = diretorio.by_detector.get(detector)
    if not arquivos:
        disponiveis = sorted(diretorio.detectors)
        raise ValueError(
            f"{pasta}: detector '{detector}' nao encontrado -- "
            f"disponiveis: {disponiveis}")

    arquivo = arquivos[0]
    X = np.asarray(arquivo.xlabels, dtype=float)
    Y = np.asarray(arquivo.data[:, 0], dtype=float)
    if X.shape != Y.shape or X.size == 0:
        raise ValueError(
            f"{pasta}: eixo/intensidade inconsistentes "
            f"(x={X.shape}, y={Y.shape})")
    return X, Y
