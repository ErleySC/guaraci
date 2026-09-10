# -*- coding: utf-8 -*-
"""eem_io.py -- Parser do dataset publico Zenodo `10.5281/zenodo.19755088`
("EEM fluorescence spectral dataset of olive-oil adulteration samples
across five adulterant systems"), Passo 149 (Fase C da auditoria das 11
tecnicas analiticas, 2026-09-04).

Formato confirmado por leitura DIRETA do dataset baixado (nao suposto):
cada amostra e' um arquivo `<n>_RM.dat`, texto separado por TABULACAO,
organizado em pastas
`data/<marca+adulterante em chines>/<rodada em chines>/<razao azeite:
adulterante>/0_RM.dat`. O `.dat` tem 3 linhas de cabecalho (nome da
tabela, lista de 35 comprimentos de onda de EXCITACAO, unidade/fator de
normalizacao) seguidas de 270 linhas de dado -- cada linha comeca com o
comprimento de onda de EMISSAO e traz 35 intensidades (uma por
excitacao). Verificado em 3 amostras de marcas/rodadas/razoes
DIFERENTES: mesma grade exata de excitacao/emissao nas 3 -- condicao
que `eem_multiway.construir_tensor_eem` exige (mesmo instrumento/
protocolo, nao precisa de interpolacao).

ESTE dataset e' o substituto real do EEM "irregular" registrado como
pendencia em `eem_multiway.py` (Mendeley `g6y69g8gwm`, formato bruto de
instrumento com numero de colunas variavel linha-a-linha) -- achado
desta rodada de busca (Passo 149): existe dataset EEM real com formato
MUITO mais regular, cobrindo a mesma lacuna (PARAFAC nunca tinha rodado
contra EEM real, so' contra dado sintetico).

Regra de parsing (nunca falhar silenciosamente nem inventar valor):
`parse_eem_dat` DESCARTA (e conta) qualquer linha de dado que nao tenha
exatamente `1 + n_excitacao` campos numericos parseaveis -- nunca
completa com zero nem interpola. `carregar_dataset_eem_azeite` agrega
esses contadores por amostra e no total, para a taxa de linhas
utilizaveis vs. descartadas poder ser reportada honestamente (medida
real neste dataset: ver `docs/VALIDACAO_PUBLICA.md` secao 2h)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "ParseEEMRelatorio",
    "parse_eem_dat",
    "carregar_dataset_eem_azeite",
]

#: Substring em chines -> tipo de adulterante (ver nomes de pasta reais,
#: verificados por leitura direta em 2026-09-04).
_MAPA_ADULTERANTE: List[Tuple[str, str]] = [
    ("玉米", "milho"),        # 玉米胚芽油 = oleo de germe de milho
    ("菜籽", "canola"),       # 菜籽油 = oleo de canola/colza
    ("花生", "amendoim"),     # 花生油 = oleo de amendoim
    ("大豆", "soja"),         # 大豆油 = oleo de soja
    ("核桃", "noz"),          # 核桃油 = oleo de noz
]

#: Prefixo em chines -> marca de azeite (3 marcas no dataset).
_MAPA_MARCA: List[Tuple[str, str]] = [
    ("伯爵", "Earl"),
    ("欧丽薇兰", "Olivoila"),
    ("鲁花", "Luhua"),
]

_MAPA_RODADA = {"第一轮": 1, "第二轮": 2, "第三轮": 3}
_RE_RATIO = re.compile(r"^\s*(\d+)\s*[:：]\s*(\d+)\s*$")

#: Achado do Passo 149 (formato irregular real, nao hipotetico): a maioria
#: das pastas de amostra tem `0_RM.dat`, mas 16/330 (4,8%) tem so' `0.dat`
#: (sem o sufixo `_RM`) -- MESMO arquivo, nome diferente, confirmado por
#: leitura direta do dataset baixado. Por isso a busca abaixo usa glob
#: (`*.dat`) em vez de um nome fixo -- ver `_localizar_arquivo_dat`.


def _adulterante_de_pasta(nome: str) -> str:
    for chave, rotulo in _MAPA_ADULTERANTE:
        if chave in nome:
            return rotulo
    raise ValueError(
        f"pasta '{nome}' nao contem nenhum token de adulterante conhecido "
        f"({[c for c, _ in _MAPA_ADULTERANTE]})")


def _marca_de_pasta(nome: str) -> str:
    for chave, rotulo in _MAPA_MARCA:
        if nome.startswith(chave):
            return rotulo
    raise ValueError(
        f"pasta '{nome}' nao comeca com nenhuma marca conhecida "
        f"({[c for c, _ in _MAPA_MARCA]})")


@dataclass
class ParseEEMRelatorio:
    """Contagem de linhas de dado de UM arquivo .dat: `n_esperado` e' o
    numero de linhas de emissao que o cabecalho implica (270 nesta
    versao do dataset, mas nunca hardcoded -- lido do arquivo);
    `n_validas`/`n_descartadas` sao medidos linha a linha.
    `motivos_descarte` lista (numero da linha, motivo) para auditoria."""
    n_esperado: int
    n_validas: int
    n_descartadas: int
    motivos_descarte: List[Tuple[int, str]] = field(default_factory=list)

    @property
    def taxa_descarte(self) -> float:
        total = self.n_validas + self.n_descartadas
        return (self.n_descartadas / total) if total > 0 else 0.0


def parse_eem_dat(caminho: "str | Path"
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ParseEEMRelatorio]:
    """Le um arquivo `<n>_RM.dat` (ver docstring do modulo). Retorna
    (excitacao (n_exc,), emissao (n_em,), matriz (n_em, n_exc),
    relatorio). `matriz[i, j]` = intensidade em (emissao[i], excitacao[j]).

    Levanta `ValueError` se o cabecalho (linha 1) nao tiver pelo menos 2
    colunas (nome + >=1 excitacao) -- sem excitacao nenhuma o arquivo
    nao e' um EEM valido, e' um erro estrutural, nao uma linha
    descartavel."""
    linhas = Path(caminho).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(linhas) < 4:
        raise ValueError(f"{caminho}: arquivo com so' {len(linhas)} linhas -- "
                          f"esperado >=4 (3 de cabecalho + >=1 de dado)")

    cabecalho = linhas[0].rstrip("\t").split("\t")
    if len(cabecalho) < 2:
        raise ValueError(f"{caminho}: cabecalho sem colunas de excitacao "
                          f"({cabecalho!r})")
    excitacao = np.array([float(v) for v in cabecalho[1:]], dtype=float)
    n_exc = len(excitacao)

    linhas_dado = linhas[3:]
    emissoes: List[float] = []
    valores: List[List[float]] = []
    n_validas = 0
    n_descartadas = 0
    motivos: List[Tuple[int, str]] = []
    for i, linha in enumerate(linhas_dado):
        bruta = linha.rstrip("\t")
        if not bruta.strip():
            continue   # linha em branco (comum ao final do arquivo) -- nao e'
                       # dado descartado, so' nao ha' nada ali pra' contar.
        campos = bruta.split("\t")
        if len(campos) != 1 + n_exc:
            n_descartadas += 1
            motivos.append((i, f"esperava {1 + n_exc} campos, achou {len(campos)}"))
            continue
        try:
            valores_linha = [float(v) for v in campos]
        except ValueError as e:
            n_descartadas += 1
            motivos.append((i, f"campo nao-numerico: {e}"))
            continue
        emissoes.append(valores_linha[0])
        valores.append(valores_linha[1:])
        n_validas += 1

    if n_validas == 0:
        raise ValueError(f"{caminho}: nenhuma linha de dado valida -- "
                          f"arquivo vazio ou formato mudou completamente")

    matriz = np.asarray(valores, dtype=float)
    emissao = np.asarray(emissoes, dtype=float)
    relatorio = ParseEEMRelatorio(
        n_esperado=n_validas + n_descartadas, n_validas=n_validas,
        n_descartadas=n_descartadas, motivos_descarte=motivos)
    return excitacao, emissao, matriz, relatorio


def _localizar_arquivo_dat(pasta_razao: Path) -> "Path | None":
    """Acha o arquivo `.dat` de uma pasta de amostra -- normalmente
    `0_RM.dat`, mas 16 pastas do dataset real usam so' `0.dat` (ver nota
    acima). Retorna `None` (nunca adivinha) se nao achar exatamente 1
    candidato `.dat`."""
    candidatos = sorted(pasta_razao.glob("*.dat"))
    return candidatos[0] if len(candidatos) == 1 else None


def carregar_dataset_eem_azeite(pasta_data: "str | Path"
                                 ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame,
                                            Dict[str, object]]:
    """Varre `pasta_data` (a subpasta `data/` extraida do zip do Zenodo
    19755088) e devolve:
      - `matrizes`: dict `{id_amostra: matriz_eem (n_exc, n_em)}` --
        formato exigido por `eem_multiway.construir_tensor_eem` (excitacao
        primeiro).
      - `metadados`: DataFrame (index = id_amostra) com `marca`,
        `adulterante`, `rodada`, `razao_azeite`, `razao_adulterante`,
        `fracao_azeite_pct` (0-100).
      - `relatorio`: dict agregado com `n_amostras_ok`, `n_amostras_ignoradas`
        (pastas que nao casaram nenhum padrao conhecido -- logadas, nunca
        silenciosas), `taxa_descarte_linhas_media` (media da taxa de
        linhas descartadas por arquivo, ver `ParseEEMRelatorio`).

    Nunca inventa amostra: uma pasta de razao sem `0_RM.dat`, ou um nome
    de pasta que nao bate com nenhum padrao conhecido (marca/adulterante/
    rodada/razao), e' IGNORADA e contada em `n_amostras_ignoradas` --
    nunca preenchida com valor inventado."""
    raiz = Path(pasta_data)
    matrizes: Dict[str, np.ndarray] = {}
    linhas_meta: List[Dict[str, object]] = []
    n_ignoradas = 0
    taxas_descarte: List[float] = []
    excitacao_ref: "np.ndarray | None" = None
    emissao_ref: "np.ndarray | None" = None

    for pasta_marca_adult in sorted(raiz.iterdir()):
        if not pasta_marca_adult.is_dir():
            continue
        try:
            marca = _marca_de_pasta(pasta_marca_adult.name)
            adulterante = _adulterante_de_pasta(pasta_marca_adult.name)
        except ValueError:
            n_ignoradas += 1
            continue

        for pasta_rodada in sorted(pasta_marca_adult.iterdir()):
            rodada = _MAPA_RODADA.get(pasta_rodada.name)
            if rodada is None:
                n_ignoradas += 1
                continue

            for pasta_razao in sorted(pasta_rodada.iterdir()):
                m = _RE_RATIO.match(pasta_razao.name)
                if not m:
                    n_ignoradas += 1
                    continue
                arquivo = _localizar_arquivo_dat(pasta_razao)
                if arquivo is None:
                    n_ignoradas += 1
                    continue

                a, b = int(m.group(1)), int(m.group(2))
                fracao_azeite = 100.0 * a / (a + b)

                excitacao, emissao, matriz, rel = parse_eem_dat(arquivo)
                taxas_descarte.append(rel.taxa_descarte)

                if excitacao_ref is None:
                    excitacao_ref, emissao_ref = excitacao, emissao
                elif (excitacao.shape != excitacao_ref.shape
                        or emissao.shape != emissao_ref.shape):
                    raise ValueError(
                        f"{arquivo}: grade excitacao/emissao ({excitacao.shape}, "
                        f"{emissao.shape}) diferente da primeira amostra "
                        f"({excitacao_ref.shape}, {emissao_ref.shape}) -- "
                        f"dataset deveria ter grade fixa (mesmo instrumento).")

                id_amostra = f"{marca}_{adulterante}_{a}-{b}_R{rodada}"
                matrizes[id_amostra] = matriz.T   # (n_em,n_exc) -> (n_exc,n_em)
                linhas_meta.append({
                    "id_amostra": id_amostra, "marca": marca,
                    "adulterante": adulterante, "rodada": rodada,
                    "razao_azeite": a, "razao_adulterante": b,
                    "fracao_azeite_pct": fracao_azeite,
                })

    if not matrizes:
        raise ValueError(f"{raiz}: nenhuma amostra EEM valida encontrada -- "
                          f"formato do dataset mudou ou pasta errada")

    metadados = pd.DataFrame(linhas_meta).set_index("id_amostra")
    relatorio: Dict[str, object] = {
        "n_amostras_ok": len(matrizes),
        "n_amostras_ignoradas": n_ignoradas,
        "taxa_descarte_linhas_media": (
            float(np.mean(taxas_descarte)) if taxas_descarte else 0.0),
        "taxa_descarte_linhas_maxima": (
            float(np.max(taxas_descarte)) if taxas_descarte else 0.0),
    }
    return matrizes, metadados, relatorio
