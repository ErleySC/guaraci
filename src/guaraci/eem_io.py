# -*- coding: utf-8 -*-
"""eem_io.py -- leitor de matrizes de excitacao-emissao (EEM) exportadas por fluorimetros em texto separado por TABULACAO.

Layout (o mesmo de exportadores de fluorimetro comerciais): 3 linhas de cabecalho (rotulo + lista de comprimentos de onda de EXCITACAO;
unidade; fator de normalizacao) seguidas de uma linha por comprimento de onda de EMISSAO (`emissao<TAB>intensidades...`, uma por
excitacao). A ordem dos eixos (crescente ou decrescente) e a grade (numero de excitacoes e de emissoes) sao lidas do arquivo, nunca
assumidas.

Regra de parsing (nunca falhar silenciosamente nem inventar valor): `parse_eem_dat` DESCARTA (e conta) qualquer linha de dado que nao
tenha exatamente `1 + n_excitacao` campos numericos parseaveis -- nunca completa com zero nem interpola. A taxa de linhas
utilizaveis vs. descartadas fica no `ParseEEMRelatorio`, para poder ser reportada honestamente.

Layouts de pasta especificos de terceiros (pastas, nomes, rodadas) nao sao tratados pelo produto.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

__all__ = [
    "ParseEEMRelatorio",
    "parse_eem_dat",
    "MODOS_ESPALHAMENTO",
    "LARGURA_ESPALHAMENTO_PADRAO_NM",
    "RAMAN_AGUA_CM1",
    "raman_agua_emissao",
    "mascara_espalhamento",
    "tratar_espalhamento",
    "forca_do_espalhamento",
    "larguras_do_config",
]


#: Achado do Passo 149 (formato irregular real, nao hipotetico): a maioria
#: das pastas de amostra tem `0_RM.dat`, mas 16/330 (4,8%) tem so' `0.dat`
#: (sem o sufixo `_RM`) -- MESMO arquivo, nome diferente, confirmado por
#: leitura direta do dataset baixado. Por isso a busca abaixo usa glob
#: (`*.dat`) em vez de um nome fixo -- ver `_localizar_arquivo_dat`.


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




# ---------------------------------------------------------------------------
# Espalhamento Rayleigh / Raman (primeira e segunda ordem)
# ---------------------------------------------------------------------------
#: Modos do tratamento (config `eem_espalhamento`). `nenhum` nao toca no dado.
MODOS_ESPALHAMENTO = ("nenhum", "nan", "zero", "interp")
#: Deslocamento Raman da agua (O-H, estiramento), em cm-1: emissao Raman = 1/(1/ex - 3400e-7) (nm).
RAMAN_AGUA_CM1 = 3400.0
#: Largura TOTAL (nm) da banda removida em torno de cada linha (+-metade para cada lado), quando o usuario nao diz.
LARGURA_ESPALHAMENTO_PADRAO_NM = 30.0
_ORDENS = ("rayleigh_1", "rayleigh_2", "raman_1", "raman_2")


def raman_agua_emissao(excitacao, deslocamento_cm1: float = RAMAN_AGUA_CM1) -> np.ndarray:
    """Comprimento de onda (nm) da banda Raman da agua (1a ordem) para cada excitacao (nm)."""
    ex = np.asarray(excitacao, dtype=float)
    return 1.0 / (1.0 / ex - float(deslocamento_cm1) * 1e-7)


def larguras_do_config(valor) -> Tuple[float, float, float, float]:
    """Larguras totais (nm) por ordem, na ordem (Rayleigh 1a, Rayleigh 2a, Raman 1a, Raman 2a).
    `valor` vem do config (`eem_espalhamento_largura_nm`): vazio = padrao para todas; 1 numero = o mesmo para todas;
    n numeros = um por ordem, e a ultima largura vale para as ordens que faltarem."""
    if valor is None or isinstance(valor, str) and not valor.strip():
        itens: List[float] = []
    elif isinstance(valor, (int, float)):
        itens = [float(valor)]
    elif isinstance(valor, str):
        return larguras_do_config([x for x in valor.split(",")])
    else:
        try:
            itens = [float(str(x).strip()) for x in valor if str(x).strip()]
        except ValueError as e:
            raise ValueError(f"eem_espalhamento_largura_nm: use numeros em nm ({e})") from None
    if any(not np.isfinite(x) or x <= 0 for x in itens):
        raise ValueError("eem_espalhamento_largura_nm: as larguras precisam ser numeros > 0 (nm)")
    if not itens:
        itens = [LARGURA_ESPALHAMENTO_PADRAO_NM]
    while len(itens) < len(_ORDENS):
        itens.append(itens[-1])
    return tuple(itens[:len(_ORDENS)])  # type: ignore[return-value]


def mascara_espalhamento(excitacao, emissao, larguras_nm=None, *, raman: bool = False,
                         raman_cm1: float = RAMAN_AGUA_CM1) -> Tuple[np.ndarray, np.ndarray]:
    """Mascaras booleanas `(n_emissao, n_excitacao)`: (bandas de espalhamento, regiao sem sinal).

    As bandas sao Rayleigh de 1a ordem (em = ex) e de 2a ordem (em = 2 ex) e, se `raman`, o Raman da agua de 1a e 2a ordem
    (em = Raman(ex) e 2 Raman(ex)), cada uma com a sua largura total em nm. A regiao sem sinal e' o triangulo `em` abaixo da banda
    de Rayleigh de 1a ordem: nao ha' emissao de comprimento de onda menor que a excitacao (lei de Stokes)."""
    ex = np.asarray(excitacao, dtype=float)[None, :]
    em = np.asarray(emissao, dtype=float)[:, None]
    w1, w2, w3, w4 = larguras_do_config(larguras_nm)
    bandas = np.abs(em - ex) <= w1 / 2.0
    bandas |= np.abs(em - 2.0 * ex) <= w2 / 2.0
    if raman:
        ram = raman_agua_emissao(ex, raman_cm1)
        bandas |= np.abs(em - ram) <= w3 / 2.0
        bandas |= np.abs(em - 2.0 * ram) <= w4 / 2.0
    abaixo = (em < ex - w1 / 2.0) & ~bandas
    return bandas, abaixo


def _interpolar_2d(matriz: np.ndarray, validos: np.ndarray, excitacao: np.ndarray, emissao: np.ndarray) -> np.ndarray:
    """Preenche as celulas `~validos` por interpolacao linear 2D (triangulacao de Delaunay, como em Zepp et al., 2004);
    o que ficar fora do casco convexo dos pontos validos usa interpolacao 1D ao longo da emissao e, por fim, o vizinho mais
    proximo. Devolve uma copia."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    out = np.array(matriz, dtype=float)
    faltam = ~validos
    if not faltam.any():
        return out
    if validos.sum() < 4:
        out[faltam] = 0.0
        return out
    EX, EM = np.meshgrid(np.asarray(excitacao, dtype=float), np.asarray(emissao, dtype=float))
    # escalas diferentes dos eixos (nm em ambos, mas grades de passos diferentes): a triangulacao usa nm reais nos dois
    pts = np.column_stack([EX[validos], EM[validos]])
    val = out[validos]
    try:
        f = LinearNDInterpolator(pts, val)
        novo = f(np.column_stack([EX[faltam], EM[faltam]]))
    except Exception:  # noqa: BLE001 -- pontos colineares etc.: cai no vizinho mais proximo
        novo = np.full(int(faltam.sum()), np.nan)
    ruim = ~np.isfinite(novo)
    if ruim.any():
        near = NearestNDInterpolator(pts, val)
        novo[ruim] = near(np.column_stack([EX[faltam][ruim], EM[faltam][ruim]]))
    out[faltam] = novo
    return out


def tratar_espalhamento(matriz, excitacao, emissao, modo: str = "nenhum", larguras_nm=None, *,
                        raman: bool = False, raman_cm1: float = RAMAN_AGUA_CM1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trata o espalhamento de UMA EEM `matriz (n_emissao, n_excitacao)`.

    Devolve `(matriz_tratada, removida, ausente)`, com `removida` e `ausente` booleanas do mesmo formato:
      * `ausente`: celulas NaN que ja vieram no arquivo (convencao de exportadores para a regiao sem sinal / espalhamento).
        Sao sempre dado faltante, em qualquer modo, e na matriz tratada valem 0 (o PARAFAC as ignora pela mascara).
      * `removida`: celulas das bandas de espalhamento e da regiao sem sinal tratadas por este passo (vazia em `nenhum`).
    Modos: `nenhum` (nao toca no dado), `nan` (a regiao removida vira dado faltante; na matriz tratada vale 0 e a mascara a
    identifica), `zero` (vira 0 como dado), `interp` (bandas por interpolacao linear 2D dos vizinhos validos, Zepp et al., 2004;
    a regiao sem sinal vira 0)."""
    if modo not in MODOS_ESPALHAMENTO:
        raise ValueError(f"eem_espalhamento '{modo}' invalido; use {list(MODOS_ESPALHAMENTO)}")
    m = np.array(matriz, dtype=float)
    ausente = ~np.isfinite(m)
    if modo == "nenhum":
        m[ausente] = 0.0
        return m, np.zeros_like(ausente), ausente
    bandas, abaixo = mascara_espalhamento(excitacao, emissao, larguras_nm, raman=raman, raman_cm1=raman_cm1)
    removida = bandas | abaixo
    if modo in ("nan", "zero"):
        m[removida | ausente] = 0.0
    else:  # interp
        validos = ~(bandas | ausente) & ~abaixo
        # a regiao sem sinal entra como 0 conhecido: ancora a interpolacao da borda inferior da banda de Rayleigh
        m[abaixo] = 0.0
        validos = validos | abaixo
        m = _interpolar_2d(np.where(validos, m, 0.0), validos, excitacao, emissao)
        m[abaixo] = 0.0
    return m, removida, ausente


def forca_do_espalhamento(matriz, excitacao, emissao, larguras_nm=None) -> float:
    """Razao tipica (mediana sobre as excitacoes) entre o maximo na banda de Rayleigh de 1a ordem e a mediana da vizinhanca
    imediata acima dela (de 1 a 2 larguras alem da borda da banda). Em EEM sem espalhamento a crista nao existe e a razao fica
    perto de 1; com espalhamento forte e' dezenas. Devolve 0.0 se a banda estiver mascarada (zerada/NaN) ou fora da grade.
    E' so' um sinal de aviso, nao uma metrica de qualidade."""
    m = np.asarray(matriz, dtype=float)
    ex = np.asarray(excitacao, dtype=float)
    em = np.asarray(emissao, dtype=float)
    w = larguras_do_config(larguras_nm)[0]
    razoes = []
    for j, x in enumerate(ex):
        banda = np.abs(em - x) <= w / 2.0
        viz = (em > x + w / 2.0) & (em <= x + 1.5 * w)
        if not banda.any() or viz.sum() < 2:
            continue
        pico = np.nanmax(np.abs(m[banda, j])) if np.isfinite(m[banda, j]).any() else 0.0
        ref = np.nanmedian(np.abs(m[viz, j])) if np.isfinite(m[viz, j]).any() else 0.0
        if ref > 0:
            razoes.append(pico / ref)
    return float(np.median(razoes)) if razoes else 0.0
