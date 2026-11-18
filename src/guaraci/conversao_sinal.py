# -*- coding: utf-8 -*-
"""conversao_sinal.py -- converte o sinal bruto do instrumento em absorbancia (ou pseudo-absorbancia)
DENTRO do Guaraci, antes de qualquer outra etapa (corte de faixa, saturacao, pre-processamento).

`sinal_entrada` (Config.signal_type) diz o que os arquivos trazem:
  - absorbancia     -- nada a fazer (padrao; comportamento anterior).
  - transmitancia   -- A = -log10(T).
  - reflectancia    -- A = log10(1/R) (pseudo-absorbancia da reflectancia difusa, padrao em NIR).
  - kubelka_munk    -- f(R) = (1 - R)^2 / (2R) (reflectancia difusa de po/solido diluido).
  - feixe_simples   -- intensidade bruta I contra a referencia I0 (fundo/branco): A = -log10(I / I0).
                       I0 vem de `sinal_referencia`: um ARQUIVO de espectro (.dx, .csv, .txt com x,y)
                       ou o NOME DE UMA CLASSE dos dados cujos espectros sao o branco (a media deles e'
                       I0; essas amostras saem do estudo).
T e R aceitam fracao (0-1) ou porcentagem (0-100): maximo acima de 1,5 e' lido como % e dividido por
100, com aviso. Valor <= 0 nao tem logaritmo: ate' 1 % dos pontos vira o menor valor positivo do
espectro (aviso com a contagem); acima disso a conversao e' recusada -- quase sempre o tipo de sinal
esta errado ou o detector saturou.

O que foi feito vai para `info` (log, resumo, pacote do modelo); a predicao reaplica a MESMA conversao
nos espectros novos (`converter_pelo_pacote`). ASCII por convencao do projeto.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["SINAIS", "converter", "converter_pelo_pacote", "info_para_pacote"]

SINAIS: Tuple[str, ...] = ("absorbancia", "transmitancia", "reflectancia", "kubelka_munk", "feixe_simples")
_PCT_LIMIAR = 1.5
LIMITE_NAO_POSITIVOS = 0.01


def _fracao(X: np.ndarray, nome: str, avisos: List[str]) -> Tuple[np.ndarray, bool]:
    finito = X[np.isfinite(X)]
    if finito.size and float(np.max(finito)) > _PCT_LIMIAR:
        if float(np.max(finito)) <= 10.0:
            raise ValueError(
                f"Conversao de sinal ({nome}): o maior valor e' {float(np.max(finito)):.3g} -- alto demais para "
                f"fracao (0-1) e baixo demais para porcentagem (0-100). Os arquivos parecem estar em "
                f"absorbancia: use sinal_entrada: absorbancia.")
        avisos.append(f"{nome}: valores ate' {float(np.max(finito)):.3g}, lidos como porcentagem (divididos por 100).")
        return X / 100.0, True
    return X, False


def _positivo(X: np.ndarray, nome: str, avisos: List[str]) -> np.ndarray:
    nao_pos = np.isfinite(X) & (X <= 0)
    frac = float(np.mean(nao_pos)) if X.size else 0.0
    if frac > LIMITE_NAO_POSITIVOS:
        raise ValueError(
            f"Conversao de sinal ({nome}): {frac:.1%} dos valores sao <= 0 e nao tem logaritmo "
            f"(limite {LIMITE_NAO_POSITIVOS:.0%}). Confira sinal_entrada: os arquivos ja' podem estar em "
            f"absorbancia (use sinal_entrada: absorbancia), ou o detector saturou nessas regioes.")
    if frac > 0:
        X = X.copy()
        for i in range(X.shape[0]):
            linha = X[i]
            pos = linha[np.isfinite(linha) & (linha > 0)]
            piso = float(pos.min()) if pos.size else 1e-6
            linha[np.isfinite(linha) & (linha <= 0)] = piso
        avisos.append(f"{nome}: {int(nao_pos.sum())} valor(es) <= 0 ({frac:.2%}) trocados pelo menor valor "
                      f"positivo do proprio espectro antes do logaritmo.")
    return X


def _referencia_no_eixo(ref_wn: np.ndarray, ref_y: np.ndarray, wn: np.ndarray) -> np.ndarray:
    ordem = np.argsort(ref_wn)
    rx, ry = np.asarray(ref_wn, dtype=float)[ordem], np.asarray(ref_y, dtype=float)[ordem]
    lo, hi = float(np.min(wn)), float(np.max(wn))
    passo = float(np.median(np.abs(np.diff(rx)))) if rx.size > 1 else 0.0
    if rx[0] > lo + passo or rx[-1] < hi - passo:
        raise ValueError(
            f"A referencia I0 cobre {rx[0]:g}-{rx[-1]:g}, mas os espectros vao de {lo:g} a {hi:g}: "
            f"meca o branco na mesma faixa (sem extrapolar).")
    return np.interp(np.asarray(wn, dtype=float), rx, ry)


def converter(X: np.ndarray, sinal: str, wn: Optional[np.ndarray] = None,
              referencia: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              origem_referencia: str = "") -> Tuple[np.ndarray, Dict[str, Any]]:
    """`(X_convertido, info)`. `referencia` = (eixo, I0) so' para `feixe_simples`."""
    sinal = (sinal or "absorbancia").strip()
    if sinal not in SINAIS:
        raise ValueError(f"sinal_entrada '{sinal}' invalido; use {list(SINAIS)}.")
    X = np.asarray(X, dtype=float)
    avisos: List[str] = []
    info: Dict[str, Any] = {"sinal": sinal, "percentual": False, "avisos": avisos}
    if sinal == "absorbancia":
        return X, info
    if sinal in ("transmitancia", "reflectancia"):
        Xf, pct = _fracao(X, sinal, avisos)
        info["percentual"] = pct
        acima = float(np.mean(Xf[np.isfinite(Xf)] > 1.05)) if Xf.size else 0.0
        if sinal == "transmitancia" and acima > LIMITE_NAO_POSITIVOS:
            raise ValueError(
                f"Conversao de sinal (transmitancia): {acima:.1%} dos valores passam de 1 (100 %), o que "
                f"transmitancia nao faz. Os arquivos provavelmente ja' estao em absorbancia: use "
                f"sinal_entrada: absorbancia.")
        if sinal == "reflectancia" and acima > 0.05:
            avisos.append(f"reflectancia: {acima:.1%} dos valores passam de 1 (100 %): confira a referencia "
                          f"branca ou se o sinal ja' esta' em absorbancia.")
        Xf = _positivo(Xf, sinal, avisos)
        info["formula"] = "A = -log10(T)" if sinal == "transmitancia" else "A = log10(1/R)"
        return -np.log10(Xf), info
    if sinal == "kubelka_munk":
        Xf, pct = _fracao(X, sinal, avisos)
        info["percentual"] = pct
        Xf = _positivo(Xf, sinal, avisos)
        acima = float(np.mean(Xf[np.isfinite(Xf)] > 1.0)) if Xf.size else 0.0
        if acima > 0.01:
            avisos.append(f"kubelka_munk: {acima:.1%} dos valores de R passam de 1 (100 %): confira se o "
                          f"sinal e' mesmo reflectancia.")
        info["formula"] = "f(R) = (1 - R)^2 / (2R)"
        return (1.0 - Xf) ** 2 / (2.0 * Xf), info
    # feixe simples
    if referencia is None or wn is None:
        raise ValueError(
            "sinal_entrada: feixe_simples precisa da referencia I0 (branco/fundo): aponte em sinal_referencia "
            "um arquivo de espectro (.dx, .csv ou .txt com x,y) ou o nome da classe dos espectros de branco.")
    I0 = _referencia_no_eixo(referencia[0], referencia[1], np.asarray(wn, dtype=float))
    if np.any(~np.isfinite(I0)) or np.any(I0 <= 0):
        raise ValueError("A referencia I0 tem valores <= 0 ou invalidos: nao serve de branco para A = -log10(I/I0).")
    Xr = _positivo(X / I0[None, :], "feixe_simples", avisos)
    info.update({"formula": "A = -log10(I / I0)", "origem_referencia": origem_referencia,
                 "referencia_eixo": [float(v) for v in np.asarray(wn, dtype=float)],
                 "referencia_I0": [float(v) for v in I0]})
    return -np.log10(Xr), info


def info_para_pacote(info: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """O que o pacote do modelo precisa para reaplicar a conversao (None = absorbancia/nada)."""
    if not info or info.get("sinal", "absorbancia") == "absorbancia":
        return None
    out = {"sinal": info["sinal"], "percentual": bool(info.get("percentual"))}
    if info["sinal"] == "feixe_simples":
        out["referencia_eixo"] = list(info["referencia_eixo"])
        out["referencia_I0"] = list(info["referencia_I0"])
        out["origem_referencia"] = info.get("origem_referencia", "")
    return out


def converter_pelo_pacote(pkg: Dict[str, Any], X_new: np.ndarray, wn_new: Optional[np.ndarray]) -> np.ndarray:
    """Reaplica nos espectros novos a conversao do treino (pacote sem a chave: nada muda). Em feixe simples
    usa o I0 do treino; o certo e' medir as amostras novas na mesma sessao do branco."""
    sinal_pkg = pkg.get("sinal_entrada") if isinstance(pkg, dict) else None
    if not sinal_pkg:
        return np.asarray(X_new, dtype=float)
    sinal = str(sinal_pkg.get("sinal", "absorbancia"))
    X = np.asarray(X_new, dtype=float)
    if sinal in ("transmitancia", "reflectancia", "kubelka_munk") and sinal_pkg.get("percentual"):
        finito = X[np.isfinite(X)]
        if finito.size and float(np.max(finito)) <= _PCT_LIMIAR:
            raise ValueError(
                f"O modelo foi treinado com {sinal} em porcentagem (0-100), mas os espectros novos parecem "
                f"fracao (0-1). Exporte as amostras novas na mesma escala do treino.")
    ref = None
    if sinal == "feixe_simples":
        ref = (np.asarray(sinal_pkg["referencia_eixo"], dtype=float),
               np.asarray(sinal_pkg["referencia_I0"], dtype=float))
    Xc, _info = converter(X, sinal, wn_new, ref, str(sinal_pkg.get("origem_referencia", "")))
    return Xc


def ler_espectro_referencia(caminho: str) -> Tuple[np.ndarray, np.ndarray]:
    """(eixo, I0) de um arquivo .dx/.jdx ou texto x,y (.csv/.txt/.spectrum)."""
    from guaraci import dados_io
    ext = os.path.splitext(caminho)[1].lower()
    if ext in (".dx", ".jdx"):
        x, y = dados_io.parse_dx(caminho)
    else:
        x, y = dados_io.parse_spectrum(caminho)
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    if x.size < 2 or x.size != y.size:
        raise ValueError(f"{os.path.basename(caminho)}: nao foi possivel ler um espectro x,y para servir de I0.")
    return x, y
