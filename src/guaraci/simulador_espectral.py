"""
simulador_espectral.py -- dados sinteticos com a estrutura de um conjunto
REAL de espectros: lote (dia de medida) -> amostra fisica (`mae_id`) ->
replicas (T1/T2/T3).

O gerador antigo ("classico", mantido para os testes) so' tinha ruido
branco entre replicas e classes separadas por um deslocamento de pico. Nao
havia espalhamento, linha de base, efeito de dia nem variacao de
composicao entre amostras da mesma classe, entao qualquer modelo acertava
tudo e a vantagem da validacao agrupada por `mae_id` nao aparecia.

Modelo (o mesmo nucleo para todas as tecnicas):

1. Mistura de Beer-Lambert: x = sum_j c_j s_j, com espectros puros s_j
   feitos de bandas (gaussianas no NIR/MIR, lorentzianas no Raman/RMN). As
   classes diferem na composicao MEDIA; cada amostra fisica sorteia a sua
   composicao em torno dela (lognormal, componentes correlacionados), e
   as classes se sobrepoem em parte.
2. Adulterante/analito entra pela propria mistura: (1 - f) * matriz +
   f * espectro do adulterante (diluicao real, nao uma banda somada).
3. Espalhamento e linha de base no modelo do MSC/EMSC (Geladi, MacDougall
   & Martens 1985, Appl. Spectrosc. 39:491; Martens & Stark 1991, J.
   Pharm. Biomed. Anal. 9:625): x = a + b * m + d * eixo + e * eixo^2.
   A amostra fisica sorteia (a, b, d, e); cada replica sorteia uma
   variacao MENOR em torno deles (reempacotar a mesma amostra muda o
   espalhamento, mas menos que trocar de amostra -- Guo et al. 2017, Anal.
   Methods 9:4410, sobre por que a validacao tem de separar por amostra).
4. Lote (dia): ganho, deslocamento do eixo em fracao de ponto e curvatura
   de linha de base comuns a todas as amostras medidas naquele dia.
5. Ruido heteroscedastico: sigma = ruido_base + ruido_prop * sqrt(|x|)
   (Wentzell et al. 1997, J. Chemometrics 11:339).
6. Artefatos por tecnica: Raman com fundo de fluorescencia (0,2 a 3 vezes
   o sinal Raman; Cao et al. 2007, J. Raman Spectrosc. 38:1199) e raios
   cosmicos de 2-3 pontos; RMN com linhas estreitas e deslocamento por pH
   por amostra (Henderson-Hasselbalch, como no MetAssimulo: Muncey et al.
   2010, BMC Bioinformatics 11:496).
7. Poucos espectros anomalos (contato ruim/espalhamento grande), como os
   que aparecem num conjunto real.

AS MAGNITUDES SAO ILUSTRATIVAS. A literatura sustenta a ESTRUTURA (mistura
bilinear, espalhamento do tipo MSC/EMSC, ruido que cresce com o sinal,
replicas mais parecidas entre si que amostras diferentes), nao valores
universais. Os valores de cada cenario foram escolhidos para dar um
problema realista (classes com sobreposicao, replicas correlacionadas,
efeito de dia visivel) e podem ser ajustados em `CENARIOS`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["CENARIOS", "CenarioSintetico", "gerar_cenario",
           "decompor_variancia"]


@dataclass(frozen=True)
class CenarioSintetico:
    """Parametros de um cenario. Desvios sao relativos ao maximo do
    espectro medio (que fica perto de 1)."""
    nome: str
    forma_banda: str              # "gauss" | "lorentz"
    largura_banda: Tuple[float, float]   # em unidades do eixo-modelo 4000->400
    n_componentes: int
    variacao_composicao: float    # cv lognormal da composicao entre amostras
    separacao_classes: float      # diferenca relativa de composicao media entre classes
    espalhamento_amostra: float   # dp de log(b) entre amostras fisicas
    espalhamento_replica: float   # dp de log(b) entre replicas da mesma amostra
    offset_amostra: float
    offset_replica: float
    inclinacao_amostra: float
    curvatura_amostra: float
    n_lotes: int
    ganho_lote: float
    deslocamento_lote_pontos: float
    curvatura_lote: float
    ruido_base: float
    ruido_prop: float
    frac_anomalos: float
    fluorescencia: Tuple[float, float] = (0.0, 0.0)   # razao F/S min, max
    raios_cosmicos_por_espectro: float = 0.0
    desloc_ph_pontos: float = 0.0
    intensidade_analito: float = 0.15   # sinal do analito a 40 % (fracao do maximo)
    # bandas proprias de cada amostra fisica (umidade, traco de contaminante,
    # origem), fora dos componentes comuns: e' o que um modelo "decora"
    # quando replicas da mesma amostra caem no treino e na validacao
    assinatura_amostra: float = 0.08


# Calibracao dos cenarios (2026-10-15, 3 sementes, PLS-DA com o melhor n de
# VLs, pre-processamento tipico de cada tecnica): balanced accuracy em CV
# AGRUPADA por amostra ~0,85-0,89, em CV ALEATORIA (vazando replicas) 3-6
# pontos acima; R2cv do teor ~0,8-0,9. No Raman, 5 % dos espectros com raio
# cosmico derrubam o R2cv de 0,80 para 0,55 sem tratamento; com
# `remover_raios_cosmicos: true` (Whitaker & Hayes 2018) ele volta ao nivel
# do dado limpo (tests/test_despike_raman.py).
CENARIOS: Dict[str, CenarioSintetico] = {
    # NIR/MIR por reflectancia difusa de amostra solida: bandas largas e
    # sobrepostas, espalhamento forte, efeito de dia moderado.
    "nir": CenarioSintetico(
        nome="nir", forma_banda="gauss", largura_banda=(40.0, 140.0),
        n_componentes=5, variacao_composicao=0.12, separacao_classes=0.20,
        espalhamento_amostra=0.08, espalhamento_replica=0.025,
        offset_amostra=0.04, offset_replica=0.012,
        inclinacao_amostra=0.03, curvatura_amostra=0.015,
        n_lotes=3, ganho_lote=0.02, deslocamento_lote_pontos=0.4,
        curvatura_lote=0.01, ruido_base=0.002, ruido_prop=0.004,
        frac_anomalos=0.02),
    # Raman: bandas estreitas, fundo de fluorescencia que varia por
    # amostra e domina o sinal, raios cosmicos.
    "raman": CenarioSintetico(
        nome="raman", forma_banda="lorentz", largura_banda=(6.0, 18.0),
        n_componentes=5, variacao_composicao=0.12, separacao_classes=0.20,
        espalhamento_amostra=0.10, espalhamento_replica=0.04,
        offset_amostra=0.01, offset_replica=0.005,
        inclinacao_amostra=0.01, curvatura_amostra=0.005,
        n_lotes=3, ganho_lote=0.05, deslocamento_lote_pontos=0.3,
        curvatura_lote=0.01, ruido_base=0.004, ruido_prop=0.012,
        frac_anomalos=0.02, fluorescencia=(0.2, 3.0),
        raios_cosmicos_por_espectro=0.05),
    # 1H-RMN de metabolomica: linhas muito estreitas, deslocamento por pH
    # entre amostras (nao entre replicas), pouca linha de base.
    "rmn": CenarioSintetico(
        nome="rmn", forma_banda="lorentz", largura_banda=(2.0, 5.0),
        n_componentes=8, variacao_composicao=0.20, separacao_classes=0.20,
        espalhamento_amostra=0.15, espalhamento_replica=0.02,
        offset_amostra=0.002, offset_replica=0.001,
        inclinacao_amostra=0.002, curvatura_amostra=0.001,
        n_lotes=2, ganho_lote=0.03, deslocamento_lote_pontos=0.2,
        curvatura_lote=0.001, ruido_base=0.003, ruido_prop=0.002,
        frac_anomalos=0.01, desloc_ph_pontos=0.5),
}


def _bandas(eixo: np.ndarray, centros: np.ndarray, larguras: np.ndarray,
            alturas: np.ndarray, forma: str) -> np.ndarray:
    d = (eixo[None, :] - centros[:, None]) / larguras[:, None]
    perfil = np.exp(-0.5 * d ** 2) if forma == "gauss" else 1.0 / (1.0 + d ** 2)
    return (alturas[:, None] * perfil).sum(axis=0)


def _espectro_puro(rng: np.random.Generator, eixo: np.ndarray,
                   cen: CenarioSintetico, n_bandas: int,
                   centros_fixos: Sequence[float] = ()) -> np.ndarray:
    lo, hi = float(eixo.min()), float(eixo.max())
    centros = np.r_[np.asarray(centros_fixos, dtype=float),
                    rng.uniform(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo),
                                max(0, n_bandas - len(centros_fixos)))]
    larguras = rng.uniform(*cen.largura_banda, centros.size)
    alturas = rng.uniform(0.3, 1.0, centros.size)
    s = _bandas(eixo, centros, larguras, alturas, cen.forma_banda)
    return s / s.max()


def _deslocar(X: np.ndarray, pontos: np.ndarray) -> np.ndarray:
    """Desloca cada linha de X por uma fracao de ponto (interpolacao
    linear, bordas repetidas)."""
    idx = np.arange(X.shape[1], dtype=float)
    out = np.empty_like(X)
    for i, d in enumerate(np.broadcast_to(pontos, (X.shape[0],))):
        out[i] = np.interp(idx - d, idx, X[i])
    return out


def gerar_cenario(cen: CenarioSintetico, *, n_pontos: int,
                  n_por_classe: int, n_replicas: int,
                  adulterantes: Sequence[str] = (), seed: int = 42,
                  classes: Sequence[str] = ("Esp_A", "Esp_B", "Esp_C"),
                  codigos: Optional[Dict[str, str]] = None,
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                             np.ndarray, np.ndarray]:
    """Gera (eixo, X, rotulos, conc, mae_id, lote).

    `mae_id` segue o formato que o resto do GUARACI interpreta: 'ESA-00'
    (gradiente de teor sem adulterante), 'ESA-P00' (puro) e 'ESA-S05.00'
    (adulterante S a 5 %). O eixo devolvido e' o eixo-modelo 4000 -> 400;
    quem chama remapeia para a faixa configurada.
    """
    rng = np.random.default_rng(seed)
    codigos = codigos or {c: "ES" + chr(ord("A") + i) for i, c in enumerate(classes)}
    eixo = np.linspace(4000.0, 400.0, int(n_pontos))
    z = (eixo - float(np.mean(eixo))) / float(np.ptp(eixo)) * 2.0   # -1..1

    k = cen.n_componentes
    puros = np.stack([_espectro_puro(rng, eixo, cen, int(rng.integers(3, 7)))
                      for _ in range(k)])
    # Componente "analito" (gradiente de teor sem adulterantes): bandas
    # sobrepostas as da matriz, para o PLS ter de separar de verdade.
    analito = _espectro_puro(rng, eixo, cen, 3)
    puros_adult = {a: _espectro_puro(rng, eixo, cen, int(rng.integers(2, 5)))
                   for a in adulterantes}

    # composicao media por classe: base comum + perturbacao por classe
    base = rng.uniform(0.5, 1.5, k)
    medias = {c: base * np.exp(rng.normal(0.0, cen.separacao_classes, k))
              for c in classes}
    # correlacao entre componentes (fechamento de massa, origem comum)
    A = rng.normal(size=(k, k)) * 0.5 + np.eye(k)
    corr = A @ A.T
    d = np.sqrt(np.diag(corr))
    chol = np.linalg.cholesky(corr / np.outer(d, d))

    linhas: List[Tuple[str, str, float, str]] = []   # classe, mae, teor, adulterante
    for c in classes:
        cod = codigos[c]
        if adulterantes:
            for i in range(3):
                linhas.append((c, f"{cod}-P{i:02d}", 0.0, ""))
            n_niveis = max(3, n_por_classe // 2)
            for a in adulterantes:
                for teor in np.linspace(5, 40, n_niveis):
                    linhas.append((c, f"{cod}-{a}{teor:.2f}", float(teor), a))
        else:
            for i, teor in enumerate(np.linspace(0, 40, n_por_classe)):
                linhas.append((c, f"{cod}-{i:02d}", float(teor), ""))

    n_am = len(linhas)
    lotes_am = rng.permutation(np.arange(n_am) % max(1, cen.n_lotes))
    ganho_lote = np.exp(rng.normal(0, cen.ganho_lote, cen.n_lotes))
    desl_lote = rng.normal(0, cen.deslocamento_lote_pontos, cen.n_lotes)
    curv_lote = rng.normal(0, cen.curvatura_lote, cen.n_lotes)

    X_am = np.empty((n_am, eixo.size))
    for i, (c, _mae, teor_i, a) in enumerate(linhas):
        comp = medias[c] * np.exp(cen.variacao_composicao * (chol @ rng.normal(size=k)))
        matriz = comp @ puros / k
        if a:
            f = teor_i / 100.0
            x = (1.0 - f) * matriz + f * puros_adult[a] * matriz.max()
        else:
            x = matriz + (teor_i / 40.0) * cen.intensidade_analito * analito * matriz.max()
        if cen.assinatura_amostra > 0:
            extra = _espectro_puro(rng, eixo, cen, int(rng.integers(1, 4)))
            x = x + cen.assinatura_amostra * rng.uniform(0.5, 1.5) * extra * matriz.max()
        if cen.desloc_ph_pontos:
            x = _deslocar(x[None, :], np.array([rng.normal(0, cen.desloc_ph_pontos)]))[0]
        X_am[i] = x
    escala = float(np.max(np.mean(np.abs(X_am), axis=0))) or 1.0
    X_am /= escala

    # parametros de espalhamento/linha de base por amostra fisica
    b_am = np.exp(rng.normal(0, cen.espalhamento_amostra, n_am))
    a_am = rng.normal(0, cen.offset_amostra, n_am)
    d_am = rng.normal(0, cen.inclinacao_amostra, n_am)
    e_am = rng.normal(0, cen.curvatura_amostra, n_am)
    fluo_am = (rng.uniform(*cen.fluorescencia, n_am)
               if cen.fluorescencia[1] > 0 else np.zeros(n_am))
    forma_fluo = np.exp(-(z + 1.0) * rng.uniform(0.3, 1.0)) if fluo_am.any() else None

    X_list, rot, conc, mae, lote = [], [], [], [], []
    for i, (c, mae_i, teor_r, _a) in enumerate(linhas):
        L = int(lotes_am[i])
        for _ in range(max(1, int(n_replicas))):
            b = b_am[i] * np.exp(rng.normal(0, cen.espalhamento_replica))
            a = a_am[i] + rng.normal(0, cen.offset_replica)
            x = (a + b * X_am[i] + d_am[i] * z + (e_am[i] + curv_lote[L]) * z ** 2)
            if forma_fluo is not None:
                x = x + fluo_am[i] * b * forma_fluo * np.exp(rng.normal(0, 0.05))
            X_list.append(ganho_lote[L] * x)
            rot.append(c); conc.append(teor_r); mae.append(mae_i); lote.append(L)
    X = np.asarray(X_list)
    X = _deslocar(X, desl_lote[np.asarray(lote)])

    sigma = cen.ruido_base + cen.ruido_prop * np.sqrt(np.abs(X))
    X = X + rng.normal(size=X.shape) * sigma

    if cen.raios_cosmicos_por_espectro > 0:
        n_raios = np.asarray(rng.poisson(cen.raios_cosmicos_por_espectro, X.shape[0]))
        for i_esp in np.flatnonzero(n_raios):
            for _ in range(int(n_raios[i_esp])):
                j = int(rng.integers(0, X.shape[1] - 3))
                X[i_esp, j:j + int(rng.integers(2, 4))] += rng.uniform(1.0, 5.0)

    n_anom = int(round(cen.frac_anomalos * X.shape[0]))
    for i_anom in rng.choice(X.shape[0], n_anom, replace=False):
        # contato ruim / apresentacao errada: espalhamento e inclinacao grandes
        X[i_anom] = X[i_anom] * rng.uniform(1.3, 1.8) + rng.normal(0, 0.15) * z + 0.1

    return (eixo, X, np.asarray(rot, dtype=str), np.asarray(conc, dtype=float),
            np.asarray(mae, dtype=str), np.asarray(lote, dtype=int))


def decompor_variancia(X: np.ndarray, grupos: np.ndarray) -> Dict[str, float]:
    """Variancia total dividida em ENTRE grupos e DENTRO deles (soma de
    quadrados por variavel, somada no espectro). Em dado real de replicas,
    `frac_dentro` fica bem abaixo de 1 mas longe de 0: e' o que faz a
    validacao por amostra fisica importar."""
    X = np.asarray(X, dtype=float)
    g = np.asarray(grupos)
    media = X.mean(axis=0)
    ss_tot = float(((X - media) ** 2).sum())
    ss_dentro = 0.0
    for u in np.unique(g):
        Xg = X[g == u]
        ss_dentro += float(((Xg - Xg.mean(axis=0)) ** 2).sum())
    return {"ss_total": ss_tot, "ss_dentro": ss_dentro,
            "frac_dentro": ss_dentro / ss_tot if ss_tot > 0 else float("nan")}


def cenario(nome: str, **ajustes) -> CenarioSintetico:
    """Cenario por nome, com ajustes opcionais de campos."""
    try:
        base = CENARIOS[nome]
    except KeyError:
        raise ValueError(f"cenario sintetico desconhecido: {nome!r} "
                         f"(opcoes: classico, {', '.join(CENARIOS)})") from None
    return replace(base, **ajustes) if ajustes else base
