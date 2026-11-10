"""saturacao.py -- detecta teto do detector (absorbancia saturada) em espectros.

Contexto: com cubeta de caminho optico longo, bandas de combinacao intensas saturam o
detector (A trava num teto, ex. 3,0). Variavel saturada nao carrega informacao
da amostra e espectro inteiro saturado e' lixo de aquisicao; deixar os dois no
modelo derruba a quantificacao e some do relatorio.

Regra (conservadora; so' age quando ha' PLATO real no teto):
  1. teto = maximo global; so' vale se muitos pontos (>= `MIN_PONTOS_NO_TETO`)
     de varios espectros (>= `MIN_FRACAO_ESPECTROS`) estao no teto -- um pico
     isolado nunca e' saturacao.
  2. variavel saturada: >= `LIMIAR_VARIAVEL` das amostras no teto -> removida.
  3. espectro saturado: depois de tirar as variaveis, >= `LIMIAR_ESPECTRO` das
     variaveis restantes no teto -> removido.
Nao age em dados que nao sao absorbancia espectral (multielementar, imagem...).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

__all__ = ["MODOS_COM_SATURACAO", "detectar_saturacao", "regioes_de_mascara"]

MODOS_COM_SATURACAO = ("dx", "opus", "spc", "sp", "csv")
MIN_VARIAVEIS = 100          # tabela com poucas colunas nao e' espectro
MIN_PONTOS_NO_TETO = 20      # pontos (no total) para considerar que ha' plato
MIN_FRACAO_ESPECTROS = 0.02  # fracao dos espectros que toca o teto
LIMIAR_VARIAVEL = 0.10       # fracao das amostras no teto para tirar a variavel
LIMIAR_ESPECTRO = 0.20       # fracao das variaveis no teto para tirar o espectro
FOLGA_PONTOS = 3             # pontos de cada lado alargando a regiao saturada
TOLERANCIA_REL = 1e-3        # distancia ao teto, relativa a amplitude dos dados


def regioes_de_mascara(wn: np.ndarray, mascara: np.ndarray) -> List[Tuple[float, float]]:
    """Intervalos contiguos (min, max) do eixo onde `mascara` e' True."""
    out: List[Tuple[float, float]] = []
    i, n = 0, len(mascara)
    while i < n:
        if mascara[i]:
            j = i
            while j + 1 < n and mascara[j + 1]:
                j += 1
            a, b = float(wn[i]), float(wn[j])
            out.append((min(a, b), max(a, b)))
            i = j + 1
        else:
            i += 1
    return out


def detectar_saturacao(X: np.ndarray, wn: np.ndarray) -> Dict[str, Any]:
    """Devolve {'ativo', 'teto', 'manter_variaveis', 'manter_espectros', ...}.

    `manter_*` sao mascaras booleanas; com `ativo=False` mantem tudo."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    tudo = {"ativo": False, "teto": None,
            "manter_variaveis": np.ones(p, dtype=bool),
            "manter_espectros": np.ones(n, dtype=bool),
            "variaveis_removidas": 0, "espectros_removidos": 0,
            "regioes": [], "frac_variavel": np.zeros(p)}
    if p < MIN_VARIAVEIS or n < 3 or not np.isfinite(X).any():
        return tudo
    amplitude = float(np.nanmax(X) - np.nanmin(X))
    if amplitude <= 0:
        return tudo
    # Teto = o valor MAIS ALTO que se repete (plato). O maximo global nao serve: um unico
    # espectro com artefato acima do teto real esconderia a saturacao de todos os outros.
    q = TOLERANCIA_REL * amplitude
    cauda = X[np.isfinite(X)]
    cauda = cauda[cauda >= np.nanpercentile(cauda, 90)]
    caixas, contagem = np.unique(np.round(cauda / q), return_counts=True)
    # Plato = caixa com MUITO mais pontos que as vizinhas (valores identicos pilhados no teto);
    # cauda suave de dado sem saturacao nao tem esse pico isolado.
    por_caixa = dict(zip(caixas.tolist(), contagem.tolist()))
    def _pico(c: float, n: int) -> bool:
        viz = [por_caixa.get(c + d, 0) for d in (-3, -2, -1, 1, 2, 3)]
        return n >= MIN_PONTOS_NO_TETO and n >= 5 * max(1.0, float(np.mean(viz))) and n >= 3 * max(viz)
    repetidas = np.array([c for c, n in por_caixa.items() if _pico(c, n)])
    if repetidas.size == 0:
        return tudo
    teto = float(repetidas.max() * q)
    no_teto = np.isfinite(X) & (X >= teto - 1.5 * q) & (X <= teto + 1.5 * q)
    por_espectro = no_teto.sum(axis=1)
    if int(no_teto.sum()) < MIN_PONTOS_NO_TETO:
        return tudo
    if float(np.mean(por_espectro >= 3)) < MIN_FRACAO_ESPECTROS:
        return tudo
    frac_var = no_teto.mean(axis=0)
    removida = frac_var >= LIMIAR_VARIAVEL
    # Vizinhas de uma banda saturada tambem estao contaminadas (o plato tem bordas e o SG usa a
    # janela): alarga a regiao removida em `FOLGA_PONTOS` pontos de cada lado.
    removida = np.convolve(removida.astype(float), np.ones(2 * FOLGA_PONTOS + 1), mode="same") > 0
    manter_var = ~removida
    if manter_var.sum() < 3:           # tudo saturado: nao ha' o que salvar, nao mexe
        return tudo
    frac_esp = no_teto[:, manter_var].mean(axis=1)
    manter_esp = frac_esp < LIMIAR_ESPECTRO
    if manter_esp.sum() < 3:
        return tudo
    return {"ativo": True, "teto": teto,
            "manter_variaveis": manter_var, "manter_espectros": manter_esp,
            "variaveis_removidas": int((~manter_var).sum()),
            "espectros_removidos": int((~manter_esp).sum()),
            "regioes": regioes_de_mascara(np.asarray(wn, dtype=float), ~manter_var),
            "frac_variavel": frac_var}
