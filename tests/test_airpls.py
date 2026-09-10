# -*- coding: utf-8 -*-
"""Testes de AirPLS (Zhang, Chen & Liang 2010) -- Passo 144/145 da
auditoria das 11 tecnicas (2026-09-04). Contra-prova sintetica (espectro
Raman simulado com fluorescencia de fundo tem que ser corrigido) +
portao de aceite contra o dataset publico Raman real (Mendeley
ctgg7k4m5g, arquivo-irmao ja integrado em
test_validacao_publica_mendeley_mir_raman.py)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from guaraci.preprocessamento import AirPLS


def _espectro_raman_sintetico(rng, n_canais=500, n_picos=4, amplitude_fundo=8.0):
    """Espectro Raman sintetico: alguns picos gaussianos estreitos
    (sinal Raman de verdade) somados a uma linha de base LARGA e suave
    (fluorescencia de fundo -- fisicamente muito mais larga que
    qualquer pico Raman, mesma premissa do airPLS/qualquer metodo de
    correcao de baseline)."""
    eixo = np.linspace(0, n_canais, n_canais)
    picos = np.zeros(n_canais)
    centros = rng.uniform(0.15, 0.85, n_picos) * n_canais
    for c in centros:
        largura = rng.uniform(3.0, 6.0)
        altura = rng.uniform(1.0, 3.0)
        picos += altura * np.exp(-0.5 * ((eixo - c) / largura) ** 2)
    # fundo de fluorescencia: soma de 2 gaussianas MUITO largas (centenas
    # de canais) -- forma tipica de fluorescencia de fundo em Raman,
    # muito mais larga que qualquer pico vibracional real.
    fundo = (amplitude_fundo * np.exp(-0.5 * ((eixo - 0.3 * n_canais) / (n_canais * 0.4)) ** 2)
             + 0.5 * amplitude_fundo * np.exp(-0.5 * ((eixo - 0.7 * n_canais) / (n_canais * 0.6)) ** 2))
    ruido = rng.normal(0, 0.03, n_canais)
    return picos, fundo, picos + fundo + ruido


def test_airpls_remove_fundo_largo_preservando_picos_estreitos():
    """Contra-prova exigida pela instrucao: espectro Raman sintetico com
    fluorescencia simulada deve ser corrigido -- a linha de base
    estimada tem que aproximar o FUNDO verdadeiro (nao os picos), e o
    residuo (espectro - baseline) tem que aproximar os PICOS puros."""
    rng = np.random.default_rng(0)
    picos, fundo, observado = _espectro_raman_sintetico(rng)

    corrigido = AirPLS(itermax=20).transform(observado[None, :])[0]

    # O residuo apos correcao deve estar MUITO mais proximo dos picos
    # puros do que o espectro bruto estava (o fundo dominava o sinal
    # bruto -- amplitude_fundo=8 vs picos ~1-3).
    erro_bruto = np.abs(observado - picos).mean()
    erro_corrigido = np.abs(corrigido - picos).mean()
    assert erro_corrigido < erro_bruto * 0.15, (
        f"erro medio contra os picos puros nao caiu o suficiente: "
        f"bruto={erro_bruto:.4f} corrigido={erro_corrigido:.4f}")

    # Nos canais de pico (onde os picos puros sao altos), o sinal
    # corrigido tem que preservar a maior parte da altura -- AirPLS nao
    # pode "comer" o sinal real junto com o fundo.
    canais_de_pico = picos > 0.5 * picos.max()
    razao_preservada = corrigido[canais_de_pico].sum() / picos[canais_de_pico].sum()
    assert 0.7 < razao_preservada < 1.3, (
        f"altura dos picos nao preservada apos correcao: razao={razao_preservada:.3f}")


def test_airpls_e_transformer_sklearn_valido():
    """fit/transform aceitam matriz (n_amostras, n_canais), sem exigir y."""
    rng = np.random.default_rng(1)
    _, _, espectros = zip(*[_espectro_raman_sintetico(rng) for _ in range(5)])
    X = np.vstack(espectros)
    t = AirPLS()
    t.fit(X)
    saida = t.transform(X)
    assert saida.shape == X.shape
    assert np.isfinite(saida).all()


# --------------------------------------------------------------------------
# Portao de aceite (Bloco 20) contra o dataset publico Raman REAL
# --------------------------------------------------------------------------

def _pasta_mendeley():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "mendeley_ctgg7k4m5g"
    return pasta if (pasta / "Raman1A.csv").is_file() else None


requer_mendeley_raman = pytest.mark.skipif(
    _pasta_mendeley() is None,
    reason=("Raman1A.csv (Mendeley ctgg7k4m5g) ausente. Baixe com "
            "'python scripts/download_datasets/baixar_mendeley_oleos.py' "
            "e aponte GUARACI_DATASETS_DIR."))


@requer_mendeley_raman
@pytest.mark.slow
def test_portao_aceite_airpls_no_raman_publico():
    """Portao de aceite (Bloco 20): roda o MESMO pipeline PLS-DA com/sem
    AirPLS sob split group-aware bloqueado, 10 seeds, Wilcoxon pareado
    -- mesma disciplina de EMSC/OSC (Passo 134). Reporta o veredito
    REAL, seja qual for -- nao ha' gate de "precisa aprovar" aqui, so'
    de "roda e produz um veredito valido" (a prosa de recomendacao so'
    acontece se `veredito == 'aprovado'`, ver docs/PROGRESSO.md)."""
    import pandas as pd

    from guaraci.portao_correcao_sinal import avaliar_correcao_sinal_pls

    pasta = _pasta_mendeley()
    df = pd.read_csv(pasta / "Raman1A.csv").dropna()
    X = df.iloc[:, 2:].to_numpy(dtype=float)
    y = np.log10(df["PeroxideValue"].to_numpy(dtype=float))
    grupos = np.arange(len(df))   # 1 linha = 1 amostra fisica, sem replica

    veredito = avaliar_correcao_sinal_pls(
        "AirPLS_raman", X, y, grupos, AirPLS(),
        metrica="RMSEP", n_componentes=7, n_seeds=10)

    assert veredito.poder_suficiente
    assert veredito.veredito in ("aprovado", "rejeitado", "neutro")
    print(veredito.resumo())
