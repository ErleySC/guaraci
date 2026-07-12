"""Testes de funcoes puras extraidas de figuras.py (matematica de plotagem,
testavel sem renderizar nenhuma figura de verdade).
"""
import numpy as np

from guaraci.figuras import _escala_vetores_biplot


def test_escala_vetores_biplot_respeita_ambos_os_eixos():
    """REGRESSAO: um bug real deixava vetores com componente forte no eixo
    de MENOR alcance (aqui, PC2) desenhados MUITO alem da area visivel --
    rotulos apareciam flutuando fora do grafico renderizado. Reproduz o
    cenario (PC1 com alcance ~10x maior que PC2, tipico quando os
    autovalores sao desiguais) e prova que a escala escolhida mantem TODO
    vetor dentro de `frac` do maior score de CADA eixo, nao so' do eixo
    dominante."""
    rng = np.random.default_rng(0)
    n, p = 60, 40
    scores2 = np.column_stack([
        rng.uniform(-0.05, 0.05, n),     # PC1: alcance grande
        rng.uniform(-0.005, 0.005, n),   # PC2: alcance 10x menor
    ])
    loadings = rng.uniform(-0.5, 0.5, size=(p, 2))
    # Garante que ALGUMAS variaveis tenham componente PC2 dominante (o
    # cenario que disparava o bug).
    loadings[0] = [0.02, 0.45]

    frac = 0.8
    escala = _escala_vetores_biplot(scores2, loadings, frac=frac)
    vx = loadings[:, 0] * escala
    vy = loadings[:, 1] * escala

    max_score_x = np.abs(scores2[:, 0]).max()
    max_score_y = np.abs(scores2[:, 1]).max()
    # Tolerancia de ponto flutuante minima -- a garantia e' "nao ultrapassa
    # frac do maior score daquele eixo", nao um limite exato.
    assert np.all(np.abs(vx) <= frac * max_score_x + 1e-12)
    assert np.all(np.abs(vy) <= frac * max_score_y + 1e-12)


def test_escala_vetores_biplot_escala_positiva_com_dados_degenerados():
    """Scores ou loadings todos zero (caso degenerado) nao devem gerar
    escala negativa, NaN ou ZeroDivisionError."""
    scores2 = np.zeros((5, 2))
    loadings = np.zeros((10, 2))
    escala = _escala_vetores_biplot(scores2, loadings)
    assert np.isfinite(escala)
    assert escala >= 0
