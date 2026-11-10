"""Testes de hsi_segmentation.py (Passo 96) -- Otsu (contra um caso com
limiar conhecido) e segmentacao PCA+Otsu (cena sintetica com objeto
claramente separavel do fundo). Dado real sem mascara de referencia
se valida por INSPECAO VISUAL DOCUMENTADA (Passo 96 da instrucao: "se nao houver [
mascara de referencia]: validar por inspecao visual documentada (salvar
mascara ao lado da imagem original) -- nao fingir metrica quantitativa
inexistente"), salvando a mascara em `resultados_hsi_segmentacao/`
(gitignorado, `resultados*/`) em vez de afirmar uma acuracia que nao
pode ser medida sem anotacao de referencia."""
from __future__ import annotations


import numpy as np
import pytest

from guaraci.hsi_segmentation import otsu_threshold, segment_object_pca_otsu


# ── otsu_threshold ───────────────────────────────────────────────────────

def test_otsu_threshold_separa_2_grupos_bem_definidos():
    """O criterio funcional de Otsu e' separar os grupos, nao cair
    necessariamente no meio geometrico do intervalo -- com um gap vazio
    largo entre os grupos, a variancia entre-classes fica CONSTANTE ao
    longo de todo o gap (nenhum dado ali para desempatar), entao o limiar
    escolhido pode cair em qualquer ponto do plato' (inclusive perto da
    borda de um dos grupos) sem prejudicar a separacao -- e' isso que
    este teste confere, nao a posicao exata do limiar."""
    rng = np.random.default_rng(0)
    grupo_baixo = rng.normal(loc=0.2, scale=0.02, size=500)
    grupo_alto = rng.normal(loc=0.8, scale=0.02, size=500)
    valores = np.concatenate([grupo_baixo, grupo_alto])
    limiar = otsu_threshold(valores)
    assert np.all(grupo_baixo <= limiar)
    assert np.all(grupo_alto > limiar)


def test_otsu_threshold_cena_constante_nao_quebra():
    valores = np.full(100, 0.5)
    limiar = otsu_threshold(valores)
    assert limiar == pytest.approx(0.5)


def test_otsu_threshold_vazio_levanta_erro():
    with pytest.raises(ValueError, match="nenhum valor"):
        otsu_threshold(np.array([]))


# ── segment_object_pca_otsu: cena sintetica com objeto separavel ────────

def _cena_com_objeto(altura: int = 40, largura: int = 40, bandas: int = 20,
                      seed: int = 0) -> tuple:
    """Fundo com 1 assinatura espectral, um bloco central (objeto, MINORIA
    da cena) com assinatura bem diferente -- caso facil, serve so' para
    confirmar que o pipeline PCA+Otsu localiza a regiao certa."""
    rng = np.random.default_rng(seed)
    espectro_fundo = rng.normal(loc=0.2, scale=0.01, size=bandas)
    espectro_objeto = rng.normal(loc=0.7, scale=0.01, size=bandas)

    cubo = np.tile(espectro_fundo, (altura, largura, 1))
    cubo += rng.normal(scale=0.005, size=cubo.shape)

    mascara_real = np.zeros((altura, largura), dtype=bool)
    y0, y1 = altura // 3, 2 * altura // 3
    x0, x1 = largura // 3, 2 * largura // 3
    mascara_real[y0:y1, x0:x1] = True
    cubo[mascara_real] = espectro_objeto + rng.normal(
        scale=0.005, size=(mascara_real.sum(), bandas))
    return cubo, mascara_real


def test_segment_object_pca_otsu_localiza_o_objeto():
    cubo, mascara_real = _cena_com_objeto()
    resultado = segment_object_pca_otsu(cubo)

    # Nao exige pixel-a-pixel identico (PCA+Otsu e' uma heuristica) --
    # exige sobreposicao forte (IoU) com a mascara real conhecida.
    intersecao = np.sum(resultado.mascara & mascara_real)
    uniao = np.sum(resultado.mascara | mascara_real)
    iou = intersecao / uniao
    assert iou > 0.8, f"IoU baixo ({iou:.3f}) -- segmentacao nao localizou o objeto."
    assert resultado.variancia_explicada_pc1 > 0.5
    assert 0.0 < resultado.fracao_objeto < 1.0


def test_segment_object_pca_otsu_objeto_ocupando_maioria_da_cena():
    """ACHADO real corrigido (2026-09-01): a primeira versao deste modulo
    assumia "objeto = minoria de pixels" e invertia a mascara quando a
    objeto ocupava a MAIORIA do quadro (caso real do dataset camera VIS,
    achado por inspecao visual). A heuristica de borda (fundo = classe
    majoritaria nos pixels mais externos) cobre este caso -- objeto
    grande e centralizado, fundo so' aparecendo nos cantos."""
    rng = np.random.default_rng(1)
    altura, largura, bandas = 40, 40, 20
    espectro_fundo = rng.normal(loc=0.2, scale=0.01, size=bandas)
    espectro_objeto = rng.normal(loc=0.7, scale=0.01, size=bandas)

    # Fundo = MOLDURA fina em volta de toda a cena (toca a borda inteira,
    # como a mesa/bancada ao redor da objeto no dataset real); objeto =
    # interior, MAIORIA da area.
    cubo = np.tile(espectro_fundo, (altura, largura, 1))
    cubo += rng.normal(scale=0.005, size=cubo.shape)
    mascara_real = np.zeros((altura, largura), dtype=bool)
    mascara_real[4:-4, 4:-4] = True
    cubo[mascara_real] = espectro_objeto + rng.normal(
        scale=0.005, size=(mascara_real.sum(), bandas))

    resultado = segment_object_pca_otsu(cubo)
    intersecao = np.sum(resultado.mascara & mascara_real)
    uniao = np.sum(resultado.mascara | mascara_real)
    iou = intersecao / uniao
    assert iou > 0.8, f"IoU baixo ({iou:.3f}) -- objeto majoritario mal segmentado."
    assert resultado.fracao_objeto > 0.5


def test_segment_object_pca_otsu_largura_borda_grande_demais_levanta_erro():
    cubo = np.zeros((6, 6, 4))
    with pytest.raises(ValueError, match="grande demais"):
        segment_object_pca_otsu(cubo, largura_borda=3)


def test_segment_object_pca_otsu_forma_errada_levanta_erro():
    with pytest.raises(ValueError, match="3D"):
        segment_object_pca_otsu(np.zeros((10, 10)))




