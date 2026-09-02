"""Testes de hsi_segmentation.py (Passo 96) -- Otsu (contra um caso com
limiar conhecido) e segmentacao PCA+Otsu (cena sintetica com objeto
claramente separavel do fundo). O teste contra o dataset publico real
(gated por GUARACI_DATASETS_DIR, mesmo padrao dos outros modulos HSI)
NAO tem mascara de referencia no dataset -- por isso valida por
INSPECAO VISUAL DOCUMENTADA (Passo 96 da instrucao: "se nao houver [
mascara de referencia]: validar por inspecao visual documentada (salvar
mascara ao lado da imagem original) -- nao fingir metrica quantitativa
inexistente"), salvando a mascara em `resultados_hsi_segmentacao/`
(gitignorado, `resultados*/`) em vez de afirmar uma acuracia que nao
pode ser medida sem anotacao de referencia."""
from __future__ import annotations

import os
from pathlib import Path

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


def test_segment_object_pca_otsu_respeita_objeto_e_pico_maior():
    """Mesma cena, mas com o objeto sendo a MAIORIA da area -- confirma
    que o parametro objeto_e_pico_maior=True inverte a escolha default
    (que assume objeto=minoria)."""
    rng = np.random.default_rng(1)
    altura, largura, bandas = 40, 40, 20
    espectro_a = rng.normal(loc=0.2, scale=0.01, size=bandas)
    espectro_b = rng.normal(loc=0.7, scale=0.01, size=bandas)
    cubo = np.tile(espectro_b, (altura, largura, 1))  # maioria = "objeto"
    cubo += rng.normal(scale=0.005, size=cubo.shape)
    cubo[:5, :5] = espectro_a  # canto pequeno = "fundo"

    resultado = segment_object_pca_otsu(cubo, objeto_e_pico_maior=True)
    assert resultado.fracao_objeto > 0.5


def test_segment_object_pca_otsu_forma_errada_levanta_erro():
    with pytest.raises(ValueError, match="3D"):
        segment_object_pca_otsu(np.zeros((10, 10)))


# ── contra o dataset publico real: inspecao visual documentada ─────────

def _pasta_deephs_kaki():
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        return None
    pasta = Path(raiz) / "deephs_kaki_vis"
    return pasta if (pasta / "manifest.json").is_file() else None


requer_deephs_kaki = pytest.mark.skipif(
    _pasta_deephs_kaki() is None,
    reason=("dataset publico DeepHS Fruit/Kaki/VIS ausente. Baixe com "
            "'python scripts/download_datasets/baixar_deephs_kaki.py' e "
            "aponte GUARACI_DATASETS_DIR."))


@requer_deephs_kaki
def test_segment_object_pca_otsu_contra_kaki_real_inspecao_visual():
    """O dataset NAO tem mascara de segmentacao de referencia -- validacao
    e' por inspecao visual documentada (Passo 96), nao por metrica
    quantitativa inventada. Salva imagem media + mascara lado a lado em
    resultados_hsi_segmentacao/ (gitignorado) para conferencia manual."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from guaraci.hsi_io import load_deephs_kaki_dataset

    pasta = str(_pasta_deephs_kaki())
    cubos, rotulos, grupos, wavelengths, meta_df = load_deephs_kaki_dataset(pasta)

    cubo = cubos[0]
    resultado = segment_object_pca_otsu(cubo)

    # Contra-prova minima automatica (nao substitui a inspecao visual, so'
    # garante que a segmentacao nao degenerou para tudo/nada -- uma fruta
    # numa cena de bancada real nunca ocupa 0% nem 100% do quadro):
    assert 0.02 < resultado.fracao_objeto < 0.98

    pasta_saida = Path("resultados_hsi_segmentacao")
    pasta_saida.mkdir(exist_ok=True)
    fig, eixos = plt.subplots(1, 2, figsize=(8, 4))
    eixos[0].imshow(np.mean(cubo, axis=2), cmap="gray")
    eixos[0].set_title("Imagem media (todas as bandas)")
    eixos[1].imshow(resultado.mascara, cmap="gray")
    eixos[1].set_title(f"Mascara PCA+Otsu (fracao={resultado.fracao_objeto:.2f})")
    for eixo in eixos:
        eixo.axis("off")
    fig.tight_layout()
    caminho_png = pasta_saida / "kaki_segmentacao_amostra.png"
    fig.savefig(caminho_png, dpi=100)
    plt.close(fig)
    assert caminho_png.is_file()
