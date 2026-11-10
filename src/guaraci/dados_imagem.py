"""
dados_imagem.py — Extracao de dados quimiometricos a partir de IMAGENS
DIGITAIS (colorimetria digital / Digital Image-Based Analysis).

Protótipo GENERICO (2026-07): converte cada imagem numa matriz de "sinal"
usando estatisticas de cor (RGB/HSV/Lab) — e opcionalmente textura (GLCM,
requer scikit-image) — analogo a um espectro: cada imagem vira UMA LINHA da
matriz X, cada estatistica de canal vira UMA VARIAVEL (coluna), exatamente
como cada comprimento de onda e uma variavel no mode .dx. A partir dai, TODA
a maquinaria quimiometrica existente (PCA, PLS-DA, DD-SIMCA, OPLS-DA, selecao
de variaveis, figuras de merito) funciona SEM alteracao — essas funcoes so
enxergam uma matriz numerica, nao sabem se a coluna 47 e um comprimento de
onda ou o canal G medio de uma foto.

Convencao de pastas: MESMA do mode .dx — uma subpasta por classe (ou pasta
unica com arquivos soltos, fallback). Extensoes aceitas: .jpg/.jpeg/.png/
.bmp/.tif/.tiff.

Limitacao conhecida deste prototipo: o "eixo de variaveis" retornado
(equivalente a `wavenumbers`) e apenas `np.arange(n_features)` — os graficos
que rotulam o eixo X como comprimento de onda (VIP, loadings, etc.) vao
mostrar indices numericos em vez do nome da feature (ex. "R_mean"). Rotular
esse eixo com os nomes reais das features e uma extensao futura (afeta varias
funcoes de figura em figuras.py, fora do escopo deste prototipo).

mae_id/concentracao: nao ha convencao de metadado equivalente ao ##TITLE=
do JCAMP-DX para imagens genericas. `conc` fica None sempre (sem
quantificacao neste protótipo) e `mae_id` depende do NIVEL DE GARANTIA DE
AGRUPAMENTO detectado automaticamente na pasta de dados (Bloco 8,
2026-08-25), nesta ordem de prioridade:

  - "high"   — subpasta por amostra fisica: cada subpasta de classe
    contem SO' subpastas (nunca arquivo solto), uma por amostra fisica;
    cada foto dentro dela e' uma replica do mesmo grupo. Sem parsing de
    nome, sem ambiguidade.
  - "medium" — CSV de associacao manual (`amostras.csv` por padrao) na
    RAIZ da pasta de dados, colunas `arquivo,id_amostra`. Usado quando o
    nivel "high" nao esta presente. TODO arquivo de imagem carregado
    precisa aparecer no CSV -- cobertura parcial e' erro, nunca
    processamento parcial em silencio.
  - "none"   — nem subpasta por amostra nem CSV presentes: aceita
    processar mesmo assim (uso pratico, "so' jogar as fotos e rodar"),
    mas `mae_id` fica None (fallback StratifiedKFold, sem protecao contra
    vazamento) e a limitacao e' declarada explicitamente em 3 saidas:
    log da execucao, model card, e manifesto do modelo -- nunca so' em
    docstring/comentario interno (ver `dados_io._leitor_imagem` e
    `pipeline.executar`).

EXIF NAO e' usado como fonte de agrupamento -- avaliado e descartado:
recompressao/edicao (WhatsApp, apps de galeria) apaga o metadado na
pratica, e mesmo quando presente, "fotos tiradas numa janela de tempo
curta" nao garante "mesma amostra fisica" (heuristica fragil demais para
uma alegacao de seguranca contra vazamento).

NORMALIZACAO POR CARTAO E SEGMENTACAO (fase 2, opt-in, R12):
  - `imagem_referencia` (x0, y0, x1, y1 relativos): regiao de um cartao cinza ou
    branco FOTOGRAFADO EM TODAS AS FOTOS, na mesma posicao. Cada canal da foto e'
    dividido pela media do canal nessa regiao (e reescalado para o nivel
    `NIVEL_REFERENCIA`), o que remove o fator multiplicativo de iluminacao e o
    desvio de balanco de branco de cada foto. FOTOGRAFE COM O MESMO CARTAO EM
    TODAS AS SESSOES: cartoes diferentes (ou o cartao em posicoes diferentes) trocam
    um erro por outro. A divisao e' feita nos valores gravados (sRGB com gama):
    como a gama e' uma potencia, o ganho de iluminacao continua um fator comum
    ao cartao e ao objeto e cancela exatamente; o cartao de reflectancia
    desconhecida nao da' reflectancia absoluta, so' relativa ao cartao.
  - `imagem_segmentacao` (nenhuma | otsu | distancia_fundo): as estatisticas de cor
    passam a ser do OBJETO, nao do fundo. Sem isso, um fundo que muda entre classes
    (mesa, papel, luz de fundo) vira "sinal" do modelo. `otsu` limiariza a luminancia;
    `distancia_fundo` limiariza a distancia de cor ao fundo estimado nas bordas.
    Em ambos o lado "fundo" e' o que domina a borda da foto.
  - Sem segmentacao nem recorte, o carregador AVISA quando o fundo (estimado nas
    bordas) ocupa mais da metade da foto e quando a cor do fundo separa as classes.

IMPORTANTE — pre-processamento: use `default_preprocessing="autoscaling"`
(ou "mc") no mode="imagem", NUNCA os presets com Savitzky-Golay
("msc_sg_mc"/"snv_sg_mc"). MSC e SG pressupoem um sinal espectral CONTINUO
ao longo do eixo de variaveis (comprimento de onda) — nao fazem sentido
cientifico p/ um vetor curto de estatisticas de cor discretas e heterogeneas
(H fica em [0,1], L*a*b* fica em dezenas/centenas) e o SG especificamente
EXIGE janela <= numero de variaveis (18 por padrao — sem textura), o que
pode nem ser satisfeito.

IMPORTANTE — faixa espectral: `load_data()` aplica o mesmo filtro
wn_min/wn_max do mode .dx sobre o eixo simbolico (indices 0..n_features-1).
Os defaults de Config (wn_min=4000, wn_max=10000) NAO cobrem esse intervalo
pequeno e descartariam TODAS as variaveis — ajuste wn_min/wn_max (ex.:
wn_min=-1, wn_max=100) ao usar mode="imagem".
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from guaraci import agrupamento_pastas as _ap

__all__ = [
    "load_image_file",
    "recortar_relativo",
    "extract_color_features",
    "extract_texture_features",
    "load_images",
    "normalizar_por_referencia",
    "segmentar_objeto",
    "extrair_features_foto",
    "parametros_imagem_de_config",
    "SEGMENTACOES_IMAGEM",
]

_EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Nomes das features de cor, na ordem em que `extract_color_features` as monta —
# usado tambem como "wavenumbers" simbolicos (ver limitacao no docstring do modulo).
_NOMES_FEATURES_COR: Tuple[str, ...] = (
    "R_media", "G_media", "B_media", "R_dp", "G_dp", "B_dp",
    "H_media", "S_media", "V_media", "H_dp", "S_dp", "V_dp",
    "L_media", "a_media", "b_media", "L_dp", "a_dp", "b_dp",
)
_NOMES_FEATURES_TEXTURA: Tuple[str, ...] = (
    "GLCM_contraste", "GLCM_homogeneidade", "GLCM_energia", "GLCM_correlacao",
)


def load_image_file(caminho: str) -> np.ndarray:
    """Le uma imagem do disco como array RGB uint8 (H, W, 3), via Pillow."""
    from PIL import Image
    with Image.open(caminho) as im:
        if im.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
            # Tons de cinza 16 bit/inteiro/float (scanner, camera cientifica):
            # `convert("RGB")` satura tudo em 255 (foto toda branca, features
            # constantes). Reescala pela profundidade nominal: 16 bit -> /257.
            a = np.asarray(im, dtype=np.float64)
            topo = 65535.0 if im.mode.startswith("I;16") or (
                im.mode == "I" and a.max(initial=0) > 255) else 255.0
            if im.mode == "F":
                topo = 1.0 if a.max(initial=0) <= 1.0 else float(a.max())
            g = np.clip(np.round(a / topo * 255.0), 0, 255).astype(np.uint8)
            return np.stack([g] * 3, axis=-1)
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def recortar_relativo(img: np.ndarray, caixa: Tuple[float, float, float, float]
                       ) -> np.ndarray:
    """Recorta a imagem por coordenadas RELATIVAS (fracao de largura/altura):
    caixa = (esquerda, topo, direita, baixo), cada uma em [0, 1].
    Default (0,0,1,1) = imagem inteira (sem recorte)."""
    h, w = img.shape[:2]
    esq, topo, dir_, baixo = caixa
    x0 = int(round(max(0.0, min(1.0, esq)) * w))
    x1 = int(round(max(0.0, min(1.0, dir_)) * w))
    y0 = int(round(max(0.0, min(1.0, topo)) * h))
    y1 = int(round(max(0.0, min(1.0, baixo)) * h))
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    if x1 <= x0 or y1 <= y0:
        return img  # recorte degenerado -> ignora, usa a imagem inteira
    return img[y0:y1, x0:x1]


#: Nivel (0-255) para o qual a media do cartao de referencia e' levada em cada canal.
NIVEL_REFERENCIA = 190.0
SEGMENTACOES_IMAGEM = ("nenhuma", "otsu", "distancia_fundo")
#: Aviso quando o fundo estimado passa desta fracao da foto (sem segmentacao nem recorte).
LIMITE_FUNDO_DOMINANTE = 0.5


def normalizar_por_referencia(img: np.ndarray, caixa: Tuple[float, float, float, float]
                               ) -> Tuple[np.ndarray, Dict[str, object]]:
    """Divide cada canal pela media do canal na regiao `caixa` (cartao cinza/branco,
    coordenadas relativas) e leva essa media a `NIVEL_REFERENCIA`. Devolve
    (imagem uint8, info) com `media_ref` (R, G, B), `cv` (desvio/media da luminancia
    na regiao: cartao uniforme da' < 0,1) e `saturado`. Levanta ValueError se a
    regiao e' escura demais (< 20/255) para dividir com seguranca."""
    regiao = recortar_relativo(img, caixa)
    if regiao is img and tuple(caixa) != (0.0, 0.0, 1.0, 1.0):
        raise ValueError("regiao de referencia degenerada")
    px = regiao.reshape(-1, 3).astype(np.float64)
    media = px.mean(axis=0)
    if float(media.min()) < 20.0:
        raise ValueError(
            f"a regiao de referencia esta escura demais (media R,G,B = "
            f"{media[0]:.0f},{media[1]:.0f},{media[2]:.0f} de 255): o cartao nao esta nessa posicao "
            f"ou a foto esta subexposta; conferir imagem_referencia.")
    lum = px @ np.array([0.2126, 0.7152, 0.0722])
    info: Dict[str, object] = {
        "media_ref": [float(v) for v in media],
        "cv": float(lum.std() / lum.mean()) if lum.mean() > 0 else 0.0,
        "saturado": bool(float(media.max()) >= 250.0),
    }
    out = np.clip(np.round(img.astype(np.float64) / media * NIVEL_REFERENCIA), 0, 255)
    return out.astype(np.uint8), info


def _luminancia(img: np.ndarray) -> np.ndarray:
    return (img[..., :3].astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722]))


def _mascara_borda(h: int, w: int) -> np.ndarray:
    largura = max(2, int(round(0.03 * min(h, w))))
    m = np.zeros((h, w), dtype=bool)
    m[:largura, :] = m[-largura:, :] = True
    m[:, :largura] = m[:, -largura:] = True
    return m


def _reduzir(img: np.ndarray, max_lado: int = 800) -> Tuple[np.ndarray, int]:
    f = max(1, int(np.ceil(max(img.shape[:2]) / max_lado)))
    return img[::f, ::f], f


def _estimar_fundo(img: np.ndarray) -> Dict[str, object]:
    """Cor do fundo = mediana dos pixels da borda; `homogeneo` diz se a borda e' uniforme
    (desvio absoluto mediano < 12/255 por canal), condicao para chamar aquilo de fundo."""
    h, w = img.shape[:2]
    borda = img[_mascara_borda(h, w)][:, :3].astype(np.float64)
    cor = np.median(borda, axis=0)
    mad = float(np.median(np.abs(borda - cor)))
    return {"cor": [float(v) for v in cor], "homogeneo": bool(mad < 12.0)}


def _limpar_mascara(m: np.ndarray) -> np.ndarray:
    """Abertura (tira ruido de 1 pixel), preenche buracos e descarta componentes
    com menos de 5% da maior (varios objetos pequenos continuam)."""
    from scipy import ndimage as ndi
    m = ndi.binary_opening(m, iterations=1)
    m = ndi.binary_fill_holes(m)
    rot, n = ndi.label(m)
    if n > 1:
        areas = np.bincount(rot.ravel())[1:]
        manter = np.where(areas >= 0.05 * areas.max())[0] + 1
        m = np.isin(rot, manter)
    return m


def segmentar_objeto(img: np.ndarray, metodo: str
                     ) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    """Mascara booleana (True = objeto) por `otsu` (luminancia) ou `distancia_fundo`
    (distancia de cor ao fundo estimado nas bordas). O lado que domina a borda e'
    o fundo. Devolve (mascara | None, info): None quando a segmentacao nao se
    sustenta (objeto < 0,5% ou > 99,5% da foto, ou sem contraste com o fundo) --
    quem chama usa a foto inteira e avisa."""
    from guaraci.hsi_segmentation import otsu_threshold
    if metodo not in ("otsu", "distancia_fundo"):
        raise ValueError(f"segmentacao '{metodo}' desconhecida; use {list(SEGMENTACOES_IMAGEM)}")
    h, w = img.shape[:2]
    peq, f = _reduzir(img)
    hp, wp = peq.shape[:2]
    borda = _mascara_borda(hp, wp)
    if metodo == "otsu":
        y = _luminancia(peq)
        if float(np.ptp(y)) < 8.0:
            return None, {"motivo": "sem contraste de luminancia"}
        acima = y > otsu_threshold(y.ravel())
        fundo_e_acima = float(np.mean(acima[borda])) >= 0.5
        obj = ~acima if fundo_e_acima else acima
    else:
        fundo = np.array(_estimar_fundo(peq)["cor"])
        d = np.sqrt(((peq[..., :3].astype(np.float64) - fundo) ** 2).sum(axis=-1))
        if float(d.max()) < 25.0:
            return None, {"motivo": "objeto sem contraste de cor com o fundo"}
        obj = d > otsu_threshold(d.ravel())
    obj = _limpar_mascara(obj)
    frac = float(obj.mean())
    if frac < 0.005 or frac > 0.995:
        return None, {"motivo": f"mascara degenerada (objeto = {frac:.1%} da foto)"}
    if f > 1:
        obj = np.repeat(np.repeat(obj, f, axis=0), f, axis=1)[:h, :w]
    return obj, {"fracao_objeto": float(obj.mean())}


def _fundo_dominante(img: np.ndarray) -> Optional[Dict[str, object]]:
    """Diagnostico barato (foto reduzida) para o aviso de fundo dominante: devolve a cor
    do fundo e a fracao dele quando a borda e' homogenea e ha' objeto distinto;
    None quando nao ha' como dizer."""
    peq, _f = _reduzir(img, 400)
    fundo = _estimar_fundo(peq)
    if not fundo["homogeneo"]:
        return {"cor": fundo["cor"], "fracao_fundo": None}
    from guaraci.hsi_segmentation import otsu_threshold
    d = np.sqrt(((peq[..., :3].astype(np.float64) - np.array(fundo["cor"])) ** 2).sum(axis=-1))
    if float(d.max()) < 30.0:
        return {"cor": fundo["cor"], "fracao_fundo": None}
    frac_obj = float(np.mean(d > otsu_threshold(d.ravel())))
    return {"cor": fundo["cor"], "fracao_fundo": 1.0 - frac_obj}


def parametros_imagem_de_config(cfg) -> Dict[str, object]:
    """Parametros de extracao de cor desta execucao, para gravar no modelo exportado e
    reaplicar IGUAL em fotos novas (`extrair_features_foto(..., **params)`)."""
    ref = tuple(getattr(cfg, "image_reference", ()) or ())
    return {"recorte": [float(v) for v in getattr(cfg, "image_crop", (0.0, 0.0, 1.0, 1.0))],
            "referencia": [float(v) for v in ref] or None,
            "nivel_referencia": NIVEL_REFERENCIA,
            "segmentacao": str(getattr(cfg, "image_segmentation", "nenhuma") or "nenhuma"),
            "incluir_textura": bool(getattr(cfg, "include_image_texture", False))}


def extrair_features_foto(
        img: np.ndarray, *,
        recorte: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        referencia: Optional[Sequence[float]] = None, segmentacao: str = "nenhuma",
        incluir_textura: bool = False, **_ignorados) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Features de uma foto ja' lida (`load_image_file`): referencia -> recorte ->
    segmentacao -> estatisticas de cor (+ textura). Mesmo caminho do treino, para a
    predicao de fotos novas. Devolve (features, info) com os diagnosticos."""
    info: Dict[str, Any] = {}
    if referencia:
        img, info["referencia"] = normalizar_por_referencia(img, tuple(referencia))   # type: ignore[arg-type]
    img = recortar_relativo(img, tuple(recorte))   # type: ignore[arg-type]
    mascara = None
    if segmentacao and segmentacao != "nenhuma":
        mascara, info["segmentacao"] = segmentar_objeto(img, segmentacao)
    elif tuple(recorte) == (0.0, 0.0, 1.0, 1.0):
        info["fundo"] = _fundo_dominante(img)
    feats = extract_color_features(img, mascara)
    if incluir_textura:
        base = img
        if mascara is not None:
            ys, xs = np.nonzero(mascara)
            base = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]   # caixa do objeto
        feats.update(extract_texture_features(base))
    return feats, info


def _rgb_para_hsv(img_rgb01: np.ndarray) -> np.ndarray:
    """RGB [0,1] -> HSV [0,1], vetorizado (evita depender de colorsys/skimage)."""
    r, g, b = img_rgb01[..., 0], img_rgb01[..., 1], img_rgb01[..., 2]
    maxc = np.max(img_rgb01, axis=-1)
    minc = np.min(img_rgb01, axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 1e-12, delta / np.where(maxc > 1e-12, maxc, 1.0), 0.0)

    rc = np.where(delta > 1e-12, (maxc - r) / np.where(delta > 1e-12, delta, 1.0), 0.0)
    gc = np.where(delta > 1e-12, (maxc - g) / np.where(delta > 1e-12, delta, 1.0), 0.0)
    bc = np.where(delta > 1e-12, (maxc - b) / np.where(delta > 1e-12, delta, 1.0), 0.0)

    h = np.zeros_like(maxc)
    h = np.where(maxc == r, bc - gc, h)
    h = np.where(maxc == g, 2.0 + rc - bc, h)
    h = np.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = np.where(delta > 1e-12, h, 0.0)
    return np.stack([h, s, v], axis=-1)


def _rgb_para_lab(img_rgb01: np.ndarray) -> np.ndarray:
    """RGB [0,1] (sRGB) -> CIE Lab, vetorizado (formulas padrao, D65)."""
    def _linearizar(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    rl, gl, bl = (_linearizar(img_rgb01[..., i]) for i in range(3))
    x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041

    # Referencia de branco D65
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def _f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)

    fx, fy, fz = _f(x / xn), _f(y / yn), _f(z / zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ = 200.0 * (fy - fz)
    return np.stack([L, a, b_], axis=-1)


def extract_color_features(img: np.ndarray, mascara: Optional[np.ndarray] = None
                           ) -> Dict[str, float]:
    """Media e desvio-padrao por canal em RGB, HSV e Lab — 18 features no
    total, na mesma ordem de `_NOMES_FEATURES_COR`. Entrada: array uint8
    (H, W, 3) ou (H, W) RGB/tons de cinza. `mascara` (bool, H x W): so' os
    pixels True (o objeto) entram nas estatisticas."""
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if mascara is not None:
        img = img[np.asarray(mascara, dtype=bool)][:, None, :]    # (N, 1, 3): as contas sao por pixel
    img01 = img[..., :3].astype(np.float64) / 255.0

    hsv = _rgb_para_hsv(img01)
    lab = _rgb_para_lab(img01)

    feats: Dict[str, float] = {}
    for nome, canal in zip(("R", "G", "B"), range(3)):
        feats[f"{nome}_media"] = float(np.mean(img01[..., canal])) * 255.0
        feats[f"{nome}_dp"] = float(np.std(img01[..., canal])) * 255.0
    # Matiz e' ANGULAR (0 e 1 sao o mesmo vermelho): media/dp aritmeticos de
    # um objeto vermelho com pixels em 0.99 e 0.01 dao ~0.5 (ciano). Usa media
    # circular e dp circular sqrt(-2 ln R) (Fisher 1993, Statistical Analysis
    # of Circular Data), ambos em fracao de volta.
    ang = 2.0 * np.pi * hsv[..., 0]
    c, s_ = float(np.mean(np.cos(ang))), float(np.mean(np.sin(ang)))
    r = min(1.0, float(np.hypot(c, s_)))
    feats["H_media"] = float((np.arctan2(s_, c) / (2.0 * np.pi)) % 1.0)
    feats["H_dp"] = float(np.sqrt(-2.0 * np.log(r)) / (2.0 * np.pi)) if r > 1e-12 else 0.5
    for nome, canal in zip(("S", "V"), (1, 2)):
        feats[f"{nome}_media"] = float(np.mean(hsv[..., canal]))
        feats[f"{nome}_dp"] = float(np.std(hsv[..., canal]))
    for nome, canal in zip(("L", "a", "b"), range(3)):
        feats[f"{nome}_media"] = float(np.mean(lab[..., canal]))
        feats[f"{nome}_dp"] = float(np.std(lab[..., canal]))
    return {k: feats[k] for k in _NOMES_FEATURES_COR}


def extract_texture_features(img: np.ndarray) -> Dict[str, float]:
    """Features de textura via GLCM (contraste/homogeneidade/energia/
    correlacao) usando scikit-image — OPCIONAL, retorna dict vazio (com
    aviso) se scikit-image nao estiver instalado. Nao e dependencia
    obrigatoria do projeto (protótipo generico)."""
    try:
        from skimage.feature import graycomatrix, graycoprops
        from skimage.color import rgb2gray
        from skimage.util import img_as_ubyte
    except ImportError:
        print("  [AVISO] scikit-image nao instalado — features de textura "
              "puladas (pip install scikit-image para habilitar).")
        return {}

    img = np.asarray(img)
    cinza = img_as_ubyte(rgb2gray(img[..., :3]) if img.ndim == 3 else img)
    glcm = graycomatrix(cinza, distances=[1], angles=[0], levels=256,
                        symmetric=True, normed=True)
    return {
        "GLCM_contraste":     float(graycoprops(glcm, "contrast")[0, 0]),
        "GLCM_homogeneidade": float(graycoprops(glcm, "homogeneity")[0, 0]),
        "GLCM_energia":       float(graycoprops(glcm, "energy")[0, 0]),
        "GLCM_correlacao":    float(graycoprops(glcm, "correlation")[0, 0]),
    }


#: Nome do CSV de associacao manual (nivel "medium"), procurado na raiz da
#: pasta de dados. Nao configuravel neste prototipo -- ver docstring do modulo.
_NOME_CSV_AMOSTRAS = _ap.NOME_CSV_AMOSTRAS

_GROUPING_HIGH = _ap.GROUPING_HIGH
_GROUPING_MEDIUM = _ap.GROUPING_MEDIUM
_GROUPING_NONE = _ap.GROUPING_NONE


def _listar_arquivos_imagem(pasta: str) -> List[str]:
    """Busca arquivos de imagem por extensao -- delega a `agrupamento_
    pastas.py` (Passo 111, extraido daqui para ser reaproveitado pelo modo
    `hsi`); mantido aqui com este nome por compatibilidade (testes e o
    resto deste modulo importam por este nome)."""
    return _ap.listar_arquivos_por_extensao(pasta, _EXTENSOES_IMAGEM)


def _detectar_subpastas_imagem(raiz: str) -> List[str]:
    """Subpastas (1 por classe) que contem >=1 arquivo de imagem, direto ou
    dentro de subpasta de amostra (nivel "high") — mesma convencao do mode
    .dx (`_detectar_subpastas_classe` em dados_io.py) generalizada para 1
    nivel extra opcional."""
    return _ap.detectar_subpastas_por_extensao(raiz, _EXTENSOES_IMAGEM)


def _detectar_nivel_high(subpastas_classe: List[str]
                          ) -> Optional[Dict[str, str]]:
    return _ap.detectar_nivel_high(subpastas_classe, _EXTENSOES_IMAGEM)


def _detectar_nivel_medium(pasta_raiz: str, arquivos: List[str]
                            ) -> Optional[Dict[str, str]]:
    return _ap.detectar_nivel_medium(pasta_raiz, arquivos, _NOME_CSV_AMOSTRAS)


def _avisos_da_leitura(infos: List[Dict[str, Any]], rotulos: np.ndarray,
                       referencia: Optional[Sequence[float]], segmentacao: str,
                       caixa_recorte: Tuple[float, float, float, float]) -> List[str]:
    """Avisos AGREGADOS (um por problema, nao um por foto) sobre cartao, mascara e fundo."""
    avisos: List[str] = []
    n = len(infos)
    if referencia:
        refs = [i["referencia"] for i in infos if i.get("referencia")]
        sat = sum(1 for r in refs if r["saturado"])
        if sat:
            avisos.append(f"o cartao de referencia esta saturado (media >= 250/255) em {sat}/{n} fotos: "
                          f"a divisao nao corrige a luz nessas fotos; exponha menos.")
        naounif = sum(1 for r in refs if r["cv"] > 0.15)
        if naounif:
            avisos.append(f"a regiao de referencia nao e' uniforme (desvio/media da luminancia > 15%) em "
                          f"{naounif}/{n} fotos: sombra, reflexo ou regiao fora do cartao; conferir imagem_referencia.")
        x0, y0, x1, y1 = referencia
        c0, c1, c2, c3 = caixa_recorte
        if segmentacao == "nenhuma" and min(x1, c2) > max(x0, c0) and min(y1, c3) > max(y0, c1):
            avisos.append("a regiao do cartao se sobrepoe a area analisada (imagem_recorte): o cartao entra nas "
                          "estatisticas de cor; recorte so' o objeto.")
    if segmentacao != "nenhuma":
        falhas = [i["segmentacao"] for i in infos if i.get("segmentacao") and
                  "motivo" in i["segmentacao"]]
        if falhas:
            avisos.append(f"a segmentacao '{segmentacao}' falhou em {len(falhas)}/{n} fotos (ex.: "
                          f"{falhas[0]['motivo']}); nelas foi usada a foto inteira.")
        return avisos
    fundos = [i.get("fundo") for i in infos if i.get("fundo")]
    fracs = [f["fracao_fundo"] for f in fundos if f.get("fracao_fundo") is not None]
    dominantes = [f for f in fracs if f > LIMITE_FUNDO_DOMINANTE]
    if dominantes:
        avisos.append(
            f"em {len(dominantes)}/{n} fotos o fundo (estimado nas bordas) ocupa mais de "
            f"{LIMITE_FUNDO_DOMINANTE:.0%} da imagem (mediana {np.median(dominantes):.0%}) e as estatisticas "
            f"de cor misturam fundo e objeto; use imagem_segmentacao: otsu (ou distancia_fundo) ou imagem_recorte.")
    # a cor do fundo separa as classes? (heuristica: distancia entre medias de classe > 4x a dispersao interna)
    cores = [(r, f["cor"]) for r, f in zip(rotulos, [i.get("fundo") for i in infos]) if f]
    classes = sorted(set(r for r, _ in cores))
    if len(classes) >= 2:
        med = {c: np.mean([cor for r, cor in cores if r == c], axis=0) for c in classes}
        interno = np.sqrt(np.mean([np.sum((np.array(cor) - med[r]) ** 2) for r, cor in cores]) / 3.0)
        pares = [(float(np.linalg.norm(med[a] - med[b])), a, b)
                 for i, a in enumerate(classes) for b in classes[i + 1:]]
        dist, a, b = max(pares)
        if dist > 4.0 * max(float(interno), 3.0):
            avisos.append(
                f"a cor do fundo (borda das fotos) difere entre classes ('{a}' x '{b}': distancia RGB "
                f"{dist:.0f} contra dispersao interna {interno:.0f}): o modelo pode estar aprendendo o fundo "
                f"e nao o objeto. Use imagem_segmentacao ou fotografe todas as classes sobre o mesmo fundo "
                f"(heuristica).")
    return avisos


def load_images(
        pasta: str,
        caixa_recorte: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        incluir_textura: bool = False,
        *, referencia: Optional[Sequence[float]] = None, segmentacao: str = "nenhuma",
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                     Optional[np.ndarray], Optional[np.ndarray],
                     Optional[pd.DataFrame]]:
    """Carrega uma pasta de imagens (uma subpasta por classe, ou pasta unica
    com arquivos soltos como fallback) e extrai features de cor (+ textura,
    se pedido) — mesmo contrato de retorno de `dados_io.load_data`:
        (wavenumbers, X, rotulos, conc, mae_id, metadados_df)
    `wavenumbers` aqui e um indice simbolico (ver limitacao no docstring do
    modulo); `conc` e sempre None (sem quantificacao neste prototipo).
    `mae_id` depende do nivel de garantia de agrupamento detectado (ver
    docstring do modulo) -- real p/ "high"/"medium", None p/ "none". O
    nivel detectado tambem vai em `metadados_df.attrs["grouping_guarantee"]`
    (o contrato de retorno de 6 posicoes nao muda; o nivel viaja junto do
    DataFrame de metadados que ja' faz parte do contrato).

    `referencia` (x0, y0, x1, y1 relativos do cartao cinza/branco) e `segmentacao`
    ("nenhuma" | "otsu" | "distancia_fundo"): ver "NORMALIZACAO POR CARTAO E
    SEGMENTACAO" no docstring do modulo; os avisos agregados vao em
    `metadados_df.attrs["avisos"]`."""
    if segmentacao not in SEGMENTACOES_IMAGEM:
        raise ValueError(f"segmentacao '{segmentacao}' invalida; use {list(SEGMENTACOES_IMAGEM)}")
    subpastas = _detectar_subpastas_imagem(pasta)
    if subpastas:
        arquivos: List[Tuple[str, str]] = []
        for sp in subpastas:
            classe = os.path.basename(sp)
            diretos = _listar_arquivos_imagem(sp)
            if diretos:
                arquivos.extend((a, classe) for a in diretos)
            else:
                # nivel "high" em potencial: sem imagem direta, mas com
                # subpastas de amostra fisica dentro da subpasta de classe.
                for n in sorted(os.listdir(sp)):
                    caminho_amostra = os.path.join(sp, n)
                    if os.path.isdir(caminho_amostra):
                        arquivos.extend(
                            (a, classe)
                            for a in _listar_arquivos_imagem(caminho_amostra))
    else:
        if not os.path.isdir(pasta):
            raise FileNotFoundError(
                f"Pasta nao existe: {pasta}\n"
                f"  -> confira cfg.input_folder (mode='imagem').")
        arqs = _listar_arquivos_imagem(pasta)
        if not arqs:
            raise FileNotFoundError(
                f"Pasta existe mas nao contem imagens conhecidas "
                f"({', '.join(_EXTENSOES_IMAGEM)}).\n  Pasta: {pasta}")
        arquivos = [(a, "") for a in arqs]

    todos_arquivos = [a for a, _ in arquivos]
    grupos_high = _detectar_nivel_high(subpastas)
    if grupos_high is not None:
        nivel_agrupamento = _GROUPING_HIGH
        mapa_grupo = grupos_high
    else:
        grupos_medium = _detectar_nivel_medium(pasta, todos_arquivos)
        if grupos_medium is not None:
            nivel_agrupamento = _GROUPING_MEDIUM
            mapa_grupo = grupos_medium
        else:
            nivel_agrupamento = _GROUPING_NONE
            mapa_grupo = {}

    if nivel_agrupamento == _GROUPING_NONE:
        print("[WARNING] Grouping guarantee: NONE -- nenhuma subpasta por "
              "amostra fisica nem CSV de associacao (amostras.csv) foi "
              "encontrado. Validacao cai em StratifiedKFold, SEM protecao "
              "contra vazamento entre fotos da mesma amostra. Resultados "
              "devem ser tratados como exploratorios. Ver docstring de "
              "dados_imagem.py para os niveis 'high'/'medium'.")
    else:
        print(f"[INFO] Grouping guarantee: {nivel_agrupamento.upper()} "
              f"({len(set(mapa_grupo.values()))} grupos de amostra fisica).")

    linhas: List[np.ndarray] = []
    rotulos: List[str] = []
    grupos_arr: List[Optional[str]] = []
    meta_rows: List[Dict[str, object]] = []
    n_falhos = 0

    if incluir_textura and not extract_texture_features(np.zeros((8, 8), dtype=np.uint8)):
        incluir_textura = False        # scikit-image ausente: segue so' com as features de cor (o aviso ja foi impresso)
    nomes_features = list(_NOMES_FEATURES_COR) + (
        list(_NOMES_FEATURES_TEXTURA) if incluir_textura else [])

    infos_foto: List[Dict[str, Any]] = []
    infos_ok: List[Dict[str, Any]] = []
    for arq, subpasta_nome in arquivos:
        try:
            img = load_image_file(arq)
            feats, info = extrair_features_foto(
                img, recorte=caixa_recorte, referencia=referencia, segmentacao=segmentacao,
                incluir_textura=incluir_textura)
            infos_foto.append(info)
        except Exception as e:  # noqa: BLE001 -- parsing defensivo de imagem
            # externa (formato/tamanho variavel); erro impresso COM NOME DO
            # ARQUIVO e contabilizado em n_falhos, nunca silencioso.
            n_falhos += 1
            print(f"  [ERROR] {os.path.basename(arq)}: {e}")
            continue

        vetor = np.array([feats[n] for n in nomes_features], dtype=float)
        linhas.append(vetor)
        classe = subpasta_nome or os.path.splitext(os.path.basename(arq))[0]
        rotulos.append(classe)
        grupos_arr.append(mapa_grupo.get(arq))
        meta_rows.append({"arquivo": os.path.basename(arq),
                           "subpasta": subpasta_nome, "especie": classe,
                           "grupo_id": mapa_grupo.get(arq)})
        infos_ok.append(infos_foto[-1])

    if not linhas:
        raise ValueError(f"Nenhuma imagem valida carregada ({n_falhos} com erro).")
    if n_falhos > 0:
        print(f"[WARNING] {n_falhos} imagens com erro de leitura — puladas.")

    X = np.array(linhas, dtype=float)
    wavenumbers = np.arange(len(nomes_features), dtype=float)
    mae_id = (np.array(grupos_arr, dtype=object)
              if nivel_agrupamento != _GROUPING_NONE else None)
    metadados_df = pd.DataFrame(meta_rows)
    metadados_df.attrs["grouping_guarantee"] = nivel_agrupamento
    avisos = _avisos_da_leitura(infos_ok, np.array(rotulos, dtype=str), referencia, segmentacao,
                                caixa_recorte)
    for a in avisos:
        print(f"[AVISO] imagem: {a}")
    metadados_df.attrs["avisos"] = avisos
    print(f"[INFO] {len(X)} imagens carregadas, {len(nomes_features)} "
          f"features ({'cor+textura' if incluir_textura else 'cor'}).")

    return (wavenumbers, X, np.array(rotulos, dtype=str), None, mae_id,
            metadados_df)
