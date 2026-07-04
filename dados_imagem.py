"""
dados_imagem.py — Extracao de dados quimiometricos a partir de IMAGENS
DIGITAIS (colorimetria digital / Digital Image-Based Analysis).

Protótipo GENERICO (2026-07): converte cada imagem numa matriz de "sinal"
usando estatisticas de cor (RGB/HSV/Lab) — e opcionalmente textura (GLCM,
requer scikit-image) — analogo a um espectro: cada imagem vira UMA LINHA da
matriz X, cada estatistica de canal vira UMA VARIAVEL (coluna), exatamente
como cada comprimento de onda e uma variavel no modo .dx. A partir dai, TODA
a maquinaria quimiometrica existente (PCA, PLS-DA, DD-SIMCA, OPLS-DA, selecao
de variaveis, figuras de merito) funciona SEM alteracao — essas funcoes so
enxergam uma matriz numerica, nao sabem se a coluna 47 e um comprimento de
onda ou o canal G medio de uma foto.

Convencao de pastas: MESMA do modo .dx — uma subpasta por classe (ou pasta
unica com arquivos soltos, fallback). Extensoes aceitas: .jpg/.jpeg/.png/
.bmp/.tif/.tiff.

Limitacao conhecida deste prototipo: o "eixo de variaveis" retornado
(equivalente a `wavenumbers`) e apenas `np.arange(n_features)` — os graficos
que rotulam o eixo X como comprimento de onda (VIP, loadings, etc.) vao
mostrar indices numericos em vez do nome da feature (ex. "R_mean"). Rotular
esse eixo com os nomes reais das features e uma extensao futura (afeta varias
funcoes de figura em figuras.py, fora do escopo deste prototipo).

mae_id/concentracao: nao ha convencao de metadado equivalente ao ##TITLE=
do JCAMP-DX para imagens genericas — mae_id fica None (sem agrupamento de
replicas) e conc fica None (sem quantificacao) neste prototipo. Rotulos vem
do nome da subpasta (mesma convencao do modo .dx).

IMPORTANTE — pre-processamento: use `preprocessamento_padrao="autoscaling"`
(ou "mc") no modo="imagem", NUNCA os presets com Savitzky-Golay
("msc_sg_mc"/"snv_sg_mc"). MSC e SG pressupoem um sinal espectral CONTINUO
ao longo do eixo de variaveis (comprimento de onda) — nao fazem sentido
cientifico p/ um vetor curto de estatisticas de cor discretas e heterogeneas
(H fica em [0,1], L*a*b* fica em dezenas/centenas) e o SG especificamente
EXIGE janela <= numero de variaveis (18 por padrao — sem textura), o que
pode nem ser satisfeito.

IMPORTANTE — faixa espectral: `carregar_dados()` aplica o mesmo filtro
wn_min/wn_max do modo .dx sobre o eixo simbolico (indices 0..n_features-1).
Os defaults de Config (wn_min=4000, wn_max=10000) NAO cobrem esse intervalo
pequeno e descartariam TODAS as variaveis — ajuste wn_min/wn_max (ex.:
wn_min=-1, wn_max=100) ao usar modo="imagem".
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Nomes das features de cor, na ordem em que `extrair_features_cor` as monta —
# usado tambem como "wavenumbers" simbolicos (ver limitacao no docstring do modulo).
NOMES_FEATURES_COR: Tuple[str, ...] = (
    "R_media", "G_media", "B_media", "R_dp", "G_dp", "B_dp",
    "H_media", "S_media", "V_media", "H_dp", "S_dp", "V_dp",
    "L_media", "a_media", "b_media", "L_dp", "a_dp", "b_dp",
)
NOMES_FEATURES_TEXTURA: Tuple[str, ...] = (
    "GLCM_contraste", "GLCM_homogeneidade", "GLCM_energia", "GLCM_correlacao",
)


def carregar_imagem_arquivo(caminho: str) -> np.ndarray:
    """Le uma imagem do disco como array RGB uint8 (H, W, 3), via Pillow."""
    from PIL import Image
    with Image.open(caminho) as im:
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


def extrair_features_cor(img: np.ndarray) -> Dict[str, float]:
    """Media e desvio-padrao por canal em RGB, HSV e Lab — 18 features no
    total, na mesma ordem de `NOMES_FEATURES_COR`. Entrada: array uint8
    (H, W, 3) ou (H, W) RGB/tons de cinza."""
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img01 = img[..., :3].astype(np.float64) / 255.0

    hsv = _rgb_para_hsv(img01)
    lab = _rgb_para_lab(img01)

    feats: Dict[str, float] = {}
    for nome, canal in zip(("R", "G", "B"), range(3)):
        feats[f"{nome}_media"] = float(np.mean(img01[..., canal])) * 255.0
        feats[f"{nome}_dp"] = float(np.std(img01[..., canal])) * 255.0
    for nome, canal in zip(("H", "S", "V"), range(3)):
        feats[f"{nome}_media"] = float(np.mean(hsv[..., canal]))
        feats[f"{nome}_dp"] = float(np.std(hsv[..., canal]))
    for nome, canal in zip(("L", "a", "b"), range(3)):
        feats[f"{nome}_media"] = float(np.mean(lab[..., canal]))
        feats[f"{nome}_dp"] = float(np.std(lab[..., canal]))
    return {k: feats[k] for k in NOMES_FEATURES_COR}


def extrair_features_textura(img: np.ndarray) -> Dict[str, float]:
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


def _listar_arquivos_imagem(pasta: str) -> List[str]:
    """Busca arquivos de imagem por extensao. Usa um set p/ deduplicar: em
    sistemas de arquivo case-insensitive (Windows, macOS default), buscar
    "*.png" e "*.PNG" separadamente devolve o MESMO arquivo duas vezes."""
    encontrados: set = set()
    for ext in _EXTENSOES_IMAGEM:
        encontrados.update(glob.glob(os.path.join(pasta, f"*{ext}")))
        encontrados.update(glob.glob(os.path.join(pasta, f"*{ext.upper()}")))
    return sorted(encontrados)


def _detectar_subpastas_imagem(raiz: str) -> List[str]:
    """Subpastas (1 por classe) que contem >=1 arquivo de imagem — mesma
    convencao do modo .dx (`_detectar_subpastas_classe` em dados_io.py)."""
    if not os.path.isdir(raiz):
        return []
    subpastas = []
    for nome in sorted(os.listdir(raiz)):
        caminho = os.path.join(raiz, nome)
        if os.path.isdir(caminho) and _listar_arquivos_imagem(caminho):
            subpastas.append(caminho)
    return subpastas


def carregar_imagens(
        pasta: str,
        caixa_recorte: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        incluir_textura: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                     Optional[np.ndarray], Optional[np.ndarray],
                     Optional[pd.DataFrame]]:
    """Carrega uma pasta de imagens (uma subpasta por classe, ou pasta unica
    com arquivos soltos como fallback) e extrai features de cor (+ textura,
    se pedido) — mesmo contrato de retorno de `dados_io.carregar_dados`:
        (wavenumbers, X, rotulos, conc, mae_id, metadados_df)
    `wavenumbers` aqui e um indice simbolico (ver limitacao no docstring do
    modulo); `conc` e `mae_id` sao sempre None neste prototipo generico
    (sem convencao de metadado equivalente ao ##TITLE= do JCAMP-DX)."""
    subpastas = _detectar_subpastas_imagem(pasta)
    if subpastas:
        arquivos: List[Tuple[str, str]] = []
        for sp in subpastas:
            arquivos.extend((a, os.path.basename(sp))
                             for a in _listar_arquivos_imagem(sp))
    else:
        if not os.path.isdir(pasta):
            raise FileNotFoundError(
                f"Pasta nao existe: {pasta}\n"
                f"  -> confira cfg.pasta_entrada (modo='imagem').")
        arqs = _listar_arquivos_imagem(pasta)
        if not arqs:
            raise FileNotFoundError(
                f"Pasta existe mas nao contem imagens conhecidas "
                f"({', '.join(_EXTENSOES_IMAGEM)}).\n  Pasta: {pasta}")
        arquivos = [(a, "") for a in arqs]

    linhas: List[np.ndarray] = []
    rotulos: List[str] = []
    meta_rows: List[Dict[str, object]] = []
    n_falhos = 0

    nomes_features = list(NOMES_FEATURES_COR) + (
        list(NOMES_FEATURES_TEXTURA) if incluir_textura else [])

    for arq, subpasta_nome in arquivos:
        try:
            img = carregar_imagem_arquivo(arq)
            img = recortar_relativo(img, caixa_recorte)
            feats = extrair_features_cor(img)
            if incluir_textura:
                feats.update(extrair_features_textura(img))
        except Exception as e:
            n_falhos += 1
            print(f"  [ERROR] {os.path.basename(arq)}: {e}")
            continue

        vetor = np.array([feats[n] for n in nomes_features], dtype=float)
        linhas.append(vetor)
        classe = subpasta_nome or os.path.splitext(os.path.basename(arq))[0]
        rotulos.append(classe)
        meta_rows.append({"arquivo": os.path.basename(arq),
                           "subpasta": subpasta_nome, "especie": classe})

    if not linhas:
        raise ValueError(f"Nenhuma imagem valida carregada ({n_falhos} com erro).")
    if n_falhos > 0:
        print(f"[WARNING] {n_falhos} imagens com erro de leitura — puladas.")

    X = np.array(linhas, dtype=float)
    wavenumbers = np.arange(len(nomes_features), dtype=float)
    metadados_df = pd.DataFrame(meta_rows)
    print(f"[INFO] {len(X)} imagens carregadas, {len(nomes_features)} "
          f"features ({'cor+textura' if incluir_textura else 'cor'}).")

    return (wavenumbers, X, np.array(rotulos, dtype=str), None, None,
            metadados_df)
