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

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Nomes das features de cor, na ordem em que `extract_color_features` as monta —
# usado tambem como "wavenumbers" simbolicos (ver limitacao no docstring do modulo).
NOMES_FEATURES_COR: Tuple[str, ...] = (
    "R_media", "G_media", "B_media", "R_dp", "G_dp", "B_dp",
    "H_media", "S_media", "V_media", "H_dp", "S_dp", "V_dp",
    "L_media", "a_media", "b_media", "L_dp", "a_dp", "b_dp",
)
NOMES_FEATURES_TEXTURA: Tuple[str, ...] = (
    "GLCM_contraste", "GLCM_homogeneidade", "GLCM_energia", "GLCM_correlacao",
)


def load_image_file(caminho: str) -> np.ndarray:
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


def extract_color_features(img: np.ndarray) -> Dict[str, float]:
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


def _listar_arquivos_imagem(pasta: str) -> List[str]:
    """Busca arquivos de imagem por extensao. Usa um set p/ deduplicar: em
    sistemas de arquivo case-insensitive (Windows, macOS default), buscar
    "*.png" e "*.PNG" separadamente devolve o MESMO arquivo duas vezes."""
    encontrados: set = set()
    for ext in _EXTENSOES_IMAGEM:
        encontrados.update(glob.glob(os.path.join(pasta, f"*{ext}")))
        encontrados.update(glob.glob(os.path.join(pasta, f"*{ext.upper()}")))
    return sorted(encontrados)


def _tem_imagem_direta_ou_em_subpasta(caminho: str) -> bool:
    """True se `caminho` tem imagem solta OU (nivel "high") subpastas de
    amostra que por sua vez tem imagem -- sem isso, uma classe organizada
    em subpasta-por-amostra seria invisivel para `_detectar_subpastas_imagem`
    (a classe pareceria vazia, ja que so' checava imagem DIRETA)."""
    if _listar_arquivos_imagem(caminho):
        return True
    return any(os.path.isdir(os.path.join(caminho, n))
               and _listar_arquivos_imagem(os.path.join(caminho, n))
               for n in os.listdir(caminho))


def _detectar_subpastas_imagem(raiz: str) -> List[str]:
    """Subpastas (1 por classe) que contem >=1 arquivo de imagem, direto ou
    dentro de subpasta de amostra (nivel "high") — mesma convencao do mode
    .dx (`_detectar_subpastas_classe` em dados_io.py) generalizada para 1
    nivel extra opcional."""
    if not os.path.isdir(raiz):
        return []
    subpastas = []
    for nome in sorted(os.listdir(raiz)):
        caminho = os.path.join(raiz, nome)
        if os.path.isdir(caminho) and _tem_imagem_direta_ou_em_subpasta(caminho):
            subpastas.append(caminho)
    return subpastas


#: Nome do CSV de associacao manual (nivel "medium"), procurado na raiz da
#: pasta de dados. Nao configuravel neste prototipo -- ver docstring do modulo.
NOME_CSV_AMOSTRAS = "amostras.csv"

GROUPING_HIGH = "high"
GROUPING_MEDIUM = "medium"
GROUPING_NONE = "none"


def _subpasta_e_grupo_de_amostras(caminho_classe: str) -> bool:
    """True se `caminho_classe` contem SO' subpastas (cada uma = 1 amostra
    fisica), nunca arquivo de imagem solto. False se tiver ao menos 1
    arquivo solto (mistura de niveis nao e' suportada -- ambigua)."""
    entradas = [os.path.join(caminho_classe, n)
                for n in os.listdir(caminho_classe)]
    arquivos_soltos = [e for e in entradas if os.path.isfile(e)
                       and e.lower().endswith(_EXTENSOES_IMAGEM)]
    subpastas_amostra = [e for e in entradas if os.path.isdir(e)
                         and _listar_arquivos_imagem(e)]
    return not arquivos_soltos and bool(subpastas_amostra)


def _detectar_nivel_high(subpastas_classe: List[str]
                          ) -> Optional[Dict[str, str]]:
    """Nivel "high": cada subpasta de CLASSE contem so' subpastas de
    AMOSTRA FISICA (nunca arquivo solto). Se TODA subpasta de classe
    satisfizer isso, devolve {caminho_arquivo: grupo_id}; senao None (cai
    p/ nivel "medium"). Grupo_id e' qualificado por classe
    ("Classe/Amostra") p/ nunca colidir entre classes com o mesmo nome de
    amostra."""
    if not subpastas_classe:
        return None
    if not all(_subpasta_e_grupo_de_amostras(sp) for sp in subpastas_classe):
        return None
    grupos: Dict[str, str] = {}
    for sp in subpastas_classe:
        classe = os.path.basename(sp)
        for nome_amostra in sorted(os.listdir(sp)):
            caminho_amostra = os.path.join(sp, nome_amostra)
            if not (os.path.isdir(caminho_amostra)
                    and _listar_arquivos_imagem(caminho_amostra)):
                continue
            grupo_id = f"{classe}/{nome_amostra}"
            for arq in _listar_arquivos_imagem(caminho_amostra):
                grupos[arq] = grupo_id
    return grupos


def _detectar_nivel_medium(pasta_raiz: str, arquivos: List[str]
                            ) -> Optional[Dict[str, str]]:
    """Nivel "medium": CSV `amostras.csv` na raiz da pasta de dados,
    colunas `arquivo,id_amostra`. `arquivo` e' o caminho RELATIVO a
    `pasta_raiz` (separador "/", como grava `os.path.relpath` normalizado).
    Cobertura parcial e' erro explicito -- nunca processamento parcial."""
    caminho_csv = os.path.join(pasta_raiz, NOME_CSV_AMOSTRAS)
    if not os.path.isfile(caminho_csv):
        return None
    df_csv = pd.read_csv(caminho_csv)
    colunas_faltando = {"arquivo", "id_amostra"} - set(df_csv.columns)
    if colunas_faltando:
        raise ValueError(
            f"{caminho_csv}: faltam as colunas {sorted(colunas_faltando)}. "
            f"Esperado: 'arquivo,id_amostra' (uma linha por imagem).")
    mapa_csv = {str(r["arquivo"]).replace("\\", "/"): str(r["id_amostra"])
                for _, r in df_csv.iterrows()}
    rel = {arq: os.path.relpath(arq, pasta_raiz).replace("\\", "/")
           for arq in arquivos}
    sem_cobertura = [rel[a] for a in arquivos if rel[a] not in mapa_csv]
    if sem_cobertura:
        raise ValueError(
            f"{caminho_csv} existe mas nao cobre {len(sem_cobertura)} "
            f"imagem(ns) do dataset -- nenhuma foi processada. Arquivos "
            f"sem entrada no CSV: {', '.join(sorted(sem_cobertura))}")
    return {arq: mapa_csv[rel[arq]] for arq in arquivos}


def load_images(
        pasta: str,
        caixa_recorte: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        incluir_textura: bool = False,
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
    DataFrame de metadados que ja' faz parte do contrato)."""
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
        nivel_agrupamento = GROUPING_HIGH
        mapa_grupo = grupos_high
    else:
        grupos_medium = _detectar_nivel_medium(pasta, todos_arquivos)
        if grupos_medium is not None:
            nivel_agrupamento = GROUPING_MEDIUM
            mapa_grupo = grupos_medium
        else:
            nivel_agrupamento = GROUPING_NONE
            mapa_grupo = {}

    if nivel_agrupamento == GROUPING_NONE:
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

    nomes_features = list(NOMES_FEATURES_COR) + (
        list(NOMES_FEATURES_TEXTURA) if incluir_textura else [])

    for arq, subpasta_nome in arquivos:
        try:
            img = load_image_file(arq)
            img = recortar_relativo(img, caixa_recorte)
            feats = extract_color_features(img)
            if incluir_textura:
                feats.update(extract_texture_features(img))
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

    if not linhas:
        raise ValueError(f"Nenhuma imagem valida carregada ({n_falhos} com erro).")
    if n_falhos > 0:
        print(f"[WARNING] {n_falhos} imagens com erro de leitura — puladas.")

    X = np.array(linhas, dtype=float)
    wavenumbers = np.arange(len(nomes_features), dtype=float)
    mae_id = (np.array(grupos_arr, dtype=object)
              if nivel_agrupamento != GROUPING_NONE else None)
    metadados_df = pd.DataFrame(meta_rows)
    metadados_df.attrs["grouping_guarantee"] = nivel_agrupamento
    print(f"[INFO] {len(X)} imagens carregadas, {len(nomes_features)} "
          f"features ({'cor+textura' if incluir_textura else 'cor'}).")

    return (wavenumbers, X, np.array(rotulos, dtype=str), None, mae_id,
            metadados_df)
