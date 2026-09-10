# -*- coding: utf-8 -*-
"""P0 (revisao 2026-08-26). Remedicao, do zero, de "espécie explica mais a
direção do desvio espectral de uma amostra adulterada que o tipo de
adulterante" -- a alegação que fundamentou a decisão de design do Bloco 9
(Identificar calibrado por combinação espécie×adulterante, não por
adulterante agregado entre espécies).

MOTIVO: `docs/MANUAL.md` (linhas 813-816, escrito ANTES desta sessão, parte
do design do Bloco 9a) citava uma tabela de R² "tipo ANOVA/PERMANOVA" (por
adulterante: 0,0034/0,0140; por espécie: 0,1866/0,1034/0,2182/0,1147) e a
frase "espécie explica 6 a 13x mais variância do delta que o tipo de
adulterante". Varredura de lastro (Passo 62) não encontrou NENHUM script em
`scripts/medicoes/` nem nos ~25 scripts privados de
`~/.guaraci_local/auditoria_privada/` que produzisse esses números. Pior:
a aritmética do próprio texto não fecha -- das 4 razões possíveis na
tabela, 0,1866/0,0034=54,9x | 0,1034/0,0034=30,4x | 0,2182/0,0140=15,6x |
0,1147/0,0140=8,2x -- NENHUMA está entre 6 e 13x.

METODOLOGIA (reconstruída -- o texto original não especifica a fórmula
exata usada; esta é a interpretação mais defensável do que ele descreve,
não uma tentativa de forçar bater com o número antigo):

1. Espectro PRÉ-PROCESSADO (preset padrão do projeto, `cfg.
   default_preprocessing`), não bruto -- espectro bruto é dominado por
   diferenças de linha de base/espalhamento entre espécies que não têm
   nada a ver com química de adulteração; comparar "direção do desvio"
   em espaço bruto confundiria instrumentação com química.
2. Réplicas técnicas (T1/T2/T3, mesmo `mae_id`) colapsadas para a MÉDIA
   antes de qualquer cálculo -- mesmo tratamento anti-vazamento/anti-
   pseudo-replicação usado no resto do projeto.
3. `delta_i = X_i - média(X | mesma espécie, PURO)` para cada amostra
   adulterada i (média de puros calculada nos mesmos dados colapsados).
4. R² tipo PERMANOVA one-way (Anderson 2001): particiona a soma de
   quadrados euclidiana total em entre-grupos e dentro-de-grupos,
   `R2 = SS_entre / SS_total` -- rodado UMA VEZ agrupando por ESPÉCIE
   (13 níveis) e UMA VEZ agrupando por ADULTERANTE (3 níveis), nos MESMOS
   deltas.
5. "Bruto" = R² sobre os deltas como estão (magnitude domina, correlaciona
   com teor). "Direção normalizada" = R² sobre os deltas normalizados
   para norma unitária (isola a FORMA/direção espectral do desvio,
   descontando "quanto" de adulterante).
6. Dois cortes de teor, para comparar com a citação original: >=0% (todas
   as amostras adulteradas) e >=10% (teor mais alto, onde o sinal químico
   deveria estar mais evidente).

Uso:
    python scripts/medicoes/medir_especie_vs_adulterante_permanova.py [pasta_dados]

`pasta_dados` (opcional): caminho para a pasta raiz do dataset .dx. Se
omitido, le' a variavel de ambiente GUARACI_DADOS_REAIS; se essa tambem
faltar, o script recusa rodar (nunca cai num caminho hardcoded do acervo
privado do autor).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "src")
from guaraci.config import Config  # noqa: E402
from guaraci.dados_io import adulterant_from_mae_id, load_data  # noqa: E402
from guaraci.preprocessamento import build_preprocessor  # noqa: E402


def _pasta_dados() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    caminho = os.environ.get("GUARACI_DADOS_REAIS")
    if not caminho:
        raise SystemExit(
            "Pasta de dados nao informada. Uso: python "
            f"{os.path.basename(__file__)} <pasta_dados>  ou defina a "
            "variavel de ambiente GUARACI_DADOS_REAIS. Nunca hardcoded "
            "aqui -- o caminho do acervo privado nao entra em codigo "
            "versionado.")
    return caminho


def _colapsar_por_grupo(X: np.ndarray, mae_id: np.ndarray) -> tuple:
    """Media por mae_id -- 1 linha por AMOSTRA FISICA, nao por espectro
    (replicas T1/T2/T3 nao contam 3x)."""
    grupos_unicos = np.unique(mae_id)
    X_col = np.array([X[mae_id == g].mean(axis=0) for g in grupos_unicos])
    return X_col, grupos_unicos


def _r2_permanova_oneway(X: np.ndarray, grupos: np.ndarray) -> float:
    """R2 tipo PERMANOVA one-way (Anderson 2001): SS_entre/SS_total, soma
    de quadrados euclidiana. Reduz ao R2 classico de MANOVA quando a
    distancia e' euclidiana (caso aqui)."""
    media_geral = X.mean(axis=0)
    ss_total = float(np.sum((X - media_geral) ** 2))
    if ss_total <= 0:
        return float("nan")
    ss_dentro = 0.0
    for g in np.unique(grupos):
        Xg = X[grupos == g]
        if len(Xg) < 2:
            continue   # grupo com 1 amostra nao contribui variancia "dentro"
        ss_dentro += float(np.sum((Xg - Xg.mean(axis=0)) ** 2))
    ss_entre = ss_total - ss_dentro
    return ss_entre / ss_total


def _normalizar_direcao(X: np.ndarray) -> np.ndarray:
    normas = np.linalg.norm(X, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return X / normas


def main():
    pasta = _pasta_dados()
    cfg = Config()
    cfg.input_folder = pasta
    cfg.mode = "dx"

    print("=" * 72)
    print("P0: R2 (tipo PERMANOVA) da direcao do delta -- especie vs adulterante")
    print("=" * 72)
    print(f"  pasta de dados: {pasta}")

    wn, X, rot, conc, mae, _meta = load_data(cfg)
    mask_wn = (wn >= cfg.wn_min) & (wn <= cfg.wn_max)
    X = X[:, mask_wn]

    preproc = build_preprocessor(cfg).fit(X)
    Xp = preproc.transform(X)

    Xp_col, mae_col_full = _colapsar_por_grupo(Xp, mae)
    # rotulo/conc/adulterante por GRUPO colapsado (todos os espectros do
    # mesmo mae_id compartilham especie/conc/adulterante por construcao).
    idx_rep = np.array([np.where(mae == g)[0][0] for g in mae_col_full])
    rot_col = rot[idx_rep]
    conc_col = np.asarray(conc, dtype=float)[idx_rep]
    conc_col = np.where(np.isnan(conc_col), 0.0, conc_col)
    adult_col = np.array(
        [adulterant_from_mae_id(m) for m in mae_col_full], dtype=object)

    print(f"  amostras fisicas (grupos mae_id) apos colapso: {len(Xp_col)}")
    print(f"  especies: {sorted(set(rot_col))}")
    print(f"  adulterantes: {sorted({a for a in adult_col if a})}")
    print()

    especies = sorted(set(rot_col))
    medias_puras = {}
    for esp in especies:
        mask_puro = (rot_col == esp) & (conc_col <= 0.0)
        if mask_puro.sum() == 0:
            continue
        medias_puras[esp] = Xp_col[mask_puro].mean(axis=0)

    resultados = {}
    for corte, rotulo_corte in ((0.0, ">=0%"), (10.0, ">=10%")):
        limiar = (conc_col >= corte) if corte > 0 else (conc_col > 0.0)
        mask_ad = (
            limiar
            & np.array([e in medias_puras for e in rot_col])
            & np.array([a is not None for a in adult_col]))
        n = int(mask_ad.sum())
        if n < 10:
            print(f"  [AVISO] corte {rotulo_corte}: apenas {n} amostras -- pulado")
            continue

        deltas = np.array([
            Xp_col[i] - medias_puras[rot_col[i]]
            for i in np.where(mask_ad)[0]])
        rot_ad = rot_col[mask_ad]
        adult_ad = adult_col[mask_ad]
        deltas_norm = _normalizar_direcao(deltas)

        r2_esp_bruto = _r2_permanova_oneway(deltas, rot_ad)
        r2_adu_bruto = _r2_permanova_oneway(deltas, adult_ad)
        r2_esp_norm = _r2_permanova_oneway(deltas_norm, rot_ad)
        r2_adu_norm = _r2_permanova_oneway(deltas_norm, adult_ad)

        resultados[rotulo_corte] = {
            "n": n, "r2_especie_bruto": r2_esp_bruto,
            "r2_adulterante_bruto": r2_adu_bruto,
            "r2_especie_norm": r2_esp_norm,
            "r2_adulterante_norm": r2_adu_norm,
        }

        print(f"  Corte {rotulo_corte} (n={n} amostras fisicas adulteradas):")
        print(f"    R2 por ADULTERANTE (bruto)     = {r2_adu_bruto:.4f}")
        print(f"    R2 por ESPECIE     (bruto)     = {r2_esp_bruto:.4f}")
        if r2_adu_bruto > 0:
            print(f"      razao especie/adulterante (bruto) = "
                  f"{r2_esp_bruto / r2_adu_bruto:.2f}x")
        print(f"    R2 por ADULTERANTE (direcao normalizada) = {r2_adu_norm:.4f}")
        print(f"    R2 por ESPECIE     (direcao normalizada) = {r2_esp_norm:.4f}")
        if r2_adu_norm > 0:
            print(f"      razao especie/adulterante (normalizada) = "
                  f"{r2_esp_norm / r2_adu_norm:.2f}x")
        print()

    print("  Comparacao com a tabela citada em docs/MANUAL.md (nao "
          "reproduzida por script ate' agora):")
    citado = {
        ">=0%": (0.0034, 0.1866, 0.1034),
        ">=10%": (0.0140, 0.2182, 0.1147),
    }
    for corte, (adu_cit, esp_bruto_cit, esp_norm_cit) in citado.items():
        if corte not in resultados:
            continue
        r = resultados[corte]
        print(f"    {corte}: citado adulterante={adu_cit:.4f} "
              f"especie(bruto)={esp_bruto_cit:.4f} "
              f"especie(norm)={esp_norm_cit:.4f}")
        print(f"    {corte}: medido  adulterante={r['r2_adulterante_bruto']:.4f} "
              f"especie(bruto)={r['r2_especie_bruto']:.4f} "
              f"especie(norm)={r['r2_especie_norm']:.4f}")


if __name__ == "__main__":
    main()
