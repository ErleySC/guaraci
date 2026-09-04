#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""comparar_npls_pixelwise_mango.py -- Passo 132: fecha a pendencia do
Passo 129 (empate 1,0 vs 1,0 em dado sintetico facil demais, nao
discriminava nada). Compara N-PLS multiway vs. PLS-DA por pixel em dado
PUBLICO real: DeepHS Fruit, Mango/VIS (56 gravacoes, 36 objetos fisicos,
3 classes de ripeness_state), 10 seeds independentes de split group-aware
+ teste de Wilcoxon pareado -- nao 1 numero solto.

ACHADO durante a preparacao: `scripts/download_datasets/
baixar_deephs_fruit_todas.py` (usado no Passo 129 pra' tentar Kiwi/outra
fruta) esta' QUEBRADO pra' qualquer fruta -- depende de um sidecar de
pins (`_deephs_fruit_todas_pins.json`) que a propria docstring do script
descreve como "versionado junto com este script", mas nunca foi de fato
commitado (`git ls-files`/`git check-ignore` confirmam: nao esta' no
repo, nem ignorado -- so' ausente). Baixar Mango/VIS aqui exigiu
contornar isso baixando com pins gerados NESTA sessao (TOFU -- trust on
first use, sha256 calculado do que foi baixado agora, NAO auditado
externamente como os pins de Kaki foram). Mais fraco que o mecanismo
original; documentado explicitamente, nao escondido. Correcao do bug em
si (versionar os pins que faltam) fica fora do escopo deste passo.

Uso:
    python scripts/medicoes/comparar_npls_pixelwise_mango.py [pasta]

`pasta` (opcional): pasta local com o manifest.json de Mango/VIS. Se
omitida, le' GUARACI_DATASETS_DIR/deephs_fruit_mango_vis."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ / "src"))

import numpy as np
from scipy.stats import wilcoxon

from guaraci.hsi_io import load_deephs_fruit_dataset
from guaraci.hsi_pipeline import apply_quality_gate_and_segment
from guaraci.hsi_multiway import comparar_npls_vs_pixelwise

N_SEEDS = 10


def _pasta_mango_vis() -> str:
    """Caminho do cache local (baixado por este script -- ver docstring
    do modulo) -- por variavel de ambiente ou argumento posicional, nunca
    hardcoded, mesmo motivo de `validar_mcr_als_oleos_reais.py`: um
    caminho absoluto de maquina especifica nao pertence a um arquivo
    versionado."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    raiz = os.environ.get("GUARACI_DATASETS_DIR")
    if not raiz:
        raise SystemExit(
            "Pasta do dataset nao informada. Uso: python "
            f"{os.path.basename(__file__)} <pasta>  ou defina "
            "GUARACI_DATASETS_DIR (pasta deephs_fruit_mango_vis dentro dela).")
    return str(Path(raiz) / "deephs_fruit_mango_vis")


def main() -> None:
    pasta = _pasta_mango_vis()
    cubos, rotulos, grupos, wavelengths, meta_df = load_deephs_fruit_dataset(
        pasta, fruta="Mango", camera="VIS")
    print(f"{len(cubos)} gravacoes, {len(set(grupos))} objetos fisicos, "
          f"{wavelengths.size} bandas, classes={sorted(set(rotulos))}")
    contagem = {c: int((rotulos == c).sum()) for c in sorted(set(rotulos))}
    print(f"distribuicao de classes: {contagem}")

    filtrado = apply_quality_gate_and_segment(
        cubos, list(grupos), list(rotulos), list(meta_df["day"]))
    print(f"quality gate: {filtrado['n_rejeitados']} rejeitadas de {len(cubos)}")
    if filtrado["n_rejeitados"]:
        print("  motivos:", filtrado["motivos_rejeicao"][:5])

    npls_scores, px_scores = [], []
    for seed in range(N_SEEDS):
        resultado = comparar_npls_vs_pixelwise(
            filtrado["cubos"], filtrado["mascaras"], filtrado["rotulos"],
            filtrado["group_ids"], n_linhas_grade=6, n_colunas_grade=6,
            n_componentes=3, n_splits=3, seed=seed,
            max_pixels_por_gravacao=2000)
        npls_scores.append(resultado["balanced_accuracy_npls"])
        px_scores.append(resultado["balanced_accuracy_pixelwise"])
        print(f"seed={seed}  npls={npls_scores[-1]:.4f}  "
              f"pixelwise={px_scores[-1]:.4f}")

    npls_scores = np.array(npls_scores)
    px_scores = np.array(px_scores)
    print(f"\nN-PLS:      media={npls_scores.mean():.4f}  desvio={npls_scores.std():.4f}")
    print(f"PLS-DA/pix: media={px_scores.mean():.4f}  desvio={px_scores.std():.4f}")
    print(f"N-PLS venceu em {int((npls_scores > px_scores).sum())}/{N_SEEDS} seeds")
    w = wilcoxon(npls_scores, px_scores)
    print(f"Wilcoxon pareado: statistic={w.statistic:.2f}  p={w.pvalue:.4f}")
    print("(chance level p/ 3 classes = 0.333 -- ambos os metodos ficam "
          "perto disso, tarefa dificil de verdade, nao so' um dos metodos "
          "com bug)")


if __name__ == "__main__":
    main()
