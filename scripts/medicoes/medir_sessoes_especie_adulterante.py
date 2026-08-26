# -*- coding: utf-8 -*-
"""Passo 64 (revisao 2026-08-26). Persiste, como script reprodutivel, a
contagem de SESSOES DE COLETA independentes por combinacao especie x
adulterante -- o numero citado em docs/MANUAL.md e no achado de
`dados_io.session_from_mae_id` (Bloco 9b): 36 das 38 combinacoes tem
exatamente 1 sessao, 2 tem exatamente 2 (Andiroba x soja, Maracuja x
algodao).

Ate' esta correcao, essa contagem so' tinha sido verificada ad hoc (heredoc
Python fora do repositorio, contra o dataset real) -- verificada duas
vezes e bateu, mas sem artefato reproduzivel versionado. Este script fecha
essa lacuna, usando `session_from_mae_id` (nao mae_id bruto -- ver
docstring da funcao em dados_io.py para o motivo: mae_id tem um token por
NIVEL DE TEOR, nao por sessao, e contar bruto infla o n).

Uso:
    python scripts/medicoes/medir_sessoes_especie_adulterante.py [pasta_dados]

`pasta_dados` (opcional): pasta raiz do dataset .dx. Se omitido, le' a
variavel de ambiente GUARACI_DADOS_REAIS. Nunca hardcoded aqui -- mesmo
padrao ja estabelecido em medir_especie_vs_adulterante_permanova.py.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "src")
from guaraci.config import Config  # noqa: E402
from guaraci.dados_io import (  # noqa: E402
    adulterant_from_mae_id,
    load_data,
    session_from_mae_id,
)


def _pasta_dados() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    caminho = os.environ.get("GUARACI_DADOS_REAIS")
    if not caminho:
        raise SystemExit(
            "Pasta de dados nao informada. Uso: python "
            f"{os.path.basename(__file__)} <pasta_dados>  ou defina a "
            "variavel de ambiente GUARACI_DADOS_REAIS.")
    return caminho


def main():
    pasta = _pasta_dados()
    cfg = Config()
    cfg.input_folder = pasta
    cfg.mode = "dx"

    print("=" * 72)
    print("Passo 64: sessoes de coleta independentes por especie x adulterante")
    print("=" * 72)
    print(f"  pasta de dados: {pasta}")

    _wn, _X, rot, conc, mae, _meta = load_data(cfg)
    conc0 = np.where(np.isnan(np.asarray(conc, dtype=float)), 0.0,
                      np.asarray(conc, dtype=float))
    adult = np.array([adulterant_from_mae_id(m) for m in mae], dtype=object)
    sessao = np.array([session_from_mae_id(m) for m in mae], dtype=str)

    especies = sorted({str(r) for r in rot})
    adulterantes = sorted({a for a in adult if a})

    combos_n: dict = {}
    for esp in especies:
        for adu in adulterantes:
            mask = (rot == esp) & (adult == adu) & (conc0 > 0.0)
            n_amostras = int(mask.sum())
            if n_amostras == 0:
                continue
            n_sessoes = len({s for s in sessao[mask]})
            combos_n[(esp, adu)] = (n_amostras, n_sessoes)

    contagem = Counter(n_sess for _n_am, n_sess in combos_n.values())
    print(f"  total de combinacoes com dados: {len(combos_n)}")
    print(f"  distribuicao de n_sessoes: {dict(sorted(contagem.items()))}")
    print()
    for (esp, adu), (n_am, n_sess) in sorted(combos_n.items()):
        print(f"    {esp:20s} x {adu:10s}  n_amostras={n_am:3d}  "
              f"n_sessoes={n_sess}")

    print()
    n1 = contagem.get(1, 0)
    n2 = contagem.get(2, 0)
    print("  Citado em docs/MANUAL.md: 36 combinacoes com 1 sessao, "
          "2 com 2 sessoes (Andiroba x soja, Maracuja x algodao).")
    print(f"  Medido agora:              {n1} combinacoes com 1 sessao, "
          f"{n2} com 2 sessoes.")
    combos_n2 = sorted(k for k, (_n_am, n_sess) in combos_n.items()
                        if n_sess == 2)
    print(f"  Combinacoes com 2 sessoes: {combos_n2}")
    if n1 == 36 and n2 == 2:
        print("  CONFIRMADO -- bate exatamente com o numero ja documentado.")
    else:
        print("  DIVERGENCIA -- nao bate com o numero documentado; tratar "
              "como achado novo, mesmo protocolo do Passo 63.")


if __name__ == "__main__":
    main()
