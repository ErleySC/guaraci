#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""amostragem_ativa_oleos_reais.py -- Passo 138 (Bloco 25): teste de
sanidade contra o acervo real -- confirma que `priorizar_amostragem`
aponta pras combinacoes espécie x adulterante já conhecidas como
não-validáveis (36 de 38, ver `scripts/medicoes/medir_sessoes_especie_
adulterante.py` e `docs/MANUAL.md`).

Nao recalibra PCA/DD-SIMCA de verdade: `ConformalOneClass._colapsar_por_
grupo` reduz o escore a UM POR SESSAO (mediana) antes de checar
`achievable_alpha` -- ou seja, `cobertura_status` depende SO' do numero
de sessoes (`n_grupos`), nunca do valor dos escores em si. Este script
reproduz a MESMA logica de classificacao de
`identificacao.train_identification_ensemble` (n_grupos<=1 ->
NOT_VALIDATED_N1; achievable_alpha(n_grupos)<=alpha -> VALIDATED; senao
NOT_VALIDATED_N2_WEAK) usando so' `session_from_mae_id`, sem precisar
treinar PCA -- mais rapido, MESMO resultado de cobertura."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ / "src"))

from guaraci.amostragem_ativa import priorizar_amostragem
from guaraci.conformal import achievable_alpha
from guaraci.dados_io import load_dx, session_from_mae_id
from guaraci.identificacao import CoverageStatus

ALPHA = 0.05


def _pasta_acervo() -> str:
    caminho = os.environ.get("GUARACI_DADOS_REAIS")
    if not caminho:
        raise SystemExit("Defina GUARACI_DADOS_REAIS.")
    return caminho


def main() -> None:
    _wn, _X, _rot, _conc, mae, meta = load_dx(_pasta_acervo())

    ensemble = {}
    combos = meta[~meta["puro"]].groupby(["especie", "adulterante_nome"])
    for (esp, adult), grupo in combos:
        sessoes = {session_from_mae_id(m) for m in mae[grupo.index]}
        n_grupos = len(sessoes)
        if n_grupos <= 1:
            status = CoverageStatus.NOT_VALIDATED_N1
        elif achievable_alpha(n_grupos) <= ALPHA:
            status = CoverageStatus.VALIDATED
        else:
            status = CoverageStatus.NOT_VALIDATED_N2_WEAK
        ensemble[(esp, adult)] = {"n_grupos": n_grupos,
                                  "cobertura_status": status}

    n_validadas = sum(1 for i in ensemble.values()
                      if i["cobertura_status"] == CoverageStatus.VALIDATED)
    n_total = len(ensemble)
    print(f"{n_total} combinacoes especie x adulterante; "
          f"{n_validadas} validadas, {n_total - n_validadas} nao-validadas "
          f"(alpha={ALPHA})")

    lista = priorizar_amostragem(ensemble, alpha=ALPHA)
    nao_validadas_na_lista = [r for r in lista if r.prioridade > 0.0]
    print(f"priorizar_amostragem: {len(nao_validadas_na_lista)} "
          f"combinacoes com prioridade > 0 (nao-validadas)")

    print("\nTop 10 prioridades (mais perto de validar):")
    for r in lista[:10]:
        print(f"  {r.especie:20s} {r.adulterante:10s} "
              f"n_atual={r.n_sessoes_atual:2d}  faltam={r.sessoes_faltantes:2d}  "
              f"status={r.cobertura_status.value if r.cobertura_status else 'None':20s} "
              f"prioridade={r.prioridade:.4f}")

    print("\nUltimas 5 (menos urgentes / ja validadas):")
    for r in lista[-5:]:
        print(f"  {r.especie:20s} {r.adulterante:10s} "
              f"n_atual={r.n_sessoes_atual:2d}  status="
              f"{r.cobertura_status.value if r.cobertura_status else 'None':20s} "
              f"prioridade={r.prioridade:.4f}")


if __name__ == "__main__":
    main()
