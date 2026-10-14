# -*- coding: utf-8 -*-
"""Grava as replicas bootstrap/jackknife que `bootstrap_bca_ci` gera, para
o `gen_oracle.R` montar o IC BCa com `boot` (Canty & Ripley) a partir
EXATAMENTE das mesmas replicas -- o gerador de numeros aleatorios do numpy
nao existe no R, entao so' a reamostragem vem daqui; z0, a aceleracao e os
quantis sao do `boot`. O teste confere que as replicas continuam as mesmas
(se o RNG do numpy mudar, o teste falha avisando, em vez de comparar com
um oraculo de outra reamostragem).

Rodar a partir DESTE diretorio, ANTES do gen_oracle.R:
    python gen_replicas.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from test_oraculo_prep_limites import CASOS_BCA, capturar_replicas  # noqa: E402

for nome, (y_true, y_pred, metrica, n_boot) in CASOS_BCA.items():
    t0, t, jack = capturar_replicas(y_true, y_pred, metrica, n_boot)
    np.savetxt(f"bca_{nome}_t.csv", t.reshape(-1, 1), delimiter=",",
               header="t", comments="", fmt="%.17g")
    np.savetxt(f"bca_{nome}_jack.csv", jack.reshape(-1, 1), delimiter=",",
               header="jack", comments="", fmt="%.17g")
    np.savetxt(f"bca_{nome}_t0.csv", np.array([[t0]]), delimiter=",",
               header="t0", comments="", fmt="%.17g")
    print(nome, t0, len(t), len(jack))
