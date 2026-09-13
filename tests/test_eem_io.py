# -*- coding: utf-8 -*-
"""Teste unitario de `eem_io.parse_eem_dat` (Grupo 1, fechamento final,
2026-09-12) -- generalizacao confirmada contra um SEGUNDO formato de
exportacao de fluorimetro, real (nao sintetico), independente do dataset
Zenodo que motivou o parser original (Passo 149; ver
`tests/test_validacao_publica_zenodo_eem_azeite.py`, gated por
`GUARACI_DATASETS_DIR` -- o dataset Zenodo completo nao esta' versionado
aqui por tamanho).

Este teste roda sempre (fixture pequena, versionada em
`tests/fixtures/eem_horiba_aqualog/`) e nao depende de nenhum dataset
externo baixado a parte."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from guaraci.eem_io import parse_eem_dat

_FIXTURA = (
    Path(__file__).parent / "fixtures" / "eem_horiba_aqualog"
    / "B1S12022-09-29-09-10-M3PEM.dat"
)


def test_parse_eem_dat_generaliza_para_formato_horiba_aqualog_real():
    excitacao, emissao, matriz, relatorio = parse_eem_dat(_FIXTURA)

    assert excitacao.shape == (106,)
    assert emissao.shape == (500,)
    assert matriz.shape == (500, 106)

    # eixo de excitacao do Aqualog e' DECRESCENTE (450->240nm) -- diferente
    # do Zenodo, mas parse_eem_dat nao assume ordem nenhuma.
    assert excitacao[0] == 450.0
    assert excitacao[-1] == 240.0
    np.testing.assert_allclose(emissao[0], 243.173)

    # 0 linhas descartadas -- confirma que a convencao de 3-linhas-de-
    # cabecalho + `emissao<TAB>valores` e' comum aos dois formatos reais,
    # nao uma coincidencia forcada.
    assert relatorio.n_validas == 500
    assert relatorio.n_descartadas == 0
