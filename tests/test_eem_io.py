# -*- coding: utf-8 -*-
"""Teste unitario de `eem_io.parse_eem_dat`: generalizacao para um segundo layout de exportacao de fluorimetro (excitacao
DECRESCENTE, grade fina de emissao), com arquivo SINTETICO de verdade conhecida. A conformidade com um arquivo real de fabricante
e' validacao do mantenedor (`validacao/tests/test_formatos_reais_locais.py`)."""
from __future__ import annotations

import numpy as np

from guaraci.eem_io import parse_eem_dat


def test_parse_eem_dat_aceita_excitacao_decrescente_e_grade_fina(tmp_path):
    from guaraci.formatos_sinteticos import escrever_eem_dat
    exc = np.arange(450.0, 239.0, -2.0)                       # 106 comprimentos de onda, decrescente
    em = np.linspace(243.173, 824.234, 500)
    verdade = np.outer(np.exp(-((em - 520.0) / 40.0) ** 2), np.exp(-((exc - 400.0) / 30.0) ** 2)) * 1.0e4
    excitacao, emissao, matriz, relatorio = parse_eem_dat(escrever_eem_dat(str(tmp_path / "a.dat"), exc, em, verdade))
    assert excitacao.shape == (106,) and emissao.shape == (500,) and matriz.shape == (500, 106)
    assert excitacao[0] == 450.0 and excitacao[-1] == 240.0   # parse_eem_dat nao assume ordem nenhuma
    np.testing.assert_allclose(emissao[0], 243.173)
    np.testing.assert_allclose(matriz, verdade, rtol=1e-5, atol=1e-6)
    assert relatorio.n_validas == 500 and relatorio.n_descartadas == 0

