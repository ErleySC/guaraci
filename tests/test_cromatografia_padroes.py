# -*- coding: utf-8 -*-
"""Padroes da familia cromatografia (HPLC, GC-MS): faixa sem corte, MC em vez de
autoscaling, tabela de compostos com cabecalho de NOMES e CSV do COW reutilizavel."""

import numpy as np

from guaraci import catalogo_tecnicas as cat
from guaraci import dados_io as dio


def test_cromatografia_nao_corta_eixo_em_segundos_nem_mz_e_usa_mc():
    for tk in ("hplc", "gc-ms"):
        ins = cat.INSTRUMENTOS[tk]
        # corrida em segundos (0-3600 s) e tabela m/z (35-550) sobrevivem inteiras
        for eixo in (np.arange(0.0, 3600.0, 0.4), np.arange(35.0, 551.0)):
            assert np.all((eixo >= ins["faixa_min"]) & (eixo <= ins["faixa_max"])), tk
        # autoscaling em cromatograma inteiro iguala o ruido da linha de base aos picos
        assert ins["preproc"] == "mc", tk


def test_tabela_de_compostos_com_nomes_no_cabecalho_carrega(tmp_path):
    p = tmp_path / "areas.csv"
    p.write_text('amostra,classe,lote,conc,composto_a,"1,8-composto_b",composto_c\n'
                 "s1,A,L1,1.0,10.5,3.2,7.0\ns2,B,L2,2.0,11.0,2.9,6.1\ns3,A,L3,1.5,9.8,3.5,7.4\n",
                 encoding="utf-8")
    eixo, X, rot, conc, mae = dio.load_csv_com_grupos(str(p), "classe", "conc", col_grupo="lote")
    assert X.shape == (3, 3)
    assert eixo.tolist() == [1.0, 2.0, 3.0]
    assert X[0].tolist() == [10.5, 3.2, 7.0]
    assert list(mae) == ["L1", "L2", "L3"]


def test_cabecalho_numerico_continua_eixo_real(tmp_path):
    p = tmp_path / "rt.csv"
    p.write_text("classe,2.10,3.60,5.00\nA,1,2,3\nB,4,5,6\n", encoding="utf-8")
    eixo, X, *_ = dio.load_csv_com_grupos(str(p), "classe", "")
    assert eixo.tolist() == [2.1, 3.6, 5.0]


def test_tabela_de_compostos_nao_usa_teor_nem_lote_como_variavel(tmp_path):
    # classificacao sem coluna_concentracao: 'conc' e 'lote' numericos nao podem virar composto
    p = tmp_path / "areas.csv"
    p.write_text("classe,conc,lote,composto_a,composto_b\nA,1.0,1,10.5,3.2\nB,2.0,2,11.0,2.9\n", encoding="utf-8")
    eixo, X, *_ = dio.load_csv_com_grupos(str(p), "classe", "")
    assert X.tolist() == [[10.5, 3.2], [11.0, 2.9]]
