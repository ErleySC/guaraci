# -*- coding: utf-8 -*-
"""Tabelas de FRX/XRF, LC-MS e GC-FID com os cabecalhos que os instrumentos e programas exportam, e as
mensagens quando a leitura nao pode seguir. Os arquivos sao escritos aqui, no layout documentado das
exportacoes (FRX portatil: coluna de incerteza ao lado de cada elemento e '< LOD'; MZmine/XCMS: um pico
por linha); nao substituem um arquivo real do instrumento do usuario."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from guaraci import dados_io as dio
from guaraci import extras_opcionais as ex
from guaraci.icp_multielementar import ler_icp_csv, normalizar_elemento


@pytest.mark.parametrize("cab,esperado", [
    ("Fe", "Fe"), ("Mg Concentration", "Mg"), ("Fe Conc", "Fe"), ("Pb (ppm)", "Pb"), ("Cd 111", "Cd 111"),
    ("Pb 220.353", "Pb 220.353"),
    ("Zr 2σ", None), ("Zr 2s", None), ("Fe +/-", None), ("Fe ±", None), ("Mo Error", None), ("Pb SD", None),
    ("Bal", None), ("LE", None), ("Sample Name", None), ("Reading No", None), ("Mode", None),
])
def test_cabecalho_de_frx_elemento_ou_incerteza(cab, esperado):
    assert normalizar_elemento(cab) == esperado


def test_frx_portatil_com_incerteza_e_lod(tmp_path):
    p = tmp_path / "frx.csv"
    p.write_text("Reading No,Sample Name,Units,Mo,Mo Error,Zr,Zr 2σ,Fe Concentration,Fe +/-,Pb,Pb Error\n"
                 "1,A1,ppm,12.3,1.1,< LOD,3.2,1200,30,ND,4\n"
                 "2,A2,ppm,10.1,1.0,22,3.1,1180,30,<LOD,4\n", encoding="utf-8")
    t = ler_icp_csv(str(p), unidade="ppm", coluna_amostra="Sample Name")
    assert t.elementos == ["Mo", "Zr", "Fe", "Pb"]
    np.testing.assert_allclose(t.valores[:, 2], [1200, 1180])
    assert t.censurado[:, 3].all() and t.censurado[0, 1] and not t.censurado[1, 1]
    assert {"Mo Error", "Zr 2σ", "Fe +/-", "Pb Error"} <= set(t.colunas_ignoradas)


def test_frx_sem_coluna_de_elemento_lista_as_colunas(tmp_path):
    p = tmp_path / "frx.csv"
    p.write_text("Amostra,Leitura,Obs\nA,1,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="colunas do arquivo"):
        ler_icp_csv(str(p), unidade="ppm", coluna_amostra="Amostra")


def test_tabela_de_picos_lcms_transposta_recebe_dica(tmp_path):
    p = tmp_path / "mzmine.csv"
    p.write_text("row ID,row m/z,row retention time,S1.mzML Peak area,S2.mzML Peak area\n"
                 "1,181.07,2.31,1000,1200\n2,203.05,3.10,500,450\n", encoding="utf-8")
    with pytest.raises(ValueError, match="transponha"):
        dio._carregar_csv_completo(str(p), "classe", "")


def test_tabela_comum_sem_classe_nao_recebe_dica_de_picos(tmp_path):
    p = tmp_path / "comum.csv"
    p.write_text("amostra,1000,1002,1004\nA,0.1,0.2,0.3\n", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        dio._carregar_csv_completo(str(p), "classe", "")
    assert "transponha" not in str(e.value)


def test_lcms_na_orientacao_certa_le_picos_como_variaveis_nomeadas(tmp_path):
    p = tmp_path / "lcms.csv"
    p.write_text("classe,M181T139,M203T186,M305T402\nA,1000,500,20\nA,1100,480,25\nB,300,900,70\n",
                 encoding="utf-8")
    _eixo, X, rot, _c, _m, nomes = dio._carregar_csv_completo(str(p), "classe", "")
    assert X.shape == (3, 3) and nomes == ["M181T139", "M203T186", "M305T402"]
    assert list(rot) == ["A", "A", "B"]


def test_gc_fid_entra_no_extra_hplc_e_mensagem_sabe_o_python(monkeypatch):
    e = next(x for x in ex.EXTRAS if x.modulo == "rainbow")
    assert "gc-fid" in e.tecnicas
    monkeypatch.setattr(sys, "version_info", (3, 14, 0, "final", 0))
    texto = ex.texto_instalar(e, "PT")
    assert "3.13" in texto and "pip install" not in texto


def test_sem_rainbow_mensagem_do_leitor_nao_manda_instalar_no_3_14(monkeypatch):
    import builtins
    from guaraci import importadores_proprietarios as ip
    real = builtins.__import__

    def falso(nome, *a, **k):
        if nome == "rainbow":
            raise ImportError("sem rainbow")
        return real(nome, *a, **k)
    monkeypatch.setattr(builtins, "__import__", falso)
    monkeypatch.setattr(sys, "version_info", (3, 14, 0, "final", 0))
    with pytest.raises(ImportError, match="Python 3.13"):
        ip.parse_cromatograma_hplc("qualquer.D", detector="FID")
