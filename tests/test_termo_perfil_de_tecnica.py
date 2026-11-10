# -*- coding: utf-8 -*-
"""Um unico termo para o perfil (V78): "perfil de tecnica".

"Perfil de matriz" e "perfil de tecnica" conviviam na mesma saida e se
contradiziam (o perfil descreve a tecnica de medida, nao a matriz). O texto
que o usuario le -- `guaraci perfis`, o manual e os catalogos -- usa so' o
primeiro termo abaixo.
"""
from __future__ import annotations

import re
from pathlib import Path

import guaraci.guaraci as g

RAIZ = Path(__file__).resolve().parents[1]
#: o termo antigo, partido em dois para este arquivo nao o citar por inteiro
_ANTIGO = re.compile("perfi(l|s) de " + "matriz", re.IGNORECASE)


def test_guaraci_perfis_diz_perfis_de_tecnica(capsys):
    g._listar_perfis()
    saida = capsys.readouterr().out
    assert "Perfis de tecnica" in saida
    assert not _ANTIGO.search(saida)


def test_manual_e_catalogos_sem_o_termo_antigo():
    alvos = [RAIZ / "MANUAL.md", *sorted((RAIZ / "src" / "guaraci").glob("catalogo_*.py")),
             RAIZ / "src" / "guaraci" / "technique_registry.py",
             RAIZ / "src" / "guaraci" / "dados_docs" / "manual.md"]
    achados = [a.name for a in alvos if _ANTIGO.search(a.read_text(encoding="utf-8"))]
    assert not achados, achados


def test_catalogo_chama_o_campo_de_perfil_de_tecnica():
    from guaraci.catalogo_opcoes import OPCOES
    nome_pt, nome_en = OPCOES["perfil_matriz"]["nome"]
    assert nome_pt == "Perfil de tecnica" and nome_en == "Technique profile"
