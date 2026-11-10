"""Navegação lateral do app web (reestruturação de 2026-09-08).

Dois contratos:

1. A estrutura (`app_nav`) e o roteamento (`app_quimiometria.py`) não podem
   divergir: toda tela declarada na barra lateral precisa ter um ramo que a
   desenhe, senão o usuário clica e cai numa página em branco.
2. Nenhuma tela ficou órfã na troca de 8 abas por navegação lateral -- as 8
   telas antigas continuam alcançáveis.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from guaraci import app_nav

_RAIZ = Path(__file__).resolve().parents[1]
_APP = _RAIZ / "app_quimiometria.py"

_TELAS_ANTIGAS = {"projeto", "dados", "preprocessamento", "modelo",
                  "validacao", "predicao", "relatorios", "sobre"}


def test_as_oito_telas_antigas_continuam_alcancaveis():
    chaves = {p.chave for p in app_nav.todas_paginas()}
    assert _TELAS_ANTIGAS <= chaves, (
        f"tela(s) perdida(s) na troca por navegação lateral: "
        f"{sorted(_TELAS_ANTIGAS - chaves)}")


def test_telas_novas_do_mockup_existem_e_ficam_fora_dos_grupos():
    fixas = {p.chave for p in app_nav.PAGINAS_FIXAS}
    assert fixas == {"inicio", "visualizacao"}
    for chave in fixas:
        assert app_nav.grupo_da_pagina(chave) is None


def test_grupos_do_web_seguem_a_taxonomia_do_catalogo():
    """Paridade CLI x web: grupos, ordem dos grupos e grupo de cada pagina vem do catalogo."""
    from guaraci import catalogo as cat
    ordem_cli = [g["nome"][0] for g in cat.grupos()]
    rotulos = [g.rotulo_pt for g in app_nav.GRUPOS]
    assert rotulos == sorted(rotulos, key=ordem_cli.index)
    assert [g.numero for g in app_nav.GRUPOS] == list(range(1, len(app_nav.GRUPOS) + 1)) and len(app_nav.GRUPOS) <= 10
    for pagina, tid in app_nav.PAGINAS_CATALOGO:
        grupo_cli = next(g["nome"][0] for g in cat.grupos() if g["id"] == cat.TELAS[tid]["grupo"])
        assert app_nav.grupo_da_pagina(pagina.chave).rotulo_pt == grupo_cli, pagina.chave


def test_toda_tela_do_cli_tem_pagina_no_web_ou_motivo_declarado():
    from guaraci import catalogo as cat
    com_pagina = {tid for _p, tid in app_nav.PAGINAS_CATALOGO}
    assert com_pagina | set(app_nav.TELAS_SO_NO_CLI) == set(cat.TELAS)
    assert not (com_pagina & set(app_nav.TELAS_SO_NO_CLI))
    assert all(len(m) > 20 for m in app_nav.TELAS_SO_NO_CLI.values())


def test_toda_pagina_declarada_tem_ramo_no_roteamento():
    """Lê o roteador por AST e confere que cada chave de tela aparece numa
    comparação `_pagina == "..."` -- sem isso a tela existiria no menu e não
    renderizaria nada."""
    fonte = (_RAIZ / "src" / "guaraci" / "app_web.py").read_text(encoding="utf-8")
    roteadas = set(re.findall(r'_pagina == "([a-z_]+)"', fonte))
    declaradas = {p.chave for p in app_nav.todas_paginas()}
    assert declaradas <= roteadas, (
        f"tela(s) sem ramo no roteamento: {sorted(declaradas - roteadas)}")


def test_chave_desconhecida_volta_ao_inicio_em_vez_de_pagina_branca():
    at = AppTest.from_file(str(_APP), default_timeout=60)
    at.session_state["pagina"] = "tela_que_nao_existe"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["pagina"] == app_nav.PAGINA_INICIAL


@pytest.mark.parametrize("chave", sorted(
    {p.chave for p in app_nav.todas_paginas()}))
def test_cada_tela_abre_sem_excecao(chave: str):
    at = AppTest.from_file(str(_APP), default_timeout=90)
    at.session_state["pagina"] = chave
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
