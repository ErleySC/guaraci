"""Seletor de paleta na aba Modelo do app web.

Contrato: a web NAO tem sistema de cor proprio -- ela le e grava o MESMO
arquivo de preferencias da CLI (`visual_config.json`) e aplica a paleta pela
MESMA funcao (`cli_assistente.apply_palette`), que e' o que faz a cor chegar
as figuras de fato.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import guaraci.preferencias_visuais as prefs

_RAIZ = Path(__file__).resolve().parents[1]
_APP = str(_RAIZ / "app_quimiometria.py")


@pytest.fixture(autouse=True)
def prefs_isoladas(tmp_path, monkeypatch):
    """Nunca tocar em ~/.guaraci/visual_config.json (estado real do usuario)."""
    monkeypatch.setattr(prefs, "VISUAL_PATH", tmp_path / "visual_config.json")
    yield tmp_path / "visual_config.json"


def test_seletor_lista_o_catalogo_real_e_mostra_a_descricao():
    from guaraci.cli_assistente import PALETAS_COR

    at = AppTest.from_file(_APP, default_timeout=60)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    sel = [s for s in at.selectbox if s.label == "Figure color palette"]
    assert sel, "seletor de paleta ausente na aba Modelo"
    # `.options` traz os rotulos ja' formatados -- tem que ser os nomes do
    # catalogo real (idioma EN por padrao), nao rotulos criados na web.
    esperados = [PALETAS_COR[k]["EN"]["nome"] for k in PALETAS_COR]
    assert list(sel[0].options) == esperados


def test_escolher_paleta_grava_no_mesmo_arquivo_da_cli(prefs_isoladas):
    at = AppTest.from_file(_APP, default_timeout=60)
    at.run()
    sel = [s for s in at.selectbox if s.label == "Figure color palette"][0]
    sel.set_value("daltonismo_safe").run()
    assert not at.exception, [str(e) for e in at.exception]

    assert prefs.load_visual_config(prefs_isoladas)["paleta"] == "daltonismo_safe"


def test_amostra_mostra_as_cores_reais_do_catalogo():
    """A previa usa os hex do catalogo -- nao cores decorativas inventadas."""
    from guaraci.cli_assistente import PALETAS_COR

    at = AppTest.from_file(_APP, default_timeout=60)
    at.run()
    sel = [s for s in at.selectbox if s.label == "Figure color palette"][0]
    sel.set_value("daltonismo_safe").run()

    markdown = "\n".join(m.value for m in at.markdown)
    for cor in PALETAS_COR["daltonismo_safe"]["cores"]:
        assert cor in markdown, f"cor {cor} do catalogo ausente na amostra"
