"""Tela Visualização: esquema de cor das figuras.

Contrato: a web NÃO tem sistema de cor próprio -- ela lê e grava o MESMO
arquivo de preferências da CLI (`visual_config.json`) e aplica a paleta pela
MESMA função (`cli_assistente.apply_palette`), que é o que faz a cor chegar
às figuras de fato.
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


def _tela_visualizacao() -> AppTest:
    at = AppTest.from_file(_APP, default_timeout=60)
    at.session_state["pagina"] = "visualizacao"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_esquemas_oferecidos_existem_no_catalogo_real():
    """Os 3 esquemas da tela apontam para entradas de PALETAS_COR -- a tela
    nao inventa paleta propria."""
    from guaraci.app_tabs.visualizacao import _ESQUEMAS
    from guaraci.cli_assistente import PALETAS_COR

    for chave in _ESQUEMAS:
        assert chave in PALETAS_COR, f"esquema '{chave}' nao esta' no catalogo"


def test_tela_mostra_os_tres_esquemas_do_mockup():
    at = _tela_visualizacao()
    radios = [r for r in at.radio if r.label == "Colour scheme"]
    assert radios, "seletor de esquema de cor ausente na tela Visualizacao"
    assert len(radios[0].options) == 3


def test_escolher_esquema_grava_no_mesmo_arquivo_da_cli(prefs_isoladas):
    at = _tela_visualizacao()
    radio = [r for r in at.radio if r.label == "Colour scheme"][0]
    alvo = [o for o in radio.options if "olourblind" in o or "daltonismo" in o][0]
    radio.set_value(alvo).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert prefs.load_visual_config(prefs_isoladas)["paleta"] == "daltonismo_safe"


def test_amostra_mostra_as_cores_reais_do_catalogo(prefs_isoladas):
    """A previa usa os hex do catalogo -- nao cores decorativas inventadas."""
    from guaraci.cli_assistente import PALETAS_COR

    at = _tela_visualizacao()
    radio = [r for r in at.radio if r.label == "Colour scheme"][0]
    alvo = [o for o in radio.options if "olourblind" in o or "daltonismo" in o][0]
    radio.set_value(alvo).run()

    markdown = "\n".join(m.value for m in at.markdown)
    for cor in PALETAS_COR["daltonismo_safe"]["cores"][:3]:
        assert cor in markdown, f"cor {cor} do catalogo ausente na amostra"


def _espectros_falsos():
    """Prévia mínima com 3 classes, no formato que a tela Dados guarda."""
    import numpy as np
    rng = np.random.default_rng(0)
    wn = np.linspace(4000, 400, 60)
    X = rng.normal(size=(9, 60))
    labels = np.array(["A"] * 3 + ["B"] * 3 + ["C"] * 3)
    return {"wn": wn, "X": X, "labels": labels}


def test_previa_desenha_com_as_cores_do_esquema_escolhido(prefs_isoladas):
    """A pré-visualização NÃO é uma simulação: ela chama a mesma função que
    colore as figuras do pipeline, então trocar o esquema tem de trocar a cor
    de fato desenhada."""
    import matplotlib
    matplotlib.use("Agg")

    from guaraci.cli_assistente import PALETAS_COR, apply_palette
    from guaraci.paleta_cores import set_active_palette
    from guaraci.spectra_preview import plot_mean_spectra

    esp = _espectros_falsos()
    try:
        apply_palette("daltonismo_safe")
        fig = plot_mean_spectra(esp["wn"], esp["X"], esp["labels"])
        cores_linhas = [ln.get_color() for ln in fig.axes[0].get_lines()]
        esperado = PALETAS_COR["daltonismo_safe"]["cores"][:3]
        assert [c.lower() for c in cores_linhas[:3]] == \
               [c.lower() for c in esperado]

        apply_palette("alto_contraste")
        fig2 = plot_mean_spectra(esp["wn"], esp["X"], esp["labels"])
        cores2 = [ln.get_color() for ln in fig2.axes[0].get_lines()]
        assert [c.lower() for c in cores2[:3]] == \
               [c.lower() for c in PALETAS_COR["alto_contraste"]["cores"][:3]]
        assert cores2[:3] != cores_linhas[:3]   # trocou de verdade
    finally:
        set_active_palette(None)


def test_cores_personalizadas_por_classe_chegam_a_figura(prefs_isoladas):
    import matplotlib
    matplotlib.use("Agg")

    from guaraci.paleta_cores import set_active_palette
    from guaraci.spectra_preview import plot_mean_spectra

    esp = _espectros_falsos()
    try:
        set_active_palette(["#112233", "#445566", "#778899"])
        fig = plot_mean_spectra(esp["wn"], esp["X"], esp["labels"])
        cores = [ln.get_color().lower() for ln in fig.axes[0].get_lines()][:3]
        assert cores == ["#112233", "#445566", "#778899"]
    finally:
        set_active_palette(None)
