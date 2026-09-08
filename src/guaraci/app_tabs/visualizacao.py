"""app_tabs/visualizacao.py — Tela (Visualisation): cor dos gráficos.

Layout do mockup de 2026-09-08: painel esquerdo com o esquema de cor e as
cores por classe, painel direito com pré-visualização ao vivo.

A pré-visualização é REAL, não simulada: ela chama
`spectra_preview.plot_mean_spectra`, que colore por
`paleta_cores.map_class_colors` — a mesma função que colore as figuras do
pipeline. Trocar o esquema aqui muda o desenho porque muda a paleta ativa,
não porque um SVG de exemplo foi repintado.

A escolha é gravada em `~/.guaraci/visual_config.json`, o mesmo arquivo que
o menu Visualização da CLI usa: web e terminal compartilham a preferência.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from guaraci.cli_assistente import PALETAS_COR, apply_palette
from guaraci.paleta_cores import PALETA, map_class_colors, set_active_palette
from guaraci.preferencias_visuais import load_visual_config, save_visual_config
from guaraci.spectra_preview import plot_mean_spectra

# Esquemas oferecidos na tela, na ordem do mockup. Cada um aponta para uma
# entrada REAL de PALETAS_COR -- a tela não inventa paleta própria.
_ESQUEMAS = ["qualitativo", "daltonismo_safe", "alto_contraste"]

_ROTULO_ESQUEMA = {
    "qualitativo":    {"PT": "Guaraci padrão", "EN": "Guaraci default"},
    "daltonismo_safe": {"PT": "Amigável a daltonismo",
                        "EN": "Colourblind friendly"},
    "alto_contraste": {"PT": "Alto contraste", "EN": "High contrast"},
}


def _paleta_escolhida() -> str:
    return str(load_visual_config().get("paleta", "qualitativo"))


def _cores_do_esquema(chave: str) -> List[str]:
    """Cores de amostra do esquema. `qualitativo` não tem lista fixa: usa a
    paleta de máxima distintividade, que é o que o pipeline realmente
    desenha nesse modo."""
    cores = PALETAS_COR.get(chave, {}).get("cores")
    return list(cores) if cores else list(PALETA)


def _cores_ativas() -> List[str]:
    """Cores em uso agora: as personalizadas por classe, se houver;
    senão as do esquema escolhido."""
    personalizadas = st.session_state.get("cores_personalizadas")
    if personalizadas:
        return list(personalizadas)
    return _cores_do_esquema(_paleta_escolhida())


def _amostra_html(cores: List[str], n: int = 3) -> str:
    return "".join(
        f"<span style='display:inline-block;width:22px;height:16px;"
        f"margin-right:3px;border-radius:3px;background:{c};"
        f"border:1px solid rgba(128,128,128,.35)'></span>"
        for c in cores[:n])


def _classes_da_previa() -> Optional[np.ndarray]:
    espectros = st.session_state.get("previa_espectros")
    if espectros is None:
        return None
    return np.unique(np.asarray(espectros["labels"]))


def _figura_previa(T: Callable[[str], str]):
    """Figura de pré-visualização com a paleta ativa. Usa os espectros reais
    da prévia quando existem; sem eles, devolve None (a tela então mostra
    apenas as amostras de cor, em vez de inventar um espectro)."""
    espectros = st.session_state.get("previa_espectros")
    if espectros is None:
        return None
    return plot_mean_spectra(espectros["wn"], espectros["X"],
                             espectros["labels"],
                             T("Preview · mean spectrum by class"))


def render(T: Callable[[str], str], tok: Callable[[], Dict[str, str]]) -> None:
    pt = st.session_state.get("lang") != "EN"
    atual = _paleta_escolhida()

    esq, dir_ = st.columns([1, 2])

    # ── Painel esquerdo: esquema + cor por classe ────────────────────────
    with esq:
        st.markdown(f"**{T('Colour scheme')}**")
        rotulos = [
            _ROTULO_ESQUEMA[c]["PT" if pt else "EN"] for c in _ESQUEMAS]
        idx = _ESQUEMAS.index(atual) if atual in _ESQUEMAS else 0
        escolha_rotulo = st.radio(
            T("Colour scheme"), rotulos, index=idx,
            key="radio_esquema_cor", label_visibility="collapsed")
        escolha = _ESQUEMAS[rotulos.index(escolha_rotulo)]
        st.markdown(_amostra_html(_cores_do_esquema(escolha)),
                    unsafe_allow_html=True)
        st.caption(PALETAS_COR.get(escolha, {}).get(
            "PT" if pt else "EN", {}).get("desc", ""))

        if escolha != atual:
            vcfg = load_visual_config()
            vcfg["paleta"] = escolha
            try:
                save_visual_config(vcfg)
            except OSError as e_vis:
                st.warning(T("Could not save the palette choice: {e}").format(
                    e=e_vis))
            st.session_state.pop("cores_personalizadas", None)
            st.rerun()

        st.divider()
        st.markdown(f"**{T('Colour per class (customise)')}**")
        classes = _classes_da_previa()
        base = _cores_do_esquema(escolha)
        if classes is None:
            st.caption(T("Load the spectra preview on the **Data** screen to "
                         "customise the colour of each real class."))
        else:
            personalizadas = list(_cores_ativas())
            while len(personalizadas) < len(classes):
                personalizadas.append(base[len(personalizadas) % len(base)])
            mudou = False
            for i, cls in enumerate(classes[:10]):
                nova = st.color_picker(
                    str(cls), personalizadas[i], key=f"cor_classe_{i}")
                if nova != personalizadas[i]:
                    personalizadas[i] = nova
                    mudou = True
            if mudou:
                st.session_state["cores_personalizadas"] = personalizadas
                st.rerun()
            if st.session_state.get("cores_personalizadas"):
                if st.button(T("Reset to the scheme"),
                             key="btn_reset_cores",
                             use_container_width=True):
                    st.session_state.pop("cores_personalizadas", None)
                    st.rerun()

        st.info(T(
            "This is the same palette catalogue the CLI offers (Visualisation "
            "menu). The choice is saved in `~/.guaraci/visual_config.json`, so "
            "terminal and web stay in sync, and it applies to **every figure "
            "of the next run**, not only to the preview below."))

    # ── Painel direito: pré-visualização ao vivo ─────────────────────────
    with dir_:
        card = st.container(border=True)
        card.markdown(f"**{T('Preview · mean spectrum by class')}**")
        # A paleta ativa é aplicada ANTES de desenhar: é o mesmo caminho que
        # o pipeline usa, por isso o desenho abaixo é o resultado de verdade.
        apply_palette(escolha)
        personalizadas = st.session_state.get("cores_personalizadas")
        if personalizadas:
            set_active_palette(personalizadas)
        fig = _figura_previa(T)
        if fig is not None:
            card.pyplot(fig, use_container_width=True)
            plt.close(fig)
            card.caption(T("Wavenumber (cm⁻¹) — decreasing"))
            classes = _classes_da_previa()
            if classes is not None:
                mapa = map_class_colors(classes)
                card.markdown(
                    " ".join(
                        f"<span style='font-size:.72rem'>"
                        f"<span style='display:inline-block;width:10px;"
                        f"height:10px;border-radius:50%;background:"
                        f"{mapa[str(c)]};margin-right:3px'></span>{c}</span>"
                        for c in classes[:10]),
                    unsafe_allow_html=True)
        else:
            card.info(T("No spectra loaded yet, so there is nothing real to "
                        "draw here. The swatches on the left already show the "
                        "exact colours this scheme will use."))
            card.markdown(_amostra_html(_cores_do_esquema(escolha), n=8),
                          unsafe_allow_html=True)
