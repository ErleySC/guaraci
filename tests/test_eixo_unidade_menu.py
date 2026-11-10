# -*- coding: utf-8 -*-
"""R6: a tela [2.4] grava a UNIDADE do eixo (e a tecnica) e o resto do programa a usa.

Antes, dado em nm saia com o eixo invertido e o rotulo "Spectral variable", e o resumo mostrava
`faixa_espectral_cm1` e "Technique (profile): -" para NIR dispersivo, UV-Vis, fluorescencia,
cromatografia, RMN, IMS e Raman. Aqui: a [2.4] grava `eixo_unidade`/`tecnica_analitica`; rotulo,
orientacao (decrescente so' para cm-1 e ppm) e resumo seguem a unidade declarada; sem unidade
declarada, o comportamento antigo (rotulo neutro, orientacao por modo) continua.
"""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from guaraci import catalogo_tecnicas as ct  # noqa: E402
from guaraci import figuras as fg  # noqa: E402
from guaraci.config import Config  # noqa: E402
from guaraci.config_io import load_config, save_config  # noqa: E402
from guaraci.eixo_espectral import (ROTULO_EIXO_NEUTRO, campos_faixa_espectral,  # noqa: E402
                                    chave_faixa_espectral, rotulo_do_eixo, tecnica_do_perfil,
                                    unidade_do_eixo)

_ROTULOS = {
    "cm-1": "Wavenumber (cm$^{-1}$)", "nm": "Wavelength (nm)", "raman_shift": "Raman shift (cm$^{-1}$)",
    "min": "Retention time (min)", "ppm": "Chemical shift (ppm)", "ms": "Drift time (ms)",
    "indice": "Variable index",
}
#: convencao de cada unidade: so' numero de onda (IR/NIR) e ppm (RMN) se escrevem do maior para o menor
_DECRESCENTE = {"cm-1": True, "ppm": True, "nm": False, "raman_shift": False, "min": False, "ms": False,
                "indice": False}


def _cfg(unidade="", modo="csv", **kw):
    return Config(output_format="png", show_plots=False, mode=modo, axis_unit=unidade, **kw)


@pytest.mark.parametrize("unidade,rotulo", sorted(_ROTULOS.items()))
def test_rotulo_do_eixo_segue_a_unidade_declarada(unidade, rotulo):
    assert rotulo_do_eixo(_cfg(unidade)) == rotulo


def test_sem_unidade_declarada_o_rotulo_continua_neutro():
    assert rotulo_do_eixo(_cfg("")) == ROTULO_EIXO_NEUTRO
    assert rotulo_do_eixo(_cfg("furlong")) == ROTULO_EIXO_NEUTRO      # valor desconhecido nao e' presumido
    assert unidade_do_eixo(_cfg("")) == ""


@pytest.mark.parametrize("unidade,decrescente", sorted(_DECRESCENTE.items()))
def test_orientacao_do_eixo_so_e_decrescente_para_cm1_e_ppm(unidade, decrescente):
    # modo csv entrava em _MODOS_EIXO_DECRESCENTE: invertia o eixo de TUDO que vinha em csv
    assert fg.eixo_x_decrescente(_cfg(unidade, "csv")) is decrescente
    assert fg.eixo_x_decrescente(_cfg(unidade, "dx")) is decrescente


def test_sem_unidade_declarada_a_orientacao_antiga_por_modo_continua():
    assert fg.eixo_x_decrescente(_cfg("", "csv")) is True       # configs antigas: a Config padrao e' FT-NIR em cm-1
    assert fg.eixo_x_decrescente(_cfg("", "hplc")) is False
    assert fg.eixo_x_decrescente(_cfg("", "gcms")) is False


def test_perfil_com_rotulo_raman_nao_inverte_o_eixo(tmp_path):
    p = tmp_path / "raman.yaml"
    p.write_text("descricao: 'Raman'\nunidade_eixo: 'cm-1'\nrotulo_eixo: 'Raman shift (cm$^{-1}$)'\n"
                 "vocabulario:\n  matriz: 'x'\n", encoding="utf-8")
    cfg = Config(mode="dx", matrix_profile=str(p))
    assert rotulo_do_eixo(cfg) == "Raman shift (cm$^{-1}$)"
    assert fg.eixo_x_decrescente(cfg) is False
    assert unidade_do_eixo(cfg) == "cm⁻¹"


def test_perfil_ppm_agora_tem_rotulo_e_continua_decrescente(tmp_path):
    p = tmp_path / "rmn.yaml"
    p.write_text("descricao: 'RMN'\nunidade_eixo: 'ppm'\nvocabulario:\n  matriz: 'x'\n", encoding="utf-8")
    cfg = Config(mode="csv", matrix_profile=str(p))
    assert rotulo_do_eixo(cfg) == "Chemical shift (ppm)"
    assert fg.eixo_x_decrescente(cfg) is True


def test_unidade_declarada_vence_a_do_perfil():
    cfg = Config(mode="csv", matrix_profile="nir_ft_difuso", axis_unit="nm")
    assert rotulo_do_eixo(cfg) == "Wavelength (nm)"
    assert unidade_do_eixo(cfg) == "nm"
    assert fg.eixo_x_decrescente(cfg) is False


def test_chave_da_faixa_no_resumo_traz_a_unidade():
    assert chave_faixa_espectral(_cfg("nm")) == "Faixa espectral (nm)"
    assert chave_faixa_espectral(_cfg("ppm")) == "Faixa espectral (ppm)"
    assert chave_faixa_espectral(_cfg("raman_shift")) == "Faixa espectral (cm⁻¹)"
    assert chave_faixa_espectral(_cfg("indice")) == "Faixa espectral"
    assert chave_faixa_espectral(_cfg("")) == "Faixa espectral"


def test_faixa_espectral_cm1_so_quando_a_unidade_e_cm1():
    nm = campos_faixa_espectral(_cfg("nm", wn_min=700.0, wn_max=2500.0))
    assert nm == {"faixa_espectral": [700.0, 2500.0], "unidade_eixo": "nm"}       # "cm1" em dado nm mentia
    assert "faixa_espectral_cm1" not in campos_faixa_espectral(_cfg("min"))
    cm = campos_faixa_espectral(_cfg("cm-1", wn_min=4000.0, wn_max=10000.0))
    assert cm["faixa_espectral_cm1"] == [4000.0, 10000.0] and cm["unidade_eixo"] == "cm-1"
    assert campos_faixa_espectral(_cfg("raman_shift"))["faixa_espectral_cm1"]      # Raman shift e' cm-1
    sem = campos_faixa_espectral(_cfg(""))                                          # resumos antigos: campo mantido
    assert sem["faixa_espectral_cm1"] and sem["unidade_eixo"] == "nao_declarada"


def test_tecnica_do_resumo_vem_da_tela_2_4():
    assert tecnica_do_perfil(Config()) == ""
    assert tecnica_do_perfil(Config(technique="nir")) == "NIR Dispersivo"
    assert tecnica_do_perfil(Config(technique="raman")) == "Raman"
    assert tecnica_do_perfil(Config(technique="tecnica_inexistente")) == ""
    assert tecnica_do_perfil(Config(matrix_profile="nir_ft_difuso")) != ""         # sem [2.4]: a descricao do perfil


def test_todo_instrumento_do_catalogo_declara_a_unidade_do_eixo():
    for k, ins in ct.INSTRUMENTOS.items():
        assert "eixo_unidade" in ins, k
        assert ins["eixo_unidade"] in ("", *_ROTULOS), (k, ins["eixo_unidade"])
    esperado = {"ft-nir": "cm-1", "nir": "nm", "mir": "cm-1", "raman": "raman_shift", "uv-vis": "nm",
                "fluorescencia": "nm", "hplc": "min", "gc-ms": "min", "nmr": "ppm", "ims": "ms"}
    for k, u in esperado.items():
        assert ct.INSTRUMENTOS[k]["eixo_unidade"] == u, k


def _aplicar_pela_tela(monkeypatch, tk, formato=""):
    """Dirige a tela [2.4] de verdade (entrada simulada): escolhe a tecnica `tk` e o formato."""
    from guaraci import guaraci as G
    cfg = Config()
    ordem = G._tecnica_ordem()
    respostas = iter([str(ordem.index(tk) + 1), formato])
    monkeypatch.setattr(G, "_input", lambda *a, **k: next(respostas))
    monkeypatch.setattr(G, "_pause", lambda *a, **k: None)
    monkeypatch.setattr(G, "_cabecalho_tela", lambda *a, **k: None)
    monkeypatch.setattr(G, "_rodape_tela", lambda *a, **k: None)
    G._menu_technique(cfg)
    return cfg


@pytest.mark.parametrize("tk", sorted(ct.INSTRUMENTOS))
def test_tela_2_4_grava_unidade_e_tecnica_e_o_yaml_devolve(monkeypatch, tmp_path, tk):
    cfg = _aplicar_pela_tela(monkeypatch, tk)
    assert cfg.axis_unit == ct.INSTRUMENTOS[tk]["eixo_unidade"]
    assert cfg.technique == tk
    caminho = tmp_path / "c.yaml"
    save_config(cfg, str(caminho))
    volta = load_config(str(caminho))
    assert (volta.axis_unit, volta.technique) == (cfg.axis_unit, tk)


def test_trocar_de_tecnica_na_tela_nao_deixa_a_unidade_da_anterior(monkeypatch):
    from guaraci import guaraci as G
    cfg = _aplicar_pela_tela(monkeypatch, "nir")
    assert cfg.axis_unit == "nm"
    ordem = G._tecnica_ordem()
    respostas = iter([str(ordem.index("generico") + 1), ""])
    monkeypatch.setattr(G, "_input", lambda *a, **k: next(respostas))
    G._menu_technique(cfg)
    assert cfg.axis_unit == "" and cfg.technique == "generico"          # generico nao sabe a unidade: nao presume


def _espectros(n_pontos=120, n=8):
    rng = np.random.default_rng(0)
    eixo = np.linspace(700.0, 2500.0, n_pontos)
    X = rng.normal(size=(n, n_pontos)).cumsum(axis=1)
    rot = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    return eixo, X, rot, {"a": "tab:blue", "b": "tab:red"}


@pytest.mark.parametrize("unidade,rotulo,decrescente", [
    ("nm", "Wavelength (nm)", False), ("cm-1", "Wavenumber (cm$^{-1}$)", True),
    ("raman_shift", "Raman shift (cm$^{-1}$)", False), ("min", "Retention time (min)", False),
    ("ppm", "Chemical shift (ppm)", True), ("ms", "Drift time (ms)", False)])
def test_figura_real_usa_rotulo_e_orientacao_da_unidade(monkeypatch, tmp_path, unidade, rotulo, decrescente):
    achados = []
    monkeypatch.setattr(fg, "save", lambda fig, nome, pasta, cfg, subpasta="": achados.append(fig))
    eixo, X, rot, cores = _espectros()
    fg.fig_class_mean_spectra(eixo, X, rot, cores, _cfg(unidade, "csv"), str(tmp_path))
    ax = achados[-1].axes[0]
    assert ax.get_xlabel() == rotulo
    lo, hi = ax.get_xlim()
    assert bool(lo > hi) is decrescente, (unidade, lo, hi)
    plt.close("all")


def test_execucao_real_em_nm_grava_resumo_e_figuras_com_a_unidade(tmp_path):
    """Ponta a ponta (pipeline real, dado sintetico): resumo com `Faixa espectral (nm)`, tecnica e unidade."""
    import os

    from guaraci import pipeline as pl
    from tests._resumo import canonico
    cfg = Config(mode="sintetico", n_per_class=8, n_synthetic_points=60, wn_min=400.0, wn_max=4001.0, n_splits_cv=2,
                 n_repeats_cv=1, n_permutations=5, n_permutations_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20,
                 max_lvs=3, output_format="png", show_plots=False, save_dpi=40, axis_unit="nm", technique="nir",
                 input_folder=str(tmp_path / "dados"), output_root_folder=str(tmp_path / "saida"))
    os.makedirs(cfg.input_folder, exist_ok=True)
    pl.executar(cfg)
    resumos = list((tmp_path / "saida").rglob("resumo_modelo.txt"))
    assert resumos
    texto = canonico(resumos[0])
    assert "Faixa espectral (nm)" in texto
    assert "Faixa espectral (cm" not in texto
    assert re.search(r"Tecnica \(perfil\)\s*: NIR Dispersivo", texto)
    efetivos = (tmp_path / "saida").rglob("model_card.md")
    card = next(iter(efetivos)).read_text(encoding="utf-8")
    assert "`unidade_eixo` | nm" in card and "faixa_espectral_cm1" not in card


@pytest.mark.parametrize("unidade,pt", [("nm", "Comprimento de onda (nm)"), ("raman_shift", "Deslocamento Raman"),
                                        ("min", "Tempo de retenção (min)"), ("ppm", "Deslocamento químico (ppm)"),
                                        ("ms", "Tempo de deriva (ms)"), ("indice", "Índice da variável")])
def test_rotulos_novos_tem_traducao_para_as_figuras_em_portugues(monkeypatch, unidade, pt):
    from guaraci.idioma_figuras import traduzir_texto
    monkeypatch.setenv("GUARACI_IDIOMA", "pt")
    assert traduzir_texto(rotulo_do_eixo(_cfg(unidade))).startswith(pt)
