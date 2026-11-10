# -*- coding: utf-8 -*-
"""Modelo LaTeX e resumo sem tecnica/material presumidos (V79, V78, V81).

O .tex era FT-NIR + "oil authentication" + instrumento e revistas fixos mesmo
para uma execucao Raman; a faixa era 4000-10000 cm-1 quando o resumo nao a
trazia, e a unidade da faixa era sempre cm-1. Agora tudo vem da execucao.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from guaraci import reports
from guaraci.config import Config
from guaraci.eixo_espectral import chave_faixa_espectral, tecnica_do_perfil
from guaraci.resultados_io import _faixa_do_resumo

_PROIBIDOS = ("FT-NIR", "oil", "Oil", "Talanta", "Food Chemistry", "ABB")
_BASE = ("Balanced Accuracy (CV): 0.912\nR2Y: 0.87\nQ2Y: 0.81\n"
         "Preprocessamento: msc_sg_mc\nN treino: 120\nN. Classes: 5\nOptimal LVs: 6\n"
         "Validation: StratifiedKFold\nGroup-aware (mae_id): sim\nPermutation n_validos: 200\n")
# O modelo sai no idioma dos relatorios (GUARACI_IDIOMA): cada teste roda nos dois.
_SECOES = {"en": ("Samples and spectral acquisition", "Spectral preprocessing",
                  "TODO: report the spectral range"),
           "pt": ("Amostras e aquisição espectral", "Pré-processamento espectral",
                  "TODO: informe a faixa espectral")}
_IDIOMA = {"atual": "pt"}


@pytest.fixture(autouse=True, params=["pt", "en"])
def _idioma(request, monkeypatch):
    monkeypatch.setenv("GUARACI_IDIOMA", request.param)
    monkeypatch.setitem(_IDIOMA, "atual", request.param)


def _pasta(tmp_path: Path, extra: str) -> str:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "resumo_modelo.txt").write_text(_BASE + extra, encoding="utf-8")
    return str(tmp_path)


def _tex(tmp_path, extra) -> str:
    return reports.generate_latex_template(_pasta(tmp_path, extra), {}).decode("utf-8")


def _trecho_faixa(tex: str) -> str:
    inicio, fim, _ = _SECOES[_IDIOMA["atual"]]
    return tex.split(inicio)[1].split(fim)[0]


def test_latex_raman_sem_ftnir_oleo_instrumento_nem_revista(tmp_path):
    tex = _tex(tmp_path, "Faixa espectral (cm⁻¹): [200, 3200]\nTecnica (perfil): Raman 785 nm\n")
    for termo in _PROIBIDOS:
        assert termo not in tex, termo
    assert "Raman 785 nm" in tex
    assert r"\SIrange{200}{3200}{\per\centi\meter}" in tex


def test_latex_faixa_em_nm_usa_nm(tmp_path):
    tex = _tex(tmp_path, "Faixa espectral (nm): [400, 1100]\nTecnica (perfil): Vis-NIR\n")
    assert r"\SIrange{400}{1100}{\nano\meter}" in tex
    assert "centi" not in _trecho_faixa(tex)


def test_latex_sem_unidade_declarada_nao_presume_unidade(tmp_path):
    tex = _tex(tmp_path, "Faixa espectral: [1, 500]\n")
    assert r"\numrange{1}{500}" in tex
    assert "centi" not in _trecho_faixa(tex) and "nano" not in _trecho_faixa(tex)
    for termo in _PROIBIDOS:
        assert termo not in tex, termo


def test_latex_sem_faixa_no_resumo_nao_inventa_4000_10000(tmp_path):
    tex = _tex(tmp_path, "")
    assert "4000" not in tex and "10000" not in tex
    assert _SECOES[_IDIOMA["atual"]][2] in tex


def test_latex_resumo_antigo_com_cm1_continua_lido(tmp_path):
    tex = _tex(tmp_path, "Faixa espectral (cm-1): [4000, 10000]\n")
    assert r"\SIrange{4000}{10000}{\per\centi\meter}" in tex


def test_chave_da_faixa_segue_o_perfil():
    assert chave_faixa_espectral(Config(matrix_profile="nir_dispersivo_1100_2498nm")) == "Faixa espectral (nm)"
    assert chave_faixa_espectral(Config(matrix_profile="nir_ft_difuso")) == "Faixa espectral (cm⁻¹)"
    assert chave_faixa_espectral(Config(matrix_profile="generico")) == "Faixa espectral"
    assert chave_faixa_espectral(Config(matrix_profile="generico"), "Faixa com sinal") == "Faixa com sinal"


def test_tecnica_do_perfil_sem_unidade_entre_parenteses():
    assert tecnica_do_perfil(Config(matrix_profile="nir_ft_difuso")) == "FT-NIR de reflectancia difusa"
    assert tecnica_do_perfil(Config(matrix_profile="generico")) == ""


def test_resumo_do_model_card_usa_a_chave_gravada():
    assert _faixa_do_resumo({"Faixa espectral (nm)": "[400, 1100]"}) == ("Faixa espectral (nm)", "[400, 1100]")
    assert _faixa_do_resumo({"Faixa espectral (cm-1)": "[1, 2]"})[0] == "Faixa espectral (cm-1)"
    assert _faixa_do_resumo({}) == ("Faixa espectral", "-")

