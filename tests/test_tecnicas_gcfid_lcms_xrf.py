# -*- coding: utf-8 -*-
"""Tecnicas acrescentadas na tela [2.4]: GC-FID, LC-MS e FRX/XRF. O que cada uma grava no config e
que o detector do leitor `hplc` acompanha a tecnica (GC-FID = FID, HPLC = UV)."""
from __future__ import annotations

import pytest

from guaraci import catalogo_tecnicas as ct
from guaraci import guaraci as G
from guaraci.config import Config


def _escolher(monkeypatch, cfg, tk, formato=""):
    ordem = G._tecnica_ordem()
    respostas = iter([str(ordem.index(tk) + 1), formato])
    monkeypatch.setattr(G, "_input", lambda *a, **k: next(respostas))
    monkeypatch.setattr(G, "_pause", lambda *a, **k: None)
    monkeypatch.setattr(G, "_cabecalho_tela", lambda *a, **k: None)
    monkeypatch.setattr(G, "_rodape_tela", lambda *a, **k: None)
    G._menu_technique(cfg)
    return cfg


def test_novas_tecnicas_estao_no_catalogo_e_na_tela_na_categoria_certa():
    ordem = G._tecnica_ordem()
    for tk in ("gc-fid", "lc-ms", "xrf"):
        assert tk in ct.INSTRUMENTOS and tk in ordem
    cats = {pt: keys for pt, _en, keys in G._TECNICA_CATEGORIAS}
    assert {"hplc", "gc-fid", "gc-ms", "lc-ms"} <= set(cats["Cromatografia"])
    assert {"icp", "xrf"} <= set(cats["Elementar"])


def test_gc_fid_grava_detector_fid_e_hplc_volta_para_uv(monkeypatch):
    cfg = _escolher(monkeypatch, Config(), "gc-fid", formato="2")      # 2 = pasta Agilent .D
    assert cfg.technique == "gc-fid" and cfg.mode == "hplc" and cfg.hplc_detector == "FID"
    _escolher(monkeypatch, cfg, "hplc", formato="2")
    assert cfg.technique == "hplc" and cfg.hplc_detector == "UV"


def test_gc_fid_por_tabela_csv_usa_mc_e_nao_corta_o_eixo(monkeypatch):
    cfg = _escolher(monkeypatch, Config(), "gc-fid")
    assert cfg.mode == "csv" and cfg.default_preprocessing == "mc"
    ins = ct.INSTRUMENTOS["gc-fid"]
    assert ins["faixa_max"] >= 1.0e9 and ins["preproc"] == "mc" and ins["eixo_unidade"] == "min"


@pytest.mark.parametrize("tk,modo,preproc", [("xrf", "multielementar", "autoscaling"), ("lc-ms", "csv", "autoscaling")])
def test_xrf_e_lc_ms_aplicam_modo_e_autoscaling(monkeypatch, tk, modo, preproc):
    cfg = _escolher(monkeypatch, Config(), tk)
    assert cfg.technique == tk and cfg.mode == modo
    assert ct.INSTRUMENTOS[tk]["preproc"] == preproc
