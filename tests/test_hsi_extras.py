# -*- coding: utf-8 -*-
"""Testes de hsi_extras.py e do submenu [E] de _menu_hsi -- dominio de
aplicabilidade, reamostragem e multiway HSI, sobre cubo 100% sintetico
(mesma montagem de test_hsi_offline_prova.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from guaraci.hsi_extras import (
    preparar_dados_hsi,
    rodar_hsi_dominio_aplicabilidade,
    rodar_hsi_multiway,
    rodar_hsi_reamostragem,
)


def _mascara(n_lin, n_col, raio_frac=0.35, largura_borda=4.0):
    yy, xx = np.ogrid[:n_lin, :n_col]
    cy, cx = (n_lin - 1) / 2.0, (n_col - 1) / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    raio = raio_frac * min(n_lin, n_col)
    return np.clip((raio - dist) / largura_borda + 0.5, 0.0, 1.0)


def _gravar(caminho: Path, cubo: np.ndarray, n_bandas: int) -> None:
    n_lin, n_col, _ = cubo.shape
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.with_suffix(".bin").write_bytes(cubo.astype("<f4").tobytes())
    caminho.with_suffix(".hdr").write_text(
        f"ENVI\nsamples = {n_col}\nlines = {n_lin}\nbands = {n_bandas}\n"
        f"header offset = 0\nfile type = ENVI Standard\ndata type = 4\n"
        f"interleave = bip\nbyte order = 0\n", encoding="utf-8")


def montar_dataset(tmp_path: Path, n_amostras=4) -> Path:
    rng = np.random.default_rng(42)
    raiz = tmp_path / "cubos"
    n_lin, n_col, n_bandas = 64, 64, 8
    alpha = _mascara(n_lin, n_col)[..., None]
    for classe, nivel in (("madura", 0.85), ("verde", 0.45)):
        n = n_amostras if classe == "madura" else max(2, n_amostras - 2)
        for a in range(n):
            fundo = rng.normal(0.05, 0.01, (n_lin, n_col, n_bandas))
            obj = rng.normal(nivel, 0.01, (n_lin, n_col, n_bandas))
            _gravar(raiz / classe / f"amostra{a}" / "vista0",
                    alpha * obj + (1.0 - alpha) * fundo, n_bandas)
    return raiz


@pytest.fixture(scope="module")
def dados(tmp_path_factory):
    return preparar_dados_hsi(str(montar_dataset(tmp_path_factory.mktemp("hsi"))))


@pytest.fixture(scope="module")
def dados_10_objetos(tmp_path_factory):
    return preparar_dados_hsi(
        str(montar_dataset(tmp_path_factory.mktemp("hsi10"), n_amostras=6)))


def test_preparar_dados_shapes_consistentes(dados):
    assert dados.X.shape[0] == len(dados.y) == len(dados.pixel_groups)
    assert len(dados.cubos) == len(dados.mascaras) == len(dados.grupos)
    assert set(dados.y.tolist()) == {"madura", "verde"}


def test_dominio_aplicabilidade_roda_e_fracao_valida(dados):
    r = rodar_hsi_dominio_aplicabilidade(dados)
    assert r["sensor_compativel"] is True
    assert 0.0 <= r["fracao_dentro"] <= 1.0
    assert r["n_objetos_teste"] >= 1
    assert r["n_objetos_treino"] >= 1


def test_dominio_aplicabilidade_split_por_objeto_nao_vaza(dados):
    r = rodar_hsi_dominio_aplicabilidade(dados, seed=1)
    assert r["n_objetos_treino"] + r["n_objetos_teste"] == len(
        set(dados.pixel_groups.tolist()))


def test_reamostragem_rebalanceia_objetos(dados):
    r = rodar_hsi_reamostragem(dados)
    # oversample duplica objetos (mesmo group_id) -> equaliza PIXELS, nao a
    # contagem de objetos DISTINTOS (estrutural, ver docstring do modulo)
    antes = {c: v["pixels"] for c, v in r["antes"].items()}
    depois = {c: v["pixels"] for c, v in r["depois"].items()}
    assert antes["madura"] > 1.5 * antes["verde"]      # dataset desbalanceado
    assert abs(depois["madura"] - depois["verde"]) < 0.05 * depois["madura"]
    assert r["depois"]["verde"]["objetos"] == r["antes"]["verde"]["objetos"]
    assert set(r["avaliabilidade"]) == {"madura", "verde"}


def test_multiway_roda_e_devolve_balanced_accuracy(dados_10_objetos):
    r = rodar_hsi_multiway(dados_10_objetos, n_splits=2)
    assert "balanced_accuracy_npls" in r
    assert "balanced_accuracy_pixelwise" in r


def test_multiway_com_poucos_objetos_da_erro_acionavel(dados):
    with pytest.raises(ValueError, match="poucos objetos"):
        rodar_hsi_multiway(dados, n_splits=2)


def test_dominio_com_um_objeto_so_da_erro_claro(dados):
    from guaraci.hsi_extras import HSIDadosPreparados
    um = dados.pixel_groups == dados.pixel_groups[0]
    pequeno = HSIDadosPreparados(
        cubos=dados.cubos[:1], mascaras=dados.mascaras[:1],
        grupos=dados.grupos[:1], rotulos=dados.rotulos[:1],
        X=dados.X[um], y=dados.y[um], pixel_groups=dados.pixel_groups[um],
        n_rejeitados=0)
    with pytest.raises(ValueError, match=">=2 objetos"):
        rodar_hsi_dominio_aplicabilidade(pequeno)


# --- CLI: submenu de extras em _menu_hsi ------------------------------------

def test_menu_hsi_extras_dominio_reamostragem_e_multiway_via_cli(
        monkeypatch, tmp_path, capsys):
    import guaraci.guaraci as guaraci_mod
    raiz = montar_dataset(tmp_path, n_amostras=6)
    # pasta; depois D, R, M; Enter sai
    respostas = iter([str(raiz), "D", "R", "M", ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
    cfg = guaraci_mod.Config(output_root_folder=str(tmp_path / "saida"))
    guaraci_mod._menu_hsi(cfg)
    saida = capsys.readouterr().out
    assert "dominio" in saida.lower() or "domain" in saida.lower()
    assert "N-PLS bal.acc" in saida
    assert "pixels" in saida


def test_menu_hsi_extras_erro_de_multiway_vira_mensagem_nao_excecao(
        monkeypatch, tmp_path, capsys):
    import guaraci.guaraci as guaraci_mod
    raiz = montar_dataset(tmp_path, n_amostras=4)   # poucos objetos p/ multiway
    respostas = iter([str(raiz), "M", ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(respostas))
    cfg = guaraci_mod.Config(output_root_folder=str(tmp_path / "saida"))
    guaraci_mod._menu_hsi(cfg)   # nao deve levantar
    assert "poucos objetos" in capsys.readouterr().out
