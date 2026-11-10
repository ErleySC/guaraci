"""V88: uma execucao interrompida nao pode parecer completa, e a CLI mostra o progresso na hora.

O marcador `EXECUCAO_INCOMPLETA.txt` nasce quando a pasta de resultados e' criada e so' e' apagado
no fim de uma execucao que terminou. Pasta com o marcador = resultado parcial, nao usar.
"""
from __future__ import annotations

import io
import os
import sys

import matplotlib
import pytest

matplotlib.use("Agg")

import guaraci.guaraci as g  # noqa: E402
from guaraci import pipeline as pl  # noqa: E402
from guaraci.config import Config  # noqa: E402

_BASE = dict(mode="sintetico", n_per_class=10, n_synthetic_points=60, wn_min=400.0, wn_max=4001.0, n_splits_cv=2, n_repeats_cv=1,
             n_permutations=5, n_permutations_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20, n_monte_carlo=3, max_lvs=4,
             output_format="png", show_plots=False, save_dpi=40, level="N2", auto_adjust=False, objective="classificacao")


def _cfg(tmp_path):
    cfg = Config(**_BASE, input_folder=str(tmp_path / "dados"), output_root_folder=str(tmp_path / "saida"))
    os.makedirs(cfg.input_folder, exist_ok=True)
    return cfg


def _marcadores(tmp_path):
    return list((tmp_path / "saida").rglob(pl.NOME_MARCADOR_INCOMPLETA))


def test_execucao_interrompida_deixa_o_marcador_na_pasta(tmp_path, monkeypatch):
    def _quebra(*a, **k):
        raise RuntimeError("interrompida de proposito")
    monkeypatch.setattr(pl, "_etapa_regressao_pls", _quebra)
    with pytest.raises(RuntimeError, match="interrompida"):
        pl.executar(_cfg(tmp_path))
    achados = _marcadores(tmp_path)
    assert len(achados) == 1
    texto = achados[0].read_text(encoding="utf-8")
    assert "NAO terminou" in texto and "PARCIAIS" in texto


def test_execucao_concluida_nao_deixa_marcador(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "should_generate", lambda cfg, nome: False)
    pl.executar(_cfg(tmp_path))
    assert list((tmp_path / "saida").rglob("modelo_plsda.joblib"))
    assert _marcadores(tmp_path) == []


def test_marcador_acompanha_a_pasta_ate_o_fim_e_so_apaga_quando_existe(tmp_path):
    pl._marcar_execucao_incompleta(str(tmp_path))
    assert (tmp_path / pl.NOME_MARCADOR_INCOMPLETA).exists()
    pl._concluir_execucao(str(tmp_path))
    assert not (tmp_path / pl.NOME_MARCADOR_INCOMPLETA).exists()
    pl._concluir_execucao(str(tmp_path))   # sem marcador: nao quebra


def test_cli_passa_a_escrever_linha_a_linha(monkeypatch):
    saida = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    assert saida.line_buffering is False
    monkeypatch.setattr(sys, "stdout", saida)
    g.main(["ajuda", "dpi"])
    assert saida.line_buffering is True
