# -*- coding: utf-8 -*-
"""Achados do primeiro uso simulado por usuario NOVO (roteiro de teste
externo, 2026-09-23), modo csv pela CLI:
  1. o checklist pre-execucao exigia a pasta `dados` (so' usada pelo modo
     dx) e BLOQUEAVA a execucao de quem so' tinha um CSV;
  2. csv (sem mae_id) mantinha grouping_guarantee="high" e a auditoria
     afirmava "mae_id confiavel para toda amostra" -- falso."""
from __future__ import annotations

import guaraci.guaraci as guaraci_mod
from guaraci.config import Config
from guaraci.pipeline import load_data


def _csv(tmp_path):
    p = tmp_path / "d.csv"
    linhas = ["4000,5000,6000,classe"]
    for i in range(6):
        linhas.append(f"{1+i*0.01},2,3,{'A' if i % 2 else 'B'}")
    p.write_text("\n".join(linhas), encoding="utf-8")
    return p


def test_checklist_csv_nao_exige_pasta_dados(tmp_path):
    cfg = Config(mode="csv", csv_file=str(_csv(tmp_path)),
                 input_folder=str(tmp_path / "pasta_que_nao_existe"))
    ok, erros, _checks = guaraci_mod._checklist(cfg)
    assert "pasta_dados" not in erros
    assert ok is True


def test_checklist_csv_sem_arquivo_ainda_falha(tmp_path):
    cfg = Config(mode="csv", csv_file=str(tmp_path / "nao_existe.csv"))
    ok, erros, _ = guaraci_mod._checklist(cfg)
    assert "arquivo_csv" in erros and ok is False


def test_checklist_dx_continua_exigindo_pasta(tmp_path):
    cfg = Config(mode="dx", input_folder=str(tmp_path / "nao_existe"))
    ok, erros, _ = guaraci_mod._checklist(cfg)
    assert "pasta_dados" in erros and ok is False


def test_csv_sem_mae_id_nao_alega_garantia_de_agrupamento(tmp_path):
    cfg = Config(mode="csv", csv_file=str(_csv(tmp_path)), class_column="classe")
    assert cfg.grouping_guarantee == "high"      # default
    load_data(cfg)
    assert cfg.grouping_guarantee == "none"


def test_sintetico_com_mae_id_mantem_high():
    cfg = Config(mode="sintetico", n_per_class=4, n_synthetic_points=20,
                 wn_min=400.0, wn_max=2001.0)
    load_data(cfg)
    assert cfg.grouping_guarantee == "high"
