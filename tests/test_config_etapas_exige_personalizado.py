# -*- coding: utf-8 -*-
from guaraci.config_io import load_config


def test_etapas_sem_personalizado_avisa(tmp_path, capsys):
    p = tmp_path / "c.yaml"
    p.write_text("etapas_pre_processamento: [emsc, mc]\n", encoding="utf-8")
    load_config(str(p))
    assert "Personalizado" in capsys.readouterr().err


def test_etapas_com_personalizado_vale(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("pre_processamento: Personalizado\netapas_pre_processamento: [emsc, mc]\n", encoding="utf-8")
    assert load_config(str(p)).etapas_pre_processamento == ("emsc", "mc")
