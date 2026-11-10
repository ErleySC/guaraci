# -*- coding: utf-8 -*-
"""LOD/LOQ/RMSEP saem na unidade do teor de referencia, nao em "%" fixo (auditoria A-07)."""
from __future__ import annotations

from guaraci import resultados_io as rio
from guaraci.resumo_parse import parse_regressao

ESP = {"especie": "Esp_A", "n_lv": 3, "rmsep": 0.5, "r2val": 0.9, "lod": 0.06, "loq": 0.18,
       "sensibilidade": 1.2, "seletividade_media": 0.4, "lod_ic_baixo": 0.04, "lod_ic_alto": 0.09,
       "loq_ic_baixo": 0.12, "loq_ic_alto": 0.27, "lod_ic_confianca": 0.95, "validacao_teor_min": 0.0,
       "validacao_teor_max": 20.0, "validacao_teor_dp": 6.1}


def test_resumo_nao_poe_percentual_nos_valores_de_merito(tmp_path):
    pasta = str(tmp_path)
    (tmp_path / "resumo_modelo.txt").write_text("Resumo\n", encoding="utf-8")
    rio.append_regression_summary(pasta, pooled={"rmsep": 0.5}, tabela_especie=[ESP],
                                  fom_pooled={"lod": 0.06, "loq": 0.18, "sensibilidade": 1.2})
    txt = "".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("resumo_modelo.*"))
    assert "LOD .......: 0.06" in txt and "LOD .......: 0.06%" not in txt
    assert "]% DP" not in txt and ("unidade do teor de referencia" in txt or "unit of the reference concentration" in txt)


def test_o_leitor_do_resumo_entende_os_dois_formatos(tmp_path):
    novo = ("Per-species figures of merit:\n"
            "    Species             LVs   RMSEP  R2val     LOD     LOQ     SEN    SEL\n"
            "    ------------------------------------------------------------------\n"
            "    Esp_A                 3    0.50  0.900    0.06    0.18   1.200  0.400\n"
            "      IC(95%): LOD [0.04-0.09]  LOQ [0.12-0.27]  |  validacao: [0.00, 20.00] DP=6.10\n"
            "      Intervalo de predicao (conforme, 95%): +/-1.20 (25 grupos de validacao)\n")
    antigo = novo.replace("20.00] DP=6.10", "20.00]% DP=6.10%").replace("+/-1.20 ", "+/-1.20% ")
    for texto in (novo, antigo):
        e = parse_regressao("  PLS Regression — Analytical Figures of Merit\n" + texto)["especies"][0]
        assert e["faixa_validacao"] == "0.00, 20.00" and e["dp_validacao"] == "6.10", texto


def test_model_card_sem_percentual_nos_valores(tmp_path):
    (tmp_path / "model_card.md").write_text("# Model Card\n", encoding="utf-8")
    rio.append_regression_model_card(str(tmp_path), pooled={"rmsep": 0.5, "r2v": 0.9}, tabela_especie=[ESP],
                                     fom_pooled={"lod": 0.06, "loq": 0.18, "sensibilidade": 1.2})
    txt = (tmp_path / "model_card.md").read_text(encoding="utf-8")
    assert "LOD=0.06 " in txt and "LOD=0.06%" not in txt and "0.27]" in txt and "0.27]%" not in txt
