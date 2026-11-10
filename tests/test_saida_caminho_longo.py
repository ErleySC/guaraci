"""M-14 do parecer de 2026-09-25: caminho de saida longo (MAX_PATH do Windows)
da erro acionavel ANTES de criar pastas, nao WinError 206 cru."""
import os

import pytest

import guaraci.pipeline as pq

BS = chr(92)


def test_caminho_longo_no_windows_e_recusado_com_mensagem_acionavel(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    longo = "C:" + BS + BS.join(["pasta_com_nome_bem_longo"] * 10)
    with pytest.raises(ValueError, match="pasta_saida mais curta"):
        pq._confere_comprimento_de_saida(longo)


def test_caminho_curto_passa(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    pq._confere_comprimento_de_saida("C:" + BS + "guaraci_saida" + BS + "x")


def test_fora_do_windows_nao_se_aplica(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    pq._confere_comprimento_de_saida("/" + "a" * 400)


@pytest.mark.skipif(os.name != "nt", reason="MAX_PATH e' do Windows")
def test_executar_com_saida_longa_falha_com_mensagem_clara(tmp_path):
    longo = str(tmp_path / ("x" * 60) / ("y" * 60) / ("z" * 60))
    cfg = pq.Config(mode="sintetico", output_root_folder=longo, show_plots=False,
                    n_permutations=3, run_benchmark=False, run_monte_carlo=False,
                    run_shap=False, run_wold=False, run_cv_anova=False, run_opls=False,
                    run_ddsimca=False, executar_etapa4=False, max_lvs=3,
                    selecao_lv_cv_aninhada=False, frac_holdout=0.0)
    with pytest.raises(ValueError, match="pasta_saida mais curta"):
        pq.executar(cfg)
