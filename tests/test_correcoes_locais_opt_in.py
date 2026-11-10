"""M-15(b) do parecer de 2026-09-25: correcoes locais de metadado so' com opcao
explicita, e sempre registradas (arquivo, sha256, n de amostras alteradas)."""
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

import guaraci.pipeline as pq
from guaraci import dados_io


def _dir_local(tmp_path):
    d = tmp_path / ".guaraci_local"
    d.mkdir()
    (d / "alias_mae_id.csv").write_text("ZZZ-01-02-2099;ZZZ-05-06-2099\n", encoding="utf-8")
    (d / "pureza_indeterminada.csv").write_text("TITULO_X;motivo teste\n", encoding="utf-8")
    return d


@pytest.fixture(autouse=True)
def _restaura():
    yield
    dados_io.definir_correcoes_locais(False)


def test_importar_o_pacote_nao_le_correcoes_do_diretorio_do_usuario(tmp_path):
    """Processo NOVO com HOME apontando para um diretorio que TEM correcoes:
    as tabelas ficam vazias (nada aplicado na importacao)."""
    _dir_local(tmp_path)
    codigo = ("import guaraci.dados_io as d;"
              "print(len(d._ALIAS_MAE_ID), len(d._TITLES_PUREZA_INDETERMINADA), "
              "len(d._CORRECOES_TITLE_CONHECIDAS))")
    import os
    env = dict(os.environ, HOME=str(tmp_path), USERPROFILE=str(tmp_path))
    env.pop("GUARACI_APLICAR_CORRECOES_LOCAIS", None)
    saida = subprocess.run([sys.executable, "-c", codigo], env=env, capture_output=True,
                           text=True, cwd=str(Path(__file__).resolve().parents[1]),
                           timeout=120)
    assert saida.stdout.split()[-3:] == ["0", "0", "0"], saida.stdout + saida.stderr


def test_com_a_opcao_explicita_aplica_e_registra_sha256_e_contagem(tmp_path, monkeypatch):
    d = _dir_local(tmp_path)
    monkeypatch.setattr(dados_io, "_DIR_LOCAL", d)
    assert pq.parse_title("ZZZ-01-02-2099-T1")["mae_id"] == "ZZZ-01-02-2099"   # desligado
    dados_io.definir_correcoes_locais(True)
    assert pq.parse_title("ZZZ-01-02-2099-T1")["mae_id"] == "ZZZ-05-06-2099"
    r = dados_io.resumo_correcoes_locais()
    assert r["ativo"] and r["alias_aplicados"] == 1
    arq = {a["arquivo"]: a for a in r["arquivos"]}
    assert arq["alias_mae_id.csv"]["sha256"] == hashlib.sha256(
        (d / "alias_mae_id.csv").read_bytes()).hexdigest()
    # desligar zera tudo
    dados_io.definir_correcoes_locais(False)
    assert dados_io.resumo_correcoes_locais()["ativo"] is False
    assert pq.parse_title("ZZZ-01-02-2099-T1")["mae_id"] == "ZZZ-01-02-2099"


def test_load_data_registra_na_config_o_que_foi_aplicado(tmp_path, monkeypatch, capsys):
    d = _dir_local(tmp_path)
    monkeypatch.setattr(dados_io, "_DIR_LOCAL", d)
    cfg = pq.Config(mode="sintetico")
    dados_io.load_data(cfg)
    assert cfg.correcoes_locais_aplicadas["ativo"] is False
    cfg2 = pq.Config(mode="sintetico", apply_local_corrections=True)
    dados_io.load_data(cfg2)
    assert cfg2.correcoes_locais_aplicadas["ativo"] is True
    assert "Correcoes de metadado LOCAIS aplicadas" in capsys.readouterr().out


@pytest.mark.slow
def test_o_registro_chega_ao_resumo_do_modelo(tmp_path):
    cfg = pq.Config(mode="sintetico", output_root_folder=str(tmp_path), show_plots=False,
                    n_permutations=5, run_benchmark=False, run_monte_carlo=False,
                    run_shap=False, run_wold=False, run_cv_anova=False, run_opls=False,
                    run_ddsimca=False, executar_etapa4=False, max_lvs=4,
                    selecao_lv_cv_aninhada=False, frac_holdout=0.0)
    pq.executar(cfg)
    r = list(tmp_path.rglob("resumo_modelo.txt"))[0].read_text(encoding="utf-8", errors="replace")
    assert "correcoes_locais" in r and "'ativo': False" in r
