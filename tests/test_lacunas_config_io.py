"""Ramos de borda de guaraci.config_io: validacao semantica, YAML sem PyYAML e checagem da fonte de dados por modo de entrada."""
from __future__ import annotations

import sys

import pytest

from guaraci import config_io as cio
from guaraci.config import Config


# ── validacao semantica ─────────────────────────────────────────────────────
def test_selecao_de_graficos_com_nome_desconhecido_e_savgol_invalido_sao_erros():
    erros = cio._validar_semantico(Config(figure_selection="pca_scores,nao_existe", sg_window=6, sg_polyorder=2))
    assert any("nome(s) desconhecido(s) ['nao_existe']" in e for e in erros)
    assert any("Savitzky-Golay invalido" in e for e in erros)
    assert not any("Savitzky" in e for e in cio._validar_semantico(Config(sg_window=11, sg_polyorder=2)))


def test_savgol_com_valor_que_nao_e_numero_nao_gera_erro_extra():
    cfg = Config()
    cfg.sg_window = "abc"                         # config manual malformada: a checagem cruzada nao tem o que comparar
    assert not any("Savitzky" in e for e in cio._validar_semantico(cfg))


# ── YAML ────────────────────────────────────────────────────────────────────
def test_sem_pyyaml_o_aspeamento_e_conservador_e_o_carregamento_recusa(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "yaml", None)                       # import yaml -> ImportError
    assert cio._yaml_reinterpretaria("qualquer") is True
    with pytest.raises(RuntimeError, match="PyYAML nao instalado"):
        cio.load_config(str(tmp_path / "config.yaml"))


def test_texto_que_nao_parseia_como_escalar_yaml_e_aspeado():
    assert cio._yaml_reinterpretaria("a: [b") is True
    assert cio._yaml_reinterpretaria("texto simples") is False


# ── fonte de dados por modo ─────────────────────────────────────────────────
def _cfg(modo, **kw):
    c = Config(mode=modo)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_multielementar_exige_arquivo_unidade_e_coluna_de_grupo(tmp_path):
    ausente = cio._validar_pasta_dados(_cfg("multielementar", csv_file=str(tmp_path / "nao.csv")))
    assert ausente[0] is False and "arquivo ICP nao encontrado" in ausente[1]
    arq = tmp_path / "icp.csv"
    arq.write_text("a;b\n1;2\n", encoding="utf-8")
    sem_unidade = cio._validar_pasta_dados(_cfg("multielementar", csv_file=str(arq), icp_unidade="", group_column="g"))
    assert sem_unidade[0] is False and "icp_unidade" in sem_unidade[1]
    sem_grupo = cio._validar_pasta_dados(_cfg("multielementar", csv_file=str(arq), icp_unidade="mg/kg", group_column=""))
    assert sem_grupo[0] is False and "coluna_grupo" in sem_grupo[1]
    assert cio._validar_pasta_dados(_cfg("multielementar", csv_file=str(arq), icp_unidade="mg/kg", group_column="g"))[0] is True


def test_imagem_com_pasta_inexistente():
    ok, msg = cio._validar_pasta_dados(_cfg("imagem", input_folder="/nao/existe/mesmo"))
    assert ok is False and "pasta nao encontrada" in msg


@pytest.mark.parametrize("modo, ext", [("spc", ".spc"), ("sp", ".sp"), ("gcms", ".cdf")])
def test_formatos_de_arquivo_por_extensao(tmp_path, modo, ext):
    assert cio._validar_pasta_dados(_cfg(modo, input_folder=str(tmp_path / "nao_existe")))[1].startswith("pasta nao encontrada")
    pasta = tmp_path / "dados"
    pasta.mkdir()
    ok_vazio, msg_vazio = cio._validar_pasta_dados(_cfg(modo, input_folder=str(pasta)))
    assert ok_vazio is False and f"nenhum arquivo {ext}" in msg_vazio
    (pasta / f"a{ext}").write_text("x", encoding="utf-8")
    (pasta / f"B{ext.upper()}").write_text("x", encoding="utf-8")
    ok, msg = cio._validar_pasta_dados(_cfg(modo, input_folder=str(pasta)))
    assert ok is True and "2 arquivos" in msg


def test_opus_exige_pasta_com_arquivos(tmp_path):
    assert cio._validar_pasta_dados(_cfg("opus", input_folder=str(tmp_path / "nao_existe")))[0] is False
    vazia = tmp_path / "vazia"
    vazia.mkdir()
    ok, msg = cio._validar_pasta_dados(_cfg("opus", input_folder=str(vazia)))
    assert ok is False and "vazia" in msg
    (vazia / "espectro.0").write_text("x", encoding="utf-8")
    assert cio._validar_pasta_dados(_cfg("opus", input_folder=str(vazia)))[0] is True


def test_rmn_hplc_e_eem_exigem_a_estrutura_de_cada_formato(tmp_path):
    for modo in ("rmn", "hplc", "eem"):
        assert cio._validar_pasta_dados(_cfg(modo, input_folder=str(tmp_path / "nao_existe")))[1].startswith("pasta nao encontrada")
        vazia = tmp_path / f"vazia_{modo}"
        vazia.mkdir()
        assert cio._validar_pasta_dados(_cfg(modo, input_folder=str(vazia)))[0] is False
    rmn = tmp_path / "rmn_ok" / "amostra" / "pdata"
    rmn.mkdir(parents=True)
    assert cio._validar_pasta_dados(_cfg("rmn", input_folder=str(tmp_path / "rmn_ok")))[0] is True
    hplc = tmp_path / "hplc_ok" / "amostra1.D"
    hplc.mkdir(parents=True)
    assert cio._validar_pasta_dados(_cfg("hplc", input_folder=str(tmp_path / "hplc_ok")))[0] is True
    eem = tmp_path / "eem_ok"
    eem.mkdir()
    (eem / "1_RM.dat").write_text("x", encoding="utf-8")
    assert cio._validar_pasta_dados(_cfg("eem", input_folder=str(eem)))[0] is True


def test_selecao_de_graficos_com_nomes_do_catalogo_nao_gera_erro():
    from guaraci.catalogo_tecnicas import GRAFICOS
    nomes = ",".join(list(GRAFICOS)[:2])
    assert not any("graficos" in e for e in cio._validar_semantico(Config(figure_selection=nomes)))
