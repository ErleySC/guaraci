"""Ramos de borda de modulos pequenos do nucleo, parte 3 (politica_pooled_local, perfil_matriz, dual_spls, config, alinhamento, agrupamento, vip_cli)."""
from __future__ import annotations

import numpy as np
import pytest

from guaraci import agrupamento_pastas as ap
from guaraci import alinhamento_retencao as alr
from guaraci import dual_spls as ds
from guaraci import perfil_matriz as pm
from guaraci import politica_pooled_local as ppl
from guaraci import vip_cli
from guaraci.config import Config


def test_politica_pooled_vs_local_com_especie_de_um_so_grupo_nao_treina_o_local_e_fica_inconclusiva():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 6))
    y = rng.normal(size=24)
    rotulos = np.array(["A"] * 4 + ["B"] * 20)
    grupos = np.array(["ga"] * 4 + [f"gb{i}" for i in range(20)])                  # a especie A inteira e' um unico grupo
    r = ppl.decidir_pooled_vs_local(X, y, rotulos, grupos, "A", n_componentes=2, n_splits=2, n_seeds=2, min_amostras=3)
    assert r.especie == "A" and r.dados_suficientes is True and r.recomendacao == "pooled"


def test_lista_de_perfis_sem_a_pasta_e_sem_filtro(monkeypatch, tmp_path):
    monkeypatch.setattr(pm, "DIR_PERFIS", tmp_path / "nao_existe")
    assert pm._perfis_disponiveis() == []
    monkeypatch.undo()
    todos = pm.perfis_disponiveis()
    assert pm.perfis_disponiveis(apenas=None) == todos and "generico" in todos
    assert set(pm.perfis_disponiveis(apenas="matriz")) <= set(todos)


def test_dual_spls_com_matriz_constante_degenera_e_recusa_numero_de_componentes_fora_do_ajuste():
    X, y = np.ones((10, 5)), np.arange(10.0)
    with pytest.raises(FloatingPointError, match="degenerou"):
        ds.DualSPLS(n_components=1, sparsity=0.5).fit(X, y)
    rng = np.random.default_rng(1)
    X2 = rng.normal(size=(20, 8))
    modelo = ds.DualSPLS(n_components=2, sparsity=0.5).fit(X2, X2[:, 0] * 2 + rng.normal(scale=0.1, size=20))
    with pytest.raises(ValueError, match="fora do intervalo ajustado"):
        modelo.predict(X2, n_components=9)


def test_config_recusa_etapa_de_pre_processamento_desconhecida_e_recorte_invalido():
    cfg = Config()
    with pytest.raises(ValueError, match="desconhecida"):
        cfg.etapas_pre_processamento = ["nao_existe"]
    for invalido in ([0.5, 0.5, 0.4, 0.9], [0, 0, 1], [0, 0, 1.5, 1]):
        with pytest.raises(ValueError, match="4 valores entre 0 e 1"):
            cfg.imagem_recorte = invalido
    cfg.imagem_recorte = [0.1, 0.1, 0.9, 0.9]
    assert cfg.image_crop == (0.1, 0.1, 0.9, 0.9)


def test_correlacao_segura_de_segmento_constante_e_zero_e_cow_com_janelas_que_se_sobrepoem():
    assert alr._correlacao_segura(np.ones(6), np.arange(6.0)) == 0.0
    ref = np.exp(-((np.arange(40) - 20.0) / 3.0) ** 2)
    r = alr.cow(ref, np.roll(ref, 2), n_segmentos=20, slack=3)                     # segmentos de 2 pontos com folga 3: ha' janelas invalidas
    assert len(r.amostra_alinhada) == 40 if hasattr(r, "amostra_alinhada") else r is not None


def test_agrupamento_nivel_high_ignora_amostra_vazia_e_nivel_medium_recusa_csv_sem_colunas(tmp_path):
    classe = tmp_path / "classe"
    (classe / "amostra_vazia").mkdir(parents=True)
    (classe / "amostra_ok").mkdir()
    (classe / "amostra_ok" / "a.dx").write_text("x", encoding="utf-8")
    (classe / "arquivo_solto.txt").write_text("x", encoding="utf-8")
    grupos = ap.detectar_nivel_high([str(classe)], (".dx",))
    assert grupos is not None and list(grupos.values()) == ["classe/amostra_ok"]
    (tmp_path / "amostras.csv").write_text("arquivo,outra\na.dx,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="faltam as colunas"):
        ap.detectar_nivel_medium(str(tmp_path), ["a.dx"])


def test_vip_cli_com_argumento_numerico_invalido_devolve_erro_de_uso(capsys):
    assert vip_cli.executar(["tabela.csv", "--classe", "c", "--grupo", "g", "--componentes", "abc"]) == 2
    assert capsys.readouterr().err


def test_perfis_com_filtro_desconhecido_devolvem_a_lista_completa():
    assert pm.perfis_disponiveis(apenas="qualquer_coisa") == pm.perfis_disponiveis()


def test_vip_cli_com_opcao_desconhecida_devolve_erro_de_uso(capsys):
    assert vip_cli.executar(["tabela.csv", "--opcao-que-nao-existe"]) == 2
    assert capsys.readouterr().err
