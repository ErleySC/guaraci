"""Ramos do orquestrador (`executar`) exercitados por execucoes sinteticas minusculas, com opcoes ligadas e com falhas injetadas.

Cada execucao e' pequena (poucos pontos, poucas permutacoes) e roda em processo (para a cobertura contar). Os testes afirmam o que o
orquestrador ENTREGA (arquivos, avisos no log), nao o desempenho do modelo. As falhas sao injetadas trocando a funcao opcional por uma
que levanta erro: o orquestrador precisa registrar o aviso e seguir.
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from guaraci import pipeline as pl  # noqa: E402
from guaraci.config import Config  # noqa: E402
from tests._resumo import canonico

_BASE = dict(mode="sintetico", n_per_class=10, n_synthetic_points=60, wn_min=400.0, wn_max=4001.0, n_splits_cv=2, n_repeats_cv=1,
             n_permutations=5, n_permutations_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20, n_monte_carlo=3, max_lvs=4,
             output_format="png", show_plots=False, save_dpi=40)


def _rodar(tmp_path, **kw):
    cfg = Config(**{**_BASE, **kw}, input_folder=str(tmp_path / "dados"), output_root_folder=str(tmp_path / "saida"))
    os.makedirs(cfg.input_folder, exist_ok=True)
    pl.executar(cfg)
    return cfg


def _quebra(msg):
    def f(*a, **k):
        raise RuntimeError(msg)
    return f


def test_exploratorio_n1_com_tudo_liberado_ddsimca_ignorado_e_hca_comparado(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pl, "should_generate", lambda cfg, nome: True)
    _rodar(tmp_path, level="N1", auto_adjust=False, objective="exploratorio", figure_selection="todos", compare_hca_pipelines=True,
           run_ddsimca=True, run_opls=True)
    assert "IGNORADO: nivel N1" in capsys.readouterr().out


def test_classificacao_n2_com_todas_as_opcoes_e_ddsimca_sobre_todas_as_amostras(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pl, "should_generate", lambda cfg, nome: True)
    _rodar(tmp_path, level="N2", auto_adjust=False, objective="classificacao", run_ddsimca=True, ddsimca_train_on="todos",
           ddsimca_pcv=True, run_martens=True, run_wold=True, run_cv_anova=True, run_opls=True)
    saida = capsys.readouterr().out
    assert "[DD-SIMCA] (n_components" in saida and "OPLS-DA" in saida


def test_anexos_opcionais_que_falham_viram_aviso_e_a_execucao_termina(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pl, "should_generate", lambda cfg, nome: True)
    monkeypatch.setattr(pl, "fig_roc_auc", _quebra("roc quebrou"))
    monkeypatch.setattr(pl, "fig_cooman_ddsimca", _quebra("cooman quebrou"))
    monkeypatch.setattr(pl, "fig_splot_opls", _quebra("splot quebrou"))
    real_boot = pl.bootstrap_vip_stratified
    monkeypatch.setattr(pl, "bootstrap_vip_stratified", lambda *a, **k: {**real_boot(*a, **k), "n_validos": 0})
    real_martens = pl.martens_uncertainty_test
    monkeypatch.setattr(pl, "martens_uncertainty_test", lambda *a, **k: {**real_martens(*a, **k), "n_folds_validos": np.asarray(2)})
    monkeypatch.setattr(pl, "ddsimca_pcv_sensitivity", lambda *a, **k: {"disponivel": False, "aviso": "prcv ausente", "sensibilidade": np.nan,
                                                                        "n_grupos": 0, "n_amostras": 0})
    _rodar(tmp_path, level="N2", auto_adjust=False, objective="classificacao", detailed_figures=True, run_ddsimca=True, ddsimca_pcv=True, run_martens=True,
           run_opls=True)
    saida = capsys.readouterr().out
    for trecho in ("ROC/AUC: roc quebrou", "Cooman's Plot: cooman quebrou", "OPLS-DA: splot quebrou", "0 iteracoes validas",
                   "(<3)", "PCV: prcv ausente"):
        assert trecho in saida, trecho


def test_com_toda_figura_dispensada_a_execucao_ainda_entrega_o_modelo(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "should_generate", lambda cfg, nome: False)
    cfg = _rodar(tmp_path, level="N2", auto_adjust=False, objective="classificacao")
    assert list((tmp_path / "saida").rglob("modelo_plsda.joblib")) and cfg is not None


def test_ddsimca_treinado_em_todas_as_amostras_fora_do_n2(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pl, "should_generate", lambda cfg, nome: True)
    _rodar(tmp_path, level="N3", auto_adjust=False, objective="classificacao", run_ddsimca=True, ddsimca_train_on="todos", ddsimca_pcv=True)
    assert "[DD-SIMCA] (n_components" in capsys.readouterr().out


def test_cargas_e_biplot_dispensados_enquanto_o_resto_e_gerado(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "should_generate", lambda cfg, nome: nome not in ("loadings", "biplot"))
    _rodar(tmp_path, level="N2", auto_adjust=False, objective="exploratorio", figure_selection="todos")
    nomes = {p.name for p in (tmp_path / "saida").rglob("*.png")}
    assert not any("loadings" in n or "biplot" in n for n in nomes) and nomes


def _csv(tmp_path, classes, nome="d.csv"):
    import pandas as pd
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(len(classes), 40)) + 1, columns=[f"{w:.1f}" for w in np.linspace(4000, 7000, 40)])
    df.insert(0, "classe", classes)
    caminho = tmp_path / nome
    df.to_csv(caminho, index=False)
    return str(caminho)


def test_csv_de_uma_so_classe_sem_teor_e_sem_grupo_ainda_roda_com_avisos(tmp_path, capsys):
    cfg = Config(mode="csv", csv_file=_csv(tmp_path, ["A"] * 12), class_column="classe", conc_column="",
                 output_root_folder=str(tmp_path / "saida"), n_splits_cv=2, n_repeats_cv=1, n_permutations=5, n_permutations_wold=5,
                 n_bootstrap_vip=3, n_bootstrap_bca=20, n_monte_carlo=3, max_lvs=3, run_ddsimca=True, run_opls=True, level="N2",
                 objective="classificacao", auto_adjust=False, wn_min=4000.0, wn_max=7000.0, output_format="png", save_dpi=40)
    pl.executar(cfg)
    saida = capsys.readouterr().out
    assert "nivel N2 sem dados de concentra" in saida and "OPLS-DA requer >= 2 classes" in saida and "mae_id ausente" in saida


# ── quantificacao (N3) por especie, com holdout ─────────────────────────────
_N3 = dict(level="N3", auto_adjust=False, objective="quantificacao", synthetic_adulterants=("A", "S"), n_per_class=6, n_synthetic_replicates=2,
           frac_holdout=0.25)


def test_quantificacao_por_especie_com_holdout_lod_benchmark_e_mapa_de_calor(tmp_path, capsys):
    _rodar(tmp_path, **_N3, run_benchmark_regression=True)
    saida = capsys.readouterr().out
    assert "RMSEP (HOLDOUT externo" in saida and "Especies modeladas" in saida
    assert list((tmp_path / "saida").rglob("*benchmark_regress*"))


def test_quantificacao_com_extras_que_falham_vira_aviso_e_termina(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pl, "avaliar_quantificacao_holdout", _quebra("holdout quebrou"))
    monkeypatch.setattr(pl, "benchmark_regression_by_species", _quebra("benchmark quebrou"))
    monkeypatch.setattr(pl, "r2cv_species_by_adulterant", _quebra("r2cv quebrou"))
    chamadas = {"n": 0}
    real = pl.save_manifest

    def segunda_vez_falha(*a, **k):
        chamadas["n"] += 1
        if chamadas["n"] >= 2:
            raise OSError("disco cheio")
        return real(*a, **k)
    monkeypatch.setattr(pl, "save_manifest", segunda_vez_falha)
    _rodar(tmp_path, **_N3, run_benchmark_regression=True)
    saida = capsys.readouterr().out
    for trecho in ("RMSEP do holdout nao calculado: holdout quebrou", "Benchmark de regressao falhou: benchmark quebrou",
                   "Heatmap especie x adulterante: r2cv quebrou", "Nao foi possivel anexar regressao por especie"):
        assert trecho in saida, trecho


def test_quantificacao_sem_mapa_de_calor_quando_o_modo_o_dispensa(tmp_path, monkeypatch):
    r2cv = {"especies": ["A"], "adulterantes": ["S"], "matriz": {("A", "S"): 0.5}, "n_ok": 0, "n_falhas": 1, "n_na": 0, "n_total": 1,
            "limiar_r2": 0.7}
    monkeypatch.setattr(pl, "r2cv_species_by_adulterant", lambda *a, **k: r2cv)
    chamadas = []
    monkeypatch.setattr(pl, "fig_heatmap_species_by_adulterant", lambda *a, **k: chamadas.append("mapa"))
    real = pl.should_generate
    monkeypatch.setattr(pl, "should_generate", lambda cfg, nome: False if nome == "heatmap_especie_adulterante" else real(cfg, nome))
    _rodar(tmp_path, **_N3)
    assert chamadas == []


def test_exportacao_do_modelo_tolera_cada_anexo_opcional_que_falha(tmp_path, capsys, monkeypatch):
    for nome, msg in (("mean_and_dof_moments", "momentos"), ("training_applicability_domain", "dominio"),
                      ("train_identification_ensemble", "ensemble"), ("DDSimca", "ddsimca")):
        monkeypatch.setattr(pl, nome, _quebra(msg + " quebrou"))
    _rodar(tmp_path, level="N1", auto_adjust=False, objective="classificacao")
    saida = capsys.readouterr().out
    for trecho in ("parametros da distancia combinada", "Dominio de aplicabilidade nao pode ser exportado", "Ensemble de identificacao",
                   "DD-SIMCA por especie nao pode ser exportado"):
        assert trecho in saida, trecho


def test_exportacao_com_referencia_de_cv_do_dominio_que_falha(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pl, "ad_rejection_rate_cv", _quebra("referencia quebrou"))
    _rodar(tmp_path, level="N2", auto_adjust=False, objective="classificacao")
    assert "referencia de CV do dominio de aplicabilidade nao calculada" in capsys.readouterr().out


def test_exportacao_que_falha_ao_gravar_o_modelo_e_registrada_e_a_execucao_continua(tmp_path, capsys, monkeypatch):
    import joblib
    monkeypatch.setattr(joblib, "dump", _quebra("disco cheio"))
    _rodar(tmp_path, level="N2", auto_adjust=False, objective="classificacao")
    assert "Exportacao do modelo pulada: disco cheio" in capsys.readouterr().out


def test_identificacao_com_cobertura_validada_nao_emite_o_aviso_de_bloqueio(tmp_path, capsys, monkeypatch):
    from guaraci.identificacao import CoverageStatus
    monkeypatch.setattr(pl, "train_identification_ensemble", lambda *a, **k: {("A", "S"): {"cobertura_status": CoverageStatus.VALIDATED}})
    monkeypatch.setattr(pl, "append_identification_model_card", lambda *a, **k: None)
    _rodar(tmp_path, level="N2", auto_adjust=False, objective="classificacao")
    assert "com cobertura VALIDADA" in capsys.readouterr().out


def test_quantificacao_sem_especie_apta_ou_com_falha_geral_e_registrada(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pl, "pls_regression_by_species", lambda *a, **k: None)
    _rodar(tmp_path, **_N3)
    assert "Nenhuma especie com amostras suficientes para regressao" in capsys.readouterr().out
    monkeypatch.setattr(pl, "pls_regression_by_species", _quebra("regressao quebrou"))
    _rodar(tmp_path / "b", **_N3)
    assert "Regressao por especie falhou: regressao quebrou" in capsys.readouterr().out


def test_quantificacao_em_modo_controle_sem_holdout_e_sem_replicas_para_lod(tmp_path, capsys):
    _rodar(tmp_path, **{**_N3, "frac_holdout": 0.0, "n_synthetic_replicates": 1, "label_mode": "controle"})
    saida = capsys.readouterr().out
    assert "LOD/LOQ: N/A" in saida or "Especies modeladas" in saida


def test_benchmark_de_regressao_sem_resultado_e_mapa_de_calor_gerado(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pl, "benchmark_regression_by_species", lambda *a, **k: None)
    r2cv = {"especies": ["A"], "adulterantes": ["S"], "matriz": {("A", "S"): 0.5}, "n_ok": 0, "n_falhas": 1, "n_na": 0, "n_total": 1,
            "limiar_r2": 0.7}
    monkeypatch.setattr(pl, "r2cv_species_by_adulterant", lambda *a, **k: r2cv)
    chamadas = []
    monkeypatch.setattr(pl, "fig_heatmap_species_by_adulterant", lambda *a, **k: chamadas.append("mapa"))
    _rodar(tmp_path, **_N3, run_benchmark_regression=True, figure_selection="todos")
    assert chamadas == ["mapa"] and "Nenhuma especie com amostras suficientes para o" in capsys.readouterr().out


def test_regressao_pulada_por_teor_ausente_constante_ou_escasso(tmp_path_factory, capsys):
    tmp_path = tmp_path_factory.mktemp("r")
    import pandas as pd
    rng = np.random.default_rng(0)

    def csv(nome, conc):
        df = pd.DataFrame(rng.normal(size=(len(conc), 30)) + 1, columns=[f"{w:.1f}" for w in np.linspace(4000, 7000, 30)])
        df.insert(0, "classe", ["A", "B"] * (len(conc) // 2))
        df.insert(1, "teor", conc)
        p = tmp_path / nome
        df.to_csv(p, index=False)
        return str(p)

    def rodar(nome, conc):
        cfg = Config(mode="csv", csv_file=csv(nome, conc), class_column="classe", conc_column="teor", output_root_folder=str(tmp_path / ("s_" + nome)),
                     n_splits_cv=2, n_repeats_cv=1, n_permutations=5, n_permutations_wold=5, n_bootstrap_vip=3, n_bootstrap_bca=20, n_monte_carlo=3,
                     max_lvs=3, level="N3", objective="quantificacao", auto_adjust=False, wn_min=4000.0, wn_max=7000.0, output_format="png", save_dpi=40)
        pl.executar(cfg)
        return capsys.readouterr().out
    assert "amostras com conc=NaN" in rodar("nan.csv", [0.0, 1.0, 2.0, np.nan] * 5)
    assert "variancia de conc ~= 0" in rodar("zero.csv", [0.0] * 20)
    assert "apenas 4 amostras com resposta > 0" in rodar("pouco.csv", [0.0] * 16 + [1.0, 2.0, 3.0, 4.0])


def test_avaliacao_do_holdout_de_quantificacao_pula_especie_com_menos_de_duas_amostras_e_sem_nenhuma_devolve_none():
    from types import SimpleNamespace
    info = {"A": {"pipeline": SimpleNamespace(predict=lambda X: np.zeros(len(X)))}}
    assert pl.avaliar_quantificacao_holdout(info, np.zeros((3, 2)), np.array(["A", "B", "B"]), np.array([1.0, 2.0, 3.0])) is None


def test_holdout_agrupado_com_um_so_grupo_e_recusado():
    with pytest.raises(ValueError, match="grupos insuficientes"):
        pl.holdout_agrupado_estratificado(np.array(list("aabb")), np.array(["g"] * 4), 0.25, 0)


def test_quantificacao_em_modo_controle_com_holdout_usa_os_rotulos_verdadeiros_e_holdout_sem_resultado_e_aceito(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pl, "avaliar_quantificacao_holdout", lambda *a, **k: None)
    _rodar(tmp_path, **{**_N3, "label_mode": "controle"})
    assert "Especies modeladas" in capsys.readouterr().out


def test_regressao_por_especie_sem_pipelines_nao_tenta_anexar_ao_modelo(tmp_path, capsys, monkeypatch):
    real = pl.pls_regression_by_species
    monkeypatch.setattr(pl, "pls_regression_by_species", lambda *a, **k: {**real(*a, **k), "pipelines_especie": {}})
    _rodar(tmp_path, **_N3)
    assert "Atualizado com regressao por especie" not in capsys.readouterr().out


def test_holdout_do_pooled_sem_dois_valores_de_teor_validos_ou_que_falha_vira_aviso(tmp_path):
    X, conc, rot, mae = (np.random.default_rng(0).normal(size=(48, 12)), np.tile(np.concatenate([np.zeros(6), np.linspace(1, 10, 18)]), 2),
                         np.repeat(["A", "B"], 24), np.array([f"{r}-{i // 3:03d}-S1.00" for r, i in zip(np.repeat(["A", "B"], 24), np.tile(np.arange(24), 2))]))
    cfg = Config(seed=0, max_lvs=3, n_splits_cv=2, frac_cal=0.7, default_preprocessing="mc")
    logs = tmp_path / "Logs"
    logs.mkdir()
    r = pl.pls_regressao_pooled(X, conc, rot, mae, cfg, str(tmp_path), str(logs), n_splits=2, X_holdout=X[:5], conc_holdout=np.full(5, np.nan))
    assert r["holdout"] is None
    r2 = pl.pls_regressao_pooled(X, conc, rot, mae, cfg, str(tmp_path), str(logs), n_splits=2, X_holdout=X[:5, :3], conc_holdout=np.arange(5.0))
    assert r2["holdout"] is None                                     # matriz de outra largura: o erro e' registrado, nao propagado


@pytest.mark.parametrize("ram_ok", [True, False])
def test_benchmark_e_monte_carlo_do_orquestrador_dependem_da_ram_e_toleram_falha(tmp_path, capsys, monkeypatch, ram_ok):
    import pandas as pd
    monkeypatch.setattr(pl, "_verificar_ram", lambda gb, rotulo: ram_ok)
    monkeypatch.setattr(pl, "benchmark_classifiers", lambda *a, **k: pd.DataFrame({"Classifier": ["PLS-DA"], "Bal.Acc mean": [0.9]}))
    monkeypatch.setattr(pl, "monte_carlo_cv", lambda *a, **k: pd.DataFrame({"Model": ["PLS-DA"], "mean": [0.9]}))
    _rodar(tmp_path, level="N2", auto_adjust=False, objective="classificacao", run_benchmark=True, run_monte_carlo=True, figure_selection="todos")
    saida = capsys.readouterr().out
    assert ("Auto-Benchmark" in saida) and (("Bal.Acc mean" in saida) is ram_ok)


def test_benchmark_e_monte_carlo_que_falham_viram_aviso(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pl, "_verificar_ram", lambda gb, rotulo: True)
    monkeypatch.setattr(pl, "benchmark_classifiers", _quebra("benchmark quebrou"))
    monkeypatch.setattr(pl, "monte_carlo_cv", _quebra("monte carlo quebrou"))
    _rodar(tmp_path, level="N2", auto_adjust=False, objective="classificacao", run_benchmark=True, run_monte_carlo=True, figure_selection="todos")
    saida = capsys.readouterr().out
    assert "Benchmark falhou: benchmark quebrou" in saida and "Monte Carlo CV falhou: monte carlo quebrou" in saida


# ── resumo, holdout e carga ─────────────────────────────────────────────────
def test_sem_holdout_o_resumo_nao_traz_bca_nem_conformal_e_o_pcv_com_aviso_vai_para_o_resumo(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "should_generate", lambda cfg, nome: True)
    monkeypatch.setattr(pl, "ddsimca_pcv_sensitivity", lambda *a, **k: {"disponivel": True, "sensibilidade": 0.5, "aviso": "exploratorio",
                                                                        "n_grupos": 3, "n_amostras": 9})
    cfg = _rodar(tmp_path, level="N2", auto_adjust=False, objective="classificacao", frac_holdout=0.0, run_ddsimca=True, ddsimca_pcv=True)
    resumo = canonico(next((tmp_path / "saida").rglob("resumo_modelo.txt")))
    assert "BCa Holdout" not in resumo and "AVISO PCV" in resumo and cfg is not None


def test_classe_com_mais_de_30_por_cento_de_outliers_t2_e_sinalizada(tmp_path, capsys, monkeypatch):
    real = pl.fig3_outliers

    def todos_outliers(*a, **k):
        T2, Q, t2_lim, q_lim, out_t2, out_q = real(*a, **k)
        return T2, Q, t2_lim, q_lim, np.arange(len(T2)), out_q
    monkeypatch.setattr(pl, "fig3_outliers", todos_outliers)
    _rodar(tmp_path, level="N1", auto_adjust=False, objective="classificacao")
    assert "possivel batch/lote anomalo" in capsys.readouterr().out


def test_holdout_legado_por_grupos_holdout_vazio_e_classe_fora_do_holdout(tmp_path, capsys, monkeypatch):
    _rodar(tmp_path / "a", level="N1", auto_adjust=False, objective="classificacao", holdout_estratificado=False)
    assert list((tmp_path / "a" / "saida").rglob("modelo_plsda.joblib"))
    capsys.readouterr()
    monkeypatch.setattr(pl, "holdout_agrupado_estratificado", lambda y, g, frac, seed: (np.arange(len(y)), np.array([], dtype=int)))
    _rodar(tmp_path / "b", level="N1", auto_adjust=False, objective="classificacao")
    assert "separar treino e holdout" in capsys.readouterr().out
    monkeypatch.setattr(pl, "holdout_agrupado_estratificado",
                        lambda y, g, frac, seed: (np.setdiff1d(np.arange(len(y)), np.flatnonzero(y == y[0])[:4]),
                                                  np.flatnonzero(y == y[0])[:4]))          # holdout so' com amostras da 1a classe
    _rodar(tmp_path / "c", level="N1", auto_adjust=False, objective="classificacao")
    assert "FORA do holdout" in capsys.readouterr().out


def test_janela_savgol_maior_que_o_numero_de_variaveis_e_recusada(tmp_path):
    with pytest.raises(ValueError, match="sg_window"):
        _rodar(tmp_path, level="N1", auto_adjust=False, objective="classificacao", sg_window=101, n_synthetic_points=60,
               default_preprocessing="msc_sg_mc")


def test_faixa_espectral_fora_do_eixo_dos_dados_e_recusada_e_exclusao_de_classe_so_age_quando_ha_amostra(tmp_path, capsys):
    cfg_csv = Config(mode="csv", csv_file=_csv(tmp_path, ["A"] * 6 + ["B"] * 6), class_column="classe", conc_column="", level="N1",
                     auto_adjust=False, objective="classificacao", wn_min=9000.0, wn_max=9500.0, output_root_folder=str(tmp_path / "fx"))
    with pytest.raises(ValueError, match="Nenhuma variavel sobrevive"):
        pl.executar(cfg_csv)
    capsys.readouterr()
    _rodar(tmp_path / "b", level="N1", auto_adjust=False, objective="classificacao", exclude_classes=["Esp_A"])
    assert "Excluindo classes" in capsys.readouterr().out
    _rodar(tmp_path / "c", level="N1", auto_adjust=False, objective="classificacao", exclude_classes=["nao_existe"])
    assert "Excluindo classes" not in capsys.readouterr().out


def test_plano_de_figuras_vazio_nao_e_listado(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pl, "describe_plan", lambda cfg: [])
    _rodar(tmp_path, level="N1", auto_adjust=False, objective="classificacao")
    assert "Figuras pertinentes a este objetivo" not in capsys.readouterr().out


@pytest.mark.parametrize("aviso_pcv", ["", "exploratorio"])
def test_ddsimca_com_pcv_sem_e_com_aviso_e_com_um_so_modelo_de_classe_no_resultado(tmp_path, monkeypatch, aviso_pcv):
    from guaraci.classificadores import DDSimca
    monkeypatch.setattr(pl, "should_generate", lambda cfg, nome: True)
    monkeypatch.setattr(pl, "ddsimca_pcv_sensitivity", lambda *a, **k: {"disponivel": True, "sensibilidade": 0.5, "aviso": aviso_pcv,
                                                                        "n_grupos": 3, "n_amostras": 9})
    real = DDSimca.score_matrix
    monkeypatch.setattr(DDSimca, "score_matrix", lambda self, *a, **k: dict(list(real(self, *a, **k).items())[:1]))
    _rodar(tmp_path, level="N2", auto_adjust=False, objective="classificacao", run_ddsimca=True, ddsimca_pcv=True, detailed_figures=True)
    resumo = canonico(next((tmp_path / "saida").rglob("resumo_modelo.txt")))
    assert ("AVISO PCV" in resumo) is bool(aviso_pcv)


def test_avaliacao_do_holdout_que_falha_vira_erro_no_log_e_some_do_resumo(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pl, "conformal_margin_classification", _quebra("conformal quebrou"))
    _rodar(tmp_path, level="N2", auto_adjust=False, objective="classificacao", run_ddsimca=True)
    assert "Avaliacao em holdout falhou: conformal quebrou" in capsys.readouterr().out
    resumo = canonico(next((tmp_path / "saida").rglob("resumo_modelo.txt")))
    assert "BCa Holdout" not in resumo


def test_lvs_otimas_no_teto_imposto_pelo_tamanho_dos_dados_geram_atencao(tmp_path, capsys, monkeypatch):
    real = pl.rmse_flat
    chamadas = {"n": 0}

    def decrescente(*a, **k):
        chamadas["n"] += 1
        return real(*a, **k) * 0.5 ** chamadas["n"]

    monkeypatch.setattr(pl, "rmse_flat", decrescente)
    _rodar(tmp_path, level="N2", auto_adjust=False, objective="classificacao", max_lvs=40)
    assert "no teto imposto pelo tamanho dos dados" in capsys.readouterr().out


def test_pasta_de_saida_que_nao_pode_ser_criada_da_mensagem_acionavel(tmp_path, monkeypatch):
    def nega(*a, **k):
        raise PermissionError("sem permissao")

    monkeypatch.setattr(pl.os, "makedirs", nega)
    with pytest.raises(ValueError, match="Nao foi possivel criar a pasta de saida"):
        pl.executar(Config(**_BASE, level="N2", auto_adjust=False, input_folder=str(tmp_path / "d"), output_root_folder=str(tmp_path / "s")))


def test_config_efetiva_e_auditoria_que_falham_sao_registradas_e_a_execucao_continua(tmp_path, capsys, monkeypatch):
    from guaraci import auditoria_delineamento as ad
    from guaraci import contexto

    monkeypatch.setattr(contexto, "gravar_efetiva", lambda *a, **k: (_ for _ in ()).throw(OSError("disco cheio")))
    monkeypatch.setattr(ad, "run_audit", _quebra("auditoria quebrou"))
    _rodar(tmp_path, level="N2", auto_adjust=False, objective="classificacao")
    saida = capsys.readouterr().out
    assert "config_efetiva (disco cheio)" in saida and "Auditoria de delineamento falhou: auditoria quebrou" in saida


# ---- menu de terminal legado e bloco `__main__` do modulo ---------------------------------------------------------------------------------


def _campo(tipo):
    return next(s for s in pl._CONFIG_SPEC if s["tipo"] == tipo)


@pytest.fixture
def mensagens(monkeypatch):
    """O menu fala pelo logger do pacote; o teste le o que ele disse trocando o logger por um coletor."""
    class Coletor:
        def __init__(self):
            self.linhas = []

        def info(self, m, *a, **k):
            self.linhas.append(str(m))

        warning = error = info

        def texto(self):
            return " | ".join(self.linhas)

    c = Coletor()
    monkeypatch.setattr(pl, "log", c)
    return c


def _entradas(monkeypatch, *respostas):
    fila = list(respostas)
    monkeypatch.setattr("builtins.input", lambda *_: fila.pop(0))


def test_editar_campo_bool_inverte_o_valor():
    cfg, s = Config(), _campo("bool")
    antes = bool(getattr(cfg, s["attr"]))
    pl._editar_campo(cfg, s)
    assert bool(getattr(cfg, s["attr"])) is (not antes)


def test_editar_campo_de_escolha_aceita_numero_valido_e_cancela_o_resto(monkeypatch):
    cfg, s = Config(), _campo("choice")
    _entradas(monkeypatch, "1", "99", "")
    pl._editar_campo(cfg, s)
    esperado = getattr(cfg, s["attr"])
    pl._editar_campo(cfg, s)
    pl._editar_campo(cfg, s)
    assert getattr(cfg, s["attr"]) == esperado


def test_editar_campo_de_escolha_com_valor_recusado_mostra_o_erro(monkeypatch, mensagens):
    cfg, s = Config(), _campo("choice")
    monkeypatch.setattr(pl, "_coagir_valor", lambda *a: (_ for _ in ()).throw(ValueError("opcao invalida")))
    _entradas(monkeypatch, "1")
    pl._editar_campo(cfg, s)
    assert "erro: opcao invalida" in mensagens.texto()


def test_editar_campo_de_texto_cancela_com_enter_aceita_valor_e_mostra_erro_de_valor_invalido(monkeypatch, mensagens):
    cfg = Config()
    s = next(x for x in pl._CONFIG_SPEC if x["tipo"] in ("int", "float"))
    _entradas(monkeypatch, "", "abc", "qualquer")
    pl._editar_campo(cfg, s)
    pl._editar_campo(cfg, s)
    monkeypatch.setattr(pl, "_coagir_valor", lambda spec, val: getattr(cfg, spec["attr"]))
    pl._editar_campo(cfg, s)
    saida = mensagens.texto()
    assert "cancelado." in saida and "erro:" in saida and "-> ok" in saida


def test_menu_interativo_percorre_salvar_carregar_editar_invalida_e_sair(tmp_path, monkeypatch, mensagens):
    caminho = str(tmp_path / "config.yaml")
    _entradas(monkeypatch, "s", "l", "1", "zzz", "q")
    monkeypatch.setattr(pl, "_editar_campo", lambda cfg, s: None)
    pl._menu_interativo(Config(), caminho)
    saida = mensagens.texto()
    assert "salvo em" in saida and "recarregado." in saida and "opcao invalida." in saida and "encerrado." in saida


def test_menu_interativo_carrega_arquivo_existente_e_mostra_erro_de_arquivo_ilegivel(tmp_path, monkeypatch, mensagens):
    caminho = tmp_path / "config.yaml"
    caminho.write_text("chave_que_nao_existe: 1\n", encoding="utf-8")
    _entradas(monkeypatch, "l", "q")
    pl._menu_interativo(Config(), str(caminho))
    saida = mensagens.texto()
    assert "nao foi possivel carregar" in saida and "erro:" in saida


def test_menu_interativo_rodar_recusa_dados_ausentes_e_erro_semantico_e_so_roda_quando_ok(tmp_path, monkeypatch, mensagens):
    caminho = str(tmp_path / "config.yaml")
    rodadas = []
    monkeypatch.setattr(pl, "executar", lambda cfg: rodadas.append(cfg))
    # o menu valida a pasta no topo de cada volta e de novo ao escolher rodar
    estados = iter([(True, "ok"), (False, "pasta ausente"), (True, "ok"), (True, "ok"), (True, "ok"), (True, "ok")])
    monkeypatch.setattr(pl, "_validar_pasta_dados", lambda cfg: next(estados))
    erros = iter([["faixa invalida"], []])
    monkeypatch.setattr(pl, "_validar_semantico", lambda cfg: next(erros))
    _entradas(monkeypatch, "r", "r", "r")
    pl._menu_interativo(Config(), caminho)
    saida = mensagens.texto()
    assert "Corrija antes de rodar." in saida and "faixa invalida" in saida and len(rodadas) == 1


def test_menu_interativo_sem_config_usa_a_padrao(tmp_path, monkeypatch):
    _entradas(monkeypatch, "q")
    pl._menu_interativo(None, str(tmp_path / "nao_existe.yaml"))


@pytest.mark.parametrize("argv", [["--rodar"], ["--codigo"]])
def test_bloco_main_despacha_rodar_e_codigo(monkeypatch, argv):
    import runpy
    import sys

    import guaraci.guaraci as g
    chamadas = []
    monkeypatch.setattr(g, "_comando_run", lambda *a, **k: chamadas.append(a))
    monkeypatch.setattr("guaraci.contexto.aplicar", lambda cfg: (_ for _ in ()).throw(RuntimeError("parou")))
    monkeypatch.setattr(sys, "argv", ["pipeline", *argv])
    if argv == ["--rodar"]:
        runpy.run_module("guaraci.pipeline", run_name="__main__")
        assert len(chamadas) == 1
    else:
        with pytest.raises(RuntimeError, match="parou"):    # `--codigo` entrega a Config embutida a `executar`
            runpy.run_module("guaraci.pipeline", run_name="__main__")


@pytest.mark.parametrize("tty, cli_ok, tem_config", [(True, True, False), (True, False, False), (False, True, False), (False, True, True)])
def test_bloco_main_sem_flags_usa_a_cli_unica_o_menu_antigo_ou_executa_direto(tmp_path, monkeypatch, tty, cli_ok, tem_config):
    import builtins
    import runpy
    import sys
    chamadas = []
    monkeypatch.chdir(tmp_path)
    if tem_config:
        pl.save_config(Config(), str(tmp_path / "config.yaml"))
    monkeypatch.setattr(sys, "argv", ["pipeline"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: tty, raising=False)
    import guaraci.guaraci as g
    monkeypatch.setattr(g, "main", lambda *a, **k: chamadas.append("cli"))
    if not cli_ok:
        real_import = builtins.__import__

        def sem_cli(nome, *a, **k):
            if nome == "guaraci.guaraci":
                raise ImportError("sem cli")
            return real_import(nome, *a, **k)

        monkeypatch.setattr(builtins, "__import__", sem_cli)
        monkeypatch.setattr("builtins.input", lambda *_: "q")
    monkeypatch.setattr("guaraci.contexto.aplicar", lambda cfg: (_ for _ in ()).throw(RuntimeError("parou")))
    if tty:
        runpy.run_module("guaraci.pipeline", run_name="__main__")
        assert chamadas == (["cli"] if cli_ok else [])
    else:
        with pytest.raises(RuntimeError, match="parou"):
            runpy.run_module("guaraci.pipeline", run_name="__main__")


def test_menu_interativo_carrega_o_arquivo_valido_ao_abrir(tmp_path, monkeypatch, mensagens):
    caminho = str(tmp_path / "config.yaml")
    pl.save_config(Config(), caminho)
    _entradas(monkeypatch, "q")
    pl._menu_interativo(Config(), caminho)
    assert "carregado de" in mensagens.texto()


def test_monte_carlo_sem_iteracoes_validas_nao_imprime_tabela_vazia(tmp_path, capsys, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(pl, "should_generate", lambda cfg, nome: True)
    monkeypatch.setattr(pl, "monte_carlo_cv", lambda *a, **k: pd.DataFrame())
    _rodar(tmp_path, level="N2", auto_adjust=False, objective="classificacao", run_monte_carlo=True)
    saida = capsys.readouterr().out
    assert "Monte Carlo CV" in saida and "Empty DataFrame" not in saida


# ── linhas condicionais do resumo do modelo (`resumo_modelo.txt`) ────────────
def _resumo(tmp_path, **kw):
    """Roda uma execucao pequena e devolve {chave: valor} do resumo; a chave perde o recuo e o valor, os espacos das pontas."""
    kw.setdefault("level", "N2")
    _rodar(tmp_path, auto_adjust=False, objective="classificacao", **kw)
    texto = canonico(next((tmp_path / "saida").rglob("resumo_modelo.txt")))
    saida = {}
    for linha in texto.splitlines():
        if " : " in linha or linha.rstrip().endswith(" :"):
            chave, _, valor = linha.partition(" :")
            saida[chave.strip()] = valor.strip()
    return saida


def test_resumo_traz_roc_por_classe_holdout_conformal_e_ddsimca_one_class(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "should_generate", lambda cfg, nome: True)
    r = _resumo(tmp_path, run_ddsimca=True, ddsimca_pcv=True)
    # ROC: macro e uma linha por classe (a media "macro" nao se repete como classe)
    assert "ROC AUC macro (OvR)" in r and "--- ROC AUC por classe ---" in r
    assert {k for k in r if k.startswith("AUC ")} == {"AUC Esp_A", "AUC Esp_B", "AUC Esp_C"}
    # holdout externo e conformal (grupos insuficientes: nao alcancavel, sem linha de cobertura)
    assert int(r["Holdout n"]) > 0 and "Holdout accuracy" in r and "BCa Holdout Accuracy" in r and "BCa Holdout Bal.acc" in r
    assert r["Conformal classif. alcancavel"] == "False" and "Conformal classif. cobertura" not in r
    assert int(r["Conformal classif. n_grupos"]) > 0
    # DD-SIMCA: componentes da configuracao; desconhecidos/ambiguos sao contagens pequenas (nao o complemento)
    assert int(r["DD-SIMCA n_components"]) == Config().ddsimca_n_components
    assert 0 <= int(r["DD-SIMCA n_desconhecidos"]) < 20 and 0 <= int(r["DD-SIMCA n_ambiguos"]) < 20
    assert r["DD-SIMCA mode treino"] == "puros (one-class)"
    for cls in ("Esp_A", "Esp_B", "Esp_C"):
        assert r[f"DD-SIMCA {cls} sens(LOGO)/esp"].startswith("n/a (nao validado) / ")        # 1 so' grupo de puros: LOGO nao valida
        assert "grupos_LOGO=1" in r[f"DD-SIMCA {cls} sens(LOGO)/esp"] and "Sensibilidade nao estimavel por LOGO" in r[f"DD-SIMCA {cls} AVISO"]
    # replicas de treino atipicas so' sao avisadas nas classes que as tem (esta semente: A e B; a C nao tem)
    assert all("replica(s) de treino atipica(s)" in r[f"DD-SIMCA {c} AVISO treino"] for c in ("Esp_A", "Esp_B"))
    assert "DD-SIMCA Esp_C AVISO treino" not in r


def test_resumo_traz_sensibilidade_pcv_exploratoria_ao_lado_do_logo(tmp_path, monkeypatch):
    pytest.importorskip("prcv")
    monkeypatch.setattr(pl, "should_generate", lambda cfg, nome: True)
    r = _resumo(tmp_path, run_ddsimca=True, ddsimca_pcv=True)
    for cls in ("Esp_A", "Esp_B", "Esp_C"):
        assert r[f"DD-SIMCA {cls} sens(PCV, exploratorio)"].endswith("%") and "PCV" in r[f"DD-SIMCA {cls} AVISO PCV"]
        assert f"DD-SIMCA {cls} sens(LOGO)/esp" in r                                             # PCV ao lado do LOGO, nunca no lugar


def test_resumo_ddsimca_treinado_em_todas_as_amostras_diz_que_nao_e_autenticacao_e_nao_tem_especificidade(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "should_generate", lambda cfg, nome: True)
    r = _resumo(tmp_path, level="N3", run_ddsimca=True, ddsimca_train_on="todos")
    assert r["DD-SIMCA mode treino"] == "todos (in-sample; sens/esp NAO sao autenticacao)"
    for cls in ("Esp_A", "Esp_B", "Esp_C"):
        sens, _, esp_e_resto = r[f"DD-SIMCA {cls} sens(LOGO)/esp"].partition(" / ")
        assert sens.endswith("%") and "n/a" not in sens                  # fracao in-sample da classe aceita
        assert esp_e_resto.startswith("n/a (")                            # sem especificidade fora do modo one-class


def test_resumo_formata_sens_esp_pcv_e_cobertura_conformal_quando_ha_valores_finitos_e_avisos(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "should_generate", lambda cfg, nome: True)
    real_logo, real_pcv, real_conf = pl.ddsimca_logo_sensitivity, pl.ddsimca_pcv_sensitivity, pl.conformal_margin_classification
    monkeypatch.setattr(pl, "ddsimca_logo_sensitivity", lambda *a, **k: {**real_logo(*a, **k), "sensibilidade": 0.9, "n_grupos": 4, "aviso": ""})
    monkeypatch.setattr(pl, "ddsimca_pcv_sensitivity",
                        lambda *a, **k: {**real_pcv(*a, **k), "disponivel": True, "sensibilidade": float("nan"), "aviso": ""})
    monkeypatch.setattr(pl, "conformal_margin_classification",
                        lambda *a, **k: {**real_conf(*a, **k), "alcancavel": True, "alpha_nominal": 0.1, "n_grupos": 25})
    r = _resumo(tmp_path / "a", run_ddsimca=True, ddsimca_pcv=True)
    linha = r["DD-SIMCA Esp_A sens(LOGO)/esp"]
    assert linha.startswith("90.0% / ") and "n/a" not in linha.split(" (")[0] and "grupos_LOGO=4" in linha
    assert "DD-SIMCA Esp_A AVISO" not in r                                   # aviso vazio: sem linha de aviso
    assert r["DD-SIMCA Esp_A sens(PCV, exploratorio)"] == "n/a" and "DD-SIMCA Esp_A AVISO PCV" not in r
    assert r["Conformal classif. alcancavel"] == "True" and r["Conformal classif. cobertura"] == "90%"
    assert r["Conformal classif. n_grupos"] == "25"
    # agora com sensibilidade nao estimavel + avisos e PCV finito
    monkeypatch.setattr(pl, "ddsimca_logo_sensitivity",
                        lambda *a, **k: {**real_logo(*a, **k), "sensibilidade": float("nan"), "n_grupos": 1, "aviso": "aviso de teste"})
    monkeypatch.setattr(pl, "ddsimca_pcv_sensitivity",
                        lambda *a, **k: {**real_pcv(*a, **k), "disponivel": True, "sensibilidade": 0.5, "aviso": "aviso pcv"})
    r2 = _resumo(tmp_path / "b", run_ddsimca=True, ddsimca_pcv=True)
    assert r2["DD-SIMCA Esp_A sens(LOGO)/esp"].startswith("n/a (nao validado) / ")
    assert r2["DD-SIMCA Esp_A AVISO"] == "aviso de teste"
    assert r2["DD-SIMCA Esp_A sens(PCV, exploratorio)"] == "50.0%" and r2["DD-SIMCA Esp_A AVISO PCV"] == "aviso pcv"


def test_resumo_sem_a_figura_do_ddsimca_nao_traz_nenhuma_linha_de_ddsimca(tmp_path, monkeypatch):
    """`run_ddsimca` ligado mas o DD-SIMCA nao executado (figura dispensada): nada de DD-SIMCA no resumo."""
    monkeypatch.setattr(pl, "should_generate", lambda cfg, nome: nome != "ddsimca")
    r = _resumo(tmp_path, run_ddsimca=True)
    assert not [k for k in r if k.startswith("DD-SIMCA")]
