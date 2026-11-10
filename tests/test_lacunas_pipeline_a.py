"""Ramos de borda de guaraci.pipeline, parte A (ambiente, validacao da entrada, comparacao de pre-processamentos, bootstrap do VIP, limpeza, R2cv)."""
from __future__ import annotations

import matplotlib
import numpy as np
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

from guaraci import pipeline as pl  # noqa: E402
from guaraci.config import Config  # noqa: E402


def _classes(n=24, p=10, seed=0, k=2):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % k
    X = rng.normal(size=(n, p)) + y[:, None] * 2.0
    return X, np.eye(k)[y], y


# ── ambiente de execucao ────────────────────────────────────────────────────
def test_ambiente_sem_backend_definido_e_com_backend_anterior_indisponivel(monkeypatch):
    def sem_backend():
        raise RuntimeError("sem backend")
    monkeypatch.setattr(pl.mpl, "get_backend", sem_backend)
    with pl._ambiente_da_execucao():
        pass
    usos = []

    def usar(nome, force=False):
        usos.append(nome)
        if nome == "TkAgg":
            raise ImportError("sem display")
    monkeypatch.setattr(pl.mpl, "get_backend", lambda: "TkAgg")
    monkeypatch.setattr(pl.mpl, "use", usar)
    with pl._ambiente_da_execucao():
        pass
    assert usos == ["Agg", "TkAgg"]                                   # tentou voltar ao anterior; o erro foi engolido


def test_identificador_do_dataset_usa_a_etiqueta_do_usuario_quando_ha():
    assert pl._dataset_id(Config(tag="Meu Dataset 2026")) not in ("", "dataset")
    assert pl._dataset_id(Config(tag="???")) == "dataset"                      # etiqueta que vira slug vazio cai no nome padrao


# ── validate_input ──────────────────────────────────────────────────────────
def test_validacao_remove_linha_com_nan_junto_com_teor_e_grupo_e_nao_avisa_duplicata_aproximada_que_ja_e_exata():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(8, 12))
    X = np.vstack([base, base[:2]])                                              # as duas ultimas linhas repetem as duas primeiras (exatas)
    X[3, 4] = np.nan
    rot = np.array(list("aabbccddab"))
    conc = np.arange(10.0)
    mae = np.array([f"g{i}" for i in range(10)])
    _, _, rot2, conc2, mae2, rel = pl.validate_input(X, np.arange(12.0), rot, conc, mae)
    assert len(rot2) == len(conc2) == len(mae2) == 9 and rel["n_amostras_removidas"] == 1
    assert rel["n_duplicatas_exatas"] == 2 and not any("correlation" in w for w in rel.get("avisos", []))


def test_validacao_sem_teor_e_sem_grupo_apenas_filtra_as_amostras():
    X = np.random.default_rng(1).normal(size=(6, 8))
    X[0, 0] = np.inf
    _, _, rot2, conc2, mae2, rel = pl.validate_input(X, np.arange(8.0), np.array(list("aabbcc")), None, None)
    assert len(rot2) == 5 and conc2 is None and mae2 is None


def test_desbalanceamento_forte_e_avisado_no_log(caplog):
    with caplog.at_level("INFO", logger=pl.log.name):
        rel = pl._check_balance(np.array(["a"] * 20 + ["b"] * 2))
    assert rel["desbalanceado"] and "Severe class imbalance" in caplog.text and "balanced_accuracy" in caplog.text


# ── compare_pipelines ───────────────────────────────────────────────────────
def test_comparacao_de_pre_processamentos_ignora_preset_que_nao_converge_e_deixa_q2_indefinido_sem_variancia(monkeypatch):
    X, Y, y = _classes()
    cv = [(np.arange(0, 24, 2), np.arange(1, 24, 2)), (np.arange(1, 24, 2), np.arange(0, 24, 2))]
    real = pl._cv_predict_manual
    chamadas = {"n": 0}

    def primeiro_preset_falha(fabrica, X_, Y_, cv_):
        chamadas["n"] += 1
        if chamadas["n"] <= 2:                                                # max_lv = 2: o 1o preset falha em todos os LVs
            raise ValueError("matriz singular")
        return real(fabrica, X_, Y_, cv_)
    monkeypatch.setattr(pl, "_cv_predict_manual", primeiro_preset_falha)
    r = pl.compare_pipelines(Config(), X, Y, y, cv, max_lv=2)
    assert r["MC only"]["accuracy"] == 0.0 and len(r) == 8                    # o preset que falhou fica na tabela zerado (e foi avisado)
    monkeypatch.setattr(pl, "_cv_predict_manual", real)
    constante = np.zeros_like(Y)
    constante[:, 0] = 1.0
    r2 = pl.compare_pipelines(Config(), X, constante, np.zeros(24, dtype=int), cv, max_lv=1)
    assert all(v["q2"] == -np.inf for v in r2.values())                     # Q2 indefinido nunca vira o "melhor"


# ── bootstrap do VIP ────────────────────────────────────────────────────────
def test_bootstrap_vip_estratificado_sem_grupo_e_com_reamostras_degeneradas(monkeypatch):
    X, Y, y = _classes()
    r = pl.bootstrap_vip_stratified(X, Y, y, n_opt=2, n_boot=5, seed=0, mae_id=None)
    assert r["mean"].shape == (X.shape[1],)
    monkeypatch.setattr(pl, "vip_scores", lambda pls: np.full(X.shape[1], np.nan))                 # VIP nao finito: contado como falha
    r_nan = pl.bootstrap_vip_stratified(X, Y, y, n_opt=2, n_boot=4, seed=0)
    assert not r_nan["mean"].any()

    def quebra(pls):
        raise ValueError("reamostra degenerada")
    monkeypatch.setattr(pl, "vip_scores", quebra)
    r_err = pl.bootstrap_vip_stratified(X, Y, y, n_opt=2, n_boot=4, seed=0)
    assert not r_err["mean"].any()


def test_bootstrap_vip_antigo_ainda_funciona_e_zera_quando_tudo_falha(monkeypatch):
    X, Y, y = _classes()
    mean, std = pl.bootstrap_vip(X, Y, 2, 4, 0)
    assert mean.shape == std.shape == (X.shape[1],)

    def quebra(pls):
        raise ValueError("degenerada")
    monkeypatch.setattr(pl, "vip_scores", quebra)
    mean0, std0 = pl.bootstrap_vip(X, Y, 2, 4, 0)
    assert not mean0.any() and not std0.any()


# ── limpeza de resultados antigos ───────────────────────────────────────────
def test_limpar_resultados_pasta_inexistente_nada_a_remover_e_remocao_com_e_sem_erro(tmp_path, monkeypatch):
    assert "nao encontrada" in pl.clear_old_results(str(tmp_path / "nao_existe"))["erro"]
    base = tmp_path / "res"
    base.mkdir()
    for i in range(5):
        pasta = base / f"run{i}"
        pasta.mkdir()
        (pasta / "a.txt").write_text("x" * 100, encoding="utf-8")
        import os
        os.utime(pasta, (1000 + i, 1000 + i))                                  # run4 = a mais recente
    assert pl.clear_old_results(str(base), manter_n=10)["removidas"] == []
    r = pl.clear_old_results(str(base), manter_n=3)
    assert len(r["removidas"]) == 2 and r["erro"] is None and {p.name for p in base.iterdir()} == {"run2", "run3", "run4"}

    def sem_permissao(*a, **k):
        raise OSError("em uso")
    monkeypatch.setattr("shutil.rmtree", sem_permissao)
    r2 = pl.clear_old_results(str(base), manter_n=1)
    assert r2["removidas"] == [] and "em uso" in r2["erro"]


# ── replicas e R2cv ─────────────────────────────────────────────────────────
def test_replicas_agrupadas_ignoram_grupo_de_uma_so_amostra():
    X = np.random.default_rng(0).normal(size=(6, 4))
    prep = StandardScaler().fit(X)
    assert pl._agrupar_replicas_processadas(X, np.array(list("abcdef")), prep) == []
    grupos = pl._agrupar_replicas_processadas(X, np.array(list("aabbcd")), prep)
    assert [g.shape[0] for g in grupos] == [2, 2]


def test_r2cv_sem_grupo_e_com_validacao_cruzada_que_falha_devolve_none(monkeypatch):
    rng = np.random.default_rng(0)
    n = 24
    rot = np.array(["E1"] * n)
    mae = np.array([f"E1-{i // 3:03d}-S{c:.2f}" for i, c in zip(range(n), np.tile([0, 0, 1, 2, 3, 4, 5, 6], 3))])
    conc = np.tile([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3)
    X = rng.normal(size=(n, 10)) + conc[:, None] * 0.2
    assert pl.r2cv_species_by_adulterant(X, conc, rot, None, Config()) is None

    def falha(*a, **k):
        raise ValueError("grupos insuficientes")
    monkeypatch.setattr(pl, "cross_val_predict", falha)
    assert pl.r2cv_species_by_adulterant(X, conc, rot, mae, Config(), min_niveis=2, min_grupos=2) is None      # todas as combinacoes viram n/a
