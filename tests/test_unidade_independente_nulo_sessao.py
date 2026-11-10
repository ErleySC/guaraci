"""B4c do parecer de 2026-09-25 -- a unidade independente da validacao.

Versao automatizada de `revisor3/nulo_sessao.py`
(teste nulo caixa-preta pelo `executar()` real): o rotulo de especie e' sorteado
POR SESSAO, sem nenhum sinal de especie; cada sessao tem assinatura espectral
propria compartilhada pelo seu puro e suas 4 diluicoes (cada diluicao = um
`mae_id`, como em `dados_io.parse_title`). Se a CV protege so' o `mae_id`,
diluicoes da mesma sessao caem nos dois lados e o PLS-DA "aprende" a sessao:
bal.acc >> 0,25 (acaso). Agrupando pela SESSAO, deve ficar no acaso.
"""
from __future__ import annotations

import re

import pytest

import numpy as np

import guaraci.pipeline as pq
from guaraci import dados_io, io_registry
from guaraci.auditoria_delineamento import check_unidade_independente
from tests._resumo import canonico

ESPECIES = ["ESA", "CAS", "ESK", "PEQ"]
N_SESSOES, NIVEIS, N_REP, P = 6, [2.0, 5.0, 10.0, 20.0], 3, 120
WN = np.linspace(4000, 10000, P)


def _gerar(seed=1):
    rng = np.random.default_rng(seed)
    base = 1.0 + 0.3 * np.sin(WN / 900.0)
    banda = np.exp(-((WN - 7000) ** 2) / (2 * 300 ** 2))
    X, conc, mae, sess = [], [], [], []
    for esp in ESPECIES:
        for s in range(N_SESSOES):
            data = f"{10 + s:02d}-06-2099"
            sessao = f"{esp}-{data}"
            assin_sess = 0.05 * rng.normal(size=P)          # so' a sessao tem sinal
            maes = [(f"{esp}-{data}", 0.0)] + [(f"{esp}-{data}-A{c:.2f}", c) for c in NIVEIS]
            for m, c in maes:
                assin_mae = 0.005 * rng.normal(size=P)
                for _ in range(N_REP):
                    x = (1.0 + 0.02 * rng.normal()) * (
                        base + assin_sess + assin_mae + 0.02 * c * banda
                        + 0.01 * rng.normal(size=P))
                    X.append(x); conc.append(c); mae.append(m); sess.append(sessao)
    X = np.asarray(X)
    # H0 verdadeiro: rotulo sorteado POR SESSAO, balanceado (6 sessoes/rotulo)
    sess_u = sorted(set(sess))
    lab = np.repeat(ESPECIES, N_SESSOES)
    rng.shuffle(lab)
    mapa = dict(zip(sess_u, lab))
    rot = np.array([mapa[s] for s in sess], dtype=str)
    return X, rot, np.asarray(conc), np.asarray(mae, dtype=str)


def _rodar(tmp_path, unidade, nome):
    X, rot, conc, mae = _gerar()
    io_registry.register_reader(
        "nulo_sessao_teste", lambda cfg: (WN.copy(), X, rot, conc, mae, None))
    cfg = pq.Config(
        mode="nulo_sessao_teste", matrix_profile="generico", level="N1",
        objective="classificacao", wn_min=4000.0, wn_max=10000.0,
        default_preprocessing="custom", apply_snv=True, apply_sg=False,
        apply_mc=False, output_root_folder=str(tmp_path / nome), show_plots=False,
        group_by_mae_id=True, group_unit=unidade, max_lvs=6, n_splits_cv=5,
        n_permutations=10, n_bootstrap_bca=30, n_bootstrap_vip=3, seed=7,
        seed_holdout=11, frac_holdout=0.2, selecao_lv_cv_aninhada=False,
        run_benchmark=False, run_monte_carlo=False, run_shap=False,
        run_wold=False, run_cv_anova=False, run_opls=False, run_ddsimca=False,
        executar_etapa4=False)
    pq.executar(cfg)
    txt = canonico(list((tmp_path / nome).rglob("resumo_modelo.txt"))[0])
    return float(re.search(r"Balanced accuracy\s*\.*:\s*([\d.]+)", txt).group(1)), txt


@pytest.mark.slow
def test_agrupar_por_sessao_da_desempenho_de_acaso_e_por_mae_id_infla(tmp_path):
    bal_mae, _ = _rodar(tmp_path, "mae_id", "por_mae")
    bal_sess, txt = _rodar(tmp_path, "sessao", "por_sessao")
    assert bal_mae > 0.65, (
        f"o fenomeno que motiva a auditoria sumiu? por mae_id bal.acc={bal_mae:.3f}")
    assert bal_sess < 0.45, (
        f"agrupando por sessao o nulo deveria ficar perto do acaso (0,25); "
        f"bal.acc={bal_sess:.3f}")
    assert "Unidade de agrupamento" in txt and "sessao" in txt


def test_auditoria_detecta_mae_id_mais_fino_que_a_sessao():
    _X, rot, _c, mae = _gerar()
    achado = check_unidade_independente(rot, mae, "mae_id")
    assert achado.nome == "unidade_independente"
    assert achado.severidade == "critico"          # cada sessao tem UMA classe
    assert "SESSAO" in achado.mensagem
    assert check_unidade_independente(rot, mae, "sessao").severidade == "ok"


def test_auditoria_aviso_quando_sessoes_tem_varias_classes():
    rot = np.array(["A", "B"] * 6)
    mae = np.array([f"ESA-04-01-2099-A{k}.00" for k in range(12)])
    # 12 mae_id, 1 sessao (mesma data) com 2 classes: mais fino, nao aninhado
    achado = check_unidade_independente(rot, mae, "mae_id")
    assert achado.severidade == "aviso"


def test_chave_de_unidade_e_identidade_para_mae_id():
    m = np.array(["ESA-04-01-2099-A1.05", "ESA-04-01-2099-A2.11"])
    assert np.array_equal(dados_io.chave_unidade_validacao(m, "mae_id"), m)
    assert list(dados_io.chave_unidade_validacao(m, "sessao")) == ["ESA-04-01-2099"] * 2
    assert dados_io.chave_unidade_validacao(None, "sessao") is None
