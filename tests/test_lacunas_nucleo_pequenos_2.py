"""Ramos de borda de modulos pequenos do nucleo, parte 2 (deriva_qc, conformal, asca, aumento_dados, hsi_extras, anova_cli)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from guaraci import anova_cli as ac  # noqa: E402
from guaraci import asca  # noqa: E402
from guaraci import aumento_dados as ad  # noqa: E402
from guaraci import conformal as cf  # noqa: E402
from guaraci import deriva_qc as dq  # noqa: E402
from guaraci import hsi_extras as hx  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── deriva_qc ───────────────────────────────────────────────────────────────
def test_deriva_qc_recusa_formas_e_ordens_inconsistentes():
    X, Q = np.ones((5, 4)), np.ones((6, 4))
    o5, o6 = np.arange(5.0), np.arange(6.0)
    with pytest.raises(ValueError, match="2D"):
        dq.corrigir_deriva_por_qc(X[:, 0], o5, Q, o6)
    with pytest.raises(ValueError, match="mesmo eixo"):
        dq.corrigir_deriva_por_qc(X, o5, np.ones((6, 3)), o6)
    with pytest.raises(ValueError, match="ordem_amostras"):
        dq.corrigir_deriva_por_qc(X, o6, Q, o6)
    with pytest.raises(ValueError, match="ordem_qc"):
        dq.corrigir_deriva_por_qc(X, o5, Q, o5)


def test_deriva_qc_com_spline_degenerada_cai_para_a_media_do_canal(monkeypatch):
    class _Quebra:
        def __init__(self, *a, **k):
            raise ValueError("spline degenerada")
    monkeypatch.setattr(dq, "UnivariateSpline", _Quebra)
    rng = np.random.default_rng(0)
    X, Q = rng.normal(size=(6, 3)), rng.normal(size=(8, 3))
    out = dq.corrigir_deriva_por_qc(X, np.arange(6.0), Q, np.arange(8.0))
    assert np.isfinite(out).all() and out.shape == X.shape


# ── conformal ───────────────────────────────────────────────────────────────
def test_limiar_conformal_sem_escore_finito_devolve_aviso():
    assert "Nenhum escore" in cf.conformal_threshold(np.array([np.nan, np.nan]))["aviso"]
    assert "Nenhum escore" in cf.conformal_threshold_agrupado(np.array([np.nan]), np.array(["g"]))["aviso"]


def test_conformal_one_class_recusa_unidade_desconhecida():
    with pytest.raises(ValueError, match="unidade"):
        cf.ConformalOneClass(unidade="outra")


# ── asca ────────────────────────────────────────────────────────────────────
def test_asca_recusa_x_nao_2d_e_unidade_de_permutacao_de_outro_tamanho():
    with pytest.raises(ValueError, match="2D"):
        asca.asca_decompose(np.ones(5), {"f": np.arange(5)})
    X = np.random.default_rng(0).normal(size=(8, 4))
    with pytest.raises(ValueError, match="unidade_permutacao"):
        asca.asca_permutation_test(X, {"f": np.array([0, 1] * 4)}, unidade_permutacao=np.arange(5), n_perm=5)


def test_asca_com_svd_que_nao_converge_zera_escores_e_variancia(monkeypatch):
    def falha(*a, **k):
        raise np.linalg.LinAlgError("nao convergiu")
    monkeypatch.setattr(np.linalg, "svd", falha)
    X = np.random.default_rng(1).normal(size=(8, 4))
    ef = asca.asca_decompose(X, {"f": np.array([0, 1] * 4)}, n_componentes=2)["efeitos"]["f"]
    assert not ef["scores"].any() and not ef["var_explicada"].any()


# ── aumento_dados ───────────────────────────────────────────────────────────
def test_aumento_recusa_x_nao_2d_e_cai_para_ruido_branco_com_poucos_canais():
    a = ad.AumentoVRM(n_aumentos=2, janela_suavizacao_ruido=10, amplitude_ruido_estruturado=0.5)
    with pytest.raises(ValueError, match="2D"):
        a.gerar(np.ones((4,)), np.arange(4), np.arange(4))
    X_novo, y_novo, g_novo = a.gerar(np.random.default_rng(0).normal(size=(5, 2)), np.arange(5), np.arange(5))           # p = 2 < janela: ruido branco
    assert X_novo.shape == (10, 2) and len(y_novo) == len(g_novo) == 10
    X2, *_ = a.gerar(np.random.default_rng(1).normal(size=(5, 60)), np.arange(5), np.arange(5))                           # janela par -> impar
    assert X2.shape == (10, 60)


# ── hsi_extras ──────────────────────────────────────────────────────────────
def _cubos_em_pasta(pasta, nivel, n=3, saturado=False):
    from test_hsi_pipeline import _gravar_cubo_sintetico, _mascara_objeto_suave
    rng = np.random.default_rng(3)
    alpha = _mascara_objeto_suave(64, 64)[..., None]
    for i in range(n):
        cubo = np.full((64, 64, 6), 5.0) if saturado else alpha * rng.normal(nivel, 0.01, (64, 64, 6)) + (1 - alpha) * rng.normal(0.05, 0.01, (64, 64, 6))
        _gravar_cubo_sintetico(pasta / f"c{i}", cubo, 6)


def test_preparar_dados_hsi_de_cubos_soltos_sem_grupo_e_com_todos_rejeitados(tmp_path):
    soltos = tmp_path / "soltos"
    soltos.mkdir()
    _cubos_em_pasta(soltos, 0.8)
    dados = hx.preparar_dados_hsi(str(soltos))
    assert len(set(dados.pixel_groups.tolist())) == 3 and all(g.startswith("__sem_grupo_") for g in dados.pixel_groups.tolist())
    ruins = tmp_path / "ruins"
    ruins.mkdir()
    _cubos_em_pasta(ruins, 0.8, saturado=True)
    with pytest.raises(ValueError, match="falharam no quality gate"):
        hx.preparar_dados_hsi(str(ruins))
    res = hx.rodar_hsi_dominio_aplicabilidade(dados)
    assert res["n_objetos_teste"] >= 1


# ── anova_cli (resumo e relatorio sem diagnostico nem comparacoes) ──────────
def _executar_e_capturar(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    linhas = ["y,g,u"]
    for g, mu in (("A", 0.0), ("B", 3.0)):
        for u in range(5):
            for r in range(2):
                linhas.append(f"{mu + rng.normal(0, .3):.3f},{g},{g}{u}")
    csv = tmp_path / "t.csv"
    csv.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    capt = {}
    real = ac.texto_resumo

    def espia(out, args, dados):
        capt.update(out=out, args=args, dados=dados)
        return real(out, args, dados)
    monkeypatch.setattr(ac, "texto_resumo", espia)
    assert ac.executar([str(csv), "--resposta", "y", "--grupo", "g", "--unidade", "u", "--saida", str(tmp_path / "res"), "--sem-bootstrap"]) == 0
    return capt


def test_resumo_e_relatorio_sem_diagnostico_de_premissas_e_sem_comparacoes(tmp_path, monkeypatch):
    c = _executar_e_capturar(tmp_path, monkeypatch)
    com_diag = ac.texto_resumo(c["out"], c["args"], c["dados"])
    sem_diag = ac.texto_resumo(dict(c["out"], diag={}), c["args"], c["dados"])
    assert "premissas: razao de variancias" in com_diag and "premissas: razao de variancias" not in sem_diag
    out = dict(c["out"], diag={})
    for r in out["resultados"].values():
        r.comparacoes = []
    blocos = ac._blocos_relatorio(out, c["args"], c["dados"], {})
    assert not any(b[0] == "h3" for b in blocos) and any(b[0] == "table" for b in blocos)
