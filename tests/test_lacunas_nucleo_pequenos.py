"""Ramos de borda de modulos pequenos do nucleo (hsi_multiway, auditoria_delineamento, transferir_cli, robustness, linearity, identificacao)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from guaraci import auditoria_delineamento as ad
from guaraci import hsi_multiway as hm
from guaraci import identificacao as ident
from guaraci import linearity as lin
from guaraci import robustness as rob
from guaraci import transferir_cli as tcli

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── hsi_multiway ────────────────────────────────────────────────────────────
def test_grade_de_bloco_recusa_cubo_nao_3d_mascara_de_outra_forma_e_mascara_vazia():
    cubo, m = np.ones((6, 6, 3)), np.ones((6, 6), dtype=bool)
    with pytest.raises(ValueError, match="3D"):
        hm._grade_bloco_medio(cubo[..., 0], m, 2, 2)
    with pytest.raises(ValueError, match="nao bate com o cubo"):
        hm._grade_bloco_medio(cubo, m[:3], 2, 2)
    with pytest.raises(ValueError, match="mascara vazia"):
        hm._grade_bloco_medio(cubo, np.zeros((6, 6), dtype=bool), 2, 2)


def test_tensor_de_amostras_recusa_comprimentos_diferentes_e_lista_vazia():
    with pytest.raises(ValueError, match="comprimentos diferentes"):
        hm.construir_tensor_amostras([np.ones((4, 4, 2))], [])
    with pytest.raises(ValueError, match="nenhuma gravacao"):
        hm.construir_tensor_amostras([], [])


def test_npls_recusa_y_de_outro_tamanho_alvo_nulo_e_laco_que_nao_converge():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3, 4))
    with pytest.raises(ValueError, match="Y tem"):
        hm.NPLS().fit(X, np.zeros((5, 1)))
    m = hm.NPLS(n_componentes=1).fit(X, np.zeros((8, 1)))                   # alvo todo zero: o escore inicial u tem norma ~0
    assert m.T_.shape == (8, 1)
    m2 = hm.NPLS(n_componentes=1, max_iter=1).fit(X, rng.normal(size=(8, 1)))
    assert m2.T_.shape == (8, 1)                                            # max_iter = 1: sai do laco sem convergir


def test_comparacao_npls_recusa_comprimentos_diferentes_e_vazamento_de_grupo(monkeypatch):
    cubos = [np.ones((6, 6, 3))] * 4
    mascaras = [np.ones((6, 6), dtype=bool)] * 4
    with pytest.raises(ValueError, match="comprimentos diferentes"):
        hm.comparar_npls_vs_pixelwise(cubos, mascaras, ["a", "b", "a"], ["g1", "g2", "g3", "g4"])
    from guaraci import validacao_estatistica as vs

    class _Vaza:
        def __init__(self, **_):
            pass

        def split(self, X, y, groups=None):
            return [(np.arange(len(X)), np.arange(len(X)))]
    monkeypatch.setattr(vs, "StableStratifiedGroupKFold", _Vaza)
    with pytest.raises(RuntimeError, match="vazamento de group_id"):
        hm.comparar_npls_vs_pixelwise(cubos, mascaras, ["a", "b", "a", "b"], ["g1", "g2", "g3", "g4"])


# ── auditoria_delineamento ──────────────────────────────────────────────────
def test_unidade_por_sessao_com_menos_de_tres_sessoes_e_aviso():
    mae = np.array(["ABC-03-01-2099-A1.00", "ABC-03-01-2099-A2.00", "ABC-05-01-2099-A1.00", "ABC-05-01-2099-A2.00"])
    f = ad.check_unidade_independente(np.array(["x"] * 4), mae, unidade="sessao")
    assert f.severidade == "aviso" and "degenerada" in f.mensagem


def test_faixa_de_validacao_sem_valor_valido_sem_faixa_declarada_dentro_e_fora(monkeypatch):
    from guaraci import perfil_matriz
    assert ad.check_validation_use_range(np.array([np.nan, np.nan]), SimpleNamespace()).severidade == "aviso"
    sem_faixa = SimpleNamespace(faixa_trabalho=None)
    monkeypatch.setattr(perfil_matriz, "cfg_profile", lambda cfg: sem_faixa)
    assert "nao declara faixa" in ad.check_validation_use_range(np.array([1.0, 2.0]), SimpleNamespace()).mensagem
    com_faixa = SimpleNamespace(faixa_trabalho=(0.0, 10.0), outside_working_range=lambda v: not (0.0 <= v <= 10.0))
    monkeypatch.setattr(perfil_matriz, "cfg_profile", lambda cfg: com_faixa)
    assert ad.check_validation_use_range(np.array([1.0, 9.0]), SimpleNamespace()).severidade == "ok"
    assert ad.check_validation_use_range(np.array([1.0, 19.0]), SimpleNamespace()).severidade == "aviso"


# ── transferir_cli ──────────────────────────────────────────────────────────
def _csv(tmp_path, nome, linhas):
    p = tmp_path / nome
    p.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return str(p)


def _csv_numerico(tmp_path, nome, n=5, k=4, seed=0):
    rng = np.random.default_rng(seed)
    cab = ",".join(f"v{j}" for j in range(k))
    return _csv(tmp_path, nome, [cab] + [",".join(f"{x:.4f}" for x in rng.normal(size=k)) for _ in range(n)])


@pytest.mark.parametrize("ajuste", ["so_cabecalho", "uma_coluna"])
def test_transferir_recusa_csv_sem_linhas_ou_com_menos_de_duas_colunas_numericas(tmp_path, capsys, ajuste):
    ruim = _csv(tmp_path, "r.csv", ["a,b,c"]) if ajuste == "so_cabecalho" else _csv(tmp_path, "r.csv", ["x,y", "1,a", "2,b"])
    ok = _csv_numerico(tmp_path, "ok.csv")
    assert tcli.executar(["--mestre", ruim, "--escravo", ok, "--novos", ok]) == 2
    assert capsys.readouterr().err


def test_transferir_recusa_janela_negativa_novos_com_colunas_a_mais_e_poucas_amostras(tmp_path, capsys):
    m, s = _csv_numerico(tmp_path, "m.csv"), _csv_numerico(tmp_path, "s.csv", seed=1)
    assert tcli.executar(["--mestre", m, "--escravo", s, "--novos", m, "--janela", "-1"]) == 2
    largo = _csv_numerico(tmp_path, "largo.csv", k=6)
    assert tcli.executar(["--mestre", m, "--escravo", s, "--novos", largo]) == 2
    curto_m, curto_s = _csv_numerico(tmp_path, "cm.csv", n=2), _csv_numerico(tmp_path, "cs.csv", n=2, seed=1)
    assert tcli.executar(["--mestre", curto_m, "--escravo", curto_s, "--novos", m]) == 2
    assert capsys.readouterr().err.count("rro") >= 3


# ── robustness ──────────────────────────────────────────────────────────────
def test_variantes_de_pre_processamento_com_janela_par_ordem_no_limite_e_janela_minima():
    from guaraci.config import Config
    v = rob.preprocessing_config_variants(Config(sg_window=8, sg_polyorder=7))          # 8-4 -> 5 (impar); 8+4 = 12 -> 13 (par -> +1)
    assert "sg_window_5" in v and "sg_window_13" in v
    assert "sg_polyorder_8" not in v                                                    # ordem >= janela nao entra
    v2 = rob.preprocessing_config_variants(Config(sg_window=5, sg_polyorder=1))
    assert "sg_window_5" not in v2                                                      # 5-4 -> 5 == atual: sem variante duplicada


def test_rmsep_e_acuracia_com_amostras_demais_para_o_split_devolvem_nan():
    X = np.random.default_rng(0).normal(size=(2, 4))
    mae = np.array(["a", "b"])
    assert np.isnan(rob.avaliar_rmsep_plsr(X, np.array([1.0, 2.0]), mae))
    assert np.isnan(rob.avaliar_bal_acc_plsda(X, np.array(["x", "y"]), mae))


def test_protocolo_pula_variante_de_pre_processamento_que_nao_se_aplica(monkeypatch):
    from guaraci import preprocessamento
    from guaraci.config import Config
    real = preprocessamento.build_preprocessor
    chamadas = {"n": 0}

    def a_segunda_variante_falha(cfg):
        chamadas["n"] += 1
        if chamadas["n"] == 3:
            raise ValueError("preset impossivel")
        return real(cfg)
    monkeypatch.setattr(preprocessamento, "build_preprocessor", a_segunda_variante_falha)
    rng = np.random.default_rng(1)
    X = np.abs(rng.normal(size=(24, 30))) + 1
    y = np.array(["a", "b"] * 12)
    mae = np.repeat(np.arange(12), 2).astype(str)
    r = rob.protocolo_padrao(X, y, mae, Config(seed=0), "classificacao", n_components=2, n_replicas=1)
    assert r                                                                            # as demais variantes rodaram


# ── linearity ───────────────────────────────────────────────────────────────
def test_faixa_de_trabalho_com_limite_ausente_mantem_todas_as_amostras():
    assert lin._filtrar_faixa_trabalho(np.arange(5.0), (None, 3.0)).all()
    assert lin._filtrar_faixa_trabalho(np.arange(5.0), (1.0, 3.0)).tolist() == [False, True, True, True, False]


def test_falta_de_ajuste_recusa_nivel_invalido_e_graus_de_liberdade_insuficientes():
    y = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    with pytest.raises(ValueError, match="nivel_por"):
        lin.lack_of_fit_test(y, y, np.array(list("aabbcc")), nivel_por="outro")
    r = lin.lack_of_fit_test(y, y, np.array(list("abcdef")))                # cada amostra e' um nivel: nenhuma replica
    assert r.computavel is False


def test_falta_de_ajuste_com_replicas_identicas_tem_erro_puro_zero(monkeypatch):
    ref = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
    grupos = np.array(list("aabbccdd"))
    curva = lin.lack_of_fit_test(ref, ref ** 2, grupos)                     # curvatura sem erro puro: F -> inf, p = 0
    assert curva.p_value == 0.0 and np.isinf(curva.F)
    monkeypatch.setattr(lin.np, "polyfit", lambda x, y, d: (1.0, 0.0))        # reta exata: sem falta de ajuste e sem erro puro
    reta = lin.lack_of_fit_test(ref, ref, grupos)
    assert reta.p_value == 1.0 and reta.F == 0.0


def test_falta_de_ajuste_ignora_o_nivel_de_uma_so_amostra_no_erro_puro():
    y = np.array([1.0, 1.1, 2.0, 2.2, 3.0, 4.0])
    r = lin.lack_of_fit_test(y, y, np.array(list("aabbcd")))
    assert r.computavel and r.n_niveis_com_replica == 2 and r.df_pe == 2


# ── identificacao ───────────────────────────────────────────────────────────
def _ensemble_pequeno():
    from sklearn.decomposition import PCA

    from test_identificacao import _monta_dataset, _pca_e_var_t
    rng = np.random.default_rng(5)
    combos = {("Classe A", "S"): 25, ("Classe D", "M"): 25}
    X, rotulos, conc, mae, centros = _monta_dataset(rng, ["Classe A", "Classe D"], combos, p=4, sep=10.0, noise=0.1)
    pca, var_t = _pca_e_var_t(X, n_components=2)
    assert isinstance(pca, PCA)
    return ident.train_identification_ensemble(pca, var_t, X, rotulos, conc, mae), pca, var_t, centros, X


def test_cobertura_loso_sem_sessoes_suficientes_devolve_nan():
    T = np.random.default_rng(0).normal(size=(6, 2))
    c, se, n = ident.cobertura_empirica_loso(T, np.array(["g"] * 6), np.ones(2))          # um so' grupo: nenhuma sessao avaliavel
    assert np.isnan(c) and np.isnan(se) and n == 0


def test_identificacao_com_zero_candidatos_nao_devolve_palpite():
    ensemble, pca, var_t, centros, X = _ensemble_pequeno()
    longe = X[0] + 1e3
    r = ident.identify_sample(ensemble, pca, var_t, longe, n_candidatos=0)
    assert r.classe_identificada is None and r.candidatos_ambiguos == [] and r.cobertura_status is None


def test_cobertura_loso_com_poucos_grupos_para_o_limiar_alcancavel_nao_avalia_nenhuma_sessao():
    T = np.random.default_rng(1).normal(size=(9, 2))
    grupos = np.repeat(["g1", "g2", "g3"], 3)                                # tirando um grupo sobram 2: o limiar a 95% nao e' alcancavel
    c, se, n = ident.cobertura_empirica_loso(T, grupos, np.ones(2))
    assert n == 0 and np.isnan(c)
