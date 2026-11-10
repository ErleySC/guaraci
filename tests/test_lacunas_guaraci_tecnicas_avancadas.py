"""Telas de tecnicas avancadas (ASCA, EPO/GLSW, MCR-ALS, fusao multibloco, PARAFAC, COW, portao, pooled x local) de `guaraci.guaraci`.

O motor de cada tecnica e' trocado por um duble que devolve um relatorio minimo (ou levanta o erro de validacao): aqui se confere o
roteiro da tela, o que ela pergunta, recusa e mostra.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

import guaraci.guaraci as g
import guaraci.mcr_als as mcr
import guaraci.tecnicas_avancadas as ta


def _dataset(monkeypatch, *, x=None, conc=True, metadados=None):
    X = np.abs(np.random.default_rng(1).normal(size=(6, 4))) if x is None else x
    pacote = (np.arange(4.0), X, np.array(list("aabbcc")), np.arange(6.0) if conc else None, np.array(list("uuvvww")),
              metadados if metadados is not None else pd.DataFrame({"f": list("xyxyxy")}))
    monkeypatch.setattr(g, "_tecavan_carregar_dataset", lambda cfg, is_pt, **_kw: pacote)
    return pacote


def _quebra(msg="recusado", exc=ValueError):
    def f(*a, **k):
        raise exc(msg)
    return f


def test_carga_do_dataset_devolve_none_com_erro_e_o_pacote_quando_ok(cli, monkeypatch):
    monkeypatch.setattr(g.pq, "load_data", _quebra("pasta vazia"))
    cli.dizer("")
    assert g._tecavan_carregar_dataset(g.Config(), True) is None and "pasta vazia" in cli.saida()
    monkeypatch.setattr(g.pq, "load_data", lambda cfg: (1, 2, 3, 4, 5, 6))
    monkeypatch.setattr(g.pq, "validate_input", lambda *a: (2, 1, 3, 4, 5, None))
    assert g._tecavan_carregar_dataset(g.Config(), False)[0] == 1


def test_escolha_de_fatores_limita_a_quantidade_e_recusa_lixo(cli, monkeypatch):
    disponiveis = {"f1": pd.Series(list("ab")), "f2": pd.Series(list("cd"))}
    monkeypatch.setattr(ta, "fatores_categoricos_disponiveis", lambda r, m: disponiveis)
    cli.dizer("1,2", "x", "9", "2,1")
    assert list(g._tecavan_escolher_fatores(None, None, True)) == ["f1", "f2"]
    assert g._tecavan_escolher_fatores(None, None, True) is None
    assert g._tecavan_escolher_fatores(None, None, False) is None
    assert list(g._tecavan_escolher_fatores(None, None, True, max_fatores=1)) == ["f2"]


@pytest.mark.parametrize("permutacao", [None, {"f": {"p_value": 0.01}}])
def test_asca_mostra_cada_efeito_com_ou_sem_permutacao(cli, monkeypatch, permutacao):
    _dataset(monkeypatch)
    monkeypatch.setattr(g, "_tecavan_escolher_fatores", lambda *a, **k: {"f": pd.Series(list("xyxyxy"))})
    rel = types.SimpleNamespace(group_aware=False, permutacao=permutacao,
                                decomposicao={"efeitos": {"f": {"fracao_ss_total": 0.4}}, "ss_desbalanco": 0.5})
    monkeypatch.setattr(ta, "rodar_asca", lambda *a, **k: rel)
    cli.dizer("")
    g._menu_tec_asca(g.Config(), True)
    assert ("SIGNIFICATIVO" in cli.saida()) is bool(permutacao)


def test_asca_sem_fator_valido_ou_com_erro_do_motor_so_avisa(cli, monkeypatch):
    _dataset(monkeypatch)
    monkeypatch.setattr(g, "_tecavan_escolher_fatores", lambda *a, **k: None)
    cli.dizer("")
    g._menu_tec_asca(g.Config(), True)
    monkeypatch.setattr(g, "_tecavan_escolher_fatores", lambda *a, **k: {"f": pd.Series(list("xy"))})
    monkeypatch.setattr(ta, "rodar_asca", _quebra("fator desbalanceado"))
    cli.dizer("")
    g._menu_tec_asca(g.Config(), True)
    monkeypatch.setattr(g, "_tecavan_carregar_dataset", lambda cfg, is_pt, **_kw: None)
    g._menu_tec_asca(g.Config(), True)
    assert "fator desbalanceado" in cli.saida()


def test_epo_glsw_cobre_cada_recusa_e_o_resultado(cli, monkeypatch, tmp_path):
    _dataset(monkeypatch)
    monkeypatch.setattr(g, "_tecavan_pasta_saida", lambda cfg, nome: (tmp_path / nome).mkdir(exist_ok=True) or str(tmp_path / nome))
    f1, f2 = {"a": pd.Series(list("xy"))}, {"b": pd.Series(list("xy"))}
    respostas = iter([None])
    monkeypatch.setattr(g, "_tecavan_escolher_fatores", lambda *a, **k: next(respostas))
    cli.dizer("")
    g._menu_tec_epo_glsw(g.Config(), True)                              # grupo_interesse sem escolha
    respostas = iter([f1, None])
    cli.dizer("")
    g._menu_tec_epo_glsw(g.Config(), True)                              # fator_incomodo sem escolha
    respostas = iter([f1, f1])
    cli.dizer("")
    g._menu_tec_epo_glsw(g.Config(), False)                             # mesmo fator
    respostas = iter([f1, f2])
    monkeypatch.setattr(ta, "rodar_epo_glsw", _quebra("poucos pares"))
    cli.dizer("", "")
    g._menu_tec_epo_glsw(g.Config(), True)
    rel = types.SimpleNamespace(metodo="GLSW", n_pares_diferenca=4, variancia_removida_fracao=0.3,
                                X_corrigido=np.zeros((6, 4)))
    chamadas = []
    monkeypatch.setattr(ta, "rodar_epo_glsw", lambda *a, **k: chamadas.append(k["metodo"]) or rel)
    for escolha in ("2", ""):
        respostas = iter([f1, f2])
        cli.dizer(escolha, "")
        g._menu_tec_epo_glsw(g.Config(), True)
    assert chamadas == ["GLSW", "EPO"] and "poucos pares" in cli.saida()
    assert (tmp_path / "glsw" / "X_corrigido_glsw.csv").exists(), "o X corrigido tem de ir para disco"


def test_mcr_als_com_e_sem_restricao_de_correlacao(cli, monkeypatch, tmp_path):
    monkeypatch.setattr(g, "_tecavan_pasta_saida", lambda cfg, nome: str(tmp_path))
    _dataset(monkeypatch, x=np.array([[1.0, -2, 3, 4]] * 6))
    simples = types.SimpleNamespace(n_iter=7, convergiu=False, lof_percent=1.5, C=np.ones((6, 2)), S=np.ones((2, 4)))

    class _Supervisionado:
        n_iter, convergiu, lof_percent, correlacao_calibracao = 5, True, 0.5, 0.97
        C, S = np.ones((6, 2)), np.ones((2, 4))

    monkeypatch.setattr(mcr, "MCRALSResultadoSupervisionado", _Supervisionado)
    monkeypatch.setattr(ta, "rodar_mcr_als", lambda *a, **k: _Supervisionado() if k.get("conc") is not None else simples)
    cli.dizer("abc", "")
    g._menu_tec_mcr_als(g.Config(), True)                               # componentes invalidos
    cli.dizer("", "", "")
    g._menu_tec_mcr_als(g.Config(), True)                               # padrao, sem restricao
    cli.dizer("2", "s", "x", "")
    g._menu_tec_mcr_als(g.Config(), False)                              # indice invalido
    cli.dizer("2", "s", "1", "")
    g._menu_tec_mcr_als(g.Config(), True)                               # com restricao
    monkeypatch.setattr(ta, "rodar_mcr_als", _quebra("nao convergiu"))
    cli.dizer("", "n", "")
    g._menu_tec_mcr_als(g.Config(), True)
    monkeypatch.setattr(g, "_tecavan_carregar_dataset", lambda cfg, is_pt, **_kw: None)
    g._menu_tec_mcr_als(g.Config(), True)
    saida = cli.saida()
    for trecho in ("Numero invalido", "0.970", "NAO convergiu", "nao convergiu"):
        assert trecho in saida, trecho
    assert (tmp_path / "mcr_concentracoes_C.csv").exists() and (tmp_path / "mcr_espectros_puros_S.csv").exists()


def test_parafac_exige_modo_eem_e_grava_o_fator_da_amostra(cli, tmp_path, monkeypatch):
    cfg = g.Config(output_root_folder=str(tmp_path / "saida"))
    cli.dizer("")
    g._menu_tec_parafac(cfg, True)
    assert "modo_entrada: eem" in cli.saida()
    cfg.mode = "eem"
    meta = pd.DataFrame({"f": list("xyxyxy")})
    meta.attrs.update(grade_excitacao=[1], grade_emissao=[2])
    _dataset(monkeypatch, metadados=meta)
    monkeypatch.setattr(ta, "rodar_parafac_eem",
                        lambda *a, **kw: types.SimpleNamespace(
                            fator_amostra=np.ones((6, 2)), erro_reconstrucao_relativo=0.1, consistencia_nucleo=90.0,
                            restricao="nao-negativo", algoritmo="als", n_faltantes=0, n_iter_em=0, nota="nota", nota_en="note"))
    cli.dizer("2", "")
    g._menu_tec_parafac(cfg, True)
    assert (tmp_path / "saida" / "TecnicasEspeciais" / "parafac" / "parafac_fator_amostra.csv").exists()
    monkeypatch.setattr(ta, "rodar_parafac_eem", _quebra("grade ausente", KeyError))
    cli.dizer("")
    g._menu_tec_parafac(cfg, False)
    monkeypatch.setattr(g, "_tecavan_carregar_dataset", lambda cfg, is_pt, **_kw: None)
    g._menu_tec_parafac(cfg, True)


def test_cow_exige_cromatograma_e_grava_os_alinhados(cli, tmp_path, monkeypatch):
    cfg = g.Config(output_root_folder=str(tmp_path / "saida"))
    cli.dizer("")
    g._menu_tec_cow(cfg, True)
    assert "hplc ou gcms" in cli.saida()
    cfg.mode = "hplc"
    pacote = _dataset(monkeypatch)
    monkeypatch.setattr(ta, "rodar_cow", lambda X, **k: types.SimpleNamespace(
        X_alinhado=pacote[1], correlacao_antes=0.5, correlacao_depois=0.9, indice_referencia=0))
    cli.dizer("", "", "")
    g._menu_tec_cow(cfg, True)
    alinhado = tmp_path / "saida" / "TecnicasEspeciais" / "cow" / "cromatogramas_alinhados.csv"
    assert alinhado.exists()
    # grupo e teor vao junto, para o CSV alinhado voltar ao modo csv com validacao por grupo
    assert alinhado.read_text(encoding="utf-8").splitlines()[0].split(";")[:3] == ["classe", "conc", "grupo"]
    monkeypatch.setattr(ta, "rodar_cow", _quebra("segmentos demais"))
    cli.dizer("99", "5", "")
    g._menu_tec_cow(cfg, False)
    monkeypatch.setattr(g, "_tecavan_carregar_dataset", lambda cfg, is_pt, **_kw: None)
    g._menu_tec_cow(cfg, True)
    assert "segmentos demais" in cli.saida()


@pytest.mark.parametrize("poder, objetivo", [(True, "classificacao"), (False, "quantificacao")])
def test_portao_de_correcao_mostra_o_veredito_e_avisa_quando_falta_poder(cli, monkeypatch, poder, objetivo):
    _dataset(monkeypatch)
    veredito = types.SimpleNamespace(resumo=lambda: "VEREDITO", poder_suficiente=poder)
    monkeypatch.setattr(ta, "rodar_portao_correcao", lambda *a, **k: veredito)
    cli.dizer("osc", "")
    g._menu_tec_portao(g.Config(objective=objetivo), True)
    assert "VEREDITO" in cli.saida() and ("Poder insuficiente" in cli.saida()) is (not poder)
    monkeypatch.setattr(ta, "rodar_portao_correcao", _quebra("poucos grupos"))
    cli.dizer("", "")
    g._menu_tec_portao(g.Config(), True)
    monkeypatch.setattr(g, "_tecavan_carregar_dataset", lambda cfg, is_pt, **_kw: None)
    g._menu_tec_portao(g.Config(), True)


def test_pooled_x_local_mostra_a_recomendacao_por_especie(cli, monkeypatch):
    _dataset(monkeypatch, conc=False)
    decisoes = [types.SimpleNamespace(especie="a", recomendacao="local", veredito=types.SimpleNamespace(resumo=lambda: "aprovado"), n_amostras=9, n_minimo=6),
                types.SimpleNamespace(especie="b", recomendacao="pooled", veredito=None, n_amostras=3, n_minimo=6)]
    monkeypatch.setattr(ta, "rodar_pooled_local", lambda *a: decisoes)
    cli.dizer("")
    g._menu_tec_pooled_local(g.Config(), True)
    assert "aprovado" in cli.saida() and "dados insuficientes" in cli.saida()
    monkeypatch.setattr(ta, "rodar_pooled_local", _quebra("sem especies"))
    cli.dizer("")
    g._menu_tec_pooled_local(g.Config(), False)
    monkeypatch.setattr(g, "_tecavan_carregar_dataset", lambda cfg, is_pt, **_kw: None)
    g._menu_tec_pooled_local(g.Config(), True)


def _blocos(monkeypatch, *, y=True, erro=None):
    def carregar(cfg):
        if erro:
            raise RuntimeError(erro)
        return (np.arange(3.0), np.ones((5, 3)), np.array(list("aabbc")), np.arange(5.0) if y else None, None, None)

    monkeypatch.setattr(g.pq, "load_data", carregar)


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_fusao_multibloco_pergunta_os_dois_blocos_e_mostra_o_resultado(cli, monkeypatch, lang):
    monkeypatch.setitem(g._STATE, "lang", lang)
    _blocos(monkeypatch)
    res = types.SimpleNamespace(rmsep=0.5, r2cal=0.9, r2val=0.8, n_lv=3, n_cal=4, n_val=1)
    monkeypatch.setattr(ta, "rodar_fusao_multibloco", lambda blocos, y: (res, ["nir", "mir"]))
    cli.dizer("nir", "dx", "pasta1", "", "csv", "arq.csv", "Class", "Teor", "")
    g._menu_tec_fusao_multibloco(g.Config(), lang == "PT")
    assert "RMSEP=0.5" in cli.saida()


def test_fusao_multibloco_recusa_carga_falha_falta_de_teor_e_blocos_incompativeis(cli, monkeypatch):
    _blocos(monkeypatch, erro="pasta sem arquivos")
    cli.dizer("a", "dx", "p", "")
    g._menu_tec_fusao_multibloco(g.Config(), True)
    _blocos(monkeypatch, y=False)
    cli.dizer("a", "dx", "p", "b", "dx", "q", "")
    g._menu_tec_fusao_multibloco(g.Config(), False)
    _blocos(monkeypatch)
    monkeypatch.setattr(ta, "rodar_fusao_multibloco", _quebra("linhas diferentes"))
    cli.dizer("a", "dx", "p", "b", "dx", "q", "")
    g._menu_tec_fusao_multibloco(g.Config(), True)
    saida = cli.saida()
    for trecho in ("pasta sem arquivos", "linhas diferentes"):
        assert trecho in saida, trecho


def test_menu_das_tecnicas_avancadas_despacha_pela_tecla(cli, monkeypatch):
    chamadas = []
    for nome in ("_menu_tec_asca", "_menu_tec_epo_glsw", "_menu_tec_mcr_als", "_menu_tec_fusao_multibloco", "_menu_tec_parafac",
                 "_menu_tec_cow", "_menu_tec_portao", "_menu_tec_pooled_local"):
        monkeypatch.setattr(g, nome, (lambda n: lambda cfg, pt: chamadas.append(n))(nome))
    for tecla in "12345678":
        cli.dizer(tecla)
        g._menu_tecnicas_avancadas(g.Config())
    cli.dizer("0")
    g._menu_tecnicas_avancadas(None)
    cli.dizer("99")
    g._menu_tecnicas_avancadas(g.Config())
    assert len(chamadas) == 8
