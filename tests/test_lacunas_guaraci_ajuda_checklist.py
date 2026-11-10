"""Telas Sobre e Ajuda, estimativa de tempo e checklist pre-execucao de `guaraci.guaraci`."""
from __future__ import annotations

import pytest

import guaraci.guaraci as g


@pytest.mark.parametrize("lang", ["PT", "EN"])
def test_sobre_percorre_cada_painel_e_recusa_o_resto(cli, monkeypatch, lang):
    monkeypatch.setitem(g._STATE, "lang", lang)
    # [1] Por que o GUARACI tem duas paginas (nome+metodo, comparativo): dois Enter
    cli.dizer("1", "", "", "2", "", "3", "", "4", "", "x", "", "i", "1", "", "0")
    g._menu_about(g.Config())
    saida = cli.saida()
    assert "GUARACI" in saida and "BibTeX" in saida


def test_sobre_sem_referencias_cadastradas_mostra_traco(cli, monkeypatch):
    monkeypatch.setattr(g, "_REFERENCIAS_GUARACI", {})
    cli.dizer("4", "", "0")
    g._menu_about(None)
    assert "Referencias" in cli.saida()


def test_ajuda_abre_por_numero_nome_busca_e_recusa_o_que_nao_existe(cli):
    cli.dizer("1", "", "help max_lvs", "", "help campo_inexistente_xyz", "", "999", "", "max_lvs", "", "permut", "", "zzzzzzzz", "",
              "i", "0")
    g._menu_help(g.Config())
    saida = cli.saida()
    assert "Nao encontrado" in saida or "Not found" in saida
    assert g._t("invalido") in saida or "invalid" in saida.lower()


def test_estimativa_de_tempo_cobre_as_faixas_e_os_modulos_opcionais(monkeypatch):
    cfg = g.Config()
    assert g._estimar_tempo(cfg, 0) is None and g._estimar_tempo(cfg, -3) is None
    assert g._estimar_tempo(cfg, 10).startswith("<")
    valores = {"max_lvs": 40, "n_splits_cv": 5, "n_permutacoes": 200, "n_jobs_permutacao": 2, "teste_wold": True, "n_permutations_wold": 50,
               "benchmark": True, "monte_carlo": True, "n_monte_carlo": 60, "shap_benchmark": True, "selecao_variaveis_etapa4": True,
               "selecao_ag": True, "selecao_spa": True, "selecao_cars": True, "selecao_uve": True}
    monkeypatch.setattr(g, "_cfgv", lambda c, k, d=None: valores.get(k, d))
    assert "h" in g._estimar_tempo(cfg, 5000)                           # horas
    assert "min" in g._estimar_tempo(cfg, 300)                          # minutos


def _checklist(monkeypatch, **valores):
    padrao = {"pasta_dados": "", "modo_entrada": "dx", "validacao_group_aware": True, "pasta_saida": "", "pre_processamento": ""}
    padrao.update(valores)
    monkeypatch.setattr(g, "_cfgv", lambda c, k, d=None: padrao.get(k, d))
    return g._checklist(g.Config())


def test_checklist_de_pasta_ausente_ou_vazia_bloqueia(cli, monkeypatch, tmp_path):
    ok, erros, _ = _checklist(monkeypatch, pasta_dados=str(tmp_path / "nao_existe"))
    assert not ok and "pasta_dados" in erros
    monkeypatch.setattr(g, "_count_dx", lambda p: 0)
    ok, erros, _ = _checklist(monkeypatch, pasta_dados=str(tmp_path))
    assert not ok and "pasta_dados vazia" in erros


@pytest.mark.parametrize("pre", [
    {"n_apos_descarte": 8, "n_fora_da_faixa": 2, "fora_por_especie": {"a": 2}, "n_sem_mae_id": 1, "n_grupos": 4},
    {"n_apos_descarte": 0, "n_fora_da_faixa": 0, "fora_por_especie": {}, "n_sem_mae_id": 0, "n_grupos": 0},
    OSError("cabecalho ilegivel"),
])
def test_checklist_de_dx_resume_o_prescan_ou_o_erro_dele(cli, monkeypatch, tmp_path, pre):
    import guaraci.dados_io as dio
    monkeypatch.setattr(g, "_count_dx", lambda p: 10)

    def prescan(pasta):
        if isinstance(pre, Exception):
            raise pre
        return pre

    monkeypatch.setattr(dio, "prescan_dx", prescan)
    ok, erros, checks = _checklist(monkeypatch, pasta_dados=str(tmp_path))
    assert ok and any(".dx" in msg for _e, msg in checks)


@pytest.mark.parametrize("nivel", ["high", "medium", "none", "outro"])
def test_checklist_traduz_a_garantia_de_agrupamento(cli, monkeypatch, tmp_path, nivel):
    import guaraci.dados_io as dio
    monkeypatch.setattr(dio, "garantia_agrupamento_prevista", lambda cfg: nivel)
    monkeypatch.setattr(g, "_count_dx", lambda p: 3)
    monkeypatch.setattr(dio, "prescan_dx", lambda p: {"n_apos_descarte": 3, "n_fora_da_faixa": 0, "fora_por_especie": {}, "n_sem_mae_id": 0, "n_grupos": 0})
    ok, erros, checks = _checklist(monkeypatch, pasta_dados=str(tmp_path))
    chave = {"high": "chk_leak_high", "medium": "chk_leak_medium", "none": "chk_leak_none"}.get(nivel, "chk_leak_verificar")
    assert g._t(chave) in [msg for _e, msg in checks] and ok is True


def test_checklist_sem_group_aware_e_um_erro(cli, monkeypatch):
    ok, erros, _ = _checklist(monkeypatch, modo_entrada="sintetico", validacao_group_aware=False)
    assert not ok and "validacao_group_aware" in erros


def test_checklist_de_hsi_csv_e_demais_modos(cli, monkeypatch, tmp_path):
    ok, erros, _ = _checklist(monkeypatch, modo_entrada="hsi", hsi_pasta_dataset=str(tmp_path))
    assert ok
    ok, erros, _ = _checklist(monkeypatch, modo_entrada="hsi", hsi_pasta_dataset=str(tmp_path / "x"))
    assert not ok and "hsi_pasta_dataset" in erros
    csv = tmp_path / "d.csv"
    csv.write_text("a\n1\n2\n", encoding="utf-8")
    ok, erros, _ = _checklist(monkeypatch, modo_entrada="csv", arquivo_csv=str(csv))
    assert ok
    ok, erros, _ = _checklist(monkeypatch, modo_entrada="csv", arquivo_csv=str(tmp_path / "nao.csv"))
    assert not ok and "arquivo_csv" in erros
    import builtins
    abrir = builtins.open

    def abrir_menos_o_csv(arquivo, *a, **k):               # so' o CSV fica bloqueado: no Linux o psutil/importlib tambem usam `open`
        if str(arquivo) == str(csv):
            raise OSError("bloqueado")
        return abrir(arquivo, *a, **k)
    monkeypatch.setattr(builtins, "open", abrir_menos_o_csv)
    _checklist(monkeypatch, modo_entrada="csv", arquivo_csv=str(csv))


@pytest.mark.parametrize("fonte_ok", [True, False])
def test_checklist_dos_demais_modos_usa_a_validacao_da_fonte(cli, monkeypatch, fonte_ok):
    monkeypatch.setattr(g.pq, "validar_fonte_de_dados", lambda cfg: (fonte_ok, "fonte"))
    ok, erros, _ = _checklist(monkeypatch, modo_entrada="spc")
    assert ok is fonte_ok


def test_checklist_avisa_memoria_curta_com_modulos_pesados_e_psutil_ausente(cli, monkeypatch):
    import sys
    import types
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(total=2 * 1024**3))
    _, _, checks = _checklist(monkeypatch, modo_entrada="sintetico", benchmark=True)
    assert any(estado is None and "2.0GB" in msg for estado, msg in checks)
    monkeypatch.setitem(sys.modules, "psutil", None)
    _, _, checks = _checklist(monkeypatch, modo_entrada="sintetico")
    assert any("psutil N/A" in msg for _e, msg in checks)


def test_checklist_impresso_marca_certo_errado_e_neutro(cli, monkeypatch):
    monkeypatch.setattr(g, "_checklist", lambda cfg: (False, ["x"], [(True, "bom"), (False, "ruim"), (None, "neutro")]))
    assert g._print_checklist(g.Config()) is False
    saida = cli.saida()
    assert "bom" in saida and "ruim" in saida and "neutro" in saida
