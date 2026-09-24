"""Regressao dos achados B2 e B4-B9 de docs/AUDITORIA_UX_2026-09-24.md.

B1 (HSI) fica em test_menu_hsi.py e B3/B11 (selecao de amostras) em
test_selecao_amostras.py, junto dos testes que ja existiam dessas telas.
"""
from __future__ import annotations

import pytest

import guaraci.guaraci as g
from guaraci import paleta_cores
from guaraci.figuras import adaptive_scatter_parameters


@pytest.fixture(autouse=True)
def _isolar_estado_do_usuario(monkeypatch, tmp_path):
    """Nada destes testes pode tocar o ~/.guaraci real (ver armadilha de
    2026-09-08): preferencias visuais e perfis vao para tmp_path."""
    monkeypatch.setattr(g, "_VISUAL_PATH", tmp_path / "visual.json")
    monkeypatch.setattr(g, "_PERFIS_DIR", tmp_path / "perfis")
    monkeypatch.setattr(g, "_cls", lambda: None)
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    yield
    paleta_cores.set_point_alpha(None)


def _entradas(monkeypatch, *valores):
    seq = iter(valores)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(seq))


# ── B2: [?] funciona em toda tela de configuracao ─────────────────────────

@pytest.mark.parametrize("menu", [
    "_menu_modeling", "_menu_preprocessing", "_menu_validation",
    "_menu_advanced", "_menu_visualization",
])
def test_tecla_ajuda_abre_ajuda_do_campo(monkeypatch, menu):
    chamados = []
    monkeypatch.setattr(g, "_mostrar_ajuda", chamados.append)
    _entradas(monkeypatch, "?", "1", "0")
    getattr(g, menu)(g.Config())
    assert len(chamados) == 1, f"{menu}: [?] anunciado no rodape mas nao abre ajuda"


@pytest.mark.parametrize("menu", [
    "_menu_preprocessing", "_menu_validation", "_menu_advanced",
    "_menu_visualization",
])
def test_enter_vazio_volta_como_nas_outras_telas(monkeypatch, menu):
    """B10: Enter vazio = voltar (antes: 'invalido' + pausa nessas telas)."""
    _entradas(monkeypatch, "")
    getattr(g, menu)(g.Config())   # StopIteration aqui = pediu mais entrada


def test_todo_campo_de_configuracao_tem_ajuda_propria():
    sem_ajuda = [s["key"] for s in g._CONFIG_SPEC if s["key"] not in g._HELP_DB]
    assert sem_ajuda == []


def test_descricoes_comecam_pelo_que_o_campo_e():
    """Nota 10 do autor: nada de 'Alem dos metodos acima, roda tambem...'."""
    for chave, h in g._HELP_DB.items():
        for lang in ("PT", "EN"):
            desc = (h.get(lang) or {}).get("desc", "").lower()
            assert not desc.startswith(("alem dos metodos", "in addition to the methods")), chave


# ── B4: transparencia da grade fora de 0-1 nao e' gravada ─────────────────

def test_grade_rejeita_transparencia_fora_da_faixa(monkeypatch):
    _entradas(monkeypatch, "D", "4", "5", "0")
    g._menu_visualization(g.Config())
    assert 0.0 < float(g._carregar_visual_cfg().get("grid_alpha", 0.4)) <= 1.0


def test_grade_aceita_virgula_decimal(monkeypatch):
    _entradas(monkeypatch, "D", "4", "0,6", "0")
    g._menu_visualization(g.Config())
    assert g._carregar_visual_cfg()["grid_alpha"] == pytest.approx(0.6)


@pytest.mark.parametrize("salvo,esperado", [
    (5.0, 0.4), (-1, 0.4), ("abc", 0.4), (None, 0.4), (0.7, 0.7), (1.0, 1.0)])
def test_valor_antigo_invalido_ja_salvo_nao_quebra_figuras(salvo, esperado):
    assert g._grid_alpha_valido(salvo) == esperado


# ── B5: [7] > [A] Transparencia dos pontos chega as figuras ───────────────

def test_transparencia_dos_pontos_padrao_continua_automatica():
    paleta_cores.set_point_alpha(paleta_cores.ALPHA_PONTOS_PRESETS["medio"])
    assert adaptive_scatter_parameters(10, 2)[1] == 0.92
    assert adaptive_scatter_parameters(2000, 2)[1] == 0.40


@pytest.mark.parametrize("preset,alfa", [("baixo", 0.9), ("alto", 0.35)])
def test_transparencia_dos_pontos_fixa_substitui_a_automatica(preset, alfa):
    paleta_cores.set_point_alpha(paleta_cores.ALPHA_PONTOS_PRESETS[preset])
    assert adaptive_scatter_parameters(10, 2)[1] == alfa
    assert adaptive_scatter_parameters(2000, 2)[1] == alfa


def test_transparencia_invalida_volta_ao_automatico():
    paleta_cores.set_point_alpha(7.0)
    assert paleta_cores.get_point_alpha() is None


# ── B6/B7/B8: Perfis ──────────────────────────────────────────────────────

_IDX_ALTA_RIGOROSIDADE = "8"
_IDX_ACESSIBILIDADE = "10"


def test_perfil_acessibilidade_nao_mexe_na_analise(monkeypatch):
    cfg = g.Config(level="N3", max_lvs=12, run_ddsimca=False)
    antes = {k: v for k, v in vars(cfg).items()}
    monkeypatch.setattr(g, "_rodar_pipeline", lambda c: pytest.fail("nao devia rodar"))
    _entradas(monkeypatch, _IDX_ACESSIBILIDADE, "")
    g._menu_profiles(cfg)
    mudou = {k for k, v in vars(cfg).items() if antes.get(k) != v}
    assert mudou <= {"show_class_markers"}, mudou
    assert g._carregar_visual_cfg()["paleta"] == "daltonismo_safe"


def test_perfil_de_rigor_em_quantificacao_nao_deixa_opcao_inerte_ligada(monkeypatch):
    cfg = g.Config(level="N3")
    monkeypatch.setattr(g, "_rodar_pipeline", lambda c: None)
    _entradas(monkeypatch, _IDX_ALTA_RIGOROSIDADE, "n")
    g._menu_profiles(cfg)
    for chave in ("benchmark", "monte_carlo", "shap_benchmark"):
        assert g._get_val(cfg, chave) is False, chave


def test_enter_apos_aplicar_perfil_nao_dispara_execucao(monkeypatch):
    rodou = []
    monkeypatch.setattr(g, "_rodar_pipeline", rodou.append)
    _entradas(monkeypatch, "1", "")
    g._menu_profiles(g.Config())
    assert rodou == []


def test_confirmar_explicitamente_ainda_roda(monkeypatch):
    rodou = []
    monkeypatch.setattr(g, "_rodar_pipeline", rodou.append)
    _entradas(monkeypatch, "1", "s")
    g._menu_profiles(g.Config())
    assert len(rodou) == 1


# ── B9: [G] na Auditoria volta para a tela ────────────────────────────────

def test_auditoria_volta_para_a_tela_depois_do_assistente(monkeypatch):
    chamados = []
    monkeypatch.setattr(g, "_abrir_assistente", lambda *a, **k: chamados.append(a))
    carregou = []
    monkeypatch.setattr(g.pq, "load_data", lambda *a, **k: carregou.append(1))
    _entradas(monkeypatch, "G", "0")
    g._menu_audit(g.Config(mode="sintetico"))
    assert len(chamados) == 1
    assert carregou == []   # saiu pelo [0], sem rodar a auditoria
