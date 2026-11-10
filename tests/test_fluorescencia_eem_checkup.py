"""Checkup fluorescencia/EEM: layout por subpasta aceito pelo validador, grade heterogenea recusada (nao truncada),
PARAFAC da tela [3.5] com regiao mascarada (zeros constantes) e sem tensorly, nao-negatividade + CORCONDIA, presets."""
import os

import numpy as np
import pytest

from guaraci.config import Config
from guaraci.formatos_sinteticos import escrever_eem_dat, gerar_eem


def _cfg_eem(pasta):
    cfg = Config()
    cfg.mode = "eem"
    cfg.input_folder = str(pasta)
    return cfg


def test_validador_aceita_o_layout_documentado_uma_subpasta_por_classe(tmp_path):
    from guaraci.config_io import validar_fonte_de_dados
    gerar_eem(str(tmp_path), n_classes=2, n_amostras=2)
    ok, msg = validar_fonte_de_dados(_cfg_eem(tmp_path))
    assert ok, msg
    assert "4 arquivos" in msg


def test_leitor_eem_recusa_grade_diferente_em_vez_de_truncar(tmp_path):
    from guaraci.leitores_avancados import _leitor_eem
    ex, em = np.arange(250.0, 301.0, 10.0), np.arange(300.0, 351.0, 5.0)
    m = np.ones((em.size, ex.size))
    escrever_eem_dat(str(tmp_path / "a" / "s1.dat"), ex, em, m)
    escrever_eem_dat(str(tmp_path / "a" / "s2.dat"), ex + 3.0, em, m)       # mesmo tamanho, outra grade
    with pytest.raises(ValueError, match="grade"):
        _leitor_eem(_cfg_eem(tmp_path))
    os.remove(tmp_path / "a" / "s2.dat")
    escrever_eem_dat(str(tmp_path / "a" / "s2.dat"), np.r_[ex, 310.0], em, np.ones((em.size, ex.size + 1)))
    with pytest.raises(ValueError, match="grade"):
        _leitor_eem(_cfg_eem(tmp_path))


def _pasta_eem_mascarada(tmp_path, n=6):
    """2 fluoroforos; a regiao em < ex + 10 vem zerada (mascara de espalhamento do exportador) -> colunas constantes."""
    rng = np.random.default_rng(1)
    ex, em = np.arange(250.0, 401.0, 10.0), np.arange(260.0, 501.0, 8.0)
    EX, EM = np.meshgrid(ex, em)
    perf = [np.outer(np.exp(-((em - 350) / 25) ** 2), np.exp(-((ex - 280) / 15) ** 2)),
            np.outer(np.exp(-((em - 440) / 35) ** 2), np.exp(-((ex - 340) / 25) ** 2))]
    for i in range(n):
        c = rng.uniform(0.2, 1.0, 2)
        m = (c[0] * perf[0] + c[1] * perf[1]) * 1000.0
        m[EM < EX + 10] = 0.0
        escrever_eem_dat(str(tmp_path / "cl" / f"s{i}.dat"), ex, em, m)
    return ex, em


def _rodar_menu_parafac(monkeypatch, cfg, resposta="2"):
    from guaraci import guaraci as g
    saida = []
    monkeypatch.setattr(g, "_ask", lambda *a, **k: resposta)
    monkeypatch.setattr(g, "_pause", lambda *a, **k: None)
    monkeypatch.setattr(g.console, "print", lambda *a, **k: saida.append(" ".join(str(x) for x in a)))
    g._menu_tec_parafac(cfg, True)
    return "\n".join(saida)


def test_menu_parafac_mantem_a_regiao_mascarada_da_grade(monkeypatch, tmp_path):
    pytest.importorskip("tensorly")
    _pasta_eem_mascarada(tmp_path)
    cfg = _cfg_eem(tmp_path)
    cfg.output_root_folder = str(tmp_path / "saida")
    texto = _rodar_menu_parafac(monkeypatch, cfg)
    assert "precisam ter a mesma grade" not in texto
    assert "CORCONDIA" in texto
    assert os.path.isfile(tmp_path / "saida" / "TecnicasEspeciais" / "parafac" / "parafac_fator_amostra.csv")


def test_menu_parafac_sem_tensorly_avisa_em_vez_de_quebrar(monkeypatch, tmp_path):
    import guaraci.tecnicas_avancadas as ta
    _pasta_eem_mascarada(tmp_path)

    def _sem_tensorly(*a, **k):
        raise ModuleNotFoundError("No module named 'tensorly'")
    monkeypatch.setattr(ta, "rodar_parafac_eem", _sem_tensorly)
    texto = _rodar_menu_parafac(monkeypatch, _cfg_eem(tmp_path))
    assert "multiway" in texto


def test_parafac_eem_nao_negativo_e_corcondia_aponta_o_numero_de_fatores():
    pytest.importorskip("tensorly")
    from guaraci.eem_multiway import parafac_eem
    rng = np.random.default_rng(0)
    ex, em = np.linspace(250, 400, 12), np.linspace(300, 500, 20)
    A = rng.uniform(0.2, 1.0, (10, 2))
    B = np.stack([np.exp(-((ex - 300) / 30) ** 2), np.exp(-((ex - 360) / 25) ** 2)], 1)
    C = np.stack([np.exp(-((em - 350) / 30) ** 2), np.exp(-((em - 450) / 35) ** 2)], 1)
    T = np.einsum("ir,jr,kr->ijk", A, B, C) + rng.normal(0, 1e-3, (10, 12, 20))
    r2 = parafac_eem(T, 2)
    assert r2.fator_amostra.min() >= 0 and r2.fator_excitacao.min() >= 0 and r2.fator_emissao.min() >= 0
    assert r2.consistencia_nucleo > 90
    assert parafac_eem(T, 3).consistencia_nucleo < 50          # componente a mais derruba o nucleo
    corr = np.corrcoef(np.hstack([A, r2.fator_amostra]).T)[:2, 2:]
    assert corr.max(axis=1).min() > 0.99                       # escores positivos e proporcionais a concentracao


def test_preset_da_fluorescencia_preserva_intensidade_e_exemplo_eem_usa_mc(tmp_path):
    from guaraci import catalogo_tecnicas as cat
    from guaraci.exemplos import gerar
    assert cat.INSTRUMENTOS["fluorescencia"]["preproc"] == "mc"
    gerar("fluorescencia_eem", str(tmp_path))
    texto = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "pre_processamento: mc" in texto
