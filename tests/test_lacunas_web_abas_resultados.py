"""Abas Validacao, Relatorios e Sobre do app web (Streamlit AppTest), com uma pasta de resultados sintetica.

Cada tela e' desenhada de verdade; os geradores de bytes dos relatorios sao trocados por funcoes que devolvem texto ou levantam erro.
"""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from tests._web_apoio import criar_pasta_de_resultados


def _textos(at):
    partes = [e.value for e in at.markdown] + [e.value for e in at.info] + [e.value for e in at.warning] + [e.value for e in at.error]
    partes += [e.value for e in at.code] + [e.value for e in at.caption]
    return "\n".join(str(p) for p in partes)


def _validacao():
    from guaraci.app_logic import list_figures, load_summary
    from guaraci.app_tabs import validacao
    from tests._web_apoio import TOKENS
    validacao.render(lambda s: s, lambda: TOKENS, load_summary, list_figures)


def test_validacao_sem_execucao_pede_para_rodar_o_pipeline():
    at = AppTest.from_function(_validacao, default_timeout=60)
    at.run()
    assert not at.exception and "No results yet" in _textos(at)


def test_validacao_mostra_resumo_notas_acuracia_por_classe_benchmark_e_galeria(tmp_path):
    pasta = criar_pasta_de_resultados(tmp_path)
    at = AppTest.from_function(_validacao, default_timeout=60)
    at.session_state["ultima_pasta"] = pasta
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    texto = _textos(at)
    assert "Methodological Notes" in texto and "figures generated" in texto
    assert len(at.dataframe) >= 2                                              # acuracia por classe + benchmark + MC
    seletor = [s for s in at.selectbox if s.key == "filtro_valid"][0]
    seletor.select("ROC / AUC")
    at.run()
    assert not at.exception
    seletor = [s for s in at.selectbox if s.key == "filtro_valid"][0]
    seletor.select("SHAP")
    at.run()
    assert "No figures found" in _textos(at)


def test_validacao_com_resumo_simples_sem_figuras_e_com_tabelas_ilegiveis(tmp_path):
    pasta = criar_pasta_de_resultados(tmp_path, figuras=(), resumo="Optimal LVs: 3\n", benchmark=False, mc=False)
    (tmp_path / "resultados" / "execucao_atual" / "Tabelas" / "benchmark_classifiers.csv").write_bytes(b"\xff\xfe\x00\x80\x81")
    at = AppTest.from_function(_validacao, default_timeout=60)
    at.session_state["ultima_pasta"] = pasta
    at.run()
    assert not at.exception and "non-displayable format" in _textos(at)


def _legenda():
    from guaraci.app_tabs.validacao import legenda_figura
    import streamlit as st
    st.write(legenda_figura("fig_pca_scores.png", True), legenda_figura("fig_roc.png", False), legenda_figura("fig_sem_chave.png", True))


def test_legenda_das_figuras_usa_o_idioma_e_cai_no_nome_do_arquivo():
    at = AppTest.from_function(_legenda, default_timeout=60)
    at.run()
    assert not at.exception
    texto = _textos(at)
    assert "Scores da PCA" in texto and "ROC curves per class" in texto and "fig_sem_chave.png" in texto


# ---- relatorios -----------------------------------------------------------------------------------------------------------------------------------


def _relatorios(falhar=False):
    import streamlit as st

    from guaraci.app_logic import list_figures, load_model_card, load_summary
    from guaraci.app_tabs import relatorios

    class _Pq:
        @staticmethod
        def clear_old_results(base, manter):
            return {"removidas": ["x"] if manter < 4 else [], "liberado_mb": 1.0, "erro": "falha parcial" if manter == 1 else ""}

    def gerar(*a):
        if st.session_state.get("falhar"):
            raise RuntimeError("gerador quebrou")
        return b"conteudo"

    relatorios.render(_Pq, {"N1": "Classificacao"}, gerar, gerar, gerar, gerar, gerar, gerar, load_summary, load_model_card, list_figures,
                      lambda s: s)


def test_relatorios_sem_execucao_pede_para_rodar_o_pipeline():
    at = AppTest.from_function(_relatorios, default_timeout=60)
    at.run()
    assert not at.exception and "Run the pipeline" in _textos(at)


def test_relatorios_oferece_downloads_model_card_resumo_galeria_e_log(tmp_path):
    pasta = criar_pasta_de_resultados(tmp_path, execucoes_irmas=1)
    at = AppTest.from_function(_relatorios, default_timeout=120)
    at.session_state["ultima_pasta"] = pasta
    at.session_state["ultimo_log"] = "linha de log"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    texto = _textos(at)
    assert "figure(s) displayed" in texto and "linha de log" in texto
    seletor = [s for s in at.selectbox if s.key == "filtro_rel"][0]
    seletor.select("ROC / AUC")
    at.run()
    assert not at.exception


def test_relatorios_com_varias_execucoes_limpa_as_antigas(tmp_path):
    pasta = criar_pasta_de_resultados(tmp_path, execucoes_irmas=4)
    at = AppTest.from_function(_relatorios, default_timeout=120)
    at.session_state["ultima_pasta"] = pasta
    at.run()
    assert not at.exception
    [s for s in at.slider if s.key == "lim_manter"][0].set_value(1)
    at.run()
    [b for b in at.button if b.key == "btn_limpar_resultados"][0].click()
    at.run()
    assert not at.exception and "falha parcial" in _textos(at)
    [s for s in at.slider if s.key == "lim_manter"][0].set_value(4)
    at.run()
    [b for b in at.button if b.key == "btn_limpar_resultados"][0].click()
    at.run()
    assert "No folders removed" in _textos(at)


def test_relatorios_com_duas_execucoes_nao_usa_slider_e_com_uma_nao_ha_o_que_limpar(tmp_path):
    pasta = criar_pasta_de_resultados(tmp_path, execucoes_irmas=1)
    at = AppTest.from_function(_relatorios, default_timeout=120)
    at.session_state["ultima_pasta"] = pasta
    at.run()
    assert not at.exception and not [s for s in at.slider if s.key == "lim_manter"]
    sozinha = criar_pasta_de_resultados(tmp_path / "outro")
    at = AppTest.from_function(_relatorios, default_timeout=120)
    at.session_state["ultima_pasta"] = sozinha
    at.run()
    assert "Nothing to clean" in _textos(at)


def test_relatorios_mostra_o_erro_de_cada_gerador_que_falha_e_a_ausencia_de_model_card(tmp_path):
    pasta = criar_pasta_de_resultados(tmp_path, figuras=(), resumo=None, model_card=None)
    at = AppTest.from_function(_relatorios, default_timeout=120)
    at.session_state["ultima_pasta"] = pasta
    at.session_state["falhar"] = True
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    texto = _textos(at)
    for trecho in ("ZIP:", "PDF:", "Word:", "Excel:", "LaTeX:", "PowerPoint:", "model_card.md not found", "resumo_modelo.txt not found",
                   "No PNG/JPG images"):
        assert trecho in texto, trecho


def test_relatorios_avisa_quando_o_python_pptx_nao_esta_instalado(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pptx", None)
    pasta = criar_pasta_de_resultados(tmp_path)
    at = AppTest.from_function(_relatorios, default_timeout=120)
    at.session_state["ultima_pasta"] = pasta
    at.run()
    assert not at.exception and "python-pptx not installed" in _textos(at)


# ---- sobre ---------------------------------------------------------------------------------------------------------------------------------------


def _sobre():
    import types

    from guaraci.app_tabs import sobre
    sobre.render(types.SimpleNamespace(__version__="9.9.9"), lambda s: s)


@pytest.mark.parametrize("idioma", ["PT", "EN"])
def test_sobre_mostra_versao_proposito_e_diferenciais_no_idioma_escolhido(idioma):
    at = AppTest.from_function(_sobre, default_timeout=60)
    at.session_state["lang"] = idioma
    at.run()
    assert not at.exception and "9.9.9" in _textos(at) + str([m.value for m in at.metric])
