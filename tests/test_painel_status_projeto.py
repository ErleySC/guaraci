"""Painel de status no topo da aba Projeto.

Contrato central (regra "evidência ou silêncio"): todo número exibido vem de
execução real ou de `st.session_state`. Sem dado carregado, o painel diz
isso em uma linha -- nunca mostra cartão vazio/zero como se fosse resultado.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_RAIZ = Path(__file__).resolve().parents[1]
_APP = str(_RAIZ / "app_quimiometria.py")

_RESUMO_FALSO = """\
============================================================
  PLS-DA Model Summary
============================================================

  Total samples                       : 137
  Total variables                     : 759
  Total classes                       : 5
  Balanced accuracy                   : 0.8123
"""

_ACHADOS_FALSOS = [
    {"nome": "agrupamento", "severidade": "ok",
     "mensagem": "mae_id presente em todas as amostras."},
    {"nome": "n_insuficiente", "severidade": "critico",
     "mensagem": "Classe X com 2 grupos -- abaixo do minimo."},
]


@pytest.fixture
def pasta_run(tmp_path: Path) -> Path:
    """Pasta de resultados minima, no layout que o pipeline grava."""
    rel = tmp_path / "Relatorios"
    rel.mkdir(parents=True)
    (rel / "resumo_modelo.txt").write_text(_RESUMO_FALSO, encoding="utf-8")
    (rel / "auditoria_delineamento.json").write_text(
        json.dumps(_ACHADOS_FALSOS), encoding="utf-8")
    return tmp_path


def _textos(at: AppTest) -> str:
    return "\n".join(
        [m.value for m in at.markdown] + [c.value for c in at.caption]
        + [i.value for i in at.info] + [e.value for e in at.error]
        + [w.value for w in at.warning] + [s.value for s in at.success])


def test_sem_dado_nenhum_o_painel_diz_por_onde_comecar_e_nao_mostra_numero():
    at = AppTest.from_file(_APP, default_timeout=60)
    at.session_state["lang"] = "PT"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    assert "Nenhum dado carregado ainda" in _textos(at)
    assert "Status do projeto" not in _textos(at)
    # Nenhuma metrica de contagem inventada antes de existir dado.
    rotulos = [m.label for m in at.metric]
    assert "espectros" not in rotulos and "classes" not in rotulos


def test_com_previa_de_dados_mostra_as_contagens_da_previa():
    at = AppTest.from_file(_APP, default_timeout=60)
    at.session_state["lang"] = "PT"
    at.session_state["previa_dados"] = {"n_espectros": 42, "n_classes": 3}
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    metricas = {m.label: m.value for m in at.metric}
    assert metricas.get("espectros") == "42"
    assert metricas.get("classes") == "3"
    # Sem execucao nao ha' contagem de variaveis -- "—", nunca 0.
    assert metricas.get("variáveis espectrais") == "—"
    assert "prévia dos dados (sem execução ainda)" in _textos(at)


def test_com_execucao_le_contagens_e_auditoria_reais_da_pasta(pasta_run: Path):
    at = AppTest.from_file(_APP, default_timeout=60)
    at.session_state["lang"] = "PT"
    at.session_state["ultima_pasta"] = str(pasta_run)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    metricas = {m.label: m.value for m in at.metric}
    assert metricas.get("espectros") == "137"
    assert metricas.get("classes") == "5"
    assert metricas.get("variáveis espectrais") == "759"

    textos = _textos(at)
    assert "última execução concluída" in textos
    # Achado critico do JSON aparece com a contagem real, nao texto fixo.
    assert "1 crítico(s)" in textos
    assert "Classe X com 2 grupos" in textos


def test_execucao_sem_json_de_auditoria_nao_afirma_que_esta_tudo_ok(
        tmp_path: Path):
    """Ausência de registro é "não sei", nunca "nenhum problema"."""
    rel = tmp_path / "Relatorios"
    rel.mkdir(parents=True)
    (rel / "resumo_modelo.txt").write_text(_RESUMO_FALSO, encoding="utf-8")

    at = AppTest.from_file(_APP, default_timeout=60)
    at.session_state["lang"] = "PT"
    at.session_state["ultima_pasta"] = str(tmp_path)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    textos = _textos(at)
    assert "Auditoria de delineamento indisponível" in textos
    assert "0 crítico(s)" not in textos
