"""Tela Início: painel de status do projeto.

Contrato central (regra "evidência ou silêncio"): todo número exibido vem de
execução real ou de `st.session_state`. Sem dado carregado, o cartão diz
isso -- nunca mostra número zerado como se fosse resultado.

O painel saiu do topo da aba Projeto e virou a tela Início na
reestruturação de 2026-09-08 (navegação lateral); os testes navegam via
`session_state["pagina"]`.
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


def _abrir_inicio(**estado) -> AppTest:
    at = AppTest.from_file(_APP, default_timeout=90)
    at.session_state["lang"] = "PT"
    at.session_state["pagina"] = "inicio"
    for chave, valor in estado.items():
        at.session_state[chave] = valor
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _textos(at: AppTest) -> str:
    return "\n".join(
        [m.value for m in at.markdown] + [c.value for c in at.caption]
        + [i.value for i in at.info] + [e.value for e in at.error]
        + [w.value for w in at.warning] + [s.value for s in at.success])


def test_sem_dado_nenhum_o_painel_diz_por_onde_comecar_e_nao_mostra_numero():
    at = _abrir_inicio()
    textos = _textos(at)
    assert "Nenhum dado carregado ainda" in textos
    # Nenhuma contagem inventada antes de existir dado. (O `>N<` é a forma
    # como o cartão renderiza um número; procurar só "934" acharia o dígito
    # dentro do base64 da logo.)
    assert ">934<" not in textos and ">311<" not in textos
    # E a proxima acao sugerida aponta para onde comecar.
    assert "Carregue os espectros" in textos


def test_com_previa_de_dados_mostra_as_contagens_da_previa():
    at = _abrir_inicio(previa_dados={"n_espectros": 42, "n_classes": 3})
    textos = _textos(at)
    assert ">42<" in textos and ">3<" in textos
    # Sem execucao nao ha' amostra fisica nem variaveis -- "—", nunca 0.
    assert ">—<" in textos
    assert "prévia dos dados (sem execução ainda)" in textos


def test_com_execucao_le_contagens_e_auditoria_reais_da_pasta(pasta_run: Path):
    at = _abrir_inicio(ultima_pasta=str(pasta_run))
    textos = _textos(at)
    assert ">137<" in textos          # espectros, do resumo_modelo.txt
    assert ">5<" in textos            # classes
    assert "última execução concluída" in textos
    # O achado critico do JSON vira a proxima acao sugerida, com o texto real.
    assert "1 achado(s) crítico(s)" in textos
    assert "Classe X com 2 grupos" in textos


def test_execucao_sem_json_de_auditoria_nao_afirma_que_esta_tudo_ok(
        tmp_path: Path):
    """Ausência de registro é "não sei", nunca "nenhum problema"."""
    rel = tmp_path / "Relatorios"
    rel.mkdir(parents=True)
    (rel / "resumo_modelo.txt").write_text(_RESUMO_FALSO, encoding="utf-8")

    at = _abrir_inicio(ultima_pasta=str(tmp_path))
    textos = _textos(at)
    # Sem registro de auditoria, nada de "0 critico" -- a tela nao afirma
    # ausencia de problema; ela sugere o proximo passo do fluxo.
    assert "0 crítico" not in textos
    assert "achado(s) crítico(s)" not in textos
