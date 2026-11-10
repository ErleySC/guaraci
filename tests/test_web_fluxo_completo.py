"""Fluxo web COMPLETO pelo ponto de entrada real (B1 e B2 do parecer 2026-09-25).

Por que existe: B1 (Config padrao x gerador sintetico) e B2 (o web executava
uma configuracao diferente da exibida) passaram por ~1.900 testes porque
ninguem exercitava o caminho do usuario -- Dados -> Pre-processamento ->
Modelo -> Run -- pelo app de verdade. Cada teste abaixo e' a versao
automatizada de um script do revisor 2 (`apptest_default.py`,
`apptest_persist.py`, `apptest_preproc.py`).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_RAIZ = Path(__file__).resolve().parents[1]

# Desliga os extras pesados: o teste exercita o FLUXO, nao o custo cientifico.
_LEVE = {
    "w_n_permutacoes": 5, "w_max_lvs": 4,
    "w_selecao_lv_cv_aninhada": False, "w_teste_wold": False,
    "w_teste_cv_anova": False, "w_teste_martens": False,
    "w_selecao_variaveis_etapa4": False, "w_ddsimca": False,
    "w_opls_da": False, "w_comparar_pre_processamentos": False,
    "w_benchmark": False, "w_benchmark_regressao": False,
    "w_monte_carlo": False, "w_shap_benchmark": False,
}


@pytest.fixture()
def app(tmp_path, monkeypatch):
    # Pasta de sessao CURTA: o guarda de MAX_PATH (M-14) recusa saidas longas, e
    # o tmp_path do pytest carrega o nome do teste (pode passar de 200 caracteres).
    import tempfile
    sessoes = tempfile.mkdtemp(prefix="gs")
    monkeypatch.setenv("GUARACI_SESSION_DIR", sessoes)
    pasta = tmp_path / "appdir"          # sem config.yaml: nunca le o do autor
    pasta.mkdir()
    shutil.copy(_RAIZ / "app_quimiometria.py", pasta / "app_quimiometria.py")
    shutil.copytree(_RAIZ / "assets", pasta / "assets")
    return str(pasta / "app_quimiometria.py")


def _set(at: AppTest, chave: str, valor) -> None:
    for colecao in (at.selectbox, at.text_input, at.number_input, at.checkbox):
        for w in colecao:
            if w.key == chave:
                w.set_value(valor)
                return
    raise AssertionError(f"widget {chave!r} nao encontrado na tela atual")


def _ir(at: AppTest, pagina: str) -> None:
    at.session_state["pagina"] = pagina
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]


def _configurar_leve(at: AppTest) -> None:
    _ir(at, "modelo")
    for chave, valor in _LEVE.items():
        try:
            _set(at, chave, valor)
        except AssertionError:
            pass     # widget ausente nesta versao da tela: nada a desligar
    at.run()


def test_web_padrao_sem_config_yaml_roda_o_demo_sintetico(app):
    """B1 (revisor2/apptest_default.py): o app SEM config.yaml -- o cenario do
    deploy publico -- roda o demo sintetico ate o fim, sem editar nada."""
    at = AppTest.from_file(app, default_timeout=600)
    at.run()
    assert not at.exception
    _configurar_leve(at)
    botao = [b for b in at.button if b.key == "btn_rodar"][0]
    assert not botao.disabled
    botao.click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error, [str(e.value) for e in at.error]
    assert at.session_state["erro_run"] is None
    pasta = at.session_state["ultima_pasta"]
    assert pasta and Path(pasta).is_dir()
    assert any(Path(pasta).rglob("resumo_modelo.txt"))


def test_web_executa_a_configuracao_exibida_preprocessamento(app, monkeypatch, tmp_path):
    """B2 (revisor2/apptest_preproc.py): o pre-processamento escolhido na tela
    Pre-processamento e' o que roda depois de passar por Modelo."""
    monkeypatch.setenv("GUARACI_LOCAL_MODE", "1")
    at = AppTest.from_file(app, default_timeout=600)
    at.run()
    _ir(at, "dados")
    _set(at, "w_pasta_saida", str(tmp_path / "saida"))   # nao suja o repositorio
    at.run()
    _ir(at, "preprocessamento")
    _set(at, "w_pre_processamento", "Autoscaling")
    at.run()
    _configurar_leve(at)
    assert at.session_state["cfg_base"].default_preprocessing == "autoscaling"
    mostra = " ".join(str(c.value) for c in list(at.caption) + list(at.markdown))
    assert "Autoscaling" in mostra
    [b for b in at.button if b.key == "btn_rodar"][0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    log = at.session_state["ultimo_log"]
    assert "pre-proc autoscaling" in log, log[:600]
    assert "msc_sg_mc" not in log.split("pre-proc", 1)[1].splitlines()[0]


def test_web_preserva_o_modo_csv_entre_telas(app, monkeypatch):
    """B2 (revisor2/apptest_persist.py): o modo/arquivo escolhidos em Dados
    continuam valendo em Modelo e ao voltar a Dados."""
    monkeypatch.setenv("GUARACI_LOCAL_MODE", "1")
    at = AppTest.from_file(app, default_timeout=120)
    at.run()
    _ir(at, "dados")
    _set(at, "w_modo_entrada", "csv")
    at.run()
    _set(at, "w_arquivo_csv", "C:/x/meu.csv")
    at.run()
    _ir(at, "modelo")
    cfg = at.session_state["cfg_base"]
    assert cfg.mode == "csv" and cfg.csv_file == "C:/x/meu.csv"
    _ir(at, "dados")
    modo = [s for s in at.selectbox if s.key == "w_modo_entrada"][0]
    assert modo.value == "csv"
    arq = [t for t in at.text_input if t.key == "w_arquivo_csv"][0]
    assert arq.value == "C:/x/meu.csv"


def test_web_valor_invalido_nao_derruba_a_configuracao(app, monkeypatch):
    monkeypatch.setenv("GUARACI_LOCAL_MODE", "1")
    at = AppTest.from_file(app, default_timeout=120)
    at.run()
    _ir(at, "preprocessamento")
    _set(at, "w_faixa_min_cm", 5000.0)
    at.run()
    _ir(at, "modelo")
    assert at.session_state["cfg_base"].wn_min == 5000.0


@pytest.fixture(autouse=True)
def _limpa_sessoes(monkeypatch):
    yield
    import os
    d = os.environ.get('GUARACI_SESSION_DIR')
    if d and os.path.basename(d).startswith('gs'):
        shutil.rmtree(d, ignore_errors=True)
