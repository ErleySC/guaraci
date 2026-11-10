"""Aba Modelo do app web (Streamlit AppTest): execucao, erro, timeout, trava de execucao simultanea, avisos de RAM e de agrupamento.

O motor (`pipeline.executar`) e' trocado por uma funcao rapida; o resto da tela e' o de verdade.
"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import guaraci.dados_io as dio
import guaraci.pipeline as pq
import guaraci.preferencias_visuais as prefs

_RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setenv("GUARACI_SESSION_DIR", tempfile.mkdtemp(prefix="gs"))
    monkeypatch.setattr(prefs, "VISUAL_PATH", tmp_path / "visual_config.json")
    pasta = tmp_path / "appdir"
    pasta.mkdir()
    shutil.copy(_RAIZ / "app_quimiometria.py", pasta / "app_quimiometria.py")
    shutil.copytree(_RAIZ / "assets", pasta / "assets")
    return str(pasta / "app_quimiometria.py")


def _texto(at):
    itens = [e.value for e in at.markdown] + [e.value for e in at.caption] + [e.value for e in at.info] + [e.value for e in at.warning]
    itens += [e.value for e in at.error] + [e.value for e in at.success] + [e.value for e in at.code] + [e.value for e in at.subheader]
    return "\n".join(str(i) for i in itens)


def _modelo(app, **estado):
    at = AppTest.from_file(app, default_timeout=120)
    for k, v in estado.items():
        at.session_state[k] = v
    at.session_state["pagina"] = "modelo"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _rodar(at, chave="btn_rodar"):
    [b for b in at.button if b.key == chave][0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_campos_das_abas_existem_no_spec():
    from guaraci.app_tabs.dados import _DADOS_KEYS
    from guaraci.app_tabs.preprocessamento import _PREPROC_KEYS
    chaves = {s["key"] for s in pq.CONFIG_SPEC}
    assert set(_DADOS_KEYS) <= chaves and set(_PREPROC_KEYS) <= chaves and {"nivel", "max_lvs", "pre_processamento"} <= chaves


def test_execucao_bem_sucedida_guarda_a_pasta_e_o_log(app, monkeypatch, tmp_path):
    def executar(cfg):
        print("[1/7] etapa\nSalvo: fig.png")
        cfg.output_folder = str(tmp_path)
    monkeypatch.setattr(pq, "executar", executar)
    at = _rodar(_modelo(app))
    assert at.session_state["ultima_pasta"] == str(tmp_path) and "Completed" in _texto(at)


def test_execucao_com_problema_critico_de_delineamento_avisa_antes_do_sucesso(app, monkeypatch):
    monkeypatch.setattr(pq, "executar", lambda cfg: print("  [AUDITORIA][CRITICO] n_insuficiente: uma classe so"))
    at = _rodar(_modelo(app))
    assert "CRITICAL design problems" in _texto(at) and "n_insuficiente" in _texto(at)


def test_execucao_que_falha_mostra_codigo_do_erro_no_servidor_e_traceback_no_local(app, monkeypatch):
    def quebra(cfg):
        raise RuntimeError("memoria insuficiente")
    monkeypatch.setattr(pq, "executar", quebra)
    at = _rodar(_modelo(app))
    assert "digo do erro" in _texto(at) and at.session_state["ultima_pasta"] is None
    monkeypatch.setenv("GUARACI_LOCAL_MODE", "1")
    at = _rodar(_modelo(app))
    assert "Error traceback" in _texto(at) and "memoria insuficiente" in _texto(at)


def test_execucao_pelo_botao_do_topo_tambem_roda(app, monkeypatch):
    chamadas = []
    monkeypatch.setattr(pq, "executar", lambda cfg: chamadas.append(1))
    _rodar(_modelo(app), "btn_run_top")
    assert chamadas == [1]


def test_nova_execucao_e_barrada_enquanto_a_anterior_ainda_roda(app, monkeypatch):
    liberar = threading.Event()
    viva = threading.Thread(target=liberar.wait, daemon=True)
    viva.start()
    try:
        at = _modelo(app, _worker_ativo=viva)
        _rodar(at)
        assert "ainda est" in _texto(at)
    finally:
        liberar.set()


def test_execucao_que_passa_do_tempo_maximo_e_encerrada_com_erro(app, monkeypatch):
    import guaraci.app_tabs.modelo as modelo
    monkeypatch.setattr(modelo, "MAX_RUNTIME_S", 0.0)
    monkeypatch.setattr(pq, "executar", lambda cfg: time.sleep(2.0))
    at = _rodar(_modelo(app))
    assert "exceeded maximum runtime" in _texto(at) + str(at.session_state["erro_run"])


def test_indicador_de_memoria_do_processo_tolera_psutil_quebrado(monkeypatch):
    import psutil

    from guaraci.app_tabs.modelo import _ram_mb
    assert _ram_mb() is None or _ram_mb() > 0
    monkeypatch.setattr(psutil, "Process", lambda: (_ for _ in ()).throw(RuntimeError("sem permissao")))
    assert _ram_mb() is None


def test_trabalhador_da_execucao_registra_pasta_erro_e_fim():
    from guaraci.app_logic import LogThreadSafe
    from guaraci.app_tabs.modelo import _rodar_worker
    import types
    ok = {}
    _rodar_worker(types.SimpleNamespace(executar=lambda cfg: None), types.SimpleNamespace(output_folder="x"), LogThreadSafe(), ok)
    assert ok["pasta"] == "x" and ok["erro"] is None and ok["fim"] is True
    ruim = {}
    _rodar_worker(types.SimpleNamespace(executar=lambda cfg: (_ for _ in ()).throw(ValueError("falhou"))), types.SimpleNamespace(),
                  LogThreadSafe(), ruim)
    assert "falhou" in ruim["erro"] and ruim["fim"] is True


@pytest.mark.parametrize("livre, trecho", [(3.0, "Benchmark and SHAP will be switched off"), (6.0, "SHAP and Monte Carlo CV will be reduced")])
def test_aviso_de_ram_baixa_na_aba_modelo(app, monkeypatch, livre, trecho):
    hw = dict(pq.hardware_probe())
    hw["ram_livre_gb"] = livre
    monkeypatch.setattr(pq, "hardware_probe", lambda: hw)
    at = _modelo(app)
    assert trecho in _texto(at)


def test_aba_modelo_tolera_sonda_de_hardware_quebrada(app, monkeypatch):
    monkeypatch.setattr(pq, "hardware_probe", lambda: (_ for _ in ()).throw(RuntimeError("sonda")))
    at = _modelo(app)
    assert not at.exception


@pytest.mark.parametrize("garantia, trecho", [("none", "No group key"), ("medium", "guarantee MEDIUM")])
def test_aba_modelo_explica_a_garantia_de_agrupamento(app, monkeypatch, garantia, trecho):
    monkeypatch.setattr(dio, "garantia_agrupamento_prevista", lambda cfg: garantia)
    assert trecho in _texto(_modelo(app))


def test_aba_modelo_sem_figuras_do_objetivo_e_sem_parametros_restantes(app, monkeypatch):
    from guaraci import catalogo
    monkeypatch.setattr(pq, "describe_plan", lambda cfg: [])
    monkeypatch.setattr(catalogo, "campos_restantes_web", lambda explicitas, specs: [])
    assert "No mode-specific figures" in _texto(_modelo(app))


def test_aba_modelo_mostra_campos_invalidos_e_bloqueia_a_execucao(app, monkeypatch):
    monkeypatch.setattr(pq, "validar_semantico", lambda cfg: ["faixa_min_cm maior que faixa_max_cm"])
    at = _modelo(app)
    assert "Invalid fields in configuration" in _texto(at) and [b for b in at.button if b.key == "btn_rodar"][0].disabled
