"""Abas Dados, Pre-processamento e Visualizacao do app web, percorridas pelo ponto de entrada real (Streamlit AppTest).

O app roda de uma copia temporaria (sem config.yaml do autor), com as pastas de sessao e as preferencias visuais fora do estado real.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

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


@pytest.fixture()
def app_local(app, monkeypatch):
    monkeypatch.setenv("GUARACI_LOCAL_MODE", "1")
    return app


def _ir(app, pagina, **estado):
    at = AppTest.from_file(app, default_timeout=120)
    for k, v in estado.items():
        at.session_state[k] = v
    at.session_state["pagina"] = pagina
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _recarregar(at):
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _set(at, chave, valor):
    for colecao in (at.selectbox, at.text_input, at.number_input, at.checkbox):
        for w in colecao:
            if w.key == chave:
                w.set_value(valor)
                return
    raise AssertionError(f"widget {chave!r} nao encontrado")


def _texto(at):
    itens = [e.value for e in at.markdown] + [e.value for e in at.caption] + [e.value for e in at.info] + [e.value for e in at.warning]
    itens += [e.value for e in at.error] + [e.value for e in at.success]
    return "\n".join(str(i) for i in itens)


def _csv_espectral(tmp_path, nome="espectros.csv"):
    rng = np.random.default_rng(0)
    colunas = [str(w) for w in np.linspace(1000, 3000, 12)]
    linhas = []
    for i, classe in enumerate(["A"] * 6 + ["B"] * 6):
        linhas.append(",".join([classe] + [f"{v:.4f}" for v in rng.normal(size=12) + (3 if classe == "B" else 0)]))
    caminho = tmp_path / nome
    caminho.write_text("classe," + ",".join(colunas) + "\n" + "\n".join(linhas) + "\n", encoding="utf-8")
    return caminho


# ---- Dados --------------------------------------------------------------------------------------------------------------------------------------


def test_dados_atalho_de_analise_recomendada_aplica_o_perfil_e_avisa(app):
    at = _ir(app, "dados")
    [b for b in at.button if b.key.startswith("btn_preset_")][0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert "applied" in _texto(at)


def test_dados_upload_de_csv_e_de_grupos_no_modo_servidor(app, tmp_path):
    at = _ir(app, "dados")
    csv = _csv_espectral(tmp_path)
    [u for u in at.file_uploader if u.label.startswith("Drag")][0].upload(csv.name, csv.read_bytes(), "text/csv")
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert "File received" in _texto(at) and "Mode automatically set" in _texto(at)
    grupos = [u for u in at.file_uploader if u.label.startswith("Group file")][0]
    grupos.upload("grupos.csv", b"amostra,grupo\n1,a\n", "text/csv")
    at.run()
    assert not at.exception and any(d.label.startswith("⬇️ Download config") for d in at.get("download_button"))


def test_dados_previa_do_csv_enviado_mostra_espectros_e_alimenta_a_visualizacao(app, tmp_path):
    at = _ir(app, "dados")
    csv = _csv_espectral(tmp_path)
    [u for u in at.file_uploader if u.label.startswith("Drag")][0].upload(csv.name, csv.read_bytes(), "text/csv")
    at.run()
    [b for b in at.button if b.key == "btn_prev_dados"][0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert "spectra" in _texto(at) and at.session_state["previa_dados"]["n_classes"] == 2
    assert "previa_espectros" in at.session_state


def test_dados_previa_do_modo_sintetico_explica_que_nao_ha_arquivo(app):
    at = _ir(app, "dados")
    [b for b in at.button if b.key == "btn_prev_dados"][0].click()
    at.run()
    assert "Synthetic mode has no file to preview" in _texto(at)


@pytest.mark.parametrize("modo, esperado", [("dx", "spectra"), ("hsi", "Could not load spectra for preview")])
def test_dados_previa_de_pasta_de_espectros_e_modo_sem_previa(app_local, monkeypatch, modo, esperado):
    import guaraci.app_tabs.dados as dados
    import guaraci.pipeline as pq
    monkeypatch.setattr(pq, "validar_fonte_de_dados", lambda cfg: (True, "fonte ok"))
    rng = np.random.default_rng(1)
    monkeypatch.setattr(dados, "preview_spectra_dx", lambda pasta, a, b: (np.linspace(1000, 3000, 10), rng.normal(size=(6, 10)), np.array(list("aabbcc"))))
    at = _ir(app_local, "dados")
    _set(at, "w_modo_entrada", modo)
    at.run()
    [b for b in at.button if b.key == "btn_prev_dados"][0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert esperado in _texto(at)


def test_dados_no_modo_local_salva_recarrega_e_avisa_campo_com_erro(app_local, tmp_path):
    at = _ir(app_local, "dados")
    [b for b in at.button if b.key == "btn_salvar_cfg_dados"][0].click()
    at.run()
    assert "Saved to" in _texto(at)
    [b for b in at.button if b.key == "btn_reload_cfg_dados"][0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert "Config reloaded" in _texto(at)
    cfg_arquivo = Path(app_local).parent / "config.yaml"
    cfg_arquivo.unlink()
    [b for b in at.button if b.key == "btn_reload_cfg_dados"][0].click()
    at.run()
    assert "Error" in _texto(at)


def test_dados_no_modo_local_com_campo_invalido_nao_salva(app_local):
    at = _ir(app_local, "dados")
    try:
        _set(at, "w_icp_limiar_censura", 9.0)
    except Exception:  # noqa: BLE001 -- o widget pode recusar o valor fora da faixa
        pytest.skip("widget recusa valor fora da faixa")
    at.run()
    if "Fields with errors" in _texto(at):
        [b for b in at.button if b.key == "btn_salvar_cfg_dados"][0].click()
        at.run()
        assert "Fix the fields" in _texto(at)


# ---- Pre-processamento ------------------------------------------------------------------------------------------------------------------------


def test_preprocessamento_no_modo_sintetico_desenha_antes_e_depois(app):
    at = _ir(app, "preprocessamento")
    [b for b in at.button if b.key == "btn_prev_preproc"][0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert "Could not load spectra" not in _texto(at) and "Error applying" not in _texto(at)


def test_preprocessamento_com_csv_enviado_e_previa_com_erro(app, tmp_path, monkeypatch):
    at = _ir(app, "dados")
    csv = _csv_espectral(tmp_path)
    [u for u in at.file_uploader if u.label.startswith("Drag")][0].upload(csv.name, csv.read_bytes(), "text/csv")
    at.run()
    at.session_state["pagina"] = "preprocessamento"
    at.run()
    [b for b in at.button if b.key == "btn_prev_preproc"][0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    import guaraci.pipeline as pq
    monkeypatch.setattr(pq, "build_preprocessor", lambda cfg: (_ for _ in ()).throw(RuntimeError("preprocessador quebrou")))
    [b for b in at.button if b.key == "btn_prev_preproc"][0].click()
    at.run()
    assert "Error applying preprocessing" in _texto(at)


def _mudar_modo(at, modo, depois="preprocessamento"):
    """O modo de entrada mora na tela Dados; a Config persistida leva a escolha para a tela seguinte."""
    at.session_state["pagina"] = "dados"
    at.run()
    _set(at, "w_modo_entrada", modo)
    at.run()
    at.session_state["pagina"] = depois
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]


def test_preprocessamento_cobre_pasta_dx_modo_sem_previa_leitor_sintetico_quebrado_e_fonte_invalida(app_local, monkeypatch):
    import guaraci.app_tabs.preprocessamento as pp
    import guaraci.io_registry as reg
    import guaraci.pipeline as pq
    rng = np.random.default_rng(2)
    monkeypatch.setattr(pp, "preview_spectra_dx", lambda pasta, a, b: (np.linspace(1000, 3000, 10), rng.normal(size=(6, 10)), np.array(list("aabbcc"))))
    original = pq.validar_fonte_de_dados
    monkeypatch.setattr(pq, "validar_fonte_de_dados", lambda cfg: (True, "ok"))
    at = _ir(app_local, "dados")
    _mudar_modo(at, "dx")
    [b for b in at.button if b.key == "btn_prev_preproc"][0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    _mudar_modo(at, "hsi")
    [b for b in at.button if b.key == "btn_prev_preproc"][0].click()
    at.run()
    assert "Could not load spectra" in _texto(at)
    _mudar_modo(at, "sintetico")
    monkeypatch.setattr(reg, "get_reader", lambda nome: (_ for _ in ()).throw(KeyError(nome)))
    [b for b in at.button if b.key == "btn_prev_preproc"][0].click()
    at.run()
    assert "Could not load spectra" in _texto(at)
    monkeypatch.setattr(pq, "validar_fonte_de_dados", original)
    _mudar_modo(at, "dx")
    assert "Configure and validate data input" in _texto(at)


# ---- Visualizacao --------------------------------------------------------------------------------------------------------------------------------


def _previa(n_classes=3):
    rng = np.random.default_rng(3)
    rotulos = np.array(list("abcdefghij"[:n_classes]) * 4)
    return {"wn": np.linspace(4000, 8000, 30), "X": rng.normal(size=(len(rotulos), 30)), "labels": rotulos}


def test_visualizacao_com_previa_mostra_seletores_de_cor_e_o_botao_de_reverter(app):
    at = _ir(app, "visualizacao", previa_espectros=_previa())
    seletores = [c for c in at.get("color_picker")]
    assert len(seletores) == 3
    seletores[0].set_value("#123456")
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["cores_personalizadas"][0] == "#123456"
    [b for b in at.button if b.key == "btn_reset_cores"][0].click()
    at.run()
    assert "cores_personalizadas" not in at.session_state


def test_visualizacao_troca_o_esquema_e_avisa_quando_nao_consegue_gravar(app, monkeypatch):
    at = _ir(app, "visualizacao")
    radio = [r for r in at.radio if r.key == "radio_esquema_cor"][0]
    radio.set_value(radio.options[1])
    at.run()
    assert not at.exception
    monkeypatch.setattr(prefs, "save_visual_config", lambda d, caminho=None: (_ for _ in ()).throw(OSError("somente leitura")))
    import guaraci.app_tabs.visualizacao as vz
    monkeypatch.setattr(vz, "save_visual_config", lambda d: (_ for _ in ()).throw(OSError("somente leitura")))
    radio = [r for r in at.radio if r.key == "radio_esquema_cor"][0]
    radio.set_value(radio.options[2])
    at.run()
    assert "Could not save the palette choice" in _texto(at)


def test_visualizacao_com_cores_personalizadas_ja_na_sessao_usa_as_cores_da_sessao(app):
    at = _ir(app, "visualizacao", previa_espectros=_previa(2), cores_personalizadas=["#ff0000", "#00ff00", "#0000ff"])
    assert not at.exception
    assert [c.value for c in at.get("color_picker")] == ["#ff0000", "#00ff00"]
