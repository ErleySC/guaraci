"""Fechamento da cobertura: os ultimos ramos de abas web, catalogo, confianca, figuras, relatorios e utilidades."""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import guaraci.catalogo as cat
import guaraci.cli_comandos as cc
import guaraci.cli_logic as cl
import guaraci.confianca as conf
import guaraci.docs_pacote as dp
import guaraci.figuras as fg
import guaraci.preferencias_visuais as prefs
import guaraci.relatorio_blocos as rb
from guaraci import reports
from guaraci.config import Config, NOME_TABELAS
from tests._web_apoio import criar_pasta_de_resultados

_RAIZ = Path(__file__).resolve().parents[1]


def _texto(at):
    itens = [e.value for e in at.markdown] + [e.value for e in at.caption] + [e.value for e in at.info] + [e.value for e in at.warning]
    itens += [e.value for e in at.error] + [e.value for e in at.success] + [e.value for e in at.subheader]
    return "\n".join(str(i) for i in itens)


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("GUARACI_SESSION_DIR", tempfile.mkdtemp(prefix="gs"))
    monkeypatch.setattr(prefs, "VISUAL_PATH", tmp_path / "visual_config.json")
    pasta = tmp_path / "appdir"
    pasta.mkdir()
    shutil.copy(_RAIZ / "app_quimiometria.py", pasta / "app_quimiometria.py")
    shutil.copytree(_RAIZ / "assets", pasta / "assets")
    return str(pasta / "app_quimiometria.py")


def _ir(app, pagina, **estado):
    at = AppTest.from_file(app, default_timeout=120)
    for k, v in estado.items():
        at.session_state[k] = v
    at.session_state["pagina"] = pagina
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _csv(tmp_path):
    rng = np.random.default_rng(0)
    linhas = ["classe," + ",".join(str(w) for w in np.linspace(1000, 3000, 12))]
    for i in range(12):
        linhas.append(("A" if i < 6 else "B") + "," + ",".join(f"{v:.3f}" for v in rng.normal(size=12)))
    caminho = tmp_path / "e.csv"
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return caminho


# ---- aba Dados ---------------------------------------------------------------------------------------------------------------------------------------


def test_dados_modo_multielementar_nao_avisa_que_virou_csv(app, tmp_path, monkeypatch):
    monkeypatch.setenv("GUARACI_LOCAL_MODE", "1")        # no servidor a politica volta o modo para sintetico antes do upload
    at = _ir(app, "dados", cfg_base=Config(mode="multielementar"))
    csv = _csv(tmp_path)
    [u for u in at.file_uploader if u.label.startswith("Drag")][0].upload(csv.name, csv.read_bytes(), "text/csv")
    at.run()
    assert not at.exception and "File received" in _texto(at) and "Mode automatically set" not in _texto(at)


def test_dados_arquivo_de_grupos_repetido_nao_e_gravado_de_novo(app):
    at = _ir(app, "dados")
    grupos = [u for u in at.file_uploader if u.label.startswith("Group file")][0]
    grupos.upload("g.csv", b"a,b\n1,x\n", "text/csv")
    at.run()
    primeiro = at.session_state["cfg_base"].group_file
    at.run()
    assert not at.exception and at.session_state["cfg_base"].group_file == primeiro


@pytest.mark.parametrize("local", [True, False])
def test_dados_com_campos_invalidos_avisa_e_nao_oferece_salvar_nem_baixar(app, monkeypatch, local):
    import guaraci.app_tabs.dados as dados
    original = dados.collect_config
    monkeypatch.setattr(dados, "collect_config", lambda base, valores: (original(base, valores)[0], ["campo_x: valor invalido"]))
    if local:
        monkeypatch.setenv("GUARACI_LOCAL_MODE", "1")
    at = _ir(app, "dados")
    assert "Fields with errors" in _texto(at)
    if local:
        [b for b in at.button if b.key == "btn_salvar_cfg_dados"][0].click()
        at.run()
        assert "Fix the fields before saving" in _texto(at)
    else:
        assert not [d for d in at.get("download_button")]


# ---- aba Inicio, Modelo, Predicao, Pre-processamento, Relatorios, Tecnicas ----------------------------------------------------------------------


def test_inicio_sem_destino_nao_mostra_botao_e_tecnica_sem_selo_de_validacao_externa(app, monkeypatch):
    import guaraci.app_tabs.inicio as inicio
    monkeypatch.setattr(inicio, "next_action", lambda **k: types.SimpleNamespace(severidade="info", titulo="t", detalhe="d", destino=None))
    at = _ir(app, "inicio", previa_dados={"n_espectros": 4, "n_classes": 2})
    texto = " ".join(str(m.value) for m in at.markdown)
    assert not [b for b in at.button if b.key == "btn_acao_sugerida"]
    assert "validada" not in texto and "validação pública" not in texto


def test_inicio_acha_a_pasta_de_relatorios_mesmo_quando_a_execucao_sumiu():
    from guaraci.app_tabs.inicio import _pasta_relatorios
    assert _pasta_relatorios("pasta/que/nao/existe") == "pasta/que/nao/existe"


def test_modelo_com_ram_folgada_nao_avisa_e_erro_repetido_reaproveita_o_codigo(app, monkeypatch):
    import guaraci.pipeline as pq
    hw = dict(pq.hardware_probe())
    hw["ram_livre_gb"] = 64.0
    monkeypatch.setattr(pq, "hardware_probe", lambda: hw)
    monkeypatch.setattr(pq, "executar", lambda cfg: (_ for _ in ()).throw(RuntimeError("quebrou")))
    at = _ir(app, "modelo")
    assert "Free RAM" not in _texto(at)
    [b for b in at.button if b.key == "btn_rodar"][0].click()
    at.run()
    primeiro = at.session_state["_erro_run_id"]
    at.run()
    assert at.session_state["_erro_run_id"] == primeiro


def test_predicao_cobre_metadados_de_outro_tamanho_valor_nao_booleano_e_quantificado_sem_faixa(monkeypatch):
    import guaraci.app_tabs.predicao as aba

    def tela():
        import streamlit as st

        from guaraci.app_tabs import predicao
        from guaraci.design_tokens import tokens
        predicao.render(bool(st.session_state.get("bloqueado", True)), lambda: tokens("light"), lambda s: s)

    meta = pd.DataFrame({"amostra": ["so_uma"]})
    monkeypatch.setattr(aba, "_carregar_csv_predicao", lambda fonte: (np.ones((3, 4)), np.arange(4.0), meta))
    monkeypatch.setattr(aba._me, "predict_portable", lambda m, X, wn: {"classe": ["a", "b", "a"]})
    at = AppTest.from_function(tela, default_timeout=60)
    at.run()
    [u for u in at.file_uploader if u.key == "pred_model_upload_json"][0].upload(
        "m.json", b'{"formato": "guaraci-modelo-portatil", "classes": ["a"], "wn_min": 1, "wn_max": 2}', "application/json")
    [u for u in at.file_uploader if u.key == "pred_csv_upload"][0].upload("n.csv", b"1;2\n", "text/csv")
    at.run()
    [b for b in at.button if b.key == "btn_predizer"][0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    cego = [types.SimpleNamespace(pureza=types.SimpleNamespace(aceito=False), identificacao=types.SimpleNamespace(classe_identificada="x"),
                                  quantificacao=types.SimpleNamespace(teor_estimado=2.0, faixa_decisao=f)) for f in ("nao_detectavel", None)]
    df = pd.DataFrame({"aceito": [True, None], "classe_identificada": ["x", "x"]})
    at = AppTest.from_function(tela, default_timeout=60)
    at.session_state["pred_resultados"] = df
    at.session_state["pred_resultados_cego"] = cego
    at.run()
    assert not at.exception and "no decision range" in _texto(at)


def test_preprocessamento_com_preset_sem_texto_explicativo_nao_mostra_o_quadro(app):
    at = _ir(app, "preprocessamento")
    [s for s in at.selectbox if s.key == "w_pre_processamento"][0].set_value("Personalizado")
    at.run()
    assert not at.exception


def test_relatorios_ignora_arquivo_que_some_ao_medir_a_pasta(monkeypatch, tmp_path):
    from guaraci.app_tabs.relatorios import _tamanho_pasta_mb
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(os.path, "getsize", lambda p: (_ for _ in ()).throw(OSError("sumiu")))
    assert _tamanho_pasta_mb(str(tmp_path)) == 0.0


def test_tecnicas_asca_sem_permutacao_nao_mostra_p_valor(monkeypatch):
    import guaraci.tecnicas_avancadas as ta

    def tela():
        import types as t

        import numpy as np
        import pandas as pd

        from guaraci.app_tabs import tecnicas
        from guaraci.config import Config as C
        classes = np.repeat(["A", "B", "C"], 6)
        pq = t.SimpleNamespace(Config=C, load_data=lambda cfg: (np.arange(4.0), np.ones((18, 4)), classes, None, None,
                                                               pd.DataFrame({"s": np.tile(["x", "y"], 9)})),
                               validate_input=lambda X, wn, r, c, m: (X, wn, r, c, m, {}))
        tecnicas.render(pq, C(), lambda s: s)

    rel = types.SimpleNamespace(group_aware=False, permutacao=None, decomposicao={"efeitos": {"s": {"fracao_ss_total": 0.2}}, "ss_desbalanco": 0.0})
    monkeypatch.setattr(ta, "rodar_asca", lambda *a, **k: rel)
    at = AppTest.from_function(tela, default_timeout=60)
    at.run()
    [b for b in at.button if b.key == "btn_tec_asca_load"][0].click()
    at.run()
    [b for b in at.button if b.key == "btn_tec_asca_run"][0].click()
    at.run()
    assert not at.exception and "p=" not in _texto(at)


def test_validacao_cobre_resumo_ausente_sem_notas_so_benchmark_e_monte_carlo_ilegivel(tmp_path):
    def tela():
        from guaraci.app_logic import list_figures, load_summary
        from guaraci.app_tabs import validacao
        from tests._web_apoio import TOKENS
        validacao.render(lambda s: s, lambda: TOKENS, load_summary, list_figures)

    sem_resumo = criar_pasta_de_resultados(tmp_path / "a", resumo=None, benchmark=False, mc=False)
    at = AppTest.from_function(tela, default_timeout=60)
    at.session_state["ultima_pasta"] = sem_resumo
    at.run()
    assert not at.exception
    sem_notas = criar_pasta_de_resultados(tmp_path / "b", resumo="Acc A: 0.9\nAcc B: 0.5\n", mc=False)
    at = AppTest.from_function(tela, default_timeout=60)
    at.session_state["ultima_pasta"] = sem_notas
    at.run()
    assert not at.exception and "Methodological Notes" not in _texto(at)
    mc_ruim = criar_pasta_de_resultados(tmp_path / "c", benchmark=False)
    (Path(mc_ruim) / "Tabelas" / "monte_carlo_cv.csv").write_bytes(b"\xff\xfe\x00\x80")
    at = AppTest.from_function(tela, default_timeout=60)
    at.session_state["ultima_pasta"] = mc_ruim
    at.run()
    assert "Error reading MC CV CSV" in _texto(at)


def test_visualizacao_completa_as_cores_quando_ha_mais_classes_que_cores_no_esquema(app):
    rng = np.random.default_rng(0)
    rotulos = np.array([f"c{i}" for i in range(10)] * 2)
    previa = {"wn": np.linspace(4000, 8000, 20), "X": rng.normal(size=(20, 20)), "labels": rotulos}
    at = _ir(app, "visualizacao", previa_espectros=previa)
    radio = [r for r in at.radio if r.key == "radio_esquema_cor"][0]
    radio.set_value(radio.options[1])
    at.run()
    assert not at.exception and len(at.get("color_picker")) == 10


# ---- catalogo, confianca, utilidades ------------------------------------------------------------------------------------------------------------


def test_confianca_do_catalogo_de_tipo_fora_da_lista_e_texto_proibido_em_campo():
    ident = next(iter(cat.OPCAO_SELO))
    pagina = cat.confianca("outro", ident, "PT") if cat._selo_da_entrada("outro", ident) else None
    assert pagina is None
    problemas = []
    cat._checar_texto(problemas, "w", "faz", "talvez ajude a decidir")
    assert problemas and "texto proibido" in problemas[0]


def test_pagina_de_confianca_com_licenca_sem_limites_e_sem_reproducao(monkeypatch):
    falsa = {"selo": "x", "estado": "validado", "limites": [],
             "evidencias": [{"tipo": "testes", "rotulo": "r", "ref": "f", "data": "d", "resumo": "s", "ressalva": "", "reproduzir": [],
                             "nao_executado": "", "licenca": "licenca a confirmar"}]}
    monkeypatch.setattr(conf.cat, "confianca", lambda tipo, ident, lang: falsa)
    texto = conf.pagina("opcao", "ddsimca", "PT")
    assert "Licenca: licenca a confirmar" in texto and "Limites:" not in texto and "Reproduzir" not in texto


def test_exemplos_sem_rodar_so_mostram_o_comando_e_a_pasta_so_com_arquivo_nao_tem_dx(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for cenario in ("vip", "monitorar", "transferir"):
        assert cc._cmd_exemplo([cenario, str(tmp_path / cenario)]) == 0
    assert "Para rodar" in capsys.readouterr().out
    (tmp_path / "so_arquivo.txt").write_text("x", encoding="utf-8")
    vazio = tmp_path / "vazio"
    vazio.mkdir()
    (vazio / "nota.txt").write_text("x", encoding="utf-8")
    assert cl.count_dx(str(vazio)) == 0


def test_manual_cuja_ultima_secao_vai_ate_o_fim_do_texto(monkeypatch):
    monkeypatch.setattr(dp, "ler", lambda topico: "## 1.1 Primeira\ntexto\n## 2.1 Segunda\nfim\n")
    assert dp.secao_do_manual("2.1").endswith("fim\n")


def test_bloco_desconhecido_e_ignorado_nos_relatorios_de_texto():
    blocos = [("p", "ok"), ("tipo_novo", "x")]
    assert rb.renderizar_md(blocos).strip() == "ok" and "tipo_novo" not in rb.renderizar_html(blocos)


# ---- figuras e relatorios -----------------------------------------------------------------------------------------------------------------------


def test_cargas_distintas_com_separacao_explicita():
    escolhidas = fg.select_distinct_loadings(np.array([5.0, 4.0, 3.0]), np.array([1000.0, 1050.0, 2000.0]), 3, sep_min_cm=100.0)
    assert list(escolhidas) == [0, 2]


def test_cooman_esconde_os_paineis_que_sobram(tmp_path):
    cfg = Config(output_format="png")
    rot = np.array(list("aaabbbcccddd"))
    rng = np.random.default_rng(0)
    scores = {c: {"Q_norm": rng.uniform(0.1, 3, 12)} for c in "abcd"}
    cores = {c: "#112233" for c in "abcd"}
    fg.resetar_falhas_salvamento()
    fg.fig_cooman_ddsimca(scores, rot, cores, cfg, str(tmp_path), max_pares=4)
    # 4 pares = 2 imagens (3 + 1): nunca mais de MAX_PAINEIS_POR_FIGURA por imagem
    assert list((tmp_path / "Graficos" / "ddsimca").rglob("fig_cooman_ddsimca_01.png"))   # na subpasta da serie
    assert list((tmp_path / "Graficos" / "ddsimca").rglob("fig_cooman_ddsimca_02.png"))
    assert fg.obter_figuras_excedentes() == []


def test_pdf_com_segunda_figura_corrompida_registra_aviso(tmp_path, caplog):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = tmp_path / "run"
    (base / "logs").mkdir(parents=True)
    (base / "logs" / "resumo_modelo.txt").write_text("Balanced Accuracy (CV): 0.9\n", encoding="utf-8")
    (base / "Graficos").mkdir()
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1])
    fig.savefig(str(base / "Graficos" / "fig1_pca_scores.png"))
    plt.close(fig)
    (base / "Graficos" / "fig2_corrompida.png").write_bytes(b"nao e imagem")
    with caplog.at_level(logging.WARNING):
        reports.generate_pdf_report(str(base), {"nome": "P", "autor": "A", "inst": "I", "tipo": "T", "objetivo": "O"}, max_figuras=14)
    assert "figura nao embutida" in caplog.text


def test_excel_sem_nenhuma_tabela_opcional_gera_so_o_resumo(tmp_path):
    base = tmp_path / "run"
    (base / "logs").mkdir(parents=True)
    (base / "logs" / "resumo_modelo.txt").write_text("Balanced Accuracy (CV): 0.9\n", encoding="utf-8")
    (base / NOME_TABELAS).mkdir()
    assert len(reports.generate_excel_report(str(base)).getvalue()) > 0
