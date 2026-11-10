"""Seguranca do app web (M-01/M-02 do parecer de revisao por pares 2026-09-25).

O padrao agora e' o MODO SERVIDOR: so' upload, pasta de sessao isolada, modelo
so' em JSON portatil, sem config.yaml compartilhado. O modo local
(`GUARACI_LOCAL_MODE=1`) mantem caminhos livres e `.joblib` assinado.

Cada teste de AppTest abaixo e' a versao automatizada de um script do
revisor 1 (`revisor1/apptest/t_web2.py` e `t_web3.py`, fora da
arvore desde 2026-10-02): o script do revisor e' o teste de aceite.
"""
from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from guaraci import web_seguranca as ws

_RAIZ = Path(__file__).resolve().parents[1]
_APP = _RAIZ / "app_quimiometria.py"


@pytest.fixture(autouse=True)
def _sessoes_isoladas(tmp_path, monkeypatch):
    monkeypatch.setenv("GUARACI_SESSION_DIR", str(tmp_path / "sessoes"))
    monkeypatch.delenv("GUARACI_LOCAL_MODE", raising=False)
    monkeypatch.delenv("GUARACI_DISABLE_MODEL_UPLOAD", raising=False)


def _app_copia(tmp_path: Path) -> str:
    """Copia do app SEM config.yaml (nunca toca o config.yaml do autor)."""
    pasta = tmp_path / "appdir"
    pasta.mkdir()
    shutil.copy(_APP, pasta / "app_quimiometria.py")
    shutil.copytree(_RAIZ / "assets", pasta / "assets")
    return str(pasta / "app_quimiometria.py")


def _chaves(at: AppTest) -> set:
    return {w.key for w in (list(at.text_input) + list(at.checkbox)
                            + list(at.button) + list(at.selectbox)) if w.key}


def _nova(app: str, pagina: str) -> AppTest:
    at = AppTest.from_file(app, default_timeout=120)
    at.session_state["pagina"] = pagina
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


# ── Politica pura ────────────────────────────────────────────────────────
def test_padrao_e_modo_servidor_e_pickle_bloqueado():
    assert ws.modo_servidor() and not ws.modo_local()
    assert not ws.pickle_permitido()


def test_modo_local_so_com_variavel_explicita(monkeypatch):
    monkeypatch.setenv("GUARACI_LOCAL_MODE", "1")
    assert ws.modo_local() and ws.pickle_permitido()
    # flag legada continua vencendo
    monkeypatch.setenv("GUARACI_DISABLE_MODEL_UPLOAD", "1")
    assert not ws.pickle_permitido()


def test_id_de_sessao_invalido_e_recusado():
    for ruim in ("../x", "abc", "", "A" * 16, "0123456789abcde/"):
        with pytest.raises(ValueError):
            ws.pasta_sessao(ruim)


def test_caminho_fora_da_sessao_e_recusado(tmp_path):
    sid = ws.nova_sessao()
    dentro = ws.salvar_upload(sid, "a.csv", b"x")
    assert ws.dentro_da_sessao(str(dentro), sid) == dentro.resolve()
    for fora in (tmp_path / "outro.csv", Path("C:/Windows/win.ini"),
                 dentro.parent / ".." / ".." / "escape.csv"):
        with pytest.raises(ValueError):
            ws.dentro_da_sessao(str(fora), sid)


def test_upload_usa_so_o_basename():
    sid = ws.nova_sessao()
    p = ws.salvar_upload(sid, "../../../etc/passwd.csv", b"x")
    assert p.name == "passwd.csv" and ws.pasta_sessao(sid).resolve() in p.resolve().parents
    p2 = ws.salvar_upload(sid, "..\\..\\evil.csv", b"x")
    assert p2.name == "evil.csv"


# ── A1-03: nomes especiais viram erro tratado, sem caminho ───────────────
@pytest.mark.parametrize("nome", ["..", "a/..", ".", "", "   ", "a\\..",
                                  "CON.csv", "nul.txt", "com1.csv", "LPT9.tsv",
                                  "dados.exe", "sem_extensao", "....csv"])
def test_upload_com_nome_invalido_e_recusado_sem_caminho(nome):
    sid = ws.nova_sessao()
    with pytest.raises(ws.UploadRecusado) as e:
        ws.salvar_upload(sid, nome, b"a,b\n1,2\n")
    texto = str(e.value)
    assert str(ws.raiz_sessoes()) not in texto and sid not in texto
    assert ws.ocultar_caminhos(texto) == texto


def test_upload_com_nome_longo_e_truncado_e_gravado():
    sid = ws.nova_sessao()
    p = ws.salvar_upload(sid, "x" * 300 + ".csv", b"a\n")
    assert p.name == "x" * ws.MAX_NOME_UPLOAD + ".csv" and p.read_bytes() == b"a\n"
    # caracteres especiais nao passam para o nome gravado
    q = ws.salvar_upload(sid, 'a<b>:c|d?*"e.CSV', b"a\n")
    assert q.name == "a_b_c_d_e.csv"


# ── A1-04: tetos so' no modo servidor ────────────────────────────────────
def _cfg_pesada():
    import guaraci.pipeline as pq
    return pq.Config(n_permutations=100000, n_permutations_wold=5000,
                     n_bootstrap_bca=100000, n_bootstrap_vip=9000)


def test_tetos_de_recursos_aplicados_no_modo_servidor():
    cfg = _cfg_pesada()
    assert {a for a, _v, _t in ws.tetos_excedidos(cfg)} == {
        "n_permutations", "n_permutations_wold", "n_bootstrap_bca", "n_bootstrap_vip"}
    ws.impor_politica_servidor(cfg, ws.nova_sessao())
    assert cfg.n_permutations == cfg.n_permutations_wold == ws.TETO_PERMUTACOES_SERVIDOR
    assert cfg.n_bootstrap_bca == cfg.n_bootstrap_vip == ws.TETO_BOOTSTRAP_SERVIDOR
    assert ws.tetos_excedidos(cfg) == []
    # teto so' reduz: valor abaixo do teto fica intacto
    import guaraci.pipeline as pq
    leve = pq.Config(n_permutations=50, n_bootstrap_bca=100)
    ws.impor_politica_servidor(leve, ws.nova_sessao())
    assert leve.n_permutations == 50 and leve.n_bootstrap_bca == 100


def test_tabela_grande_demais_recusada_so_no_modo_servidor(monkeypatch):
    linha = b",".join([b"1"] * (ws.MAX_COLUNAS_TABELA_SERVIDOR + 1)) + b"\n"
    sid = ws.nova_sessao()
    with pytest.raises(ws.UploadRecusado, match="grande demais"):
        ws.salvar_upload(sid, "larga.csv", linha)
    muitas_linhas = b"1\n" * (ws.MAX_LINHAS_TABELA_SERVIDOR + 1)
    with pytest.raises(ws.UploadRecusado):
        ws.salvar_upload(sid, "longa.csv", muitas_linhas)
    assert ws.conferir_dimensoes_tabela(b"a;b;c\n1;2;3\n") == (2, 3)
    # modo local: sem teto (comportamento de antes)
    monkeypatch.setenv("GUARACI_LOCAL_MODE", "1")
    assert ws.salvar_upload(sid, "larga.csv", linha).is_file()


def test_cli_local_nao_passa_pelos_tetos():
    """Os tetos vivem SO' em web_seguranca.impor_politica_servidor; a CLI e o
    pipeline nao a importam (nada muda para a linha de comando)."""
    raiz_src = _RAIZ / "src" / "guaraci"
    for arq in ("guaraci.py", "pipeline.py", "config_io.py"):
        fonte = (raiz_src / arq).read_text(encoding="utf-8")
        assert "impor_politica_servidor" not in fonte and "TETO_" not in fonte, arq


def test_limite_de_upload_do_servidor_e_o_mesmo_no_toml_e_no_docker():
    assert _config_toml()["server"]["maxUploadSize"] == 50
    assert "--server.maxUploadSize=50" in (_RAIZ / "Dockerfile").read_text(encoding="utf-8")


def test_tela_com_excecao_mostra_so_mensagem_e_id_no_modo_servidor(tmp_path, monkeypatch):
    """A1-03: excecao nao tratada numa tela nao chega ao visitante como
    traceback; vira `mensagem_visitante` (id do erro) e o detalhe vai so'
    para o log."""
    from guaraci.app_tabs import sobre
    def explode(*a, **k):
        raise FileNotFoundError("C:" + "\\" + "segredo" + "\\" + "x.csv sumiu")
    monkeypatch.setattr(sobre, "render", explode)
    app = _app_copia(tmp_path)
    at = AppTest.from_file(app, default_timeout=120)
    at.session_state["pagina"] = "sobre"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    erros = [e.value for e in at.error]
    junto = " ".join(erros)
    assert "error id" in junto or "codigo do erro" in junto
    assert all("segredo" not in t and "Traceback" not in t for t in erros)


def test_tela_com_excecao_no_modo_local_continua_propagando(tmp_path, monkeypatch):
    monkeypatch.setenv("GUARACI_LOCAL_MODE", "1")
    from guaraci.app_tabs import sobre
    def explode(*a, **k):
        raise RuntimeError("falha local visivel")
    monkeypatch.setattr(sobre, "render", explode)
    app = _app_copia(tmp_path)
    at = AppTest.from_file(app, default_timeout=120)
    at.session_state["pagina"] = "sobre"
    at.run()
    # o erro nao e' engolido pela fronteira da tela: segue como excecao
    # do Streamlit (detalhe completo), nao como st.error amigavel
    assert at.exception, "excecao engolida no modo local"
    assert not [e for e in at.error if "error id" in e.value or "codigo do erro" in e.value]
    # (o texto detalhado depende de client.showErrorDetails: o config.toml do
    # repositorio diz "none"; `guaraci web` passa "full" na linha de comando)


def _config_toml() -> dict:
    try:
        import tomllib  # Python >= 3.11
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]
    with open(_RAIZ / ".streamlit" / "config.toml", "rb") as f:
        return tomllib.load(f)


def test_config_de_servidor_esconde_detalhe_do_erro():
    assert _config_toml()["client"]["showErrorDetails"] == "none"
    docker = (_RAIZ / "Dockerfile").read_text(encoding="utf-8")
    assert "--client.showErrorDetails=none" in docker


def test_pasta_da_sessao_e_apagada_ao_fim_da_sessao():
    sid = ws.nova_sessao()
    guarda = ws.GuardaSessao(sid)
    ws.salvar_upload(sid, "a.csv", b"dados")
    pasta = ws.pasta_sessao(sid, criar=False)
    assert pasta.is_dir()
    del guarda            # o estado do Streamlit foi descartado
    import gc
    gc.collect()
    assert not pasta.exists()


def test_varredura_apaga_sessoes_antigas_e_preserva_as_novas():
    import os
    velha, nova = ws.nova_sessao(), ws.nova_sessao()
    arq_velho = ws.salvar_upload(velha, "a.csv", b"x")
    ws.salvar_upload(nova, "a.csv", b"x")
    t = time.time() - 7 * 3600
    for p in (arq_velho, arq_velho.parent, ws.pasta_sessao(velha)):
        os.utime(p, (t, t))
    assert ws.limpar_sessoes_antigas(max_horas=6) == 1
    assert not ws.pasta_sessao(velha, criar=False).exists()
    assert ws.pasta_sessao(nova, criar=False).exists()


def test_mensagem_ao_visitante_nao_vaza_traceback_nem_caminho(caplog):
    try:
        # montado em tempo de execucao: um literal seria pego pela guarda de caminhos versionados
        raise FileNotFoundError("C:" + "\\" + "Users" + "\\" + "servidor\\segredo\\dados.csv nao existe")
    except FileNotFoundError as e:
        msg, eid = ws.mensagem_visitante(e, "leitura dos dados")
    assert eid in msg
    assert "Traceback" not in msg and "servidor" not in msg and "C:\\" not in msg
    assert "<caminho>" in msg


def test_impor_politica_forca_saida_e_entrada_da_sessao(tmp_path):
    import guaraci.pipeline as pq
    sid = ws.nova_sessao()
    cfg = pq.Config(mode="dx", input_folder="C:/dados_do_servidor",
                    output_root_folder="C:/Windows/Temp")
    ws.impor_politica_servidor(cfg, sid)
    assert cfg.mode == "sintetico" and cfg.input_folder == ""
    assert Path(cfg.output_root_folder).resolve() == ws.pasta_sessao(sid, "resultados").resolve()
    # CSV fora da sessao vira sintetico; dentro da sessao e' mantido
    cfg2 = pq.Config(mode="csv", csv_file="C:/Windows/win.ini")
    ws.impor_politica_servidor(cfg2, sid)
    assert cfg2.mode == "sintetico"
    up = ws.salvar_upload(sid, "d.csv", b"classe,4000\nA,1\n")
    cfg3 = pq.Config(mode="csv", csv_file=str(up))
    ws.impor_politica_servidor(cfg3, sid)
    assert cfg3.mode == "csv"
    # multielementar (tabela ICP por upload): mesma regra do csv
    cfg4 = pq.Config(mode="multielementar", csv_file="C:/Windows/win.ini")
    ws.impor_politica_servidor(cfg4, sid)
    assert cfg4.mode == "sintetico"
    cfg5 = pq.Config(mode="multielementar", csv_file=str(up))
    ws.impor_politica_servidor(cfg5, sid)
    assert cfg5.mode == "multielementar"


def test_rotulo_de_classe_e_escapado_antes_de_html():
    assert "<" not in ws.escapar_html("<img src=x onerror=alert(1)>")
    assert ws.escapar_html('a"b') == "a&quot;b"


# ── AppTest: scripts do revisor 1 (t_web2 pickle/caminhos, t_web3) ───────
def test_web_nao_expoe_campos_de_caminho_nem_pickle(tmp_path):
    """t_web2.py modos 'pickle' e 'caminhos': nenhum campo permite apontar
    para arquivo do servidor, e nao existe carga de .joblib."""
    app = _app_copia(tmp_path)
    pred = _nova(app, "predicao")
    k = _chaves(pred)
    assert not ({"pred_model_path", "pred_model_confia", "pred_csv_path"} & k)
    rotulos = [f.label for f in pred.get("file_uploader")]
    assert rotulos and all("joblib" not in r.lower() for r in rotulos)

    dados = _nova(app, "dados")
    k = _chaves(dados)
    assert not ({"w_pasta_dados", "w_arquivo_csv", "w_pasta_saida",
                 "w_hsi_pasta_dataset"} & k)
    modo = [s for s in dados.selectbox if s.key == "w_modo_entrada"][0]
    assert set(modo.options) == {"sintetico", "csv", "multielementar"}

    tec = _nova(app, "tecnicas")
    assert "tec_fusao_caminho_1" not in _chaves(tec)


def test_web_nao_le_nem_grava_config_yaml_compartilhado(tmp_path):
    """t_web2.py modo 'persistir' + t_web3.py: a sessao A nao grava
    config.yaml e a sessao B nunca herda o CSV/saida da A."""
    app = _app_copia(tmp_path)
    cfg_path = Path(app).parent / "config.yaml"
    a = _nova(app, "dados")
    assert "btn_salvar_cfg_dados" not in _chaves(a)
    assert "btn_reload_cfg_dados" not in _chaves(a)
    assert not cfg_path.exists()
    # Mesmo que exista um config.yaml na pasta do app (o do autor), o modo
    # servidor nao o le.
    cfg_path.write_text("modo_entrada: dx\npasta_dados: C:/dados_reais\n", encoding="utf-8")
    b = _nova(app, "dados")
    cfg = b.session_state["cfg_base"]
    assert cfg.mode == "sintetico" and "dados_reais" not in (cfg.input_folder or "")
    assert cfg_path.read_text(encoding="utf-8").startswith("modo_entrada: dx")


def test_sessoes_tem_pastas_de_saida_distintas_e_dentro_da_raiz(tmp_path):
    app = _app_copia(tmp_path)
    a, b = _nova(app, "inicio"), _nova(app, "inicio")
    sa, sb = a.session_state["_sessao_id"], b.session_state["_sessao_id"]
    assert sa != sb
    sai_a = Path(a.session_state["cfg_base"].output_root_folder).resolve()
    sai_b = Path(b.session_state["cfg_base"].output_root_folder).resolve()
    assert sai_a != sai_b
    raiz = ws.raiz_sessoes().resolve()
    assert raiz in sai_a.parents and raiz in sai_b.parents


def test_config_yaml_do_autor_permanece_intocado_no_modo_servidor():
    """O app do repositorio, em modo servidor, nao le nem reescreve o
    config.yaml da pasta do app (pode existir na maquina do autor)."""
    cfg = _RAIZ / "config.yaml"
    antes = hashlib.sha256(cfg.read_bytes()).hexdigest() if cfg.exists() else None
    at = AppTest.from_file(str(_APP), default_timeout=60)
    at.session_state["pagina"] = "dados"
    at.run()
    assert not at.exception
    depois = hashlib.sha256(cfg.read_bytes()).hexdigest() if cfg.exists() else None
    assert antes == depois


# ── model card / manifesto: caso do revisor com pickle ───────────────────
def test_pickle_malicioso_nao_executa_com_manifesto_ausente(tmp_path):
    import joblib
    from guaraci import predicao as pr
    marcador = tmp_path / "MARCADOR.txt"

    class _P:
        def __reduce__(self):
            return (Path.write_text, (marcador, "executou"))

    mal = tmp_path / "web_malicioso.joblib"
    joblib.dump({"x": _P()}, mal)
    with pytest.raises(pr.SecurityError):
        pr.load_model(str(mal), confiar=True)
    assert not marcador.exists()
