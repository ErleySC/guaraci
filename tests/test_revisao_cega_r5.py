"""Achados da revisao cega de 2026-09-26 (R5-02/03/06/08/09/16, R2-09; o R5-10 saiu com o rotulo vindo do catalogo)."""
import os
import subprocess
import sys

import numpy as np
import pytest

import guaraci.guaraci as g
import guaraci.pipeline as pq
from guaraci.config_io import _validar_pasta_dados
from guaraci.design_tokens import TOKENS, contraste
from guaraci.spectra_preview import preview_spectra_csv

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _csv(tmp_path, nome, texto):
    p = tmp_path / nome
    p.write_text(texto, encoding="utf-8")
    return str(p)


def test_status_csv_lixo_ou_vazio_nao_e_ok(tmp_path):
    for nome, txt in (("lixo.csv", "\x00\x01\x02 nada"), ("vazio.csv", "")):
        ok, msg = _validar_pasta_dados(pq.Config(mode="csv", csv_file=_csv(tmp_path, nome, txt)))
        assert not ok, (nome, msg)


def test_status_csv_sem_coluna_de_classe_diz_qual_falta(tmp_path):
    arq = _csv(tmp_path, "a.csv", "x,1000,1010\nA,1,2\n")
    ok, msg = _validar_pasta_dados(pq.Config(mode="csv", csv_file=arq, class_column="classe"))
    assert not ok and "classe" in msg


def test_previa_ignora_coluna_alvo_numerica(tmp_path):
    arq = _csv(tmp_path, "m.csv", "Class,PeroxideValue,4000,4010,4020\nA,1.5,0.1,0.2,0.3\nB,2.5,0.3,0.2,0.1\n")
    wn, X, labs = preview_spectra_csv(arq, "Class", 3900, 4100)
    assert wn is not None and X.shape == (2, 3) and list(labs) == ["A", "B"]


def test_rodape_de_prontidao_concorda_com_modo_csv_e_sintetico(tmp_path):
    assert g._fonte_de_dados_ok(g.Config(mode="sintetico"))
    arq = _csv(tmp_path, "d.csv", "classe,1000,1010\nA,1,2\n")
    assert g._fonte_de_dados_ok(g.Config(mode="csv", csv_file=arq))
    assert not g._fonte_de_dados_ok(g.Config(mode="csv", csv_file=str(tmp_path / "nao.csv")))


@pytest.mark.parametrize("tema", ["light", "dark"])
def test_texto_secundario_do_app_passa_aa(tema):
    fundo = TOKENS[tema]["bg"]
    assert contraste(TOKENS[tema]["text_muted"], fundo) >= 4.5
    css = open(os.path.join(_RAIZ, "src", "guaraci", "app_web.py"), encoding="utf-8").read()
    assert "color: rgba(128,128,128,1)" not in css and "color:rgba(128,128,128,1)" not in css


def test_saida_em_pipe_nao_tem_sequencias_ansi():
    cod = "import guaraci.guaraci as g; g.console.print('[bold red]ola[/]')"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-c", cod], capture_output=True, text=True, env=env)
    assert "\x1b[" not in r.stdout and "ola" in r.stdout, repr(r.stdout[:80])


def test_cv_com_uma_amostra_por_classe_e_recusada_antes_de_rodar(tmp_path):
    wn = np.linspace(4000, 4110, 60)
    linhas = ["classe," + ",".join(f"{w:.2f}" for w in wn)]
    rng = np.random.default_rng(0)
    for c in "ABCD":
        linhas.append(c + "," + ",".join(f"{v:.4f}" for v in rng.random(60)))
    arq = _csv(tmp_path, "uma.csv", "\n".join(linhas) + "\n")
    cfg = pq.Config(mode="csv", csv_file=arq, class_column="classe", show_plots=False,
                    output_root_folder=str(tmp_path / "o"), n_permutations=3,
                    wn_min=4000, wn_max=4110, frac_holdout=0.0)
    with pytest.raises(ValueError, match="Validacao cruzada impossivel"):
        pq.executar(cfg)


def test_grupo_declarado_com_id_nao_parseavel_e_aviso_de_nao_verificavel():
    """R3-01: antes dizia '[OK] mae_id e sessao coincidem' para qualquer id
    declarado pelo usuario que o software nao sabe interpretar."""
    from guaraci.auditoria_delineamento import check_unidade_independente
    rot = np.array(["A"] * 6 + ["B"] * 6)
    mae = np.array([f"amostra{i}" for i in range(12)])
    r = check_unidade_independente(rot, mae, "mae_id", garantia="medium")
    assert r.severidade == "aviso" and "NAO VERIFICAVEL" in r.mensagem
    assert check_unidade_independente(rot, mae, "mae_id", garantia="high").severidade == "ok"


def test_cli_em_terminal_claro_tem_contraste_aa():
    """M-18c: GUARACI_TERMINAL=claro troca a paleta da CLI por tons escuros."""
    cod = ("import guaraci.guaraci_theme as t; from guaraci.design_tokens import contraste; "
           "print(min(contraste(c, '#FFFFFF') for c in (t.PA, t.PF, t.PS, t.PR, t.PW, t.PM, t.PG)))")
    env = dict(os.environ, GUARACI_TERMINAL="claro")
    r = subprocess.run([sys.executable, "-c", cod], capture_output=True, text=True, env=env)
    assert float(r.stdout.strip()) >= 4.5, r.stderr[-300:]


def test_interrogacao_nas_telas_mostra_ajuda_e_pergunta_de_novo(monkeypatch):
    """R5-17: em B/K/J/U/T/9 o '?' virava caminho/nome de classe."""
    import io as _io
    from rich.console import Console
    respostas = iter(["?", "meu_modelo.joblib"])
    monkeypatch.setattr("builtins.input", lambda *a: next(respostas))
    buf = _io.StringIO()
    monkeypatch.setattr(g, "console", Console(file=buf, width=120))
    monkeypatch.setattr(g, "_AJUDA_TELA", g._AJUDAS_TELAS["3.3"])
    assert g._input("caminho: ") == "meu_modelo.joblib"
    assert "Predi" in buf.getvalue() or "prediction" in buf.getvalue().lower()
    assert set(g._AJUDAS_TELAS) >= {"3.3", "2.8", "2.7", "2.6", "3.5", "2.9"}


def test_sair_com_edicao_nao_salva_pergunta_antes(monkeypatch, tmp_path):
    """R5-22: o aviso vinha so' DEPOIS de sair."""
    cfg = g.Config()
    foto = g._foto_cfg(cfg)
    assert g._confirmar_saida(cfg, foto) is True          # nada mudou: sai direto
    cfg.max_lvs = cfg.max_lvs + 1
    for resp, esperado in (("", False), ("S", True)):
        monkeypatch.setattr("builtins.input", lambda *a, _r=resp: _r)
        assert g._confirmar_saida(cfg, foto) is esperado
    monkeypatch.setattr(g, "_USER_DIR", tmp_path)
    monkeypatch.setattr(g, "_CFG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr("builtins.input", lambda *a: "G")
    assert g._confirmar_saida(cfg, foto) is True
    assert (tmp_path / "config.yaml").is_file()


def test_uma_amostra_nao_culpa_a_faixa_espectral(tmp_path):
    """R2-08: com n=1 a mensagem mandava mexer na faixa espectral."""
    wn = np.linspace(4000, 4100, 60)
    arq = _csv(tmp_path, "um.csv", "classe," + ",".join(f"{w:.1f}" for w in wn)
               + "\nA," + ",".join("0.5" for _ in wn) + "\n")
    cfg = pq.Config(mode="csv", csv_file=arq, class_column="classe", show_plots=False,
                    output_root_folder=str(tmp_path / "o"), wn_min=4000, wn_max=4100)
    with pytest.raises(ValueError, match="pelo menos 3"):
        pq.executar(cfg)


def test_no_color_tira_todo_estilo():
    """R5-16: NO_COLOR mantinha negrito e italico."""
    cod = ("import io, guaraci.guaraci_theme as t; t.console.file = io.StringIO(); "
           "t.console.print('[bold italic red]x[/]'); print(repr(t.console.file.getvalue()))")
    env = dict(os.environ, NO_COLOR="1")
    r = subprocess.run([sys.executable, "-c", cod], capture_output=True, text=True, env=env)
    assert "\x1b" not in r.stdout and "x" in r.stdout, r.stdout


def test_modo_linear_para_leitor_de_tela_sem_caixas():
    """R5-S1: GUARACI_MODO_LINEAR=1 tira os caracteres de desenho de caixa."""
    cod = ("import io, guaraci.guaraci as g; from rich.console import Console; "
           "g._ativar_modo_linear(); b = io.StringIO(); "
           "g.console = Console(file=b, width=80, force_terminal=False); "
           "g._print_status(g.Config(mode='sintetico')); v = b.getvalue(); "
           "print(g.modo_linear(), any(c in v for c in '┌─│╔═'), 'Status' in v)")
    env = dict(os.environ, GUARACI_MODO_LINEAR="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-c", cod], capture_output=True, text=True,
                       env=env, encoding="utf-8")
    assert r.stdout.split() == ["True", "False", "True"], r.stdout + r.stderr[-300:]
