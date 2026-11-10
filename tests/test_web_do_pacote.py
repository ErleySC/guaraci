"""`guaraci web` a partir do pacote instalado (V07): o app mora em `guaraci/app_web.py`, achado por
`importlib.resources`, e as opcoes do Streamlit vao na linha de comando (o `.streamlit/config.toml` do
repositorio nao viaja no wheel e so' vale com a pasta atual na raiz dele)."""
from __future__ import annotations

import pathlib
import subprocess

import pytest

import guaraci
import guaraci.guaraci as g
from guaraci.design_tokens import tokens

RAIZ = pathlib.Path(__file__).resolve().parents[1]


def test_o_app_esta_dentro_do_pacote_e_o_da_raiz_e_so_um_ponto_de_entrada():
    pacote = pathlib.Path(guaraci.__file__).resolve().parent
    assert (pacote / "app_web.py").is_file()
    assert (pacote / "assets" / "guaraci_icon.png").is_file()
    stub = (RAIZ / "app_quimiometria.py").read_text(encoding="utf-8")
    assert len(stub.splitlines()) < 40 and "runpy.run_path" in stub


def test_icone_do_pacote_e_o_mesmo_arquivo_da_identidade_visual():
    original = RAIZ / "assets" / "guaraci_icon.png"
    if not original.is_file():
        pytest.skip("sem a pasta assets/ do repositorio (copia do pacote)")
    assert original.read_bytes() == (pathlib.Path(guaraci.__file__).parent / "assets" / "guaraci_icon.png").read_bytes()


def test_caminho_do_app_nao_depende_da_raiz_do_repositorio(tmp_path, monkeypatch):
    """Simula a instalacao por pip: nenhum `app_quimiometria.py` ao lado de `src/`."""
    falso = tmp_path / "site-packages" / "guaraci" / "guaraci.py"
    falso.parent.mkdir(parents=True)
    monkeypatch.setattr(g, "__file__", str(falso))
    assert not (tmp_path / "app_quimiometria.py").exists()
    app = g._caminho_app_web()
    assert app.name == "app_web.py" and app.is_file()


def test_web_chama_o_streamlit_com_o_app_do_pacote_e_opcoes_na_linha_de_comando(monkeypatch):
    chamadas = []
    monkeypatch.setattr(subprocess, "call", lambda cmd, env=None: chamadas.append((cmd, env)) or 0)
    with pytest.raises(SystemExit) as e:
        g._comando_web()
    cmd = chamadas[0][0]
    assert e.value.code == 0 and chamadas[0][1]["GUARACI_LOCAL_MODE"] == "1"
    assert str(g._caminho_app_web()) in cmd
    for opcao in ("--server.address=127.0.0.1", "--server.maxUploadSize=500", "--server.headless=true",
                  "--server.enableXsrfProtection=true", "--browser.gatherUsageStats=false",
                  "--client.showErrorDetails=full", "--theme.base=light",
                  f"--theme.primaryColor={tokens('light')['primary']}",
                  f"--theme.dark.backgroundColor={tokens('dark')['bg']}"):
        assert opcao in cmd, opcao


def test_web_sem_streamlit_explica_o_extra(monkeypatch, capsys):
    import builtins
    real = builtins.__import__

    def sem_streamlit(nome, *a, **k):
        if nome == "streamlit":
            raise ImportError(nome)
        return real(nome, *a, **k)

    def nao_deve_lancar(*a, **k):
        raise AssertionError("lancou o Streamlit mesmo sem ele instalado")

    monkeypatch.setattr(subprocess, "call", nao_deve_lancar)
    monkeypatch.setattr(builtins, "__import__", sem_streamlit)
    with pytest.raises(SystemExit) as e:
        g._comando_web()
    assert e.value.code == 1 and "[web]" in capsys.readouterr().err


def test_o_app_tira_a_pasta_do_pacote_do_caminho_antes_dos_imports():
    """O Streamlit poe a pasta do script em sys.path[0]; na pasta do pacote `guaraci/guaraci.py` sombreia o pacote
    `guaraci` (ImportError circular visto num wheel instalado). O app remove essa pasta antes de importar."""
    fonte = (pathlib.Path(guaraci.__file__).parent / "app_web.py").read_text(encoding="utf-8")
    poda = fonte.index("sys.path[:] =")
    assert poda < fonte.index("from guaraci import catalogo")
    # a mesma poda, aplicada a um sys.path com a pasta do pacote na frente, deixa `guaraci` importavel
    import sys
    pasta = str(pathlib.Path(guaraci.__file__).resolve().parent)
    codigo = ("import sys, os;"
              "sys.path.insert(0, sys.argv[1]);"
              "sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != sys.argv[1]];"
              "from guaraci import config_io")
    r = subprocess.run([sys.executable, "-c", codigo, pasta], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    sem_poda = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); from guaraci import config_io",
                               pasta], capture_output=True, text=True, timeout=120)
    assert sem_poda.returncode != 0
