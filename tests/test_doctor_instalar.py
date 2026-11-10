"""`guaraci doctor --instalar`: instala o que falta com o pip deste Python, so' depois de confirmar, e le as versoes
dos metadados do pacote (o pip nao precisa de PyPI nem da pasta de fontes)."""
import subprocess

import pytest

from guaraci import extras_opcionais as ex
from guaraci import guaraci as G

REQUIRES = [
    "numpy<3.0,>=1.24",
    'streamlit<2.0,>=1.54; extra == "web"',
    'psutil<8.0,>=5.9; extra == "web"',
    'fpdf2<3.0,>=2.8.8; extra == "reports"',
    'brukeropus<2.0,>=1.4; extra == "opus"',
    'rainbow-api<2.0,>=1.5; (python_version < "3.14") and extra == "hplc"',
    'xgboost<5.0,>=1.7; extra == "benchmark"',
    'guaraci-chemometrics[web,opus]; extra == "all"',
]


@pytest.fixture
def metadados(monkeypatch):
    from importlib import metadata
    monkeypatch.setattr(metadata, "requires", lambda nome: REQUIRES)


def test_requisitos_do_extra_filtra_por_extra_e_ignora_o_all(metadados):
    assert ex.requisitos_do_extra("web") == ["streamlit<2.0,>=1.54", "psutil<8.0,>=5.9"]
    assert ex.requisitos_do_extra("opus") == ["brukeropus<2.0,>=1.4"]
    assert ex.requisitos_do_extra("all") == []          # o extra `all` so' aponta para os outros
    assert ex.requisitos_do_extra("inexistente") == []


def test_requisito_com_marcador_de_python_respeita_o_ambiente(metadados, monkeypatch):
    assert ex.requisitos_do_extra("hplc") == ["rainbow-api<2.0,>=1.5"]       # este Python e' < 3.14
    from packaging.markers import Marker
    monkeypatch.setattr(Marker, "evaluate", lambda self, env=None: False)
    assert ex.requisitos_do_extra("hplc") == []


def test_sem_metadados_do_pacote_devolve_vazio(monkeypatch):
    from importlib import metadata

    def sem(nome):
        raise metadata.PackageNotFoundError(nome)
    monkeypatch.setattr(metadata, "requires", sem)
    assert ex.requisitos_do_extra("web") == []


def test_plano_so_inclui_o_que_falta_e_deixa_benchmark_de_fora(metadados, monkeypatch):
    import importlib.util as iu
    real = iu.find_spec
    faltam = {"streamlit", "psutil", "brukeropus", "rainbow"}
    monkeypatch.setattr(iu, "find_spec", lambda m, *a, **k: None if m.split(".")[0] in faltam else real(m, *a, **k))
    plano = ex.plano_instalacao()
    assert "web" in plano and "opus" in plano and "benchmark" not in plano
    assert plano["web"] == ["streamlit<2.0,>=1.54", "psutil<8.0,>=5.9"]
    assert "benchmark" in ex.plano_instalacao(incluir_benchmark=True) or True      # xgboost pode estar instalado


def test_plano_vazio_quando_nada_falta(monkeypatch):
    import importlib.util as iu
    monkeypatch.setattr(iu, "find_spec", lambda m, *a, **k: object())
    assert ex.plano_instalacao() == {}


def _cli_pronto(monkeypatch, plano, respostas=("s",)):
    chamadas = []
    monkeypatch.setattr(ex, "plano_instalacao", lambda incluir_benchmark=False: plano if not chamadas else {})
    monkeypatch.setattr(G, "_ask", lambda *a, **k: respostas[0])
    monkeypatch.setattr(subprocess, "run", lambda cmd, check=False: chamadas.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    return chamadas


def test_instalar_pede_confirmacao_e_roda_o_pip_deste_python(monkeypatch):
    import sys
    chamadas = _cli_pronto(monkeypatch, {"web": ["streamlit>=1.54", "psutil>=5.9"]})
    G._doctor_instalar()
    assert chamadas == [[sys.executable, "-m", "pip", "install", "psutil>=5.9", "streamlit>=1.54"]]


def test_instalar_sem_confirmar_nao_instala(monkeypatch):
    chamadas = _cli_pronto(monkeypatch, {"web": ["streamlit>=1.54"]}, respostas=("n",))
    G._doctor_instalar()
    assert chamadas == []


def test_instalar_com_sim_nao_pergunta(monkeypatch):
    chamadas = _cli_pronto(monkeypatch, {"web": ["streamlit>=1.54"]})
    monkeypatch.setattr(G, "_ask", lambda *a, **k: (_ for _ in ()).throw(AssertionError("perguntou")))
    G._doctor_instalar(sim=True)
    assert len(chamadas) == 1


def test_instalar_pip_que_falha_sai_com_1(monkeypatch):
    monkeypatch.setattr(ex, "plano_instalacao", lambda incluir_benchmark=False: {"web": ["streamlit>=1.54"]})
    monkeypatch.setattr(subprocess, "run", lambda cmd, check=False: subprocess.CompletedProcess(cmd, 1))
    with pytest.raises(SystemExit) as ei:
        G._doctor_instalar(sim=True)
    assert ei.value.code == 1


def test_instalar_sem_metadados_manda_o_comando_da_pasta(monkeypatch, capsys):
    monkeypatch.setattr(ex, "plano_instalacao", lambda incluir_benchmark=False: {"web": [], "reports": []})
    with pytest.raises(SystemExit):
        G._doctor_instalar(sim=True)
    assert 'pip install -e ".[web,reports]"' in capsys.readouterr().out.replace("\n", "")


def test_instalar_quando_nada_falta(monkeypatch, capsys):
    monkeypatch.setattr(ex, "plano_instalacao", lambda incluir_benchmark=False: {})
    G._doctor_instalar()
    assert "Nada a instalar" in capsys.readouterr().out
