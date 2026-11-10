"""m-20 do parecer de 2026-09-25: usar so' `guaraci.dados_io.load_data` (sem
importar `guaraci.pipeline`) ja' encontra os leitores avancados."""
import subprocess
import sys


def test_load_data_registra_leitores_avancados_sem_importar_pipeline():
    codigo = (
        "import sys\n"
        "import guaraci.dados_io as d\n"
        "from guaraci.config import Config\n"
        "cfg = Config(mode='opus', input_folder='/nao/existe')\n"
        "try:\n"
        "    d.load_data(cfg)\n"
        "except Exception as e:\n"
        "    assert 'guaraci.pipeline' not in sys.modules\n"
        "    assert 'nao registrado' not in str(e).lower() and 'desconhecido' not in str(e).lower(), e\n"
        "print('ok')\n")
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-400:]
