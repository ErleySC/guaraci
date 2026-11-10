"""Guarda estatica: nenhum teste usa RAM livre, CPU, relogio, caminho de maquina, ambiente ou rede sem uma decisao registrada."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import varrer_dependencias_de_maquina as vd  # noqa: E402


def test_todo_uso_tem_decisao_nenhuma_esta_obsoleta_e_a_tabela_esta_em_dia():
    r = subprocess.run([sys.executable, str(RAIZ / "scripts" / "varrer_dependencias_de_maquina.py"), "--verificar"],
                       cwd=RAIZ, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout


@pytest.mark.parametrize("categoria, linha", [
    ("ram", "livre = psutil.virtual_memory().available"),
    ("cpu", "n = os.cpu_count()"),
    ("relogio", "t0 = time.perf_counter()"),
    ("relogio", "time.sleep(0.5)"),
    ("caminho", 'p = "C:/Users/alguem/x"'),
    ("caminho", "h = Path.home()"),
    ("ambiente", 'v = os.environ.get("X")'),
    ("rede", "import socket"),
])
def test_cada_categoria_e_detectada_num_teste_novo(tmp_path, categoria, linha):
    """Mutacao da guarda: um teste NOVO com a dependencia aparece como `sem_decisao` (se a regex perdesse o padrao, este teste cairia)."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "validacao" / "tests").mkdir(parents=True)
    (tmp_path / "tests" / "test_novo.py").write_text(f"def test_x():\n    {linha}\n", encoding="utf-8")
    a = vd.achados(tmp_path)
    assert ("tests/test_novo.py", categoria) in a
    assert any("tests/test_novo.py" in x for x in vd.comparar(a)["sem_decisao"])


def test_linha_de_comentario_nao_conta(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "validacao" / "tests").mkdir(parents=True)
    (tmp_path / "tests" / "test_c.py").write_text("# usa psutil e time.sleep so' no comentario\n", encoding="utf-8")
    assert vd.achados(tmp_path) == {}


def test_decisao_sem_uso_correspondente_e_obsoleta():
    d = {("tests/test_que_nao_existe.py", "ram"): ("injetado", "x")}
    assert vd.comparar({}, d)["decisao_obsoleta"] == ["tests/test_que_nao_existe.py [ram]"]


def test_validacao_local_fora_do_repositorio_nao_e_varrida_nem_tem_regra_implicita(tmp_path):
    """A validacao do mantenedor saiu do repositorio: mesmo se posta em `validacao/` localmente, a guarda so' olha `tests/`
    e nenhum caminho ganha decisao sem registro explicito."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "validacao" / "tests").mkdir(parents=True)
    (tmp_path / "validacao" / "tests" / "test_v.py").write_text("import os\nv = os.environ.get('X')\n", encoding="utf-8")
    assert vd.achados(tmp_path) == {}
    assert vd.decisao(("validacao/tests/test_qualquer.py", "ambiente")) is None
    assert vd.decisao(("tests/test_qualquer.py", "ambiente")) is None


def test_todo_corrigido_tem_motivo_que_diz_o_que_mudou():
    for (_arq, _cat), (estado, motivo) in vd.DECISOES.items():
        assert estado in ("injetado", "bloqueado", "isolado", "corrigido", "legitimo")
        assert len(motivo) > 20
