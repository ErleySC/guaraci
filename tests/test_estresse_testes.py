"""scripts/estresse_testes.py: leitura do resumo do pytest, veredito e tabela (a carga em si roda fora da suite)."""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import estresse_testes as et  # noqa: E402


def _r(estado):
    return {"n": 1, "condicao": "x", "estado": estado, "detalhe": "d"}


def test_ler_resumo_do_pytest():
    assert et.ler_resumo("...\n==== 3 passed in 1.2s ====\n") == (3, 0, "3 passed in 1.2s")
    assert et.ler_resumo("== 2 failed, 5 passed in 9s ==")[:2] == (5, 2)
    assert et.ler_resumo("") == (0, 0, "")


def test_veredito_so_e_zero_com_todas_ok_e_pelo_menos_uma_rodada():
    assert et.veredito([_r("ok"), _r("ok")]) == 0
    assert et.veredito([_r("ok"), _r("falhou")]) == 1
    assert et.veredito([_r("ok"), _r("nao_executada")]) == 1          # nao executada nunca e' aprovacao
    assert et.veredito([]) == 1


def test_tabela_traz_hash_curto_condicao_e_resultado():
    t = et.tabela([_r("ok")], "a" * 40)
    assert "`" + "a" * 12 + "`" in t and "| 1 | x | ok | d |" in t and t.startswith("# Estresse")


def test_subconjunto_existe():
    for a in et.SUBCONJUNTO:
        assert (RAIZ / a).is_file(), a


def test_comando_pytest_nao_usa_cache_e_lista_os_arquivos():
    c = et.comando_pytest("py", ["a.py", "b.py"])
    assert c[:3] == ["py", "-m", "pytest"] and "a.py" in c and "b.py" in c and "no:cacheprovider" in c


def test_descrever_saida_diz_o_codigo_e_o_ultimo_teste_quando_o_processo_morre():
    saida = "tests/test_a.py::test_um PASSED\ntests/test_a.py::test_dois"
    d = et.descrever_saida(3221226525, saida)
    assert "0xc000041d" in d and "tests/test_a.py::test_dois" in d
    assert et.descrever_saida(0, "== 3 passed in 1s ==") == "3 passed in 1s"
    assert "anormal" in et.descrever_saida(-11, "")


def _rod(amb, estado):
    return {"n": 1, "ambiente": amb, "condicao": "x", "estado": estado, "detalhe": "d"}


def test_leitura_linux_que_falha_e_defeito_e_windows_que_falha_e_ambiente():
    assert et.leitura([_rod("linux", "ok"), _rod("windows", "ok")])[0] == 0
    rc, texto = et.leitura([_rod("linux", "falhou"), _rod("windows", "falhou")])
    assert rc == 1 and "DEFEITO" in texto
    rc, texto = et.leitura([_rod("linux", "ok"), _rod("windows", "falhou")])
    assert rc == 0 and "AMBIENTE" in texto and "hipotese" in texto
    assert et.leitura([_rod("linux", "nao_executada")])[0] == 1                 # nao executada nunca e' aprovacao
    assert et.leitura([_rod("windows", "ok")])[0] == 1                           # sem Linux nao ha conclusao possivel
    assert et.leitura([])[0] == 1


def test_comando_do_wsl_leva_hash_limite_saturacao_e_arquivos():
    cmd = et.comando_wsl("a" * 40, "/mnt/c/repo", "/mnt/c/repo/scripts/estresse_wsl.sh", "unlimited", True, ["tests/a.py", "tests/b.py"])
    texto = cmd[-1]
    assert cmd[:4] == ["wsl", "-d", et.DISTRO, "--"] and f"{'a' * 12} '/mnt/c/repo' unlimited 1 tests/a.py tests/b.py" in texto
    assert "unlimited 0 " in et.comando_wsl("b" * 40, "/r", "/s", "unlimited", False, [])[-1] + " "


def test_o_script_do_wsl_satura_a_cpu_so_quando_pedido_e_encerra_os_lacos():
    sh = (RAIZ / "scripts" / "estresse_wsl.sh").read_text(encoding="utf-8")
    assert 'SAT="${4:?saturar}"' in sh and 'if [ "$SAT" = "1" ]' in sh and "kill $PIDS" in sh
