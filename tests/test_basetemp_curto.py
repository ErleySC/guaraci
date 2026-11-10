"""A3-05 (vistoria 2026-10): no Windows a sessao troca um basetemp longo por uma pasta curta (tests/_basetemp_curto.py + conftest)."""
from __future__ import annotations

import os
import time

from tests import _basetemp_curto as bc


def test_so_o_windows_troca_e_so_acima_do_limite(monkeypatch):
    # `os.path.abspath("C:/...")` em Linux acrescenta a pasta de trabalho: o resultado dependia do comprimento dela
    # (falhou num clone em pasta funda do WSL). O caminho de teste entra como esta, sem abspath.
    monkeypatch.setattr(bc, "basetemp_previsto", lambda opcao: opcao)
    longo = "C:/" + "p" * 120
    assert bc.precisa_trocar(longo, "nt") and not bc.precisa_trocar(longo, "posix")
    assert not bc.precisa_trocar("C:/gt/x", "nt")


def test_sem_basetemp_dado_mede_o_padrao_do_pytest_na_pasta_temporaria(monkeypatch, tmp_path):
    monkeypatch.setattr(bc.tempfile, "gettempdir", lambda: "C:/" + "t" * 60)
    assert bc.precisa_trocar(None, "nt")
    monkeypatch.setattr(bc.tempfile, "gettempdir", lambda: "C:/t")
    monkeypatch.setattr(bc.getpass, "getuser", lambda: "u")
    assert not bc.precisa_trocar(None, "nt")


def test_raiz_curta_usa_a_unidade_do_sistema():
    assert str(bc.raiz_curta({"SystemDrive": "D:"})).replace("\\", "/") == "D:/gt"


def test_pasta_nova_e_unica_e_so_as_automaticas_antigas_saem(tmp_path):
    alheia = tmp_path / "camada"
    alheia.mkdir()   # pasta de outro script na mesma raiz: nunca e' apagada
    criadas = []
    for _ in range(5):
        criadas.append(bc.criar_basetemp_curto(tmp_path, manter=3, idade_minima_s=0))
        time.sleep(0.02)
    assert len(set(criadas)) == 5
    restantes = sorted(p.name for p in tmp_path.iterdir())
    assert "camada" in restantes
    assert sorted(os.path.basename(c) for c in criadas[-3:]) == sorted(n for n in restantes if n.startswith(bc.PREFIXO))


def test_pasta_recente_de_outra_sessao_nunca_e_apagada(tmp_path):
    """Sessoes de pytest simultaneas: a pasta viva de uma nao pode sumir quando outra comeca (2021 erros de setup sem isto)."""
    criadas = [bc.criar_basetemp_curto(tmp_path, manter=1) for _ in range(6)]
    assert all(os.path.isdir(c) for c in criadas)
    velha = criadas[0]
    os.utime(velha, (time.time() - 3 * 86400,) * 2)
    bc.criar_basetemp_curto(tmp_path, manter=1)
    assert not os.path.exists(velha) and all(os.path.isdir(c) for c in criadas[1:])


def test_a_sessao_roda_com_basetemp_curto_no_windows(tmp_path_factory):
    """Prova de ponta a ponta: no Windows, mesmo com `--basetemp` numa pasta funda, a pasta temporaria da sessao fica abaixo do
    limite. Fora do Windows nada e' trocado (nao ha MAX_PATH)."""
    base = str(tmp_path_factory.getbasetemp())
    if os.environ.get("PYTEST_XDIST_WORKER"):
        base = os.path.dirname(base)   # o trabalhador do xdist recebe <basetemp>/popen-gwN
    if os.name == "nt":
        assert len(base) <= bc.LIMITE, base
    else:
        assert not bc.precisa_trocar(base)
