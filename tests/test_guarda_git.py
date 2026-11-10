"""Guarda de sessao do repositorio real (tests/_guarda_git.py), provada num
repositorio TEMPORARIO -- nunca no repositorio de verdade."""
from __future__ import annotations

import subprocess

from tests import _guarda_git as gg


def _g(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=teste",
                    "-c", "user.email=teste",
                    "-c", "commit.gpgsign=false", *args],
                   check=True, capture_output=True)


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _g(repo, "init", "-q")
    (repo / "a.txt").write_text("1\n", encoding="utf-8")
    _g(repo, "add", "a.txt")
    _g(repo, "commit", "-q", "-m", "inicial")
    return repo


def test_estado_igual_nao_acusa_nada_e_arquivo_nao_rastreado_e_ignorado(tmp_path):
    repo = _repo(tmp_path)
    antes = gg.fotografar(str(repo))
    assert antes and len(antes["HEAD"]) >= 40
    (repo / "novo_nao_rastreado.txt").write_text("x", encoding="utf-8")
    assert gg.comparar(antes, gg.fotografar(str(repo))) == []


def test_edicao_desfeita_stash_e_commit_sao_acusados(tmp_path):
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("trabalho nao commitado\n", encoding="utf-8")
    antes = gg.fotografar(str(repo))
    # um teste "esperto" guarda o trabalho do desenvolvedor no stash
    _g(repo, "stash", "-q")
    probs = gg.comparar(antes, gg.fotografar(str(repo)))
    assert any("stash" in p for p in probs)
    assert any("rastreados" in p for p in probs)
    _g(repo, "stash", "pop", "-q")
    assert gg.comparar(antes, gg.fotografar(str(repo))) == []
    # commit feito por teste muda HEAD
    _g(repo, "commit", "-q", "-am", "commit de teste")
    probs = gg.comparar(antes, gg.fotografar(str(repo)))
    assert any(p.startswith("HEAD") for p in probs)


def test_troca_de_ramo_e_acusada(tmp_path):
    repo = _repo(tmp_path)
    antes = gg.fotografar(str(repo))
    _g(repo, "checkout", "-q", "-b", "outro")
    assert any("ramo" in p for p in gg.comparar(antes, gg.fotografar(str(repo))))


def test_sem_git_na_pasta_ou_sem_executavel_a_guarda_pula_sem_erro(tmp_path, monkeypatch):
    # copia exportada (sem .git): nada a fotografar, nada a acusar
    assert gg.fotografar(str(tmp_path)) is None
    assert gg.comparar(None, None) == [] and gg.comparar({"HEAD": "x"}, None) == []
    # git ausente do sistema
    repo = _repo(tmp_path)

    def sem_git(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(subprocess, "run", sem_git)
    assert gg.fotografar(str(repo)) is None


def test_guarda_e_so_leitura():
    """Nenhum subcomando que escreve no repositorio."""
    from pathlib import Path
    fonte = Path(gg.__file__).read_text(encoding="utf-8")
    for proibido in ('"add"', '"commit"', '"checkout"', '"reset"', '"restore"',
                     '"stash", "push"', '"stash", "pop"', '"clean"'):
        assert proibido not in fonte, proibido
    assert '"--no-optional-locks"' in fonte
