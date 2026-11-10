"""scripts/medir_vault.py: contagens do vault sobre um vault sintetico; nunca imprime caminho; sem a variavel nao e' aprovacao."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import medir_vault as mv  # noqa: E402

LIVRO = ("| id | origem | descricao | dono | estado | evidencia | bloqueia |\n|---|---|---|---|---|---|---|\n"
         "| A-1 | x | d | agente | FEITO | e | - |\n| A-2 | x | d | autor | ABERTO | e | - |\n| lixo | x | d | a | OUTRO | e | - |\n")
EMAIL = "pessoa" + "@" + "empresa.net"                    # montado em tempo de execucao: o auditor de exposicao le' este arquivo


def _nota(p: Path, texto: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(texto, encoding="utf-8")


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "v"
    _nota(v / "65-Livro-Razao" / "lr-A-1.md", "---\nid_livro: 'A-1'\n---\nvolta a [[Estado-Atual]]\n")
    _nota(v / "65-Livro-Razao" / "lr-A-2.md", "---\nid_livro: 'A-2'\n---\nvolta a [[Estado-Atual|estado]]\n")
    _nota(v / "Estado-Atual.md", "# Estado\n[[lr-A-1]] [[lr-A-2#secao]]\n")
    _nota(v / "70-Notas-Pessoais" / "diario.md", f"meu contato {EMAIL} e [[inexistente]]\n")
    _nota(v / "99-Arquivo" / "velho.md", "[[nada]]\n")
    livro = tmp_path / "livro.md"
    livro.write_text(LIVRO, encoding="utf-8")
    return v, livro


def test_ids_do_livro_ignora_linha_com_estado_invalido(vault):
    assert mv.ids_do_livro(vault[1]) == {"A-1": "FEITO", "A-2": "ABERTO"}


def test_vault_integro_tem_tudo_zerado_e_as_pastas_do_autor_nao_contam(vault):
    m = mv.medir(vault[0], vault[1], None)
    assert m["notas"] == 3 and m["ids_no_livro"] == 2 and m["notas_do_livro"] == 2
    assert mv.zerado(m), m


def test_id_sem_nota_nota_sem_id_e_link_quebrado_sao_contados(vault):
    v, livro = vault
    (v / "65-Livro-Razao" / "lr-A-2.md").unlink()
    _nota(v / "65-Livro-Razao" / "lr-B-9.md", "sem frontmatter [[nao-existe]]\n")
    _nota(v / "65-Livro-Razao" / "lr-C-1.md", "---\nid_livro: 'C-1'\n---\n")
    m = mv.medir(v, livro, None)
    assert m["ids_sem_nota"] == 1
    assert m["notas_sem_id"] == 2                  # sem frontmatter + id fora do livro
    assert m["wikilinks_quebrados"] >= 2           # a nota B-9 e o link da Estado-Atual para a nota apagada
    assert not mv.zerado(m)


def test_dado_sensivel_em_nota_do_vault_e_contado_mas_nao_em_nota_pessoal(vault):
    v, livro = vault
    assert mv.medir(v, livro, None)["notas_com_dado_sensivel"] == 0
    _nota(v / "Estado-Atual.md", f"contato {EMAIL}\n")
    assert mv.medir(v, livro, None)["notas_com_dado_sensivel"] == 1


@pytest.fixture
def repo_dois_commits(tmp_path):
    """Repositorio git temporario com dois commits em `src/`: o teste nao depende do historico do repositorio real
    (que pode ter um commit so', ou vir de um clone raso na CI)."""
    r = tmp_path / "repo"
    (r / "src").mkdir(parents=True)

    def git(*args):
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                              cwd=str(r), capture_output=True, text=True, check=True).stdout

    git("init", "-q")
    for i in (1, 2):
        (r / "src" / "a.py").write_text(f"x = {i}\n", encoding="utf-8")
        git("add", "-A")
        git("commit", "-q", "-m", f"c{i}")
    commits = git("log", "--format=%H").split()       # mais novo primeiro
    return r, commits


def test_nota_gerada_em_commit_anterior_ao_relevante_e_defasada(repo_dois_commits):
    raiz, commits = repo_dois_commits
    assert mv.cobre_o_commit(commits[0], commits[0], raiz) is True
    assert mv.cobre_o_commit(commits[1], commits[0], raiz) is False
    assert mv.ultimo_commit_relevante(raiz).split()[0] == commits[0]


def test_defasagem_por_carimbo_de_commit(vault, repo_dois_commits, monkeypatch):
    v, livro = vault
    raiz, commits = repo_dois_commits
    original = mv.cobre_o_commit
    monkeypatch.setattr(mv, "cobre_o_commit", lambda carimbo, relevante: original(carimbo, relevante, raiz))
    relevante = f"{commits[0]} 0"
    _nota(v / "Estado-Atual.md", f"---\ncommit: {commits[0]}\n---\n[[lr-A-1]]\n")
    assert mv.medir(v, livro, relevante)["notas_defasadas"] == 0
    _nota(v / "Estado-Atual.md", f"---\ncommit: {commits[1]}\n---\n[[lr-A-1]]\n")
    assert mv.medir(v, livro, relevante)["notas_defasadas"] == 1


def test_sem_a_variavel_o_resultado_e_nao_executado_e_nao_zero(monkeypatch, capsys):
    monkeypatch.delenv("GUARACI_VAULT", raising=False)
    assert mv.main([]) == 3
    assert "nao executado" in capsys.readouterr().out


def test_a_saida_traz_contagens_e_nunca_o_caminho(vault, monkeypatch, capsys):
    v, livro = vault
    monkeypatch.setenv("GUARACI_VAULT", str(v))
    monkeypatch.setenv("GUARACI_LIVRO_RAZAO", str(livro))
    rc = mv.main([])
    saida = capsys.readouterr().out
    assert rc in (0, 1) and "notas 3" in saida
    assert str(v) not in saida and v.name + "/" not in saida


def test_gerar_sem_gerador_configurado_e_nao_executado(vault, monkeypatch, capsys):
    monkeypatch.setenv("GUARACI_VAULT", str(vault[0]))
    monkeypatch.delenv("GUARACI_VAULT_GERADOR", raising=False)
    assert mv.main(["--gerar"]) == 3
    assert "GERADOR" in capsys.readouterr().out


def test_gerar_roda_o_gerador_com_a_pasta_do_vault_e_propaga_a_falha(vault, monkeypatch, tmp_path, capsys):
    v, livro = vault
    ok = tmp_path / "ger" / "scripts" / "g.py"
    ok.parent.mkdir(parents=True)
    ok.write_text("import os, pathlib; pathlib.Path(os.environ['GUARACI_VAULT_DIR'], 'gerada.md').write_text('# g')\n", encoding="utf-8")
    monkeypatch.setenv("GUARACI_VAULT", str(v))
    monkeypatch.setenv("GUARACI_LIVRO_RAZAO", str(livro))
    monkeypatch.setenv("GUARACI_VAULT_GERADOR", str(ok))
    mv.main(["--gerar"])
    assert (v / "gerada.md").is_file()
    ok.write_text("raise SystemExit(5)\n", encoding="utf-8")
    assert mv.main(["--gerar"]) == 1
    assert "gerador falhou" in capsys.readouterr().out
