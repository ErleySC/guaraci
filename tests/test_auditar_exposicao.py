"""`scripts/auditar_exposicao.py`: guarda generica (na CI) + guarda local por lista do autor; nunca mostra o trecho."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
_B = chr(92)                                   # barra invertida


def _win(usuario, resto="x"):
    """Caminho de usuario do Windows montado em tempo de execucao (o arquivo nao carrega nenhum caminho literal)."""
    return "C:" + _B + "Users" + _B + usuario + _B + resto


def _posix(raiz, usuario, resto="x/"):
    return "/" + raiz + "/" + usuario + "/" + resto


@pytest.fixture(scope="module")
def aud():
    spec = importlib.util.spec_from_file_location("auditar_exposicao", RAIZ / "scripts" / "auditar_exposicao.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _repo(tmp_path, arquivos: dict, commits: list = ()) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    for nome, texto in arquivos.items():
        p = tmp_path / nome
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(texto, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    if commits:
        subprocess.run(["git", "commit", "-q", "-m", "c"], cwd=tmp_path, check=True)
    return tmp_path


def _regras(aud, linha, locais=()):
    return aud.achar_na_linha(linha, aud._regexes(), list(locais))


@pytest.mark.parametrize("linha, regra", [
    ("contato: fulano.silva@instituicao.edu.br", "email"),
    ("tel (91) 98888-7777", "telefone"),
    ("tel +55 91 98888 7777", "telefone"),
    (_win("mariadasilva", "Documents" + _B + "x"), "caminho_usuario_windows"),
    (_win("mariadasilva", "x").replace(_B, _B + _B), "caminho_usuario_windows"),
    (_posix("home", "mariadasilva", "projeto/x"), "caminho_usuario_posix"),
    (_posix("Users", "mariadasilva", "x/"), "caminho_usuario_posix"),
    ("caminho One" + "Drive/Documentos", "onedrive"),
    ("token ghp_" + "a" * 30, "chave_ou_token"),
    ("AKIA" + "A" * 16, "chave_ou_token"),
    ("-----BEGIN RSA PRIVATE KEY-----", "chave_ou_token"),
    ("http://192.168.0.10:8080/x", "url_interna"),
    ("https://servidor.intranet/x", "url_interna"),
    ("Rascunho de e-mail de contato com a instituicao", "rascunho_de_contato"),
    ("minuta da carta ao parceiro", "rascunho_de_contato"),
])
def test_cada_regra_generica_pega_o_seu_caso(aud, linha, regra):
    # a linha de e-mail de exemplo precisa ser um e-mail real de usuario, nao um dos permitidos
    assert regra in _regras(aud, linha), (linha, regra)


@pytest.mark.parametrize("linha", [
    "Co-Authored-By: x <noreply@github.com>", "user@example.com",
    _win("<usuario>"), _win("alguem"), _posix("home", "usuario"), _posix("home", "runner", "work/x/"),
    "http://github.com/x", "o numero 12345678 e 2026-10-07", "versao 1.2.3", "rascunho do texto do manual",
    "http://127.0.0.1:8501/health", "http://localhost:8501/x",
    "pastas dentro do One" + "Drive seguram arquivos", "id 2af35fd2-d3e1-4e27-9712-6593b42c39b6",
])
def test_casos_legitimos_nao_sao_pegos(aud, linha):
    assert _regras(aud, linha) == [], linha


@pytest.mark.parametrize("arquivo", ["pyproject.toml", "CITATION.cff", "SECURITY.md", "src/guaraci/dados_docs/seguranca.md", "README.md"])
def test_nenhum_arquivo_tem_excecao_para_email_de_contato(aud, arquivo):
    """Decisao do autor (lote 10): o projeto nao publica e-mail em arquivo nenhum, nem nos de metadado."""
    assert aud.EMAIL_PERMITIDO_EM == ()
    linha = "contato: fulano.silva@instituicao.edu.br"
    assert aud.achar_na_linha(linha, aud._regexes(), [], arquivo) == ["email"]


def test_termo_local_casa_sem_acento_e_sem_caixa_e_nunca_vem_da_regra_generica(aud):
    locais = [aud._norm("Fulano de Tal")]
    assert _regras(aud, "obrigado ao FULANO DE TAL pela ajuda", locais) == ["termo_local"]
    assert _regras(aud, "obrigado a todos", locais) == []


def test_lista_local_e_criada_vazia_e_nunca_escrita_com_termo(aud, tmp_path):
    arq = tmp_path / "x" / "termos.txt"
    assert aud.termos_locais(arq) == []
    assert arq.exists() and "NAO e' versionado" in arq.read_text(encoding="utf-8")
    arq.write_text("# comentario\n\nTermo Um\n  outro termo  \n", encoding="utf-8")
    assert aud.termos_locais(arq) == ["termo um", "outro termo"]


def test_varrer_arvore_acha_e_ignora_o_proprio_auditor(aud, tmp_path):
    raiz = _repo(tmp_path, {"a.md": "ver " + _win("mariadasilva") + "\n", "b.md": "texto neutro\n",
                            "scripts/auditar_exposicao.py": _win("mariadasilva") + "\n"})
    achados = aud.varrer_arvore(raiz, locais=[])
    assert set(achados) == {"a.md"} and achados["a.md"] == [(1, "caminho_usuario_windows")]


def test_historico_conta_linhas_adicionadas_e_removidas_sem_guardar_o_trecho(aud, tmp_path):
    raiz = _repo(tmp_path, {"a.md": "limpo\n"}, commits=[1])
    (raiz / "a.md").write_text("contato fulano.x@inst.edu.br\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-qam", "d"], cwd=raiz, check=True)
    (raiz / "a.md").write_text("limpo de novo\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-qam", "e"], cwd=raiz, check=True)
    assert aud.varrer_arvore(raiz, locais=[]) == {}               # a arvore final esta limpa...
    cont = aud.varrer_historico("HEAD~2..HEAD", raiz, locais=[], incluir_removidas=True)    # ...o historico do intervalo nao
    assert cont == {"email": 2}                                    # +linha e -linha
    assert aud.varrer_historico("HEAD~2..HEAD", raiz, locais=[]) == {"email": 1}               # por padrao so' o adicionado


def test_historico_atribui_a_linha_removida_ao_arquivo_apagado(aud, tmp_path):
    """Linha de arquivo APAGADO tem o nome do arquivo no cabecalho `--- a/` (o `+++` e' /dev/null): a regra do e-mail por arquivo vale."""
    raiz = _repo(tmp_path, {"pyproject.toml": "contato fulano.x@inst.edu.br\n", "b.md": "texto\n"}, commits=[1])
    subprocess.run(["git", "rm", "-q", "pyproject.toml"], cwd=raiz, check=True)
    subprocess.run(["git", "commit", "-qm", "apaga"], cwd=raiz, check=True)
    assert aud.varrer_historico("HEAD~1..HEAD", raiz, locais=[], incluir_removidas=True) == {"email": 1}     # nenhum arquivo e' isento


def test_main_nunca_imprime_o_trecho_e_sai_1(aud, tmp_path, monkeypatch, capsys):
    raiz = _repo(tmp_path, {"a.md": "contato segredo.pessoa@inst.edu.br\n"})
    monkeypatch.setattr(aud, "RAIZ", raiz)
    assert aud.main(["--local", str(tmp_path / "termos.txt")]) == 1
    saida = capsys.readouterr().out
    assert "a.md" in saida and "[email]" in saida and "segredo" not in saida and "inst.edu" not in saida


def test_main_com_arvore_limpa_sai_0(aud, tmp_path, monkeypatch, capsys):
    raiz = _repo(tmp_path, {"a.md": "texto neutro\n"})
    monkeypatch.setattr(aud, "RAIZ", raiz)
    assert aud.main(["--local", str(tmp_path / "termos.txt")]) == 0
    assert "0 ocorrencia(s)" in capsys.readouterr().out


def test_lista_local_do_autor_vira_regra(aud, tmp_path):
    arq = tmp_path / "local.txt"
    arq.write_text("palavra_que_so_o_autor_conhece_xyz\n", encoding="utf-8")
    achados = aud.achar_na_linha("a palavra_que_so_o_autor_conhece_xyz aqui", aud._regexes(), aud.termos_locais(arq))
    assert achados == ["termo_local"]


def test_main_avisa_que_r5_nao_foi_executado_quando_a_lista_local_esta_vazia(aud, tmp_path, capsys):
    vazia = tmp_path / "termos.txt"
    vazia.write_text("# so comentario\n", encoding="utf-8")
    aud.main(["--local", str(vazia)])
    assert "R5 (lista local de termos) NAO EXECUTADO" in capsys.readouterr().out
    cheia = tmp_path / "termos2.txt"
    cheia.write_text("termo_que_nao_aparece_no_repositorio_xyz\n", encoding="utf-8")
    aud.main(["--local", str(cheia), "--json"])
    assert '"r5": "executado"' in capsys.readouterr().out


def test_main_sem_git_falha_alto_em_vez_de_dizer_zero(aud, tmp_path, monkeypatch, capsys):
    """Arvore sem `.git` (zip, sdist): a varredura nao olhou nada, entao NAO pode sair '0 ocorrencias' com codigo 0."""
    (tmp_path / "a.md").write_text("contato segredo.pessoa@inst.edu.br\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.setattr(aud, "RAIZ", tmp_path)
    assert aud.main(["--local", str(tmp_path / "termos.txt")]) == 2
    assert "nada foi verificado" in capsys.readouterr().err
