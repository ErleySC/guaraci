"""Testes de scripts/estado_do_projeto.py: o que o relatorio de estado conta tem de bater com a entrada."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import estado_do_projeto as ep  # noqa: E402

JUNIT = """<?xml version="1.0"?>
<testsuites><testsuite name="pytest" tests="5">
<testcase classname="a" name="t1"/>
<testcase classname="a" name="t2"/>
<testcase classname="a" name="t3"><failure message="x">boom</failure></testcase>
<testcase classname="a" name="t4"><error message="y">boom</error></testcase>
<testcase classname="a" name="t5"><skipped type="pytest.skip" message="dataset X ausente">s</skipped></testcase>
</testsuite></testsuites>"""

BASE = {"git": {"ramo": "r", "hash": "h", "sujo": False}, "versoes": {"pyproject": "1"}, "versoes_ok": True,
        "privacidade": 0, "catalogo": [], "cenarios_total": 19, "exposicao": 0, "termos_de_matriz": 0,
        "links_quebrados": 0, "referencias_nao_conferidas": 0, "documentos": {"cientifico": 7, "privado-local (fora do git)": 3}}


def test_junit_conta_passados_falhas_erros_e_pulados(tmp_path):
    p = tmp_path / "r.xml"
    p.write_text(JUNIT, encoding="utf-8")
    r = ep.ler_junit(p)
    assert (r["passaram"], r["falharam"], r["erros"], r["pulados"]) == (2, 1, 1, 1)
    assert r["motivos"] == ["dataset X ausente"]


def test_cenarios_distingue_passou_nao_executado_e_falhou():
    t = ("| cenario | grupo | dataset | caminho | resultado | nota |\n|---|---|---|---|---|---|\n"
         "| a | g | d | c | passou | ok |\n| b | g | d | c | NAO EXECUTADO | falta |\n| c | g | d | c | FALHOU | rc=1 |\n")
    assert ep.contar_cenarios(t) == {"passou": 1, "nao executado": 1, "falhou": 1}


def test_versao_inconsistente_ou_ausente_nao_e_consistente():
    assert ep.versoes_consistentes({"a": "1.0", "b": "1.0"})
    assert not ep.versoes_consistentes({"a": "1.0", "b": "1.1"})
    assert not ep.versoes_consistentes({"a": "1.0", "b": None})


def test_pulado_com_motivo_novo_e_apontado():
    assert ep.pulados_fora_do_previsto(["um motivo que ninguem previu"])
    assert ep.pulados_fora_do_previsto([""])


def test_falhas_lista_o_que_reprova():
    assert ep.falhas(BASE) == []
    assert ep.falhas({**BASE, "versoes_ok": False})
    assert ep.falhas({**BASE, "privacidade": 2})
    assert ep.falhas({**BASE, "catalogo": ["x"]})
    assert ep.falhas({**BASE, "ruff": "FALHOU: x"})
    assert ep.falhas({**BASE, "testes": {"falharam": 1, "erros": 0}})
    assert ep.falhas({**BASE, "testes": {"falharam": 0, "erros": 0}, "pulados_problemas": ["x"]})


@pytest.mark.parametrize("chave", ["exposicao", "termos_de_matriz", "links_quebrados", "referencias_nao_conferidas"])
def test_cada_auditor_com_ocorrencia_reprova(chave):
    assert ep.falhas({**BASE, chave: 1}) and ep.falhas({**BASE, chave: 0}) == []


def test_verificacao_final_com_falha_reprova_e_sem_falha_passa():
    ok = {"hash": "abc1234", "falhas": 0, "etapas": 5, "reprovadas": [], "versoes_python": ["3.10", "3.13"]}
    assert ep.falhas({**BASE, "verificacao": ok}) == []
    assert ep.falhas({**BASE, "verificacao": {**ok, "falhas": 2, "reprovadas": ["ruff"]}})


def test_ler_resultado_resume_o_json_do_verificar_hash_final(tmp_path):
    p = tmp_path / "resultado_abc1234.json"
    p.write_text(json.dumps({"hash": "abc1234", "falhas": 1, "versoes_python": ["3.12"],
                             "etapas": [{"nome": "ruff", "estado": "ok"}, {"nome": "mypy", "estado": "falhou"}]}), encoding="utf-8")
    r = ep.ler_resultado(p)
    assert (r["hash"], r["falhas"], r["etapas"], r["reprovadas"]) == ("abc1234", 1, 2, ["mypy"])


def test_vault_nao_se_aplica_ou_pendente_nunca_aparece_como_reprovada(tmp_path):
    p = tmp_path / "resultado_abc1234.json"
    p.write_text(json.dumps({"hash": "abc1234", "falhas": 0, "versoes_python": ["3.12"],
                             "etapas": [{"nome": "ruff", "estado": "ok"}, {"nome": "vault", "estado": "nao_se_aplica"},
                                        {"nome": "r5", "estado": "pendente_do_autor"}]}), encoding="utf-8")
    r = ep.ler_resultado(p)
    assert (r["reprovadas"], r["nao_se_aplica"], r["pendentes_do_autor"]) == ([], ["vault"], ["r5"])
    linha = next(x for x in ep.formatar({**BASE, "verificacao": r}).splitlines() if x.startswith("VERIFICACAO FINAL"))
    assert "reprovadas" not in linha and "nao se aplica: vault" in linha and "pendentes do autor: r5" in linha
    assert ep.falhas({**BASE, "verificacao": r}) == []


def test_o_repositorio_atual_tem_versao_consistente_e_catalogo_sem_violacao():
    assert ep.versoes_consistentes(ep.versoes())
    assert ep.violacoes_do_catalogo() == []


def test_documentos_por_destino_conta_publico_por_pasta_e_privado_so_a_contagem():
    arq = ["docs/cientifico/a.md", "docs/cientifico/b.yaml", "docs/projeto/c.md", "docs/index.md", "README.md", "src/x.py"]
    d = ep.documentos_por_destino(arq, privados=4)
    assert d == {"cientifico": 2, "docs (raiz)": 1, "entrada (raiz)": 1, "projeto": 1, "privado-local (fora do git)": 4}


def test_saida_de_uma_tela_nao_mede_o_que_nao_foi_dado():
    s = ep.formatar(BASE)
    assert "TESTES nao medido" in s and "COBERTURA nao medido" in s and "VERIFICACAO FINAL nao medido" in s
    assert s.rstrip().endswith("RESULTADO OK")
    assert len(s.splitlines()) <= 16
    assert "privado-local (fora do git) 3" in s


def test_documento_gerado_traz_o_hash_e_nao_vaza_caminho():
    n = ep.documento_gerado({**BASE, "git": {"ramo": "r", "hash": "abc1234", "sujo": False}})
    assert "`abc1234`" in n and n.startswith("# Estado do projeto")
    from privacidade_amostras import caminhos_absolutos_em_texto
    assert caminhos_absolutos_em_texto(n) == []


@pytest.mark.parametrize("texto,esperado", [("x", None)])
def test_achar_sem_casamento_devolve_none(texto, esperado):
    assert ep._achar(r"version=(\d)", texto) is esperado


def test_privacidade_do_relatorio_concorda_com_a_guarda_e_ignora_a_sentinela():
    assert ep.ocorrencias_de_privacidade() == 0


def test_reexecucao_focada_substitui_so_o_arquivo_reexecutado(tmp_path):
    base = tmp_path / "b.xml"
    base.write_text('<?xml version="1.0"?><testsuites><testsuite name="pytest">'
                    '<testcase classname="tests.test_a" name="x"><failure message="m">m</failure></testcase>'
                    '<testcase classname="tests.test_a" name="y"/>'
                    '<testcase classname="tests.test_b" name="z"><failure message="m">m</failure></testcase>'
                    '</testsuite></testsuites>', encoding="utf-8")
    re_ = tmp_path / "r.xml"
    re_.write_text('<?xml version="1.0"?><testsuites><testsuite name="pytest">'
                   '<testcase classname="tests.test_a" name="x"/><testcase classname="tests.test_a" name="y"/>'
                   '</testsuite></testsuites>', encoding="utf-8")
    sem = ep.ler_junit(base)
    com = ep.ler_junit(base, [re_])
    assert (sem["passaram"], sem["falharam"]) == (1, 2)
    assert (com["passaram"], com["falharam"]) == (2, 1)  # test_a corrigido; test_b continua falhando


def _repo_git(pasta, sujo: bool):
    import subprocess
    pasta.mkdir(parents=True, exist_ok=True)
    env = ["-c", "user.name=t", "-c", "user.email=sem-email", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q"], cwd=pasta, check=True)
    (pasta / "a.txt").write_text("1\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=pasta, check=True)
    subprocess.run(["git", *env, "commit", "-q", "-m", "x"], cwd=pasta, check=True)
    if sujo:
        (pasta / "a.txt").write_text("2\n", encoding="utf-8")
    return pasta


def test_gravar_em_arvore_suja_e_recusado_e_em_limpa_e_aceito(tmp_path, monkeypatch, capsys):
    sujo = _repo_git(tmp_path / "suja", True)
    limpo = _repo_git(tmp_path / "limpa", False)
    assert ep.git_estado(sujo)["sujo"] and not ep.git_estado(limpo)["sujo"]
    assert ep.recusa_de_gravar({**BASE, "git": ep.git_estado(sujo)}) and ep.recusa_de_gravar({**BASE, "git": ep.git_estado(limpo)}) is None
    for pasta, esperado in ((sujo, 2), (limpo, 0)):
        monkeypatch.setattr(ep, "RAIZ", pasta)
        monkeypatch.setattr(ep, "medir", lambda _a, p=pasta: {**BASE, "git": ep.git_estado(p)})
        assert ep.main(["--gravar", "--sem-lint"]) == esperado
        assert (pasta / ep.SAIDA).exists() == (esperado == 0)
    assert "recusado" in capsys.readouterr().err


def test_documento_versionado_omite_a_contagem_privada_e_a_tela_mantem():
    n = ep.documento_gerado(BASE)
    assert "privado-local" not in n and "cientifico 7" in n
    assert "privado-local (fora do git) 3" in ep.formatar(BASE)
