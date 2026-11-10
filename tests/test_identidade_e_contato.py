"""Lote 10 da vistoria: identidade e contato (V103, V110, linha de contato de V106).

Decisao do autor: dono = conta `ErleySC` (sem organizacao); duvidas = GitHub Discussions; vulnerabilidade = relato privado do
GitHub; NENHUM e-mail em arquivo versionado; sem link de curriculo; o nome, o ORCID, a licenca e os links do repositorio aparecem
no `--version`, no painel da CLI e na tela Sobre. Nada aqui contem o e-mail nem o seu padrao pessoal: a varredura usa o padrao
generico de `scripts/auditar_exposicao.py` e a lista privada, quando existir.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:        # Python 3.10: o pytest ja traz o tomli
    import tomli as tomllib

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import auditar_exposicao as ae  # noqa: E402

from guaraci import config as cfg  # noqa: E402

#: Onde a URL do repositorio aparece escrita (alem de `guaraci/config.py`, que e' a fonte no codigo). Numa renomeacao do
#: repositorio: troque `REPO_URL` e estes arquivos; o teste falha se aparecer um lugar novo sem entrar na lista.
ONDE_A_URL_APARECE = {
    ".github/ISSUE_TEMPLATE/config.yml", "CITATION.cff", "CONTRIBUTING.md", "MANUAL.md", "README.md", "README.pt-BR.md",
    "docs/index.md", "mkdocs.yml", "notebooks/guaraci_5_minutos.ipynb", "pyproject.toml", "scripts/verificar_citacoes.py",
    "src/guaraci/config.py", "src/guaraci/dados_docs/manual.md", "SECURITY.md", "SUPPORT.md", "CODE_OF_CONDUCT.md",
    "docs/usuario/privacidade.md", "docs/usuario/termos_de_uso.md", "src/guaraci/dados_docs/seguranca.md",
    "src/guaraci/dados_docs/suporte.md", "src/guaraci/dados_docs/privacidade.md", "src/guaraci/dados_docs/termos_de_uso.md",
}


def _ler(rel: str) -> str:
    return (RAIZ / rel).read_text(encoding="utf-8")


def _rastreados():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=RAIZ, capture_output=True).stdout.split(b"\0")
    return [f.decode() for f in out if f]


def test_constantes_apontam_para_a_conta_sem_organizacao():
    assert cfg.AUTOR_GITHUB_URL == "https://github.com/ErleySC"
    assert cfg.REPO_URL == "https://github.com/ErleySC/guaraci"
    assert cfg.DISCUSSIONS_URL == cfg.REPO_URL + "/discussions"
    assert cfg.SEGURANCA_URL.startswith(cfg.REPO_URL + "/security/")


def test_nenhum_arquivo_versionado_tem_email_nem_padrao_generico():
    """Varredura da arvore inteira com os padroes genericos (e-mail sem isencao por arquivo). Roda em qualquer maquina."""
    assert ae.EMAIL_PERMITIDO_EM == ()
    achados = ae.varrer_arvore(RAIZ, [])
    assert achados == {}, {f: len(v) for f, v in achados.items()}


def _lista_privada_ou_pula():
    locais = ae.termos_locais_se_existirem()
    if not locais:
        pytest.skip("lista privada de termos ausente ou vazia nesta maquina: so' as verificacoes genericas rodam")
    return locais


def test_nenhum_arquivo_versionado_tem_termo_da_lista_privada():
    """Com a lista do autor: a arvore (as guardas incluidas, que o auditor isenta dos padroes genericos) tem 0 termos."""
    locais = _lista_privada_ou_pula()
    ruins = {}
    for f in _rastreados():
        try:
            linhas = (RAIZ / f).read_bytes().decode("utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        n = sum(1 for l in linhas if "termo_local" in ae.achar_na_linha(l, {}, locais, f)) + (
            1 if "termo_local" in ae.achar_na_linha(f, {}, locais, f) else 0)
        if n:
            ruins[f] = n
    assert ruins == {}, ruins


def test_ferramental_de_reescrita_do_historico_nao_e_versionado():
    """A ferramenta que reescreve o historico (e os seus testes) mora fora do repositorio, no ferramental privado do mantenedor."""
    fora = {"scripts/reconstruir_historico_publicavel.py", "scripts/precheck_publicavel.py",
            "scripts/localizar_identificadores_no_historico.py", "scripts/varrer_privacidade_intervalo.py",
            "tests/test_reconstruir_historico_publicavel.py", "tests/test_precheck_publicavel.py",
            "tests/test_varrer_privacidade_intervalo.py"}
    assert fora & set(_rastreados()) == set()


def test_ler_a_lista_privada_ausente_nao_cria_o_arquivo(tmp_path):
    arq = tmp_path / "nao_existe" / "termos.txt"
    assert ae.termos_locais_se_existirem(arq) is None
    assert not arq.exists() and not arq.parent.exists()
    arq.parent.mkdir()
    arq.write_text("# c\ntermo sintetico\n", encoding="utf-8")
    assert ae.termos_locais_se_existirem(arq) == ["termo sintetico"]


def test_email_em_qualquer_arquivo_rastreado_de_texto_reprova():
    """Mais largo que o auditor (que so' le' certas extensoes): qualquer arquivo de texto rastreado, sem isencao de arquivo."""
    rx = re.compile(ae.GENERICOS["email"])
    ruins = []
    for f in _rastreados():
        if f in ae.IGNORAR or f.startswith("tests/"):
            continue                                    # carregam e-mails SINTETICOS de proposito (testes do auditor)
        try:
            texto = (RAIZ / f).read_bytes().decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for m in rx.finditer(texto):
            e = m.group(0).lower()
            if e in ae.EMAILS_PERMITIDOS or e.endswith(ae.DOMINIOS_DE_EXEMPLO):
                continue
            ruins.append(f)
    assert ruins == []


def test_pyproject_tem_so_o_nome_e_os_enderecos_do_repositorio():
    p = tomllib.loads(_ler("pyproject.toml"))["project"]
    assert p["authors"] == [{"name": cfg.AUTOR_NOME}] and p["maintainers"] == [{"name": cfg.AUTOR_NOME}]
    u = p["urls"]
    assert u["Homepage"] == u["Repository"] == cfg.REPO_URL
    assert u["Issues"] == cfg.REPO_URL + "/issues" and u["Discussions"] == cfg.DISCUSSIONS_URL
    assert u["Security"].startswith(cfg.REPO_URL + "/security")
    assert "Documentation" in u


def test_citation_sem_email_e_sem_curriculo():
    c = _ler("CITATION.cff")
    assert "email:" not in c and "latt" "es" not in c.lower() and "cnp" "q" not in c.lower()
    assert cfg.AUTOR_ORCID in c and f'url: "{cfg.REPO_URL}"' in c


def test_nenhum_arquivo_versionado_aponta_curriculo_em_bases_de_vinculo():
    ruins = []
    for f in _rastreados():
        if f.startswith(("docs/auditoria", "tests/fixtures")) or f == "tests/test_identidade_e_contato.py":
            continue
        try:
            t = (RAIZ / f).read_bytes().decode("utf-8").lower()
        except (UnicodeDecodeError, OSError):
            continue
        if "latt" "es" + "." in t or "cnp" "q" + ".br" in t:
            ruins.append(f)
    assert ruins == []


@pytest.mark.parametrize("rel", ["SECURITY.md", "SUPPORT.md", "CODE_OF_CONDUCT.md", "docs/usuario/privacidade.md", "docs/usuario/termos_de_uso.md"])
def test_documentos_apontam_para_discussions_ou_canal_privado_sem_marcador_pendente(rel):
    t = _ler(rel)
    assert "definir no lote" not in t and "ainda não há um canal privado" not in t
    assert "Discussions" in t or "Report a vulnerability" in t


def test_seguranca_manda_para_o_canal_privado_do_github():
    t = _ler("SECURITY.md")
    assert "Report a vulnerability" in t and cfg.SEGURANCA_URL in t
    assert "Nunca em issue pública" not in t and "7 dias úteis" in t
    assert not re.search(ae.GENERICOS["email"], t)


def test_suporte_e_codigo_de_conduta_usam_o_canal_do_github_e_nao_o_citation():
    s = _ler("SUPPORT.md")
    assert "Report a vulnerability" in s and cfg.DISCUSSIONS_URL in s
    assert "contato do mantenedor" not in s and "maintainer contact" not in s
    assert "contact channel listed" not in _ler("CODE_OF_CONDUCT.md")


def test_texto_de_versao_mostra_nome_orcid_licenca_repositorio_e_discussions_sem_afiliacao_nem_email():
    from guaraci.guaraci import _texto_versao
    t = _texto_versao()
    for esperado in (cfg.AUTOR_NOME, cfg.AUTOR_ORCID, cfg.LICENCA_SPDX, cfg.REPO_URL, cfg.DISCUSSIONS_URL):
        assert esperado in t
    assert "@" not in t.replace(str(Path(sys.modules["guaraci.guaraci"].__file__).resolve().parent), "")


def test_tela_sobre_da_web_usa_as_constantes_sem_curriculo_nem_pagina_de_citation_como_contato():
    s = _ler("src/guaraci/app_tabs/sobre.py")
    assert "REPO_URL" in s and "DISCUSSIONS_URL" in s and "SEGURANCA_URL" in s and "AUTOR_GITHUB_URL" in s
    assert "_AUTOR_LAT" "TES" not in s and "github.com/ErleySC" not in s
    assert "blob/master/CITATION.cff)" not in s            # o antigo link "Contato" apontava para o CITATION.cff


def test_painel_da_cli_mostra_orcid_github_projeto_e_discussions():
    g = _ler("src/guaraci/guaraci.py")
    assert "_AUTOR_LAT" "TES" not in g and "Latt" "es" not in g
    assert 't.add_row("ORCID:"' in g and 't.add_row("Discussions:"' in g and "github.com/ErleySC" not in g


def test_url_do_repositorio_so_aparece_nos_lugares_listados():
    onde = set()
    for f in _rastreados():
        if f.startswith(("docs/auditoria", "tests/", "CHANGELOG.md")) or f.endswith((".png", ".ico", ".pdf", ".whl")):
            continue
        try:
            t = (RAIZ / f).read_bytes().decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "github.com/ErleySC" in t:
            onde.add(f)
    assert onde <= ONDE_A_URL_APARECE, sorted(onde - ONDE_A_URL_APARECE)
    assert {"pyproject.toml", "CITATION.cff", "src/guaraci/config.py"} <= onde


def test_paper_nao_agradece_pessoas_instituicoes_amostras_nem_dados_e_nao_cita_vinculo():
    """V109 (decisao do autor): o agradecimento do paper e' so' a software de codigo aberto; sem vinculo com laboratorio."""
    txt = _ler("paper/paper.md")
    ag = txt.split("# Acknowledgements", 1)[1].split("# References", 1)[0].lower()
    assert "open-source" in ag and "stack" in ag
    for proibido in ("sample", "instrument", "acquisition", "dataset", "laborator", "fund", "grant", "supervis", "advis"):
        assert proibido not in ag, proibido
    assert "own laboratory" not in txt


def test_guarda_de_terceiros_cobre_o_paper_e_o_texto_nao_cita_conjunto_publico():
    """Politica 3.7: dado de terceiros so' calibra, sem mencao. O paper deixa de ser excecao da guarda."""
    guarda = _ler("tests/test_guarda_dados_de_terceiros.py")
    assert re.search(r"^DOCS_FORA = \(\"tests/\",\)$", guarda, re.M)
    ae_t = __import__("auditar_termos_matriz")
    assert "paper/paper.md" not in ae_t.PERMITIDO
    txt = _ler("paper/paper.md").lower()
    assert "public dataset" not in txt and "tec" "ator" not in txt and "thod" "berg" not in txt


#: Conselho tecnico sobre pasta sincronizada diz "pasta sincronizada com a nuvem (servicos de sincronizacao)", sem nomear produto.
#: So' as DEFINICOES do auditor carregam o nome, porque os padroes precisam dele.
NOME_DO_SERVICO_PERMITIDO_EM = {"scripts/auditar_exposicao.py", "tests/test_auditar_exposicao.py"}


def test_nenhum_arquivo_versionado_nomeia_o_servico_de_sincronizacao_fora_das_definicoes_do_auditor():
    nome = "one" + "drive"
    ruins = []
    for f in _rastreados():
        if f in NOME_DO_SERVICO_PERMITIDO_EM or f.startswith("docs/auditoria") or f.endswith((".png", ".ico", ".pdf", ".whl")):
            continue
        try:
            t = (RAIZ / f).read_bytes().decode("utf-8").lower()
        except (UnicodeDecodeError, OSError):
            continue
        if nome in t:
            ruins.append(f)
    assert ruins == []


def test_nenhum_arquivo_versionado_cita_a_sigla_do_trabalho_academico():
    """O produto nao carrega rotulo de trabalho de curso: nem na pasta de saida padrao, nem em nome de script, nem em texto.
    (Literal partido: este arquivo tambem passa pela varredura.)"""
    sigla = re.compile(r"(?<![a-z0-9])" + "t" + "cc" + r"(?![a-z0-9])", re.I)
    ruins = []
    for f in _rastreados():
        if f.startswith("docs/auditoria") or f.endswith((".png", ".ico", ".pdf", ".whl")):
            continue
        try:
            t = (RAIZ / f).read_bytes().decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if sigla.search(t) or sigla.search(f):
            ruins.append(f)
    assert ruins == []


#: O deposito do proprio software foi retirado; so' a "lapide" (registro de referencias, exigido pela guarda de citacoes) o nomeia.
LAPIDE_DO_DEPOSITO_EM = {"docs/cientifico/referencias_verificadas.yaml", "docs/cientifico/referencias.md"}


def test_identificador_do_deposito_retirado_so_aparece_na_lapide_do_registro_de_referencias():
    trecho = "5281/" + "zeno" + "do.2131"
    ruins = []
    for f in _rastreados():
        if f in LAPIDE_DO_DEPOSITO_EM or f.startswith(("tests/", "docs/auditoria")) or f.endswith((".png", ".ico", ".pdf", ".whl")):
            continue
        try:
            t = (RAIZ / f).read_bytes().decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if trecho in t:
            ruins.append(f)
    assert ruins == []
