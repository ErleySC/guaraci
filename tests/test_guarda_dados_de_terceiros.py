"""Guarda da politica de dados de terceiros (so' calibracao, sem mencao): nada visivel ao usuario cita conjunto de dados.

Varre o que o usuario ve -- `guaraci confianca` (lista e todas as tecnicas/opcoes, PT e EN), `guaraci manual` (todos os
topicos), os relatorios de uma execucao sintetica (model card, resumo, auditoria), o codigo-fonte do pacote (docstrings,
comentarios e textos de `src/guaraci/**/*.py`) e o wheel construido (nomes de arquivo e todos os arquivos) -- e reprova
se aparecer:
  (a) prefixo de DOI de repositorio de dados;
  (b) "dataset publico" / "public dataset";
  (c) nome de conjunto de dados de terceiros. Os nomes NAO ficam neste arquivo: vem do dossie privado do mantenedor
      (fora do git), lido so' quando existe -- o mesmo padrao da lista privada de `scripts/auditar_exposicao.py`.
      Sem o dossie (CI, clone de terceiro), valem so' (a) e (b).
"""
from __future__ import annotations

import importlib.util
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Pattern

import pytest

RAIZ = Path(__file__).resolve().parents[1]
DOSSIE_PADRAO = Path(os.path.expanduser("~")) / ".guaraci_local" / "validacao" / "DOSSIE_CALIBRACAO.md"

# a extracao dos nomes do dossie mora no auditor de termos (fonte unica, usada tambem por ele)
_spec = importlib.util.spec_from_file_location("_auditar_termos_matriz", RAIZ / "scripts" / "auditar_termos_matriz.py")
_aud = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aud)
termos_do_dossie = _aud.termos_do_dossie

# (a) prefixos de DOI de repositorios de dados; (b) frase generica. O software nao tem DOI proprio hoje.
PADROES_PUBLICOS = (
    r"10\.17632/", r"10\.6084/", r"10\.25678/", r"10\.5281/zenodo\.", r"10\.5061/dryad\.", r"10\.7910/dvn/",
    r"\bdatasets?\s+p[uú]blicos?\b", r"\bpublic\s+data\s?sets?\b",
    r"\bdados?\s+p[uú]blicos?\b", r"\bpublic\s+data\b",
)
#: Documentos versionados fora da varredura: so' `tests/` (entradas dos detectores; o codigo do pacote tem teste proprio).
#: O `paper/` entra na varredura (lote 10): o manuscrito segue a mesma politica de dados de terceiros.
DOCS_FORA = ("tests/",)


def padroes(dossie: Path = DOSSIE_PADRAO) -> List[Pattern[str]]:
    ps = [re.compile(p, re.I) for p in PADROES_PUBLICOS]
    ps += [re.compile(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])") for t in termos_do_dossie(dossie)]
    return ps


def achados(textos: Dict[str, str], ps: List[Pattern[str]]) -> List[str]:
    """'origem: trecho' de cada ocorrencia (o trecho fica so' na mensagem de falha local, nunca em arquivo)."""
    out = []
    for origem, texto in textos.items():
        for p in ps:
            m = p.search(texto)
            if m:
                out.append(f"{origem}: ...{texto[max(0, m.start() - 30):m.end() + 10]!r}...")
    return out


# ---------------------------------------------------------------- o mecanismo
# (DOIs de exemplo partidos em dois literais: o verificador de citacoes do repositorio nao os toma por citacao)
def test_o_detector_pega_doi_de_dados_e_a_frase_generica_e_ignora_texto_neutro():
    ps = padroes(Path("/caminho/que/nao/existe.md"))
    assert achados({"x": "veja doi:10." + "17632/abcd.1"}, ps) and achados({"x": "validado em dataset publico"}, ps)
    assert achados({"x": "tested on a public data set"}, ps)
    assert achados({"x": "na validacao com dado publico"}, ps) and achados({"x": "against results on public data"}, ps)
    assert achados({"x": "valores padrao calibrados em dados de referencia"}, ps) == []
    # texto legitimo vizinho: privacidade e dados do usuario nao sao "dado publico"
    assert achados({"x": "dados pessoais nao sao publicados; a publicacao de dados e' do usuario"}, ps) == []


def test_os_nomes_vem_do_dossie_privado_quando_ele_existe(tmp_path):
    d = tmp_path / "d.md"
    d.write_text("# t\n## Intro\n## Conjuntos sem licenca\n## Fulanoset Xy (NIR, 10 amostras)\n"
                 "## Repo 10." "9999/abc.1 (x)\n## Arquivos de terceiros\n## Depoisset\n", encoding="utf-8")
    assert termos_do_dossie(d) == ["10." "9999/abc.1", "Fulanoset", "Repo"]
    ps = padroes(d)
    assert achados({"x": "rodado no Fulanoset"}, ps) and achados({"x": "Depoisset"}, ps) == []


# ---------------------------------------------------------------- CLI
def _cli(capsys, nome: str, args: List[str]) -> str:
    from guaraci import cli_comandos
    cli_comandos.subcomando(nome, args)
    sai = capsys.readouterr()
    return sai.out + sai.err


def test_guaraci_confianca_nao_cita_conjunto_de_terceiros(capsys):
    from guaraci import confianca as conf
    textos = {"confianca (lista)": _cli(capsys, "confianca", [])}
    for tipo, ident, _, _ in conf.entradas():
        textos[f"confianca {ident}"] = _cli(capsys, "confianca", [ident])
        for lang in ("PT", "EN"):
            for ascii_only in (True, False):
                textos[f"pagina {tipo} {ident} {lang} {ascii_only}"] = conf.pagina(tipo, ident, lang, ascii_only)
        textos[f"lista EN {ident}"] = conf.lista(None, "EN", True)
    assert len(textos) > 50
    assert achados(textos, padroes()) == []


def test_guaraci_manual_nao_cita_conjunto_de_terceiros(capsys):
    from guaraci import docs_pacote as dp
    textos = {"manual (lista)": _cli(capsys, "manual", [])}
    for topico in dp.TOPICOS:
        textos[f"manual {topico}"] = _cli(capsys, "manual", [topico])
    assert all(len(t) > 200 for k, t in textos.items() if k != "manual (lista)")
    assert achados(textos, padroes()) == []


# ---------------------------------------------------------------- codigo-fonte do pacote
def _fontes_do_pacote() -> Dict[str, str]:
    return {p.relative_to(RAIZ).as_posix(): p.read_text(encoding="utf-8")
            for p in sorted((RAIZ / "src" / "guaraci").rglob("*.py"))}


def test_codigo_fonte_do_pacote_nao_cita_conjunto_de_terceiros():
    """Docstrings e comentarios vao no wheel e aparecem em `help()`: valem as mesmas regras do texto de tela."""
    textos = _fontes_do_pacote()
    assert len(textos) > 100
    assert achados(textos, padroes()) == []


def test_o_detector_de_texto_proibido_do_catalogo_usa_padroes_genericos():
    """Sem nome de conjunto no codigo, o detector do catalogo ainda pega prefixo de DOI de repositorio e a frase."""
    from guaraci import catalogo as cat
    assert cat.texto_proibido("rodado no conjunto doi:10." + "17632/abcd.1") and cat.texto_proibido("com dados publicos")


# ---------------------------------------------------------------- documentos publicos do repositorio
def _documentos_publicos() -> Dict[str, str]:
    """README, MANUAL, CHANGELOG, docs/, configuracoes e comentarios versionados (o que vai no sdist ou e' lido no GitHub)."""
    nomes = [f for f in _aud.arquivos(RAIZ) if not f.endswith(".py")] + [".gitignore"]
    textos = {}
    for f in nomes:
        if f.startswith(DOCS_FORA) or not (RAIZ / f).is_file():
            continue
        textos[f] = (RAIZ / f).read_text(encoding="utf-8", errors="replace")
    return textos


def test_documentos_publicos_nao_citam_conjunto_de_terceiros():
    textos = _documentos_publicos()
    assert {"README.md", "MANUAL.md", "ACKNOWLEDGMENTS.md", "CHANGELOG.md", ".gitignore"} <= set(textos)
    assert achados(textos, padroes()) == []


# ---------------------------------------------------------------- relatorios de uma execucao sintetica
@pytest.mark.slow
def test_relatorios_de_execucao_sintetica_nao_citam_conjunto_de_terceiros(pq, tmp_path):
    cfg = pq.Config(input_folder=str(tmp_path / "dados"), output_root_folder=str(tmp_path / "saida"),
                    mode="sintetico", n_per_class=10, n_synthetic_points=60, wn_min=400.0, wn_max=4001.0,
                    n_splits_cv=2, n_repeats_cv=1, n_permutations=5, n_permutations_wold=5, n_bootstrap_vip=3,
                    n_bootstrap_bca=20, n_monte_carlo=3, max_lvs=5)
    os.makedirs(cfg.input_folder, exist_ok=True)
    pq.executar(cfg)
    textos = {}
    for arq in Path(cfg.output_root_folder).rglob("*"):
        if arq.is_file() and arq.suffix.lower() in (".md", ".txt", ".json", ".csv", ".html"):
            textos[arq.name] = arq.read_text(encoding="utf-8", errors="replace")
    assert any(n == "model_card.md" for n in textos), sorted(textos)
    assert achados(textos, padroes()) == []


# ---------------------------------------------------------------- wheel construido
@pytest.mark.slow
def test_wheel_construido_nao_cita_conjunto_de_terceiros(tmp_path):
    import sys
    sys.path.insert(0, str(RAIZ / "scripts"))
    import auditar_pacote_construido as apc
    assert apc.construir(tmp_path) == 0
    (whl,) = tmp_path.glob("*.whl")
    textos = {}
    with zipfile.ZipFile(whl) as z:
        nomes = z.namelist()
        textos["(nomes de arquivo)"] = "\n".join(nomes)
        for n in nomes:
            if not n.endswith(".pyc"):
                textos[n] = z.read(n).decode("utf-8", errors="replace")
    assert achados(textos, padroes()) == []
