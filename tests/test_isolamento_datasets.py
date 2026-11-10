"""test_isolamento_datasets.py -- Passo 118
(INSTRUCAO_PUSH_HIPOTESE_D_...md): prova P0 de que nenhum conjunto de
dados de terceiro esta ou jamais esteve versionado neste repositorio
-- checagem por COMANDO DIRETO (`git ls-files` + tamanho de blob),
nunca por alegacao. Roda em toda suite (nao gated por
GUARACI_DATASETS_DIR) -- e' uma checagem sobre o REPOSITORIO em si, nao
sobre um dataset baixado.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

_RAIZ_REPO = Path(__file__).resolve().parent.parent

#: Extensoes que so' fazem sentido como dado bruto de instrumento/dataset
#: publico (nunca como codigo/doc/config deste projeto) -- qualquer
#: arquivo com uma delas rastreado pelo git e' achado grave (P0).
_EXTENSOES_DADO_TERCEIRO = (
    ".mat", ".hdr", ".bin", ".raw", ".zip",
)

#: Limite de tamanho (bytes) acima do qual QUALQUER arquivo versionado
#: (de qualquer extensao) exige justificativa -- um conjunto de dados real
#: raramente cabe em menos que isso, mas um binario legitimo do projeto (icone,
#: por ex.) pode passar perto. 1 MiB da' folga generosa pro que o
#: projeto ja tem (o maior binario hoje e' guaraci_icon.png, ~2.7MB --
#: ver excecao explicita abaixo) sem abrir margem pra um dataset inteiro
#: colar sem ser pego.
_LIMITE_BYTES = 1 * 1024 * 1024

#: Arquivos legitimos do proprio projeto que passam do limite -- cada um
#: com justificativa, nunca uma allowlist muda em silencio.
_EXCECOES_TAMANHO = {
    "guaraci_icon.png",  # icone do projeto, nao dado de terceiro
}


def _git(args: list) -> str:
    return subprocess.run(
        ["git", *args], cwd=_RAIZ_REPO, capture_output=True,
        text=True, check=True).stdout


def test_nenhum_arquivo_de_dataset_publico_esta_versionado_hoje():
    """`git ls-files`: nenhum arquivo rastreado hoje tem extensao de
    dado bruto de terceiro, e nenhum passa do limite de tamanho sem
    estar na lista de excecoes justificadas."""
    arquivos = [l for l in _git(["ls-files"]).splitlines() if l.strip()]
    assert arquivos, "git ls-files nao devolveu nada -- comando/cwd errado?"

    com_extensao_suspeita = [
        f for f in arquivos if f.lower().endswith(_EXTENSOES_DADO_TERCEIRO)]
    assert not com_extensao_suspeita, (
        f"arquivo(s) com extensao de dado bruto de terceiro VERSIONADO(S): "
        f"{com_extensao_suspeita} -- achado P0, investigar antes de "
        f"prosseguir com qualquer outra coisa.")

    grandes_nao_justificados = []
    for f in arquivos:
        caminho = _RAIZ_REPO / f
        if not caminho.is_file():
            continue  # symlink quebrado ou caso de borda, nao e' o que procuramos
        if caminho.stat().st_size > _LIMITE_BYTES and caminho.name not in _EXCECOES_TAMANHO:
            grandes_nao_justificados.append((f, caminho.stat().st_size))
    assert not grandes_nao_justificados, (
        f"arquivo(s) versionado(s) acima de {_LIMITE_BYTES} bytes sem "
        f"justificativa em _EXCECOES_TAMANHO: {grandes_nao_justificados} "
        f"-- pode ser um dataset colado por engano.")


def test_nenhum_blob_de_dataset_publico_no_historico_completo():
    """Mesma checagem, mas em TODO o historico do git (`rev-list --all` +
    `cat-file --batch-check`) -- um arquivo removido de um commit
    POSTERIOR ainda vive no historico/objetos do repositorio e ainda
    seria baixado por qualquer `git clone` completo. So' roda quando
    `git` esta disponivel com o historico completo (pula em clone raso,
    onde a checagem nao seria confiavel)."""
    import pytest

    raso = _git(["rev-parse", "--is-shallow-repository"]).strip()
    if raso == "true":
        pytest.skip("clone raso -- checagem de historico completo nao "
                    "seria confiavel aqui (CI usa fetch-depth padrao "
                    "que pode ser raso).")

    saida = subprocess.run(
        ["git", "rev-list", "--objects", "--all"], cwd=_RAIZ_REPO,
        capture_output=True, text=True, check=True).stdout

    linhas = [l.split(" ", 1) for l in saida.splitlines() if " " in l]
    nomes_por_sha = {sha: nome for sha, nome in linhas}
    if not nomes_por_sha:
        pytest.skip("rev-list nao devolveu objetos com nome de arquivo "
                    "(historico vazio/raso?) -- nao ha o que checar.")

    batch_check = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objecttype) %(objectname) %(objectsize)"],
        cwd=_RAIZ_REPO, input="\n".join(nomes_por_sha) + "\n",
        capture_output=True, text=True, check=True).stdout

    grandes_no_historico = []
    for linha in batch_check.splitlines():
        partes = linha.split()
        if len(partes) != 3 or partes[0] != "blob":
            continue
        tipo, sha, tamanho = partes
        tamanho = int(tamanho)
        nome = nomes_por_sha.get(sha, "")
        if tamanho > _LIMITE_BYTES and Path(nome).name not in _EXCECOES_TAMANHO:
            grandes_no_historico.append((nome, tamanho))

    assert not grandes_no_historico, (
        f"blob(s) grande(s) no HISTORICO do git (mesmo que ja removidos "
        f"da arvore atual): {grandes_no_historico[:10]} -- ainda seriam "
        f"baixados por um clone completo, investigar antes de prosseguir.")


def test_gitignore_cobre_o_cache_local_de_datasets():
    """`.gitignore` precisa cobrir a pasta default de cache de datasets (`datasets_publicos/`) e a pasta `validacao/`
    da raiz, onde o mantenedor pode por a sua validacao local (que fica fora do repositorio)."""
    gitignore = (_RAIZ_REPO / ".gitignore").read_text(encoding="utf-8")
    assert "datasets_publicos" in gitignore
    assert re.search(r"^/validacao/\s*$", gitignore, re.M), "falta /validacao/ no .gitignore"


def test_nenhum_script_de_download_nem_validacao_de_terceiros_versionada():
    """Politica 3.7 (lote 2.1 da vistoria): scripts de download de conjuntos de terceiros e a validacao do mantenedor saem do
    repositorio (ficam no espaco privado). Nenhum arquivo versionado em `validacao/`, nenhum `baixar_*.py` em pasta alguma,
    nenhuma pasta `baixar/`."""
    versionados = _versionados()
    assert versionados, "git ls-files vazio -- caminho mudou?"
    fora = [f for f in versionados
            if f.startswith("validacao/") or Path(f).name.startswith("baixar_") or "/baixar/" in f"/{f}"]
    assert not fora, f"script de download ou validacao de terceiros versionado: {fora}"


# --- Politica de dados de terceiros (2026-10-07): o produto, os testes e os exemplos usam so' dado SINTETICO ----------------------
#: Extensoes que um arquivo versionado pode ter. Qualquer outra (.spc, .sp, .dat, .ch, .cdf, .jdx, .mat, sem extensao como `fid`/`acqus`)
#: e' dado ou arquivo de instrumento e so' entra se estiver na lista de excecoes abaixo.
_EXTENSOES_PERMITIDAS = {"py", "md", "json", "yaml", "yml", "txt", "toml", "sh", "png", "ico", "ipynb", "cff", "bib", "bat", "R",
                         "gitignore", "dockerignore", "mailmap", "spls", "csv", "cfg", "ini", "in"}
#: Sem extensao, versionados de proposito (nao sao dado).
_SEM_EXTENSAO_PERMITIDOS = {"Dockerfile", "LICENSE"}
#: .csv so' nestas pastas: saidas de um pacote de terceiros (licenca no proprio diretorio) calculadas sobre uma matriz X SINTETICA
#: (`rnorm` com semente fixa, ver gen_oracle.R) -- nao e' dado de terceiros.
_PASTAS_COM_CSV = ("tests/fixtures/dual_spls_oracle/",)
#: .tsv so' aqui: perguntas e respostas esperadas do assistente, escritas a mao (texto, nao medida).
_PASTAS_COM_TSV = ("tests/perguntas_assistente/",)


#: So' o seletor de perfil de pulados (que escolhe `local-completo` quando a variavel existe) e este teste a mencionam.
_CONHECEM_A_VARIAVEL = {"tests/test_isolamento_datasets.py", "tests/_pulados.py", "tests/test_pulados_esperados.py"}


def _versionados() -> list:
    return [l for l in _git(["ls-files"]).splitlines() if l.strip()]


def test_so_ha_dado_sintetico_versionado_nenhum_arquivo_de_instrumento_ou_tabela_de_terceiros():
    fora = []
    for f in _versionados():
        nome = Path(f).name
        ext = nome.rsplit(".", 1)[-1] if "." in nome else ""
        if not ext:
            if nome not in _SEM_EXTENSAO_PERMITIDOS:
                fora.append(f)
        elif ext == "tsv" and f.startswith(_PASTAS_COM_TSV):
            continue
        elif ext not in _EXTENSOES_PERMITIDAS:
            fora.append(f)
        elif ext == "csv" and not f.startswith(_PASTAS_COM_CSV):
            fora.append(f)
    assert not fora, f"arquivo(s) versionado(s) que parecem dado/instrumento de terceiros: {fora}"


def test_a_suite_padrao_nao_le_dado_de_terceiros_nem_a_variavel_de_datasets():
    """`GUARACI_DATASETS_DIR` e os downloads existem so' na validacao do mantenedor (fora do repositorio); `tests/` e `src/` nao os conhecem."""
    achados = []
    for f in _versionados():
        if f.startswith(("tests/", "src/")) and f.endswith(".py") and f not in _CONHECEM_A_VARIAVEL:
            texto = (_RAIZ_REPO / f).read_text(encoding="utf-8", errors="replace")
            if "GUARACI_DATASETS_DIR" in texto or "datasets_publicos" in texto:
                achados.append(f)
    assert not achados, f"tests/ e src/ nao podem referenciar dataset de terceiros: {achados}"


def test_nenhum_ponteiro_para_medicoes_ou_scripts_de_download_que_sairam_do_repositorio():
    """`validacao/medicoes/`, `validacao/baixar/` e `validacao/README.md` sairam do repositorio: o pacote e os documentos
    nao podem mandar o leitor a um arquivo que nao existe (texto de tela inclusive)."""
    rx = re.compile(r"validacao/(?:medicoes|baixar|README\.md)")
    achados = [f for f in _versionados()
               if f != "tests/test_isolamento_datasets.py" and f != "tests/test_links_docs.py"
               and f.endswith((".py", ".md", ".yml", ".yaml", ".toml", ".txt"))
               and rx.search((_RAIZ_REPO / f).read_text(encoding="utf-8", errors="replace"))]
    assert not achados, f"ponteiro para arquivo que saiu do repositorio: {achados}"


def test_o_pacote_nao_leva_validacao_nem_dados():
    """pyproject lista so' os pacotes do produto; MANIFEST.in poda a validacao do sdist."""
    pyproject = (_RAIZ_REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'packages = ["guaraci", "guaraci.app_tabs"]' in pyproject
    manifest = (_RAIZ_REPO / "MANIFEST.in").read_text(encoding="utf-8")
    for pasta in ("validacao", "datasets_publicos", "docs/qualidade"):
        assert f"prune {pasta}" in manifest, pasta
