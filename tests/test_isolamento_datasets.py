"""test_isolamento_datasets.py -- Passo 118
(INSTRUCAO_PUSH_HIPOTESE_D_...md): prova P0 de que nenhum dataset
publico de terceiro esta ou jamais esteve versionado neste repositorio
-- checagem por COMANDO DIRETO (`git ls-files` + tamanho de blob),
nunca por alegacao. Roda em toda suite (nao gated por
GUARACI_DATASETS_DIR) -- e' uma checagem sobre o REPOSITORIO em si, nao
sobre um dataset baixado.
"""
from __future__ import annotations

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
#: (de qualquer extensao) exige justificativa -- nenhum dataset publico
#: cabe em menos que isso, mas um binario legitimo do projeto (icone,
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
    """`.gitignore` precisa cobrir explicitamente a pasta default de
    cache de datasets (`datasets_publicos/`, usada como default por
    TODOS os scripts de scripts/download_datasets/ quando
    GUARACI_DATASETS_DIR nao esta setado)."""
    gitignore = (_RAIZ_REPO / ".gitignore").read_text(encoding="utf-8")
    assert "datasets_publicos" in gitignore


def test_todos_os_scripts_de_download_usam_o_mesmo_mecanismo():
    """Nenhum script de download deve inventar um segundo mecanismo de
    cache -- todos devem ler `GUARACI_DATASETS_DIR`, com o MESMO
    fallback `datasets_publicos` (pasta solta na raiz, ja coberta pelo
    `.gitignore`) quando a variavel nao esta setada. So' checa a linha
    de fallback em si (regex), nao qualquer mencao textual a "tests/"
    em docstring/comentario (ex.: "ver tests/test_x.py para..." e'
    documentacao legitima, nao um destino de escrita)."""
    import re

    pasta_scripts = _RAIZ_REPO / "scripts" / "download_datasets"
    scripts_py = sorted(p for p in pasta_scripts.glob("*.py"))
    assert scripts_py, "nenhum script de download encontrado -- caminho mudou?"

    padrao_fallback = re.compile(
        r'os\.environ\.get\(\s*["\']GUARACI_DATASETS_DIR["\']\s*,\s*'
        r'["\']([^"\']+)["\']\s*\)')

    for script in scripts_py:
        texto = script.read_text(encoding="utf-8")
        assert "GUARACI_DATASETS_DIR" in texto, (
            f"{script.name} nao referencia GUARACI_DATASETS_DIR -- "
            f"pode estar usando um mecanismo de cache diferente.")
        m = padrao_fallback.search(texto)
        assert m, (
            f"{script.name}: nao achei o padrao "
            f"os.environ.get('GUARACI_DATASETS_DIR', '<fallback>') -- "
            f"confirme manualmente que o destino nao mudou de mecanismo.")
        fallback = m.group(1)
        assert fallback == "datasets_publicos", (
            f"{script.name}: fallback de destino e' {fallback!r}, "
            f"esperado 'datasets_publicos' (mesmo default dos outros "
            f"scripts, ja coberto pelo .gitignore) -- um valor diferente "
            f"pode escapar da cobertura do .gitignore ou apontar pra "
            f"dentro da arvore versionada.")
