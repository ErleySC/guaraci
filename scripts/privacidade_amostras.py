# -*- coding: utf-8 -*-
"""Lógica de detecção de identificador de amostra real, compartilhada entre
`tests/test_sem_identificador_real.py` (varre o repositório) e
`scripts/gerar_vault_obsidian.py` (varre o vault gerado, Passo 169).

Extraído de `tests/test_sem_identificador_real.py` sem alterar o padrão nem
a exceção -- só o lugar onde moram, para que o gerador do vault reaproveite
a MESMA regra em vez de duplicá-la (duplicar é como uma das duas varreduras
manuais anteriores já falhou, ver docstring do teste).
"""
from __future__ import annotations

import re
from pathlib import Path

#: `COD-DD-MM-AAAA` -- especie + data de coleta, o formato de `mae_id` do
#: acervo de origem. Ver `tests/test_sem_identificador_real.py` para o
#: histórico completo de por que cada detalhe do padrão existe.
PADRAO_IDENTIFICADOR = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]{2,5}[-_]\d{2}-\d{2}-(?:19|20)\d{2}(?!\d)",
    re.IGNORECASE,
)

#: Único ano aceito em identificador de exemplo/fixture.
ANO_SENTINELA = "2099"

#: Caminho absoluto de máquina: unidade Windows (`C:\Users\nome\...`) ou
#: home Unix (`/home/nome/...`, `/Users/nome/...`). O vault é gerado fora do
#: repositório (pasta pessoal do usuário) e não pode vazar ESSE caminho nem
#: nenhum outro caminho absoluto da máquina que o gerou.
PADRAO_CAMINHO_ABSOLUTO = re.compile(
    r"[A-Za-z]:[\\/](?:Users|home)[\\/][^\s\\/\"'`)]+"
    r"|/(?:home|Users)/[^\s/\"'`)]+",
)

_BINARIO = {".png", ".ico", ".jpg", ".jpeg", ".gif", ".pdf", ".joblib",
            ".xlsx", ".docx", ".pptx", ".woff", ".woff2", ".zip", ".gz"}


def identificadores_em_texto(texto: str) -> list[str]:
    """Retorna cada identificador de amostra real (exceto o ano sentinela)
    encontrado em `texto`."""
    return [
        m.group(0)
        for m in PADRAO_IDENTIFICADOR.finditer(texto)
        if not m.group(0).endswith(ANO_SENTINELA)
    ]


def caminhos_absolutos_em_texto(texto: str) -> list[str]:
    """Retorna cada caminho absoluto de máquina encontrado em `texto`."""
    return [m.group(0) for m in PADRAO_CAMINHO_ABSOLUTO.finditer(texto)]


class VazamentoDePrivacidade(RuntimeError):
    """Levantado quando a varredura encontra identificador real ou caminho
    absoluto de máquina em conteúdo que seria publicado/gerado."""


def varrer_diretorio(raiz: Path) -> list[tuple[str, int, str]]:
    """Varre todo arquivo de texto sob `raiz` (recursivo) e retorna
    `(caminho_relativo, linha, achado)` para cada identificador real ou
    caminho absoluto encontrado -- no CONTEÚDO e no NOME de cada arquivo."""
    achados: list[tuple[str, int, str]] = []
    for caminho in sorted(raiz.rglob("*")):
        if not caminho.is_file():
            continue
        rel = str(caminho.relative_to(raiz)).replace("\\", "/")
        for ident in identificadores_em_texto(rel):
            achados.append((rel, 0, ident))
        for cam in caminhos_absolutos_em_texto(rel):
            achados.append((rel, 0, cam))
        if caminho.suffix.lower() in _BINARIO:
            continue
        try:
            texto = caminho.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            achados.append((rel, 0, "<ILEGIVEL: nao e' UTF-8 nem binario declarado>"))
            continue
        for n, linha in enumerate(texto.splitlines(), start=1):
            for ident in identificadores_em_texto(linha):
                achados.append((rel, n, ident))
            for cam in caminhos_absolutos_em_texto(linha):
                achados.append((rel, n, cam))
    return achados


def varrer_conteudos(itens: dict[str, str]) -> list[tuple[str, int, str]]:
    """Mesma varredura de `varrer_diretorio`, mas sobre conteúdo em memória
    (`{caminho_relativo: texto}`) em vez de arquivos em disco. Usado pelo
    gerador do vault para checar o plano de escrita ANTES de gravar
    qualquer arquivo -- se algo vazar, nada é escrito."""
    achados: list[tuple[str, int, str]] = []
    for rel, texto in sorted(itens.items()):
        rel_norm = rel.replace("\\", "/")
        for ident in identificadores_em_texto(rel_norm):
            achados.append((rel_norm, 0, ident))
        for cam in caminhos_absolutos_em_texto(rel_norm):
            achados.append((rel_norm, 0, cam))
        for n, linha in enumerate(texto.splitlines(), start=1):
            for ident in identificadores_em_texto(linha):
                achados.append((rel_norm, n, ident))
            for cam in caminhos_absolutos_em_texto(linha):
                achados.append((rel_norm, n, cam))
    return achados


def checar_conteudos_ou_falhar(itens: dict[str, str]) -> None:
    """Levanta `VazamentoDePrivacidade` se `varrer_conteudos` encontrar
    qualquer achado no plano de escrita do vault."""
    achados = varrer_conteudos(itens)
    if achados:
        linhas = "\n".join(f"  {a}:{n}  ->  {ident}" for a, n, ident in achados)
        raise VazamentoDePrivacidade(
            f"{len(achados)} vazamento(s) de privacidade no PLANO do vault "
            f"(nada foi escrito em disco):\n{linhas}"
        )


def checar_diretorio_ou_falhar(raiz: Path) -> None:
    """Levanta `VazamentoDePrivacidade` se `varrer_diretorio` encontrar
    qualquer achado. Usado pelo gerador do vault (Passo 169): o gerador
    FALHA, não avisa e continua."""
    achados = varrer_diretorio(raiz)
    if achados:
        linhas = "\n".join(f"  {a}:{n}  ->  {ident}" for a, n, ident in achados)
        raise VazamentoDePrivacidade(
            f"{len(achados)} vazamento(s) de privacidade no vault gerado em "
            f"{raiz}:\n{linhas}"
        )
