# -*- coding: utf-8 -*-
"""Lógica de detecção de identificador de amostra real, compartilhada entre
`tests/test_sem_identificador_real.py` (varre o repositório) e
`scripts/estado_do_projeto.py` (varre as saídas geradas).

Extraído de `tests/test_sem_identificador_real.py` sem alterar o padrão nem
a exceção -- só o lugar onde moram, para que todos reaproveitem a MESMA
regra em vez de duplicá-la (duplicar é como uma das duas varreduras
manuais anteriores já falhou, ver docstring do teste).
"""
from __future__ import annotations

import re

#: `COD-DD-MM-AAAA` -- especie + data de coleta, o formato de `mae_id` do
#: colecao de origem. Ver `tests/test_sem_identificador_real.py` para o
#: histórico completo de por que cada detalhe do padrão existe.
PADRAO_IDENTIFICADOR = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]{2,5}[-_]\d{2}-\d{2}-(?:19|20)\d{2}(?!\d)",
    re.IGNORECASE,
)

#: Único ano aceito em identificador de exemplo/fixture.
ANO_SENTINELA = "2099"

#: Caminho absoluto de máquina: unidade Windows (`C:\Users\alguem\...`) ou
#: home Unix (`/home/alguem/...`, `/Users/alguem/...`). O vault é gerado fora do
#: repositório (pasta pessoal do usuário) e não pode vazar ESSE caminho nem
#: nenhum outro caminho absoluto da máquina que o gerou.
PADRAO_CAMINHO_ABSOLUTO = re.compile(
    # `[\\/]+`: separador repetido, como em string escapada (`C:\\Users\\alguem`
    # dentro de printf/YAML/JSON). Achado de 2026-10-04: um caminho real assim,
    # num documento versionado, passava pela guarda que so' aceitava 1 barra.
    r"[A-Za-z]:[\\/]+(?:Users|home)[\\/]+[^\s\\/\"'`)]+"
    r"|/+(?:home|Users)/+[^\s/\"'`)]+"
    # Forma "achatada" com hifens que ferramentas usam para nomear pastas
    # temporarias a partir do caminho do projeto (`<letra>--Users-<usuario>-...`).
    # Achado de 2026-10-02: 5 saidas versionadas a tinham e a varredura so'
    # conhecia as formas com barra. Nao ha' como separar o usuario do resto
    # do nome, entao a forma achatada nunca tem excecao (nem a sentinela).
    r"|[A-Za-z]--(?:Users|home)-[^\s\\/\"'`)]+",
)

#: Único usuário aceito em caminho absoluto de EXEMPLO dentro de arquivo
#: versionado (fixture de teste, documentação do próprio padrão). Mesma
#: lógica do `ANO_SENTINELA`: não se julga caso a caso se um nome é real.
#: A exceção vale só para a varredura do repositório
#: (`tests/test_sem_identificador_real.py`) -- a guarda do vault NÃO a
#: aplica, e as contra-provas do vault usam justamente este nome.
USUARIO_SENTINELA = "alguem"

def identificadores_em_texto(texto: str) -> list[str]:
    """Retorna cada identificador de amostra real (exceto o ano sentinela)
    encontrado em `texto`."""
    return [
        m.group(0)
        for m in PADRAO_IDENTIFICADOR.finditer(texto)
        if not m.group(0).endswith(ANO_SENTINELA)
    ]


#: Categorias da classificacao (`classificar_ocorrencia`). So' as de `CATEGORIAS_REAIS`
#: reprovam: as demais sao exemplo declarado e passam.
CATEGORIA_PLACEHOLDER = "placeholder"
CATEGORIA_CAMINHO_REAL = "caminho real"
CATEGORIA_FORMA_ACHATADA = "caminho real (forma achatada)"
CATEGORIA_CODIGO_REAL = "codigo de amostra real"
CATEGORIA_CODIGO_EXEMPLO = "codigo de exemplo (ano sentinela)"
CATEGORIAS_REAIS = frozenset({CATEGORIA_CAMINHO_REAL, CATEGORIA_FORMA_ACHATADA, CATEGORIA_CODIGO_REAL})


def usuario_do_caminho(caminho: str) -> str:
    """Ultimo segmento de um caminho de usuario do Windows ou do Unix (separador repetido aceito)."""
    return re.split(r"[\\/]+", caminho.rstrip("\\/"))[-1]


def classificar_ocorrencia(achado: str) -> str:
    """Categoria de um trecho que casou com `PADRAO_CAMINHO_ABSOLUTO` ou `PADRAO_IDENTIFICADOR`.
    Mesma regra da varredura do repositorio: a unica folga e' a sentinela declarada
    (usuario `alguem`, ano 2099); a forma achatada nunca tem folga."""
    if PADRAO_IDENTIFICADOR.fullmatch(achado):
        return CATEGORIA_CODIGO_EXEMPLO if achado.endswith(ANO_SENTINELA) else CATEGORIA_CODIGO_REAL
    if re.match(r"[A-Za-z]--", achado):
        return CATEGORIA_FORMA_ACHATADA
    return CATEGORIA_PLACEHOLDER if usuario_do_caminho(achado) == USUARIO_SENTINELA else CATEGORIA_CAMINHO_REAL


def caminhos_absolutos_em_texto(texto: str) -> list[str]:
    """Retorna cada caminho absoluto de máquina encontrado em `texto`."""
    return [m.group(0) for m in PADRAO_CAMINHO_ABSOLUTO.finditer(texto)]


class VazamentoDePrivacidade(RuntimeError):
    """Levantado quando a varredura encontra identificador real ou caminho
    absoluto de máquina em conteúdo que seria publicado/gerado."""


def varrer_conteudos(itens: dict[str, str]) -> list[tuple[str, int, str]]:
    """Varre conteúdo em memória (`{caminho_relativo: texto}`), no CONTEÚDO e
    no NOME de cada item, e retorna `(caminho, linha, achado)`. Usado pelo
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


