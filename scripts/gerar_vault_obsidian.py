# -*- coding: utf-8 -*-
"""Gera o vault Obsidian do Guaraci a partir das fontes de verdade do
repositório (Passos 166-171 da auditoria de 2026-09-06).

NÃO edite o vault gerado à mão (exceto a pasta `70-Notas-Pessoais/`, que
este script nunca toca) -- rode este script de novo para regenerar. Ver
`README-VAULT.md` (escrito dentro do vault) para os detalhes.

Uso:
    python scripts/gerar_vault_obsidian.py [--out CAMINHO] [--check]

Destino, em ordem de prioridade: `--out`, variável de ambiente
`GUARACI_VAULT_DIR`, `~/GuaraciVault`.

`--check` só roda a geração em memória e a guarda de privacidade (Passo
169), sem escrever nada em disco -- usado por CI/verificação rápida.

Princípio central (não-negociável): todo número específico (RMSEP,
balanced accuracy, contagem de testes, etc.) vem de uma leitura de
arquivo real, nunca de um literal escrito neste script. Onde a fonte não
permite gerar uma categoria com confiança, a categoria fica vazia --
"evidência ou silêncio" (CLAUDE.md) se aplica ao vault tanto quanto ao
código.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_AQUI = Path(__file__).resolve()
_SCRIPTS_DIR = _AQUI.parent
_RAIZ = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from privacidade_amostras import (  # noqa: E402
    VazamentoDePrivacidade,
    checar_conteudos_ou_falhar,
)

_MANIFESTO = ".vault_manifest.json"
_PASTA_PROTEGIDA = "70-Notas-Pessoais"


# ═════════════════════════════════════════════════════════════════════════
#  Utilidades gerais
# ═════════════════════════════════════════════════════════════════════════

def _slug(texto: str, max_len: int = 70) -> str:
    """`Título Longo, com Pontuação!` -> `titulo-longo-com-pontuacao`."""
    nfkd = unicodedata.normalize("NFKD", texto)
    sem_acento = "".join(c for c in nfkd if not unicodedata.combining(c))
    sem_acento = sem_acento.lower()
    sem_acento = re.sub(r"[^a-z0-9]+", "-", sem_acento).strip("-")
    return sem_acento[:max_len].strip("-") or "nota"


def _ler(rel: str) -> str:
    return (_RAIZ / rel).read_text(encoding="utf-8")


def _git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=_RAIZ, capture_output=True, check=True)
    return r.stdout.decode("utf-8", "replace").strip()


def _head_info() -> tuple[str, str]:
    """`(hash_curto, data_iso)` do commit HEAD -- usado como `gerado_em`
    para que rodar o gerador duas vezes sem o repositório mudar produza
    exatamente o mesmo conteúdo (idempotência real, Passo 167)."""
    try:
        h = _git("rev-parse", "--short", "HEAD")
        d = _git("log", "-1", "--format=%cI")
        return h, d
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "sem-git", "desconhecido"


_HEAD_HASH, _HEAD_DATA = _head_info()


def _frontmatter(tags: list[str], fonte: str | list[str], **extra: Any) -> str:
    linhas = ["---"]
    linhas.append("tags: [" + ", ".join(tags) + "]")
    if isinstance(fonte, list):
        linhas.append("fonte:")
        for f in fonte:
            linhas.append(f"  - {f}")
    else:
        linhas.append(f"fonte: {fonte}")
    linhas.append(f"gerado_em: {_HEAD_DATA}")
    linhas.append(f"commit: {_HEAD_HASH}")
    for k, v in extra.items():
        if v is None:
            continue
        linhas.append(f"{k}: {v}")
    linhas.append("---")
    return "\n".join(linhas) + "\n"


def _nota(titulo: str, tags: list[str], fonte: str | list[str],
          corpo: str, **extra: Any) -> str:
    return _frontmatter(tags, fonte, **extra) + f"\n# {titulo}\n\n" + corpo.strip() + "\n"


def _wikilink(nome: str) -> str:
    """Link para uma nota cujo NOME DE ARQUIVO já é `nome` (módulos
    `x.py.md`, MOCs, `Estado-Atual.md`, ou qualquer stem já calculado por
    `_slug` em outro lugar). Não usar com título humano livre -- ver
    `_wikilink_titulo`."""
    return f"[[{nome}]]"


def _wikilink_titulo(titulo: str) -> str:
    """Link para uma nota cujo arquivo foi nomeado com `_slug(titulo)`
    (técnicas, conceitos) -- usa alias `[[slug|Título Humano]]` para que o
    link resolva ao arquivo real e ainda mostre o título legível."""
    return f"[[{_slug(titulo)}|{titulo}]]"


_PADRAO_MODULO_EM_CRASE = re.compile(r"`(\w+)\.py`")


def _autolinkar_modulos(texto: str, modulos_conhecidos: set[str]) -> str:
    """Troca `` `nome.py` `` por `[[nome.py]]` quando `nome` é um módulo
    real de `src/guaraci/`. Aplica-se a texto extraído de fonte (docstring,
    parágrafo de achado/decisão) que já cita o módulo por convenção própria
    da documentação -- transforma essa menção em aresta real do grafo
    (Passo 173) sem inventar relação nenhuma: só liga ao que a própria
    fonte já escreveu explicitamente entre crases."""
    def _sub(m: re.Match[str]) -> str:
        nome = m.group(1)
        return f"[[{nome}.py]]" if nome in modulos_conhecidos else m.group(0)
    return _PADRAO_MODULO_EM_CRASE.sub(_sub, texto)


# ═════════════════════════════════════════════════════════════════════════
#  Parsing das fontes de verdade
# ═════════════════════════════════════════════════════════════════════════

def parse_tecnicas_catalog() -> dict[str, dict[str, Any]]:
    """Extrai `TECNICAS` de `src/guaraci/cli_assistente.py` por AST
    (sem importar o módulo -- `ast.literal_eval`, não `exec`/`import`)."""
    fonte = _ler("src/guaraci/cli_assistente.py")
    arvore = ast.parse(fonte)
    for node in ast.walk(arvore):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "TECNICAS":
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", None) == "TECNICAS":
                    return ast.literal_eval(node.value)
    raise RuntimeError("TECNICAS não encontrado em cli_assistente.py — fonte mudou de forma, "
                        "gerador precisa ser ajustado (não gere nota com dado velho).")


def parse_status_tecnicas_11() -> list[dict[str, str]]:
    """Extrai a tabela 'Fechamento honesto do estado final das 11 técnicas'
    de `docs/PROGRESSO.md` (Passo 160): índice, nome, status, dataset,
    métrica, notas -- por técnica, na mesma ordem de `TECNICAS`."""
    texto = _ler("docs/PROGRESSO.md")
    m = re.search(r"^## Passo 160 —.*$", texto, re.M)
    if not m:
        return []
    bloco = texto[m.end():]
    linhas = [ln for ln in bloco.splitlines() if ln.startswith("| ")]
    linhas_dados = [ln for ln in linhas if re.match(r"^\|\s*\d+\s*\|", ln)]
    resultado = []
    for ln in linhas_dados:
        campos = [c.strip() for c in ln.strip("|").split("|")]
        if len(campos) < 6:
            continue
        indice, nome, status, dataset, metrica, notas = (
            re.sub(r"\*\*", "", c).strip() for c in campos[:6])
        resultado.append({
            "indice": indice, "nome": nome,
            "status": status, "dataset": dataset,
            "metrica": metrica, "notas": notas,
        })
    return resultado


@dataclass
class ModuloInfo:
    nome: str
    caminho: str
    docstring: str
    all_publico: list[str]
    depende_de: set[str] = field(default_factory=set)
    usado_por: set[str] = field(default_factory=set)


def parse_modulos() -> dict[str, ModuloInfo]:
    """1 entrada por `src/guaraci/*.py` (não-recursivo, Passo 167):
    docstring de módulo, `__all__` e dependências internas, todos por AST
    ou regex sobre o texto real -- nenhum nome de módulo é inventado."""
    modulos: dict[str, ModuloInfo] = {}
    caminhos = sorted((_RAIZ / "src" / "guaraci").glob("*.py"))
    nomes_validos = {p.stem for p in caminhos}
    for caminho in caminhos:
        nome = caminho.stem
        texto = caminho.read_text(encoding="utf-8")
        try:
            arvore = ast.parse(texto)
            doc = ast.get_docstring(arvore) or ""
        except SyntaxError:
            doc = ""
        all_publico: list[str] = []
        try:
            for node in ast.walk(arvore):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if getattr(t, "id", None) == "__all__":
                            try:
                                all_publico = list(ast.literal_eval(node.value))
                            except (ValueError, SyntaxError):
                                pass
        except NameError:
            pass
        deps: set[str] = set()
        for m in re.finditer(r"^from guaraci\.(\w+) import", texto, re.M):
            if m.group(1) in nomes_validos and m.group(1) != nome:
                deps.add(m.group(1))
        for m in re.finditer(r"^from \.(\w+) import", texto, re.M):
            if m.group(1) in nomes_validos and m.group(1) != nome:
                deps.add(m.group(1))
        modulos[nome] = ModuloInfo(
            nome=nome, caminho=f"src/guaraci/{caminho.name}",
            docstring=doc.strip(), all_publico=all_publico, depende_de=deps,
        )
    for info in modulos.values():
        for dep in info.depende_de:
            if dep in modulos:
                modulos[dep].usado_por.add(info.nome)
    return modulos


def _resumo_docstring(doc: str, max_linhas: int = 6) -> str:
    if not doc:
        return "(módulo sem docstring)"
    linhas = [ln for ln in doc.splitlines()]
    return "\n".join(linhas[:max_linhas]).strip()


@dataclass
class ParagrafoMarcado:
    arquivo: str
    linha: int
    marcador: str
    titulo: str
    texto: str


def _titulo_paragrafo(marcador: str, resto_paragrafo: str) -> str:
    """`**Decisão**` sozinho não distingue duas decisões diferentes -- se o
    marcador tem poucas palavras próprias, completa o título com as
    primeiras palavras do texto que vem depois dele na mesma frase."""
    marcador_limpo = re.sub(r"\*\*", "", marcador).strip(" :—-")
    if len(re.findall(r"\w+", marcador_limpo)) >= 4:
        return marcador_limpo
    resto = re.sub(r"^[:\s—-]+", "", resto_paragrafo)
    trecho = " ".join(re.findall(r"\S+", resto)[:10])
    trecho = re.sub(r"[.,;:]+$", "", trecho)
    return f"{marcador_limpo}: {trecho}" if trecho else marcador_limpo


def parse_paragrafos_marcados(padrao: re.Pattern, arquivos: list[str],
                               max_linhas: int = 8) -> list[ParagrafoMarcado]:
    """Varre `arquivos` por parágrafos (separados por linha em branco) cujo
    início casa `padrao` (ex.: `**Achado ...**`, `**RETRATAÇÃO ...**`).
    Cada parágrafo vira 1 candidato a nota -- resumido em até `max_linhas`
    (a fonte já escreve esses parágrafos curtos por convenção própria)."""
    achados: list[ParagrafoMarcado] = []
    for rel in arquivos:
        texto = _ler(rel)
        offset = 0
        for bloco in re.split(r"\n\s*\n", texto):
            linha_no = texto.count("\n", 0, offset) + 1
            offset += len(bloco) + 2
            bloco_stripped = bloco.strip()
            m = padrao.match(bloco_stripped)
            if not m:
                continue
            linhas = bloco_stripped.splitlines()[:max_linhas]
            titulo = _titulo_paragrafo(m.group(0), bloco_stripped[m.end():])
            achados.append(ParagrafoMarcado(
                arquivo=rel, linha=linha_no, marcador=m.group(0), titulo=titulo,
                texto="\n".join(linhas),
            ))
    return achados


def parse_compatibility_casos_especiais() -> list[dict[str, str]]:
    """`## Casos especiais documentados (não são dívida, são decisão)` de
    `docs/COMPATIBILITY.md` -- decisões já pré-curadas pela própria fonte,
    uma por item de lista de nível superior."""
    texto = _ler("docs/COMPATIBILITY.md")
    m = re.search(r"^## Casos especiais documentados.*?\n(.*?)(?=\n## )", texto,
                  re.S | re.M)
    if not m:
        return []
    corpo = m.group(1)
    itens = re.split(r"\n(?=- \*\*)", corpo.strip())
    resultado = []
    for item in itens:
        item = item.strip()
        if not item.startswith("- "):
            continue
        titulo_m = re.match(r"-\s+\*\*([^*]+)\*\*", item)
        titulo = titulo_m.group(1).strip(" —-*") if titulo_m else item[:60]
        resultado.append({"titulo": titulo, "texto": item})
    return resultado


# ═════════════════════════════════════════════════════════════════════════
#  Catálogo de conceitos (vocabulário do domínio -> módulo(s) que o
#  implementam). Nenhum número vive aqui -- só o mapeamento nome->módulo,
#  verificado contra o disco em tempo de geração.
# ═════════════════════════════════════════════════════════════════════════

CONCEITOS: list[dict[str, Any]] = [
    {"titulo": "Validação group-aware / anti-vazamento",
     "modulos": ["validacao_estatistica", "avaliacao_modelos"]},
    {"titulo": "Domínio de aplicabilidade",
     "modulos": ["hsi_applicability"]},
    {"titulo": "Conjunto aberto / conformal",
     "modulos": ["conformal"]},
    {"titulo": "Propagação de incerteza (Bonferroni)",
     "modulos": ["hsi_uncertainty"]},
    {"titulo": "Cobertura não-validável",
     "modulos": ["identificacao", "conformal"]},
    {"titulo": "Portão de aceite de correção de sinal",
     "modulos": ["portao_correcao_sinal"], "tecnicas": ["raman"]},
    {"titulo": "Transferência de calibração",
     "modulos": ["transferencia_calibracao"], "tecnicas": ["nir"]},
    {"titulo": "LOD/LOQ multivariado",
     "modulos": ["linearity"]},
    {"titulo": "Faixa de decisão",
     "modulos": ["predicao"]},
    {"titulo": "Amostragem ativa",
     "modulos": ["amostragem_ativa"]},
    {"titulo": "Multiway / PARAFAC",
     "modulos": ["eem_multiway", "hsi_multiway"], "tecnicas": ["fluorescencia"]},
    {"titulo": "MCR-ALS (aviso de escopo)",
     "modulos": ["mcr_als"]},
    {"titulo": "Seleção de variáveis",
     "modulos": ["selecao_variaveis"]},
    {"titulo": "Quality gate",
     "modulos": ["hsi_quality"]},
    {"titulo": "Sentinela de deriva",
     "modulos": ["sentinela_deriva"]},
]


# ═════════════════════════════════════════════════════════════════════════
#  Modalidades de entrada fora do catálogo `cli_assistente.TECNICAS`
#
#  `TECNICAS` (11 chaves) não é a lista completa do que o Guaraci aceita
#  como entrada -- `Config.mode` (src/guaraci/config.py) declara
#  "dx" | "csv" | "imagem" | "sintetico" | "hsi". `imagem` (colorimetria
#  digital) e `hsi` (imageamento hiperespectral) são tecnicamente reais
#  (módulo próprio, e no caso de HSI dataset público validado) mas nunca
#  apareceriam em 10-Tecnicas/ sem isto, porque não estão em `TECNICAS`.
#  `sintetico` fica de fora de propósito: é dado simulado para teste/
#  demonstração (`docs/MANUAL.md`: "Para testes/demonstração"), não uma
#  técnica analítica -- incluí-la aqui seria o tipo de afirmação que
#  "evidência ou silêncio" proíbe.
# ═════════════════════════════════════════════════════════════════════════

MODOS_FORA_DO_CATALOGO: list[dict[str, Any]] = [
    {"chave": "hsi", "nome": "HSI (Imageamento Hiperespectral)",
     "modulo_principal": "hsi_pipeline", "busca_validacao": "deephs",
     "modulos_relacionados": [
         "hsi_io", "hsi_quality", "hsi_segmentation", "hsi_pixels",
         "hsi_classification", "hsi_chemistry", "hsi_validation",
         "hsi_applicability", "hsi_uncertainty", "hsi_multiway",
         "hsi_resampling", "hsi_figures", "hsi_identification",
     ]},
    {"chave": "imagem", "nome": "Imagem (Colorimetria Digital)",
     "modulo_principal": "dados_imagem", "busca_validacao": None,
     "modulos_relacionados": []},
]


def gerar_tecnicas_fora_do_catalogo(modulos: dict[str, ModuloInfo],
                                     validacoes: dict[str, str]) -> dict[str, str]:
    plano: dict[str, str] = {}
    for item in MODOS_FORA_DO_CATALOGO:
        principal = modulos.get(item["modulo_principal"])
        if principal is None:
            continue
        corpo = [_resumo_docstring(principal.docstring, max_linhas=10)]
        relacionados = [m for m in item["modulos_relacionados"] if m in modulos]
        if relacionados:
            corpo.append("\n## Módulos relacionados\n" +
                          "\n".join(f"- {_wikilink(f'{m}.py')}" for m in relacionados))
        nota_validacao_rel = None
        if item["busca_validacao"]:
            nota_validacao_rel = next(
                (rel for rel in validacoes if item["busca_validacao"] in rel.lower()), None)
        if nota_validacao_rel:
            corpo.append("\n## Validação pública\n- " +
                          _wikilink(Path(nota_validacao_rel).stem))
        else:
            corpo.append("\n## Validação pública\nSem dataset público validado registrado "
                          "para esta modalidade em `docs/VALIDACAO_PUBLICA.md` nesta rodada.")
        link_principal = _wikilink(f"{item['modulo_principal']}.py")
        corpo.append(f"\n## Ver também\n- {link_principal}")
        tags = ["tecnica", "fora-do-catalogo"]
        if not nota_validacao_rel and item["busca_validacao"] is None:
            tags.append("pendente")
        conteudo = _nota(
            titulo=item["nome"], tags=tags,
            fonte=[principal.caminho, "src/guaraci/config.py"],
            corpo="\n".join(corpo),
        )
        plano[f"10-Tecnicas/{_slug(item['nome'])}.md"] = conteudo
    return plano


# ═════════════════════════════════════════════════════════════════════════
#  Geração de notas — 10-Tecnicas
# ═════════════════════════════════════════════════════════════════════════

def gerar_tecnicas(tecnicas: dict[str, Any], status11: list[dict[str, str]],
                    modulos: dict[str, ModuloInfo]) -> dict[str, str]:
    plano: dict[str, str] = {}
    conceitos_por_tecnica: dict[str, list[str]] = {}
    for c in CONCEITOS:
        for t in c.get("tecnicas", []):
            conceitos_por_tecnica.setdefault(t, []).append(c["titulo"])
    ordem = list(tecnicas.keys())
    for i, chave in enumerate(ordem):
        dados = tecnicas[chave]
        pt = dados.get("PT", {})
        nome = pt.get("nome", chave)
        status_row = status11[i] if i < len(status11) else None
        corpo = [
            f"**Faixa:** {pt.get('faixa', '—')}",
            f"**Pré-processamento recomendado:** {pt.get('preproc_rec', '—')}",
            "",
            pt.get("desc", ""),
        ]
        if status_row:
            corpo += [
                "",
                f"## Estado de validação (técnica #{status_row['indice']} do menu)",
                f"**Status:** {status_row['status']}",
                f"**Dataset:** {status_row['dataset']}",
                f"**Métrica:** {status_row['metrica']}",
            ]
            if status_row["notas"].strip(" —"):
                corpo.append(f"**Notas:** {status_row['notas']}")
        corpo += [
            "",
            "Detalhe completo, licenças e reprodução: `docs/VALIDACAO_PUBLICA.md` "
            "e `docs/PROGRESSO.md` (Passo 160).",
        ]
        links = [_wikilink("preprocessamento.py")]
        links += [_wikilink_titulo(c) for c in conceitos_por_tecnica.get(chave, [])]
        corpo.append("\n## Ver também\n" + "\n".join(f"- {l}" for l in links))
        conteudo = _nota(
            titulo=nome,
            tags=["tecnica"] + (["pendente"] if status_row and "aguardando" in
                                 status_row["dataset"].lower() else []),
            fonte=["src/guaraci/cli_assistente.py", "docs/PROGRESSO.md#Passo-160",
                   "docs/VALIDACAO_PUBLICA.md"],
            corpo="\n".join(corpo),
            status=status_row["status"].replace(":", "").strip() if status_row else None,
        )
        plano[f"10-Tecnicas/{_slug(nome)}.md"] = conteudo
    return plano


# ═════════════════════════════════════════════════════════════════════════
#  Geração de notas — 20-Modulos
# ═════════════════════════════════════════════════════════════════════════

def gerar_modulos(modulos: dict[str, ModuloInfo]) -> dict[str, str]:
    plano: dict[str, str] = {}
    conceitos_por_modulo: dict[str, list[str]] = {}
    for c in CONCEITOS:
        for m in c["modulos"]:
            conceitos_por_modulo.setdefault(m, []).append(c["titulo"])
    nomes_modulos = set(modulos)
    for nome, info in sorted(modulos.items()):
        corpo = [_autolinkar_modulos(_resumo_docstring(info.docstring), nomes_modulos)]
        if info.all_publico:
            corpo.append("\n## Exporta (`__all__`)\n" +
                          "\n".join(f"- `{n}`" for n in info.all_publico))
        if info.depende_de:
            corpo.append("\n## Depende de\n" +
                          "\n".join(f"- {_wikilink(f'{d}.py')}" for d in sorted(info.depende_de)))
        if info.usado_por:
            corpo.append("\n## Usado por\n" +
                          "\n".join(f"- {_wikilink(f'{u}.py')}" for u in sorted(info.usado_por)))
        conceitos = conceitos_por_modulo.get(nome, [])
        if conceitos:
            corpo.append("\n## Conceitos implementados\n" +
                          "\n".join(f"- {_wikilink_titulo(c)}" for c in conceitos))
        conteudo = _nota(
            titulo=f"{nome}.py", tags=["modulo"], fonte=info.caminho,
            corpo="\n".join(corpo),
        )
        plano[f"20-Modulos/{nome}.py.md"] = conteudo
    return plano


# ═════════════════════════════════════════════════════════════════════════
#  Geração de notas — 30-Conceitos
# ═════════════════════════════════════════════════════════════════════════

def gerar_conceitos(modulos: dict[str, ModuloInfo],
                     tecnicas: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    plano: dict[str, str] = {}
    avisos: list[str] = []
    nomes_modulos = set(modulos)
    for c in CONCEITOS:
        mods_existentes = [m for m in c["modulos"] if m in modulos]
        if not mods_existentes:
            avisos.append(f"conceito '{c['titulo']}': nenhum dos módulos "
                           f"{c['modulos']} existe mais em src/guaraci/ — nota omitida")
            continue
        partes = []
        for m in mods_existentes:
            resumo = _resumo_docstring(modulos[m].docstring, max_linhas=4)
            partes.append(_autolinkar_modulos(resumo, nomes_modulos))
        corpo = "\n\n".join(partes)
        corpo += "\n\n## Implementado em\n" + "\n".join(
            f"- {_wikilink(f'{m}.py')}" for m in mods_existentes)
        tecnicas_relacionadas = [t for t in c.get("tecnicas", []) if t in tecnicas]
        if tecnicas_relacionadas:
            corpo += "\n\n## Aplicável às técnicas\n" + "\n".join(
                f"- {_wikilink_titulo(tecnicas[t].get('PT', {}).get('nome', t))}"
                for t in tecnicas_relacionadas)
        conteudo = _nota(
            titulo=c["titulo"], tags=["conceito"],
            fonte=[modulos[m].caminho for m in mods_existentes],
            corpo=corpo,
        )
        plano[f"30-Conceitos/{_slug(c['titulo'])}.md"] = conteudo
    return plano, avisos


# ═════════════════════════════════════════════════════════════════════════
#  Geração de notas — 40-Validacoes
# ═════════════════════════════════════════════════════════════════════════

def parse_tabela_consolidada() -> list[dict[str, str]]:
    texto = _ler("docs/VALIDACAO_PUBLICA.md")
    m = re.search(r"^## 1\. Tabela consolidada\n\n(.*?)\n\n", texto, re.S | re.M)
    if not m:
        return []
    linhas = [ln for ln in m.group(1).splitlines() if ln.startswith("|")]
    linhas_dados = linhas[2:]  # pula cabeçalho + separador
    cabecalho = ["dataset", "matriz", "n", "canais", "alvo", "metrica",
                 "referencia", "estado"]
    resultado = []
    for ln in linhas_dados:
        campos = [c.strip() for c in ln.strip("|").split("|")]
        if len(campos) < len(cabecalho):
            continue
        resultado.append(dict(zip(cabecalho, campos)))
    return resultado


def gerar_validacoes(tabela: list[dict[str, str]]) -> dict[str, str]:
    plano: dict[str, str] = {}
    for row in tabela:
        titulo = re.sub(r"\*\*", "", row["dataset"])
        titulo = re.sub(r"\s+", " ", titulo).strip()
        corpo = [
            f"**Matriz:** {row['matriz']}",
            f"**n:** {row['n']}",
            f"**Canais/faixa:** {row['canais']}",
            f"**Alvo:** {row['alvo']}",
            f"**Métrica obtida:** {row['metrica']}",
            f"**Referência:** {row['referencia']}",
            f"**Estado:** {row['estado']}",
            "",
            "Detalhe completo (licença, arquivos, reprodução): "
            "`docs/VALIDACAO_PUBLICA.md` §1, e a subseção correspondente "
            "(`datasets/README.md` para o script de download).",
        ]
        conteudo = _nota(
            titulo=titulo[:80], tags=["validacao"],
            fonte="docs/VALIDACAO_PUBLICA.md#1-Tabela-consolidada",
            corpo="\n".join(corpo),
        )
        plano[f"40-Validacoes/{_slug(titulo)}.md"] = conteudo
    return plano


# ═════════════════════════════════════════════════════════════════════════
#  Geração de notas — 50-Decisoes e 60-Achados
# ═════════════════════════════════════════════════════════════════════════

_PADRAO_ACHADO = re.compile(
    r"^-?\s*\*\*(RETRATA[CÇ][AÃ]O[^*]*|Retrata[çc][aã]o[^*]*|Achado[^*]*"
    r"|Bug real[^*]*|BUG REAL[^*]*)\*\*", re.M)
_PADRAO_DECISAO = re.compile(
    r"^-?\s*\*\*(Decis[aã]o[^*]*)\*\*", re.M)

_ARQUIVOS_ACHADOS_DECISOES = ["docs/PROGRESSO.md", "docs/VALIDACAO_PUBLICA.md"]


@dataclass
class BlocoPasso:
    titulo: str
    linha_inicio: int  # 1-based, linha do cabeçalho "## Passo ..."
    linha_fim: int  # 1-based, exclusiva (linha do próximo cabeçalho ou EOF+1)
    corpo: str


def parse_blocos_passo(rel: str = "docs/PROGRESSO.md") -> list[BlocoPasso]:
    """Cada `## Passo ...` de `docs/PROGRESSO.md` vira 1 bloco, com o range
    de linhas até o próximo `## ` (ou fim do arquivo). Usado para garantir
    que TODO passo tenha alguma nota em `50-Decisoes/`/`60-Achados/`
    (Passo 172) -- mesmo que só um resumo mínimo quando nenhum parágrafo
    marcado (`**Achado**`/`**Decisão**`/...) cair dentro do bloco."""
    texto = _ler(rel)
    linhas = texto.splitlines()
    indices = [i for i, ln in enumerate(linhas) if ln.startswith("## Passo")]
    blocos = []
    for pos, i in enumerate(indices):
        fim = indices[pos + 1] if pos + 1 < len(indices) else len(linhas)
        titulo = linhas[i][3:].strip()
        corpo = "\n".join(linhas[i + 1:fim]).strip()
        blocos.append(BlocoPasso(titulo=titulo, linha_inicio=i + 1, linha_fim=fim + 1,
                                  corpo=corpo))
    return blocos


def _notas_de_paragrafos(paragrafos: list[ParagrafoMarcado], pasta: str,
                          tag: str, modulos_conhecidos: set[str]) -> dict[str, str]:
    plano: dict[str, str] = {}
    vistos: set[str] = set()
    for p in paragrafos:
        # Chave de deduplicação = conteúdo do parágrafo (não só o marcador
        # -- "**Decisão**"/"**Achado**" sozinhos se repetem para achados
        # completamente diferentes; o texto todo é que distingue repetição
        # real (mesmo achado citado em dois arquivos) de coincidência).
        chave = re.sub(r"\s+", " ", p.texto).strip().lower()[:200]
        if chave in vistos:
            continue
        vistos.add(chave)
        titulo = re.sub(r"\s+", " ", p.titulo).strip()
        conteudo = _nota(
            titulo=titulo, tags=[tag],
            fonte=f"{p.arquivo}:{p.linha}",
            corpo=_autolinkar_modulos(p.texto, modulos_conhecidos),
        )
        rel = f"{pasta}/{_slug(titulo)}.md"
        if rel in plano:
            rel = f"{pasta}/{_slug(titulo)}-{p.linha}.md"
        plano[rel] = conteudo
    return plano


def gerar_achados_e_decisoes(modulos_conhecidos: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Retorna `(achados, decisoes)`. Todo `## Passo` de `docs/PROGRESSO.md`
    (Passo 172 -- auditoria de cobertura) acaba representado em pelo menos
    uma nota de uma das duas pastas: os parágrafos já marcados pela própria
    fonte (`**Achado**`, `**RETRATAÇÃO**`, `**Decisão**`, ...) viram notas
    ricas; qualquer passo que não caia em nenhum desses parágrafos ganha
    uma nota-resumo mínima (tag `passo`, não `achado`) em `60-Achados/` --
    nunca fica sem nenhuma nota, mas também não finge ser um achado real."""
    achado_paragrafos = parse_paragrafos_marcados(_PADRAO_ACHADO, _ARQUIVOS_ACHADOS_DECISOES)
    decisao_paragrafos = parse_paragrafos_marcados(_PADRAO_DECISAO, _ARQUIVOS_ACHADOS_DECISOES)
    achados = _notas_de_paragrafos(achado_paragrafos, "60-Achados", "achado", modulos_conhecidos)
    decisoes = _notas_de_paragrafos(decisao_paragrafos, "50-Decisoes", "decisao", modulos_conhecidos)
    for item in parse_compatibility_casos_especiais():
        conteudo = _nota(
            titulo=item["titulo"], tags=["decisao"],
            fonte="docs/COMPATIBILITY.md#Casos-especiais-documentados",
            corpo=_autolinkar_modulos(item["texto"], modulos_conhecidos),
        )
        decisoes[f"50-Decisoes/{_slug(item['titulo'])}.md"] = conteudo

    linhas_cobertas = {
        p.linha for p in achado_paragrafos + decisao_paragrafos
        if p.arquivo == "docs/PROGRESSO.md"
    }
    for bloco in parse_blocos_passo():
        if any(bloco.linha_inicio <= ln < bloco.linha_fim for ln in linhas_cobertas):
            continue
        resumo = "\n".join(bloco.corpo.splitlines()[:5]).strip() or "(bloco sem corpo)"
        conteudo = _nota(
            titulo=bloco.titulo, tags=["passo"],
            fonte=f"docs/PROGRESSO.md:{bloco.linha_inicio}",
            corpo=_autolinkar_modulos(resumo, modulos_conhecidos),
        )
        rel = f"60-Achados/{_slug(bloco.titulo)}.md"
        if rel in achados:
            rel = f"60-Achados/{_slug(bloco.titulo)}-{bloco.linha_inicio}.md"
        achados[rel] = conteudo
    return achados, decisoes


# ═════════════════════════════════════════════════════════════════════════
#  MOCs (Maps of Content)
# ═════════════════════════════════════════════════════════════════════════

def gerar_mocs(tecnicas: dict[str, Any], modulos: dict[str, ModuloInfo],
                tecnicas_notas: dict[str, str], validacoes: dict[str, str],
                decisoes: dict[str, str], achados: dict[str, str]) -> dict[str, str]:
    plano: dict[str, str] = {}

    def _links(pasta_plano: dict[str, str]) -> str:
        nomes = sorted(Path(p).stem for p in pasta_plano)
        return "\n".join(f"- {_wikilink(n)}" for n in nomes)

    n_extras = len(MODOS_FORA_DO_CATALOGO)
    plano["00-MOC/MOC-Tecnicas.md"] = _nota(
        "MOC — Técnicas", ["moc"], "src/guaraci/cli_assistente.py",
        f"As {len(tecnicas)} técnicas do catálogo `cli_assistente.TECNICAS` "
        f"+ {n_extras} modalidade(s) de entrada real(is) que existem em "
        "`Config.mode` mas não estão nesse catálogo (tag `fora-do-catalogo`"
        " nas notas correspondentes).\n\n" + _links(tecnicas_notas))

    plano["00-MOC/MOC-Arquitetura.md"] = _nota(
        "MOC — Arquitetura", ["moc"], "src/guaraci/",
        f"{len(modulos)} módulos em `src/guaraci/*.py`. Grafo completo de "
        "dependências: ver backlinks de cada nota de módulo.\n\n" +
        "\n".join(f"- {_wikilink(f'{n}.py')}" for n in sorted(modulos)))

    plano["00-MOC/MOC-Validacoes.md"] = _nota(
        "MOC — Validações públicas", ["moc"], "docs/VALIDACAO_PUBLICA.md",
        "Um dataset público por linha da tabela consolidada "
        "(`docs/VALIDACAO_PUBLICA.md` §1).\n\n" + _links(validacoes))

    plano["00-MOC/MOC-Decisoes.md"] = _nota(
        "MOC — Decisões", ["moc"], ["docs/PROGRESSO.md", "docs/COMPATIBILITY.md"],
        "Decisões de design com razão registrada.\n\n### Decisões\n" +
        _links(decisoes) + "\n\n### Achados e retratações\n" +
        _links(achados))

    plano["00-MOC/MOC-Guaraci.md"] = _nota(
        "MOC — Guaraci", ["moc"], "docs/INDICE_PROJETO.md",
        "Ponto de entrada. Este vault é GERADO por "
        "`scripts/gerar_vault_obsidian.py` a partir do repositório do "
        "Guaraci — ver `README-VAULT.md`.\n\n"
        f"- {_wikilink('Estado-Atual')}\n"
        f"- {_wikilink('MOC-Tecnicas')}\n"
        f"- {_wikilink('MOC-Arquitetura')}\n"
        f"- {_wikilink('MOC-Validacoes')}\n"
        f"- {_wikilink('MOC-Decisoes')}\n")

    return plano


# ═════════════════════════════════════════════════════════════════════════
#  Estado-Atual.md — snapshot vivo
# ═════════════════════════════════════════════════════════════════════════

def _contar_testes() -> str:
    total = 0
    for caminho in (_RAIZ / "tests").rglob("test_*.py"):
        try:
            texto = caminho.read_text(encoding="utf-8")
        except OSError:
            continue
        total += len(re.findall(r"^\s*def test_", texto, re.M))
    return str(total)


def gerar_estado_atual(status11: list[dict[str, str]]) -> dict[str, str]:
    funcionais = sum(1 for r in status11 if "Funcional" in r["status"])
    log_recente = _git("log", "-1", "--format=%h %ci %s")
    corpo = [
        f"**Commit mais recente:** `{log_recente}`",
        f"**Funções de teste (`def test_*`) em `tests/`:** {_contar_testes()}",
        f"**Técnicas com status 'Funcional' (de {len(status11)}):** {funcionais}",
        "",
        "## Estado por técnica (Passo 160)",
    ]
    for r in status11:
        corpo.append(f"- **{r['nome']}**: {r['status']}")
    corpo += [
        "",
        "Este é um snapshot no momento do commit acima — regenere o vault "
        "(`python scripts/gerar_vault_obsidian.py`) para atualizar.",
    ]
    conteudo = _nota(
        "Estado Atual", ["estado-vivo"],
        ["docs/PROGRESSO.md#Passo-160", "tests/"],
        "\n".join(corpo))
    return {"Estado-Atual.md": conteudo}


# ═════════════════════════════════════════════════════════════════════════
#  Canvas (Passo 170)
# ═════════════════════════════════════════════════════════════════════════

def _canvas_no(id_: str, x: int, y: int, w: int, h: int, texto: str,
               cor: str | None = None) -> dict[str, Any]:
    no = {"id": id_, "type": "text", "x": x, "y": y, "width": w, "height": h,
          "text": texto}
    if cor:
        no["color"] = cor
    return no


def _canvas_edge(id_: str, de: str, para: str, label: str | None = None) -> dict[str, Any]:
    e = {"id": id_, "fromNode": de, "fromSide": "right", "toNode": para,
         "toSide": "left"}
    if label:
        e["label"] = label
    return e


def gerar_canvas_fluxo_cego() -> dict[str, str]:
    nos = [
        _canvas_no("detectar", 0, 0, 260, 140,
                   "## Detectar\nTécnica + faixa espectral compatível "
                   "(`cli_assistente.TECNICAS`)"),
        _canvas_no("identificar", 340, 0, 260, 140,
                   "## Identificar\nEspécie / classe — DD-SIMCA, conjunto "
                   "aberto (`conformal.py`, `identificacao.py`)"),
        _canvas_no("gate1", 340, 180, 260, 80,
                   "⛔ Gate: cobertura não-validável reportada honestamente "
                   "quando n é pequeno", "1"),
        _canvas_no("quantificar", 680, 0, 260, 140,
                   "## Quantificar\nPLS/PLS-DA por espécie "
                   "(`pipeline.pls_regression_by_species`, `predicao.py`)"),
        _canvas_no("gate2", 680, 180, 260, 100,
                   "⛔ Gate: faixa de decisão contra LOD/LOQ "
                   "(`linearity.py`) + propagação de incerteza "
                   "(`hsi_uncertainty.py`, Bonferroni)", "1"),
        _canvas_no("saida", 1020, 0, 260, 140,
                   "## Saída\nRelatório com incerteza propagada "
                   "(`reports.py`, `resultados_io.py`)"),
    ]
    arestas = [
        _canvas_edge("e1", "detectar", "identificar"),
        _canvas_edge("e2", "identificar", "quantificar"),
        _canvas_edge("e3", "quantificar", "saida"),
        _canvas_edge("e4", "identificar", "gate1"),
        _canvas_edge("e5", "quantificar", "gate2"),
    ]
    return {"90-Canvas/Fluxo-Cego.canvas": json.dumps(
        {"nodes": nos, "edges": arestas}, ensure_ascii=False, indent=2)}


def gerar_canvas_arquitetura(modulos: dict[str, ModuloInfo]) -> dict[str, str]:
    etapas = [
        ("entrada", "Entrada", ["dados_io", "hsi_io", "eem_io", "gcms_io",
                                 "importadores_proprietarios"]),
        ("preproc", "Pré-processamento", ["preprocessamento", "mcr_als",
                                           "alinhamento_retencao"]),
        ("modelagem", "Modelagem", ["chemometric_stats", "classificadores",
                                     "predicao", "conformal"]),
        ("validacao", "Validação", ["validacao_estatistica", "avaliacao_modelos",
                                     "robustness", "linearity"]),
        ("saida", "Saída", ["reports", "resultados_io", "figuras"]),
    ]
    nos = []
    arestas = []
    x = 0
    anterior = None
    for chave, titulo, mods in etapas:
        existentes = [m for m in mods if m in modulos]
        texto = f"## {titulo}\n" + "\n".join(f"- `{m}.py`" for m in existentes)
        nos.append(_canvas_no(chave, x, 0, 280, 60 + 24 * len(existentes), texto))
        if anterior:
            arestas.append(_canvas_edge(f"e-{anterior}-{chave}", anterior, chave))
        anterior = chave
        x += 340
    return {"90-Canvas/Arquitetura-Geral.canvas": json.dumps(
        {"nodes": nos, "edges": arestas}, ensure_ascii=False, indent=2)}


_FAMILIAS_TECNICAS = [
    ("Vibracional", ["ft-nir", "nir", "mir", "raman"]),
    ("Luminescência", ["uv-vis", "fluorescencia"]),
    ("Cromatografia", ["hplc", "gc-ms", "ims"]),
    ("Ressonância / outras", ["nmr", "generico"]),
]


def gerar_canvas_mapa_tecnicas(tecnicas: dict[str, Any],
                                status11: list[dict[str, str]]) -> dict[str, str]:
    status_por_nome = {re.sub(r"[^a-z0-9]", "", r["nome"].lower()): r["status"]
                        for r in status11}

    nos = []
    y = 0
    for familia, chaves in _FAMILIAS_TECNICAS:
        x = 0
        for chave in chaves:
            if chave not in tecnicas:
                continue
            nome = tecnicas[chave].get("PT", {}).get("nome", chave)
            chave_norm = re.sub(r"[^a-z0-9]", "", nome.lower())
            status = next((s for k, s in status_por_nome.items() if k in chave_norm
                           or chave_norm in k), "status desconhecido")
            cor = "4" if "Funcional" in status else "5"
            nos.append(_canvas_no(f"{familia}-{chave}", x, y, 260, 100,
                                   f"**{nome}**\n{status}", cor))
            x += 300
        y += 160
    return {"90-Canvas/Mapa-Tecnicas.canvas": json.dumps(
        {"nodes": nos, "edges": []}, ensure_ascii=False, indent=2)}


# ═════════════════════════════════════════════════════════════════════════
#  README-VAULT.md
# ═════════════════════════════════════════════════════════════════════════

def gerar_readme_vault(contagens: dict[str, int]) -> dict[str, str]:
    corpo = f"""# README do Vault

Este vault é **gerado inteiramente por script**
(`scripts/gerar_vault_obsidian.py`, versionado no repositório do Guaraci)
a partir das fontes de verdade do repositório: `docs/PROGRESSO.md`,
`docs/VALIDACAO_PUBLICA.md`, `docs/COMPATIBILITY.md`,
`src/guaraci/cli_assistente.py` e `src/guaraci/*.py`.

**Não edite nada fora de `{_PASTA_PROTEGIDA}/` à mão.** Qualquer edição
manual em outra pasta é perdida na próxima regeneração — o script
recalcula o conteúdo inteiro a cada execução e arquiva (nunca apaga) o
que não existir mais nas fontes.

## Quando regenerar

```
python scripts/gerar_vault_obsidian.py
```

Regenere depois de: um bloco/passo novo em `docs/PROGRESSO.md`, uma
retratação, uma validação pública nova ou atualizada, ou uma mudança de
`__all__`/docstring em `src/guaraci/`.

## Como consultar

```
python scripts/consultar_vault.py "termo ou nome de nota"
python scripts/consultar_vault.py --cobertura
```

Navega o grafo de wikilinks a partir da nota mais relevante (não só
busca de texto isolada) e avisa explicitamente se o vault parece
desatualizado em relação ao commit atual do repositório.

## Pasta protegida

`{_PASTA_PROTEGIDA}/` é sua — o gerador nunca lê nem escreve nada lá.
Use-a para anotações pessoais, rascunhos, ligações manuais extras.

## Como o conteúdo é derivado

| Pasta | Fonte |
|---|---|
| `10-Tecnicas/` | `src/guaraci/cli_assistente.py` (catálogo `TECNICAS`, 11) + `docs/PROGRESSO.md` (Passo 160) + `Config.mode` (`config.py`) para as modalidades fora do catálogo (`imagem`, `hsi`) |
| `20-Modulos/` | `src/guaraci/*.py` — docstring de módulo, `__all__`, imports internos (introspecção estática) |
| `30-Conceitos/` | mapeamento conceito→módulo (neste script) + docstring do(s) módulo(s) |
| `40-Validacoes/` | `docs/VALIDACAO_PUBLICA.md` §1 (tabela consolidada), 1 nota por linha |
| `50-Decisoes/` | parágrafos `**Decisão...**` em `docs/PROGRESSO.md`/`docs/VALIDACAO_PUBLICA.md` + `docs/COMPATIBILITY.md` (casos especiais) |
| `60-Achados/` | parágrafos `**Achado...**`/`**RETRATAÇÃO...**`/`**Bug real...**` em `docs/PROGRESSO.md`/`docs/VALIDACAO_PUBLICA.md` + 1 resumo mínimo (tag `passo`, não `achado`) por `## Passo` que não caiu em nenhum parágrafo marcado — garante que todo passo tenha alguma nota (Passo 172) |
| `90-Canvas/` | gerados programaticamente a partir dos mesmos dados acima |
| `Estado-Atual.md` | commit HEAD + contagem de `def test_*` em `tests/` + Passo 160 |

Nenhum número (RMSEP, balanced accuracy, contagem de testes etc.)
aparece hardcoded no gerador — todos vêm de uma leitura de arquivo real
no momento da geração.

## Contagens desta geração

- {contagens.get('10-Tecnicas', 0)} técnicas
- {contagens.get('20-Modulos', 0)} módulos
- {contagens.get('30-Conceitos', 0)} conceitos
- {contagens.get('40-Validacoes', 0)} validações públicas
- {contagens.get('50-Decisoes', 0)} decisões
- {contagens.get('60-Achados', 0)} achados/retratações

Commit: `{_HEAD_HASH}` ({_HEAD_DATA}).
"""
    return {"README-VAULT.md": corpo}


# ═════════════════════════════════════════════════════════════════════════
#  Orquestração
# ═════════════════════════════════════════════════════════════════════════

def _resolve_vault_dir(cli_out: str | None) -> Path:
    import os
    if cli_out:
        return Path(cli_out).expanduser().resolve()
    if os.environ.get("GUARACI_VAULT_DIR"):
        return Path(os.environ["GUARACI_VAULT_DIR"]).expanduser().resolve()
    return (Path.home() / "GuaraciVault").resolve()


def montar_plano() -> tuple[dict[str, str], dict[str, int], list[str]]:
    tecnicas = parse_tecnicas_catalog()
    status11 = parse_status_tecnicas_11()
    modulos = parse_modulos()
    tabela_validacoes = parse_tabela_consolidada()

    plano: dict[str, str] = {}
    avisos: list[str] = []

    p_tecnicas = gerar_tecnicas(tecnicas, status11, modulos)
    p_modulos = gerar_modulos(modulos)
    p_conceitos, avisos_conceitos = gerar_conceitos(modulos, tecnicas)
    p_validacoes = gerar_validacoes(tabela_validacoes)
    p_tecnicas.update(gerar_tecnicas_fora_do_catalogo(modulos, p_validacoes))
    p_achados, p_decisoes = gerar_achados_e_decisoes(set(modulos))
    p_mocs = gerar_mocs(tecnicas, modulos, p_tecnicas, p_validacoes, p_decisoes, p_achados)
    p_estado = gerar_estado_atual(status11)
    p_canvas = {}
    p_canvas.update(gerar_canvas_fluxo_cego())
    p_canvas.update(gerar_canvas_arquitetura(modulos))
    p_canvas.update(gerar_canvas_mapa_tecnicas(tecnicas, status11))

    contagens = {
        "10-Tecnicas": len(p_tecnicas), "20-Modulos": len(p_modulos),
        "30-Conceitos": len(p_conceitos), "40-Validacoes": len(p_validacoes),
        "50-Decisoes": len(p_decisoes), "60-Achados": len(p_achados),
    }
    p_readme = gerar_readme_vault(contagens)

    for parte in (p_tecnicas, p_modulos, p_conceitos, p_validacoes, p_decisoes,
                  p_achados, p_mocs, p_estado, p_canvas, p_readme):
        plano.update(parte)

    avisos.extend(avisos_conceitos)
    if not tabela_validacoes:
        avisos.append("Tabela consolidada de docs/VALIDACAO_PUBLICA.md não encontrada "
                       "ou vazia — 40-Validacoes/ ficará vazio.")
    if not status11:
        avisos.append("Tabela do Passo 160 não encontrada em docs/PROGRESSO.md — "
                       "10-Tecnicas/ e Estado-Atual.md ficarão sem status por técnica.")

    return plano, contagens, avisos


def _garantir_pasta_protegida(vault_dir: Path) -> None:
    """Cria `70-Notas-Pessoais/` com um placeholder SÓ na primeira execução
    (a pasta ainda não existe). Depois disso o gerador nunca mais toca
    nela -- nem para criar, nem para ler, nem para escrever."""
    pasta = vault_dir / _PASTA_PROTEGIDA
    if pasta.exists():
        return
    pasta.mkdir(parents=True)
    (pasta / "README.md").write_text(
        "# Notas pessoais\n\n"
        "Esta pasta é sua. `scripts/gerar_vault_obsidian.py` nunca lê nem "
        "escreve nada aqui, em nenhuma execução -- pode apagar este "
        "arquivo e usar a pasta como quiser.\n",
        encoding="utf-8",
    )


def escrever_vault(vault_dir: Path, plano: dict[str, str]) -> tuple[int, int, int]:
    vault_dir.mkdir(parents=True, exist_ok=True)
    _garantir_pasta_protegida(vault_dir)
    manifesto_path = vault_dir / _MANIFESTO
    manifesto_antigo: list[str] = []
    if manifesto_path.exists():
        try:
            manifesto_antigo = json.loads(manifesto_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifesto_antigo = []

    escritos = 0
    inalterados = 0
    for rel, conteudo in plano.items():
        destino = vault_dir / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        if destino.exists() and destino.read_text(encoding="utf-8") == conteudo:
            inalterados += 1
            continue
        destino.write_text(conteudo, encoding="utf-8")
        escritos += 1

    novo_conjunto = set(plano.keys())
    arquivados = 0
    for rel in manifesto_antigo:
        if rel in novo_conjunto or rel.startswith(_PASTA_PROTEGIDA + "/"):
            continue
        origem = vault_dir / rel
        if not origem.exists():
            continue
        destino = vault_dir / "99-Arquivo" / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        origem.replace(destino)
        arquivados += 1

    manifesto_path.write_text(
        json.dumps(sorted(novo_conjunto), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return escritos, inalterados, arquivados


def _fixar_utf8_console() -> None:
    """No Windows, o console pode estar num codepage que não é UTF-8 e
    mostra acento como `�`. Mesmo ajuste de `src/guaraci/guaraci.py` --
    chamado só em `main()`, nunca na importação do módulo, para não afetar
    a captura de stdout dos testes."""
    if sys.platform != "win32":
        return
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (OSError, ValueError):
            pass


def main() -> int:
    _fixar_utf8_console()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="pasta de destino do vault")
    ap.add_argument("--check", action="store_true",
                     help="só gera em memória + guarda de privacidade, não escreve")
    args = ap.parse_args()

    plano, contagens, avisos = montar_plano()

    try:
        checar_conteudos_ou_falhar(plano)
    except VazamentoDePrivacidade as e:
        print("FALHA — guarda de privacidade acusou vazamento. Nada foi escrito.",
              file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 1

    for a in avisos:
        print(f"AVISO: {a}", file=sys.stderr)

    if args.check:
        print(f"OK (--check): {len(plano)} notas planejadas, guarda de privacidade limpa.")
        return 0

    vault_dir = _resolve_vault_dir(args.out)
    escritos, inalterados, arquivados = escrever_vault(vault_dir, plano)
    print(f"Vault gerado em {vault_dir}")
    print(f"  {escritos} arquivo(s) escrito(s)/atualizado(s)")
    print(f"  {inalterados} arquivo(s) sem mudança")
    print(f"  {arquivados} arquivo(s) arquivado(s) em 99-Arquivo/")
    for cat, n in contagens.items():
        print(f"  {cat}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
