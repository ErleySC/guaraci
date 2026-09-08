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

if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))

from guaraci.validacao_publica import parse_consolidated_table  # noqa: E402

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


#: Toda nota gerada (exceto `Autoria.md` nela mesma e `README-VAULT.md`,
#: que é documentação SOBRE o vault, não uma nota de conteúdo) carrega
#: este campo -- Passo 183/184: proveniência universal, "linkar tudo de
#: volta à autoria" mesmo que a nota seja extraída/copiada isoladamente.
_LINK_AUTORIA = '"[[Autoria]]"'


def _frontmatter(tags: list[str], fonte: str | list[str],
                  autor: str | None = _LINK_AUTORIA, **extra: Any) -> str:
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
    if autor is not None:
        linhas.append(f"autor: {autor}")
    for k, v in extra.items():
        if v is None:
            continue
        linhas.append(f"{k}: {v}")
    linhas.append("---")
    return "\n".join(linhas) + "\n"


def _nota(titulo: str, tags: list[str], fonte: str | list[str],
          corpo: str, autor: str | None = _LINK_AUTORIA, **extra: Any) -> str:
    return (_frontmatter(tags, fonte, autor=autor, **extra) +
            f"\n# {titulo}\n\n" + corpo.strip() + "\n")


def _yaml_str(s: str) -> str:
    """Escapa `s` como string YAML de aspas simples (`'a''b'` para `a'b`)
    -- usado para valores de frontmatter que podem conter `:`/`[`/`]`
    (assinatura de função, wikilink)."""
    return "'" + s.replace("'", "''") + "'"


def _wikilink(nome: str, relacao: str | None = None) -> str:
    """Link para uma nota cujo NOME DE ARQUIVO já é `nome` (módulos
    `x.py.md`, MOCs, `Estado-Atual.md`, ou qualquer stem já calculado por
    `_slug` em outro lugar). Não usar com título humano livre -- ver
    `_wikilink_titulo`.

    `relacao`, quando dado, é anexado como `— relação` (Passo 179/180:
    todo link do vault explica por que existe, não fica nu)."""
    base = f"[[{nome}]]"
    return f"{base} — {relacao}" if relacao else base


def _wikilink_titulo(titulo: str, relacao: str | None = None) -> str:
    """Link para uma nota cujo arquivo foi nomeado com `_slug(titulo)`
    (técnicas, conceitos) -- usa alias `[[slug|Título Humano]]` para que o
    link resolva ao arquivo real e ainda mostre o título legível. Ver
    `_wikilink` para `relacao`."""
    base = f"[[{_slug(titulo)}|{titulo}]]"
    return f"{base} — {relacao}" if relacao else base


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
                          "\n".join(f"- {_wikilink(f'{m}.py', 'implementa parte desta modalidade')}"
                                    for m in relacionados))
        nota_validacao_rel = None
        if item["busca_validacao"]:
            nota_validacao_rel = next(
                (rel for rel in validacoes if item["busca_validacao"] in rel.lower()), None)
        if nota_validacao_rel:
            corpo.append("\n## Validação pública\n- " +
                          _wikilink(Path(nota_validacao_rel).stem, "valida esta modalidade"))
        else:
            corpo.append("\n## Validação pública\nSem dataset público validado registrado "
                          "para esta modalidade em `docs/VALIDACAO_PUBLICA.md` nesta rodada.")
        link_principal = _wikilink(f"{item['modulo_principal']}.py", "módulo de entrada desta modalidade")
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
        links = [_wikilink("preprocessamento.py", "pré-processamento recomendado para esta técnica")]
        links += [_wikilink_titulo(c, "conceito aplicável a esta técnica")
                  for c in conceitos_por_tecnica.get(chave, [])]
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

def gerar_modulos(modulos: dict[str, ModuloInfo],
                   funcoes_por_modulo: dict[str, str] | None = None) -> dict[str, str]:
    plano: dict[str, str] = {}
    conceitos_por_modulo: dict[str, list[str]] = {}
    for c in CONCEITOS:
        for m in c["modulos"]:
            conceitos_por_modulo.setdefault(m, []).append(c["titulo"])
    funcoes_por_modulo = funcoes_por_modulo or {}
    nomes_modulos = set(modulos)
    for nome, info in sorted(modulos.items()):
        corpo = [_autolinkar_modulos(_resumo_docstring(info.docstring), nomes_modulos)]
        if info.all_publico:
            corpo.append("\n## Exporta (`__all__`)\n" +
                          "\n".join(f"- `{n}`" for n in info.all_publico))
        if info.depende_de:
            corpo.append("\n## Depende de\n" +
                          "\n".join(f"- {_wikilink(f'{d}.py', 'depende de')}"
                                    for d in sorted(info.depende_de)))
        if info.usado_por:
            corpo.append("\n## Usado por\n" +
                          "\n".join(f"- {_wikilink(f'{u}.py', 'usado por')}"
                                    for u in sorted(info.usado_por)))
        conceitos = conceitos_por_modulo.get(nome, [])
        if conceitos:
            corpo.append("\n## Conceitos implementados\n" +
                          "\n".join(f"- {_wikilink_titulo(c, 'implementa')}" for c in conceitos))
        funcao_principal = funcoes_por_modulo.get(nome)
        if funcao_principal:
            corpo.append("\n## Função de entrada\n- " +
                          _wikilink(funcao_principal, "função pública principal deste módulo"))
        conteudo = _nota(
            titulo=f"{nome}.py", tags=["modulo"], fonte=info.caminho,
            corpo="\n".join(corpo),
        )
        plano[f"20-Modulos/{nome}.py.md"] = conteudo
    return plano


# ═════════════════════════════════════════════════════════════════════════
#  Geração de notas — 25-Funcoes
#
#  Escopo: a instrução original pedia 1 nota por nome em `__all__` que
#  fosse função (não classe/constante) -- medido: 290 funções em 65
#  módulos (`figuras.py` sozinho: 38). Volume muito maior que o esperado
#  pela granularidade proposta -- decisão tomada COM o usuário em
#  2026-09-06 (regra de pausa (a) da instrução): reduzir para 1 função de
#  entrada por módulo, a PRIMEIRA função de `__all__` na ordem em que o
#  próprio módulo a declara (critério verificável no código-fonte, não
#  suposição de "qual é a mais importante").
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class FuncaoInfo:
    nome: str
    modulo: str
    caminho: str
    linha: int
    assinatura: str
    resumo_docstring: str
    testada_por: list[str]


def _assinatura_funcao(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Assinatura completa (parâmetros com tipo, retorno) por AST +
    `ast.unparse` -- nunca por `inspect`/import (mesma disciplina de
    `parse_tecnicas_catalog`: nunca importar o módulo que está sendo
    descrito)."""
    args = node.args
    partes: list[str] = []
    n_sem_default = len(args.args) - len(args.defaults)
    defaults = [None] * n_sem_default + list(args.defaults)
    for arg, default in zip(args.args, defaults):
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        dflt = f" = {ast.unparse(default)}" if default is not None else ""
        partes.append(f"{arg.arg}{ann}{dflt}")
    if args.vararg:
        ann = f": {ast.unparse(args.vararg.annotation)}" if args.vararg.annotation else ""
        partes.append(f"*{args.vararg.arg}{ann}")
    elif args.kwonlyargs:
        partes.append("*")
    for kwarg, default in zip(args.kwonlyargs, args.kw_defaults):
        ann = f": {ast.unparse(kwarg.annotation)}" if kwarg.annotation else ""
        dflt = f" = {ast.unparse(default)}" if default is not None else ""
        partes.append(f"{kwarg.arg}{ann}{dflt}")
    if args.kwarg:
        ann = f": {ast.unparse(args.kwarg.annotation)}" if args.kwarg.annotation else ""
        partes.append(f"**{args.kwarg.arg}{ann}")
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    prefixo = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefixo} {node.name}(" + ", ".join(partes) + f"){ret}"


def _primeiro_paragrafo_docstring(doc: str) -> str:
    if not doc:
        return "(função sem docstring)"
    paragrafos = re.split(r"\n\s*\n", doc.strip())
    return re.sub(r"\s+", " ", paragrafos[0]).strip()


def _carregar_arquivos_teste() -> list[tuple[str, str]]:
    resultado = []
    for caminho in sorted((_RAIZ / "tests").rglob("test_*.py")):
        try:
            texto = caminho.read_text(encoding="utf-8")
        except OSError:
            continue
        resultado.append((str(caminho.relative_to(_RAIZ)).replace("\\", "/"), texto))
    return resultado


def _testada_por(nome_funcao: str, arquivos_teste: list[tuple[str, str]]) -> list[str]:
    """Chamada DIRETA ao nome da função em algum `tests/test_*.py` -- real
    (busca em texto), não suposição. Não pega cobertura indireta (função
    chamada só por outra função de produção, nunca pelo nome dela num
    teste)."""
    padrao = re.compile(rf"(?<!\w){re.escape(nome_funcao)}\s*\(")
    return [rel for rel, texto in arquivos_teste if padrao.search(texto)]


def parse_funcoes_principais(modulos: dict[str, ModuloInfo]) -> list[FuncaoInfo]:
    arquivos_teste = _carregar_arquivos_teste()
    resultado: list[FuncaoInfo] = []
    for nome, info in sorted(modulos.items()):
        caminho_abs = _RAIZ / info.caminho
        try:
            texto = caminho_abs.read_text(encoding="utf-8")
            arvore = ast.parse(texto)
        except (OSError, SyntaxError):
            continue
        funcoes_top_level = {
            n.name: n for n in arvore.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        principal_nome = next((n for n in info.all_publico if n in funcoes_top_level), None)
        if principal_nome is None:
            continue
        node = funcoes_top_level[principal_nome]
        doc = ast.get_docstring(node) or ""
        resultado.append(FuncaoInfo(
            nome=principal_nome, modulo=nome, caminho=info.caminho,
            linha=node.lineno, assinatura=_assinatura_funcao(node),
            resumo_docstring=_primeiro_paragrafo_docstring(doc),
            testada_por=_testada_por(principal_nome, arquivos_teste),
        ))
    return resultado


def gerar_funcoes(funcoes: list[FuncaoInfo]) -> dict[str, str]:
    plano: dict[str, str] = {}
    for f in funcoes:
        corpo = [
            f"`{f.assinatura}`",
            "",
            f.resumo_docstring,
            "",
            f"**Localização:** `{f.caminho}:{f.linha}`",
            "",
            "## Módulo\n- " + _wikilink(f"{f.modulo}.py", "módulo que define esta função"),
        ]
        if f.testada_por:
            corpo.append("\n## Testada por\n" + "\n".join(
                f"- `{t}` — chama `{f.nome}(` diretamente" for t in f.testada_por))
        else:
            corpo.append(
                f"\n## Testada por\nNenhuma chamada direta a `{f.nome}(` encontrada em "
                "`tests/test_*.py` -- pode estar coberta indiretamente (via outra função "
                "de produção que a chama) ou sem teste direto."
            )
        conteudo = _nota(
            titulo=f.nome, tags=["funcao"], fonte=f"{f.caminho}:{f.linha}",
            corpo="\n".join(corpo),
            modulo=_yaml_str(f"[[{f.modulo}.py]]"),
            assinatura=_yaml_str(f.assinatura),
        )
        plano[f"25-Funcoes/{f.nome}.md"] = conteudo
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
            f"- {_wikilink(f'{m}.py', 'implementa este conceito')}" for m in mods_existentes)
        tecnicas_relacionadas = [t for t in c.get("tecnicas", []) if t in tecnicas]
        if tecnicas_relacionadas:
            corpo += "\n\n## Aplicável às técnicas\n" + "\n".join(
                f"- {_wikilink_titulo(tecnicas[t].get('PT', {}).get('nome', t), 'usa este conceito')}"
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
    """Wrapper de I/O sobre `guaraci.validacao_publica` — o parsing em si
    mora la' desde 2026-09-08, porque o painel de status do app web precisa
    da MESMA leitura (duas cópias divergiriam na primeira mudança da
    tabela)."""
    return parse_consolidated_table(_ler("docs/VALIDACAO_PUBLICA.md"))


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
#  Geração de notas — 05-Identidade
# ═════════════════════════════════════════════════════════════════════════

def _secao_markdown(texto: str, cabecalho: str) -> str:
    """Corpo de uma seção markdown identificada por `cabecalho` (regex de
    início de linha, ex. `r"## 1\\. Paleta"`) até o próximo cabeçalho de
    qualquer nível (ou fim do arquivo)."""
    m = re.search(rf"^{cabecalho}[^\n]*\n(.*?)(?=\n#{{1,6}} |\Z)", texto, re.M | re.S)
    return m.group(1).strip() if m else ""


def parse_paleta_mascote() -> list[dict[str, str]]:
    """Tabela de `docs/DESIGN.md` §1 -- valores medidos por amostragem de
    pixel real do ícone, não estimados. Este parser NUNCA reescreve os
    hex aqui; se a tabela mudar de forma, a nota fica sem paleta (fail
    silencioso controlado) em vez de mostrar um valor desatualizado."""
    corpo = _secao_markdown(_ler("docs/DESIGN.md"), r"## 1\. Paleta extra.da da mascote")
    linhas = [ln for ln in corpo.splitlines() if ln.startswith("|")][2:]
    resultado = []
    for ln in linhas:
        campos = [c.strip() for c in ln.strip("|").split("|")]
        if len(campos) < 4:
            continue
        resultado.append({"papel": campos[0], "hex_moda": campos[1].strip("`"),
                           "hex_media": campos[2].strip("`"), "onde": campos[3]})
    return resultado


def parse_regras_uso_cor() -> list[str]:
    """Bullets de `### 1.2 Regras de uso` -- cada um pode quebrar em 2+
    linhas no markdown fonte (linha de continuação sem `- `); junta as
    continuações na mesma entrada, senão a regra fica cortada no meio."""
    corpo = _secao_markdown(_ler("docs/DESIGN.md"), r"### 1\.2 Regras de uso")
    bullets: list[str] = []
    for ln in corpo.splitlines():
        ln_stripped = ln.strip()
        if not ln_stripped:
            continue
        if ln_stripped.startswith("- "):
            bullets.append(ln_stripped)
        elif bullets:
            bullets[-1] += " " + ln_stripped
    return bullets


def parse_tipografia_design() -> str:
    return _secao_markdown(_ler("docs/DESIGN.md"), r"## 2\. Tipografia")


def parse_tom_de_voz_evidencias() -> dict[str, str]:
    """Exemplos reais de tom direto/honestidade de `docs/VALIDACAO_PUBLICA.md`
    -- lidos do arquivo, não retdigitados de memória (Passo 177)."""
    texto = _ler("docs/VALIDACAO_PUBLICA.md")
    abertura = _secao_markdown(texto, r"# Valida[cç][aã]o p[uú]blica")
    primeiro_paragrafo = abertura.split("\n\nReproduzir:")[0].strip()
    m_fecho = re.search(
        r"^O RMSEP do Corn está no meio da faixa publicada.*?desempenho\.",
        texto, re.M | re.S)
    return {
        "abertura": primeiro_paragrafo,
        "fechamento": m_fecho.group(0).strip() if m_fecho else "",
    }


def gerar_identidade(achados: dict[str, str], decisoes: dict[str, str]) -> dict[str, str]:
    plano: dict[str, str] = {}

    # -- Missao.md ----------------------------------------------------------
    # Os 3 compromissos são citados verbatim pela própria instrução desta
    # auditoria (2026-09-06) -- "Plano de Evolução" como documento próprio
    # não existe no repositório (verificado por grep antes de escrever
    # esta nota); em vez de inventar essa fonte, cada compromisso é
    # ancorado a uma nota real de 60-Achados/50-Decisoes que o comprova.
    compromissos = [
        ("Nunca reportar métrica potencialmente inflada por vazamento",
         next((Path(r).stem for r in achados
               if "retratacao-metodologica-interna-antes-de-publicar-qualquer-numero" in r),
              None)),
        ("Nunca recomendar método sem prova sob validação bloqueada",
         next((Path(r).stem for r in decisoes
               if "decisao-nenhum-dataset-hplc-compativel" in r), None)),
        ("Nunca esconder resultado negativo",
         next((Path(r).stem for r in achados
               if "achado-real-negativo-registrado-honesto" in r), None)),
    ]
    linhas_compromissos = []
    for texto_compromisso, stem in compromissos:
        if stem:
            linhas_compromissos.append(
                f"- **{texto_compromisso}** — exemplo real: "
                f"{_wikilink(stem, 'prova este compromisso')}")
        else:
            linhas_compromissos.append(
                f"- **{texto_compromisso}** — nenhuma nota-âncora encontrada nesta "
                "geração (achado/decisão pode ter sido reformulado; revisar).")
    corpo_missao = (
        "Não existe, no repositório, um documento único chamado \"Plano de "
        "Evolução\" (verificado por busca de texto antes de escrever esta "
        "nota). Os três compromissos abaixo vêm diretamente da instrução "
        "desta auditoria do vault (2026-09-06) -- cada um ancorado a uma "
        "nota real, não repetido.\n\n" + "\n".join(linhas_compromissos))
    plano["05-Identidade/Missao.md"] = _nota(
        "Missão", ["identidade"], "docs/VALIDACAO_PUBLICA.md", corpo_missao)

    # -- Identidade-Visual.md ------------------------------------------------
    paleta = parse_paleta_mascote()
    linhas_paleta = ["| Papel | Hex (moda) | Hex (média) | Onde aparece na mascote |",
                      "|---|---|---|---|"]
    for p in paleta:
        linhas_paleta.append(
            f"| {p['papel']} | `{p['hex_moda']}` | `{p['hex_media']}` | {p['onde']} |")
    regras = parse_regras_uso_cor()
    tipografia = parse_tipografia_design()
    corpo_visual = (
        "Paleta medida por amostragem de pixel real do ícone "
        "(`assets/guaraci_icon.png`), documentada em `docs/DESIGN.md` §1 -- "
        "não são valores \"de olho\". **Atualização registrada em "
        "`docs/DESIGN.md`, 2026-09-01:** o caminho A (migração completa "
        "para esta paleta, laranja como cor primária) foi aprovado e já "
        "está implementado em `design_tokens.py`/`guaraci_theme.py` -- "
        "não é mais só uma proposta.\n\n"
        "## Paleta\n" + "\n".join(linhas_paleta) +
        ("\n\n## Regras de uso (a cor carrega significado, não decoração)\n" +
         "\n".join(regras) if regras else "") +
        (f"\n\n## Tipografia\n{tipografia}" if tipografia else "") +
        "\n\n## Ver também\n"
        f"- {_wikilink('design_tokens.py', 'implementa esta paleta em código')}\n"
        f"- {_wikilink('guaraci_theme.py', 'reexporta a paleta para o tema Rich do CLI')}\n"
        f"- {_wikilink('Mascote', 'origem visual desta paleta')}")
    plano["05-Identidade/Identidade-Visual.md"] = _nota(
        "Identidade Visual", ["identidade"],
        ["docs/DESIGN.md", "src/guaraci/design_tokens.py"], corpo_visual)

    # -- Mascote.md -----------------------------------------------------------
    corpo_mascote = (
        "Cachorro cientista de jaleco branco, cocar com folha/flor "
        "(elemento indígena/amazônico), sentado com um notebook (tela "
        "mostrando um sol estilizado e um gráfico de barras) e um "
        "erlenmeyer ao lado -- tudo dentro da silhueta maior de um "
        "frasco/balão com gradiente laranja→verde e moldura dourada. "
        "Descrição confirmada por inspeção direta de "
        "`assets/guaraci_icon.png` (não só a partir do texto de "
        "`docs/DESIGN.md`).\n\n"
        "**O que representa:** identidade regional amazônica (o cocar, a "
        "paleta laranja/verde) combinada com ciência aplicada (jaleco, "
        "notebook, gráfico, vidraria de laboratório) -- a mesma "
        "combinação que dá nome ao projeto (Guaraci, divindade solar "
        "Tupi-Guarani) e à sua origem real (autenticação de óleos "
        "amazônicos por FT-NIR).\n\n"
        "**Elemento reaproveitável:** o sol no notebook é o elemento mais "
        "simples de isolar como marca gráfica -- silhueta reconhecível em "
        "tamanho pequeno, já carrega a cor dourada (`docs/DESIGN.md` "
        "§1.3) -- e já é o favicon/ícone do app através deste mesmo "
        "arquivo.\n\n"
        "A imagem está versionada em `assets/guaraci_icon.png` (e "
        "`assets/guaraci_icon.ico` para o ícone do executável Windows) -- "
        "link ao arquivo, não embutida nesta nota.\n\n"
        "## Ver também\n"
        f"- {_wikilink('Identidade-Visual', 'paleta extraída desta mascote')}")
    plano["05-Identidade/Mascote.md"] = _nota(
        "Mascote", ["identidade"], "docs/DESIGN.md", corpo_mascote)

    # -- Tom-de-Voz.md --------------------------------------------------------
    evidencias = parse_tom_de_voz_evidencias()
    partes_tom = [
        "Direto, sem alegação sem prova, nunca insinua conformidade "
        "regulatória -- resultado negativo é publicado com o mesmo peso "
        "que resultado positivo (mesma disciplina de \"evidência ou "
        "silêncio\" que rege o código e este próprio vault).",
    ]
    if evidencias["abertura"]:
        partes_tom.append(
            "## Exemplo real — abertura de `docs/VALIDACAO_PUBLICA.md`\n> " +
            evidencias["abertura"].replace("\n", "\n> "))
    if evidencias["fechamento"]:
        partes_tom.append(
            "## Exemplo real — fechamento da tabela consolidada\n> " +
            evidencias["fechamento"].replace("\n", "\n> "))
    stem_negativo = next((Path(r).stem for r in achados
                          if "achado-real-negativo-registrado-honesto" in r), None)
    stem_retratacao = next((Path(r).stem for r in achados
                            if "retratacao-metodologica-interna-antes-de-publicar-qualquer-numero"
                            in r), None)
    links_tom = []
    if stem_negativo:
        links_tom.append(f"- {_wikilink(stem_negativo, 'resultado fraco publicado, não escondido')}")
    if stem_retratacao:
        links_tom.append(f"- {_wikilink(stem_retratacao, 'retratado antes de publicar, não depois')}")
    if links_tom:
        partes_tom.append("## Ver também\n" + "\n".join(links_tom))
    plano["05-Identidade/Tom-de-Voz.md"] = _nota(
        "Tom de Voz", ["identidade"], "docs/VALIDACAO_PUBLICA.md", "\n\n".join(partes_tom))

    # -- MOC-Identidade.md ------------------------------------------------
    plano["05-Identidade/MOC-Identidade.md"] = _nota(
        "MOC — Identidade", ["moc"], "05-Identidade/",
        "Ponto de entrada da identidade do projeto.\n\n"
        f"- {_wikilink('Missao', 'os 3 compromissos do projeto')}\n"
        f"- {_wikilink('Identidade-Visual', 'paleta, regras de uso e tipografia')}\n"
        f"- {_wikilink('Mascote', 'origem da paleta e do símbolo do sol')}\n"
        f"- {_wikilink('Tom-de-Voz', 'como o projeto se comunica')}\n")

    return plano


# ═════════════════════════════════════════════════════════════════════════
#  Geração de notas — 06-Autoria-e-Seguranca
# ═════════════════════════════════════════════════════════════════════════

def parse_citation_cff() -> dict[str, Any]:
    import yaml
    return yaml.safe_load(_ler("CITATION.cff"))


def gerar_autoria_e_seguranca() -> dict[str, str]:
    plano: dict[str, str] = {}
    citation = parse_citation_cff()
    autor_cff = (citation.get("authors") or [{}])[0]
    nome = f"{autor_cff.get('given-names', '')} {autor_cff.get('family-names', '')}".strip()
    email = autor_cff.get("email", "—")
    orcid = autor_cff.get("orcid", "—")
    licenca = citation.get("license", "—")
    autores_git = sorted({ln for ln in _git("log", "--format=%an <%ae>").splitlines()})

    # -- Autoria.md -----------------------------------------------------------
    corpo_autoria = (
        f"**Nome:** {nome}\n\n"
        f"**E-mail:** {email}\n\n"
        f"**ORCID:** {orcid}\n\n"
        f"**Licença do projeto:** {licenca} (dual licensing -- ver "
        "`docs/COMMERCIAL.md` e `README.md`).\n\n"
        "## Afiliação institucional e orientação acadêmica\n"
        "`CITATION.cff` não tem campo de afiliação institucional, e o "
        f"histórico de commits (`git log`) só registra {len(autores_git)} "
        "identidade(s) de autor -- nome/e-mail, sem afiliação nem "
        "orientador(a). Em vez de preencher por suposição (a instrução "
        "desta auditoria pedia para reportar antes de inventar), o autor "
        "confirmou diretamente, nesta auditoria (2026-09-06): **projeto "
        "pessoal, idealizado e desenvolvido inteiramente por ele, sem "
        "vínculo institucional ou orientação acadêmica associada ao "
        "código deste repositório.**\n\n"
        "## Autoria dos commits\n" +
        "\n".join(f"- `{a}`" for a in autores_git) +
        "\n\n## Licenciamento (correção de premissa)\n"
        "A instrução original desta auditoria presumia uma \"decisão de "
        "dual-licensing com CLA já registrada em `50-Decisoes/`\" -- "
        "checado: não existe menção a CLA (Contributor License Agreement) "
        "em nenhum arquivo `.md` deste repositório, e nenhuma das duas "
        "fontes que alimentam `50-Decisoes/` (`docs/PROGRESSO.md`, "
        "`docs/VALIDACAO_PUBLICA.md`) tem um parágrafo `**Decisão**` "
        "sobre isso. O que existe de fato, em `docs/COMMERCIAL.md` e "
        "`README.md`, é dual licensing simples: GPLv3 para uso "
        "acadêmico/pesquisa, licença comercial separada (contato direto "
        "com o autor) para uso proprietário -- copyright integral "
        "retido pelo autor, sem CLA.\n\n"
        "## Ver também\n"
        f"- {_wikilink('Proveniencia', 'como o histórico do git sustenta esta autoria')}\n"
        f"- {_wikilink('Seguranca-de-Dados', 'o que é protegido antes de qualquer nota ser gravada')}")
    plano["06-Autoria-e-Seguranca/Autoria.md"] = _nota(
        "Autoria", ["autoria"], ["CITATION.cff", "docs/COMMERCIAL.md", "README.md"],
        corpo_autoria, autor=None)

    # -- Proveniencia.md --------------------------------------------------
    corpo_proveniencia = (
        "**Isto é prova de proveniência informacional, não parecer "
        "jurídico sobre propriedade intelectual.**\n\n"
        "Cada regeneração deste vault grava, no frontmatter de toda "
        f"nota, o commit HEAD do repositório no momento da geração "
        f"(`commit: {_HEAD_HASH}` nesta rodada) -- não um valor fixo "
        "escrito à mão. O histórico do git por trás desse commit é "
        "timestampado e distribuído: cada commit carrega autor e data, "
        "e reescrever um commit já publicado (`git commit --amend`, "
        "`rebase`) muda o hash de tudo que vem depois -- é detectável, "
        "não silencioso. É essa cadeia verificável, não esta nota em si, "
        "que dá peso a uma afirmação de autoria.\n\n"
        "## Ver também\n"
        f"- {_wikilink('Autoria', 'a quem esta cadeia de proveniência aponta')}")
    plano["06-Autoria-e-Seguranca/Proveniencia.md"] = _nota(
        "Proveniência", ["autoria"], ["scripts/gerar_vault_obsidian.py"],
        corpo_proveniencia)

    # -- Seguranca-de-Dados.md --------------------------------------------
    corpo_seguranca = (
        "## O que a guarda de privacidade protege\n"
        "`scripts/privacidade_amostras.py` varre todo conteúdo ANTES de "
        "gravar (Passo 169) por dois padrões: identificador real de "
        "amostra (`COD-DD-MM-AAAA`, formato de `mae_id` do acervo de "
        "origem) e caminho absoluto de máquina (unidade Windows seguida "
        "de pasta de usuário, ou raiz Unix de pasta pessoal -- ver o "
        "regex `PADRAO_CAMINHO_ABSOLUTO` no próprio módulo para o padrão "
        "exato; não reproduzido aqui de propósito, para esta nota não "
        "acionar a própria guarda que descreve).\n\n"
        "## Como é aplicada\n"
        "O gerador FALHA (`VazamentoDePrivacidade`, nada é escrito em "
        "disco) se qualquer achado aparecer no plano em memória -- não "
        "avisa e continua. Contra-prova real (não só a regra escrita): "
        "`tests/test_gerador_vault_obsidian.py` monta um identificador "
        "sintético em tempo de execução e confirma que a guarda realmente "
        "levanta `VazamentoDePrivacidade` -- um guarda que nunca casasse "
        "nada passaria despercebido sem este teste.\n\n"
        "## Isolamento de dado de terceiro\n"
        "Nenhum dataset público usado nas validações é versionado neste "
        "repositório -- confirmado em `docs/VALIDACAO_PUBLICA.md` "
        "(seção de reprodução: os datasets são baixados via "
        "`GUARACI_DATASETS_DIR`/`scripts/download_datasets/`, não "
        "commitados). Política de terceiro, não de amostra própria -- "
        "mecanismo diferente do acima, mesma disciplina.\n\n"
        "## Ver também\n"
        f"- {_wikilink('Autoria', 'quem é responsável por esta política')}")
    plano["06-Autoria-e-Seguranca/Seguranca-de-Dados.md"] = _nota(
        "Segurança de Dados", ["autoria"],
        ["scripts/privacidade_amostras.py", "docs/VALIDACAO_PUBLICA.md"],
        corpo_seguranca)

    # -- MOC-Autoria-Seguranca.md ------------------------------------------
    plano["06-Autoria-e-Seguranca/MOC-Autoria-Seguranca.md"] = _nota(
        "MOC — Autoria e Segurança", ["moc"], "06-Autoria-e-Seguranca/",
        "Ponto de entrada de autoria, proveniência e segurança de dados.\n\n"
        f"- {_wikilink('Autoria', 'quem fez, com qual licença')}\n"
        f"- {_wikilink('Proveniencia', 'por que a autoria é verificável')}\n"
        f"- {_wikilink('Seguranca-de-Dados', 'o que nunca vaza para o vault')}\n")

    return plano


# ═════════════════════════════════════════════════════════════════════════
#  Geração de notas — 35-Telas-e-Fluxos (CLI + Web) — Passo 186
#
#  Fonte real do dispatch da CLI: `_SECOES_NAVEGAVEIS` (guaraci.py) -- lista
#  de 17 teclas -> (chave t_, chave d_) usada pelo próprio assistente ([G] ->
#  "navegar seções") desde o achado do Agente 6 (docstring de
#  `_guaraci_navegar_secoes`: um dict estático anterior tinha só 8 das 18
#  abas reais). "A" (Sobre) é tratada à parte no próprio código-fonte, sem
#  chave t_/d_ própria -- reproduzido aqui do mesmo jeito. G/M/I/S/L/R/N são
#  ações do loop de `main()` que NÃO aparecem nesses 17+1 -- ou porque não
#  têm submenu navegável (G é o próprio assistente, circular) ou porque
#  ficam fora dos 6 grupos visuais de `_print_main_menu` (S/L/R/N, achado
#  desta auditoria do vault, Passo 186 -- ver nota de cada uma abaixo).
# ═════════════════════════════════════════════════════════════════════════

def parse_i18n_menu() -> dict[str, dict[str, str]]:
    """Extrai `_I18N` (dict PT/EN de rótulos/descrições de menu) de
    `guaraci.py` por AST -- mesma disciplina de `parse_tecnicas_catalog`:
    nunca importar o módulo (evita efeitos de import de `guaraci.py`, que
    carrega Rich/console)."""
    fonte = _ler("src/guaraci/guaraci.py")
    arvore = ast.parse(fonte)
    for node in ast.walk(arvore):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "_I18N":
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", None) == "_I18N":
                    return ast.literal_eval(node.value)
    raise RuntimeError("_I18N não encontrado em guaraci.py — fonte mudou de "
                        "forma, gerador precisa ser ajustado.")


def parse_secoes_navegaveis() -> list[tuple[str, str, str]]:
    """Extrai `_SECOES_NAVEGAVEIS` (tecla, chave t_, chave d_) de
    `guaraci.py` por AST -- a mesma fonte que já alimenta
    `_guaraci_navegar_secoes` no assistente (tecla G)."""
    fonte = _ler("src/guaraci/guaraci.py")
    arvore = ast.parse(fonte)
    for node in ast.walk(arvore):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "_SECOES_NAVEGAVEIS":
            return [tuple(x) for x in ast.literal_eval(node.value)]
    raise RuntimeError("_SECOES_NAVEGAVEIS não encontrado em guaraci.py — "
                        "fonte mudou de forma, gerador precisa ser ajustado.")


#: tecla -> função `_menu_*` real que a implementa (verificado por leitura
#: direta do dispatch de `main()`, guaraci.py — Passo 186). "A" não está em
#: `_SECOES_NAVEGAVEIS` mas tem função própria (`_menu_about`).
_FUNCAO_POR_TECLA: dict[str, str] = {
    "1": "_menu_project", "2": "_menu_data", "3": "_menu_preprocessing",
    "4": "_menu_modeling", "5": "_menu_validation", "6": "_menu_advanced",
    "7": "_menu_visualization", "8": "_menu_technique", "9": "_menu_encoding",
    "H": "_menu_hardware", "B": "_menu_prediction", "X": "_menu_hsi",
    "J": "_menu_plan", "U": "_menu_audit", "K": "_menu_selecao_amostras",
    "P": "_menu_profiles", "?": "_menu_help", "A": "_menu_about",
}

#: Grupos reais do painel principal (`_print_main_menu`, guaraci.py:
#: ~1820-1856, docs/DESIGN.md seção 4) -- confirmado por leitura direta,
#: não pela ordem do dispatch. Teclas que não aparecem nesse painel (S/L/R/N)
#: ficam de fora deste dict de propósito -- ver `_ACOES_SEM_PAINEL`.
_GRUPO_POR_TECLA: dict[str, str] = {
    "2": "Preparar", "3": "Preparar", "9": "Preparar", "P": "Preparar",
    "J": "Planejar", "K": "Planejar", "U": "Planejar",
    "4": "Modelar", "6": "Modelar", "8": "Modelar",
    "5": "Validar", "7": "Validar",
    "B": "Prever", "X": "Prever",
    "1": "Sistema", "H": "Sistema", "G": "Sistema", "M": "Sistema",
    "I": "Sistema", "?": "Sistema", "A": "Sistema", "Q": "Sistema",
}

#: Só estas 3 teclas têm subseção própria em `docs/MANUAL.md` citando a
#: tecla explicitamente ("CLI — menu principal, tecla [X]") -- verificado
#: por leitura direta (Passo 186). O manual é organizado por tópico
#: científico, não por tela; as ~23 entradas restantes não têm
#: correspondência 1:1 confiável, então ficam sem este campo em vez de
#: inventar uma seção (acionamento da regra "evidência ou silêncio").
_SECAO_MANUAL_POR_TECLA: dict[str, str] = {
    "J": "`docs/MANUAL.md` §2.3 — Planejamento de coleta",
    "K": "`docs/MANUAL.md` §2.2c — Seleção de amostras de calibração",
    "U": "`docs/MANUAL.md` §2.4 — Auditoria de delineamento",
}

#: Teclas do dispatch de `main()` sem função `_menu_*` dedicada e sem par
#: t_/d_ em `_SECOES_NAVEGAVEIS` -- cada uma descrita a partir de leitura
#: direta do código (função real quando existe; senão, do bloco inline em
#: `main()`). G/M/I ficam no grupo "Sistema" do painel principal mas
#: chamam uma função de nome diferente do padrão `_menu_*`; S/L/R/N não
#: aparecem em nenhum dos 6 grupos visuais (achado desta auditoria,
#: registrado também em `Paridade-CLI-Web.md`).
_ACOES_SEM_MENU_DEDICADO: dict[str, dict[str, Any]] = {
    "G": {"titulo": "Assistente Guaraci", "funcao": "_abrir_assistente",
          "grupo": "Sistema"},
    "M": {"titulo": "Alternar Modo (Iniciante/Avançado)",
          "funcao": "_toggle_modo_usuario", "grupo": "Sistema",
          "resumo_extra": "Alterna globalmente entre os modos 'Iniciante' e "
                           "'Avançado' -- controla se os submenus escondem "
                           "campos avançados por padrão. Persistido em "
                           "arquivo-flag próprio (mesmo padrão do idioma)."},
    "I": {"titulo": "Alternar Idioma (PT/EN)", "funcao": "_toggle_idioma",
          "grupo": "Sistema",
          "resumo_extra": "Alterna a interface entre Português e Inglês, "
                           "persistido em arquivo-flag."},
    "S": {"titulo": "Salvar Configuração como Perfil", "funcao": "_salvar_yaml",
          "grupo": None,
          "resumo_extra": "Salva a configuração (`cfg`) atual como um perfil "
                           "`.yaml` nomeado pelo usuário, em `_PERFIS_DIR` -- "
                           "reaproveitável depois via `[L]`. Não aparece nos "
                           "6 grupos do painel principal (achado desta "
                           "auditoria do vault, Passo 186)."},
    "L": {"titulo": "Carregar Configuração de Perfil", "funcao": "_carregar_yaml",
          "grupo": None,
          "resumo_extra": "Lista os perfis `.yaml` salvos em `_PERFIS_DIR` e "
                           "carrega o escolhido na configuração ativa. Não "
                           "aparece nos 6 grupos do painel principal (achado "
                           "desta auditoria do vault, Passo 186)."},
    "R": {"titulo": "Executar Pipeline", "funcao": "_rodar_pipeline",
          "grupo": None,
          "resumo_extra": "Roda o checklist pré-execução (faixa espectral "
                           "incompatível, amostras sem `mae_id`, estimativa "
                           "de tempo — `docs/MANUAL.md` §1.1) e, se os dados "
                           "estiverem prontos, executa o pipeline "
                           "(`pipeline.executar()`) com o painel de "
                           "acompanhamento ao vivo (§1.2). Destacada em caixa "
                           "própria (`_print_run_box`), não nos 6 grupos do "
                           "painel principal (achado desta auditoria do "
                           "vault, Passo 186)."},
    "N": {"titulo": "Definir Tag/ID da Execução", "funcao": None, "grupo": None,
          "resumo_extra": "Define um identificador (`cfg.tag`) usado para "
                           "nomear a pasta de resultados da próxima "
                           "execução, sanitizado para `[\\w\\-_]`. Ação "
                           "inline no loop de `main()` (sem função própria) "
                           "e sem entrada nos 6 grupos do painel principal "
                           "(achado desta auditoria do vault, Passo 186)."},
    "Q": {"titulo": "Sair", "funcao": None, "grupo": "Sistema",
          "resumo_extra": "Encerra o assistente (`_exibir_despedida()` + "
                           "`break` no loop de `main()`)."},
}


def _funcoes_de(fonte: str) -> dict[str, ast.FunctionDef]:
    arvore = ast.parse(fonte)
    return {n.name: n for n in ast.walk(arvore) if isinstance(n, ast.FunctionDef)}


_CHAMADAS_COM_CAMPOS = {"_loop_menu", "_print_submenu_compact"}


def _campos_de_chamada(node: ast.FunctionDef) -> list[str] | None:
    """1ª lista literal de strings passada a `_loop_menu`/
    `_print_submenu_compact` dentro do corpo de `node` -- os campos DE
    VERDADE que aquele submenu exibe (ex.: `_loop_menu(t, d, ["pasta_dados",
    "pasta_saida", "tag"], cfg)`), nunca uma suposição do que a tela mostra."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) in _CHAMADAS_COM_CAMPOS:
            for arg in n.args:
                if isinstance(arg, ast.List):
                    try:
                        valores = [ast.literal_eval(e) for e in arg.elts]
                    except ValueError:
                        continue
                    if valores and all(isinstance(v, str) for v in valores):
                        return valores
    return None


def _modulos_citados_por(fonte_completa: str, node: ast.AST,
                          modulos_conhecidos: set[str]) -> list[str]:
    """Módulos reais de `src/guaraci/` importados dentro do corpo de `node`
    (`from guaraci.X import ...`) -- mesmo padrão de dependência interna já
    usado por `parse_modulos`, aplicado aqui a uma função/tela em vez de a
    um módulo inteiro."""
    trecho = ast.get_source_segment(fonte_completa, node) or ""
    citados = set(re.findall(r"from guaraci\.(\w+) import", trecho))
    return sorted(m for m in citados if m in modulos_conhecidos)


def gerar_telas_cli(modulos_conhecidos: set[str]) -> dict[str, str]:
    plano: dict[str, str] = {}
    fonte = _ler("src/guaraci/guaraci.py")
    funcoes = _funcoes_de(fonte)
    i18n_pt = parse_i18n_menu()["PT"]
    secoes = parse_secoes_navegaveis()

    entradas: list[tuple[str, str, str]] = [
        (tecla, i18n_pt[t_key], i18n_pt[d_key]) for tecla, t_key, d_key in secoes
    ]
    entradas.append(("A", "Sobre", i18n_pt["d_sobre"]))

    for tecla, titulo, descricao in entradas:
        nome_funcao = _FUNCAO_POR_TECLA.get(tecla)
        node = funcoes.get(nome_funcao) if nome_funcao else None
        corpo = [
            f"**Tecla:** `[{tecla}]`  ·  **Grupo no painel principal:** "
            f"{_GRUPO_POR_TECLA.get(tecla, '(não aparece nos 6 grupos — ver Paridade-CLI-Web)')}",
            "", descricao,
        ]
        fonte_campos = ["src/guaraci/guaraci.py (_SECOES_NAVEGAVEIS)"]
        if node is not None:
            fonte_campos = [f"src/guaraci/guaraci.py:{node.lineno} ({nome_funcao})"]
            doc = ast.get_docstring(node)
            if doc:
                corpo.append("\n## Detalhe (docstring real da função)\n" +
                              _resumo_docstring(doc, max_linhas=8))
            campos = _campos_de_chamada(node)
            if campos:
                corpo.append("\n## Campos de configuração exibidos\n" +
                              "\n".join(f"- `{c}`" for c in campos))
            mods = _modulos_citados_por(fonte, node, modulos_conhecidos)
            if mods:
                corpo.append("\n## Chama\n" + "\n".join(
                    f"- {_wikilink(f'{m}.py', 'chamado por esta tela da CLI')}" for m in mods))
        secao_manual = _SECAO_MANUAL_POR_TECLA.get(tecla)
        if secao_manual:
            corpo.append(f"\n## Documentação de referência\n{secao_manual}")
        corpo.append(
            "\n## Ver também\n"
            f"- {_wikilink('MOC-Telas-e-Fluxos', 'ponto de entrada de telas e fluxos')}\n"
            f"- {_wikilink('Paridade-CLI-Web', 'cobertura desta tela na interface web')}")
        conteudo = _nota(titulo=f"CLI — {titulo}", tags=["tela", "cli"],
                          fonte=fonte_campos, corpo="\n".join(corpo))
        plano[f"35-Telas-e-Fluxos/{_slug(f'CLI {titulo}')}.md"] = conteudo

    for tecla, info in _ACOES_SEM_MENU_DEDICADO.items():
        nome_funcao = info["funcao"]
        node = funcoes.get(nome_funcao) if nome_funcao else None
        corpo = [
            f"**Tecla:** `[{tecla}]`  ·  **Grupo no painel principal:** "
            f"{info['grupo'] or '(não aparece nos 6 grupos — achado desta auditoria, ver Paridade-CLI-Web)'}",
        ]
        if info.get("resumo_extra"):
            corpo += ["", info["resumo_extra"]]
        fonte_campos = ["src/guaraci/guaraci.py::main() (ação inline)"]
        if node is not None:
            fonte_campos = [f"src/guaraci/guaraci.py:{node.lineno} ({nome_funcao})"]
            doc = ast.get_docstring(node)
            if doc:
                corpo.append("\n## Detalhe (docstring real da função)\n" +
                              _resumo_docstring(doc, max_linhas=8))
            mods = _modulos_citados_por(fonte, node, modulos_conhecidos)
            if mods:
                corpo.append("\n## Chama\n" + "\n".join(
                    f"- {_wikilink(f'{m}.py', 'chamado por esta ação da CLI')}" for m in mods))
        corpo.append(
            "\n## Ver também\n"
            f"- {_wikilink('MOC-Telas-e-Fluxos', 'ponto de entrada de telas e fluxos')}\n"
            f"- {_wikilink('Paridade-CLI-Web', 'cobertura desta ação na interface web')}")
        titulo_nota = f"CLI — {info['titulo']}"
        conteudo = _nota(titulo=titulo_nota, tags=["tela", "cli"],
                          fonte=fonte_campos, corpo="\n".join(corpo))
        plano[f"35-Telas-e-Fluxos/{_slug(titulo_nota)}.md"] = conteudo

    return plano


#: Ordem real das 8 abas (`app_quimiometria.py`, `st.tabs([...])` — Passo 186).
_ABAS_WEB_ORDEM: list[str] = [
    "projeto", "dados", "preprocessamento", "modelo",
    "validacao", "predicao", "relatorios", "sobre",
]

_PADRAO_ABA_DOCSTRING = re.compile(r"Aba\s+(\d+)\s*\(([^)]+)\)\s*:\s*(.+)", re.S)


def parse_abas_web(modulos_conhecidos: set[str]) -> list[dict[str, Any]]:
    """1 entrada por `src/guaraci/app_tabs/*.py` -- todos seguem a mesma
    convenção real de docstring de módulo ("app_tabs/x.py — Aba N (Nome):
    descrição"), verificada por leitura direta das 8 abas (Passo 186).
    Nenhuma prosa é inventada aqui: número, nome e resumo vêm do parse
    desse docstring; se um módulo novo não seguir a convenção, o resumo cai
    no docstring bruto em vez de quebrar a geração."""
    resultado = []
    for nome in _ABAS_WEB_ORDEM:
        rel = f"src/guaraci/app_tabs/{nome}.py"
        texto = _ler(rel)
        arvore = ast.parse(texto)
        doc_modulo = ast.get_docstring(arvore) or ""
        m = _PADRAO_ABA_DOCSTRING.search(doc_modulo)
        numero = m.group(1) if m else "?"
        titulo_en = m.group(2).strip() if m else nome
        resumo = re.sub(r"\s+", " ", m.group(3)).strip() if m else doc_modulo.strip()
        render_node = next(
            (n for n in ast.walk(arvore)
             if isinstance(n, ast.FunctionDef) and n.name == "render"), None)
        mods = set(re.findall(r"from guaraci\.(\w+) import", texto))
        resultado.append({
            "modulo": nome, "caminho": rel, "numero": numero, "titulo_en": titulo_en,
            "resumo": resumo,
            "assinatura": _assinatura_funcao(render_node) if render_node else None,
            "doc_render": (ast.get_docstring(render_node) or "") if render_node else "",
            "linha_render": render_node.lineno if render_node else None,
            "modulos_chamados": sorted(m for m in mods if m in modulos_conhecidos),
        })
    return resultado


def gerar_abas_web(abas: list[dict[str, Any]]) -> dict[str, str]:
    plano: dict[str, str] = {}
    for aba in abas:
        corpo = [f"**Ordem:** aba {aba['numero']} de {len(abas)}  ·  **Módulo:** "
                 f"`{aba['caminho']}`", "", aba["resumo"]]
        if aba["assinatura"]:
            corpo.append(f"\n## Função de renderização\n`{aba['assinatura']}`")
        if aba["doc_render"]:
            corpo.append("\n" + _resumo_docstring(aba["doc_render"], max_linhas=8))
        if aba["modulos_chamados"]:
            corpo.append("\n## Chama\n" + "\n".join(
                f"- {_wikilink(f'{m}.py', 'chamado por esta aba web')}"
                for m in aba["modulos_chamados"]))
        corpo.append(
            "\n## Documentação de referência\n`docs/MANUAL.md` §6 (Fluxo "
            "típico na interface web) -- lista consolidada das 8 abas, sem "
            "subseção própria por aba (ver `Paridade-CLI-Web.md`).")
        corpo.append(
            "\n## Ver também\n"
            f"- {_wikilink('MOC-Telas-e-Fluxos', 'ponto de entrada de telas e fluxos')}\n"
            f"- {_wikilink('Paridade-CLI-Web', 'cobertura desta aba na CLI')}")
        titulo = f"Web — {aba['titulo_en']}"
        fonte = [aba["caminho"]]
        if aba["linha_render"]:
            fonte.append(f"{aba['caminho']}:{aba['linha_render']}")
        conteudo = _nota(titulo=titulo, tags=["tela", "web"], fonte=fonte,
                          corpo="\n".join(corpo))
        plano[f"35-Telas-e-Fluxos/{_slug(titulo)}.md"] = conteudo
    return plano


#: Achado real desta auditoria do vault (Passo 186): estes 6 fluxos da CLI
#: não têm equivalente na web, e `docs/MANUAL.md` os documenta sem
#: qualquer nota de "falta portar" -- verificado por leitura direta das 8
#: abas de `app_quimiometria.py` e do dispatch completo de `guaraci.py`.
#: Decisão do autor (2026-09-07, resposta direta a esta auditoria):
#: limitação de escopo aceita, não pendência a abrir.
_GAPS_CLI_ONLY: list[tuple[str, str]] = [
    ("Planejamento de Coleta (`[J]`)",
     "tamanho amostral e distribuição de sessões de coleta; sem tela "
     "equivalente na web."),
    ("Seleção de Amostras (`[K]`)",
     "divide um CSV em calibração/validação (Kennard-Stone/Duplex/SPXY); "
     "sem tela equivalente na web."),
    ("Auditoria de Delineamento como tela dedicada (`[U]`)",
     "a auditoria em si roda embutida em toda execução, CLI e web "
     "(`auditoria_delineamento.py`) — só a tela para rodá-la isoladamente, "
     "sem treinar o modelo inteiro, é exclusiva da CLI."),
    ("HSI — Imageamento Hiperespectral (`[X]`)",
     "a web só expõe o campo de configuração `hsi_pasta_dataset` (aba "
     "Dados), sem tela de resultado/classificação por pixel."),
    ("Perfis Prontos (`[P]`)",
     "tela de listar/aplicar perfis de matriz/técnica prontos; a web usa "
     "perfis via `selectbox` dentro de cada campo, mas não tem esta tela "
     "de gestão dedicada."),
    ("Codificação DX (`[9]`)",
     "nomenclatura JCAMP-DX e códigos de espécie; sem tela equivalente na web."),
]


def gerar_paridade_cli_web(modulos_conhecidos: set[str]) -> dict[str, str]:
    corpo = [
        "Cruzamento entre as telas/fluxos reais da CLI "
        "(`_SECOES_NAVEGAVEIS` + ações diretas de `main()`, `guaraci.py`) e "
        "as 8 abas do aplicativo web (`app_quimiometria.py`).",
        "",
        "## Fluxos só na CLI (achado real, Passo 186 desta auditoria do "
        "vault — limitação de escopo aceita, não pendência aberta)",
    ]
    for titulo, texto in _GAPS_CLI_ONLY:
        corpo.append(f"- **{titulo}** — {texto}")
    if "auditoria_delineamento" in modulos_conhecidos:
        corpo.append(f"\n({_wikilink('auditoria_delineamento.py', 'módulo que roda embutido nas duas interfaces')})")
    corpo += [
        "",
        "## Fluxos com paridade real (presentes nas duas interfaces)",
        "Entrada de dados, pré-processamento, modelagem, validação, "
        "predição em amostras novas, relatórios e identificação do "
        "projeto existem nas duas interfaces (menu `[1]`-`[8]`/`[B]` da "
        "CLI ↔ as 8 abas da web), compartilhando o mesmo motor "
        f"({_wikilink('pipeline.py', 'motor único das duas interfaces') if 'pipeline' in modulos_conhecidos else '`pipeline.py`'}) "
        "e a mesma `Config`/`config.yaml`.",
        "",
        "## Gap de paridade já corrigido nesta sessão (evidência real, "
        "verificado em código, não suposição)",
        "- Textos de i18n (PT/EN) que faltavam em abas específicas (Data/"
        "Preprocessing/Prediction/Reports) — comentário em "
        "`app_quimiometria.py`: essas 4 abas \"não passavam por T() "
        "nenhum, ficavam sempre em inglês mesmo com idioma=PT "
        "selecionado\", fechado em 2026-09-01.",
        "- O fluxo cego (Detectar → Identificar → Quantificar, "
        "`predict_blind`) está presente hoje na aba Prediction da web "
        "(`app_tabs/predicao.py` importa `predict_blind` de "
        "`guaraci.predicao`, e `app_quimiometria.py` traduz os textos "
        "\"Blind flow — Detect → Identify → Quantify\").",
        "",
        "## Decisão de escopo (Passo 186 desta auditoria do vault)",
        "Os 6 fluxos CLI-only acima são tratados como **limitação de "
        "escopo aceita**, não pendência a abrir — decisão do autor "
        "nesta auditoria (2026-09-07), no mesmo espírito de outras "
        "decisões de escopo já registradas no projeto.",
    ]
    conteudo = _nota(
        "Paridade CLI ↔ Web", ["paridade"],
        ["src/guaraci/guaraci.py", "app_quimiometria.py", "docs/MANUAL.md"],
        "\n".join(corpo))
    return {"35-Telas-e-Fluxos/Paridade-CLI-Web.md": conteudo}


_GRUPOS_TELAS_FLUXOS = [
    ("Preparar", ["Dados", "Pré-processamento", "Codificação DX", "Perfis Prontos"]),
    ("Planejar", ["Planejamento de Coleta", "Seleção de Amostras",
                  "Auditoria de Delineamento"]),
    ("Modelar", ["Modelagem", "Métodos Avançados", "Técnica Analítica"]),
    ("Validar", ["Validação", "Visualização"]),
    ("Prever", ["Predição em Lote", "Imageamento Hiperespectral"]),
]


def gerar_moc_telas_fluxos(plano_cli: dict[str, str], plano_web: dict[str, str]) -> dict[str, str]:
    stems_cli = {Path(p).stem: p for p in plano_cli}
    corpo = [
        f"Ponto de entrada das {len(plano_cli)} telas/ações da CLI e das "
        f"{len(plano_web)} abas da web (Passo 186 desta auditoria do "
        f"vault) — agrupadas pelo mesmo fluxo de trabalho já usado no "
        "painel principal da CLI (`docs/DESIGN.md` seção 4).",
        "",
        f"- {_wikilink('Paridade-CLI-Web', 'o que existe só numa interface, e por quê')}",
        "",
        "## Web (8 abas, ordem real de `app_quimiometria.py`)",
    ]
    corpo += [f"- {_wikilink(Path(p).stem, 'aba do aplicativo web')}" for p in sorted(
        plano_web, key=lambda p: Path(p).stem)]
    corpo.append("\n## CLI, por grupo de fluxo de trabalho (painel principal)")
    usados: set[str] = set()
    for grupo, titulos in _GRUPOS_TELAS_FLUXOS:
        linhas_grupo = []
        for titulo in titulos:
            stem = _slug(f"CLI {titulo}")
            if stem in stems_cli:
                linhas_grupo.append(f"- {_wikilink(stem, f'tela do grupo {grupo}')}")
                usados.add(stems_cli[stem])
        if linhas_grupo:
            corpo.append(f"\n### {grupo}\n" + "\n".join(linhas_grupo))
    restantes = sorted(set(plano_cli) - usados, key=lambda p: Path(p).stem)
    if restantes:
        corpo.append("\n### Sistema / ações diretas\n" + "\n".join(
            f"- {_wikilink(Path(p).stem, 'ação/tela de sistema da CLI')}" for p in restantes))
    conteudo = _nota(
        "MOC — Telas e Fluxos", ["moc"],
        ["src/guaraci/guaraci.py", "app_quimiometria.py"], "\n".join(corpo))
    return {"35-Telas-e-Fluxos/MOC-Telas-e-Fluxos.md": conteudo}


# ═════════════════════════════════════════════════════════════════════════
#  Geração de notas — 08-Documentos — Passo 187
#
#  1 nota por documento REAL e VERSIONADO do projeto -- "versionado" é
#  verificado por `git ls-files` no momento da geração (não pela presença
#  neste catálogo), então um arquivo removido/nunca commitado nunca vira
#  nota. `para_quem`/`deriva_para` são julgamento curado sobre a ESTRUTURA
#  do vault (mesmo padrão de `CONCEITOS`/`MODOS_FORA_DO_CATALOGO`); o
#  conteúdo de cada nota (resumo, data) vem sempre de uma leitura real do
#  arquivo/git no momento da geração.
# ═════════════════════════════════════════════════════════════════════════

_CATALOGO_DOCUMENTOS: list[dict[str, Any]] = [
    {"rel": "README.md", "para_quem": "usuário final / quem descobre o projeto no GitHub",
     "deriva_para": [("MOC-Guaraci", "ponto de entrada equivalente em prosa")]},
    {"rel": "README.pt-br.md", "para_quem": "usuário final em português",
     "deriva_para": [("MOC-Guaraci", "ponto de entrada equivalente em prosa")]},
    {"rel": "docs/MANUAL.md", "para_quem": "usuário final e contribuidor — manual funcional completo",
     "deriva_para": [("MOC-Telas-e-Fluxos", "fonte de descrição de telas/abas")]},
    {"rel": "docs/VALIDACAO_PUBLICA.md", "para_quem": "revisor científico / contribuidor",
     "deriva_para": [("MOC-Validacoes", "fonte da tabela consolidada")]},
    {"rel": "docs/COMPATIBILITY.md", "para_quem": "contribuidor — contrato de não-regressão",
     "deriva_para": [("MOC-Decisoes", "fonte dos casos especiais documentados")]},
    {"rel": "docs/PROGRESSO.md", "para_quem": "o próprio autor / uma sessão futura do agente",
     "deriva_para": [("MOC-Decisoes", "fonte de achados e decisões")]},
    {"rel": "docs/INDICE_PROJETO.md", "para_quem": "contribuidor / agente — mapa de onde está cada coisa",
     "deriva_para": [("MOC-Guaraci", "índice equivalente em prosa")]},
    {"rel": "docs/DESIGN.md", "para_quem": "contribuidor — identidade visual e UX",
     "deriva_para": [("MOC-Identidade", "fonte de paleta/tipografia/regras de uso")]},
    {"rel": "SECURITY.md", "para_quem": "quem opera um deploy público / contribuidor",
     "deriva_para": [("Seguranca-de-Dados", "documento formal da política; esta nota resume")]},
    {"rel": "CITATION.cff", "para_quem": "quem cita o projeto em trabalho científico",
     "deriva_para": [("Autoria", "fonte de nome/e-mail/ORCID/licença")]},
    {"rel": "datasets/README.md", "para_quem": "contribuidor / revisor reproduzindo validações públicas",
     "deriva_para": [("MOC-Validacoes", "instruções de download dos datasets validados")]},
    {"rel": "docs/RASCUNHOS_CONTATO.md", "para_quem": "o próprio autor (rascunhos de contato/divulgação)",
     "deriva_para": []},
    {"rel": "docs/COMMERCIAL.md", "para_quem": "interessado em uso comercial/proprietário",
     "deriva_para": [("Autoria", "fonte do modelo de dual licensing")]},
    {"rel": "CONTRIBUTING.md", "para_quem": "contribuidor externo em potencial",
     "deriva_para": []},
    {"rel": "CODE_OF_CONDUCT.md", "para_quem": "qualquer participante da comunidade do projeto",
     "deriva_para": []},
    {"rel": "ACKNOWLEDGMENTS.md", "para_quem": "usuário/contribuidor — créditos de terceiros",
     "deriva_para": []},
    {"rel": "docs/BENCHMARK_TECATOR.md", "para_quem": "revisor científico — benchmark externo público",
     "deriva_para": [("MOC-Validacoes", "detalha o benchmark Tecator")]},
    {"rel": "docs/CHANGELOG.md", "para_quem": "usuário atualizando de versão / contribuidor",
     "deriva_para": [("Estado-Atual", "versão atual, fonte única em pipeline.__version__")]},
    {"rel": "docs/VALIDATION.md", "para_quem": "revisor científico — cartão de visita técnico",
     "deriva_para": [("MOC-Validacoes", "complementar a VALIDACAO_PUBLICA.md")]},
    {"rel": "docs/index.md", "para_quem": "visitante da página do projeto (GitHub Pages)",
     "deriva_para": [("MOC-Guaraci", "página de entrada equivalente")]},
]

_PADRAO_LINHA_RUIDO_DOC = re.compile(r"^(<|\[!\[)")


def _resumo_documento(texto: str, min_chars: int = 90) -> str:
    """Primeiro parágrafo REAL e substancial do documento (>= `min_chars`),
    pulando título, cabeçalhos, HTML cru e badges -- nunca reescrito à mão.
    Blocos de blockquote (`>`) contam como parágrafo (convenção já usada
    por vários docs do projeto, ex. `docs/MANUAL.md`/`VALIDATION.md`)."""
    linhas = texto.splitlines()
    corpo = linhas[1:] if linhas and linhas[0].startswith("#") else linhas
    blocos: list[str] = []
    atual: list[str] = []
    for ln in corpo:
        s = ln.strip()
        if _PADRAO_LINHA_RUIDO_DOC.match(s) or s.startswith("#"):
            continue
        if not s:
            if atual:
                blocos.append(" ".join(atual)); atual = []
            continue
        atual.append(s.lstrip(">").strip() if s.startswith(">") else s)
    if atual:
        blocos.append(" ".join(atual))
    for b in blocos:
        if len(b) >= min_chars:
            return " ".join(b.split()[:120])
    return blocos[0] if blocos else "(sem parágrafo extraível na abertura do arquivo)"


def gerar_documentos() -> dict[str, str]:
    plano: dict[str, str] = {}
    rastreados = set(_git("ls-files").splitlines())
    for item in _CATALOGO_DOCUMENTOS:
        rel = item["rel"]
        caminho_abs = _RAIZ / rel
        if rel not in rastreados or not caminho_abs.exists():
            continue  # evidência ou silêncio: doc não versionado/sumiu, nota omitida
        if rel.endswith(".cff"):
            cff = parse_citation_cff()
            resumo = (f"Metadados de citação (Citation File Format) -- "
                      f"versão do software `{cff.get('version', '—')}`, "
                      f"licença `{cff.get('license', '—')}`, "
                      f"{len(cff.get('authors') or [])} autor(es) registrado(s).")
        else:
            resumo = _resumo_documento(caminho_abs.read_text(encoding="utf-8"))
        try:
            ultima_att = _git("log", "-1", "--format=%ci", "--", rel)
        except subprocess.CalledProcessError:
            ultima_att = ""
        corpo = [
            resumo, "",
            f"**Para quem:** {item['para_quem']}",
            f"**Última atualização (git log):** {ultima_att or 'sem histórico git'}",
            f"**Arquivo:** `{rel}`",
        ]
        if item["deriva_para"]:
            corpo.append("\n## Fonte de\n" + "\n".join(
                f"- {_wikilink(alvo, relacao)}" for alvo, relacao in item["deriva_para"]))
        titulo = f"Documento — {rel}"
        conteudo = _nota(titulo=titulo, tags=["documento"], fonte=rel,
                          corpo="\n".join(corpo))
        plano[f"08-Documentos/{_slug(titulo)}.md"] = conteudo

    corpo_moc = [
        f"{len(plano)} documento(s) real(is) e versionado(s) do projeto — "
        "antes só apareciam como referência de rodapé dentro de outras "
        "notas; agora cada um é uma nota navegável que linka de volta às "
        "categorias do vault que dele derivam (Passo 187 desta auditoria "
        "do vault).",
        "",
    ]
    corpo_moc += [f"- {_wikilink(Path(p).stem, 'documento do projeto')}"
                  for p in sorted(plano, key=lambda p: Path(p).stem)]
    plano["08-Documentos/MOC-Documentos.md"] = _nota(
        "MOC — Documentos do Projeto", ["moc"], "git ls-files", "\n".join(corpo_moc))
    return plano


# ═════════════════════════════════════════════════════════════════════════
#  MOCs (Maps of Content)
# ═════════════════════════════════════════════════════════════════════════

def gerar_mocs(tecnicas: dict[str, Any], modulos: dict[str, ModuloInfo],
                tecnicas_notas: dict[str, str], validacoes: dict[str, str],
                decisoes: dict[str, str], achados: dict[str, str]) -> dict[str, str]:
    plano: dict[str, str] = {}

    def _links(pasta_plano: dict[str, str], relacao: str) -> str:
        nomes = sorted(Path(p).stem for p in pasta_plano)
        return "\n".join(f"- {_wikilink(n, relacao)}" for n in nomes)

    n_extras = len(MODOS_FORA_DO_CATALOGO)
    plano["00-MOC/MOC-Tecnicas.md"] = _nota(
        "MOC — Técnicas", ["moc"], "src/guaraci/cli_assistente.py",
        f"As {len(tecnicas)} técnicas do catálogo `cli_assistente.TECNICAS` "
        f"+ {n_extras} modalidade(s) de entrada real(is) que existem em "
        "`Config.mode` mas não estão nesse catálogo (tag `fora-do-catalogo`"
        " nas notas correspondentes).\n\n" + _links(tecnicas_notas, "técnica catalogada"))

    plano["00-MOC/MOC-Arquitetura.md"] = _nota(
        "MOC — Arquitetura", ["moc"], "src/guaraci/",
        f"{len(modulos)} módulos em `src/guaraci/*.py`. Grafo completo de "
        "dependências: ver backlinks de cada nota de módulo.\n\n" +
        "\n".join(f"- {_wikilink(f'{n}.py', 'módulo do pipeline')}" for n in sorted(modulos)))

    plano["00-MOC/MOC-Validacoes.md"] = _nota(
        "MOC — Validações públicas", ["moc"], "docs/VALIDACAO_PUBLICA.md",
        "Um dataset público por linha da tabela consolidada "
        "(`docs/VALIDACAO_PUBLICA.md` §1).\n\n" +
        _links(validacoes, "dataset público validado"))

    plano["00-MOC/MOC-Decisoes.md"] = _nota(
        "MOC — Decisões", ["moc"], ["docs/PROGRESSO.md", "docs/COMPATIBILITY.md"],
        "Decisões de design com razão registrada.\n\n### Decisões\n" +
        _links(decisoes, "decisão registrada") + "\n\n### Achados e retratações\n" +
        _links(achados, "achado ou retratação registrado"))

    plano["00-MOC/MOC-Guaraci.md"] = _nota(
        "MOC — Guaraci", ["moc"], "docs/INDICE_PROJETO.md",
        "Ponto de entrada. Este vault é GERADO por "
        "`scripts/gerar_vault_obsidian.py` a partir do repositório do "
        "Guaraci — ver `README-VAULT.md`.\n\n"
        f"- {_wikilink('Estado-Atual', 'snapshot vivo do estado do projeto')}\n"
        f"- {_wikilink('MOC-Identidade', 'missão, identidade visual, mascote e tom de voz')}\n"
        f"- {_wikilink('MOC-Tecnicas', 'catálogo de técnicas')}\n"
        f"- {_wikilink('MOC-Arquitetura', 'mapa de módulos')}\n"
        f"- {_wikilink('MOC-Validacoes', 'validações públicas')}\n"
        f"- {_wikilink('MOC-Decisoes', 'decisões e achados')}\n"
        f"- {_wikilink('MOC-Autoria-Seguranca', 'autoria, proveniência e segurança de dados')}\n"
        f"- {_wikilink('MOC-Telas-e-Fluxos', 'telas da CLI e abas da web, com paridade documentada')}\n"
        f"- {_wikilink('MOC-Documentos', 'documentos reais do projeto como notas navegáveis')}\n")

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
`docs/VALIDACAO_PUBLICA.md`, `docs/COMPATIBILITY.md`, `docs/DESIGN.md`,
`CITATION.cff`, `src/guaraci/cli_assistente.py`, `src/guaraci/*.py`,
`src/guaraci/guaraci.py` (dispatch da CLI), `app_quimiometria.py` +
`src/guaraci/app_tabs/*.py` (abas da web), e os próprios documentos
versionados do projeto (`08-Documentos/`, via `git ls-files`).

**Não edite nada fora de `{_PASTA_PROTEGIDA}/` à mão.** Qualquer edição
manual em outra pasta é perdida na próxima regeneração — o script
recalcula o conteúdo inteiro a cada execução e arquiva (nunca apaga) o
que não existir mais nas fontes.

## Método (por que o vault é estruturado assim)

Duas ideias com precedente real, não invenção deste projeto:

- **Zettelkasten** — cada nota é atômica (uma ideia só) e autônoma (faz
  sentido lida isolada, fora do vault). A parte mais aplicada aqui é a
  disciplina de conexão: um wikilink sem texto ao lado é uma conexão
  fraca -- todo link deste vault carrega a relação por extenso logo
  depois (travessão + poucas palavras, ex. "depende de", "implementa",
  "testado por", "valida"), gerado já com a relação (não editado depois
  à mão), para que o grafo em si comunique POR QUE duas notas estão
  ligadas.
- **Diátaxis** — separa conteúdo por propósito de leitura: `referência`
  (consulta rápida, "o que é/o que faz" -- técnicas, módulos, funções,
  validações) vs. `explicação` (entendimento profundo, "por quê" --
  conceitos, decisões, achados, identidade, autoria). Não reorganiza
  pasta nenhuma -- entra como o campo `modo:` no frontmatter de toda
  nota, calculado pela categoria de origem.

## Quando regenerar

```
python scripts/gerar_vault_obsidian.py
```

Regenere depois de: um bloco/passo novo em `docs/PROGRESSO.md`, uma
retratação, uma validação pública nova ou atualizada, uma mudança de
`__all__`/docstring em `src/guaraci/`, ou uma atualização em
`docs/DESIGN.md`/`CITATION.cff`.

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

| Pasta | Fonte | `modo:` |
|---|---|---|
| `05-Identidade/` | `docs/DESIGN.md` (paleta/tipografia), `docs/VALIDACAO_PUBLICA.md` (tom de voz), `assets/guaraci_icon.png` (mascote, inspecionada diretamente) | `explicacao` |
| `10-Tecnicas/` | `src/guaraci/cli_assistente.py` (catálogo `TECNICAS`, 11) + `docs/PROGRESSO.md` (Passo 160) + `Config.mode` (`config.py`) para as modalidades fora do catálogo (`imagem`, `hsi`) | `referencia` |
| `20-Modulos/` | `src/guaraci/*.py` — docstring de módulo, `__all__`, imports internos (introspecção estática) | `referencia` |
| `25-Funcoes/` | `src/guaraci/*.py` por AST — 1 função por módulo: a primeira função de `__all__` na ordem declarada (não toda função exportada -- ver nota no próprio script sobre o Passo 178) | `referencia` |
| `30-Conceitos/` | mapeamento conceito→módulo (neste script) + docstring do(s) módulo(s) | `explicacao` |
| `40-Validacoes/` | `docs/VALIDACAO_PUBLICA.md` §1 (tabela consolidada), 1 nota por linha | `referencia` |
| `50-Decisoes/` | parágrafos `**Decisão...**` em `docs/PROGRESSO.md`/`docs/VALIDACAO_PUBLICA.md` + `docs/COMPATIBILITY.md` (casos especiais) | `explicacao` |
| `60-Achados/` | parágrafos `**Achado...**`/`**RETRATAÇÃO...**`/`**Bug real...**` em `docs/PROGRESSO.md`/`docs/VALIDACAO_PUBLICA.md` + 1 resumo mínimo (tag `passo`, não `achado`) por `## Passo` que não caiu em nenhum parágrafo marcado — garante que todo passo tenha alguma nota (Passo 172) | `explicacao` (`referencia` se tag `passo`) |
| `06-Autoria-e-Seguranca/` | `CITATION.cff`, `git log`, `scripts/privacidade_amostras.py`, `docs/VALIDACAO_PUBLICA.md` | `explicacao` |
| `35-Telas-e-Fluxos/` | `_SECOES_NAVEGAVEIS`/`_I18N`/dispatch de `main()` (`guaraci.py`) para a CLI + docstring de módulo de `src/guaraci/app_tabs/*.py` para a web -- 1 nota por tela/ação real, mais `Paridade-CLI-Web.md` (cruzamento das duas interfaces) | `referencia` (`explicacao` para `Paridade-CLI-Web.md`) |
| `08-Documentos/` | 1 nota por documento real e VERSIONADO do projeto (`git ls-files` confirma no momento da geração); resumo = primeiro parágrafo real do arquivo | `referencia` |
| `00-MOC/` + MOCs de categoria | gerados a partir dos planos acima | `referencia` |
| `90-Canvas/` | gerados programaticamente a partir dos mesmos dados acima | — |
| `Estado-Atual.md` | commit HEAD + contagem de `def test_*` em `tests/` + Passo 160 | — |

Nenhum número (RMSEP, balanced accuracy, contagem de testes, hex de
cor etc.) aparece hardcoded no gerador — todos vêm de uma leitura de
arquivo real no momento da geração.

## Proveniência universal

Toda nota (exceto `06-Autoria-e-Seguranca/Autoria.md`, que seria uma
autorreferência sem sentido, e este arquivo, que é documentação SOBRE o
vault) carrega `autor: "[[Autoria]]"` no frontmatter -- mesmo extraída
ou copiada isoladamente do vault, a nota aponta de volta à origem e à
autoria. Ver `06-Autoria-e-Seguranca/Autoria.md` e `Proveniencia.md`.

## Densidade de grafo e notas órfãs (Passo 179/180)

O gerador produz todo link já com a relação (`— relação`), e a suíte de
testes (`tests/test_proveniencia_e_densidade_vault.py`) calcula o grau
de saída médio por categoria e sinaliza qualquer nota sem link de saída
OU sem link de entrada. Duas notas ficam legitimamente órfãs de entrada
(nada precisa linkar de volta para elas) e isso é esperado, não um bug:
`00-MOC/MOC-Guaraci.md` (ponto de entrada do vault) e este próprio
`README-VAULT.md` (documentação sobre o vault, não conteúdo do grafo).

## Contagens desta geração

- {contagens.get('05-Identidade', 0)} notas de identidade
- {contagens.get('10-Tecnicas', 0)} técnicas
- {contagens.get('20-Modulos', 0)} módulos
- {contagens.get('25-Funcoes', 0)} funções de entrada
- {contagens.get('30-Conceitos', 0)} conceitos
- {contagens.get('40-Validacoes', 0)} validações públicas
- {contagens.get('50-Decisoes', 0)} decisões
- {contagens.get('60-Achados', 0)} achados/retratações
- {contagens.get('06-Autoria-e-Seguranca', 0)} notas de autoria/segurança
- {contagens.get('35-Telas-e-Fluxos', 0)} telas/ações (CLI + web) + paridade
- {contagens.get('08-Documentos', 0)} documentos do projeto

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


#: Diátaxis (Passo 180): `referencia` = consulta rápida ("o que é/o que
#: faz"), `explicacao` = entendimento profundo ("por quê"). Mapeamento
#: automático por pasta de origem -- nunca decisão manual por nota.
_MODO_DIATAXIS_POR_PASTA = {
    "10-Tecnicas": "referencia", "20-Modulos": "referencia",
    "25-Funcoes": "referencia", "40-Validacoes": "referencia",
    "30-Conceitos": "explicacao", "50-Decisoes": "explicacao",
    "05-Identidade": "explicacao", "06-Autoria-e-Seguranca": "explicacao",
    # MOCs (00-MOC/ e os das categorias novas) são consulta rápida de
    # links, mesma natureza de 10-Tecnicas/20-Modulos -- não estavam na
    # lista original da instrução (escrita antes de 06-Autoria existir);
    # extensão razoável, documentada aqui.
    "00-MOC": "referencia",
    # 35-Telas-e-Fluxos (Passo 186): "o que essa tela faz" é consulta
    # rápida, como 10-Tecnicas/20-Modulos -- exceto Paridade-CLI-Web.md,
    # que é análise/decisão de escopo (explicacao), tratada à parte abaixo.
    "35-Telas-e-Fluxos": "referencia",
    # 08-Documentos (Passo 187): mesma natureza de referência rápida.
    "08-Documentos": "referencia",
}


def _modo_diataxis_para(rel: str, conteudo: str) -> str | None:
    pasta = rel.split("/")[0]
    if pasta == "60-Achados":
        m = re.search(r"^tags: \[(.*?)\]", conteudo, re.M)
        tags = [t.strip() for t in (m.group(1).split(",") if m else [])]
        return "referencia" if "passo" in tags else "explicacao"
    if rel == "35-Telas-e-Fluxos/Paridade-CLI-Web.md":
        return "explicacao"
    if pasta.endswith(".md"):  # arquivo solto na raiz do vault (Estado-Atual.md)
        return None
    return _MODO_DIATAXIS_POR_PASTA.get(pasta)


def _aplicar_modo_diataxis(plano: dict[str, str]) -> dict[str, str]:
    resultado = {}
    for rel, conteudo in plano.items():
        modo = None if rel == "README-VAULT.md" else _modo_diataxis_para(rel, conteudo)
        if modo and re.search(r"^tags: \[.*?\]\n", conteudo, re.M):
            conteudo = re.sub(r"(^tags: \[.*?\]\n)", rf"\1modo: {modo}\n",
                               conteudo, count=1, flags=re.M)
        resultado[rel] = conteudo
    return resultado


def montar_plano() -> tuple[dict[str, str], dict[str, int], list[str]]:
    tecnicas = parse_tecnicas_catalog()
    status11 = parse_status_tecnicas_11()
    modulos = parse_modulos()
    tabela_validacoes = parse_tabela_consolidada()
    funcoes_principais = parse_funcoes_principais(modulos)
    funcoes_por_modulo = {f.modulo: f.nome for f in funcoes_principais}

    plano: dict[str, str] = {}
    avisos: list[str] = []

    p_tecnicas = gerar_tecnicas(tecnicas, status11, modulos)
    p_modulos = gerar_modulos(modulos, funcoes_por_modulo)
    p_funcoes = gerar_funcoes(funcoes_principais)
    p_conceitos, avisos_conceitos = gerar_conceitos(modulos, tecnicas)
    p_validacoes = gerar_validacoes(tabela_validacoes)
    p_tecnicas.update(gerar_tecnicas_fora_do_catalogo(modulos, p_validacoes))
    p_achados, p_decisoes = gerar_achados_e_decisoes(set(modulos))
    p_identidade = gerar_identidade(p_achados, p_decisoes)
    p_autoria = gerar_autoria_e_seguranca()
    p_mocs = gerar_mocs(tecnicas, modulos, p_tecnicas, p_validacoes, p_decisoes, p_achados)
    p_estado = gerar_estado_atual(status11)
    p_canvas = {}
    p_canvas.update(gerar_canvas_fluxo_cego())
    p_canvas.update(gerar_canvas_arquitetura(modulos))
    p_canvas.update(gerar_canvas_mapa_tecnicas(tecnicas, status11))

    # 35-Telas-e-Fluxos (Passo 186) e 08-Documentos (Passo 187) -- ver
    # docstrings das seções correspondentes acima.
    p_telas_cli = gerar_telas_cli(set(modulos))
    abas_web = parse_abas_web(set(modulos))
    p_abas_web = gerar_abas_web(abas_web)
    p_paridade = gerar_paridade_cli_web(set(modulos))
    p_moc_telas = gerar_moc_telas_fluxos(p_telas_cli, p_abas_web)
    p_telas_fluxos: dict[str, str] = {}
    for parte in (p_telas_cli, p_abas_web, p_paridade, p_moc_telas):
        p_telas_fluxos.update(parte)
    p_documentos = gerar_documentos()

    contagens = {
        "05-Identidade": len(p_identidade),
        "10-Tecnicas": len(p_tecnicas), "20-Modulos": len(p_modulos),
        "25-Funcoes": len(p_funcoes),
        "30-Conceitos": len(p_conceitos), "40-Validacoes": len(p_validacoes),
        "50-Decisoes": len(p_decisoes), "60-Achados": len(p_achados),
        "06-Autoria-e-Seguranca": len(p_autoria),
        "35-Telas-e-Fluxos": len(p_telas_fluxos), "08-Documentos": len(p_documentos),
    }
    p_readme = gerar_readme_vault(contagens)

    for parte in (p_tecnicas, p_modulos, p_funcoes, p_conceitos, p_validacoes, p_decisoes,
                  p_achados, p_identidade, p_autoria, p_mocs, p_estado, p_canvas,
                  p_telas_fluxos, p_documentos, p_readme):
        plano.update(parte)

    plano = _aplicar_modo_diataxis(plano)

    avisos.extend(avisos_conceitos)
    if not tabela_validacoes:
        avisos.append("Tabela consolidada de docs/VALIDACAO_PUBLICA.md não encontrada "
                       "ou vazia — 40-Validacoes/ ficará vazio.")
    if not status11:
        avisos.append("Tabela do Passo 160 não encontrada em docs/PROGRESSO.md — "
                       "10-Tecnicas/ e Estado-Atual.md ficarão sem status por técnica.")
    if not parse_paleta_mascote():
        avisos.append("Tabela de paleta em docs/DESIGN.md §1 não encontrada/vazia — "
                       "05-Identidade/Identidade-Visual.md ficará sem tabela de cores.")

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
