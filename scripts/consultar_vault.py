# -*- coding: utf-8 -*-
"""Motor de consulta por grafo sobre o vault Obsidian do Guaraci (Passos
173-175). Lê as notas já geradas por `gerar_vault_obsidian.py`, monta o
grafo de wikilinks em memória, e para uma consulta devolve a nota mais
relevante + o contexto dela (1+ saltos de link) -- não só a nota isolada.

Uso:
    python scripts/consultar_vault.py "MCR-ALS"
    python scripts/consultar_vault.py "conformal.py" --saltos 2
    python scripts/consultar_vault.py --cobertura
    python scripts/consultar_vault.py --nota "raman-espectroscopia-raman"

Evidência ou silêncio: toda nota devolvida cita seu `fonte:`; se nada for
encontrado, o motor diz isso explicitamente -- nunca inventa uma resposta
fora do vault. `70-Notas-Pessoais/` e `99-Arquivo/` nunca são lidas (a
mesma fronteira de privacidade do gerador, Passo 169, se aplica aqui: o
motor só indexa o que o gerador já checou).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_AQUI = Path(__file__).resolve()
_SCRIPTS_DIR = _AQUI.parent
_RAIZ = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import gerar_vault_obsidian as gvo  # noqa: E402
from privacidade_amostras import (  # noqa: E402
    VazamentoDePrivacidade,
    checar_conteudos_ou_falhar,
)

_PASTAS_EXCLUIDAS = {gvo._PASTA_PROTEGIDA, "99-Arquivo"}
_PADRAO_WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_PADRAO_FONTE_PROGRESSO = re.compile(r"docs/PROGRESSO\.md:(\d+)")


# ═════════════════════════════════════════════════════════════════════════
#  Carregamento do vault e grafo de wikilinks
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class Nota:
    rel: str
    stem: str
    titulo: str
    tags: list[str]
    fonte: list[str]
    commit: str | None
    gerado_em: str | None
    corpo: str
    links_saida: set[str] = field(default_factory=set)
    links_entrada: set[str] = field(default_factory=set)

    def resumo(self, max_chars: int = 140) -> str:
        for linha in self.corpo.splitlines():
            # Só remove marcação markdown (crase, negrito, cabeçalho) --
            # NÃO underscore: `mcr_als.py` viraria "mcrals.py" e mancharia
            # o próprio nome do módulo no resumo.
            limpa = re.sub(r"[`*#]", "", linha).strip()
            if limpa and not limpa.startswith("---"):
                return limpa[:max_chars] + ("…" if len(limpa) > max_chars else "")
        return "(sem resumo)"


def _separar_frontmatter(texto: str) -> tuple[dict, str]:
    if not texto.startswith("---"):
        return {}, texto
    fim = texto.find("\n---", 3)
    if fim == -1:
        return {}, texto
    try:
        fm = yaml.safe_load(texto[3:fim]) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, texto[fim + 4:].lstrip("\n")


def carregar_vault(vault_dir: Path) -> dict[str, Nota]:
    """1 `Nota` por arquivo `.md` do vault, exceto `70-Notas-Pessoais/` e
    `99-Arquivo/` (nunca lidas -- mesma fronteira de privacidade do
    gerador). Constrói também os links de ENTRADA (backlinks), calculados
    depois de carregar tudo."""
    notas: dict[str, Nota] = {}
    for caminho in sorted(vault_dir.rglob("*.md")):
        partes = caminho.relative_to(vault_dir).parts
        if partes and partes[0] in _PASTAS_EXCLUIDAS:
            continue
        texto = caminho.read_text(encoding="utf-8")
        fm, corpo_completo = _separar_frontmatter(texto)
        titulo_m = re.search(r"^#\s+(.+)$", corpo_completo, re.M)
        if titulo_m:
            titulo = titulo_m.group(1).strip()
            corpo = corpo_completo[titulo_m.end():].strip()
        else:
            titulo = caminho.stem
            corpo = corpo_completo.strip()
        fonte_bruta = fm.get("fonte")
        fonte = fonte_bruta if isinstance(fonte_bruta, list) else (
            [str(fonte_bruta)] if fonte_bruta else [])
        links = {m.group(1).strip() for m in _PADRAO_WIKILINK.finditer(corpo)}
        notas[caminho.stem] = Nota(
            rel="/".join(partes), stem=caminho.stem, titulo=titulo,
            tags=[str(t) for t in (fm.get("tags") or [])], fonte=fonte,
            commit=fm.get("commit"), gerado_em=fm.get("gerado_em"),
            corpo=corpo, links_saida=links,
        )
    for nota in notas.values():
        for alvo in nota.links_saida:
            if alvo in notas:
                notas[alvo].links_entrada.add(nota.stem)
    return notas


# ═════════════════════════════════════════════════════════════════════════
#  Consulta: busca textual + navegação por grafo (Passo 173)
# ═════════════════════════════════════════════════════════════════════════

def buscar(termo: str, notas: dict[str, Nota]) -> list[Nota]:
    """Busca por (a) nome exato de nota, (b) título/tag, (c) corpo --
    nessa ordem de relevância. Não é só "grep": o resultado alimenta a
    expansão por grafo em `consultar()`, que é o que traz o contexto."""
    termo_low = termo.lower()
    grupos = [
        [n for n in notas.values() if n.stem.lower() == termo_low],
        [n for n in notas.values() if n.titulo.lower() == termo_low],
        [n for n in notas.values() if termo_low in n.titulo.lower()],
        [n for n in notas.values() if any(termo_low in t.lower() for t in n.tags)],
        [n for n in notas.values() if termo_low in n.corpo.lower()],
    ]
    vistos: set[str] = set()
    ordenadas: list[Nota] = []
    for grupo in grupos:
        for n in sorted(grupo, key=lambda x: x.titulo):
            if n.stem not in vistos:
                vistos.add(n.stem)
                ordenadas.append(n)
    return ordenadas


@dataclass
class ResultadoConsulta:
    termo: str
    principal: Nota | None
    correspondencias: list[Nota]
    vizinhos: list[tuple[Nota, int]]  # (nota, distância em saltos)


def consultar(termo: str, notas: dict[str, Nota], saltos: int = 1) -> ResultadoConsulta:
    correspondencias = buscar(termo, notas)
    principal = correspondencias[0] if correspondencias else None
    vizinhos: list[tuple[Nota, int]] = []
    if principal is not None:
        visitado = {principal.stem}
        fronteira = [principal.stem]
        for distancia in range(1, max(saltos, 0) + 1):
            proxima = []
            for stem in fronteira:
                atual = notas[stem]
                for alvo in atual.links_saida | atual.links_entrada:
                    if alvo in notas and alvo not in visitado:
                        visitado.add(alvo)
                        vizinhos.append((notas[alvo], distancia))
                        proxima.append(alvo)
            fronteira = proxima
    return ResultadoConsulta(termo=termo, principal=principal,
                              correspondencias=correspondencias, vizinhos=vizinhos)


# ═════════════════════════════════════════════════════════════════════════
#  Alerta de desatualização (Passo 174)
# ═════════════════════════════════════════════════════════════════════════

def head_atual(raiz: Path = gvo._RAIZ) -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=raiz,
                            capture_output=True, check=True)
        return r.stdout.decode("utf-8", "replace").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def aviso_desatualizacao(notas_envolvidas: list[Nota], head: str | None) -> str | None:
    if head is None:
        return None
    commits = {n.commit for n in notas_envolvidas if n.commit}
    divergentes = sorted(c for c in commits if c != head)
    if not divergentes:
        return None
    return (f"⚠️  AVISO: vault gerado no(s) commit(s) {', '.join(divergentes)}, "
            f"repositório está em {head} -- pode estar desatualizado, considere "
            "rodar `python scripts/gerar_vault_obsidian.py`.")


# ═════════════════════════════════════════════════════════════════════════
#  Formatação da resposta
# ═════════════════════════════════════════════════════════════════════════

def formatar_resultado(resultado: ResultadoConsulta, head: str | None) -> str:
    if resultado.principal is None:
        return (f"Nenhuma nota encontrada para '{resultado.termo}'. "
                "Evidência ou silêncio: sem nota no vault, sem resposta inventada.")

    todas = ([resultado.principal]
             + [n for n in resultado.correspondencias if n is not resultado.principal]
             + [n for n, _ in resultado.vizinhos])
    linhas: list[str] = []
    aviso = aviso_desatualizacao(todas, head)
    if aviso:
        linhas.append(aviso + "\n")

    linhas.append(f"# Consulta: {resultado.termo}\n")
    p = resultado.principal
    linhas.append(f"## Nota principal\n- **{p.titulo}** (`{p.rel}`) — {p.resumo()}")

    outras = [n for n in resultado.correspondencias if n is not resultado.principal]
    if outras:
        linhas.append("\n## Outras correspondências textuais")
        for n in outras:
            linhas.append(f"- **{n.titulo}** (`{n.rel}`) — {n.resumo()}")

    # Vizinho que já apareceu como correspondência textual não repete --
    # o leitor já viu a nota, repetir só adiciona ruído.
    ja_mostrados = {p.stem} | {n.stem for n in outras}
    vizinhos_novos = [(n, d) for n, d in resultado.vizinhos if n.stem not in ja_mostrados]
    if vizinhos_novos:
        max_d = max(d for _, d in vizinhos_novos)
        linhas.append(f"\n## Vizinhos por grafo (até {max_d} salto(s) de wikilink)")
        for n, d in vizinhos_novos:
            linhas.append(f"- [{d} salto{'s' if d > 1 else ''}] **{n.titulo}** "
                           f"(`{n.rel}`) — {n.resumo()}")

    return "\n".join(linhas)


# ═════════════════════════════════════════════════════════════════════════
#  Verificação de cobertura sob demanda (Passo 175)
# ═════════════════════════════════════════════════════════════════════════

def checar_cobertura_do_vault(notas: dict[str, Nota]) -> str:
    """Mesma auditoria do Passo 172 (`tests/test_cobertura_vault_obsidian.py`),
    rodada sob demanda contra o vault REALMENTE em disco -- não contra o
    plano em memória que o gerador produziria. Detecta tanto módulo/passo
    faltando quanto vault desatualizado em relação às fontes atuais."""
    linhas = ["# Cobertura do vault\n"]
    completa = True

    modulos_reais = {p.stem for p in (gvo._RAIZ / "src" / "guaraci").glob("*.py")}
    modulos_vault = {
        n.stem[:-3] for n in notas.values()
        if n.rel.startswith("20-Modulos/") and n.stem.endswith(".py")
    }
    faltando_mod = sorted(modulos_reais - modulos_vault)
    sobrando_mod = sorted(modulos_vault - modulos_reais)
    ok_mod = not faltando_mod and not sobrando_mod
    completa &= ok_mod
    linhas.append(f"- **Módulos**: {len(modulos_vault)}/{len(modulos_reais)} "
                   + ("OK" if ok_mod else f"FALTANDO {faltando_mod} SOBRANDO {sobrando_mod}"))

    tecnicas_reais = gvo.parse_tecnicas_catalog()
    n_tecnicas_vault = sum(1 for n in notas.values() if n.rel.startswith("10-Tecnicas/"))
    ok_tec = n_tecnicas_vault == len(tecnicas_reais)
    completa &= ok_tec
    linhas.append(f"- **Técnicas**: {n_tecnicas_vault}/{len(tecnicas_reais)} "
                   + ("OK" if ok_tec else "CONTAGEM DIVERGENTE"))

    blocos = gvo.parse_blocos_passo()
    linhas_citadas: set[int] = set()
    for n in notas.values():
        if n.rel.startswith(("60-Achados/", "50-Decisoes/")):
            for f in n.fonte:
                m = _PADRAO_FONTE_PROGRESSO.search(f)
                if m:
                    linhas_citadas.add(int(m.group(1)))
    sem_rastro = [b.titulo for b in blocos
                  if not any(b.linha_inicio <= ln < b.linha_fim for ln in linhas_citadas)]
    ok_passos = not sem_rastro
    completa &= ok_passos
    linhas.append(f"- **Passos de PROGRESSO.md**: {len(blocos) - len(sem_rastro)}/{len(blocos)} "
                   + ("OK" if ok_passos else
                      f"SEM RASTRO: {sem_rastro[:5]}{' ...' if len(sem_rastro) > 5 else ''}"))

    tabela = gvo.parse_tabela_consolidada()
    n_validacoes_vault = sum(1 for n in notas.values() if n.rel.startswith("40-Validacoes/"))
    ok_val = n_validacoes_vault == len(tabela)
    completa &= ok_val
    linhas.append(f"- **Datasets (VALIDACAO_PUBLICA.md)**: {n_validacoes_vault}/{len(tabela)} "
                   + ("OK" if ok_val else "CONTAGEM DIVERGENTE"))

    head = head_atual()
    aviso = aviso_desatualizacao(list(notas.values()), head)
    if aviso:
        linhas.append(f"\n{aviso}")
        completa = False

    linhas.append(f"\n**Cobertura {'COMPLETA' if completa else 'INCOMPLETA'}.**")
    return "\n".join(linhas)


# ═════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════

def main() -> int:
    gvo._fixar_utf8_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("termo", nargs="?", help="termo, título ou nome de nota a consultar")
    ap.add_argument("--vault", default=None,
                     help="pasta do vault (senão GUARACI_VAULT_DIR / ~/GuaraciVault)")
    ap.add_argument("--saltos", type=int, default=1,
                     help="quantos saltos de wikilink expandir a partir da nota principal")
    ap.add_argument("--cobertura", action="store_true",
                     help="roda a auditoria de cobertura do vault (Passo 175) e sai")
    args = ap.parse_args()

    vault_dir = gvo._resolve_vault_dir(args.vault)
    if not vault_dir.exists():
        print(f"Vault não encontrado em {vault_dir}. Rode "
              "`python scripts/gerar_vault_obsidian.py` primeiro.", file=sys.stderr)
        return 1
    notas = carregar_vault(vault_dir)

    if args.cobertura:
        saida = checar_cobertura_do_vault(notas)
        codigo_sucesso = 0
    elif not args.termo:
        ap.print_usage(sys.stderr)
        return 2
    else:
        resultado = consultar(args.termo, notas, saltos=args.saltos)
        saida = formatar_resultado(resultado, head_atual())
        codigo_sucesso = 0 if resultado.principal else 1

    try:
        checar_conteudos_ou_falhar({"resposta-do-motor-de-consulta": saida})
    except VazamentoDePrivacidade as e:
        print("FALHA — a guarda de privacidade acusou algo na resposta; "
              "resposta suprimida (Passo 169 se aplica também ao motor de consulta).",
              file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 1

    print(saida)
    return codigo_sucesso


if __name__ == "__main__":
    raise SystemExit(main())
