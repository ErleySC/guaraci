#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Remove metadados de PROVENIENCIA de arquivos JCAMP-DX, em copias.

POR QUE. Um `.dx` gravado por espectrometro carrega, no cabecalho, muito
mais do que o espectro: `##AUDIT TRAIL` grava data/hora, **operador** e
**local** de cada leitura; `##$Detector model` e `##$Spectrometer model`
identificam o equipamento. Nada disso e' necessario para a analise
quimiometrica, e tudo isso e' dado sobre PESSOAS e INSTITUICOES quando os
arquivos vem de um acervo compartilhado.

O GUARACI nunca le esses campos (ver `dados_io.parse_dx`), entao rodar o
software sobre os arquivos e' seguro. Este script serve para o outro caso:
quando os PROPRIOS ARQUIVOS vao sair da sua maquina -- deposito num
repositorio de dados, material suplementar de artigo, envio a um
colaborador.

GARANTIA DE SEGURANCA: escreve SEMPRE em um diretorio de saida separado e
**recusa** sobrescrever os originais. Um script de sanitizacao que apaga o
original em caso de bug destroi a unica copia da evidencia.

Uso:
    python scripts/sanitizar_dx.py <pasta_entrada> <pasta_saida>
    python scripts/sanitizar_dx.py <pasta_entrada> <pasta_saida> --conferir

    --conferir   apos escrever, varre a saida e falha se qualquer campo
                 sensivel sobreviveu (recomendado antes de publicar)

O `##TITLE` NAO e' removido por padrao: dele saem classe, teor e o
agrupamento de replicas, sem os quais os arquivos deixam de ser
analisaveis. Se o proprio identificador de amostra for sensivel no seu
caso, use `--anonimizar-titulo`, que o substitui por um rotulo sequencial
(`AMOSTRA_0001_T1`) preservando a estrutura de replicas -- e grava o mapa
de correspondencia em `mapa_titulos.csv`, na saida, para que voce possa
reverter localmente. NUNCA publique esse mapa junto com os espectros.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

#: Prefixos de linha JCAMP removidos por completo. `AUDIT TRAIL` e' um
#: bloco multilinha: as linhas seguintes que comecam com '(' fazem parte
#: dele e tambem saem (ver `_e_continuacao_de_audit`).
PREFIXOS_REMOVIDOS = (
    "##AUDIT TRAIL",
    "##$DETECTOR MODEL",
    "##$SPECTROMETER MODEL",
    "##$ACCESSORY MODEL",
    "##OWNER",
    "##ORIGIN",
    "##OPERATOR",
    "##$OPERATOR",
    "##SOURCE REFERENCE",
    "##SAMPLE DESCRIPTION",
    "##$PATH",
    "##$FILENAME",
)

#: Padroes que denunciam sobrevivencia de proveniencia, usados por
#: --conferir.
#:
#: PRECISO, nao amplo. A primeira versao alarmava em qualquer `##$Detector*`
#: e disparava em 100% dos arquivos reais por causa de `##$Detector Gain=1`
#: -- que e' PARAMETRO DE AQUISICAO (ganho do detector, metodologicamente
#: relevante, deve ser preservado), nao identificacao. Um alarme que dispara
#: sempre ensina quem o le' a ignora-lo, e ai ele nao protege mais nada.
#: `model` e' o campo que carrega numero de serie; `Gain`/`Resolution`/
#: `Scans` descrevem a medida e ficam.
_RE_SUSPEITO = re.compile(
    r"##AUDIT|##\$DETECTOR\s+MODEL|##\$SPECTROMETER\s+MODEL|"
    r"##\$ACCESSORY\s+MODEL|##OWNER|##ORIGIN|##OPERATOR|##\$OPERATOR|"
    r"##\$PATH|##\$FILENAME|Instrument ID|Detector ID",
    re.IGNORECASE)

_RE_TITULO = re.compile(r"^\s*##TITLE\s*=\s*(?P<valor>.*)$", re.IGNORECASE)
#: Sufixo de replica no fim do TITLE (T1/T2/T3 ou T_1), preservado na
#: anonimizacao: sem ele, `mae_id` nao consegue reagrupar as replicas e a
#: validacao group-aware -- a razao de o arquivo existir -- vai embora.
_RE_REPLICA = re.compile(r"[-_]?T_?(?P<n>\d+)\s*$", re.IGNORECASE)


def _e_continuacao_de_audit(linha: str) -> bool:
    """Linhas do bloco AUDIT TRAIL comecam com '(' — ex.:
    `( 1, <2021/01/06 14:15:37 -03>, <fulano>, <instituicao>, <...>)`."""
    return linha.lstrip().startswith("(")


def _chave_de_grupo(titulo: str) -> str:
    """Parte do TITLE que identifica o PONTO FISICO (sem a replica)."""
    return _RE_REPLICA.sub("", titulo).strip()


def sanitizar_texto(texto: str, novo_titulo: str | None = None) -> str:
    """Devolve o conteudo do `.dx` sem os campos de proveniencia.

    `novo_titulo`, se dado, substitui o valor de `##TITLE=` (o campo
    permanece: removE-lo quebraria o parsing de classe/teor).
    """
    saida: list[str] = []
    dentro_de_audit = False
    for linha in texto.splitlines():
        nu = linha.lstrip().upper()
        if any(nu.startswith(p) for p in PREFIXOS_REMOVIDOS):
            dentro_de_audit = nu.startswith("##AUDIT TRAIL")
            continue
        if dentro_de_audit:
            if _e_continuacao_de_audit(linha):
                continue
            dentro_de_audit = False
        if novo_titulo is not None:
            m = _RE_TITULO.match(linha)
            if m:
                saida.append(f"##TITLE={novo_titulo}")
                continue
        saida.append(linha)
    return "\n".join(saida) + "\n"


def _ler_titulo(texto: str) -> str | None:
    for linha in texto.splitlines():
        m = _RE_TITULO.match(linha)
        if m:
            return m.group("valor").strip()
        if linha.lstrip().upper().startswith(("##XYDATA", "##XYPOINTS")):
            break
    return None


def sanitizar_pasta(entrada: Path, saida: Path, *,
                    anonimizar_titulo: bool = False) -> dict:
    if not entrada.is_dir():
        raise SystemExit(f"Erro: '{entrada}' nao e' um diretorio.")
    entrada = entrada.resolve()
    saida = saida.resolve()
    if saida == entrada or entrada in saida.parents:
        raise SystemExit(
            "Erro: a pasta de saida nao pode ser a de entrada nem estar "
            "dentro dela. Este script NUNCA sobrescreve os originais -- eles "
            "sao a unica copia da evidencia.")

    arquivos = sorted(entrada.rglob("*.dx"))
    if not arquivos:
        raise SystemExit(f"Nenhum .dx encontrado em '{entrada}'.")

    mapa: dict[str, str] = {}       # titulo original -> titulo anonimo
    grupos: dict[str, int] = {}     # chave de ponto fisico -> indice
    escritos = 0

    for arq in arquivos:
        texto = arq.read_text(encoding="latin-1", errors="replace")
        novo = None
        if anonimizar_titulo:
            titulo = _ler_titulo(texto)
            if titulo:
                if titulo in mapa:
                    novo = mapa[titulo]
                else:
                    chave = _chave_de_grupo(titulo)
                    if chave not in grupos:
                        grupos[chave] = len(grupos) + 1
                    m = _RE_REPLICA.search(titulo)
                    rep = m.group("n") if m else "1"
                    novo = f"AMOSTRA_{grupos[chave]:04d}_T{rep}"
                    mapa[titulo] = novo

        destino = saida / arq.relative_to(entrada)
        if anonimizar_titulo and novo:
            destino = destino.with_name(novo + ".dx")
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(sanitizar_texto(texto, novo),
                           encoding="latin-1")
        escritos += 1

    if anonimizar_titulo and mapa:
        caminho_mapa = saida / "mapa_titulos.csv"
        with open(caminho_mapa, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["titulo_original", "titulo_anonimo"])
            w.writerows(sorted(mapa.items()))
        print(f"  mapa de correspondencia: {caminho_mapa}")
        print("  ATENCAO: NUNCA publique mapa_titulos.csv junto com os "
              "espectros -- ele desfaz a anonimizacao.")
    return {"lidos": len(arquivos), "escritos": escritos,
            "grupos": len(grupos)}


def conferir(saida: Path) -> int:
    """Varre a saida e devolve o numero de arquivos com campo suspeito."""
    problemas = 0
    for arq in sorted(saida.rglob("*.dx")):
        texto = arq.read_text(encoding="latin-1", errors="replace")
        achados = sorted({m.group(0) for m in _RE_SUSPEITO.finditer(texto)})
        if achados:
            problemas += 1
            if problemas <= 10:
                print(f"  [SUSPEITO] {arq.name}: {achados}")
    return problemas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Sanitiza metadados de proveniencia de arquivos JCAMP-DX "
                    "(escreve COPIAS; nunca sobrescreve os originais).")
    ap.add_argument("entrada", type=Path, help="pasta com os .dx originais")
    ap.add_argument("saida", type=Path, help="pasta de destino (sera criada)")
    ap.add_argument("--anonimizar-titulo", action="store_true",
                    help="substitui ##TITLE= por rotulo sequencial, "
                         "preservando o agrupamento de replicas")
    ap.add_argument("--conferir", action="store_true",
                    help="apos escrever, varre a saida e falha se algum "
                         "campo sensivel sobreviveu")
    a = ap.parse_args(argv)

    print(f"Entrada : {a.entrada}")
    print(f"Saida   : {a.saida}")
    res = sanitizar_pasta(a.entrada, a.saida,
                          anonimizar_titulo=a.anonimizar_titulo)
    print(f"  {res['escritos']} de {res['lidos']} arquivos escritos"
          + (f" | {res['grupos']} pontos fisicos anonimizados"
             if a.anonimizar_titulo else ""))

    if a.conferir:
        print("Conferindo a saida...")
        problemas = conferir(a.saida)
        if problemas:
            print(f"FALHOU: {problemas} arquivo(s) ainda contem campo "
                  f"sensivel. NAO publique.", file=sys.stderr)
            return 1
        print("  OK: nenhum campo de proveniencia encontrado na saida.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
