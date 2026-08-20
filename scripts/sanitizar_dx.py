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
de correspondencia num CSV **irmao** da saida
(`<saida>_mapa_titulos.csv`, fora da pasta que se publica), para que voce possa
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
    # Prosa livre. Nos arquivos reais carrega descricao da amostra e, em
    # alguns gravadores, quem leu e onde. Nao e' parametro de aquisicao.
    "##COMMENTS",
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

#: Formato de identificador de amostra do acervo de origem
#: (`COD-DD-MM-AAAA`). Usado SO' por `conferir()`, sobre nomes de arquivo e
#: de pasta -- nao para remover, e sim para LISTAR o que um humano precisa
#: olhar. Mesmo padrao de `tests/test_sem_identificador_real.py`.
_RE_ID_AMOSTRA = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]{2,5}[-_]\d{2}-\d{2}-(?:19|20)\d{2}(?!\d)",
    re.IGNORECASE,
)

_RE_TITULO = re.compile(r"^\s*##TITLE\s*=\s*(?P<valor>.*)$", re.IGNORECASE)
#: Sufixo de replica no fim do TITLE (T1/T2/T3 ou T_1), preservado na
#: anonimizacao: sem ele, `mae_id` nao consegue reagrupar as replicas e a
#: validacao group-aware -- a razao de o arquivo existir -- vai embora.
_RE_REPLICA = re.compile(r"[-_]?T_?(?P<n>\d+)\s*$", re.IGNORECASE)


#: JCAMP-DX manda ignorar espaco, hifen, sublinhado e caixa em nomes de
#: rotulo (LABEL EQUIVALENCE). O gravador pode emitir `##AUDIT TRAIL`,
#: `##AUDITTRAIL` ou `##AUDIT-TRAIL` para o mesmo campo. A primeira versao
#: comparava a string crua e so' pegava a forma canonica -- as outras duas
#: passavam inteiras. Achado na verificacao independente de 2026-08-20.
_RE_NAO_SIGNIFICATIVO = re.compile(r"[\s\-_/]+")


def _normalizar_rotulo(linha: str) -> str:
    """Rotulo em forma canonica para comparacao: sem espaco/hifen/sublinhado,
    em maiuscula. `##AUDIT-TRAIL=` e `##audit trail =` viram `##AUDITTRAIL`."""
    cabeca = linha.lstrip().split("=", 1)[0]
    return _RE_NAO_SIGNIFICATIVO.sub("", cabeca).upper()


#: Prefixos em forma normalizada, derivados de PREFIXOS_REMOVIDOS -- para nao
#: existirem duas listas que possam divergir.
#:
#: Cada entrada gera as DUAS formas, `##X` e `##$X`. O prefixo `##$` marca
#: rotulo especifico do fabricante e aparece no dado real: `##$OWNER=`
#: escapava porque `"##$OWNER".startswith("##OWNER")` e' False. A lista ja'
#: trazia `##$OPERATOR` e `##$PATH` escritos a mao -- prova de que a variante
#: existe e de que enumera-la manualmente por rotulo nao escala. Achado na
#: verificacao independente de 2026-08-20.
def _formas_normalizadas(prefixo: str) -> tuple[str, str]:
    base = _RE_NAO_SIGNIFICATIVO.sub("", prefixo).upper()
    sem = base.replace("$", "", 1) if base.startswith("##$") else base
    com = sem.replace("##", "##$", 1)
    return sem, com


_PREFIXOS_NORM = tuple(sorted({
    forma for p in PREFIXOS_REMOVIDOS for forma in _formas_normalizadas(p)
}))

#: Uma linha de continuacao do AUDIT TRAIL comeca com '(' na forma canonica.
#: Assumir que SEMPRE comeca era o furo mais grave: se nao comecasse, o
#: cabecalho `##AUDIT TRAIL` saia, a linha com operador e local FICAVA, e o
#: --conferir reportava limpo porque o padrao que ele procurava tinha saido
#: junto. Agora a continuacao e' tudo que nao abre um rotulo novo (`##`).
#:
#: E vale para TODO rotulo removido, nao so' o AUDIT TRAIL. A versao anterior
#: ligava o rastreio apenas para `##AUDITTRAIL`, entao `##COMMENTS` e
#: `##SAMPLE DESCRIPTION` multilinha perdiam o cabecalho e mantinham a linha
#: seguinte -- com operador e local -- exatamente o bug que esta funcao
#: declarava corrigido, cometido para outros rotulos.
def _e_continuacao_de_audit(linha: str) -> bool:
    """Continuacao de um rotulo de valor longo: qualquer linha que nao inicie
    um rotulo JCAMP novo. Inclui a forma canonica do AUDIT TRAIL
    `( 1, <data>, <operador>, ...)` e as variantes sem parentese."""
    s = linha.lstrip()
    if not s:
        return False
    return not s.startswith("##")


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
        rotulo = _normalizar_rotulo(linha)
        if any(rotulo.startswith(p) for p in _PREFIXOS_NORM):
            # Todo rotulo removido pode ter valor multilinha, nao so' o
            # AUDIT TRAIL -- ver `_e_continuacao_de_audit`.
            dentro_de_audit = True
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


def _pasta_anonima(nome: str, mapa: dict[str, str]) -> str:
    """Nome de pasta anonimo e ESTAVEL para a mesma pasta de origem.

    A arvore do acervo e' organizada por classe, e o nome da pasta costuma
    repetir o identificador ou a especie. Preservar a estrutura importa (o
    agrupamento por pasta e' informacao metodologica); preservar o NOME, nao.
    """
    if nome not in mapa:
        mapa[nome] = f"GRUPO_{len(mapa) + 1:03d}"
    return mapa[nome]


def sanitizar_pasta(entrada: Path, saida: Path, *,
                    anonimizar_titulo: bool = False) -> dict:
    pastas: dict[str, str] = {}
    sem_titulo = 0
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

        relativo = arq.relative_to(entrada)
        if anonimizar_titulo:
            # O CONTEUDO era anonimizado e o CAMINHO nao: a arvore de saida
            # reproduzia `<identificador_real>/<arquivo>.dx`, e `conferir()`
            # so' lia conteudo, entao nada acusava. Achado na verificacao
            # independente de 2026-08-20.
            partes = [_pasta_anonima(p, pastas) for p in relativo.parts[:-1]]
            # Sem TITLE parseavel `novo` fica None -- antes disso mantinha o
            # nome ORIGINAL, que e' identificador tanto quanto o TITLE.
            if novo is None:
                sem_titulo += 1
                novo = f"AMOSTRA_SEM_TITULO_{sem_titulo:04d}"
            destino = saida.joinpath(*partes, novo + ".dx")
        else:
            destino = saida / relativo
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(sanitizar_texto(texto, novo),
                           encoding="latin-1")
        escritos += 1

    if anonimizar_titulo and mapa:
        # FORA de `saida`, de proposito: `saida` e' a pasta que se
        # publica. Grava-lo dentro dela era um vazamento estrutural que
        # nenhuma lista de rotulos protegidos cobria -- `conferir()` so'
        # varre *.dx dentro de `saida`. Um sibling de `saida` nao pode
        # ser incluido por acidente quando alguem publica o CONTEUDO de
        # `saida`. Achado na verificacao independente de 2026-08-20.
        caminho_mapa = saida.parent / f"{saida.name}_mapa_titulos.csv"
        with open(caminho_mapa, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["titulo_original", "titulo_anonimo"])
            w.writerows(sorted(mapa.items()))
        print(f"  mapa de correspondencia: {caminho_mapa}")
        print("  ATENCAO: NUNCA copie esse arquivo para dentro da pasta "
              "que sera publicada -- ele desfaz a anonimizacao.")
    return {"lidos": len(arquivos), "escritos": escritos,
            "grupos": len(grupos)}


#: Rotulos JCAMP que sao PARAMETRO da medida e devem sobreviver. Tudo que
#: nao estiver aqui, e nao for dado numerico, e' listado para inspecao
#: humana -- a lista e' de PERMISSAO, nao de proibicao.
_ROTULOS_ESPERADOS = frozenset(_RE_NAO_SIGNIFICATIVO.sub("", r).upper() for r in (
    "##TITLE", "##JCAMP-DX", "##DATA TYPE", "##DATATYPE", "##CLASS",
    "##DATE", "##TIME", "##LONGDATE", "##XUNITS", "##YUNITS", "##XFACTOR",
    "##YFACTOR", "##FIRSTX", "##LASTX", "##NPOINTS", "##FIRSTY", "##MAXY",
    "##MINY", "##DELTAX", "##RESOLUTION", "##XYDATA", "##XYPOINTS", "##END",
    "##$RESOLUTION", "##$SCANS", "##$DETECTOR GAIN", "##$ZERO FILLING",
    "##$APODIZATION", "##$PHASE CORRECTION", "##$ACQUISITION MODE",
    "##$SAMPLE SCANS", "##$BACKGROUND SCANS", "##$LASER FREQUENCY",
    "##$HIGH FOLDING LIMIT", "##$LOW FOLDING LIMIT",
))


def conferir(saida: Path) -> tuple[int, list[str]]:
    """Verificacao INDEPENDENTE da regra de remocao.

    POR QUE NAO REUSA `_RE_SUSPEITO` SOZINHO. A versao anterior procurava os
    MESMOS padroes que `sanitizar_texto` removia. Quando a remocao errava
    tirando de menos -- o caso medido: cabecalho `##AUDIT TRAIL` removido e a
    linha de continuacao com operador e local deixada para tras -- o
    conferidor reportava LIMPO, porque o padrao que ele procurava tinha saido
    junto com o cabecalho. Um conferidor que so' ve' o que o removedor ve'
    nao confere nada.

    Agora sao duas verificacoes independentes:

    1. `_RE_SUSPEITO` -- padroes conhecidos de proveniencia (mantido).
    2. **Lista de permissao de rotulos.** Todo rotulo `##...` que nao esteja
       em `_ROTULOS_ESPERADOS`, e toda linha de texto livre fora do bloco
       de dados, sao LISTADOS para inspecao humana.

    Devolve `(n_arquivos_com_suspeita, linhas_para_inspecao)`. Nao devolve
    veredito binario "limpo": a segunda verificacao produz material para
    alguem olhar, nao uma aprovacao.
    """
    problemas = 0
    inspecionar: list[str] = []
    for arq in sorted(saida.rglob("*.dx")):
        # O CAMINHO tambem e' metadado. `--anonimizar-titulo` limpava o
        # conteudo e deixava `<identificador_real>/<arquivo>.dx` de pe',
        # porque esta funcao so' lia o conteudo.
        rel = arq.relative_to(saida).as_posix()
        achados_caminho = sorted({m.group(0) for m in _RE_SUSPEITO.finditer(rel)})
        if achados_caminho:
            problemas += 1
            print(f"  [SUSPEITO NO CAMINHO] {rel}: {achados_caminho}")
        for m in _RE_ID_AMOSTRA.finditer(rel):
            inspecionar.append(
                f"{rel}: nome/caminho parece identificador de amostra "
                f"({m.group(0)})")
        texto = arq.read_text(encoding="latin-1", errors="replace")
        achados = sorted({m.group(0) for m in _RE_SUSPEITO.finditer(texto)})
        if achados:
            problemas += 1
            if problemas <= 10:
                print(f"  [SUSPEITO] {arq.name}: {achados}")

        em_dados = False
        for n, linha in enumerate(texto.splitlines(), start=1):
            s = linha.strip()
            if not s:
                continue
            rotulo = _normalizar_rotulo(linha)
            if rotulo.startswith("##XYDATA") or rotulo.startswith("##XYPOINTS"):
                em_dados = True
                continue
            if rotulo.startswith("##END"):
                em_dados = False
                continue
            if em_dados:
                continue
            if s.startswith("##"):
                if rotulo not in _ROTULOS_ESPERADOS:
                    inspecionar.append(f"{arq.name}:{n}: {s[:100]}")
            else:
                # Texto livre fora do bloco de dados: continuacao de algum
                # rotulo. E' exatamente onde operador e local sobrevivem.
                inspecionar.append(f"{arq.name}:{n}: {s[:100]}")
    return problemas, inspecionar


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
        problemas, inspecionar = conferir(a.saida)
        if problemas:
            print(f"FALHOU: {problemas} arquivo(s) ainda contem campo "
                  f"sensivel conhecido. NAO publique.", file=sys.stderr)
            return 1
        print("  Os padroes conhecidos de proveniencia nao aparecem na "
              "saida. Isso nao e' um atestado de limpeza: cobre o que este "
              "script conhece, e nada mais.")
        if inspecionar:
            print(f"\n  {len(inspecionar)} linha(s) fora da lista de rotulos "
                  f"esperados. Isto NAO e' um veredito: e' material para voce")
            print("  olhar antes de publicar. Um campo novo do gravador cai "
                  "aqui, e nenhuma lista de proibicao o pegaria.")
            for linha in inspecionar[:40]:
                print(f"    {linha}")
            if len(inspecionar) > 40:
                print(f"    ... e mais {len(inspecionar) - 40} linha(s).")
        else:
            print("  Nenhuma linha fora da lista de rotulos esperados.")
        print("\n  Confira alguns arquivos a mao mesmo assim: este script "
              "nao substitui inspecao humana.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
