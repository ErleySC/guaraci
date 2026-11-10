#!/usr/bin/env python3
"""percorrer_ajuda_instalada.py -- percorre TODA a ajuda do programa INSTALADO e reprova se algo aponta para fora do pacote.

    <python-do-venv-com-guaraci-instalado> -I scripts/percorrer_ajuda_instalada.py [--permitir-repo]

Roda no interprete onde o `guaraci` foi instalado (wheel/sdist, nao o checkout). Para cada topico do catalogo (opcoes, tecnicas,
graficos, telas, comandos, termos), cada topico/secao de `guaraci manual`, cada `--help` de subcomando e cada regra do assistente,
executa a resposta e procura: ponteiro para `docs/`, `MANUAL.md`/`SECURITY.md`, "nao encontrado", Traceback, codigo de saida inesperado.
Imprime UMA linha `RESUMO ...` com contagens; sai com 1 se houve problema. Sem rede, sem escrita em disco.
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path
from typing import Callable, List, Tuple

PROIBIDO = re.compile(r"docs/|MANUAL\.md|SECURITY\.md|CONTRIBUTING\.md|n[aã]o encontrad|Traceback", re.IGNORECASE)


def _capturar(fn: Callable[[], object]) -> Tuple[object, str]:
    saida, erro = io.StringIO(), io.StringIO()
    rc: object = None
    with contextlib.redirect_stdout(saida), contextlib.redirect_stderr(erro):
        try:
            rc = fn()
        except SystemExit as e:
            rc = e.code if e.code is not None else 0
    return rc, saida.getvalue() + erro.getvalue()


def percorrer() -> Tuple[List[str], dict]:
    from guaraci import assistente as ass
    from guaraci import catalogo as cat
    from guaraci import cli_comandos as cmd
    from guaraci import docs_pacote as dp
    from guaraci.catalogo_duvidas import DUVIDAS

    problemas: List[str] = []
    contagem = {"ajuda": 0, "manual": 0, "help": 0, "assistente": 0}

    def checar(rotulo: str, rc: object, texto: str, esperado=(0, None)) -> None:
        if rc not in esperado:
            problemas.append(f"{rotulo}: codigo de saida {rc!r}")
        achado = PROIBIDO.search(texto)
        if achado:
            problemas.append(f"{rotulo}: texto proibido {achado.group(0)!r}")
        if not texto.strip():
            problemas.append(f"{rotulo}: resposta vazia")

    rc, txt = _capturar(lambda: cmd.subcomando("ajuda", []))
    checar("ajuda (mapa)", rc, txt)
    ids = (list(cat.OPCOES) + [t["id"] for t in cat.tecnicas()] + [g["id"] for g in cat.graficos()] + list(cat.TELAS)
           + [c["id"] for c in cat.comandos()] + list(cat.TERMOS))
    for i in ids:
        rc, txt = _capturar(lambda i=i: cmd.subcomando("ajuda", [i]))
        checar(f"ajuda {i}", rc, txt)
        contagem["ajuda"] += 1

    rc, txt = _capturar(lambda: cmd.subcomando("manual", []))
    checar("manual (lista)", rc, txt)
    alvos = list(dp.TOPICOS) + sorted({ident for d in DUVIDAS.values() for tipo, ident in d["onde"] if tipo == "manual"})
    for a in alvos:
        rc, txt = _capturar(lambda a=a: cmd.subcomando("manual", [a]))
        # o texto do manual e dos documentos pode mencionar caminhos do repositorio: so' o codigo de saida conta
        if rc != 0:
            problemas.append(f"manual {a}: codigo de saida {rc!r}")
        if not txt.strip():
            problemas.append(f"manual {a}: vazio")
        contagem["manual"] += 1

    import guaraci.guaraci as g
    for c in cmd.SUBCOMANDOS:
        if c in ("ajuda", "buscar", "manual", "assistente", "confianca", "exemplo"):
            continue                                   # estes recebem argumento proprio; ja percorridos acima / abaixo
        rc, txt = _capturar(lambda c=c: g.main([c, "--help"]))
        checar(f"{c} --help", rc, txt)
        contagem["help"] += 1

    for args in (["--help"], ["--version"], ["perfis"], ["confianca"], ["confianca", "pls_da"]):
        rc, txt = _capturar(lambda args=args: g.main(list(args)))
        checar(" ".join(["guaraci", *args]), rc, txt)
        contagem["help"] += 1

    perguntas = [d["gatilhos"][0] for d in DUVIDAS.values()] + ["qual a capital da franca"]
    for q in perguntas:
        rc, txt = _capturar(lambda q=q: print(ass.responder(q, "PT")))
        checar(f"assistente {q!r}", rc, txt)
        contagem["assistente"] += 1
    return problemas, contagem


def main() -> int:
    import guaraci
    if "--permitir-repo" not in sys.argv and "site-packages" not in Path(guaraci.__file__).as_posix():
        print("RESUMO ajuda_instalada rc=2 resumo=\"o guaraci importado nao esta instalado (veio do checkout)\"")
        return 2
    problemas, n = percorrer()
    for p in problemas[:30]:
        print("PROBLEMA", p)
    print(f"RESUMO ajuda_instalada rc={1 if problemas else 0} resumo=\"{n['ajuda']} topicos, {n['manual']} manual, {n['help']} --help, "
          f"{n['assistente']} regras; {len(problemas)} problema(s)\"")
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
