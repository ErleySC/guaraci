"""relatorio_cli.py -- `guaraci relatorio`: gera os relatorios de uma execucao pela CLI.

    guaraci relatorio PASTA_DA_EXECUCAO [--formato pdf,docx,xlsx,tex,pptx | todos]
                      [--nome TEXTO] [--autor TEXTO] [--instituicao TEXTO] [--tipo TEXTO]

Usa os mesmos geradores da aba de relatorios do web (`guaraci.reports`): nada de
segundo gerador. A pasta e' a de uma execucao ja feita (a que tem `Relatorios/`).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["executar", "FORMATOS"]

#: formato -> (funcao de guaraci.reports, usa dados do projeto?, extensao)
FORMATOS: Dict[str, Tuple[str, bool, str]] = {
    "pdf": ("generate_pdf_report", True, "pdf"),
    "docx": ("generate_word_report", True, "docx"),
    "xlsx": ("generate_excel_report", False, "xlsx"),
    "tex": ("generate_latex_template", True, "tex"),
    "pptx": ("generate_pptx_report", True, "pptx"),
}

#: formatos que o relatorio de uma execucao HSI ainda nao gera
FORMATOS_SEM_HSI = ("pptx",)

USO = ("Uso: guaraci relatorio PASTA_DA_EXECUCAO [--formato pdf,docx,xlsx,tex,pptx | todos]\n"
       "                        [--nome TEXTO] [--autor TEXTO] [--instituicao TEXTO] [--tipo TEXTO]\n"
       "  PASTA_DA_EXECUCAO  a pasta criada por uma execucao (a que tem Relatorios/); aceita a de uma execucao HSI\n"
       "  --formato          um ou mais, separados por virgula (padrao: pdf)\n"
       "  --nome, --autor, --instituicao, --tipo  dados do projeto na capa\n")


class _Uso(Exception):
    """Erro de uso (exit 2)."""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="guaraci relatorio", add_help=False, exit_on_error=False)
    p.add_argument("pasta", nargs="?")
    p.add_argument("--formato", default="pdf")
    p.add_argument("--nome", default="")
    p.add_argument("--autor", default="")
    p.add_argument("--instituicao", default="")
    p.add_argument("--tipo", default="")
    p.add_argument("--idioma", default=None)
    p.add_argument("--ajuda", "--help", "-h", action="store_true", dest="ajuda")
    return p


def formatos_pedidos(texto: str) -> List[str]:
    pedidos = list(FORMATOS) if texto.strip().lower() in ("todos", "all") else [f.strip().lower() for f in texto.split(",") if f.strip()]
    ruins = [f for f in pedidos if f not in FORMATOS]
    if ruins or not pedidos:
        raise _Uso(f"formato invalido {ruins or texto!r}; use {', '.join(FORMATOS)} ou todos.")
    return pedidos


def executar(argv: Optional[Sequence[str]] = None) -> int:
    """Roda `guaraci relatorio`. Devolve 0 (ok), 1 (falha ao gerar) ou 2 (uso)."""
    from guaraci import erros_entrada
    from guaraci.config import NOME_RELATORIOS, pasta_execucao
    args_l = list(sys.argv[1:] if argv is None else argv)
    lang = erros_entrada.idioma(args_l)
    try:
        try:
            args = _parser().parse_args(args_l)
        except argparse.ArgumentError as e:
            raise _Uso(f"argumento invalido: {e}") from e
        if args.ajuda:
            print(USO)
            return 0
        if not args.pasta:
            raise _Uso("informe a pasta de uma execucao.\n\n" + USO)
        rel = pasta_execucao(args.pasta, NOME_RELATORIOS)
        # execucao HSI: nao ha' resumo_modelo.txt; o relatorio parte do hsi_relatorio.txt
        e_hsi = (not os.path.isfile(os.path.join(rel, "resumo_modelo.txt"))
                 and os.path.isfile(os.path.join(rel, "hsi_relatorio.txt")))
        if not e_hsi and not os.path.isfile(os.path.join(rel, "resumo_modelo.txt")):
            raise _Uso(f"{args.pasta} nao parece a pasta de uma execucao (falta {NOME_RELATORIOS}/resumo_modelo.txt). "
                       "Rode a analise antes (guaraci run) e aponte a pasta que ela criou." if lang == "pt" else
                       f"{args.pasta} does not look like the folder of a run (missing {NOME_RELATORIOS}/resumo_modelo.txt). "
                       "Run the analysis first (guaraci run) and point to the folder it created.")
        pedidos = formatos_pedidos(args.formato)
        todos = args.formato.strip().lower() in ("todos", "all")
    except _Uso as e:
        print(f"{'Error' if lang == 'en' else 'Erro'}: {e}", file=sys.stderr)
        return 2
    from guaraci import reports
    projeto = {"nome": args.nome or os.path.basename(os.path.normpath(args.pasta)), "autor": args.autor,
               "inst": args.instituicao, "tipo": args.tipo}
    falhas = 0
    for fmt in pedidos:
        if e_hsi and fmt in FORMATOS_SEM_HSI:
            msg = (f"{fmt}: nao disponivel para execucoes HSI (use pdf, docx, xlsx ou tex)." if lang == "pt" else
                   f"{fmt}: not available for HSI runs (use pdf, docx, xlsx or tex).")
            print(f"  {msg}", file=sys.stderr)
            falhas += 0 if todos else 1
            continue
        nome_fn, usa_projeto, ext = FORMATOS[fmt]
        fn: Callable = getattr(reports, nome_fn)
        destino = os.path.join(rel, f"relatorio.{ext}")
        try:
            saida = fn(args.pasta, projeto) if usa_projeto else fn(args.pasta)
            dados = saida if isinstance(saida, (bytes, bytearray)) else saida.read()
            with open(destino, "wb") as fh:
                fh.write(dados)
            print(f"  {fmt}: {destino}")
        except Exception as e:  # noqa: BLE001 -- um formato que falha nao derruba os outros; o codigo de saida diz
            falhas += 1
            print(f"  {fmt}: FALHOU ({type(e).__name__}: {e})", file=sys.stderr)
    return 1 if falhas else 0
