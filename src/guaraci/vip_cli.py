"""vip_cli.py -- `guaraci vip`: VIP do PLS com estabilidade nas dobras agrupadas.

    guaraci vip TABELA.csv (--classe COL | --resposta COL) --grupo COL
                [--variaveis a,b,c] [--componentes 2] [--limiar 1] [--divisoes 5]
                [--repeticoes 10] [--semente 0] [--saida PASTA] [--sem-figuras]
                [--idioma pt|en]

`--classe` roda PLS-DA (Y = classes em one-hot); `--resposta` roda PLS de regressao.
`--grupo` e' o id da amostra fisica/mae de cada linha: replicas com o MESMO id
ficam juntas em cada dobra. Sem `--grupo` o comando roda, mas AVISA que o
resultado fica otimista. As colunas restantes sao as variaveis (todas numericas),
salvo `--variaveis`. O calculo e' o de `guaraci.vip_estabilidade`.
"""
from __future__ import annotations

import argparse
import os

from guaraci.config import pasta_execucao, nome_pasta
import sys
from typing import Any, Dict, List, Optional, Sequence

from guaraci import erros_entrada as _erros
from guaraci.relatorio_blocos import renderizar_html, renderizar_md, renderizar_tex

__all__ = ["executar"]

_ESTADO: Dict[str, str] = {"lang": "pt"}

_MSG: Dict[str, Any] = {
    "arquivo": ("arquivo nao encontrado: {c}", "file not found: {c}"),
    "leitura": ("nao consegui ler {c} como CSV ({e})", "could not read {c} as CSV ({e})"),
    "arg": ("argumento invalido: {e}", "invalid argument: {e}"),
    "sem_tabela": ("informe a tabela CSV.", "provide the CSV table."),
    "alvo": ("informe --classe COL (PLS-DA) ou --resposta COL (PLS de regressao), um dos dois.",
             "provide --classe COL (PLS-DA) or --resposta COL (PLS regression), one of the two."),
    "alvo_dois": ("use --classe OU --resposta, nao os dois.", "use --classe OR --resposta, not both."),
    "coluna": ("coluna {papel} {nome!r} nao existe na tabela; colunas: {cols}. Escolha um destes nomes (a comparacao e' exata, "
               "inclusive maiusculas) ou corrija o cabecalho do arquivo.",
               "{papel} column {nome!r} does not exist in the table; columns: {cols}. Pick one of these names (the match is exact, "
               "including case) or fix the header of the file."),
    "nao_numerico": ("coluna {nome!r}, linha {i}: valor nao numerico ou vazio {v!r}",
                     "column {nome!r}, row {i}: non-numeric or empty value {v!r}"),
    "sem_variaveis": ("nenhuma coluna de variavel sobrou depois de tirar classe/resposta/grupo.",
                      "no variable column left after removing class/response/group."),
    "numeros": ("--componentes, --divisoes, --repeticoes devem ser inteiros positivos e --limiar > 0.",
                "--componentes, --divisoes, --repeticoes must be positive integers and --limiar > 0."),
    "recusa": ("recusado pelo calculo: {m}", "refused by the calculation: {m}"),
    "cab": ("VIP com estabilidade ({t}); {p} variaveis, {n} linhas, {g} grupos; {c} componente(s); "
            "{a} ajustes de dobra ({d} dobras x {r} repeticoes)",
            "VIP with stability ({t}); {p} variables, {n} rows, {g} groups; {c} component(s); "
            "{a} fold fits ({d} folds x {r} repeats)"),
    "soma": ("conferencia: soma de VIP^2 = {s:.4f} (esperado: {p} variaveis)",
             "check: sum of VIP^2 = {s:.4f} (expected: {p} variables)"),
    "estavel": ("{n} variavel(is) com VIP > {l:g} em >= 80% das dobras (estavel acima): {v}",
                "{n} variable(s) with VIP > {l:g} in >= 80% of folds (stably above): {v}"),
    "instavel": ("{n} variavel(is) instavel(is) (VIP > {l:g} em 20% a 80% das dobras): {v}",
                 "{n} unstable variable(s) (VIP > {l:g} in 20% to 80% of folds): {v}"),
    "aviso": ("aviso: {a}", "warning: {a}"),
    "arquivos": ("Arquivos gravados:", "Files written:"),
    "rel_titulo": ("Relatorio de VIP com estabilidade", "VIP stability report"),
    "h_tab": ("Variaveis mais importantes", "Most important variables"),
    "c_var": ("variavel", "variable"), "c_vip": ("VIP (todos os dados)", "VIP (all data)"),
    "c_med": ("VIP medio das dobras", "mean fold VIP"), "c_ic": ("intervalo 95%", "95% interval"),
    "c_freq": ("frequencia VIP > limiar", "freq. VIP > threshold"), "c_cls": ("classe", "class"),
    "figura": ("Figura", "Figure"),
    "mais": ("(mostradas as 20 primeiras de {n}; a tabela completa esta no CSV)",
             "(first 20 of {n} shown; the full table is in the CSV)"),
    "cls_estavel_acima": ("estavel acima", "stably above"), "cls_estavel_abaixo": ("estavel abaixo", "stably below"),
    "cls_instavel": ("instavel", "unstable"),
}

# trecho do erro do nucleo (portugues) -> ingles
_ERROS_NUCLEO = (
    ("menos de 2 grupos", "VIP refused: fewer than 2 independent groups; stability is undefined."),
    ("menos de 2 classes", "VIP refused: the class column has fewer than 2 classes."),
    ("constante", "VIP refused: the numeric response is constant."),
    ("NaN", "VIP refused: the data contain NaN/inf; remove or impute them first."),
    ("n_componentes", "VIP refused: number of components out of range for this table."),
    ("nenhum ajuste", "VIP refused: no fold fit converged; reduce the number of components."),
    (">= 4 linhas", "VIP refused: need at least 4 rows."),
)


def _m(chave: str, **kw: Any) -> str:
    par = _MSG[chave]
    t = par[1] if _ESTADO["lang"] == "en" else par[0]
    return t.format(**kw) if kw else t


class _Uso(Exception):
    """Erro de uso (exit 2)."""


def _erro_nucleo(e: Exception) -> _Uso:
    msg = str(e)
    if _ESTADO["lang"] == "en":
        for trecho, en in _ERROS_NUCLEO:
            if trecho in msg:
                return _Uso(en)
        return _Uso(_m("recusa", m=msg))
    return _Uso(msg)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="guaraci vip", add_help=False, exit_on_error=False)
    p.add_argument("tabela", nargs="?")
    p.add_argument("--classe")
    p.add_argument("--resposta")
    p.add_argument("--grupo")
    p.add_argument("--variaveis")
    p.add_argument("--componentes", type=int, default=2)
    p.add_argument("--limiar", type=float, default=1.0)
    p.add_argument("--divisoes", type=int, default=5)
    p.add_argument("--repeticoes", type=int, default=10)
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--saida", default=nome_pasta("resultados_vip"))
    p.add_argument("--sem-figuras", action="store_true")
    p.add_argument("--idioma", default=None)
    p.add_argument("--ajuda", "--help", "-h", action="store_true", dest="ajuda")
    return p


USO = (
    "Uso: guaraci vip TABELA.csv (--classe COL | --resposta COL) --grupo COL\n"
    "                  [--variaveis a,b,c] [--componentes 2] [--limiar 1] [--divisoes 5]\n"
    "                  [--repeticoes 10] [--semente 0] [--saida PASTA] [--sem-figuras] [--idioma pt|en]\n"
    "  --classe     coluna de classe (PLS-DA)           --resposta  coluna numerica (PLS de regressao)\n"
    "  --grupo      id da amostra fisica/mae por linha (replicas com o MESMO id ficam juntas nas dobras)\n"
    "  --variaveis  colunas das variaveis, separadas por virgula (padrao: todas as outras, numericas)\n"
    "  O VIP > 1 e' regra pratica, nao teste de significancia; a estabilidade mostra em quantas dobras\n"
    "  a variavel passa do limiar. Escolher variaveis pelo VIP e validar com os mesmos dados da resultado otimista.\n")
_USO_EN = (
    "Usage: guaraci vip TABLE.csv (--classe COL | --resposta COL) --grupo COL\n"
    "                  [--variaveis a,b,c] [--componentes 2] [--limiar 1] [--divisoes 5]\n"
    "                  [--repeticoes 10] [--semente 0] [--saida FOLDER] [--sem-figuras] [--idioma pt|en]\n"
    "  --classe     class column (PLS-DA)               --resposta  numeric column (PLS regression)\n"
    "  --grupo      physical sample/parent id per row (replicates with the SAME id stay together in folds)\n"
    "  --variaveis  comma-separated variable columns (default: all other columns, numeric)\n"
    "  VIP > 1 is a rule of thumb, not a significance test; stability shows in how many folds the\n"
    "  variable passes the threshold. Choosing variables by VIP and validating on the same data is optimistic.\n")


def _idioma_de(args_l: Sequence[str]) -> str:
    for i, a in enumerate(args_l):
        if a == "--idioma" and i + 1 < len(args_l):
            return "en" if args_l[i + 1].lower().startswith("en") else "pt"
        if a.startswith("--idioma="):
            return "en" if a.split("=", 1)[1].lower().startswith("en") else "pt"
    return "en" if os.environ.get("GUARACI_IDIOMA", "").lower().startswith("en") else "pt"


def _coluna(df, nome: str, papel: str):
    if nome not in df.columns:
        raise _Uso(_m("coluna", papel=papel, nome=nome, cols=", ".join(map(str, df.columns))))
    return df[nome]


def _matriz(df, colunas: List[str]):
    import numpy as np
    X = np.empty((len(df), len(colunas)))
    for j, c in enumerate(colunas):
        for i, v in enumerate(df[c], start=2):
            try:
                X[i - 2, j] = float(str(v).strip().replace(",", "."))
            except ValueError:
                raise _Uso(_m("nao_numerico", nome=c, i=i, v=str(v))) from None
    return X


def _classe_pt(c: str) -> str:
    return _m("cls_" + c)


def _resumo(res: Dict[str, Any], nomes: List[str], args: argparse.Namespace, n: int) -> List[str]:
    L = [_m("cab", t=("PLS-DA" if res["tarefa"] == "classificacao" else "PLS"), p=res["n_variaveis"], n=n,
            g=res["n_grupos"], c=args.componentes, a=res["n_ajustes"], d=res["n_divisoes"], r=res["n_repeticoes"]),
         _m("soma", s=res["soma_vip2"], p=res["n_variaveis"])]
    for cls, chave in (("estavel_acima", "estavel"), ("instavel", "instavel")):
        idx = [i for i in range(len(nomes)) if res["classe"][i] == cls]
        idx.sort(key=lambda i: -res["media"][i])
        if idx:
            L.append(_m(chave, n=len(idx), l=res["limiar"], v=", ".join(nomes[i] for i in idx[:15])
                        + (" ..." if len(idx) > 15 else "")))
    for a in res["avisos"]:
        L.append(_m("aviso", a=a))
    return L


def _blocos(res: Dict[str, Any], nomes: List[str], args: argparse.Namespace, n: int,
            figura: Optional[str]) -> List[tuple]:
    B: List[tuple] = [("h1", _m("rel_titulo"))]
    B += [("p", t) for t in _resumo(res, nomes, args, n)]
    ordem = sorted(range(len(nomes)), key=lambda i: -res["media"][i])
    B.append(("h2", _m("h_tab")))
    B.append(("table", [_m("c_var"), _m("c_vip"), _m("c_med"), _m("c_ic"), _m("c_freq"), _m("c_cls")],
              [[nomes[i], f"{res['vip'][i]:.3f}", f"{res['media'][i]:.3f}",
                f"{res['ic95_inf'][i]:.2f} - {res['ic95_sup'][i]:.2f}", f"{res['frequencia'][i]:.2f}",
                _classe_pt(res["classe"][i])] for i in ordem[:20]]))
    if len(nomes) > 20:
        B.append(("p", _m("mais", n=len(nomes))))
    if figura:
        B.append(("img", "../figuras/" + os.path.basename(figura), _m("figura")))
    B.append(("p", res["texto"]))
    return B


def executar(argv: Optional[Sequence[str]] = None) -> int:
    """Roda `guaraci vip`. Codigo de saida: 0 ok, 2 uso/recusa."""
    import numpy as np
    args_l = list(sys.argv[1:] if argv is None else argv)
    _ESTADO["lang"] = _idioma_de(args_l)
    try:
        try:
            args = _parser().parse_args(args_l)
        except argparse.ArgumentError as e:
            raise _Uso(_m("arg", e=e)) from e
        except SystemExit as e:
            # compatibilidade: no Python 3.10 e 3.11 o argparse ainda chama `parser.error` (e sai com codigo 2) para argumento
            # desconhecido ou valor invalido, mesmo com `exit_on_error=False`; so' o 3.12+ levanta ArgumentError. NAO e' codigo morto.
            raise _Uso(_m("arg", e="--componentes/--divisoes/--repeticoes/--semente/--limiar")) from e
        if args.ajuda:
            print(_USO_EN if _ESTADO["lang"] == "en" else USO)
            return 0
        if not args.tabela:
            raise _Uso(_m("sem_tabela") + "\n\n" + (_USO_EN if _ESTADO["lang"] == "en" else USO))
        if not args.classe and not args.resposta:
            raise _Uso(_m("alvo"))
        if args.classe and args.resposta:
            raise _Uso(_m("alvo_dois"))
        if min(args.componentes, args.divisoes, args.repeticoes) < 1 or not args.limiar > 0:
            raise _Uso(_m("numeros"))
        if not os.path.isfile(args.tabela):
            raise _Uso(_erros.arquivo_nao_encontrado(args.tabela, _ESTADO["lang"], "vip"))
        import pandas as pd
        try:
            df = pd.read_csv(args.tabela, sep=None, engine="python", encoding="utf-8-sig", dtype=str,
                             keep_default_na=False)
        except Exception as e:   # noqa: BLE001 -- qualquer falha de leitura vira mensagem de uso
            raise _Uso(_erros.leitura_csv(args.tabela, e, _ESTADO["lang"], "vip")) from e
        df.columns = [str(c).strip() for c in df.columns]
        alvo_nome = args.classe or args.resposta
        alvo = np.asarray(_coluna(df, alvo_nome, "classe" if args.classe else "resposta").astype(str).str.strip(),
                          dtype=object)
        grupos = (list(np.asarray(_coluna(df, args.grupo, "grupo").astype(str).str.strip(), dtype=object))
                  if args.grupo else None)
        if args.variaveis:
            nomes = [v.strip() for v in args.variaveis.split(",") if v.strip()]
            for v in nomes:
                _coluna(df, v, "variavel" if _ESTADO["lang"] == "pt" else "variable")
        else:
            usadas = {alvo_nome, args.grupo}
            nomes = [c for c in df.columns if c not in usadas]
        if not nomes:
            raise _Uso(_m("sem_variaveis"))
        X = _matriz(df, nomes)
        y: Any = alvo
        if args.resposta:
            y = _matriz(df.assign(**{alvo_nome: alvo}), [alvo_nome]).ravel()
        from guaraci.vip_estabilidade import figura_vip_estabilidade, vip_estabilidade
        try:
            res = vip_estabilidade(X, y, args.componentes, grupos,
                                   tarefa="classificacao" if args.classe else "regressao",
                                   n_divisoes=args.divisoes, n_repeticoes=args.repeticoes,
                                   limiar=args.limiar, seed=args.semente)
        except ValueError as e:
            raise _erro_nucleo(e) from e
        pasta = args.saida
        for sub in ("tabelas", "relatorio"):
            os.makedirs(os.path.join(pasta, sub), exist_ok=True)
        escritos: List[str] = []
        t = os.path.join(pasta_execucao(pasta, "tabelas"), "vip_estabilidade.csv")
        pd.DataFrame({"variavel": nomes, "vip": res["vip"], "vip_medio_dobras": res["media"],
                      "dp": res["dp"], "ic95_inf": res["ic95_inf"], "ic95_sup": res["ic95_sup"],
                      "frequencia_vip_maior_limiar": res["frequencia"], "classe": res["classe"]}
                     ).to_csv(t, index=False, sep=";", decimal=",")
        escritos.append(t)
        figura: Optional[str] = None
        if not args.sem_figuras:
            os.makedirs(pasta_execucao(pasta, "figuras"), exist_ok=True)
            figura = figura_vip_estabilidade(res, os.path.join(pasta_execucao(pasta, "figuras"), "vip_estabilidade.png"), nomes)
            escritos.append(figura)
        linhas = _resumo(res, nomes, args, len(df))
        texto = "\n".join(linhas + ["", res["texto"]])
        r = os.path.join(pasta_execucao(pasta, "relatorio"), "vip_resumo.txt")
        with open(r, "w", encoding="utf-8") as fh:
            fh.write(texto + "\n")
        escritos.append(r)
        blocos = _blocos(res, nomes, args, len(df), figura)
        for ext, f in (("md", renderizar_md), ("html", renderizar_html), ("tex", renderizar_tex)):
            c = os.path.join(pasta_execucao(pasta, "relatorio"), f"vip_relatorio.{ext}")
            with open(c, "w", encoding="utf-8") as fh:
                fh.write(f(blocos))
            escritos.append(c)
        cfg = os.path.join(pasta, "config_efetiva.yaml")
        with open(cfg, "w", encoding="utf-8") as fh:
            fh.write("comando: vip\n")
            for k in ("tabela", "classe", "resposta", "grupo", "componentes", "limiar", "divisoes",
                      "repeticoes", "semente"):
                fh.write(f"{k}: {getattr(args, k)}\n")
            fh.write(f"figuras: {str(not args.sem_figuras).lower()}\n")
            fh.write(f"idioma: {_ESTADO['lang']}\n")
            fh.write("variaveis: [" + ", ".join(nomes[:50]) + (", ..." if len(nomes) > 50 else "") + "]\n")
        escritos.append(cfg)
        print(texto)
        print("\n" + _m("arquivos"))
        for a in escritos:
            print(f"  {a}")
        return 0
    except _Uso as e:
        print(f"Erro: {e}" if _ESTADO["lang"] == "pt" else f"Error: {e}", file=sys.stderr)
        return 2
