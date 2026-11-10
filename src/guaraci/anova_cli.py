"""anova_cli.py -- `guaraci anova`: comparacao de grupos sobre a unidade independente, pela CLI.

    guaraci anova TABELA.csv --resposta COL [--resposta COL2 ...] --grupo COL --unidade COL
                  [--bloco COL] [--metodo auto|tukey|games-howell|kruskal-dunn|recomendar]
                  [--ajuste holm|bh] [--alpha 0.05] [--saida PASTA] [--sem-bootstrap]
                  [--sem-figuras] [--idioma pt|en]

Regra fail-closed: sem `--unidade` (a amostra fisica, o lote ou a sessao de cada
linha) o comando RECUSA. Replicas, espectros e dobras de validacao cruzada nao
sao unidades independentes; tratar replicas como independentes da p-valor
otimista (pseudo-replicacao). O calculo e' o de `guaraci.anova_grupos`.

Padrao: Games-Howell (sem `--bloco`) -- escolhido por simulacao com verdade conhecida
(`docs/qualidade/politicas_anova_simulacao.md`): entre as politicas testadas e' a que menos
passa do limite de erro por familia. Nenhuma politica cumpre o criterio em todos os cenarios;
a falha se concentra em variancia alta no grupo de poucas unidades. Com `--bloco` o padrao e'
`auto` (Games-Howell nao tem versao com bloco aqui; o pos-teste com bloco supoe variancias
iguais e a saida AVISA quando a razao de variancias passa de 4 ou o Brown-Forsythe rejeita).
Tukey so' roda a pedido (`--metodo tukey`) ou quando o `auto` o escolhe pelo Brown-Forsythe
(Levene com centro = mediana) e o resultado diz. A saida descreve o metodo que de fato rodou.

O metodo NUNCA e' trocado em silencio: `auto` mantem a regra visivel do Brown-Forsythe; `recomendar`
mostra o diagnostico de premissas e a razao em uma linha, mas NAO troca o metodo (a
recomendacao nao venceu o padrao na simulacao): roda o padrao e diz o que foi sugerido; os
demais rodam exatamente o que foi pedido e, se o diagnostico sugeria outro, dizem isso.
"""
from __future__ import annotations

import argparse
import os

from guaraci.config import pasta_execucao, nome_pasta
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from guaraci import erros_entrada as _erros
from guaraci.relatorio_blocos import renderizar_html as _html, renderizar_md as _md, renderizar_tex as _tex

__all__ = ["executar", "REGRA_UNIDADE", "METODOS", "letras_de_grupos"]

METODOS = ("auto", "tukey", "games-howell", "kruskal-dunn", "recomendar")
METODO_PADRAO = "games-howell"
METODO_PADRAO_COM_BLOCO = "auto"


def metodo_padrao(com_bloco: bool) -> str:
    """Metodo quando o usuario nao pede nenhum: Games-Howell (simulacao), ou `auto` com bloco."""
    return METODO_PADRAO_COM_BLOCO if com_bloco else METODO_PADRAO


_MAX_FIGURAS = 30
_NOME_METODO = {"tukey": "ANOVA/Tukey", "games-howell": "ANOVA/Games-Howell",
                "kruskal-dunn": "Kruskal-Wallis/Dunn"}

REGRA_UNIDADE = (
    "ANOVA recusada: informe --unidade COL, a coluna com o id da amostra fisica (ou do lote, "
    "ou da sessao) de cada linha. Replicas, espectros e dobras de validacao cruzada nao sao "
    "unidades independentes: tratados como tal, o p-valor sai otimista (pseudo-replicacao). "
    "Linhas da mesma amostra fisica precisam ter o MESMO id.")
_REGRA_UNIDADE_EN = (
    "ANOVA refused: provide --unidade COL, the column holding the id of the physical sample (or "
    "batch, or session) of each row. Replicates, spectra and cross-validation folds are not "
    "independent units: treated as such, the p-value comes out optimistic (pseudo-replication). "
    "Rows from the same physical sample must share the SAME id.")

_ESTADO: Dict[str, str] = {"lang": "pt"}

# chave -> (portugues, ingles)
_MSG: Dict[str, Tuple[str, str]] = {
    "arquivo": ("arquivo nao encontrado: {c}", "file not found: {c}"),
    "leitura": ("nao consegui ler {c} como CSV ({e})", "could not read {c} as CSV ({e})"),
    "coluna": ("coluna {papel} {nome!r} nao existe na tabela; colunas: {cols}. Escolha um destes nomes (a comparacao e' exata, "
               "inclusive maiusculas) ou corrija o cabecalho do arquivo.",
               "{papel} column {nome!r} does not exist in the table; columns: {cols}. Pick one of these names (the match is exact, "
               "including case) or fix the header of the file."),
    "p_grupo": ("de grupo", "group"), "p_unidade": ("de unidade", "unit"),
    "p_bloco": ("de bloco", "block"), "p_resposta": ("de resposta", "response"),
    "nao_numerico": ("coluna {nome!r}, linha {i}: valor nao numerico {v!r}",
                     "column {nome!r}, row {i}: non-numeric value {v!r}"),
    "arg": ("argumento invalido: {e}", "invalid argument: {e}"),
    "sem_tabela": ("informe a tabela CSV.", "provide the CSV table."),
    "sem_grupo": ("informe --grupo COL (a coluna que define os grupos a comparar).",
                  "provide --grupo COL (the column that defines the groups to compare)."),
    "sem_resposta": ("informe ao menos uma --resposta COL (coluna numerica).",
                     "provide at least one --resposta COL (numeric column)."),
    "alpha": ("--alpha deve estar entre 0 e 1.", "--alpha must be between 0 and 1."),
    "metodo": ("--metodo deve ser um de: {lista}.", "--metodo must be one of: {lista}."),
    "ajuste": ("--ajuste deve ser holm ou bh.", "--ajuste must be holm or bh."),
    "bloco_metodo": ("--bloco so' combina com --metodo auto, tukey ou recomendar (o Games-Howell e o "
                     "Kruskal-Wallis nao tem versao com bloco aqui). Tire o --bloco ou troque o metodo.",
                     "--bloco only works with --metodo auto, tukey or recomendar (Games-Howell and "
                     "Kruskal-Wallis have no blocked version here). Drop --bloco or change the method."),
    "nenhuma": ("nenhuma resposta pode ser testada: {det}", "no response could be tested: {det}"),
    "cab": ("ANOVA sobre a unidade independente '{u}' (grupo: {g}); alpha = {a}",
            "Group comparison over the independent unit '{u}' (group: {g}); alpha = {a}"),
    "linhas": ("{n} linha(s) lida(s); {c}", "{n} row(s) read; {c}"),
    "nao_testada": ("{n}: NAO TESTADA -- {m}", "{n}: NOT TESTED -- {m}"),
    "rodape": ("Leia o tamanho de efeito e as comparacoes par a par, nao so o p-valor. Com menos de 6 unidades "
               "por grupo o poder e' baixo.",
               "Read the effect size and the pairwise comparisons, not just the p-value. With fewer than 6 "
               "units per group the power is low."),
    "dif": ("diferenca entre grupos", "difference between groups"),
    "sem_dif": ("sem diferenca detectavel", "no detectable difference"),
    "unidades": ("unidades", "units"),
    # so' quando o metodo rodado foi a Welch ANOVA (o texto descreve o que rodou)
    "var_desig": ("variancias desiguais (Brown-Forsythe, centro = mediana, p = {p:.3g}): Welch + {pos}",
                  "unequal variances (Brown-Forsythe, median-centred, p = {p:.3g}): Welch + {pos}"),
    # ANOVA classica rodou com variancias desiguais (pedido de Tukey, ou o `auto` sem como usar o Welch)
    "var_desig_classica": ("variancias desiguais (Brown-Forsythe, centro = mediana, p = {p:.3g}): rodou {met} + {pos}, "
                           "que supoe variancias iguais{pedido}; considere --metodo games-howell",
                           "unequal variances (Brown-Forsythe, median-centred, p = {p:.3g}): ran {met} + {pos}, "
                           "which assumes equal variances{pedido}; consider --metodo games-howell"),
    "mantido_pedido": (" (mantido a pedido: --metodo tukey)", " (kept as requested: --metodo tukey)"),
    "padrao_gh": ("metodo: padrao do programa (Welch + Games-Howell); para Tukey use --metodo tukey",
                  "method: program default (Welch + Games-Howell); for Tukey pass --metodo tukey"),
    "ic": ("IC", "CI"),
    "aviso": ("aviso: {a}", "warning: {a}"),
    "premissas": ("premissas: razao de variancias {r:.2f}; Brown-Forsythe p = {bf:.3g}; Shapiro p = {sh} "
                  "(informacao, nao porteiro); assimetria {a:.2f}; curtose {k:.2f}",
                  "assumptions: variance ratio {r:.2f}; Brown-Forsythe p = {bf:.3g}; Shapiro p = {sh} "
                  "(information, not a gatekeeper); skewness {a:.2f}; kurtosis {k:.2f}"),
    "recomendado": ("recomendacao: {r}", "recommendation: {r}"),
    "pediu_outro": ("metodo pedido: {pedido}; o diagnostico sugeria {rec} -- mantido o pedido",
                    "requested method: {pedido}; the diagnostics suggested {rec} -- request kept"),
    "usou_rec": ("recomendacao nao aplicada: sugerido {m}; rodado o padrao {p} (peca --metodo {m_flag} para usar a sugestao)",
                 "recommendation not applied: suggested {m}; ran the default {p} (pass --metodo {m_flag} to use the suggestion)"),
    "padrao_bloco": ("com --bloco o padrao e' auto (Games-Howell nao tem versao com bloco aqui)",
                     "with --bloco the default is auto (Games-Howell has no blocked version here)"),
    "bloco_sem_rec": ("com --bloco a recomendacao nao se aplica; usada a ANOVA com bloco",
                      "with --bloco the recommendation does not apply; blocked ANOVA used"),
    "tab_grupos": ("Grupos", "Groups"), "col_grupo": ("grupo", "group"), "col_n": ("n", "n"),
    "col_media": ("media", "mean"), "col_posto": ("posto medio", "mean rank"),
    "col_letra": ("letra", "letter"), "col_dif": ("diferenca", "difference"),
    "col_dif_postos": ("diferenca de postos", "rank difference"),
    "col_p_aj": ("p ajustado", "adjusted p"), "col_sig": ("significativo", "significant"),
    "sim": ("sim", "yes"), "nao": ("nao", "no"),
    "comparacoes": ("Comparacoes par a par", "Pairwise comparisons"),
    "letras_nota": ("Grupos com a mesma letra nao diferem significativamente ao nivel alpha = {a:g} "
                    "(p ajustado >= alpha; isso nao prova igualdade).",
                    "Groups sharing a letter do not differ significantly at level alpha = {a:g} "
                    "(adjusted p >= alpha; this does not prove equality)."),
    "dunn_regra": ("Dunn com postos globais e correcao de empates; ajuste: {r}",
                   "Dunn with global ranks and tie correction; adjustment: {r}"),
    "arquivos": ("Arquivos gravados:", "Files written:"),
    "rel_titulo": ("Relatorio de comparacao de grupos", "Group comparison report"),
    "figura": ("Figura", "Figure"),
    "fig_caixas": ("Valores por unidade, por grupo", "Values per unit, by group"),
    "fig_qq": ("Quantil-quantil dos residuos padronizados (reta y = x)",
               "Standardized-residual quantile-quantile (line y = x)"),
    "fig_teo": ("quantil teorico (normal)", "theoretical quantile (normal)"),
    "fig_res": ("residuo padronizado (residuo / DP)", "standardized residual (residual / SD)"),
    "fig_mais": ("figuras geradas so' para as {n} primeiras respostas",
                 "figures generated only for the first {n} responses"),
    "met": ("Metodo", "Method"), "efeito": ("Tamanho de efeito", "Effect size"),
}

# trecho da mensagem do nucleo (portugues) -> versao em ingles
_ERROS_NUCLEO: Tuple[Tuple[str, str], ...] = (
    ("menos de 2 grupos", "ANOVA refused: fewer than 2 groups with >= 2 independent units. "
                          "Bring more physical samples per group."),
    ("variancia dentro dos grupos e' zero", "ANOVA refused: within-group variance is zero (constant "
                                            "responses per group); the F test is undefined."),
    ("todos os valores sao iguais", "Kruskal-Wallis refused: all values are equal (all tied); the H "
                                    "statistic is undefined."),
    ("aparece em mais de um grupo", "a unit appears in more than one group: the independent unit "
                                    "belongs to a single group. Check the ids."),
    ("aparece em mais de um bloco", "a unit appears in more than one block. Check the ids."),
    ("sem id de unidade", "ANOVA refused: some rows have no unit id. Fill the unit column; replicates "
                          "of one physical sample must share the SAME id."),
    ("sem graus de liberdade", "blocked design without enough residual degrees of freedom (too few "
                               "units per group x block combination)."),
    ("unidade independente nao informada", _REGRA_UNIDADE_EN),
)


def _m(chave: str, **kw: Any) -> str:
    par = _MSG[chave]
    texto = par[1] if _ESTADO["lang"] == "en" else par[0]
    return texto.format(**kw) if kw else texto


class _Uso(Exception):
    """Erro de uso (exit 2): mensagem com a causa e o que fazer, no idioma escolhido."""


def _erro_nucleo(e: Exception) -> "_Uso":
    msg = str(e)
    if _ESTADO["lang"] == "en":
        for trecho, en in _ERROS_NUCLEO:
            if trecho in msg:
                return _Uso(en)
        return _Uso(f"refused by the calculation core (message in Portuguese): {msg}")
    return _Uso(msg)


def _ler_tabela(caminho: str):
    import pandas as pd
    if not os.path.isfile(caminho):
        raise _Uso(_erros.arquivo_nao_encontrado(caminho, _ESTADO["lang"]))
    try:
        df = pd.read_csv(caminho, sep=None, engine="python", encoding="utf-8-sig", dtype=str,
                         keep_default_na=False)
    except Exception as e:   # noqa: BLE001 -- qualquer falha de leitura vira mensagem de uso
        raise _Uso(_erros.leitura_csv(caminho, e, _ESTADO["lang"])) from e
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _coluna(df, nome: str, papel: str):
    if nome not in df.columns:
        raise _Uso(_m("coluna", papel=_m(papel), nome=nome, cols=", ".join(map(str, df.columns))))
    return df[nome]


def _numerica(serie, nome: str):
    import numpy as np
    saida = []
    for i, v in enumerate(serie, start=2):       # linha 1 = cabecalho
        t = str(v).strip().replace(",", ".")
        if t == "":
            saida.append(np.nan)
            continue
        try:
            saida.append(float(t))
        except ValueError:
            raise _Uso(_m("nao_numerico", nome=nome, i=i, v=str(v))) from None
    return np.asarray(saida, dtype=float)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="guaraci anova", add_help=False, exit_on_error=False)
    p.add_argument("tabela", nargs="?")
    p.add_argument("--resposta", action="append", default=[])
    p.add_argument("--grupo")
    p.add_argument("--unidade")
    p.add_argument("--bloco")
    p.add_argument("--metodo", default=None)
    p.add_argument("--ajuste", default="holm")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--saida", default=nome_pasta("resultados_anova"))
    p.add_argument("--sem-bootstrap", action="store_true")
    p.add_argument("--sem-figuras", action="store_true")
    p.add_argument("--idioma", default=None)
    p.add_argument("--ajuda", "--help", "-h", action="store_true", dest="ajuda")
    return p


USO = (
    "Uso: guaraci anova TABELA.csv --resposta COL [--resposta COL2 ...] --grupo COL --unidade COL\n"
    "                  [--bloco COL] [--metodo auto|tukey|games-howell|kruskal-dunn|recomendar]\n"
    "                  [--ajuste holm|bh] [--alpha 0.05] [--saida PASTA] [--sem-bootstrap]\n"
    "                  [--sem-figuras] [--idioma pt|en]\n"
    "  --resposta  coluna numerica a comparar (repita para varias; FDR entre elas)\n"
    "  --grupo     coluna do grupo a comparar (origem, tratamento, produtor)\n"
    "  --unidade   coluna com o id da unidade independente (OBRIGATORIA: amostra fisica, lote ou sessao)\n"
    "  --bloco     coluna de bloco opcional (ex.: sessao): ANOVA de duas vias sem interacao\n"
    "  --metodo    games-howell (padrao; com --bloco o padrao e' auto), auto (Brown-Forsythe decide\n"
    "              Tukey x Games-Howell e o resultado diz qual), tukey (so' a pedido), kruskal-dunn (nao\n"
    "              parametrico) ou recomendar (mostra o diagnostico e a sugestao, mas roda o padrao).\n"
    "              Nunca troca em silencio.\n"
    "  --ajuste    correcao do Dunn: holm (padrao) ou bh\n"
    "  --saida     pasta dos resultados (padrao: resultados_anova)\n"
    "  --sem-figuras  nao gera as figuras (caixas por grupo com letras + quantil-quantil dos residuos)\n")

_USO_EN = (
    "Usage: guaraci anova TABLE.csv --resposta COL [--resposta COL2 ...] --grupo COL --unidade COL\n"
    "                  [--bloco COL] [--metodo auto|tukey|games-howell|kruskal-dunn|recomendar]\n"
    "                  [--ajuste holm|bh] [--alpha 0.05] [--saida FOLDER] [--sem-bootstrap]\n"
    "                  [--sem-figuras] [--idioma pt|en]\n"
    "  --resposta  numeric column to compare (repeat for several; FDR across them)\n"
    "  --grupo     column defining the groups to compare\n"
    "  --unidade   column with the independent unit id (REQUIRED: physical sample, batch or session)\n"
    "  --bloco     optional block column (e.g. session): two-way ANOVA without interaction\n"
    "  --metodo    games-howell (default; with --bloco the default is auto), auto (Brown-Forsythe decides\n"
    "              Tukey vs Games-Howell and the result says which), tukey (on request only), kruskal-dunn\n"
    "              (non-parametric) or recomendar (shows the diagnostics and the suggestion, but runs the\n"
    "              default). Never switches silently.\n"
    "  --ajuste    Dunn correction: holm (default) or bh\n"
    "  --saida     results folder (default: resultados_anova)\n"
    "  --sem-figuras  skip the figures (boxes per group with letters + residual quantile-quantile)\n")


# --------------------------------------------------------------- letras (CLD)

def letras_de_grupos(nomes: Sequence[str], medias: Dict[str, float],
                     comparacoes: Sequence[Dict[str, Any]], alpha: float) -> Dict[str, str]:
    """Letras de grupos homogeneos (insercao-absorcao, Piepho 2004): grupos com
    ao menos uma letra em comum NAO diferem (p ajustado >= alpha). Ordem das
    letras pela media decrescente do grupo."""
    ordem = sorted(nomes, key=lambda n: (-medias[n], n))
    sig = {frozenset((c["a"], c["b"])) for c in comparacoes if c["p_ajustado"] < alpha}
    colunas: List[set] = [set(ordem)]
    for par in sig:
        i, j = tuple(par)
        novas: List[set] = []
        for col in colunas:
            if i in col and j in col:
                novas.append(col - {i})
                novas.append(col - {j})
            else:
                novas.append(col)
        colunas = [c for k, c in enumerate(novas)
                   if c and not any(c < o or (c == o and k > idx) for idx, o in enumerate(novas) if idx != k)]
    colunas.sort(key=lambda c: min(ordem.index(n) for n in c))
    letras = "abcdefghijklmnopqrstuvwxyz"
    return {n: "".join(letras[k % 26] for k, c in enumerate(colunas) if n in c) for n in nomes}


# ----------------------------------------------------------------- calculo

def _fdr(resultados: Dict[str, Any]) -> None:
    from scipy import stats
    if resultados:
        nomes = list(resultados)
        ajust = stats.false_discovery_control([resultados[n].p for n in nomes], method="bh")
        for n, pa in zip(nomes, ajust):
            resultados[n].p_fdr = float(pa)


def _calcular(respostas: Dict[str, Any], grupo: List[Any], unidade: List[Any],
              bloco: Optional[Any], args: argparse.Namespace) -> Dict[str, Any]:
    from guaraci.anova_grupos import (
        anova_uma_via, diagnosticar_premissas, kruskal_dunn, recomendar_metodo)
    resultados: Dict[str, Any] = {}
    recusadas: Dict[str, str] = {}
    diag: Dict[str, Dict[str, Any]] = {}
    rec: Dict[str, Dict[str, str]] = {}
    usado: Dict[str, str] = {}
    notas: Dict[str, List[str]] = {}
    n_boot = 0 if args.sem_bootstrap else 500
    for nome, y in respostas.items():
        notas[nome] = []
        try:
            d = diagnosticar_premissas(y, grupo, unidade, alpha=args.alpha)
            diag[nome], rec[nome] = d, recomendar_metodo(d)
        except ValueError:
            pass                                    # o calculo abaixo recusa com a mensagem certa
        metodo = args.metodo
        if metodo == "recomendar":
            metodo = metodo_padrao(bloco is not None)
            if bloco is not None:
                notas[nome].append(_m("bloco_sem_rec"))
            elif nome in rec and rec[nome]["metodo"] != metodo:
                notas[nome].append(_m("usou_rec", m=_NOME_METODO[rec[nome]["metodo"]], p=_NOME_METODO[metodo],
                                      m_flag=rec[nome]["metodo"]))
        elif nome in rec and args.metodo_pedido is not None and args.metodo != "auto" and rec[nome]["metodo"] != args.metodo:
            notas[nome].append(_m("pediu_outro", pedido=_NOME_METODO[args.metodo],
                                  rec=_NOME_METODO[rec[nome]["metodo"]]))
        if metodo == "games-howell" and args.metodo_pedido in (None, "recomendar"):
            notas[nome].append(_m("padrao_gh"))          # nao foi pedido: e' o padrao, e o texto diz isso
        try:
            r: Any
            if metodo == "kruskal-dunn":
                r = kruskal_dunn(y, grupo, unidade, nome_resposta=nome, ajuste=args.ajuste)
            else:
                r = anova_uma_via(y, grupo, unidade, nome_resposta=nome, bloco=bloco, alpha=args.alpha,
                                  n_boot=n_boot, variancias={"auto": "auto", "tukey": "iguais",
                                                             "games-howell": "desiguais"}[metodo])
        except ValueError as e:
            recusadas[nome] = str(_erro_nucleo(e))
            continue
        resultados[nome] = r
        usado[nome] = metodo
    _fdr(resultados)
    return {"resultados": resultados, "recusadas": recusadas, "diag": diag, "rec": rec,
            "usado": usado, "notas": notas,
            "correcao": ("Tukey/Games-Howell/Dunn dentro de cada resposta; "
                         "Benjamini-Hochberg (FDR) entre respostas") if _ESTADO["lang"] == "pt" else
                        ("Tukey/Games-Howell/Dunn within each response; "
                         "Benjamini-Hochberg (FDR) across responses")}


def _finito(x: Any) -> bool:
    return x is not None and x == x and x not in (float("inf"), float("-inf"))


def _eh_kruskal(r: Any) -> bool:
    return hasattr(r, "H")


def _metodo_texto(r: Any) -> str:
    return "Kruskal-Wallis" if _eh_kruskal(r) else r.metodo


def _pos_texto(r: Any) -> str:
    return f"Dunn ({r.ajuste})" if _eh_kruskal(r) else r.pos_teste


# ----------------------------------------------------------- saidas tabulares

def _escrever_tabelas(pasta: str, out: Dict[str, Any]) -> List[str]:
    import pandas as pd
    os.makedirs(pasta_execucao(pasta, "tabelas"), exist_ok=True)
    nan = float("nan")
    linhas, comps = [], []
    for nome, r in out["resultados"].items():
        kw = _eh_kruskal(r)
        rc = out["rec"].get(nome, {})
        linhas.append({"resposta": nome, "metodo": _metodo_texto(r), "pos_teste": _pos_texto(r),
                       "n_unidades": r.n_unidades, "F": nan if kw else r.F, "p": r.p, "p_fdr": r.p_fdr,
                       "eta2": nan if kw else r.eta2,
                       "eta2_ic95_inf": nan if kw else r.eta2_ic95[0],
                       "eta2_ic95_sup": nan if kw else r.eta2_ic95[1],
                       "omega2": nan if kw else r.omega2, "levene_p": nan if kw else r.levene_p,
                       "avisos": " | ".join(list(r.avisos) + out["notas"].get(nome, [])),
                       "H": r.H if kw else nan, "epsilon2": r.epsilon2 if kw else nan,
                       "metodo_recomendado": rc.get("metodo", ""), "razao_recomendacao": rc.get("razao", "")})
        for c in r.comparacoes:
            comps.append({"resposta": nome, "grupo_a": c["a"], "grupo_b": c["b"],
                          "diferenca": c.get("diferenca", c.get("diferenca_postos")),
                          "ic95_inf": nan if kw else c["ic95"][0], "ic95_sup": nan if kw else c["ic95"][1],
                          "p_ajustado": c["p_ajustado"], "z": c.get("z", nan), "p_bruto": c.get("p_bruto", nan)})
    t1 = os.path.join(pasta_execucao(pasta, "tabelas"), "anova_por_resposta.csv")
    t2 = os.path.join(pasta_execucao(pasta, "tabelas"), "anova_comparacoes.csv")
    pd.DataFrame(linhas).to_csv(t1, index=False, sep=";", decimal=",")
    pd.DataFrame(comps).to_csv(t2, index=False, sep=";", decimal=",")
    return [t1, t2]


def _linha_premissas(d: Dict[str, Any]) -> str:
    sh = "n/a" if d["shapiro_p"] is None else f"{d['shapiro_p']:.3g}"
    return _m("premissas", r=d["razao_variancias"], bf=d["brown_forsythe_p"], sh=sh,
              a=d["assimetria"], k=d["curtose_excesso"])


def texto_resumo(out: Dict[str, Any], args: argparse.Namespace, dados: Dict[str, Any]) -> str:
    from guaraci.anova_grupos import TEXTO_KRUSKAL, TEXTO_PREMISSAS
    L = [_m("cab", u=args.unidade, g=args.grupo, a=args.alpha),
         _m("linhas", n=dados["n_linhas"], c=out["correcao"]), ""]
    usou_kw = False
    for nome, r in out["resultados"].items():
        sig = _m("dif") if (r.p_fdr is not None and r.p_fdr < args.alpha) else _m("sem_dif")
        if _eh_kruskal(r):
            usou_kw = True
            L.append(f"- {nome}: {_metodo_texto(r)}, H = {r.H:.3f}, gl = {r.gl}, p = {r.p:.4g}, "
                     f"p(FDR) = {r.p_fdr:.4g}, epsilon2 = {r.epsilon2:.3f}, {r.n_unidades} {_m('unidades')} -> {sig}")
            for c in r.comparacoes:
                if c["p_ajustado"] < args.alpha:
                    L.append(f"    {c['a']} x {c['b']}: z = {c['z']:.3f}; p ajustado {c['p_ajustado']:.3g} [{_pos_texto(r)}]")
        else:
            # IC de eta2: bootstrap percentil 2,5-97,5 (sempre 95%, independe de alpha); omitido sem bootstrap
            ic_eta = ("" if args.sem_bootstrap or not all(map(_finito, r.eta2_ic95)) else
                      f" ({_m('ic')}95 bootstrap {r.eta2_ic95[0]:.2f} a {r.eta2_ic95[1]:.2f})")
            L.append(f"- {nome}: {r.metodo}, F = {r.F:.3f}, p = {r.p:.4g}, p(FDR) = {r.p_fdr:.4g}, "
                     f"eta2 = {r.eta2:.3f}{ic_eta}, "
                     f"{r.n_unidades} {_m('unidades')} -> {sig}")
            if r.levene_p is not None and r.levene_p < args.alpha:
                if r.metodo == "Welch ANOVA":
                    L.append("    " + _m("var_desig", p=r.levene_p, pos=r.pos_teste))
                elif r.metodo == "ANOVA classica":
                    pedido = _m("mantido_pedido") if args.metodo_pedido == "tukey" else ""
                    L.append("    " + _m("var_desig_classica", p=r.levene_p, met=r.metodo, pos=r.pos_teste, pedido=pedido))
                # com bloco: o aviso vem do nucleo (o pos-teste com bloco supoe variancias iguais)
            # IC simultaneo do pos-teste: nivel 1 - alpha (o mesmo do p ajustado)
            rot_ic = f"{_m('ic')}{100 * (1 - args.alpha):g}"
            for c in r.comparacoes:
                if c["p_ajustado"] < args.alpha:
                    L.append(f"    {c['a']} x {c['b']}: diferenca {c['diferenca']:.4g} "
                             f"({rot_ic} {c['ic95'][0]:.4g} a {c['ic95'][1]:.4g}; p ajustado {c['p_ajustado']:.3g}) [{r.pos_teste}]")
        if nome in out["diag"]:
            L.append("    " + _linha_premissas(out["diag"][nome]))
            L.append("    " + _m("recomendado", r=out["rec"][nome]["razao"]))
        for a in out["notas"].get(nome, []):
            L.append(f"    {a}")
        for a in r.avisos:
            L.append("    " + _m("aviso", a=a))
    for nome, motivo in out["recusadas"].items():
        L.append("- " + _m("nao_testada", n=nome, m=motivo))
    L.append("")
    if usou_kw:
        L.append(TEXTO_KRUSKAL)
    L.append(TEXTO_PREMISSAS)
    L.append(_m("rodape"))
    return "\n".join(L)


# ------------------------------------------------------------------ figuras

def _slug(nome: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", nome).strip("_") or "resposta"


def _qq_padronizado(d: Dict[str, Any]):
    """Pontos do Q-Q com os residuos PADRONIZADOS (divididos pelo DP amostral,
    ddof=1): so' assim a reta y = x contra os quantis da N(0, 1) e' a
    referencia certa. So' a figura usa isto; os numeros de `d` nao mudam."""
    import numpy as np
    teo = np.asarray(d["qq_teorico"], dtype=float)
    amo = np.asarray(d["qq_amostral"], dtype=float)
    dp = float(np.std(amo, ddof=1)) if amo.size > 1 else 0.0
    if dp > 0 and np.isfinite(dp):
        amo = (amo - float(np.mean(amo))) / dp
    return teo, amo


def _figura(nome: str, y_u, g_u, r: Any, d: Optional[Dict[str, Any]], caminho: str, alpha: float) -> str:
    """Caixas por grupo (pontos = unidades) com letras de grupos homogeneos +
    quantil-quantil dos residuos. Figure/Agg direto: sem estado global do pyplot."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9.0, 3.8))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 2, 1)
    dados = [y_u[g_u == n] for n in r.grupos]
    ax.boxplot(dados, showfliers=False)
    rng = np.random.default_rng(0)
    for i, v in enumerate(dados, start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color="#4C72B0", alpha=0.7, zorder=3)
    ax.set_xticks(range(1, len(dados) + 1))
    ax.set_xticklabels(r.grupos, rotation=30, ha="right", fontsize=8)
    medias = ({n: float(np.mean(y_u[g_u == n])) for n in r.grupos})
    letras = letras_de_grupos(r.grupos, medias, r.comparacoes, alpha) if r.comparacoes else {}
    topo = max(float(np.max(v)) for v in dados)
    base = min(float(np.min(v)) for v in dados)
    folga = (topo - base) * 0.06      # amplitude > 0: dados constantes sao recusados antes
    for i, n in enumerate(r.grupos, start=1):
        if letras:
            ax.text(i, float(np.max(dados[i - 1])) + folga, letras[n], ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{nome}  ({_metodo_texto(r)})", fontsize=9)
    ax.set_ylabel(_m("fig_caixas"), fontsize=8)
    if d is not None:
        bx = fig.add_subplot(1, 2, 2)
        teo, amo = _qq_padronizado(d)
        bx.scatter(teo, amo, s=12, color="#4C72B0")
        lim = [min(teo.min(), amo.min()), max(teo.max(), amo.max())]
        bx.plot(lim, lim, color="#999999", lw=1)
        bx.set_title(_m("fig_qq"), fontsize=9)
        bx.set_xlabel(_m("fig_teo"), fontsize=8)
        bx.set_ylabel(_m("fig_res"), fontsize=8)
    from guaraci.idioma_figuras import traduzir_figura   # texto da figura no idioma da sessao
    traduzir_figura(fig)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    return caminho


def _escrever_figuras(pasta: str, out: Dict[str, Any], dados_res: Dict[str, Any], grupo, unidade,
                      args: argparse.Namespace) -> Dict[str, str]:
    import numpy as np
    from guaraci.anova_grupos import agregar_por_unidade
    os.makedirs(pasta_execucao(pasta, "figuras"), exist_ok=True)
    feitas: Dict[str, str] = {}
    for nome, r in list(out["resultados"].items())[:_MAX_FIGURAS]:
        y_u, g_u, _, _ = agregar_por_unidade(dados_res[nome], grupo, unidade)
        mant = np.asarray([str(x) in r.grupos for x in g_u])
        caminho = os.path.join(pasta_execucao(pasta, "figuras"), f"anova_{_slug(nome)}.png")
        feitas[nome] = _figura(nome, y_u[mant], g_u[mant].astype(str), r, out["diag"].get(nome), caminho, args.alpha)
    return feitas


# ---------------------------------------------------------------- relatorios

def _blocos_relatorio(out: Dict[str, Any], args: argparse.Namespace, dados: Dict[str, Any],
                      figuras: Dict[str, str]) -> List[tuple]:
    from guaraci.anova_grupos import TEXTO_KRUSKAL, TEXTO_PREMISSAS
    B: List[tuple] = [("h1", _m("rel_titulo")),
                      ("p", _m("cab", u=args.unidade, g=args.grupo, a=args.alpha)),
                      ("p", _m("linhas", n=dados["n_linhas"], c=out["correcao"]))]
    usou_kw = False
    for nome, r in out["resultados"].items():
        kw = _eh_kruskal(r)
        usou_kw = usou_kw or kw
        B.append(("h2", nome))
        if kw:
            B.append(("p", f"{_m('met')}: {_metodo_texto(r)} + {_pos_texto(r)}; H = {r.H:.3f}, gl = {r.gl}, "
                           f"p = {r.p:.4g}, p(FDR) = {r.p_fdr:.4g}; {_m('efeito')}: epsilon2 = {r.epsilon2:.3f} "
                           f"(epsilon2 = H/(n-1))."))
            B.append(("p", _m("dunn_regra", r=r.regra_ajuste)))
            medias = r.posto_medio_por_grupo
        else:
            B.append(("p", f"{_m('met')}: {r.metodo} + {r.pos_teste}; F = {r.F:.3f}, p = {r.p:.4g}, "
                           f"p(FDR) = {r.p_fdr:.4g}; {_m('efeito')}: eta2 = {r.eta2:.3f}, omega2 = {r.omega2:.3f}."))
            medias = r.media_por_grupo
        letras = (letras_de_grupos(r.grupos, medias, r.comparacoes, args.alpha)
                  if r.comparacoes else {})
        cab = [_m("col_grupo"), _m("col_n"), _m("col_posto") if kw else _m("col_media")]
        if letras:
            cab.append(_m("col_letra"))
        linhas = [[n, str(r.n_por_grupo[n]), f"{medias[n]:.4g}"] + ([letras[n]] if letras else [])
                  for n in r.grupos]
        B.append(("table", cab, linhas))
        if letras:
            B.append(("p", _m("letras_nota", a=args.alpha)))
        if r.comparacoes:
            B.append(("h3", _m("comparacoes")))
            cab2 = [_m("col_grupo") + " A", _m("col_grupo") + " B",
                    _m("col_dif_postos") if kw else _m("col_dif"), _m("col_p_aj"), _m("col_sig")]
            lin2 = [[c["a"], c["b"], f"{c.get('diferenca', c.get('diferenca_postos')):.4g}",
                     f"{c['p_ajustado']:.3g}", _m("sim") if c["p_ajustado"] < args.alpha else _m("nao")]
                    for c in r.comparacoes]
            B.append(("table", cab2, lin2))
        if nome in out["diag"]:
            B.append(("p", _linha_premissas(out["diag"][nome])))
            B.append(("p", _m("recomendado", r=out["rec"][nome]["razao"])))
        for a in out["notas"].get(nome, []) + [_m("aviso", a=x) for x in r.avisos]:
            B.append(("p", a))
        if nome in figuras:
            B.append(("img", "../figuras/" + os.path.basename(figuras[nome]), f"{_m('figura')}: {nome}"))
    for nome, motivo in out["recusadas"].items():
        B.append(("p", _m("nao_testada", n=nome, m=motivo)))
    if usou_kw:
        B.append(("p", TEXTO_KRUSKAL))
    B.append(("p", TEXTO_PREMISSAS))
    B.append(("p", _m("rodape")))
    return B


def _escrever_resultados(pasta: str, out: Dict[str, Any], args: argparse.Namespace, dados: Dict[str, Any],
                         respostas: Dict[str, Any], grupo: Sequence[Any], unidade: Sequence[Any]) -> List[str]:
    os.makedirs(pasta_execucao(pasta, "relatorio"), exist_ok=True)
    escritos = _escrever_tabelas(pasta, out)
    figuras: Dict[str, str] = {}
    if not args.sem_figuras:
        figuras = _escrever_figuras(pasta, out, respostas, grupo, unidade, args)
        escritos += list(figuras.values())
        if len(out["resultados"]) > _MAX_FIGURAS:
            print(_m("fig_mais", n=_MAX_FIGURAS))
    rel = os.path.join(pasta_execucao(pasta, "relatorio"), "anova_resumo.txt")
    with open(rel, "w", encoding="utf-8") as fh:
        fh.write(texto_resumo(out, args, dados) + "\n")
    escritos.append(rel)
    blocos = _blocos_relatorio(out, args, dados, figuras)
    for ext, render in (("md", _md), ("html", _html), ("tex", _tex)):
        caminho = os.path.join(pasta_execucao(pasta, "relatorio"), f"anova_relatorio.{ext}")
        with open(caminho, "w", encoding="utf-8") as fh:
            fh.write(render(blocos))
        escritos.append(caminho)
    cfg = os.path.join(pasta, "config_efetiva.yaml")
    with open(cfg, "w", encoding="utf-8") as fh:
        fh.write("comando: anova\n")
        for k in ("tabela", "grupo", "unidade", "bloco", "alpha", "metodo", "ajuste"):
            fh.write(f"{k}: {getattr(args, k)}\n")
        fh.write(f"metodo_pedido_pelo_usuario: {'sim' if args.metodo_pedido is not None else 'nao (padrao)'}\n")
        fh.write(f"figuras: {str(not args.sem_figuras).lower()}\n")
        fh.write(f"idioma: {_ESTADO['lang']}\n")
        fh.write("respostas: [" + ", ".join(args.resposta) + "]\n")
        fh.write("metodo_efetivo_por_resposta:\n")
        for nome, m in out["usado"].items():
            fh.write(f"  {nome}: {m}\n")
        fh.write("troca_automatica_de_metodo: nenhuma (o padrao e o que foi sugerido ficam declarados na saida)\n")
    escritos.append(cfg)
    return escritos


def _idioma_de(args_l: Sequence[str]) -> str:
    for i, a in enumerate(args_l):
        if a == "--idioma" and i + 1 < len(args_l):
            return "en" if args_l[i + 1].lower().startswith("en") else "pt"
        if a.startswith("--idioma="):
            return "en" if a.split("=", 1)[1].lower().startswith("en") else "pt"
    from guaraci.idioma_relatorio import idioma as _idioma_padrao   # GUARACI_IDIOMA, escolha do menu, ingles
    return _idioma_padrao()


def executar(argv: Optional[Sequence[str]] = None) -> int:
    """Roda `guaraci anova`. Devolve o codigo de saida (0 ok, 2 uso/recusa)."""
    import numpy as np
    args_l = list(sys.argv[1:] if argv is None else argv)
    _ESTADO["lang"] = _idioma_de(args_l)
    try:
        try:
            args = _parser().parse_args(args_l)
        except argparse.ArgumentError as e:
            raise _Uso(_m("arg", e=e)) from e
        if args.ajuda:
            print(_USO_EN if _ESTADO["lang"] == "en" else USO)
            return 0
        if not args.tabela:
            raise _Uso(_m("sem_tabela") + "\n\n" + (_USO_EN if _ESTADO["lang"] == "en" else USO))
        if not args.grupo:
            raise _Uso(_m("sem_grupo"))
        if not args.unidade:
            raise _Uso(_REGRA_UNIDADE_EN if _ESTADO["lang"] == "en" else REGRA_UNIDADE)
        if not args.resposta:
            raise _Uso(_m("sem_resposta"))
        if not 0.0 < args.alpha < 1.0:
            raise _Uso(_m("alpha"))
        args.metodo_pedido = args.metodo
        if args.metodo is None:
            args.metodo = metodo_padrao(bool(args.bloco))
        if args.metodo not in METODOS:
            raise _Uso(_m("metodo", lista=", ".join(METODOS)))
        if args.ajuste not in ("holm", "bh"):
            raise _Uso(_m("ajuste"))
        if args.bloco and args.metodo in ("games-howell", "kruskal-dunn"):
            raise _Uso(_m("bloco_metodo"))
        df = _ler_tabela(args.tabela)
        grupo = np.asarray(_coluna(df, args.grupo, "p_grupo").astype(str).str.strip(), dtype=object)
        unidade = np.asarray(_coluna(df, args.unidade, "p_unidade").astype(str).str.strip(), dtype=object)
        bloco = (np.asarray(_coluna(df, args.bloco, "p_bloco").astype(str).str.strip(), dtype=object)
                 if args.bloco else None)
        respostas = {c: _numerica(_coluna(df, c, "p_resposta"), c) for c in args.resposta}
        out = _calcular(respostas, list(grupo), list(unidade), bloco, args)
        if not out["resultados"]:
            raise _Uso(_m("nenhuma", det="; ".join(f"{k}: {v}" for k, v in out["recusadas"].items())))
        dados = {"n_linhas": len(df)}
        escritos = _escrever_resultados(args.saida, out, args, dados, respostas, list(grupo), list(unidade))
        print(texto_resumo(out, args, dados))
        print("\n" + _m("arquivos"))
        for arquivo in escritos:
            print(f"  {arquivo}")
        return 0
    except _Uso as e:
        print(f"Erro: {e}" if _ESTADO["lang"] == "pt" else f"Error: {e}", file=sys.stderr)
        return 2
