"""monitorar_cli.py -- `guaraci monitorar`: monitorar lotes de um processo (MSPC) pela CLI.

    guaraci monitorar --referencia REF.csv --novos NOVOS.csv --lote COL [--id COL]
                      [--pcs auto|N] [--alpha 0.05] [--min-lotes 3] [--autoescala]
                      [--mais-graficos] [--saida PASTA]

Calibra o dominio do processo (PCA + distancia combinada T2/Q) nos lotes de
referencia, julga cada amostra nova e testa, lote a lote, se a taxa de rejeicao
subiu em relacao a taxa que a propria calibracao rejeita em validacao cruzada
(teste R2 de `sentinela_deriva`). Reaproveita `chemometric_stats` e
`sentinela_deriva`; nao reimplementa o dominio nem o teste.
"""
from __future__ import annotations

import argparse
import os

from guaraci.config import NOME_GRAFICOS, pasta_execucao, nome_pasta
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from guaraci import erros_entrada as _erros

__all__ = ["executar", "escolher_pcs", "conferir_referencia", "MIN_LOTES_PADRAO"]

MIN_LOTES_PADRAO = 3
VAR_EXPLICADA_AUTO = 0.90
MAX_PCS_AUTO = 10

USO = (
    "Uso: guaraci monitorar --referencia REF.csv --novos NOVOS.csv --lote COL [--id COL]\n"
    "                       [--pcs auto|N] [--alpha 0.05] [--min-lotes 3] [--autoescala]\n"
    "                       [--mais-graficos] [--saida PASTA]\n"
    "  --referencia  CSV com lotes produzidos em controle (uma linha por amostra, colunas numericas = variaveis)\n"
    "  --novos       CSV com os lotes a monitorar (mesmas colunas numericas da referencia)\n"
    "  --lote        coluna com o id do lote ou da sessao (recomendada: sem ela, os novos viram um lote so')\n"
    "  --id          coluna com o id da amostra (opcional, so' para as tabelas)\n"
    "  --pcs         componentes do PCA: auto (90% da variancia, no maximo 10) ou um inteiro\n"
    "  --alpha       nivel do dominio e do teste de deriva (padrao 0.05)\n"
    "  --min-lotes   minimo de lotes de referencia (padrao 3); abaixo disso o comando recusa\n"
    "  --saida       pasta dos resultados (padrao: resultados_monitoramento)\n")


class _Uso(Exception):
    """Erro de uso ou recusa (exit 2): mensagem com a causa e o que fazer."""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="guaraci monitorar", add_help=False, exit_on_error=False)
    p.add_argument("--referencia")
    p.add_argument("--novos")
    p.add_argument("--lote")
    p.add_argument("--id", dest="coluna_id")
    p.add_argument("--pcs", default="auto")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-lotes", type=int, default=MIN_LOTES_PADRAO, dest="min_lotes")
    p.add_argument("--autoescala", action="store_true")
    p.add_argument("--mais-graficos", action="store_true", dest="mais_graficos")
    p.add_argument("--saida", default=nome_pasta("resultados_monitoramento"))
    p.add_argument("--idioma", default=None)
    p.add_argument("--ajuda", "--help", "-h", action="store_true", dest="ajuda")
    return p


_LANG = {"v": "pt"}


def _ler(caminho: str, meta: Sequence[str]):
    """Le o CSV e separa metadados (colunas pedidas) das variaveis numericas."""
    import numpy as np
    import pandas as pd
    if not caminho or not os.path.isfile(caminho):
        raise _Uso(_erros.arquivo_nao_encontrado(caminho, _LANG["v"], "monitorar"))
    try:
        df = pd.read_csv(caminho, sep=None, engine="python", encoding="utf-8-sig")
        if len(df) == 0:
            raise ValueError("Could not determine delimiter")        # so' cabecalho ou nada: tratado como arquivo vazio
    except Exception as e:   # noqa: BLE001 -- qualquer falha de leitura vira mensagem de uso
        raise _Uso(_erros.leitura_csv(caminho, e, _LANG["v"], "monitorar")) from e
    df.columns = [str(c).strip() for c in df.columns]
    for c in meta:
        if c not in df.columns:
            raise _Uso(_erros.coluna_ausente("de lote/id" if _LANG["v"] == "pt" else "batch/id", c, list(df.columns), _LANG["v"],
                                             os.path.basename(caminho)))
    variaveis = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    if len(variaveis) < 2:
        raise _Uso(f"{os.path.basename(caminho)}: menos de 2 colunas numericas (variaveis do processo).")
    X = df[variaveis].to_numpy(dtype=float)
    if not np.all(np.isfinite(X)):
        raise _Uso(f"{os.path.basename(caminho)}: ha celulas vazias ou nao numericas nas variaveis; "
                   "o monitoramento exige a tabela completa.")
    return df, variaveis, X


def conferir_referencia(lotes_ref: Optional[Sequence[Any]], n_ref: int, min_lotes: int) -> Optional[str]:
    """Regra de contexto: devolve a mensagem de recusa, ou None se a referencia serve.

    Sem coluna de lote nao ha como contar lotes: nao recusa (vira sugestao no relatorio).
    Com coluna de lote, menos de `min_lotes` lotes distintos recusa: o dominio calibrado
    num lote so' confunde a variacao normal entre lotes com deriva.
    """
    if n_ref < 10:
        return (f"a referencia tem {n_ref} amostra(s); sao precisas pelo menos 10 para calibrar o "
                "dominio. Junte mais amostras produzidas em controle.")
    if lotes_ref is None:
        return None
    n_lotes = len(set(map(str, lotes_ref)))
    if n_lotes < min_lotes:
        return (f"a referencia tem {n_lotes} lote(s) distinto(s); o minimo e' {min_lotes}. Com menos lotes, "
                "a variacao normal entre lotes vira alarme falso. Junte lotes produzidos em controle de "
                "dias ou sessoes diferentes (ou reduza --min-lotes sabendo desse risco).")
    return None


def escolher_pcs(var_explicada: Sequence[float], pedido: str, n_ref: int, n_var: int) -> int:
    """Numero de componentes: inteiro pedido (validado) ou o menor k com 90% da variancia."""
    import numpy as np
    teto = max(1, min(n_ref - 2, n_var, MAX_PCS_AUTO if pedido == "auto" else n_var))
    if pedido != "auto":
        try:
            k = int(pedido)
        except ValueError:
            raise _Uso(f"--pcs deve ser 'auto' ou um inteiro, nao {pedido!r}.") from None
        if not 1 <= k <= min(n_ref - 2, n_var):
            raise _Uso(f"--pcs {k} fora do intervalo 1..{min(n_ref - 2, n_var)} para esta referencia.")
        return k
    acum = np.cumsum(np.asarray(var_explicada, dtype=float))
    k = int(np.searchsorted(acum, VAR_EXPLICADA_AUTO) + 1)
    return int(min(max(k, 1), teto))


def _calibrar(X_ref, lotes_ref, pcs: str, alpha: float, autoescala: bool):
    import numpy as np
    from sklearn.decomposition import PCA
    from guaraci.chemometric_stats import ad_rejection_rate_cv, training_applicability_domain
    escala = X_ref.std(axis=0, ddof=1) if autoescala else np.ones(X_ref.shape[1])
    escala[escala == 0] = 1.0
    Z = X_ref / escala
    k = escolher_pcs(PCA().fit(Z).explained_variance_ratio_, pcs, Z.shape[0], Z.shape[1])
    pca = PCA(n_components=k).fit(Z)
    grupos = None if lotes_ref is None else np.asarray(list(map(str, lotes_ref)))
    dominio = training_applicability_domain(pca, Z, alpha=alpha, groups=grupos)
    ref_r, ref_n = ad_rejection_rate_cv(Z, k, alpha=alpha, groups=grupos)
    return pca, escala, dominio, (ref_r, ref_n), k


def _julgar(pca, escala, dominio, X_novos):
    from guaraci.chemometric_stats import applicability_domain_new_samples
    d = dominio
    return applicability_domain_new_samples(pca, X_novos / escala, d["var_t"], d["h0"], d["q0"],
                                            d["Nh"], d["Nq"], d["f_crit"])


def _por_lote(lotes: List[str], dentro, ref: Tuple[int, int], alpha: float) -> List[Dict[str, Any]]:
    from guaraci.sentinela_deriva import SentinelState, check_drift
    ordem: List[str] = []
    for lt in lotes:
        if lt not in ordem:
            ordem.append(lt)
    linhas = []
    for lt in ordem:
        est = SentinelState(alpha_nominal=alpha)
        est.ref_rejeitadas, est.ref_n = ref
        for lote_i, d in zip(lotes, dentro):
            if lote_i == lt:
                est.registrar(bool(d))
        al = check_drift(est, significancia=alpha)
        linhas.append({"lote": lt, "n": al.n, "n_fora": est.n_fora_do_dominio,
                       "taxa_fora": al.taxa_rejeicao_observada, "taxa_referencia": al.taxa_referencia,
                       "p": al.p_valor, "alarme": bool(al.alerta),
                       "testado": bool(al.p_valor == al.p_valor), "mensagem": al.mensagem})
    return linhas


def _figuras(pasta: str, res: Dict[str, Any], lotes: List[str], mais: bool, Z_novos, pca,
             alpha: Optional[float] = None) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from guaraci.config import Config
    from guaraci.design_tokens import tokens
    from guaraci.figuras import save, setup_matplotlib
    cfg = Config()
    setup_matplotlib(cfg)
    tk = tokens("light")
    base = pasta_execucao(pasta, NOME_GRAFICOS)
    feitos: List[str] = []
    idx = np.arange(len(lotes))
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 5.4), sharex=True)
    txt_alpha = f"alpha = {alpha:g}" if alpha is not None else "alpha"
    for ax, chave, rotulo in ((axs[0], "t2", "T2 de Hotelling"), (axs[1], "q", "Q (residuo)")):
        v = np.asarray(res[chave], dtype=float)
        lim = res.get("t2_limite" if chave == "t2" else "q_limite")
        # Marca POR ESTATISTICA: no painel T2 so' T2 > limite de T2; no Q so'
        # Q > limite de Q (antes usava o status combinado, e pontos abaixo do
        # limite apareciam como "fora"). A decisao final de dominio da tabela
        # continua sendo a distancia combinada.
        acima = (v > float(lim)) if lim is not None and np.isfinite(lim) else np.zeros(v.size, bool)
        nome_est = "T2" if chave == "t2" else "Q"
        ax.plot(idx, v, color=tk["text_muted"], lw=0.8, zorder=1)
        ax.scatter(idx[~acima], v[~acima], s=14, color=tk["success"],
                   label=f"{nome_est} <= limite", zorder=2)
        ax.scatter(idx[acima], v[acima], s=22, color=tk["error"], marker="x",
                   label=f"{nome_est} > limite", zorder=3)
        if lim is not None and np.isfinite(lim):
            ax.axhline(float(lim), color=tk["warn"], ls="--", lw=1.0,
                       label=f"limite de {nome_est} ({txt_alpha})")
        ax.set_ylabel(rotulo)
        ax.legend(loc="upper left", fontsize=7)
    for i in range(1, len(lotes)):
        if lotes[i] != lotes[i - 1]:
            for ax in axs:
                ax.axvline(i - 0.5, color=tk["border"], lw=0.8)
    axs[0].set_title("Carta T2 e Q por amostra (linhas verticais separam lotes)\n"
                     "decisao de dominio da tabela: distancia combinada T2+Q", fontsize=9)
    if len(set(lotes)) > 1:
        # nome de cada lote no centro do seu trecho contiguo
        ini = [0] + [i for i in range(1, len(lotes)) if lotes[i] != lotes[i - 1]]
        fim = ini[1:] + [len(lotes)]
        axs[1].set_xticks([(a + b - 1) / 2 for a, b in zip(ini, fim)])
        axs[1].set_xticklabels([lotes[a] for a in ini], rotation=45, ha="right", fontsize=7)
        axs[1].set_xlabel("lote (amostras na ordem do arquivo)")
    else:
        axs[1].set_xlabel("amostra (ordem do arquivo)")
    save(fig, "carta_t2_q", pasta, cfg)
    feitos.append(os.path.join(base, f"carta_t2_q.{cfg.output_format}"))
    if mais:
        linhas = res["por_lote"]
        fig, ax = plt.subplots(figsize=(7.0, 3.4))
        xs = np.arange(len(linhas))
        ax.bar(xs, [r["taxa_fora"] for r in linhas],
               color=[tk["error"] if r["alarme"] else tk["success"] for r in linhas])
        ref = linhas[0]["taxa_referencia"] if linhas else None
        if ref is not None:
            ax.axhline(ref, color=tk["warn"], ls="--", lw=1.0, label="taxa da calibracao (CV)")
            ax.legend(fontsize=7)
        ax.set_xticks(xs, [r["lote"] for r in linhas], rotation=45, ha="right")
        ax.set_ylabel("fracao fora do dominio")
        ax.set_title("Tendencia por lote (vermelho = alarme)")
        save(fig, "tendencia_por_lote", pasta, cfg)
        feitos.append(os.path.join(base, f"tendencia_por_lote.{cfg.output_format}"))
        i_max = int(np.argmax(res["q"]))
        rec = pca.inverse_transform(pca.transform(Z_novos[i_max:i_max + 1]))[0]
        contrib = (Z_novos[i_max] - rec) ** 2
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        ax.bar(np.arange(contrib.size), contrib, color=tk["primary"])
        ax.set_xlabel("variavel (ordem das colunas)")
        ax.set_ylabel("contribuicao ao Q")
        ax.set_title(f"Contribuicao ao Q da amostra com maior Q (linha {i_max + 1}, lote {lotes[i_max]})")
        save(fig, "contribuicao_q", pasta, cfg)
        feitos.append(os.path.join(base, f"contribuicao_q.{cfg.output_format}"))
    return feitos


def texto_resumo(info: Dict[str, Any]) -> str:
    L = [f"Monitoramento: {info['n_ref']} amostra(s) de referencia em {info['n_lotes_ref']} lote(s); "
         f"{info['k']} componente(s) do PCA; alpha = {info['alpha']}",
         f"Taxa que a calibracao rejeita em validacao cruzada: {info['ref'][0]}/{info['ref'][1]}", ""]
    for r in info["por_lote"]:
        if not r["testado"]:
            estado = f"sem poder para testar ({r['n']} amostra(s))"
        else:
            estado = "ALARME" if r["alarme"] else "em controle"
        L.append(f"- lote {r['lote']}: {r['n_fora']}/{r['n']} fora do dominio -> {estado}"
                 + (f" (p = {r['p']:.3g})" if r["testado"] else ""))
    L += [""] + [f"Sugestao: {s}" for s in info["sugestoes"]]
    L.append("Alarme diz que o processo mudou em relacao a referencia; a causa (instrumento, materia-prima, "
             "operacao) precisa ser investigada no processo.")
    return "\n".join(L)


def executar(argv: Optional[Sequence[str]] = None) -> int:
    """Roda `guaraci monitorar`. Devolve o codigo de saida (0 ok, 2 uso/recusa)."""
    import numpy as np
    import pandas as pd
    args_l = list(sys.argv[1:] if argv is None else argv)
    _LANG["v"] = _erros.idioma(args_l)
    try:
        try:
            args = _parser().parse_args(args_l)
        except argparse.ArgumentError as e:
            raise _Uso(f"argumento invalido: {e}") from e
        if args.ajuda:
            print(USO)
            return 0
        if not args.referencia or not args.novos:
            raise _Uso("informe --referencia e --novos.\n\n" + USO)
        if not 0.0 < args.alpha < 0.5:
            raise _Uso("--alpha deve estar entre 0 e 0.5.")
        meta = [c for c in (args.lote, args.coluna_id) if c]
        df_ref, var_ref, X_ref = _ler(args.referencia, meta)
        df_nov, var_nov, X_nov = _ler(args.novos, meta)
        if var_ref != var_nov:
            faltam = sorted(set(var_ref) ^ set(var_nov))[:8]
            raise _Uso(f"as colunas numericas de referencia e novos diferem (ex.: {', '.join(faltam)}); "
                       "use as mesmas variaveis, na mesma ordem.")
        lotes_ref = list(df_ref[args.lote].astype(str)) if args.lote else None
        recusa = conferir_referencia(lotes_ref, len(df_ref), args.min_lotes)
        if recusa:
            raise _Uso("monitoramento recusado: " + recusa)
        sugestoes: List[str] = []
        if not args.lote:
            sugestoes.append("informe --lote COL: sem lote, os novos sao testados como um lote unico e a "
                             "referencia nao separa variacao entre lotes de deriva.")
        pca, escala, dominio, ref, k = _calibrar(X_ref, lotes_ref, args.pcs, args.alpha, args.autoescala)
        res: Dict[str, Any] = dict(_julgar(pca, escala, dominio, X_nov))
        res["t2_limite"] = dominio.get("t2_limite")
        res["q_limite"] = dominio.get("q_limite")
        lotes = list(df_nov[args.lote].astype(str)) if args.lote else ["unico"] * len(df_nov)
        res["por_lote"] = _por_lote(lotes, res["dentro_dominio"], ref, args.alpha)
        os.makedirs(pasta_execucao(args.saida, "tabelas"), exist_ok=True)
        os.makedirs(pasta_execucao(args.saida, "relatorio"), exist_ok=True)
        amostras = pd.DataFrame({"lote": lotes, "T2": res["t2"], "Q": res["q"], "distancia": res["f"],
                                 "fora_do_dominio": ~np.asarray(res["dentro_dominio"], dtype=bool)})
        if args.coluna_id:
            amostras.insert(0, "id", list(df_nov[args.coluna_id].astype(str)))
        t1 = os.path.join(pasta_execucao(args.saida, "tabelas"), "monitor_amostras.csv")
        t2 = os.path.join(pasta_execucao(args.saida, "tabelas"), "monitor_lotes.csv")
        amostras.to_csv(t1, index=False, sep=";", decimal=",")
        pd.DataFrame(res["por_lote"]).drop(columns=["mensagem"]).to_csv(t2, index=False, sep=";", decimal=",")
        info = {"n_ref": len(df_ref), "n_lotes_ref": len(set(lotes_ref)) if lotes_ref else 1, "k": k,
                "alpha": args.alpha, "ref": ref, "por_lote": res["por_lote"], "sugestoes": sugestoes}
        rel = os.path.join(pasta_execucao(args.saida, "relatorio"), "monitor_resumo.txt")
        with open(rel, "w", encoding="utf-8") as fh:
            fh.write(texto_resumo(info) + "\n")
        cfg_ef = os.path.join(args.saida, "config_efetiva.yaml")
        with open(cfg_ef, "w", encoding="utf-8") as fh:
            fh.write("comando: monitorar\n")
            for chave, valor in (("referencia", os.path.basename(args.referencia)),
                                 ("novos", os.path.basename(args.novos)), ("lote", args.lote),
                                 ("pcs", args.pcs), ("pcs_usados", k), ("alpha", args.alpha),
                                 ("min_lotes", args.min_lotes), ("autoescala", args.autoescala)):
                fh.write(f"{chave}: {valor}\n")
        figs = _figuras(args.saida, res, lotes, args.mais_graficos, X_nov / escala, pca,
                        alpha=args.alpha)
        print(texto_resumo(info))
        print("\nArquivos gravados:")
        for arquivo in [t1, t2, rel, cfg_ef, *figs]:
            print(f"  {arquivo}")
        return 0
    except _Uso as e:
        print(f"{'Error' if _LANG['v'] == 'en' else 'Erro'}: {e}", file=sys.stderr)
        return 2
