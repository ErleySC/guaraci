"""transferir_cli.py -- `guaraci transferir`: transferencia de calibracao entre instrumentos (DS/PDS).

    guaraci transferir --mestre MESTRE.csv --escravo ESCRAVO.csv --novos NOVOS.csv
                       [--metodo ds|pds] [--janela 5] [--alpha X] [--saida PASTA]

MESTRE e ESCRAVO tem as MESMAS amostras de transferencia, na mesma ordem de linha, medidas no
instrumento em que o modelo foi calibrado (mestre) e no novo (escravo). NOVOS sao espectros do
escravo a converter para o espaco do mestre (para usar com o modelo do mestre). Reaproveita
`transferencia_calibracao`; nao reimplementa DS nem PDS.
"""
from __future__ import annotations

import argparse
import os

from guaraci.config import nome_pasta
import sys
from typing import Any, List, Optional, Sequence, Tuple

from guaraci import erros_entrada as _erros

__all__ = ["executar", "erro_loo"]

USO = ("Uso: guaraci transferir --mestre MESTRE.csv --escravo ESCRAVO.csv --novos NOVOS.csv\n"
       "                        [--metodo ds|pds] [--janela 5] [--alpha X] [--saida PASTA]\n"
       "  --mestre    amostras de transferencia medidas no instrumento do modelo\n"
       "  --escravo   as MESMAS amostras, mesma ordem, medidas no instrumento novo\n"
       "  --novos     espectros do instrumento novo a converter\n"
       "  --metodo    pds (padrao: janelas locais, pede menos amostras) ou ds (global)\n"
       "  --janela    meia-largura da janela do PDS, em canais (padrao 5)\n"
       "  --alpha     regularizacao ridge (padrao: escolhida por validacao deixando uma amostra de fora)\n"
       "  --saida     pasta dos resultados (padrao: resultados_transferencia)\n")


class _Uso(Exception):
    """Erro de uso (exit 2)."""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="guaraci transferir", add_help=False, exit_on_error=False)
    for a in ("--mestre", "--escravo", "--novos"):
        p.add_argument(a)
    p.add_argument("--metodo", default="pds", choices=("ds", "pds"))
    p.add_argument("--janela", type=int, default=5)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--saida", default=nome_pasta("resultados_transferencia"))
    p.add_argument("--idioma", default=None)
    p.add_argument("--ajuda", "--help", "-h", action="store_true", dest="ajuda")
    return p


_LANG = {"v": "pt"}


def _ler(caminho: Optional[str]) -> Tuple[Any, List[str], Any]:
    import pandas as pd
    if not caminho or not os.path.isfile(caminho):
        raise _Uso(_erros.arquivo_nao_encontrado(caminho, _LANG["v"], "transferir"))
    try:
        df = pd.read_csv(caminho, sep=None, engine="python", encoding="utf-8-sig")
        if len(df) == 0:
            raise ValueError("Could not determine delimiter")        # so' cabecalho ou nada: tratado como arquivo vazio
    except Exception as e:   # noqa: BLE001 -- qualquer falha de leitura vira mensagem de uso
        raise _Uso(_erros.leitura_csv(caminho, e, _LANG["v"], "transferir")) from e
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num) < 2:
        raise _Uso(f"{os.path.basename(caminho)}: menos de 2 colunas numericas.")
    return df, num, df[num].to_numpy(dtype=float)


def _ajustar(metodo: str, Xm, Xs, janela: int, alpha: Optional[float]):
    from guaraci.transferencia_calibracao import direct_standardization, piecewise_direct_standardization
    if metodo == "ds":
        return direct_standardization(Xm, Xs, alpha=alpha)
    return piecewise_direct_standardization(Xm, Xs, janela=janela, alpha=alpha)


def erro_loo(metodo: str, Xm, Xs, janela: int, alpha: float) -> Tuple[float, float]:
    """(RMS sem correcao, RMS com correcao) do escravo contra o mestre, deixando cada amostra de
    transferencia de fora do ajuste (a correcao nunca e' julgada na amostra que a ajustou)."""
    import numpy as np
    from guaraci.transferencia_calibracao import apply_standardization
    n = Xm.shape[0]
    antes, depois = [], []
    for i in range(n):
        dentro = np.arange(n) != i
        t = _ajustar(metodo, Xm[dentro], Xs[dentro], janela, alpha)
        antes.append(np.mean((Xs[i] - Xm[i]) ** 2))
        depois.append(np.mean((apply_standardization(Xs[i:i + 1], t)[0] - Xm[i]) ** 2))
    return float(np.sqrt(np.mean(antes))), float(np.sqrt(np.mean(depois)))


def executar(argv: Optional[Sequence[str]] = None) -> int:
    """Roda `guaraci transferir`. Devolve 0 (ok) ou 2 (uso)."""
    import pandas as pd
    args_l = list(sys.argv[1:] if argv is None else argv)
    _LANG["v"] = _erros.idioma(args_l)
    try:
        try:
            args = _parser().parse_args(args_l)
        except (argparse.ArgumentError, SystemExit) as e:
            raise _Uso(f"argumento invalido: {e}") from None
        if args.ajuda:
            print(USO)
            return 0
        if not (args.mestre and args.escravo and args.novos):
            raise _Uso("informe --mestre, --escravo e --novos.\n\n" + USO)
        if args.janela < 0 or (args.alpha is not None and args.alpha < 0):
            raise _Uso("--janela e --alpha devem ser >= 0.")
        _dm, col_m, Xm = _ler(args.mestre)
        _ds, col_s, Xs = _ler(args.escravo)
        df_n, col_n, Xn = _ler(args.novos)
        if Xm.shape != Xs.shape:
            raise _Uso(f"mestre {Xm.shape} e escravo {Xs.shape} precisam ter as mesmas amostras e colunas.")
        if Xn.shape[1] != Xs.shape[1]:
            raise _Uso(f"novos tem {Xn.shape[1]} colunas numericas; o escravo tem {Xs.shape[1]}.")
        if Xm.shape[0] < 3:
            raise _Uso("sao precisas pelo menos 3 amostras de transferencia.")
    except _Uso as e:
        print(f"{'Error' if _LANG['v'] == 'en' else 'Erro'}: {e}", file=sys.stderr)
        return 2
    from guaraci.transferencia_calibracao import apply_standardization
    t = _ajustar(args.metodo, Xm, Xs, args.janela, args.alpha)
    alpha_usado = t.alpha
    rms_antes, rms_depois = erro_loo(args.metodo, Xm, Xs, args.janela, alpha_usado)
    convertidos = apply_standardization(Xn, t)
    os.makedirs(args.saida, exist_ok=True)
    meta = [c for c in df_n.columns if c not in col_n]
    out = pd.concat([df_n[meta].reset_index(drop=True), pd.DataFrame(convertidos, columns=col_m)], axis=1)
    destino = os.path.join(args.saida, "novos_no_espaco_do_mestre.csv")
    out.to_csv(destino, index=False)
    resumo = (f"Transferencia {args.metodo.upper()}"
              + (f" (janela {args.janela})" if args.metodo == "pds" else "")
              + f", {Xm.shape[0]} amostras de transferencia, alpha {alpha_usado:.4g}"
              + (" (escolhido por validacao)" if args.alpha is None else "") + "\n"
              f"Diferenca escravo x mestre (RMS, deixando cada amostra de fora): {rms_antes:.4g} sem correcao, "
              f"{rms_depois:.4g} com correcao\n"
              f"{Xn.shape[0]} espectro(s) convertido(s): {destino}\n"
              "Use o CSV convertido com o modelo calibrado no mestre e confira o erro em amostras de referencia.")
    with open(os.path.join(args.saida, "transferencia_resumo.txt"), "w", encoding="utf-8") as fh:
        fh.write(resumo + "\n")
    with open(os.path.join(args.saida, "config_efetiva.yaml"), "w", encoding="utf-8") as fh:
        fh.write(f"comando: transferir\nmetodo: {args.metodo}\njanela: {args.janela}\nalpha: {alpha_usado}\n"
                 f"mestre: {os.path.basename(args.mestre)}\nescravo: {os.path.basename(args.escravo)}\n"
                 f"novos: {os.path.basename(args.novos)}\n")
    print(resumo)
    return 0
