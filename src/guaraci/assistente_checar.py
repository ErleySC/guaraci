"""assistente_checar.py -- pre-voo e leitura de resultado pela linha de comando, sem abrir as telas.

    python -m guaraci.assistente_checar config.yaml               # so' a configuracao
    python -m guaraci.assistente_checar config.yaml --dados       # carrega os dados e faz o pre-voo
    python -m guaraci.assistente_checar config.yaml --resultado PASTA_DA_EXECUCAO [--ficha]

`--resultado` le PASTA/Relatorios/resumo_modelo.txt (ou logs/, do layout antigo, ou o proprio arquivo).
`--ficha` imprime a ficha anti-vazamento em Markdown. Saida 1 se ha achado critico, 0 caso contrario. Nada e' alterado.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, List, Optional, Sequence

from guaraci import assistente_regras as regras
from guaraci import assistente_vazamento as vazamento

__all__ = ["main", "resumo_dos_dados", "ler_resultado"]


def resumo_dos_dados(cfg: Any) -> regras.ResumoDados:
    """Carrega os dados como o pipeline carrega e devolve o resumo (importa o motor)."""
    from guaraci import pipeline as pq
    eixo, X, rotulos, _conc, mae_id, _meta = pq.load_data(cfg)
    return regras.resumir_dados(X, eixo, rotulos, mae_id)


def _pasta_do_relatorio_hsi(caminho: str) -> Optional[str]:
    """Pasta com o hsi_relatorio.(json|txt) de uma execucao HSI (que nao grava resumo_modelo.txt), ou None."""
    if not os.path.isdir(caminho):
        return None
    for sub in ("Relatorios", "Reports", "logs", ""):
        pasta = os.path.join(caminho, sub)
        if any(os.path.isfile(os.path.join(pasta, f)) for f in ("hsi_relatorio.json", "hsi_relatorio.txt")):
            return pasta
    return None


def ler_resultado(caminho: str) -> dict:
    arq = caminho
    if os.path.isdir(caminho):
        # Relatorios/ e' a pasta do resumo desde o layout novo (config.NOME_RELATORIOS); logs/ e' o antigo.
        for sub in ("Relatorios", "Reports", "logs", ""):   # PT, EN (sessao em ingles), antigo
            arq = os.path.join(caminho, sub, "resumo_modelo.txt")
            if os.path.exists(arq):
                break
        else:
            # execucao HSI: nao ha' resumo_modelo.txt; o resultado sao as chaves `hsi_*` do hsi_relatorio (a ficha as le)
            rel = _pasta_do_relatorio_hsi(caminho)
            if rel is not None:
                from guaraci.resumo_parse import ler_hsi_pasta
                return {f"hsi_{k}": v for k, v in ler_hsi_pasta(rel).items()}
    from guaraci.resumo_parse import ler_resumo_canonico   # chaves fixas (o .txt sai no idioma da sessao)
    texto = ler_resumo_canonico(arq)
    if texto is None:
        raise FileNotFoundError(arq)
    return regras.resultado_de_resumo(texto)


def _imprimir_extras_ausentes(lang: str) -> None:
    """Lista os pacotes opcionais que faltam e o que cada um deixa de habilitar (nada se estao todos)."""
    from guaraci import extras_opcionais as ext
    faltam = ext.ausentes()
    if not faltam:
        return
    en = lang.upper() == "EN"
    print()
    print("[INFO] " + ("Optional packages not installed (what each one enables):" if en
                       else "Pacotes opcionais nao instalados (o que cada um habilita):"))
    for e in faltam:
        tec = ", ".join(e.tecnicas) or "-"
        print(f"  - {e.pacote}: {e.afeta[1] if en else e.afeta[0]} | " + ("techniques" if en else "tecnicas") + f": {tec} | "
              + ext.texto_instalar(e, lang))
    print("  " + ("Install everything: " if en else "Instalar tudo: ") + ext.COMANDO_TUDO)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m guaraci.assistente_checar",
                                 description="Pre-voo e leitura de resultado do assistente Guaraci (offline).")
    ap.add_argument("config", help="config.yaml da analise")
    ap.add_argument("--dados", action="store_true", help="carrega os dados e aplica as regras de dados")
    ap.add_argument("--resultado", metavar="PASTA", help="pasta da execucao ou resumo_modelo.txt")
    ap.add_argument("--ficha", action="store_true", help="imprime a ficha anti-vazamento")
    ap.add_argument("--idioma", default="PT", choices=("PT", "EN", "pt", "en"))
    a = ap.parse_args(argv)
    lang = a.idioma.upper()

    from guaraci.config_io import load_config
    try:
        cfg = load_config(a.config)
    except (OSError, ValueError) as e:
        from guaraci.assistente_erros import diagnosticar_excecao
        print(diagnosticar_excecao(e, lang), file=sys.stderr)
        return 2
    dados = None
    if a.dados:
        try:
            dados = resumo_dos_dados(cfg)
        except Exception as e:  # noqa: BLE001 -- dado externo falha de muitas formas; o diagnostico mostra a causa
            from guaraci.assistente_erros import diagnosticar_excecao
            print(diagnosticar_excecao(e, lang), file=sys.stderr)
            return 2
    resultado = None
    if a.resultado:
        try:
            resultado = ler_resultado(a.resultado)
        except OSError as e:
            from guaraci.assistente_erros import diagnosticar_excecao
            print(diagnosticar_excecao(e, lang), file=sys.stderr)
            return 2

    achados: List[regras.Achado] = regras.avaliar(cfg, dados, resultado, lang)
    print(regras.formatar_achados(achados, lang))
    _imprimir_extras_ausentes(lang)
    if a.ficha:
        print()
        print(vazamento.ficha(cfg, dados, resultado, lang))
    return 1 if any(x.severidade == "critico" for x in achados) else 0


if __name__ == "__main__":
    sys.exit(main())
