"""quantificacao_pares.py -- um modelo de quantificacao (N3) por par OLEO x ADULTERANTE.

A regressao por especie do pipeline junta os adulterantes de uma mesma especie num modelo so'; quem
precisa da figura de merito DE CADA PAR (ex.: LOD de coco+soja contra coco+milho) roda uma analise por
par. Este modulo faz isso sem mudar o motor: para cada subpasta de `pasta_dados` (uma por especie/classe)
e cada adulterante achado nos nomes dos arquivos (`..._AD-<letra>-<teor>%_T<n>.dx`) monta uma pasta de
entrada com os arquivos desse par MAIS os puros da especie e roda `pipeline.executar` com nivel N3. Cada
corrida guarda o seu proprio `config_efetiva.yaml` e `Tables/regressao_figuras_de_merito.csv`.

Saida: `<saida>/tabela_pares.csv` (uma linha por par; par que falhou fica com `erro`, nunca some) e,
se pedido, a correlacao de Spearman entre o LOD e a diferenca de indice de iodo.

Uso: `python -m guaraci.quantificacao_pares PASTA_DADOS SAIDA [--config config.yaml] [--iodo iodo.csv]`.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["achar_pares", "rodar_pares", "correlacao_lod_iodo"]

_RE_AD = re.compile(r"AD[-_]([A-Za-z])[-_]")


def _adulterante_do_nome(nome: str) -> Optional[str]:
    m = _RE_AD.search(nome)
    return m.group(1).upper() if m else None


def achar_pares(pasta_dados: str) -> Dict[Tuple[str, str], Dict[str, List[str]]]:
    """{(especie, adulterante): {"misturas": [arquivos], "puros": [arquivos]}} a partir das subpastas.

    Especie = nome da subpasta; adulterante = letra do token `AD-<letra>-` no nome do arquivo; sem token = puro.
    So' entram pares com mistura e com ao menos um puro (sem o ponto 0 nao ha' curva)."""
    pares: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
    for esp in sorted(os.listdir(pasta_dados)):
        d = os.path.join(pasta_dados, esp)
        if not os.path.isdir(d):
            continue
        arqs = sorted(f for f in os.listdir(d) if f.lower().endswith(".dx"))
        puros = [os.path.join(d, f) for f in arqs if _adulterante_do_nome(f) is None]
        por_ad: Dict[str, List[str]] = {}
        for f in arqs:
            ad = _adulterante_do_nome(f)
            if ad is not None:
                por_ad.setdefault(ad, []).append(os.path.join(d, f))
        for ad, misturas in sorted(por_ad.items()):
            if misturas and puros:
                pares[(esp, ad)] = {"misturas": misturas, "puros": puros}
    return pares


def _montar_entrada(destino: str, arquivos: Sequence[str]) -> None:
    os.makedirs(destino, exist_ok=True)
    for a in arquivos:
        alvo = os.path.join(destino, os.path.basename(a))
        try:
            os.symlink(os.path.abspath(a), alvo)
        except (OSError, NotImplementedError):      # Windows sem privilegio de link: copia
            shutil.copy2(a, alvo)


def rodar_pares(pasta_dados: str, saida: str, cfg_base: Optional[Any] = None, **campos: Any) -> List[Dict[str, Any]]:
    """Roda um N3 por par e grava `tabela_pares.csv`. `campos` sobrescrevem a Config (ex.: wn_min=4540.0).

    `cfg_base` (opcional) e' uma Config ja' pronta (ex.: `load_config`); cada par recebe uma copia dela com
    `input_folder`, `output_root_folder`, nivel N3 e objetivo quantificacao."""
    import copy

    from guaraci import pipeline as pl
    from guaraci.config import NOME_TABELAS, Config, pasta_execucao
    from guaraci.resultados_io import COLUNAS_MERITO

    os.makedirs(saida, exist_ok=True)
    pares = achar_pares(pasta_dados)
    if not pares:
        raise ValueError(f"Nenhum par especie x adulterante em {pasta_dados}: espere subpastas por especie com "
                         f"arquivos .dx nomeados '..._AD-<letra>-<teor>%_T<n>.dx' e os puros sem o token AD.")
    linhas: List[Dict[str, Any]] = []
    for (esp, ad), g in pares.items():
        nome = f"{esp}_{ad}".replace(" ", "_")
        entrada = os.path.join(saida, "_entradas", nome)
        _montar_entrada(entrada, g["misturas"] + g["puros"])
        cfg = copy.deepcopy(cfg_base) if cfg_base is not None else Config()
        for k, v in dict(mode="dx", level="N3", objective="quantificacao", extract_conc_from_filename=True,
                         input_folder=entrada, output_root_folder=os.path.join(saida, nome), **campos).items():
            setattr(cfg, k, v)
        linha: Dict[str, Any] = {"especie": esp, "adulterante": ad, "n_misturas": len(g["misturas"]),
                                 "n_puros": len(g["puros"]), "pasta": "", "erro": ""}
        try:
            pl.executar(cfg)
            pasta_run = str(getattr(cfg, "output_folder", "") or "")
            linha["pasta"] = os.path.relpath(pasta_run, saida) if pasta_run else ""
            csv_m = os.path.join(pasta_execucao(pasta_run, NOME_TABELAS), "regressao_figuras_de_merito.csv")
            if not os.path.isfile(csv_m):
                linha["erro"] = "a corrida terminou sem regressao_figuras_de_merito.csv (regressao nao rodou)"
            else:
                with open(csv_m, encoding="utf-8", newline="") as fh:
                    linhas_m = list(csv.DictReader(fh))
                ref = next((r for r in linhas_m if r["modelo"] == "pooled"), linhas_m[0])
                linha.update({k: ref.get(k, "") for k in COLUNAS_MERITO if k != "modelo"})
        except Exception as e:  # noqa: BLE001 -- um par que falha nao derruba os outros; o erro vai na tabela
            linha["erro"] = f"{type(e).__name__}: {str(e)[:300]}"
        linhas.append(linha)
        print(f"[PAR] {esp} x {ad}: " + (linha["erro"] or f"LV={linha.get('n_lv')} RMSECV={linha.get('rmsecv')} "
                                                        f"R2cv={linha.get('r2cv')} LOD={linha.get('lod')}"))
    colunas = ["especie", "adulterante", "n_misturas", "n_puros", *[c for c in COLUNAS_MERITO if c != "modelo"],
               "pasta", "erro"]
    with open(os.path.join(saida, "tabela_pares.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=colunas, extrasaction="ignore")
        w.writeheader()
        w.writerows(linhas)
    return linhas


def correlacao_lod_iodo(linhas: Sequence[Dict[str, Any]], iodo_oleo: Dict[str, float],
                        iodo_adulterante: Dict[str, float]) -> Dict[str, Any]:
    """Spearman entre o LOD de cada par e |iodo do oleo - iodo do adulterante|.

    Pares sem LOD numerico, ou sem indice de iodo conhecido, ficam de fora e sao contados."""
    from scipy.stats import spearmanr
    xs: List[float] = []
    ys: List[float] = []
    fora = 0
    for r in linhas:
        try:
            lod = float(r.get("lod", ""))
            d = abs(float(iodo_oleo[str(r["especie"])]) - float(iodo_adulterante[str(r["adulterante"])]))
        except (TypeError, ValueError, KeyError):
            fora += 1
            continue
        if lod == lod:
            xs.append(d)
            ys.append(lod)
        else:
            fora += 1
    if len(xs) < 4:
        return {"n": len(xs), "fora": fora, "rho": float("nan"), "p": float("nan"),
                "aviso": "menos de 4 pares com LOD e iodo: correlacao nao calculada"}
    rho, p = spearmanr(xs, ys)
    return {"n": len(xs), "fora": fora, "rho": float(rho), "p": float(p), "aviso": ""}


def _ler_mapa(caminho: str) -> Dict[str, float]:
    with open(caminho, encoding="utf-8", newline="") as fh:
        return {r[0].strip(): float(r[1].replace(",", ".")) for r in csv.reader(fh)
                if len(r) >= 2 and r[0].strip() and r[0].strip().lower() not in ("oleo", "especie", "adulterante")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="guaraci pares", description=__doc__.split("\n\n")[0])
    ap.add_argument("pasta_dados")
    ap.add_argument("saida")
    ap.add_argument("--config", help="config.yaml base (faixa, pre-processamento, validacao...)")
    ap.add_argument("--iodo-especies", help="CSV especie,iodo (nome = nome da subpasta)")
    ap.add_argument("--iodo-adulterantes", help="CSV letra,iodo (ex.: S,130)")
    a = ap.parse_args(argv)
    base = None
    if a.config:
        from guaraci.config_io import load_config
        base = load_config(a.config)
    linhas = rodar_pares(a.pasta_dados, a.saida, base)
    if a.iodo_especies and a.iodo_adulterantes:
        c = correlacao_lod_iodo(linhas, _ler_mapa(a.iodo_especies), _ler_mapa(a.iodo_adulterantes))
        print(f"[IODO] Spearman LOD x |delta iodo|: rho={c['rho']:.3f} p={c['p']:.4f} n={c['n']} "
              f"(fora: {c['fora']}) {c['aviso']}")
    return 1 if any(r["erro"] for r in linhas) else 0


if __name__ == "__main__":
    raise SystemExit(main())
