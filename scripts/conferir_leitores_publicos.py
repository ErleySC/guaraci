"""Confere os leitores de .mat (MATLAB v4 a v7.3) e de GC-FID (pasta Agilent .D) com arquivos REAIS publicos.

Nada aqui entra no pacote e nenhum arquivo de terceiros vai para o repositorio (licencas diferentes). Os
arquivos sao gravados pelo proprio MATLAB ou por cromatografos Agilent; nenhum e' imagem hiperespectral nem
cromatograma de amostra de alimento: o que se confere e' o FORMATO, nao o desempenho do modelo.

Fontes:
  - scipy (instalado): ~110 .mat do MATLAB 4.2c a 8 em scipy/io/matlab/tests/data (BSD-3).
  - mat7.3 (MIT): .mat v7.3 gravados pelo MATLAB, https://github.com/skjerns/mat7.3 (tests/testfile*.mat).
  - rainbow (LGPL): yellow.D, GC com detector FID, https://github.com/evanyeyeye/rainbow (tests/inputs).

Uso (de fora do repositorio):
    mkdir dados && cd dados
    git clone --depth 1 https://github.com/skjerns/mat7.3.git mat73
    git clone --depth 1 https://github.com/evanyeyeye/rainbow.git rb
    python conferir_leitores_publicos.py --dados . [--json saida.json]

Saida: uma linha por controle e codigo 1 se algum falhar. Com --json, grava contagens e o sha256 de cada
arquivo conferido (o registro em docs/qualidade/leitores_publicos.json foi gerado assim)."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import warnings
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")
FALHAS: List[str] = []
ARQUIVOS: Dict[str, str] = {}


def controle(ok: bool, texto: str) -> None:
    print(("  ok    " if ok else "  FALHA ") + texto)
    if not ok:
        FALHAS.append(texto)


def _sha(caminho: str) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _varrer_mat(arquivos: List[str], rotulo: str) -> Dict[str, int]:
    from guaraci import arquivo_matrizes as am
    cont = {"arquivos": 0, "lidos": 0, "recusados_com_mensagem": 0, "variaveis": 0, "ignoradas": 0}
    for p in arquivos:
        cont["arquivos"] += 1
        ARQUIVOS[f"{rotulo}/{os.path.basename(p)}"] = _sha(p)
        nome = os.path.basename(p)
        try:
            _fmt, vs = am.listar_variaveis(p)
        except ValueError as e:
            controle(nome in str(e), f"{nome}: recusado com mensagem que cita o arquivo")
            cont["recusados_com_mensagem"] += 1
            continue
        except Exception as e:  # noqa: BLE001 -- e' exatamente o que se quer pegar
            controle(False, f"{nome}: excecao crua {type(e).__name__}: {e}")
            continue
        ok = True
        for v in vs:
            try:
                a = np.asarray(v.ler())
            except ValueError as e:
                ok = ok and nome in str(e)
                continue
            except Exception as e:  # noqa: BLE001
                controle(False, f"{nome}:{v.nome}: excecao crua {type(e).__name__}: {e}")
                ok = False
                continue
            ok = ok and tuple(a.shape) == v.forma and a.dtype.kind in "iufb"
        cont["lidos"] += 1
        cont["variaveis"] += len(vs)
        cont["ignoradas"] += len(getattr(vs, "ignoradas", []))
        controle(ok, f"{nome}: {len(vs)} variavel(is) numerica(s) com a forma anunciada")
    return cont


def mat_scipy() -> Dict[str, int]:
    import scipy.io.matlab as m
    pasta = os.path.join(os.path.dirname(m.__file__), "tests", "data")
    return _varrer_mat(sorted(glob.glob(os.path.join(pasta, "*.mat"))), "scipy")


def mat73(d: str) -> Dict[str, int]:
    from guaraci import arquivo_matrizes as am
    arqs = sorted(glob.glob(os.path.join(d, "mat73", "tests", "*.mat")))
    cont = _varrer_mat(arqs, "mat7.3")
    p = os.path.join(d, "mat73", "tests", "testfile14.mat")
    if os.path.isfile(p):
        _f, vs = am.listar_variaveis(p)
        controle([v.forma for v in vs] == [(3, 1, 4, 2)], "testfile14.mat: matriz 4D na ordem do MATLAB (3x1x4x2)")
    return cont


def gc_fid(d: str) -> Dict[str, int]:
    from guaraci.importadores_proprietarios import parse_cromatograma_hplc
    pasta = os.path.join(d, "rb", "tests", "inputs", "yellow.D")
    ref = os.path.join(d, "rb", "tests", "outputs", "yellow", "FID1A.csv")
    ARQUIVOS["rainbow/yellow.D/FID1A.ch"] = _sha(os.path.join(pasta, "FID1A.ch"))
    x, y = parse_cromatograma_hplc(pasta, detector="FID")
    esperado = np.loadtxt(ref, delimiter=",", skiprows=1)
    controle(x.size == esperado.shape[0], f"yellow.D: {x.size} pontos de FID (esperado {esperado.shape[0]})")
    controle(bool(np.allclose(x, esperado[:, 0]) and np.allclose(y, esperado[:, 1])),
             "yellow.D: tempo e sinal iguais a saida de referencia do rainbow")
    controle(bool(np.all(np.diff(x) > 0)) and 8.0 < x[-1] < 9.0, f"yellow.D: tempo crescente ate {x[-1]:.2f} min")
    try:
        parse_cromatograma_hplc(pasta, detector="UV")
        controle(False, "yellow.D: pedir UV num GC-FID devia falhar")
    except ValueError as e:
        controle("FID" in str(e), "yellow.D: pedir UV lista os detectores que existem (FID, MS)")
    return {"pastas": 1, "pontos": int(x.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dados", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    resumo = {"mat_scipy": mat_scipy(), "mat73": mat73(a.dados), "gc_fid_agilent": gc_fid(a.dados)}
    resumo["controles_falhos"] = len(FALHAS)
    print(json.dumps(resumo, indent=1))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"resumo": resumo, "sha256": ARQUIVOS}, f, indent=1, ensure_ascii=True, sort_keys=True)
            f.write("\n")
    return 1 if FALHAS else 0


if __name__ == "__main__":
    sys.exit(main())
