# -*- coding: utf-8 -*-
"""Um modelo de quantificacao por par especie x adulterante e a tabela de figuras de merito em CSV."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from guaraci import quantificacao_pares as qp


def _dx(path, titulo, y, wn):
    n = len(wn)
    corpo = "\n".join(f"{wn[k]:.4f} " + " ".join(f"{v:.6f}" for v in y[k:k + 5]) for k in range(0, n, 5))
    Path(path).write_text(
        f"##TITLE={titulo}\n##JCAMP-DX=4.24\n##DATA TYPE=NEAR INFRARED SPECTRUM\n##XUNITS=1/CM\n##YUNITS=ABSORBANCE\n"
        f"##XFACTOR=1\n##YFACTOR=1\n##FIRSTX={wn[0]}\n##LASTX={wn[-1]}\n##NPOINTS={n}\n##XYDATA=(X++(Y..Y))\n{corpo}\n##END=\n",
        encoding="latin-1")


def _raiz(tmp_path, adulterantes=("S", "M")):
    rng = np.random.default_rng(0)
    wn = np.linspace(4000, 10000, 200)
    base = np.exp(-((wn - 7000) / 1500.0) ** 2) + 0.3
    raiz = tmp_path / "dados"
    d = raiz / "CLA"
    d.mkdir(parents=True)
    for r in range(3):
        _dx(d / f"CLA-01-01-2099_T{r + 1}.dx", f"CLA-01-01-2099_T{r + 1}",
            base + rng.normal(0, 0.002, wn.size), wn)
    for k, ad in enumerate(adulterantes):
        banda = np.exp(-((wn - (5200 + 900 * k)) / 200.0) ** 2)
        for teor in range(1, 11):
            for r in range(3):
                nome = f"CLA-01-01-2099-AD-{ad}-{teor},0%_T{r + 1}"
                _dx(d / f"{nome}.dx", nome, base + 0.05 * teor * banda + rng.normal(0, 0.002, wn.size), wn)
    return str(raiz)


def test_achar_pares_separa_por_adulterante_e_exige_puros(tmp_path):
    raiz = _raiz(tmp_path)
    pares = qp.achar_pares(raiz)
    assert set(pares) == {("CLA", "S"), ("CLA", "M")}
    assert len(pares[("CLA", "S")]["misturas"]) == 30 and len(pares[("CLA", "S")]["puros"]) == 3
    for f in list(Path(raiz, "CLA").glob("CLA-01-01-2099_T*.dx")):
        f.unlink()
    assert qp.achar_pares(raiz) == {}          # sem puro nao ha' ponto zero: o par nao e' tentado


def test_um_modelo_por_par_grava_tabela_e_csv_de_merito(tmp_path):
    raiz = _raiz(tmp_path)
    saida = str(tmp_path / "saida")
    linhas = qp.rodar_pares(raiz, saida, None, wn_min=4000.0, wn_max=10000.0, sg_window=11, default_preprocessing="msc_sg_mc",
                            n_permutations=3, n_permutations_wold=3, n_bootstrap_vip=3, n_bootstrap_bca=10,
                            n_monte_carlo=2, max_lvs=6)
    assert [(r["especie"], r["adulterante"]) for r in linhas] == [("CLA", "M"), ("CLA", "S")]
    for r in linhas:
        assert r["erro"] == "", r["erro"]
        assert float(r["r2cv"]) > 0.9 and float(r["lod"]) > 0 and int(r["n_lv"]) >= 1
        assert os.path.isfile(os.path.join(saida, r["pasta"], "config_efetiva.yaml"))     # para o apendice
        vs = list(Path(saida, r["pasta"]).rglob("regressao_vip_sr.csv"))
        assert len(vs) == 1                                        # VIP e SR da quantificacao
        with open(vs[0], encoding="utf-8") as fh:
            lin = list(csv.DictReader(fh))
        assert len(lin) > 50 and float(lin[0]["VIP"]) >= 0
        assert max(float(x["VIP"]) for x in lin) > 1.0
    with open(os.path.join(saida, "tabela_pares.csv"), encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 2


def test_par_que_falha_vai_para_a_tabela_com_o_erro(tmp_path, monkeypatch):
    raiz = _raiz(tmp_path, ("S",))
    import guaraci.pipeline as pl

    def estoura(cfg):
        raise ValueError("falha de proposito")
    monkeypatch.setattr(pl, "executar", estoura)
    linhas = qp.rodar_pares(raiz, str(tmp_path / "s"))
    assert len(linhas) == 1 and "falha de proposito" in linhas[0]["erro"]


def test_correlacao_lod_iodo_ignora_o_que_nao_tem_dado():
    linhas = [{"especie": e, "adulterante": "S", "lod": lod} for e, lod in
              (("A", "0.1"), ("B", "0.2"), ("C", "0.3"), ("D", "0.4"), ("E", ""), ("F", "0.9"))]
    iodo = {"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0, "E": 50.0}
    r = qp.correlacao_lod_iodo(linhas, iodo, {"S": 130.0})
    assert r["n"] == 4 and r["fora"] == 2                 # E sem LOD, F sem iodo
    assert r["rho"] == -1.0                               # LOD cresce quando |delta iodo| cai
    assert qp.correlacao_lod_iodo(linhas[:2], iodo, {"S": 130.0})["aviso"]


# ------------------------------------------------------------- linha de comando (mesmo padrao dos irmaos)
def test_cli_ajuda_pt_en_e_erros_de_uso(capsys, tmp_path):
    assert qp.executar(["--ajuda", "--idioma", "pt"]) == 0
    assert "Uso: guaraci pares" in capsys.readouterr().out
    assert qp.executar(["--help", "--idioma", "en"]) == 0
    assert "Usage: guaraci pares" in capsys.readouterr().out
    assert qp.executar([str(tmp_path / "nao_existe"), "--idioma", "pt"]) == 2
    assert "nao existe" in capsys.readouterr().err
    assert qp.executar([str(tmp_path), "--iodo-especies", "x.csv", "--idioma", "pt"]) == 2     # as duas tabelas de iodo vao juntas
    assert "iodo" in capsys.readouterr().err.lower()


def test_cli_roda_e_grava_a_tabela(tmp_path, capsys):
    raiz = _raiz(tmp_path, adulterantes=("S",))
    saida = tmp_path / "saida"
    assert qp.executar([raiz, "--saida", str(saida), "--idioma", "pt"]) == 0
    assert (saida / "tabela_pares.csv").is_file()
    assert "CLA + S" in capsys.readouterr().out
