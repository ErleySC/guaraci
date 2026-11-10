"""`guaraci monitorar`: regras de contexto, escolha de componentes, ponta a ponta pelo CLI,
taxa de falso alarme em controle (com IC)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta

from guaraci import exemplos, monitorar_cli as mon

RAIZ = Path(__file__).resolve().parents[1]


def _cli(args, tmp_path):
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
           "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    r = subprocess.run([sys.executable, "-m", "guaraci", "monitorar", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, cwd=str(tmp_path), timeout=300)
    return r.returncode, r.stdout + r.stderr


def _csv(caminho, X, lotes):
    linhas = ["lote," + ",".join(f"v{i}" for i in range(X.shape[1]))]
    linhas += [f"{lt}," + ",".join(f"{v:.6f}" for v in x) for lt, x in zip(lotes, X)]
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return str(caminho)


# ---------------------------------------------------------------- regras de contexto
def test_conferir_referencia_recusa_poucos_lotes_e_poucas_amostras():
    assert "minimo e' 3" in mon.conferir_referencia(["a"] * 6 + ["b"] * 6, 12, 3)
    assert "pelo menos 10" in mon.conferir_referencia(["a", "b", "c"] * 3, 9, 3)
    assert mon.conferir_referencia(["a", "b", "c"] * 4, 12, 3) is None
    assert mon.conferir_referencia(None, 12, 3) is None            # sem lote: vira sugestao, nao recusa
    assert mon.conferir_referencia(["a", "b"] * 6, 12, 2) is None   # o usuario reduziu --min-lotes


def test_escolher_pcs_auto_explicito_e_limites():
    assert mon.escolher_pcs([0.6, 0.25, 0.1, 0.05], "auto", 50, 40) == 3     # 0.6+0.25 < 0.90 <= +0.1
    assert mon.escolher_pcs([0.95, 0.05], "auto", 50, 40) == 1
    assert mon.escolher_pcs([0.01] * 100, "auto", 50, 100) == mon.MAX_PCS_AUTO
    assert mon.escolher_pcs([0.5] * 2, "auto", 4, 2) == 2                     # teto n_ref - 2
    assert mon.escolher_pcs([0.5], "4", 50, 40) == 4
    with pytest.raises(mon._Uso):
        mon.escolher_pcs([0.5], "0", 50, 40)
    with pytest.raises(mon._Uso):
        mon.escolher_pcs([0.5], "49", 50, 40)
    with pytest.raises(mon._Uso):
        mon.escolher_pcs([0.5], "dois", 50, 40)


def test_cli_recusa_referencia_com_dois_lotes(tmp_path):
    rng = np.random.default_rng(0)
    ref = _csv(tmp_path / "ref.csv", rng.normal(size=(24, 8)), ["a"] * 12 + ["b"] * 12)
    nov = _csv(tmp_path / "nov.csv", rng.normal(size=(20, 8)), ["n"] * 20)
    rc, out = _cli(["--referencia", ref, "--novos", nov, "--lote", "lote"], tmp_path)
    assert rc == 2 and "monitoramento recusado" in out and "2 lote(s)" in out


def test_cli_sem_lote_roda_e_sugere(tmp_path):
    rng = np.random.default_rng(1)
    ref = _csv(tmp_path / "ref.csv", rng.normal(size=(30, 8)), ["a"] * 30)
    nov = _csv(tmp_path / "nov.csv", rng.normal(size=(20, 8)), ["n"] * 20)
    rc, out = _cli(["--referencia", ref, "--novos", nov, "--saida", str(tmp_path / "s")], tmp_path)
    assert rc == 0, out
    assert "Sugestao: informe --lote COL" in out and "lote unico" in out


def test_cli_colunas_diferentes_e_ajuda(tmp_path):
    rng = np.random.default_rng(2)
    ref = _csv(tmp_path / "ref.csv", rng.normal(size=(30, 8)), list("abc") * 10)
    nov = _csv(tmp_path / "nov.csv", rng.normal(size=(20, 7)), ["n"] * 20)
    rc, out = _cli(["--referencia", ref, "--novos", nov, "--lote", "lote"], tmp_path)
    assert rc == 2 and "diferem" in out
    rc, out = _cli(["--ajuda"], tmp_path)
    assert rc == 0 and "--referencia" in out


# ---------------------------------------------------------------- ponta a ponta
def test_exemplo_monitorar_alarma_so_os_lotes_com_deriva(tmp_path):
    exemplos.gerar("monitorar", str(tmp_path / "ex"))
    p = tmp_path / "ex"
    rc, out = _cli(["--referencia", str(p / "referencia.csv"), "--novos", str(p / "novos.csv"), "--lote", "lote",
                    "--id", "amostra", "--mais-graficos", "--saida", str(p / "res")], tmp_path)
    assert rc == 0, out
    import pandas as pd
    lotes = pd.read_csv(p / "res" / "tabelas" / "monitor_lotes.csv", sep=";", decimal=",")
    alarme = dict(zip(lotes["lote"], lotes["alarme"]))
    assert [alarme[f"N{i:02d}"] for i in range(6)] == [False, False, False, False, True, True]
    for f in ("carta_t2_q", "tendencia_por_lote", "contribuicao_q"):
        assert (p / "res" / "Graficos" / f"{f}.png").is_file()
    assert (p / "res" / "config_efetiva.yaml").is_file() and (p / "res" / "relatorio" / "monitor_resumo.txt").is_file()


def _processo(rng, n_lotes, n_por, p=40, delta=0.0):
    eixo = np.linspace(0, 1, p)
    base = np.exp(-((eixo - 0.5) / 0.1) ** 2)
    linhas, lotes = [], []
    for lt in range(n_lotes):
        dl = rng.normal(0, 0.01, p)
        for _ in range(n_por):
            linhas.append(base * rng.normal(1, 0.02) + dl + delta * eixo + rng.normal(0, 0.01, p))
            lotes.append(f"L{lt}")
    return np.asarray(linhas), lotes


def test_falso_alarme_em_controle_fica_perto_do_nominal():
    """Lotes novos do MESMO processo: fracao de lotes alarmados com alpha 0,05, em 30 calibracoes."""
    alarmes, total = 0, 0
    for seed in range(30):
        rng = np.random.default_rng(seed)
        X_ref, l_ref = _processo(rng, 5, 12)
        X_nov, l_nov = _processo(rng, 3, 20)
        pca, esc, dom, ref, _k = mon._calibrar(X_ref, l_ref, "auto", 0.05, False)
        res = mon._julgar(pca, esc, dom, X_nov)
        for r in mon._por_lote(l_nov, res["dentro_dominio"], ref, 0.05):
            alarmes += int(r["alarme"])
            total += 1
    ic_sup = beta.ppf(0.975, alarmes + 1, total - alarmes) if alarmes < total else 1.0
    assert alarmes / total <= 0.10 and ic_sup < 0.20, f"falso alarme {alarmes}/{total}, IC95 sup {ic_sup:.3f}"


def test_deriva_pequena_e_detectada_na_maioria_das_calibracoes():
    detect = 0
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        X_ref, l_ref = _processo(rng, 5, 12)
        X_nov, l_nov = _processo(rng, 1, 20, delta=0.03)
        pca, esc, dom, ref, _k = mon._calibrar(X_ref, l_ref, "auto", 0.05, False)
        res = mon._julgar(pca, esc, dom, X_nov)
        detect += int(mon._por_lote(l_nov, res["dentro_dominio"], ref, 0.05)[0]["alarme"])
    assert detect >= 16, f"poder {detect}/20"


def test_lote_pequeno_nao_finge_testar():
    rng = np.random.default_rng(3)
    X_ref, l_ref = _processo(rng, 5, 12)
    X_nov, l_nov = _processo(rng, 1, 8, delta=0.5)
    pca, esc, dom, ref, _k = mon._calibrar(X_ref, l_ref, "auto", 0.05, False)
    r = mon._por_lote(l_nov, mon._julgar(pca, esc, dom, X_nov)["dentro_dominio"], ref, 0.05)[0]
    assert r["testado"] is False and r["alarme"] is False and "abaixo do minimo" in r["mensagem"]




