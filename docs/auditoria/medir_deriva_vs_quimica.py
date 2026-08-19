# -*- coding: utf-8 -*-
"""O sinal de teor que o PLS-R aprende e' QUIMICA ou DERIVA DA SESSAO?

Contexto (medido em `medir_ordem_leitura.py`): dentro de cada bloco
especie x adulterante x sessao, a ordem de leitura e' monotonica com o
teor (rho medio +0,997 em 47/47 blocos). Logo, dentro de UMA sessao,
"tempo desde o inicio da sessao" e "teor" sao a mesma variavel: qualquer
deriva instrumental (aquecimento da fonte, purga de CO2/H2O, sujeira
acumulando na janela) entra no modelo como se fosse sinal de adulteracao,
e a validacao cruzada dentro da sessao NAO consegue separar os dois.

Teste que separa: SEIS especies tem o mesmo adulterante (soja) lido em
DUAS sessoes distintas, com semanas de intervalo. A deriva de uma sessao
nao se repete na outra; a quimica sim. Entao:

  - treina o PLS-R na sessao A, prediz a sessao B (e vice-versa);
  - compara com o R2 obtido DENTRO da sessao (CV group-aware).

Se o R2 desaba na transferencia entre sessoes, o que o modelo aprendeu e'
majoritariamente deriva. Se sobrevive, e' quimica.

Uso:
    python docs/auditoria/medir_deriva_vs_quimica.py <raiz_dos_dx>
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from guaraci.dados_io import (extrair_title_do_dx, parse_dx,  # noqa: E402
                              parse_title)
from guaraci.preprocessamento import MSC, SavGol  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

WN_MIN, WN_MAX = 4000.0, 10000.0
N_GRADE = 1200

_RE_AUDIT = re.compile(
    r"\(\s*1\s*,\s*<(?P<ts>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})")


def _ts(caminho):
    with open(caminho, "r", encoding="latin-1", errors="replace") as f:
        for linha in f:
            m = _RE_AUDIT.search(linha)
            if m:
                return datetime.strptime(m.group("ts"), "%Y/%m/%d %H:%M:%S")
            if linha.startswith("##XYDATA"):
                break
    return None


def _preproc():
    """Mesma receita do pipeline em producao: cfg.preprocessamento_padrao
    = 'msc_sg_mc' com sg_window=25, polyorder=2, deriv=1 (config.py:125-131).
    MSC e' stateful (referencia = media do TREINO), por isso vive dentro do
    Pipeline do sklearn e nunca ve o conjunto de teste."""
    return Pipeline([("msc", MSC()),
                     ("sg", SavGol(25, 2, 1)),
                     ("mc", StandardScaler(with_std=False))])


def _ajustar_prever(Xtr, ytr, Xte, n_lv):
    pipe = Pipeline([("pre", _preproc()),
                     ("pls", PLSRegression(n_components=n_lv, scale=False))])
    pipe.fit(Xtr, ytr.reshape(-1, 1))
    return np.asarray(pipe.predict(Xte), dtype=float).ravel()


def _r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def carregar(raiz):
    grade = np.linspace(WN_MIN, WN_MAX, N_GRADE)
    regs = []
    for p in sorted(Path(raiz).rglob("*.dx")):
        titulo = extrair_title_do_dx(str(p))
        if not titulo:
            continue
        info = parse_title(titulo)
        if not info:
            continue
        ts = _ts(str(p))
        if ts is None:
            continue
        try:
            wn, y = parse_dx(str(p))
        except Exception:                                  # noqa: BLE001
            continue
        if wn.size < 100:
            continue
        ordem = np.argsort(wn)
        yi = np.interp(grade, wn[ordem], y[ordem])
        if not np.isfinite(yi).all():
            continue
        regs.append({
            "x": yi, "especie": info["especie"],
            "adulterante": info["adulterante_nome"],
            "teor": 0.0 if info["puro"] else float(info["teor"]),
            "puro": info["puro"], "mae_id": info["mae_id"],
            "dia": ts.date(), "ts": ts,
        })
    return grade, regs


def main(raiz):
    _grade, regs = carregar(raiz)
    print("espectros carregados: %d" % len(regs))

    # blocos: (especie, adulterante) -> {dia: [regs]}
    blocos = defaultdict(lambda: defaultdict(list))
    for r in regs:
        if r["adulterante"]:
            blocos[(r["especie"], r["adulterante"])][r["dia"]].append(r)

    alvos = [k for k, v in blocos.items() if len(v) >= 2]
    print("pares (especie, adulterante) com >= 2 sessoes: %d" % len(alvos))
    if not alvos:
        print("Sem par de sessoes -- teste nao aplicavel neste dataset.")
        return 1

    print("\n%-18s %-8s %-11s %-11s %5s %5s %8s %8s %8s %8s"
          % ("especie", "adult", "sessao_tr", "sessao_te", "n_tr", "n_te",
             "R2_int", "R2_ext", "RMSE_in", "RMSE_ex"))

    linhas = []
    for (esp, adu) in sorted(alvos):
        dias = sorted(blocos[(esp, adu)])
        # usa as duas maiores sessoes
        dias = sorted(dias, key=lambda d: -len(blocos[(esp, adu)][d]))[:2]
        for i in (0, 1):
            d_tr, d_te = dias[i], dias[1 - i]
            tr = blocos[(esp, adu)][d_tr]
            te = blocos[(esp, adu)][d_te]
            # adiciona os PUROS da mesma especie lidos na mesma sessao
            # (ancora teor=0 da calibracao), como faz o pipeline
            tr = tr + [r for r in regs
                       if r["puro"] and r["especie"] == esp
                       and r["dia"] == d_tr]
            te = te + [r for r in regs
                       if r["puro"] and r["especie"] == esp
                       and r["dia"] == d_te]
            Xtr = np.array([r["x"] for r in tr])
            ytr = np.array([r["teor"] for r in tr])
            gtr = np.array([r["mae_id"] for r in tr])
            Xte = np.array([r["x"] for r in te])
            yte = np.array([r["teor"] for r in te])
            if len(np.unique(gtr)) < 4 or np.std(ytr) < 1e-9:
                continue
            if len(yte) < 3 or np.std(yte) < 1e-9:
                continue
            # mesma regra de LV do pipeline (r2cv_especie_adulterante,
            # pipeline.py:836): min(max_lvs, n//5, n_niveis-1)
            n_niveis = int(len(np.unique(ytr)))
            n_lv = int(max(1, min(40, Xtr.shape[0] // 5,
                                  n_niveis - 1)))

            # (a) DENTRO da sessao: CV group-aware, como o pipeline faz
            cv = GroupKFold(n_splits=min(5, len(np.unique(gtr))))
            pipe = Pipeline([("pre", _preproc()),
                             ("pls", PLSRegression(n_components=n_lv,
                                                   scale=False))])
            yhat_in = np.asarray(
                cross_val_predict(pipe, Xtr, ytr.reshape(-1, 1),
                                  cv=cv, groups=gtr), dtype=float).ravel()
            r2_in, rm_in = _r2(ytr, yhat_in), _rmse(ytr, yhat_in)

            # (b) ENTRE sessoes
            yhat_ex = _ajustar_prever(Xtr, ytr, Xte, n_lv)
            r2_ex, rm_ex = _r2(yte, yhat_ex), _rmse(yte, yhat_ex)

            print("%-18s %-8s %-11s %-11s %5d %5d %8.3f %8.3f %8.2f %8.2f"
                  % (esp[:18], adu[:8], str(d_tr), str(d_te), len(ytr),
                     len(yte), r2_in, r2_ex, rm_in, rm_ex))
            linhas.append((r2_in, r2_ex, rm_in, rm_ex))

    if linhas:
        a = np.array(linhas, dtype=float)
        print("\n== RESUMO (%d transferencias) ==" % len(a))
        print("   R2 DENTRO da sessao (CV group-aware): mediana %+.3f  "
              "media %+.3f" % (np.median(a[:, 0]), np.mean(a[:, 0])))
        print("   R2 ENTRE sessoes (mesma especie+adulterante): "
              "mediana %+.3f  media %+.3f" % (np.median(a[:, 1]),
                                              np.mean(a[:, 1])))
        print("   RMSE dentro: mediana %.2f   |   RMSE entre: mediana %.2f "
              "(razao %.1fx)"
              % (np.median(a[:, 2]), np.median(a[:, 3]),
                 np.median(a[:, 3]) / max(np.median(a[:, 2]), 1e-9)))
        pior = int(np.sum(a[:, 1] < 0))
        print("   transferencias com R2 < 0 (pior que prever a media): "
              "%d/%d" % (pior, len(a)))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
