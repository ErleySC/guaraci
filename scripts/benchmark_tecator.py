"""benchmark_tecator.py — Validacao externa do motor de pre-processamento +
regressao PLS do GUARACI contra o dataset publico Tecator (NIR, teor de
gordura em carne), fora do dataset proprio do autor (auditoria de
2026-07-12, item "sem benchmark contra dataset publico externo";
docs/VALIDATION.md, docs/BENCHMARK_TECATOR.md).

Fonte dos dados: StatLib (CMU), http://lib.stat.cmu.edu/datasets/tecator
Dominio publico -- nota de permissao original preservada abaixo.
Referencia primaria: THODBERG, H. H. A review of Bayesian neural networks
with an application to near infrared spectroscopy. IEEE Transactions on
Neural Networks, v. 7, n. 1, p. 56-72, 1996. doi:10.1109/72.478392

Uso:
    python scripts/benchmark_tecator.py

Nao baixa/comita o dataset no repositorio (evita redistribuir 240KB de
dado de terceiro); baixa da fonte original a cada execucao.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from guaraci.chemometric_stats import rmse_flat  # noqa: E402
from guaraci.config import Config  # noqa: E402
from guaraci.preprocessamento import construir_preprocessador  # noqa: E402

TECATOR_URL = "http://lib.stat.cmu.edu/datasets/tecator"
N_SAMPLES = 240
N_WAVELENGTHS = 100
LINES_PER_SAMPLE = 25
DATA_START_MARKER = "extrapolation_examples=25"


def _baixar_tecator_raw() -> str:
    with urllib.request.urlopen(TECATOR_URL, timeout=30) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _parsear_tecator(texto: str) -> pd.DataFrame:
    """Extrai (100 absorbancias, umidade, gordura, proteina) por amostra.

    Formato do arquivo (ver header do proprio dataset): 25 linhas por
    amostra -- 20 linhas com 100 absorbancias (5 por linha), 4 linhas com
    22 componentes principais (nao usados aqui -- GUARACI usa o espectro
    bruto + seu proprio pre-processamento), 1 linha final com os ultimos
    2 componentes + umidade/gordura/proteina.
    """
    # rfind: o marcador tambem aparece 1x na prosa descritiva do arquivo,
    # antes do cabecalho real que precede os dados numericos.
    marcador = texto.rfind(DATA_START_MARKER)
    if marcador < 0:
        raise ValueError("Marcador de inicio dos dados nao encontrado no "
                          "arquivo Tecator -- formato da fonte pode ter mudado.")
    linhas = [ln.strip() for ln in texto[marcador:].splitlines() if ln.strip()]
    linhas = linhas[1:]  # pula a propria linha do marcador
    if len(linhas) % LINES_PER_SAMPLE != 0:
        raise ValueError(f"Numero de linhas de dado ({len(linhas)}) nao e' "
                          f"multiplo de {LINES_PER_SAMPLE} (linhas/amostra) "
                          "-- arquivo truncado ou formato mudou.")
    n_amostras = len(linhas) // LINES_PER_SAMPLE

    wn = np.linspace(850, 1050, N_WAVELENGTHS)
    linhas_espectro, gorduras, splits = [], [], []
    for i in range(n_amostras):
        bloco = linhas[i * LINES_PER_SAMPLE:(i + 1) * LINES_PER_SAMPLE]
        absorb = []
        for ln in bloco[:20]:
            absorb.extend(float(x) for x in ln.split())
        if len(absorb) != N_WAVELENGTHS:
            raise ValueError(f"Amostra {i}: esperado {N_WAVELENGTHS} "
                              f"absorbancias, achei {len(absorb)}.")
        _moisture, fat, _protein = (float(x) for x in bloco[24].split()[-3:])
        linhas_espectro.append(absorb)
        gorduras.append(fat)
        if i < 172:
            splits.append("train")
        elif i < 215:
            splits.append("test")
        else:
            splits.append("extrapolation")

    df = pd.DataFrame(linhas_espectro, columns=[f"{w:.2f}" for w in wn])
    df["teor_gordura"] = gorduras
    df["split_original"] = splits
    return df


def _rmsecv_por_lv(X: np.ndarray, y: np.ndarray, cfg: Config,
                    lv_max: int, n_splits: int = 5) -> list:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    erros = []
    for n in range(1, lv_max + 1):
        pipe = Pipeline([
            ("preproc", construir_preprocessador(cfg)),
            ("pls", PLSRegression(n_components=n, scale=False)),
        ])
        try:
            y_hat_cv = cross_val_predict(pipe, X, y.reshape(-1, 1), cv=cv)
            erros.append(rmse_flat(y, y_hat_cv))
        except (ValueError, np.linalg.LinAlgError):
            erros.append(float("inf"))
    return erros


def rodar_benchmark(presets=("msc_sg_mc", "snv_sg_mc", "mc", "autoscaling"),
                     max_lvs: int = 20, seed: int = 42) -> dict:
    """Roda o motor real de pre-processamento+PLS do GUARACI (nao uma
    reimplementacao) no split oficial do Tecator (172 treino / 43 teste),
    selecionando LVs por CV no treino -- mesma metodologia usada
    internamente por `pipeline.pls_regressao_por_especie`."""
    df = _parsear_tecator(_baixar_tecator_raw())
    if len(df) != N_SAMPLES:
        raise ValueError(f"Esperado {N_SAMPLES} amostras do Tecator (fonte "
                          f"conhecida e estavel), achei {len(df)} -- "
                          "verificar se a fonte mudou antes de confiar no "
                          "resultado do benchmark.")
    wn_cols = [c for c in df.columns if c not in ("teor_gordura", "split_original")]
    X = df[wn_cols].values.astype(float)
    y = df["teor_gordura"].values.astype(float)
    treino, teste = df["split_original"] == "train", df["split_original"] == "test"
    X_tr, y_tr, X_te, y_te = X[treino], y[treino], X[teste], y[teste]

    resultados = {}
    for preset in presets:
        cfg = Config(preprocessamento_padrao=preset, sg_window=15,
                     sg_polyorder=2, sg_deriv=1, max_lvs=max_lvs, seed=seed)
        lv_max = min(cfg.max_lvs, X_tr.shape[0] // 5)
        erros_cv = _rmsecv_por_lv(X_tr, y_tr, cfg, lv_max)
        n_opt = int(np.argmin(erros_cv)) + 1

        pipe_final = Pipeline([
            ("preproc", construir_preprocessador(cfg)),
            ("pls", PLSRegression(n_components=n_opt, scale=False)),
        ]).fit(X_tr, y_tr.reshape(-1, 1))
        y_pred = np.asarray(pipe_final.predict(X_te)).flatten()
        rmsep = rmse_flat(y_te, y_pred)
        ss_res = float(np.sum((y_te - y_pred) ** 2))
        ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
        resultados[preset] = {
            "n_lv": n_opt,
            "rmsecv": float(erros_cv[n_opt - 1]),
            "rmsep": float(rmsep),
            "r2_pred": 1.0 - ss_res / ss_tot,
        }
    return resultados


if __name__ == "__main__":
    res = rodar_benchmark()
    print(f"{'preset':14s} {'n_lv':>5s} {'RMSECV':>8s} {'RMSEP':>8s} {'R2pred':>8s}")
    for preset, m in res.items():
        print(f"{preset:14s} {m['n_lv']:5d} {m['rmsecv']:8.3f} "
              f"{m['rmsep']:8.3f} {m['r2_pred']:8.4f}")
    out = Path(__file__).resolve().parent.parent / "resultados_tecator.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nSalvo em {out}")
