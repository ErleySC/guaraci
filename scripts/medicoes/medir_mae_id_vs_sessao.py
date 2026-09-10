# -*- coding: utf-8 -*-
"""Mede o efeito de agrupar a CV/holdout por `mae_id` (produção atual) vs.
por `session_from_mae_id` (achado do Agente 3, rodada multiagente de
2026-09-10, Passo 200).

NÃO altera o pipeline de produção. Reusa as MESMAS funções que
`pipeline.executar()` usa (build_preprocessor, StableStratifiedGroupKFold,
classification_metrics, expandir_binario_um_quente) para que o número seja
uma medição fiel, não uma reimplementação aproximada.

Escopo (confirmado antes de medir, por instrução): os testes de validação
pública passam `group_by_mae_id=False` (ver tests/test_validacao_publica*.py,
grep confirmado) -- nenhum dos 14 datasets públicos usa agrupamento por
mae_id. O achado é exclusivo do dataset PRIVADO de óleos adulterados, onde
`mae_id` de amostra adulterada é um por nível de teor (dados_io.py:
session_from_mae_id).

Uso:
    python scripts/medicoes/medir_mae_id_vs_sessao.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import threadpoolctl
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

_RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ / "src"))

from guaraci.chemometric_stats import expandir_binario_um_quente, rmse_flat  # noqa: E402
from guaraci.config import Config  # noqa: E402
from guaraci.config_io import load_config  # noqa: E402
from guaraci.dados_io import load_data, session_from_mae_id  # noqa: E402
from guaraci.pipeline import classification_metrics, validate_input  # noqa: E402
from guaraci.preprocessamento import build_preprocessor  # noqa: E402
from guaraci.validacao_estatistica import StableStratifiedGroupKFold  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402


def _carregar_dados_reais() -> tuple:
    cfg = load_config(str(_RAIZ / "config.yaml"), base=Config())
    wavenumbers, X_raw, rotulos, conc, mae_id, _meta = load_data(cfg)
    X_raw = np.asarray(X_raw, dtype=float)
    rotulos = np.asarray(rotulos, dtype=str)
    mae_id = np.asarray(mae_id, dtype=str)
    X_raw, wavenumbers, rotulos, conc, mae_id, _rel = validate_input(
        X_raw, wavenumbers, rotulos, conc, mae_id)
    mask_wn = (wavenumbers >= cfg.wn_min) & (wavenumbers <= cfg.wn_max)
    X_raw = X_raw[:, mask_wn]
    return cfg, X_raw, rotulos, conc, mae_id


class GrupoInviavel(RuntimeError):
    """Levantado quando a chave de agrupamento não sustenta uma CV
    estratificada: alguma classe existe em grupos de menos, ou o número de
    grupos é pequeno e desbalanceado demais e produz fold(s) vazio(s)."""


def _checar_viabilidade_grupos(rotulos: np.ndarray, grupos: np.ndarray,
                                n_splits: int, rotulo_grupo: str) -> None:
    """Verifica ANTES de rodar a CV se a chave de agrupamento sustenta uma
    partição estratificada em `n_splits` folds sem fold vazio nem classe
    confinada a 1 grupo. Achado real (rodada multiagente 2026-09-10, Passo
    201): com `groups=session_from_mae_id` neste dataset, 8/13 espécies têm
    exatamente 1 sessão -- StableStratifiedGroupKFold produz fold(s) VAZIO(S)
    (medido: partição 1432/144/96/0/0 amostras em 5 folds), e o
    `savgol_filter` do pré-processador estoura com `ValueError: Internal
    LAPACK errors` ao tentar transformar um lote vazio. Preferimos essa
    checagem explícita a deixar o erro de baixo nível se propagar sem
    explicação."""
    n_grupos_por_classe = {
        c: len(np.unique(grupos[rotulos == c])) for c in np.unique(rotulos)
    }
    classes_1_grupo = sorted(c for c, n in n_grupos_por_classe.items() if n < 2)
    if classes_1_grupo:
        raise GrupoInviavel(
            f"[{rotulo_grupo}] {len(classes_1_grupo)}/{len(n_grupos_por_classe)} "
            f"classes têm menos de 2 grupos únicos, então não podem aparecer "
            f"em treino E validação ao mesmo tempo: {classes_1_grupo}. "
            f"CV estratificada por grupo não é executável com esta chave "
            f"neste dataset -- não é um bug de código, é o delineamento "
            f"real da coleta (ver docs/RELATORIO_MULTIAGENTE_2026-09-10.md "
            f"e ~/.guaraci_local/CLAUDE.md P12).")


def _rodar_cv_classificacao(cfg: Config, X_raw: np.ndarray, rotulos: np.ndarray,
                             grupos: np.ndarray, rotulo_grupo: str) -> dict:
    """Reproduz EXATAMENTE o laço de seleção de LV + predição de CV de
    `pipeline.executar()` (l.1753-1809), variando só `groups`."""
    lb = LabelBinarizer()
    Y_bin = np.asarray(lb.fit_transform(rotulos), dtype=float)
    Y_bin = expandir_binario_um_quente(Y_bin)
    y_int = np.argmax(Y_bin, axis=1)
    classes_unicas = np.unique(rotulos)

    n_grupos_unicos = len(np.unique(grupos))
    n_splits = min(cfg.n_splits_cv, n_grupos_unicos)
    _checar_viabilidade_grupos(rotulos, grupos, max(n_splits, 2), rotulo_grupo)
    cv = StableStratifiedGroupKFold(n_splits=max(n_splits, 2), seed=cfg.seed)
    cv_indices = list(cv.split(X_raw, y_int, groups=grupos))

    def fabrica_pipeline(n_lv: int) -> Pipeline:
        return Pipeline([
            ("preproc", build_preprocessor(cfg)),
            ("pls", PLSRegression(n_components=n_lv, scale=False)),
        ])

    erros_rmsecv = []
    preds_por_lv = {}
    for n in range(1, cfg.max_lvs + 1):
        y_hat = np.zeros_like(Y_bin)
        contador = np.zeros(len(Y_bin), dtype=int)
        for tr, va in cv_indices:
            pipe = fabrica_pipeline(n)
            pipe.fit(X_raw[tr], Y_bin[tr])
            y_hat[va] += pipe.predict(X_raw[va])
            contador[va] += 1
        contador[contador == 0] = 1
        y_hat = y_hat / contador[:, None]
        erros_rmsecv.append(rmse_flat(Y_bin, y_hat))
        preds_por_lv[n] = y_hat

    rmsecv_arr = np.array(erros_rmsecv)
    rmsecv_min = float(rmsecv_arr.min())
    tol_wold = rmsecv_min * 1.02
    candidatos_wold = np.where(rmsecv_arr <= tol_wold)[0]
    n_opt = int(candidatos_wold[0]) + 1

    Y_cv = preds_por_lv[n_opt]
    pred_lab_int = np.argmax(Y_cv, axis=1)
    m = classification_metrics(y_int, pred_lab_int, np.arange(len(classes_unicas)))
    return {
        "rotulo_grupo": rotulo_grupo,
        "n_grupos": n_grupos_unicos,
        "n_splits": n_splits,
        "n_opt_lvs": n_opt,
        "rmsecv": rmsecv_min,
        **m,
    }


def _rodar_holdout_classificacao(cfg: Config, X_raw: np.ndarray, rotulos: np.ndarray,
                                  grupos: np.ndarray, rotulo_grupo: str) -> dict:
    """Reproduz o split de holdout de `pipeline.executar()` (l.1609-1620) e
    treina/avalia com o n_opt de LVs escolhido pela CV da MESMA chave de
    agrupamento."""
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.frac_holdout,
                             random_state=cfg.seed_holdout)
    tr_idx, ho_idx = next(gss.split(X_raw, rotulos, groups=grupos))
    X_tr, rot_tr, grp_tr = X_raw[tr_idx], rotulos[tr_idx], grupos[tr_idx]
    X_ho, rot_ho = X_raw[ho_idx], rotulos[ho_idx]

    cv_treino = _rodar_cv_classificacao(cfg, X_tr, rot_tr, grp_tr,
                                         rotulo_grupo + " (CV no treino p/ LVs)")
    n_opt = cv_treino["n_opt_lvs"]

    lb = LabelBinarizer()
    Y_bin_tr = np.asarray(lb.fit_transform(rot_tr), dtype=float)
    Y_bin_tr = expandir_binario_um_quente(Y_bin_tr)

    pipe = Pipeline([
        ("preproc", build_preprocessor(cfg)),
        ("pls", PLSRegression(n_components=n_opt, scale=False)),
    ])
    pipe.fit(X_tr, Y_bin_tr)
    y_hat_ho = pipe.predict(X_ho)
    pred_lab_ho = lb.classes_[np.argmax(y_hat_ho, axis=1)]
    y_true_int = np.array([np.where(lb.classes_ == r)[0][0] for r in rot_ho])
    y_pred_int = np.array([np.where(lb.classes_ == r)[0][0] for r in pred_lab_ho])
    m = classification_metrics(y_true_int, y_pred_int, np.arange(len(lb.classes_)))
    return {
        "rotulo_grupo": rotulo_grupo,
        "n_treino": len(tr_idx),
        "n_holdout": len(ho_idx),
        "n_opt_lvs_usado": n_opt,
        **m,
    }


def main() -> None:
    print("[1/4] Carregando dataset privado (config.yaml) ...")
    cfg, X_raw, rotulos, conc, mae_id = _carregar_dados_reais()
    # threadpool_limits(1): mesmo padrao de validacao_estatistica.py --
    # evita oversubscription do BLAS interno (OpenBLAS multithread) durante
    # os ~9600 ajustes de PLS desta medicao (2 groupings x 40 LVs x folds,
    # mais o holdout). Sem isto, um LAPACK "info=-4" apareceu de forma
    # nao-deterministica dentro do savgol_filter em 2026-09-10.
    sessao = np.array([session_from_mae_id(m) for m in mae_id])

    n_mae_id = len(np.unique(mae_id))
    n_sessao = len(np.unique(sessao))
    print(f"    {len(rotulos)} espectros, {len(np.unique(rotulos))} classes, "
          f"{n_mae_id} mae_id unicos, {n_sessao} sessoes unicas "
          f"(razao {n_mae_id / n_sessao:.2f}x)")

    n_sessoes_por_classe = {
        c: len(np.unique(sessao[rotulos == c])) for c in np.unique(rotulos)
    }
    classes_1_sessao = sorted(c for c, n in n_sessoes_por_classe.items() if n < 2)
    print(f"    classes com 1 unica sessao (nunca poderiam aparecer em "
          f"treino E validacao ao mesmo tempo se agrupadas por sessao): "
          f"{len(classes_1_sessao)}/{len(n_sessoes_por_classe)} -> "
          f"{classes_1_sessao}")

    with threadpoolctl.threadpool_limits(1):
        print("\n[2/4] CV de classificacao (N1) agrupando por mae_id (producao) ...")
        cv_mae = _rodar_cv_classificacao(cfg, X_raw, rotulos, mae_id, "mae_id")
        print(f"    {cv_mae}")

        print("\n[3/4] CV de classificacao (N1) agrupando por sessao (proposto) ...")
        try:
            cv_ses = _rodar_cv_classificacao(cfg, X_raw, rotulos, sessao, "sessao")
            print(f"    {cv_ses}")
        except GrupoInviavel as e:
            print(f"    NAO EXECUTAVEL: {e}")
            print("\n===================== RESULTADO =====================")
            print(f"n_mae_id={n_mae_id}  n_sessoes={n_sessao}")
            print(f"classes com 1 unica sessao: {len(classes_1_sessao)}/"
                  f"{len(n_sessoes_por_classe)} -> {classes_1_sessao}")
            print("CV/holdout classificacao (N1) agrupados por SESSAO: "
                  "ESTRUTURALMENTE NAO EXECUTAVEL neste dataset -- nao e' "
                  "uma diferenca numerica para medir, e' impossibilidade de "
                  "particao (ver docstring de GrupoInviavel acima).")
            print(f"Para referencia, CV atual (mae_id): "
                  f"balanced_accuracy={cv_mae['balanced_accuracy']:.4f}  "
                  f"n_opt_lvs={cv_mae['n_opt_lvs']}")
            return

        print("\n[4/4] Holdout externo, cada um com sua propria chave de grupo ...")
        ho_mae = _rodar_holdout_classificacao(cfg, X_raw, rotulos, mae_id, "mae_id")
        print(f"    {ho_mae}")
        ho_ses = _rodar_holdout_classificacao(cfg, X_raw, rotulos, sessao, "sessao")
        print(f"    {ho_ses}")

    print("\n===================== RESULTADO =====================")
    print(f"n_mae_id={n_mae_id}  n_sessoes={n_sessao}")
    print(f"CV balanced_accuracy  mae_id={cv_mae['balanced_accuracy']:.4f}  "
          f"sessao={cv_ses['balanced_accuracy']:.4f}  "
          f"delta={cv_ses['balanced_accuracy'] - cv_mae['balanced_accuracy']:+.4f}")
    print(f"CV n_opt_lvs          mae_id={cv_mae['n_opt_lvs']}  "
          f"sessao={cv_ses['n_opt_lvs']}")
    print(f"Holdout balanced_acc  mae_id={ho_mae['balanced_accuracy']:.4f}  "
          f"sessao={ho_ses['balanced_accuracy']:.4f}  "
          f"delta={ho_ses['balanced_accuracy'] - ho_mae['balanced_accuracy']:+.4f}")
    print(f"Holdout n (treino/ho) mae_id={ho_mae['n_treino']}/{ho_mae['n_holdout']}  "
          f"sessao={ho_ses['n_treino']}/{ho_ses['n_holdout']}")


if __name__ == "__main__":
    main()
