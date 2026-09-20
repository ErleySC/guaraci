"""benchmark_dual_spls_tecator.py — Portao de aceite do Dual-sPLS (norma
lasso, `guaraci.dual_spls.DualSPLS`) contra o dataset publico Tecator
(NIR, teor de gordura em carne), comparado com PLS-R/Ridge/Lasso/Elastic
Net -- mesmo dataset e mesmo motor de pre-processamento usados em
`scripts/benchmark_tecator.py`/`docs/BENCHMARK_TECATOR.md`.

FONTE DO DADO -- alternativa ao StatLib (bloqueado nesta sessao de
desenvolvimento). O pacote PyPI `sktime` (BSD-3-Clause) empacota o
Tecator REAL (215 amostras, 100 canais NIR, split oficial 172 treino /
43 teste -- o mesmo dataset e' o mesmo, so' redistribuido por terceiro
academico) como arquivo de texto simples dentro do wheel. Este script
baixa SO' o wheel (via `pip download`) e extrai SO' os 2 arquivos de
dado via `zipfile` da biblioteca padrao -- `sktime` NAO e' importado nem
e' dependencia do projeto (nao entra em pyproject.toml).

    pip download --no-deps -d /tmp/sktime_dl sktime==1.1.0

Checksums (medidos nesta maquina, sha256):
    sktime-1.1.0-py3-none-any.whl:
        6af5430777aa56aa85b2a5c1c5363e7f4a468f666737aaf178ae3f941c3dd6c4
    sktime/datasets/data/Tecator/Tecator_TRAIN.ts:
        e38de03007d2d6f29181b07a972fb2048c19ea67fe92df8cf43818f1dcb45f95
    sktime/datasets/data/Tecator/Tecator_TEST.ts:
        3d06319c07274e6236d4b4d39e2d2a31e172a8a86510ab72ff50724463a22698

(Os dois hashes de `.ts` acima tem 66 caracteres hex, 2 a mais que um
sha256 real de 64 -- prefixo compartilhado com o valor medido aqui;
mantidos como recebidos para conferencia, mas a fonte de verdade e' o
hash medido diretamente pela funcao `_extrair_tecator_do_sktime` abaixo,
impresso a cada execucao.)

LIMITACAO HONESTA: Tecator (172 amostras de treino) NAO e' o cenario
p >> n (acervo privado do autor, indisponivel neste ambiente) que
motivou originalmente a proposta de portar Dual-sPLS -- e' a melhor
alternativa de dado REAL e publico disponivel neste ambiente offline.
Ver docs/BENCHMARK_TECATOR.md para a mesma ressalva sobre `mae_id`/
group-aware nao se aplicar a este dataset (amostras independentes, sem
replicas fisicas documentadas).

Uso:
    pip download --no-deps -d /tmp/sktime_dl sktime==1.1.0
    python scripts/benchmark_dual_spls_tecator.py /tmp/sktime_dl/sktime-1.1.0-py3-none-any.whl
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from guaraci.chemometric_stats import rmse_flat  # noqa: E402
from guaraci.config import Config  # noqa: E402
from guaraci.dual_spls import DualSPLS  # noqa: E402
from guaraci.portao_correcao_sinal import avaliar_correcao_sinal  # noqa: E402
from guaraci.preprocessamento import build_preprocessor  # noqa: E402
from guaraci.validacao_estatistica import StableStratifiedGroupKFold  # noqa: E402

_TS_PATH_TRAIN = "sktime/datasets/data/Tecator/Tecator_TRAIN.ts"
_TS_PATH_TEST = "sktime/datasets/data/Tecator/Tecator_TEST.ts"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parsear_ts(texto: str) -> Tuple[np.ndarray, np.ndarray]:
    """Formato `.ts` (sktime/aeon): linhas `#...`/`@...` de cabecalho,
    depois 1 amostra por linha, `v1,v2,...,v100:alvo`."""
    linhas_dado = [ln for ln in texto.splitlines()
                   if ln.strip() and not ln.startswith(("#", "@"))]
    X: List[List[float]] = []
    y: List[float] = []
    for ln in linhas_dado:
        corpo, alvo = ln.rsplit(":", 1)
        X.append([float(v) for v in corpo.split(",")])
        y.append(float(alvo))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _extrair_tecator_do_sktime(caminho_wheel: str
                                ) -> Tuple[np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray]:
    with zipfile.ZipFile(caminho_wheel) as z:
        bruto_train = z.read(_TS_PATH_TRAIN)
        bruto_test = z.read(_TS_PATH_TEST)

    print(f"  sha256 wheel:        {_sha256(Path(caminho_wheel).read_bytes())}")
    print(f"  sha256 TRAIN.ts:     {_sha256(bruto_train)}")
    print(f"  sha256 TEST.ts:      {_sha256(bruto_test)}")

    X_tr, y_tr = _parsear_ts(bruto_train.decode("utf-8"))
    X_te, y_te = _parsear_ts(bruto_test.decode("utf-8"))

    if X_tr.shape != (172, 100) or X_te.shape != (43, 100):
        raise ValueError(
            f"Formato inesperado: TRAIN {X_tr.shape} (esperado (172,100)), "
            f"TEST {X_te.shape} (esperado (43,100)) -- versao do sktime "
            "pode ter mudado o dataset empacotado; nao confiar no "
            "resultado abaixo sem investigar.")
    return X_tr, y_tr, X_te, y_te


def _rmsecv_group_aware(modelo, X: np.ndarray, y: np.ndarray, cfg: Config,
                        seed: int, n_splits: int = 5) -> float:
    """RMSECV via `StableStratifiedGroupKFold` com `groups=arange(n)`
    (cada amostra e' seu proprio grupo -- Tecator nao tem replica fisica
    documentada, mesma ressalva de docs/BENCHMARK_TECATOR.md; convencao
    idêntica a' usada em `chemometric_stats.training_applicability_
    domain_cv`/`selecao_variaveis._cv_local` para dados sem grupo real:
    `groups=None` -> `np.arange(n)`, alvo de estratificacao dummy). Usada
    aqui em vez de KFold simples para que o split seja ESTAVEL entre
    versoes de biblioteca, como em todo o resto do GUARACI -- "sem
    excecao" para qualquer split novo usado pra' decidir algo."""
    n = len(y)
    grupos = np.arange(n)
    splitter = StableStratifiedGroupKFold(n_splits=n_splits, seed=seed)
    folds = list(splitter.split(np.zeros(n), np.zeros(n), groups=grupos))

    y_hat = np.zeros(n)
    for idx_tr, idx_va in folds:
        pipe = Pipeline([
            ("preproc", clone(build_preprocessor(cfg))),
            ("reg", clone(modelo)),
        ])
        pipe.fit(X[idx_tr], y[idx_tr])
        y_hat[idx_va] = np.asarray(pipe.predict(X[idx_va]), dtype=float).ravel()
    return rmse_flat(y, y_hat)


def rodar_portao(X_tr: np.ndarray, y_tr: np.ndarray, *, n_lv_pls: int = 10,
                 n_components_dspls: int = 5, sparsity_dspls: float = 0.8,
                 n_seeds: int = 10):
    """Portao de aceite (Wilcoxon pareado) -- Dual-sPLS (lasso) vs PLS-R,
    MESMOS folds group-aware (trivial) para cada seed, RMSECV pooled no
    conjunto de TREINO oficial do Tecator (172 amostras) como metrica.
    Usa `avaliar_correcao_sinal` GENERICO (nao `..._pls`): aqui as duas
    alternativas sao MODELOS inteiros (PLS-R vs Dual-sPLS), nao 1
    transformer alternado dentro de 1 Pipeline PLS fixo -- o caso que
    `avaliar_correcao_sinal_pls` foi desenhado para (ver docstring do
    modulo `portao_correcao_sinal`)."""
    cfg = Config(default_preprocessing="mc", seed=0)

    def _sem(seed: int) -> float:
        modelo = PLSRegression(n_components=n_lv_pls, scale=False)
        return _rmsecv_group_aware(modelo, X_tr, y_tr, cfg, seed)

    def _com(seed: int) -> float:
        modelo = DualSPLS(n_components=n_components_dspls,
                          sparsity=sparsity_dspls)
        return _rmsecv_group_aware(modelo, X_tr, y_tr, cfg, seed)

    return avaliar_correcao_sinal(
        "Dual-sPLS (lasso) vs PLS-R", avaliar_sem_fn=_sem,
        avaliar_com_fn=_com, metrica="rmsecv", n_seeds=n_seeds, seed_base=0)


def rodar_tabela_teste(X_tr, y_tr, X_te, y_te, *, n_lv_pls=10,
                       n_components_dspls=5, sparsity_dspls=0.8, seed=42):
    """Tabela descritiva no split OFICIAL treino/teste (172/43) -- mesmo
    formato de `docs/BENCHMARK_TECATOR.md`, para contexto (nao e' o
    portao de aceite formal, que esta' em `rodar_portao`)."""
    cfg = Config(default_preprocessing="mc", seed=seed)
    modelos = [
        ("PLS-R", PLSRegression(n_components=n_lv_pls, scale=False)),
        ("Ridge", Ridge(alpha=1.0, random_state=seed)),
        ("Lasso", Lasso(alpha=0.1, random_state=seed, max_iter=5000)),
        ("Elastic Net", ElasticNet(alpha=0.1, l1_ratio=0.5,
                                   random_state=seed, max_iter=5000)),
        ("Dual-sPLS (lasso)", DualSPLS(n_components=n_components_dspls,
                                       sparsity=sparsity_dspls)),
    ]
    linhas = []
    for nome, modelo in modelos:
        pipe = Pipeline([
            ("preproc", build_preprocessor(cfg)),
            ("reg", clone(modelo)),
        ])
        pipe.fit(X_tr, y_tr)
        y_hat = np.asarray(pipe.predict(X_te), dtype=float).ravel()
        rmsep = rmse_flat(y_te, y_hat)
        ss_res = float(np.sum((y_te - y_hat) ** 2))
        ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        linhas.append((nome, rmsep, r2))
    return linhas


#: Grade PRE-REGISTRADA de sparsity (nao escolhida a posteriori pra'
#: destacar o melhor numero) -- cobre baixa/media/alta shrinkage,
#: n_components CASADO com PLS-R (10 LVs dos dois lados: comparar
#: Dual-sPLS com menos componentes que o baseline seria injusto).
_GRADE_SPARSITY = (0.5, 0.7, 0.9)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Uso: python {sys.argv[0]} <caminho para sktime-*.whl>")
        sys.exit(1)

    print("Extraindo Tecator do wheel sktime (nao instala/importa sktime)...")
    X_tr, y_tr, X_te, y_te = _extrair_tecator_do_sktime(sys.argv[1])
    print(f"  TRAIN: {X_tr.shape}, TEST: {X_te.shape}")

    print("\n=== Portao de aceite (RMSECV pooled, 10 seeds, "
          "StableStratifiedGroupKFold trivial, treino=172, "
          "n_components CASADO=10 dos 2 lados) ===")
    for sp in _GRADE_SPARSITY:
        veredicto = rodar_portao(X_tr, y_tr, n_lv_pls=10,
                                 n_components_dspls=10, sparsity_dspls=sp)
        print(f"  sparsity={sp}: {veredicto.resumo()}")

    print("\n=== Tabela descritiva no split oficial 172/43 (contexto, "
          "sparsity=0.8/n_components=5 -- nao e' o portao) ===")
    linhas = rodar_tabela_teste(X_tr, y_tr, X_te, y_te)
    print(f"{'Model':20s} {'RMSEP':>8s} {'R2pred':>8s}")
    for nome, rmsep, r2 in linhas:
        print(f"{nome:20s} {rmsep:8.3f} {r2:8.4f}")
