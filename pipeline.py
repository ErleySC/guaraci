# =============================================================================
# CHANGELOG
# v08  base: Sprints 1-3, GroupKFold mae_id, spectral truncation
# v10  2026-05-28  max_lvs=30; ddsimca_n_components=7;
#                    C2: comparar_pipelines uses max_lv=cfg.max_lvs (was min(8,..))
# v11 — 2026-05-28 — C3: HCA dendrogram (Ward); C4: DD-SIMCA one-class
#                    (trains only on pure samples, sens/spec); C5: N3 PLS reg GroupKFold
#                    by mae_id; C6: T2 outliers per class in model summary
# v12 — 2026-05-28 — M1: pure(*)/adulterated(o) markers in score plots;
#                    M2: sens/spec in DD-SIMCA acceptance plot titles
# v13 — 2026-05-28 — M3: chemical annotation of VIP bands; M4: accuracy per
#                    class in resumo_modelo.txt
# v14 — 2026-05-28 — FINDING: MSC->SG+MC = 0.923 bal.acc on full dataset
#                    (1807) vs autoscaling 0.472 (AUTO advantage was
#                    artifact of 80% subset). Changes:
#                    (1) preset "msc_sg_mc" in construir_preprocessador;
#                    (2) preprocessamento_padrao default = "msc_sg_mc";
#                    (3) frac_holdout default = 0.20;
#                    (4) gerar_nome_saida case "msc_sg_mc" -> "MSC-SGd-MC";
#                    (5) M1: stars -> circle with black edge (avoids cluttering 1807pts);
#                    (6) DD-SIMCA reverts to training on ALL samples (3 pure/class
#                        makes one-class infeasible; requires >=15 pure/class)
# v15 — 2026-05-28 — (1) holdout_preserva_puros=True: pure samples always in training
#                        (fixes "pure=0" in 4 classes after holdout);
#                    (2) automatic warning "LVs at ceiling" (console + summary);
#                    (3) DD-SIMCA acceptance plot in LOG-LOG scale
#                        (fixes data squeezed in corner; Pomerantsev standard)
# v16 — 2026-05-28 — Organization/visualization:
#                    (1) salvar() accepts subfolder; (2) fig3 Hotelling T2 in
#                    log scale (Y) and T2vsQ in log-log (centers the cloud);
#                    (3) score_contribution split into 2 figs (spectrum +
#                    top-discriminant tall/readable with side legend);
#                    (4) DD-SIMCA: 14 individual plots in ddsimca/ subfolder
# v17 — 2026-05-28 — MAXIMUM PERCEPTUAL DISTINCTIVENESS color system:
#                    (1) PALETTE Trubetskoy/Glasbey 20 colors (deltaE_min 27.4
#                        vs ~15 before; eliminates 3 near-identical blues/2 greens);
#                    (2) optional detection of glasbey/colorcet libs;
#                    (3) SEQUENTIAL deterministic assignment (adjacent contrast)
#                        replaces hash; (4) secondary SHAPE channel
#                        (mapear_marcadores_classes, 14 shapes) for
#                        colorblindness/B&W; (5) edge_para_cor by luminance
# v18 — 2026-05-28 — Axis readability: _ticks_x_inteiros() applies
#                    MaxNLocator(integer, nbins=10) when >15 ticks
#                    (LV selection and PLS regression with 30-50 LVs no longer
#                    overlap numbers); <=15 shows all values.
# v19 — 2026-05-28 — V3 HCA/VIP:
#                    (1) HCA on centroids in PCA(hca_n_pcs=65) — reduces
#                        noise; (2) dendrogram axes inverted
#                        (orientation=top: species on lower X axis colored
#                        and rotated, distance on left Y axis);
#                    (3) fig_hca_comparacao_pipelines: dendrogram panel
#                        (raw/SNV/MSC/SG1/SG2/SNV+SG1/MSC+SG1/norm);
#                    (4) automatic cluster interpretation (k=2);
#                    (5) VIP: y-lim on real range + statistics box
#                        (min/max/mean/std/n>=1) — checks real dispersion
#                    + Config flags: mostrar_marcadores_classe/elipses_grupo
# v20 — 2026-05-28 — Organization Q1: folder PLSDA_OE_{level}_{preproc}_
#                    {YYYYMMDD_HHMMSS} with subfolders dados/ figuras/
#                    modelos/ logs/. Figures->figuras/; metadata,
#                    identifiers, comparison->dados/; summary->logs/;
#                    final model (joblib: preproc+PLS+LB+wavenumbers)
#                    ->modelos/. Sprint1 audit (A1,A2,A3,A5,A6,A11):
#                    confirmed ALREADY implemented in previous versions.
# v22 — 2026-05-29 — Phase 0 (rigor fixes):
#                    B1: validar_entrada synchronizes mae_id with the SAME mask
#                        for NaN/Inf removal (before, 1 NaN silently disabled
#                        GroupKFold = replica leakage);
#                    B4: DD-SIMCA 'todos' mode no longer reports misleading
#                        in-sample "spec" (spec=n/a; mode label in
#                        summary makes clear that sens/spec != authentication);
#                    B7: Q-residual in summary with adaptive notation (:.4g
#                        when <1e-3) instead of displaying 0.0000.
# v21 — 2026-05-28 — STAGE 4 (variable selection) + class exclusion:
#                    (1) Config.excluir_classes (e.g. Copaiba anomalous batch);
#                    (2) iPLS (intervals), selection by VIP>=threshold, by SR
#                        (top fraction), sPLS-DA (NIPALS soft-selection);
#                    (3) single evaluator _avaliar_subset_cv (group-aware CV,
#                        MC re-fitted per fold = no leakage);
#                    (4) figures fig_etapa4_ipls_intervalos +
#                        fig_etapa4_comparacao_metodos; CSVs in dados/;
#                    (5) most PARSIMONIOUS method selected (bal.acc within
#                        1% of max, fewer variables) in summary.
# v24 — 2026-05-29 — Sprint v24: Publication Figures:
#                    (1) fig_loadings_pca: PCA Loading Plot PC1/PC2 (bars
#                        colored by sign, NIR inverted X axis);
#                    (2) fig_roc_auc: Multiclass ROC curves OvR (scores
#                        group-aware CV; macro AUC in title and summary);
#                    (3) fig_splot_opls: OPLS-DA S-Plot (covariance x
#                        correlation with t_pred; top-N annotated; colormap
#                        RdBu_r; ref. Bylesjo 2006);
#                    (4) fig_cooman_ddsimca: DD-SIMCA Cooman's Plot (pairs
#                        A x B; sqrt(dQ) scale; subplot grid;
#                        ref. Pomerantsev 2020).
#                    Integration: aucs_roc added to resumo_modelo.txt.
# v23 — 2026-05-29 — ACCESSIBLE LAYER (no code editing):
#                    (1) _CONFIG_SPEC: single source mapping friendly names
#                        <-> Config attributes, with type,
#                        description and options for validation;
#                    (2) salvar_config/carregar_config: commented YAML in
#                        plain language; defaults preserved for missing keys;
#                        unknown keys ignored;
#                    (3) menu_interativo: terminal assistant (CMD-style)
#                        to edit fields, save/load and run without opening
#                        the code editor;
#                    (4) new __main__: --rodar (uses config.yaml), --codigo
#                        (legacy CFG), or interactive menu when in terminal;
#                    (5) config.yaml template generated (excludes Copaiba
#                        anomalous batch, max_lvs=40). Pipeline logic INTACT.
# v27  benchmark_classificadores integrated into executar()
# v28  Monte Carlo CV (IC95%); SHAP TreeExplainer; DET curves (linear+log)
# v29  hardware_probe; auto RAM tiers (4 levels); RAM guards; cleanup util
# v30  PowerPoint export; .streamlit/config.toml; CLAUDE.md; English i18n
# v31  Bugfix: (1) iPLS/comparar_pipelines Q2 overflow guard (ss_res non-finite
#              → q2=nan; eliminates -3.9e31 artifact in narrow intervals);
#           (2) Wold permutation loop: filter non-finite r2/q2 before polyfit
#              (fixes NaN intercept when degenerate model produces blow-up);
#           (3) Wold fig: correct N/A status when intercept is NaN;
#           (4) resumo_modelo.txt: Q2 NaN shown as "n/a" instead of crash;
#           (5) config.yaml: N1 14-class, pasta dados/, ddsimca todos
# =============================================================================

"""Chemometric pipeline for FT-NIR spectroscopy of Amazonian vegetable oils.

Publication-quality implementation — GEAAp / UFPA (PIBIC).

Capabilities:
    Preprocessing    : SNV, MSC (stateful), SG, MC, Autoscaling
    Classification   : PLS-DA, OPLS-DA, DD-SIMCA (GroupKFold anti-leakage)
    Variable sel.    : iPLS, VIP >= 1, SR top-fraction, sPLS-DA
    Validation       : Y-permutation, Wold R2Y/Q2Y, CV-ANOVA, BCa bootstrap
    Benchmark        : PLS-DA vs SVM vs RF vs GBM vs XGBoost (OOF GroupKFold)
    Interpretability : SHAP TreeExplainer, DET curves (linear + log)
    Uncertainty      : Monte Carlo CV — N x StratifiedGroupShuffleSplit IC95%
    Reports          : PDF, Word, Excel (5 sheets), LaTeX, PowerPoint

Best result: MSC -> SG -> MC, balanced accuracy = 0.923
(GroupKFold, 1807 samples, 14 Amazonian oil species).
"""

__version__ = "31.0.0"

import os
import glob
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Callable, cast, Any

import numpy as np
import pandas as pd
import matplotlib as mpl
# Backend headless (Agg) forcado sempre: sem isso o matplotlib usa TkAgg (se
# disponivel), que NAO e thread-safe. Combinado com a execucao paralela dos
# testes de permutacao/Wold (Fase E, n_jobs_permutacao>1) ou com processos de
# longa duracao rodando varias analises (Streamlit), objetos Tk pendentes de
# coleta de lixo podem ser finalizados fora da thread principal e derrubar o
# processo ("main thread is not in main loop"). Agg tambem e o backend correto
# para servidor/Cloud (sem display). Efeito colateral: a opcao
# 'abrir_figuras_na_tela'/mostrar_graficos nao abre mais janela — as figuras
# continuam sendo sempre salvas em disco normalmente.
mpl.use("Agg")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_val_predict, RepeatedStratifiedKFold,
    StratifiedShuffleSplit, GroupKFold, GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    balanced_accuracy_score, cohen_kappa_score, f1_score,
    precision_score, recall_score, r2_score,
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                SETTINGS — edit ONLY here                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Nome amigavel de cada nivel de analise (valor interno N1/N2/N3 inalterado).
_NIVEL_NOME = {
    "N1": "Classificacao por especie",
    "N2": "Discriminacao puro/adulterado",
    "N3": "Quantificacao de teor",
}


@dataclass
class Config:
    modo: str = "dx"                          # "dx" | "csv" | "sintetico"

    # ---- INPUT ----
    # Root folder with subfolders per species (each subfolder = 1 class) OR
    # single folder with .dx files (legacy mode). Auto-detected.
    # DO NOT version personal paths: the real path goes in config.yaml
    # (gitignored). This default is just a portable example.
    pasta_entrada: str = r"dados"
    parte_classe: int = 0                     # fallback if no subfolders
    extrair_conc_filename: bool = True        # only used in old fallback
    arquivo_csv: str = "seus_espectros.csv"
    coluna_classe: str = "classe"
    coluna_conc: Optional[str] = None
    usar_parse_title: bool = True             # uses ##TITLE= from JCAMP-DX

    # ---- OUTPUT (auto-generated by gerar_nome_saida) ----
    pasta_saida_raiz: str = "resultados_tcc"
    nivel: str = "N1"                         # "N1" | "N2" | "N3"
    tag:   str = ""                            # free label e.g. "pure", "soybean"
    pasta_saida: str = ""                     # DO NOT edit manually
    formato_saida: str = "png"                # png | pdf | svg
    dpi_salvar: int = 600
    mostrar_graficos: bool = False

    # ---- Score plot style (PCA / PLS-DA / OPLS) ----
    # Disable for "clean" scatter (color by class only, no shapes/contours).
    mostrar_marcadores_classe: bool = False   # False -> all circles 'o'
    mostrar_elipses_grupo:     bool = False   # False -> no T2/convex hull ellipse

    # ---- Figure detail level ----
    # False (default): gera apenas o conjunto ESSENCIAL de figuras (scores PCA/
    #   PLS-DA, confusao, ROC/AUC, selecao de LVs, SR+VIP, outliers T2/Q, e a
    #   aceitacao DD-SIMCA quando aplicavel). Mais rapido e menos arquivos.
    # True: gera tambem as figuras EXPLORATORIAS/detalhadas (HCA, loadings PCA,
    #   pre-processamento, contribuicao de score, DD-SIMCA por classe, Cooman).
    figuras_detalhadas: bool = False

    # ---- Group-aware validation (Q1 differentiator) ----
    # When True and mae_id is available, uses GroupKFold/GroupShuffleSplit
    # so that T1/T2/T3 of the same physical point stay in the same fold/holdout.
    agrupar_por_mae_id: bool = True

    # ---- Spectral truncation (useful FT-NIR range: 4000-10000 cm-1) -------
    # Removes FFT edge noise (0/8/16/24 cm-1 appear as false top
    # VIP when the SG derivative amplifies that region). Applied BEFORE
    # any preprocessing.
    wn_min: float = 4000.0
    wn_max: float = 10000.0

    # ---- Preprocessing ---------------------------------------------------
    # Quick preset. When != 'custom', overrides individual flags.
    #   'msc_sg_mc'   : MSC -> SG -> mean-centering (BEST: 0.923 bal.acc full)
    #   'snv_sg_mc'   : SNV -> SG -> mean-centering (Rinnan et al.)
    #   'autoscaling' : StandardScaler only (good on subset, poor on full: 0.472)
    #   'mc'          : mean-centering only
    #   'custom'      : honors aplicar_snv / aplicar_sg / aplicar_mc below
    preprocessamento_padrao: str = "msc_sg_mc"   # v14: best on full dataset (1807)

    aplicar_snv: bool = True
    aplicar_sg: bool = True
    sg_window: int = 25
    sg_polyorder: int = 2
    sg_deriv: int = 1
    aplicar_mc: bool = True

    max_lvs: int = 40
    n_pcs_pca: int = 10
    # HCA: dendrogram uses PCA scores with hca_n_pcs components
    # (reduces spectral noise before clustering). comparar_hca_pipelines
    # generates a dendrogram panel per preprocessing method.
    hca_n_pcs: int = 65
    comparar_hca_pipelines: bool = True
    n_splits_cv: int = 5
    n_repeats_cv: int = 3

    frac_cal: float = 0.70

    n_permutacoes: int = 200
    n_permutacoes_wold: int = 200        # 200 for publication; 50 is minimum for diagnostic
    n_bootstrap_vip: int = 30
    n_bootstrap_bca: int = 500
    comparar_pipelines: bool = True
    executar_wold: bool = True
    executar_cv_anova: bool = True
    # Paralelismo (processos, via joblib/loky) para os testes de permutacao/
    # Wold — cada iteracao e independente (mesma X, so o rotulo e
    # reembaralhado), entao rodar em paralelo NAO altera nenhum resultado
    # numerico, so o tempo de execucao (verificado por teste de regressao).
    # Padrao 1 (sequencial) preserva o comportamento atual sem mudanca.
    # Medido: n_jobs=4 ~2x mais rapido no pipeline completo (threading NAO
    # ajuda aqui — overhead do sklearn e Python/GIL-bound; processos sim).
    # Em ambientes com pouca RAM/CPU (ex.: Streamlit Cloud gratuito), deixe em 1
    # (processos extras multiplicam o uso de memoria).
    n_jobs_permutacao: int = 1

    frac_holdout: float = 0.20         # v14: external holdout by default
    # v15: pure samples (conc==0) ALWAYS stay in training — they are scarce (3/class)
    # and precious for DD-SIMCA/interpretation. Only adulterated go to holdout.
    holdout_preserva_puros: bool = True
    seed_holdout: int = 42

    n_por_classe: int = 20
    n_pontos_sint: int = 1000

    seed: int = 42

    # Sprint 3
    executar_ddsimca: bool = True
    ddsimca_n_components: int = 7       # PCA LVs per DD-SIMCA model (3 was insufficient — UCL poorly calibrated)
    ddsimca_ucl_method: str = "empirical"  # 'empirical' | 'theoretical' | 'chi2'
    # v14: 'todos' trains each model with ALL samples of the class
    # (exploratory; works with few pure samples). 'puros' = true one-class N2
    # but requires >=15 pure/class (current data: 3/class).
    ddsimca_treinar_em: str = "puros"   # 'todos' | 'puros'
    executar_opls: bool = True
    n_ortho_opls: int = 1               # OPLS-DA orthogonal components
    executar_benchmark: bool = False    # v27: SVM / RF / XGBoost vs PLS-DA (same CV)
    executar_monte_carlo: bool = False      # v28: MC CV (N × GroupShuffleSplit) for 95% CI
    n_monte_carlo: int = 100               # number of MC repetitions
    monte_carlo_test_size: float = 0.25    # test fraction per MC repetition
    monte_carlo_incluir_todos: bool = False # v28: run all benchmark models in MC CV
    executar_shap: bool = False            # v28: SHAP values (TreeExplainer) for ensemble
    shap_max_amostras: int = 500        # sample limit for SHAP (memory control)

    # ---- Class exclusion (e.g. Copaiba with anomalous batch) ----
    excluir_classes: Tuple[str, ...] = ()

    # ---- STAGE 4: Variable selection ----
    executar_etapa4: bool = True
    ipls_n_intervalos: int = 20         # iPLS: number of intervals
    vip_threshold_sel: float = 1.0      # selection by VIP >= threshold
    sr_top_frac: float = 0.20           # selection by SR: top fraction
    splsda_keep_por_comp: int = 50      # sPLS-DA: non-zero variables/component
    # SPA/APS (Successive Projections Algorithm, Araujo et al. 2001) and
    # AG (Genetic Algorithm, GA-PLS) — opt-in (default False) because they
    # run many more CV evaluations than iPLS/VIP/SR/sPLS-DA (heavier, like
    # executar_benchmark/executar_shap).
    executar_spa: bool = False
    spa_n_vars_max: int = 20            # SPA: max chain length (variables)
    spa_n_starts: int = 25              # SPA: starting variables sampled (evenly across spectrum)
    executar_ag: bool = False
    ag_tam_populacao: int = 20          # AG: chromosomes per generation
    ag_n_geracoes: int = 15             # AG: number of generations
    ag_prob_mutacao: float = 0.02       # AG: per-bit mutation probability
    ag_frac_inicial: float = 0.10       # AG: initial fraction of variables "on" per chromosome


CFG = Config()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                END OF SETTINGS                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝


# parse_title/JCAMP-DX (CODIGO_ESPECIE, ADULTERANTE_NOME, parse_title,
# extrair_title_do_dx) extraidos p/ dados_io.py (Fase H). Reexportados aqui
# para nao quebrar `pipeline.parse_title(...)` nem o restante do modulo.
from dados_io import (   # noqa: E402
    CODIGO_ESPECIE,
    ADULTERANTE_NOME,
    parse_title,
    extrair_title_do_dx,
)


def gerar_nome_saida(cfg: Config, n_classes: int, n_amostras: int) -> str:
    """Default output path (v20): project prefix + analysis type
    (level + preprocessing) + compact date/time.
        {root}/PLSDA_OE_{level}_{preproc}_{YYYYMMDD_HHMMSS}
    Example: resultados_tcc/PLSDA_OE_N1_MSC-SG1-MC_20260528_191500
    Subfolders (created in executar): dados/ figuras/ modelos/ logs/
    """
    preset = (cfg.preprocessamento_padrao or "custom").lower()
    if preset == "autoscaling":
        preproc = ["AUTO"]
    elif preset == "mc":
        preproc = ["MC"]
    elif preset == "snv_sg_mc":
        preproc = ["SNV", f"SG{cfg.sg_deriv}", "MC"]
    elif preset == "msc_sg_mc":
        preproc = ["MSC", f"SG{cfg.sg_deriv}", "MC"]
    else:  # custom
        preproc = []
        if cfg.aplicar_snv: preproc.append("SNV")
        if cfg.aplicar_sg:  preproc.append(f"SG{cfg.sg_deriv}")
        if cfg.aplicar_mc:  preproc.append("MC")
        if not preproc:     preproc.append("raw")
    partes = ["PLSDA_OE", cfg.nivel]
    if cfg.tag.strip():
        partes.append(cfg.tag.strip().replace(" ", "_"))
    partes.append("-".join(preproc))
    partes.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return os.path.join(cfg.pasta_saida_raiz, "_".join(partes))


# Paleta/marcadores de classes extraidos p/ paleta_cores.py (Fase H). Reexpor-
# tados aqui para nao quebrar `pipeline.cor(...)`/`pipeline.PALETA` nem as
# ~52 chamadas internas nas funcoes de figura.
from paleta_cores import (   # noqa: E402
    PALETA,
    MARCADORES,
    _paleta_externa,
    _luminancia,
    edge_para_cor,
    cor,
    mapear_cores_classes,
    mapear_marcadores_classes,
)


# Camada de plotagem (setup_matplotlib, salvar, especificidade_por_classe,
# helpers de plot e ~30 funcoes fig_*) extraida p/ figuras.py (Fase H).
# Reexportada aqui para nao quebrar pipeline.fig1_pca_scores(...),
# pipeline.salvar(...) nem as chamadas de executar()/figuras remanescentes.
from figuras import (   # noqa: E402
    setup_matplotlib,
    salvar,
    especificidade_por_classe,
    elipse_t2,
    convex_hull_contorno,
    parametros_scatter_adaptativos,
    _ticks_x_inteiros,
    plot_scores_panel,
    fig1_selecao_lvs,
    fig_hca_dendrograma,
    fig_hca_comparacao_pipelines,
    fig1_pca_scores,
    fig2_plsda_scores,
    fig3_outliers,
    fig4_confusao,
    fig5_vip,
    fig6_preprocessamento,
    fig_extra_wold,
    fig_extra_holdout,
    fig_extra_comparacao_pipelines,
    fig5b_vip_estabilidade,
    fig7_pls_regressao,
    fig_sprint3_sr_vip,
    fig_sprint3_score_contribution,
    fig_sprint3_ddsimca_acceptance,
    fig_ddsimca_individuais,
    fig_sprint3_opls_scores,
    fig_loadings_pca,
    fig_roc_auc,
    fig_splot_opls,
    fig_cooman_ddsimca,
)


# Transformers de pre-processamento (SNV/SavGol/MSC) + construir_preprocessador
# extraidos p/ preprocessamento.py (Fase H). Reexportados aqui para nao quebrar
# pipeline.SNV / pipeline.construir_preprocessador(...) nem o restante do modulo.
from preprocessamento import (   # noqa: E402
    SNV,
    SavGol,
    MSC,
    construir_preprocessador,
)


# =========================================================================
#  Chemometric diagnostics
#  Extraidos para chemometric_stats.py (Fase H — modularizacao). Reexportados
#  aqui para nao quebrar `pipeline.vip_scores(...)` nem as chamadas internas.
# =========================================================================
from chemometric_stats import (   # noqa: E402
    vip_scores,
    calcular_selectivity_ratio,
    hotelling_t2,
    hotelling_t2_limite,
    q_residuos,
    q_residuos_limite,
    variancia_explicada,
    figuras_merito_regressao,
)


# DDSimca / OPLSDAWrapper extraidos p/ classificadores.py (Fase H).
# Reexportados aqui para nao quebrar pipeline.DDSimca(...) /
# pipeline.OPLSDAWrapper(...) nem as chamadas internas em executar().
from classificadores import (   # noqa: E402
    DDSimca,
    OPLSDAWrapper,
)


def metricas_modelo_pls(modelo: PLSRegression, X: np.ndarray, Y: np.ndarray,
                         Y_cv: np.ndarray) -> Tuple[float, float, float]:
    """R2X via 1 - SS(X - T @ P^T) / SS(X_centered); R2Y via 1 - SS(res)/SS(Y);
    Q2 likewise using CV predictions. Rigorous reconstruction formula."""
    X = np.asarray(X, dtype=float)
    T = np.asarray(modelo.x_scores_,   dtype=float)
    P = np.asarray(modelo.x_loadings_, dtype=float)

    X_c = X - X.mean(axis=0)
    ss_x_total = float(np.sum(X_c ** 2))
    ss_x_res   = float(np.sum((X_c - T @ P.T) ** 2))
    r2x = float(1.0 - ss_x_res / ss_x_total) if ss_x_total > 0 else 0.0

    Y = np.asarray(Y, dtype=float)
    Y_cv = np.asarray(Y_cv, dtype=float)
    Y_hat = np.asarray(modelo.predict(X), dtype=float)
    ss_total = float(np.sum((Y - Y.mean(axis=0)) ** 2))
    if ss_total <= 0:
        return r2x, 0.0, 0.0
    r2y = float(1.0 - float(np.sum((Y - Y_hat) ** 2)) / ss_total)
    q2  = float(1.0 - float(np.sum((Y - Y_cv)  ** 2)) / ss_total)
    return r2x, r2y, q2


def salvar_identificadores(rotulos: np.ndarray, pred_lab: np.ndarray,
                            scores_pls: np.ndarray, T2: np.ndarray,
                            Q: np.ndarray, t2_lim: float, q_lim: float,
                            pasta: str) -> None:
    """Table with IDs, true/predicted classes and per-sample diagnostics."""
    n = len(rotulos)
    n_lvs_save = min(3, scores_pls.shape[1])
    dados = {
        "ID":             [f"S{i:03d}" for i in range(n)],
        "indice":         np.arange(n),
        "classe_real":    rotulos,
        "classe_predita": pred_lab,
        "correto":        rotulos == pred_lab,
    }
    for k in range(n_lvs_save):
        dados[f"LV{k+1}"] = scores_pls[:, k]
    dados["T2"]          = T2
    dados["T2_outlier"]  = T2 > t2_lim
    dados["Q"]           = Q
    dados["Q_outlier"]   = Q > q_lim
    pd.DataFrame(dados).to_csv(
        os.path.join(pasta, "amostras_identificadores.csv"),
        index=False, sep=";", decimal=",")


def salvar_resumo_modelo(pasta: str, info: Dict[str, object]) -> None:
    caminho = os.path.join(pasta, "resumo_modelo.txt")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  PLS-DA Model Summary\n")
        f.write("=" * 60 + "\n\n")
        largura = max(len(k) for k in info.keys()) + 2
        for k, v in info.items():
            if isinstance(v, float):
                f.write(f"  {k:<{largura}s}: {v:.4f}\n")
            else:
                f.write(f"  {k:<{largura}s}: {v}\n")

        # Methodological transparency notes (for article Methods section)
        f.write("\n" + "-" * 60 + "\n")
        f.write("  Methodological Notes (for article peer review)\n")
        f.write("-" * 60 + "\n")
        notes = [
            ("LV selection", "Wold parsimony criterion (2% RMSECV tolerance above "
             "minimum). Prevents overfitting on broad RMSECV plateaus with "
             "high max_lvs. Reference: Wold (1978) Technometrics 20:397-405."),
            ("CV-ANOVA F-test", "Eriksson et al. (2008) formula applied to one-hot "
             "Y_bin matrix. With K>2 classes, columns of Y_bin are not independent "
             "(row-sums constrained to 1), so the F-statistic degrees of freedom "
             "are approximate. Report as global omnibus test only; do not "
             "interpret individual class F-values."),
            ("SHAP values", "Computed on the full training set (in-sample) using "
             "TreeExplainer. Represent feature importance for the fitted model, "
             "not out-of-sample generalization. For publication, state: "
             "'SHAP analysis was performed on training data (n=X); "
             "importance rankings may be optimistic.'"),
            ("Benchmark hyperparameters", "SVM (C=10, gamma=scale), RF (300 trees), "
             "GBM (200 trees, lr=0.05), XGBoost (300 trees, lr=0.05) use "
             "literature-based heuristics, not cross-validated tuning. "
             "Comparisons are indicative of order-of-magnitude differences; "
             "nested-CV would be required for rigorous pairwise claims."),
            ("VIP bootstrap", "Group-aware (mae_id): resamples physical measurement "
             "points (T1/T2/T3 kept together), preventing inflated stability "
             "from correlated replicates. Standard stratified bootstrap "
             "(per-sample) would overestimate VIP confidence intervals."),
        ]
        for title, note in notes:
            f.write(f"\n  [{title}]\n")
            # Word-wrap at 70 chars
            words = note.split()
            line = "    "
            for w in words:
                if len(line) + len(w) + 1 > 70:
                    f.write(line + "\n")
                    line = "    " + w + " "
                else:
                    line += w + " "
            if line.strip():
                f.write(line + "\n")


def validar_entrada(X: np.ndarray, wavenumbers: np.ndarray,
                     rotulos: np.ndarray, conc: Optional[np.ndarray] = None,
                     mae_id: Optional[np.ndarray] = None,
                     tol_const: float = 1e-12,
                     limiar_correlacao: float = 0.99995,
                     max_n_para_corr: int = 500
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                  Optional[np.ndarray], Optional[np.ndarray],
                                  Dict[str, object]]:
    """Robust validation. Removes NaN/Inf, constant variables, and detects
    exact/approximate duplicates. Returns cleaned data + report.

    mae_id (B1): synchronized with the SAME NaN/Inf removal mask, so that
    group-aware validation survives sample removal (previously, a single
    NaN sample would silently disable mae_id and GroupKFold).

    Returns (X, wavenumbers, rotulos, conc, mae_id, relatorio).
    """
    X = np.asarray(X, dtype=float)
    rotulos = np.asarray(rotulos, dtype=str)
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    if conc is not None:
        conc = np.asarray(conc, dtype=float)
    if mae_id is not None:
        mae_id = np.asarray(mae_id, dtype=str)

    warnings_list: List[str] = []
    relatorio: Dict[str, object] = {
        "n_inicial":               int(len(X)),
        "n_variaveis_inicial":     int(X.shape[1]),
        "n_nan_amostras":          0,
        "n_inf_amostras":          0,
        "n_amostras_removidas":    0,
        "n_constantes_removidas":  0,
        "n_duplicatas_exatas":     0,
        "n_duplicatas_aproximadas": 0,
    }

    # --- NaN / Inf per sample (entire row eliminated) --------------------
    nan_mask = np.any(np.isnan(X), axis=1)
    inf_mask = np.any(np.isinf(X), axis=1)
    bad_mask = nan_mask | inf_mask
    n_nan = int(nan_mask.sum())
    n_inf = int(inf_mask.sum())
    relatorio["n_nan_amostras"] = n_nan
    relatorio["n_inf_amostras"] = n_inf
    if bad_mask.any():
        n_rem = int(bad_mask.sum())
        relatorio["n_amostras_removidas"] = n_rem
        warnings_list.append(
            f"{n_rem} samples removed due to NaN ({n_nan}) or Inf ({n_inf})")
        print(f"[WARNING] Removed {n_rem} samples with NaN/Inf.")
        keep = ~bad_mask
        X = X[keep]; rotulos = rotulos[keep]
        if conc is not None:
            conc = conc[keep]
        if mae_id is not None:
            mae_id = mae_id[keep]   # B1: keep group-aware in sync

    # --- Constant columns -----------------------------------------------
    var_cols = np.var(X, axis=0)
    mask_var = var_cols > tol_const
    n_const = int((~mask_var).sum())
    relatorio["n_constantes_removidas"] = n_const
    if n_const > 0:
        warnings_list.append(
            f"{n_const} variables with variance ~= 0 removed")
        print(f"[WARNING] Removed {n_const} constant variables.")
        X = X[:, mask_var]
        wavenumbers = wavenumbers[mask_var]

    # --- Exact duplicates -----------------------------------------------
    _, idx_unique = np.unique(X, axis=0, return_index=True)
    n_dup_exatas = int(len(X) - len(idx_unique))
    relatorio["n_duplicatas_exatas"] = n_dup_exatas
    if n_dup_exatas > 0:
        msg = (f"CAUTION: {n_dup_exatas} EXACT duplicate samples "
                f"detected. Possible train/validation leakage if "
                f"copies fall in different folds.")
        warnings_list.append(msg)
        print(f"[CAUTION] {msg}")

    # --- Approximate duplicates (high correlation) ----------------------
    if len(X) <= max_n_para_corr and len(X) >= 2:
        Xc = X - X.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(Xc, axis=1)
        norms[norms == 0] = 1.0
        Xn = Xc / norms[:, None]
        corr = Xn @ Xn.T
        np.fill_diagonal(corr, 0.0)
        pares_altos = np.argwhere(corr > limiar_correlacao)
        if len(pares_altos) > 0:
            envolvidas = np.unique(pares_altos.flatten())
            n_aprox = int(len(envolvidas))
            relatorio["n_duplicatas_aproximadas"] = n_aprox
            if n_aprox > n_dup_exatas:
                warnings_list.append(
                    f"CAUTION: {n_aprox} samples with correlation > "
                    f"{limiar_correlacao} (possible technical replicates). "
                    f"Consider GroupKFold to avoid leakage.")
                print(f"[CAUTION] {n_aprox} samples with corr > "
                      f"{limiar_correlacao:.5f}.")

    relatorio["n_final"] = int(len(X))
    relatorio["n_variaveis_final"] = int(X.shape[1])
    relatorio["warnings"] = warnings_list
    return X, wavenumbers, rotulos, conc, mae_id, relatorio


def verificar_balanceamento(rotulos: np.ndarray, ratio_alvo: float = 5.0
                              ) -> Dict[str, object]:
    """Detects severe class imbalance."""
    cls_unicas, counts = np.unique(rotulos, return_counts=True)
    n_max = int(counts.max())
    n_min = int(counts.min())
    ratio = n_max / max(n_min, 1)
    rel: Dict[str, object] = {
        "imbalance_ratio": float(ratio),
        "n_max":           n_max,
        "n_min":           n_min,
        "distribuicao":    {str(c): int(n) for c, n in zip(cls_unicas, counts)},
        "desbalanceado":   ratio > ratio_alvo,
    }
    if ratio > ratio_alvo:
        print(f"[WARNING] Severe class imbalance: max/min ratio = "
              f"{ratio:.2f} (max={n_max}, min={n_min}).")
        print("          Suggestions: prioritize balanced_accuracy/F1-macro "
              "over accuracy; consider class_weight or subsampling.")
    return rel


def metricas_classificacao(y_true, y_pred, classes) -> Dict[str, float]:
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa":       float(cohen_kappa_score(y_true, y_pred)),
        "f1_macro":          float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "precision_macro":   float(precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "recall_macro":      float(recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
    }


def comparar_pipelines(cfg: Config, X_raw: np.ndarray, Y_bin: np.ndarray,
                        y_int: np.ndarray, cv_indices: list,
                        max_lv: int = 8) -> Dict[str, Dict[str, float]]:
    """Evaluates several preprocessing pipelines via CV. For each
    pipeline finds the best n_lv by RMSECV and reports accuracy,
    balanced accuracy and Q2."""

    def _etapas_sg():
        return ("sg", SavGol(cfg.sg_window, cfg.sg_polyorder, cfg.sg_deriv))
    def _mc():
        return ("mc", StandardScaler(with_std=False))

    presets: Dict[str, Callable[[], List]] = {
        "Apenas MC":              lambda: [_mc()],
        "Autoscaling":            lambda: [("auto", StandardScaler())],
        "SNV + MC":               lambda: [("snv", SNV()), _mc()],
        "MSC + MC":               lambda: [("msc", MSC()), _mc()],
        "SG + MC":                lambda: [_etapas_sg(), _mc()],
        "SG -> SNV + MC":         lambda: [_etapas_sg(), ("snv", SNV()), _mc()],
        "SNV -> SG + MC (atual)": lambda: [("snv", SNV()), _etapas_sg(), _mc()],
        "MSC -> SG + MC":         lambda: [("msc", MSC()), _etapas_sg(), _mc()],
    }

    ss_total = float(np.sum((Y_bin - Y_bin.mean(axis=0)) ** 2))
    resultados: Dict[str, Dict[str, float]] = {}

    for nome, build_etapas in presets.items():
        melhor = {"q2": -np.inf, "accuracy": 0.0, "balanced_acc": 0.0, "n_lv": 1}
        n_lv_ok = 0
        for n_lv in range(1, max_lv + 1):
            def factory(_etapas=build_etapas, _n=n_lv):
                return Pipeline(_etapas() + [
                    ("pls", PLSRegression(n_components=_n, scale=False))])
            try:
                y_cv = _cv_predict_manual(factory, X_raw, Y_bin, cv_indices)
            except Exception:
                continue
            n_lv_ok += 1
            ss_res = float(np.sum((Y_bin - y_cv) ** 2))
            if ss_total < 1e-12 or not np.isfinite(ss_res):
                q2 = float("nan")
            else:
                q2 = max(-1.0, 1.0 - ss_res / ss_total)
            y_int_hat = np.argmax(y_cv, axis=1)
            acc = float(accuracy_score(y_int, y_int_hat))
            bal = float(balanced_accuracy_score(y_int, y_int_hat))
            if q2 > melhor["q2"]:
                melhor = {"q2": q2, "accuracy": acc,
                          "balanced_acc": bal, "n_lv": n_lv}
        resultados[nome] = melhor
        if n_lv_ok == 0:
            # Nenhum numero de LVs convergiu: o preset falhou por completo.
            # Sem este aviso, Acc=0.000 seria exibido como se fosse um
            # resultado ruim de verdade, mascarando a falha numerica.
            print(f"  {nome:<26s} -> [AVISO] nenhum modelo convergiu "
                  f"({max_lv} LVs testados) — preset ignorado")
            continue
        _q2_disp = f"{melhor['q2']:.3f}" if np.isfinite(melhor['q2']) else "n/a"
        print(f"  {nome:<26s} -> LVs={melhor['n_lv']:2d}  "
              f"Acc={melhor['accuracy']:.3f}  "
              f"BalAcc={melhor['balanced_acc']:.3f}  "
              f"Q2={_q2_disp}")

    return resultados


def bootstrap_vip_estratificado(X_processed: np.ndarray, Y_bin: np.ndarray,
                                  y_int: np.ndarray, n_opt: int, n_boot: int,
                                  seed: int, vip_threshold: float = 1.0,
                                  mae_id: Optional[np.ndarray] = None,
                                  ) -> Dict[str, object]:
    """GROUP-AWARE stratified VIP bootstrap (anti-leakage).

    When mae_id is provided (recommended), resamples mae_id GROUPS with
    replacement within each class — all samples from a physical measurement
    point (T1/T2/T3 replicates) are kept together. This prevents inflated
    VIP stability caused by partial inclusion of correlated replicates.

    Without mae_id (fallback), resamples individual samples per class.

    Returns dict with:
        mean, std, ci95_low, ci95_high  - point statistics per variable
        selection_frequency             - fraction of bootstraps with VIP >= threshold
        n_validos, n_falhos             - iteration counts
    """
    rng = np.random.default_rng(seed)
    n_var = X_processed.shape[1]
    classes = np.unique(y_int)
    indices_por_classe = {int(c): np.where(y_int == c)[0] for c in classes}

    # Pre-compute per-class groups when mae_id is available
    grupos_por_classe: Optional[Dict[int, np.ndarray]] = None
    if mae_id is not None:
        mae_id = np.asarray(mae_id)
        grupos_por_classe = {
            int(c): np.unique(mae_id[indices_por_classe[int(c)]])
            for c in classes
        }

    vips_arr: List[np.ndarray] = []
    n_validos = 0
    n_falhos = 0

    for _ in range(n_boot):
        partes = []
        if grupos_por_classe is not None:
            # Group-aware bootstrap: resample groups, include all their samples
            for c in classes:
                grps = grupos_por_classe[int(c)]
                grps_boot = rng.choice(grps, size=len(grps), replace=True)
                for g in grps_boot:
                    mask = (y_int == c) & (mae_id == g)
                    partes.append(np.where(mask)[0])
        else:
            for c in classes:
                idx_c = indices_por_classe[int(c)]
                partes.append(rng.choice(idx_c, size=len(idx_c), replace=True))
        idx = np.concatenate(partes)
        try:
            pls = PLSRegression(n_components=n_opt, scale=False)
            pls.fit(X_processed[idx], Y_bin[idx])
            v = vip_scores(pls)
            if np.all(np.isfinite(v)):
                vips_arr.append(v)
                n_validos += 1
            else:
                n_falhos += 1
        except Exception:
            n_falhos += 1

    if not vips_arr:
        zeros = np.zeros(n_var)
        return {
            "mean": zeros, "std": zeros,
            "ci95_low": zeros, "ci95_high": zeros,
            "selection_frequency": zeros,
            "n_validos": 0, "n_falhos": n_falhos,
        }

    arr = np.asarray(vips_arr)
    return {
        "mean":      arr.mean(axis=0),
        "std":       arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros(n_var),
        "ci95_low":  np.percentile(arr, 2.5,  axis=0),
        "ci95_high": np.percentile(arr, 97.5, axis=0),
        "selection_frequency": (arr >= vip_threshold).mean(axis=0),
        "n_validos": n_validos,
        "n_falhos":  n_falhos,
    }


def bootstrap_vip(X_processed, Y_bin, n_opt, n_boot, seed):
    """DEPRECATED: use bootstrap_vip_estratificado. Kept for
    backward compatibility. Does NOT use per-class stratification."""
    print("[WARNING] Non-stratified bootstrap_vip is DEPRECATED. "
          "Use bootstrap_vip_estratificado.")
    rng = np.random.default_rng(seed)
    n = len(X_processed)
    vips = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            pls = PLSRegression(n_components=n_opt, scale=False)
            pls.fit(X_processed[idx], Y_bin[idx])
            vips.append(vip_scores(pls))
        except Exception:
            continue
    if not vips:
        p = X_processed.shape[1]
        return np.zeros(p), np.zeros(p)
    vips = np.asarray(vips)
    return vips.mean(axis=0), vips.std(axis=0)


# Validacao estatistica (cross_val_predict manual, BCa, CV-ANOVA, teste de
# permutacao e de Wold) extraida p/ validacao_estatistica.py (Fase H).
# Reexportada para nao quebrar pipeline.teste_permutacao(...),
# pipeline._cv_predict_manual(...) nem as chamadas em comparar_pipelines/
# etapa4/executar.
from validacao_estatistica import (   # noqa: E402
    _cv_predict_manual,
    bootstrap_bca_ci,
    cv_anova_eriksson,
    _iter_wold,
    teste_wold,
    _iter_permutacao,
    teste_permutacao,
)


# Carregamento de dados (.dx JCAMP-DX/ASDF, CSV, sintetico) extraido p/
# dados_io.py (Fase H). Reexportado aqui para nao quebrar as chamadas de
# `executar()` (`pipeline.carregar_dados(cfg)` etc.).
from dados_io import (   # noqa: E402
    gerar_dados_sinteticos,
    carregar_csv,
    _flush_asdf,
    _decodificar_linha_asdf,
    parse_dx,
    parse_spectrum,
    _extrair_conc_filename,
    _listar_arquivos_espectro,
    _detectar_subpastas_classe,
    carregar_dx,
    carregar_dados,
)


# (Camada de plotagem extraida p/ figuras.py — ver reexport acima.)
# =========================================================================
#  STAGE 4 — Variable Selection (iPLS, VIP, SR, sPLS-DA)
# =========================================================================

# STAGE 4 — Selecao de variaveis (iPLS/sPLS-DA) + figuras da etapa extraidas
# p/ selecao_variaveis.py (Fase H). Reexportadas p/ nao quebrar as chamadas em
# executar() (etapa4_selecao_variaveis, selecao_ipls, ...).
from selecao_variaveis import (   # noqa: E402
    _avaliar_subset_cv,
    selecao_ipls,
    sparse_plsda_mask,
    _spa_cadeia,
    selecao_spa,
    selecao_ag,
    fig_etapa4_ipls,
    fig_etapa4_ag_convergencia,
    fig_etapa4_comparacao,
    etapa4_selecao_variaveis,
)


# Auto-Benchmark / Monte Carlo CV / DET / SHAP + PLSDAClassifier extraidos p/
# avaliacao_modelos.py (Fase H). Reexportados p/ nao quebrar as chamadas em
# executar() e o uso de PLSDAClassifier.
from avaliacao_modelos import (   # noqa: E402
    PLSDAClassifier,
    fig_benchmark_classificadores,
    benchmark_classificadores,
    fig_monte_carlo_distribuicao,
    _stratified_group_shuffle_splits,
    monte_carlo_cv,
    fig_det_curvas,
    fig_shap_benchmark,
)


# =========================================================================
#  Orquestrador
# =========================================================================

# =========================================================================
#  v29: Compatibilidade de hardware — probe, auto-ajuste, guardas de RAM
# =========================================================================

# Compatibilidade de hardware (probe, auto-ajuste, guarda de RAM) extraida p/
# hardware.py (Fase H). Reexportada p/ nao quebrar pipeline.hardware_probe(),
# pipeline.auto_ajustar_config_hardware(...) nem o uso em app_quimiometria.py.
from hardware import (   # noqa: E402
    hardware_probe,
    auto_ajustar_config_hardware,
    _verificar_ram,
)
def limpar_resultados_antigos(pasta_base: str,
                               manter_n: int = 3) -> Dict[str, Any]:
    """
    Remove as pastas de resultados mais antigas dentro de pasta_base,
    mantendo apenas as manter_n mais recentes (por data de modificacao).
    Retorna {'removidas': [paths], 'liberado_mb': float, 'erro': str|None}.
    """
    import shutil
    from pathlib import Path

    resultado: Dict[str, Any] = {"removidas": [], "liberado_mb": 0.0, "erro": None}
    base = Path(pasta_base)
    if not base.exists():
        resultado["erro"] = f"Pasta '{pasta_base}' nao encontrada."
        return resultado

    pastas = sorted(
        [p for p in base.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    a_remover = pastas[manter_n:]
    if not a_remover:
        return resultado

    for p in a_remover:
        try:
            tam = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            shutil.rmtree(p)
            resultado["removidas"].append(str(p))
            resultado["liberado_mb"] += tam / (1024 * 1024)
        except Exception as _e:
            resultado["erro"] = (resultado["erro"] or "") + f"\n{p}: {_e}"
    resultado["liberado_mb"] = round(resultado["liberado_mb"], 1)
    return resultado


def rmse_flat(a, b):
    return float(np.sqrt(np.mean((np.asarray(a).flatten()
                                  - np.asarray(b).flatten()) ** 2)))


def _agrupar_replicas_processadas(X_raw_subset: np.ndarray,
                                   mae_subset: Optional[np.ndarray],
                                   preproc_ajustado) -> List[np.ndarray]:
    """Agrupa espectros por mae_id (fisicas T1/T2/T3 do mesmo ponto, >=2
    replicas) e aplica o pre-processador JA AJUSTADO (so .transform(), nunca
    reajusta) a cada grupo — usado para estimar ruido instrumental/
    repetibilidade nas figuras de merito (Valderrama et al. 2009)."""
    if mae_subset is None:
        return []
    grupos: List[np.ndarray] = []
    for g in np.unique(mae_subset):
        idxg = np.where(mae_subset == g)[0]
        if len(idxg) >= 2:
            grupos.append(np.asarray(preproc_ajustado.transform(X_raw_subset[idxg])))
    return grupos


def pls_regressao_por_especie(
        X_raw: np.ndarray, conc: np.ndarray, rotulos: np.ndarray,
        mae_id: Optional[np.ndarray], classes_unicas: np.ndarray,
        cfg: "Config", pasta: str, n_splits: int,
        min_amostras_adult: int = 6) -> Optional[Dict[str, Any]]:
    """Regressão PLS do teor de adulteração, calibrada SEPARADAMENTE por espécie.

    Motivo: um único modelo multi-espécie é confundido — a variação inter-
    espécies (~90% da variância espectral) domina o sinal de adulteração,
    produzindo R²≈0 e um gráfico em banda vertical ("prevê a média").
    Calibrando DENTRO de cada espécie, o espectro varia apenas com o teor,
    e a regressão recupera a diagonal correta.

    Para cada espécie com >= `min_amostras_adult` amostras adulteradas e
    variância de teor > 0: faz split cal/val group-aware (mae_id), seleciona
    LVs por CV, ajusta, prediz. As predições são agrupadas (pooled) entre
    espécies e plotadas num único gráfico measured-vs-predicted (diagonal).

    Retorna dict com métricas pooled + tabela por espécie, ou None se nenhuma
    espécie tiver dados suficientes.
    """
    conc = np.asarray(conc, dtype=float)
    Yc_all, Ych_all, Yv_all, Yvh_all = [], [], [], []
    tabela_esp: List[Dict[str, Any]] = []
    erros_reg_repr: List[float] = []   # RMSECV curve from the largest species
    n_opt_repr = 1
    n_max_amostras = -1

    for cls in classes_unicas:
        idx = np.where(rotulos == cls)[0]
        if idx.size == 0:
            continue
        conc_c = conc[idx].copy()
        # Pure samples are stored as NaN (loaded from None); treat as 0% adulteration.
        # Do NOT skip species with NaN — that would exclude all pure reference points.
        conc_c = np.where(np.isnan(conc_c), 0.0, conc_c)
        # need variation in concentration (pure + adulterated of THIS species)
        n_adult_c = int(np.sum(conc_c > 0))
        if n_adult_c < min_amostras_adult or float(conc_c.std()) < 1e-8:
            continue

        X_c = X_raw[idx]
        mae_c = mae_id[idx] if mae_id is not None else None
        Y_c = conc_c.reshape(-1, 1)

        # group-aware cal/val split (replicates never split)
        try:
            if mae_c is not None and len(np.unique(mae_c)) >= 4:
                gss = GroupShuffleSplit(n_splits=1, train_size=cfg.frac_cal,
                                        random_state=cfg.seed)
                ic, iv = next(gss.split(X_c, Y_c, groups=mae_c))
            else:
                rng = np.random.default_rng(cfg.seed)
                perm = rng.permutation(len(conc_c))
                ncal = max(2, int(cfg.frac_cal * len(conc_c)))
                ic, iv = perm[:ncal], perm[ncal:]
        except Exception:
            continue
        if len(ic) < 4 or len(iv) < 2:
            continue

        Xc, Yc = X_c[ic], Y_c[ic]
        Xv, Yv = X_c[iv], Y_c[iv]
        lv_max = min(cfg.max_lvs, max(2, Xc.shape[0] // 5))

        # internal CV for LV selection (group-aware)
        if mae_c is not None:
            grupos_cal = mae_c[ic]
            n_g = int(len(np.unique(grupos_cal)))
            n_sp = max(2, min(n_splits, n_g))
            cv_reg = GroupKFold(n_splits=n_sp)
            grp = grupos_cal
        else:
            n_sp = max(2, min(n_splits, Xc.shape[0] // 2))
            cv_reg = KFold(n_splits=n_sp, shuffle=True, random_state=cfg.seed)
            grp = None

        erros_reg: List[float] = []
        try:
            for n in range(1, lv_max + 1):
                pipe = Pipeline([
                    ("preproc", construir_preprocessador(cfg)),
                    ("pls", PLSRegression(n_components=n, scale=False)),
                ])
                Y_hat = cross_val_predict(pipe, Xc, Yc, cv=cv_reg, groups=grp)
                erros_reg.append(rmse_flat(Yc, Y_hat))
        except Exception:
            continue
        if not erros_reg:
            continue

        n_opt_reg = int(np.argmin(erros_reg)) + 1
        pipe_final = Pipeline([
            ("preproc", construir_preprocessador(cfg)),
            ("pls", PLSRegression(n_components=n_opt_reg, scale=False)),
        ]).fit(Xc, Yc)
        Yc_hat = np.asarray(pipe_final.predict(Xc)).flatten()
        Yv_hat = np.asarray(pipe_final.predict(Xv)).flatten()

        rmsep_c = rmse_flat(Yv, Yv_hat)
        r2v_c = float(r2_score(Yv, Yv_hat)) if len(np.unique(Yv)) > 1 else float("nan")

        # Figuras de merito analiticas (Valderrama, Braga & Poppi, 2009):
        # ruido instrumental estimado a partir de replicas fisicas (T1/T2/T3
        # via mae_id) SOMENTE do lado de calibracao (nao usa dados de validacao).
        _preproc_ajustado = pipe_final.named_steps["preproc"]
        _X_cal_proc = np.asarray(_preproc_ajustado.transform(Xc))
        _grupos_rep = _agrupar_replicas_processadas(
            Xc, mae_c[ic] if mae_c is not None else None, _preproc_ajustado)
        _fom = figuras_merito_regressao(
            pipe_final.named_steps["pls"], _X_cal_proc, _grupos_rep)

        Yc_all.append(np.asarray(Yc).flatten())
        Ych_all.append(Yc_hat)
        Yv_all.append(np.asarray(Yv).flatten())
        Yvh_all.append(Yv_hat)
        tabela_esp.append({
            "especie": str(cls), "n_lv": n_opt_reg,
            "n_cal": int(len(ic)), "n_val": int(len(iv)),
            "rmsep": rmsep_c, "r2val": r2v_c,
            "lod": _fom["lod"], "loq": _fom["loq"],
            "sensibilidade": _fom["sensibilidade"],
            "sensibilidade_analitica": _fom["sensibilidade_analitica"],
            "seletividade_media": _fom["seletividade_media"],
        })

        # keep the RMSECV curve of the species with most samples (for panel a)
        if Xc.shape[0] > n_max_amostras:
            n_max_amostras = Xc.shape[0]
            erros_reg_repr = erros_reg
            n_opt_repr = n_opt_reg

    if not Yc_all:
        return None

    Yc_p = np.concatenate(Yc_all)
    Ych_p = np.concatenate(Ych_all)
    Yv_p = np.concatenate(Yv_all)
    Yvh_p = np.concatenate(Yvh_all)

    r2c = float(r2_score(Yc_p, Ych_p)) if len(np.unique(Yc_p)) > 1 else float("nan")
    r2v = float(r2_score(Yv_p, Yvh_p)) if len(np.unique(Yv_p)) > 1 else float("nan")
    rmsec = rmse_flat(Yc_p, Ych_p)
    rmsep = rmse_flat(Yv_p, Yvh_p)
    rmsecv = float(np.min(erros_reg_repr)) if erros_reg_repr else rmsec
    bias_v = float(np.mean(Yvh_p - Yv_p))

    # pooled diagonal figure (proper diagonal: within-species calibration)
    fig7_pls_regressao(Yc_p, Ych_p, Yv_p, Yvh_p, erros_reg_repr or [rmsec],
                       n_opt_repr, r2c, r2v, rmsec, rmsecv, rmsep, bias_v,
                       cfg, pasta)

    return {
        "tabela_especie": tabela_esp,
        "r2c": r2c, "r2v": r2v, "rmsec": rmsec, "rmsecv": rmsecv,
        "rmsep": rmsep, "bias": bias_v, "n_especies": len(tabela_esp),
    }


def executar(cfg: Config):
    setup_matplotlib(cfg)

    # --- 0. Hardware probe + auto-ajuste preventivo -------------------------
    _hw = hardware_probe()
    _avisos_hw = auto_ajustar_config_hardware(cfg, _hw)
    print(f"[HARDWARE] RAM total: {_hw['ram_total_gb']:.1f} GB  "
          f"livre: {_hw['ram_livre_gb']:.1f} GB  "
          f"CPU: {_hw['cpu_fisicos']}f/{_hw['cpu_logicos']}l")
    for _av in _avisos_hw:
        print(f"  [AUTO-AJUSTE] {_av}")

    # --- 1. Carregamento (6-tupla com mae_id + metadados) ------------------
    wavenumbers, X_raw, rotulos, conc, mae_id, metadados_df = carregar_dados(cfg)
    X_raw   = np.asarray(X_raw,   dtype=float)
    rotulos = np.asarray(rotulos, dtype=str)
    if conc is not None:
        conc = np.asarray(conc, dtype=float)
    if mae_id is not None:
        mae_id = np.asarray(mae_id, dtype=str)

    # --- 1a0. nivel N2: autenticação por espécie (DD-SIMCA one-class) ------
    # DESIGN (escolha do usuário — opção A):
    #   N1 = identificar a espécie (PLS-DA 13 classes, bal.acc≈0.906).
    #   N2 = autenticar pureza POR ESPÉCIE via DD-SIMCA one-class. Treina um
    #        modelo do "puro" para cada espécie e testa se cada amostra é
    #        pura (aceita) ou adulterada (rejeitada). É o método-padrão de
    #        autenticação e funciona (sens≈90%, esp=100%).
    #
    # CRÍTICO: NÃO remapeamos rotulos para puro/adulterado. Os rótulos de
    # ESPÉCIE são preservados — o DD-SIMCA precisa deles para construir um
    # modelo por espécie. (O remap binário anterior fazia o DD-SIMCA treinar
    # um único modelo "puro" genérico, perdendo a autenticação por espécie.)
    # A distinção puro/adulterado vem de `conc` (0 = puro), usada dentro do
    # bloco DD-SIMCA. Não há undersampling: o DD-SIMCA precisa das amostras
    # adulteradas para medir a especificidade.
    if cfg.nivel == "N2":
        if conc is not None:
            n_puro = int(np.sum(np.isnan(conc) | (conc == 0.0)))
            n_adul = int(np.sum(~(np.isnan(conc) | (conc == 0.0))))
            print(f"[INFO] N2 (autenticação por espécie): rótulos de espécie "
                  f"preservados para DD-SIMCA one-class "
                  f"(puros={n_puro} | adulterados={n_adul}). "
                  f"DD-SIMCA forçado para modo 'puros'.")
            # Force per-species one-class authentication for N2
            cfg.ddsimca_treinar_em = "puros"
            cfg.executar_ddsimca = True
        else:
            print("[AVISO] nivel=N2 sem dados de concentração (##TITLE= sem "
                  "adulterante). Não é possível separar puro/adulterado — "
                  "verifique os arquivos .dx.")

    # --- 1a. Truncamento espectral: remove ruido de borda da FFT ----------
    # SG derivativo amplifica os ultimos pontos da FFT (proximos a 0 cm-1
    # e ao final do interferograma). Sem truncar, esses pontos viram falsos
    # top-VIP com estabilidade 100% (artefato, nao quimica).
    n_orig = X_raw.shape[1]
    mask_wn = (wavenumbers >= cfg.wn_min) & (wavenumbers <= cfg.wn_max)
    if not mask_wn.any():
        raise ValueError(
            f"Nenhuma variavel sobrevive ao filtro "
            f"[{cfg.wn_min}, {cfg.wn_max}] cm-1. "
            f"Faixa real dos dados: [{wavenumbers.min():.1f}, "
            f"{wavenumbers.max():.1f}].")
    n_removidas = n_orig - int(mask_wn.sum())
    X_raw       = X_raw[:, mask_wn]
    wavenumbers = wavenumbers[mask_wn]
    print(f"[INFO] Truncamento espectral [{cfg.wn_min:.0f}, {cfg.wn_max:.0f}]"
          f" cm-1: {n_removidas}/{n_orig} variaveis removidas, "
          f"restam {len(wavenumbers)} ({wavenumbers.min():.1f}-"
          f"{wavenumbers.max():.1f}).")

    # --- 1a2. Exclusao de classes (ex: Copaiba com lote anomalo) ----------
    if cfg.excluir_classes:
        excl = set(str(c) for c in cfg.excluir_classes)
        mask_keep = ~np.isin(rotulos, list(excl))
        n_rem = int((~mask_keep).sum())
        if n_rem > 0:
            print(f"[INFO] Excluindo classes {sorted(excl)}: "
                  f"{n_rem} amostras removidas.")
            X_raw   = X_raw[mask_keep]
            rotulos = rotulos[mask_keep]
            conc    = conc[mask_keep] if conc is not None else None
            mae_id  = mae_id[mask_keep] if mae_id is not None else None

    # --- 1b. Pasta de saida descritiva -------------------------------------
    cfg.pasta_saida = gerar_nome_saida(cfg, len(np.unique(rotulos)),
                                         X_raw.shape[0])
    pasta = cfg.pasta_saida
    # Estrutura de subpastas (v20): dados/ figuras/ modelos/ logs/
    pasta_dados   = os.path.join(pasta, "dados")
    pasta_modelos = os.path.join(pasta, "modelos")
    pasta_logs    = os.path.join(pasta, "logs")
    for _p in (pasta, pasta_dados, os.path.join(pasta, "figuras"),
               pasta_modelos, pasta_logs):
        os.makedirs(_p, exist_ok=True)
    print(f"[INFO] Saida: {pasta}")
    print("[INFO] Subpastas: dados/ figuras/ modelos/ logs/")
    if metadados_df is not None:
        cam_meta = os.path.join(pasta_dados, "metadados.csv")
        metadados_df.to_csv(cam_meta, index=False, sep=";", decimal=",")
        print(f"[INFO] Metadados salvos: {cam_meta}")

    # --- 1c. Input integrity validation -----------------------------------
    print("\n[0/7] Input integrity validation...")
    X_raw, wavenumbers, rotulos, conc, mae_id, relatorio_entrada = validar_entrada(
        X_raw, wavenumbers, rotulos, conc, mae_id)
    relatorio_balanco = verificar_balanceamento(rotulos)

    # B1: mae_id is now synchronized INSIDE validar_entrada (same NaN/Inf
    # removal mask). Group-aware validation survives removals —
    # no more silent GroupKFold disabling due to a single NaN.
    if mae_id is not None:
        mae_id = np.asarray(mae_id, dtype=str)

    # --- Validation strategy: group-aware if mae_id available -------------
    usar_grupos = (cfg.agrupar_por_mae_id and mae_id is not None
                   and len(np.unique(mae_id)) >= 3)
    if cfg.agrupar_por_mae_id and not usar_grupos:
        print("[INFO] GroupKFold desabilitado: mae_id indisponivel ou "
              "grupos insuficientes — usando StratifiedKFold (estratificada).")
    if usar_grupos and mae_id is not None:
        n_grupos = int(len(np.unique(mae_id)))
        print(f"[INFO] Validacao group-aware ATIVA: {n_grupos} grupos "
              f"unicos via mae_id.")

    # --- 1d. Hold-out externo (group-aware se possivel) --------------------
    X_holdout = None; rotulos_holdout = None; _conc_holdout = None
    _mae_id_holdout: Optional[np.ndarray] = None
    n_holdout = 0
    if cfg.frac_holdout > 0:
        try:
            # v15: optionally excludes pure samples from the draw — they always
            # stay in training. Split runs only on the eligible subset (adulterated).
            n_all = len(rotulos)
            if (cfg.holdout_preserva_puros and conc is not None):
                elegiveis = np.where(np.asarray(conc, dtype=float) > 0)[0]
                n_puros_reserv = n_all - len(elegiveis)
            else:
                elegiveis = np.arange(n_all)
                n_puros_reserv = 0

            Xe, rote = X_raw[elegiveis], rotulos[elegiveis]
            maee = mae_id[elegiveis] if mae_id is not None else None

            if usar_grupos and maee is not None:
                gss = GroupShuffleSplit(
                    n_splits=1, test_size=cfg.frac_holdout,
                    random_state=cfg.seed_holdout)
                tr_e, ho_e = next(gss.split(Xe, rote, groups=maee))
                tipo_ho = "GroupShuffleSplit (replicas juntas)"
            else:
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=cfg.frac_holdout,
                    random_state=cfg.seed_holdout)
                tr_e, ho_e = next(sss.split(Xe, rote))
                tipo_ho = "StratifiedShuffleSplit"
            # Remap indices from eligible subset back to global indices;
            # pure (non-eligible) samples are added entirely to training.
            nao_elegiveis = np.setdiff1d(np.arange(n_all), elegiveis)
            tr_idx = np.concatenate([elegiveis[tr_e], nao_elegiveis])
            ho_idx = elegiveis[ho_e]
            if n_puros_reserv > 0:
                tipo_ho += f" + {n_puros_reserv} puros preservados"
            X_holdout       = X_raw[ho_idx]
            rotulos_holdout = rotulos[ho_idx]
            # Reservados junto do holdout (ainda nao consumidos na avaliacao —
            # prefixo _ sinaliza intencional e evita falso-positivo de lint).
            _conc_holdout   = conc[ho_idx] if conc is not None else None
            if mae_id is not None:
                _mae_id_holdout = mae_id[ho_idx]
            X_raw   = X_raw[tr_idx]
            rotulos = rotulos[tr_idx]
            conc    = conc[tr_idx] if conc is not None else None
            if mae_id is not None:
                mae_id = mae_id[tr_idx]
            n_holdout = int(len(ho_idx))
            print(f"[INFO] Hold-out ({tipo_ho}): {n_holdout} amostras "
                  f"reservadas (frac={cfg.frac_holdout:.2f}). "
                  f"Pipeline rodara em {len(tr_idx)} amostras.")
        except Exception as e:
            print(f"[AVISO] Hold-out falhou ({e}). Continuando sem holdout.")
            X_holdout = None

    if cfg.aplicar_sg and cfg.sg_window >= X_raw.shape[1]:
        raise ValueError(
            f"sg_window ({cfg.sg_window}) deve ser menor que o numero de "
            f"pontos espectrais ({X_raw.shape[1]}).")

    classes_unicas = np.unique(rotulos)
    mapa_cores     = mapear_cores_classes(classes_unicas)
    mapa_marcadores = mapear_marcadores_classes(classes_unicas)

    print(f"\n[INFO] Amostras : {X_raw.shape[0]}")
    print(f"[INFO] Variaveis: {X_raw.shape[1]}")
    print(f"[INFO] Classes  : {classes_unicas.tolist()}")
    print(f"[INFO] Conc.    : {'sim' if conc is not None else 'nao'}")
    _mae_info = (f"sim ({int(len(np.unique(mae_id)))} grupos)"
                  if mae_id is not None else "nao")
    print(f"[INFO] mae_id   : {_mae_info}")
    print(f"[INFO] Imbalance ratio: {relatorio_balanco['imbalance_ratio']:.2f}  "
          f"(max={relatorio_balanco['n_max']}, min={relatorio_balanco['n_min']})")

    contagem = {cls: int(np.sum(rotulos == cls)) for cls in classes_unicas}
    min_por_classe = min(contagem.values())
    n_splits = min(cfg.n_splits_cv, min_por_classe)
    if usar_grupos and mae_id is not None:
        # Numero de grupos por classe limita k
        n_grupos_por_classe = []
        for cls in classes_unicas:
            n_grupos_por_classe.append(
                int(len(np.unique(mae_id[rotulos == cls]))))
        min_grupos_classe = min(n_grupos_por_classe)
        n_splits = min(n_splits, min_grupos_classe)
        if n_splits < cfg.n_splits_cv:
            print(f"[AVISO] n_splits ajustado para {n_splits} "
                  f"(min de grupos/classe = {min_grupos_classe}).")
    elif n_splits < cfg.n_splits_cv:
        print(f"[AVISO] n_splits_cv={cfg.n_splits_cv} excede minimo por "
              f"classe ({min_por_classe}). Usando {n_splits}.")

    if usar_grupos:
        # StratifiedGroupKFold: estratifica por classe E agrupa por mae_id
        cv = StratifiedGroupKFold(n_splits=max(n_splits, 2), shuffle=True,
                                    random_state=cfg.seed)
        cv_label = f"StratifiedGroupKFold n_splits={n_splits}"
    elif cfg.n_repeats_cv > 1:
        cv = RepeatedStratifiedKFold(n_splits=n_splits,
                                      n_repeats=cfg.n_repeats_cv,
                                      random_state=cfg.seed)
        cv_label = f"RepeatedStratifiedKFold n_splits={n_splits} repeats={cfg.n_repeats_cv}"
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=cfg.seed)
        cv_label = f"StratifiedKFold n_splits={n_splits}"

    lb = LabelBinarizer()
    Y_bin = np.asarray(lb.fit_transform(rotulos), dtype=float)
    if Y_bin.ndim == 1:
        Y_bin = np.column_stack([1 - Y_bin, Y_bin])
    y_int = np.argmax(Y_bin, axis=1)

    # --- 2. Pre-processamento (uma vez, para visualizacao e PCA) -----------
    print(f"\n[1/7] Pre-processamento (preset='{cfg.preprocessamento_padrao}')")
    preproc_full = construir_preprocessador(cfg).fit(X_raw)
    X_processed  = np.asarray(preproc_full.transform(X_raw), dtype=float)

    # --- 3. LV selection by CV (no leakage, group-aware if possible) -------
    print(f"\n[2/7] LV selection by CV ({cv_label})")

    def fabrica_pipeline(n_lv: int):
        return Pipeline([
            ("preproc", construir_preprocessador(cfg)),
            ("pls", PLSRegression(n_components=n_lv, scale=False)),
        ])

    erros_rmsecv: List[float] = []
    metricas_por_lv: List[Dict[str, float]] = []
    preds_por_lv: Dict[int, np.ndarray] = {}

    grupos_cv = mae_id if usar_grupos else None
    cv_indices = list(cv.split(X_raw, y_int, groups=grupos_cv))

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
        y_hat_int = np.argmax(y_hat, axis=1)
        m = metricas_classificacao(y_int, y_hat_int, np.arange(len(classes_unicas)))
        metricas_por_lv.append(m)
        preds_por_lv[n] = y_hat

    rmsecv_arr = np.array(erros_rmsecv)
    rmsecv_min = float(rmsecv_arr.min())
    n_opt_minrmsecv = int(np.argmin(rmsecv_arr)) + 1

    # Wold parsimony criterion (Wold 1978): select the SMALLEST number of LVs
    # whose RMSECV is within 2% of the minimum. Avoids overfitting when
    # RMSECV plateau is broad (common with max_lvs=40 and noisy FT-NIR data).
    tol_wold = rmsecv_min * 1.02
    candidatos_wold = np.where(rmsecv_arr <= tol_wold)[0]
    n_opt = int(candidatos_wold[0]) + 1   # smallest LV satisfying criterion

    if n_opt < n_opt_minrmsecv:
        print(f"  LVs otimas (Wold parcimonia): {n_opt} "
              f"(min-RMSECV em {n_opt_minrmsecv} LVs, delta <2%)")
    else:
        print(f"  LVs otimas: {n_opt}")

    Y_cv  = preds_por_lv[n_opt]
    pred_lab = lb.classes_[np.argmax(Y_cv, axis=1)]
    lvs_no_teto = (n_opt >= cfg.max_lvs)
    if lvs_no_teto:
        print(f"  [ATENCAO] LVs otimas ({n_opt}) == max_lvs ({cfg.max_lvs}): "
              f"RMSECV ainda nao atingiu plateau. Aumente max_lvs "
              f"(ex: {cfg.max_lvs + 10}) e rode novamente.")

    # --- 4. Modelo final (para VIP, scores, T2/Q) --------------------------
    pls_final = PLSRegression(n_components=n_opt, scale=False)
    pls_final.fit(X_processed, Y_bin)
    T_pls = np.asarray(pls_final.x_scores_,  dtype=float)
    P_pls = np.asarray(pls_final.x_loadings_, dtype=float).T
    var_lv_pls = variancia_explicada(X_processed, T_pls)
    vip = vip_scores(pls_final)

    # --- 5. PCA exploratoria -----------------------------------------------
    print("\n[3/7] PCA exploratoria")
    pca = PCA(n_components=min(cfg.n_pcs_pca, X_processed.shape[1],
                                X_processed.shape[0]))
    scores_pca = pca.fit_transform(X_processed)
    var_pca = pca.explained_variance_ratio_ * 100
    print(f"  PC1: {var_pca[0]:.2f}%  PC2: {var_pca[1]:.2f}%  "
          f"acumulado(2): {sum(var_pca[:2]):.2f}%")

    # --- 6. Permutation test (Y-randomization) ----------------------------
    print(f"\n[4/7] Teste de permutacao (Y-randomization, "
          f"n={cfg.n_permutacoes})")
    if usar_grupos:
        cv_perm = StratifiedGroupKFold(n_splits=max(n_splits, 2),
                                         shuffle=True,
                                         random_state=cfg.seed)
    else:
        cv_perm = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=cfg.seed)
    perm_res = teste_permutacao(
        lambda: fabrica_pipeline(n_opt),
        X_raw, Y_bin, y_int, cv_perm, cfg.n_permutacoes, cfg.seed,
        groups=grupos_cv, n_jobs=cfg.n_jobs_permutacao)
    perm_obs : float      = cast(float, perm_res["acc_observada"])
    perm_dist: np.ndarray = cast(np.ndarray, perm_res["accs_permutadas"])
    perm_p   : float      = cast(float, perm_res["p_value"])
    media_h0 = float(perm_dist.mean()) if len(perm_dist) > 0 else float("nan")
    print(f"  Bal.Acc observada = {perm_obs:.4f}  |  p = {perm_p:.4f}  "
          f"|  bal.acc media H0 = {media_h0:.4f}")
    print(f"  Iteracoes validas: {cast(int, perm_res['n_validos'])}/"
          f"{cfg.n_permutacoes}  "
          f"(failure_rate = {cast(float, perm_res['failure_rate']):.1%})")

    # --- 6b. Teste de Wold (R2Y / Q2Y intercept) --------------------------
    wold_res: Optional[Dict[str, object]] = None
    if cfg.executar_wold:
        print(f"\n[4b/7] Teste de Wold (R2Y/Q2Y intercept, "
              f"n={cfg.n_permutacoes_wold})")
        wold_res = teste_wold(
            lambda: fabrica_pipeline(n_opt),
            X_raw, Y_bin, y_int, cv_perm, cfg.n_permutacoes_wold, cfg.seed,
            groups=grupos_cv, n_jobs=cfg.n_jobs_permutacao)
        _wr2 = cast(float, wold_res['intercept_r2'])
        _wq2 = cast(float, wold_res['intercept_q2'])
        _wr2_s = f"{_wr2:.4f}" if np.isfinite(_wr2) else "n/a (permutacoes insuficientes)"
        _wq2_s = f"{_wq2:.4f}" if np.isfinite(_wq2) else "n/a (permutacoes insuficientes)"
        print(f"  R2Y obs = {cast(float, wold_res['r2_obs']):.4f}  |  "
              f"intercepto = {_wr2_s}  "
              f"{'VALIDO' if wold_res['valid_r2'] else 'FALHA'} (limiar < 0.40)")
        print(f"  Q2Y obs = {cast(float, wold_res['q2_obs']):.4f}  |  "
              f"intercepto = {_wq2_s}  "
              f"{'VALIDO' if wold_res['valid_q2'] else 'FALHA'} (limiar < 0.05)")

    # --- 6c. CV-ANOVA Eriksson --------------------------------------------
    cv_anova_res: Optional[Dict[str, float]] = None
    if cfg.executar_cv_anova:
        cv_anova_res = cv_anova_eriksson(Y_bin, Y_cv, n_opt)
        print(f"\n[4c/7] CV-ANOVA (Eriksson): F = {cv_anova_res['F']:.3f}  "
              f"p = {cv_anova_res['p_value']:.4g}  "
              f"(df = {cv_anova_res['df_model']}, {cv_anova_res['df_resid']})")

    # --- 7. Metricas e relatorio -------------------------------------------
    cm_mat = confusion_matrix(rotulos, pred_lab, labels=lb.classes_)
    metricas_finais = metricas_classificacao(rotulos, pred_lab, lb.classes_)
    print("\n[5/7] Metricas finais (CV):")
    for k, v in metricas_finais.items():
        print(f"  {k:>22s}: {v:.4f}")

    # --- 5b. BCa CI 95% para metricas via bootstrap estratificado ----------
    print(f"\n[5b/7] BCa CI 95% (n_boot={cfg.n_bootstrap_bca})")
    bca: Dict[str, Tuple[float, float, float]] = {}
    cls_arr = np.asarray(lb.classes_)
    metricas_funcoes = {
        "accuracy":          lambda yt, yp: accuracy_score(yt, yp),
        "balanced_accuracy": lambda yt, yp: balanced_accuracy_score(yt, yp),
        "f1_macro":          lambda yt, yp: f1_score(yt, yp, labels=cls_arr,
                                                       average="macro", zero_division=0),
        "cohen_kappa":       lambda yt, yp: cohen_kappa_score(yt, yp),
    }
    for nome, fn in metricas_funcoes.items():
        lo, hi, obs = bootstrap_bca_ci(rotulos, pred_lab, fn,
                                         n_boot=cfg.n_bootstrap_bca,
                                         alpha=0.05, seed=cfg.seed)
        bca[nome] = (lo, hi, obs)
        print(f"  {nome:>22s}: {obs:.4f}  [{lo:.4f}, {hi:.4f}]")
    print("\n" + str(classification_report(rotulos, pred_lab,
                                             target_names=lb.classes_,
                                             zero_division=0)))

    # --- 7b. R²X, R²Y, Q² ---------------------------------------------------
    r2x, r2y, q2 = metricas_modelo_pls(pls_final, X_processed, Y_bin, Y_cv)
    print(f"\n[5b/7] R2X = {r2x:.4f}  |  R2Y = {r2y:.4f}  |  Q2 = {q2:.4f}")

    # --- 8. Figuras --------------------------------------------------------
    print("\n[6/7] Gerando figuras...")
    aucs_roc: Dict[str, float] = {}
    # M1: mascara de puros para marcadores diferenciados
    # Pure samples: conc loaded as None -> NaN after asarray(float), OR stored as 0.0
    # Must handle BOTH cases to avoid false-negative mask for pure samples.
    _conc_f = np.asarray(conc, dtype=float) if conc is not None else None
    puros_mask_fig = (np.isnan(_conc_f) | (_conc_f == 0.0)) if _conc_f is not None else None
    # Flag de simbolos por classe (None -> todos circulo 'o')
    marcadores_fig = (mapa_marcadores if cfg.mostrar_marcadores_classe
                      else None)
    # ---- ESSENCIAIS (sempre) ----
    fig1_pca_scores(scores_pca, var_pca, rotulos, mapa_cores, cfg, pasta,
                     puros_mask=puros_mask_fig, mapa_marcadores=marcadores_fig)
    # ---- EXPLORATORIAS (so com figuras_detalhadas) ----
    if cfg.figuras_detalhadas:
        fig_hca_dendrograma(X_processed, rotulos, mapa_cores, cfg, pasta)
        fig_loadings_pca(pca, wavenumbers, cfg, pasta, n_pcs=2)
    if cfg.comparar_hca_pipelines:
        fig_hca_comparacao_pipelines(X_raw, rotulos, mapa_cores, cfg, pasta)
    fig2_plsda_scores(T_pls, var_lv_pls, rotulos, mapa_cores, cfg, pasta,
                       puros_mask=puros_mask_fig, mapa_marcadores=marcadores_fig)
    T2, Q, t2_lim, q_lim, out_t2, out_q = fig3_outliers(
        T_pls, P_pls, X_processed, rotulos, mapa_cores, n_opt, cfg, pasta)
    fig4_confusao(cm_mat, lb.classes_, rotulos, pred_lab, cfg, pasta)
    try:
        aucs_roc = fig_roc_auc(Y_bin, Y_cv, lb.classes_, cfg, pasta)
    except Exception as _e_roc:
        print(f"  [AVISO] ROC/AUC: {_e_roc}")
    # fig4b_metricas_globais e fig5_vip removidas: a primeira e redundante com
    # resumo_modelo.txt; a segunda (VIP puro) esta contida em fig_sprint3_sr_vip,
    # que mostra VIP + Selectivity Ratio lado a lado (ver abaixo).

    if cfg.n_bootstrap_vip > 0:
        print(f"  [bootstrap VIP estratificado, n={cfg.n_bootstrap_vip}]")
        boot = bootstrap_vip_estratificado(
            X_processed, Y_bin, y_int, n_opt,
            cfg.n_bootstrap_vip, cfg.seed,
            mae_id=grupos_cv)   # group-aware: respects mae_id replicates
        boot_validos = cast(int, boot["n_validos"])
        print(f"  Iteracoes validas: {boot_validos}/"
              f"{cfg.n_bootstrap_vip}  (falhos: {cast(int, boot['n_falhos'])})")
        if boot_validos > 0:
            fig5b_vip_estabilidade(boot, wavenumbers, top_n=20,
                                     cfg=cfg, pasta=pasta)
        else:
            print("  [AVISO] Bootstrap VIP: 0 iteracoes validas — fig5b pulada.")

    if cfg.figuras_detalhadas:
        fig6_preprocessamento(wavenumbers, X_raw, X_processed, rotulos,
                               mapa_cores, cfg, pasta)
    fig1_selecao_lvs(erros_rmsecv, metricas_por_lv, n_opt, cfg, pasta)

    # ---- Sprint 3 — SR (essencial) + Score Contribution (detalhada) -----
    print("\n[Sprint3] Selectivity Ratio + Score Contribution...")
    sr = calcular_selectivity_ratio(pls_final, X_processed)
    fig_sprint3_sr_vip(vip, sr, wavenumbers, top_n=20, cfg=cfg, pasta=pasta)
    if cfg.figuras_detalhadas:
        fig_sprint3_score_contribution(pls_final, X_processed, rotulos,
                                        wavenumbers, mapa_cores, top_n=20,
                                        cfg=cfg, pasta=pasta)

    # DD-SIMCA — configurable training mode (v14).
    #   'todos' (default): trains each model on all class samples
    #     (exploratory; robust with few pure samples). sens = fraction of
    #     own class accepted.
    #   'puros': true one-class N2 (trains only on pure samples) — requires
    #     >=15 pure samples/class. With 3/class it generates a tiny region
    #     and ~all samples become 'Unknown'. sens = pure accepted; esp = adult rejected.
    ddsimca_res: Optional[Dict[str, Dict[str, Any]]] = None
    simca_pred: np.ndarray = np.array([], dtype=str)
    ddsimca_sens_esp: Dict[str, Tuple[float, float, int, int]] = {}
    modo_dd: str = "todos"  # default; overwritten if executar_ddsimca=True
    if cfg.executar_ddsimca:
        modo_dd = (cfg.ddsimca_treinar_em or "todos").lower()
        if conc is not None:
            # Pure samples: conc loaded as None -> NaN after asarray(float), OR 0.0.
            # Use both conditions — NaN == 0.0 is False, causing pure samples to be
            # silently excluded from DD-SIMCA training (sens=0% for all classes).
            _conc_dd = np.asarray(conc, dtype=float)
            mask_puros = np.isnan(_conc_dd) | (_conc_dd == 0.0)
        else:
            mask_puros = np.ones(len(rotulos), dtype=bool)

        if modo_dd == "puros":
            mask_treino = mask_puros
            n_treino = int(mask_puros.sum())
        else:
            mask_treino = np.ones(len(rotulos), dtype=bool)
            n_treino = len(rotulos)

        print(f"\n[Sprint3] DD-SIMCA "
              f"(n_components={cfg.ddsimca_n_components}, alpha=0.05, "
              f"ucl={cfg.ddsimca_ucl_method}, treino='{modo_dd}') "
              f"— {n_treino} amostras de treino")
        if modo_dd == "puros" and n_treino < (cfg.ddsimca_n_components + 2) * 2:
            print(f"  [AVISO] Poucos puros ({n_treino}) para one-class robusto"
                  f" (recomendado >=15/classe). Resultados exploratorios.")

        ddsimca = DDSimca(n_components=cfg.ddsimca_n_components, alpha=0.05,
                           ucl_method=cfg.ddsimca_ucl_method)
        ddsimca.fit(X_processed[mask_treino], rotulos[mask_treino])
        ddsimca_res   = ddsimca.score_matrix(X_processed)   # prediz em TODOS
        simca_pred    = ddsimca.predict(X_processed)
        n_unknown     = int(np.sum(simca_pred == "Desconhecido"))
        n_ambig       = int(np.sum(simca_pred == "Ambiguo"))

        print(f"  {'Classe':18s} {'sens':>7s} {'esp(adult)':>11s}")
        for cls in classes_unicas:
            if cls not in ddsimca_res:
                continue
            m = ddsimca_res[cls]
            aceito = ((np.asarray(m["T2_norm"]) <= 1.0) &
                      (np.asarray(m["Q_norm"])  <= 1.0))
            idx_puro_c  = (rotulos == cls) & mask_puros
            idx_adult_c = (rotulos == cls) & (~mask_puros)
            idx_cls     = (rotulos == cls)
            n_puro_c    = int(idx_puro_c.sum())
            n_adult_c   = int(idx_adult_c.sum())
            # sensitivity: 'puros' mode uses pure samples; 'todos' uses the whole class
            if modo_dd == "puros" and n_puro_c > 0:
                sens = float(np.mean(aceito[idx_puro_c]))
            else:
                sens = float(np.mean(aceito[idx_cls]))
            # B4: in 'todos' mode adulterated samples are IN TRAINING, so
            # "specificity" would be in-sample (not authentication) and misleading.
            # Only reported in 'puros' mode (true one-class).
            if modo_dd == "puros":
                esp = (float(np.mean(~aceito[idx_adult_c]))
                       if n_adult_c > 0 else float("nan"))
            else:
                esp = float("nan")
            ddsimca_sens_esp[cls] = (sens, esp, n_puro_c, n_adult_c)
            esp_txt = f"{esp*100:9.1f}%" if esp == esp else "      n/a"
            print(f"  {cls:18s} {sens*100:6.1f}% {esp_txt}"
                  f"   (puros={n_puro_c}, adult={n_adult_c})")
        print(f"  Desconhecidos: {n_unknown}  |  Ambiguos: {n_ambig}")
        if ddsimca_res:
            # Essencial: painel de aceitacao consolidado (todas as classes).
            fig_sprint3_ddsimca_acceptance(
                ddsimca_res, rotulos, mapa_cores, cfg, pasta,
                sens_esp=ddsimca_sens_esp)
            if cfg.figuras_detalhadas:
                # Detalhadas: um plot por classe (subpasta ddsimca/) + Cooman.
                fig_ddsimca_individuais(
                    ddsimca_res, rotulos, mapa_cores, cfg, pasta,
                    sens_esp=ddsimca_sens_esp)
                if len(ddsimca_res) >= 2:
                    try:
                        fig_cooman_ddsimca(ddsimca_res, rotulos, mapa_cores,
                                           cfg, pasta)
                    except Exception as _e_coom:
                        print(f"  [AVISO] Cooman's Plot: {_e_coom}")

    # OPLS-DA
    _opls_n_ortho: Optional[int] = None
    if cfg.executar_opls:
        n_cls_opls = len(classes_unicas)
        print(f"\n[Sprint3] OPLS-DA "
              f"(n_ortho={cfg.n_ortho_opls}, {n_cls_opls} classes)...")
        if n_cls_opls < 2:
            print("  [AVISO] OPLS-DA requer >= 2 classes.")
        else:
            try:
                opls = OPLSDAWrapper(n_ortho=cfg.n_ortho_opls)
                opls.fit(X_processed, Y_bin)
                t_pred_opls, t_orth_opls = opls.transform(X_processed)
                fig_sprint3_opls_scores(
                    t_pred_opls, t_orth_opls, rotulos, mapa_cores,
                    opls.n_ortho_fitted_, cfg=cfg, pasta=pasta)
                fig_splot_opls(X_processed, t_pred_opls, wavenumbers, cfg, pasta)
                _opls_n_ortho = opls.n_ortho_fitted_
                print(f"  Componentes ortogonais ajustados: {_opls_n_ortho}")
            except Exception as _e_opls:
                print(f"  [ERRO] OPLS-DA: {_e_opls}")

    # --- STAGE 4: Variable Selection ------------------------------------
    etapa4_res: Optional[Dict[str, Any]] = None
    if cfg.executar_etapa4:
        try:
            etapa4_res = etapa4_selecao_variaveis(
                X_processed, Y_bin, y_int, vip, sr, wavenumbers,
                cv_indices, n_opt, cfg, pasta, pasta_dados)
        except Exception as _e_e4:
            print(f"  [ERRO] Etapa 4: {_e_e4}")

    if cfg.comparar_pipelines:
        print("\n[6b/7] Comparacao de pipelines de pre-processamento...")
        comp = comparar_pipelines(cfg, X_raw, Y_bin, y_int, cv_indices,
                                    max_lv=cfg.max_lvs)
        fig_extra_comparacao_pipelines(comp, cfg, pasta)
        pd.DataFrame(comp).T.to_csv(
            os.path.join(pasta_dados, "comparacao_pipelines.csv"),
            sep=";", decimal=",")

    if wold_res is not None and cast(int, wold_res["n_validos"]) > 2:
        fig_extra_wold(wold_res, cfg, pasta)

    # --- 8b. Avaliacao em holdout independente ----------------------------
    metricas_holdout: Optional[Dict[str, float]] = None
    bca_holdout:      Optional[Dict[str, Tuple[float, float, float]]] = None
    if X_holdout is not None and rotulos_holdout is not None:
        rot_ho: np.ndarray = rotulos_holdout
        print(f"\n[6c/7] Avaliacao em holdout ({n_holdout} amostras)...")
        try:
            X_holdout_proc = preproc_full.transform(X_holdout)
            Y_holdout_hat = pls_final.predict(X_holdout_proc)
            pred_holdout = lb.classes_[np.argmax(Y_holdout_hat, axis=1)]
            cm_holdout = confusion_matrix(rot_ho, pred_holdout,
                                            labels=lb.classes_)
            metricas_holdout = metricas_classificacao(
                rot_ho, pred_holdout, lb.classes_)
            for k, v in metricas_holdout.items():
                print(f"  {k:>22s}: {v:.4f}")
            fig_extra_holdout(metricas_finais, metricas_holdout,
                                cm_holdout, lb.classes_, n_holdout, cfg, pasta)
            # BCa CI no holdout tambem
            bca_holdout = {}
            for nome, fn in metricas_funcoes.items():
                lo, hi, obs = bootstrap_bca_ci(
                    rot_ho, pred_holdout, fn,
                    n_boot=cfg.n_bootstrap_bca, alpha=0.05,
                    seed=cfg.seed + 1)
                bca_holdout[nome] = (lo, hi, obs)
        except Exception as e:
            print(f"  [ERRO] Avaliacao em holdout falhou: {e}")
            metricas_holdout = None

    # --- 9. Identificadores e resumo (separados dos graficos) --------------
    salvar_identificadores(rotulos, pred_lab, T_pls, T2, Q,
                            t2_lim, q_lim, pasta_dados)
    print(f"  -> {os.path.join(pasta_dados, 'amostras_identificadores.csv')}")

    # Formata strings de CI para o resumo
    def _ci_str(b):
        if b is None: return "-"
        lo, hi, obs = b
        if not np.isfinite(lo): return f"{obs:.4f} [CI indisponivel]"
        return f"{obs:.4f} [{lo:.4f}, {hi:.4f}]"

    _preset_str = (cfg.preprocessamento_padrao or "custom").lower()
    if _preset_str in ("snv_sg_mc", "msc_sg_mc"):
        _scat = "SNV" if _preset_str == "snv_sg_mc" else "MSC"
        _pp_descr = (f"{_scat} -> SG(w={cfg.sg_window},p={cfg.sg_polyorder},"
                      f"d={cfg.sg_deriv}) -> mean-centering")
    elif _preset_str == "autoscaling":
        _pp_descr = "Autoscaling (mean + unit variance)"
    elif _preset_str == "mc":
        _pp_descr = "Mean-centering"
    else:
        _pp_descr = " -> ".join(
            (["SNV"] if cfg.aplicar_snv else []) +
            ([f"SG(w={cfg.sg_window},p={cfg.sg_polyorder},d={cfg.sg_deriv})"]
             if cfg.aplicar_sg else []) +
            (["mean-centering"] if cfg.aplicar_mc else []))
    resumo = {
        "Total de amostras":      int(X_raw.shape[0]),
        "Total de variaveis":     int(X_raw.shape[1]),
        "Total de classes":       int(len(classes_unicas)),
        "Metodo":                 "PLS-DA",
        "Pre-processamento":      _pp_descr,
        "Faixa espectral (cm-1)": f"[{cfg.wn_min:.0f}, {cfg.wn_max:.0f}]",
        "LVs otimas":             int(n_opt),
        "LVs no teto (max_lvs)":  ("SIM - aumente max_lvs" if lvs_no_teto
                                    else "nao"),
        "Validacao":              cv_label + (
            f", repeats={cfg.n_repeats_cv}" if cfg.n_repeats_cv > 1 and
            not usar_grupos else ""),
        "Group-aware (mae_id)":   "sim" if usar_grupos else "nao",
        "N grupos mae_id":        (int(len(np.unique(mae_id)))
                                    if mae_id is not None else 0),
        "Nivel":                  f"{cfg.nivel} ({_NIVEL_NOME.get(cfg.nivel, cfg.nivel)})",
        "Tag":                    cfg.tag if cfg.tag else "-",
        "Accuracy (CV)":          float(metricas_finais["accuracy"]),
        "Balanced accuracy":      float(metricas_finais["balanced_accuracy"]),
        "F1 (macro)":             float(metricas_finais["f1_macro"]),
        "Cohen's kappa":          float(metricas_finais["cohen_kappa"]),
        "R2X":                    float(r2x),
        "R2Y":                    float(r2y),
        "Q2":                     float(q2),
        "Permutation p-value":    float(perm_p),
        "Permutation n_validos":  cast(int, perm_res["n_validos"]),
        "Permutation n_falhos":   cast(int, perm_res["n_falhos"]),
        "Permutation failure_rate": cast(float, perm_res["failure_rate"]),
        "Hotelling T2 (95%)":     float(t2_lim),
        # B7: notacao adaptativa — com SG-derivada + muitas LVs, Q_lim e
        # minusculo e ":.4f" exibia 0.0000 (mascarando o valor real).
        "Q-residual (95%)":       (f"{q_lim:.4g}" if abs(q_lim) < 1e-3
                                    else f"{q_lim:.4f}"),
        "N outliers T2":          int(out_t2.size),
        "N outliers Q":           int(out_q.size),
        "Imbalance ratio":        cast(float, relatorio_balanco["imbalance_ratio"]),
        "Classe maior":           cast(int, relatorio_balanco["n_max"]),
        "Classe menor":           cast(int, relatorio_balanco["n_min"]),
        "Integridade NaN":        cast(int, relatorio_entrada["n_nan_amostras"]),
        "Integridade Inf":        cast(int, relatorio_entrada["n_inf_amostras"]),
        "Variaveis constantes":   cast(int, relatorio_entrada["n_constantes_removidas"]),
        "Duplicatas exatas":      cast(int, relatorio_entrada["n_duplicatas_exatas"]),
        "Duplicatas aproximadas": cast(int, relatorio_entrada["n_duplicatas_aproximadas"]),
        "BCa Accuracy":           _ci_str(bca.get("accuracy")),
        "BCa Balanced acc.":      _ci_str(bca.get("balanced_accuracy")),
        "BCa F1 (macro)":         _ci_str(bca.get("f1_macro")),
        "BCa Cohen's kappa":      _ci_str(bca.get("cohen_kappa")),
    }
    # ROC/AUC (v24)
    if aucs_roc:
        resumo["ROC AUC macro (OvR)"] = float(aucs_roc.get("macro", float("nan")))
        resumo["--- ROC AUC por classe ---"] = ""
        for _cls, _auc in aucs_roc.items():
            if _cls != "macro":
                resumo[f"  AUC {_cls}"] = float(_auc)
    if wold_res is not None:
        resumo["Wold R2Y intercept"]    = cast(float, wold_res["intercept_r2"])
        resumo["Wold Q2Y intercept"]    = cast(float, wold_res["intercept_q2"])
        resumo["Wold R2Y valido (<.40)"] = cast(bool,  wold_res["valid_r2"])
        resumo["Wold Q2Y valido (<.05)"] = cast(bool,  wold_res["valid_q2"])
    if cv_anova_res is not None:
        resumo["CV-ANOVA F"]      = float(cv_anova_res["F"])
        resumo["CV-ANOVA p"]      = float(cv_anova_res["p_value"])
    if X_holdout is not None and metricas_holdout is not None:
        resumo["Holdout n"]          = int(n_holdout)
        resumo["Holdout accuracy"]   = float(metricas_holdout["accuracy"])
        resumo["Holdout balanced acc"] = float(metricas_holdout["balanced_accuracy"])
        resumo["Holdout F1 (macro)"] = float(metricas_holdout["f1_macro"])
        resumo["Holdout Cohen kappa"] = float(metricas_holdout["cohen_kappa"])
        if bca_holdout is not None:
            resumo["BCa Holdout Accuracy"] = _ci_str(bca_holdout.get("accuracy"))
            resumo["BCa Holdout Bal.acc"]  = _ci_str(bca_holdout.get("balanced_accuracy"))
    # Sprint 3 — append to summary after dict already exists
    if cfg.executar_ddsimca and ddsimca_res is not None:
        resumo["DD-SIMCA n_components"]    = int(cfg.ddsimca_n_components)
        resumo["DD-SIMCA n_desconhecidos"] = int(np.sum(simca_pred == "Desconhecido"))
        resumo["DD-SIMCA n_ambiguos"]      = int(np.sum(simca_pred == "Ambiguo"))
        # B4 — honest training mode label. In 'todos' mode, sens/spec
        # are IN-SAMPLE acceptance, NOT one-class authentication metrics.
        resumo["DD-SIMCA modo treino"] = (
            modo_dd + (" (one-class)" if modo_dd == "puros"
                       else " (in-sample; sens/esp NAO sao autenticacao)"))
        # C4 — sensibilidade/especificidade one-class por classe
        for cls in classes_unicas:
            if cls in ddsimca_sens_esp:
                s_c, e_c, npc, nac = ddsimca_sens_esp[cls]
                esp_s = f"{e_c*100:.1f}%" if e_c == e_c else "n/a"
                resumo[f"DD-SIMCA {cls} sens/esp"] = (
                    f"{s_c*100:.1f}% / {esp_s} "
                    f"(puros={npc}, adult={nac})")
    if _opls_n_ortho is not None:
        resumo["OPLS-DA n_ortho"] = int(_opls_n_ortho)

    # Stage 4 — variable selection
    if etapa4_res is not None:
        resumo["--- Etapa 4: selecao de variaveis ---"] = ""
        for t in etapa4_res["tabela"]:
            _q2_str = f"{t['q2']:.3f}" if np.isfinite(t['q2']) else "n/a"
            resumo[f"  {t['metodo']}"] = (
                f"bal.acc={t['balanced_accuracy']:.3f} | "
                f"Q2={_q2_str} | {t['n_vars']} vars | {t['n_lv']} LVs")
        mlhr = etapa4_res["melhor"]
        resumo["  >> Mais parcimonioso"] = (
            f"{mlhr['metodo']} ({mlhr['n_vars']} vars, "
            f"bal.acc={mlhr['balanced_accuracy']:.3f})")

    # C6 — Outliers T2 por classe (diagnostico de batch/lote)
    resumo["--- Outliers T2 por classe ---"] = ""
    set_out_t2 = set(out_t2.tolist())
    for cls in classes_unicas:
        idx_cls = np.where(rotulos == cls)[0]
        n_out_c = int(sum(1 for i in idx_cls if i in set_out_t2))
        n_tot_c = int(len(idx_cls))
        frac_c  = n_out_c / max(n_tot_c, 1)
        flag = "  <-- ANOMALO (>30%)" if frac_c > 0.30 else ""
        resumo[f"  T2-outliers {cls}"] = (
            f"{n_out_c}/{n_tot_c} ({frac_c*100:.1f}%){flag}")
        if frac_c > 0.30:
            print(f"[ATENCAO] Classe '{cls}': {frac_c*100:.0f}% outliers T2 "
                  f"({n_out_c}/{n_tot_c}) — possivel batch/lote anomalo.")

    # M4 — Accuracy (recall) por classe
    resumo["--- Accuracy por classe ---"] = ""
    rec_por_classe = recall_score(rotulos, pred_lab, labels=lb.classes_,
                                   average=None, zero_division=0)
    for cls, rec_c in zip(lb.classes_, np.asarray(rec_por_classe)):
        resumo[f"  Acc {cls}"] = float(rec_c)

    salvar_resumo_modelo(pasta_logs, resumo)
    print(f"  -> {os.path.join(pasta_logs, 'resumo_modelo.txt')}")

    # --- 9a. Auto-Benchmark (opcional) ─────────────────────────────────────
    if cfg.executar_benchmark:
        print("\n[7b/7] Auto-Benchmark (SVM / RF / XGBoost vs PLS-DA)...")
        # Guarda: ~1.2 GB pico (SVM kernel matrix + OOF proba)
        if _verificar_ram(1.2, "Auto-Benchmark"):
            try:
                bench_df = benchmark_classificadores(
                    X_raw, y_int, grupos_cv, lb, n_opt, cfg, pasta,
                    wavenumbers=wavenumbers)
                print(bench_df.to_string(index=False))
            except Exception as _e_bench:
                print(f"  [AVISO] Benchmark falhou: {_e_bench}")

    # --- 9a2. Monte Carlo CV (opcional) ────────────────────────────────────
    if cfg.executar_monte_carlo:
        print("\n[7c/7] Monte Carlo CV (IC95% por percentil)...")
        # Guarda: ~400 MB (PLS-DA x N splits em serie)
        if _verificar_ram(0.5, "Monte Carlo CV"):
            try:
                mc_df = monte_carlo_cv(
                    X_raw, y_int, grupos_cv, lb, n_opt, cfg, pasta)
                print(mc_df.to_string(index=False))
            except Exception as _e_mc:
                print(f"  [AVISO] Monte Carlo CV falhou: {_e_mc}")

    # --- 9b. Exportar modelo final (modelos/) — joblib opcional -----------
    try:
        import joblib
        preproc_export = construir_preprocessador(cfg).fit(X_raw)
        pacote_modelo = {
            "preprocessador": preproc_export,
            "pls_final":      pls_final,
            "label_binarizer": lb,
            "classes":        list(lb.classes_),
            "wavenumbers":    wavenumbers,
            "n_opt":          int(n_opt),
            "preset":         cfg.preprocessamento_padrao,
            "wn_min":         cfg.wn_min, "wn_max": cfg.wn_max,
            # v25: limites para diagnosticos em novos dados (Aba Predicao)
            "t2_ucl":         float(t2_lim),
            "q_ucl":          float(q_lim),
        }
        cam_modelo = os.path.join(pasta_modelos, "modelo_plsda.joblib")
        joblib.dump(pacote_modelo, cam_modelo)
        print(f"  -> {cam_modelo}")
    except Exception as _e_mod:
        print(f"  [AVISO] Exportacao do modelo pulada: {_e_mod}")

    if out_t2.size or out_q.size:
        print(f"\n[INFO] Outliers (T2 > lim): {out_t2.tolist()}")
        print(f"[INFO] Outliers (Q  > lim): {out_q.tolist()}")

    # --- 9. PLS regression (optional) -------------------------------------
    # Guard: only runs if (1) concentration data present, (2) no NaN,
    # (3) variance > 0, (4) at least 10 samples with concentration > 0,
    # (5) nivel is N2 or N3 — in N1 the inter-species spectral variation
    #     (~90 % of variance) completely overwhelms the adulteration signal,
    #     producing R²≈0 and a "predict-the-mean" model.
    # (6) Single-species data (mae_id prefix check) — multi-species N2 is
    #     also confounded; the model sees species identity as a stronger
    #     signal than adulteration level.
    _pls_reg_ok = False
    if conc is not None:
        conc_arr = np.asarray(conc, dtype=float)
        n_nan    = int(np.isnan(conc_arr).sum())
        n_nonzero = int(np.sum(conc_arr > 0))

        if cfg.nivel == "N1":
            print(f"\n[7/7] PLS regressao — PULADA: nivel=N1. "
                  f"Regressao de concentracao requer nivel=N2 ou N3 "
                  f"(N1 mistura {len(classes_unicas)} especies cujos "
                  f"espectros dominam o sinal de adulteracao).")
        elif n_nan > 0:
            print(f"\n[7/7] PLS regressao — PULADA: {n_nan} amostras com "
                  f"conc=NaN.")
        elif float(conc_arr.std()) < 1e-8:
            print("\n[7/7] PLS regressao — PULADA: variancia de conc ~= 0.")
        elif n_nonzero < 10:
            print(f"\n[7/7] PLS regressao — PULADA: apenas {n_nonzero} "
                  f"amostras com teor > 0 (precisa >= 10).")
        else:
            # Guard (6): check number of unique species via mae_id prefix
            # (first 3 uppercase chars = species code: AND, ACA, BCB …)
            n_especies = 1
            if mae_id is not None:
                prefixos = {str(m)[:3].upper() for m in mae_id}
                n_especies = len(prefixos)
            elif rotulos is not None:
                n_especies = len(np.unique(rotulos))

            if n_especies > 1:
                # Multi-species: a SINGLE pooled model is confounded by inter-
                # species variation. Instead, calibrate PER SPECIES and pool
                # the predictions — recovers the proper diagonal.
                print(f"\n[7/7] PLS regressao POR ESPECIE "
                      f"({n_especies} especies — calibracao separada para "
                      f"evitar confusao inter-especies)")
                try:
                    reg_esp = pls_regressao_por_especie(
                        X_raw, conc_arr, rotulos, mae_id, classes_unicas,
                        cfg, pasta, n_splits)
                    if reg_esp is not None:
                        print(f"  Especies modeladas: {reg_esp['n_especies']}")
                        print(f"  R2cal (pooled): {reg_esp['r2c']:.4f}  |  "
                              f"R2val (pooled): {reg_esp['r2v']:.4f}")
                        print(f"  RMSEP (pooled): {reg_esp['rmsep']:.3f}")
                        for t in reg_esp["tabela_especie"]:
                            print(f"    {t['especie']:18s} "
                                  f"LV={t['n_lv']:2d}  RMSEP={t['rmsep']:.2f}  "
                                  f"R2val={t['r2val']:.3f}  "
                                  f"(cal={t['n_cal']}, val={t['n_val']})")
                            _lod_t, _loq_t = t.get("lod"), t.get("loq")
                            if _lod_t is not None and np.isfinite(_lod_t):
                                print(f"      LOD={_lod_t:.2f}%  LOQ={_loq_t:.2f}%  "
                                      f"SEN={t['sensibilidade']:.3f}  "
                                      f"gamma={t['sensibilidade_analitica']:.2f}  "
                                      f"SEL={t['seletividade_media']:.3f}")
                            else:
                                print("      LOD/LOQ: N/A (sem replicas fisicas "
                                      "suficientes para estimar ruido instrumental)")
                    else:
                        print("  [AVISO] Nenhuma especie com amostras "
                              "suficientes para regressao (>= 6 adulteradas "
                              "e variancia de teor > 0).")
                except Exception as _e_reg_esp:
                    print(f"  [AVISO] Regressao por especie falhou: {_e_reg_esp}")
                _pls_reg_ok = False   # per-species path handled the figure
            else:
                _pls_reg_ok = True

    if _pls_reg_ok and conc is not None:
        print(f"\n[7/7] PLS regressao "
              f"(target=teor%, {int(np.sum(conc > 0))} adulterados + "
              f"{int(np.sum(conc == 0))} puros)")
        Y_reg = np.asarray(conc, dtype=float).reshape(-1, 1)

        # Calibration/validation split — group-aware if mae_id available
        # (T1/T2/T3 replicates of the same sample point never split between cal/val).
        if mae_id is not None:
            gss_reg = GroupShuffleSplit(n_splits=1, train_size=cfg.frac_cal,
                                         random_state=cfg.seed)
            ic, iv = next(gss_reg.split(X_raw, Y_reg, groups=mae_id))
            print(f"  Split cal/val: GroupShuffleSplit por mae_id "
                  f"({len(ic)} cal / {len(iv)} val)")
        else:
            rng   = np.random.default_rng(cfg.seed)
            idx_p = rng.permutation(len(conc))
            n_cal = int(cfg.frac_cal * len(conc))
            ic, iv = idx_p[:n_cal], idx_p[n_cal:]
        Xc_raw, Yc = X_raw[ic], Y_reg[ic]
        Xv_raw, Yv = X_raw[iv], Y_reg[iv]

        lv_max = min(cfg.max_lvs, max(2, Xc_raw.shape[0] // 5))

        # CV interna — GroupKFold por mae_id (C5: nao vaza replicas)
        if mae_id is not None:
            grupos_cal   = mae_id[ic]
            n_grupos_cal = int(len(np.unique(grupos_cal)))
            n_splits_reg = max(2, min(n_splits, n_grupos_cal))
            cv_reg = GroupKFold(n_splits=n_splits_reg)
            grupos_cv_reg: Optional[np.ndarray] = grupos_cal
            print(f"  CV interna: GroupKFold n_splits={n_splits_reg} "
                  f"({n_grupos_cal} grupos na calibracao)")
        else:
            n_splits_reg = max(2, min(n_splits, Xc_raw.shape[0] // 2))
            cv_reg = KFold(n_splits=n_splits_reg, shuffle=True,
                            random_state=cfg.seed)
            grupos_cv_reg = None

        erros_reg = []
        preds_reg = []
        for n in range(1, lv_max + 1):
            pipe = Pipeline([
                ("preproc", construir_preprocessador(cfg)),
                ("pls", PLSRegression(n_components=n, scale=False)),
            ])
            Y_hat = cross_val_predict(pipe, Xc_raw, Yc, cv=cv_reg,
                                       groups=grupos_cv_reg)
            erros_reg.append(rmse_flat(Yc, Y_hat))
            preds_reg.append(Y_hat)

        n_opt_reg = int(np.argmin(erros_reg)) + 1
        pipe_final = Pipeline([
            ("preproc", construir_preprocessador(cfg)),
            ("pls", PLSRegression(n_components=n_opt_reg, scale=False)),
        ]).fit(Xc_raw, Yc)
        Yc_hat = pipe_final.predict(Xc_raw)
        Yv_hat = pipe_final.predict(Xv_raw)
        Yc_cv  = preds_reg[n_opt_reg - 1]

        rmsec  = rmse_flat(Yc, Yc_hat)
        rmsecv = rmse_flat(Yc, Yc_cv)
        rmsep  = rmse_flat(Yv, Yv_hat)
        bias_v = float(np.mean(np.asarray(Yv_hat).flatten()
                                - np.asarray(Yv).flatten()))
        r2c    = float(r2_score(Yc, Yc_hat))
        r2v    = float(r2_score(Yv, Yv_hat))

        fig7_pls_regressao(Yc, Yc_hat, Yv, Yv_hat, erros_reg, n_opt_reg,
                            r2c, r2v, rmsec, rmsecv, rmsep, bias_v, cfg, pasta)

        print(f"  LVs    : {n_opt_reg}")
        print(f"  RMSEC  : {rmsec:.3f}  |  RMSECV: {rmsecv:.3f}  "
              f"|  RMSEP: {rmsep:.3f}")
        print(f"  R2cal  : {r2c:.4f}  |  R2val : {r2v:.4f}  "
              f"|  Bias: {bias_v:.4f}")

        # Figuras de merito analiticas (Valderrama, Braga & Poppi, 2009):
        # ruido instrumental estimado a partir de replicas fisicas (T1/T2/T3
        # via mae_id) SOMENTE do lado de calibracao.
        _preproc_ajustado_reg = pipe_final.named_steps["preproc"]
        _X_cal_proc_reg = np.asarray(_preproc_ajustado_reg.transform(Xc_raw))
        _grupos_rep_reg = _agrupar_replicas_processadas(
            Xc_raw, mae_id[ic] if mae_id is not None else None,
            _preproc_ajustado_reg)
        _fom_reg = figuras_merito_regressao(
            pipe_final.named_steps["pls"], _X_cal_proc_reg, _grupos_rep_reg)
        if np.isfinite(_fom_reg["lod"]):
            print(f"  LOD    : {_fom_reg['lod']:.2f}%  |  "
                  f"LOQ: {_fom_reg['loq']:.2f}%")
            print(f"  SEN    : {_fom_reg['sensibilidade']:.3f}  |  "
                  f"gamma: {_fom_reg['sensibilidade_analitica']:.2f}  |  "
                  f"SEL: {_fom_reg['seletividade_media']:.3f}")
        else:
            print("  LOD/LOQ: N/A (sem replicas fisicas suficientes para "
                  "estimar ruido instrumental)")
    elif conc is None:
        print("\n[7/7] PLS regressao — pulado (sem coluna de concentracao)")

    print(f"\n{'=' * 60}")
    print("  Pipeline concluido.")
    print(f"  Resultados em: {pasta}")
    print(f"{'=' * 60}")


# =========================================================================
#  CAMADA ACESSIVEL (v23) — configuracao em YAML + assistente de terminal
#  Objetivo: usar o pipeline SEM editar o codigo. Toda a configuracao de
#  usuario fica em config.yaml (linguagem simples). O assistente CMD
#  (menu_interativo) le/edita/salva esse arquivo e dispara executar().
#  Fonte UNICA de verdade: _CONFIG_SPEC mapeia campo amigavel <-> Config,
#  e alimenta tanto o YAML quanto o menu (e, depois, o app web).
# =========================================================================

_PRE_PROC_FRIENDLY: Dict[str, str] = {
    "MSC+SG+MC":      "msc_sg_mc",
    "SNV+SG+MC":      "snv_sg_mc",
    "Autoscaling":    "autoscaling",
    "Mean-centering": "mc",
}
_PRE_PROC_INV: Dict[str, str] = {v: k for k, v in _PRE_PROC_FRIENDLY.items()}

# Cada campo: chave_yaml, atributo da Config, tipo, descricao, opcoes.
#   tipo in {"str","int","float","bool","list","choice","preproc"}
_CONFIG_SPEC: List[Dict[str, Any]] = [
    {"key": "modo_entrada", "attr": "modo", "tipo": "choice",
     "desc": "Origem dos dados: dx (JCAMP-DX, FT-NIR) | csv (tabela generica) | sintetico (teste)",
     "opcoes": ["dx", "csv", "sintetico"]},
    {"key": "pasta_dados", "attr": "pasta_entrada", "tipo": "str",
     "desc": "Pasta com os arquivos .dx (modo dx; uma subpasta por classe)", "opcoes": None},
    {"key": "arquivo_csv", "attr": "arquivo_csv", "tipo": "str",
     "desc": "Caminho do CSV (modo csv): colunas espectrais/variaveis + 1 coluna de classe", "opcoes": None},
    {"key": "coluna_classe", "attr": "coluna_classe", "tipo": "str",
     "desc": "Nome da coluna de classe/rotulo no CSV (modo csv)", "opcoes": None},
    {"key": "coluna_concentracao", "attr": "coluna_conc", "tipo": "str_opcional",
     "desc": "Nome da coluna de concentracao no CSV (vazio se nao houver; modo csv)", "opcoes": None},
    {"key": "pasta_saida", "attr": "pasta_saida_raiz", "tipo": "str",
     "desc": "Pasta onde os resultados serao gravados", "opcoes": None},
    {"key": "nivel", "attr": "nivel", "tipo": "choice",
     "desc": "Modo de analise: Classificacao (especie) | Discriminacao "
             "(puro vs. adulterado) | Quantificacao (teor de adulterante)",
     "opcoes": ["N1", "N2", "N3"]},
    {"key": "pre_processamento", "attr": "preprocessamento_padrao", "tipo": "preproc",
     "desc": "Pre-processamento espectral", "opcoes": list(_PRE_PROC_FRIENDLY)},
    {"key": "faixa_min_cm", "attr": "wn_min", "tipo": "float",
     "desc": "Inicio da faixa espectral util (cm-1)", "opcoes": None, "min": 0.0},
    {"key": "faixa_max_cm", "attr": "wn_max", "tipo": "float",
     "desc": "Fim da faixa espectral util (cm-1)", "opcoes": None, "min": 0.0},
    {"key": "excluir_classes", "attr": "excluir_classes", "tipo": "list",
     "desc": "Especies a remover da analise (ex: [Copaiba])", "opcoes": None},
    {"key": "max_lvs", "attr": "max_lvs", "tipo": "int",
     "desc": "Numero maximo de variaveis latentes (LVs) testadas", "opcoes": None,
     "min": 1, "max": 200},
    {"key": "holdout_fracao", "attr": "frac_holdout", "tipo": "float",
     "desc": "Fracao reservada para teste externo (0 a 0.5)", "opcoes": None,
     "min": 0.0, "max": 0.5},
    {"key": "validacao_group_aware", "attr": "agrupar_por_mae_id", "tipo": "bool",
     "desc": "Manter replicas (T1/T2/T3) juntas na validacao (evita vazamento)", "opcoes": None},
    {"key": "n_permutacoes", "attr": "n_permutacoes", "tipo": "int",
     "desc": "Iteracoes do teste de permutacao", "opcoes": None, "min": 1, "max": 100000},
    {"key": "teste_wold", "attr": "executar_wold", "tipo": "bool",
     "desc": "Rodar teste de Wold (intercepts R2Y/Q2Y)", "opcoes": None},
    {"key": "n_jobs_permutacao", "attr": "n_jobs_permutacao", "tipo": "int",
     "desc": "Processos paralelos para os testes de permutacao/Wold "
             "(1 = sequencial, resultado identico; so muda o tempo). "
             "Medido: 4 processos = ~2x mais rapido no pipeline completo; "
             "acima disso o ganho cai (overhead de criar processos). "
             "Use 1 em ambientes com pouca RAM/CPU (ex.: Streamlit Cloud gratuito)",
     "opcoes": None, "min": 1, "max": 64},
    {"key": "teste_cv_anova", "attr": "executar_cv_anova", "tipo": "bool",
     "desc": "Rodar CV-ANOVA (Eriksson)", "opcoes": None},
    {"key": "selecao_variaveis_etapa4", "attr": "executar_etapa4", "tipo": "bool",
     "desc": "Rodar Etapa 4 (iPLS / VIP / SR / sPLS-DA)", "opcoes": None},
    {"key": "selecao_spa", "attr": "executar_spa", "tipo": "bool",
     "desc": "Etapa 4: rodar tambem SPA/APS (Algoritmo das Projecoes "
             "Sucessivas, Araujo et al. 2001) — mais lento que iPLS/VIP/SR",
     "opcoes": None},
    {"key": "selecao_ag", "attr": "executar_ag", "tipo": "bool",
     "desc": "Etapa 4: rodar tambem AG (Algoritmo Genetico, GA-PLS) — o mais "
             "lento dos metodos de selecao de variaveis (populacao x geracoes "
             "avaliacoes de CV)",
     "opcoes": None},
    {"key": "ddsimca", "attr": "executar_ddsimca", "tipo": "bool",
     "desc": "Rodar DD-SIMCA (classificacao one-class)", "opcoes": None},
    {"key": "modo_ddsimca", "attr": "ddsimca_treinar_em", "tipo": "choice",
     "desc": "DD-SIMCA: 'puros' treina so em amostras puras (autenticacao); 'todos' usa todas",
     "opcoes": ["puros", "todos"]},
    {"key": "opls_da", "attr": "executar_opls", "tipo": "bool",
     "desc": "Rodar OPLS-DA", "opcoes": None},
    {"key": "comparar_pre_processamentos", "attr": "comparar_pipelines", "tipo": "bool",
     "desc": "Comparar varios pre-processamentos", "opcoes": None},
    {"key": "benchmark", "attr": "executar_benchmark", "tipo": "bool",
     "desc": "Auto-Benchmark: SVM RBF / RF / XGBoost vs PLS-DA (mesma CV group-aware)", "opcoes": None},
    {"key": "monte_carlo", "attr": "executar_monte_carlo", "tipo": "bool",
     "desc": "Monte Carlo CV: IC95% por percentil (N repeticoes estratificadas por grupo)", "opcoes": None},
    {"key": "n_monte_carlo", "attr": "n_monte_carlo", "tipo": "int",
     "desc": "Numero de repeticoes do Monte Carlo CV", "opcoes": None, "min": 1, "max": 100000},
    {"key": "monte_carlo_incluir_todos", "attr": "monte_carlo_incluir_todos", "tipo": "bool",
     "desc": "MC CV: incluir SVM RBF / RF / XGBoost alem do PLS-DA (mais lento)", "opcoes": None},
    {"key": "shap_benchmark", "attr": "executar_shap", "tipo": "bool",
     "desc": "SHAP values (TreeExplainer) para RF/XGBoost/GBM — interpretabilidade espectral", "opcoes": None},
    {"key": "shap_max_amostras", "attr": "shap_max_amostras", "tipo": "int",
     "desc": "Limite de amostras para calculo de SHAP (controle de memoria)", "opcoes": None,
     "min": 1, "max": 1000000},
    {"key": "figuras_detalhadas", "attr": "figuras_detalhadas", "tipo": "bool",
     "desc": "Gerar tambem as figuras exploratorias/detalhadas (HCA, loadings PCA, "
             "pre-processamento, contribuicao de score, DD-SIMCA por classe, Cooman). "
             "Desligado = so o conjunto essencial (mais rapido, menos arquivos)",
     "opcoes": None},
    {"key": "figuras_mostrar_marcadores", "attr": "mostrar_marcadores_classe", "tipo": "bool",
     "desc": "Usar formas diferentes por classe nos graficos de score", "opcoes": None},
    {"key": "figuras_mostrar_elipses", "attr": "mostrar_elipses_grupo", "tipo": "bool",
     "desc": "Desenhar elipses de confianca por grupo", "opcoes": None},
    {"key": "formato_figura", "attr": "formato_saida", "tipo": "choice",
     "desc": "Formato das figuras", "opcoes": ["png", "pdf", "svg"]},
    {"key": "dpi", "attr": "dpi_salvar", "tipo": "int",
     "desc": "Resolucao das figuras (DPI)", "opcoes": None, "min": 50, "max": 1200},
    {"key": "abrir_figuras_na_tela", "attr": "mostrar_graficos", "tipo": "bool",
     "desc": "[Nao disponivel: o backend grafico e sempre headless/Agg, por "
             "estabilidade em execucao paralela e no servidor web] Figuras "
             "continuam sendo sempre salvas em disco normalmente",
     "opcoes": None},
]


def _attr_para_yaml(spec: Dict[str, Any], cfg: Config) -> Any:
    """Le o atributo da Config e converte para a forma amigavel do YAML."""
    v = getattr(cfg, spec["attr"], getattr(Config(), spec["attr"], None))
    if spec["tipo"] == "preproc":
        return _PRE_PROC_INV.get(str(v).lower(), str(v))
    if spec["tipo"] == "list":
        return list(v) if v is not None else []
    if spec["tipo"] == "str_opcional":
        return "" if v is None else str(v)
    return v


def _checar_faixa(spec: Dict[str, Any], v: float) -> float:
    """Valida um numero contra 'min'/'max' opcionais do spec, com mensagem
    amigavel. Impede que valores impossiveis (fracao negativa, contagem zero)
    passem silenciosamente e so quebrem — ou pior, distorcam o resultado —
    no meio da execucao."""
    lo, hi = spec.get("min"), spec.get("max")
    if lo is not None and v < lo:
        raise ValueError(
            f"{spec['key']}={v}: valor minimo permitido e {lo}")
    if hi is not None and v > hi:
        raise ValueError(
            f"{spec['key']}={v}: valor maximo permitido e {hi}")
    return v


def _coagir_valor(spec: Dict[str, Any], val: Any) -> Any:
    """Converte um valor do YAML/menu para o tipo correto do atributo Config,
    validando opcoes. Lanca ValueError com mensagem amigavel se invalido."""
    t = spec["tipo"]
    if t == "str_opcional":
        s2 = str(val).strip()
        return s2 if s2 else None
    if t == "bool":
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in ("true", "sim", "1", "yes", "s", "v")
    if t == "int":
        return int(_checar_faixa(spec, int(val)))
    if t == "float":
        return _checar_faixa(spec, float(val))
    if t == "list":
        if val is None:
            return ()
        if isinstance(val, (list, tuple)):
            return tuple(str(x).strip() for x in val if str(x).strip())
        return tuple(x.strip() for x in str(val).split(",") if x.strip())
    if t == "choice":
        sv = str(val)
        if spec["opcoes"] and sv not in spec["opcoes"]:
            raise ValueError(f"valor '{sv}' invalido; use {spec['opcoes']}")
        return sv
    if t == "preproc":
        sv = str(val)
        if sv in _PRE_PROC_FRIENDLY:
            return _PRE_PROC_FRIENDLY[sv]
        if sv.lower() in _PRE_PROC_INV:
            return sv.lower()
        raise ValueError(
            f"pre-processamento '{sv}' invalido; use {list(_PRE_PROC_FRIENDLY)}")
    return str(val)


def _validar_semantico(cfg: "Config") -> List[str]:
    """Validacoes CRUZADAS entre campos, que 'min'/'max' isolados nao pegam.

    Retorna lista de mensagens amigaveis (vazia = tudo certo). Deve ser chamada
    ANTES de rodar (app e CLI) para falhar cedo, com mensagem clara, em vez de
    quebrar so no meio da execucao (depois de minutos carregando/processando).
    """
    erros: List[str] = []

    # Faixa espectral: o inicio precisa ser menor que o fim. Se invertido, o
    # filtro [wn_min, wn_max] remove TODAS as variaveis e o pipeline so
    # quebraria depois de carregar os dados (crash tardio, confuso).
    try:
        if float(cfg.wn_min) >= float(cfg.wn_max):
            erros.append(
                f"Faixa espectral invalida: o inicio ({cfg.wn_min:.0f}) deve "
                f"ser MENOR que o fim ({cfg.wn_max:.0f}) cm-1.")
    except (TypeError, ValueError):
        pass

    # Holdout: precisa sobrar treino suficiente. 'max'=0.5 ja barra pelo widget,
    # mas um config.yaml editado a mao pode trazer valor fora da faixa — aqui e
    # o backstop para esse caminho (Config cru, sem passar por _coagir_valor).
    try:
        fh = float(cfg.frac_holdout)
        if fh < 0.0 or fh > 0.5:
            erros.append(
                f"Fracao de holdout invalida ({fh}): use um valor entre 0 e 0.5 "
                "(0 = sem teste externo).")
    except (TypeError, ValueError):
        pass

    return erros


def _fmt_yaml(v: Any) -> str:
    """Formata um valor Python como YAML simples (para o arquivo comentado).
    Usa ASPAS SIMPLES quando precisa citar: em YAML, dentro de aspas simples a
    barra invertida e literal — essencial p/ caminhos do Windows (C:\\Users\\...).
    Em aspas duplas, '\\U' / '\\D' seriam escapes e quebrariam a leitura."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_yaml(x) for x in v) + "]"
    if isinstance(v, str):
        precisa = (v == "" or v != v.strip()
                   or v.lower() in ("true", "false", "null", "yes", "no", "~")
                   or any(c in v for c in ':#,[]{}&*!|>%@`"\''))
        if precisa:
            return "'" + v.replace("'", "''") + "'"
        return v
    return str(v)


def salvar_config(cfg: Config, caminho: str) -> None:
    """Escreve config.yaml em linguagem simples, com um comentario explicativo
    acima de cada campo. Regenera os comentarios a cada salvamento."""
    linhas = [
        "# " + "=" * 60,
        "#  CONFIGURACAO DO PIPELINE QUIMIOMETRICO",
        "#  Edite os valores abaixo. Cada campo tem uma explicacao no",
        "#  comentario logo acima. Voce NAO precisa abrir o codigo.",
        "#  Para rodar:   python pipeline.py --rodar",
        "#  Assistente:   python guaraci.py",
        "# " + "=" * 60,
        "",
    ]
    for s in _CONFIG_SPEC:
        linhas.append(f"# {s['desc']}")
        if s["opcoes"]:
            linhas.append(f"#   opcoes: {' | '.join(map(str, s['opcoes']))}")
        linhas.append(f"{s['key']}: {_fmt_yaml(_attr_para_yaml(s, cfg))}")
        linhas.append("")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))


def carregar_config(caminho: str, base: Optional[Config] = None) -> Config:
    """Le config.yaml e devolve uma Config. Mantem os defaults para chaves
    ausentes; ignora chaves desconhecidas; reune erros numa mensagem clara."""
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML nao instalado. Rode: pip install pyyaml")
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Config nao encontrado: {caminho}. Gere um pelo assistente "
            f"(opcao [S] salvar) antes de usar --rodar.")
    with open(caminho, "r", encoding="utf-8") as f:
        dados = yaml.safe_load(f) or {}
    cfg = base if base is not None else Config()
    spec_por_key = {s["key"]: s for s in _CONFIG_SPEC}
    erros: List[str] = []
    for key, val in dados.items():
        s = spec_por_key.get(key)
        if s is None:
            continue
        try:
            setattr(cfg, s["attr"], _coagir_valor(s, val))
        except Exception as e:
            erros.append(f"  - '{key}': {e}")
    if erros:
        raise ValueError("Problemas no config.yaml:\n" + "\n".join(erros))
    return cfg


def _validar_pasta_dados(cfg: Config) -> Tuple[bool, str]:
    """Checagem amigavel da fonte de dados, ciente do modo de entrada.
    Generico: serve para .dx (FT-NIR), CSV (qualquer dado tabular) ou
    dados sinteticos de teste."""
    modo = getattr(cfg, "modo", "dx")
    if modo == "sintetico":
        return True, "OK — modo sintetico (dados gerados em memoria, sem arquivo)"
    if modo == "csv":
        cam = cfg.arquivo_csv
        if not cam or not os.path.isfile(cam):
            return False, f"CSV nao encontrado: '{cam}' (confira o caminho)"
        return True, f"OK — CSV: {os.path.basename(cam)}"
    # modo dx (padrao)
    p = cfg.pasta_entrada
    if not p or not os.path.isdir(p):
        return False, f"pasta nao encontrada: '{p}' (confira o caminho)"
    n_dx = len(glob.glob(os.path.join(p, "**", "*.dx"), recursive=True))
    if n_dx == 0:
        return False, f"nenhum arquivo .dx em '{p}' (nem nas subpastas)"
    return True, f"OK — {n_dx} arquivos .dx encontrados"


def _editar_campo(cfg: Config, s: Dict[str, Any]) -> None:
    """Edita um campo via terminal, com validacao."""
    print(f"\n  {s['desc']}")
    print(f"  valor atual: {_fmt_yaml(_attr_para_yaml(s, cfg))}")
    if s["tipo"] == "bool":
        setattr(cfg, s["attr"], not bool(getattr(cfg, s["attr"])))
        print(f"  -> alterado para {getattr(cfg, s['attr'])}")
        return
    if s["tipo"] in ("choice", "preproc"):
        ops = s["opcoes"] or []
        for i, o in enumerate(ops, 1):
            print(f"    ({i}) {o}")
        r = input("  escolha o numero (Enter cancela): ").strip()
        if r.isdigit() and 1 <= int(r) <= len(ops):
            try:
                setattr(cfg, s["attr"], _coagir_valor(s, ops[int(r) - 1]))
                print("  -> ok")
            except Exception as e:
                print(f"  erro: {e}")
        else:
            print("  cancelado.")
        return
    novo = input("  novo valor (Enter cancela): ").strip()
    if novo == "":
        print("  cancelado.")
        return
    try:
        setattr(cfg, s["attr"], _coagir_valor(s, novo))
        print("  -> ok")
    except Exception as e:
        print(f"  erro: {e}")


def menu_interativo(cfg: Optional[Config] = None,
                    caminho_cfg: str = "config.yaml") -> None:
    """Assistente de terminal: visualiza/edita a configuracao, salva/carrega
    o config.yaml e dispara o pipeline — tudo sem editar o codigo."""
    cfg = cfg if cfg is not None else Config()
    if os.path.exists(caminho_cfg):
        try:
            cfg = carregar_config(caminho_cfg, base=cfg)
            print(f"[config] carregado de {caminho_cfg}")
        except Exception as e:
            print(f"[config] nao foi possivel carregar ({e}). Usando padroes.")

    while True:
        print("\n" + "=" * 62)
        print("  ASSISTENTE DO PIPELINE QUIMIOMETRICO")
        print("=" * 62)
        for i, s in enumerate(_CONFIG_SPEC, 1):
            print(f"  [{i:2d}] {s['key']:<28s}: "
                  f"{_fmt_yaml(_attr_para_yaml(s, cfg))}")
        ok, msg = _validar_pasta_dados(cfg)
        print("-" * 62)
        print(f"  dados: {msg}")
        print("-" * 62)
        print("  digite o NUMERO p/ editar  |  [S]alvar  [L]carregar  "
              "[R]odar  [Q]sair")
        escolha = input("  > ").strip().lower()

        if escolha in ("q", "sair", "quit"):
            print("  encerrado."); return
        if escolha in ("s", "salvar"):
            salvar_config(cfg, caminho_cfg)
            print(f"  salvo em {caminho_cfg}"); continue
        if escolha in ("l", "carregar"):
            try:
                cfg = carregar_config(caminho_cfg, base=cfg)
                print("  recarregado.")
            except Exception as e:
                print(f"  erro: {e}")
            continue
        if escolha in ("r", "rodar", "run"):
            ok, msg = _validar_pasta_dados(cfg)
            if not ok:
                print(f"  [!] {msg}. Corrija antes de rodar."); continue
            _erros_sem = _validar_semantico(cfg)
            if _erros_sem:
                for _e in _erros_sem:
                    print(f"  [!] {_e}")
                print("  Corrija antes de rodar."); continue
            salvar_config(cfg, caminho_cfg)
            print("  iniciando pipeline...\n")
            executar(cfg); return
        if escolha.isdigit() and 1 <= int(escolha) <= len(_CONFIG_SPEC):
            _editar_campo(cfg, _CONFIG_SPEC[int(escolha) - 1])
        else:
            print("  opcao invalida.")


if __name__ == "__main__":
    import sys
    _CFG_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    if "--rodar" in sys.argv:
        # Modo direto: usa config.yaml se existir, senao a Config do codigo.
        _cfg = carregar_config(_CFG_PATH) if os.path.exists(_CFG_PATH) else CFG
        executar(_cfg)
    elif "--codigo" in sys.argv:
        executar(CFG)                       # modo legado (Config embutida)
    elif sys.stdin is not None and sys.stdin.isatty():
        try:
            from cli_assistente import main as _cli_main
            _cli_main()
        except ImportError:
            menu_interativo(CFG, _CFG_PATH)  # fallback para o menu antigo
    else:
        _cfg = carregar_config(_CFG_PATH) if os.path.exists(_CFG_PATH) else CFG
        executar(_cfg)
