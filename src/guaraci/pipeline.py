"""Chemometric pipeline for multivariate analytical data (FT-NIR and other techniques).

Publication-quality implementation.

Capabilities:
    Preprocessing    : SNV, MSC (stateful), SG, MC, Autoscaling
    Classification   : PLS-DA, OPLS-DA, DD-SIMCA (GroupKFold anti-leakage)
    Variable sel.    : iPLS, VIP >= 1, SR top-fraction, sPLS-DA
    Validation       : Y-permutation, Wold R2Y/Q2Y, CV-ANOVA, BCa bootstrap
    Benchmark        : PLS-DA vs SVM vs RF vs GBM vs XGBoost (OOF GroupKFold)
    Interpretability : SHAP TreeExplainer, DET curves (linear + log)
    Uncertainty      : Monte Carlo CV — N x StratifiedGroupShuffleSplit IC95%
    Reports          : PDF, Word, Excel (5 sheets), LaTeX, PowerPoint

Preset padrao: MSC -> SG -> MC (FT-NIR difuso com espalhamento forte).
(GroupKFold por amostra fisica).
"""

# __version__ e _NIVEL_NOME sao a fonte unica em config.py (modulo sem
# dependencias); reexportados aqui p/ `pipeline.__version__` e `pq._NIVEL_NOME`.
from guaraci.config import __version__, _NIVEL_NOME, NIVEL_NOME, _NIVEL_SLUG_PASTA  # noqa: F401,E402
from guaraci.config import (   # noqa: F401,E402
    NOME_GRAFICOS,
    NOME_TABELAS,
    NOME_RELATORIOS,
    NOME_MODELOS,
    pasta_execucao,
)

import copy
import logging
import os
import glob
import time
import contextlib
import functools
import warnings
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Callable, cast, Any

import numpy as np
import pandas as pd
import matplotlib as mpl
# M-15(a) do parecer de 2026-09-25: IMPORTAR este modulo nao altera mais o
# backend do matplotlib nem os filtros de warnings do processo inteiro. Isso
# vive em `_ambiente_da_execucao()`, aplicado so' DURANTE `executar()` (e
# restaurado ao sair). Motivo do backend Agg (mantido durante a execucao): sem
# ele o matplotlib usa TkAgg, que NAO e thread-safe -- com a execucao paralela
# dos testes de permutacao/Wold ou processos longos (Streamlit), objetos Tk
# pendentes de coleta podem ser finalizados fora da thread principal e derrubar
# o processo ("main thread is not in main loop"). Efeito colateral inerente:
# 'abrir_figuras_na_tela'/show_plots nao abre janela -- as figuras sao sempre
# salvas em disco.
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


_VARS_THREADS_BLAS = ("GUARACI_BLAS_THREADS", "OMP_NUM_THREADS",
                      "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _limite_threads_blas() -> Any:
    """BLAS com 1 thread durante a execucao, salvo se o usuario escolheu.

    As matrizes do pipeline sao pequenas (centenas de amostras) e as contas
    vem em milhares de chamadas curtas (um PCA/PLS por fold, por VL, por
    permutacao). Nesse regime o BLAS multithread gasta mais sincronizando
    threads do que calculando. Medido (2026-10-15, 243 x 1500, OpenBLAS com
    4 threads): execucao completa de 623 s para 194 s com 1 thread, o mesmo
    `resumo_modelo.txt`. Num micro-benchmark de PCA/PLS/eigh: 3,8 s com 4
    threads, 1,2 s com 1 ou 2. O paralelismo que vale a pena aqui e' o de
    processos (permutacoes), que ja' limita cada processo a 1 thread.

    `GUARACI_BLAS_THREADS=n` (ou OMP/OPENBLAS/MKL_NUM_THREADS) mantem a
    escolha do usuario.
    """
    if any(os.environ.get(v) for v in _VARS_THREADS_BLAS):
        n_env = os.environ.get("GUARACI_BLAS_THREADS")
        if n_env and n_env.isdigit() and int(n_env) > 0:
            import threadpoolctl
            return threadpoolctl.threadpool_limits(int(n_env), user_api="blas")
        return contextlib.nullcontext()
    try:
        import threadpoolctl
    except ImportError:  # dependencia declarada; sem ela, so' nao limita
        return contextlib.nullcontext()
    return threadpoolctl.threadpool_limits(1, user_api="blas")


@contextlib.contextmanager
def _ambiente_da_execucao():
    """Backend headless + filtros de warnings do sklearn SO' durante uma
    execucao, restaurados ao sair (nada de efeito global na importacao)."""
    try:
        anterior = mpl.get_backend()
    except Exception:  # noqa: BLE001 -- sem backend definido ainda
        anterior = None
    mpl.use("Agg", force=True)
    with warnings.catch_warnings(), _limite_threads_blas():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        try:
            yield
        finally:
            if anterior and anterior.lower() != "agg":
                try:
                    mpl.use(anterior, force=True)
                except Exception:  # noqa: BLE001 -- backend interativo indisponivel
                    pass


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                SETTINGS — edit ONLY here                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Config extraida para config.py (item 10 da auditoria): dataclass pura de
# ~80 campos, sem dependencia deste modulo. Reexportada aqui para nao quebrar
# `pipeline.Config` nem os modulos que fazem `from pipeline import Config`.
from guaraci.config import Config  # noqa: F401


_CFG = Config()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                END OF SETTINGS                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝


# parse_title/JCAMP-DX (CODIGO_ESPECIE, ADULTERANTE_NOME, parse_title,
# extract_dx_title) extraidos p/ dados_io.py (Fase H). Reexportados aqui
# para nao quebrar `pipeline.parse_title(...)` nem o restante do modulo.
from guaraci.dados_io import (   # noqa: E402
    CODIGO_ESPECIE,
    ADULTERANTE_NOME,
    adulterant_from_mae_id,
    chave_unidade_validacao,
    parse_title,
    extract_dx_title,
    sanitizar_metadados,
)
from guaraci.perfil_matriz import apply_profile, cfg_profile  # noqa: E402
from guaraci.eixo_espectral import (   # noqa: E402
    aviso_de_corte_de_faixa, campos_faixa_espectral, chave_faixa_espectral, tecnica_do_perfil,
    unidade_do_eixo)
# Registra os leitores de formato de instrumento (opus/spc/sp/rmn/hplc/
# gcms/eem) no io_registry -- import so' por efeito colateral (achado da
# auditoria de acessibilidade CLI: os parsers existiam testados, mas nenhum
# cfg.mode os alcancava). Ver docstring de leitores_avancados.py.
from guaraci import leitores_avancados as _leitores_avancados  # noqa: E402,F401


log = logging.getLogger(__name__)


class ValidacaoCruzadaImpossivel(ValueError):
    """Alguma classe nao tem 2 amostras (ou 2 grupos independentes): nao ha' como validar. `classes` diz quais,
    para a interface oferecer retira-las e rodar de novo."""

    def __init__(self, classes, mensagem: str):
        super().__init__(mensagem)
        self.classes = [str(c) for c in classes]

# __all__ lista so' o que pipeline.py DEFINE. A fachada reexporta dezenas de
# nomes de outros modulos (chemometric_stats, dados_io, preprocessamento,
# classificadores, figuras, validacao_estatistica, hardware,
# selecao_variaveis, avaliacao_modelos, paleta_cores, config, perfil_matriz)
# -- esse contrato de reexport ja' e' protegido por
# tests/test_fachada_reexport.py (CONTRATO), que verifica identidade de cada
# simbolo; nao duplicado aqui para nao manter duas fontes de verdade da
# mesma lista.
__all__ = [
    "generate_output_name",
    "validate_input",
    "classification_metrics",
    "compare_pipelines",
    "bootstrap_vip_stratified",
    "bootstrap_vip",
    "clear_old_results",
    "labels_for_quantification",
    "r2cv_species_by_adulterant",
    "pls_regression_by_species",
    "pls_regressao_pooled",
    "executar",
]


NOME_MARCADOR_INCOMPLETA = "EXECUCAO_INCOMPLETA.txt"

_TEXTO_MARCADOR_INCOMPLETA = (
    "Esta execucao NAO terminou.\n"
    "O arquivo existe desde o inicio e so' e' apagado quando a execucao conclui. "
    "Se voce o esta lendo, a execucao foi interrompida (erro, Ctrl+C, falta de "
    "memoria ou computador desligado) e as tabelas, figuras e modelos desta pasta "
    "estao PARCIAIS. Nao use estes arquivos como resultado: rode de novo.\n")


def _marcar_execucao_incompleta(pasta: str) -> None:
    """Cria o marcador de execucao incompleta na pasta de resultados."""
    try:
        with open(os.path.join(pasta, NOME_MARCADOR_INCOMPLETA), "w",
                  encoding="utf-8") as f:
            f.write(_TEXTO_MARCADOR_INCOMPLETA)
    except OSError as e:
        log.warning(f"  [AVISO] nao foi possivel gravar o marcador "
                    f"{NOME_MARCADOR_INCOMPLETA}: {e}")


def _concluir_execucao(pasta: str) -> None:
    """Apaga o marcador: so' a execucao que chegou ao fim fica sem ele."""
    try:
        os.remove(os.path.join(pasta, NOME_MARCADOR_INCOMPLETA))
    except FileNotFoundError:
        pass
    except OSError as e:
        log.warning(f"  [AVISO] nao foi possivel apagar o marcador "
                    f"{NOME_MARCADOR_INCOMPLETA}: {e}")


def _slug(texto: str) -> str:
    """Normaliza um texto livre para nome de pasta seguro (sem espacos/barras)."""
    import re as _re
    limpo = _re.sub(r"[^\w\-]+", "_", texto.strip())
    return limpo.strip("_") or ""


def _dataset_id(cfg: Config) -> str:
    """Identificador do DATASET/AMOSTRA (nivel de topo da saida, auditoria
    jul/2026 item 4: Resultados/Amostra/Modo/...). Prioridade:
    1) cfg.tag (rotulo livre ja existente, ex.: 'lote_a') quando
       preenchido — o usuario esta nomeando o conjunto explicitamente;
    2) senao, deriva do mode de entrada (nome do CSV, da pasta de dados, ou
       'sintetico' para dados de teste) — sempre disponivel, nunca vazio.
    """
    if cfg.tag.strip():
        return _slug(cfg.tag) or "dataset"
    if cfg.mode == "csv":
        base = os.path.splitext(os.path.basename(cfg.csv_file or ""))[0]
    elif cfg.mode == "sintetico":
        base = "sintetico"
    else:  # "dx" | "imagem"
        base = os.path.basename(os.path.normpath(cfg.input_folder or ""))
    return _slug(base) or "dataset"


def generate_output_name(cfg: Config, n_classes: int, n_amostras: int) -> str:
    """Default output path (auditoria jul/2026, item 4): reestrutura a saida
    em Amostra/Modo/Execucao para nao misturar resultados de objetivos
    diferentes na mesma pasta.
        {root}/{dataset}/{Modo}/PLSDA_OE_{slug}_{preproc}_{YYYYMMDD_HHMMSS}
    Example: resultados_guaraci/lote_a/Classificacao/PLSDA_OE_Autenticacao_MSC-SG1-MC_20260528_191500
    'dataset' vem de cfg.tag (se preenchido) ou e' derivado do mode de
    entrada (ver _dataset_id). 'Modo' e' o rotulo amigavel do objetivo
    cientifico resolvido (ver modos_analise.resolve_objective) — Exploratorio
    | Classificacao | Quantificacao. '{slug}' e' o nome amigavel de cfg.level
    (_NIVEL_SLUG_PASTA) -- necessario porque N1 e N2 caem no MESMO 'Modo'
    (Classificacao) mas sao analises distintas (por-especie vs autenticacao);
    corrigido em 2026-07-13 (P8 residual: pasta ainda expunha N1/N2/N3 cru).
    Subfolders (created in executar): Graficos/ Tabelas/ Relatorios/ Modelos/
    """
    from guaraci.preprocessamento import descrever_preprocessamento
    preproc, _descr, _usa_sg = descrever_preprocessamento(cfg)
    partes = ["PLSDA_OE", _NIVEL_SLUG_PASTA.get(cfg.level, cfg.level)]
    partes.append("-".join(preproc))
    partes.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    dataset_id = _dataset_id(cfg)
    modo_pasta = OBJETIVO_ROTULO.get(resolve_objective(cfg), "Analise")
    # Nomes de pasta no idioma da sessao (pedido de 2026-11-07); nada le' estas duas pastas pelo nome.
    from guaraci.idioma_relatorio import idioma
    if idioma() == "en":
        modo_pasta = _MODO_PASTA_EN.get(modo_pasta, modo_pasta)
        partes[1] = _NIVEL_SLUG_PASTA_EN.get(partes[1], partes[1])
    return os.path.join(cfg.output_root_folder, dataset_id, modo_pasta,
                         "_".join(partes))


#: pasta do Modo e slug do nivel em ingles (sessao EN).
_MODO_PASTA_EN = {"Exploratorio": "Exploratory", "Classificacao": "Classification",
                  "Quantificacao": "Quantification", "Analise": "Analysis"}
_NIVEL_SLUG_PASTA_EN = {"PorEspecie": "PerSpecies", "Autenticacao": "Authentication",
                        "Quantificacao": "Quantification"}


# Paleta/marcadores de classes extraidos p/ paleta_cores.py (Fase H). Reexpor-
# tados aqui para nao quebrar `pipeline.color(...)`/`pipeline.PALETA` nem as
# ~52 chamadas internas nas funcoes de figura.
from guaraci.paleta_cores import (   # noqa: E402
    PALETA,
    MARCADORES,
    _paleta_externa,
    _luminancia,
    get_edge_color,
    color,
    map_class_colors,
    map_class_markers,
)


# Camada de plotagem (setup_matplotlib, save, specificity_by_class,
# helpers de plot e ~30 funcoes fig_*) extraida p/ figuras.py (Fase H).
# Reexportada aqui para nao quebrar pipeline.fig1_pca_scores(...),
# pipeline.save(...) nem as chamadas de executar()/figuras remanescentes.
from guaraci.figuras import (   # noqa: E402
    setup_matplotlib,
    save,
    resetar_falhas_salvamento,
    obter_falhas_salvamento,
    registrar_falha_salvamento,
    specificity_by_class,
    ellipse_t2,
    convex_hull_contorno,
    adaptive_scatter_parameters,
    _ticks_x_inteiros,
    plot_scores_panel,
    fig1_selecao_lvs,
    fig_hca_dendrogram,
    fig_hca_comparacao_pipelines,
    fig1_pca_scores,
    fig2_plsda_scores,
    fig3_outliers,
    fig4_confusao,
    fig6_preprocessing,
    fig_extra_wold,
    fig_extra_holdout,
    fig_extra_comparacao_pipelines,
    fig5b_vip_stability,
    fig7_pls_regression,
    fig_sprint3_sr_vip,
    fig_sprint3_score_contribution,
    fig_sprint3_ddsimca_acceptance,
    fig_ddsimca_individuais,
    fig_sprint3_opls_scores,
    fig_loadings_pca,
    fig_roc_auc,
    fig_splot_opls,
    fig_cooman_ddsimca,
    fig_regression_merit,
    fig_heatmap_species_by_adulterant,
    fig_class_mean_spectra,
    fig_biplot_pca,
)

# Camada de objetivo cientifico (Exploratorio/Classificacao/Quantificacao):
# fonte unica que decide QUAIS figuras/relatorios cada mode gera, para que
# um run so' produza os resultados pertinentes ao seu objetivo (ver
# modos_analise.py). Reexportado para pipeline.resolve_objective(...) etc.
from guaraci.modos_analise import (   # noqa: E402
    resolve_objective,
    should_generate,
    exploratory_figures_enabled,
    figure_plan,
    describe_plan,
    OBJETIVO_ROTULO,
    EXPLORATORIO,
    CLASSIFICACAO,
    QUANTIFICACAO,
)


# Transformers de pre-processamento (SNV/SavGol/MSC) + build_preprocessor
# extraidos p/ preprocessamento.py (Fase H). Reexportados aqui para nao quebrar
# pipeline.SNV / pipeline.build_preprocessor(...) nem o restante do modulo.
from guaraci.preprocessamento import (   # noqa: E402
    SNV,
    SavGol,
    MSC,
    build_preprocessor,
    preparar_eixo_espectral,
)


# =========================================================================
#  Chemometric diagnostics
#  Extraidos para chemometric_stats.py (Fase H — modularizacao). Reexportados
#  aqui para nao quebrar `pipeline.vip_scores(...)` nem as chamadas internas.
# =========================================================================
from guaraci.saturacao import MODOS_COM_SATURACAO, detectar_saturacao  # noqa: E402
from guaraci.chemometric_stats import (   # noqa: E402
    vip_scores,
    compute_selectivity_ratio,
    martens_uncertainty_test,
    hotelling_t2,
    hotelling_t2_limit,
    q_residuals,
    q_residuals_limit,
    q_residuals_pls_loo,
    dmodx,
    dmody,
    explained_variance,
    regression_figures_of_merit,
    faixa_decisao,
    rpd_rer,
    interpret_rpd,
    applicability_domain,
    training_applicability_domain,
    ad_rejection_rate_cv,
    ad_small_n_warning,
    applicability_domain_new_samples,
    mean_and_dof_moments,
    rmse_flat,
    diagnose_spectral_range,
    expandir_binario_um_quente,
)


# DDSimca / OPLSDAWrapper extraidos p/ classificadores.py (Fase H).
# Reexportados aqui para nao quebrar pipeline.DDSimca(...) /
# pipeline.OPLSDAWrapper(...) nem as chamadas internas em executar().
from guaraci.classificadores import (   # noqa: E402
    DDSimca,
    OPLSDAWrapper,
    PLSLDAClassifier,
    detectar_classes_em_rampa,
    PUREZA_MIN_GRUPOS_LOGO,
    ddsimca_alpha_sustentado,
    ddsimca_logo_sensitivity,
    ddsimca_pcv_sensitivity,
)


# Escritores de resultados (resumo/model-card/identificadores) extraidos para
# resultados_io.py (dividida tecnica). Reexportados: executar() os chama e os
# testes/consumidores usam via `pipeline.X`.
from guaraci.resultados_io import (   # noqa: F401
    pls_model_metrics, save_identifiers, _NOTAS_METODOLOGICAS,
    save_model_summary, append_regression_summary, _md_tabela,
    generate_model_card, append_regression_model_card, append_heatmap_summary,
    append_identification_model_card, append_purity_model_card,
    append_linearity_robustness_model_card,
    save_design_audit, load_design_audit,
)

# Linearidade formal (lack-of-fit) e protocolo de robustez (Bloco 13d).
from guaraci.linearity import LackOfFitResult, lack_of_fit_test   # noqa: E402
from guaraci.robustness import (   # noqa: E402
    RobustnessResult, gaussian_noise_variants, baseline_drift_variants,
    preprocessing_config_variants, run_robustness_protocol,
    avaliar_rmsep_plsr, avaliar_bal_acc_plsda,
)

# Identificacao especie x adulterante (Bloco 9b) -- ensemble conformal +
# limite de uniao para o mode cego (Detectar -> Identificar -> Quantificar).
from guaraci.identificacao import (   # noqa: E402
    CoverageStatus,
    combine_alpha_bonferroni,
    train_identification_ensemble,
)
from guaraci.conformal import (  # noqa: E402
    conformal_margin_classification,
    conformal_margin_regression,
    n_minimum_for_alpha,
)


def validate_input(X: np.ndarray, wavenumbers: np.ndarray,
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
        log.info(f"[WARNING] Removed {n_rem} samples with NaN/Inf.")
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
        log.info(f"[WARNING] Removed {n_const} constant variables.")
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
        log.info(f"[CAUTION] {msg}")

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
                log.info(f"[CAUTION] {n_aprox} samples with corr > "
                      f"{limiar_correlacao:.5f}.")

    relatorio["n_final"] = int(len(X))
    relatorio["n_variaveis_final"] = int(X.shape[1])
    relatorio["warnings"] = warnings_list
    return X, wavenumbers, rotulos, conc, mae_id, relatorio


def _check_balance(rotulos: np.ndarray, ratio_alvo: float = 5.0
                              ) -> Dict[str, object]:
    """Detects severe class imbalance."""
    cls_unicas, counts = np.unique(rotulos, return_counts=True)
    n_max = int(np.max(counts))
    n_min = int(np.min(counts))
    ratio = n_max / max(n_min, 1)
    rel: Dict[str, object] = {
        "imbalance_ratio": float(ratio),
        "n_max":           n_max,
        "n_min":           n_min,
        "distribuicao":    {str(c): int(n) for c, n in zip(cls_unicas, counts)},
        "desbalanceado":   ratio > ratio_alvo,
    }
    if ratio > ratio_alvo:
        log.info(f"[WARNING] Severe class imbalance: max/min ratio = "
              f"{ratio:.2f} (max={n_max}, min={n_min}).")
        log.info("          Suggestions: prioritize balanced_accuracy/F1-macro "
              "over accuracy; consider class_weight or subsampling.")
    return rel


def classification_metrics(y_true, y_pred, classes) -> Dict[str, float]:
    """Metricas globais de classificacao.

    Com UMA unica classe no conjunto, toda metrica aqui e' degenerada por
    construcao: accuracy = 1,0, kappa = 0,0, recall = 1,0 -- nao porque o
    modelo acertou, mas porque nao havia o que errar. A versao anterior
    devolvia esses numeros sem nenhuma marca, e eles apareciam no
    resumo/model card como se fossem desempenho (medido rodando o pipeline
    sobre dados de classe unica -- auditoria mestre de 2026-08-17). A chave `degenerada_uma_classe` deixa isso
    explicito para quem consome o dicionario.
    """
    if len(set(map(str, np.asarray(y_true).ravel().tolist()))) < 2:
        log.info("  [AVISO] Apenas UMA classe presente: as metricas globais "
                 "de classificacao (accuracy, kappa, F1) sao degeneradas e "
                 "nao medem desempenho. Ignore-as neste conjunto.")
        base = {
            "accuracy":          float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "cohen_kappa":       float(cohen_kappa_score(y_true, y_pred)),
            "f1_macro":          float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
            "precision_macro":   float(precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
            "recall_macro":      float(recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        }
        base["degenerada_uma_classe"] = 1.0
        return base
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa":       float(cohen_kappa_score(y_true, y_pred)),
        "f1_macro":          float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "precision_macro":   float(precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "recall_macro":      float(recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
    }


def compare_pipelines(cfg: Config, X_raw: np.ndarray, Y_bin: np.ndarray,
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
        "MC only":              lambda: [_mc()],
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
            except (ValueError, np.linalg.LinAlgError):
                continue   # fold/preset degenerado (matriz singular etc.)
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
            log.info(f"  {nome:<26s} -> [AVISO] nenhum modelo convergiu "
                  f"({max_lv} LVs testados) — preset ignorado")
            continue
        _q2_disp = f"{melhor['q2']:.3f}" if np.isfinite(melhor['q2']) else "n/a"
        log.info(f"  {nome:<26s} -> LVs={melhor['n_lv']:2d}  "
              f"Acc={melhor['accuracy']:.3f}  "
              f"BalAcc={melhor['balanced_acc']:.3f}  "
              f"Q2={_q2_disp}")

    return resultados


def bootstrap_vip_stratified(X_processed: np.ndarray, Y_bin: np.ndarray,
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
        except (ValueError, np.linalg.LinAlgError):
            # Grupo/amostra reamostrada degenerada -- contado, nao mascarado
            # (n_falhos reportado ao chamador).
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
    """DEPRECATED: use bootstrap_vip_stratified. Kept for
    backward compatibility. Does NOT use per-class stratification."""
    log.info("[WARNING] Non-stratified bootstrap_vip is DEPRECATED. "
          "Use bootstrap_vip_stratified.")
    rng = np.random.default_rng(seed)
    n = len(X_processed)
    vips = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            pls = PLSRegression(n_components=n_opt, scale=False)
            pls.fit(X_processed[idx], Y_bin[idx])
            vips.append(vip_scores(pls))
        except (ValueError, np.linalg.LinAlgError):
            continue   # amostra reamostrada degenerada
    if not vips:
        p = X_processed.shape[1]
        return np.zeros(p), np.zeros(p)
    vips_arr = np.asarray(vips)
    return vips_arr.mean(axis=0), vips_arr.std(axis=0)


# Validacao estatistica (cross_val_predict manual, BCa, CV-ANOVA, teste de
# permutacao e de Wold) extraida p/ validacao_estatistica.py (Fase H).
# Reexportada para nao quebrar pipeline.permutation_test(...),
# pipeline._cv_predict_manual(...) nem as chamadas em compare_pipelines/
# etapa4/executar.
from guaraci.validacao_estatistica import (   # noqa: E402
    _cv_predict_manual,
    bootstrap_bca_ci,
    cv_anova_eriksson,
    _iter_wold,
    wold_test,
    _iter_permutacao,
    permutation_test,
    StableStratifiedGroupKFold,
)


# Carregamento de dados (.dx JCAMP-DX/ASDF, CSV, sintetico) extraido p/
# dados_io.py (Fase H). Reexportado aqui para nao quebrar as chamadas de
# `executar()` (`pipeline.load_data(cfg)` etc.).
from guaraci.dados_io import (   # noqa: E402
    generate_synthetic_data,
    kennard_stone,
    kennard_stone_split,
    kennard_stone_split_group_aware,
    _split_cal_val_por_especie,
    duplex_split,
    duplex_split_group_aware,
    spxy_split,
    spxy_split_group_aware,
    load_csv,
    _flush_asdf,
    _decodificar_linha_asdf,
    parse_dx,
    parse_spectrum,
    _extrair_conc_filename,
    _listar_arquivos_espectro,
    _detectar_subpastas_classe,
    prescan_dx,
    load_dx,
    load_data,
)

# Colorimetria digital (mode="imagem", prototipo) extraida p/ dados_imagem.py
# (Fase de expansao pos-H). Reexportada p/ pipeline.load_images(...) etc.
from guaraci.dados_imagem import (   # noqa: E402
    load_images,
    load_image_file,
    recortar_relativo,
    extract_color_features,
    extract_texture_features,
)

# Manifesto de proveniencia/integridade do modelo exportado (P5). Reexportado
# para nao quebrar pipeline.save_manifest(...) nem pipeline.load_model(...).
from guaraci.predicao import (   # noqa: E402
    save_manifest,
    load_model,
    SecurityError,
)


# (Camada de plotagem extraida p/ figuras.py — ver reexport acima.)
# =========================================================================
#  STAGE 4 — Variable Selection (iPLS, VIP, SR, sPLS-DA)
# =========================================================================

# STAGE 4 — Selecao de variaveis (iPLS/sPLS-DA) + figuras da etapa extraidas
# p/ selecao_variaveis.py (Fase H). Reexportadas p/ nao quebrar as chamadas em
# executar() (etapa4_selecao_variaveis, selecao_ipls, ...).
from guaraci.selecao_variaveis import (   # noqa: E402
    _avaliar_subset_cv,
    _avaliar_subset_nested_cv,
    _avaliar_busca_nested_cv,
    _cv_local,
    _mask_vip_threshold,
    _mask_sr_top_frac,
    selecao_ipls,
    sparse_plsda_mask,
    _spa_cadeia,
    selecao_spa,
    ga_selection,
    fig_etapa4_ipls,
    fig_etapa4_ag_convergencia,
    fig_etapa4_comparacao,
    etapa4_selecao_variaveis,
)


# Auto-Benchmark / Monte Carlo CV / DET / SHAP + PLSDAClassifier extraidos p/
# avaliacao_modelos.py (Fase H). Reexportados p/ nao quebrar as chamadas em
# executar() e o uso de PLSDAClassifier.
from guaraci.avaliacao_modelos import (   # noqa: E402
    PLSDAClassifier,
    fig_benchmark_classifiers,
    benchmark_classifiers,
    fig_monte_carlo_distribution,
    _stratified_group_shuffle_splits,
    monte_carlo_cv,
    fig_det_curvas,
    fig_shap_benchmark,
    benchmark_regression_by_species,
    fig_benchmark_regressors,
)


# =========================================================================
#  Orquestrador
# =========================================================================

# =========================================================================
#  v29: Compatibilidade de hardware — probe, auto-ajuste, guardas de RAM
# =========================================================================

# Compatibilidade de hardware (probe, auto-ajuste, guarda de RAM) extraida p/
# hardware.py (Fase H). Reexportada p/ nao quebrar pipeline.hardware_probe(),
# pipeline.auto_adjust_hardware_config(...) nem o uso em app_quimiometria.py.
from guaraci.hardware import (   # noqa: E402
    hardware_probe,
    auto_adjust_hardware_config,
    _verificar_ram,
)
# transferencia_calibracao.py (Passo 86): Direct/Piecewise Direct
# Standardization -- corrige espectros de um instrumento "escravo" para
# se parecerem com o instrumento "mestre" em que o modelo foi calibrado.
from guaraci.transferencia_calibracao import (   # noqa: E402
    StandardizationTransform,
    direct_standardization,
    piecewise_direct_standardization,
    apply_standardization,
)
def clear_old_results(pasta_base: str,
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
        except OSError as _e:   # permissao/arquivo em uso/etc.
            resultado["erro"] = (resultado["erro"] or "") + f"\n{p}: {_e}"
    resultado["liberado_mb"] = round(resultado["liberado_mb"], 1)
    return resultado




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


def labels_for_quantification(cfg: "Config", rotulos_verdadeiros: np.ndarray,
                                rotulos_preditos: Optional[np.ndarray]
                                ) -> Tuple[np.ndarray, str]:
    """Quais rotulos a calibracao por classe pode ver. Devolve (rotulos, mode).

    O QUE ESTA EM JOGO. A quantificacao do GUARACI calibra SEPARADAMENTE por
    classe, porque um unico modelo multi-classe e' dominado pela variacao
    entre matrizes. Isso obriga a responder: de onde vem a classe de cada
    amostra na hora de calibrar?

    - `cego` (PADRAO): da PREDICAO do classificador. E' o unico mode que
      corresponde ao uso real -- quem envia uma amostra desconhecida nao
      sabe a classe dela. Se o classificador erra, o erro se propaga para a
      quantificacao, e e' correto que se propague: e' o que aconteceria em
      producao.
    - `controle`: da verdade. Util para UMA coisa so' -- isolar o erro de
      quantificacao do erro de classificacao durante o desenvolvimento.
      Qualquer numero obtido assim descreve um cenario que o usuario final
      nunca tera', e por isso a saida marca o mode explicitamente.

    Sem rotulos preditos disponiveis (ex.: objetivo puramente exploratorio,
    sem classificador ajustado), o mode cego nao tem como operar; a funcao
    devolve o mode `controle-forcado`, que o chamador deve registrar na
    saida -- silenciar isso transformaria um resultado de controle num
    resultado aparentemente cego.
    """
    mode = str(getattr(cfg, "label_mode", "cego") or "cego").lower()
    if mode not in ("cego", "controle"):
        raise ValueError(
            f"cfg.label_mode='{mode}' invalido. Use 'cego' (padrao, usa a "
            f"classe predita) ou 'controle' (usa a classe verdadeira, so' "
            f"para diagnostico interno).")
    if mode == "controle":
        return np.asarray(rotulos_verdadeiros, dtype=str), "controle"
    if rotulos_preditos is None:
        return np.asarray(rotulos_verdadeiros, dtype=str), "controle-forcado"
    return np.asarray(rotulos_preditos, dtype=str), "cego"


def r2cv_species_by_adulterant(
        X: np.ndarray, conc: np.ndarray, rotulos: np.ndarray,
        mae_id: Optional[np.ndarray], cfg: "Config", *,
        limiar_r2: float = 0.70, min_niveis: int = 3,
        min_grupos: int = 3,
        mapa_adulterante: Optional[Dict[str, str]] = None,
        ) -> Optional[Dict[str, Any]]:
    """R2 em validacao cruzada (group-aware) do teor, POR especie x adulterante.

    A regressao agrupando ESPECIES falha (a matriz vegetal domina o sinal,
    R2~0); a que agrupa ADULTERANTES dentro da especie mascara que alguns
    adulterantes nao sao quantificaveis. A granularidade honesta e
    especie x adulterante: para cada combinacao, junta os PUROS da especie
    (teor=0, ancora da calibracao) + as amostras daquele adulterante, faz
    cross_val_predict group-aware por mae_id (replicas nunca se separam entre
    treino e teste) e calcula o R2cv. Combinacoes sem dados/replicas
    suficientes viram 'n/a' (nunca inventam numero).

    O adulterante de cada amostra vem do mae_id (que sobrevive alinhado a
    validate_input) via adulterant_from_mae_id -- evita desalinhar com o
    metadados_df, que NAO passa pela remocao de NaN/Inf.

    `mapa_adulterante` (Passo 122): repassado a `adulterant_from_mae_id`
    -- ver docstring de `identificacao.train_identification_ensemble`.

    Returns dict {especies, adulterantes, matriz{(esp,adult): r2|nan}, n_ok,
    n_falhas, n_na, n_total, limiar_r2} ou None se nao ha adulterante/combinacao.
    """
    if mae_id is None:
        return None
    conc = np.asarray(conc, dtype=float)
    conc = np.where(np.isnan(conc), 0.0, conc)
    adult_por_amostra = np.array(
        [adulterant_from_mae_id(m, mapa_adulterante) for m in mae_id],
        dtype=object)
    especies = sorted({str(r) for r in rotulos})
    adulterantes = sorted({a for a in adult_por_amostra if a})
    if not adulterantes:
        return None

    matriz: Dict[Tuple[str, str], float] = {}
    n_ok = n_falhas = n_na = 0
    for esp in especies:
        mask_esp = (rotulos == esp)
        mask_puro = mask_esp & (conc <= 0.0)
        for adult in adulterantes:
            mask_ad = mask_esp & (adult_por_amostra == adult) & (conc > 0.0)
            mask = mask_puro | mask_ad
            X_c, Y_c = X[mask], conc[mask]
            # grupos = unidade independente (B4c); o adulterante ja' foi
            # derivado do mae_id ORIGINAL acima.
            mae_c = chave_unidade_validacao(
                mae_id[mask], getattr(cfg, "group_unit", "mae_id"))
            n_niveis = int(len(np.unique(Y_c)))
            n_grp = int(len(np.unique(mae_c)))
            if (int(mask_ad.sum()) == 0 or n_niveis < min_niveis
                    or n_grp < min_grupos or float(Y_c.std()) < 1e-8):
                matriz[(esp, adult)] = float("nan")
                n_na += 1
                continue
            n_sp = max(2, min(cfg.n_splits_cv, n_grp))
            lv = max(1, min(cfg.max_lvs, X_c.shape[0] // 5, n_niveis - 1))
            pipe = Pipeline([
                ("preproc", build_preprocessor(cfg)),
                ("pls", PLSRegression(n_components=lv, scale=False)),
            ])
            try:
                Y_hat = cross_val_predict(pipe, X_c, Y_c,
                                          cv=GroupKFold(n_splits=n_sp),
                                          groups=mae_c)
            except (ValueError, np.linalg.LinAlgError) as _e_cv:
                # Combinacao degenerada apesar dos guards acima (grupos
                # insuficientes p/ o n_splits escolhido etc.) -- vira n/a,
                # nunca um R2 inventado.
                log.info(f"  [AVISO] R2cv {esp} x {adult}: {_e_cv}")
                matriz[(esp, adult)] = float("nan")
                n_na += 1
                continue
            r2 = float(r2_score(Y_c, Y_hat))
            matriz[(esp, adult)] = r2
            if r2 >= limiar_r2:
                n_ok += 1
            else:
                n_falhas += 1

    n_total = n_ok + n_falhas
    if n_total == 0:
        return None
    return {"especies": especies, "adulterantes": adulterantes,
            "matriz": matriz, "n_ok": n_ok, "n_falhas": n_falhas,
            "n_na": n_na, "n_total": n_total, "limiar_r2": limiar_r2}


def pls_regression_by_species(
        X_raw: np.ndarray, conc: np.ndarray, rotulos: np.ndarray,
        mae_id: Optional[np.ndarray], classes_unicas: np.ndarray,
        cfg: "Config", pasta: str, n_splits: int,
        min_amostras_adult: int = 6,
        rotulos_treino: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
    """Regressão PLS do teor de adulteração, calibrada SEPARADAMENTE por espécie.

    `rotulos` e' a classe que ROTEIA cada amostra para um modelo de teor (a PREVISTA no uso
    real). Por padrao e' tambem a que seleciona as amostras de treino de cada modelo (v1.0).
    Com `rotulos_treino` (a classe VERDADEIRA; opt-in `quantificacao_rotulo_treino: verdadeiro`)
    cada modelo treina so' com amostras da sua classe verdadeira, e a validacao reportada
    (R2val/RMSEP/vies agregados) roteia cada amostra de validacao pelo modelo da classe em
    `rotulos`, o que o usuario obtem de fato. A tabela por classe continua sendo o desempenho de
    cada modelo nas amostras de validacao da sua classe verdadeira.

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
    Ycv_all, Ycv_ref_all = [], []   # predicoes de CV por classe (RMSECV pooled)
    Gv_all: List[np.ndarray] = []
    tabela_esp: List[Dict[str, Any]] = []
    # Pipelines FINAIS (preproc+PLS ajustados na calibracao, mesma
    # combinacao ja validada contra o split de teste abaixo) por especie --
    # usadas para persistir Quantificar em amostra nova (Bloco 9b), nao so'
    # para reportar metricas de CV. Deploy o que foi validado, nao um refit
    # em cal+val pooled (que quebraria "isto e' exatamente o que foi testado").
    pipelines_especie: Dict[str, Dict[str, Any]] = {}
    erros_reg_repr: List[float] = []   # RMSECV curve from the largest species
    n_opt_repr = 1
    n_max_amostras = -1
    treino_verdadeiro = rotulos_treino is not None
    rot_selecao = np.asarray(rotulos_treino, dtype=str) if treino_verdadeiro else rotulos
    idx_val_glob: List[np.ndarray] = []   # indices globais das amostras de validacao (treino verdadeiro)

    for cls in classes_unicas:
        idx = np.where(rot_selecao == cls)[0]
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
        _split = _split_cal_val_por_especie(X_c, Y_c, mae_c, cfg)
        if _split is None:
            continue   # especie sem amostras/grupos suficientes p/ o split
        ic, iv = _split

        Xc, Yc = X_c[ic], Y_c[ic]
        Xv, Yv = X_c[iv], Y_c[iv]
        lv_max = min(cfg.max_lvs, max(2, Xc.shape[0] // 5))

        # internal CV for LV selection (group-aware)
        if mae_c is not None:
            grupos_cal = mae_c[ic]
            n_g = int(len(np.unique(grupos_cal)))
            n_sp = max(2, min(_dobras_regressao(cfg, n_splits), n_g))
            cv_reg = GroupKFold(n_splits=n_sp)
            grp = grupos_cal
        else:
            n_sp = max(2, min(_dobras_regressao(cfg, n_splits), Xc.shape[0] // 2))
            cv_reg = KFold(n_splits=n_sp, shuffle=True, random_state=cfg.seed)
            grp = None

        erros_reg: List[float] = []
        preds_cv_reg: List[np.ndarray] = []
        try:
            for n in range(1, lv_max + 1):
                pipe = Pipeline([
                    ("preproc", build_preprocessor(cfg)),
                    ("pls", PLSRegression(n_components=n, scale=False)),
                ])
                Y_hat = cross_val_predict(pipe, Xc, Yc, cv=cv_reg, groups=grp)
                erros_reg.append(rmse_flat(Yc, Y_hat))
                preds_cv_reg.append(np.asarray(Y_hat).flatten())
        except (ValueError, np.linalg.LinAlgError):
            continue   # LV degenerado p/ esta especie
        # (lv_max >= 2: o laco acima sempre acrescenta ao menos um erro, entao `erros_reg` nao fica vazio)

        n_opt_reg = int(np.argmin(erros_reg)) + 1
        # RMSECV pooled (V73): guarda as predicoes de CV desta classe no seu
        # proprio n_opt, para juntar todas as classes no fim.
        Ycv_all.append(preds_cv_reg[n_opt_reg - 1])
        Ycv_ref_all.append(np.asarray(Yc).flatten())
        pipe_final = Pipeline([
            ("preproc", build_preprocessor(cfg)),
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
        _fom = regression_figures_of_merit(
            pipe_final.named_steps["pls"], _X_cal_proc, _grupos_rep)

        # T1 (rodada multiagente 2026-09-10): margem de predicao conforme
        # para o teor de adulterante, fechando a lacuna #10 (Quantificacao
        # sem intervalo por amostra). O split de VALIDACAO (Xv/Yv/Yv_hat)
        # nunca foi visto pelo ajuste do modelo -- e' exatamente o
        # conjunto de calibracao que split-conformal exige (Lei et al.
        # 2018). Group-aware: cada mae_id colapsa ao PIOR residuo entre
        # replicas (ver conformal.conformal_margin_regression).
        _grupos_val = mae_c[iv] if mae_c is not None else None
        _conformal_reg = conformal_margin_regression(
            Yv.flatten(), Yv_hat, groups=_grupos_val, alpha=0.05)

        Yc_all.append(np.asarray(Yc).flatten())
        Ych_all.append(Yc_hat)
        Yv_all.append(np.asarray(Yv).flatten())
        Yvh_all.append(Yv_hat)
        idx_val_glob.append(idx[iv])
        Gv_all.append(np.array([f"{cls}|{g}" for g in (_grupos_val if _grupos_val is not None
                                                        else np.arange(len(iv)).astype(str))]))
        # Bloco 12 (D-like: reportar LOD/LOQ nunca isolado): faixa E
        # desvio-padrao do CONJUNTO DE VALIDACAO desta especie, para
        # contextualizar se o LOD/LOQ e' pequeno/grande frente ao que a
        # amostra real varia -- um LOD de 0,5% nao diz nada sozinho se
        # o teor de validacao varia so' entre 0% e 1%.
        _Yv_flat = np.asarray(Yv).flatten()
        tabela_esp.append({
            "especie": str(cls), "n_lv": n_opt_reg,
            "n_cal": int(len(ic)), "n_val": int(len(iv)),
            "rmsecv": float(erros_reg[n_opt_reg - 1]),
            "rmsep": rmsep_c, "r2val": r2v_c,
            "lod": _fom["lod"], "loq": _fom["loq"],
            "lod_ic_baixo": _fom["lod_ic_baixo"],
            "lod_ic_alto": _fom["lod_ic_alto"],
            "loq_ic_baixo": _fom["loq_ic_baixo"],
            "loq_ic_alto": _fom["loq_ic_alto"],
            "lod_ic_confianca": _fom["lod_ic_confianca"],
            "sensibilidade": _fom["sensibilidade"],
            "sensibilidade_analitica": _fom["sensibilidade_analitica"],
            "seletividade_media": _fom["seletividade_media"],
            "validacao_teor_min": float(_Yv_flat.min()),
            "validacao_teor_max": float(_Yv_flat.max()),
            "validacao_teor_dp": float(_Yv_flat.std(ddof=1)) if len(_Yv_flat) > 1 else float("nan"),
            "conformal_alcancavel": _conformal_reg["alcancavel"],
            "conformal_margem": _conformal_reg["limiar"],
            "conformal_n_grupos": _conformal_reg["n_grupos"],
        })
        pipelines_especie[str(cls)] = {
            "pipeline": pipe_final, "n_lv": n_opt_reg,
            "rmsep": rmsep_c, "r2val": r2v_c,
            # Bloco 24: LOD/LOQ (ja calculados acima em `_fom`) persistidos
            # junto com o pipeline -- sem isso, `quantify_sample` (que so
            # recebe UMA amostra nova, sem dado de calibracao/replicas) nao
            # teria como categorizar o teor predito em faixa de decisao.
            "lod": _fom["lod"], "loq": _fom["loq"],
            # T1: margem de predicao conforme persistida junto com o
            # pipeline -- `quantify_sample` so' recebe UMA amostra nova,
            # sem dado de calibracao, entao a margem precisa vir pronta.
            "conformal": _conformal_reg,
        }

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

    roteamento: Optional[Dict[str, Any]] = None
    manter_grupos = None
    if treino_verdadeiro:
        # Validacao pelo ROTEAMENTO PREVISTO: cada amostra de validacao (de qualquer classe
        # verdadeira) e' predita pelo modelo da classe que o classificador lhe atribuiu.
        iv_all = np.concatenate(idx_val_glob)
        y_ver = np.where(np.isnan(conc[iv_all]), 0.0, conc[iv_all])
        rota = np.asarray(rotulos, dtype=str)[iv_all]
        hat = np.full(len(iv_all), np.nan)
        for cls_prev, info in pipelines_especie.items():
            m = rota == cls_prev
            if m.any():
                hat[m] = np.asarray(info["pipeline"].predict(X_raw[iv_all[m]])).ravel()
        ok = np.isfinite(hat)
        verdadeira_val = np.asarray(rot_selecao, dtype=str)[iv_all]
        roteamento = {
            "modo": "verdadeiro", "n_val": int(ok.sum()), "n_sem_modelo": int((~ok).sum()),
            "acuracia_roteamento": float(np.mean(rota[ok] == verdadeira_val[ok])) if ok.any() else float("nan"),
            "r2v_classe_verdadeira": (float(r2_score(Yv_p, Yvh_p)) if len(np.unique(Yv_p)) > 1
                                      else float("nan")),
            "rmsep_classe_verdadeira": rmse_flat(Yv_p, Yvh_p),
        }
        if ok.sum() >= 2:
            Yv_p, Yvh_p = y_ver[ok], hat[ok]
            manter_grupos = ok

    r2c = float(r2_score(Yc_p, Ych_p)) if len(np.unique(Yc_p)) > 1 else float("nan")
    r2v = float(r2_score(Yv_p, Yvh_p)) if len(np.unique(Yv_p)) > 1 else float("nan")
    rmsec = rmse_flat(Yc_p, Ych_p)
    rmsep = rmse_flat(Yv_p, Yvh_p)
    # RMSECV pooled: predicoes de CV de TODAS as classes, cada uma no seu
    # n_opt (mesma populacao do RMSEC e do RMSEP pooled). Ate a v1.0-rc
    # anterior era o minimo da curva so' da classe com mais amostras (V73).
    rmsecv = (rmse_flat(np.concatenate(Ycv_ref_all), np.concatenate(Ycv_all))
              if Ycv_all else rmsec)
    # R2cv (Q2 da regressao): 1 - PRESS/SS_tot sobre as MESMAS predicoes de CV do RMSECV acima.
    _ycv_ref = np.concatenate(Ycv_ref_all) if Ycv_all else np.array([])
    r2cv = (float(r2_score(_ycv_ref, np.concatenate(Ycv_all)))
            if Ycv_all and len(np.unique(_ycv_ref)) > 1 else float("nan"))
    bias_v = float(np.mean(Yvh_p - Yv_p))

    # pooled diagonal figure (proper diagonal: within-species calibration)
    if should_generate(cfg, "regressao"):
        fig7_pls_regression(Yc_p, Ych_p, Yv_p, Yvh_p, erros_reg_repr or [rmsec],
                            n_opt_repr, r2c, r2v, rmsec, rmsecv, rmsep, bias_v,
                            cfg, pasta)
        # Figura de merito analitica dedicada (auditoria jul/2026, item 5):
        # LOD/LOQ/Seletividade por especie, ate aqui so' em texto no resumo.
        fig_regression_merit(tabela_esp, cfg, pasta)

    # DModY (Eriksson et al. 2006) -- mesma reapresentacao do residuo de
    # validacao ja usado no RMSEP/bias acima, na nomenclatura SIMCA-P/
    # Unscrambler. n_opt_repr (LVs da especie com mais amostras) usado como
    # A representativo do pool multi-especie.
    _dmody_res = dmody(Yvh_p - Yv_p, n_componentes=n_opt_repr,
                       n_amostras=len(Yv_p))

    # RPD/RER: um RMSEP sozinho nao diz se o modelo serve para alguma coisa.
    # As razoes normalizam o erro pela variacao do proprio conjunto de
    # referencia e vem com a faixa de uso da literatura ao lado.
    _razoes = rpd_rer(Yv_p, Yvh_p)

    return {
        "tabela_especie": tabela_esp,
        "pipelines_especie": pipelines_especie,
        "r2c": r2c, "r2v": r2v, "rmsec": rmsec, "rmsecv": rmsecv, "r2cv": r2cv,
        "rmsep": rmsep, "bias": bias_v, "n_especies": len(tabela_esp),
        "sep": _razoes["sep"], "rpd": _razoes["rpd"], "rer": _razoes["rer"],
        "rpd_faixa": interpret_rpd(_razoes["rpd"]),
        "dmody_crit": _dmody_res["dmody_crit"],
        "n_fora_do_dmody": _dmody_res["n_fora_do_modelo"],
        "validacao_y": Yv_p, "validacao_y_hat": Yvh_p, "n_lv_repr": n_opt_repr,
        "validacao_grupos": (None if not all(g is not None for g in Gv_all) else
                             (np.concatenate(Gv_all) if manter_grupos is None
                              else np.concatenate(Gv_all)[manter_grupos])),
        "roteamento": roteamento,
    }


def _anexar_linearidade_robustez(pasta_logs: str, y_val: np.ndarray, y_val_hat: np.ndarray,
                                 grupos_val: Optional[np.ndarray], cfg: "Config", X_raw: np.ndarray,
                                 conc: np.ndarray, mae_id: Optional[np.ndarray], n_lv: int) -> Dict[str, Any]:
    """Linearidade formal (falta de ajuste) nas predicoes de validacao -- cada amostra fisica e' um
    nivel da curva; sem grupos, os niveis sao os valores de referencia -- e, se `protocolo_robustez`
    estiver ligado, o protocolo de robustez. Anexa ao model card; nao muda o modelo."""
    lin = lack_of_fit_test(np.ravel(y_val), np.ravel(y_val_hat),
                           grupos_val if grupos_val is not None else np.ravel(y_val),
                           nivel_por="grupo" if grupos_val is not None else "conc")
    log.info("  Linearidade (falta de ajuste): " + (f"F={lin.F:.3f}, p={lin.p_value:.4g}" if lin.computavel
                                                     else f"nao computavel -- {lin.motivo}"))
    if lin.computavel and not lin.linear and lin.aviso_erro_puro:
        log.info("    [RESSALVA] rejeicao NAO conclusiva: " + lin.aviso_erro_puro)
    rob = None
    if getattr(cfg, "robustness_protocol", False):
        from guaraci.robustness import protocolo_padrao
        log.info("  Protocolo de robustez (ruido, deriva, variantes de pre-processamento)...")
        rob = protocolo_padrao(X_raw, np.asarray(conc, dtype=float), mae_id, cfg, "quantificacao", n_lv)
    append_linearity_robustness_model_card(pasta_logs, linearidade=lin, robustez=rob)
    return {"linearidade": lin, "robustez": rob}


def _dobras_regressao(cfg: "Config", n_splits_classificacao: int) -> int:
    """Dobras da CV da regressao (teor). `n_splits` vem da CLASSIFICACAO e e' limitado pela
    classe com menos grupos (ex.: 2 com uma especie de 2 sessoes); a regressao nao estratifica
    por classe, entao o limite dela e' so' o numero de grupos do proprio ajuste. Com 2 dobras o
    modelo treina com metade dos niveis de teor e o R2cv sai pessimista (medido: mediana do
    R2cv 0,33 com 2 dobras contra 0,47 com deixa-um-nivel-fora no mesmo dado)."""
    return max(int(n_splits_classificacao), int(getattr(cfg, "n_splits_cv", 5) or 5))


def pls_regressao_pooled(
        X_raw: np.ndarray, conc: np.ndarray, rotulos: np.ndarray,
        mae_id: Optional[np.ndarray], cfg: "Config", pasta: str,
        pasta_logs: str, n_splits: int,
        X_holdout: Optional[np.ndarray] = None,
        conc_holdout: Optional[np.ndarray] = None,
        wavenumbers: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Regressão PLS do teor de adulteração, UM UNICO MODELO sobre todas as
    amostras (sem separar por especie).

    Extraída de `executar()` em 2026-08-20 (Passo 27 do plano de dívida de
    `executar()`/`guaraci.py`): antes vivia inline, sem função própria e
    sem teste de contrato dedicado. É o caminho que roda sempre que
    `pls_regression_by_species` devolve `None` -- dataset de espécie
    única, nível N1, ou nenhuma espécie com amostras adulteradas
    suficientes -- não é caso raro, é o caminho normal para dataset
    pequeno ou de espécie única.

    Split cal/val e CV interna de seleção de LVs são group-aware por
    `mae_id`, no mesmo padrão de `pls_regression_by_species` -- ver
    `tests/test_contrato_validacao_agrupada.py`.
    """
    log.info(f"\n[7/7] PLS regressao "
          f"(alvo=resposta, {int(np.sum(conc > 0))} com resposta > 0 + "
          f"{int(np.sum(conc == 0))} com resposta = 0)")
    Y_reg = np.asarray(conc, dtype=float).reshape(-1, 1)

    # Calibration/validation split — group-aware if mae_id available
    # (T1/T2/T3 replicates of the same sample point never split between cal/val).
    if cfg.cal_val_split == "kennard_stone":
        ic, iv = kennard_stone_split_group_aware(
            X_raw, mae_id, cfg.frac_cal)
        log.info(f"  Split cal/val: Kennard-Stone "
              f"({len(ic)} cal / {len(iv)} val)")
    elif mae_id is not None:
        gss_reg = GroupShuffleSplit(n_splits=1, train_size=cfg.frac_cal,
                                     random_state=cfg.seed)
        ic, iv = next(gss_reg.split(X_raw, Y_reg, groups=mae_id))
        log.info(f"  Split cal/val: GroupShuffleSplit por mae_id "
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
        n_splits_reg = max(2, min(_dobras_regressao(cfg, n_splits), n_grupos_cal))
        cv_reg = GroupKFold(n_splits=n_splits_reg)
        grupos_cv_reg: Optional[np.ndarray] = grupos_cal
        log.info(f"  CV interna: GroupKFold n_splits={n_splits_reg} "
              f"({n_grupos_cal} grupos na calibracao)")
    else:
        n_splits_reg = max(2, min(_dobras_regressao(cfg, n_splits), Xc_raw.shape[0] // 2))
        cv_reg = KFold(n_splits=n_splits_reg, shuffle=True,
                        random_state=cfg.seed)
        grupos_cv_reg = None

    erros_reg = []
    preds_reg = []
    for n in range(1, lv_max + 1):
        pipe = Pipeline([
            ("preproc", build_preprocessor(cfg)),
            ("pls", PLSRegression(n_components=n, scale=False)),
        ])
        Y_hat = cross_val_predict(pipe, Xc_raw, Yc, cv=cv_reg,
                                   groups=grupos_cv_reg)
        erros_reg.append(rmse_flat(Yc, Y_hat))
        preds_reg.append(Y_hat)

    n_opt_reg = int(np.argmin(erros_reg)) + 1
    pipe_final = Pipeline([
        ("preproc", build_preprocessor(cfg)),
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
    r2cv   = float(r2_score(Yc, Yc_cv)) if len(np.unique(np.asarray(Yc))) > 1 else float("nan")

    if should_generate(cfg, "regressao"):
        # Pontos do HOLDOUT externo so' para as figuras (o RMSEP publicavel
        # do holdout continua sendo calculado no bloco M-05b abaixo).
        _ho_fig = _predicoes_holdout_para_figura(pipe_final, X_holdout,
                                                 conc_holdout)
        fig7_pls_regression(Yc, Yc_hat, Yv, Yv_hat, erros_reg, n_opt_reg,
                            r2c, r2v, rmsec, rmsecv, rmsep, bias_v, cfg, pasta,
                            **cast(Dict[str, Any], {} if _ho_fig is None else {
                                "y_holdout": _ho_fig[0],
                                "yhat_holdout": _ho_fig[1],
                                "rmsep_holdout": rmse_flat(*_ho_fig)}))
        _figuras_modelo_unico_regressao(
            pipe_final, Xc_raw, Yc, Yc_hat, Xv_raw, Yv, Yv_hat,
            X_holdout, conc_holdout, _ho_fig, wavenumbers, cfg, pasta)

    # DModY (Eriksson et al. 2006) -- mesma reapresentacao do residuo de
    # validacao ja usado no RMSEP/bias acima.
    _dmody_res_pooled = dmody(
        np.asarray(Yv_hat).flatten() - np.asarray(Yv).flatten(),
        n_componentes=n_opt_reg, n_amostras=len(Yv))

    log.info(f"  LVs    : {n_opt_reg}")
    log.info(f"  RMSEC  : {rmsec:.3f}  |  RMSECV: {rmsecv:.3f}  "
          f"|  RMSEP: {rmsep:.3f}")
    log.info(f"  R2cal  : {r2c:.4f}  |  R2val : {r2v:.4f}  "
          f"|  Bias: {bias_v:.4f}")
    # RPD/RER da regressao pooled: um RMSEP sozinho nao diz se o modelo
    # serve para alguma coisa -- as razoes normalizam o erro pela
    # variacao do proprio conjunto de referencia, e a faixa de uso vem
    # junto para o numero nao sair nu (Williams 2014; AACC 39-00.01).
    _razoes_pooled = rpd_rer(Yv, Yv_hat)
    _faixa_pooled = interpret_rpd(_razoes_pooled["rpd"])
    log.info(f"  SEP    : {_razoes_pooled['sep']:.3f}  "
             f"|  RPD: {_razoes_pooled['rpd']:.2f}  "
             f"|  RER: {_razoes_pooled['rer']:.1f}  ->  {_faixa_pooled}")

    # Figuras de merito analiticas (Valderrama, Braga & Poppi, 2009):
    # ruido instrumental estimado a partir de replicas fisicas (T1/T2/T3
    # via mae_id) SOMENTE do lado de calibracao.
    _preproc_ajustado_reg = pipe_final.named_steps["preproc"]
    _X_cal_proc_reg = np.asarray(_preproc_ajustado_reg.transform(Xc_raw))
    _grupos_rep_reg = _agrupar_replicas_processadas(
        Xc_raw, mae_id[ic] if mae_id is not None else None,
        _preproc_ajustado_reg)
    _fom_reg = regression_figures_of_merit(
        pipe_final.named_steps["pls"], _X_cal_proc_reg, _grupos_rep_reg)
    # Figura de merito dedicada (auditoria jul/2026, item 5): caminho
    # single-especie so' tem 1 modelo pooled, entao a "tabela" tem 1 linha.
    _especies_unicas_reg = np.unique(rotulos)
    _nome_esp_pooled = (str(_especies_unicas_reg[0])
                         if len(_especies_unicas_reg) == 1 else "Pooled")
    if should_generate(cfg, "regressao"):
        fig_regression_merit([{
            "especie": _nome_esp_pooled,
            "lod": _fom_reg["lod"], "loq": _fom_reg["loq"],
            "seletividade_media": _fom_reg["seletividade_media"],
        }], cfg, pasta)
    if np.isfinite(_fom_reg["lod"]):
        log.info(f"  LOD    : {_fom_reg['lod']:.2f}  |  "
              f"LOQ: {_fom_reg['loq']:.2f}  (unidade do teor; aproximado: so' o ruido de "
              f"replica; sem leverage nem erro de referencia -> tende a "
              f"SUBestimar)")
        log.info(f"  SEN    : {_fom_reg['sensibilidade']:.3f}  |  "
              f"gamma: {_fom_reg['sensibilidade_analitica']:.2f}  |  "
              f"SEL: {_fom_reg['seletividade_media']:.3f}")
        # R3-03 (revisao cega): LOD abaixo do erro de predicao e' incompativel --
        # com modelo sem poder preditivo o LOD saia 0,01-0,02% com RMSEP ~ SD(y).
        if _fom_reg["lod"] < rmsep:
            log.warning(
                f"  [!] LOD ({_fom_reg['lod']:.2f}) esta ABAIXO do RMSEP "
                f"({rmsep:.2f}): o LOD calculado so' com ruido de replica e' "
                f"irreal. NAO o cite como limite de deteccao; use o RMSEP como "
                f"piso do que o modelo distingue.")
    else:
        log.info("  LOD/LOQ: N/A (sem replicas fisicas suficientes para "
              "estimar ruido instrumental)")
    log.info(f"  DModY critico (SIMCA): {_dmody_res_pooled['dmody_crit']:.3f}"
          f"  |  amostras fora: {_dmody_res_pooled['n_fora_do_modelo']}")
    # M-05b: RMSEP no HOLDOUT externo (o rmsep acima vem do split interno).
    _ho_pooled: Optional[Dict[str, Any]] = None
    if X_holdout is not None and conc_holdout is not None:
        try:
            _ch = np.asarray(conc_holdout, dtype=float)
            _ok = ~np.isnan(_ch)
            if int(_ok.sum()) >= 2:
                _yh = np.asarray(pipe_final.predict(X_holdout[_ok])).ravel()
                _ho_pooled = {
                    "n": int(_ok.sum()), "n_sem_modelo": 0,
                    "rmsep": rmse_flat(_ch[_ok], _yh),
                    "r2": (float(r2_score(_ch[_ok], _yh))
                           if len(np.unique(_ch[_ok])) > 1 else float("nan")),
                    "bias": float(np.mean(_yh - _ch[_ok])),
                    "tabela_especie": []}
                log.info(f"  RMSEP (HOLDOUT externo, n={_ho_pooled['n']}): "
                         f"{_ho_pooled['rmsep']:.3f}  [numero publicavel; o "
                         f"RMSEP interno vem de split 70/30]")
        except Exception as _e_hp:  # noqa: BLE001 -- extra de avaliacao
            log.info(f"  [AVISO] RMSEP do holdout nao calculado: {_e_hp}")
    resultado_pooled = {
        "r2c": r2c, "r2v": r2v, "rmsec": rmsec, "r2cv": r2cv,
        "rmsecv": rmsecv, "rmsep": rmsep, "bias": bias_v,
        "sep": _razoes_pooled["sep"],
        "rpd": _razoes_pooled["rpd"],
        "rer": _razoes_pooled["rer"],
        "rpd_faixa": _faixa_pooled,
        "dmody_crit": _dmody_res_pooled["dmody_crit"],
        "n_fora_do_dmody": _dmody_res_pooled["n_fora_do_modelo"],
        "holdout": _ho_pooled,
    }
    append_regression_summary(pasta_logs, pooled=resultado_pooled,
                            fom_pooled=_fom_reg, holdout=_ho_pooled)
    append_regression_model_card(pasta_logs, pooled=resultado_pooled,
                                fom_pooled=_fom_reg, holdout=_ho_pooled)
    resultado_pooled.update(_anexar_linearidade_robustez(
        pasta_logs, Yv, Yv_hat, mae_id[iv] if mae_id is not None else None, cfg, X_raw, conc, mae_id, n_opt_reg))
    return resultado_pooled


def _predicoes_holdout_para_figura(
        pipe_final: Pipeline, X_holdout: Optional[np.ndarray],
        conc_holdout: Optional[np.ndarray]
        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(y, y_hat) do holdout externo pelo modelo final, so' para desenhar.

    Mesmo filtro do RMSEP do holdout (resposta nao-NaN, n >= 2). Qualquer
    falha devolve None: a figura sai como antes, sem os pontos externos."""
    if X_holdout is None or conc_holdout is None:
        return None
    try:
        ch = np.asarray(conc_holdout, dtype=float).ravel()
        ok = ~np.isnan(ch)
        if int(ok.sum()) < 2:
            return None
        yh = np.asarray(pipe_final.predict(np.asarray(X_holdout)[ok])).ravel()
        return ch[ok], yh
    except Exception as e:  # noqa: BLE001 -- so' figura
        log.info(f"  [AVISO] pontos do holdout fora da figura: {e}")
        return None


def _figuras_modelo_unico_regressao(
        pipe_final: Pipeline, Xc_raw: np.ndarray, Yc: np.ndarray,
        Yc_hat: np.ndarray, Xv_raw: np.ndarray, Yv: np.ndarray,
        Yv_hat: np.ndarray, X_holdout: Optional[np.ndarray],
        conc_holdout: Optional[np.ndarray],
        ho_fig: Optional[Tuple[np.ndarray, np.ndarray]],
        wavenumbers: Optional[np.ndarray], cfg: "Config", pasta: str) -> None:
    """Coeficientes b x eixo espectral e grafico de Williams do modelo PLS
    UNICO (pooled) final. So' desenha: nada aqui entra em CSV/JSON, e uma
    falha e' registrada sem derrubar a corrida (mesmo padrao das figuras
    complementares). Validacao e holdout sao projetados no modelo e
    padronizados pelo s da calibracao."""
    try:
        from guaraci import figuras_extras as fx
        pls = pipe_final.named_steps["pls"]
        prep = pipe_final.named_steps["preproc"]
        A = int(pls.n_components)
        b = np.ravel(np.asarray(pls.coef_, dtype=float))
        if wavenumbers is not None and np.asarray(wavenumbers).size == b.size:
            fx.fig_coeficientes_regressao(
                np.asarray(wavenumbers, dtype=float), {"PLS": b}, cfg, pasta,
                n_lv={"PLS": A})
        else:
            log.info("  [AVISO] coeficientes b sem eixo espectral compativel; "
                     "figura nao gerada")
        T_cal = np.asarray(pls.transform(prep.transform(Xc_raw)))
        r_cal, s_cal = fx.residuos_padronizados(Yc, Yc_hat, A)
        mod: Dict[str, Any] = {
            "h_cal": fx.alavancagem_scores(T_cal), "r_cal": r_cal, "n_lv": A}
        if len(np.ravel(Yv)) > 0:
            T_val = np.asarray(pls.transform(prep.transform(Xv_raw)))
            mod["h_val"] = fx.alavancagem_scores(T_val, T_cal)
            mod["r_val"] = fx.residuos_padronizados(Yv, Yv_hat, A, s=s_cal)[0]
        if ho_fig is not None and X_holdout is not None and conc_holdout is not None:
            ok = ~np.isnan(np.asarray(conc_holdout, dtype=float).ravel())
            T_ho = np.asarray(pls.transform(
                prep.transform(np.asarray(X_holdout)[ok])))
            mod["h_holdout"] = fx.alavancagem_scores(T_ho, T_cal)
            mod["r_holdout"] = fx.residuos_padronizados(
                ho_fig[0], ho_fig[1], A, s=s_cal)[0]
        fx.fig_williams({"PLS": mod}, cfg, pasta)
    except Exception as e:  # noqa: BLE001 -- figura complementar, opcional
        log.info(f"  [AVISO] coeficientes/Williams do modelo final nao "
                 f"gerados: {e}")
        import matplotlib.pyplot as plt
        plt.close("all")


def _novo_pls_classificador(cfg: "Config", n_componentes: int) -> PLSRegression:
    """O PLS do classificador principal: PLS-DA (padrao) ou PLS-LDA (`cfg.classificador`)."""
    nome = str(getattr(cfg, "classificador", "plsda") or "plsda").lower()
    if nome == "plslda":
        return PLSLDAClassifier(n_components=n_componentes, scale=False)
    if nome != "plsda":
        raise ValueError(f"classificador='{nome}' invalido. Use 'plsda' (padrao) ou 'plslda'.")
    return PLSRegression(n_components=n_componentes, scale=False)


def _predicoes_pls_ate_k(pls: PLSRegression, Xp_tr: np.ndarray,
                         Y_tr: np.ndarray, Xp_va: np.ndarray
                         ) -> List[np.ndarray]:
    """Predicoes de validacao para 1..K VLs a partir de UM PLS com K VLs.

    O NIPALS do sklearn extrai as VLs uma a uma, com deflacao: as k
    primeiras de um ajuste com K VLs sao as mesmas de um ajuste com k. O
    modelo de k VLs e' entao B_k = W_k (P_k^T W_k)^-1 C_k^T (a mesma
    formula que o sklearn usa para `coef_`), com as medias de treino como
    centro (scale=False). So' atributos publicos do PLS ajustado."""
    W, P, C = pls.x_weights_, pls.x_loadings_, pls.y_loadings_
    x_media = Xp_tr.mean(axis=0)
    y_media = Y_tr.mean(axis=0)
    Xc = Xp_va - x_media
    saidas = []
    for k in range(1, W.shape[1] + 1):
        R = W[:, :k] @ np.linalg.pinv(P[:, :k].T @ W[:, :k])
        saidas.append(Xc @ (R @ C[:, :k].T) + y_media)
    return saidas


def _predicoes_cv_por_lv(fabrica_pipeline: Callable[[int], Pipeline],
                         X: np.ndarray, Y_bin: np.ndarray,
                         cv_indices: List[Tuple[np.ndarray, np.ndarray]],
                         n_max: int) -> Dict[int, np.ndarray]:
    """Predicoes de CV (media entre repeticoes) para cada n de 1 a n_max.

    Antes: um ajuste de pre-processamento + PLS por (n, fold), n_max vezes
    o trabalho. Agora: um ajuste por fold com n_max VLs, e as predicoes
    para cada n menor saem desse ajuste (`_predicoes_pls_ate_k`). Medido
    (243 x 1500, 40 VLs): 29,3 s -> 1,8 s, mesmo n_opt. Se o pipeline nao
    for o par ('preproc', PLSRegression(scale=False)), cai no caminho
    antigo."""
    modelo = fabrica_pipeline(n_max)
    pls_modelo = modelo.steps[-1][1] if len(modelo.steps) == 2 else None
    rapido = (isinstance(pls_modelo, PLSRegression)
              and not pls_modelo.scale and modelo.steps[0][0] == "preproc")
    somas = {n: np.zeros_like(Y_bin, dtype=float) for n in range(1, n_max + 1)}
    contador = np.zeros(len(Y_bin), dtype=int)
    for tr, va in cv_indices:
        if rapido:
            pipe = fabrica_pipeline(n_max)
            pre = pipe.steps[0][1]
            Xp_tr = pre.fit_transform(X[tr], Y_bin[tr])
            pls = pipe.steps[1][1].fit(Xp_tr, Y_bin[tr])
            if hasattr(pls, "predicoes_ate_k"):      # PLS-LDA: LDA sobre os escores de cada k
                preds = pls.predicoes_ate_k(Xp_tr, Y_bin[tr], pre.transform(X[va]))
            else:
                preds = _predicoes_pls_ate_k(pls, Xp_tr, Y_bin[tr],
                                             pre.transform(X[va]))
            for n in range(1, n_max + 1):
                somas[n][va] += preds[n - 1]
        else:
            for n in range(1, n_max + 1):
                pipe = fabrica_pipeline(n)
                pipe.fit(X[tr], Y_bin[tr])
                somas[n][va] += pipe.predict(X[va])
        contador[va] += 1
    contador[contador == 0] = 1
    return {n: somas[n] / contador[:, None] for n in somas}


def _selecionar_n_opt_wold(fabrica_pipeline: Callable[[int], Pipeline],
                            X: np.ndarray, Y_bin: np.ndarray,
                            cv_indices: List[Tuple[np.ndarray, np.ndarray]],
                            max_lvs: int) -> Tuple[int, np.ndarray]:
    """Escolhe `n_opt` pela regra pratica de parcimonia (tolerancia de 2%) sobre
    `cv_indices` -- extraido do corpo de `executar()` (Passo 213, correcao
    do achado #12/Passo 211) para ser reusado tanto na CV EXTERNA (n_opt do
    modelo final, como sempre foi) quanto, quando `cfg.selecao_lv_cv_
    aninhada=True`, na CV INTERNA de cada fold externo (nunca vendo a
    validacao daquele fold -- e' isso que remove o vies de reusar o mesmo
    dado pra' escolher `n_opt` E avaliar a metrica reportada).

    `max_lvs` e' sempre limitado pelo menor fold de treino em `cv_indices`
    (mesmo padrao de `hsi_classification.select_n_components_wold`,
    `X.shape[0] - 1`) -- achado real ao aplicar isso na CV INTERNA (um
    subconjunto ja' reduzido do treino externo): `PLSRegression` levanta
    `ValueError` se `n_components` exceder `min(n_amostras, n_variaveis)`
    do fold, algo que o treino externo (maior) tolerava mas um fold interno
    pequeno nao."""
    min_tr = min(len(tr) for tr, _ in cv_indices)
    max_lvs_viavel = max(1, min(max_lvs, X.shape[1], min_tr - 1))
    preds = _predicoes_cv_por_lv(fabrica_pipeline, X, Y_bin, cv_indices,
                                 max_lvs_viavel)
    erros = [rmse_flat(Y_bin, preds[n]) for n in range(1, max_lvs_viavel + 1)]
    erros_arr = np.array(erros)
    tol = erros_arr.min() * 1.02
    n_opt = int(np.where(erros_arr <= tol)[0][0]) + 1
    return n_opt, preds[n_opt]


def _construir_cv_interno(y_int_sub: np.ndarray, grupos_sub: Optional[np.ndarray],
                           usar_grupos: bool, n_splits_alvo: int, seed: int):
    """CV interna de 1 fold externo, para a selecao aninhada de `n_opt`
    (Passo 213). Mesma logica de ajuste de `n_splits` por contagem minima de
    classe/grupo que a CV externa (`executar()`, bloco "2. LV selection"),
    aplicada ao subconjunto de treino do fold externo -- sempre 1 passada
    (nunca `RepeatedStratifiedKFold`: repeticoes sao uma tecnica de
    ESTABILIDADE da estimativa externa, nao fazem sentido pra' escolher um
    hiperparametro dentro de 1 fold ja' interno, e multiplicariam ainda mais
    o custo computacional, ja' ~4x maior so' com 1 passada)."""
    classes_sub, contagem_sub = np.unique(y_int_sub, return_counts=True)
    n_splits = min(n_splits_alvo, int(np.min(contagem_sub)))
    if usar_grupos and grupos_sub is not None:
        n_grupos_por_classe = [int(len(np.unique(grupos_sub[y_int_sub == c])))
                                for c in classes_sub]
        n_splits = min(n_splits, min(n_grupos_por_classe))
        return StableStratifiedGroupKFold(n_splits=max(n_splits, 2), seed=seed)
    return StratifiedKFold(n_splits=max(n_splits, 2), shuffle=True,
                            random_state=seed)


def avaliar_quantificacao_holdout(
        pipelines_especie: Dict[str, Dict[str, Any]], X_holdout: np.ndarray,
        rotulos_usados: np.ndarray, conc_holdout: np.ndarray,
        ) -> Optional[Dict[str, Any]]:
    """RMSEP no HOLDOUT externo.

    Antes, o holdout de 25% era reservado e NUNCA avaliado pela regressao: o
    "RMSEP" publicado vinha de um split interno 70/30 (n_val pequeno).
    Aqui os pipelines FINAIS por especie (os mesmos que vao para o modelo
    exportado) predizem o holdout, que nao participou de nenhuma escolha
    (LVs, pre-processamento, calibracao). `rotulos_usados` sao os rotulos que
    a quantificacao enxerga -- preditos no modo cego, verdadeiros no de
    controle. Amostras cuja classe nao tem modelo de regressao ficam de fora
    e sao contadas.
    """
    conc = np.asarray(conc_holdout, dtype=float)
    conc = np.where(np.isnan(conc), 0.0, conc)
    rot = np.asarray(rotulos_usados, dtype=str)
    linhas: List[Dict[str, Any]] = []
    Ys: List[np.ndarray] = []
    Yh: List[np.ndarray] = []
    for cls, info in pipelines_especie.items():
        m = rot == str(cls)
        if int(m.sum()) < 2:
            continue
        y = conc[m]
        yhat = np.asarray(info["pipeline"].predict(X_holdout[m])).ravel()
        linhas.append({
            "especie": str(cls), "n": int(m.sum()),
            "rmsep": rmse_flat(y, yhat),
            "r2": (float(r2_score(y, yhat)) if len(np.unique(y)) > 1
                   else float("nan")),
            "bias": float(np.mean(yhat - y))})
        Ys.append(y)
        Yh.append(yhat)
    if not Ys:
        return None
    y_all, yh_all = np.concatenate(Ys), np.concatenate(Yh)
    return {
        "n": int(len(y_all)),
        "n_sem_modelo": int(len(rot) - len(y_all)),
        "rmsep": rmse_flat(y_all, yh_all),
        "r2": (float(r2_score(y_all, yh_all)) if len(np.unique(y_all)) > 1
               else float("nan")),
        "bias": float(np.mean(yh_all - y_all)),
        "tabela_especie": linhas,
    }


_MAIOR_SUBCAMINHO_DE_SAIDA = 75    # ex.: Modelos/modelo_plsda.portatil.json + folga


def _confere_comprimento_de_saida(pasta: str) -> None:
    """Recusa cedo, com mensagem acionavel, uma pasta de saida cujo caminho
    mais longo previsivel passaria do limite MAX_PATH do Windows (260)."""
    if os.name != "nt":
        return
    total = len(os.path.abspath(pasta)) + _MAIOR_SUBCAMINHO_DE_SAIDA
    if total >= 260:
        raise ValueError(
            f"O caminho da pasta de saida tem {len(os.path.abspath(pasta))} "
            f"caracteres; com os arquivos internos passaria de 260, o limite do "
            f"Windows. Escolha uma pasta_saida mais curta (ex.: "
            f"C:/guaraci_saida) -- pastas sincronizadas com a nuvem e nomes longos "
            f"de projeto/perfil somam rapido.")


def holdout_agrupado_estratificado(y: np.ndarray, groups: np.ndarray,
                                   frac: float, seed: int
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Holdout que mantem os grupos inteiros E estratifica por classe.

    Usa o `StableStratifiedGroupKFold` do projeto com `k = round(1/frac)`
    dobras e toma como holdout a dobra de tamanho mais proximo de `frac*n`
    -- a mesma heuristica gulosa da CV (particao estavel entre versoes de
    biblioteca). Retorna (indices_treino, indices_holdout) relativos a `y`.
    Se uma classe tem um unico grupo, ela so' cabe de um lado; quem chama
    confere a cobertura de classes e avisa."""
    y = np.asarray(y)
    k = max(2, int(round(1.0 / max(frac, 1e-6))))
    k = min(k, len(np.unique(groups)))
    if k < 2:
        raise ValueError("grupos insuficientes para um holdout agrupado")
    dobras = [va for _tr, va in StableStratifiedGroupKFold(
        n_splits=k, seed=seed).split(np.zeros(len(y)), y, groups=groups)]
    alvo = frac * len(y)
    ho = min(dobras, key=lambda v: (abs(len(v) - alvo), len(v)))
    groups = np.asarray(groups)
    # Cobertura de classes: a heuristica gulosa pode deixar de fora do holdout
    # uma classe que TEM >=2 grupos (o holdout ficaria cego a ela). Move para
    # o holdout o MENOR grupo dessa classe (desempate por hash estavel), e a
    # classe continua com >=1 grupo no treino. O holdout pode passar um pouco
    # de `frac`; isso e' preferivel a uma classe invisivel.
    ho_set = set(ho.tolist())
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        if any(i in ho_set for i in idx_c):
            continue
        grupos_c = np.unique(groups[idx_c])
        if len(grupos_c) < 2:
            continue                      # um grupo so': nao ha' como dividir
        escolhido = min(
            grupos_c,
            key=lambda gg: (int(np.sum(groups[idx_c] == gg)),
                            StableStratifiedGroupKFold._chave_ordem(gg, seed)))
        ho_set.update(idx_c[groups[idx_c] == escolhido].tolist())
    ho = np.array(sorted(ho_set))
    tr = np.setdiff1d(np.arange(len(y)), ho, assume_unique=False)
    return tr, ho


def _avisar_classes_em_rampa(*, cfg: "Config", T_pls: np.ndarray, rotulos: np.ndarray,
                             pred_lab: np.ndarray, X_raw: np.ndarray, Y_bin: np.ndarray, y_int: np.ndarray,
                             cv_indices: Any, lv_cap: int, n_opt: int,
                             metricas_por_lv: List[Dict[str, float]], pasta_logs: str) -> Optional[Dict[str, Any]]:
    """Classes ordenadas (niveis de uma mesma propriedade): o PLS-DA mascara a classe do meio.
    Detecta (`classificadores.detectar_classes_em_rampa`: medias das classes colineares nos escores E
    sensibilidade da classe do meio muito menor que a das pontas) e avisa no console, no resumo e no
    model card. Com `classificador: plsda`, compara com o PLS-LDA na MESMA validacao cruzada (nenhuma
    das duas aninhada, para a comparacao ser justa). Nao muda nenhum resultado."""
    from guaraci.resultados_io import anexar_bloco_modelo
    try:
        det = detectar_classes_em_rampa(T_pls, rotulos, pred_lab)
        if not det["detectado"]:
            return det
        usa_lda = str(getattr(cfg, "classificador", "plsda")).lower() == "plslda"
        ordem = " < ".join(det["ordem"])
        meio, ponta = det["sens_meio"], det["sens_pontas"]
        comp_pt = comp_en = ""
        if not usa_lda:
            try:
                def _fab_lda(n_lv: int):
                    return Pipeline([("preproc", build_preprocessor(cfg)),
                                     ("pls", PLSLDAClassifier(n_components=n_lv, scale=False))])
                _n_lda, _pred_lda = _selecionar_n_opt_wold(_fab_lda, X_raw, Y_bin, cv_indices, lv_cap)
                ba_lda = float(classification_metrics(
                    y_int, np.argmax(_pred_lda, axis=1),
                    np.arange(Y_bin.shape[1]))["balanced_accuracy"])
                ba_da = float(metricas_por_lv[n_opt - 1]["balanced_accuracy"])
                comp_pt = (f" Na mesma validacao cruzada (sem aninhar nas duas), o PLS-LDA da acuracia "
                           f"balanceada {ba_lda:.2f} contra {ba_da:.2f} do PLS-DA.")
                comp_en = (f" In the same cross-validation (neither nested), PLS-LDA gives balanced accuracy "
                           f"{ba_lda:.2f} against {ba_da:.2f} for PLS-DA.")
                det["ba_plslda_naive"], det["ba_plsda_naive"] = ba_lda, ba_da
            except Exception as _e_cmp:  # noqa: BLE001 -- comparacao e' um extra do aviso
                log.info(f"  [INFO] Comparacao com PLS-LDA nao calculada: {_e_cmp}")
        sug_pt = ("" if usa_lda else " Contorno: classificador: plslda (LDA sobre os escores do PLS).")
        sug_en = ("" if usa_lda else " Workaround: classificador: plslda (LDA on the PLS scores).")
        pt = (f"Classes em rampa (ordenadas): as medias das {len(det['ordem'])} classes ficam alinhadas em uma "
              f"direcao ({ordem}; {det['frac_colinear'] * 100:.0f}% da variancia entre as medias) e a "
              f"sensibilidade da classe do meio '{det['classe_mascarada']}' ({meio:.2f}) e' "
              f"{ponta - meio:.2f} menor que a da pior ponta ({ponta:.2f}). E' o mascaramento do PLS-DA "
              f"quando as classes sao niveis de uma mesma propriedade (teor, cor, grau): a acuracia "
              f"balanceada subestima a separacao que existe nos dados.{comp_pt}{sug_pt} Se o que importa e' "
              f"o nivel numerico, a regressao PLS no valor e' mais adequada que classificar.")
        en = (f"Ramp-like (ordered) classes: the means of the {len(det['ordem'])} classes lie along one "
              f"direction ({ordem}; {det['frac_colinear'] * 100:.0f}% of the variance between means) and the "
              f"sensitivity of the middle class '{det['classe_mascarada']}' ({meio:.2f}) is "
              f"{ponta - meio:.2f} lower than that of the worst end class ({ponta:.2f}). This is PLS-DA "
              f"masking when the classes are levels of one property (content, colour, grade): the balanced "
              f"accuracy underestimates the separation present in the data.{comp_en}{sug_en} If the "
              f"numeric level is what matters, PLS regression on the value is more appropriate than "
              f"classifying.")
        log.info("\n  [AVISO] " + pt)
        anexar_bloco_modelo(pasta_logs, ("Avisos do modelo: classificacao", "Model warnings: classification"),
                            [("classes_em_rampa", pt, en)], aviso=True)
        return det
    except Exception as _e_rampa:  # noqa: BLE001 -- aviso e' um extra; nunca derruba a classificacao
        log.info(f"  [AVISO] Deteccao de classes em rampa nao calculada: {_e_rampa}")
        return None


def _registrar_alinhamento(preproc_full: Pipeline, mae_id: Optional[np.ndarray], pasta_logs: Optional[str],
                           anexar: bool) -> None:
    """Registra no log (e, com `anexar`, no resumo e no model card) o alinhamento COW do modelo final:
    a referencia (escolhida so' com o treino), parametros e a correlacao media par a par antes/depois."""
    passo = getattr(preproc_full, "named_steps", {}).get("cow")
    if passo is None:
        return
    ref = int(passo.indice_referencia_)
    quem = f" (grupo {mae_id[ref]})" if mae_id is not None and ref < len(mae_id) else ""
    c0, c1 = passo.correlacao_antes_, passo.correlacao_depois_
    pt = (f"Alinhamento de retencao COW (Nielsen et al. 1998): {passo.n_segmentos} segmentos, folga de "
          f"{passo.folga} pontos; referencia = cromatograma #{ref} do treino{quem}, escolhida so' com o treino "
          f"(em cada dobra da validacao a referencia e' escolhida so' com o treino da dobra e a mesma alinha o "
          f"teste; o modelo exportado leva a referencia para alinhar amostras novas). Correlacao media par a "
          f"par no treino: {c0:.3f} antes, {c1:.3f} depois.")
    en = (f"COW retention alignment (Nielsen et al. 1998): {passo.n_segmentos} segments, slack of {passo.folga} "
          f"points; reference = training chromatogram #{ref}{quem}, chosen from the training data only (in each "
          f"validation fold the reference is chosen from the fold training data only and aligns the test data "
          f"too; the exported model carries the reference to align new samples). Mean pairwise correlation in "
          f"the training set: {c0:.3f} before, {c1:.3f} after.")
    log.info("  [Alinhamento] " + pt)
    if anexar and pasta_logs:
        from guaraci.resultados_io import anexar_bloco_modelo
        anexar_bloco_modelo(pasta_logs, ("Alinhamento de retencao", "Retention alignment"),
                            [("alinhamento_cow", pt, en)], aviso=False)


#: Abaixo desta acuracia balanceada (CV) da classificacao que ROTEIA a quantificacao, avisa. Criterio:
#: acuracia balanceada = recall medio por classe, entao < 0,80 quer dizer que, em media, mais de
#: 1 amostra em cada 5 de cada classe vai para o modelo de teor errado. O acaso e' 1/K (0,50 com 2
#: classes, 0,33 com 3, menos com mais): um limiar "acaso + margem" so' superaria 0,80 com margem
#: acima de 0,30 em 2 classes, entao 0,80 e' o criterio unico. O aviso diz que a validacao PODE ficar
#: instavel, nao que ficara: numa varredura sintetica (15 execucoes, 3 classes, teor independente da
#: classe, 16 amostras por classe) o R2 de validacao agregado foi de -2,2 a 0,55 em 3 execucoes (todas
#: com acuracia balanceada <= 0,42) e de 0,96 a 1,00 nas outras 12; com o treino pela classe
#: verdadeira (quantificacao_rotulo_treino: verdadeiro) ficou em 0,994 ou mais nas 15 (ver CHANGELOG).
LIMIAR_BA_ROTEAMENTO_QUANT = 0.80


def _quant_treina_com_classe_verdadeira(cfg: "Config", modo_rotulo: str) -> bool:
    """`quantificacao_rotulo_treino: verdadeiro` so' vale no modo cego (no modo controle a classe ja'
    e' a verdadeira)."""
    escolha = str(getattr(cfg, "quant_rotulo_treino", "previsto") or "previsto").lower()
    if escolha not in ("previsto", "verdadeiro"):
        raise ValueError(f"quantificacao_rotulo_treino='{escolha}' invalido. Use 'previsto' (padrao) "
                         f"ou 'verdadeiro'.")
    return escolha == "verdadeiro" and modo_rotulo == "cego"


def _avisar_roteamento_da_quantificacao(*, pasta_logs: str, rotulos: np.ndarray, pred_lab: np.ndarray,
                                        modo_rotulo: str, treino_verdadeiro: bool,
                                        reg_esp: Dict[str, Any]) -> None:
    """Quantificacao com varias classes: o modelo de teor de cada amostra e' escolhido pela classe
    PREVISTA. Com a classificacao fraca isso derruba o R2 do teor (modelos com poucas amostras,
    misturadas e roteamento errado), mesmo que o teor nao dependa da classe. Avisa no console,
    no resumo e no model card; registra no resumo o treino com a classe verdadeira, se ligado."""
    from sklearn.metrics import balanced_accuracy_score
    from guaraci.resultados_io import anexar_bloco_modelo
    try:
        ba = float(balanced_accuracy_score(np.asarray(rotulos, dtype=str), np.asarray(pred_lab, dtype=str)))
        k = int(len(np.unique(np.asarray(rotulos, dtype=str))))
        r2v = reg_esp.get("r2v")
        r2_txt = f"{float(r2v):.2f}" if r2v is not None and np.isfinite(r2v) else "n/a"
        rot = reg_esp.get("roteamento")
        if treino_verdadeiro and rot:
            _acc = rot["acuracia_roteamento"]
            _acc_txt = f"{_acc * 100:.0f}%" if np.isfinite(_acc) else "n/a"
            _r2c = rot["r2v_classe_verdadeira"]
            _r2c_txt = f"{_r2c:.2f}" if np.isfinite(_r2c) else "n/a"
            pt = (f"Teor treinado com a classe VERDADEIRA (quantificacao_rotulo_treino: verdadeiro). "
                  f"R2 de validacao agregado {r2_txt}, roteando cada amostra de validacao pela classe "
                  f"PREVISTA ({_acc_txt} foram para o modelo da sua classe verdadeira, {rot['n_val']} "
                  f"amostras). Com a classe verdadeira em todas (cenario de controle, nao e' o uso real) "
                  f"o R2 seria {_r2c_txt}.")
            en = (f"Content trained with the TRUE class (quantificacao_rotulo_treino: verdadeiro). "
                  f"Aggregate validation R2 {r2_txt}, routing each validation sample by its PREDICTED "
                  f"class ({_acc_txt} went to the model of their true class, {rot['n_val']} samples). "
                  f"With the true class on all of them (control scenario, not real use) R2 would be "
                  f"{_r2c_txt}.")
            log.info("  " + pt)
            anexar_bloco_modelo(pasta_logs, ("Quantificacao: classe usada no treino do teor",
                                             "Quantification: class used to train the content model"),
                                [("quant_treino_verdadeiro", pt, en)], aviso=False)
        if modo_rotulo != "cego" or ba >= LIMIAR_BA_ROTEAMENTO_QUANT:
            return
        if treino_verdadeiro:
            pt = (f"Quantificacao com {k} classes: a acuracia balanceada da classificacao que escolhe o modelo "
                  f"de teor e' {ba:.2f} (abaixo de {LIMIAR_BA_ROTEAMENTO_QUANT:.2f}; acaso {1.0 / k:.2f}). O treino "
                  f"usa a classe verdadeira e o R2 de validacao agregado ({r2_txt}) ja inclui o roteamento pela "
                  f"classe prevista, entao e' o numero do uso real; se o teor depende da classe, as amostras "
                  f"roteadas errado serao quantificadas com o modelo de outra matriz.")
            en = (f"Quantification with {k} classes: the balanced accuracy of the classification that chooses the "
                  f"content model is {ba:.2f} (below {LIMIAR_BA_ROTEAMENTO_QUANT:.2f}; chance {1.0 / k:.2f}). Training "
                  f"uses the true class and the aggregate validation R2 ({r2_txt}) already includes routing by the "
                  f"predicted class, so it is the real-use number; if the content depends on the class, "
                  f"wrongly routed samples will be quantified with the model of another matrix.")
        else:
            pt = (f"Quantificacao com {k} classes: o modelo de teor de cada amostra e' escolhido pela classe "
                  f"PREVISTA, e a acuracia balanceada dessa classificacao e' {ba:.2f} (abaixo de "
                  f"{LIMIAR_BA_ROTEAMENTO_QUANT:.2f}; acaso {1.0 / k:.2f}). Amostras na classe errada vao para o "
                  f"modelo de teor errado, e os modelos de teor recebem amostras misturadas e em numero menor: a "
                  f"validacao por classe e a agregada podem ficar pessimistas ou instaveis (R2 de validacao agregado "
                  f"{r2_txt}). Vale tambem para o uso real, em que a classe de uma amostra nova tambem vem do "
                  f"classificador. Contornos: (1) se o teor nao depende da classe, use uma classe so "
                  f"(coluna_classe constante) para ter um modelo unico; (2) quantificacao_rotulo_treino: "
                  f"verdadeiro treina cada modelo com a classe verdadeira e valida roteando pela classe prevista.")
            en = (f"Quantification with {k} classes: the content model of each sample is chosen by the PREDICTED "
                  f"class, and the balanced accuracy of that classification is {ba:.2f} (below "
                  f"{LIMIAR_BA_ROTEAMENTO_QUANT:.2f}; chance {1.0 / k:.2f}). Samples in the wrong class go to the "
                  f"wrong content model, and the content models receive mixed and fewer samples: the per-class "
                  f"and aggregate validation can become pessimistic or unstable (aggregate validation R2 {r2_txt}). "
                  f"It also applies to real use, where the class of a new sample also comes from the classifier. "
                  f"Workarounds: (1) if the content does not depend on the class, use a single class (constant "
                  f"coluna_classe) to get one pooled model; (2) quantificacao_rotulo_treino: verdadeiro trains "
                  f"each model with the true class and validates by routing with the predicted class.")
        log.info("  [AVISO] " + pt)
        anexar_bloco_modelo(pasta_logs, ("Avisos do modelo: quantificacao", "Model warnings: quantification"),
                            [("roteamento_fraco", pt, en)], aviso=True)
    except Exception as _e_av:  # noqa: BLE001 -- aviso e' um extra; nunca derruba a quantificacao
        log.info(f"  [AVISO] Aviso de roteamento da quantificacao nao gerado: {_e_av}")


def _etapa_regressao_pls(*, X_holdout, X_raw, _G, _conc_holdout, cam_modelo, cfg, classes_unicas, conc, joblib, lb, mae_id, n_splits, objetivo, pacote_modelo, pasta, pasta_logs, perfil, pls_final, pred_lab, preproc_full, rotulos, rotulos_holdout, wavenumbers=None):
    """Etapa [7/7]: regressao PLS (pooled e por especie), holdout de quantificacao, figuras de merito e addendum do model card.

    Extraido de `_executar_impl` (P6, 2026-09-26) sem mudanca de logica:
    entradas e saidas explicitas; paridade bit a bit conferida."""
    _pls_reg_ok = False
    if conc is not None:
        conc_arr = np.asarray(conc, dtype=float)
        n_nan    = int(np.isnan(conc_arr).sum())
        n_nonzero = int(np.sum(conc_arr > 0))

        if objetivo != QUANTIFICACAO:
            log.info(f"\n[7/7] PLS regressao — PULADA: objective="
                  f"{OBJETIVO_ROTULO.get(objetivo, objetivo)} "
                  f"(level={cfg.level}). A regressao da resposta pertence "
                  f"ao Modo Quantificacao (N3); nos modos Exploratorio/"
                  f"Classificacao ela seria um resultado fora de escopo (e "
                  f"em N1 as {len(classes_unicas)} classes dominam o sinal "
                  f"da resposta, produzindo R2~0).")
        elif n_nan > 0:
            log.info(f"\n[7/7] PLS regressao — PULADA: {n_nan} amostras com "
                  f"conc=NaN.")
        elif float(conc_arr.std()) < 1e-8:
            log.info("\n[7/7] PLS regressao — PULADA: variancia de conc ~= 0.")
        elif n_nonzero < 10:
            log.info(f"\n[7/7] PLS regressao — PULADA: apenas {n_nonzero} "
                  f"amostras com resposta > 0 (precisa >= 10).")
        else:
            # Guard (6): numero de especies = numero de CLASSES (rotulos). Antes
            # vinha do prefixo de 3 caracteres do mae_id (uma convencao de
            # nomenclatura especifica), que com um mae_id de outra
            # convencao (ex.: "grupo_1", "grupo_2" -> um unico prefixo "GRU")
            # contava 1 especie e mandava o pooled unico sobre matrizes
            # diferentes.
            n_especies = int(len(np.unique(rotulos))) if rotulos is not None else 1

            if n_especies > 1:
                # Multi-species: a SINGLE pooled model is confounded by inter-
                # species variation. Instead, calibrate PER SPECIES and pool
                # the predictions — recovers the proper diagonal.
                log.info(f"\n[7/7] PLS regressao POR ESPECIE "
                      f"({n_especies} especies — calibracao separada para "
                      f"evitar confusao inter-especies)")
                # Rotulos da calibracao por classe: PREDITOS no mode cego
                # (padrao), verdadeiros so' em mode controle. Ver
                # labels_for_quantification -- e' a diferenca entre medir o
                # que o usuario vai obter e medir um cenario que ele nunca
                # tera'.
                _rot_quant, _modo_quant = labels_for_quantification(
                    cfg, rotulos, pred_lab)
                log.info(f"  Modo de rotulo na quantificacao: {_modo_quant}"
                         + ("  [os numeros abaixo NAO representam uso real]"
                            if _modo_quant != "cego" else ""))
                _treino_verd = _quant_treina_com_classe_verdadeira(cfg, _modo_quant)
                if _treino_verd:
                    log.info("  Treino do teor por classe: classe VERDADEIRA "
                             "(quantificacao_rotulo_treino=verdadeiro); a validacao roteia "
                             "cada amostra pela classe PREVISTA.")
                try:
                    reg_esp = pls_regression_by_species(
                        X_raw, conc_arr, _rot_quant, _G(mae_id),
                        np.unique(rotulos if _treino_verd else _rot_quant), cfg, pasta, n_splits,
                        rotulos_treino=(rotulos if _treino_verd else None))
                    if reg_esp is not None:
                        # M-05b: RMSEP do HOLDOUT externo (o "RMSEP" interno
                        # abaixo vem de um split 70/30 dos dados de treino).
                        reg_esp["holdout"] = None
                        if (X_holdout is not None and _conc_holdout is not None
                                and reg_esp.get("pipelines_especie")):
                            try:
                                if _modo_quant == "cego":
                                    _rot_ho_q = lb.classes_[np.argmax(
                                        pls_final.predict(
                                            preproc_full.transform(X_holdout)),
                                        axis=1)]
                                else:
                                    _rot_ho_q = rotulos_holdout
                                reg_esp["holdout"] = avaliar_quantificacao_holdout(
                                    reg_esp["pipelines_especie"], X_holdout,
                                    _rot_ho_q, _conc_holdout)
                                if reg_esp["holdout"] is not None:
                                    _h = reg_esp["holdout"]
                                    log.info(
                                        f"  RMSEP (HOLDOUT externo, n={_h['n']}): "
                                        f"{_h['rmsep']:.3f}  R2={_h['r2']:.3f}  "
                                        f"bias={_h['bias']:.3f}  [numero publicavel; "
                                        f"o RMSEP interno abaixo vem de split 70/30]")
                            except Exception as _e_hq:  # noqa: BLE001 -- extra
                                # de avaliacao; falha aqui nunca derruba a
                                # quantificacao ja calculada.
                                log.info(f"  [AVISO] RMSEP do holdout nao calculado: {_e_hq}")
                        log.info(f"  Especies modeladas: {reg_esp['n_especies']}")
                        log.info(f"  R2cal (pooled): {reg_esp['r2c']:.4f}  |  "
                              f"R2val (pooled): {reg_esp['r2v']:.4f}")
                        log.info(f"  RMSEP (pooled): {reg_esp['rmsep']:.3f}")
                        log.info(f"  DModY critico (SIMCA): "
                              f"{reg_esp['dmody_crit']:.3f}  |  "
                              f"amostras fora: {reg_esp['n_fora_do_dmody']}")
                        # Bloco 9b: persiste os pipelines de regressao POR
                        # ESPECIE (ja validados acima) no MESMO pacote de
                        # modelo exportado em 9b -- sem isso, Quantificar em
                        # amostra nova nao teria modelo nenhum para aplicar
                        # (esta funcao ate' aqui so' calculava metricas de
                        # CV/relatorio, nunca persistia o modelo ajustado).
                        # `pacote_modelo`/`cam_modelo` chegam como parametros.
                        if reg_esp.get("pipelines_especie"):
                            try:
                                pacote_modelo["regressao_por_especie"] = (
                                    reg_esp["pipelines_especie"])
                                joblib.dump(pacote_modelo, cam_modelo)
                                save_manifest(cam_modelo, pacote_modelo)
                                log.info(
                                    f"  [Modelo] Atualizado com "
                                    f"regressao por especie "
                                    f"({len(reg_esp['pipelines_especie'])} "
                                    f"especies) -> {cam_modelo}")
                            except Exception as _e_upd:  # noqa: BLE001 --
                                # anexo opcional; N1/N2 exportados continuam
                                # validos mesmo se esta atualizacao falhar.
                                log.info(
                                    f"  [AVISO] Nao foi possivel anexar "
                                    f"regressao por especie ao pacote de "
                                    f"modelo: {_e_upd}")
                        for t in reg_esp["tabela_especie"]:
                            log.info(f"    {t['especie']:18s} "
                                  f"LV={t['n_lv']:2d}  RMSEP={t['rmsep']:.2f}  "
                                  f"R2val={t['r2val']:.3f}  "
                                  f"(cal={t['n_cal']}, val={t['n_val']})")
                            _lod_t, _loq_t = t.get("lod"), t.get("loq")
                            if _lod_t is not None and np.isfinite(_lod_t):
                                _lod_lo, _lod_hi = t.get("lod_ic_baixo"), t.get("lod_ic_alto")
                                _loq_lo, _loq_hi = t.get("loq_ic_baixo"), t.get("loq_ic_alto")
                                _conf = t.get("lod_ic_confianca")
                                log.info(f"      LOD={_lod_t:.2f} "
                                      f"[{_lod_lo:.2f}-{_lod_hi:.2f}, "
                                      f"{_conf*100:.0f}% IC]  "
                                      f"LOQ={_loq_t:.2f} "
                                      f"[{_loq_lo:.2f}-{_loq_hi:.2f}, "
                                      f"{_conf*100:.0f}% IC]  "
                                      f"SEN={t['sensibilidade']:.3f}  "
                                      f"gamma={t['sensibilidade_analitica']:.2f}  "
                                      f"SEL={t['seletividade_media']:.3f}")
                                # Bloco 12: LOD/LOQ nunca reportado isolado --
                                # faixa e desvio-padrao do CONJUNTO DE
                                # VALIDACAO ao lado, para contextualizar se
                                # o LOD/LOQ e' pequeno/grande frente ao que
                                # a amostra real varia.
                                log.info(f"      Faixa de validacao: "
                                      f"[{t['validacao_teor_min']:.2f}, "
                                      f"{t['validacao_teor_max']:.2f}]  "
                                      f"DP={t['validacao_teor_dp']:.2f}")
                            else:
                                log.info("      LOD/LOQ: N/A (sem replicas fisicas "
                                      "suficientes para estimar ruido instrumental)")
                        append_regression_summary(
                            pasta_logs,
                            pooled={k: reg_esp.get(k) for k in
                                    ("r2c", "r2v", "rmsec", "rmsecv", "r2cv",
                                     "rmsep", "bias", "dmody_crit",
                                     "n_fora_do_dmody")},
                            tabela_especie=reg_esp["tabela_especie"],
                            holdout=reg_esp.get("holdout"))
                        append_regression_model_card(
                            pasta_logs,
                            pooled={k: reg_esp.get(k) for k in
                                    ("r2c", "r2v", "rmsec", "rmsecv",
                                     "rmsep", "bias", "dmody_crit",
                                     "n_fora_do_dmody")},
                            tabela_especie=reg_esp["tabela_especie"],
                            holdout=reg_esp.get("holdout"))
                        reg_esp.update(_anexar_linearidade_robustez(
                            pasta_logs, reg_esp["validacao_y"], reg_esp["validacao_y_hat"],
                            reg_esp["validacao_grupos"] if mae_id is not None else None,
                            cfg, X_raw, conc, mae_id, int(reg_esp["n_lv_repr"])))
                        _avisar_roteamento_da_quantificacao(
                            pasta_logs=pasta_logs, rotulos=rotulos, pred_lab=pred_lab,
                            modo_rotulo=_modo_quant, treino_verdadeiro=_treino_verd,
                            reg_esp=reg_esp)

                        # --- Auto-Benchmark de regressao (opcional) ------
                        if cfg.run_benchmark_regression and should_generate(cfg, "benchmark_regressao"):
                            log.info("\n[7b/7] Auto-Benchmark de regressao "
                                  "(Ridge/Lasso/Elastic Net/Dual-sPLS/SVR/RF "
                                  "vs PLS-R)...")
                            if _verificar_ram(0.6, "Auto-Benchmark de regressao"):
                                try:
                                    bench_reg_df = benchmark_regression_by_species(
                                        X_raw, conc_arr, rotulos, _G(mae_id),
                                        classes_unicas, cfg, pasta, reg_esp)
                                    if bench_reg_df is not None:
                                        log.info(bench_reg_df.to_string(index=False))
                                    else:
                                        log.info("  [AVISO] Nenhuma especie com "
                                              "amostras suficientes para o "
                                              "benchmark de regressao.")
                                except Exception as _e_bench_reg:  # noqa: BLE001
                                    # modulo opcional (compara PLS-R com
                                    # outros regressores); erro impresso,
                                    # regressao principal (reg_esp) intacta.
                                    log.info(f"  [AVISO] Benchmark de regressao "
                                          f"falhou: {_e_bench_reg}")
                    else:
                        log.info("  [AVISO] Nenhuma especie com amostras "
                              "suficientes para regressao (>= 6 adulteradas "
                              "e variancia de teor > 0).")
                except Exception as _e_reg_esp:  # noqa: BLE001 -- rede de
                    # seguranca do bloco inteiro de regressao (multi-etapa:
                    # split + selecao de LV + fit + figuras); erro impresso,
                    # classificacao (resultado central, calculada antes) intacta.
                    log.info(f"  [AVISO] Regressao por especie falhou: {_e_reg_esp}")

                # Heatmap R2cv especie x adulterante: granularidade honesta da
                # quantificacao. A regressao pooled por especie junta os
                # adulterantes e esconde que alguns nao sao quantificaveis; o
                # heatmap expoe cada combinacao e MARCA as que falham (so roda
                # em Quantificacao, ja garantido pelo guard objetivo acima).
                try:
                    _rot_hm, _ = labels_for_quantification(
                        cfg, rotulos, pred_lab)
                    _r2cv = r2cv_species_by_adulterant(
                        X_raw, conc_arr, _rot_hm, mae_id, cfg,
                        mapa_adulterante=(perfil.codigos_adulterante or None))
                    if _r2cv is not None:
                        log.info(f"\n[7c/7] R2cv por especie x adulterante — "
                              f"{_r2cv['n_falhas']}/{_r2cv['n_total']} "
                              f"combinacoes abaixo de R2cv="
                              f"{_r2cv['limiar_r2']:.2f}  (n/a: {_r2cv['n_na']})")
                        if should_generate(cfg, "heatmap_especie_adulterante"):
                            fig_heatmap_species_by_adulterant(_r2cv, cfg, pasta)
                        append_heatmap_summary(pasta_logs, _r2cv)
                except Exception as _e_hm:  # noqa: BLE001 -- figura/relatorio
                    # opcional (o calculo R2cv em si ja tem tratamento
                    # granular por combinacao); erro impresso, resto da
                    # corrida intacto.
                    log.info(f"  [AVISO] Heatmap especie x adulterante: {_e_hm}")
                _pls_reg_ok = False   # per-species path handled the figure
            else:
                _pls_reg_ok = True

    if _pls_reg_ok and conc is not None:
        pls_regressao_pooled(X_raw, conc, rotulos, _G(mae_id), cfg, pasta,
                             pasta_logs, n_splits,
                             X_holdout=X_holdout,
                             conc_holdout=_conc_holdout,
                             wavenumbers=wavenumbers)
    elif conc is None:
        log.info("\n[7/7] PLS regressao — pulado (sem coluna de concentracao)")


def _etapa_exportar_modelo(*, Q, T2, X_processed, X_raw, _G, cfg, conc, conformal_classificacao, grupos_cv, lb, mae_id, n_opt, out_q, out_t2, pasta_logs, pasta_modelos, pca, perfil, pls_final, q_lim, resumo, rotulos, t2_lim, usar_grupos, wavenumbers):
    """Etapa 9b: monta o pacote do modelo (dominio de aplicabilidade, DD-SIMCA por especie, ensemble de identificacao), grava .joblib + manifesto assinado + JSON portatil.

    Extraido de `_executar_impl` (P6, 2026-09-26) sem mudanca de logica:
    entradas e saidas explicitas; paridade bit a bit conferida."""
    cam_modelo: Any = None  # saida condicional (preserva o comportamento original)
    joblib: Any = None  # saida condicional (preserva o comportamento original)
    pacote_modelo: Any = None  # saida condicional (preserva o comportamento original)
    try:
        import joblib  # type: ignore[no-redef]
        preproc_export = build_preprocessor(cfg).fit(X_raw)
        pacote_modelo = {
            "parametros_efetivos": resumo.get("parametros_efetivos"),
            "preprocessador": preproc_export,
            "pls_final":      pls_final,
            "label_binarizer": lb,
            "classes":        list(lb.classes_),
            "wavenumbers":    wavenumbers,
            "n_opt":          int(n_opt),
            "preset":         cfg.default_preprocessing,
            "wn_min":         cfg.wn_min, "wn_max": cfg.wn_max,
            # Faixas excluidas e normalizacao por espectro (a predicao reaplica as MESMAS):
            # `wavenumbers` acima ja' e' o eixo SEM as faixas excluidas.
            "faixas_excluir": list((getattr(cfg, "exclusion_info", None) or {}).get("faixas", [])),
            "normalizacao":   str((getattr(cfg, "exclusion_info", None) or {}).get("normalizacao", "nenhuma")),
            "ims_rip_janela": (getattr(cfg, "exclusion_info", None) or {}).get("janela_rip"),
            # v25: limites para diagnosticos em novos dados (Aba Predicao)
            "t2_ucl":         float(t2_lim),
            "q_ucl":          float(q_lim),
            "grouping_guarantee": str(cfg.grouping_guarantee),
            # Q de treino do espaco PLS fora da amostra e com X centrado pela
            # media do PLS (A4-03). `predict_samples` so' centra X quando esta
            # marca existe: pacotes antigos calibraram Q sem centrar, e centrar
            # so' a predicao os tornaria incoerentes.
            "pls_q_centrado": True,
            # Passo 57 (achado na revisao com o usuario): sem esta marca,
            # uma execucao com dado sintetico (metricas quase sempre
            # perfeitas -- a banda marcadora de `generate_synthetic_data`
            # e' limpa por construcao) vira template de referencia sem
            # aviso, indistinguivel de dado real no manifesto/model card.
            "dados_sinteticos": bool(cfg.mode == "sintetico"),
        }
        # Perfil multielementar: o modelo exportado precisa dizer em que
        # unidade, com que elementos (na ordem das colunas), com que politica
        # de censura e transformacao foi treinado -- sem isso, uma amostra
        # nova seria pre-processada diferente sem ninguem saber.
        if getattr(cfg, "multielementar_resumo", None):
            _mr = cfg.multielementar_resumo
            pacote_modelo["multielementar"] = {
                k: _mr[k] for k in ("selo", "unidade", "politica_censura", "limiar_censura",
                                    "divisor_lod", "transformacao", "elementos_modelados",
                                    "elementos_excluidos")}
        if cfg.mode == "imagem":   # R12: recorte/cartao/segmentacao para reaplicar IGUAL em fotos novas
            from guaraci.dados_imagem import parametros_imagem_de_config
            pacote_modelo["imagem"] = parametros_imagem_de_config(cfg)
        # Conjunto de predicao conforme p/ classificacao (Grupo 2): so'
        # persistido quando o holdout calibrou (pode ser None se o holdout
        # falhou ou nao foi gerado -- predicao.predict_samples cai no
        # comportamento antigo, argmax puro, sem esta chave).
        if conformal_classificacao is not None:
            pacote_modelo["conformal_classificacao"] = conformal_classificacao
        # Parametros da distancia combinada NO ESPACO PLS. Sem eles,
        # predict_samples() decidia "aceito" pela regra RETANGULAR
        # (T2<=lim E Q<=lim, alpha independente por eixo) -- a mesma ja
        # corrigida no DD-SIMCA (2026-08-08) e no dominio de aplicabilidade
        # (achado A3), com alpha conjunto efetivo ~0,0975 em vez de 0,05.
        # As colunas AD_* ja usavam a regra certa; a coluna `aceito`, nao.
        try:
            _h0, _Nh = mean_and_dof_moments(T2)
            _q0, _Nq = mean_and_dof_moments(Q)
            pacote_modelo["pls_h0"] = float(_h0)
            pacote_modelo["pls_q0"] = float(_q0)
            pacote_modelo["pls_Nh"] = float(_Nh)
            pacote_modelo["pls_Nq"] = float(_Nq)
            from scipy.stats import chi2 as _chi2
            pacote_modelo["pls_f_crit"] = float(
                _chi2.ppf(0.95, _Nh + _Nq))
        except Exception as _e_comb:  # noqa: BLE001 -- anexo opcional do
            # pacote; sem ele predicao.py cai na regra por eixo e avisa.
            log.info(f"  [AVISO] parametros da distancia combinada no "
                     f"espaco PLS nao exportados: {_e_comb}")
        # Dominio de Aplicabilidade (Jaworska et al. 2005): reaproveita o PCA
        # exploratorio ja ajustado (fig1_pca_scores) para avisar, na predicao
        # em amostras novas, quando o espectro cai fora do espaco coberto pela
        # calibracao. So' salva var_t + parametros da distancia combinada
        # (leve, ~poucos floats) em vez de X_processed inteiro (que pode
        # pesar dezenas de MB em dados reais). h0/q0/Nh/Nq/f_crit (em vez de
        # t2_limite/q_limite) desde a correcao do achado A3 (auditoria
        # 2026-08-07): a decisao dentro/fora usa a distancia combinada do
        # DD-SIMCA, nao mais o teste retangular por eixo.
        log.info("\n[7d/7] Dominio de aplicabilidade: calibrando...")
        try:
            _ad_treino = cast(Dict[str, Any], training_applicability_domain(
                pca, X_processed, alpha=0.05,
                groups=grupos_cv if usar_grupos else None))
            pacote_modelo["pca"] = pca
            pacote_modelo["ad_var_t"] = _ad_treino["var_t"]
            pacote_modelo["ad_h0"] = _ad_treino["h0"]
            pacote_modelo["ad_q0"] = _ad_treino["q0"]
            pacote_modelo["ad_Nh"] = _ad_treino["Nh"]
            pacote_modelo["ad_Nq"] = _ad_treino["Nq"]
            pacote_modelo["ad_f_crit"] = _ad_treino["f_crit"]
            # A4-02: dominio calibrado com poucos grupos por classe rejeita
            # muitas amostras legitimas (9% a 98% medido); avisar o n.
            _g_ad = _G(mae_id) if (usar_grupos and mae_id is not None) else None
            _aviso_ad = ad_small_n_warning(
                {str(c): (int(len(np.unique(_g_ad[rotulos == c])))
                          if _g_ad is not None else int(np.sum(rotulos == c)))
                 for c in np.unique(rotulos)},
                por_amostra=_g_ad is None)
            if _aviso_ad:
                pacote_modelo["ad_aviso_n_pequeno"] = _aviso_ad
                resumo["AD aviso n pequeno"] = _aviso_ad
                log.info(f"  [AVISO] {_aviso_ad}")
            # Referencia do sentinela de deriva (Passo 220, correcao R2):
            # quanto ESTA calibracao rejeita em dados que nao viu (CV
            # group-aware por mae_id). O sentinela testa a taxa em producao
            # contra isto, nao contra o alpha nominal -- ver
            # sentinela_deriva.py. Falha aqui nao derruba o AD: o pacote so'
            # perde a referencia e o sentinela cai no teste legado (avisado).
            try:
                _t0_ad = time.time()

                def _progresso_ad(k: int, total: int) -> None:
                    # ~20 linhas no maximo, mesmo com 300 folds
                    if k == total or k % max(1, total // 20) == 0:
                        log.info(f"  [7d/7] Dominio de aplicabilidade: fold "
                                 f"{k}/{total} concluido "
                                 f"({time.time() - _t0_ad:.0f}s)")

                _ref_r, _ref_n = ad_rejection_rate_cv(
                    X_processed, int(pca.n_components_), alpha=0.05,
                    groups=_G(mae_id), progresso=_progresso_ad)
                if _ref_n > 0:
                    pacote_modelo["ad_cv_rejeitadas"] = int(_ref_r)
                    pacote_modelo["ad_cv_n"] = int(_ref_n)
                    resumo["AD taxa de rejeicao em CV (referencia do sentinela)"] = (
                        f"{_ref_r}/{_ref_n} = {_ref_r / _ref_n:.3f}")
                    resumo["AD taxa esperada de alarmes em amostras legitimas"] = (
                        f"{_ref_r / _ref_n:.3f} (rejeicao em CV; nominal 0.05)")
                    log.info(
                        f"  [AD] Taxa esperada de alarmes em amostras "
                        f"legitimas (rejeicao do dominio em CV por grupo): "
                        f"{_ref_r}/{_ref_n} = {_ref_r / _ref_n:.3f} "
                        f"(nominal 0.05).")
            except Exception as _e_ref:  # noqa: BLE001 -- anexo opcional
                log.info(f"  [AVISO] referencia de CV do dominio de "
                         f"aplicabilidade nao calculada (sentinela usara o "
                         f"teste legado): {_e_ref}")
        except Exception as _e_ad:  # noqa: BLE001 -- anexo opcional do
            # pacote de modelo; erro impresso, modelo principal (pls_final)
            # exportado normalmente logo abaixo mesmo sem o AD.
            log.info(f"  [AVISO] Dominio de aplicabilidade nao pode ser "
                  f"exportado: {_e_ad}")
        # Ensemble de Identificacao especie x adulterante (Bloco 9b): reusa
        # o MESMO PCA/var_t do dominio de aplicabilidade acima -- ver
        # docstring de identificacao.py sobre por que nao ajusta um espaco
        # novo por combinacao (a maioria tem poucos espectros de 1-2
        # sessoes). Persistido mesmo quando toda combinacao sai
        # nao-validada: e' o que permite ao Quantificar bloquear com motivo
        # explicito, em vez de rodar sem checagem nenhuma.
        try:
            _ensemble_id = train_identification_ensemble(
                pca, _ad_treino["var_t"], X_processed, rotulos, conc, mae_id,
                mapa_adulterante=(perfil.codigos_adulterante or None))
            pacote_modelo["identification_ensemble"] = _ensemble_id
            resumo["Identificacao n_combinacoes"] = len(_ensemble_id)
            if _ensemble_id:
                _n_validado = sum(
                    1 for v in _ensemble_id.values()
                    if v["cobertura_status"] == CoverageStatus.VALIDATED)
                resumo["Identificacao n_validado"] = _n_validado
                log.info(
                    f"  [Identificacao] {len(_ensemble_id)} "
                    f"combinacoes especie x adulterante calibradas, "
                    f"{_n_validado} com cobertura VALIDADA. Ver model card "
                    f"para o detalhe por combinacao.")
                if _n_validado == 0:
                    log.warning(
                        "  [Identificacao] NENHUMA combinacao especie x adulterante "
                        "tem cobertura estatistica validada neste dataset -- "
                        "Identificar/Quantificar em amostra nova ficam "
                        "bloqueados por padrao (ver cobertura_status).")
            # D5: a ressalva de (nao-)validacao vai para o model card TAMBEM
            # (nao so' o log acima) -- mesmo padrao ja usado para
            # grouping_guarantee (Bloco 8) e para a regressao por especie
            # (append_regression_model_card).
            append_identification_model_card(pasta_logs, _ensemble_id)
        except Exception as _e_ident:  # noqa: BLE001 -- anexo opcional;
            # sem ele, o mode cego (Identificar/Quantificar) simplesmente
            # nao fica disponivel na predicao, sem afetar N1/N2 exportados.
            log.info(f"  [AVISO] Ensemble de identificacao nao pode ser "
                  f"exportado: {_e_ident}")
        # DD-SIMCA POR ESPECIE (purity gate, Bloco 9b -- fechando o gap do
        # Detectar): o dominio de aplicabilidade acima e' ajustado em TODA
        # a amostragem (pura + adulterada) -- responde "isto e' parecido
        # com algo que vimos", NAO "isto e' puro para a especie que o
        # classificador disse". Uma amostra adulterada passa tranquilamente
        # pelo AD (ela FAZ parte do treino do AD). O DD-SIMCA por especie,
        # ajustado SO' nos puros, e' quem responde a pergunta de pureza --
        # ate' aqui, so' existia para o relatorio de UMA rodada N2, nunca
        # persistido para aplicar depois a amostra nova. Calculado aqui
        # SEMPRE (independente de cfg.level/cfg.run_ddsimca), mesmo
        # tratamento incondicional ja dado ao AD.
        try:
            _mask_puro = np.isnan(conc) | (conc == 0.0) if conc is not None \
                else np.ones(len(rotulos), dtype=bool)
            _ddsimca_export = DDSimca(
                n_components=cfg.ddsimca_n_components, alpha=0.05,
                ucl_method=cfg.ddsimca_ucl_method)
            _ddsimca_export.fit(
                X_processed[_mask_puro], rotulos[_mask_puro],
                mae_id=_G(mae_id)[_mask_puro] if mae_id is not None else None)
            _ddsimca_por_especie: Dict[str, Dict[str, Any]] = {}
            for _esp, _m in _ddsimca_export._modelos.items():
                _ddsimca_por_especie[str(_esp)] = {
                    "pca": _m["pca"], "var_t": _m["var_t"],
                    "h0": _m["h0"], "q0": _m["q0"],
                    "Nh": _m["Nh"], "Nq": _m["Nq"], "f_crit": _m["f_crit"],
                    "n_grupos_calibracao": _m["n_grupos_calibracao"],
                    "calibrado_por_amostra": _m["calibrado_por_amostra"],
                }
                # A4-01: sem n >= n_minimum_for_alpha, o alpha so' e' declarado
                # com sensibilidade LOGO medida (ddsimca_alpha_sustentado).
                # Com < PUREZA_MIN_GRUPOS_LOGO grupos o LOGO nao e' evidencia.
                _ng_esp = int(_m["n_grupos_calibracao"])
                if (PUREZA_MIN_GRUPOS_LOGO <= _ng_esp < n_minimum_for_alpha(0.05)
                        and mae_id is not None):
                    try:
                        _idx_esp = _mask_puro & (rotulos == _esp)
                        _logo_esp = ddsimca_logo_sensitivity(
                            X_processed[_idx_esp], _G(mae_id)[_idx_esp],
                            n_components=cfg.ddsimca_n_components,
                            alpha=0.05, ucl_method=cfg.ddsimca_ucl_method,
                            X_brutos=X_raw[_idx_esp],
                            fabrica_preproc=lambda: build_preprocessor(cfg),
                            classe=str(_esp))
                        _s_logo = float(_logo_esp["sensibilidade"])
                        if np.isfinite(_s_logo):
                            _ddsimca_por_especie[str(_esp)]["sens_logo"] = _s_logo
                    except Exception as _e_lg:  # noqa: BLE001 -- anexo opcional
                        log.info(f"  [AVISO] LOGO da pureza de '{_esp}' nao "
                                 f"calculado (alpha ficara sem lastro): {_e_lg}")
            pacote_modelo["ddsimca_por_especie"] = _ddsimca_por_especie
            resumo["Pureza n_especies"] = len(_ddsimca_por_especie)
            _n_confiavel_pureza = sum(
                1 for v in _ddsimca_por_especie.values()
                if ddsimca_alpha_sustentado(
                    v["n_grupos_calibracao"], v.get("sens_logo")))
            resumo["Pureza n_confiavel"] = _n_confiavel_pureza
            log.info(
                f"  [Pureza] DD-SIMCA por especie (Detectar/pureza): "
                f"{len(_ddsimca_por_especie)} especies calibradas, "
                f"{_n_confiavel_pureza} com alpha sustentado por evidencia "
                f"(LOGO medido ou >= {n_minimum_for_alpha(0.05)} grupos); nas "
                f"demais a taxa de falsa rejeicao de puros novos e' "
                f"desconhecida e pode ser alta.")
            # D5: ressalva tambem no model card (log acima + manifesto em
            # predicao.generate_manifest ja' cobrem os outros 2 lugares).
            append_purity_model_card(pasta_logs, _ddsimca_por_especie)
        except Exception as _e_dds:  # noqa: BLE001 -- anexo opcional; sem
            # ele, Detectar cai so' no AD (comportamento anterior).
            log.info(f"  [AVISO] DD-SIMCA por especie nao pode ser "
                  f"exportado para predicao: {_e_dds}")
        cam_modelo = os.path.join(pasta_modelos, "modelo_plsda.joblib")
        joblib.dump(pacote_modelo, cam_modelo)
        log.info(f"  -> {cam_modelo}")
        # Manifesto de proveniencia/integridade: sha256 do
        # arquivo + versoes de biblioteca, usado por predicao.load_model
        # para detectar arquivo trocado/corrompido ANTES de executar o pickle.
        cam_manifesto = save_manifest(cam_modelo, pacote_modelo)
        log.info(f"  -> {cam_manifesto}")
        # Exportacao portatil (Grupo 2, JSON puro sem pickle/joblib --
        # ver model_export.py): sidecar OPCIONAL ao lado do .joblib,
        # so' quando o preset/config do modelo e' suportado (silenciosa
        # p/ os que nao sao ainda -- ex.: preset 'custom'/'airpls_sg_mc'
        # -- comportamento IDENTICO ao anterior, so' sem o arquivo extra).
        try:
            from guaraci.model_export import save_portable_json
            cam_portatil = cam_modelo.replace(".joblib", ".portatil.json")
            save_portable_json(pacote_modelo, cam_portatil)
            log.info(f"  -> {cam_portatil} (formato portatil, sem pickle)")
        except Exception as _e_port:  # noqa: BLE001 -- conveniencia
            # opcional; preset ainda nao suportado ou qualquer outra
            # falha nao afeta o .joblib (resultado principal) ja salvo.
            log.info(f"  [INFO] Exportacao portatil (JSON) nao gerada: "
                      f"{_e_port}")
    except Exception as _e_mod:  # noqa: BLE001 -- exportacao opcional
        # (predicao em amostra nova); erro impresso, nao afeta as figuras/
        # relatorios ja gerados desta corrida.
        log.info(f"  [AVISO] Exportacao do modelo pulada: {_e_mod}")
        registrar_falha_salvamento(f"modelo/manifesto ({cam_modelo}): {_e_mod}")

    if out_t2.size or out_q.size:
        log.info(f"\n[INFO] Outliers (T2 > lim): {out_t2.tolist()}")
        log.info(f"[INFO] Outliers (Q  > lim): {out_q.tolist()}")
    return (cam_modelo, joblib, pacote_modelo,)


def _etapa_benchmark_monte_carlo(*, X_holdout, X_raw, cfg, grupos_cv, lb, n_opt, pasta, rotulos_holdout, wavenumbers, y_int):
    """Etapas 9a/9a2 (opcionais): comparacao automatica de classificadores e CV Monte Carlo.

    Extraido de `_executar_impl` (P6, 2026-09-26) sem mudanca de logica:
    entradas e saidas explicitas; paridade bit a bit conferida."""
    if cfg.run_benchmark and should_generate(cfg, "benchmark"):
        log.info("\n[7b/7] Auto-Benchmark (SVM / RF / XGBoost vs PLS-DA)...")
        # Guarda: ~1.2 GB pico (SVM kernel matrix + OOF proba)
        if _verificar_ram(1.2, "Auto-Benchmark"):
            try:
                # Explicabilidade fora do treino (Grupo 2): quando o
                # holdout existe, SHAP explica ELE (nunca visto no ajuste
                # das arvores) em vez de uma subamostra do proprio treino
                # -- ver docstring de avaliacao_modelos.fig_shap_benchmark.
                _y_holdout_int = (
                    np.argmax(lb.transform(rotulos_holdout), axis=1)
                    if X_holdout is not None and rotulos_holdout is not None
                    else None)
                bench_df = benchmark_classifiers(
                    X_raw, y_int, grupos_cv, lb, n_opt, cfg, pasta,
                    wavenumbers=wavenumbers,
                    X_holdout_raw=X_holdout, y_holdout_int=_y_holdout_int)
                log.info(bench_df.to_string(index=False))
            except Exception as _e_bench:  # noqa: BLE001 -- modulo opcional
                # (comparacao com outros classificadores); erro impresso,
                # bench_df so' usado neste bloco; PLS-DA (resultado central)
                # ja calculado.
                log.info(f"  [AVISO] Benchmark falhou: {_e_bench}")

    # --- 9a2. Monte Carlo CV (opcional) ────────────────────────────────────
    if cfg.run_monte_carlo and should_generate(cfg, "monte_carlo"):
        log.info("\n[7c/7] Monte Carlo CV (IC95% por percentil)...")
        # Guarda: ~400 MB (PLS-DA x N splits em serie)
        if _verificar_ram(0.5, "Monte Carlo CV"):
            try:
                mc_df = monte_carlo_cv(
                    X_raw, y_int, grupos_cv, lb, n_opt, cfg, pasta)
                # DataFrame vazio = MC CV pulado por poucos grupos de
                # replica (aviso ja emitido la' dentro, B2-1b) -- nao
                # imprimir "Empty DataFrame" por cima da explicacao.
                if not mc_df.empty:
                    log.info(mc_df.to_string(index=False))
            except Exception as _e_mc:  # noqa: BLE001 -- modulo opcional;
                # erro impresso, mc_df so' usado neste bloco.
                log.info(f"  [AVISO] Monte Carlo CV falhou: {_e_mc}")


def _etapa_resumo_e_relatorios(*, Q, T2, T_pls, X_holdout, X_raw, _G, _achados_auditoria, _classes_fora_holdout, _dmodx_res, _hw, _martens_n_folds, _martens_n_sig, _opls_n_ortho, _unidade_grupo, aucs_roc, bca, bca_holdout, cfg, classes_unicas, conformal_classificacao, cv_anova_res, cv_label, ddsimca_pcv_esp, ddsimca_res, ddsimca_sens_esp, diag_faixa, etapa4_res, lb, lvs_no_teto, mae_id, metricas_finais, metricas_holdout, modo_dd, n_holdout, n_opt, n_splits, objetivo, out_q, out_t2, pasta_dados, pasta_logs, perm_p, perm_res, pred_lab, q2, q_lim, r2x, r2y, relatorio_balanco, relatorio_entrada, rotulos, simca_pred, t2_lim, usar_grupos, wold_res):
    """Etapa 9: identificadores da execucao, dicionario resumo, resumo_modelo.txt, model card, tabelas e metadados.

    Extraido de `_executar_impl` (P6, 2026-09-26) sem mudanca de logica:
    entradas e saidas explicitas; paridade bit a bit conferida."""
    save_identifiers(rotulos, pred_lab, T_pls, T2, Q,
                            t2_lim, q_lim, pasta_dados)
    log.info(f"  -> {os.path.join(pasta_dados, 'amostras_identificadores.csv')}")

    # Formata strings de CI para o resumo
    _n_grupos_bca = (int(len(np.unique(mae_id))) if (usar_grupos and mae_id is not None)
                     else None)

    def _ci_str(b):
        # `bca` e `bca_holdout` trazem sempre as quatro metricas de `metricas_funcoes`: `b` nunca e' None.
        lo, hi, obs = b
        if not np.isfinite(lo):
            return (f"{obs:.4f} [CI nao estimavel: sem variacao entre "
                    f"reamostragens -- grupos insuficientes por classe]")
        # R3-05 (revisao cega): com poucos grupos o IC BCa agrupado SUBCOBRE
        # (cobertura medida 0,73 com 6 grupos, 0,86 com 10, 0,95 com 40).
        if _n_grupos_bca is not None and _n_grupos_bca < 20:
            return (f"{obs:.4f} [{lo:.4f}, {hi:.4f}] (so' {_n_grupos_bca} grupos: "
                    f"este IC tende a ser estreito demais)")
        return f"{obs:.4f} [{lo:.4f}, {hi:.4f}]"

    from guaraci.preprocessamento import descrever_preprocessamento
    _pp_descr = descrever_preprocessamento(cfg)[1]
    _NA_1CLS = "n/a (uma unica classe)"
    resumo = {
        # Modo de entrada no resumo (achado B4-1): e' o que permite aos
        # geradores de relatorio carimbarem "PROTOTIPO -- NAO VALIDADO"
        # quando mode="imagem", em vez de produzirem um PDF/LaTeX
        # tipograficamente identico ao de uma analise FT-NIR validada.
        "Input mode":        str(cfg.mode),
        # Bloco 8 (2026-08-25): nivel de garantia do agrupamento group-aware
        # desta execucao -- "high" (mae_id real) ate' "none" (sem fonte de
        # agrupamento, fallback StratifiedKFold). So' difere de "high" hoje
        # no mode="imagem". Ver dados_imagem.py para os 3 niveis.
        "Grouping guarantee": str(cfg.grouping_guarantee),
        # M-03: TODO parametro que define o numero reportado, como foi
        # EFETIVAMENTE usado (o model card e o manifesto o repetem).
        "parametros_efetivos": {
            "n_splits_cv_configurado": int(cfg.n_splits_cv),
            "n_splits_cv_efetivo": int(n_splits),
            "n_repeats_cv": int(cfg.n_repeats_cv),
            "n_permutations": int(cfg.n_permutations),
            "n_bootstrap_bca": int(cfg.n_bootstrap_bca),
            "max_lvs": int(cfg.max_lvs),
            "seed": int(cfg.seed),
            "seed_holdout": int(cfg.seed_holdout),
            "frac_holdout": float(cfg.frac_holdout),
            "unidade_agrupamento": _unidade_grupo,
            "pre_processamento": str(cfg.default_preprocessing),
            **campos_faixa_espectral(cfg),
            "faixas_excluir": list((getattr(cfg, "exclusion_info", None) or {}).get("faixas", [])),
            "normalizacao": str((getattr(cfg, "exclusion_info", None) or {}).get("normalizacao", "nenhuma")),
            "ajustes_hardware": list(getattr(cfg, "ajustes_hardware", [])),
            "correcoes_locais": dict(getattr(cfg, "correcoes_locais_aplicadas", {})),
            "multielementar": ({k: v for k, v in cfg.multielementar_resumo.items()
                                if k in ("selo", "unidade", "politica_censura", "limiar_censura",
                                         "divisor_lod", "transformacao", "elementos_modelados",
                                         "elementos_excluidos")}
                               if getattr(cfg, "multielementar_resumo", None) else {}),
        },
        # Bloco 11: lista de dicts simples (nao os dataclasses de
        # auditoria_delineamento.py) -- resultados_io.py nao precisa
        # importar aquele modulo so' para renderizar o model card.
        "auditoria_delineamento": [
            {"nome": a.nome, "severidade": a.severidade, "mensagem": a.mensagem}
            for a in _achados_auditoria],
        "Total samples":      int(X_raw.shape[0]),
        "Total variables":     int(X_raw.shape[1]),
        **({"Variaveis (nomes, na ordem do arquivo)": ", ".join(cfg.nomes_variaveis)}
           if getattr(cfg, "nomes_variaveis", None)
           and len(cfg.nomes_variaveis) == X_raw.shape[1] else {}),
        **({"Reamostragem do eixo (n original -> n final)": (
                f"{cfg.reamostragem_aplicada['n_original_min']}"
                + (f"-{cfg.reamostragem_aplicada['n_original_max']}"
                   if cfg.reamostragem_aplicada['n_original_max'] != cfg.reamostragem_aplicada['n_original_min'] else "")
                + f" -> {cfg.reamostragem_aplicada['n_final']} pontos, {cfg.reamostragem_aplicada['metodo']}, "
                  f"passo {cfg.reamostragem_aplicada['largura_bin']:.6g}")}
           if getattr(cfg, "reamostragem_aplicada", None) else {}),
        **({"Canais DAD lidos": (", ".join(cfg.canais_dad_lidos[:12])
                                 + (f" ... (+{len(cfg.canais_dad_lidos) - 12})" if len(cfg.canais_dad_lidos) > 12 else ""))}
           if getattr(cfg, "canais_dad_lidos", None) else {}),
        "Total classes":       int(len(classes_unicas)),
        "Method":                 "PLS-DA",
        "Pre-processamento":      _pp_descr,
        chave_faixa_espectral(cfg): f"[{cfg.wn_min:.0f}, {cfg.wn_max:.0f}]",
        # So' aparecem quando ligadas: o resumo de quem nao usa nao muda.
        **({"Faixas excluidas": (f"{', '.join(cfg.exclusion_info['faixas'])} "
                                 f"({cfg.exclusion_info['n_removidas']} var.)")}
           if (getattr(cfg, "exclusion_info", None) or {}).get("faixas") else {}),
        **({"Normalizacao por espectro": (cfg.exclusion_info["normalizacao"]
                                          + (f" (janela {cfg.exclusion_info['janela_rip']})"
                                             if cfg.exclusion_info.get("janela_rip") else ""))}
           if (getattr(cfg, "exclusion_info", None) or {}).get("normalizacao", "nenhuma") != "nenhuma" else {}),
        **({"Saturacao do detector": (
                f"{cfg.saturacao_info['variaveis_removidas']} variaveis e "
                f"{cfg.saturacao_info['espectros_removidos']} espectro(s) no teto "
                f"({cfg.saturacao_info['teto']:.3g})"
                + ("" if cfg.saturacao_info.get("aplicado") else " -- NAO removidos (saturacao=avisar)"))}
           if getattr(cfg, "saturacao_info", None) else {}),
        "Tecnica (perfil)":       tecnica_do_perfil(cfg) or "-",
        # Diagnostico de faixa: fica no relatorio (nao so' no terminal) para
        # que a decisao de manter/reduzir a faixa seja auditavel depois.
        "Variaveis com sinal (SNR>=3)": (
            f"{diag_faixa['frac_util'] * 100:.0f}%"),
        chave_faixa_espectral(cfg, "Faixa com sinal"): (
            f"[{diag_faixa['faixa_sugerida'][0]:.0f}, "
            f"{diag_faixa['faixa_sugerida'][1]:.0f}]"
            if diag_faixa.get("faixa_sugerida") else "faixa toda util"),
        "Optimal LVs":             int(n_opt),
        "LVs no teto (max_lvs)":  ("SIM - aumente max_lvs" if lvs_no_teto
                                    else "nao"),
        "Validation":             cv_label + (
            f", repeats={cfg.n_repeats_cv}" if cfg.n_repeats_cv > 1 and
            not usar_grupos else ""),
        "Group-aware (mae_id)":   "sim" if usar_grupos else "nao",
        "N grupos mae_id":        (int(len(np.unique(mae_id)))
                                    if mae_id is not None else 0),
        "Unidade de agrupamento": _unidade_grupo,
        "Holdout classes ausentes": (", ".join(_classes_fora_holdout)
                                     if _classes_fora_holdout else "nenhuma"),
        "N unidades independentes": (int(len(np.unique(_G(mae_id))))
                                     if mae_id is not None else 0),
        "Nivel":                  f"{cfg.level} ({_NIVEL_NOME.get(cfg.level, cfg.level)})",
        "Tag":                    cfg.tag if cfg.tag else "-",
        # R3-08: com UMA classe as metricas de classificacao sao degeneradas
        # (acc 1,0 e kappa nan por construcao): nao aparecem como numero.
        "Accuracy (CV)":          (float(metricas_finais["accuracy"]) if len(classes_unicas) > 1 else _NA_1CLS),
        "Balanced accuracy":      (float(metricas_finais["balanced_accuracy"]) if len(classes_unicas) > 1 else _NA_1CLS),
        "F1 (macro)":             (float(metricas_finais["f1_macro"]) if len(classes_unicas) > 1 else _NA_1CLS),
        "Cohen's kappa":          (float(metricas_finais["cohen_kappa"]) if len(classes_unicas) > 1 else _NA_1CLS),
        "R2X":                    float(r2x),
        "R2Y":                    float(r2y),
        "Q2":                     float(q2),
        "Hotelling T2 (95%)":     float(t2_lim),
        # B7: notacao adaptativa — com SG-derivada + muitas LVs, Q_lim e
        # minusculo e ":.4f" exibia 0.0000 (mascarando o valor real).
        "Q-residual (95%)":       (f"{q_lim:.4g}" if abs(q_lim) < 1e-3
                                    else f"{q_lim:.4f}"),
        "N outliers T2":          int(out_t2.size),
        "N outliers Q":           int(out_q.size),
        "DModX critical (SIMCA)":  round(float(_dmodx_res["dmodx_crit"]), 4),
        "N samples outside DModX": int(_dmodx_res["n_fora_do_modelo"]),
        "Imbalance ratio":        cast(float, relatorio_balanco["imbalance_ratio"]),
        "Largest class":           cast(int, relatorio_balanco["n_max"]),
        "Smallest class":           cast(int, relatorio_balanco["n_min"]),
        "NaN integrity":        cast(int, relatorio_entrada["n_nan_amostras"]),
        "Integridade Inf":        cast(int, relatorio_entrada["n_inf_amostras"]),
        "Constant variables":   cast(int, relatorio_entrada["n_constantes_removidas"]),
        "Exact duplicates":      cast(int, relatorio_entrada["n_duplicatas_exatas"]),
        "Approximate duplicates": cast(int, relatorio_entrada["n_duplicatas_aproximadas"]),
    }
    # Permutacao + BCa (v.jul/2026): computados so' em objetivo Classificacao
    # (ver otimizacao de desempenho acima) -- por isso so' aparecem no resumo
    # quando pertinentes, em vez de "nan"/"CI indisponivel" fora de escopo.
    if objetivo == CLASSIFICACAO:
        resumo["Permutation p-value"]      = float(perm_p)
        resumo["Permutation n_validos"]    = cast(int, perm_res["n_validos"])
        resumo["Permutation n_falhos"]     = cast(int, perm_res["n_falhos"])
        resumo["Permutation failure_rate"] = cast(float, perm_res["failure_rate"])
        resumo["BCa Accuracy"]      = _ci_str(bca["accuracy"])
        resumo["BCa Balanced acc."] = _ci_str(bca["balanced_accuracy"])
        resumo["BCa F1 (macro)"]    = _ci_str(bca["f1_macro"])
        resumo["BCa Cohen's kappa"] = _ci_str(bca["cohen_kappa"])
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
        # `bca_holdout` e `conformal_classificacao` nascem no mesmo bloco que `metricas_holdout` (qualquer falha
        # ali zera `metricas_holdout`), entao aqui nenhum dos dois e' None.
        assert bca_holdout is not None and conformal_classificacao is not None
        resumo["BCa Holdout Accuracy"] = _ci_str(bca_holdout["accuracy"])
        resumo["BCa Holdout Bal.acc"]  = _ci_str(bca_holdout["balanced_accuracy"])
        # Nomes de chave curtos DE PROPOSITO: `save_model_summary`
        # calcula a largura da coluna de TODO o resumo a partir da
        # MAIOR chave presente -- uma chave longa aqui alargaria a
        # formatacao do arquivo inteiro (achado real desta rodada,
        # ver test_dados_imagem.py).
        resumo["Conformal classif. alcancavel"] = bool(
            conformal_classificacao["alcancavel"])
        if conformal_classificacao["alcancavel"]:
            resumo["Conformal classif. cobertura"] = (
                f"{1 - conformal_classificacao['alpha_nominal']:.0%}")
        resumo["Conformal classif. n_grupos"] = int(
            conformal_classificacao["n_grupos"])
    # Sprint 3 — append to summary after dict already exists
    if cfg.run_ddsimca and ddsimca_res is not None:
        resumo["DD-SIMCA n_components"]    = int(cfg.ddsimca_n_components)
        resumo["DD-SIMCA n_desconhecidos"] = int(np.sum(simca_pred == "Desconhecido"))
        resumo["DD-SIMCA n_ambiguos"]      = int(np.sum(simca_pred == "Ambiguo"))
        # B4 — honest training mode label. In 'todos' mode, sens/spec
        # are IN-SAMPLE acceptance, NOT one-class authentication metrics.
        resumo["DD-SIMCA mode treino"] = (
            modo_dd + (" (one-class)" if modo_dd == "puros"
                       else " (in-sample; sens/esp NAO sao autenticacao)"))
        # C4 — sensibilidade (LOGO honesto por mae_id) / especificidade one-class.
        # A sensibilidade NUNCA aparece sem o numero de grupos LOGO ao lado:
        # com poucos grupos ela e exploratoria e o AVISO explicita a incerteza.
        for cls in classes_unicas:
            if cls in ddsimca_sens_esp:
                s_c, e_c, npc, nac, ng, av = ddsimca_sens_esp[cls]
                sens_s = (f"{s_c*100:.1f}%" if s_c == s_c
                          else "n/a (nao validado)")
                esp_s = f"{e_c*100:.1f}%" if e_c == e_c else "n/a"
                resumo[f"DD-SIMCA {cls} sens(LOGO)/esp"] = (
                    f"{sens_s} / {esp_s} "
                    f"(grupos_LOGO={ng}, puros={npc}, adult={nac})")
                if av:
                    resumo[f"DD-SIMCA {cls} AVISO"] = av
                # PCV: diagnostico complementar opt-in -- SEMPRE ao lado do
                # LOGO, nunca em vez dele (ver ddsimca_pcv_sensitivity).
                if cls in ddsimca_pcv_esp:
                    s_pcv, av_pcv = ddsimca_pcv_esp[cls]
                    sens_pcv_s = f"{s_pcv*100:.1f}%" if s_pcv == s_pcv else "n/a"
                    resumo[f"DD-SIMCA {cls} sens(PCV, exploratorio)"] = sens_pcv_s
                    if av_pcv:
                        resumo[f"DD-SIMCA {cls} AVISO PCV"] = av_pcv
                # Diagnostico robusto (mediana/MAD): replicas de treino
                # atipicas, so' sinalizadas -- nunca removidas sozinhas (ver
                # DDSimca._outliers_robustos_mad).
                # `ddsimca_sens_esp` so' tem classes presentes em `ddsimca_res`.
                _out = ddsimca_res[cls].get("outliers_treino") or []
                if _out:
                    resumo[f"DD-SIMCA {cls} AVISO treino"] = (
                        f"{len(_out)} replica(s) de treino atipica(s) "
                        f"(indices {list(_out)}, deteccao robusta "
                        "mediana/MAD) -- nao removidas automaticamente, "
                        "considere investigar.")
    if _opls_n_ortho is not None:
        resumo["OPLS-DA n_ortho"] = int(_opls_n_ortho)
    if _martens_n_sig is not None:
        resumo["Martens n_significativas"] = _martens_n_sig
        resumo["Martens n_folds_validos"] = cast(int, _martens_n_folds)

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
            log.info(f"[ATENCAO] Classe '{cls}': {frac_c*100:.0f}% outliers T2 "
                  f"({n_out_c}/{n_tot_c}) — possivel batch/lote anomalo.")

    # M4 — Accuracy (recall) por classe
    resumo["--- Accuracy por classe ---"] = ""
    rec_por_classe = recall_score(rotulos, pred_lab, labels=lb.classes_,
                                   average=None, zero_division=0)
    for cls, rec_c in zip(lb.classes_, np.asarray(rec_por_classe)):
        resumo[f"  Acc {cls}"] = float(rec_c)

    save_model_summary(pasta_logs, resumo)
    log.info(f"  -> {os.path.join(pasta_logs, 'resumo_modelo.txt')}")

    # Model Card (Mitchell et al. 2019) -- mesmo ponto/dados do resumo acima.
    generate_model_card(pasta_logs, cfg, resumo, _hw, classes_unicas)
    log.info(f"  -> {os.path.join(pasta_logs, 'model_card.md')}")

    # Auditoria de delineamento tambem em JSON (mesmos achados do model card,
    # ja calculados na etapa [0/7]): o painel de status da aba Projeto mostra
    # severidade por checagem, e ler isso de volta do Markdown seria
    # reparsear um texto que ja' foi serializado uma vez.
    save_design_audit(
        pasta_logs,
        cast(List[Dict[str, str]], resumo["auditoria_delineamento"]))
    log.info(f"  -> {os.path.join(pasta_logs, 'auditoria_delineamento.json')}")
    return resumo


def _etapa_avaliar_holdout(*, X_holdout, _mae_id_holdout, cfg, lb, metricas_finais, metricas_funcoes, n_holdout, pasta, pls_final, preproc_full, rotulos_holdout):
    """Etapa 8b: avaliacao do modelo no holdout externo (metricas, BCa, conformal de classificacao, figura).

    Extraido de `_executar_impl` (P6, 2026-09-26) sem mudanca de logica:
    entradas e saidas explicitas; paridade bit a bit conferida."""
    metricas_holdout: Optional[Dict[str, float]] = None
    bca_holdout:      Optional[Dict[str, Tuple[float, float, float]]] = None
    conformal_classificacao: Optional[Dict[str, Any]] = None
    if (X_holdout is not None and rotulos_holdout is not None
            and should_generate(cfg, "holdout")):
        rot_ho: np.ndarray = rotulos_holdout
        log.info(f"\n[6c/7] Avaliacao em holdout ({n_holdout} amostras)...")
        try:
            X_holdout_proc = preproc_full.transform(X_holdout)
            Y_holdout_hat = pls_final.predict(X_holdout_proc)
            pred_holdout = lb.classes_[np.argmax(Y_holdout_hat, axis=1)]
            cm_holdout = confusion_matrix(rot_ho, pred_holdout,
                                            labels=lb.classes_)
            metricas_holdout = classification_metrics(
                rot_ho, pred_holdout, lb.classes_)
            for k, v in metricas_holdout.items():
                log.info(f"  {k:>22s}: {v:.4f}")
            fig_extra_holdout(metricas_finais, metricas_holdout,
                                cm_holdout, lb.classes_, n_holdout, cfg, pasta)
            # BCa CI no holdout tambem -- groups=_mae_id_holdout, mesmo
            # motivo do bloco de CV acima (achado #13).
            bca_holdout = {}
            for nome, fn in metricas_funcoes.items():
                lo, hi, obs = bootstrap_bca_ci(
                    rot_ho, pred_holdout, fn,
                    n_boot=cfg.n_bootstrap_bca, alpha=0.05,
                    seed=cfg.seed + 1, groups=_mae_id_holdout)
                bca_holdout[nome] = (lo, hi, obs)
            # Conjunto de predicao conforme p/ classificacao (fechamento do
            # Grupo 2 -- analogo classificatorio do T1 de regressao):
            # mesmo softmax-like clip+normalize de `predicao.predict_
            # samples`, calibrado no HOLDOUT (nunca visto pelo ajuste de
            # `pls_final`, exatamente o requisito de split-conformal). Sem
            # `_mae_id_holdout`, cada espectro do holdout conta como
            # amostra independente -- so' correto se nao houver replicas
            # (mesmo aviso de `conformal_margin_regression`/
            # `ConformalOneClass`).
            _Yho_clip = np.clip(np.asarray(Y_holdout_hat, dtype=float), 0.0, 1.0)
            _Yho_tot = _Yho_clip.sum(axis=1, keepdims=True)
            _Yho_tot[_Yho_tot < 1e-12] = 1.0
            Y_norm_holdout = _Yho_clip / _Yho_tot
            conformal_classificacao = conformal_margin_classification(
                rot_ho, Y_norm_holdout, lb.classes_,
                groups=_mae_id_holdout, alpha=0.05)
            if not conformal_classificacao["alcancavel"]:
                log.info(f"  [AVISO] {conformal_classificacao['aviso']}")
        except Exception as e:  # noqa: BLE001 -- avaliacao externa opcional;
            # erro impresso, metricas_holdout fica None e some do resumo
            # (nunca um valor inventado); metricas de CV (resultado central)
            # ja calculadas antes deste bloco.
            log.info(f"  [ERRO] Avaliacao em holdout falhou: {e}")
            metricas_holdout = None
    return (bca_holdout, conformal_classificacao, metricas_holdout,)


def _etapa_diagnosticos_e_figuras(*, P_pls, T_pls, X_processed, X_raw, Y_bin, Y_cv, _G, _fig_explor_on, cfg, classes_unicas, cm_mat, conc, cv_indices, erros_rmsecv, grupos_cv, lb, lv_cap, mae_id, mapa_cores, mapa_marcadores, metricas_por_lv, n_opt, pasta, pasta_dados, pca, pls_final, pred_lab, rotulos, scores_pca, var_lv_pls, var_pca, vip, wavenumbers, wold_res, y_int):
    """Etapa [6/7]: diagnosticos (T2/Q, DModX, DD-SIMCA, Etapa 4) e figuras (visao geral, exploratorias, classificacao, SR, DD-SIMCA, Etapa 4 de selecao de variaveis) filtradas pelo objetivo.

    Extraido de `_executar_impl` (P6, 2026-09-26) sem mudanca de logica:
    entradas e saidas explicitas; paridade bit a bit conferida."""
    log.info("\n[6/7] Gerando figuras...")
    aucs_roc: Dict[str, float] = {}
    # M1: mascara de puros para marcadores diferenciados
    # Pure samples: conc loaded as None -> NaN after asarray(float), OR stored as 0.0
    # Must handle BOTH cases to avoid false-negative mask for pure samples.
    _conc_f = np.asarray(conc, dtype=float) if conc is not None else None
    puros_mask_fig = (np.isnan(_conc_f) | (_conc_f == 0.0)) if _conc_f is not None else None
    # Flag de simbolos por classe (None -> todos circulo 'o')
    marcadores_fig = (mapa_marcadores if cfg.show_class_markers
                      else None)
    # ---- OVERVIEW (sempre — contexto valido em qualquer objetivo) ----
    # Espectros medios por classe: dado BRUTO, antes de qualquer modelagem
    # -- mesma logica de "sempre" do PCA/T2-Q abaixo (nao e' uma analise
    # exploratoria opcional, e' contexto quimico minimo para ler o resto).
    if should_generate(cfg, "espectros_medios"):
        fig_class_mean_spectra(wavenumbers, X_raw, rotulos, mapa_cores,
                               cfg, pasta)
    if should_generate(cfg, "pca_scores"):
        fig1_pca_scores(scores_pca, var_pca, rotulos, mapa_cores, cfg, pasta,
                        puros_mask=puros_mask_fig, mapa_marcadores=marcadores_fig)
    # ---- EXPLORATORIAS: nucleo do Modo Exploratorio; escotilha detalhada
    # dentro de Classificacao; FILTRADAS em Quantificacao. ----
    if _fig_explor_on:
        if should_generate(cfg, "hca"):
            fig_hca_dendrogram(X_processed, rotulos, mapa_cores, cfg, pasta)
        if should_generate(cfg, "loadings"):
            fig_loadings_pca(pca, wavenumbers, cfg, pasta, n_pcs=2)
        if should_generate(cfg, "biplot"):
            fig_biplot_pca(pca, scores_pca, wavenumbers, rotulos, mapa_cores,
                           cfg, pasta)
    if cfg.compare_hca_pipelines and _fig_explor_on and should_generate(cfg, "hca_comparacao"):
        fig_hca_comparacao_pipelines(X_raw, rotulos, mapa_cores, cfg, pasta)
    # ---- CLASSIFICACAO (supervisionada) — filtrada fora de N1/N2 ----
    if should_generate(cfg, "plsda_scores"):
        fig2_plsda_scores(T_pls, var_lv_pls, rotulos, mapa_cores, cfg, pasta,
                           puros_mask=puros_mask_fig,
                           mapa_marcadores=marcadores_fig)
    # Q de treino FORA da amostra, no espaco PLS (achado A4-03): o Q do proprio
    # ajuste e' otimista e o limite derivado dele (q_ucl, pls_q0/Nq e a coluna
    # `aceito` da predicao) rejeitava amostras novas legitimas. Leave-one-group-
    # out com o mesmo agrupamento da CV (mae_id) e X centrado pela media do PLS.
    q_treino = q_residuals_pls_loo(X_processed, Y_bin, n_opt, groups=grupos_cv)
    T2, Q, t2_lim, q_lim, out_t2, out_q = fig3_outliers(
        T_pls, P_pls, X_processed, rotulos, mapa_cores, n_opt, cfg, pasta,
        gerar_figura=should_generate(cfg, "outliers_t2q"), q_treino=q_treino)
    # DModX (Eriksson et al. 2006) -- mesma reapresentacao do Q-residuo
    # acima, na nomenclatura/escala que usuarios de SIMCA-P/Unscrambler
    # esperam. Nao gera figura nova (seria redundante com o painel T2/Q
    # acima); reportado no resumo/console/model card.
    _dmodx_res = cast(Dict[str, Any], dmodx(
        Q, n_variaveis=X_processed.shape[1],
        n_componentes=n_opt, n_amostras=X_processed.shape[0]))
    if should_generate(cfg, "confusao"):
        fig4_confusao(cm_mat, lb.classes_, rotulos, pred_lab, cfg, pasta)
    if should_generate(cfg, "roc"):
        try:
            aucs_roc = fig_roc_auc(Y_bin, Y_cv, lb.classes_, cfg, pasta)
        except Exception as _e_roc:  # noqa: BLE001 -- figura opcional (curva
            # ROC/AUC); erro impresso, resultado central da corrida intacto.
            log.info(f"  [AVISO] ROC/AUC: {_e_roc}")
    # fig4b_metricas_globais e fig5_vip removidas: a primeira e redundante com
    # resumo_modelo.txt; a segunda (VIP puro) esta contida em fig_selectivity_ratio_vip,
    # que mostra VIP + Selectivity Ratio lado a lado (ver abaixo).

    if cfg.n_bootstrap_vip > 0 and should_generate(cfg, "vip"):
        log.info(f"  [bootstrap VIP estratificado, n={cfg.n_bootstrap_vip}]")
        boot = bootstrap_vip_stratified(
            X_processed, Y_bin, y_int, n_opt,
            cfg.n_bootstrap_vip, cfg.seed,
            mae_id=grupos_cv)   # group-aware: respects mae_id replicates
        boot_validos = cast(int, boot["n_validos"])
        log.info(f"  Iteracoes validas: {boot_validos}/"
              f"{cfg.n_bootstrap_vip}  (falhos: {cast(int, boot['n_falhos'])})")
        if boot_validos > 0:
            fig5b_vip_stability(boot, wavenumbers, top_n=20,
                                     cfg=cfg, pasta=pasta)
        else:
            log.info("  [AVISO] Bootstrap VIP: 0 iteracoes validas — fig5b pulada.")

    if _fig_explor_on and should_generate(cfg, "preprocessamento"):
        fig6_preprocessing(wavenumbers, X_raw, X_processed, rotulos,
                               mapa_cores, cfg, pasta)
    if should_generate(cfg, "selecao_lvs"):
        fig1_selecao_lvs(erros_rmsecv, metricas_por_lv, n_opt, cfg, pasta)

    # ---- Sprint 3 — SR (essencial) + Score Contribution (detalhada) -----
    # sr e' computado SEMPRE (consumido tambem pela Etapa 4); apenas as
    # FIGURAS de SR/VIP sao filtradas por objetivo (classificacao).
    log.info("\n[Razao de seletividade] Selectivity Ratio + contribuicao dos escores...")
    sr = compute_selectivity_ratio(pls_final, X_processed)
    if should_generate(cfg, "sr_vip"):
        fig_sprint3_sr_vip(vip, sr, wavenumbers, top_n=20, cfg=cfg, pasta=pasta)
    if cfg.detailed_figures and should_generate(cfg, "score_contribution"):
        fig_sprint3_score_contribution(pls_final, X_processed, rotulos,
                                        wavenumbers, mapa_cores, top_n=20,
                                        cfg=cfg, pasta=pasta)

    # Teste de incerteza de Martens (opcional) -- jackknifing group-aware
    # dos coeficientes PLS, complementa VIP/SR com um teste de hipotese
    # formal (p-valor) de significancia por variavel.
    _martens_n_sig: Optional[int] = None
    _martens_n_folds: Optional[int] = None
    if cfg.run_martens and should_generate(cfg, "martens"):
        log.info("  [Martens] Jackknifing group-aware dos coeficientes PLS...")
        martens = martens_uncertainty_test(
            X_processed, Y_bin, n_opt, cv_indices, pls_final.coef_)
        _martens_n_folds = int(martens["n_folds_validos"])
        if _martens_n_folds >= 3:
            _martens_n_sig = int(np.sum(martens["significativo"]))
            log.info(f"  {_martens_n_sig}/{len(wavenumbers)} variaveis "
                  f"significativas (p<0.05, {_martens_n_folds} folds validos)")
        else:
            log.info(f"  [AVISO] Apenas {_martens_n_folds} folds validos "
                  "(<3) -- jackknife nao pode estimar variancia.")
        df_martens = pd.DataFrame({
            "wavenumber":    wavenumbers,
            "t_valor":       martens["t_valores"],
            "p_valor":       martens["p_valores"],
            "significativo": martens["significativo"],
        })
        cam_martens = os.path.join(pasta_dados, "teste_martens.csv")
        df_martens.to_csv(cam_martens, index=False, sep=";", decimal=",")
        log.info(f"  -> {cam_martens}")

    # DD-SIMCA — configurable training mode (v14).
    #   'todos' (default): trains each model on all class samples
    #     (exploratory; robust with few pure samples). sens = fraction of
    #     own class accepted.
    #   'puros': true one-class N2 (trains only on pure samples) — requires
    #     >=15 pure samples/class. With 3/class it generates a tiny region
    #     and ~all samples become 'Unknown'. sens = pure accepted; esp = adult rejected.
    ddsimca_res: Optional[Dict[str, Dict[str, Any]]] = None
    simca_pred: np.ndarray = np.array([], dtype=str)
    # (sens_LOGO, esp, n_puros, n_adult, n_grupos_LOGO, aviso)
    ddsimca_sens_esp: Dict[
        str, Tuple[float, float, int, int, int, Optional[str]]] = {}
    # PCV: diagnostico complementar, opt-in via cfg.ddsimca_pcv (ver
    # ddsimca_pcv_sensitivity). (sens_PCV, aviso)
    ddsimca_pcv_esp: Dict[str, Tuple[float, Optional[str]]] = {}
    _pcv_indisponivel_avisado = False
    modo_dd: str = "todos"  # default; overwritten if run_ddsimca=True
    # DD-SIMCA e' um diagnostico de AUTENTICACAO DE PUREZA (N2): pergunta se
    # a amostra pertence a regiao de aceitacao da sua propria especie/classe.
    # Em N1 (identificacao de especie), essa pergunta nao agrega -- o pipeline
    # ja classifica a especie via PLS-DA, e um grafico de "aceito/rejeitado"
    # sem eixo de pureza confunde mais do que esclarece num estudo de N1. Por
    # isso o toggle e' ignorado (nao bloqueado no Config, so' na execucao) com
    # aviso explicito, mesmo que o usuario tenha ligado manualmente.
    if cfg.run_ddsimca and cfg.level == "N1":
        log.info("\n[DD-SIMCA] IGNORADO: nivel N1 (identificacao de "
              "especie). DD-SIMCA e um diagnostico de autenticacao de pureza "
              "(conceito de N2); nao agrega a este tipo de analise. Troque "
              "para nivel N2 se quiser autenticar pureza por especie.")
    elif cfg.run_ddsimca and should_generate(cfg, "ddsimca"):
        modo_dd = (cfg.ddsimca_train_on or "todos").lower()
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

        log.info(f"\n[DD-SIMCA] "
              f"(n_components={cfg.ddsimca_n_components}, alpha=0.05, "
              f"ucl={cfg.ddsimca_ucl_method}, treino='{modo_dd}') "
              f"— {n_treino} amostras de treino")
        if modo_dd == "puros" and n_treino < (cfg.ddsimca_n_components + 2) * 2:
            log.info(f"  [AVISO] Poucos puros ({n_treino}) para one-class robusto"
                  f" (recomendado >=15/classe). Resultados exploratorios.")

        ddsimca = DDSimca(n_components=cfg.ddsimca_n_components, alpha=0.05,
                           ucl_method=cfg.ddsimca_ucl_method)
        # mae_id (achado F1/A2-3, auditoria 2026-08-16): calibra h0/q0/Nh/Nq
        # por AMOSTRA FISICA, nao por espectro -- ver docstring de
        # DDSimca.fit(). Ausente em modos sem identificador de replica
        # (ex.: modo_entrada="imagem"), onde None preserva o comportamento
        # anterior (Nh/Nq por espectro) com aviso explicito no log.
        ddsimca.fit(X_processed[mask_treino], rotulos[mask_treino],
                    mae_id=_G(mae_id)[mask_treino] if mae_id is not None else None)
        # mask_treino+rotulos (achado A1): pontos de TREINO recebem o Q
        # leave-one-out usado para calibrar a fronteira, em vez do Q
        # in-sample -- sem isso, pontos e fronteira ficam em escalas
        # diferentes na figura de aceitacao (Q in-sample medido 10-15x
        # menor no regime real). Amostras novas seguem in-sample, que e' o
        # valor correto para elas.
        ddsimca_res   = ddsimca.score_matrix(       # pontua em TODOS
            X_processed, mask_treino=mask_treino, y=rotulos)
        simca_pred    = ddsimca.predict(X_processed)
        n_unknown     = int(np.sum(simca_pred == "Desconhecido"))
        n_ambig       = int(np.sum(simca_pred == "Ambiguo"))

        log.info(f"  {'Classe':18s} {'sens(LOGO)':>11s} {'grupos':>7s} "
              f"{'esp(adult)':>11s}")
        for cls in classes_unicas:
            if cls not in ddsimca_res:
                continue
            m = ddsimca_res[cls]
            aceito = np.asarray(m["f"]) <= m["f_crit"]
            idx_puro_c  = (rotulos == cls) & mask_puros
            idx_adult_c = (rotulos == cls) & (~mask_puros)
            idx_cls     = (rotulos == cls)
            n_puro_c    = int(idx_puro_c.sum())
            n_adult_c   = int(idx_adult_c.sum())
            n_grupos_c  = 0
            aviso_sens: Optional[str] = None
            # Sensibilidade:
            #   'puros' (one-class N2): leave-one-group-out HONESTO por mae_id.
            #     Re-substituicao (media sobre os proprios puros de treino) infla
            #     para ~100% e NAO e evidencia de autenticacao -- mede o modelo
            #     reconhecendo dados que ja viu. LOGO retreina sem um grupo de
            #     replica e testa os puros retidos (ver ddsimca_logo_sensitivity).
            #   'todos': fracao in-sample da classe inteira aceita (ja rotulada
            #     como nao-autenticacao no resumo/figuras).
            if modo_dd == "puros" and n_puro_c > 0:
                if mae_id is not None:
                    _logo = ddsimca_logo_sensitivity(
                        X_processed[idx_puro_c], _G(mae_id)[idx_puro_c],
                        n_components=cfg.ddsimca_n_components,
                        alpha=0.05, ucl_method=cfg.ddsimca_ucl_method,
                        # m-06: pre-processamento reajustado dentro de cada dobra
                        X_brutos=X_raw[idx_puro_c],
                        fabrica_preproc=lambda: build_preprocessor(cfg),
                        classe=str(cls))
                    sens       = _logo["sensibilidade"]
                    n_grupos_c = int(_logo["n_grupos"])
                    aviso_sens = _logo["aviso"]
                    # PCV: diagnostico complementar opt-in (nunca substitui
                    # o LOGO acima) -- ver ddsimca_pcv_sensitivity().
                    if cfg.ddsimca_pcv:
                        _pcv = ddsimca_pcv_sensitivity(
                            X_processed[idx_puro_c], _G(mae_id)[idx_puro_c],
                            n_components=cfg.ddsimca_n_components,
                            alpha=0.05, ucl_method=cfg.ddsimca_ucl_method,
                            classe=str(cls))
                        if _pcv["disponivel"]:
                            ddsimca_pcv_esp[cls] = (
                                _pcv["sensibilidade"], _pcv["aviso"])
                        elif not _pcv_indisponivel_avisado:
                            log.info(f"    [AVISO] PCV: {_pcv['aviso']}")
                            _pcv_indisponivel_avisado = True
                else:
                    sens = float("nan")
                    aviso_sens = ("Sensibilidade nao estimavel: mae_id ausente "
                                  "(sem grupos de replica para LOGO).")
            else:
                sens = float(np.mean(aceito[idx_cls]))
                n_grupos_c = (int(len(np.unique(_G(mae_id)[idx_cls])))
                              if mae_id is not None else 0)
            # B4: in 'todos' mode adulterated samples are IN TRAINING, so
            # "specificity" would be in-sample (not authentication) and misleading.
            # Only reported in 'puros' mode (true one-class).
            if modo_dd == "puros":
                esp = (float(np.mean(~aceito[idx_adult_c]))
                       if n_adult_c > 0 else float("nan"))
            else:
                esp = float("nan")
            ddsimca_sens_esp[cls] = (sens, esp, n_puro_c, n_adult_c,
                                     n_grupos_c, aviso_sens)
            sens_txt = f"{sens*100:.1f}%" if sens == sens else "n/a"
            esp_txt  = f"{esp*100:9.1f}%" if esp == esp else "      n/a"
            log.info(f"  {cls:18s} {sens_txt:>11s} {n_grupos_c:>7d} {esp_txt}"
                  f"   (puros={n_puro_c}, adult={n_adult_c})")
            if aviso_sens:
                log.info(f"    [AVISO] {cls}: {aviso_sens}")
        log.info(f"  Desconhecidos: {n_unknown}  |  Ambiguos: {n_ambig}")
        if ddsimca_res:
            # Essencial: painel de aceitacao consolidado (todas as classes).
            fig_sprint3_ddsimca_acceptance(
                ddsimca_res, rotulos, mapa_cores, cfg, pasta,
                sens_esp=ddsimca_sens_esp)
            if cfg.detailed_figures:
                # Detalhadas: um plot por classe (subpasta ddsimca/) + Cooman.
                fig_ddsimca_individuais(
                    ddsimca_res, rotulos, mapa_cores, cfg, pasta,
                    sens_esp=ddsimca_sens_esp)
                if len(ddsimca_res) >= 2:
                    try:
                        fig_cooman_ddsimca(ddsimca_res, rotulos, mapa_cores,
                                           cfg, pasta)
                    except Exception as _e_coom:  # noqa: BLE001 -- figura
                        # opcional (Cooman's Plot); erro impresso, DD-SIMCA
                        # ja calculado e reportado independentemente.
                        log.info(f"  [AVISO] Cooman's Plot: {_e_coom}")

    # OPLS-DA
    _opls_n_ortho: Optional[int] = None
    if cfg.run_opls and should_generate(cfg, "opls"):
        n_cls_opls = len(classes_unicas)
        log.info(f"\n[OPLS-DA] "
              f"(n_ortho={cfg.n_ortho_opls}, {n_cls_opls} classes)...")
        if n_cls_opls < 2:
            log.info("  [AVISO] OPLS-DA requer >= 2 classes.")
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
                log.info(f"  Componentes ortogonais ajustados: {_opls_n_ortho}")
            except Exception as _e_opls:  # noqa: BLE001 -- modulo opcional
                # (OPLS-DA/S-Plot); erro impresso, _opls_n_ortho fica None e
                # some do resumo em vez de exibir um valor inventado; PLS-DA
                # (resultado central) ja calculado antes deste bloco.
                log.info(f"  [ERRO] OPLS-DA: {_e_opls}")

    # --- STAGE 4: Variable Selection ------------------------------------
    etapa4_res: Optional[Dict[str, Any]] = None
    if cfg.executar_etapa4 and should_generate(cfg, "etapa4"):
        try:
            etapa4_res = etapa4_selecao_variaveis(
                X_processed, Y_bin, y_int, wavenumbers,
                cv_indices, n_opt, cfg, pasta, pasta_dados,
                # mae_id (achado B1-3): torna group-aware tambem a CV
                # INTERNA que guia as buscas SPA/AG -- sem isso, a busca
                # escolhe variaveis que exploram similaridade entre
                # replicas, ainda que o numero reportado seja honesto.
                mae_id=_G(mae_id))
        except Exception as _e_e4:  # noqa: BLE001 -- modulo opcional (selecao
            # de variaveis); erro impresso, etapa4_res fica None e some do
            # resumo; PLS-DA (resultado central) ja calculado antes.
            log.info(f"  [ERRO] Etapa 4: {_e_e4}")

    if cfg.comparar_pipelines and should_generate(cfg, "comparar_pipelines"):
        log.info("\n[6b/7] Comparacao de pipelines de pre-processamento...")
        comp = compare_pipelines(cfg, X_raw, Y_bin, y_int, cv_indices,
                                    max_lv=lv_cap)
        fig_extra_comparacao_pipelines(comp, cfg, pasta)
        pd.DataFrame(comp).T.to_csv(
            os.path.join(pasta_dados, "comparacao_pipelines.csv"),
            sep=";", decimal=",")

    if (wold_res is not None and cast(int, wold_res["n_validos"]) > 2
            and should_generate(cfg, "wold")):
        fig_extra_wold(wold_res, cfg, pasta)
    return (Q, T2, _dmodx_res, _martens_n_folds, _martens_n_sig, _opls_n_ortho, aucs_roc, ddsimca_pcv_esp, ddsimca_res, ddsimca_sens_esp, etapa4_res, modo_dd, out_q, out_t2, q_lim, simca_pred, t2_lim,)


def _etapa_holdout_e_estrategia_cv(*, X_raw, _G, _eixo_orig, cfg, conc, mae_id, n_orig, relatorio_balanco, rotulos, usar_grupos):
    """Etapa 1d: estrategia de CV (agrupada quando ha grupos) e reserva do holdout externo (agrupado e estratificado quando ha' grupos) ANTES de qualquer escolha de modelo.

    Extraido de `_executar_impl` (P6, 2026-09-26) sem mudanca de logica:
    entradas e saidas explicitas; paridade bit a bit conferida."""
    cv: Any = None  # saida condicional (preserva o comportamento original)
    cv_label: Any = None  # saida condicional (preserva o comportamento original)
    X_holdout = None; rotulos_holdout = None; _conc_holdout = None
    _mae_id_holdout: Optional[np.ndarray] = None
    _classes_fora_holdout: List[str] = []
    n_holdout = 0
    # Objetivo exploratorio so' gera PCA/HCA/loadings: nao avalia modelo, entao reservar holdout tiraria amostras
    # dos graficos sem servir a nada, e classe com uma so' unidade nao e' motivo para parar (ver n_splits abaixo).
    _exploratorio = resolve_objective(cfg) == EXPLORATORIO
    if _exploratorio and cfg.frac_holdout > 0:
        log.info("[INFO] Objetivo exploratorio: sem holdout (os graficos usam todas as amostras).")
    if cfg.frac_holdout > 0 and not _exploratorio:
        try:
            # v15: optionally excludes pure samples from the draw — they always
            # stay in training. Split runs only on the eligible subset (adulterated).
            n_all = len(rotulos)
            if (cfg.holdout_preserves_pure and conc is not None):
                elegiveis = np.where(np.asarray(conc, dtype=float) > 0)[0]
                n_puros_reserv = n_all - len(elegiveis)
            else:
                elegiveis = np.arange(n_all)
                n_puros_reserv = 0

            Xe, rote = X_raw[elegiveis], rotulos[elegiveis]
            maee = _G(mae_id)[elegiveis] if mae_id is not None else None

            if usar_grupos and maee is not None:
                if getattr(cfg, "holdout_estratificado", True):
                    tr_e, ho_e = holdout_agrupado_estratificado(
                        rote, maee, cfg.frac_holdout, cfg.seed_holdout)
                    tipo_ho = "StratifiedGroup (replicas juntas, por classe)"
                else:
                    gss = GroupShuffleSplit(
                        n_splits=1, test_size=cfg.frac_holdout,
                        random_state=cfg.seed_holdout)
                    tr_e, ho_e = next(gss.split(Xe, rote, groups=maee))
                    tipo_ho = "GroupShuffleSplit (replicas juntas; legado)"
            else:
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=cfg.frac_holdout,
                    random_state=cfg.seed_holdout)
                tr_e, ho_e = next(sss.split(Xe, rote))
                tipo_ho = "StratifiedShuffleSplit"
            # R2-07: dataset minusculo devolvia indices vazios (float) e estourava
            # IndexError opaco; vira ValueError, tratado abaixo (segue sem holdout).
            tr_e = np.asarray(tr_e, dtype=int); ho_e = np.asarray(ho_e, dtype=int)
            if tr_e.size == 0 or ho_e.size == 0:
                raise ValueError("amostras/grupos insuficientes para separar treino e holdout")
            # Remap indices from eligible subset back to global indices;
            # pure (non-eligible) samples are added entirely to training.
            nao_elegiveis = np.setdiff1d(np.arange(n_all), elegiveis)
            tr_idx = np.concatenate([elegiveis[tr_e], nao_elegiveis])
            ho_idx = elegiveis[ho_e]
            if n_puros_reserv > 0:
                tipo_ho += f" + {n_puros_reserv} puros preservados"
            # M-05a: classe que ficou FORA do holdout e' um fato que o resumo
            # precisa dizer (bal.acc do holdout so' cobre as classes presentes).
            _cls_todas = sorted({str(c) for c in rotulos})
            _cls_ho = {str(c) for c in rotulos[ho_idx]}
            _classes_fora_holdout = [c for c in _cls_todas if c not in _cls_ho]
            if _classes_fora_holdout:
                log.warning(
                    f"[AVISO] {len(_classes_fora_holdout)} classe(s) FORA do "
                    f"holdout: {_classes_fora_holdout} (grupos insuficientes "
                    f"para representa-las nos dois lados). As metricas do "
                    f"holdout NAO cobrem essas classes.")
            X_holdout       = X_raw[ho_idx]
            rotulos_holdout = rotulos[ho_idx]
            # Reservados junto do holdout (ainda nao consumidos na avaliacao —
            # prefixo _ sinaliza intencional e evita falso-positivo de lint).
            _conc_holdout   = conc[ho_idx] if conc is not None else None
            if mae_id is not None:
                _mae_id_holdout = _G(mae_id)[ho_idx]
            X_raw   = X_raw[tr_idx]
            rotulos = rotulos[tr_idx]
            conc    = conc[tr_idx] if conc is not None else None
            if mae_id is not None:
                mae_id = mae_id[tr_idx]
            n_holdout = int(len(ho_idx))
            log.info(f"[INFO] Hold-out ({tipo_ho}): {n_holdout} amostras "
                  f"reservadas (frac={cfg.frac_holdout:.2f}). "
                  f"Pipeline rodara em {len(tr_idx)} amostras.")
        except ValueError as e:   # test_size/n_splits incompativel com os dados
            log.info(f"[AVISO] Hold-out falhou ({e}). Continuando sem holdout.")
            X_holdout = None

    # SG so entra no preprocessador ativo dependendo do PRESET (ver
    # build_preprocessor em preprocessamento.py) — "autoscaling"/"mc"
    # NUNCA usam SG (a flag cfg.apply_sg so vale p/ o preset "custom"),
    # enquanto "snv_sg_mc"/"msc_sg_mc" SEMPRE usam SG independente da flag.
    # Checar so `cfg.apply_sg` aqui gerava falso-positivo com presets sem
    # SG (ex.: autoscaling com poucas variaveis, como no mode="imagem").
    from guaraci.preprocessamento import descrever_preprocessamento
    _sg_ativo = descrever_preprocessamento(cfg)[2]
    # R2-08 (revisao cega): com 1-2 amostras toda variavel e' constante e sai no
    # filtro de integridade; a mensagem antiga culpava a faixa espectral.
    if X_raw.shape[0] < 3:
        raise ValueError(
            f"So' {X_raw.shape[0]} amostra(s) depois da validacao de entrada: "
            "e' preciso pelo menos 3 (e, para validar, 2 por classe). A faixa "
            "espectral nao e' o problema.")
    if _sg_ativo and cfg.sg_window >= X_raw.shape[1]:
        _restam = X_raw.shape[1]
        _un_msg = f" {unidade_do_eixo(cfg) or 'cm-1'}"   # sem unidade declarada: o texto antigo (cm-1)
        _msg_faixa = (
            f"A faixa espectral configurada [{cfg.wn_min:.0f}, {cfg.wn_max:.0f}]"
            f"{_un_msg} deixou apenas {_restam} de {n_orig} pontos do dado (eixo "
            f"real [{_eixo_orig[0]:.0f}, {_eixo_orig[1]:.0f}]{_un_msg}). "
            if _restam < n_orig else "")
        raise ValueError(
            f"{_msg_faixa}sg_window ({cfg.sg_window}) deve ser menor que o "
            f"numero de pontos espectrais ({_restam}). Ajuste "
            f"faixa_min_cm/faixa_max_cm para cobrir o eixo dos dados "
            f"(ex.: {_eixo_orig[0]:.0f} a {_eixo_orig[1]:.0f}) ou reduza "
            f"sg_window.")

    classes_unicas = np.unique(rotulos)
    mapa_cores     = map_class_colors(classes_unicas)
    mapa_marcadores = map_class_markers(classes_unicas)

    log.info(f"\n[INFO] Amostras : {X_raw.shape[0]}")
    log.info(f"[INFO] Variaveis: {X_raw.shape[1]}")
    log.info(f"[INFO] Classes  : {classes_unicas.tolist()}")
    log.info(f"[INFO] Conc.    : {'sim' if conc is not None else 'nao'}")
    _mae_info = (f"sim ({int(len(np.unique(mae_id)))} grupos)"
                  if mae_id is not None else "nao")
    log.info(f"[INFO] mae_id   : {_mae_info}")
    log.info(f"[INFO] Imbalance ratio: {relatorio_balanco['imbalance_ratio']:.2f}  "
          f"(max={relatorio_balanco['n_max']}, min={relatorio_balanco['n_min']})")

    contagem = {cls: int(np.sum(rotulos == cls)) for cls in classes_unicas}
    min_por_classe = min(contagem.values())
    n_splits = min(cfg.n_splits_cv, min_por_classe)
    if usar_grupos and mae_id is not None:
        # Numero de grupos por classe limita k
        n_grupos_por_classe = []
        for cls in classes_unicas:
            n_grupos_por_classe.append(
                int(len(np.unique(_G(mae_id)[rotulos == cls]))))
        min_grupos_classe = min(n_grupos_por_classe)
        n_splits = min(n_splits, min_grupos_classe)
        if n_splits < cfg.n_splits_cv:
            log.info(f"[AVISO] n_splits ajustado para {n_splits} "
                  f"(min de grupos/classe = {min_grupos_classe}).")
    elif n_splits < cfg.n_splits_cv:
        log.info(f"[AVISO] n_splits_cv={cfg.n_splits_cv} excede minimo por "
              f"classe ({min_por_classe}). Usando {n_splits}.")
    # R5-06/R2-17: com 1 so' fold NAO ha' validacao cruzada -- a "CV" reportada
    # (accuracy 0,975 com n_splits=1) nao valida nada e a etapa de LV rodava
    # ~9 min sem sinal de vida. Recusa antes, dizendo quais classes faltam.
    _cv_exploratoria = False
    if n_splits < 2 and _exploratorio:
        _n_unid = int(len(np.unique(_G(mae_id)))) if (usar_grupos and mae_id is not None) else len(rotulos)
        n_splits = max(2, min(cfg.n_splits_cv, _n_unid))
        _cv_exploratoria = True
        log.info("[AVISO] Ha' classe com uma so' unidade: sem validacao por classe. No objetivo exploratorio isso "
                 "nao impede os graficos; o modelo PLS-DA interno usa divisao simples e nao e' validado.")
    if n_splits < 2:
        _pequenas = sorted(c for c in classes_unicas if contagem[c] < 2)
        if usar_grupos and mae_id is not None:
            # com validacao por grupo, o que falta e' grupo: classe com 3 replicas de UM ponto tambem conta
            _pequenas = sorted(c for c, n in zip(classes_unicas, n_grupos_por_classe) if n < 2)
        raise ValidacaoCruzadaImpossivel(_pequenas,
            "Validacao cruzada impossivel: e' preciso pelo menos 2 amostras "
            "(ou 2 grupos independentes) em CADA classe. Classe(s) com so' 1: "
            f"{', '.join(map(str, _pequenas[:10])) or 'ver grupos por classe'}"
            f"{'...' if len(_pequenas) > 10 else ''}. Junte classes, remova as "
            "menores (excluir_classes) ou traga mais amostras.")

    if _cv_exploratoria:
        # GroupKFold/KFold sem estratificar: deterministico e aceita classe que cai inteira num fold so'.
        cv = GroupKFold(n_splits=n_splits) if (usar_grupos and mae_id is not None) else KFold(n_splits=n_splits)
        cv_label = f"{type(cv).__name__} n_splits={n_splits} (exploratorio: nao valida o modelo)"
    elif usar_grupos:
        # Estratifica por classe E agrupa por mae_id. Implementacao PROPRIA
        # (validacao_estatistica.StableStratifiedGroupKFold) em vez da do
        # scikit-learn: a do sklearn muda a particao entre versoes mesmo com
        # random_state fixo -- medido em 2026-08-05, 42% das amostras trocaram
        # de fold entre 1.7.2 e 1.9.0. Isso fazia Q2/RMSECV/acuracia/F1 e ate'
        # o numero de LVs otimas dependerem da versao instalada, contradizendo
        # a reprodutibilidade que e' o argumento central do projeto.
        cv = StableStratifiedGroupKFold(n_splits=max(n_splits, 2),
                                         seed=cfg.seed)
        cv_label = f"StableStratifiedGroupKFold n_splits={n_splits}"
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
    # Achado do Passo 148 (2026-09-04): faltava reconstruir a
    # 2a coluna one-hot para EXATAMENTE 2 classes; sem ela, toda predicao
    # colapsava na 1a classe. Extraido para funcao compartilhada no Passo
    # 156 -- ver docstring de `expandir_binario_um_quente`.
    Y_bin = expandir_binario_um_quente(Y_bin)
    y_int = np.argmax(Y_bin, axis=1)
    return (X_holdout, X_raw, Y_bin, _classes_fora_holdout, _conc_holdout, _mae_id_holdout, classes_unicas, conc, cv, cv_label, lb, mae_id, mapa_cores, mapa_marcadores, n_holdout, n_splits, rotulos, rotulos_holdout, y_int,)


def _etapa_selecao_lvs(*, X_raw, Y_bin, _G, cfg, classes_unicas, cv, cv_label, lb, mae_id, n_splits, usar_grupos, y_int):
    """Etapa [2/7]: selecao do numero de variaveis latentes por CV (agrupada quando possivel) e CV aninhada honesta.

    Extraido de `_executar_impl` (P6, 2026-09-26) sem mudanca de logica:
    entradas e saidas explicitas; paridade bit a bit conferida."""
    log.info(f"\n[2/7] LV selection by CV ({cv_label})")

    def fabrica_pipeline(n_lv: int):
        return Pipeline([
            ("preproc", build_preprocessor(cfg)),
            ("pls", _novo_pls_classificador(cfg, n_lv)),
        ])

    erros_rmsecv: List[float] = []
    metricas_por_lv: List[Dict[str, float]] = []
    preds_por_lv: Dict[int, np.ndarray] = {}

    grupos_cv = _G(mae_id) if usar_grupos else None
    cv_indices = list(cv.split(X_raw, y_int, groups=grupos_cv))

    # Achado real (1o uso de dado pequeno pelo roteiro de teste externo,
    # 2026-09-23): com max_lvs=40 (default) e um fold de treino de 38
    # amostras, PLSRegression(n_components=39) estourava "n_components
    # upper bound is 38" e a execucao inteira falhava. O teto de LVs
    # nunca pode passar do menor fold de treino - 1 (mesma defesa ja'
    # existente em `_selecionar_n_opt_wold`, que esta etapa nao tinha).
    _min_tr = min(len(tr) for tr, _ in cv_indices)
    lv_cap = max(1, min(cfg.max_lvs, _min_tr - 1, X_raw.shape[1]))
    if lv_cap < cfg.max_lvs:
        log.info(f"  [INFO] max_lvs={cfg.max_lvs} limitado a {lv_cap} pelo "
                 f"tamanho do menor fold de treino ({_min_tr} amostras).")

    _preds_cv = _predicoes_cv_por_lv(fabrica_pipeline, X_raw, Y_bin,
                                     cv_indices, lv_cap)
    for n in range(1, lv_cap + 1):
        y_hat = _preds_cv[n]

        erros_rmsecv.append(rmse_flat(Y_bin, y_hat))
        y_hat_int = np.argmax(y_hat, axis=1)
        m = classification_metrics(y_int, y_hat_int, np.arange(len(classes_unicas)))
        metricas_por_lv.append(m)
        preds_por_lv[n] = y_hat

    rmsecv_arr = np.array(erros_rmsecv)
    rmsecv_min = float(rmsecv_arr.min())
    n_opt_minrmsecv = int(np.argmin(rmsecv_arr)) + 1

    # Parsimony rule of thumb (2% tolerance; NOT a published Wold criterion): select the SMALLEST number of LVs
    # whose RMSECV is within 2% of the minimum. Avoids overfitting when
    # RMSECV plateau is broad (common with max_lvs=40 and noisy FT-NIR data).
    tol_wold = rmsecv_min * 1.02
    candidatos_wold = np.where(rmsecv_arr <= tol_wold)[0]
    n_opt = int(candidatos_wold[0]) + 1   # smallest LV satisfying criterion

    if n_opt < n_opt_minrmsecv:
        log.info(f"  LVs otimas (Wold parcimonia): {n_opt} "
              f"(min-RMSECV em {n_opt_minrmsecv} LVs, delta <2%)")
    else:
        log.info(f"  LVs otimas: {n_opt}")

    Y_cv  = preds_por_lv[n_opt]

    # --- 3b. CV aninhada (correcao do achado #12/Passo 211, opt-out) -------
    # A selecao acima (Wold) e a metrica de CV acima (Y_cv) usam os MESMOS
    # folds: o mesmo dado escolhe n_opt E avalia a metrica reportada, o que
    # infla a metrica de forma otimista (medido, 10 seeds replicadas,
    # Wilcoxon p=0,0020). A correcao
    # NAO muda `n_opt` do modelo final (`pls_final` abaixo continua com o
    # `n_opt` escolhido acima, sobre TODO o dado -- e' a escolha certa pra'
    # UM modelo a ser implantado; nao existe "n_opt aninhado" pra' um unico
    # modelo, CV aninhada e' tecnica de AVALIACAO honesta, nao de selecao de
    # hiperparametro do artefato final). O que muda e' de ONDE vem a metrica
    # REPORTADA (`Y_cv`/`pred_lab`, usados por Q2, classification_metrics,
    # ROC AUC, CV-ANOVA, matriz de confusao, bootstrap CI, resumo/model
    # card): quando ativada, `Y_cv` passa a vir de predicoes onde `n_opt` de
    # cada fold externo foi escolhido SO' com o treino daquele fold (CV
    # interna, `_construir_cv_interno`), nunca vendo a propria validacao.
    metricas_cv_naive = metricas_por_lv[n_opt - 1]
    if cfg.selecao_lv_cv_aninhada:
        log.info("  [INFO] CV aninhada ativada (selecao_lv_cv_aninhada="
                 "True, default a partir da v1.0 -- corrige o vies de "
                 "reusar o mesmo dado p/ escolher n_opt e avaliar a "
                 "metrica reportada). Pode levar "
                 "~4-5x mais tempo nesta etapa; desative no config.yaml "
                 "(selecao_lv_cv_aninhada: false) p/ iteracao rapida, "
                 "aceitando a metrica otimista.")
        t0_aninhada = time.time()
        Y_cv_aninhado = np.zeros_like(Y_bin)
        n_opts_aninhado = []
        n_folds_aninhado = len(cv_indices)
        for _i_fold_aninhado, (tr, va) in enumerate(cv_indices):
            grupos_tr = grupos_cv[tr] if grupos_cv is not None else None
            y_tr = y_int[tr]
            classes_tr = np.unique(y_tr)
            if usar_grupos and grupos_tr is not None:
                min_disponivel = min(
                    int(len(np.unique(grupos_tr[y_tr == c]))) for c in classes_tr)
            else:
                min_disponivel = int(np.min(np.unique(y_tr, return_counts=True)[1]))
            if min_disponivel < 2:
                # Treino do fold externo pequeno demais p/ sustentar uma CV
                # interna de verdade (< 2 grupos/amostras da classe minoritaria
                # -- forcar `max(n_splits, 2)` aqui produziria um fold interno
                # com uma classe VAZIA, e dai' um X vazio chegando no
                # pre-processador -- achado real, LAPACK crashava em
                # savgol_filter num fixture sintetico de 12 amostras). Honesto
                # cai de volta pro n_opt global (naive) so' NESTE fold, em vez
                # de forcar um split degenerado ou quebrar.
                n_opt_i = n_opt
            else:
                cv_interna = _construir_cv_interno(
                    y_tr, grupos_tr, usar_grupos, n_splits, cfg.seed)
                cv_idx_interna = list(
                    cv_interna.split(X_raw[tr], y_tr, groups=grupos_tr))
                n_opt_i, _ = _selecionar_n_opt_wold(
                    fabrica_pipeline, X_raw[tr], Y_bin[tr], cv_idx_interna,
                    cfg.max_lvs)
            n_opts_aninhado.append(n_opt_i)
            pipe_i = fabrica_pipeline(n_opt_i)
            pipe_i.fit(X_raw[tr], Y_bin[tr])
            Y_cv_aninhado[va] = pipe_i.predict(X_raw[va])
            # Indicacao de progresso (Grupo 3 do MAPA_COMPLETUDE_V1.md,
            # 2026-09-11): sem isto, esta etapa fica ate' ~16min em
            # silencio total entre o log de "ativada" e o de resultado
            # final -- medido, nao estimado (medicao do mantenedor,
            # script nao versionado). "[2b/7]" segue a mesma
            # convencao de sub-passo ja usada em "[6b/7]"/"[6c/7]"
            # (ver app_logic.log_progress); o total de folds vai
            # embutido na propria linha para o parser nao precisar de
            # um parametro novo, ao contrario do bonus de "[6/7]".
            log.info(f"  [2b/7] CV aninhada: fold externo "
                     f"{_i_fold_aninhado + 1}/{n_folds_aninhado} concluido "
                     f"({time.time() - t0_aninhada:.0f}s)")
        metricas_cv_aninhada = classification_metrics(
            y_int, np.argmax(Y_cv_aninhado, axis=1), np.arange(len(classes_unicas)))
        log.info(f"  CV aninhada: n_opts por fold externo = {n_opts_aninhado}"
                 f"  ({time.time() - t0_aninhada:.0f}s)")
        log.info(f"  Balanced accuracy  NAIVE (n_opt={n_opt} fixo, "
                 f"reportada ate' v1.0)      = "
                 f"{metricas_cv_naive['balanced_accuracy']:.4f}")
        log.info(f"  Balanced accuracy  ANINHADA (n_opt por fold, honesta, "
                 f"reportada a partir da v1.0) = "
                 f"{metricas_cv_aninhada['balanced_accuracy']:.4f}")
        Y_cv = Y_cv_aninhado
    else:
        log.info(f"  [AVISO] CV aninhada DESATIVADA (selecao_lv_cv_aninhada="
                 f"False) -- a metrica de CV reportada abaixo pode estar "
                 f"otimisticamente enviesada (a escolha de LVs reusou o mesmo dado, "
                 f"da avaliacao). Balanced accuracy "
                 f"naive = {metricas_cv_naive['balanced_accuracy']:.4f}.")

    pred_lab = lb.classes_[np.argmax(Y_cv, axis=1)]
    lvs_no_teto = (n_opt >= lv_cap)
    if lvs_no_teto and lv_cap < cfg.max_lvs:
        log.info(f"  [ATENCAO] LVs otimas ({n_opt}) no teto imposto pelo "
                 f"tamanho dos dados ({lv_cap}): poucas amostras para "
                 f"mais componentes -- mais amostras ajudariam.")
    elif lvs_no_teto:
        log.info(f"  [ATENCAO] LVs otimas ({n_opt}) == max_lvs ({cfg.max_lvs}): "
              f"RMSECV ainda nao atingiu plateau. Aumente max_lvs "
              f"(ex: {cfg.max_lvs + 10}) e rode novamente.")
    return (Y_cv, cv_indices, erros_rmsecv, fabrica_pipeline, grupos_cv, lv_cap, lvs_no_teto, metricas_por_lv, n_opt, pred_lab,)


def _etapa_testes_estatisticos(*, X_raw, Y_bin, Y_cv, _G, cfg, fabrica_pipeline, grupos_cv, lb, mae_id, n_opt, n_splits, objetivo, pred_lab, rotulos, usar_grupos, y_int):
    """Etapas [4/7]-[5/7]: teste de permutacao, Wold, CV-ANOVA, metricas finais de CV e IC BCa.

    Extraido de `_executar_impl` (P6, 2026-09-26) sem mudanca de logica:
    entradas e saidas explicitas; paridade bit a bit conferida."""
    perm_p: Any = None  # saida condicional (preserva o comportamento original)
    perm_res: Any = None  # saida condicional (preserva o comportamento original)
    if usar_grupos:
        # Mesmo motivo do bloco 5 (ver comentario la'): particao estavel entre
        # versoes, senao o p-valor de permutacao tambem mudaria com o sklearn.
        cv_perm = StableStratifiedGroupKFold(n_splits=max(n_splits, 2),
                                              seed=cfg.seed)
    else:
        cv_perm = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=cfg.seed)
    # Otimizacao de desempenho (auditoria jul/2026, item 8): o teste de
    # permutacao (200 refits de CV por padrao) so' alimenta o p-valor de
    # SIGNIFICANCIA DE CLASSIFICACAO no resumo — sem sentido cientifico fora
    # do objetivo Classificacao (ver should_generate/_FIG_OBJETIVOS). Pular a
    # computacao (nao so' a figura/linha do resumo) evita o refit mais caro
    # do pipeline quando o run e' Exploratorio/Quantificacao.
    if objetivo == CLASSIFICACAO:
        log.info(f"\n[4/7] Teste de permutacao (Y-randomization, "
              f"n={cfg.n_permutations})")
        perm_res = permutation_test(
            lambda: fabrica_pipeline(n_opt),
            X_raw, Y_bin, y_int, cv_perm, cfg.n_permutations, cfg.seed,
            groups=grupos_cv, n_jobs=cfg.n_jobs_permutation)
        perm_obs : float      = cast(float, perm_res["acc_observada"])
        perm_dist: np.ndarray = cast(np.ndarray, perm_res["accs_permutadas"])
        perm_p = cast(float, perm_res["p_value"])
        media_h0 = float(perm_dist.mean()) if len(perm_dist) > 0 else float("nan")
        log.info(f"  Bal.Acc observada = {perm_obs:.4f}  |  p = {perm_p:.4f}  "
              f"|  bal.acc media H0 = {media_h0:.4f}")
        log.info(f"  Iteracoes validas: {cast(int, perm_res['n_validos'])}/"
              f"{cfg.n_permutations}  "
              f"(failure_rate = {cast(float, perm_res['failure_rate']):.1%})")
    else:
        log.info(f"\n[4/7] Teste de permutacao — PULADO: objective="
              f"{OBJETIVO_ROTULO.get(objetivo, objetivo)}. Significancia de "
              f"classificacao nao e' pertinente fora do modo Classificacao "
              f"(economiza {cfg.n_permutations} refits de CV).")
        perm_res = {"acc_observada": float("nan"),
                    "accs_permutadas": np.array([], dtype=float),
                    "p_value": float("nan"), "n_validos": 0, "n_falhos": 0,
                    "failure_rate": float("nan")}
        perm_p = float("nan")

    # --- 6b. Teste de Wold (R2Y / Q2Y intercept) --------------------------
    # Gated por objetivo == CLASSIFICACAO, no MESMO padrao do teste de
    # permutacao acima (bloco 6/6a). Achado em 2026-08-06: faltava esse
    # guard aqui -- o bloco rodava incondicionalmente mesmo em objective=
    # Quantificacao/Exploratorio, refazendo n_permutations_wold refits de CV
    # usando Y_bin (rotulos de CLASSE, one-hot) para um run que nao classifica
    # nada, e escrevendo "Wold R2Y/Q2Y intercept" no resumo sem nenhum
    # sentido nesse contexto -- nem gasto de tempo silencioso, nem numero
    # espurio no relatorio sao aceitaveis em software cientifico.
    wold_res: Optional[Dict[str, object]] = None
    if cfg.run_wold and objetivo == CLASSIFICACAO:
        log.info(f"\n[4b/7] Teste de Wold (R2Y/Q2Y intercept, "
              f"n={cfg.n_permutations_wold})")
        wold_res = wold_test(
            lambda: fabrica_pipeline(n_opt),
            X_raw, Y_bin, y_int, cv_perm, cfg.n_permutations_wold, cfg.seed,
            groups=grupos_cv, n_jobs=cfg.n_jobs_permutation)
        _wr2 = cast(float, wold_res['intercept_r2'])
        _wq2 = cast(float, wold_res['intercept_q2'])
        _wr2_s = f"{_wr2:.4f}" if np.isfinite(_wr2) else "n/a (permutacoes insuficientes)"
        _wq2_s = f"{_wq2:.4f}" if np.isfinite(_wq2) else "n/a (permutacoes insuficientes)"
        log.info(f"  R2Y obs = {cast(float, wold_res['r2_obs']):.4f}  |  "
              f"intercepto = {_wr2_s}  "
              f"{'VALIDO' if wold_res['valid_r2'] else 'FALHA'} (limiar < 0.40)")
        log.info(f"  Q2Y obs = {cast(float, wold_res['q2_obs']):.4f}  |  "
              f"intercepto = {_wq2_s}  "
              f"{'VALIDO' if wold_res['valid_q2'] else 'FALHA'} (limiar < 0.05)")
    elif cfg.run_wold:
        log.info(f"\n[4b/7] Teste de Wold — PULADO: objective="
              f"{OBJETIVO_ROTULO.get(objetivo, objetivo)}. Intercepto R2Y/Q2Y "
              f"e' diagnostico de CLASSIFICACAO (usa rotulos de especie/classe "
              f"one-hot); nao pertinente fora desse mode "
              f"(economiza {cfg.n_permutations_wold} refits de CV).")

    # --- 6c. CV-ANOVA Eriksson --------------------------------------------
    # Mesmo motivo e mesmo achado do bloco 6b acima: sem o guard de objetivo,
    # rodava (barato, mas gerava numero sem sentido no resumo) mesmo fora de
    # Classificacao -- Y_bin/Y_cv sao rotulos de classe.
    cv_anova_res: Optional[Dict[str, float]] = None
    if cfg.run_cv_anova and objetivo == CLASSIFICACAO:
        cv_anova_res = cv_anova_eriksson(Y_bin, Y_cv, n_opt)
        log.info(f"\n[4c/7] CV-ANOVA (Eriksson): F = {cv_anova_res['F']:.3f}  "
              f"p = {cv_anova_res['p_value']:.4g}  "
              f"(df = {cv_anova_res['df_model']}, {cv_anova_res['df_resid']})")
    elif cfg.run_cv_anova:
        log.info(f"\n[4c/7] CV-ANOVA — PULADO: objective="
              f"{OBJETIVO_ROTULO.get(objetivo, objetivo)}. Teste de "
              f"significancia de CLASSIFICACAO (Eriksson et al. 2008); "
              f"nao pertinente fora desse mode.")

    # --- 7. Metricas e relatorio -------------------------------------------
    cm_mat = confusion_matrix(rotulos, pred_lab, labels=lb.classes_)
    metricas_finais = classification_metrics(rotulos, pred_lab, lb.classes_)
    log.info("\n[5/7] Metricas finais (CV):")
    for k, v in metricas_finais.items():
        log.info(f"  {k:>22s}: {v:.4f}")

    # --- 5b. BCa CI 95% para metricas via bootstrap estratificado ----------
    # metricas_funcoes tambem e' reaproveitado pelo bloco de holdout (8b),
    # que ja e' filtrado por should_generate(cfg,"holdout") = so' Classificacao —
    # entao o dict de lambdas (barato: so' fecha funcoes, nao executa nada)
    # pode ficar definido sempre; o CUSTO real esta no LOOP de bootstrap
    # abaixo, que e' o que a otimizacao de desempenho pula fora do objetivo
    # Classificacao (economiza n_bootstrap_bca resamples x 4 metricas).
    bca: Dict[str, Tuple[float, float, float]] = {}
    cls_arr = np.asarray(lb.classes_)
    metricas_funcoes = {
        "accuracy":          lambda yt, yp: accuracy_score(yt, yp),
        "balanced_accuracy": lambda yt, yp: balanced_accuracy_score(yt, yp),
        "f1_macro":          lambda yt, yp: f1_score(yt, yp, labels=cls_arr,
                                                       average="macro", zero_division=0),
        "cohen_kappa":       lambda yt, yp: cohen_kappa_score(yt, yp),
    }
    if objetivo != CLASSIFICACAO:
        log.info(f"\n[5b/7] BCa CI 95% — PULADO: objective="
              f"{OBJETIVO_ROTULO.get(objetivo, objetivo)}. Intervalo de "
              f"confianca de metricas de classificacao nao e' pertinente "
              f"fora do modo Classificacao.")
    else:
        log.info(f"\n[5b/7] BCa CI 95% (n_boot={cfg.n_bootstrap_bca})")
        # groups=mae_id (achado #13, rodada multiagente 2026-09-10): sem
        # isto, o bootstrap reamostrava ESPECTROS individuais -- replicas
        # fisicas (T1/T2/T3) do mesmo mae_id podiam cair em lados diferentes,
        # deixando o IC mais estreito do que a independencia real sustenta.
        # Com replicas, o IC group-aware e' mais largo: o comportamento
        # antigo subestimava a incerteza.
        for nome, fn in metricas_funcoes.items():
            lo, hi, obs = bootstrap_bca_ci(rotulos, pred_lab, fn,
                                            n_boot=cfg.n_bootstrap_bca,
                                            alpha=0.05, seed=cfg.seed,
                                            groups=_G(mae_id))
            bca[nome] = (lo, hi, obs)
            log.info(f"  {nome:>22s}: {obs:.4f}  [{lo:.4f}, {hi:.4f}]")
    log.info("\n" + str(classification_report(rotulos, pred_lab,
                                             target_names=lb.classes_,
                                             zero_division=0)))
    return (bca, cm_mat, cv_anova_res, metricas_finais, metricas_funcoes, perm_p, perm_res, wold_res,)


def _etapa_carregar_e_validar(*, cfg):
    """Etapas 1-1c2: carrega os dados (mae_id + metadados), trunca a faixa espectral, exclui classes, cria as pastas de saida, valida a integridade e roda a auditoria de delineamento.

    Extraido de `_executar_impl` (P6, 2026-09-26) sem mudanca de logica:
    entradas e saidas explicitas; paridade bit a bit conferida."""
    _a: Any = None  # saida condicional (preserva o comportamento original)
    _achados_auditoria: Any = None  # saida condicional (preserva o comportamento original)
    wavenumbers, X_raw, rotulos, conc, mae_id, metadados_df = load_data(cfg)
    X_raw   = np.asarray(X_raw,   dtype=float)
    rotulos = np.asarray(rotulos, dtype=str)
    if conc is not None:
        conc = np.asarray(conc, dtype=float)
    if mae_id is not None:
        mae_id = np.asarray(mae_id, dtype=str)

    if cfg.grouping_guarantee != "high":
        log.info(f"[WARNING] Grouping guarantee: {cfg.grouping_guarantee.upper()} "
              f"-- ver 'Grouping guarantee' no resumo/model card para o que "
              f"isso significa para a validacao desta execucao.")

    # --- 1a0. nivel N2: autenticação por espécie (DD-SIMCA one-class) ------
    # DESIGN (escolha do usuário — opção A):
    #   N1 = identificar a classe (PLS-DA multiclasse).
    #   N2 = autenticar pureza POR ESPÉCIE via DD-SIMCA one-class. Treina um
    #        modelo do "puro" para cada espécie e testa se cada amostra é
    #        pura (aceita) ou adulterada (rejeitada). É o método-padrão de
    #        autenticação one-class na literatura de espectroscopia.
    #
    # CRÍTICO: NÃO remapeamos rotulos para puro/adulterado. Os rótulos de
    # ESPÉCIE são preservados — o DD-SIMCA precisa deles para construir um
    # modelo por espécie. (O remap binário anterior fazia o DD-SIMCA treinar
    # um único modelo "puro" genérico, perdendo a autenticação por espécie.)
    # A distinção puro/adulterado vem de `conc` (0 = puro), usada dentro do
    # bloco DD-SIMCA. Não há undersampling: o DD-SIMCA precisa das amostras
    # adulteradas para medir a especificidade.
    if cfg.level == "N2":
        if conc is not None:
            n_puro = int(np.sum(np.isnan(conc) | (conc == 0.0)))
            n_adul = int(np.sum(~(np.isnan(conc) | (conc == 0.0))))
            log.info(f"[INFO] N2 (autenticação por espécie): rótulos de espécie "
                  f"preservados para DD-SIMCA one-class "
                  f"(puros={n_puro} | adulterados={n_adul}). "
                  f"DD-SIMCA forçado para mode 'puros'.")
            # Force per-species one-class authentication for N2
            cfg.ddsimca_train_on = "puros"
            cfg.run_ddsimca = True
        else:
            log.info("[AVISO] nivel N2 sem dados de concentração (##TITLE= sem "
                  "adulterante). Não é possível separar puro/adulterado — "
                  "verifique os arquivos .dx.")

    # --- 1a. Truncamento espectral: remove ruido de borda da FFT ----------
    # SG derivativo amplifica os ultimos pontos da FFT (proximos a 0 cm-1
    # e ao final do interferograma). Sem truncar, esses pontos viram falsos
    # top-VIP com estabilidade 100% (artefato, nao quimica).
    n_orig = X_raw.shape[1]
    _eixo_orig = (float(wavenumbers.min()), float(wavenumbers.max()))
    mask_wn = (wavenumbers >= cfg.wn_min) & (wavenumbers <= cfg.wn_max)
    if cfg.mode == "multielementar":
        # tabela amostra x elemento: o "eixo" e' so' o indice do elemento;
        # faixa espectral nao se aplica (nada e' truncado).
        mask_wn = np.ones_like(wavenumbers, dtype=bool)
    if not mask_wn.any():
        raise ValueError(
            f"Nenhuma variavel sobrevive ao filtro "
            f"[{cfg.wn_min}, {cfg.wn_max}] {unidade_do_eixo(cfg) or 'cm-1'}. "
            f"Faixa real dos dados: [{_eixo_orig[0]:.1f}, {_eixo_orig[1]:.1f}]. "
            f"Ajuste faixa_min_cm/faixa_max_cm (ou wn_min/wn_max) para "
            f"cobrir o eixo dos dados.")
    n_removidas = n_orig - int(mask_wn.sum())
    X_raw       = X_raw[:, mask_wn]
    _wn_dados   = wavenumbers
    wavenumbers = wavenumbers[mask_wn]
    _un_faixa = unidade_do_eixo(cfg)
    log.info(f"[INFO] Truncamento espectral [{cfg.wn_min:.0f}, {cfg.wn_max:.0f}]"
          f"{' ' + _un_faixa if _un_faixa else ''}: {n_removidas}/{n_orig} variaveis removidas, "
          f"restam {len(wavenumbers)} ({wavenumbers.min():.1f}-"
          f"{wavenumbers.max():.1f}).")
    _aviso_corte = aviso_de_corte_de_faixa(
        n_orig, n_removidas, (cfg.wn_min, cfg.wn_max), _eixo_orig, _un_faixa)
    if _aviso_corte:
        log.warning(f"[AVISO] {_aviso_corte}")

    # --- 1a1. Faixas excluidas (agua/TSP em RMN...) e normalizacao por espectro -------
    # Logo depois do corte de faixa e ANTES de qualquer pre-processamento (SNV/PQN dividem
    # pelo proprio espectro: uma regiao dominada por agua residual domina o divisor).
    X_raw, wavenumbers, _info_eixo = preparar_eixo_espectral(cfg, X_raw, _wn_dados, wavenumbers)
    cfg.exclusion_info = _info_eixo   # atributo de execucao (nao e' campo da Config nem chave do YAML)
    if _info_eixo["faixas"]:
        log.info(f"[INFO] Faixas excluidas {_info_eixo['faixas']}: {_info_eixo['n_removidas']} variaveis "
                 f"removidas ({_info_eixo['fracao'] * 100:.0f}%), restam {len(wavenumbers)}.")
    if _info_eixo["normalizacao"] != "nenhuma":
        log.info(f"[INFO] Normalizacao por espectro: {_info_eixo['normalizacao']}"
                 + (f" (janela {_info_eixo['janela_rip']})" if _info_eixo["janela_rip"] else "") + ".")
    for _av_eixo in _info_eixo["avisos"]:
        log.warning(f"[AVISO] {_av_eixo}")

    # --- 1a1b. Saturacao do detector (teto de absorbancia) -------------------
    # Variavel/espectro no teto nao carrega informacao da amostra. `saturacao`: auto = remove e
    # registra (saturacao.json + resumo); avisar = so' avisa; nenhuma = nem procura.
    cfg.saturacao_info = {}
    _modo_sat = str(getattr(cfg, "saturacao", "auto") or "auto")
    if _modo_sat != "nenhuma" and cfg.mode in MODOS_COM_SATURACAO and cfg.mode != "multielementar":
        _sat = detectar_saturacao(X_raw, wavenumbers)
        if _sat["ativo"] and (_sat["variaveis_removidas"] or _sat["espectros_removidos"]):
            _un_sat = unidade_do_eixo(cfg) or "cm-1"
            _regs = ", ".join(f"{a:.0f}-{b:.0f}" for a, b in _sat["regioes"][:8])
            _aplicar_sat = _modo_sat == "auto"
            log.warning(f"[AVISO] Saturacao do detector (teto {_sat['teto']:.3g}): "
                        f"{_sat['variaveis_removidas']} variaveis ({_regs} {_un_sat}) e "
                        f"{_sat['espectros_removidos']} espectro(s) no teto."
                        + ("" if _aplicar_sat else "saturacao=avisar: nada removido; "
                           "reduza faixa_min_cm/faixa_max_cm ou refaca as medidas."))
            cfg.saturacao_info = {
                "teto": _sat["teto"], "aplicado": _aplicar_sat,
                "variaveis_removidas": _sat["variaveis_removidas"],
                "espectros_removidos": _sat["espectros_removidos"],
                "regioes": [[a, b] for a, b in _sat["regioes"]],
                "unidade_eixo": _un_sat,
                "espectros_removidos_ids": ([str(mae_id[i]) if mae_id is not None else int(i)
                                             for i in np.where(~_sat["manter_espectros"])[0][:200]]),
            }
            if _aplicar_sat:
                _mv, _me = _sat["manter_variaveis"], _sat["manter_espectros"]
                X_raw = X_raw[_me][:, _mv]
                wavenumbers = wavenumbers[_mv]
                rotulos = rotulos[_me]
                conc = conc[_me] if conc is not None else None
                mae_id = mae_id[_me] if mae_id is not None else None
                if metadados_df is not None and len(metadados_df) == len(_me):
                    metadados_df = metadados_df.loc[_me].reset_index(drop=True)
                if _sat["espectros_removidos"]:
                    log.warning(f"[AVISO] {_sat['espectros_removidos']} espectro(s) saturado(s) fora do "
                                f"modelo (ver saturacao.json em Relatorios). Refaca essas medidas "
                                f"com menos caminho optico ou diluicao.")

    # --- 1a2. Exclusao de classes (ex: uma classe com lote anomalo) ----------
    if cfg.exclude_classes:
        excl = set(str(c) for c in cfg.exclude_classes)
        mask_keep = ~np.isin(rotulos, list(excl))
        n_rem = int((~mask_keep).sum())
        if n_rem > 0:
            log.info(f"[INFO] Excluindo classes {sorted(excl)}: "
                  f"{n_rem} amostras removidas.")
            X_raw   = X_raw[mask_keep]
            rotulos = rotulos[mask_keep]
            conc    = conc[mask_keep] if conc is not None else None
            mae_id  = mae_id[mask_keep] if mae_id is not None else None

    # --- 1b. Pasta de saida descritiva -------------------------------------
    # Layout (auditoria jul/2026, item 4): output_root_folder/Amostra/Modo/
    # Execucao/{Graficos,Tabelas,Relatorios,Modelos} — separa fisicamente os
    # resultados por objetivo cientifico, alem do gating de conteudo (ver
    # modos_analise.py) que ja impede a figura errada de ser GERADA.
    cfg.output_folder = generate_output_name(cfg, len(np.unique(rotulos)),
                                         X_raw.shape[0])
    pasta = cfg.output_folder
    pasta_dados   = pasta_execucao(pasta, NOME_TABELAS)
    pasta_modelos = pasta_execucao(pasta, NOME_MODELOS)
    pasta_logs    = pasta_execucao(pasta, NOME_RELATORIOS)
    # M-14: no Windows o caminho total nao pode passar de 260 caracteres. Antes
    # o pipeline morria em `os.makedirs` com "WinError 206" (e `guaraci demo`/o
    # app web mostravam o erro cru). Confere ANTES de criar e diz o que fazer.
    _confere_comprimento_de_saida(pasta)
    for _p in (pasta, pasta_dados, pasta_execucao(pasta, NOME_GRAFICOS),
               pasta_modelos, pasta_logs):
        try:
            os.makedirs(_p, exist_ok=True)
        except OSError as _e_mk:
            raise ValueError(
                f"Nao foi possivel criar a pasta de saida ({type(_e_mk).__name__}: "
                f"{_e_mk}). Se o caminho e' longo (limite do Windows: 260 "
                f"caracteres), escolha uma pasta_saida mais curta, ex.: "
                f"C:/guaraci_saida.") from _e_mk
    _marcar_execucao_incompleta(pasta)
    try:
        from guaraci import contexto as _contexto
        _contexto.gravar_efetiva(cfg, pasta, pasta_logs)
    except OSError as _e_ef:
        registrar_falha_salvamento(f"config_efetiva ({_e_ef})")
    if getattr(cfg, "saturacao_info", None):
        try:
            import json as _json
            with open(os.path.join(pasta_logs, "saturacao.json"), "w", encoding="utf-8") as _fh_sat:
                _json.dump(cfg.saturacao_info, _fh_sat, ensure_ascii=False, indent=2, default=str)
        except OSError as _e_sat:
            registrar_falha_salvamento(f"saturacao.json ({_e_sat})")
    log.info(f"[INFO] Saida: {pasta}")
    log.info("[INFO] Subpastas: " + " ".join(
        os.path.basename(pasta_execucao(pasta, c)) + "/"
        for c in (NOME_GRAFICOS, NOME_TABELAS, NOME_MODELOS, NOME_RELATORIOS)))
    if metadados_df is not None:
        cam_meta = os.path.join(pasta_dados, "metadados.csv")
        # Sanitizado antes de tocar o disco: o identificador da amostra de
        # origem (title/arquivo/cod/data/mae_id) fica em memoria, onde o
        # pipeline precisa dele, e nunca vai para um arquivo que depois
        # viaja junto com os resultados. `grupo_replica` preserva o
        # agrupamento de forma anonima. Ver dados_io.sanitizar_metadados.
        sanitizar_metadados(metadados_df).to_csv(
            cam_meta, index=False, sep=";", decimal=",")
        log.info(f"[INFO] Metadados salvos: {cam_meta}")
    cfg.multielementar_resumo = {}
    if cfg.mode == "multielementar" and metadados_df is not None             and "multielementar" in metadados_df.attrs:
        from guaraci.icp_multielementar import relatorio_multielementar
        cfg.multielementar_resumo = relatorio_multielementar(
            metadados_df.attrs["multielementar"], pasta_dados, pasta_logs,
            pasta_execucao(pasta, NOME_GRAFICOS))
        _ra = cfg.multielementar_resumo["anova"]
        log.info(f"[INFO] Multielementar: ANOVA por elemento ({_ra['correcao']}); "
                 f"{len(_ra['elementos_com_diferenca_entre_grupos'])} elemento(s) com "
                 f"diferenca entre grupos (FDR < {_ra['alpha']}); sensibilidade da "
                 f"censura -- muda: {cfg.multielementar_resumo['sensibilidade_censura']['muda']}")

    # --- 1c. Input integrity validation -----------------------------------
    log.info("\n[0/7] Input integrity validation...")
    X_raw, wavenumbers, rotulos, conc, mae_id, relatorio_entrada = validate_input(
        X_raw, wavenumbers, rotulos, conc, mae_id)
    relatorio_balanco = _check_balance(rotulos)

    # --- 1c2. Auditoria de delineamento (Bloco 11) -- roda por padrao,
    # nunca opt-in (mesmo espirito do grouping_guarantee do Bloco 8).
    # Calculada aqui (dados ja validados/limpos) e anexada ao `resumo`
    # mais abaixo, para o model card renderizar uma secao propria.
    from guaraci.auditoria_delineamento import resolver_unidade_agrupamento
    _unidade_grupo, _motivo_unidade = resolver_unidade_agrupamento(
        rotulos, mae_id, str(getattr(cfg, "group_unit", "auto") or "auto"),
        str(getattr(cfg, "grouping_guarantee", "high")))
    if _motivo_unidade:
        log.info(f"[INFO] Unidade de agrupamento '{_unidade_grupo}': {_motivo_unidade}")
    try:
        from guaraci.auditoria_delineamento import run_audit as _rodar_auditoria
        _achados_auditoria = _rodar_auditoria(
            X_raw, wavenumbers, rotulos, cfg, conc, mae_id, unidade=_unidade_grupo)
        _n_criticos_audit = sum(1 for a in _achados_auditoria
                                 if a.severidade == "critico")
        _n_avisos_audit = sum(1 for a in _achados_auditoria
                               if a.severidade == "aviso")
        log.info(f"[INFO] Auditoria de delineamento: {_n_criticos_audit} "
              f"critico(s), {_n_avisos_audit} aviso(s) -- ver model card "
              f"para o detalhe por checagem.")
        for _a in _achados_auditoria:
            if _a.severidade == "critico":
                log.warning(f"  [AUDITORIA][CRITICO] {_a.nome}: {_a.mensagem}")
            elif _a.severidade == "aviso":
                log.info(f"  [AUDITORIA][AVISO] {_a.nome}: {_a.mensagem}")
    except Exception as _e_audit:  # noqa: BLE001 -- auditoria e' diagnostico
        # opcional; erro impresso, resto da execucao continua normalmente.
        log.info(f"  [AVISO] Auditoria de delineamento falhou: {_e_audit}")
        _achados_auditoria = []

    # B1: mae_id is now synchronized INSIDE validate_input (same NaN/Inf
    # removal mask). Group-aware validation survives removals —
    # no more silent GroupKFold disabling due to a single NaN.
    if mae_id is not None:
        mae_id = np.asarray(mae_id, dtype=str)

    # --- Unidade independente da validacao (B4c) -----------------------------
    # `mae_id` continua sendo a identidade da amostra (parse do adulterante,
    # metadados); a chave que os SPLITTERS usam vem de `_G(...)`: o proprio
    # mae_id (padrao) ou a sessao de coleta (`group_unit="sessao"`).
    def _G(m):
        return chave_unidade_validacao(m, _unidade_grupo)

    # --- Validation strategy: group-aware if mae_id available -------------
    usar_grupos = (cfg.group_by_mae_id and mae_id is not None
                   and len(np.unique(_G(mae_id))) >= 3)
    if cfg.group_by_mae_id and not usar_grupos:
        log.info("[INFO] GroupKFold desabilitado: mae_id indisponivel ou "
              "grupos insuficientes — usando StratifiedKFold (estratificada).")
    if usar_grupos and mae_id is not None:
        n_grupos = int(len(np.unique(_G(mae_id))))
        log.info(f"[INFO] Validacao group-aware ATIVA: {n_grupos} grupos "
              f"unicos via {'sessao de coleta' if _unidade_grupo == 'sessao' else 'mae_id'}.")
    return (X_raw, _G, _a, _achados_auditoria, _eixo_orig, _unidade_grupo, conc, mae_id, n_orig, pasta, pasta_dados, pasta_logs, pasta_modelos, relatorio_balanco, relatorio_entrada, rotulos, usar_grupos, wavenumbers,)


def _executar_impl(cfg: Config):
    from guaraci.log import configurar as _configurar_log
    # Chamado aqui (nao em nivel de modulo) para rodar DENTRO de qualquer
    # contextlib.redirect_stdout ja ativo no chamador (CLI/app web) --
    # ver guaraci/log.py:_StdoutHandler.
    _configurar_log()
    setup_matplotlib(cfg)
    # Zera o acumulador de falhas de salvamento (figuras/manifesto) desta
    # corrida -- ver resumo final, achado de auditoria sobre falha
    # silenciosa em pasta de saida profunda (MAX_PATH do Windows).
    resetar_falhas_salvamento()

    # --- 0a. Objetivo cientifico do run (Exploratorio/Classificacao/
    # Quantificacao). Decide quais figuras/relatorios serao gerados, para
    # que cada mode produza EXCLUSIVAMENTE o que e' pertinente ao seu
    # objetivo (ver modos_analise.py). Preserva N1/N2/N3 quando objective=auto.
    objetivo = resolve_objective(cfg)
    _fig_explor_on = exploratory_figures_enabled(cfg)
    log.info(f"\n[MODO] Objetivo cientifico: "
             f"{OBJETIVO_ROTULO.get(objetivo, objetivo)}  "
             f"(nivel {cfg.level}, objetivo configurado: {cfg.objective})")
    _plano = describe_plan(cfg)
    if _plano:
        log.info(f"[MODO] Figuras pertinentes a este objetivo ({len(_plano)}): "
              + "; ".join(_plano))

    # Aviso de PROTOTIPO (achado B4-1): o mode imagem (colorimetria digital)
    # nunca foi validado com dataset real, e -- pior -- `dados_imagem` devolve
    # mae_id=None sempre, o que desliga a validacao group-aware (o diferencial
    # central do projeto) sem que nada no relatorio dissesse isso. O aviso vai
    # como WARNING (nao INFO), e o mode tambem entra no resumo_modelo.txt para
    # que os geradores de relatorio possam carimbar a saida.
    if cfg.mode == "imagem":
        log.warning(
            "[PROTOTIPO] mode='imagem' (colorimetria digital) NAO e validado "
            "com dataset real. A validacao so' e' agrupada por amostra fisica "
            "com subpasta por amostra ou amostras.csv (ver 'Grouping "
            "guarantee' abaixo); sem isso os relatorios sairao carimbados.")

    # --- 0. Hardware probe + auto-ajuste preventivo -------------------------
    _hw = hardware_probe()
    _avisos_hw = auto_adjust_hardware_config(cfg, _hw)
    if _hw.get("ram_total_gb") is None or _hw.get("ram_livre_gb") is None:
        log.info(f"[HARDWARE] RAM nao medida (nenhuma fonte a le); os extras "
                 f"pedidos nao foram ajustados  "
                 f"CPU: {_hw['cpu_fisicos']}f/{_hw['cpu_logicos']}l")
    else:
        log.info(f"[HARDWARE] RAM total: {_hw['ram_total_gb']:.1f} GB  "
                 f"livre: {_hw['ram_livre_gb']:.1f} GB  "
                 f"CPU: {_hw['cpu_fisicos']}f/{_hw['cpu_logicos']}l")
    for _av in _avisos_hw:
        log.warning(f"  [AUTO-AJUSTE DE EXTRA POR RAM] {_av}  (registrado no "
                    f"model card; nenhum parametro cientifico e' alterado)")

    # --- 0z. Perfil da matriz ---------------------------------------------
    # ANTES do carregamento: o perfil define a faixa do eixo espectral, que
    # e' aplicada na leitura. Um nome desconhecido estoura aqui, com a lista
    # de perfis existentes -- nunca cai num padrao de outra matriz, porque
    # isso produziria faixa e vocabulario errados sem nenhum sinal.
    perfil = cfg_profile(cfg)
    cfg = apply_profile(cfg, perfil)

    # --- 1. Carregamento (6-tupla com mae_id + metadados) ------------------
    X_raw, _G, _a, _achados_auditoria, _eixo_orig, _unidade_grupo, conc, mae_id, n_orig, pasta, pasta_dados, pasta_logs, pasta_modelos, relatorio_balanco, relatorio_entrada, rotulos, usar_grupos, wavenumbers = _etapa_carregar_e_validar(cfg=cfg)
    if getattr(cfg, "group_unit", None) != _unidade_grupo:
        # "auto" ja' resolvido: o resto da execucao (matriz especie x adulterante etc.) ve' a unidade real
        cfg = copy.copy(cfg)             # copia rasa: atributos soltos (nao-campos) da config continuam
        cfg.group_unit = _unidade_grupo

    # --- 1d. Hold-out externo (group-aware se possivel) --------------------
    X_holdout, X_raw, Y_bin, _classes_fora_holdout, _conc_holdout, _mae_id_holdout, classes_unicas, conc, cv, cv_label, lb, mae_id, mapa_cores, mapa_marcadores, n_holdout, n_splits, rotulos, rotulos_holdout, y_int = _etapa_holdout_e_estrategia_cv(X_raw=X_raw, _G=_G, _eixo_orig=_eixo_orig, cfg=cfg, conc=conc, mae_id=mae_id, n_orig=n_orig, relatorio_balanco=relatorio_balanco, rotulos=rotulos, usar_grupos=usar_grupos)

    # --- 2. Pre-processamento (uma vez, para visualizacao e PCA) -----------
    log.info(f"\n[1/7] Pre-processamento (preset='{cfg.default_preprocessing}')")
    preproc_full = build_preprocessor(cfg).fit(X_raw)
    X_processed  = np.asarray(preproc_full.transform(X_raw), dtype=float)

    # Diagnostico de faixa espectral (2026-08-07): avisa quando a faixa
    # configurada inclui regiao sem sinal analitico. Rodar com faixa larga
    # demais nao e' inofensivo -- infla o n de variaveis, dilui VIP/SR,
    # encarece a CV e da' ao modelo espaco para ajustar ruido. E' um AVISO,
    # nunca um corte automatico: mudar a faixa muda o resultado, e essa
    # decisao e' do usuario.
    # cast: o retorno e' Dict[str, object] (valores heterogeneos por chave);
    # os tipos concretos sao garantidos na construcao do dict.
    diag_faixa = cast(Dict[str, Any],
                      diagnose_spectral_range(X_processed, wavenumbers))
    if diag_faixa.get("faixa_sugerida") and diag_faixa["frac_util"] < 0.95:
        _fu = float(diag_faixa["frac_util"])
        _sug = diag_faixa["faixa_sugerida"]
        log.info(f"  [AVISO] Faixa espectral: so' {_fu * 100:.0f}% das "
                 f"{X_processed.shape[1]} variaveis carregam sinal (SNR>=3).")
        for _a, _b, _t in diag_faixa["regioes_ruins"]:
            log.info(f"          regiao {_t}: {_a:.0f}-{_b:.0f} {unidade_do_eixo(cfg) or 'cm-1'}")
        log.info(f"          faixa com sinal: [{_sug[0]:.0f}, {_sug[1]:.0f}] "
                 f"{unidade_do_eixo(cfg) or 'cm-1'} (atual: [{cfg.wn_min:.0f}, {cfg.wn_max:.0f}])")
        log.info("          Considere reduzir faixa_min_cm/faixa_max_cm e "
                 "reexecutar; compare Q2 antes de adotar.")

    # --- 3. LV selection by CV (no leakage, group-aware if possible) -------
    Y_cv, cv_indices, erros_rmsecv, fabrica_pipeline, grupos_cv, lv_cap, lvs_no_teto, metricas_por_lv, n_opt, pred_lab = _etapa_selecao_lvs(X_raw=X_raw, Y_bin=Y_bin, _G=_G, cfg=cfg, classes_unicas=classes_unicas, cv=cv, cv_label=cv_label, lb=lb, mae_id=mae_id, n_splits=n_splits, usar_grupos=usar_grupos, y_int=y_int)

    # --- 4. Modelo final (para VIP, scores, T2/Q) --------------------------
    pls_final = _novo_pls_classificador(cfg, n_opt)
    pls_final.fit(X_processed, Y_bin)
    T_pls = np.asarray(pls_final.x_scores_,  dtype=float)
    P_pls = np.asarray(pls_final.x_loadings_, dtype=float).T
    var_lv_pls = explained_variance(X_processed, T_pls)
    vip = vip_scores(pls_final)

    # --- 5. PCA exploratoria -----------------------------------------------
    log.info("\n[3/7] PCA exploratoria")
    pca = PCA(n_components=min(cfg.n_pcs_pca, X_processed.shape[1],
                                X_processed.shape[0]))
    scores_pca = pca.fit_transform(X_processed)
    var_pca = pca.explained_variance_ratio_ * 100
    log.info(f"  PC1: {var_pca[0]:.2f}%  PC2: {var_pca[1]:.2f}%  "
          f"acumulado(2): {sum(var_pca[:2]):.2f}%")

    # --- 6. Permutation test (Y-randomization) ----------------------------
    # cv_perm e' construido sempre (barato, so' um objeto de split) pois o
    # teste de Wold (bloco 6b, opt-in) o reaproveita mesmo quando o teste de
    # permutacao abaixo e' pulado.
    bca, cm_mat, cv_anova_res, metricas_finais, metricas_funcoes, perm_p, perm_res, wold_res = _etapa_testes_estatisticos(X_raw=X_raw, Y_bin=Y_bin, Y_cv=Y_cv, _G=_G, cfg=cfg, fabrica_pipeline=fabrica_pipeline, grupos_cv=grupos_cv, lb=lb, mae_id=mae_id, n_opt=n_opt, n_splits=n_splits, objetivo=objetivo, pred_lab=pred_lab, rotulos=rotulos, usar_grupos=usar_grupos, y_int=y_int)

    # --- 7b. R²X, R²Y, Q² ---------------------------------------------------
    r2x, r2y, q2 = pls_model_metrics(pls_final, X_processed, Y_bin, Y_cv)
    log.info(f"\n[5b/7] R2X = {r2x:.4f}  |  R2Y = {r2y:.4f}  |  Q2 = {q2:.4f}")

    # --- 8. Figuras --------------------------------------------------------
    Q, T2, _dmodx_res, _martens_n_folds, _martens_n_sig, _opls_n_ortho, aucs_roc, ddsimca_pcv_esp, ddsimca_res, ddsimca_sens_esp, etapa4_res, modo_dd, out_q, out_t2, q_lim, simca_pred, t2_lim = _etapa_diagnosticos_e_figuras(P_pls=P_pls, T_pls=T_pls, X_processed=X_processed, X_raw=X_raw, Y_bin=Y_bin, Y_cv=Y_cv, _G=_G, _fig_explor_on=_fig_explor_on, cfg=cfg, classes_unicas=classes_unicas, cm_mat=cm_mat, conc=conc, cv_indices=cv_indices, erros_rmsecv=erros_rmsecv, grupos_cv=grupos_cv, lb=lb, lv_cap=lv_cap, mae_id=mae_id, mapa_cores=mapa_cores, mapa_marcadores=mapa_marcadores, metricas_por_lv=metricas_por_lv, n_opt=n_opt, pasta=pasta, pasta_dados=pasta_dados, pca=pca, pls_final=pls_final, pred_lab=pred_lab, rotulos=rotulos, scores_pca=scores_pca, var_lv_pls=var_lv_pls, var_pca=var_pca, vip=vip, wavenumbers=wavenumbers, wold_res=wold_res, y_int=y_int)

    # --- 8b. Avaliacao em holdout independente ----------------------------
    bca_holdout, conformal_classificacao, metricas_holdout = _etapa_avaliar_holdout(X_holdout=X_holdout, _mae_id_holdout=_mae_id_holdout, cfg=cfg, lb=lb, metricas_finais=metricas_finais, metricas_funcoes=metricas_funcoes, n_holdout=n_holdout, pasta=pasta, pls_final=pls_final, preproc_full=preproc_full, rotulos_holdout=rotulos_holdout)

    # --- 9. Identificadores e resumo (separados dos graficos) --------------
    resumo = _etapa_resumo_e_relatorios(Q=Q, T2=T2, T_pls=T_pls, X_holdout=X_holdout, X_raw=X_raw, _G=_G, _achados_auditoria=locals().get('_achados_auditoria'), _classes_fora_holdout=_classes_fora_holdout, _dmodx_res=_dmodx_res, _hw=_hw, _martens_n_folds=_martens_n_folds, _martens_n_sig=_martens_n_sig, _opls_n_ortho=_opls_n_ortho, _unidade_grupo=_unidade_grupo, aucs_roc=aucs_roc, bca=bca, bca_holdout=bca_holdout, cfg=cfg, classes_unicas=classes_unicas, conformal_classificacao=conformal_classificacao, cv_anova_res=cv_anova_res, cv_label=locals().get('cv_label'), ddsimca_pcv_esp=ddsimca_pcv_esp, ddsimca_res=ddsimca_res, ddsimca_sens_esp=ddsimca_sens_esp, diag_faixa=diag_faixa, etapa4_res=etapa4_res, lb=lb, lvs_no_teto=lvs_no_teto, mae_id=mae_id, metricas_finais=metricas_finais, metricas_holdout=metricas_holdout, modo_dd=modo_dd, n_holdout=n_holdout, n_opt=n_opt, n_splits=n_splits, objetivo=objetivo, out_q=out_q, out_t2=out_t2, pasta_dados=pasta_dados, pasta_logs=pasta_logs, perm_p=locals().get('perm_p'), perm_res=locals().get('perm_res'), pred_lab=pred_lab, q2=q2, q_lim=q_lim, r2x=r2x, r2y=r2y, relatorio_balanco=relatorio_balanco, relatorio_entrada=relatorio_entrada, rotulos=rotulos, simca_pred=simca_pred, t2_lim=t2_lim, usar_grupos=usar_grupos, wold_res=wold_res)

    # --- 9-aviso. Classes em rampa (R14) e registro do alinhamento COW (R9) ---
    _avisar_classes_em_rampa(cfg=cfg, T_pls=T_pls, rotulos=rotulos, pred_lab=pred_lab, X_raw=X_raw,
                             Y_bin=Y_bin, y_int=y_int, cv_indices=cv_indices, lv_cap=lv_cap, n_opt=n_opt,
                             metricas_por_lv=metricas_por_lv, pasta_logs=pasta_logs)
    _registrar_alinhamento(preproc_full, mae_id, pasta_logs, anexar=True)

    # --- 9a. Auto-Benchmark (opcional) ─────────────────────────────────────
    _etapa_benchmark_monte_carlo(X_holdout=X_holdout, X_raw=X_raw, cfg=cfg, grupos_cv=grupos_cv, lb=lb, n_opt=n_opt, pasta=pasta, rotulos_holdout=rotulos_holdout, wavenumbers=wavenumbers, y_int=y_int)

    # --- 9b. Exportar modelo final (modelos/) — joblib opcional -----------
    cam_modelo, joblib, pacote_modelo = _etapa_exportar_modelo(Q=Q, T2=T2, X_processed=X_processed, X_raw=X_raw, _G=_G, cfg=cfg, conc=conc, conformal_classificacao=conformal_classificacao, grupos_cv=grupos_cv, lb=lb, mae_id=mae_id, n_opt=n_opt, out_q=out_q, out_t2=out_t2, pasta_logs=pasta_logs, pasta_modelos=pasta_modelos, pca=pca, perfil=perfil, pls_final=pls_final, q_lim=q_lim, resumo=resumo, rotulos=rotulos, t2_lim=t2_lim, usar_grupos=usar_grupos, wavenumbers=wavenumbers)

    # --- 9. PLS regression (optional) -------------------------------------
    # Guard: only runs if (1) concentration data present, (2) no NaN,
    # (3) variance > 0, (4) at least 10 samples with concentration > 0,
    # (5) nivel is N2 or N3 — in N1 the inter-species spectral variation
    #     (~90 % of variance) completely overwhelms the adulteration signal,
    #     producing R²≈0 and a "predict-the-mean" model.
    # (6) Single-species data (mae_id prefix check) — multi-species N2 is
    #     also confounded; the model sees species identity as a stronger
    #     signal than adulteration level.
    _etapa_regressao_pls(X_holdout=X_holdout, X_raw=X_raw, _G=_G, _conc_holdout=_conc_holdout, cam_modelo=locals().get('cam_modelo'), cfg=cfg, classes_unicas=classes_unicas, conc=conc, joblib=locals().get('joblib'), lb=lb, mae_id=mae_id, n_splits=n_splits, objetivo=objetivo, pacote_modelo=locals().get('pacote_modelo'), pasta=pasta, pasta_logs=pasta_logs, perfil=perfil, pls_final=pls_final, pred_lab=pred_lab, preproc_full=preproc_full, rotulos=rotulos, rotulos_holdout=rotulos_holdout, wavenumbers=wavenumbers)

    _concluir_execucao(pasta)
    _falhas_salvamento = obter_falhas_salvamento()
    log.info(f"\n{'=' * 60}")
    if _falhas_salvamento:
        # Achado de auditoria funcional (2026-09-01): em pasta de saida
        # muito profunda (MAX_PATH do Windows, 260 chars), ate 3 arquivos
        # falhavam ao salvar e o pipeline terminava dizendo "concluido"
        # do mesmo jeito -- um "[ERROR]" perdido em centenas de linhas de
        # log e' facil de nao notar. Nao falha a corrida (o resto do
        # resultado continua valido e usavel), mas o resumo final NUNCA
        # pode dizer "concluido" sem ressalva quando algo nao foi salvo.
        log.warning(f"  Pipeline concluido COM {len(_falhas_salvamento)} "
                    "FALHA(S) DE SALVAMENTO:")
        for _f in _falhas_salvamento:
            log.warning(f"    - {_f}")
        log.warning("  O resto do resultado desta corrida e' valido, mas "
                    "os arquivos acima NAO foram gravados -- confira "
                    "espaco em disco/permissao, e se o caminho de saida "
                    "nao excede o limite do sistema operacional "
                    "(Windows: 260 caracteres sem prefixo \\\\?\\).")
    else:
        log.info("  Pipeline concluido.")
    log.info(f"  Resultados em: {pasta}")
    if getattr(cfg, "grouping_guarantee", "high") in ("none", "medium"):
        # Ressalva PERSISTENTE (B4b): repetida no fim, onde o usuario olha,
        # nao so' no log do inicio.
        if cfg.grouping_guarantee == "none":
            log.warning(
                "  [!] VALIDACAO NAO AGRUPADA: sem chave de grupo "
                "(coluna_grupo/arquivo_grupos), a CV usou StratifiedKFold e "
                "replicas da mesma amostra podem estar nos dois lados. Trate "
                "as metricas como EXPLORATORIAS.")
        else:
            log.warning(
                "  [!] Validacao agrupada pelo grupo DECLARADO pelo usuario "
                "(garantia MEDIA): o software nao verifica que os grupos "
                "sao de fato independentes.")
    log.info(f"{'=' * 60}")


@functools.wraps(_executar_impl)
def executar(cfg: Config) -> Any:
    """Executa o pipeline completo (ver `_executar_impl` para as etapas). O
    backend Agg e os filtros de warnings valem so' durante a chamada."""
    from guaraci import contexto as _contexto
    res = _contexto.aplicar(cfg)
    if res.erros:
        raise ValueError("Configuracao recusada: " + " | ".join(res.erros))
    try:
        with _ambiente_da_execucao():
            for linha in res.linhas():
                log.info(f"[CONTEXTO] {linha}")
            if getattr(cfg, "mode", "") == "hsi":
                # cubos ENVI: fluxo proprio (segmentacao, pixel -> objeto), o mesmo da tela [X]
                from guaraci.hsi_pipeline import gravar_relatorio_hsi, run_hsi_pipeline
                resumo = run_hsi_pipeline(cfg)
                log.info(f"[HSI] relatorio: {gravar_relatorio_hsi(cfg, resumo)}")
                return resumo
            return _executar_impl(cfg)
    finally:
        # A execucao usou a configuracao efetiva (gravada nos resultados); o `cfg`
        # do usuario volta aos valores que ele escolheu.
        _contexto.desfazer(cfg, manter_registro=True)


# =========================================================================
#  CAMADA ACESSIVEL (v23) — configuracao em YAML + assistente de terminal
#  Objetivo: usar o pipeline SEM editar o codigo. Toda a configuracao de
#  usuario fica em config.yaml (linguagem simples). O assistente CMD
#  (_menu_interativo) le/edita/salva esse arquivo e dispara executar().
#  Fonte UNICA de verdade: _CONFIG_SPEC mapeia campo amigavel <-> Config,
#  e alimenta tanto o YAML quanto o menu (e, depois, o app web).
# =========================================================================

# Config IO / _CONFIG_SPEC extraidos para config_io.py (dividida tecnica pos-
# auditoria). Reexportados aqui: o menu de terminal legado (_menu_interativo/
# _editar_campo, abaixo), executar() e os consumidores externos (guaraci.py,
# app, testes) usam estes nomes via `pipeline.X`.
from guaraci.config_io import (   # noqa: F401
    _PRE_PROC_FRIENDLY, _PRE_PROC_INV, _CONFIG_SPEC,
    _attr_para_yaml, _checar_faixa, _coagir_valor, _validar_semantico,
    _fmt_yaml, save_config, load_config, _validar_pasta_dados,
    CONFIG_SPEC, attr_para_yaml, coagir_valor, validar_semantico, validar_fonte_de_dados,
)

def _editar_campo(cfg: Config, s: Dict[str, Any]) -> None:
    """Edita um campo via terminal, com validacao."""
    log.info(f"\n  {s['desc']}")
    log.info(f"  valor atual: {_fmt_yaml(_attr_para_yaml(s, cfg))}")
    if s["tipo"] == "bool":
        setattr(cfg, s["attr"], not bool(getattr(cfg, s["attr"])))
        log.info(f"  -> alterado para {getattr(cfg, s['attr'])}")
        return
    if s["tipo"] in ("choice", "preproc"):
        ops = s["opcoes"] or []
        for i, o in enumerate(ops, 1):
            log.info(f"    ({i}) {o}")
        r = input("  escolha o numero (Enter cancela): ").strip()
        if r.isdigit() and 1 <= int(r) <= len(ops):
            try:
                setattr(cfg, s["attr"], _coagir_valor(s, ops[int(r) - 1]))
                log.info("  -> ok")
            except ValueError as e:   # _coagir_valor: opcao/faixa invalida
                log.info(f"  erro: {e}")
        else:
            log.info("  cancelado.")
        return
    novo = input("  novo valor (Enter cancela): ").strip()
    if novo == "":
        log.info("  cancelado.")
        return
    try:
        setattr(cfg, s["attr"], _coagir_valor(s, novo))
        log.info("  -> ok")
    except ValueError as e:   # _coagir_valor: valor digitado invalido
        log.info(f"  erro: {e}")


def _menu_interativo(cfg: Optional[Config] = None,
                    caminho_cfg: str = "config.yaml") -> None:
    """Assistente de terminal: visualiza/edita a configuracao, salva/carrega
    o config.yaml e dispara o pipeline — tudo sem editar o codigo."""
    cfg = cfg if cfg is not None else Config()
    if os.path.exists(caminho_cfg):
        try:
            cfg = load_config(caminho_cfg, base=cfg)
            log.info(f"[config] carregado de {caminho_cfg}")
        except (RuntimeError, FileNotFoundError, ValueError) as e:
            # load_config so' lanca esses 3 tipos (PyYAML ausente,
            # arquivo ausente, chaves invalidas) -- ver config_io.py.
            log.info(f"[config] nao foi possivel carregar ({e}). Usando padroes.")

    while True:
        log.info("\n" + "=" * 62)
        log.info("  ASSISTENTE DO PIPELINE QUIMIOMETRICO")
        log.info("=" * 62)
        for i, s in enumerate(_CONFIG_SPEC, 1):
            log.info(f"  [{i:2d}] {s['key']:<28s}: "
                  f"{_fmt_yaml(_attr_para_yaml(s, cfg))}")
        ok, msg = _validar_pasta_dados(cfg)
        log.info("-" * 62)
        log.info(f"  dados: {msg}")
        log.info("-" * 62)
        log.info("  digite o NUMERO p/ editar  |  [S]alvar  [L]carregar  "
              "[R]odar  [Q]sair")
        escolha = input("  > ").strip().lower()

        if escolha in ("q", "sair", "quit"):
            log.info("  encerrado."); return
        if escolha in ("s", "salvar"):
            save_config(cfg, caminho_cfg)
            log.info(f"  salvo em {caminho_cfg}"); continue
        if escolha in ("l", "carregar"):
            try:
                cfg = load_config(caminho_cfg, base=cfg)
                log.info("  recarregado.")
            except (RuntimeError, FileNotFoundError, ValueError) as e:
                log.info(f"  erro: {e}")
            continue
        if escolha in ("r", "rodar", "run"):
            ok, msg = _validar_pasta_dados(cfg)
            if not ok:
                log.info(f"  [!] {msg}. Corrija antes de rodar."); continue
            _erros_sem = _validar_semantico(cfg)
            if _erros_sem:
                for _e in _erros_sem:
                    log.info(f"  [!] {_e}")
                log.info("  Corrija antes de rodar."); continue
            save_config(cfg, caminho_cfg)
            log.info("  iniciando pipeline...\n")
            executar(cfg); return
        if escolha.isdigit() and 1 <= int(escolha) <= len(_CONFIG_SPEC):
            _editar_campo(cfg, _CONFIG_SPEC[int(escolha) - 1])
        else:
            log.info("  opcao invalida.")


if __name__ == "__main__":
    import sys
    # config.yaml e um arquivo de runtime do usuario: resolvido relativo ao
    # diretorio de trabalho (raiz do projeto), nao ao pacote instalado.
    _CFG_PATH = os.path.join(os.getcwd(), "config.yaml")
    if "--rodar" in sys.argv:
        # R2-09: sem config.yaml o modo direto caia na Config embutida (pasta
        # 'dados' inexistente) e terminava em traceback. Agora delega ao mesmo
        # caminho de `guaraci run`: mensagem clara e codigo de saida 2.
        from guaraci.guaraci import _comando_run
        _comando_run(_CFG_PATH)
    elif "--codigo" in sys.argv:
        executar(_CFG)                       # mode legado (Config embutida)
    elif sys.stdin is not None and sys.stdin.isatty():
        # CLI unica (item 16 da auditoria): guaraci.py e' o unico ponto de
        # entrada interativo (cli_assistente.py virou modulo de dados/i18n).
        try:
            from guaraci.guaraci import main as _cli_main
            _cli_main()
        except ImportError:
            _menu_interativo(_CFG, _CFG_PATH)  # fallback para o menu antigo
    else:
        _cfg = load_config(_CFG_PATH) if os.path.exists(_CFG_PATH) else _CFG
        executar(_cfg)
