# Changelog — GUARACI

Histórico de versões do pipeline quimiométrico. Extraído do cabeçalho de
`pipeline.py` (a versão atual vive em `pipeline.__version__`).

> Ordem histórica original preservada como estava no código-fonte.

```
v08  base: Sprints 1-3, GroupKFold mae_id, spectral truncation
v10  2026-05-28  max_lvs=30; ddsimca_n_components=7;
                   C2: comparar_pipelines uses max_lv=cfg.max_lvs (was min(8,..))
v11 — 2026-05-28 — C3: HCA dendrogram (Ward); C4: DD-SIMCA one-class
                   (trains only on pure samples, sens/spec); C5: N3 PLS reg GroupKFold
                   by mae_id; C6: T2 outliers per class in model summary
v12 — 2026-05-28 — M1: pure(*)/adulterated(o) markers in score plots;
                   M2: sens/spec in DD-SIMCA acceptance plot titles
v13 — 2026-05-28 — M3: chemical annotation of VIP bands; M4: accuracy per
                   class in resumo_modelo.txt
v14 — 2026-05-28 — FINDING: MSC->SG+MC = 0.923 bal.acc on full dataset
                   (1807) vs autoscaling 0.472 (AUTO advantage was
                   artifact of 80% subset). Changes:
                   (1) preset "msc_sg_mc" in construir_preprocessador;
                   (2) preprocessamento_padrao default = "msc_sg_mc";
                   (3) frac_holdout default = 0.20;
                   (4) gerar_nome_saida case "msc_sg_mc" -> "MSC-SGd-MC";
                   (5) M1: stars -> circle with black edge (avoids cluttering 1807pts);
                   (6) DD-SIMCA reverts to training on ALL samples (3 pure/class
                       makes one-class infeasible; requires >=15 pure/class)
v15 — 2026-05-28 — (1) holdout_preserva_puros=True: pure samples always in training
                       (fixes "pure=0" in 4 classes after holdout);
                   (2) automatic warning "LVs at ceiling" (console + summary);
                   (3) DD-SIMCA acceptance plot in LOG-LOG scale
                       (fixes data squeezed in corner; Pomerantsev standard)
v16 — 2026-05-28 — Organization/visualization:
                   (1) salvar() accepts subfolder; (2) fig3 Hotelling T2 in
                   log scale (Y) and T2vsQ in log-log (centers the cloud);
                   (3) score_contribution split into 2 figs (spectrum +
                   top-discriminant tall/readable with side legend);
                   (4) DD-SIMCA: 14 individual plots in ddsimca/ subfolder
v17 — 2026-05-28 — MAXIMUM PERCEPTUAL DISTINCTIVENESS color system:
                   (1) PALETTE Trubetskoy/Glasbey 20 colors (deltaE_min 27.4
                       vs ~15 before; eliminates 3 near-identical blues/2 greens);
                   (2) optional detection of glasbey/colorcet libs;
                   (3) SEQUENTIAL deterministic assignment (adjacent contrast)
                       replaces hash; (4) secondary SHAPE channel
                       (mapear_marcadores_classes, 14 shapes) for
                       colorblindness/B&W; (5) edge_para_cor by luminance
v18 — 2026-05-28 — Axis readability: _ticks_x_inteiros() applies
                   MaxNLocator(integer, nbins=10) when >15 ticks
                   (LV selection and PLS regression with 30-50 LVs no longer
                   overlap numbers); <=15 shows all values.
v19 — 2026-05-28 — V3 HCA/VIP:
                   (1) HCA on centroids in PCA(hca_n_pcs=65) — reduces
                       noise; (2) dendrogram axes inverted
                       (orientation=top: species on lower X axis colored
                       and rotated, distance on left Y axis);
                   (3) fig_hca_comparacao_pipelines: dendrogram panel
                       (raw/SNV/MSC/SG1/SG2/SNV+SG1/MSC+SG1/norm);
                   (4) automatic cluster interpretation (k=2);
                   (5) VIP: y-lim on real range + statistics box
                       (min/max/mean/std/n>=1) — checks real dispersion
                   + Config flags: mostrar_marcadores_classe/elipses_grupo
v20 — 2026-05-28 — Organization Q1: folder PLSDA_OE_{level}_{preproc}_
                   {YYYYMMDD_HHMMSS} with subfolders dados/ figuras/
                   modelos/ logs/. Figures->figuras/; metadata,
                   identifiers, comparison->dados/; summary->logs/;
                   final model (joblib: preproc+PLS+LB+wavenumbers)
                   ->modelos/. Sprint1 audit (A1,A2,A3,A5,A6,A11):
                   confirmed ALREADY implemented in previous versions.
v22 — 2026-05-29 — Phase 0 (rigor fixes):
                   B1: validar_entrada synchronizes mae_id with the SAME mask
                       for NaN/Inf removal (before, 1 NaN silently disabled
                       GroupKFold = replica leakage);
                   B4: DD-SIMCA 'todos' mode no longer reports misleading
                       in-sample "spec" (spec=n/a; mode label in
                       summary makes clear that sens/spec != authentication);
                   B7: Q-residual in summary with adaptive notation (:.4g
                       when <1e-3) instead of displaying 0.0000.
v21 — 2026-05-28 — STAGE 4 (variable selection) + class exclusion:
                   (1) Config.excluir_classes (e.g. Copaiba anomalous batch);
                   (2) iPLS (intervals), selection by VIP>=threshold, by SR
                       (top fraction), sPLS-DA (NIPALS soft-selection);
                   (3) single evaluator _avaliar_subset_cv (group-aware CV,
                       MC re-fitted per fold = no leakage);
                   (4) figures fig_etapa4_ipls_intervalos +
                       fig_etapa4_comparacao_metodos; CSVs in dados/;
                   (5) most PARSIMONIOUS method selected (bal.acc within
                       1% of max, fewer variables) in summary.
v24 — 2026-05-29 — Sprint v24: Publication Figures:
                   (1) fig_loadings_pca: PCA Loading Plot PC1/PC2 (bars
                       colored by sign, NIR inverted X axis);
                   (2) fig_roc_auc: Multiclass ROC curves OvR (scores
                       group-aware CV; macro AUC in title and summary);
                   (3) fig_splot_opls: OPLS-DA S-Plot (covariance x
                       correlation with t_pred; top-N annotated; colormap
                       RdBu_r; ref. Bylesjo 2006);
                   (4) fig_cooman_ddsimca: DD-SIMCA Cooman's Plot (pairs
                       A x B; sqrt(dQ) scale; subplot grid;
                       ref. Pomerantsev 2020).
                   Integration: aucs_roc added to resumo_modelo.txt.
v23 — 2026-05-29 — ACCESSIBLE LAYER (no code editing):
                   (1) _CONFIG_SPEC: single source mapping friendly names
                       <-> Config attributes, with type,
                       description and options for validation;
                   (2) salvar_config/carregar_config: commented YAML in
                       plain language; defaults preserved for missing keys;
                       unknown keys ignored;
                   (3) menu_interativo: terminal assistant (CMD-style)
                       to edit fields, save/load and run without opening
                       the code editor;
                   (4) new __main__: --rodar (uses config.yaml), --codigo
                       (legacy CFG), or interactive menu when in terminal;
                   (5) config.yaml template generated (excludes Copaiba
                       anomalous batch, max_lvs=40). Pipeline logic INTACT.
v27  benchmark_classificadores integrated into executar()
v28  Monte Carlo CV (IC95%); SHAP TreeExplainer; DET curves (linear+log)
v29  hardware_probe; auto RAM tiers (4 levels); RAM guards; cleanup util
v30  PowerPoint export; .streamlit/config.toml; CLAUDE.md; English i18n
v31  Bugfix: (1) iPLS/comparar_pipelines Q2 overflow guard (ss_res non-finite
             → q2=nan; eliminates -3.9e31 artifact in narrow intervals);
          (2) Wold permutation loop: filter non-finite r2/q2 before polyfit
             (fixes NaN intercept when degenerate model produces blow-up);
          (3) Wold fig: correct N/A status when intercept is NaN;
          (4) resumo_modelo.txt: Q2 NaN shown as "n/a" instead of crash;
          (5) config.yaml: N1 14-class, pasta dados/, ddsimca todos
```
