# Changelog — GUARACI

Histórico de versões do pipeline quimiométrico. Extraído do cabeçalho de
`pipeline.py` (a versão atual vive em `pipeline.__version__`).

> Ordem histórica original preservada como estava no código-fonte.

```
v31.5.0 — 2026-07-13 — print() -> logging em pipeline.py (CLAUDE.md P6, parcial):
             (1) 164 chamadas `print()` em `pipeline.py` migradas para
                 `log.info()` (`log = logging.getLogger(__name__)`).
                 `src/guaraci/log.py` novo: ponto único de configuração,
                 com handler que escreve em `sys.stdout` NO MOMENTO do
                 emit (não uma referência capturada na importação) --
                 necessário para continuar funcionando dentro do
                 `contextlib.redirect_stdout` que o CLI e o worker do app
                 web usam para capturar o log e alimentar o painel de
                 progresso ao vivo. Verificado com teste de integração
                 dedicado que roda o pipeline sintético de verdade e
                 confirma que os regex do painel (`app_logic.py`) ainda
                 casam com o texto capturado antes E depois da migração;
             (2) PARCIAL DE PROPÓSITO: o painel do CLI/app web continua
                 fazendo parsing de texto por regex, não consumindo
                 registros de logging estruturados -- essa reescrita
                 (a solução completa que o CLAUDE.md P6 propõe) é um
                 projeto à parte, não feito aqui. `log.info()` preserva o
                 mesmo texto que `print()` produzia, então resolve a
                 inconsistência entre módulos mas não a fragilidade de
                 fundo (mudar uma string ainda quebraria o painel).
```

```
v31.4.0 — 2026-07-13 — Preparação para submissão JOSS:
             (1) Benchmark contra dataset público externo (Tecator, NIR,
                 teor de gordura em carne — Thodberg 1996): roda o motor
                 real de pré-processamento + regressão PLS do GUARACI
                 (não uma reimplementação) no split oficial 172/43,
                 RMSEP 2,0-2,3% / R²pred 0,97-0,98, dentro da faixa
                 esperada da literatura. Script reprodutível
                 (`scripts/benchmark_tecator.py`, baixa o dado da fonte
                 original a cada execução) + write-up completo
                 (`docs/BENCHMARK_TECATOR.md`) com metodologia, resultados
                 e limitações honestas (não cobre DD-SIMCA/classificação/
                 group-aware — Tecator não tem réplicas físicas). Fecha a
                 lacuna citada em VALIDATION.md/MANUAL.md;
             (2) fix(web): mesmo bug de sincronização de widget do preset
                 (v31.3.0) também corrigido no botão pré-existente
                 "↺ Reload config.yaml" — extraído para
                 `_sincronizar_widgets_com_cfg()` compartilhada;
             (3) docs: `paper.md`/`CONTRIBUTING.md` sincronizados com a
                 contagem real de testes (550+, não 525+/498+);
                 `paper.bib` ganha a referência Thodberg1996.
```

```
v31.3.0 — 2026-07-13 — Correções da auditoria multidisciplinar de 15 etapas (2026-07-12):
             (1) BREAKING: Etapa 4 (seleção de variáveis) corrige viés de
                 seleção não-aninhada. VIP>=threshold, SR top-fração e
                 sPLS-DA calculavam a máscara de variáveis a partir de um
                 modelo ajustado no DATASET INTEIRO (double dipping —
                 Ambroise & McLachlan, 2002, PNAS) antes de avaliar por CV.
                 Agora usam nested-CV (`_avaliar_subset_nested_cv`,
                 `selecao_variaveis.py`): a máscara é recalculada a cada
                 fold usando só os dados de treino daquele fold. Resultados
                 numéricos de balanced_accuracy da Etapa 4 para esses 3
                 métodos NÃO são comparáveis com versões anteriores (tende
                 a cair, o que é o objetivo — número anterior era otimista).
                 iPLS não precisou de correção: a partição em intervalos não
                 usa rótulo, só a escolha do "melhor intervalo" usa CV (viés
                 padrão de qualquer seleção de hiperparâmetro, não double
                 dipping). `etapa4_selecao_variaveis()` não recebe mais
                 `vip`/`sr` pré-calculados como parâmetro;
             (2) menu "Visualização" da CLI (`guaraci.py`) tinha 4 opções
                 (H/M/B/V — heatmap espectral, matriz de confusão, biplot,
                 variância×wavelength) que sempre falhavam: apontavam para
                 funções que nunca existiram em nenhum módulo do projeto,
                 mascaradas por um `except Exception` genérico. Opções
                 removidas do menu (gerar essas figuras fora de uma
                 execução completa é feature nova, não bugfix — fica para
                 quando for implementada de verdade);
             (3) tooltip do assistente "G" (`GUARACI_TIPS["nivel"]`) não
                 seguia a convenção "nome amigável primeiro, código entre
                 parênteses" já usada nos rótulos de menu (P8) — corrigido;
             (4) CLAUDE.md sincronizado com o estado real do código (a
                 tabela "estado alegado" e a lista de problemas P1-P9
                 estavam desatualizadas desde a v31.2.0);
             (5) BREAKING: mesma correção de (1), agora para AG/SPA (Etapa
                 4, opt-in) — achado colateral mais grave que o de VIP/SR:
                 a fitness do AG e a pontuação do SPA usavam a MESMA CV do
                 número final reportado (double dipping por construção,
                 não só double dipping indireto via modelo pré-ajustado).
                 Corrigido com `_avaliar_busca_nested_cv`: busca refeita a
                 cada fold externo, usando só o treino daquele fold; custo
                 de execução sobe ~N vezes (N = nº de folds), aceitável
                 pois ambos já são opt-in/documentados como lentos;
             (6) nomes de pasta de saída (`PLSDA_OE_N2_...`) e de figura
                 (`figN3_heatmap_...`) não expõem mais N1/N2/N3 cru — slugs
                 amigáveis (`_NIVEL_SLUG_PASTA`, `config.py`):
                 `PLSDA_OE_Autenticacao_...`, `fig_heatmap_especie_adulterante.png`.
                 `cfg.nivel` continua "N1"/"N2"/"N3" internamente; só o nome
                 em disco mudou (P8 residual, decisão aprovada explicitamente
                 por ser mudança de formato de saída);
             (7) 3 presets por objetivo científico — "Explorar Dados" /
                 "Autenticar Pureza" / "Quantificar Teor" (CLI: `menu_perfis`;
                 app web: aba Dados) — reaproveitam `PROFILES`
                 (`cli_assistente.py`), mesma fonte usada pelos perfis de
                 rigor já existentes. CLI: aplicar um perfil agora pergunta
                 "Rodar agora?" e chama `_rodar_pipeline` direto. App web:
                 corrigido também um bug de sincronização de estado dos
                 widgets do Streamlit (`key=` estático só honra o valor novo
                 se escrito direto em `st.session_state[key]`, não bastando
                 apagar a chave) — sem essa correção os presets mudavam
                 `cfg_base` mas os widgets da aba Model continuavam
                 mostrando o valor antigo.
```

```
v31.2.0 — 2026-07-12 — Mudanças de COMPORTAMENTO CIENTÍFICO (CLAUDE.md P1/P2/P5):
             (1) BREAKING: sensibilidade DD-SIMCA deixa de ser re-substituição
                 (treino==teste, inflava para ~100%) e passa a ser estimada por
                 leave-one-group-out (LOGO) sobre mae_id. O dict de resultado
                 passa a expor `n_grupos` e um `aviso` quando n_grupos<10.
                 Resultados numéricos de sensibilidade gerados por versões
                 anteriores NÃO são comparáveis com esta versão;
             (2) heatmap espécie×adulterante (R²cv) passa a ser figura nativa
                 de `executar()` no objetivo Quantificação, com contagem de
                 combinações abaixo de R²cv=0.70 no título;
             (3) `predicao.carregar_modelo` passa a exigir `confiar=True`
                 explícito (joblib/pickle executa código arbitrário) e cada
                 modelo salvo passa a vir com manifesto SHA-256
                 (docs/SECURITY.md);
             (4) auditoria dos blocos `except Exception`/`except:` (P3):
                 maioria estreitada para o tipo real de erro; 1 bug real
                 corrigido (fallback silencioso LDA→PLS2 no OPLS-DA sem log);
             (5) 2 figuras novas: espectros médios por classe (±1 desvio) e
                 biplot PCA (scores+loadings);
             (6) vocabulário N1/N2/N3 aposentado da UI (P8; mantido como
                 apelido interno) — ver tabela de equivalência no MANUAL;
             (7) docs/VALIDATION.md e seção "Limitações conhecidas" no MANUAL.
```

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
